"""Experiment_Memory: every judged Strategy, with its effect, its cost and a description of the dataset.

Version 1 (Requirement 14). The memory is one JSON object per line in
``experiments/memory/memory.jsonl``, written append-only, so earlier records are never rewritten; a
damaged line (for example the last one after a crash) is skipped with a warning. It can find the
past datasets most similar to a new one and summarise how each Strategy did on them. In Phase 2 it
is only written and shown; Phase 3's learned strategy choice reads it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.logging_config import get_logger

logger = get_logger(__name__)

MEMORY_SCHEMA_VERSION = 1
KEPT_VERDICTS = ("keep", "keep_simpler")
NO_EFFECT_TOLERANCE = 1e-12  # the Assessor's refit tolerance (REPEAT_TOLERANCE)


def dataset_fingerprint(dataframe: pd.DataFrame, target_column: str) -> str:
    """SHA-256 of the column names, their dtypes, the number of rows and the target name."""
    parts = [f"{column}:{dtype}" for column, dtype in zip(dataframe.columns, dataframe.dtypes.astype(str))]
    text = json.dumps({"columns": parts, "rows": len(dataframe), "target": target_column}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class MemoryRecord:
    record_id: str
    created_at: str
    session_id: str
    experiment_id: str
    dataset_name: str
    dataset_fingerprint: str  # dataset_fingerprint(): column names, dtypes, rows and target name
    task_type: str
    primary_metric: str
    meta_features: dict[str, float]  # of the Training_Part
    diagnoses: list[dict[str, str]]  # kind and severity of the Diagnoses answered
    strategy: dict[str, Any]  # StrategySpec as a dict
    verdict: str
    mean_difference: float
    std_difference: float
    n_folds: int
    cost_seconds: float
    utility: float | None
    engine: str
    schema_version: int = MEMORY_SCHEMA_VERSION

    @property
    def kept(self) -> bool:
        return self.verdict in KEPT_VERDICTS

    @property
    def no_effect(self) -> bool:
        """The Judge's fold differences were all zero within the refit tolerance: the change gave the same
        fold scores (in practice the same model), so it is no evidence that the Strategy helps."""
        return abs(self.mean_difference) <= NO_EFFECT_TOLERANCE and abs(self.std_difference) <= NO_EFFECT_TOLERANCE

    @property
    def counts_as_kept(self) -> bool:
        """What the arm statistics count as a keep: kept, and with a measurable effect (since 31 October
        2026; before, a kept step that changed nothing counted as a keep too)."""
        return self.kept and not self.no_effect

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        """Unknown keys (from a later schema) are ignored; missing ones raise."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class StrategyStats:
    strategy_id: str
    diagnosis_kind: str
    observations: int
    weighted_keeps: float  # sum of 1 / (1 + distance) over observations that count as kept
    weighted_discards: float  # over the others, including kept steps that changed nothing
    mean_gain_when_kept: float | None  # mean of the Judge's mean difference over observations that count as kept
    mean_cost_seconds: float
    datasets: int = 0  # distinct datasets among the observations
    no_effect: int = 0  # observations whose change made no difference to any fold score

    @property
    def keep_rate(self) -> float:
        total = self.weighted_keeps + self.weighted_discards
        return self.weighted_keeps / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "keep_rate": self.keep_rate}


class ExperimentMemory:
    def __init__(self, path: Path, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self.skipped_lines = 0  # damaged lines found by the last records() call

    # ------------------------------------------------------------- records
    def add(self, record: MemoryRecord) -> None:
        """Append one line; the file is flushed to disk so that a crash cannot lose earlier records."""
        if self.read_only:
            raise PermissionError(f"The memory {self.path} was opened read-only.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), sort_keys=True, default=str, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())

    def records(self) -> list[MemoryRecord]:
        """Every readable record, in the order written; damaged lines are skipped with a warning."""
        self.skipped_lines = 0
        if not self.path.exists():
            return []
        records = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(MemoryRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                self.skipped_lines += 1
                logger.warning("Skipping damaged line %d of %s: %s", number, self.path, exc)
        return records

    def snapshot_hash(self) -> str:
        """SHA-256 of the memory file as it is now (of nothing, if it does not exist)."""
        digest = hashlib.sha256()
        if self.path.exists():
            digest.update(self.path.read_bytes())
        return digest.hexdigest()

    # ------------------------------------------------------------ similarity
    def similar(self, meta_features: dict[str, float], task_type: str, k: int = 5,
                exclude_fingerprint: str | None = None) -> list[tuple[str, float]]:
        """The k past datasets of the same task type nearest to ``meta_features``: (fingerprint, distance).

        Each stored dataset is the average of its records' Meta_Features. Every feature is standardised
        with its mean and standard deviation over the stored datasets; features with no spread are left
        out, and only features that both datasets have are compared (Euclidean distance). Ties are
        broken by fingerprint.
        """
        return self._nearest(self.records(), meta_features, task_type, k, exclude_fingerprint)

    def strategy_stats(self, meta_features: dict[str, float], task_type: str, k: int = 5,
                       exclude_fingerprint: str | None = None) -> list[StrategyStats]:
        """How each (Strategy, diagnosis kind) did on the k nearest datasets; each observation is weighted
        by 1 / (1 + distance of its dataset). A kept step that changed no fold score counts as not kept
        (``MemoryRecord.counts_as_kept``). Sorted by Strategy and kind."""
        records = self.records()
        nearest = dict(self._nearest(records, meta_features, task_type, k, exclude_fingerprint))
        arms: dict[tuple[str, str], list[tuple[MemoryRecord, float]]] = {}
        for record in records:
            if record.dataset_fingerprint not in nearest or record.task_type != task_type:
                continue
            weight = 1.0 / (1.0 + nearest[record.dataset_fingerprint])
            for kind in sorted({d["kind"] for d in record.diagnoses}) or ["none"]:
                arms.setdefault((record.strategy["strategy_id"], kind), []).append((record, weight))
        stats = []
        for (strategy_id, kind), observed in sorted(arms.items()):
            kept = [r for r, _ in observed if r.counts_as_kept]
            stats.append(StrategyStats(
                strategy_id=strategy_id,
                diagnosis_kind=kind,
                observations=len(observed),
                weighted_keeps=sum(w for r, w in observed if r.counts_as_kept),
                weighted_discards=sum(w for r, w in observed if not r.counts_as_kept),
                mean_gain_when_kept=sum(r.mean_difference for r in kept) / len(kept) if kept else None,
                mean_cost_seconds=sum(r.cost_seconds for r, _ in observed) / len(observed),
                datasets=len({r.dataset_fingerprint for r, _ in observed}),
                no_effect=sum(1 for r, _ in observed if r.no_effect),
            ))
        return stats

    @staticmethod
    def _nearest(records: list[MemoryRecord], meta_features: dict[str, float], task_type: str, k: int,
                 exclude_fingerprint: str | None) -> list[tuple[str, float]]:
        datasets: dict[str, dict[str, list[float]]] = {}
        for record in records:
            if record.task_type != task_type or record.dataset_fingerprint == exclude_fingerprint:
                continue
            features = datasets.setdefault(record.dataset_fingerprint, {})
            for name, value in record.meta_features.items():
                if value is not None and math.isfinite(float(value)):
                    features.setdefault(name, []).append(float(value))
        if not datasets:
            return []
        means = {fp: {name: sum(v) / len(v) for name, v in features.items()} for fp, features in datasets.items()}
        scale: dict[str, tuple[float, float]] = {}
        for name in sorted({name for features in means.values() for name in features}):
            values = [features[name] for features in means.values() if name in features]
            centre = sum(values) / len(values)
            spread = math.sqrt(sum((v - centre) ** 2 for v in values) / len(values))
            if spread > 0:
                scale[name] = (centre, spread)

        def distance(features: dict[str, float]) -> float:
            shared = [n for n in scale if n in features and n in meta_features and meta_features[n] is not None
                      and math.isfinite(float(meta_features[n]))]
            return math.sqrt(sum(((features[n] - float(meta_features[n])) / scale[n][1]) ** 2 for n in shared))

        ranked = sorted(((fp, distance(features)) for fp, features in means.items()), key=lambda x: (x[1], x[0]))
        return ranked[:k]
