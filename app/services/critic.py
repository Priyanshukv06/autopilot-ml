"""Critic: finds what is wrong with the data or with an Experiment, with evidence (Requirement 5).

The Critic reads only Training_Part information: the profile of the Training_Part, the Setup, the
Assessment on the Session_Folds and the search summary. It is a pure function of these inputs, so
the Diagnoses of a finished Session can be recomputed from its records. All thresholds come from
``critic_thresholds.toml``, and every Diagnosis records that file's version label.

Thin version (task 6): overfitting, class_imbalance and high_cardinality. The other kinds are
added in task 13.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from app.services.assessor import Assessment
from app.services.automl_runner import ExperimentConfig
from app.services.profiler import DatasetProfile
from app.services.search_summary import SearchSummary
from app.services.task_detector import TaskType

DEFAULT_THRESHOLDS_PATH = Path(__file__).with_name("critic_thresholds.toml")

# Metrics that are bounded and higher-is-better; any other primary metric is replaced by F1
# (macro F1 for multiclass) or R2 for the model-level rules, so that gaps have one scale.
_BOUNDED_METRICS = ("f1", "accuracy", "roc_auc", "r2")


class DiagnosisKind(str, Enum):
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    BUDGET_LIMITED = "budget_limited"
    WASTED_BUDGET = "wasted_budget"
    CLASS_IMBALANCE = "class_imbalance"
    MISSING_VALUES = "missing_values"
    HIGH_CARDINALITY = "high_cardinality"
    IDENTIFIER_COLUMN = "identifier_column"
    CONSTANT_COLUMN = "constant_column"
    REDUNDANT_FEATURES = "redundant_features"
    UNRELIABLE_VALIDATION = "unreliable_validation"
    WEAK_MODEL_FAMILIES = "weak_model_families"


class Severity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass(frozen=True)
class Diagnosis:
    kind: DiagnosisKind
    severity: Severity
    columns: tuple[str, ...]
    evidence: dict[str, float | str]
    message: str
    thresholds_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "columns": list(self.columns),
            "evidence": dict(self.evidence),
            "message": self.message,
            "thresholds_version": self.thresholds_version,
        }


class ThresholdsError(ValueError):
    """The thresholds file is missing or invalid."""


# Every threshold the implemented rules read, as "section.name".
REQUIRED_THRESHOLDS = (
    "overfitting.moderate_gap",
    "overfitting.high_gap",
    "class_imbalance.moderate_ratio",
    "class_imbalance.high_ratio",
    "class_imbalance.moderate_recall",
    "class_imbalance.high_recall",
    "class_imbalance.min_ratio_for_recall",
    "high_cardinality.moderate_distinct",
    "high_cardinality.moderate_unseen",
    "high_cardinality.high_unseen",
)
# Pairs (less severe, more severe) that must be in this order: (a, b) means a <= b.
_ORDERED = (
    ("overfitting.moderate_gap", "overfitting.high_gap"),
    ("class_imbalance.moderate_ratio", "class_imbalance.high_ratio"),
    ("class_imbalance.high_recall", "class_imbalance.moderate_recall"),
    ("high_cardinality.moderate_unseen", "high_cardinality.high_unseen"),
)


@dataclass(frozen=True)
class CriticThresholds:
    version: str
    values: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> CriticThresholds:
        """Read and check a thresholds file (the default one next to this module if not given)."""
        path = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ThresholdsError(f"Critic thresholds file not found: {path}") from exc
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ThresholdsError(f"Critic thresholds file {path} cannot be read: {exc}") from exc
        return cls.from_mapping(data, source=str(path))

    @classmethod
    def from_mapping(cls, data: dict[str, Any], source: str = "thresholds") -> CriticThresholds:
        version = data.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ThresholdsError(f"{source}: 'version' must be a non-empty text label.")
        values: dict[str, float] = {}
        for section, table in data.items():
            if section == "version":
                continue
            if not isinstance(table, dict):
                raise ThresholdsError(f"{source}: '{section}' must be a table of thresholds.")
            for name, value in table.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ThresholdsError(f"{source}: '{section}.{name}' must be a number.")
                values[f"{section}.{name}"] = float(value)
        missing = [key for key in REQUIRED_THRESHOLDS if key not in values]
        unknown = sorted(set(values) - set(REQUIRED_THRESHOLDS))
        if missing:
            raise ThresholdsError(f"{source}: missing thresholds {missing}.")
        if unknown:
            raise ThresholdsError(f"{source}: unknown thresholds {unknown} (a typing mistake?).")
        for low, high in _ORDERED:
            if values[low] > values[high]:
                raise ThresholdsError(f"{source}: '{low}' ({values[low]:g}) must not be above '{high}' "
                                      f"({values[high]:g}).")
        return cls(version=version.strip(), values=values)

    def __getitem__(self, key: str) -> float:
        return self.values[key]


def diagnostic_metric(task_type: TaskType, primary_metric: str) -> str:
    """The metric of the model-level rules: the primary metric if it is bounded and higher is
    better, otherwise F1 (macro F1 for multiclass) or R2."""
    if primary_metric in _BOUNDED_METRICS:
        return primary_metric
    return "f1" if task_type.is_classification else "r2"


class Critic:
    def __init__(self, thresholds: CriticThresholds) -> None:
        self.thresholds = thresholds

    def diagnose(
        self,
        *,
        profile: DatasetProfile,
        config: ExperimentConfig,
        assessment: Assessment,
        search: SearchSummary | None,
        task_type: TaskType,
        primary_metric: str,
    ) -> list[Diagnosis]:
        """Diagnoses in a fixed order: by rule (the order of design.md's table), then by column name.

        ``profile`` must describe the Training_Part only, and ``assessment`` must be complete.
        ``search`` is read by the budget rules of the full version.
        """
        if assessment.failed or not assessment.folds:
            raise ValueError("The Critic needs a completed Assessment.")
        used = self._used_columns(profile, config)
        metric = diagnostic_metric(task_type, primary_metric)
        found: list[Diagnosis | None] = [self._overfitting(assessment, metric)]
        if task_type.is_classification:
            found.append(self._class_imbalance(profile, assessment))
        found.extend(self._high_cardinality(profile, assessment, column) for column in used
                     if column in profile.categorical_features)
        return [d for d in found if d is not None]

    # ------------------------------------------------------------------ rules
    def _overfitting(self, assessment: Assessment, metric: str) -> Diagnosis | None:
        train, validation = assessment.values(metric, "train"), assessment.values(metric, "validation")
        both = np.isfinite(train) & np.isfinite(validation)
        if not both.any():
            return None
        gap = float(np.mean(train[both] - validation[both]))
        severity = self._level(gap >= self.thresholds["overfitting.high_gap"],
                               gap >= self.thresholds["overfitting.moderate_gap"])
        if severity is None:
            return None
        train_mean, validation_mean = float(train[both].mean()), float(validation[both].mean())
        return self._make(
            DiagnosisKind.OVERFITTING, severity, (),
            {"metric": metric, "gap": gap, "train_mean": train_mean, "validation_mean": validation_mean,
             "folds": int(both.sum())},
            f"The model scores {train_mean:.3f} on the rows it was trained on but {validation_mean:.3f} on "
            f"validation rows ({metric}, mean over {int(both.sum())} folds), a gap of {gap:.3f}: it fits "
            "details of the training rows that do not carry over to new rows.",
        )

    def _class_imbalance(self, profile: DatasetProfile, assessment: Assessment) -> Diagnosis | None:
        balance = profile.class_balance
        if balance is None:
            return None
        ratio = float(balance.imbalance_ratio)
        recalls = [fold.class_recall.get(balance.minority_class) for fold in assessment.folds if fold.class_recall]
        recalls = [float(r) for r in recalls if r is not None]
        recall = float(np.mean(recalls)) if recalls else None
        use_recall = recall is not None and ratio >= self.thresholds["class_imbalance.min_ratio_for_recall"]
        severity = self._level(
            ratio >= self.thresholds["class_imbalance.high_ratio"]
            or (use_recall and recall < self.thresholds["class_imbalance.high_recall"]),
            ratio >= self.thresholds["class_imbalance.moderate_ratio"]
            or (use_recall and recall < self.thresholds["class_imbalance.moderate_recall"]),
        )
        if severity is None:
            return None
        evidence: dict[str, float | str] = {"imbalance_ratio": ratio, "minority_class": balance.minority_class,
                                            "minority_share": float(balance.minority_ratio)}
        message = (f"The smallest class '{balance.minority_class}' has 1 training row for every {ratio:.1f} rows "
                   "of the largest class")
        if recall is not None:
            evidence.update({"minority_recall": recall, "folds": len(recalls)})
            message += f", and the model finds {recall:.0%} of its rows in validation (mean over {len(recalls)} folds)"
        return self._make(DiagnosisKind.CLASS_IMBALANCE, severity, (str(profile.target_column),), evidence,
                          message + ".")

    def _high_cardinality(self, profile: DatasetProfile, assessment: Assessment, column: str) -> Diagnosis | None:
        distinct = int(profile.unique_counts.get(column, 0))
        shares = [fold.unseen_share[column] for fold in assessment.folds if column in fold.unseen_share]
        unseen = float(np.mean(shares)) if shares else None
        severity = self._level(
            unseen is not None and unseen >= self.thresholds["high_cardinality.high_unseen"],
            distinct > self.thresholds["high_cardinality.moderate_distinct"]
            or (unseen is not None and unseen >= self.thresholds["high_cardinality.moderate_unseen"]),
        )
        if severity is None:
            return None
        evidence: dict[str, float | str] = {"distinct_values": distinct}
        message = f"'{column}' has {distinct} different values in the training part."
        if unseen is not None:
            evidence.update({"unseen_share": unseen, "folds": len(shares)})
            message += (f" On average {unseen:.0%} of validation rows have a value that the fold's training rows "
                        f"never had (mean over {len(shares)} folds); the model gets no information from this "
                        "column for those rows.")
        return self._make(DiagnosisKind.HIGH_CARDINALITY, severity, (column,), evidence, message)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _used_columns(profile: DatasetProfile, config: ExperimentConfig) -> list[str]:
        """Feature columns of the Setup, sorted by name."""
        skipped = {profile.target_column, *config.excluded_columns}
        return sorted(str(c) for c in profile.column_names if c not in skipped)

    @staticmethod
    def _level(high: bool, moderate: bool) -> Severity | None:
        if high:
            return Severity.HIGH
        if moderate:
            return Severity.MODERATE
        return None

    def _make(self, kind: DiagnosisKind, severity: Severity, columns: tuple[str, ...],
              evidence: dict[str, float | str], message: str) -> Diagnosis:
        return Diagnosis(kind, severity, columns, evidence, message, self.thresholds.version)
