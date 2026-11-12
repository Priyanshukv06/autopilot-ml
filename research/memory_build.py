"""The frozen Experiment_Memory of the Phase 3 benchmark (spec task 26, Requirement 22.3).

The memory is built from Sessions on the 10 development datasets only, then frozen as a copy whose
SHA-256 is recorded in every benchmark row. Decisions, fixed on 27 October 2026 before the build:

- Sessions: every development dataset with seeds 1, 2 and 3 (30 Sessions), 300 s each (the benchmark's
  middle budget), at most 8 Experiments, the default Session settings otherwise, all CPU cores, MLflow off.
- Order of the changes: the ``random`` selector. The rule order always tries the same changes first on the
  same data, so a memory built with it would hold few records for the other changes, and a learned selector
  needs 3 records of an arm before it moves that arm (Requirement 19.4). A random order spreads the records
  over every change that applies. No benchmark system uses this order except the ``random_choice`` ablation.
- The build starts from a new, empty memory file; the Phase 2 memory is not used, so the frozen memory comes
  from one documented procedure.
- Each finished Session is written as one row (the benchmark's row format) and a restart skips finished
  Sessions. A Session stopped in the middle may already have written records; ``freeze`` keeps only the
  records of finished Sessions, so the frozen memory is exactly the records of the rows.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from app.core.config import Settings, build_settings
from app.services.autopilot import CODE_ROOT, code_version
from app.services.experiment_memory import ExperimentMemory
from research import benchmark
from research.datasets import datasets
from research.systems import AutopilotSystem, System

SYSTEM = "memory_build"
OPTIONS = {"selector": "random", "use_memory": False}
SEEDS = (1, 2, 3)
SECONDS = 300


def folders(project_root: Path, label: str = "") -> dict[str, Any]:
    """Where the build writes: the Sessions' records (``experiments/benchmark-memory/``, models in
    ``models/benchmark-memory/``), the growing memory inside them, the rows, and the frozen copy. With a
    ``label`` (12 November 2026), ``benchmark`` becomes ``benchmark-<label>`` in these names."""
    run = benchmark.run_name(label)
    base = build_settings(Path(project_root))
    settings = replace(base, experiments_dir=base.experiments_dir / f"{run}-memory",
                       models_dir=base.models_dir / f"{run}-memory")
    results = base.project_root / "results" / run
    return {"settings": settings, "memory": settings.experiments_dir / "memory" / "memory.jsonl",
            "rows": results / "memory_sessions.csv", "frozen": results / "frozen_memory.jsonl",
            "summary": results / "frozen_memory_summary.csv"}


def cells(names: Iterable[str] | None = None, seeds: Iterable[int] = SEEDS, seconds: int = SECONDS) -> list[benchmark.Cell]:
    """Seed by seed, then dataset by dataset (the registry's order)."""
    names = list(names) if names is not None else [e.name for e in datasets("development")]
    return [benchmark.Cell(SYSTEM, name, seconds, seed) for seed in seeds for name in names]


def make(memory_path: Path) -> System:
    return AutopilotSystem(SYSTEM, memory_path, options=OPTIONS, read_only=False)


def build(cells_: Sequence[benchmark.Cell], settings: Settings, rows_path: Path, memory_path: Path, *,
          n_jobs: int = -1, make_system: Callable[[str], System] | None = None,
          load: Callable[[str], benchmark.Dataset] = benchmark.load_dataset,
          report: Callable[[dict[str, Any], int, int], None] | None = None) -> list[dict[str, Any]]:
    """Run the Sessions that have no row yet; each row records the memory as its Session found it."""
    benchmark.check_rows_file(rows_path)
    entries = {e.name for e in datasets("development")}
    if outside := sorted({c.dataset for c in cells_} - entries):
        raise ValueError(f"The memory is built from development datasets only, not {', '.join(outside)}.")
    rows = benchmark.read_cells(rows_path)
    todo = benchmark.pending(cells_, rows)
    version = code_version(CODE_ROOT)
    settings.experiments_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    make_system = make_system or (lambda name: make(memory_path))
    loaded: dict[str, benchmark.Dataset] = {}
    written = []
    for number, cell in enumerate(todo, start=1):
        if cell.dataset not in loaded:
            loaded[cell.dataset] = load(cell.dataset)
        memory = ExperimentMemory(memory_path, read_only=True)
        context = {"memory_file": benchmark._relative(memory_path), "memory_hash": memory.snapshot_hash(),
                   "memory_records": len(memory.records()), "version": version}
        row = benchmark.run_cell(cell, loaded[cell.dataset], make_system(cell.system), settings, n_jobs=n_jobs,
                                 context=context)
        benchmark.append_row(rows_path, row)
        written.append(row)
        if report is not None:
            report(row, number, len(todo))
    return written


def freeze(memory_path: Path, rows: Sequence[dict[str, Any]], frozen_path: Path) -> dict[str, Any]:
    """Copy the records of the finished Sessions (rows with status completed) to ``frozen_path``, which must
    not exist yet: a frozen memory is never rewritten. Returns the counts and the SHA-256 of the copy."""
    if frozen_path.exists():
        raise FileExistsError(f"{frozen_path} exists; a frozen memory is never rewritten.")
    finished = {r["run_id"] for r in benchmark.latest(rows).values() if r.get("status") == "completed" and r.get("run_id")}
    records = ExperimentMemory(memory_path, read_only=True).records()
    kept = [r for r in records if r.session_id in finished]
    frozen = ExperimentMemory(frozen_path)
    for record in kept:
        frozen.add(record)
    return {"sessions": len(finished), "records": len(kept), "left_out": len(records) - len(kept),
            "sha256": frozen.snapshot_hash()}


def summary(memory_path: Path) -> pd.DataFrame:
    """Per task type, Strategy and diagnosis kind (the selectors' arms, counted as ``strategy_stats`` counts
    them: a record under every kind it answered): records, datasets, keeps (verdict keep or keep_simpler),
    records without effect, keeps that the arm statistics count (since 31 October 2026), and the mean Judge
    difference and cost."""
    rows = []
    for record in ExperimentMemory(memory_path, read_only=True).records():
        kinds = sorted({d["kind"] for d in record.diagnoses}) or ["none"]
        for kind in kinds:
            rows.append({"task_type": record.task_type, "strategy": record.strategy["strategy_id"],
                         "diagnosis_kind": kind, "dataset": record.dataset_name, "kept": record.kept,
                         "no_effect": record.no_effect, "counts_as_kept": record.counts_as_kept,
                         "mean_difference": record.mean_difference, "cost_seconds": record.cost_seconds})
    if not rows:
        return pd.DataFrame(columns=["task_type", "strategy", "diagnosis_kind", "records", "datasets", "keeps",
                                     "no_effect", "counted_keeps", "mean_difference", "mean_cost_seconds"])
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["task_type", "strategy", "diagnosis_kind"], sort=True)
    return grouped.agg(records=("kept", "size"), datasets=("dataset", "nunique"), keeps=("kept", "sum"),
                       no_effect=("no_effect", "sum"), counted_keeps=("counts_as_kept", "sum"),
                       mean_difference=("mean_difference", "mean"),
                       mean_cost_seconds=("cost_seconds", "mean")).reset_index()


__all__ = ["OPTIONS", "SEEDS", "SECONDS", "SYSTEM", "build", "cells", "folders", "freeze", "make", "summary"]
