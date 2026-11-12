"""Phase 3 benchmark harness (spec task 24, Requirement 23).

Runs every comparison system of ``research/systems.py`` on every dataset, budget and seed. A cell is one
(system, dataset, budget, seed). In a cell, the system gets the Training_Part, Test_Part and columns that
every other system gets for that dataset and seed (``systems.prepare``), and the same CPU cores.

The default grid, fixed with the user on 27 October 2026 before any evaluation run:

- the main systems ``manual``, ``flaml``, ``autogluon``, ``autopilot_rules`` and ``autopilot_full`` at
  total budgets of 120 and 300 s;
- the ablations ``random_choice``, ``no_critic`` and ``no_resource_manager`` at 300 s only (``no_memory``
  is the same Session as ``autopilot_rules``, so it is not run twice);
- the 6 evaluation datasets of ``research/datasets.json`` and seeds 1 to 5.

That is 300 main cells and 90 ablation cells. design.md planned a third budget of 600 s. It was dropped to
fit the laptop: it would add about 20 hours (4 systems that use their whole budget x 600 s x 30 pairs of
dataset and seed). ``budgets`` can still include it.

Each finished cell is appended at once as one CSV row: the test score and every test metric, the wall
time, the number of Experiments, the CPU cores, the system's settings, the code version and the SHA-256 of
the frozen Experiment_Memory. A restart skips cells that have a row; an interrupted cell has none, so it
runs again. Cells run seed by seed, so a run stopped early still has complete seeds. A system that fails
gives a row with status ``failed``; with ``retry_failed`` such cells run again, and the analysis uses the
last row of each cell.

The smoke setting runs every system once on one small development dataset (titanic), with seed 1 and the
smallest budget, in its own folders.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from app.core.config import Settings, build_settings
from app.services.autopilot import CODE_ROOT, code_version
from app.services.evaluator import METRICS
from app.services.experiment_memory import ExperimentMemory, dataset_fingerprint
from app.services.task_detector import TaskType
from research import datasets as dataset_registry
from research.systems import AUTOPILOT_VARIANTS, AutoGluonSystem, Problem, System, SystemResult, make_system, prepare

MAIN_SYSTEMS = ("manual", "flaml", "autogluon", "autopilot_rules", "autopilot_full")
ABLATIONS = ("random_choice", "no_critic", "no_resource_manager")
BUDGETS = (120, 300)
ABLATION_BUDGET = 300
SEEDS = (1, 2, 3, 4, 5)
SMOKE_DATASET = "titanic"
SMOKE_SEED = 1
MEMORY_SYSTEMS = tuple(name for name, options in AUTOPILOT_VARIANTS.items() if options.get("use_memory"))
ROWS_FILE = "benchmark_cells.csv"
EXCLUSIONS_FILE = "excluded_cells.csv"  # rows that must not be scored, with the reason (next to the rows file)
EXCLUSION_COLUMNS = ("cell_id", "finished_at", "reason")
POWER_POLL_SECONDS = 30

COLUMNS = (
    "cell_id", "system", "dataset", "task_type", "metric", "budget_seconds", "seed", "status", "test_score",
    *(f"test_{name}" for name in METRICS),
    "wall_seconds", "n_experiments", "model", "stop_reason", "first_test_score", "kept", "run_id",
    "n_train", "n_test", "excluded_columns", "split", "n_jobs", "cpu_count", "system_settings", "details",
    "memory_file", "memory_hash", "memory_records", "git_commit", "git_dirty", "versions", "error",
    "started_at", "finished_at",
)
NUMBER_COLUMNS = {"budget_seconds", "seed", "test_score", *(f"test_{name}" for name in METRICS), "wall_seconds",
                  "n_experiments", "first_test_score", "n_train", "n_test", "n_jobs", "cpu_count", "memory_records"}


@dataclass(frozen=True)
class Cell:
    system: str
    dataset: str
    budget_seconds: int
    seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.system}-{self.dataset}-{self.budget_seconds}s-s{self.seed}"


@dataclass(frozen=True)
class Dataset:
    name: str
    frame: pd.DataFrame
    target: str
    task_type: TaskType  # OpenML's task, from the registry (two regression targets look multiclass to the app)


# --------------------------------------------------------------------------- the grid
def grid(datasets: Iterable[str], seeds: Iterable[int] = SEEDS, budgets: Iterable[int] = BUDGETS,
         systems: Iterable[str] = MAIN_SYSTEMS, ablations: Iterable[str] = ABLATIONS,
         ablation_budget: int = ABLATION_BUDGET) -> list[Cell]:
    """Seed by seed, then dataset by dataset and budget by budget; the ablations only at ``ablation_budget``."""
    datasets, budgets = list(dict.fromkeys(datasets)), sorted(set(budgets))
    systems, ablations = list(dict.fromkeys(systems)), list(dict.fromkeys(ablations))
    cells = []
    for seed in dict.fromkeys(seeds):
        for dataset in datasets:
            for budget in budgets:
                names = systems + [a for a in ablations if a not in systems] if budget == ablation_budget else systems
                cells += [Cell(name, dataset, budget, seed) for name in names]
    return cells


def smoke_grid(datasets: Iterable[str] = (SMOKE_DATASET,), budgets: Iterable[int] = BUDGETS,
               systems: Iterable[str] = MAIN_SYSTEMS, ablations: Iterable[str] = ABLATIONS) -> list[Cell]:
    """Every system once: the smoke dataset, seed 1 and the smallest budget."""
    names = list(dict.fromkeys([*systems, *ablations]))
    return [Cell(name, dataset, min(budgets), SMOKE_SEED) for dataset in dict.fromkeys(datasets) for name in names]


LABEL = re.compile(r"[a-z0-9][a-z0-9-]{0,30}")


def run_name(label: str = "") -> str:
    """``benchmark``, or ``benchmark-<label>`` for a later run (for example ``v2``), so a new run never writes
    into the folders of an earlier one."""
    if label and not LABEL.fullmatch(label):
        raise ValueError(f"A run label is lower-case letters, digits and '-', at most 31 characters: {label!r}")
    return f"benchmark-{label}" if label else "benchmark"


def folders(project_root: Path, smoke: bool = False, label: str = "") -> tuple[Settings, Path]:
    """The settings the systems write their records with, and the CSV file of finished cells.

    Records go to ``experiments/benchmark/`` and ``models/benchmark/`` (``benchmark-smoke`` for the smoke
    setting), apart from the dashboard's history; rows go to ``results/benchmark/`` (``results/benchmark/smoke/``).
    With a ``label`` (12 November 2026), every ``benchmark`` in these names becomes ``benchmark-<label>``.
    """
    run = run_name(label)
    name = f"{run}-smoke" if smoke else run
    base = build_settings(Path(project_root))
    settings = replace(base, experiments_dir=base.experiments_dir / name, models_dir=base.models_dir / name)
    results = base.project_root / "results" / run
    return settings, (results / "smoke" if smoke else results) / ROWS_FILE


def load_dataset(name: str) -> Dataset:
    entry = dataset_registry.registry()[name]
    return Dataset(name, dataset_registry.load(entry), entry.target, TaskType(entry.task))


# --------------------------------------------------------------------------- checks before a run
def preflight(cells: Sequence[Cell], memory_path: Path | None, *, require_memory: bool,
              evaluation_only: bool) -> list[str]:
    """Problems that would spoil a run, found before it starts (an empty list means none)."""
    problems = []
    entries = dataset_registry.registry()
    names = list(dict.fromkeys(c.dataset for c in cells))
    for name in names:
        entry = entries.get(name)
        if entry is None:
            problems.append(f"{name} is not in research/datasets.json.")
        elif evaluation_only and entry.split != "evaluation":
            problems.append(f"{name} is a {entry.split} dataset; the benchmark grid uses evaluation datasets only "
                            "(use the smoke setting for a trial on other datasets).")
        elif not (dataset_registry.ROOT / entry.file).exists():
            problems.append(f"{entry.file} is missing: run python scripts/prepare_datasets.py download")
    if any(c.system == "autogluon" for c in cells) and not AutoGluonSystem().python.exists():
        problems.append(f"The AutoGluon Python {AutoGluonSystem().python} does not exist (decision D4; set "
                        "AUTOPILOT_AUTOGLUON_PYTHON).")
    readers = sorted({c.system for c in cells if c.system in MEMORY_SYSTEMS})
    if memory_path is None:
        if require_memory and readers:
            problems.append(f"{', '.join(readers)} read the frozen Experiment_Memory: give its file with --memory.")
    elif not Path(memory_path).exists():
        problems.append(f"The memory file {memory_path} does not exist.")
    else:
        present = {split: [load_dataset(e.name) for e in entries.values()
                           if e.split == split and (dataset_registry.ROOT / e.file).exists()]
                   for split in ("development", "evaluation")}
        from_evaluation, unknown = memory_sources(Path(memory_path), present["development"], present["evaluation"])
        if from_evaluation:
            problems.append(f"{len(from_evaluation)} memory records come from evaluation datasets (Requirement "
                            f"22.3), for example {from_evaluation[0]}.")
        if unknown:
            problems.append(f"{len(unknown)} memory records come from datasets that are not development datasets "
                            f"(Requirement 22.3), for example {unknown[0]}.")
    return problems


def memory_sources(memory_path: Path, development: Iterable[Dataset],
                   evaluation: Iterable[Dataset]) -> tuple[list[str], list[str]]:
    """IDs of the memory records made on an evaluation dataset, and of those made on no development dataset.
    A record's dataset is matched by name or by its dataset fingerprint."""
    def keys(group: Iterable[Dataset]) -> tuple[set[str], set[str]]:
        group = list(group)
        return {d.name for d in group}, {dataset_fingerprint(d.frame, d.target) for d in group}

    (dev_names, dev_prints), (eval_names, eval_prints) = keys(development), keys(evaluation)
    from_evaluation, unknown = [], []
    for record in ExperimentMemory(memory_path, read_only=True).records():
        if record.dataset_name in eval_names or record.dataset_fingerprint in eval_prints:
            from_evaluation.append(record.record_id)
        elif record.dataset_name not in dev_names and record.dataset_fingerprint not in dev_prints:
            unknown.append(record.record_id)
    return from_evaluation, unknown


def read_cells(path: Path) -> list[dict[str, Any]]:
    """The rows of a rows file, in the order written; empty cells are None. Columns other than the numbers
    stay text, so that a commit such as ``1e84091`` or a hash is not read as a number."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path, dtype={c: str for c in COLUMNS if c not in NUMBER_COLUMNS})
    rows = []
    for record in frame.to_dict("records"):
        row = {k: (None if isinstance(v, float) and v != v else v) for k, v in record.items()}
        for key in ("budget_seconds", "seed"):
            row[key] = int(row[key])
        rows.append(row)
    return rows


def read_exclusions(path: Path) -> dict[tuple[str, str], str]:
    """(cell_id, finished_at) of each excluded row, with its reason. Rows are never edited; a row that must not
    be scored (for example one measured while the laptop was on battery) is listed here instead."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    return {(r["cell_id"], r["finished_at"]): r["reason"] for r in frame.to_dict("records")}


def add_exclusion(path: Path, row: dict[str, Any], reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    line = {"cell_id": row["cell_id"], "finished_at": str(row["finished_at"]), "reason": reason}
    text = pd.DataFrame([line], columns=list(EXCLUSION_COLUMNS)).to_csv(index=False, header=header,
                                                                         lineterminator="\n")
    with path.open("a", encoding="utf-8", newline="") as file:
        file.write(text)


def usable(rows: Iterable[dict[str, Any]], exclusions: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    """The rows that are not excluded; an excluded cell counts as not run, so it runs again."""
    return [r for r in rows if (r["cell_id"], str(r["finished_at"])) not in exclusions]


def on_battery() -> bool | None:
    """True on battery, False on mains power, None when unknown (not Windows, or no answer)."""
    if os.name != "nt":
        return None
    import ctypes

    class PowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    status = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return {0: True, 1: False}.get(status.ACLineStatus)


def wait_for_power(power: Callable[[], bool | None] = on_battery, sleep: Callable[[float], None] = time.sleep,
                   on_wait: Callable[[], None] | None = None, poll_seconds: float = POWER_POLL_SECONDS) -> float:
    """Wait while the laptop runs on battery (Windows slows it down and puts it into standby there, which
    would cut time-limited searches short); returns the seconds waited."""
    waited = 0.0
    while power() is True:
        if waited == 0 and on_wait is not None:
            on_wait()
        sleep(poll_seconds)
        waited += poll_seconds
    return waited


def check_rows_file(path: Path) -> None:
    """Refuse a rows file whose header is not ``COLUMNS`` or whose last line was cut off."""
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise ValueError(f"The last line of {path} is incomplete (the run stopped while writing it). Remove that "
                         "line by hand and start again; the cell will run again.")
    header = data.split(b"\n", 1)[0].decode("utf-8").strip().split(",")
    if header != list(COLUMNS):
        raise ValueError(f"{path} has other columns than this version of the harness writes; use a new folder.")


# --------------------------------------------------------------------------- running
def pending(cells: Sequence[Cell], rows: Sequence[dict[str, Any]], retry_failed: bool = False) -> list[Cell]:
    """The cells without a row; with ``retry_failed`` also those whose last row failed."""
    last = latest(rows)
    return [c for c in dict.fromkeys(cells)
            if c.cell_id not in last or (retry_failed and last[c.cell_id].get("status") != "completed")]


def latest(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The last row of each cell."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        out[row["cell_id"]] = row
    return out


def run(cells: Sequence[Cell], settings: Settings, rows_path: Path, *, memory_path: Path | None = None,
        n_jobs: int = -1, make: Callable[[str], System] | None = None,
        load: Callable[[str], Dataset] = load_dataset, retry_failed: bool = False, limit: int = 0,
        report: Callable[[dict[str, Any], int, int, float], None] | None = None,
        power: Callable[[], bool | None] | None = None, sleep: Callable[[float], None] = time.sleep,
        on_wait: Callable[[], None] | None = None) -> list[dict[str, Any]]:
    """Run the cells that have no usable row yet, each written as one row when it finishes; returns the new rows.

    ``report(row, number, total, seconds_left)`` is called after each cell; ``seconds_left`` is an estimate
    from the wall times measured so far. With ``power`` (for example ``on_battery``; the command line passes
    it), the run waits before each cell while the laptop is on battery, and a cell that ends on battery is
    written and also listed in the exclusions file, so it is not scored and runs again.
    """
    check_rows_file(rows_path)
    exclusions_path = rows_path.with_name(EXCLUSIONS_FILE)
    all_rows = read_cells(rows_path)
    rows = usable(all_rows, read_exclusions(exclusions_path))
    memory_file = Path(memory_path) if memory_path else settings.experiments_dir / "memory" / "empty.jsonl"
    memory = ExperimentMemory(memory_file, read_only=True)
    memory_hash = memory.snapshot_hash()
    earlier = {r["memory_hash"] for r in all_rows if r.get("memory_hash")} - {memory_hash}
    if earlier:
        raise ValueError(f"The rows in {rows_path} were made with another memory (SHA-256 {sorted(earlier)[0]}); "
                         f"this one is {memory_hash}. Use the same frozen memory, or a new folder.")
    make = make or (lambda name: make_system(name, Path(memory_path) if memory_path else None))
    context = {
        "memory_file": _relative(memory_file), "memory_hash": memory_hash, "memory_records": len(memory.records()),
        "version": code_version(CODE_ROOT),
    }
    todo = pending(cells, rows, retry_failed)[: limit or None]
    settings.experiments_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, Dataset] = {}
    written: list[dict[str, Any]] = []
    for number, cell in enumerate(todo, start=1):
        if cell.dataset not in loaded:
            loaded[cell.dataset] = load(cell.dataset)
        waited = wait_for_power(power, sleep, on_wait) if power is not None else 0.0
        row = run_cell(cell, loaded[cell.dataset], make(cell.system), settings, n_jobs=n_jobs, context=context)
        if memory.snapshot_hash() != memory_hash:  # checked before the row is written, so this cell runs again
            raise RuntimeError(f"The memory file {memory_file} changed during the run; it must stay frozen.")
        ended_on_battery = power is not None and power() is True
        row["details"] = json.dumps({**json.loads(row["details"]), "harness": {
            "waited_for_power_seconds": waited, "on_battery_at_end": ended_on_battery}}, sort_keys=True, default=str)
        append_row(rows_path, row)
        if ended_on_battery:
            add_exclusion(exclusions_path, row, "the laptop was on battery at the end of the cell")
        written.append(row)
        if report is not None:
            report(row, number, len(todo), estimate_seconds(todo[number:], [*rows, *written]))
    return written


def run_cell(cell: Cell, dataset: Dataset, system: System, settings: Settings, *, n_jobs: int,
             context: dict[str, Any]) -> dict[str, Any]:
    """One cell as one row. A failure is a row with status ``failed``, never an exception."""
    started_at = _now()
    started = time.perf_counter()
    problem = Problem(cell.dataset, dataset.frame, dataset.target, dataset.task_type, cell.seed, cell.budget_seconds,
                      n_jobs)
    prepared = None
    try:
        prepared = prepare(problem, settings)
        result = system.run(problem, prepared, settings)
    except Exception as exc:  # the systems return their failures; this catches a failing split
        result = SystemResult(cell.system, "failed", None, {}, round(time.perf_counter() - started, 2), 0, None,
                              error=f"{type(exc).__name__}: {exc}")
    details = dict(result.details or {})
    version = context["version"]
    describe = getattr(system, "describe", None)
    return {
        "cell_id": cell.cell_id, "system": cell.system, "dataset": cell.dataset,
        "task_type": problem.task_type.value, "metric": problem.primary_metric,
        "budget_seconds": cell.budget_seconds, "seed": cell.seed, "status": result.status,
        "test_score": result.test_score,
        **{f"test_{name}": (result.test_metrics or {}).get(name) for name in METRICS},
        "wall_seconds": result.wall_seconds, "n_experiments": result.n_experiments, "model": result.model,
        "stop_reason": details.get("stop_reason"), "first_test_score": details.get("first_test_score"),
        "kept": "; ".join(details.get("kept") or []),
        "run_id": details.get("session_id") or details.get("experiment_id"),
        "n_train": len(prepared.y_train) if prepared else None, "n_test": len(prepared.y_test) if prepared else None,
        "excluded_columns": json.dumps(prepared.excluded_columns) if prepared else None,
        "split": prepared.split.fingerprint() if prepared else None,
        "n_jobs": n_jobs, "cpu_count": os.cpu_count(),
        "system_settings": json.dumps(describe() if describe else {}, sort_keys=True, default=str),
        "details": json.dumps(details, sort_keys=True, default=str),
        "memory_file": context["memory_file"], "memory_hash": context["memory_hash"],
        "memory_records": context["memory_records"],
        "git_commit": (version.get("git_commit") or "")[:7] or None, "git_dirty": version.get("git_dirty"),
        "versions": json.dumps({k: v for k, v in version.items() if not k.startswith("git_")}, sort_keys=True),
        "error": result.error, "started_at": started_at, "finished_at": _now(),
    }


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one row and force it to disk, so that a crash cannot lose a finished cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    text = pd.DataFrame([row], columns=list(COLUMNS)).to_csv(index=False, header=header, lineterminator="\n")
    with path.open("a", encoding="utf-8", newline="") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())


# --------------------------------------------------------------------------- progress
def estimate_seconds(cells: Sequence[Cell], rows: Sequence[dict[str, Any]]) -> float:
    """Time the cells will take: the mean wall time of finished cells of the same system and budget, or the
    budget itself while there are none. An estimate, not a measurement."""
    walls: dict[tuple[str, int], list[float]] = {}
    for row in latest(rows).values():
        if row.get("wall_seconds") is not None:
            walls.setdefault((row["system"], int(row["budget_seconds"])), []).append(float(row["wall_seconds"]))
    means = {key: sum(values) / len(values) for key, values in walls.items()}
    return float(sum(means.get((c.system, c.budget_seconds), c.budget_seconds) for c in cells))


def timing(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per system and budget: cells finished, completed and failed, and the wall time (mean, largest, and the
    mean time above the budget)."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in latest(rows).values():
        groups.setdefault((row["system"], int(row["budget_seconds"])), []).append(row)
    out = []
    for (system, budget), group in sorted(groups.items(), key=lambda item: (item[0][1], _order(item[0][0]))):
        walls = [float(r["wall_seconds"]) for r in group if r.get("wall_seconds") is not None]
        out.append({
            "system": system, "budget_seconds": budget, "cells": len(group),
            "completed": sum(r.get("status") == "completed" for r in group),
            "failed": sum(r.get("status") != "completed" for r in group),
            "mean_wall_seconds": round(sum(walls) / len(walls), 1) if walls else None,
            "max_wall_seconds": round(max(walls), 1) if walls else None,
            "mean_seconds_above_budget": round(sum(walls) / len(walls) - budget, 1) if walls else None,
        })
    return out


def _order(system: str) -> int:
    names = [*MAIN_SYSTEMS, *ABLATIONS]
    return names.index(system) if system in names else len(names)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(CODE_ROOT).as_posix()
    except ValueError:
        return str(path)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


__all__ = ["ABLATIONS", "BUDGETS", "COLUMNS", "Cell", "Dataset", "EXCLUSIONS_FILE", "MAIN_SYSTEMS", "SEEDS",
           "add_exclusion", "folders", "grid", "latest", "load_dataset", "on_battery", "pending", "preflight",
           "read_cells", "read_exclusions", "run", "run_name", "smoke_grid", "timing", "usable", "wait_for_power"]
