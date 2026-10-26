"""Phase 2 measurement (spec task 19, Requirement 16.2): the rule-based loop against FLAML alone.

For every development dataset available (the four bundled samples) and every seed, two runs with the
same total time budget, on the same Training_Part and Test_Part (the Session's outer split is the
Phase 1 split, Requirement 2.2):

- ``session``: one AutopilotSession with the default settings (rule selector, 3 x 5 folds, Judge alpha
  0.20, the current Critic thresholds). Its score is the final model's test score from the
  Final_Evaluation; the first model's test score is kept too.
- ``flaml``: one FLAML search with the whole budget (the Phase 1 runner). It leaves out the same columns
  as the Session's first Setup (the identifier-like and constant columns of the Training_Part), so the
  two runs differ only in the loop.

Decisions fixed before running: 300 s per run (the middle budget of the Phase 3 grid, as in SES-001), at
most 8 Experiments per Session, seeds 1, 2 and 3, test share 0.2, each task's default primary metric
(F1, macro F1, R2), all CPU cores, MLflow on, and the records in the project folders, because they are
the raw results. Every finished run is written at once as one CSV row, and a restart skips finished runs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from app.core.config import Settings
from app.services.automl_runner import AutoMLEngine, AutoMLRunner, ExperimentConfig
from app.services.autopilot import CODE_ROOT, AutopilotSession, SessionSettings, code_version
from app.services.evaluator import METRICS
from app.services.resource_manager import SessionBudget

SAMPLES = CODE_ROOT / "data" / "sample"
DATASETS: dict[str, tuple[str, str]] = {  # name: (file in data/sample, target)
    "titanic": ("titanic.csv", "Survived"),
    "breast_cancer": ("breast_cancer.csv", "diagnosis"),
    "penguins": ("penguins.csv", "species"),
    "california_housing": ("california_housing.csv", "MedHouseVal"),
}
SEEDS = (1, 2, 3)
SECONDS = 300
MAX_EXPERIMENTS = 8
TEST_SIZE = 0.2
TIE = 1e-12


@dataclass(frozen=True)
class Cell:
    system: str  # "session" or "flaml"
    dataset: str
    seed: int
    seconds: int

    @property
    def cell_id(self) -> str:
        return f"{self.system}-{self.dataset}-s{self.seed}-{self.seconds}s"


def cells(datasets: Iterable[str], seeds: Iterable[int], seconds: int) -> list[Cell]:
    """Seed by seed and dataset by dataset; the Session first, since the FLAML run uses its exclusions."""
    return [Cell(system, dataset, seed, seconds) for seed in seeds for dataset in datasets
            for system in ("session", "flaml")]


def load(dataset: str) -> tuple[pd.DataFrame, str]:
    file_name, target = DATASETS[dataset]
    return pd.read_csv(SAMPLES / file_name), target


def run_session(cell: Cell, settings: Settings, *, n_jobs: int = -1, enable_mlflow: bool = True,
                engine: AutoMLEngine | None = None, max_experiments: int = MAX_EXPERIMENTS) -> dict[str, Any]:
    df, target = load(cell.dataset)
    plan = SessionSettings(
        base=ExperimentConfig(target_column=target, random_seed=cell.seed, test_size=TEST_SIZE,
                              dataset_name=cell.dataset),
        budget=SessionBudget(total_seconds=cell.seconds, max_experiments=max_experiments, n_jobs=n_jobs),
    )
    started = time.perf_counter()
    result = AutopilotSession(settings=settings, engine=engine, enable_mlflow=enable_mlflow).run(df, plan)
    wall = time.perf_counter() - started
    record = json.loads((settings.experiments_dir / result.session_id / "session.json").read_text(encoding="utf-8"))
    base, steps = record["settings"]["base"], record["steps"]
    final = record.get("final_evaluation") or {}
    first = _summary(settings, record.get("first_experiment_id"))
    last = _summary(settings, record.get("final_experiment_id"))
    candidates = [s for s in steps if s["index"] > 0]
    return {
        **_common(cell, base["task_type"], base["primary_metric"]),
        "status": record["status"],
        "test_score": (final.get("final") or {}).get("test_score"),
        "first_test_score": (final.get("first") or {}).get("test_score"),
        "wall_seconds": round(wall, 2),
        "n_experiments": len(steps),
        "stop_reason": record["stop_reason"],
        "kept": "; ".join(strategy_text(s) for s in candidates if s.get("accepted")),
        "tried": "; ".join(f"{strategy_text(s)}: {verdict_text(s)}" for s in candidates),
        "excluded_columns": json.dumps(first.get("excluded_columns") or []),
        "model": last.get("best_model_name"),
        "first_model": first.get("best_model_name"),
        "n_trials": None,
        "run_id": result.session_id,
        "error": record.get("error"),
    }


def run_flaml(cell: Cell, settings: Settings, excluded: list[str], *, n_jobs: int = -1,
              enable_mlflow: bool = True, engine: AutoMLEngine | None = None) -> dict[str, Any]:
    df, target = load(cell.dataset)
    config = ExperimentConfig(target_column=target, time_budget=cell.seconds, random_seed=cell.seed,
                              test_size=TEST_SIZE, excluded_columns=list(excluded), dataset_name=cell.dataset,
                              n_jobs=n_jobs)
    started = time.perf_counter()
    result = AutoMLRunner(settings=settings, engine=engine, enable_mlflow=enable_mlflow).run_experiment(df, config)
    wall = time.perf_counter() - started
    return {
        **_common(cell, result.task_type.value, result.primary_metric),
        "status": result.status, "test_score": result.test_score, "first_test_score": None,
        "wall_seconds": round(wall, 2), "n_experiments": 1, "stop_reason": None, "kept": "", "tried": "",
        "excluded_columns": json.dumps(list(excluded)), "model": result.best_model_name, "first_model": None,
        "n_trials": result.n_trials, "run_id": result.experiment_id, "error": None,
    }


def strategy_text(step: dict[str, Any]) -> str:
    spec = step["proposal"]["spec"]
    params = ", ".join(f"{k}={'+'.join(map(str, v)) if isinstance(v, list) else v}"
                       for k, v in sorted((spec.get("params") or {}).items()))
    return f"{spec['strategy_id']}({params})" if params else spec["strategy_id"]


def verdict_text(step: dict[str, Any]) -> str:
    return "failed" if step["status"] == "failed" else (step.get("decision") or {}).get("verdict", "n/a")


# --------------------------------------------------------------------------- analysis
def pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per dataset and seed with both runs. Differences are signed so that positive means the
    Session did better (for error metrics the sign is flipped)."""
    by_key = {(r["system"], r["dataset"], int(r["seed"]), int(r["budget_seconds"])): r for r in rows}
    out = []
    for (system, dataset, seed, budget), s in sorted(by_key.items()):
        f = by_key.get(("flaml", dataset, seed, budget))
        if system != "session" or f is None:
            continue
        sign = 1.0 if METRICS[s["metric"]].greater_is_better else -1.0
        out.append({
            "dataset": dataset, "seed": seed, "budget_seconds": budget, "metric": s["metric"],
            "session_test": s["test_score"], "flaml_test": f["test_score"],
            "difference": _signed(sign, s["test_score"], f["test_score"]),
            "session_first_test": s["first_test_score"],
            "loop_gain": _signed(sign, s["test_score"], s["first_test_score"]),
            "session_seconds": s["wall_seconds"], "flaml_seconds": f["wall_seconds"],
            "session_experiments": s["n_experiments"], "stop_reason": s["stop_reason"], "kept": s["kept"] or "",
            "session_model": s["model"], "flaml_model": f["model"], "flaml_trials": f["n_trials"],
            "session_id": s["run_id"], "flaml_experiment_id": f["run_id"],
        })
    return out


def summary(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per dataset and over all pairs: mean difference, wins, ties and losses of the Session, and costs."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        groups.setdefault(row["dataset"], []).append(row)
    out = []
    for name, group in [*sorted(groups.items()), ("all", pair_rows)]:
        diffs = [r["difference"] for r in group if r["difference"] is not None]
        gains = [r["loop_gain"] for r in group if r["loop_gain"] is not None]
        out.append({
            "dataset": name, "pairs": len(group),
            "mean_difference": _mean(diffs), "min_difference": min(diffs, default=None),
            "max_difference": max(diffs, default=None),
            "session_wins": sum(d > TIE for d in diffs), "ties": sum(abs(d) <= TIE for d in diffs),
            "flaml_wins": sum(d < -TIE for d in diffs),
            "mean_loop_gain": _mean(gains),
            "mean_session_seconds": _mean([r["session_seconds"] for r in group]),
            "mean_flaml_seconds": _mean([r["flaml_seconds"] for r in group]),
            "mean_session_experiments": _mean([r["session_experiments"] for r in group]),
        })
    return out


def strategy_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """How often each Strategy was tried and kept, per dataset, over the Sessions."""
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        if row["system"] != "session" or not row.get("tried"):
            continue
        for item in str(row["tried"]).split("; "):
            strategy, _, verdict = item.rpartition(": ")
            entry = counts.setdefault((row["dataset"], strategy.split("(")[0]),
                                      {"tried": 0, "keep": 0, "keep_simpler": 0, "discard": 0, "failed": 0})
            entry["tried"] += 1
            entry[verdict] = entry.get(verdict, 0) + 1
    return [{"dataset": d, "strategy": s, **c} for (d, s), c in sorted(counts.items())]


# --------------------------------------------------------------------------- helpers
def _common(cell: Cell, task_type: str, metric: str) -> dict[str, Any]:
    version = code_version(CODE_ROOT)
    return {
        "cell_id": cell.cell_id, "system": cell.system, "dataset": cell.dataset, "seed": cell.seed,
        "budget_seconds": cell.seconds, "task_type": task_type, "metric": metric,
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git_commit": (version.get("git_commit") or "")[:7], "git_dirty": version.get("git_dirty"),
    }


def _summary(settings: Settings, experiment_id: str | None) -> dict[str, Any]:
    if not experiment_id:
        return {}
    path = settings.experiments_dir / experiment_id / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _signed(sign: float, a: float | None, b: float | None) -> float | None:
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return None
    return sign * (float(a) - float(b))


def _mean(values: list[Any]) -> float | None:
    values = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else None
