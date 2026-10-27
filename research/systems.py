"""Comparison systems for the Phase 3 benchmark (spec task 23, Requirement 21).

Every system gets the same Training_Part and Test_Part for a dataset and seed (the Phase 1 split,
``make_outer_split``), the same columns (the identifier-like and constant columns of the Training_Part
left out, as in a Session's first Setup), the same time budget and CPU cores, and is scored by the same
``Evaluator`` on the same test rows, so all systems report the same metrics (Requirement 21.4).

- ``manual``: LightGBM with its library default settings (100 trees, 31 leaves, learning rate 0.1, ...),
  trained once through ``FLAMLEngine.fit_config`` (FLAML's data preparation, no search).
- ``flaml``: one FLAML search with the whole budget (the Phase 1 runner).
- ``autogluon``: AutoGluon-Tabular 1.6.3 with its default preset, CPU only, in its own conda environment
  (decision D4: it needs pandas 2.3, the project pins pandas 3.0), called as a separate process.
- AutoPilot variants: a Session with the settings of ``AUTOPILOT_VARIANTS``, reading a frozen,
  read-only Experiment_Memory. ``no_memory`` (the full system without the memory) is the same Session
  as ``autopilot_rules``: without records the learned selector keeps the rule order.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from app.core.config import Settings
from app.services.automl_runner import AutoMLEngine, AutoMLRunner, ExperimentConfig, FLAMLEngine
from app.services.autopilot import AutopilotSession, SessionSettings
from app.services.evaluator import Evaluator, default_primary_metric
from app.services.experiment_memory import ExperimentMemory
from app.services.experiment_planner import Planner
from app.services.profiler import DatasetProfiler
from app.services.resource_manager import SessionBudget
from app.services.splits import OuterSplit, make_outer_split
from app.services.task_detector import TaskType

TEST_SIZE = 0.2
MAX_EXPERIMENTS = 8
# LightGBM's library defaults, in FLAML's names (max_bin 255 = 2^8 - 1). Checked against
# lightgbm.LGBMClassifier().get_params() on 27 October 2026.
LIGHTGBM_DEFAULTS = {"n_estimators": 100, "num_leaves": 31, "min_child_samples": 20, "learning_rate": 0.1,
                     "log_max_bin": 8, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0}
AUTOGLUON_PRESET = "medium_quality"
AUTOGLUON_METRICS = {  # (task, our metric) -> AutoGluon's eval_metric
    ("binary", "f1"): "f1", ("binary", "accuracy"): "accuracy", ("binary", "roc_auc"): "roc_auc",
    ("binary", "log_loss"): "log_loss", ("multiclass", "f1"): "f1_macro", ("multiclass", "accuracy"): "accuracy",
    ("multiclass", "log_loss"): "log_loss", ("regression", "r2"): "r2",
    ("regression", "rmse"): "root_mean_squared_error", ("regression", "mae"): "mean_absolute_error",
    ("regression", "mse"): "mean_squared_error",
}
AUTOPILOT_VARIANTS: dict[str, dict[str, Any]] = {
    "autopilot_rules": {"selector": "rules", "use_memory": False},
    "autopilot_full": {"selector": "thompson", "use_memory": True},
    "random_choice": {"selector": "random", "use_memory": False},
    "no_critic": {"selector": "thompson", "use_memory": True, "use_critic": False},
    "no_resource_manager": {"selector": "thompson", "use_memory": True, "use_resource_manager": False},
}


@dataclass(frozen=True)
class Problem:
    """One benchmark cell, as every system sees it."""

    dataset: str
    frame: pd.DataFrame
    target: str
    task_type: TaskType
    seed: int
    budget_seconds: int
    n_jobs: int = -1
    metric: str | None = None  # None: the task's default primary metric

    @property
    def primary_metric(self) -> str:
        return self.metric or default_primary_metric(self.task_type)


@dataclass
class Prepared:
    """The shared split and columns of a Problem."""

    split: OuterSplit
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    excluded_columns: list[str]


@dataclass
class SystemResult:
    system: str
    status: str  # "completed" or "failed"
    test_score: float | None
    test_metrics: dict[str, float | None]
    wall_seconds: float
    n_experiments: int
    model: str | None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class System(Protocol):
    name: str

    def run(self, problem: Problem, prepared: Prepared, settings: Settings) -> SystemResult: ...


def base_config(problem: Problem, excluded: list[str] | None = None, time_budget: int = 60) -> ExperimentConfig:
    return ExperimentConfig(target_column=problem.target, task_type=problem.task_type,
                            primary_metric=problem.primary_metric, time_budget=time_budget,
                            random_seed=problem.seed, test_size=TEST_SIZE, excluded_columns=list(excluded or []),
                            dataset_name=problem.dataset, n_jobs=problem.n_jobs)


def prepare(problem: Problem, settings: Settings) -> Prepared:
    """The split and columns every system uses: the Phase 1 split of the labelled rows, and the columns of a
    Session's first Setup (identifier-like and constant columns of the Training_Part left out)."""
    config = base_config(problem)
    _, y, task_type, _, _ = AutoMLRunner(settings=settings, enable_mlflow=False).prepare(problem.frame, config)
    split, _ = make_outer_split(y, task_type, TEST_SIZE, problem.seed)
    labelled = problem.frame.loc[y.index]
    train, test = labelled.iloc[split.train_positions], labelled.iloc[split.test_positions]
    profile = DatasetProfiler().profile(train, problem.target, problem.dataset, task_type)
    excluded = Planner.safe_exclusions(config, profile)
    drop = [problem.target, *excluded]
    return Prepared(split=split, X_train=train.drop(columns=drop), y_train=y.iloc[split.train_positions],
                    X_test=test.drop(columns=drop), y_test=y.iloc[split.test_positions], excluded_columns=excluded)


def _score(problem: Problem, prepared: Prepared, model: Any, classes: list[Any] | None = None) -> tuple[float | None, dict]:
    evaluation = Evaluator(problem.task_type, problem.primary_metric).evaluate(model, prepared.X_test,
                                                                               prepared.y_test, classes)
    return evaluation.primary_score, dict(evaluation.metrics)


def _failed(name: str, started: float, exc: Exception) -> SystemResult:
    return SystemResult(name, "failed", None, {}, round(time.perf_counter() - started, 2), 0, None,
                        error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- the systems
class ManualSystem:
    """LightGBM with its library default settings, no search (Requirement 21.1)."""

    name = "manual"

    def __init__(self, engine: AutoMLEngine | None = None) -> None:
        self.engine = engine or FLAMLEngine()

    def run(self, problem: Problem, prepared: Prepared, settings: Settings) -> SystemResult:
        started = time.perf_counter()
        try:
            model = self.engine.fit_config(prepared.X_train, prepared.y_train, task_type=problem.task_type,
                                           estimator="lgbm", config=dict(LIGHTGBM_DEFAULTS), seed=problem.seed,
                                           n_jobs=problem.n_jobs)
            score, metrics = _score(problem, prepared, model)
        except Exception as exc:
            return _failed(self.name, started, exc)
        return SystemResult(self.name, "completed", score, metrics, round(time.perf_counter() - started, 2), 1,
                            "LightGBM (library defaults)", {"config": dict(LIGHTGBM_DEFAULTS)})


class FlamlSystem:
    """One FLAML search with the whole budget on the shared Training_Part (Requirement 21.2)."""

    name = "flaml"

    def __init__(self, engine: AutoMLEngine | None = None) -> None:
        self.engine = engine

    def run(self, problem: Problem, prepared: Prepared, settings: Settings) -> SystemResult:
        started = time.perf_counter()
        try:
            config = base_config(problem, prepared.excluded_columns, problem.budget_seconds)
            result = AutoMLRunner(settings=settings, engine=self.engine, enable_mlflow=False).run_experiment(
                problem.frame, config, split=prepared.split, compute_importance=False)
        except Exception as exc:
            return _failed(self.name, started, exc)
        return SystemResult(self.name, "completed", result.test_score, dict(result.test_metrics),
                            round(time.perf_counter() - started, 2), 1, result.best_model_name,
                            {"experiment_id": result.experiment_id, "n_trials": result.n_trials,
                             "training_time": result.training_time})


class AutoGluonSystem:
    """AutoGluon-Tabular in its own environment, as a separate process (Requirement 21.3, decision D4)."""

    name = "autogluon"

    def __init__(self, python: str | Path | None = None, worker: Path | None = None,
                 extra_seconds: int = 300) -> None:
        self.python = Path(python or os.environ.get("AUTOPILOT_AUTOGLUON_PYTHON") or default_autogluon_python())
        self.worker = worker or Path(__file__).with_name("autogluon_worker.py")
        self.extra_seconds = extra_seconds  # a run that takes this much longer than its budget is stopped

    def run(self, problem: Problem, prepared: Prepared, settings: Settings) -> SystemResult:
        started = time.perf_counter()
        task = problem.task_type.value
        classes = sorted(pd.unique(prepared.y_train).tolist()) if problem.task_type.is_classification else None
        try:
            with tempfile.TemporaryDirectory(prefix="autogluon_") as folder:
                folder_path = Path(folder)
                prepared.X_train.assign(**{problem.target: prepared.y_train.to_numpy()}).to_csv(
                    folder_path / "train.csv", index=False)
                prepared.X_test.to_csv(folder_path / "test.csv", index=False)
                (folder_path / "settings.json").write_text(json.dumps({
                    "label": problem.target, "problem_type": task,
                    "eval_metric": AUTOGLUON_METRICS[(task, problem.primary_metric)],
                    "positive_class": _plain(classes[-1]) if task == "binary" else None,
                    "time_limit": problem.budget_seconds, "presets": AUTOGLUON_PRESET,
                    "num_cpus": "auto" if problem.n_jobs == -1 else problem.n_jobs,
                }), encoding="utf-8")
                completed = subprocess.run([str(self.python), str(self.worker), str(folder_path)], capture_output=True,
                                           text=True, timeout=problem.budget_seconds + self.extra_seconds)
                if completed.returncode != 0:
                    raise RuntimeError(f"the AutoGluon worker failed: {completed.stderr.strip()[-2000:]}")
                predictions = pd.read_csv(folder_path / "predictions.csv")
                info = json.loads((folder_path / "result.json").read_text(encoding="utf-8"))
            model = _FixedPredictions(predictions, classes, prepared.y_test)
            score, metrics = _score(problem, prepared, model, classes)
        except Exception as exc:
            return _failed(self.name, started, exc)
        return SystemResult(self.name, "completed", score, metrics, round(time.perf_counter() - started, 2), 1,
                            f"AutoGluon {info['best_model']}", info)


class AutopilotSystem:
    """A Session with the variant's settings; the memory is opened read-only, so it stays frozen."""

    def __init__(self, name: str, memory_path: Path | None, engine: AutoMLEngine | None = None,
                 max_experiments: int = MAX_EXPERIMENTS, session_factory: Any = AutopilotSession) -> None:
        self.name = name
        self.options = AUTOPILOT_VARIANTS[name]
        self.memory_path = memory_path
        self.engine = engine
        self.max_experiments = max_experiments
        self.session_factory = session_factory

    def run(self, problem: Problem, prepared: Prepared, settings: Settings) -> SystemResult:
        started = time.perf_counter()
        try:
            memory_file = self.memory_path or (settings.experiments_dir / "memory" / "empty.jsonl")
            session = self.session_factory(settings=settings, engine=self.engine, enable_mlflow=False,
                                           memory=ExperimentMemory(memory_file, read_only=True))
            plan = SessionSettings(base=base_config(problem),
                                   budget=SessionBudget(total_seconds=problem.budget_seconds,
                                                        max_experiments=self.max_experiments, n_jobs=problem.n_jobs),
                                   **self.options)
            result = session.run(problem.frame, plan)
            if result.status != "completed" or result.final_evaluation is None:
                raise RuntimeError(result.error or "the Session did not finish")
            final = result.final_evaluation["final"]
        except Exception as exc:
            return _failed(self.name, started, exc)
        kept = [s.proposal["spec"]["strategy_id"] for s in result.steps[1:] if s.accepted and s.proposal]
        return SystemResult(self.name, "completed", final["test_score"], dict(final["test_metrics"]),
                            round(time.perf_counter() - started, 2), len(result.steps), None, {
                                "session_id": result.session_id, "stop_reason": result.stop_reason.value,
                                "first_test_score": result.final_evaluation["first"]["test_score"],
                                "kept": kept, "options": dict(self.options)})


def make_system(name: str, memory_path: Path | None = None, engine: AutoMLEngine | None = None) -> System:
    if name == "manual":
        return ManualSystem(engine)
    if name == "flaml":
        return FlamlSystem(engine)
    if name == "autogluon":
        return AutoGluonSystem()
    if name in AUTOPILOT_VARIANTS:
        return AutopilotSystem(name, memory_path, engine)
    raise ValueError(f"Unknown system '{name}'.")


def default_autogluon_python() -> Path:
    """The ``autogluon`` conda environment next to the current one (``envs/automl`` -> ``envs/autogluon``)."""
    here = Path(sys.executable).resolve()
    name = "python.exe" if os.name == "nt" else "bin/python"
    return here.parent.parent / "autogluon" / name


class _FixedPredictions:
    """A stand-in model that returns predictions made elsewhere, for the shared Evaluator."""

    def __init__(self, predictions: pd.DataFrame, classes: list[Any] | None, y_like: pd.Series) -> None:
        self._predictions = _like(predictions["prediction"], y_like)
        self.classes_ = np.asarray(classes) if classes is not None else None
        columns = [f"proba::{c}" for c in classes] if classes is not None else []
        self._proba = predictions[columns].to_numpy() if columns and all(c in predictions for c in columns) else None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) != len(self._predictions):
            raise ValueError("The predictions do not match the test rows.")
        return self._predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._proba is None:
            raise AttributeError("no probabilities")
        return self._proba


def _like(values: pd.Series, y_like: pd.Series) -> np.ndarray:
    """Predictions read back from CSV, in the label type of the true values."""
    if pd.api.types.is_numeric_dtype(y_like) and not pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values).to_numpy()
    if not pd.api.types.is_numeric_dtype(y_like):
        return values.astype(str).to_numpy()
    return values.to_numpy()


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


__all__ = ["AUTOPILOT_VARIANTS", "AutoGluonSystem", "AutopilotSystem", "FlamlSystem", "ManualSystem", "Prepared",
           "Problem", "System", "SystemResult", "make_system", "prepare"]
