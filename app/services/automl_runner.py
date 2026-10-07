"""AutoML orchestration: split -> engine search -> evaluation -> explainability -> tracking.

The rest of the application talks to ``AutoMLRunner`` and ``ExperimentResult`` only.
FLAML specifics are contained in ``FLAMLEngine``; another engine (e.g. AutoGluon for
the planned baseline comparison) can be added by implementing ``AutoMLEngine``.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
import warnings as py_warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.services.evaluator import (
    METRICS,
    EvaluationResult,
    Evaluator,
    default_primary_metric,
    metric_warnings,
    validate_primary_metric,
)
from app.services.experiment_tracker import ExperimentTracker
from app.services.profiler import DatasetProfile, DatasetProfiler, analyze_class_balance
from app.services.task_detector import TaskDetector, TaskType

logger = get_logger(__name__)

ProgressCallback = Callable[[str, float, str], None]
"""Signature: (stage, fraction_complete, message)."""

MIN_ROWS = 20

# Two kinds of warnings are pure noise here and can flood the terminal:
# - FLAML warns on every predict() call when a categorical column has values it did not see
#   during fit (dozens of times during permutation importance). The runner reports this once
#   per column instead, see `unseen_category_warnings`.
# - ConvergenceWarning from the short linear-model trials FLAML runs on purpose.
# A warnings filter alone is not enough: scikit-learn's joblib wrapper calls
# warnings.resetwarnings() inside worker threads, and since the filter list is shared by the
# whole process this can wipe it in the middle of a search. Dropping these two in
# showwarning() as well survives that reset.
_UNSEEN_CATEGORY_MESSAGE = re.compile(r"Column '.*' contains values unseen at fit time")
py_warnings.filterwarnings("ignore", message=_UNSEEN_CATEGORY_MESSAGE.pattern, category=UserWarning)
_default_showwarning = py_warnings.showwarning


def _is_search_noise(message: Any, category: type) -> bool:
    if issubclass(category, ConvergenceWarning):
        return True
    return issubclass(category, UserWarning) and bool(_UNSEEN_CATEGORY_MESSAGE.match(str(message)))


def _showwarning_without_search_noise(message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
    if not _is_search_noise(message, category):
        _default_showwarning(message, category, filename, lineno, file, line)


py_warnings.showwarning = _showwarning_without_search_noise


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be configured or completed."""


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas / enum values into JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def unseen_category_warnings(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
    """Report categorical columns whose test values never appear in the training split."""
    messages = []
    for column in X_train.columns:
        train_col = X_train[column]
        if is_numeric_dtype(train_col) and not is_bool_dtype(train_col):
            continue
        test_values = X_test[column].dropna()
        unseen = ~test_values.isin(set(train_col.dropna().unique()))
        if len(X_test) and unseen.any():
            messages.append(
                f"'{column}': {unseen.sum() / len(X_test):.0%} of test rows have categories never seen in "
                "training, so the model cannot use them (typical for high-cardinality columns)."
            )
    return messages


# --------------------------------------------------------------------------- config
@dataclass
class ExperimentConfig:
    """Everything needed to reproduce one experiment (future planner output)."""

    target_column: str
    task_type: TaskType | None = None  # None -> detect automatically
    primary_metric: str | None = None  # None -> task default
    time_budget: int = 60
    random_seed: int = 42
    test_size: float = 0.2
    excluded_columns: list[str] = field(default_factory=list)
    estimator_list: list[str] | None = None  # None -> engine default search space
    dataset_name: str = "dataset"
    n_jobs: int = -1


# --------------------------------------------------------------------------- engine
@dataclass
class EngineResult:
    model: Any
    engine_name: str
    engine_version: str
    best_estimator: str
    best_model_name: str
    best_config: dict[str, Any]
    best_validation_score: float | None
    validation_method: str
    training_time: float
    search_history: list[dict[str, Any]]
    leaderboard: list[dict[str, Any]]
    native_importance: list[dict[str, Any]] | None
    classes: list[Any] | None


class AutoMLEngine(Protocol):
    """Minimal contract an AutoML backend must satisfy."""

    name: str

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        task_type: TaskType,
        metric: str,
        time_budget: int,
        seed: int,
        n_jobs: int,
        estimator_list: list[str] | None,
        work_dir: Path,
    ) -> EngineResult: ...


ESTIMATOR_DISPLAY_NAMES = {
    "lgbm": "LightGBM",
    "xgboost": "XGBoost",
    "xgb_limitdepth": "XGBoost (depth-limited)",
    "rf": "Random Forest",
    "extra_tree": "Extra Trees",
    "histgb": "HistGradientBoosting",
    "catboost": "CatBoost",
    "kneighbor": "k-Nearest Neighbors",
    "lrl1": "Logistic Regression (L1)",
    "lrl2": "Logistic Regression (L2)",
    "sgd": "SGD Linear Model",
    "svc": "Linear SVM",
    "enet": "Elastic Net",
    "lassolars": "Lasso LARS",
}


def estimator_display_name(key: str) -> str:
    return ESTIMATOR_DISPLAY_NAMES.get(key, key)


class FLAMLEngine:
    """FLAML-backed AutoML engine. FLAML is used as an unmodified external dependency."""

    name = "flaml"

    _TASKS = {
        TaskType.BINARY: "classification",
        TaskType.MULTICLASS: "classification",
        TaskType.REGRESSION: "regression",
    }
    _METRICS = {
        (TaskType.BINARY, "f1"): "f1",
        (TaskType.BINARY, "roc_auc"): "roc_auc",
        (TaskType.BINARY, "accuracy"): "accuracy",
        (TaskType.BINARY, "log_loss"): "log_loss",
        (TaskType.MULTICLASS, "f1"): "macro_f1",
        (TaskType.MULTICLASS, "roc_auc"): "roc_auc_ovr",
        (TaskType.MULTICLASS, "accuracy"): "accuracy",
        (TaskType.MULTICLASS, "log_loss"): "log_loss",
        (TaskType.REGRESSION, "r2"): "r2",
        (TaskType.REGRESSION, "rmse"): "rmse",
        (TaskType.REGRESSION, "mae"): "mae",
        (TaskType.REGRESSION, "mse"): "mse",
    }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        task_type: TaskType,
        metric: str,
        time_budget: int,
        seed: int,
        n_jobs: int,
        estimator_list: list[str] | None,
        work_dir: Path,
    ) -> EngineResult:
        # FLAML imports MLflow; silence MLflow's import-time "agent hint" banner.
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import flaml
        from flaml import AutoML

        try:
            flaml_metric = self._METRICS[(task_type, metric)]
        except KeyError as exc:
            raise ExperimentError(f"Metric '{metric}' is not supported by FLAML for {task_type.value}.") from exc

        log_path = Path(work_dir) / "flaml_search.log"
        automl = AutoML()
        start = time.perf_counter()
        try:
            automl.fit(
                X_train=X_train,
                y_train=y_train,
                task=self._TASKS[task_type],
                metric=flaml_metric,
                time_budget=time_budget,
                seed=seed,
                n_jobs=n_jobs,
                estimator_list=estimator_list or "auto",
                eval_method="auto",
                log_file_name=str(log_path),
                log_type="all",
                retrain_full=True,
                early_stop=True,
                verbose=0,
                mlflow_logging=False,  # tracking is handled by ExperimentTracker
            )
        except Exception as exc:
            raise ExperimentError(f"FLAML search failed: {exc}") from exc
        training_time = time.perf_counter() - start

        if automl.best_estimator is None or automl.model is None:
            raise ExperimentError(
                "FLAML did not complete any trial within the time budget. Increase the time budget."
            )

        greater_is_better = METRICS[metric].greater_is_better

        def to_score(loss: Any) -> float | None:
            if loss is None or not math.isfinite(float(loss)):
                return None
            return 1.0 - float(loss) if greater_is_better else float(loss)

        history = self._read_history(log_path, to_score)
        classes = None
        if task_type.is_classification and automl.classes_ is not None:
            classes = list(automl.classes_)

        return EngineResult(
            model=automl,
            engine_name=self.name,
            engine_version=flaml.__version__,
            best_estimator=automl.best_estimator,
            best_model_name=estimator_display_name(automl.best_estimator),
            best_config=to_jsonable(automl.best_config or {}),
            best_validation_score=to_score(automl.best_loss),
            validation_method=str(getattr(getattr(automl, "_state", None), "eval_method", "auto")),
            training_time=training_time,
            search_history=history,
            leaderboard=self._leaderboard(automl, history, to_score, greater_is_better),
            native_importance=self._native_importance(automl),
            classes=classes,
        )

    @staticmethod
    def _read_history(log_path: Path, to_score: Callable[[Any], float | None]) -> list[dict[str, Any]]:
        """Parse FLAML's JSON-lines search log into engine-neutral trial records."""
        history: list[dict[str, Any]] = []
        if not log_path.exists():
            return history
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "record_id" not in record:
                continue
            learner = record.get("learner")
            history.append(
                {
                    "trial": int(record["record_id"]),
                    "estimator": learner,
                    "model_name": estimator_display_name(learner),
                    "validation_score": to_score(record.get("validation_loss")),
                    "trial_time": record.get("trial_time"),
                    "wall_clock_time": record.get("wall_clock_time"),
                    "sample_size": record.get("sample_size"),
                    "config": to_jsonable(record.get("config") or {}),
                }
            )
        return history

    @staticmethod
    def _leaderboard(
        automl: Any,
        history: list[dict[str, Any]],
        to_score: Callable[[Any], float | None],
        greater_is_better: bool,
    ) -> list[dict[str, Any]]:
        trials = Counter(h["estimator"] for h in history)
        configs = automl.best_config_per_estimator or {}
        rows = []
        for estimator, loss in (automl.best_loss_per_estimator or {}).items():
            rows.append(
                {
                    "estimator": estimator,
                    "model_name": estimator_display_name(estimator),
                    "best_validation_score": to_score(loss),
                    "trials": trials.get(estimator, 0),
                    "best_config": to_jsonable(configs.get(estimator) or {}),
                }
            )

        def sort_key(row: dict[str, Any]) -> tuple[bool, float]:
            score = row["best_validation_score"]
            if score is None:
                return (True, 0.0)
            return (False, -score if greater_is_better else score)

        rows.sort(key=sort_key)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank if row["best_validation_score"] is not None else None
        return rows

    @staticmethod
    def _native_importance(automl: Any) -> list[dict[str, Any]] | None:
        """Model-native importances (on FLAML-transformed features), if the model has them."""
        try:
            importances = automl.feature_importances_
            names = automl.feature_names_in_
        except Exception:  # not every estimator exposes these
            return None
        if importances is None or names is None:
            return None
        values = np.abs(np.asarray(importances, dtype=float))
        if values.ndim > 1:  # e.g. per-class coefficients
            values = values.mean(axis=0)
        names = [str(n) for n in names]
        if len(values) != len(names):
            return None
        total = float(values.sum()) or 1.0
        rows = [
            {"feature": name, "importance": float(v), "importance_share": float(v) / total}
            for name, v in zip(names, values)
        ]
        return sorted(rows, key=lambda r: r["importance"], reverse=True)


# --------------------------------------------------------------------------- result
@dataclass
class ExperimentResult:
    experiment_id: str
    created_at: str
    dataset_name: str
    target_column: str
    task_type: TaskType
    primary_metric: str
    time_budget: int
    random_seed: int
    test_size: float
    engine: str
    engine_version: str
    n_train: int
    n_test: int
    feature_columns: list[str]
    excluded_columns: list[str]
    best_estimator: str
    best_model_name: str
    best_config: dict[str, Any]
    best_validation_score: float | None
    validation_method: str
    test_score: float | None
    train_score: float | None
    test_metrics: dict[str, Any]
    train_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    training_time: float
    total_time: float
    n_trials: int
    leaderboard: list[dict[str, Any]]
    feature_importance: list[dict[str, Any]]
    native_feature_importance: list[dict[str, Any]] | None
    dataset_meta_features: dict[str, float]
    warnings: list[str]
    model_path: str
    artifact_dir: str
    mlflow_run_id: str | None = None
    status: str = "completed"
    # Rich objects for the UI; not part of the JSON summary.
    search_history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    test_evaluation: EvaluationResult | None = field(default=None, repr=False)
    profile: DatasetProfile | None = field(default=None, repr=False)

    _NON_SUMMARY_FIELDS = ("search_history", "test_evaluation", "profile")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary (stored in experiments/EXP-###/summary.json)."""
        data = {k: v for k, v in self.__dict__.items() if k not in self._NON_SUMMARY_FIELDS}
        return to_jsonable(data)


# --------------------------------------------------------------------------- runner
@dataclass
class _PreparedData:
    """Validated features/target and the train/test split for one experiment."""

    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    task_type: TaskType
    metric: str
    warnings: list[str]


class AutoMLRunner:
    """Run one complete baseline experiment on a DataFrame."""

    def __init__(
        self,
        settings: Settings | None = None,
        engine: AutoMLEngine | None = None,
        tracker: ExperimentTracker | None = None,
        profiler: DatasetProfiler | None = None,
        task_detector: TaskDetector | None = None,
        enable_mlflow: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or FLAMLEngine()
        self.tracker = tracker or ExperimentTracker.from_settings(self.settings)
        self.profiler = profiler or DatasetProfiler()
        self.task_detector = task_detector or TaskDetector()
        self.enable_mlflow = enable_mlflow

    def run(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        task: TaskType | str | None = None,
        time_budget: int | None = None,
        primary_metric: str | None = None,
        random_seed: int | None = None,
        test_size: float | None = None,
        excluded_columns: list[str] | None = None,
        dataset_name: str = "dataset",
        estimator_list: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ExperimentResult:
        config = ExperimentConfig(
            target_column=target_column,
            task_type=TaskType(task) if task is not None else None,
            primary_metric=primary_metric,
            time_budget=time_budget if time_budget is not None else self.settings.default_time_budget,
            random_seed=random_seed if random_seed is not None else self.settings.default_random_seed,
            test_size=test_size if test_size is not None else self.settings.default_test_size,
            excluded_columns=list(excluded_columns or []),
            estimator_list=estimator_list,
            dataset_name=dataset_name,
            n_jobs=self.settings.n_jobs,
        )
        return self.run_experiment(dataframe, config, progress_callback)

    # ----------------------------------------------------------- main flow
    def run_experiment(
        self, dataframe: pd.DataFrame, config: ExperimentConfig, progress_callback: ProgressCallback | None = None
    ) -> ExperimentResult:
        notify = progress_callback or (lambda stage, fraction, message: None)
        total_start = time.perf_counter()

        notify("preparing", 0.02, "Validating data and configuration")
        X, y, task_type, metric, warnings = self._prepare(dataframe, config)

        X_train, X_test, y_train, y_test = self._split(X, y, task_type, config, warnings)
        warnings.extend(unseen_category_warnings(X_train, X_test))
        experiment_id = self.tracker.allocate_experiment_id()
        work_dir = self.tracker.experiment_dir(experiment_id)
        logger.info(
            "%s: %s | target=%s | task=%s | metric=%s | budget=%ss | train=%d test=%d",
            experiment_id, config.dataset_name, config.target_column, task_type.value, metric,
            config.time_budget, len(X_train), len(X_test),
        )
        data = _PreparedData(X, y, X_train, X_test, y_train, y_test, task_type, metric, warnings)
        try:
            return self._execute(experiment_id, work_dir, dataframe, config, data, notify, total_start)
        except Exception as exc:
            # Keep a trace of failed runs (useful for debugging and, later, experiment memory).
            failure = {"experiment_id": experiment_id, "status": "failed", "error": str(exc), "config": config.__dict__}
            (work_dir / "error.json").write_text(json.dumps(to_jsonable(failure), indent=2, default=str), "utf-8")
            logger.error("%s failed: %s", experiment_id, exc)
            raise

    def _execute(
        self,
        experiment_id: str,
        work_dir: Path,
        dataframe: pd.DataFrame,
        config: ExperimentConfig,
        data: _PreparedData,
        notify: ProgressCallback,
        total_start: float,
    ) -> ExperimentResult:
        X, y, task_type, metric, warnings = data.X, data.y, data.task_type, data.metric, data.warnings
        X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test

        notify("searching", 0.05, f"AutoML search running ({config.time_budget}s budget)")
        engine_result = self.engine.fit(
            X_train,
            y_train,
            task_type=task_type,
            metric=metric,
            time_budget=config.time_budget,
            seed=config.random_seed,
            n_jobs=config.n_jobs,
            estimator_list=config.estimator_list,
            work_dir=work_dir,
        )

        notify("evaluating", 0.85, "Evaluating the best model on the held-out test set")
        evaluator = Evaluator(task_type, metric)
        model, classes = engine_result.model, engine_result.classes
        test_eval = evaluator.evaluate(model, X_test, y_test, classes)
        train_eval = evaluator.evaluate(model, X_train, y_train, classes)
        baseline = evaluator.baseline_metrics(y_train, y_test)
        warnings.extend(test_eval.notes)

        notify("explaining", 0.9, "Computing permutation feature importance")
        try:
            importance = evaluator.feature_importance(model, X_test, y_test, random_state=config.random_seed)
            importance_rows = to_jsonable(importance.to_dict(orient="records"))
        except Exception as exc:
            logger.warning("Permutation importance failed: %s", exc)
            warnings.append(f"Feature importance could not be computed: {exc}")
            importance_rows = []

        notify("saving", 0.95, "Saving the model and logging to MLflow")
        clean_df = dataframe.loc[y.index]
        profile = self.profiler.profile(clean_df, config.target_column, config.dataset_name, task_type)
        model_path = self._save_model(model, experiment_id, X.columns.tolist(), config, task_type, classes)

        result = ExperimentResult(
            experiment_id=experiment_id,
            created_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            dataset_name=config.dataset_name,
            target_column=config.target_column,
            task_type=task_type,
            primary_metric=metric,
            time_budget=config.time_budget,
            random_seed=config.random_seed,
            test_size=config.test_size,
            engine=engine_result.engine_name,
            engine_version=engine_result.engine_version,
            n_train=len(X_train),
            n_test=len(X_test),
            feature_columns=X.columns.tolist(),
            excluded_columns=list(config.excluded_columns),
            best_estimator=engine_result.best_estimator,
            best_model_name=engine_result.best_model_name,
            best_config=engine_result.best_config,
            best_validation_score=engine_result.best_validation_score,
            validation_method=engine_result.validation_method,
            test_score=test_eval.primary_score,
            train_score=train_eval.primary_score,
            test_metrics=to_jsonable(test_eval.metrics),
            train_metrics=to_jsonable(train_eval.metrics),
            baseline_metrics=to_jsonable(baseline),
            training_time=round(engine_result.training_time, 2),
            total_time=0.0,
            n_trials=len(engine_result.search_history),
            leaderboard=engine_result.leaderboard,
            feature_importance=importance_rows,
            native_feature_importance=engine_result.native_importance,
            dataset_meta_features=to_jsonable(profile.meta_features()),
            warnings=list(dict.fromkeys(warnings)),
            model_path=str(model_path),
            artifact_dir=str(work_dir),
            search_history=engine_result.search_history,
            test_evaluation=test_eval,
            profile=profile,
        )
        artifacts = self._write_artifacts(result, work_dir)
        result.total_time = round(time.perf_counter() - total_start, 2)

        if self.enable_mlflow:
            try:
                result.mlflow_run_id = self._log_to_mlflow(result, artifacts, Path(model_path))
            except Exception as exc:  # tracking must never lose a finished experiment
                logger.exception("MLflow logging failed")
                result.warnings.append(f"MLflow logging failed: {exc}")
        self.tracker.save_record(experiment_id, result.to_dict())

        notify("done", 1.0, "Experiment complete")
        logger.info(
            "%s finished: best=%s test_%s=%s in %.1fs",
            experiment_id, result.best_model_name, metric, result.test_score, result.total_time,
        )
        return result

    # ------------------------------------------------------------- helpers
    def _prepare(
        self, df: pd.DataFrame, config: ExperimentConfig
    ) -> tuple[pd.DataFrame, pd.Series, TaskType, str, list[str]]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ExperimentError("The dataset is empty.")
        target = config.target_column
        if target not in df.columns:
            raise ExperimentError(f"Target column '{target}' not found in the dataset.")
        unknown = [c for c in config.excluded_columns if c not in df.columns]
        if unknown:
            raise ExperimentError(f"Excluded columns not found in the dataset: {unknown}")
        if not self.settings.min_time_budget <= config.time_budget <= self.settings.max_time_budget:
            raise ExperimentError(
                f"Time budget must be between {self.settings.min_time_budget} and "
                f"{self.settings.max_time_budget} seconds."
            )
        if not 0.05 <= config.test_size <= 0.5:
            raise ExperimentError("Test size must be between 0.05 and 0.5.")

        warnings: list[str] = []
        y_full = df[target]
        y = y_full.dropna()
        X = df.loc[y.index].drop(columns=[target, *[c for c in config.excluded_columns if c != target]])
        if X.shape[1] == 0:
            raise ExperimentError("No feature columns left after excluding columns.")
        if len(X) < MIN_ROWS:
            raise ExperimentError(f"At least {MIN_ROWS} labelled rows are required (found {len(X)}).")

        try:
            if config.task_type is None:
                detection = self.task_detector.detect(y_full)
                task_type = detection.task_type
                warnings.append(f"Task auto-detected as {task_type.display_name}: {detection.reason}")
            else:
                task_type = config.task_type
            warnings.extend(self.task_detector.validate(y_full, task_type))
            metric = validate_primary_metric(task_type, config.primary_metric or default_primary_metric(task_type))
        except ValueError as exc:
            raise ExperimentError(str(exc)) from exc

        if task_type.is_classification:
            warnings.extend(metric_warnings(task_type, metric, analyze_class_balance(y)))
        return X, y, task_type, metric, warnings

    @staticmethod
    def _split(
        X: pd.DataFrame, y: pd.Series, task_type: TaskType, config: ExperimentConfig, warnings: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        stratify = None
        if task_type.is_classification:
            if y.value_counts().min() >= 2:
                stratify = y
            else:
                warnings.append("Stratified split disabled because some classes have fewer than 2 samples.")
        try:
            return train_test_split(
                X, y, test_size=config.test_size, random_state=config.random_seed, stratify=stratify
            )
        except ValueError as exc:
            raise ExperimentError(f"Could not split the data: {exc}") from exc

    def _save_model(
        self,
        model: Any,
        experiment_id: str,
        feature_columns: list[str],
        config: ExperimentConfig,
        task_type: TaskType,
        classes: list[Any] | None,
    ) -> Path:
        model_dir = self.settings.models_dir / experiment_id
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / "model.pkl"
        bundle = {
            "experiment_id": experiment_id,
            "engine": self.engine.name,
            "model": model,
            "feature_columns": feature_columns,
            "target_column": config.target_column,
            "task_type": task_type.value,
            "classes": classes,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with path.open("wb") as fh:
            pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load_model(path: str | Path) -> dict[str, Any]:
        """Load a saved model bundle.

        Security: pickle can execute arbitrary code. Only load files produced by
        this application on this machine.
        """
        with Path(path).open("rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def _write_artifacts(result: ExperimentResult, work_dir: Path) -> list[Path]:
        paths = []

        def write_json(name: str, payload: Any) -> None:
            path = work_dir / name
            path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str), encoding="utf-8")
            paths.append(path)

        def write_csv(name: str, frame: pd.DataFrame) -> None:
            if frame.empty:
                return
            path = work_dir / name
            frame.to_csv(path, index=False)
            paths.append(path)

        if result.profile is not None:
            write_json("profile.json", result.profile.to_dict())
        evaluation = {
            "test": result.test_evaluation.to_dict() if result.test_evaluation else None,
            "train_metrics": result.train_metrics,
            "baseline_metrics": result.baseline_metrics,
        }
        write_json("evaluation.json", evaluation)
        write_csv("leaderboard.csv", pd.DataFrame(result.leaderboard).drop(columns=["best_config"], errors="ignore"))
        history = pd.DataFrame(result.search_history)
        if not history.empty:
            history["config"] = history["config"].apply(json.dumps)
        write_csv("search_history.csv", history)
        write_csv("feature_importance.csv", pd.DataFrame(result.feature_importance))
        if result.native_feature_importance:
            write_csv("native_feature_importance.csv", pd.DataFrame(result.native_feature_importance))
        if result.test_evaluation is not None:
            write_csv("predictions.csv", result.test_evaluation.predictions_frame())
        flaml_log = work_dir / "flaml_search.log"
        if flaml_log.exists():
            paths.append(flaml_log)
        return paths

    def _log_to_mlflow(self, result: ExperimentResult, artifacts: list[Path], model_path: Path) -> str:
        params: dict[str, Any] = {
            "experiment_id": result.experiment_id,
            "dataset": result.dataset_name,
            "target": result.target_column,
            "task": result.task_type.value,
            "primary_metric": result.primary_metric,
            "time_budget_s": result.time_budget,
            "random_seed": result.random_seed,
            "test_size": result.test_size,
            "engine": f"{result.engine}=={result.engine_version}",
            "validation_method": result.validation_method,
            "best_estimator": result.best_estimator,
            "n_train": result.n_train,
            "n_test": result.n_test,
            "n_features": len(result.feature_columns),
            "excluded_columns": ",".join(result.excluded_columns) or "none",
        }
        params.update({f"hp.{k}": v for k, v in result.best_config.items()})
        params.update({f"meta.{k}": round(v, 6) for k, v in result.dataset_meta_features.items()})

        metrics: dict[str, Any] = {
            "primary_test_score": result.test_score,
            "primary_train_score": result.train_score,
            "validation_score": result.best_validation_score,
            "training_time_s": result.training_time,
            "total_time_s": result.total_time,
            "n_trials": result.n_trials,
        }
        metrics.update({f"test_{k}": v for k, v in result.test_metrics.items()})
        metrics.update({f"train_{k}": v for k, v in result.train_metrics.items()})
        metrics.update({f"baseline_{k}": v for k, v in result.baseline_metrics.items() if k != "strategy"})

        tags = {
            "experiment_id": result.experiment_id,
            "dataset": result.dataset_name,
            "task": result.task_type.value,
            "best_model": result.best_model_name,
            "project": "autopilot-ml",
            "phase": "baseline",
        }
        artifact_list: list[tuple[Path, str | None]] = [(model_path, "model")]
        artifact_list += [(path, "results") for path in artifacts]
        return self.tracker.log_run(result.experiment_id, params, metrics, tags, artifact_list)
