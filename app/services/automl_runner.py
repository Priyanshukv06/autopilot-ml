"""AutoML orchestration: split -> engine search -> evaluation -> explainability -> tracking.

The rest of the application talks to ``AutoMLRunner`` and ``ExperimentResult`` only.
FLAML specifics are contained in ``FLAMLEngine``; another engine (e.g. AutoGluon for
the planned baseline comparison) can be added by implementing ``AutoMLEngine``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import re
import time
import warnings as py_warnings
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.class_weight import compute_sample_weight

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
from app.services.search_summary import summarize_search
from app.services.splits import OuterSplit, SplitError, make_outer_split
from app.services.task_detector import TaskDetector, TaskType
from app.services.transforms import DataPipeline, DataStepError, PipelineModel, as_specs

logger = get_logger(__name__)

ProgressCallback = Callable[[str, float, str], None]
"""Signature: (stage, fraction_complete, message)."""

MIN_ROWS = 20
SCHEMA_VERSION = 2  # version of summary.json; Phase 1 records have no version field (version 1)
CLASS_WEIGHT_OPTIONS = (None, "balanced")

# Two kinds of warnings are pure noise here and can flood the terminal:
# - FLAML warns on every predict() call when a categorical column has values it did not see
#   during fit (dozens of times during permutation importance). The runner reports this once
#   per column instead, see `unseen_category_warnings`.
# - ConvergenceWarning from the short linear-model trials FLAML runs on purpose.
# - scikit-learn's "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`"
#   (added 31 October 2026). scikit-learn shows it when a job finds the warning filters it was given
#   empty; the reset described below can empty them while another thread hands out jobs. This is the
#   likely cause, not a confirmed one (a test with threads did not reproduce it). The project sets no
#   scikit-learn configuration, so the jobs compute the same; the warning only filled a smoke run's
#   log with about 37,800 lines.
# A warnings filter alone is not enough: scikit-learn's joblib wrapper calls
# warnings.resetwarnings() inside worker threads, and since the filter list is shared by the
# whole process this can wipe it in the middle of a search. Dropping these in showwarning()
# as well survives that reset.
_UNSEEN_CATEGORY_MESSAGE = re.compile(r"Column '.*' contains values unseen at fit time")
_PARALLEL_CONFIG_MESSAGE = re.compile(r"`sklearn\.utils\.parallel\.delayed` should be used with")
py_warnings.filterwarnings("ignore", message=_UNSEEN_CATEGORY_MESSAGE.pattern, category=UserWarning)
py_warnings.filterwarnings("ignore", message=_PARALLEL_CONFIG_MESSAGE.pattern, category=UserWarning)
_default_showwarning = py_warnings.showwarning


def _is_search_noise(message: Any, category: type) -> bool:
    if issubclass(category, ConvergenceWarning):
        return True
    text = str(message)
    return issubclass(category, UserWarning) and bool(_UNSEEN_CATEGORY_MESSAGE.match(text)
                                                      or _PARALLEL_CONFIG_MESSAGE.match(text))


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


def unseen_category_shares(X_train: pd.DataFrame, X_other: pd.DataFrame) -> dict[str, float]:
    """For each categorical column, the share of ``X_other`` rows whose value never appears in ``X_train``.

    Empty cells count as seen (FLAML fills them). The share is over all rows of ``X_other``.
    """
    shares: dict[str, float] = {}
    for column in X_train.columns:
        train_col = X_train[column]
        if is_numeric_dtype(train_col) and not is_bool_dtype(train_col):
            continue
        other_values = X_other[column].dropna()
        unseen = ~other_values.isin(set(train_col.dropna().unique()))
        shares[column] = float(unseen.sum() / len(X_other)) if len(X_other) else 0.0
    return shares


def unseen_category_warnings(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
    """Report categorical columns whose test values never appear in the training split."""
    return [
        f"'{column}': {share:.0%} of test rows have categories never seen in "
        "training, so the model cannot use them (typical for high-cardinality columns)."
        for column, share in unseen_category_shares(X_train, X_test).items()
        if share > 0
    ]


# --------------------------------------------------------------------------- config
@dataclass
class ExperimentConfig:
    """Everything needed to reproduce one experiment.

    The fields after ``n_jobs`` are used by the autonomous layer. Their defaults give the
    Phase 1 behaviour.
    """

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
    data_steps: list[Any] = field(default_factory=list)  # Data_Step specifications (task 12)
    class_weight: str | None = None  # None or "balanced"
    search_metric: str | None = None  # metric FLAML optimises; None means the primary metric
    # Plain JSON, for example {"lgbm": {"num_leaves": {"upper": 32}}}; bounds are inclusive.
    search_space_limits: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    starting_points: dict[str, dict[str, Any]] | None = None  # learner family -> configuration
    max_iter: int | None = None  # fixed number of trials (tests only)

    def setup(self) -> dict[str, Any]:
        """The settings that define the Setup, as JSON-safe values.

        Left out: the time budget, ``max_iter`` and the starting points (they change how long or
        from where FLAML searches, not what it searches), and ``dataset_name`` and ``n_jobs``.
        Column and learner lists are sorted, so their order does not matter.
        """
        task = self.task_type.value if isinstance(self.task_type, TaskType) else self.task_type
        steps = [step.to_dict() if hasattr(step, "to_dict") else step for step in self.data_steps]
        return to_jsonable(
            {
                "target_column": self.target_column,
                "task_type": task,
                "primary_metric": self.primary_metric,
                "random_seed": self.random_seed,
                "test_size": self.test_size,
                "excluded_columns": sorted(self.excluded_columns),
                "estimator_list": sorted(self.estimator_list) if self.estimator_list else None,
                "data_steps": steps,
                "class_weight": self.class_weight,
                "search_metric": self.search_metric,
                "search_space_limits": self.search_space_limits or {},
            }
        )

    def setup_key(self) -> str:
        """Short stable hash of ``setup()``: equal for two configs that define the same Setup."""
        text = json.dumps(self.setup(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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
    # Best configuration of every learner family that finished a trial (for warm starts).
    best_config_per_estimator: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class AutoMLEngine(Protocol):
    """Minimal contract an AutoML backend must satisfy.

    ``metric`` is the metric the search optimises (the Search_Metric). The runner scores the
    final model with the primary metric itself.
    """

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
        starting_points: dict[str, dict[str, Any]] | None = None,
        search_space_limits: dict[str, dict[str, dict[str, float]]] | None = None,
        sample_weight: np.ndarray | None = None,
        max_iter: int | None = None,
    ) -> EngineResult: ...

    def fit_config(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        task_type: TaskType,
        estimator: str,
        config: dict[str, Any],
        seed: int,
        n_jobs: int,
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        """Train exactly one configuration. Returns a model with predict (and predict_proba, classes_)."""
        ...


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


# FLAML keys that are not hyperparameters (removed by FLAML's AutoMLState.sanitize as well).
_NON_HYPERPARAMETER_KEYS = ("FLAML_sample_size", "learner", "_choice_")


def _upper_is_exclusive(domain: Any) -> bool:
    """True when FLAML never searches the domain's ``upper`` value itself.

    FLAML 2.7.0 maps a non-quantized integer setting onto ``lower .. upper - 1`` during the
    search (``normalize`` and ``denormalize`` in ``flaml/tune/space.py``). FLAML's own search
    spaces follow this: LightGBM's ``min_child_samples`` uses ``upper=2**7 + 1`` for "at most 128".
    """
    from flaml.tune import sample

    return isinstance(domain, sample.Integer) and not isinstance(domain.get_sampler(), sample.Quantized)


def _searched_range(domain: Any) -> tuple[float, float]:
    """Smallest and largest value FLAML can try for a numeric domain (both inclusive)."""
    upper = domain.upper - 1 if _upper_is_exclusive(domain) else domain.upper
    return domain.lower, upper


def value_in_search_space(value: Any, spec: dict[str, Any] | None) -> bool:
    """Whether a starting value is one FLAML could try for this setting.

    Follows FLAML's own check (``AutoMLState.valid_starting_point_one_dim``): the value must have
    the domain's type and lie inside it. For integer settings it uses the range FLAML really
    searches, which ends one below ``upper`` (see ``_upper_is_exclusive``).
    """
    from flaml.tune import sample

    if not spec or "domain" not in spec:
        return False
    domain = spec["domain"]
    if isinstance(domain, sample.Categorical):
        return value in domain.categories
    if isinstance(domain, sample.Integer):
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        lower, upper = _searched_range(domain)
        return lower <= value <= upper
    if isinstance(domain, sample.Float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return domain.lower <= value <= domain.upper
    if isinstance(domain, sample.Domain):
        return bool(domain.is_valid(value))
    return value == domain  # a fixed value, not searched


def limited_search_space_spec(spec: dict[str, Any] | None, bounds: dict[str, float], label: str) -> dict[str, Any]:
    """Copy FLAML's spec of one numeric setting with tighter bounds (inclusive, as written by the user).

    The domain type (for example ``tune.lograndint``) and the other keys are kept. The starting
    values ``init_value`` and ``low_cost_init_value`` are moved inside the new bounds. A limit can
    only narrow the range, never widen it.
    """
    from flaml.tune import sample

    domain = (spec or {}).get("domain")
    if not isinstance(domain, (sample.Integer, sample.Float)):
        raise ExperimentError(f"'{label}' is not a numeric setting of that learner, so it cannot be limited.")
    unknown = sorted(set(bounds) - {"lower", "upper"})
    if unknown:
        raise ExperimentError(f"Unknown search-space limit {unknown} for '{label}'. Use 'lower' and 'upper'.")
    is_integer = isinstance(domain, sample.Integer)
    lower, upper = _searched_range(domain)
    if bounds.get("lower") is not None:
        lower = max(lower, math.ceil(bounds["lower"]) if is_integer else float(bounds["lower"]))
    if bounds.get("upper") is not None:
        upper = min(upper, math.floor(bounds["upper"]) if is_integer else float(bounds["upper"]))
    if not lower < upper:
        raise ExperimentError(f"The search-space limits for '{label}' leave fewer than two values to search.")

    limited = copy.deepcopy(domain)
    limited.lower = lower
    limited.upper = upper + 1 if _upper_is_exclusive(domain) else upper
    new_spec = dict(spec, domain=limited)
    for key in ("init_value", "low_cost_init_value"):
        if new_spec.get(key) is not None:
            new_spec[key] = min(max(new_spec[key], lower), upper)
    return new_spec


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
        starting_points: dict[str, dict[str, Any]] | None = None,
        search_space_limits: dict[str, dict[str, dict[str, float]]] | None = None,
        sample_weight: np.ndarray | None = None,
        max_iter: int | None = None,
    ) -> EngineResult:
        """Run a FLAML search. With none of the optional arguments this is the Phase 1 search."""
        # FLAML imports MLflow; silence MLflow's import-time "agent hint" banner.
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import flaml
        from flaml import AutoML

        try:
            flaml_metric = self._METRICS[(task_type, metric)]
        except KeyError as exc:
            raise ExperimentError(f"Metric '{metric}' is not supported by FLAML for {task_type.value}.") from exc

        # Optional settings are passed only when given, so that the Phase 1 call stays unchanged.
        options: dict[str, Any] = {}
        engine_warnings: list[str] = []
        custom_hp = self.custom_hp(search_space_limits, task_type, X_train.shape) if search_space_limits else {}
        if custom_hp:
            options["custom_hp"] = custom_hp
        if starting_points:
            points, engine_warnings = self.check_starting_points(starting_points, task_type, X_train.shape, custom_hp)
            if points:
                options["starting_points"] = points
        if sample_weight is not None:
            options["sample_weight"] = self._checked_weights(sample_weight, len(y_train))
        if max_iter is not None:
            options["max_iter"] = int(max_iter)

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
                **options,
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
            best_config_per_estimator={
                estimator: to_jsonable(config)
                for estimator, config in (automl.best_config_per_estimator or {}).items()
                if config
            },
            warnings=engine_warnings,
        )

    def fit_config(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        task_type: TaskType,
        estimator: str,
        config: dict[str, Any],
        seed: int,
        n_jobs: int,
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        """Train one learner family with one fixed configuration on all the given rows.

        Uses ``max_iter=1`` with the configuration as the starting point: FLAML then skips the
        search and trains exactly that configuration, with the same data preparation as a search
        (checked in ``tests/test_flaml_capabilities.py``). Returns FLAML's ``AutoML`` object, which
        has ``predict``, ``predict_proba`` and ``classes_`` and predicts the original labels.
        """
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        from flaml import AutoML

        options: dict[str, Any] = {}
        if sample_weight is not None:
            options["sample_weight"] = self._checked_weights(sample_weight, len(y_train))
        automl = AutoML()
        try:
            automl.fit(
                X_train=X_train,
                y_train=y_train,
                task=self._TASKS[task_type],
                estimator_list=[estimator],
                starting_points={estimator: dict(config)},
                max_iter=1,
                retrain_full=True,
                seed=seed,
                n_jobs=n_jobs,
                verbose=0,
                mlflow_logging=False,
                **options,
            )
        except Exception as exc:
            raise ExperimentError(f"Training {estimator_display_name(estimator)} with a fixed configuration "
                                  f"failed: {exc}") from exc
        if automl.model is None:
            raise ExperimentError(f"FLAML did not train {estimator_display_name(estimator)}.")
        return automl

    def search_space(self, estimator: str, task_type: TaskType, data_size: tuple[int, int]) -> dict[str, Any]:
        """FLAML's default search space of one learner family for data of this size."""
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        from flaml.automl.task.factory import task_factory

        task = task_factory(self._TASKS[task_type])
        try:
            learner = task.estimator_class_from_str(estimator)
        except Exception as exc:
            raise ExperimentError(f"Unknown learner family '{estimator}'.") from exc
        return learner.search_space(data_size=data_size, task=task)

    def custom_hp(
        self,
        limits: dict[str, dict[str, dict[str, float]]],
        task_type: TaskType,
        data_size: tuple[int, int],
    ) -> dict[str, dict[str, Any]]:
        """Turn plain JSON limits into FLAML's ``custom_hp`` (one copied spec per limited setting).

        The default space depends on the number of rows (for example LightGBM's ``num_leaves``
        goes up to the number of rows), so ``data_size`` should be the shape FLAML searches on.
        """
        custom: dict[str, dict[str, Any]] = {}
        for estimator, settings in limits.items():
            space = self.search_space(estimator, task_type, data_size)
            for name, bounds in settings.items():
                custom.setdefault(estimator, {})[name] = limited_search_space_spec(
                    space.get(name), bounds, f"{estimator}.{name}"
                )
        return custom

    def check_starting_points(
        self,
        starting_points: dict[str, dict[str, Any]],
        task_type: TaskType,
        data_size: tuple[int, int],
        custom_hp: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Keep the starting configurations that fit the (possibly limited) search space.

        FLAML drops an invalid starting point itself, but only logs it, and that log could not be
        captured (task 1). So invalid configurations are left out here, with a warning. FLAML builds
        its space from the rows it searches on; with a holdout split those are slightly fewer than
        ``data_size``, so a value at the very top of a row-dependent range may still be dropped by
        FLAML without a warning.
        """
        valid: dict[str, dict[str, Any]] = {}
        messages: list[str] = []
        for estimator, raw in starting_points.items():
            config = {k: v for k, v in to_jsonable(raw or {}).items() if k not in _NON_HYPERPARAMETER_KEYS}
            if not config:
                continue
            space = self.search_space(estimator, task_type, data_size)
            space.update((custom_hp or {}).get(estimator, {}))
            outside = sorted(name for name, value in config.items() if not value_in_search_space(value, space.get(name)))
            if outside:
                messages.append(
                    f"The starting configuration for {estimator_display_name(estimator)} was not used because "
                    f"{', '.join(outside)} {'is' if len(outside) == 1 else 'are'} outside the search space."
                )
            else:
                valid[estimator] = config
        return valid, messages

    @staticmethod
    def _checked_weights(sample_weight: Any, n_rows: int) -> np.ndarray:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (n_rows,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ExperimentError("Sample weights must be one finite, non-negative number per training row.")
        return weights

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
    schema_version: int = SCHEMA_VERSION
    # Metric of best_validation_score, the leaderboard and the search history. It differs from
    # primary_metric only when the Setup sets a separate Search_Metric.
    search_metric: str | None = None
    test_evaluated: bool = True  # False inside a Session: no test score, predictions or importance
    best_config_per_estimator: dict[str, dict[str, Any]] = field(default_factory=dict)
    search_summary: dict[str, Any] | None = None
    session_id: str | None = None
    parent_experiment_id: str | None = None
    proposal: dict[str, Any] | None = None
    setup_key: str | None = None
    setup: dict[str, Any] = field(default_factory=dict)
    profile_rows: str = "all"  # rows behind the profile and meta-features: "all" labelled rows or "train"
    # Rich objects for the UI; not part of the JSON summary.
    search_history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    test_evaluation: EvaluationResult | None = field(default=None, repr=False)
    profile: DatasetProfile | None = field(default=None, repr=False)

    _NON_SUMMARY_FIELDS = ("search_history", "test_evaluation", "profile")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary (stored in experiments/EXP-###/summary.json)."""
        data = {k: v for k, v in self.__dict__.items() if k not in self._NON_SUMMARY_FIELDS}
        return to_jsonable(data)


# Fields that hold Test_Part results or may mention them (warnings include test-row notes).
_TEST_PART_FIELDS = ("test_score", "test_metrics", "baseline_metrics", "feature_importance", "warnings")


def decision_view(result: ExperimentResult) -> dict[str, Any]:
    """The part of a result that decision components may read (Requirement 1.6).

    It is ``result.to_dict()`` without any Test_Part metric, prediction or count derived from
    Test_Part labels. Meta-features are kept only when the profile was computed on the
    Training_Part alone (``profile_rows == "train"``).
    """
    view = result.to_dict()
    for key in _TEST_PART_FIELDS:
        view.pop(key, None)
    if result.profile_rows != "train":
        view.pop("dataset_meta_features", None)
    return view


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
    split: OuterSplit
    search_metric: str


@dataclass(frozen=True)
class _RunOptions:
    """How one run uses the Test_Part and how it is linked to a Session."""

    evaluate_test: bool = True
    compute_importance: bool = True
    profile_rows: str = "all"
    session_id: str | None = None
    parent_experiment_id: str | None = None
    proposal: dict[str, Any] | None = None


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
        self,
        dataframe: pd.DataFrame,
        config: ExperimentConfig,
        progress_callback: ProgressCallback | None = None,
        *,
        split: OuterSplit | None = None,
        evaluate_test: bool = True,
        compute_importance: bool = True,
        profile_rows: Literal["all", "train"] = "all",
        session_id: str | None = None,
        parent_experiment_id: str | None = None,
        proposal: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        """Run one experiment. The keyword arguments are used by Sessions; their defaults give Phase 1.

        - ``split``: use exactly these training and test rows (positions of the labelled rows).
          Without it the split is made with the Phase 1 rule.
        - ``evaluate_test=False``: the Test_Part is not used at all: no test metrics, baseline,
          predictions, permutation importance or notes about test rows (``test_evaluated`` is False).
        - ``compute_importance=False``: skip permutation importance.
        - ``profile_rows="train"``: profile and meta-features from the Training_Part only.
        """
        if profile_rows not in ("all", "train"):
            raise ExperimentError(f"profile_rows must be 'all' or 'train', not '{profile_rows}'.")
        options = _RunOptions(evaluate_test, compute_importance, profile_rows, session_id, parent_experiment_id,
                              proposal)
        notify = progress_callback or (lambda stage, fraction, message: None)
        total_start = time.perf_counter()

        notify("preparing", 0.02, "Validating data and configuration")
        X, y, task_type, metric, warnings = self._prepare(dataframe, config)
        search_metric = self._search_metric(config, task_type, metric)

        split, X_train, X_test, y_train, y_test = self._split(X, y, task_type, config, warnings, split)
        if evaluate_test:
            warnings.extend(unseen_category_warnings(X_train, X_test))
        experiment_id = self.tracker.allocate_experiment_id()
        work_dir = self.tracker.experiment_dir(experiment_id)
        logger.info(
            "%s: %s | target=%s | task=%s | metric=%s | budget=%ss | train=%d test=%d",
            experiment_id, config.dataset_name, config.target_column, task_type.value, metric,
            config.time_budget, len(X_train), len(X_test),
        )
        data = _PreparedData(X, y, X_train, X_test, y_train, y_test, task_type, metric, warnings, split, search_metric)
        try:
            return self._execute(experiment_id, work_dir, dataframe, config, data, notify, total_start, options)
        except Exception as exc:
            # Keep a trace of failed runs (useful for debugging and, later, experiment memory).
            failure = {"experiment_id": experiment_id, "status": "failed", "error": str(exc), "config": config.__dict__}
            (work_dir / "error.json").write_text(json.dumps(to_jsonable(failure), indent=2, default=str), "utf-8")
            logger.error("%s failed: %s", experiment_id, exc)
            try:  # lets a Session name the failed Experiment in its record
                exc.experiment_id = experiment_id
            except AttributeError:
                pass
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
        options: _RunOptions,
    ) -> ExperimentResult:
        X, y, task_type, metric, warnings = data.X, data.y, data.task_type, data.metric, data.warnings
        X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test

        sample_weight = None
        if config.class_weight == "balanced":
            sample_weight = compute_sample_weight("balanced", y_train)
        # Data_Steps are learned from the Training_Part only, before the search (Requirement 3.3).
        steps = as_specs(config.data_steps)
        pipeline = DataPipeline(steps).fit(X_train) if steps else None
        X_search = pipeline.transform(X_train) if pipeline is not None else X_train
        notify("searching", 0.05, f"AutoML search running ({config.time_budget}s budget)")
        engine_result = self.engine.fit(
            X_search,
            y_train,
            task_type=task_type,
            metric=data.search_metric,
            time_budget=config.time_budget,
            seed=config.random_seed,
            n_jobs=config.n_jobs,
            estimator_list=config.estimator_list,
            work_dir=work_dir,
            starting_points=config.starting_points,
            search_space_limits=config.search_space_limits or None,
            sample_weight=sample_weight,
            max_iter=config.max_iter,
        )
        warnings.extend(engine_result.warnings)

        evaluator = Evaluator(task_type, metric)
        model, classes = engine_result.model, engine_result.classes
        if pipeline is not None:  # evaluate, explain and save on the original columns
            model = PipelineModel(pipeline, model)
        test_eval: EvaluationResult | None = None
        baseline: dict[str, Any] = {}
        importance_rows: list[dict[str, Any]] = []
        if options.evaluate_test:
            notify("evaluating", 0.85, "Evaluating the best model on the held-out test set")
            test_eval = evaluator.evaluate(model, X_test, y_test, classes)
            baseline = evaluator.baseline_metrics(y_train, y_test)
            warnings.extend(test_eval.notes)
        else:
            notify("evaluating", 0.85, "Scoring the best model on the training part (test part not used)")
        train_eval = evaluator.evaluate(model, X_train, y_train, classes)

        if options.evaluate_test and options.compute_importance:
            notify("explaining", 0.9, "Computing permutation feature importance")
            try:
                importance = evaluator.feature_importance(model, X_test, y_test, random_state=config.random_seed)
                importance_rows = to_jsonable(importance.to_dict(orient="records"))
            except Exception as exc:
                logger.warning("Permutation importance failed: %s", exc)
                warnings.append(f"Feature importance could not be computed: {exc}")

        notify("saving", 0.95, "Saving the model and logging to MLflow")
        clean_df = dataframe.loc[y.index]
        if options.profile_rows == "train":
            clean_df = clean_df.iloc[data.split.train_positions]
        profile = self.profiler.profile(clean_df, config.target_column, config.dataset_name, task_type)
        model_path = self._save_model(model, experiment_id, X.columns.tolist(), config, task_type, classes, pipeline)
        resolved = replace(
            config,
            task_type=task_type,
            primary_metric=metric,
            search_metric=None if data.search_metric == metric else data.search_metric,
        )
        search_summary = None
        if engine_result.search_history:  # engines without a trial history have no summary
            try:
                search_summary = summarize_search(engine_result.search_history, data.search_metric).to_dict()
            except ValueError as exc:
                logger.warning("Search summary not available: %s", exc)

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
            test_score=test_eval.primary_score if test_eval is not None else None,
            train_score=train_eval.primary_score,
            test_metrics=to_jsonable(test_eval.metrics) if test_eval is not None else {},
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
            search_metric=data.search_metric,
            test_evaluated=test_eval is not None,
            best_config_per_estimator=engine_result.best_config_per_estimator,
            search_summary=search_summary,
            session_id=options.session_id,
            parent_experiment_id=options.parent_experiment_id,
            proposal=to_jsonable(options.proposal) if options.proposal is not None else None,
            setup_key=resolved.setup_key(),
            setup=resolved.setup(),
            profile_rows=options.profile_rows,
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
    def prepare(
        self, dataframe: pd.DataFrame, config: ExperimentConfig
    ) -> tuple[pd.DataFrame, pd.Series, TaskType, str, list[str]]:
        """Check the data and configuration as a run does, and return the features, the labelled target
        values (rows without a target dropped, in their original order), the task type, the primary
        metric and notes. Sessions use it to make their splits on exactly the rows a run will use."""
        return self._prepare(dataframe, config)

    def _prepare(
        self, df: pd.DataFrame, config: ExperimentConfig
    ) -> tuple[pd.DataFrame, pd.Series, TaskType, str, list[str]]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ExperimentError("The dataset is empty.")
        # Rows are matched with their target values by label (df.loc[y.index]) and splits are row
        # positions, so repeated labels would silently pair features with the wrong rows.
        if df.index.has_duplicates:
            raise ExperimentError(
                "The dataset has repeated row labels, so rows cannot be matched with their target values. "
                "Reset the index first, for example with df.reset_index(drop=True)."
            )
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
        try:
            as_specs(config.data_steps)
        except (DataStepError, KeyError, TypeError) as exc:
            raise ExperimentError(f"Invalid Data_Step: {exc}") from exc
        if config.class_weight not in CLASS_WEIGHT_OPTIONS:
            raise ExperimentError(f"Unknown class weight option '{config.class_weight}'. Use None or 'balanced'.")
        if config.max_iter is not None and config.max_iter < 1:
            raise ExperimentError("max_iter must be at least 1.")

        warnings: list[str] = []
        y_full = df[target]
        y = y_full.dropna()
        X = df.loc[y.index].drop(columns=[target, *[c for c in config.excluded_columns if c != target]])
        if X.shape[1] == 0:
            raise ExperimentError("No feature columns left after excluding columns.")
        if len(X) < MIN_ROWS:
            raise ExperimentError(f"At least {MIN_ROWS} labelled rows are required (found {len(X)}).")
        if config.data_steps:
            # A dry run on no rows checks the columns and types of every step without reading any data.
            try:
                if DataPipeline(config.data_steps).fit_transform(X.head(0)).shape[1] == 0:
                    raise ExperimentError("The Data_Steps leave no feature columns.")
            except DataStepError as exc:
                raise ExperimentError(f"Invalid Data_Step: {exc}") from exc

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
        elif config.class_weight is not None:
            raise ExperimentError("Class weights apply to classification tasks only.")
        return X, y, task_type, metric, warnings

    @staticmethod
    def _search_metric(config: ExperimentConfig, task_type: TaskType, primary_metric: str) -> str:
        """The metric FLAML optimises: the Setup's Search_Metric, or else the primary metric."""
        if config.search_metric is None:
            return primary_metric
        try:
            return validate_primary_metric(task_type, config.search_metric)
        except ValueError as exc:
            raise ExperimentError(f"Search metric: {exc}") from exc

    @staticmethod
    def _split(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: TaskType,
        config: ExperimentConfig,
        warnings: list[str],
        split: OuterSplit | None = None,
    ) -> tuple[OuterSplit, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Use the given split, or make one with the shared rule in ``splits.py`` (the same rows as Phase 1)."""
        if split is None:
            try:
                split, split_warnings = make_outer_split(y, task_type, config.test_size, config.random_seed)
            except SplitError as exc:
                raise ExperimentError(str(exc)) from exc
            warnings.extend(split_warnings)
        train, test = np.asarray(split.train_positions), np.asarray(split.test_positions)
        positions = np.concatenate([train, test])
        if (
            len(train) == 0
            or len(test) == 0
            or positions.min() < 0
            or positions.max() >= len(y)
            or len(np.unique(positions)) != len(positions)
        ):
            raise ExperimentError(
                f"The given split does not fit this dataset ({len(y)} labelled rows): positions must be "
                "inside the data, both parts non-empty, and no row in both parts."
            )
        return split, X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]

    def _save_model(
        self,
        model: Any,
        experiment_id: str,
        feature_columns: list[str],
        config: ExperimentConfig,
        task_type: TaskType,
        classes: list[Any] | None,
        pipeline: DataPipeline | None = None,
    ) -> Path:
        """Save the model bundle. With Data_Steps, ``model`` is a PipelineModel that predicts from raw
        rows, and the bundle also holds the step specifications and the fitted pipeline."""
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
        if pipeline is not None:
            bundle["data_steps"] = [step.to_dict() for step in pipeline.steps]
            bundle["pipeline"] = pipeline
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
            "search_metric": result.search_metric,
            "class_weight": result.setup.get("class_weight") or "none",
            "search_space_limits": json.dumps(result.setup.get("search_space_limits") or {}, sort_keys=True),
            "data_steps": json.dumps(result.setup.get("data_steps") or []),
            "setup_key": result.setup_key,
            "test_evaluated": result.test_evaluated,
            "profile_rows": result.profile_rows,
            "session_id": result.session_id or "none",
            "parent_experiment_id": result.parent_experiment_id or "none",
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
            "phase": "session" if result.session_id else "baseline",
        }
        if result.session_id:
            tags["session_id"] = result.session_id
        artifact_list: list[tuple[Path, str | None]] = [(model_path, "model")]
        artifact_list += [(path, "results") for path in artifacts]
        return self.tracker.log_run(result.experiment_id, params, metrics, tags, artifact_list,
                                    parent_run_id=self.tracker.session_run_id(result.session_id))
