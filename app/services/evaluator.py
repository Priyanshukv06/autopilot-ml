"""Model-agnostic evaluation and basic explainability.

The evaluator only relies on ``predict`` (and ``predict_proba`` when available),
so it works for FLAML models today and for any other engine later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from app.core.logging_config import get_logger
from app.services.profiler import ClassBalance
from app.services.task_detector import TaskType

logger = get_logger(__name__)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    display_name: str
    greater_is_better: bool
    needs_proba: bool = False


METRICS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec("accuracy", "Accuracy", True),
    "balanced_accuracy": MetricSpec("balanced_accuracy", "Balanced Accuracy", True),
    "precision": MetricSpec("precision", "Precision", True),
    "recall": MetricSpec("recall", "Recall", True),
    "f1": MetricSpec("f1", "F1", True),
    "roc_auc": MetricSpec("roc_auc", "ROC-AUC", True, needs_proba=True),
    "log_loss": MetricSpec("log_loss", "Log Loss", False, needs_proba=True),
    "mae": MetricSpec("mae", "MAE", False),
    "mse": MetricSpec("mse", "MSE", False),
    "rmse": MetricSpec("rmse", "RMSE", False),
    "r2": MetricSpec("r2", "R²", True),
}

# Metrics that can be selected as the optimisation (primary) metric, default first.
PRIMARY_METRICS: dict[TaskType, tuple[str, ...]] = {
    TaskType.BINARY: ("f1", "roc_auc", "accuracy", "log_loss"),
    TaskType.MULTICLASS: ("f1", "accuracy", "roc_auc", "log_loss"),
    TaskType.REGRESSION: ("r2", "rmse", "mae", "mse"),
}

_MACRO_AVERAGED = {"precision", "recall", "f1", "roc_auc"}


def default_primary_metric(task_type: TaskType) -> str:
    return PRIMARY_METRICS[task_type][0]


def metric_display_name(metric: str, task_type: TaskType | None = None) -> str:
    name = METRICS[metric].display_name if metric in METRICS else metric
    if task_type is TaskType.MULTICLASS and metric in _MACRO_AVERAGED:
        name += " (macro)"
    return name


def validate_primary_metric(task_type: TaskType, metric: str) -> str:
    if metric not in PRIMARY_METRICS[task_type]:
        allowed = ", ".join(PRIMARY_METRICS[task_type])
        raise ValueError(f"Metric '{metric}' is not supported for {task_type.value}. Choose one of: {allowed}.")
    return metric


def metric_warnings(task_type: TaskType, primary_metric: str, class_balance: ClassBalance | None) -> list[str]:
    """Warn when the chosen metric can be misleading for this data."""
    warnings: list[str] = []
    if not task_type.is_classification or class_balance is None:
        return warnings
    if primary_metric == "accuracy" and class_balance.severity in ("moderate", "severe"):
        warnings.append(
            f"Accuracy may be misleading: the majority class '{class_balance.majority_class}' is "
            f"{class_balance.majority_ratio:.0%} of rows, so always predicting it already scores "
            f"{class_balance.majority_ratio:.2f}. Prefer F1 or ROC-AUC."
        )
    elif class_balance.severity == "severe":
        warnings.append(
            f"Severe class imbalance (ratio {class_balance.imbalance_ratio:.1f}:1). Check per-class recall "
            "in the confusion matrix, not just the aggregate score."
        )
    return warnings


@dataclass
class EvaluationResult:
    task_type: TaskType
    primary_metric: str
    metrics: dict[str, float | None]
    n_samples: int
    labels: list[str] | None = None
    positive_label: str | None = None
    confusion_matrix: list[list[int]] | None = None
    per_class: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    # Raw arrays for plots; excluded from to_dict().
    y_true: np.ndarray | None = field(default=None, repr=False)
    y_pred: np.ndarray | None = field(default=None, repr=False)
    y_proba: np.ndarray | None = field(default=None, repr=False)

    @property
    def primary_score(self) -> float | None:
        return self.metrics.get(self.primary_metric)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "metrics": self.metrics,
            "n_samples": self.n_samples,
            "labels": self.labels,
            "positive_label": self.positive_label,
            "confusion_matrix": self.confusion_matrix,
            "per_class": self.per_class,
            "notes": self.notes,
        }

    def predictions_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({"y_true": self.y_true, "y_pred": self.y_pred})
        if self.task_type is TaskType.REGRESSION:
            frame["residual"] = frame["y_true"] - frame["y_pred"]
        elif self.y_proba is not None and self.labels is not None:
            for i, label in enumerate(self.labels):
                frame[f"proba_{label}"] = self.y_proba[:, i]
        return frame


class Evaluator:
    """Compute task-appropriate metrics for any fitted model."""

    def __init__(self, task_type: TaskType, primary_metric: str | None = None) -> None:
        self.task_type = task_type
        self.primary_metric = validate_primary_metric(task_type, primary_metric or default_primary_metric(task_type))

    # --------------------------------------------------------------- metrics
    def compute_metrics(
        self,
        y_true: Any,
        y_pred: Any,
        y_proba: np.ndarray | None = None,
        classes: list[Any] | None = None,
        notes: list[str] | None = None,
    ) -> dict[str, float | None]:
        notes = notes if notes is not None else []
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if self.task_type is TaskType.REGRESSION:
            y_true = y_true.astype(float)
            y_pred = y_pred.astype(float)
            mse = mean_squared_error(y_true, y_pred)
            return {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(y_true, y_pred)),
            }

        classes = list(classes) if classes is not None else sorted(pd.unique(np.concatenate([y_true, y_pred])))
        metrics: dict[str, float | None] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }
        if self.task_type is TaskType.BINARY:
            positive = classes[-1]
            kwargs: dict[str, Any] = {"pos_label": positive, "zero_division": 0}
        else:
            kwargs = {"average": "macro", "labels": classes, "zero_division": 0}
        metrics["precision"] = float(precision_score(y_true, y_pred, **kwargs))
        metrics["recall"] = float(recall_score(y_true, y_pred, **kwargs))
        metrics["f1"] = float(f1_score(y_true, y_pred, **kwargs))

        if y_proba is None:
            notes.append("Model does not provide probabilities; ROC-AUC and log loss were not computed.")
            return metrics
        try:
            if self.task_type is TaskType.BINARY:
                metrics["roc_auc"] = float(roc_auc_score((y_true == classes[-1]).astype(int), y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", labels=classes))
        except ValueError as exc:
            metrics["roc_auc"] = None
            notes.append(f"ROC-AUC not applicable on this split: {exc}")
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=classes))
        except ValueError as exc:
            metrics["log_loss"] = None
            notes.append(f"Log loss not applicable on this split: {exc}")
        return metrics

    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series, classes: list[Any] | None = None) -> EvaluationResult:
        """Evaluate a fitted model on (X, y)."""
        if len(X) == 0:
            raise ValueError("Cannot evaluate on an empty dataset.")
        y_true = np.asarray(y)
        y_pred = np.asarray(model.predict(X))
        notes: list[str] = []

        if self.task_type is TaskType.REGRESSION:
            metrics = self.compute_metrics(y_true, y_pred, notes=notes)
            return EvaluationResult(
                self.task_type, self.primary_metric, metrics, len(y_true), notes=notes,
                y_true=y_true.astype(float), y_pred=y_pred.astype(float),
            )

        if classes is None:
            model_classes = getattr(model, "classes_", None)
            classes = list(model_classes) if model_classes is not None else sorted(pd.unique(y_true))
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = np.asarray(model.predict_proba(X))
                if y_proba.ndim != 2 or y_proba.shape[1] != len(classes):
                    notes.append("Probability output shape does not match the class list; ignored.")
                    y_proba = None
            except Exception as exc:  # some estimators do not support probabilities
                logger.warning("predict_proba failed: %s", exc)
                y_proba = None

        metrics = self.compute_metrics(y_true, y_pred, y_proba, classes, notes)
        labels = [str(c) for c in classes]
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        report = classification_report(
            y_true, y_pred, labels=classes, target_names=labels, output_dict=True, zero_division=0
        )
        return EvaluationResult(
            task_type=self.task_type,
            primary_metric=self.primary_metric,
            metrics=metrics,
            n_samples=len(y_true),
            labels=labels,
            positive_label=labels[-1] if self.task_type is TaskType.BINARY else None,
            confusion_matrix=cm.astype(int).tolist(),
            per_class={k: v for k, v in report.items() if k in labels},
            notes=notes,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    def baseline_metrics(self, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
        """Scores of a trivial predictor (majority class / training mean) for context."""
        y_test_arr = np.asarray(y_test)
        if self.task_type is TaskType.REGRESSION:
            constant = float(np.mean(np.asarray(y_train, dtype=float)))
            preds = np.full(len(y_test_arr), constant)
            metrics = self.compute_metrics(y_test_arr, preds)
            return {"strategy": "predict training mean", **metrics}
        majority = pd.Series(y_train).value_counts().index[0]
        preds = np.array([majority] * len(y_test_arr))
        classes = sorted(pd.unique(np.asarray(y_train)))
        metrics = self.compute_metrics(y_test_arr, preds, None, classes, notes=[])
        return {"strategy": f"predict majority class '{majority}'", **metrics}

    # -------------------------------------------------------- explainability
    def _scorer(self) -> Callable[[Any, pd.DataFrame, pd.Series], float]:
        spec = METRICS[self.primary_metric]
        sign = 1.0 if spec.greater_is_better else -1.0

        def score(model: Any, X: pd.DataFrame, y: pd.Series) -> float:
            classes = getattr(model, "classes_", None)
            proba = model.predict_proba(X) if spec.needs_proba else None
            metrics = self.compute_metrics(y, model.predict(X), proba, classes, notes=[])
            value = metrics.get(self.primary_metric)
            return sign * float(value) if value is not None else float("nan")

        return score

    def feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5,
        max_rows: int = 2000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Permutation importance on raw input columns, measured with the primary metric.

        Importance = average drop in the primary metric when a column is shuffled
        (positive = the model relies on that column).
        """
        if len(X) > max_rows:
            sample = X.sample(n=max_rows, random_state=random_state)
            X, y = sample, y.loc[sample.index]
        result = permutation_importance(
            model, X, y, scoring=self._scorer(), n_repeats=n_repeats, random_state=random_state, n_jobs=1
        )
        frame = pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        return frame.sort_values("importance", ascending=False, ignore_index=True)
