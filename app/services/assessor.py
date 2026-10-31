"""Assessor: scores a Setup by refitting its winning configuration on the Session_Folds.

FLAML's own validation score is optimistic and comes from FLAML's folds, so the decision steps
use these refit scores instead (Requirement 4). Every Assessment of a Session uses the same
folds of the Training_Part, so two Setups are always compared on the same rows. The Assessor is
only ever given the Training_Part.

A Setup's Data_Steps are fitted again on each fold's training rows, so no fold's validation rows
shape the data its model is trained on.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from app.core.logging_config import get_logger
from app.services.automl_runner import (
    AutoMLEngine,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    to_jsonable,
    unseen_category_shares,
)
from app.services.evaluator import Evaluator
from app.services.splits import SessionFolds
from app.services.task_detector import TaskType
from app.services.transforms import DataPipeline, PipelineModel, as_specs

logger = get_logger(__name__)

# Two refits of the same configuration gave identical predictions in task 1 (largest difference
# 3.3e-16), so scores of a repeated Assessment must agree to this tolerance.
REPEAT_TOLERANCE = 1e-12

Part = Literal["train", "validation", "baseline"]
_PART_FIELDS = {"train": "train_metrics", "validation": "validation_metrics", "baseline": "baseline_validation"}


@dataclass
class FoldScore:
    repeat: int
    fold: int
    train_metrics: dict[str, float | None]  # every task metric on the fold's training rows
    validation_metrics: dict[str, float | None]  # every task metric on the fold's validation rows
    class_recall: dict[str, float] | None  # validation recall per class (classification)
    unseen_share: dict[str, float]  # per categorical column: share of validation rows not seen in training rows
    baseline_validation: dict[str, float | None]  # trivial baseline on the same fold
    fit_seconds: float


@dataclass
class Assessment:
    setup_key: str
    experiment_id: str
    estimator: str
    config: dict[str, Any]
    primary_metric: str
    folds_fingerprint: str
    folds: list[FoldScore] = field(default_factory=list)
    seconds: float = 0.0
    failed: bool = False
    error: str | None = None
    reused_from: str | None = None  # the Experiment whose identical refits these fold scores are (since 31 Oct 2026)

    def values(self, metric: str, part: Part) -> np.ndarray:
        """One value per fold, in fold order; NaN where the metric could not be computed."""
        key = _PART_FIELDS[part]
        values = [getattr(fold, key).get(metric) for fold in self.folds]
        return np.array([np.nan if v is None else float(v) for v in values], dtype=float)

    def summary(self) -> dict[str, Any]:
        """Means and standard deviations (ddof=1) over the folds, for records."""
        metrics = sorted({m for fold in self.folds for m in fold.validation_metrics})
        parts: dict[str, Any] = {}
        for part in _PART_FIELDS:
            parts[part] = {}
            for metric in metrics:
                values = self.values(metric, part)
                values = values[np.isfinite(values)]
                parts[part][metric] = {
                    "mean": float(values.mean()) if len(values) else None,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                }
        return {
            "setup_key": self.setup_key,
            "experiment_id": self.experiment_id,
            "estimator": self.estimator,
            "primary_metric": self.primary_metric,
            "n_folds": len(self.folds),
            "seconds": round(self.seconds, 2),
            "failed": self.failed,
            "error": self.error,
            **parts,
        }

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Assessment:
        """Rebuild an Assessment from ``to_dict()`` output (an Experiment's ``assessment.json``)."""
        folds = [FoldScore(**fold) for fold in data.get("folds", [])]
        return cls(**{**{k: v for k, v in data.items() if k != "folds"}, "folds": folds})


class Assessor:
    """Refits Setups on fixed folds and keeps the Assessments of one Session."""

    def __init__(
        self,
        engine: AutoMLEngine,
        folds: SessionFolds,
        task_type: TaskType,
        primary_metric: str,
        n_jobs: int,
        seed: int,
    ) -> None:
        self.engine = engine
        self.folds = folds
        self.task_type = task_type
        self.primary_metric = primary_metric
        self.n_jobs = n_jobs
        self.seed = seed
        self.n_rows = sum(len(validation) for _, validation in folds.folds[: folds.n_splits])
        self._cache: dict[str, Assessment] = {}

    @staticmethod
    def refit_key(columns: Any, config: ExperimentConfig, estimator: str, best_config: dict[str, Any]) -> str:
        """What the refits depend on: the feature columns, the Setup's Data_Steps and class weights, and the
        winning learner and configuration. The rest of a Setup (learner list, search metric, search-space
        limits, time) only changes what the search finds, so two Setups with the same key give the same fold
        scores. Since 31 October 2026; before, the whole setup key was used, so a Setup that only searched
        fewer learners and found the same model was refitted 15 times for the same scores."""
        steps = [spec.to_dict() for spec in as_specs(config.data_steps)]
        return json.dumps([[str(c) for c in columns], steps, config.class_weight, estimator, to_jsonable(best_config)],
                          sort_keys=True)

    def cached(self, X_train: pd.DataFrame, config: ExperimentConfig, result: ExperimentResult) -> Assessment | None:
        return self._cache.get(self.refit_key(X_train.columns, config, result.best_estimator, result.best_config))

    def assess(
        self, X_train: pd.DataFrame, y_train: pd.Series, config: ExperimentConfig, result: ExperimentResult
    ) -> Assessment:
        """Assess the winning configuration of ``result``, or reuse the fold scores of an earlier Assessment
        with the same refits (a copy that names this Experiment and the one it was made for).

        ``X_train`` and ``y_train`` must be the Training_Part the folds were made on, in the same order.
        """
        key = self.refit_key(X_train.columns, config, result.best_estimator, result.best_config)
        if key in self._cache:
            earlier = self._cache[key]
            if earlier.experiment_id == result.experiment_id:
                return earlier
            logger.info("%s refits like %s: reusing its %d fold scores", result.experiment_id,
                        earlier.experiment_id, len(earlier.folds))
            return replace(earlier, setup_key=result.setup_key or "", experiment_id=result.experiment_id,
                           folds=list(earlier.folds), seconds=0.0, reused_from=earlier.experiment_id)
        if len(X_train) != self.n_rows or len(y_train) != self.n_rows:
            raise ExperimentError(
                f"The Assessor needs the {self.n_rows} training rows its folds were made on "
                f"(got {len(X_train)} rows and {len(y_train)} labels)."
            )

        evaluator = Evaluator(self.task_type, self.primary_metric)
        assessment = Assessment(
            setup_key=result.setup_key or "",
            experiment_id=result.experiment_id,
            estimator=result.best_estimator,
            config=dict(result.best_config),
            primary_metric=self.primary_metric,
            folds_fingerprint=self.folds.fingerprint(),
        )
        start = time.perf_counter()
        for index, (train_positions, validation_positions) in enumerate(self.folds.folds):
            repeat, fold = divmod(index, self.folds.n_splits)
            try:
                assessment.folds.append(
                    self._score_fold(evaluator, X_train, y_train, config, result, repeat, fold,
                                     train_positions, validation_positions)
                )
            except Exception as exc:  # one failed fold fails the Assessment (Requirement 4.6)
                assessment.failed = True
                assessment.error = f"Fold {fold + 1} of repeat {repeat + 1} failed: {exc}"
                logger.warning("Assessment of %s failed: %s", result.experiment_id, assessment.error)
                break
        assessment.seconds = time.perf_counter() - start
        self._cache[key] = assessment
        logger.info(
            "Assessed %s (%s): %d folds in %.1fs%s", result.experiment_id, result.best_estimator,
            len(assessment.folds), assessment.seconds, " (failed)" if assessment.failed else "",
        )
        return assessment

    def _score_fold(
        self,
        evaluator: Evaluator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: ExperimentConfig,
        result: ExperimentResult,
        repeat: int,
        fold: int,
        train_positions: np.ndarray,
        validation_positions: np.ndarray,
    ) -> FoldScore:
        X_fit, y_fit = X_train.iloc[train_positions], y_train.iloc[train_positions]
        X_valid, y_valid = X_train.iloc[validation_positions], y_train.iloc[validation_positions]
        weights = compute_sample_weight("balanced", y_fit) if config.class_weight == "balanced" else None

        start = time.perf_counter()
        # The Setup's Data_Steps are fitted again on this fold's training rows only (Requirement 3.3).
        steps = as_specs(config.data_steps)
        pipeline = DataPipeline(steps).fit(X_fit) if steps else None
        model = self.engine.fit_config(
            pipeline.transform(X_fit) if pipeline is not None else X_fit, y_fit, task_type=self.task_type,
            estimator=result.best_estimator, config=result.best_config, seed=self.seed, n_jobs=self.n_jobs,
            sample_weight=weights,
        )
        if pipeline is not None:  # scored, like the search's model, on the original columns
            model = PipelineModel(pipeline, model)
        fit_seconds = time.perf_counter() - start

        train_eval = evaluator.evaluate(model, X_fit, y_fit)
        valid_eval = evaluator.evaluate(model, X_valid, y_valid)
        baseline = evaluator.baseline_metrics(y_fit, y_valid)
        recall = None
        if self.task_type.is_classification:
            recall = {label: float(stats["recall"]) for label, stats in valid_eval.per_class.items()}
        return FoldScore(
            repeat=repeat,
            fold=fold,
            train_metrics=dict(train_eval.metrics),
            validation_metrics=dict(valid_eval.metrics),
            class_recall=recall,
            unseen_share=unseen_category_shares(X_fit, X_valid),
            baseline_validation={k: v for k, v in baseline.items() if k != "strategy"},
            fit_seconds=fit_seconds,
        )
