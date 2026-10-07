"""Tests for the AutoML runner, the FLAML engine integration and the evaluator.

FLAML runs use very small time budgets so the whole suite stays fast; they check
the plumbing (split -> search -> evaluate -> save -> track), not model quality.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.core.config import build_settings
from app.services.automl_runner import AutoMLRunner, EngineResult, ExperimentError
from app.services.evaluator import Evaluator, metric_warnings, validate_primary_metric
from app.services.profiler import ClassBalance
from app.services.task_detector import TaskType


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def settings(tmp_path):
    """Settings rooted in a temporary directory (no writes to the real project)."""
    s = build_settings(tmp_path)
    s.ensure_directories()
    return s


def make_classification_frame(n: int = 300, n_classes: int = 2, seed: int = 0) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=0, n_classes=n_classes,
        n_clusters_per_class=1, random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["segment"] = np.where(df["f0"] > 0, "high", "low")  # categorical feature
    df.loc[::17, "f1"] = np.nan  # some missing values
    df["row_id"] = np.arange(n)
    labels = np.array(["neg", "pos"]) if n_classes == 2 else np.array(["a", "b", "c"])
    df["label"] = labels[y]
    return df


def breast_cancer_frame() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["diagnosis"] = df.pop("target").map({0: "malignant", 1: "benign"})
    return df


# --------------------------------------------------------------------------- FLAML runs
def test_binary_classification_end_to_end(settings):
    df = make_classification_frame()
    runner = AutoMLRunner(settings=settings)
    stages: list[str] = []
    result = runner.run(
        df, "label", time_budget=5, excluded_columns=["row_id"], dataset_name="synthetic",
        random_seed=0, progress_callback=lambda stage, fraction, message: stages.append(stage),
    )

    assert result.experiment_id == "EXP-001"
    assert result.task_type is TaskType.BINARY
    assert result.primary_metric == "f1"
    assert result.engine == "flaml" and result.best_estimator
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"} <= set(result.test_metrics)
    assert 0.0 <= result.test_score <= 1.0
    assert result.test_metrics["roc_auc"] > 0.6  # informative synthetic data
    assert result.n_trials > 0 and result.leaderboard and result.search_history
    assert "row_id" not in result.feature_columns
    assert result.feature_importance and {"feature", "importance", "std"} <= set(result.feature_importance[0])
    assert stages[0] == "preparing" and stages[-1] == "done"

    # Saved model reloads and predicts original labels.
    bundle = AutoMLRunner.load_model(result.model_path)
    predictions = bundle["model"].predict(df[bundle["feature_columns"]].head(10))
    assert set(predictions) <= {"neg", "pos"}

    # Local experiment record and artifacts.
    artifact_dir = Path(result.artifact_dir)
    for name in ("summary.json", "profile.json", "evaluation.json", "predictions.csv", "leaderboard.csv"):
        assert (artifact_dir / name).exists(), name
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["test_score"] == pytest.approx(result.test_score)
    assert list(runner.tracker.history_frame()["Experiment"]) == ["EXP-001"]

    # MLflow run (local SQLite store inside tmp_path).
    assert result.mlflow_run_id
    run = runner.tracker.client.get_run(result.mlflow_run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["target"] == "label"
    assert run.data.metrics["test_f1"] == pytest.approx(result.test_metrics["f1"])


def test_multiclass_with_roc_auc_metric(settings):
    df = make_classification_frame(n=300, n_classes=3)
    runner = AutoMLRunner(settings=settings, enable_mlflow=False)
    result = runner.run(
        df, "label", primary_metric="roc_auc", time_budget=5, estimator_list=["lgbm"],
        excluded_columns=["row_id"],
    )
    assert result.task_type is TaskType.MULTICLASS
    assert result.primary_metric == "roc_auc"
    assert result.best_estimator == "lgbm"
    assert result.test_evaluation.labels == ["a", "b", "c"]
    assert np.array(result.test_evaluation.confusion_matrix).shape == (3, 3)
    assert result.mlflow_run_id is None


def test_regression_without_mlflow(settings):
    X, y = make_regression(n_samples=300, n_features=5, noise=10.0, random_state=0)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    df["price"] = y
    runner = AutoMLRunner(settings=settings, enable_mlflow=False)
    result = runner.run(df, "price", time_budget=5, estimator_list=["lgbm"], dataset_name="synthetic-reg")

    assert result.task_type is TaskType.REGRESSION
    assert result.primary_metric == "r2"
    assert set(result.test_metrics) == {"mae", "mse", "rmse", "r2"}
    assert result.test_metrics["rmse"] == pytest.approx(np.sqrt(result.test_metrics["mse"]))
    assert result.baseline_metrics["strategy"] == "predict training mean"
    assert "residual" in result.test_evaluation.predictions_frame()


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_column": "missing"},
        {"target_column": "label", "excluded_columns": ["not_a_column"]},
        {"target_column": "label", "task": "regression"},  # string target
        {"target_column": "label", "primary_metric": "r2"},  # metric of another task
        {"target_column": "label", "time_budget": 1},
        {"target_column": "label", "test_size": 0.9},
        {"target_column": "label", "excluded_columns": ["f0", "f1", "f2", "f3", "f4", "f5", "segment", "row_id"]},
    ],
)
def test_invalid_configurations_fail_fast(settings, kwargs):
    runner = AutoMLRunner(settings=settings, enable_mlflow=False)
    with pytest.raises(ExperimentError):
        runner.run(make_classification_frame(), **kwargs)


def test_binary_override_on_multiclass_target_fails(settings):
    runner = AutoMLRunner(settings=settings, enable_mlflow=False)
    with pytest.raises(ExperimentError, match="exactly 2 classes"):
        runner.run(make_classification_frame(n_classes=3), "label", task="binary")


def test_too_few_rows(settings):
    runner = AutoMLRunner(settings=settings, enable_mlflow=False)
    with pytest.raises(ExperimentError, match="At least"):
        runner.run(make_classification_frame().head(10), "label")


# --------------------------------------------------------------------------- engine abstraction
class LogisticEngine:
    """Minimal non-FLAML engine: proves the runner is engine-agnostic."""

    name = "sklearn-logistic"

    def fit(self, X_train, y_train, *, task_type, metric, time_budget, seed, n_jobs, estimator_list, work_dir):
        model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
        return EngineResult(
            model=model, engine_name=self.name, engine_version="test", best_estimator="logreg",
            best_model_name="Logistic Regression", best_config={"C": 1.0}, best_validation_score=None,
            validation_method="none", training_time=0.0, search_history=[], leaderboard=[],
            native_importance=None, classes=list(model.classes_),
        )


class FailingEngine:
    name = "failing"

    def fit(self, *args, **kwargs):
        raise RuntimeError("engine exploded")


def test_runner_accepts_custom_engine(settings):
    runner = AutoMLRunner(settings=settings, engine=LogisticEngine(), enable_mlflow=False)
    result = runner.run(breast_cancer_frame(), "diagnosis", task="binary", primary_metric="roc_auc")
    assert result.engine == "sklearn-logistic"
    assert result.test_evaluation.positive_label == "malignant"
    assert result.test_metrics["roc_auc"] > 0.9


def test_failed_run_leaves_error_record(settings):
    runner = AutoMLRunner(settings=settings, engine=FailingEngine(), enable_mlflow=False)
    with pytest.raises(RuntimeError, match="engine exploded"):
        runner.run(breast_cancer_frame(), "diagnosis")
    error = json.loads((settings.experiments_dir / "EXP-001" / "error.json").read_text(encoding="utf-8"))
    assert error["status"] == "failed" and "engine exploded" in error["error"]
    assert runner.tracker.history_frame().empty  # failed runs are not listed as completed


# --------------------------------------------------------------------------- evaluator
def test_evaluator_binary_metrics_match_sklearn():
    df = breast_cancer_frame()
    X, y = df.drop(columns="diagnosis"), df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=5000).fit(X_train, y_train)

    result = Evaluator(TaskType.BINARY, "f1").evaluate(model, X_test, y_test)
    proba = model.predict_proba(X_test)[:, list(model.classes_).index("malignant")]
    assert result.positive_label == "malignant"
    assert result.metrics["accuracy"] == pytest.approx(accuracy_score(y_test, model.predict(X_test)))
    assert result.metrics["f1"] == pytest.approx(f1_score(y_test, model.predict(X_test), pos_label="malignant"))
    assert result.metrics["roc_auc"] == pytest.approx(roc_auc_score(y_test == "malignant", proba))
    assert np.array(result.confusion_matrix).sum() == len(y_test)
    assert result.primary_score == result.metrics["f1"]


def test_evaluator_regression_metrics_and_baseline():
    X, y = make_regression(n_samples=200, n_features=3, noise=5.0, random_state=1)
    X = pd.DataFrame(X, columns=["a", "b", "c"])
    y = pd.Series(y)
    model = LinearRegression().fit(X[:150], y[:150])
    evaluator = Evaluator(TaskType.REGRESSION)
    result = evaluator.evaluate(model, X[150:], y[150:])
    predictions = model.predict(X[150:])
    assert evaluator.primary_metric == "r2"
    assert result.metrics["r2"] == pytest.approx(r2_score(y[150:], predictions))
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(mean_squared_error(y[150:], predictions)))
    baseline = evaluator.baseline_metrics(y[:150], y[150:])
    assert baseline["r2"] < result.metrics["r2"]


def test_majority_baseline_and_feature_importance():
    df = breast_cancer_frame()
    X, y = df.drop(columns="diagnosis"), df["diagnosis"]
    model = LogisticRegression(max_iter=5000).fit(X, y)
    evaluator = Evaluator(TaskType.BINARY, "accuracy")
    baseline = evaluator.baseline_metrics(y, y)
    assert baseline["accuracy"] == pytest.approx((y == "benign").mean())
    importance = evaluator.feature_importance(model, X, y, n_repeats=2)
    assert len(importance) == X.shape[1]
    assert importance["importance"].is_monotonic_decreasing


def test_metric_validation_and_imbalance_warning():
    with pytest.raises(ValueError):
        validate_primary_metric(TaskType.REGRESSION, "accuracy")
    severe = ClassBalance("no", 0.92, "yes", 0.08, 11.5, "severe")
    mild = ClassBalance("no", 0.6, "yes", 0.4, 1.5, "mild")
    assert any("misleading" in w for w in metric_warnings(TaskType.BINARY, "accuracy", severe))
    assert metric_warnings(TaskType.BINARY, "f1", mild) == []
    assert metric_warnings(TaskType.REGRESSION, "r2", None) == []
