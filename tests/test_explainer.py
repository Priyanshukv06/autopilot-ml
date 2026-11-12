"""Tests for the SHAP explanations of a Session's final model (optional task 32, Requirement 29)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from app.core.config import build_settings
from app.services.automl_runner import AutoMLRunner, ExperimentConfig, FLAMLEngine
from app.services.explainer import PERMUTATION_ROWS, ExplanationError, explain, explain_session, model_input
from app.services.splits import make_outer_split
from app.services.task_detector import TaskType
from app.services.transforms import DataPipeline, DataStepSpec, PipelineModel
from tests.test_transforms import NumericEngine, frame

ROOT = Path(__file__).resolve().parents[1]


def numeric(n=300, seed=0, classes=2):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "noise": rng.normal(size=n)})
    score = 2 * X["a"] - X["b"]
    y = pd.cut(score, classes, labels=False) if classes > 2 else (score > 0).astype(int)
    return X, pd.Series(y, name="y")


def test_tree_models_get_exact_values_that_add_up_to_the_prediction():
    X, y = numeric()
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    explanation = explain(model, X, "binary", [0, 1], seed=1, max_rows=80)
    assert explanation.method == "tree" and explanation.units == "probability" and explanation.outputs == ["1"]
    assert explanation.values.shape == (80, 3, 1) and len(explanation.row_labels) == 80
    rows = X.loc[explanation.row_labels]
    np.testing.assert_allclose(explanation.values[:, :, 0].sum(axis=1) + explanation.base_values[0],
                               model.predict_proba(rows)[:, 1], atol=1e-6)
    assert explanation.importance()["column"].tolist()[:2] == ["a", "b"]  # the noise column comes last
    one = explanation.row(0)
    assert list(one.columns) == ["column", "value", "shap"] and one["shap"].abs().is_monotonic_decreasing
    summary = explanation.summary()
    assert summary["rows"] == 80 and summary["importance"][0]["column"] == "a" and summary["method"] == "tree"


def test_other_models_use_the_permutation_explainer_on_probabilities():
    X, y = numeric()
    model = LogisticRegression().fit(X, y)
    explanation = explain(model, X, "binary", [0, 1], seed=1)
    assert explanation.method == "permutation" and explanation.units == "probability"
    assert explanation.values.shape == (PERMUTATION_ROWS, 3, 1)
    rows = X.loc[explanation.row_labels]
    # contributions add up to the prediction minus the average prediction over the background rows
    np.testing.assert_allclose(explanation.values[:, :, 0].sum(axis=1) + explanation.base_values[0],
                               model.predict_proba(rows)[:, 1], atol=1e-6)
    assert explanation.importance()["column"].iloc[-1] == "noise"


def test_multiclass_and_regression_outputs():
    X, y = numeric(classes=3)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    multi = explain(model, X, "multiclass", [0, 1, 2], max_rows=50)
    assert multi.outputs == ["0", "1", "2"] and multi.values.shape == (50, 3, 3) and len(multi.base_values) == 3
    assert multi.importance(2)["mean_abs_shap"].sum() > 0
    target = 2 * X["a"] - X["b"]
    regression = explain(RandomForestRegressor(n_estimators=20, random_state=0).fit(X, target), X, "regression",
                         max_rows=50)
    assert regression.outputs == ["prediction"] and regression.units == "predicted value"
    assert regression.values.shape == (50, 3, 1)


def test_the_columns_are_the_models_inputs_after_the_data_steps_and_flaml(tmp_path):
    df = pd.read_csv(ROOT / "data" / "sample" / "titanic.csv").drop(columns=["PassengerId", "Ticket"])
    X, y = df.drop(columns="Survived"), df["Survived"]
    pipeline = DataPipeline([DataStepSpec("token_features", ("Name",))], task_type="binary")
    X_used = pipeline.fit_transform(X, y)
    result = FLAMLEngine().fit(X_used, y, task_type=TaskType.BINARY, metric="f1", time_budget=10, seed=1, n_jobs=2,
                               estimator_list=["lgbm"], work_dir=tmp_path, max_iter=3)
    model = PipelineModel(pipeline, result.model)
    estimator, data = model_input(model, X.head(5))
    assert type(estimator).__name__ == "LGBMClassifier" and "Name__tok_mr" in data.columns and "Name" not in data
    explanation = explain(model, X, "binary", result.classes, max_rows=60)
    assert explanation.method == "tree" and explanation.units == "log-odds" and "Name__tok_mr" in explanation.columns
    assert explanation.importance()["column"].iloc[0] in ("Sex", "Name__tok_mr")


def test_a_sessions_final_model_is_explained_on_its_training_part_only(tmp_path):
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    df = frame(200)
    config = ExperimentConfig(target_column="label", task_type=TaskType.BINARY, time_budget=10,
                              data_steps=[DataStepSpec("frequency_encode", ("city",))])
    result = AutoMLRunner(settings=settings, engine=NumericEngine(), enable_mlflow=False).run_experiment(df, config)
    split, _ = make_outer_split(df["label"], TaskType.BINARY, 0.2, 42)
    record = {"session_id": "SES-001", "final_experiment_id": result.experiment_id,
              "settings": {"base": {"target_column": "label", "task_type": "binary", "primary_metric": "f1",
                                    "test_size": 0.2, "random_seed": 42, "excluded_columns": []}},
              "splits": {"outer": split.fingerprint()}}
    explanation = explain_session(record, df, settings)
    train_labels = set(df.index[split.train_positions])
    assert set(explanation.row_labels) <= train_labels  # never a Test_Part row
    assert explanation.columns == ["city", "age"] and explanation.method == "permutation"
    with pytest.raises(ExplanationError, match="not the Session's data"):
        explain_session(record, df.iloc[::-1].reset_index(drop=True), settings)
    with pytest.raises(ExplanationError, match="no final model"):
        explain_session({**record, "final_experiment_id": None}, df, settings)
