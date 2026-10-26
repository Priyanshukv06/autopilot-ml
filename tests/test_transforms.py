"""Tests for the Data_Steps (task 12.1 of the spec)."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression

from app.core.config import build_settings
from app.services.assessor import Assessor
from app.services.automl_runner import AutoMLRunner, EngineResult, ExperimentConfig
from app.services.splits import make_outer_split, make_session_folds
from app.services.task_detector import TaskType
from app.services.transforms import (
    OTHER,
    DataPipeline,
    DataStepError,
    DataStepSpec,
    PipelineModel,
    as_specs,
)

FREQ = DataStepSpec("frequency_encode", ("city",))
RARE = DataStepSpec("group_rare_categories", ("city",), {"min_share": 0.25})
MISSING = DataStepSpec("missing_indicator", ("age",))
DROP = DataStepSpec("drop_columns", ("note",))


def train_rows():
    return pd.DataFrame({
        "city": ["a", "a", "a", "b", "b", "c", None, "a"],
        "age": [20.0, np.nan, 31.0, 40.0, np.nan, 18.0, 25.0, 60.0],
        "note": ["x"] * 8,
    })


def other_rows():
    return pd.DataFrame({"city": ["a", "c", "z", None], "age": [np.nan, 1.0, 2.0, 3.0], "note": ["y"] * 4})


# --------------------------------------------------------------------------- each step
def test_frequency_encode():
    pipeline = DataPipeline([FREQ]).fit(train_rows())
    out = pipeline.transform(other_rows())
    # Shares of the 8 training rows: a 4/8, b 2/8, c 1/8 (the empty cell counts in the total only).
    assert out["city"].tolist()[:3] == [0.5, 0.125, 0.0]  # "z" was never seen: 0
    assert np.isnan(out["city"].iloc[3])  # empty stays empty
    assert is_numeric_dtype(out["city"]) and out["age"].equals(other_rows()["age"])


def test_group_rare_categories():
    pipeline = DataPipeline([RARE]).fit(train_rows())
    out = pipeline.transform(other_rows())
    # Kept: values in at least 25% of training rows, so only "a" (4 of 8) and "b" (2 of 8).
    assert out["city"].tolist()[:3] == ["a", OTHER, OTHER]
    assert pd.isna(out["city"].iloc[3])
    assert DataPipeline([DataStepSpec("group_rare_categories", ("city",))]).fit(train_rows()).state[0]["kept"] == {
        "city": ["a", "b", "c"]}  # the default 1% keeps every value seen


def test_missing_indicator_and_drop_columns():
    out = DataPipeline([MISSING, DROP]).fit(train_rows()).transform(other_rows())
    assert out["age__missing"].tolist() == [1, 0, 0, 0]
    assert out["age"].isna().tolist() == [True, False, False, False]  # the original column is kept
    assert "note" not in out.columns and list(out.columns) == ["city", "age", "age__missing"]


def test_steps_run_in_order_on_the_output_of_the_steps_before():
    pipeline = DataPipeline([RARE, FREQ]).fit(train_rows())  # rare values grouped, then encoded
    out = pipeline.transform(other_rows())
    # Grouping keeps "a" (4 of 8 rows) and "b" (2 of 8) and turns "c" into __other__ (1 of 8), so the
    # encoding learns a 0.5, b 0.25, __other__ 0.125. "c" and the unseen "z" both become __other__.
    assert out["city"].tolist()[:3] == [0.5, 0.125, 0.125]


# --------------------------------------------------------------------------- fitted rows only
def test_a_step_never_reads_rows_it_was_not_fitted_on():
    train = train_rows()
    pipeline = DataPipeline([RARE, FREQ, MISSING, DROP]).fit(train)
    state = copy.deepcopy(pipeline.state)
    other = other_rows()
    together = pipeline.transform(other)
    assert pipeline.state == state  # applying the steps learns nothing
    # Each output row depends only on its own input row and the fitted state.
    one_by_one = pd.concat([pipeline.transform(other.iloc[[i]]) for i in range(len(other))])
    pd.testing.assert_frame_equal(together, one_by_one)
    # Changing the other rows changes nothing that was learned: refitting on the same training rows gives the same state.
    assert DataPipeline([RARE, FREQ, MISSING, DROP]).fit(train).state == state


def test_the_assessor_fits_the_steps_on_each_folds_training_rows():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"city": rng.choice(list("abcdefgh"), size=80), "x": rng.normal(size=80)})
    y = pd.Series((X["x"] > 0).astype(int))
    seen = []

    class RecordingEngine:
        def fit_config(self, X_train, y_train, **kwargs):
            seen.append(X_train)
            return LogisticRegression().fit(X_train, y_train)

    folds, _ = make_session_folds(y, TaskType.BINARY, n_splits=4, n_repeats=1, seed=0)
    config = ExperimentConfig(target_column="t", data_steps=[FREQ])
    result = type("R", (), {"setup_key": "k", "experiment_id": "EXP-1", "best_estimator": "lr", "best_config": {}})()
    assessment = Assessor(RecordingEngine(), folds, TaskType.BINARY, "f1", 1, 0).assess(X, y, config, result)
    assert not assessment.failed and len(seen) == 4
    for fitted, (train_positions, _) in zip(seen, folds.folds):
        fold_rows = X.iloc[train_positions]
        expected = (fold_rows["city"].value_counts() / len(fold_rows)).to_dict()
        assert fitted["city"].tolist() == [expected[v] for v in fold_rows["city"]]  # shares of this fold only


# --------------------------------------------------------------------------- specs and errors
def test_specs_are_json_safe_and_round_trip():
    for spec in (FREQ, RARE, MISSING, DROP):
        data = spec.to_dict()
        assert DataStepSpec.from_dict(json.loads(json.dumps(data))) == spec
    assert as_specs([RARE.to_dict(), FREQ]) == [RARE, FREQ]
    assert ExperimentConfig(target_column="t", data_steps=[RARE]).setup_key() == \
        ExperimentConfig(target_column="t", data_steps=[RARE.to_dict()]).setup_key()
    assert ExperimentConfig(target_column="t", data_steps=[RARE]).setup_key() != \
        ExperimentConfig(target_column="t", data_steps=[FREQ]).setup_key()
    assert "fewer than 25%" in RARE.describe()


@pytest.mark.parametrize(
    "make",
    [
        lambda: DataStepSpec("scale", ("a",)),
        lambda: DataStepSpec("frequency_encode", ()),
        lambda: DataStepSpec("frequency_encode", ("a", "a")),
        lambda: DataStepSpec("frequency_encode", ("a",), {"min_share": 0.1}),
        lambda: DataStepSpec("group_rare_categories", ("a",), {"min_share": 1.5}),
    ],
)
def test_invalid_specs_are_refused(make):
    with pytest.raises(DataStepError):
        make()


@pytest.mark.parametrize(
    "step, frame, message",
    [
        (DataStepSpec("frequency_encode", ("nope",)), train_rows(), "not in the data"),
        (DataStepSpec("group_rare_categories", ("age",)), train_rows(), "is numeric"),
        (DataStepSpec("missing_indicator", ("city",)), train_rows(), "not numeric"),
        (MISSING, train_rows().assign(age__missing=0), "already exists"),
    ],
)
def test_steps_that_do_not_fit_the_data_are_refused(step, frame, message):
    with pytest.raises(DataStepError, match=message):
        DataPipeline([step]).fit(frame)


def test_applying_unfitted_steps_or_to_rows_without_the_column_fails():
    with pytest.raises(DataStepError, match="fitted"):
        DataPipeline([FREQ]).transform(other_rows())
    pipeline = DataPipeline([FREQ]).fit(train_rows())
    with pytest.raises(DataStepError, match="not in the data"):
        pipeline.transform(other_rows().drop(columns="city"))


# --------------------------------------------------------------------------- the runner
class NumericEngine:
    """Logistic regression; the Data_Steps must make every feature numeric for it."""

    name = "numeric"

    def fit(self, X_train, y_train, **kwargs):
        self.searched_columns = list(X_train.columns)
        model = LogisticRegression(max_iter=2000).fit(X_train.fillna(0), y_train)
        wrapped = FillModel(model)
        return EngineResult(model=wrapped, engine_name=self.name, engine_version="test", best_estimator="lr",
                            best_model_name="Logistic", best_config={}, best_validation_score=None,
                            validation_method="none", training_time=0.0, search_history=[], leaderboard=[],
                            native_importance=None, classes=list(model.classes_))


class FillModel:
    def __init__(self, model):
        self.model, self.classes_ = model, model.classes_

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.model.predict(X.fillna(0))

    def predict_proba(self, X):
        return self.model.predict_proba(X.fillna(0))


def frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"city": rng.choice(["north", "south", "east", "west", "rare"], size=n, p=[.4, .3, .2, .09, .01]),
                       "age": rng.normal(40, 10, size=n)})
    df.loc[::7, "age"] = np.nan
    df["label"] = np.where((df["city"] == "north") ^ (rng.random(n) < 0.2), "yes", "no")
    return df


@pytest.fixture
def runner(tmp_path):
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    return AutoMLRunner(settings=settings, engine=NumericEngine(), enable_mlflow=False)


def test_a_reloaded_bundle_predicts_the_same_from_raw_rows(runner):
    df = frame()
    config = ExperimentConfig(target_column="label", task_type=TaskType.BINARY, time_budget=10,
                              data_steps=[DataStepSpec("frequency_encode", ("city",)), MISSING])
    result = runner.run_experiment(df, config)
    assert runner.engine.searched_columns == ["city", "age", "age__missing"]  # the search saw the changed columns
    assert result.feature_columns == ["city", "age"]  # the record keeps the original columns
    bundle = AutoMLRunner.load_model(result.model_path)
    assert isinstance(bundle["model"], PipelineModel)
    assert [s["kind"] for s in bundle["data_steps"]] == ["frequency_encode", "missing_indicator"]
    # Before saving, the runner predicted the test rows with the in-memory model; the reloaded model,
    # given the same raw test rows, predicts exactly the same.
    split, _ = make_outer_split(df["label"], TaskType.BINARY, 0.2, 42)
    raw_test = df.drop(columns="label").iloc[split.test_positions]
    np.testing.assert_array_equal(bundle["model"].predict(raw_test), result.test_evaluation.y_pred)
    np.testing.assert_allclose(bundle["model"].predict_proba(raw_test), result.test_evaluation.y_proba)


def test_importance_is_reported_on_the_original_columns(runner):
    config = ExperimentConfig(target_column="label", task_type=TaskType.BINARY, time_budget=10,
                              data_steps=[DataStepSpec("frequency_encode", ("city",)), MISSING])
    result = runner.run_experiment(frame(), config)
    assert sorted(row["feature"] for row in result.feature_importance) == ["age", "city"]


def test_without_data_steps_nothing_changes(runner, tmp_path):
    df = frame().assign(city=lambda d: d["city"].map({"north": 1.0, "south": 2.0, "east": 3.0, "west": 4.0,
                                                      "rare": 5.0}))
    config = ExperimentConfig(target_column="label", task_type=TaskType.BINARY, time_budget=10)
    result = runner.run_experiment(df, config)
    bundle = AutoMLRunner.load_model(result.model_path)
    assert not isinstance(bundle["model"], PipelineModel)
    assert "data_steps" not in bundle and "pipeline" not in bundle
    assert runner.engine.searched_columns == ["city", "age"] and result.setup["data_steps"] == []
