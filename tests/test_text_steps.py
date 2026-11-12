"""Tests for target encoding and word features (optional task 30, Requirement 27)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.core.config import build_settings
from app.services.assessor import Assessor
from app.services.automl_runner import AutoMLRunner, ExperimentConfig, ExperimentError
from app.services.critic import Critic, CriticThresholds, DiagnosisKind, Severity
from app.services.experiment_planner import Planner
from app.services.profiler import DatasetProfiler
from app.services.splits import make_session_folds
from app.services.strategies import CATALOGUE
from app.services.strategy_engine import propose
from app.services.task_detector import TaskType
from app.services.transforms import (
    DEFAULT_SMOOTHING,
    DataPipeline,
    DataStepError,
    DataStepSpec,
    frequent_tokens,
    tokenize,
)
from tests.test_critic import assessment, fold
from tests.test_transforms import NumericEngine

ROOT = Path(__file__).resolve().parents[1]
TARGET = DataStepSpec("target_encode", ("city",))
TOKENS = DataStepSpec("token_features", ("name",))
ENCODE, WORDS = CATALOGUE["target_encode"], CATALOGUE["token_features"]


def rows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    city = rng.choice(["north", "south", "east"], size=n).astype(object)
    city[::9] = None
    y = pd.Series(np.where((city == "north") ^ (rng.random(n) < 0.1), "yes", "no"), name="y")
    return pd.DataFrame({"city": city, "x": rng.normal(size=n)}), y


def titanic():
    return pd.read_csv(ROOT / "data" / "sample" / "titanic.csv")


# --------------------------------------------------------------------------- target_encode
def test_no_rows_encoding_uses_its_own_label():
    X, y = rows()
    crossed = DataPipeline([TARGET], task_type="binary").fit_transform(X, y)
    for i in range(len(X)):  # flipping one label never changes that row's own encoding
        flipped = y.copy()
        flipped.iloc[i] = "no" if y.iloc[i] == "yes" else "yes"
        again = DataPipeline([TARGET], task_type="binary").fit_transform(X, flipped)
        assert again["city"].iloc[i] == crossed["city"].iloc[i] or np.isnan(crossed["city"].iloc[i])
    assert crossed["city"].isna().tolist() == X["city"].isna().tolist()  # empty cells stay empty


def test_other_rows_get_the_smoothed_mean_of_all_fitted_rows():
    X, y = rows()
    pipeline = DataPipeline([TARGET], task_type="binary").fit(X, y)
    positive = (y == "yes").astype(float)  # "yes" is the last sorted label, FLAML's positive class
    prior = positive.mean()
    north = X["city"] == "north"
    expected = (positive[north].sum() + DEFAULT_SMOOTHING * prior) / (north.sum() + DEFAULT_SMOOTHING)
    out = pipeline.transform(pd.DataFrame({"city": ["north", "west", None], "x": [0.0, 0.0, 0.0]}))
    assert out["city"].iloc[0] == pytest.approx(expected)
    assert out["city"].iloc[1] == pytest.approx(prior) and np.isnan(out["city"].iloc[2])  # unseen, empty
    # fit_transform (cross-fitted) and fit + transform (all rows) differ on the fitted rows
    assert not np.allclose(pipeline.transform(X)["city"].fillna(0), DataPipeline([TARGET], task_type="binary")
                           .fit_transform(X, y)["city"].fillna(0))


def test_multiclass_gets_one_column_per_class_and_regression_the_mean():
    df = pd.read_csv(ROOT / "data" / "sample" / "penguins.csv").dropna(subset=["species"])
    step = DataStepSpec("target_encode", ("island",))
    out = DataPipeline([step], task_type="multiclass").fit_transform(df.drop(columns="species"), df["species"])
    names = ["island__te_Adelie", "island__te_Chinstrap", "island__te_Gentoo"]
    assert "island" not in out.columns and list(out.columns[-3:]) == names
    np.testing.assert_allclose(out[names].sum(axis=1), 1.0)  # class shares of a row add up to 1
    X, _ = rows()
    value = pd.Series(np.arange(len(X), dtype=float))
    pipeline = DataPipeline([TARGET], task_type="regression").fit(X, value)
    north = X["city"] == "north"
    expected = (value[north].sum() + DEFAULT_SMOOTHING * value.mean()) / (north.sum() + DEFAULT_SMOOTHING)
    out = pipeline.transform(pd.DataFrame({"city": ["west", "north"], "x": [0.0, 0.0]}))
    assert out["city"].tolist() == pytest.approx([value.mean(), expected])


def test_target_encode_needs_the_target_the_task_and_a_text_column():
    X, y = rows()
    with pytest.raises(DataStepError, match="target values"):
        DataPipeline([TARGET], task_type="binary").fit(X)
    with pytest.raises(DataStepError, match="task type"):
        DataPipeline([TARGET]).fit(X, y)
    with pytest.raises(DataStepError, match="numeric"):
        DataPipeline([DataStepSpec("target_encode", ("x",))], task_type="binary").fit(X, y)
    with pytest.raises(DataStepError, match="smoothing"):
        DataStepSpec("target_encode", ("city",), {"smoothing": -1})
    with pytest.raises(DataStepError, match="values for"):
        DataPipeline([TARGET], task_type="binary").fit(X, y.head(5))
    assert DataStepSpec.from_dict(json.loads(json.dumps(TARGET.to_dict()))) == TARGET


# --------------------------------------------------------------------------- token_features
def test_words_of_titanic_names():
    df = titanic()
    assert tokenize("Braund, Mr. Owen Harris") == ["braund", "mr", "owen", "harris"] and tokenize("C85") == ["c", "85"]
    words = frequent_tokens(df["Name"])
    assert words[:3] == ["mr", "miss", "mrs"] and "master" in words and len(words) <= 20
    assert frequent_tokens(df["Sex"]) == [] and frequent_tokens(df["Embarked"]) == []  # each word is one value
    out = DataPipeline([DataStepSpec("token_features", ("Name",))]).fit_transform(df.drop(columns="Survived"))
    assert "Name" not in out.columns and out["Name__tok_mr"].sum() == df["Name"].str.contains(r"\bMr\b").sum()


def test_token_features_on_other_rows():
    train = pd.DataFrame({"name": ["Smith, Mr. A", "Jones, Mr. B", "Brown, Miss. C", "Green, Miss. D", None],
                          "x": np.arange(5.0)})
    pipeline = DataPipeline([DataStepSpec("token_features", ("name",), {"min_share": 0.3})]).fit(train)
    assert pipeline.state[0]["tokens"] == {"name": ["miss", "mr"]}  # each in 2 of 5 rows and 2 different values
    out = pipeline.transform(pd.DataFrame({"name": ["Lee, Mr. Q", "Dr. Z", None], "x": [0.0, 0.0, 0.0]}))
    assert list(out.columns) == ["x", "name__tok_miss", "name__tok_mr"]
    assert out["name__tok_mr"].tolist() == [1, 0, 0] and out["name__tok_miss"].tolist() == [0, 0, 0]
    with pytest.raises(DataStepError, match="numeric"):
        DataPipeline([DataStepSpec("token_features", ("x",))]).fit(train)
    with pytest.raises(DataStepError, match="already exist"):
        DataPipeline([DataStepSpec("token_features", ("name",), {"min_share": 0.3})]).fit(
            train.assign(name__tok_mr=0))


# --------------------------------------------------------------------------- runner and Assessor
class SearchRecordingEngine(NumericEngine):
    def fit(self, X_train, y_train, **kwargs):
        self.searched = (X_train, y_train)
        return super().fit(X_train, y_train, **kwargs)


def test_the_runner_searches_cross_fitted_rows_and_the_bundle_predicts_from_raw_rows(tmp_path):
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    engine = SearchRecordingEngine()
    runner = AutoMLRunner(settings=settings, engine=engine, enable_mlflow=False)
    X, y = rows(200)
    df = X.assign(y=y)
    config = ExperimentConfig(target_column="y", task_type=TaskType.BINARY, time_budget=10, data_steps=[TARGET])
    result = runner.run_experiment(df, config)
    searched, labels = engine.searched
    expected = DataPipeline([TARGET], task_type="binary").fit_transform(X.loc[searched.index], labels)
    pd.testing.assert_frame_equal(searched, expected)  # the cross-fitted encodings of the Training_Part
    bundle = AutoMLRunner.load_model(result.model_path)
    model = bundle["model"]
    assert model.pipeline.task_type == "binary" and result.feature_columns == ["city", "x"]
    raw = df.drop(columns="y")
    np.testing.assert_array_equal(model.predict(raw), model.model.predict(model.pipeline.transform(raw)))
    # the dry run of the Data_Steps comes after the task type, which target_encode needs
    with pytest.raises(ExperimentError, match="Invalid Data_Step"):
        runner.prepare(df, ExperimentConfig(target_column="y", data_steps=[DataStepSpec("target_encode", ("x",))]))
    words_only = ExperimentConfig(target_column="y", excluded_columns=["x"], data_steps=[
        DataStepSpec("token_features", ("city",))])
    runner.prepare(df, words_only)  # its columns are learned from rows, so an empty dry run proves nothing


class _Fill:
    """Logistic regression that fills empty cells (the encodings keep them empty)."""

    def __init__(self, X, y):
        self.model = LogisticRegression().fit(X.fillna(0.5), y)
        self.classes_ = self.model.classes_

    def predict(self, X):
        return self.model.predict(X.fillna(0.5))

    def predict_proba(self, X):
        return self.model.predict_proba(X.fillna(0.5))


def test_the_assessor_cross_fits_within_each_folds_training_rows():
    X, y = rows(80)
    seen = []

    class RecordingEngine:
        def fit_config(self, X_train, y_train, **kwargs):
            seen.append((X_train, y_train))
            return _Fill(X_train, y_train)

    folds, _ = make_session_folds(y, TaskType.BINARY, n_splits=4, n_repeats=1, seed=0)
    config = ExperimentConfig(target_column="y", data_steps=[TARGET])
    result = type("R", (), {"setup_key": "k", "experiment_id": "EXP-1", "best_estimator": "lr", "best_config": {}})()
    scored = Assessor(RecordingEngine(), folds, TaskType.BINARY, "f1", 1, 0).assess(X, y, config, result)
    assert not scored.failed and len(seen) == 4
    for (fitted, labels), (train_positions, _) in zip(seen, folds.folds):
        expected = DataPipeline([TARGET], task_type="binary").fit_transform(X.iloc[train_positions],
                                                                            y.iloc[train_positions])
        pd.testing.assert_series_equal(fitted["city"], expected["city"])  # this fold's labels only, cross-fitted
        assert labels.index.equals(X.index[train_positions])


# --------------------------------------------------------------------------- Strategies, Critic, rules
@pytest.fixture(scope="module")
def titanic_profile():
    return DatasetProfiler().profile(titanic(), "Survived", task_type=TaskType.BINARY)


def test_the_profile_lists_the_frequent_words_of_text_columns(titanic_profile):
    assert set(titanic_profile.frequent_words) == {"Name", "Ticket", "Cabin"}
    assert titanic_profile.frequent_words["Cabin"][:2] == ["c", "b"]
    assert DatasetProfiler().profile(titanic(), "Survived").to_dict()["frequent_words"] == \
        titanic_profile.frequent_words


def test_token_features_brings_back_a_left_out_identifier_column(titanic_profile):
    first = Planner().first_setup(ExperimentConfig(target_column="Survived"), titanic_profile)
    assert "Name" in first.excluded_columns  # identifier-like: left out by the first Setup
    params = {"columns": ["Name"]}
    assert WORDS.applicable(first, titanic_profile, TaskType.BINARY, params)
    changed = WORDS.apply(first, params)
    assert "Name" not in changed.excluded_columns and changed.data_steps == [DataStepSpec("token_features", ("Name",))]
    assert not WORDS.applicable(changed, titanic_profile, TaskType.BINARY, params)  # once only
    assert not ENCODE.applicable(changed, titanic_profile, TaskType.BINARY, params)  # one encoding per column
    assert not WORDS.applicable(first, titanic_profile, TaskType.BINARY, {"columns": ["Sex"]})  # no shared words
    assert ENCODE.applicable(first, titanic_profile, TaskType.REGRESSION, {"columns": ["Cabin"]})
    assert not ENCODE.applicable(first, titanic_profile, TaskType.BINARY, {"columns": ["Name"]})  # left out


def test_unused_words_diagnosis_and_its_rule(titanic_profile):
    critic = Critic(CriticThresholds.load())
    first = Planner().first_setup(ExperimentConfig(target_column="Survived"), titanic_profile)
    restorable = Planner.safe_exclusions(ExperimentConfig(target_column="Survived"), titanic_profile)

    def words(config, restorable=()):
        found = critic.diagnose(profile=titanic_profile, config=config, assessment=assessment([fold(0.8, 0.8)] * 3),
                                search=None, task_type=TaskType.BINARY, primary_metric="f1", restorable=restorable)
        return {d.columns[0]: d for d in found if d.kind is DiagnosisKind.UNUSED_WORDS}

    found = words(first, restorable)
    assert set(found) == {"Cabin", "Name", "Ticket"}
    assert found["Name"].severity is Severity.MODERATE and found["Cabin"].severity is Severity.LOW
    assert found["Name"].evidence["words"].startswith("mr,miss,mrs") and found["Name"].evidence["column_used"] == 0
    assert set(words(first)) == {"Cabin", "Ticket"}  # without restorable: the user's own exclusions stay out
    changed = WORDS.apply(first, {"columns": ["Name"]})
    assert set(words(changed, restorable)) == {"Cabin", "Ticket"}  # treated, and no identifier Diagnosis either
    assert not [d for d in critic.diagnose(profile=titanic_profile, config=changed,
                                           assessment=assessment([fold(0.8, 0.8)] * 3), search=None,
                                           task_type=TaskType.BINARY, primary_metric="f1")
                if d.kind is DiagnosisKind.IDENTIFIER_COLUMN]
    proposals = propose(list(found.values()), first, titanic_profile, TaskType.BINARY, set())
    assert [p.spec.to_dict() for p in proposals][0] == {"strategy_id": "token_features", "params": {"columns": ["Name"]}}
    assert {p.spec.params["columns"][0] for p in proposals} == {"Name", "Cabin", "Ticket"}
