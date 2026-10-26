"""Tests for the engine and runner extensions used by Sessions (task 3.1 of the spec).

Runner tests use a small logistic-regression engine that records what it is given, so they need
no FLAML search. The FLAML tests are short (a few trials on 240 rows).
"""

import json
from dataclasses import replace

import flaml
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from app.core.config import build_settings
from app.services.automl_runner import (
    AutoMLRunner,
    EngineResult,
    ExperimentConfig,
    ExperimentError,
    FLAMLEngine,
    decision_view,
    limited_search_space_spec,
    value_in_search_space,
)
from app.services.profiler import DatasetProfiler
from app.services.splits import OuterSplit, make_outer_split
from app.services.task_detector import TaskType

# The call FLAML received in Phase 1. With no new option the engine must make exactly this call.
PHASE1_FLAML_ARGUMENTS = {
    "X_train", "y_train", "task", "metric", "time_budget", "seed", "n_jobs", "estimator_list", "eval_method",
    "log_file_name", "log_type", "retrain_full", "early_stop", "verbose", "mlflow_logging",
}
LGBM_START = {
    "n_estimators": 12, "num_leaves": 5, "min_child_samples": 10, "learning_rate": 0.2,
    "log_max_bin": 8, "colsample_bytree": 0.9, "reg_alpha": 0.01, "reg_lambda": 1.0,
}
TITANIC_SIZE = (712, 9)


# --------------------------------------------------------------------------- helpers
@pytest.fixture
def settings(tmp_path):
    s = build_settings(tmp_path)
    s.ensure_directories()
    return s


def cancer_frame() -> pd.DataFrame:
    df = load_breast_cancer(as_frame=True).frame.copy()
    df["diagnosis"] = df.pop("target").map({0: "malignant", 1: "benign"})
    return df


def labelled_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Numbers, one category column and string labels (about a quarter "yes")."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["colour"] = np.where(X[:, 2] > 0, "red", "blue")
    score = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n)
    df["label"] = np.where(score > 0.8, "yes", "no")
    return df


class RecordingModel:
    """A fitted model that records which rows it is asked to predict."""

    def __init__(self, model):
        self.model = model
        self.classes_ = model.classes_
        self.seen = []

    def predict(self, X):
        self.seen.append(X.index)
        return self.model.predict(X)

    def predict_proba(self, X):
        self.seen.append(X.index)
        return self.model.predict_proba(X)


class RecordingEngine:
    """Logistic regression engine that records the arguments the runner passes."""

    name = "recording"

    def __init__(self, record_predictions=False, warnings=()):
        self.calls = []
        self.record_predictions = record_predictions
        self.extra_warnings = list(warnings)

    def fit(self, X_train, y_train, **kwargs):
        self.calls.append(dict(kwargs, X_train=X_train, y_train=y_train))
        model = self.fit_config(X_train, y_train, task_type=kwargs["task_type"], estimator="logreg",
                                config={"C": 1.0}, seed=kwargs["seed"], n_jobs=kwargs["n_jobs"],
                                sample_weight=kwargs["sample_weight"])
        if self.record_predictions:
            model = RecordingModel(model)
        self.last_model = model
        return EngineResult(
            model=model, engine_name=self.name, engine_version="test", best_estimator="logreg",
            best_model_name="Logistic Regression", best_config={"C": 1.0}, best_validation_score=None,
            validation_method="none", training_time=0.0, search_history=[], leaderboard=[],
            native_importance=None, classes=list(model.classes_), best_config_per_estimator={"logreg": {"C": 1.0}},
            warnings=list(self.extra_warnings),
        )

    def fit_config(self, X_train, y_train, *, task_type, estimator, config, seed, n_jobs, sample_weight=None):
        return LogisticRegression(max_iter=5000, **config).fit(X_train, y_train, sample_weight=sample_weight)


class RecordingProfiler(DatasetProfiler):
    """Records which rows were profiled."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def profile(self, df, *args, **kwargs):
        self.seen.append(df.index)
        return super().profile(df, *args, **kwargs)


def cancer_config(**changes) -> ExperimentConfig:
    base = ExperimentConfig(target_column="diagnosis", task_type=TaskType.BINARY, primary_metric="f1", time_budget=10)
    return replace(base, **changes)


@pytest.fixture
def flaml_calls(monkeypatch):
    """Record the keyword arguments of every AutoML.fit call made by FLAMLEngine."""
    calls = []

    class SpyAutoML(flaml.AutoML):
        def fit(self, *args, **kwargs):
            calls.append(kwargs)
            return super().fit(*args, **kwargs)

    monkeypatch.setattr(flaml, "AutoML", SpyAutoML)
    return calls


# --------------------------------------------------------------------------- FLAML engine
def test_without_new_options_flaml_gets_the_phase1_call(flaml_calls, tmp_path):
    df = labelled_frame()
    result = FLAMLEngine().fit(
        df.drop(columns="label"), df["label"], task_type=TaskType.BINARY, metric="f1", time_budget=2, seed=0,
        n_jobs=1, estimator_list=["lgbm"], work_dir=tmp_path,
    )
    assert set(flaml_calls[0]) == PHASE1_FLAML_ARGUMENTS
    assert result.warnings == []
    assert result.best_config_per_estimator == {"lgbm": result.best_config}


def test_starting_points_limits_weights_and_max_iter_reach_flaml(flaml_calls, tmp_path):
    df = labelled_frame()
    X, y = df.drop(columns="label"), df["label"]
    weights = compute_sample_weight("balanced", y)
    result = FLAMLEngine().fit(
        X, y, task_type=TaskType.BINARY, metric="roc_auc", time_budget=60, seed=0, n_jobs=1,
        estimator_list=["lgbm"], work_dir=tmp_path,
        starting_points={"lgbm": LGBM_START, "xgboost": {"max_leaves": 10**6}},
        search_space_limits={"lgbm": {"num_leaves": {"upper": 6}}},
        sample_weight=weights, max_iter=6,
    )
    call = flaml_calls[0]
    assert set(call) == PHASE1_FLAML_ARGUMENTS | {"custom_hp", "starting_points", "sample_weight", "max_iter"}
    assert call["metric"] == "roc_auc" and call["max_iter"] == 6
    np.testing.assert_array_equal(call["sample_weight"], weights)
    assert call["custom_hp"]["lgbm"]["num_leaves"]["domain"].upper == 7  # "at most 6"
    assert call["starting_points"] == {"lgbm": LGBM_START}  # the invalid XGBoost start is left out
    assert len(result.warnings) == 1 and "XGBoost" in result.warnings[0] and "max_leaves" in result.warnings[0]

    trials = result.search_history
    assert 1 <= len(trials) <= 6
    assert trials[0]["config"] == pytest.approx(LGBM_START)  # the search starts from the given point
    assert all(4 <= trial["config"]["num_leaves"] <= 6 for trial in trials)
    assert result.best_config_per_estimator["lgbm"] == result.best_config


def test_fit_config_trains_the_given_configuration_and_predicts_original_labels():
    df = labelled_frame()
    X, y = df.drop(columns="label"), df["label"]
    engine = FLAMLEngine()
    model = engine.fit_config(X, y, task_type=TaskType.BINARY, estimator="lgbm", config=LGBM_START, seed=0, n_jobs=1)
    assert model.best_estimator == "lgbm"
    assert model.best_config == pytest.approx(LGBM_START)
    assert list(model.classes_) == ["no", "yes"]
    assert set(model.predict(X)) <= {"no", "yes"}
    plain = model.predict_proba(X)
    assert plain.shape == (len(X), 2)

    weighted = engine.fit_config(X, y, task_type=TaskType.BINARY, estimator="lgbm", config=LGBM_START, seed=0,
                                 n_jobs=1, sample_weight=compute_sample_weight("balanced", y))
    # Balanced weights raise the weight of the smaller class ("yes"), so its predicted share rises.
    assert weighted.predict_proba(X)[:, 1].mean() > plain[:, 1].mean()
    with pytest.raises(ExperimentError, match="one finite"):
        engine.fit_config(X, y, task_type=TaskType.BINARY, estimator="lgbm", config=LGBM_START, seed=0, n_jobs=1,
                          sample_weight=np.ones(3))


# --------------------------------------------------------------------------- search-space helpers
def lgbm_space() -> dict:
    return FLAMLEngine().search_space("lgbm", TaskType.BINARY, TITANIC_SIZE)


def test_limits_are_inclusive_and_keep_flaml_domain_type():
    spec = lgbm_space()["num_leaves"]
    limited = limited_search_space_spec(spec, {"upper": 32}, "lgbm.num_leaves")
    assert type(limited["domain"]) is type(spec["domain"])
    assert type(limited["domain"].get_sampler()) is type(spec["domain"].get_sampler())
    assert (limited["domain"].lower, limited["domain"].upper) == (4, 33)  # FLAML searches 4 to 32
    assert limited["init_value"] == limited["low_cost_init_value"] == 4
    assert spec["domain"].upper == 712  # FLAML's spec is copied, not changed


def test_limits_move_starting_values_and_never_widen():
    space = lgbm_space()
    raised = limited_search_space_spec(space["min_child_samples"], {"lower": 30}, "lgbm.min_child_samples")
    assert (raised["domain"].lower, raised["domain"].upper, raised["init_value"]) == (30, 129, 30)
    wide = limited_search_space_spec(space["num_leaves"], {"upper": 10**6}, "lgbm.num_leaves")
    assert wide["domain"].upper == 712
    xgb = FLAMLEngine().search_space("xgboost", TaskType.BINARY, TITANIC_SIZE)
    weight = limited_search_space_spec(xgb["min_child_weight"], {"upper": 0.5}, "xgboost.min_child_weight")
    assert (weight["domain"].upper, weight["init_value"]) == (0.5, 0.5)  # a float range keeps its upper value


@pytest.mark.parametrize(
    "learner, setting, bounds, message",
    [
        ("lgbm", "num_leaves", {"upper": 4}, "fewer than two values"),
        ("lgbm", "num_leaves", {"max": 32}, "Unknown search-space limit"),
        ("lgbm", "no_such_setting", {"upper": 3}, "not a numeric setting"),
        ("rf", "criterion", {"upper": 1}, "not a numeric setting"),
    ],
)
def test_impossible_limits_are_rejected(learner, setting, bounds, message):
    space = FLAMLEngine().search_space(learner, TaskType.BINARY, TITANIC_SIZE)
    with pytest.raises(ExperimentError, match=message):
        limited_search_space_spec(space.get(setting), bounds, f"{learner}.{setting}")


def test_unknown_learner_family_is_rejected():
    with pytest.raises(ExperimentError, match="Unknown learner family"):
        FLAMLEngine().search_space("no_such_learner", TaskType.BINARY, TITANIC_SIZE)


def test_starting_values_are_checked_against_the_searched_range():
    space = lgbm_space()
    limited = limited_search_space_spec(space["num_leaves"], {"upper": 32}, "lgbm.num_leaves")
    assert value_in_search_space(32, limited) and value_in_search_space(4, limited)
    assert not value_in_search_space(33, limited)
    assert limited["domain"].is_valid(33)  # FLAML's own check would accept it
    assert not value_in_search_space(3, limited)
    assert not value_in_search_space(8.0, limited)  # wrong type for an integer setting
    assert not value_in_search_space(True, limited)
    assert value_in_search_space(1, space["learning_rate"])  # an integer is fine for a float setting
    assert not value_in_search_space(2.0, space["learning_rate"])
    criterion = FLAMLEngine().search_space("rf", TaskType.BINARY, TITANIC_SIZE)["criterion"]
    assert value_in_search_space("gini", criterion) and not value_in_search_space("poisson", criterion)
    assert not value_in_search_space(1, None)  # a setting the learner does not have
    assert value_in_search_space(3, {"domain": 3}) and not value_in_search_space(4, {"domain": 3})


def test_check_starting_points_uses_the_limited_space():
    engine = FLAMLEngine()
    custom = engine.custom_hp({"lgbm": {"num_leaves": {"upper": 8}}}, TaskType.BINARY, TITANIC_SIZE)
    points = {
        "lgbm": {"num_leaves": 20, "FLAML_sample_size": 712},  # fine by default, but above the limit
        "xgboost": {"max_leaves": 10, "FLAML_sample_size": 712},
        "rf": {},
    }
    valid, messages = engine.check_starting_points(points, TaskType.BINARY, TITANIC_SIZE, custom)
    assert valid == {"xgboost": {"max_leaves": 10}}
    assert len(messages) == 1 and "LightGBM" in messages[0] and "num_leaves" in messages[0]


# --------------------------------------------------------------------------- runner
def test_runner_defaults_keep_the_phase1_behaviour(settings):
    engine = RecordingEngine()
    runner = AutoMLRunner(settings=settings, engine=engine, enable_mlflow=False)
    result = runner.run(cancer_frame(), "diagnosis", task="binary", primary_metric="f1", time_budget=10)
    call = engine.calls[0]
    assert call["metric"] == "f1"
    assert all(call[key] is None for key in ("starting_points", "search_space_limits", "sample_weight", "max_iter"))
    assert result.test_evaluated and result.test_score == result.test_metrics["f1"]
    assert result.baseline_metrics and result.feature_importance
    assert (result.search_metric, result.schema_version, result.profile_rows) == ("f1", 2, "all")
    assert result.session_id is None and result.parent_experiment_id is None and result.proposal is None
    assert result.best_config_per_estimator == {"logreg": {"C": 1.0}}
    files = {path.name for path in (settings.experiments_dir / result.experiment_id).iterdir()}
    assert {"predictions.csv", "feature_importance.csv", "summary.json"} <= files


def test_runner_passes_class_weights_search_metric_and_options(settings):
    engine = RecordingEngine(warnings=["engine note"])
    runner = AutoMLRunner(settings=settings, engine=engine, enable_mlflow=False)
    config = cancer_config(
        class_weight="balanced", search_metric="roc_auc", search_space_limits={"lgbm": {"num_leaves": {"upper": 32}}},
        starting_points={"lgbm": {"num_leaves": 8}}, max_iter=3,
    )
    result = runner.run_experiment(cancer_frame(), config, compute_importance=False, session_id="SES-001",
                                   parent_experiment_id="EXP-001", proposal={"strategy": "class_weight_balanced"})
    call = engine.calls[0]
    np.testing.assert_allclose(call["sample_weight"], compute_sample_weight("balanced", call["y_train"]))
    assert call["metric"] == "roc_auc"
    assert call["search_space_limits"] == {"lgbm": {"num_leaves": {"upper": 32}}}
    assert call["starting_points"] == {"lgbm": {"num_leaves": 8}} and call["max_iter"] == 3
    # Reported scores still use the primary metric.
    assert result.primary_metric == "f1" and result.search_metric == "roc_auc"
    assert result.train_score == result.train_metrics["f1"]
    assert "engine note" in result.warnings
    assert (result.session_id, result.parent_experiment_id) == ("SES-001", "EXP-001")
    assert result.proposal == {"strategy": "class_weight_balanced"}
    assert result.setup["class_weight"] == "balanced" and result.setup["search_metric"] == "roc_auc"
    saved = json.loads((settings.experiments_dir / result.experiment_id / "summary.json").read_text("utf-8"))
    assert saved["session_id"] == "SES-001" and saved["setup_key"] == result.setup_key


def test_evaluate_test_false_leaves_the_test_part_untouched(settings):
    df = cancer_frame()
    split, _ = make_outer_split(df["diagnosis"], TaskType.BINARY, 0.2, 7)  # not the config seed
    engine, profiler = RecordingEngine(record_predictions=True), RecordingProfiler()
    runner = AutoMLRunner(settings=settings, engine=engine, profiler=profiler, enable_mlflow=False)
    result = runner.run_experiment(df, cancer_config(), split=split, evaluate_test=False, profile_rows="train")

    train_rows, test_rows = set(df.index[split.train_positions]), set(df.index[split.test_positions])
    assert set(engine.calls[0]["X_train"].index) == train_rows  # exactly the given training rows
    seen = engine.last_model.seen  # every predict or predict_proba call on the final model
    assert seen and all(set(index) <= train_rows for index in seen)
    assert [set(index) for index in profiler.seen] == [train_rows]

    assert result.test_score is None and result.test_metrics == {} and result.baseline_metrics == {}
    assert result.feature_importance == [] and result.test_evaluation is None
    assert result.test_evaluated is False and result.n_test == len(test_rows)
    assert result.train_metrics and result.profile.rows == len(train_rows)
    work_dir = settings.experiments_dir / result.experiment_id
    files = {path.name for path in work_dir.iterdir()}
    assert "predictions.csv" not in files and "feature_importance.csv" not in files
    assert json.loads((work_dir / "evaluation.json").read_text("utf-8"))["test"] is None
    assert json.loads((work_dir / "summary.json").read_text("utf-8"))["test_evaluated"] is False


def test_compute_importance_false_skips_only_the_importance(settings):
    runner = AutoMLRunner(settings=settings, engine=RecordingEngine(), enable_mlflow=False)
    result = runner.run_experiment(cancer_frame(), cancer_config(), compute_importance=False)
    assert result.test_evaluated and result.test_metrics and result.feature_importance == []


def test_decision_view_contains_no_test_part_information(settings):
    runner = AutoMLRunner(settings=settings, engine=RecordingEngine(), enable_mlflow=False)
    result = runner.run_experiment(cancer_frame(), cancer_config())
    assert result.test_metrics and result.feature_importance and result.baseline_metrics
    view = decision_view(result)
    assert {key for key in view if key.startswith("test_")} == {"test_size", "test_evaluated"}
    for key in ("baseline_metrics", "feature_importance", "warnings", "dataset_meta_features"):
        assert key not in view
    json.dumps(view)  # JSON-safe
    # Meta-features are kept when the profile comes from the training part only.
    train_only = runner.run_experiment(cancer_frame(), cancer_config(), evaluate_test=False, profile_rows="train")
    assert decision_view(train_only)["dataset_meta_features"] == train_only.dataset_meta_features


def test_setup_key_is_stable_and_ignores_search_effort():
    base = ExperimentConfig(target_column="y", excluded_columns=["a", "b"], estimator_list=["xgboost", "lgbm"])
    same = [
        replace(base, time_budget=300), replace(base, n_jobs=1), replace(base, dataset_name="other"),
        replace(base, max_iter=5), replace(base, starting_points={"lgbm": {"num_leaves": 8}}),
        replace(base, excluded_columns=["b", "a"]), replace(base, estimator_list=["lgbm", "xgboost"]),
    ]
    different = [
        replace(base, class_weight="balanced"), replace(base, search_metric="roc_auc"),
        replace(base, search_space_limits={"lgbm": {"num_leaves": {"upper": 32}}}),
        replace(base, excluded_columns=["a"]), replace(base, random_seed=1), replace(base, primary_metric="roc_auc"),
        replace(base, target_column="z"), replace(base, test_size=0.3), replace(base, estimator_list=None),
        replace(base, task_type=TaskType.BINARY),
    ]
    assert {config.setup_key() for config in same} == {base.setup_key()}
    keys = {config.setup_key() for config in different}
    assert len(keys) == len(different) and base.setup_key() not in keys
    # Pinned, so that keys saved in old Session records stay valid. Change only on purpose.
    assert base.setup_key() == "c1e409aa1c5216c9"


def test_runner_setup_key_uses_the_resolved_task_and_metrics(settings):
    runner = AutoMLRunner(settings=settings, engine=RecordingEngine(), enable_mlflow=False)
    df = cancer_frame()
    auto = runner.run_experiment(df, ExperimentConfig(target_column="diagnosis", time_budget=10), evaluate_test=False)
    explicit = runner.run_experiment(df, cancer_config(primary_metric=auto.primary_metric), evaluate_test=False)
    same_metric = runner.run_experiment(df, cancer_config(primary_metric=auto.primary_metric,
                                                          search_metric=auto.primary_metric), evaluate_test=False)
    assert auto.setup_key == explicit.setup_key == same_metric.setup_key
    assert same_metric.setup["search_metric"] is None


@pytest.mark.parametrize(
    "changes, kwargs, message",
    [
        ({"data_steps": [{"kind": "drop_columns"}]}, {}, "Invalid Data_Step"),  # no columns given
        ({"data_steps": [{"kind": "frequency_encode", "columns": ["nope"]}]}, {}, "Invalid Data_Step"),
        ({"class_weight": "heavy"}, {}, "class weight"),
        ({"search_metric": "rmse"}, {}, "Search metric"),
        ({"max_iter": 0}, {}, "max_iter"),
        ({}, {"profile_rows": "test"}, "profile_rows"),
        ({}, {"split": OuterSplit(np.array([0, 1, 2]), np.array([2, 3]), 0, 0.2, False)}, "split"),
        ({}, {"split": OuterSplit(np.array([0, 1]), np.array([10**6]), 0, 0.2, False)}, "split"),
        ({}, {"split": OuterSplit(np.array([0, 1]), np.array([], dtype=int), 0, 0.2, False)}, "split"),
    ],
)
def test_invalid_session_options_fail_fast(settings, changes, kwargs, message):
    runner = AutoMLRunner(settings=settings, engine=RecordingEngine(), enable_mlflow=False)
    with pytest.raises(ExperimentError, match=message):
        runner.run_experiment(cancer_frame(), cancer_config(**changes), **kwargs)
    assert not any(settings.experiments_dir.glob("EXP-*"))  # rejected before an ID is given out


def test_class_weights_are_rejected_for_regression(settings):
    X, y = make_regression(n_samples=60, n_features=3, noise=1.0, random_state=0)
    df = pd.DataFrame(X, columns=["a", "b", "c"]).assign(target=y)
    runner = AutoMLRunner(settings=settings, engine=RecordingEngine(), enable_mlflow=False)
    config = ExperimentConfig(target_column="target", task_type=TaskType.REGRESSION, time_budget=10,
                              class_weight="balanced")
    with pytest.raises(ExperimentError, match="classification tasks only"):
        runner.run_experiment(df, config)
