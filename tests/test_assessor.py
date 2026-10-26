"""Tests for the Assessor (task 5.1 of the spec).

Most tests use a small logistic-regression engine that records each refit, so they need no
FLAML. The repeatability test uses real FLAML refits.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from app.services.assessor import REPEAT_TOLERANCE, Assessor
from app.services.automl_runner import ExperimentConfig, ExperimentError, FLAMLEngine
from app.services.splits import make_outer_split, make_session_folds
from app.services.task_detector import TaskType

LGBM_CONFIG = {
    "n_estimators": 12, "num_leaves": 5, "min_child_samples": 10, "learning_rate": 0.2,
    "log_max_bin": 8, "colsample_bytree": 0.9, "reg_alpha": 0.01, "reg_lambda": 1.0,
}
BINARY_METRICS = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"}

# Validation rows have cities the fold's training rows never saw (on purpose), and FLAML warns on
# each prediction. pytest records warnings itself, so the runner's warning filter does not apply.
pytestmark = pytest.mark.filterwarnings("ignore:Column '.*' contains values unseen at fit time:UserWarning")


class NumericLogistic:
    """Logistic regression on the numeric columns only."""

    def __init__(self, X, y, sample_weight):
        self.columns = [c for c in X.columns if is_numeric_dtype(X[c])]
        self.model = LogisticRegression(max_iter=1000).fit(X[self.columns], y, sample_weight=sample_weight)
        self.classes_ = self.model.classes_

    def predict(self, X):
        return self.model.predict(X[self.columns])

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.columns])


class RecordingEngine:
    """Records every refit; can fail on a chosen call."""

    name = "recording"

    def __init__(self, fail_on_call=None):
        self.calls = []
        self.fail_on_call = fail_on_call

    def fit_config(self, X_train, y_train, *, task_type, estimator, config, seed, n_jobs, sample_weight=None):
        self.calls.append({"index": X_train.index, "y": y_train, "estimator": estimator, "config": config,
                           "sample_weight": sample_weight})
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("refit exploded")
        return NumericLogistic(X_train, y_train, sample_weight)


def training_part(n=300, seed=0):
    """The Training_Part of a small dataset (index labels kept), and its test rows."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    df["city"] = "common"
    rare = rng.choice(n, size=n // 10, replace=False)
    df.loc[rare, "city"] = [f"rare{i}" for i in range(len(rare))]  # every rare value appears once
    df["label"] = np.where(df["a"] + rng.normal(scale=0.7, size=n) > 0.6, "pos", "neg")
    split, _ = make_outer_split(df["label"], TaskType.BINARY, 0.2, 42)
    train, test = df.iloc[split.train_positions], df.iloc[split.test_positions]
    return train.drop(columns="label"), train["label"], test


def result_for(estimator="logreg", config=None, setup_key="setup-a", experiment_id="EXP-001"):
    """The parts of an ExperimentResult the Assessor reads."""
    return SimpleNamespace(setup_key=setup_key, experiment_id=experiment_id, best_estimator=estimator,
                           best_config=dict(config or {"C": 1.0}))


def make_assessor(engine, y_train, task_type=TaskType.BINARY, metric="f1"):
    folds, _ = make_session_folds(y_train, task_type, seed=1)
    return Assessor(engine, folds, task_type, metric, n_jobs=1, seed=0), folds


CONFIG = ExperimentConfig(target_column="label", task_type=TaskType.BINARY, primary_metric="f1")


def test_one_score_per_fold_with_every_metric():
    X, y, _ = training_part()
    assessor, folds = make_assessor(RecordingEngine(), y)
    assessment = assessor.assess(X, y, CONFIG, result_for())
    assert not assessment.failed and len(assessment.folds) == 15
    assert [(f.repeat, f.fold) for f in assessment.folds] == [(r, k) for r in range(3) for k in range(5)]
    for fold in assessment.folds:
        assert set(fold.train_metrics) == set(fold.validation_metrics) == BINARY_METRICS
        assert set(fold.baseline_validation) == {"accuracy", "balanced_accuracy", "precision", "recall", "f1"}
        assert set(fold.class_recall) == {"neg", "pos"} and fold.fit_seconds >= 0
    assert assessment.folds_fingerprint == folds.fingerprint() and assessment.seconds > 0
    assert (assessment.setup_key, assessment.experiment_id, assessment.estimator) == ("setup-a", "EXP-001", "logreg")


def test_values_and_summary():
    X, y, _ = training_part()
    assessor, _ = make_assessor(RecordingEngine(), y)
    assessment = assessor.assess(X, y, CONFIG, result_for())
    f1 = assessment.values("f1", "validation")
    assert f1.shape == (15,) and np.isfinite(f1).all()
    summary = assessment.summary()
    assert summary["validation"]["f1"]["mean"] == pytest.approx(f1.mean())
    assert summary["validation"]["f1"]["std"] == pytest.approx(f1.std(ddof=1))
    assert summary["n_folds"] == 15 and summary["failed"] is False
    # The trivial baseline predicts the majority class ("neg"), so its F1 for "pos" is 0.
    assert (assessment.values("f1", "baseline") == 0).all()
    assert assessment.to_dict()["folds"][0]["repeat"] == 0


def test_the_assessor_only_ever_sees_the_fold_rows_of_the_training_part():
    X, y, test = training_part()
    engine = RecordingEngine()
    assessor, folds = make_assessor(engine, y)
    with pytest.raises(ExperimentError, match="training rows"):  # the whole dataset is refused
        assessor.assess(pd.concat([X, test.drop(columns="label")]), pd.concat([y, test["label"]]), CONFIG,
                        result_for())
    assessor.assess(X, y, CONFIG, result_for())
    test_rows = set(test.index)
    for call, (train_positions, _) in zip(engine.calls, folds.folds):
        assert list(call["index"]) == list(X.index[train_positions])  # exactly the fold's training rows
        assert not set(call["index"]) & test_rows


def test_class_weights_come_from_each_folds_training_rows():
    X, y, _ = training_part()
    engine = RecordingEngine()
    assessor, _ = make_assessor(engine, y)
    assessor.assess(X, y, ExperimentConfig(target_column="label", class_weight="balanced"), result_for())
    for call in engine.calls:
        np.testing.assert_allclose(call["sample_weight"], compute_sample_weight("balanced", call["y"]))
    plain = RecordingEngine()
    make_assessor(plain, y)[0].assess(X, y, CONFIG, result_for())
    assert all(call["sample_weight"] is None for call in plain.calls)


def test_a_cached_setup_is_not_refitted_but_a_new_configuration_is():
    X, y, _ = training_part()
    engine = RecordingEngine()
    assessor, _ = make_assessor(engine, y)
    first = assessor.assess(X, y, CONFIG, result_for())
    assert assessor.assess(X, y, CONFIG, result_for(experiment_id="EXP-002")) is first
    assert assessor.cached(result_for()) is first and len(engine.calls) == 15
    # Same Setup, other configuration (for example after more search time): assessed again.
    second = assessor.assess(X, y, CONFIG, result_for(config={"C": 0.1}))
    assert second is not first and len(engine.calls) == 30
    assert assessor.cached(result_for(setup_key="setup-b")) is None


def test_a_failing_fold_marks_the_assessment_as_failed():
    X, y, _ = training_part()
    assessor, _ = make_assessor(RecordingEngine(fail_on_call=3), y)
    assessment = assessor.assess(X, y, CONFIG, result_for())
    assert assessment.failed and len(assessment.folds) == 2
    assert "Fold 3 of repeat 1" in assessment.error and "refit exploded" in assessment.error
    assert assessment.summary()["failed"] is True


def test_unseen_shares_are_measured_between_each_folds_training_and_validation_rows():
    X, y, _ = training_part()
    assessor, folds = make_assessor(RecordingEngine(), y)
    assessment = assessor.assess(X, y, CONFIG, result_for())
    rare = X["city"].str.startswith("rare").to_numpy()
    for fold, (_, validation_positions) in zip(assessment.folds, folds.folds):
        assert set(fold.unseen_share) == {"city"}  # numeric columns are not checked
        # Each rare value appears once, so it is unseen exactly when its row is a validation row.
        assert fold.unseen_share["city"] == pytest.approx(rare[validation_positions].mean())


def test_regression_assessment():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(120, 2)), columns=["a", "b"])
    y = pd.Series(2 * X["a"] - X["b"] + rng.normal(scale=0.1, size=120))

    class LinearEngine:
        def fit_config(self, X_train, y_train, **kwargs):
            from sklearn.linear_model import LinearRegression
            return LinearRegression().fit(X_train, y_train)

    assessor, _ = make_assessor(LinearEngine(), y, TaskType.REGRESSION, "r2")
    assessment = assessor.assess(X, y, ExperimentConfig(target_column="y"), result_for())
    fold = assessment.folds[0]
    assert set(fold.validation_metrics) == {"mae", "mse", "rmse", "r2"} and fold.class_recall is None
    assert assessment.values("r2", "validation").min() > 0.95
    assert abs(assessment.values("r2", "baseline")).max() < 0.2  # predicting the training mean


def test_repeated_flaml_assessments_agree_within_the_task1_tolerance():
    X, y, _ = training_part(n=240)
    result = result_for(estimator="lgbm", config=LGBM_CONFIG)
    first = make_assessor(FLAMLEngine(), y)[0].assess(X, y, CONFIG, result)
    second = make_assessor(FLAMLEngine(), y)[0].assess(X, y, CONFIG, result)  # a new Assessor: no cache
    assert not first.failed and not second.failed
    for part in ("train", "validation"):
        for metric in BINARY_METRICS:
            difference = np.nan_to_num(first.values(metric, part) - second.values(metric, part))
            assert np.abs(difference).max() <= REPEAT_TOLERANCE
