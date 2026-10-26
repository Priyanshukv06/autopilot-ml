"""Checks of the FLAML 2.7.0 features that the autonomous layer relies on (task 1 of the spec).

Each test pins down one behaviour that later code depends on. If FLAML changes, these tests
fail first, before the code built on them breaks. Findings are recorded in
docs/imp/report-notes.md, section 5.2.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import pytest
from flaml import AutoML, tune
from flaml.automl.model import LGBMEstimator
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# pytest records warnings itself, so the runner's warning filter does not apply here.
pytestmark = pytest.mark.filterwarnings("ignore:Column '.*' contains values unseen at fit time:UserWarning")

ROOT = Path(__file__).resolve().parents[1]
COMMON = {"task": "classification", "metric": "f1", "verbose": 0, "mlflow_logging": False, "n_jobs": 1, "seed": 42}
LGBM_CONFIG = {
    "n_estimators": 37, "num_leaves": 9, "min_child_samples": 15, "learning_rate": 0.07,
    "log_max_bin": 7, "colsample_bytree": 0.8, "reg_alpha": 0.01, "reg_lambda": 0.5,
}


@pytest.fixture(scope="module")
def titanic():
    df = pd.read_csv(ROOT / "data" / "sample" / "titanic.csv")
    y = df["Survived"]
    X = df.drop(columns=["Survived", "PassengerId", "Name"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def logged_trials(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if '"record_id"' in line]


def fit_one(X, y, estimator="lgbm", config=None, **kwargs) -> AutoML:
    automl = AutoML()
    automl.fit(X, y, estimator_list=[estimator], starting_points={estimator: config or LGBM_CONFIG},
               max_iter=1, retrain_full=True, **COMMON, **kwargs)
    return automl


def test_search_starts_from_the_given_configuration(titanic, tmp_path):
    X_train, _, y_train, _ = titanic
    automl = AutoML()
    automl.fit(X_train, y_train, estimator_list=["lgbm"], starting_points={"lgbm": LGBM_CONFIG}, max_iter=3,
               eval_method="cv", log_file_name=str(tmp_path / "search.log"), log_type="all", **COMMON)
    first = logged_trials(tmp_path / "search.log")[0]["config"]
    for key, value in LGBM_CONFIG.items():
        assert first[key] == pytest.approx(value), key


def test_max_iter_one_trains_exactly_the_given_configuration(titanic):
    X_train, _, y_train, _ = titanic
    automl = fit_one(X_train, y_train)
    params = automl.model.estimator.get_params()
    assert automl.best_estimator == "lgbm"
    for key in ("n_estimators", "num_leaves", "min_child_samples", "learning_rate", "colsample_bytree"):
        assert params[key] == pytest.approx(LGBM_CONFIG[key]), key


def test_refitting_the_same_configuration_is_repeatable(titanic):
    X_train, X_test, y_train, _ = titanic
    first = fit_one(X_train, y_train).predict_proba(X_test)
    second = fit_one(X_train, y_train).predict_proba(X_test)
    assert np.max(np.abs(first - second)) <= 1e-12


def test_sample_weights_change_the_model(titanic):
    X_train, X_test, y_train, _ = titanic
    weights = compute_sample_weight("balanced", y_train)
    plain = fit_one(X_train, y_train).predict_proba(X_test)[:, 1].mean()
    weighted = fit_one(X_train, y_train, sample_weight=weights).predict_proba(X_test)[:, 1].mean()
    # Balanced weights up-weight the smaller class (survivors), so its predicted share rises.
    assert weighted > plain + 0.02


def test_search_space_upper_limit_is_respected(titanic, tmp_path):
    X_train, _, y_train, _ = titanic
    spec = dict(LGBMEstimator.search_space(data_size=X_train.shape, task=None)["num_leaves"])
    spec["domain"] = tune.lograndint(lower=4, upper=32)
    automl = AutoML()
    automl.fit(X_train, y_train, estimator_list=["lgbm"], custom_hp={"lgbm": {"num_leaves": spec}}, max_iter=20,
               eval_method="cv", log_file_name=str(tmp_path / "limited.log"), log_type="all", **COMMON)
    leaves = [trial["config"]["num_leaves"] for trial in logged_trials(tmp_path / "limited.log")]
    assert leaves[0] == 4  # FLAML's cheap starting value is kept when only the bound changes
    assert max(leaves) <= 32


def test_starting_configuration_outside_the_space_is_dropped(titanic, tmp_path):
    X_train, _, y_train, _ = titanic
    bad = dict(LGBM_CONFIG, num_leaves=10**6)  # far above the upper bound (number of rows)
    automl = AutoML()
    automl.fit(X_train, y_train, estimator_list=["lgbm"], starting_points={"lgbm": bad}, max_iter=3,
               eval_method="cv", log_file_name=str(tmp_path / "bad.log"), log_type="all", **COMMON)
    trials = logged_trials(tmp_path / "bad.log")
    assert len(trials) >= 2  # the search carries on
    assert trials[0]["config"]["num_leaves"] != bad["num_leaves"]


class RecordingResampler(BaseEstimator):
    """Stand-in for an imbalanced-learn resampler: records how many rows it is given."""

    calls: list[int] = []

    def fit_resample(self, X, y):
        RecordingResampler.calls.append(len(y))
        return X, y


def test_resampler_only_sees_training_rows(titanic):
    X_train, _, y_train, _ = titanic
    RecordingResampler.calls = []
    automl = AutoML()
    automl.fit(X_train, y_train, estimator_list=["lgbm"], resampler=RecordingResampler(), max_iter=2,
               eval_method="cv", **COMMON)
    n = len(y_train)
    fold_train_sizes = {n - n // 5, n - n // 5 - 1, n - (n + 4) // 5}
    # Called on the training rows of each fold and once on all training rows; never on a validation fold.
    assert set(RecordingResampler.calls) <= fold_train_sizes | {n}
    assert n in RecordingResampler.calls


def test_resampler_cannot_be_combined_with_sample_weights(titanic):
    X_train, _, y_train, _ = titanic
    weights = compute_sample_weight("balanced", y_train)
    with pytest.raises(ValueError):
        AutoML().fit(X_train, y_train, estimator_list=["lgbm"], resampler=RecordingResampler(),
                     sample_weight=weights, max_iter=2, **COMMON)


def test_integer_upper_bound_is_never_searched():
    from flaml.tune.space import denormalize

    # The top of the scaled 0-to-1 range maps to upper - 1 for plain integer ranges, so a limit of
    # "at most 32" must be written upper=33 (FLAMLEngine.custom_hp does this). Quantized ranges
    # include their upper value.
    space = {
        "leaves": tune.lograndint(lower=4, upper=33),
        "depth": tune.randint(lower=1, upper=9),
        "even": tune.qrandint(lower=2, upper=10, q=2),
    }
    top = denormalize({"leaves": 1.0, "depth": 1.0, "even": 1.0}, space, {}, {}, np.random.RandomState(0))
    assert top == {"leaves": 32, "depth": 8, "even": 10}
    # FLAML's own validity check of a starting value still accepts the upper value itself.
    assert space["leaves"].is_valid(33)
