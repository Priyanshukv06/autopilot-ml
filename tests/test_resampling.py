"""Tests for SMOTE through FLAML's resampler (optional task 29, Requirement 26): the resampler itself, the
Strategy and its rule, the checks of a Setup, and that validation rows are never resampled."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from app.core.config import build_settings
from app.services.assessor import Assessor
from app.services.automl_runner import AutoMLRunner, ExperimentConfig, ExperimentError, FLAMLEngine
from app.services.critic import Diagnosis, DiagnosisKind, Severity
from app.services.profiler import DatasetProfiler
from app.services.resampling import FoldSafeSmote, make_resampler
from app.services.splits import make_session_folds
from app.services.strategies import CATALOGUE, StrategySpec
from app.services.strategy_engine import propose
from app.services.task_detector import TaskType
from tests.test_assessor import CONFIG, NumericLogistic, result_for, training_part

SMOTE, WEIGHTS = CATALOGUE["smote"], CATALOGUE["class_weight_balanced"]


def imbalanced(n=200, ratio=4, seed=0, text=True):
    """Two numeric columns, a text column and an imbalanced binary target."""
    rng = np.random.default_rng(seed)
    y = np.array(["pos"] * (n // (ratio + 1)) + ["neg"] * (n - n // (ratio + 1)))
    frame = pd.DataFrame({"a": rng.normal(size=n) + (y == "pos"), "b": rng.normal(size=n)})
    if text:
        frame["city"] = pd.Categorical(rng.choice(["x", "y", "z"], size=n))
    return frame, pd.Series(y, name="label")


# --------------------------------------------------------------------------- the resampler
@pytest.mark.parametrize("text", [True, False])
def test_smote_balances_the_classes_and_keeps_the_original_rows_first(text):
    X, y = imbalanced(text=text)
    out_X, out_y = FoldSafeSmote(random_state=0).fit_resample(X, y)
    counts = pd.Series(out_y).value_counts()
    assert counts["pos"] == counts["neg"] == 160 and len(out_X) == 320
    assert out_X.iloc[:200].equals(X.reset_index(drop=True)) and list(out_y[:200]) == list(y)
    assert dict(out_X.dtypes) == dict(X.dtypes)  # FLAML's column types are kept
    if text:
        assert set(out_X["city"].iloc[200:]) <= {"x", "y", "z"}  # only values seen in the rows given
    again_X, _ = FoldSafeSmote(random_state=0).fit_resample(X, y)
    assert again_X.equals(out_X)  # the same seed gives the same rows


def test_smote_on_multiclass_text_only_and_empty_cells():
    X, y = imbalanced(n=120)
    y = pd.Series(np.where(np.arange(120) % 10 == 0, "rare", y), name="label")
    out_X, out_y = FoldSafeSmote(random_state=1).fit_resample(X, y)
    assert set(pd.Series(out_y).value_counts()) == {int((y == "neg").sum())}  # every class as large as the largest
    only_text = X[["city"]]
    out_text, _ = FoldSafeSmote(random_state=1).fit_resample(only_text, y)  # SMOTEN for text columns only
    assert len(out_text) > len(only_text) and set(out_text["city"]) <= {"x", "y", "z"}
    holes = X.copy()
    holes.loc[[1, 5, 9], "a"] = np.nan  # empty cells: used for finding neighbours only
    out_holes, _ = FoldSafeSmote(random_state=1).fit_resample(holes, y)
    assert out_holes.iloc[:120]["a"].isna().sum() == 3 and not out_holes.iloc[120:]["a"].isna().any()


def test_small_classes_and_balanced_data():
    X, y = imbalanced(n=60, ratio=19)  # 3 rows of "pos": 2 neighbours instead of 5
    out_X, out_y = FoldSafeSmote(random_state=0).fit_resample(X, y)
    assert pd.Series(out_y).value_counts()["pos"] == 57
    one = y.copy()
    one.iloc[:3] = "neg"
    one.iloc[0] = "pos"  # a class of one row cannot be oversampled: the rows come back unchanged
    same_X, same_y = FoldSafeSmote(random_state=0).fit_resample(X, one)
    assert same_X is X and same_y is one
    balanced = pd.Series(["a", "b"] * 30)
    assert FoldSafeSmote().fit_resample(X, balanced)[1] is balanced
    assert clone(FoldSafeSmote(k_neighbors=3, random_state=7)).get_params() == {"k_neighbors": 3, "random_state": 7}
    assert make_resampler(None, 0) is None and make_resampler("smote", 4).random_state == 4
    with pytest.raises(ValueError, match="Unknown resampler"):
        make_resampler("adasyn", 0)


# --------------------------------------------------------------------------- the Strategy and its rule
def test_the_smote_strategy_and_class_weights_exclude_each_other():
    frame, y = imbalanced()
    profile = DatasetProfiler().profile(frame.assign(label=y), "label")
    plain = ExperimentConfig(target_column="label")
    assert SMOTE.applicable(plain, profile, TaskType.BINARY, {})
    assert not SMOTE.applicable(plain, profile, TaskType.REGRESSION, {})
    changed = SMOTE.apply(plain, {})
    assert changed.resampler == "smote" and plain.resampler is None
    assert changed.__dict__ == {**plain.__dict__, "resampler": "smote"}
    assert not SMOTE.applicable(changed, profile, TaskType.BINARY, {})  # once
    assert not WEIGHTS.applicable(changed, profile, TaskType.BINARY, {})  # FLAML forbids both
    assert not SMOTE.applicable(WEIGHTS.apply(plain, {}), profile, TaskType.BINARY, {})
    with pytest.raises(ValueError, match="takes no parameters"):
        SMOTE.applicable(plain, profile, TaskType.BINARY, {"k": 3})
    # The Setup's key changes only when a resampler is set, so earlier Setups keep their keys
    assert "resampler" not in plain.setup() and changed.setup()["resampler"] == "smote"
    assert changed.setup_key() != plain.setup_key()


def test_class_imbalance_proposes_smote_after_the_other_answers():
    frame, y = imbalanced()
    profile = DatasetProfiler().profile(frame.assign(label=y), "label")
    imbalance = Diagnosis(DiagnosisKind.CLASS_IMBALANCE, Severity.HIGH, (), {"ratio": 4.0}, "imbalanced", "v")
    plain = ExperimentConfig(target_column="label", primary_metric="f1")
    ids = [p.spec.strategy_id for p in propose([imbalance], plain, profile, TaskType.BINARY, set())]
    assert ids == ["class_weight_balanced", "smoother_search_metric", "smote"]
    weighted = ExperimentConfig(target_column="label", primary_metric="f1", class_weight="balanced")
    assert "smote" not in [p.spec.strategy_id for p in propose([imbalance], weighted, profile, TaskType.BINARY, set())]
    tried = {StrategySpec("class_weight_balanced").key()}
    resampled = SMOTE.apply(plain, {})
    assert [p.spec.strategy_id for p in propose([imbalance], resampled, profile, TaskType.BINARY, tried)] == [
        "smoother_search_metric"]


def test_a_setup_with_a_resampler_is_checked(tmp_path):
    frame, y = imbalanced(text=False)
    data = frame.assign(label=y)
    runner = AutoMLRunner(settings=build_settings(tmp_path), enable_mlflow=False)
    for config, message in [
        (ExperimentConfig(target_column="label", resampler="adasyn"), "Unknown resampler"),
        (ExperimentConfig(target_column="label", resampler="smote", class_weight="balanced"), "class weights"),
        (ExperimentConfig(target_column="a", resampler="smote", task_type=TaskType.REGRESSION), "classification"),
    ]:
        with pytest.raises(ExperimentError, match=message):
            runner.prepare(data, config)


# --------------------------------------------------------------------------- validation rows never resampled
class RecordingResamplerEngine:
    """Records the rows of every refit and the resampler it was asked for."""

    name = "recording"

    def __init__(self):
        self.calls = []

    def fit_config(self, X_train, y_train, *, task_type, estimator, config, seed, n_jobs, sample_weight=None,
                   resampler=None):
        self.calls.append({"index": X_train.index, "resampler": resampler})
        return NumericLogistic(X_train, y_train, sample_weight)


def test_the_assessor_resamples_only_the_training_rows_of_each_fold():
    X, y, _ = training_part()
    engine = RecordingResamplerEngine()
    folds, _ = make_session_folds(y, TaskType.BINARY, seed=1)
    assessor = Assessor(engine, folds, TaskType.BINARY, "f1", n_jobs=1, seed=0)
    smote = ExperimentConfig(**{**CONFIG.__dict__, "resampler": "smote"})
    assessment = assessor.assess(X, y, smote, result_for())
    assert len(engine.calls) == 15 and {c["resampler"] for c in engine.calls} == {"smote"}
    for call, (train_positions, validation_positions) in zip(engine.calls, folds.folds):
        assert list(call["index"]) == list(X.index[train_positions])  # the resampler gets these rows only
        assert not set(call["index"]) & set(X.index[validation_positions])
    # the refits without a resampler are other refits: assessed again, and the engine is asked for none
    plain = assessor.assess(X, y, CONFIG, result_for(experiment_id="EXP-002"))
    assert plain.reused_from is None and engine.calls[-1]["resampler"] is None and not assessment.failed


@pytest.mark.slow
def test_flaml_resamples_fold_training_rows_and_never_validation_rows(monkeypatch):
    """With the real engine: FoldSafeSmote is called with the rows of each fold's training part (and the
    refit's rows), never with a number of rows that includes a validation part."""
    sizes = []
    original = FoldSafeSmote.fit_resample

    def recording(self, X, y):
        sizes.append(len(y))
        return original(self, X, y)

    monkeypatch.setattr(FoldSafeSmote, "fit_resample", recording)
    X, y = imbalanced(n=300)
    model = FLAMLEngine().fit_config(X, y, task_type=TaskType.BINARY, estimator="lgbm",
                                     config={"n_estimators": 8, "num_leaves": 4}, seed=0, n_jobs=1, resampler="smote")
    assert sizes == [300] and set(model.predict(X)) <= {"pos", "neg"}
    sizes.clear()
    result = FLAMLEngine().fit(X, y, task_type=TaskType.BINARY, metric="f1", time_budget=5, seed=0, n_jobs=1,
                               estimator_list=["lgbm"], work_dir=__import__("tempfile").mkdtemp(), max_iter=3,
                               resampler="smote")
    assert result.best_estimator == "lgbm" and sizes and max(sizes) <= 300 and 300 in sizes
    assert all(s < 300 for s in sizes[:-1])  # the search's training partitions are smaller than all rows
