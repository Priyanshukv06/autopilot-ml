"""Tests for the session splits (spec task 2.1)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from app.services.splits import SplitError, make_outer_split, make_session_folds
from app.services.task_detector import TaskType

ROOT = Path(__file__).resolve().parents[1]


def titanic_target() -> pd.Series:
    return pd.read_csv(ROOT / "data" / "sample" / "titanic.csv")["Survived"]


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_outer_split_matches_phase1_rule(seed):
    """The Phase 1 runner called train_test_split on the data itself; positions give the same rows."""
    df = pd.read_csv(ROOT / "data" / "sample" / "titanic.csv")
    y = df["Survived"]
    X = df.drop(columns=["Survived"])
    _, X_test_old, _, _ = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    split, warnings = make_outer_split(y, TaskType.BINARY, 0.2, seed)
    assert warnings == []
    assert split.stratified
    assert list(X.index[split.test_positions]) == list(X_test_old.index)


def test_titanic_seed_42_sizes_and_disjoint_parts():
    split, _ = make_outer_split(titanic_target(), TaskType.BINARY, 0.2, 42)
    assert (split.n_train, split.n_test) == (712, 179)
    assert not set(split.train_positions) & set(split.test_positions)
    assert sorted([*split.train_positions, *split.test_positions]) == list(range(891))


def test_outer_split_order_matches_saved_exp001_predictions():
    """EXP-001 (Phase 1) saved its test answers in split order; the new rule must give the same order."""
    predictions = ROOT / "experiments" / "EXP-001" / "predictions.csv"
    if not predictions.exists():
        pytest.skip("EXP-001 results are not in this copy of the project")
    y = titanic_target()
    split, _ = make_outer_split(y, TaskType.BINARY, 0.2, 42)
    saved = pd.read_csv(predictions)["y_true"].tolist()
    assert y.iloc[split.test_positions].tolist() == saved


def test_regression_split_is_not_stratified():
    y = pd.Series(np.linspace(0, 1, 100))
    split, _ = make_outer_split(y, TaskType.REGRESSION, 0.25, 0)
    assert not split.stratified
    assert split.n_test == 25


def test_outer_split_without_stratification_when_a_class_is_tiny():
    y = pd.Series(["a"] * 30 + ["b"] * 30 + ["c"])
    split, warnings = make_outer_split(y, TaskType.MULTICLASS, 0.2, 0)
    assert not split.stratified
    assert any("Stratified split disabled" in w for w in warnings)


def test_session_folds_cover_the_training_part():
    y_train = titanic_target().iloc[:712].reset_index(drop=True)
    folds, warnings = make_session_folds(y_train, TaskType.BINARY, n_splits=5, n_repeats=3, seed=43)
    assert warnings == []
    assert len(folds) == 15 and folds.stratified
    assert folds.test_train_ratio == pytest.approx(0.25)
    for repeat in range(3):
        validation = np.concatenate([v for _, v in folds.folds[repeat * 5:(repeat + 1) * 5]])
        assert sorted(validation) == list(range(712))  # each row is validated once per repeat
    for train, validation in folds.folds:
        assert not set(train) & set(validation)
        assert max(train.max(), validation.max()) < 712  # only training-part positions


def test_session_folds_are_stratified():
    y_train = pd.Series([1] * 40 + [0] * 160)
    folds, _ = make_session_folds(y_train, TaskType.BINARY, n_splits=5, n_repeats=1, seed=0)
    for _, validation in folds.folds:
        assert y_train.iloc[validation].mean() == pytest.approx(0.2)


def test_session_folds_for_regression_are_plain():
    folds, _ = make_session_folds(pd.Series(np.arange(50.0)), TaskType.REGRESSION, n_splits=5, n_repeats=2, seed=0)
    assert not folds.stratified
    assert len(folds) == 10


def test_session_folds_fall_back_when_a_class_is_too_small():
    y_train = pd.Series(["a"] * 20 + ["b"] * 20 + ["c"] * 3)
    folds, warnings = make_session_folds(y_train, TaskType.MULTICLASS, n_splits=5, n_repeats=1, seed=0)
    assert not folds.stratified
    assert any("not stratified" in w for w in warnings)


def test_fingerprints_are_stable_and_sensitive():
    y = titanic_target()
    a, _ = make_outer_split(y, TaskType.BINARY, 0.2, 42)
    b, _ = make_outer_split(y, TaskType.BINARY, 0.2, 42)
    c, _ = make_outer_split(y, TaskType.BINARY, 0.2, 43)
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()
    y_train = y.iloc[a.train_positions].reset_index(drop=True)
    f1, _ = make_session_folds(y_train, TaskType.BINARY, seed=1)
    f2, _ = make_session_folds(y_train, TaskType.BINARY, seed=1)
    f3, _ = make_session_folds(y_train, TaskType.BINARY, seed=2)
    assert f1.fingerprint() == f2.fingerprint() != f3.fingerprint()
    assert a.to_dict()["fingerprint"] == a.fingerprint()


def test_invalid_fold_settings_raise():
    y_train = pd.Series([0, 1] * 10)
    with pytest.raises(SplitError):
        make_session_folds(y_train, TaskType.BINARY, n_splits=1)
    with pytest.raises(SplitError):
        make_session_folds(y_train, TaskType.BINARY, n_repeats=0)
    with pytest.raises(SplitError):
        make_session_folds(y_train.iloc[:3], TaskType.BINARY, n_splits=5)
