"""Tests for the synthetic data generator (task 4.1 of the spec)."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from app.services.profiler import DatasetProfiler
from app.services.task_detector import TaskType
from research.synthetic import BASE_COLUMNS, PROBLEM_KINDS, TARGET, make_synthetic

ALL_PROBLEMS = [
    ("class_imbalance", 4), ("label_noise", 0.1), ("duplicated_columns", 0.9), ("useless_features", 3),
    ("identifier_column", 1), ("constant_column", 1), ("high_cardinality", 2), ("missing_values", 0.3),
]


def both_parts(data):
    return [data.frame, data.fresh]


# --------------------------------------------------------------------------- repeatability
def test_same_seed_gives_the_same_data_and_another_seed_does_not():
    a = make_synthetic("binary", 300, ALL_PROBLEMS, seed=5, fresh_rows=1000)
    b = make_synthetic("binary", 300, ALL_PROBLEMS, seed=5, fresh_rows=1000)
    c = make_synthetic("binary", 300, ALL_PROBLEMS, seed=6, fresh_rows=1000)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    pd.testing.assert_frame_equal(a.fresh, b.fresh)
    assert a.planted == b.planted and a.name == b.name
    assert not a.frame[list(BASE_COLUMNS)].equals(c.frame[list(BASE_COLUMNS)])


def test_problem_order_does_not_matter_and_base_rows_are_shared():
    problems = [("missing_values", 0.2), ("useless_features", 2), ("identifier_column", 0.5)]
    forward = make_synthetic("binary", 200, problems, seed=3, fresh_rows=500)
    backward = make_synthetic("binary", 200, problems[::-1], seed=3, fresh_rows=500)
    clean = make_synthetic("binary", 200, [], seed=3, fresh_rows=500)
    pd.testing.assert_frame_equal(forward.frame, backward.frame)
    untouched = [c for c in BASE_COLUMNS if c not in forward.problem("missing_values").columns] + [TARGET]
    pd.testing.assert_frame_equal(forward.frame[untouched], clean.frame[untouched])


# --------------------------------------------------------------------------- planted problems
@pytest.mark.parametrize("task, ratio", [("binary", 4), ("multiclass", 9)])
def test_class_imbalance_ratio(task, ratio):
    data = make_synthetic(task, 2000, [("class_imbalance", ratio)], seed=0, fresh_rows=4000)
    for part in both_parts(data):
        counts = part[TARGET].value_counts()
        assert counts.max() / counts.min() == pytest.approx(ratio, rel=0.05)
    assert data.problem("class_imbalance").columns == (TARGET,)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_label_noise_changes_exactly_the_requested_share(task):
    noisy = make_synthetic(task, 1000, [("label_noise", 0.15)], seed=1, fresh_rows=2000)
    clean = make_synthetic(task, 1000, [], seed=1, fresh_rows=2000)
    for noisy_part, clean_part in zip(both_parts(noisy), both_parts(clean)):
        assert (noisy_part[TARGET] != clean_part[TARGET]).mean() == pytest.approx(0.15)
        if task != "regression":  # labels move to another existing class
            assert set(noisy_part[TARGET]) <= set(clean_part[TARGET])


def test_duplicated_columns_have_the_requested_correlation():
    data = make_synthetic("binary", 2000, [("duplicated_columns", 0.9)], seed=2, fresh_rows=4000)
    columns = data.problem("duplicated_columns").columns
    assert len(columns) == 4 and columns[1::2] == ("x8", "x9")
    for part in both_parts(data):
        for source, copy in zip(columns[::2], columns[1::2]):
            assert part[source].corr(part[copy]) == pytest.approx(0.9, abs=0.02)


def test_useless_features_are_added_and_unrelated_to_the_target():
    data = make_synthetic("binary", 500, [("useless_features", 3)], seed=4)
    assert data.problem("useless_features").columns == ("x8", "x9", "x10")
    for column in ("x8", "x9", "x10"):
        assert abs(data.fresh[column].corr(data.fresh[TARGET])) < 0.05  # 20,000 rows


@pytest.mark.parametrize("share", [0.5, 1.0])
def test_identifier_column_share_of_distinct_values(share):
    data = make_synthetic("binary", 400, [("identifier_column", share)], seed=5, fresh_rows=800)
    for part in both_parts(data):
        assert part["record_id"].nunique() / len(part) == pytest.approx(share)
        assert part["record_id"].dtype == np.int64
    assert not set(data.frame["record_id"]) & set(data.fresh["record_id"])  # new rows, new IDs
    if share == 1.0:
        assert DatasetProfiler().profile(data.frame, TARGET).id_like_columns == ["record_id"]


@pytest.mark.parametrize("share", [0.8, 1.0])
def test_constant_column_share_of_the_main_value(share):
    data = make_synthetic("binary", 400, [("constant_column", share)], seed=6, fresh_rows=800)
    for part in both_parts(data):
        assert (part["flag"] == 1).mean() == pytest.approx(share)
    if share == 1.0:
        assert DatasetProfiler().profile(data.frame, TARGET).constant_columns == ["flag"]


def test_high_cardinality_column_gives_the_expected_unseen_share():
    strength, rows = 2, 1000
    data = make_synthetic("binary", rows, [("high_cardinality", strength)], seed=7)
    profile = DatasetProfiler().profile(data.frame, TARGET)
    assert profile.high_cardinality_columns == ["code"]
    train, valid = train_test_split(data.frame["code"], test_size=0.2, random_state=0)
    unseen = (~valid.isin(set(train))).mean()
    assert unseen == pytest.approx(math.exp(-0.8 / strength), abs=0.1)  # 0.67
    # Both parts draw from the same strength x rows categories.
    assert data.fresh["code"].nunique() <= strength * rows
    assert set(data.frame["code"]) | set(data.fresh["code"]) <= {f"C{i:04d}" for i in range(strength * rows)}


def test_missing_values_share_is_exact_in_the_chosen_columns():
    data = make_synthetic("regression", 500, [("missing_values", 0.3)], seed=8, fresh_rows=1000)
    columns = list(data.problem("missing_values").columns)
    assert len(columns) == 2
    for part in both_parts(data):
        assert part[columns].isna().mean().tolist() == pytest.approx([0.3, 0.3])
        assert part.drop(columns=columns).notna().all().all()


def test_every_problem_kind_at_once():
    data = make_synthetic("binary", 1000, ALL_PROBLEMS, seed=0, fresh_rows=2000)
    assert [p.kind for p in data.planted] == list(PROBLEM_KINDS)  # every kind, in the fixed order
    assert list(data.frame.columns) == list(data.fresh.columns)
    assert data.name.startswith("synthetic-binary-class_imbalance4-") and data.name.endswith("-seed0")


# --------------------------------------------------------------------------- clean data
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_clean_datasets_have_no_planted_problem(task):
    data = make_synthetic(task, 1000, [], seed=9, fresh_rows=2000)
    assert data.planted == () and data.problem("label_noise") is None
    assert data.task_type == TaskType(task) and data.name == f"synthetic-{task}-clean-seed9"
    assert list(data.frame.columns) == [*BASE_COLUMNS, TARGET]
    assert data.frame.notna().all().all()
    profile = DatasetProfiler().profile(data.frame, TARGET, task_type=data.task_type)
    assert profile.id_like_columns == profile.constant_columns == profile.high_cardinality_columns == []
    correlations = data.frame[list(BASE_COLUMNS)].corr().abs().to_numpy()
    assert correlations[~np.eye(len(BASE_COLUMNS), dtype=bool)].max() < 0.95  # no near-copies
    if task != "regression":
        counts = data.frame[TARGET].value_counts()
        assert counts.max() / counts.min() < 1.1


# --------------------------------------------------------------------------- fresh sample
def test_fresh_sample_comes_from_the_same_generator():
    data = make_synthetic("binary", 1000, [], seed=10)
    assert len(data.fresh) == 20_000 and (data.fresh.dtypes == data.frame.dtypes).all()
    X, y = data.frame[list(BASE_COLUMNS)], data.frame[TARGET]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    cv_accuracy = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5).mean()
    fresh_accuracy = accuracy_score(data.fresh[TARGET], model.predict(data.fresh[list(BASE_COLUMNS)]))
    assert fresh_accuracy == pytest.approx(cv_accuracy, abs=0.05)


def test_regression_fresh_sample_reaches_the_designed_best_score():
    data = make_synthetic("regression", 1000, [], seed=12)
    model = LinearRegression().fit(data.frame[list(BASE_COLUMNS)], data.frame[TARGET])
    assert r2_score(data.fresh[TARGET], model.predict(data.fresh[list(BASE_COLUMNS)])) == pytest.approx(0.9, abs=0.02)


# --------------------------------------------------------------------------- invalid requests
@pytest.mark.parametrize(
    "task, rows, problems, message",
    [
        ("ranking", 100, [], "Unknown task"),
        ("binary", 10, [], "at least"),
        ("binary", 100, [("no_such_problem", 1)], "Unknown problem"),
        ("binary", 100, [("label_noise", 0.1), ("label_noise", 0.2)], "listed twice"),
        ("regression", 100, [("class_imbalance", 3)], "not valid"),
        ("binary", 100, [("class_imbalance", 1)], "not valid"),
        ("binary", 100, [("label_noise", 0.5)], "not valid"),
        ("binary", 100, [("useless_features", 1.5)], "not valid"),
        ("binary", 100, [("constant_column", 0.5)], "not valid"),
        ("binary", 100, [("missing_values", 1)], "not valid"),
        ("binary", 30, [("class_imbalance", 50)], "Too few rows"),
    ],
)
def test_invalid_requests_are_rejected(task, rows, problems, message):
    with pytest.raises(ValueError, match=message):
        make_synthetic(task, rows, problems, seed=0, fresh_rows=100)
