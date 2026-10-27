"""The dataset registry (task 20): the selection rules on hand-made candidates, and the committed lists."""

import pandas as pd
import pytest

from research import datasets as ds


def candidate(openml_id, classes=2, missing=0, categorical=0, ratio=1.2, rows=1000, features=10):
    return ds.Candidate(openml_id, f"d{openml_id}", 1, "suite", rows, features, classes, missing, categorical,
                        None if classes == 0 else ratio)


def test_each_candidate_falls_into_at_most_one_group():
    assert ds.group_of(candidate(1, ratio=3.0)) == "binary_imbalanced"
    assert ds.group_of(candidate(2, missing=5)) == "binary_messy"
    assert ds.group_of(candidate(3)) is None  # balanced binary with only numeric columns
    assert ds.group_of(candidate(4, classes=4)) == "multiclass_clean"
    assert ds.group_of(candidate(5, classes=4, ratio=5.0)) == "multiclass_messy"
    assert ds.group_of(candidate(6, classes=0)) == "regression_clean"
    assert ds.group_of(candidate(7, classes=0, categorical=2)) == "regression_messy"
    assert not ds.eligible(candidate(8, rows=499)) and not ds.eligible(candidate(9, features=101))
    assert not ds.eligible(candidate(1510))  # the breast_cancer sample's data


def pool():
    kinds = [dict(ratio=4.0), dict(categorical=3), dict(classes=3), dict(classes=3, missing=1), dict(classes=0),
             dict(classes=0, categorical=1)]
    return [candidate(100 * g + i, **kind) for g, kind in enumerate(kinds, start=1) for i in range(4)]


def test_selection_is_fixed_by_the_seed_and_never_overlaps():
    development, evaluation = ds.select(pool())
    assert (development, evaluation) == ds.select(list(reversed(pool())))  # the order of the pool does not matter
    assert [ds.group_of(c) for c in evaluation] == list(ds.GROUPS) == [ds.group_of(c) for c in development]
    assert not {c.openml_id for c in development} & {c.openml_id for c in evaluation}
    assert ds.select(pool(), seed=1) != (development, evaluation)
    with pytest.raises(ValueError, match="2 are needed"):
        ds.select([c for c in pool() if c.openml_id != 101 and ds.group_of(c) != "binary_imbalanced"] + [candidate(101, ratio=4.0)])


def test_describe_uses_the_given_task():
    frame = pd.DataFrame({"x": [1.0, None, 3.0, 4.0], "c": ["a", "b", "a", "b"], "y": [1, 1, 1, 2]})
    stats = ds.describe(frame, "y", "regression")
    assert stats["imbalance_ratio"] is None and stats["detected_task"] == "binary"
    assert stats["missing_share"] == 0.125 and stats["categorical_columns"] == 1 and stats["columns"] == 2
    assert ds.describe(frame, "y", "binary")["imbalance_ratio"] == 3.0


def test_the_committed_registry():
    entries = ds.registry()
    development, evaluation = ds.datasets("development"), ds.datasets("evaluation")
    assert len(development) == 10 and len(evaluation) == 6 and len(entries) == 16
    assert {e.source for e in evaluation} <= set(ds.SUITES.values())  # no sample is used for evaluation
    assert not {e.openml_id for e in development} & {e.openml_id for e in evaluation}
    for entry in entries.values():
        assert entry.target and entry.task in ("binary", "multiclass", "regression") and entry.licence
        if entry.source != "sample":
            assert entry.openml_id and entry.version and entry.url.endswith(str(entry.openml_id))
            assert ds.MIN_ROWS <= entry.rows <= ds.MAX_ROWS and entry.columns <= ds.MAX_FEATURES
    assert {e.task for e in evaluation} == {"binary", "multiclass", "regression"}
