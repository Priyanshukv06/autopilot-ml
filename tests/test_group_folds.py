"""Tests for group-aware splits and folds (optional task 31, Requirement 28)."""

import numpy as np
import pandas as pd
import pytest

from app.services.automl_runner import ExperimentConfig, ExperimentError
from app.services.splits import SplitError, make_outer_split, make_session_folds
from app.services.task_detector import TaskType
from tests.test_autopilot import DATA, run_session


def grouped(n=240, n_groups=40, seed=0):
    rng = np.random.default_rng(seed)
    groups = pd.Series(rng.integers(0, n_groups, n), name="site").map(lambda g: f"site{g}")
    y = pd.Series(np.where(rng.random(n) < 0.3, "b", "a"), name="y")
    return y, groups


def test_the_outer_split_keeps_every_group_on_one_side():
    y, groups = grouped()
    split, notes = make_outer_split(y, TaskType.BINARY, 0.2, 7, groups=groups)
    train, test = set(groups.iloc[split.train_positions]), set(groups.iloc[split.test_positions])
    assert not train & test and len(train | test) == groups.nunique()
    assert sorted([*split.train_positions, *split.test_positions]) == list(range(len(y)))
    assert len(test) == int(np.ceil(0.2 * groups.nunique()))  # test_size is the share of groups
    assert split.grouped and not split.stratified and split.to_dict()["grouped"] is True
    assert "Grouped split" in notes[0]
    plain, _ = make_outer_split(y, TaskType.BINARY, 0.2, 7)
    assert "grouped" not in plain.to_dict() and plain.fingerprint() != split.fingerprint()
    again, _ = make_outer_split(y, TaskType.BINARY, 0.2, 7, groups=groups)
    assert again.fingerprint() == split.fingerprint()  # fixed by the seed


@pytest.mark.parametrize("task_type", [TaskType.BINARY, TaskType.REGRESSION])
def test_every_session_fold_keeps_each_group_inside_one_fold(task_type):
    y, groups = grouped()
    if task_type is TaskType.REGRESSION:
        y = pd.Series(np.arange(len(y), dtype=float))
    folds, notes = make_session_folds(y, task_type, n_splits=5, n_repeats=3, seed=1, groups=groups)
    assert len(folds) == 15 and folds.grouped and folds.stratified is task_type.is_classification
    for repeat in range(3):
        seen = []
        for train, validation in folds.folds[5 * repeat:5 * repeat + 5]:
            assert not set(groups.iloc[train]) & set(groups.iloc[validation])
            seen.extend(validation)
        assert sorted(seen) == list(range(len(y)))  # each row is a validation row once per repeat
    assert [list(v) for _, v in folds.folds[:5]] != [list(v) for _, v in folds.folds[5:10]]  # repeats differ
    plain, _ = make_session_folds(y, task_type, n_splits=5, n_repeats=3, seed=1)
    assert folds.fingerprint() != plain.fingerprint() and "grouped" not in plain.to_dict()
    assert f"Grouped folds: {groups.nunique()} groups" in notes[-1]


def test_group_problems_are_refused():
    y, groups = grouped(n_groups=4)
    with pytest.raises(SplitError, match="at least 5 groups"):
        make_session_folds(y, TaskType.BINARY, n_splits=5, groups=groups)
    with pytest.raises(SplitError, match="empty cells"):
        make_session_folds(y, TaskType.BINARY, n_splits=2, groups=groups.mask(groups.index < 3))
    with pytest.raises(SplitError, match="group values for"):
        make_outer_split(y, TaskType.BINARY, 0.2, 0, groups=groups.head(10))
    with pytest.raises(SplitError, match="at least 2 groups"):
        make_outer_split(y, TaskType.BINARY, 0.2, 0, groups=pd.Series(["one"] * len(y)))


def test_a_session_with_a_group_column_records_it_and_never_uses_it(tmp_path):
    data = DATA.assign(site=[f"s{i % 30}" for i in range(len(DATA))])
    result, record, engine, assessor, _ = run_session(tmp_path, data=data, group_column="site",
                                                      shifts={1: 0.0, 2: 0.0, 3: -0.05})
    assert result.status == "completed"
    assert record["settings"]["group_column"] == "site" and record["splits"]["grouped"] is True
    assert "site" in record["settings"]["base"]["excluded_columns"]
    assert all("site" not in fit["columns"] for fit in engine.fits)  # never a feature
    assert any("grouped by 'site'" in note for note in record["notes"])
    labelled = data.loc[data["target"].notna()]
    train_sites = set(labelled.loc[list(engine.fits[0]["index"]), "site"])
    test_sites = set(labelled["site"]) - train_sites
    assert test_sites and not train_sites & test_sites  # the test part is made of whole groups
    split, _ = make_outer_split(labelled["target"], TaskType.BINARY, 0.2, 42, groups=labelled["site"])
    assert split.fingerprint() == record["splits"]["outer"]
    sites = labelled["site"].iloc[split.train_positions].to_numpy()
    assert assessor.folds.grouped and len(assessor.folds) == 15
    for train, validation in assessor.folds.folds:  # the Assessor's folds keep each site inside one fold
        assert not set(sites[train]) & set(sites[validation])
    plain = run_session(tmp_path / "plain", shifts={1: 0.0, 2: 0.0, 3: -0.05})[1]
    assert plain["settings"]["group_column"] is None and plain["splits"]["grouped"] is False


@pytest.mark.parametrize("column, message", [("nope", "not found"), ("target", "cannot be the target")])
def test_a_wrong_group_column_is_refused_before_anything_is_written(tmp_path, column, message):
    with pytest.raises(ExperimentError, match=message):
        run_session(tmp_path, group_column=column)
    assert not any((tmp_path / "experiments").iterdir())
