"""Tests for the Strategy_Catalogue, thin version (task 7.1 of the spec).

Requirement 7.6: every Strategy has a test of the exact change it makes to a Setup.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from app.services.automl_runner import ExperimentConfig, FLAMLEngine
from app.services.profiler import DatasetProfiler
from app.services.strategies import (
    CATALOGUE,
    REGULARISED_LIMITS,
    SEARCH_SPACE_LIMITS_ALLOWED,
    StrategySpec,
    get_strategy,
    merge_limits,
)
from app.services.task_detector import TaskType

DROP, WEIGHTS, SMALLER = (CATALOGUE[i] for i in ("drop_columns", "class_weight_balanced", "regularised_search_space"))


def profile(columns=("a", "b", "code"), target="y"):
    df = pd.DataFrame({c: np.arange(20) for c in columns})
    df[target] = [0, 1] * 10
    return DatasetProfiler().profile(df, target)


def config(**changes):
    base = ExperimentConfig(target_column="y", excluded_columns=["a"], estimator_list=["lgbm", "xgboost"])
    return ExperimentConfig(**{**base.__dict__, **changes})


# --------------------------------------------------------------------------- catalogue
def test_the_thin_catalogue():
    assert list(CATALOGUE) == ["drop_columns", "class_weight_balanced", "regularised_search_space"]
    described = {s.strategy_id: (s.kind, s.complexity, s.cost_class) for s in CATALOGUE.values()}
    assert described == {
        "drop_columns": ("data", "simpler", "cheap"),
        "class_weight_balanced": ("model", "same", "cheap"),
        "regularised_search_space": ("search", "simpler", "cheap"),
    }
    assert all(s.title for s in CATALOGUE.values())
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("no_such_strategy")


def test_spec_key_is_canonical():
    a = StrategySpec("drop_columns", {"columns": ["b"]})
    assert a.key() == '{"params":{"columns":["b"]},"strategy":"drop_columns"}'
    assert StrategySpec("x", {"p": 1, "q": 2}).key() == StrategySpec("x", {"q": 2, "p": 1}).key()
    assert a.key() != StrategySpec("drop_columns", {"columns": ["code"]}).key()
    assert a.to_dict() == {"strategy_id": "drop_columns", "params": {"columns": ["b"]}}


@pytest.mark.parametrize(
    "strategy_id, params",
    [("drop_columns", {"columns": ["b"]}), ("class_weight_balanced", {}), ("regularised_search_space", {})],
)
def test_apply_returns_a_new_setup_and_leaves_the_original_unchanged(strategy_id, params):
    original = config(search_space_limits={"lgbm": {"num_leaves": {"upper": 16}}})
    before = copy.deepcopy(original)
    changed = get_strategy(strategy_id).apply(original, params)
    assert changed is not original and original == before
    assert changed.setup_key() != original.setup_key()
    # Nothing is shared: changing the new Setup's lists and dicts leaves the original as it was.
    changed.excluded_columns.append("zzz")
    changed.search_space_limits.setdefault("rf", {})["max_leaves"] = {"upper": 1}
    changed.estimator_list.append("rf")
    assert original == before


# --------------------------------------------------------------------------- drop_columns
def test_drop_columns_exact_change():
    changed = DROP.apply(config(), {"columns": ["code"]})
    assert changed.excluded_columns == ["a", "code"]
    assert changed.__dict__ == {**config().__dict__, "excluded_columns": ["a", "code"]}
    assert DROP.describe({"columns": ["code"]}) == "Stop using the column 'code'"


def test_drop_columns_applies_only_to_columns_in_use():
    p = profile()
    assert DROP.applicable(config(), p, TaskType.BINARY, {"columns": ["code"]})
    assert not DROP.applicable(config(), p, TaskType.BINARY, {"columns": ["a"]})  # already excluded
    assert not DROP.applicable(config(), p, TaskType.BINARY, {"columns": ["y"]})  # the target
    assert not DROP.applicable(config(), p, TaskType.BINARY, {"columns": ["missing"]})
    assert not DROP.applicable(config(), p, TaskType.BINARY, {"columns": ["b", "code"]})  # nothing would remain


@pytest.mark.parametrize("params", [{}, {"columns": []}, {"columns": "code"}, {"columns": ["b", "b"]},
                                    {"columns": ["b"], "extra": 1}, {"columns": [3]}])
def test_drop_columns_rejects_malformed_parameters(params):
    with pytest.raises(ValueError, match="drop_columns needs"):
        DROP.applicable(config(), profile(), TaskType.BINARY, params)


# --------------------------------------------------------------------------- class weights
def test_class_weight_balanced_exact_change():
    changed = WEIGHTS.apply(config(), {})
    assert changed.__dict__ == {**config().__dict__, "class_weight": "balanced"}


def test_class_weights_apply_only_to_classification_without_weights():
    p = profile()
    assert WEIGHTS.applicable(config(), p, TaskType.BINARY, {})
    assert WEIGHTS.applicable(config(), p, TaskType.MULTICLASS, {})
    assert not WEIGHTS.applicable(config(), p, TaskType.REGRESSION, {})
    assert not WEIGHTS.applicable(config(class_weight="balanced"), p, TaskType.BINARY, {})
    with pytest.raises(ValueError, match="no parameters"):
        WEIGHTS.applicable(config(), p, TaskType.BINARY, {"weight": 2})


# --------------------------------------------------------------------------- smaller trees
def test_regularised_search_space_exact_change():
    changed = SMALLER.apply(config(), {})
    assert changed.search_space_limits == {
        "lgbm": {"num_leaves": {"upper": 32}, "min_child_samples": {"lower": 10}},
        "xgboost": {"max_leaves": {"upper": 32}, "min_child_weight": {"lower": 1}},
        "xgb_limitdepth": {"max_depth": {"upper": 4}},
        "rf": {"max_leaves": {"upper": 64}},
        "extra_tree": {"max_leaves": {"upper": 64}},
    }
    assert changed.__dict__ == {**config().__dict__, "search_space_limits": changed.search_space_limits}


def test_regularised_search_space_keeps_tighter_limits_and_applies_once():
    tighter = config(search_space_limits={"lgbm": {"num_leaves": {"upper": 8}, "min_child_samples": {"lower": 40}}})
    changed = SMALLER.apply(tighter, {})
    assert changed.search_space_limits["lgbm"] == {"num_leaves": {"upper": 8}, "min_child_samples": {"lower": 40}}
    assert changed.search_space_limits["rf"] == {"max_leaves": {"upper": 64}}
    p = profile()
    assert SMALLER.applicable(tighter, p, TaskType.BINARY, {})
    assert not SMALLER.applicable(changed, p, TaskType.BINARY, {})  # applying it again changes nothing


def test_merge_limits_never_widens():
    current = {"lgbm": {"num_leaves": {"upper": 8, "lower": 4}}}
    merged = merge_limits(current, {"lgbm": {"num_leaves": {"upper": 64, "lower": 2}}})
    assert merged == current and merged is not current


@pytest.mark.parametrize("task_type", [TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION])
def test_regularised_limits_fit_flamls_search_spaces(task_type):
    """Every limited setting exists in FLAML's space, is numeric and really narrows it on Titanic-sized data."""
    custom = FLAMLEngine().custom_hp(REGULARISED_LIMITS, task_type, (712, 9))
    searched_max = {(e, s): spec["domain"].upper - 1 for e, settings in custom.items() for s, spec in settings.items()}
    assert searched_max[("lgbm", "num_leaves")] == 32 and searched_max[("xgboost", "max_leaves")] == 32
    assert searched_max[("xgb_limitdepth", "max_depth")] == 4
    assert searched_max[("rf", "max_leaves")] == searched_max[("extra_tree", "max_leaves")] == 64
    assert custom["lgbm"]["min_child_samples"]["domain"].lower == 10
    assert custom["xgboost"]["min_child_weight"]["domain"].lower == 1.0
    assert SEARCH_SPACE_LIMITS_ALLOWED == {e: tuple(s) for e, s in REGULARISED_LIMITS.items()}
