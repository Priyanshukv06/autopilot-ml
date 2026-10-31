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
def test_the_catalogue():
    described = {s.strategy_id: (s.kind, s.complexity, s.cost_class) for s in CATALOGUE.values()}
    assert described == {  # the table of design.md section 7
        "drop_columns": ("data", "simpler", "cheap"),
        "class_weight_balanced": ("model", "same", "cheap"),
        "regularised_search_space": ("search", "simpler", "cheap"),
        "frequency_encode": ("data", "same", "cheap"),
        "group_rare_categories": ("data", "simpler", "cheap"),
        "missing_indicator": ("data", "more_complex", "cheap"),
        "restrict_model_families": ("model", "simpler", "cheap"),
        "more_search_time": ("budget", "same", "expensive"),
        "smoother_search_metric": ("search", "same", "normal"),
        "smote": ("data", "more_complex", "normal"),  # optional task 29
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


# =========================================================================== the full catalogue (task 14.1)
from app.core.config import build_settings  # noqa: E402
from app.services.automl_runner import AutoMLRunner  # noqa: E402
from app.services.transforms import DataStepSpec  # noqa: E402

ENCODE, GROUP, INDICATOR, FAMILIES, MORE, SMOOTH = (CATALOGUE[i] for i in (
    "frequency_encode", "group_rare_categories", "missing_indicator", "restrict_model_families", "more_search_time",
    "smoother_search_metric"))


def mixed_frame():
    """20 rows: a text column, a numeric column with empty cells, a full numeric column and the target."""
    return pd.DataFrame({
        "code": [f"c{i % 7}" for i in range(20)],
        "age": [np.nan if i % 4 == 0 else float(i) for i in range(20)],
        "a": np.arange(20, dtype=float),
        "y": [0, 1] * 10,
    })


def mixed_profile():
    return DatasetProfiler().profile(mixed_frame(), "y")


def setup(**changes):
    return ExperimentConfig(**{**ExperimentConfig(target_column="y").__dict__, **changes})


@pytest.mark.parametrize("strategy, step_kind", [(ENCODE, "frequency_encode"), (GROUP, "group_rare_categories"),
                                                 (INDICATOR, "missing_indicator")])
def test_column_strategies_add_exactly_one_data_step(strategy, step_kind, tmp_path):
    column = "age" if strategy is INDICATOR else "code"
    original = setup(data_steps=[DataStepSpec("drop_columns", ("a",))])
    changed = strategy.apply(original, {"columns": [column]})
    assert changed.data_steps == [DataStepSpec("drop_columns", ("a",)), DataStepSpec(step_kind, (column,))]
    assert changed.__dict__ == {**original.__dict__, "data_steps": changed.data_steps}
    assert original.data_steps == [DataStepSpec("drop_columns", ("a",))]
    # The changed Setup is runnable: the runner's check of the Data_Steps accepts it.
    AutoMLRunner(settings=build_settings(tmp_path), enable_mlflow=False).prepare(mixed_frame(), changed)


def test_encodings_apply_to_text_columns_in_use_only_once():
    p = mixed_profile()
    for strategy in (ENCODE, GROUP):
        assert strategy.applicable(setup(), p, TaskType.BINARY, {"columns": ["code"]})
        assert not strategy.applicable(setup(), p, TaskType.BINARY, {"columns": ["a"]})  # numeric
        assert not strategy.applicable(setup(excluded_columns=["code"]), p, TaskType.BINARY, {"columns": ["code"]})
        dropped = setup(data_steps=[DataStepSpec("drop_columns", ("code",))])
        assert not strategy.applicable(dropped, p, TaskType.BINARY, {"columns": ["code"]})
    encoded = ENCODE.apply(setup(), {"columns": ["code"]})
    grouped = GROUP.apply(setup(), {"columns": ["code"]})
    # One encoding per column: neither can follow the other or itself.
    for config in (encoded, grouped):
        assert not ENCODE.applicable(config, p, TaskType.BINARY, {"columns": ["code"]})
        assert not GROUP.applicable(config, p, TaskType.BINARY, {"columns": ["code"]})


def test_missing_indicator_needs_a_numeric_column_with_empty_cells():
    p = mixed_profile()
    assert INDICATOR.applicable(setup(), p, TaskType.BINARY, {"columns": ["age"]})
    assert not INDICATOR.applicable(setup(), p, TaskType.BINARY, {"columns": ["a"]})  # no empty cells
    assert not INDICATOR.applicable(setup(), p, TaskType.BINARY, {"columns": ["code"]})  # text
    indicated = INDICATOR.apply(setup(), {"columns": ["age"]})
    assert not INDICATOR.applicable(indicated, p, TaskType.BINARY, {"columns": ["age"]})


@pytest.mark.parametrize("strategy", [ENCODE, GROUP, INDICATOR])
@pytest.mark.parametrize("params", [{}, {"columns": []}, {"columns": ["a", "age"]}, {"columns": "code"},
                                    {"columns": ["code"], "extra": 1}])
def test_column_strategies_reject_malformed_parameters(strategy, params):
    with pytest.raises(ValueError, match="one column name"):
        strategy.applicable(setup(), mixed_profile(), TaskType.BINARY, params)


def test_dropping_a_column_also_removes_its_data_steps(tmp_path):
    config = setup(data_steps=[DataStepSpec("frequency_encode", ("code",)), DataStepSpec("missing_indicator", ("age",)),
                               DataStepSpec("frequency_encode", ("code", "a"))])
    changed = DROP.apply(config, {"columns": ["code"]})
    assert changed.excluded_columns == ["code"]
    assert changed.data_steps == [DataStepSpec("missing_indicator", ("age",)), DataStepSpec("frequency_encode", ("a",))]
    # No step points at the dropped column, so the runner's check accepts the Setup.
    AutoMLRunner(settings=build_settings(tmp_path), enable_mlflow=False).prepare(mixed_frame(), changed)


def test_restrict_model_families_exact_change_and_applicability():
    p = mixed_profile()
    explicit = setup(estimator_list=["lgbm", "rf", "sgd", "xgboost"])
    changed = FAMILIES.apply(explicit, {"keep": ["xgboost", "lgbm"]})
    assert changed.estimator_list == ["lgbm", "xgboost"]  # the Setup's own order is kept
    assert changed.__dict__ == {**explicit.__dict__, "estimator_list": ["lgbm", "xgboost"]}
    assert FAMILIES.applicable(explicit, p, TaskType.BINARY, {"keep": ["lgbm", "xgboost"]})
    assert not FAMILIES.applicable(explicit, p, TaskType.BINARY, {"keep": ["lgbm", "rf", "sgd", "xgboost"]})  # no change
    assert not FAMILIES.applicable(explicit, p, TaskType.BINARY, {"keep": ["lgbm", "catboost"]})  # adds a family
    # With FLAML's own list, the families to keep become the list.
    assert FAMILIES.apply(setup(), {"keep": ["lgbm", "xgboost"]}).estimator_list == ["lgbm", "xgboost"]
    assert FAMILIES.applicable(setup(), p, TaskType.BINARY, {"keep": ["lgbm"]})
    for params in ({}, {"keep": []}, {"keep": ["lgbm", "lgbm"]}, {"keep": "lgbm"}):
        with pytest.raises(ValueError, match="distinct family names"):
            FAMILIES.apply(setup(), params)


def test_more_search_time_keeps_the_setup():
    original = setup(excluded_columns=["a"], class_weight="balanced")
    changed = MORE.apply(original, {})
    assert changed == original and changed is not original and changed.setup_key() == original.setup_key()
    assert MORE.applicable(original, mixed_profile(), TaskType.REGRESSION, {})
    with pytest.raises(ValueError, match="no parameters"):
        MORE.apply(original, {"factor": 3})


def test_smoother_search_metric_exact_change_and_applicability():
    p = mixed_profile()
    changed = SMOOTH.apply(setup(primary_metric="f1"), {})
    assert changed.__dict__ == {**setup(primary_metric="f1").__dict__, "search_metric": "log_loss"}
    assert changed.setup_key() != setup(primary_metric="f1").setup_key()
    assert SMOOTH.applicable(setup(primary_metric="f1"), p, TaskType.BINARY, {})
    assert SMOOTH.applicable(setup(primary_metric="accuracy"), p, TaskType.MULTICLASS, {})
    assert not SMOOTH.applicable(setup(primary_metric="r2"), p, TaskType.REGRESSION, {})
    assert not SMOOTH.applicable(changed, p, TaskType.BINARY, {})  # already log loss
    assert not SMOOTH.applicable(setup(primary_metric="log_loss"), p, TaskType.BINARY, {})


@pytest.mark.parametrize(
    "strategy_id, params",
    [("frequency_encode", {"columns": ["code"]}), ("group_rare_categories", {"columns": ["code"]}),
     ("missing_indicator", {"columns": ["age"]}), ("restrict_model_families", {"keep": ["lgbm"]}),
     ("more_search_time", {}), ("smoother_search_metric", {})],
)
def test_new_strategies_leave_the_original_unchanged(strategy_id, params):
    original = setup(estimator_list=["lgbm", "rf"], data_steps=[DataStepSpec("missing_indicator", ("a",))],
                     search_space_limits={"lgbm": {"num_leaves": {"upper": 16}}})
    before = copy.deepcopy(original)
    changed = get_strategy(strategy_id).apply(original, params)
    assert changed is not original and original == before
    changed.data_steps.append(DataStepSpec("drop_columns", ("code",)))
    changed.estimator_list.append("xgboost")
    changed.search_space_limits["rf"] = {}
    assert original == before
    assert get_strategy(strategy_id).describe(params)
