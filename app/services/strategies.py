"""Strategy_Catalogue: every change the autonomous loop may make to a Setup (Requirement 7).

A Strategy is a small, tested unit: it says when it applies and returns a changed copy of the
Setup (an ``ExperimentConfig``), never changing the one it was given. Strategies are named by a
``StrategySpec`` (an ID and JSON-safe parameters), whose ``key()`` lets a Session avoid repeats.

The catalogue has the nine Strategies of design.md: drop_columns, class_weight_balanced and
regularised_search_space (thin version, task 7), and frequency_encode, group_rare_categories,
missing_indicator, restrict_model_families, more_search_time and smoother_search_metric (task 14).
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from app.services.automl_runner import ExperimentConfig, to_jsonable
from app.services.profiler import DatasetProfile
from app.services.task_detector import TaskType
from app.services.transforms import DataStepSpec, as_specs

Kind = Literal["data", "model", "search", "budget"]
Complexity = Literal["simpler", "same", "more_complex"]
CostClass = Literal["cheap", "normal", "expensive"]

# Starting limits of regularised_search_space (design choices, not measured values). Bounds are
# inclusive; FLAMLEngine turns them into FLAML's custom_hp.
REGULARISED_LIMITS: dict[str, dict[str, dict[str, float]]] = {
    "lgbm": {"num_leaves": {"upper": 32}, "min_child_samples": {"lower": 10}},
    "xgboost": {"max_leaves": {"upper": 32}, "min_child_weight": {"lower": 1}},
    "xgb_limitdepth": {"max_depth": {"upper": 4}},
    "rf": {"max_leaves": {"upper": 64}},
    "extra_tree": {"max_leaves": {"upper": 64}},
}
# The only settings Strategies may limit.
SEARCH_SPACE_LIMITS_ALLOWED: dict[str, tuple[str, ...]] = {
    estimator: tuple(settings) for estimator, settings in REGULARISED_LIMITS.items()
}


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        """Canonical JSON of the ID and parameters: equal for the same change."""
        return json.dumps({"strategy": self.strategy_id, "params": to_jsonable(self.params)}, sort_keys=True,
                          separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return {"strategy_id": self.strategy_id, "params": to_jsonable(self.params)}


class Strategy(Protocol):
    strategy_id: str
    title: str
    kind: Kind
    complexity: Complexity
    cost_class: CostClass

    def applicable(self, config: ExperimentConfig, profile: DatasetProfile, task_type: TaskType,
                   params: dict[str, Any]) -> bool: ...

    def apply(self, config: ExperimentConfig, params: dict[str, Any]) -> ExperimentConfig:
        """A changed copy of ``config``; ``config`` itself is not changed."""
        ...

    def describe(self, params: dict[str, Any]) -> str:
        """The change in plain words, for records and the dashboard."""
        ...


def _copy(config: ExperimentConfig, **changes: Any) -> ExperimentConfig:
    """A deep copy with some fields changed, so that no list or dict is shared with ``config``."""
    return replace(copy.deepcopy(config), **changes)


def _no_params(strategy_id: str, params: dict[str, Any]) -> None:
    if params:
        raise ValueError(f"{strategy_id} takes no parameters, got {sorted(params)}.")


def used_columns(config: ExperimentConfig, profile: DatasetProfile) -> list[str]:
    """Feature columns of a Setup in data order: not the target, not excluded, not dropped by a Data_Step."""
    dropped = {c for step in as_specs(config.data_steps) if step.kind == "drop_columns" for c in step.columns}
    return [c for c in profile.column_names
            if c != profile.target_column and c not in config.excluded_columns and c not in dropped]


def treated_columns(config: ExperimentConfig, kinds: tuple[str, ...]) -> set[str]:
    """Columns that a Data_Step of one of these kinds already changes."""
    return {c for step in as_specs(config.data_steps) if step.kind in kinds for c in step.columns}


def _one_column(strategy_id: str, params: dict[str, Any]) -> str:
    columns = params.get("columns")
    if set(params) != {"columns"} or not isinstance(columns, list) or len(columns) != 1 \
            or not isinstance(columns[0], str):
        raise ValueError(f"{strategy_id} needs {{'columns': [one column name]}}, got {params}.")
    return columns[0]


# --------------------------------------------------------------------------- strategies
class DropColumns:
    strategy_id = "drop_columns"
    title = "Drop columns"
    kind: Kind = "data"
    complexity: Complexity = "simpler"
    cost_class: CostClass = "cheap"

    @staticmethod
    def columns(params: dict[str, Any]) -> list[str]:
        columns = params.get("columns")
        if set(params) != {"columns"} or not isinstance(columns, list) or not columns \
                or not all(isinstance(c, str) for c in columns) or len(set(columns)) != len(columns):
            raise ValueError(f"drop_columns needs {{'columns': [distinct column names]}}, got {params}.")
        return columns

    def applicable(self, config, profile, task_type, params) -> bool:
        columns = self.columns(params)
        features = used_columns(config, profile)
        # Every column must still be in use, and at least one feature must remain.
        return all(c in features for c in columns) and len(features) > len(columns)

    def apply(self, config, params) -> ExperimentConfig:
        """Exclude the columns, and remove them from any Data_Step (a step left without columns goes)."""
        columns = self.columns(params)
        added = [c for c in columns if c not in config.excluded_columns]
        steps = []
        for step in as_specs(config.data_steps):
            kept = tuple(c for c in step.columns if c not in columns)
            if kept:
                steps.append(step if kept == step.columns else DataStepSpec(step.kind, kept, dict(step.params)))
        return _copy(config, excluded_columns=[*config.excluded_columns, *added], data_steps=steps)

    def describe(self, params) -> str:
        names = ", ".join(f"'{c}'" for c in self.columns(params))
        return f"Stop using the column{'s' if len(self.columns(params)) > 1 else ''} {names}"


class ClassWeightBalanced:
    strategy_id = "class_weight_balanced"
    title = "Balanced class weights"
    kind: Kind = "model"
    complexity: Complexity = "same"
    cost_class: CostClass = "cheap"

    def applicable(self, config, profile, task_type, params) -> bool:
        _no_params(self.strategy_id, params)
        return task_type.is_classification and config.class_weight != "balanced"

    def apply(self, config, params) -> ExperimentConfig:
        _no_params(self.strategy_id, params)
        return _copy(config, class_weight="balanced")

    def describe(self, params) -> str:
        return "Weight the training rows so that every class counts equally"


def merge_limits(current: dict[str, dict[str, dict[str, float]]],
                 new: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    """Both sets of limits together: for each setting the lower of the upper bounds and the higher of
    the lower bounds, so that merging never widens a range."""
    merged = copy.deepcopy(current)
    for estimator, settings in new.items():
        for name, bounds in settings.items():
            target = merged.setdefault(estimator, {}).setdefault(name, {})
            if bounds.get("upper") is not None:
                target["upper"] = min(target.get("upper", math.inf), bounds["upper"])
            if bounds.get("lower") is not None:
                target["lower"] = max(target.get("lower", -math.inf), bounds["lower"])
    return merged


class RegularisedSearchSpace:
    strategy_id = "regularised_search_space"
    title = "Smaller trees"
    kind: Kind = "search"
    complexity: Complexity = "simpler"
    cost_class: CostClass = "cheap"

    def applicable(self, config, profile, task_type, params) -> bool:
        _no_params(self.strategy_id, params)
        return merge_limits(config.search_space_limits, REGULARISED_LIMITS) != config.search_space_limits

    def apply(self, config, params) -> ExperimentConfig:
        _no_params(self.strategy_id, params)
        return _copy(config, search_space_limits=merge_limits(config.search_space_limits, REGULARISED_LIMITS))

    def describe(self, params) -> str:
        return ("Search only smaller trees: at most 32 leaves for LightGBM and XGBoost, depth at most 4 for "
                "depth-limited XGBoost, at most 64 leaves for the forests, and larger minimum leaf sizes")


_ENCODINGS = ("frequency_encode", "group_rare_categories")


class _ColumnStep:
    """A Strategy that adds one Data_Step for one column of the Setup."""

    strategy_id: str
    title: str
    kind: Kind = "data"
    complexity: Complexity
    cost_class: CostClass = "cheap"
    step_kind: str

    def _column_fits(self, column: str, config: ExperimentConfig, profile: DatasetProfile) -> bool:
        raise NotImplementedError

    def applicable(self, config, profile, task_type, params) -> bool:
        column = _one_column(self.strategy_id, params)
        return column in used_columns(config, profile) and self._column_fits(column, config, profile)

    def apply(self, config, params) -> ExperimentConfig:
        column = _one_column(self.strategy_id, params)
        steps = [*as_specs(config.data_steps), DataStepSpec(self.step_kind, (column,))]
        return _copy(config, data_steps=steps)


class FrequencyEncode(_ColumnStep):
    strategy_id = "frequency_encode"
    title = "Frequency encoding"
    complexity: Complexity = "same"
    step_kind = "frequency_encode"

    def _column_fits(self, column, config, profile) -> bool:
        # A text column that no other encoding already changes.
        return column in profile.categorical_features and column not in treated_columns(config, _ENCODINGS)

    def describe(self, params) -> str:
        return f"Replace '{_one_column(self.strategy_id, params)}' by how often each value occurs in the training rows"


class GroupRareCategories(_ColumnStep):
    strategy_id = "group_rare_categories"
    title = "Group rare values"
    complexity: Complexity = "simpler"
    step_kind = "group_rare_categories"

    def _column_fits(self, column, config, profile) -> bool:
        return column in profile.categorical_features and column not in treated_columns(config, _ENCODINGS)

    def describe(self, params) -> str:
        return (f"Group the values of '{_one_column(self.strategy_id, params)}' seen in fewer than 1% of the training "
                "rows into one value")


class MissingIndicator(_ColumnStep):
    strategy_id = "missing_indicator"
    title = "Missing-value indicator"
    complexity: Complexity = "more_complex"
    step_kind = "missing_indicator"

    def _column_fits(self, column, config, profile) -> bool:
        # A numeric column with empty cells and no indicator yet.
        return (column in profile.numerical_features and profile.missing_ratio.get(column, 0.0) > 0
                and column not in treated_columns(config, ("missing_indicator",)))

    def describe(self, params) -> str:
        return f"Add a 0/1 column that marks the empty cells of '{_one_column(self.strategy_id, params)}'"


class RestrictModelFamilies:
    strategy_id = "restrict_model_families"
    title = "Fewer learner families"
    kind: Kind = "model"
    complexity: Complexity = "simpler"
    cost_class: CostClass = "cheap"

    @staticmethod
    def keep(params: dict[str, Any]) -> list[str]:
        keep = params.get("keep")
        if set(params) != {"keep"} or not isinstance(keep, list) or not keep \
                or not all(isinstance(f, str) for f in keep) or len(set(keep)) != len(keep):
            raise ValueError(f"restrict_model_families needs {{'keep': [distinct family names]}}, got {params}.")
        return keep

    def applicable(self, config, profile, task_type, params) -> bool:
        keep = self.keep(params)
        if config.estimator_list is None:  # FLAML's own list; ``keep`` comes from the families it tried
            return True
        return set(keep) < set(config.estimator_list)  # removes at least one, adds none

    def apply(self, config, params) -> ExperimentConfig:
        keep = self.keep(params)
        current = config.estimator_list or keep
        return _copy(config, estimator_list=[f for f in current if f in keep])

    def describe(self, params) -> str:
        return f"Search only these learner families: {', '.join(self.keep(params))}"


class MoreSearchTime:
    """The Setup stays the same; the Resource_Manager gives the search twice the Accepted_Setup's time, and
    the Planner starts it from the current best configurations."""

    strategy_id = "more_search_time"
    title = "More search time"
    kind: Kind = "budget"
    complexity: Complexity = "same"
    cost_class: CostClass = "expensive"

    def applicable(self, config, profile, task_type, params) -> bool:
        _no_params(self.strategy_id, params)
        return True

    def apply(self, config, params) -> ExperimentConfig:
        _no_params(self.strategy_id, params)
        return _copy(config)

    def describe(self, params) -> str:
        return "Search for longer, starting from the current best configurations"


class SmootherSearchMetric:
    """FLAML optimises log loss, which changes smoothly with every prediction; the Judge still compares the
    primary metric."""

    strategy_id = "smoother_search_metric"
    title = "Smoother search metric"
    kind: Kind = "search"
    complexity: Complexity = "same"
    cost_class: CostClass = "normal"

    def applicable(self, config, profile, task_type, params) -> bool:
        _no_params(self.strategy_id, params)
        current = config.search_metric or config.primary_metric
        return task_type.is_classification and current != "log_loss"

    def apply(self, config, params) -> ExperimentConfig:
        _no_params(self.strategy_id, params)
        return _copy(config, search_metric="log_loss")

    def describe(self, params) -> str:
        return "Let the search optimise log loss, which rewards better probabilities, while the Judge keeps the primary metric"


CATALOGUE: dict[str, Strategy] = {
    strategy.strategy_id: strategy
    for strategy in (
        DropColumns(), ClassWeightBalanced(), RegularisedSearchSpace(), FrequencyEncode(), GroupRareCategories(),
        MissingIndicator(), RestrictModelFamilies(), MoreSearchTime(), SmootherSearchMetric(),
    )
}


def get_strategy(strategy_id: str) -> Strategy:
    try:
        return CATALOGUE[strategy_id]
    except KeyError:
        raise ValueError(f"Unknown strategy '{strategy_id}'. Known: {', '.join(CATALOGUE)}.") from None
