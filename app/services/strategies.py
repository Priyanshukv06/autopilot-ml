"""Strategy_Catalogue: every change the autonomous loop may make to a Setup (Requirement 7).

A Strategy is a small, tested unit: it says when it applies and returns a changed copy of the
Setup (an ``ExperimentConfig``), never changing the one it was given. Strategies are named by a
``StrategySpec`` (an ID and JSON-safe parameters), whose ``key()`` lets a Session avoid repeats.

Thin version (task 7): drop_columns, class_weight_balanced and regularised_search_space. The
full version (task 14) adds the others listed in design.md.
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
        features = [c for c in profile.column_names if c != profile.target_column and c not in config.excluded_columns]
        # Every column must still be in use, and at least one feature must remain.
        return all(c in features for c in columns) and len(features) > len(columns)

    def apply(self, config, params) -> ExperimentConfig:
        columns = self.columns(params)
        added = [c for c in columns if c not in config.excluded_columns]
        return _copy(config, excluded_columns=[*config.excluded_columns, *added])

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


CATALOGUE: dict[str, Strategy] = {
    strategy.strategy_id: strategy for strategy in (DropColumns(), ClassWeightBalanced(), RegularisedSearchSpace())
}


def get_strategy(strategy_id: str) -> Strategy:
    try:
        return CATALOGUE[strategy_id]
    except KeyError:
        raise ValueError(f"Unknown strategy '{strategy_id}'. Known: {', '.join(CATALOGUE)}.") from None
