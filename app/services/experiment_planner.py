"""Planner: builds the Setups that a Session runs (Requirement 12.1 and 12.2)."""

from __future__ import annotations

import copy
from dataclasses import replace

from app.services.automl_runner import ExperimentConfig, ExperimentResult
from app.services.profiler import DatasetProfile
from app.services.strategies import get_strategy
from app.services.strategy_engine import Proposal


class Planner:
    @staticmethod
    def safe_exclusions(base: ExperimentConfig, train_profile: DatasetProfile) -> list[str]:
        """Identifier-like and constant columns of the Training_Part that the user did not exclude already."""
        return [c for c in train_profile.suggested_exclusions()
                if c != base.target_column and c not in base.excluded_columns]

    def first_setup(self, base: ExperimentConfig, train_profile: DatasetProfile) -> ExperimentConfig:
        """User settings + exclude identifier-like and constant columns found on the Training_Part
        (the Baseline's safe defaults)."""
        excluded = [*base.excluded_columns, *self.safe_exclusions(base, train_profile)]
        return replace(copy.deepcopy(base), excluded_columns=excluded)

    def candidate(self, accepted: ExperimentConfig, accepted_result: ExperimentResult, proposal: Proposal,
                  budget_seconds: int) -> ExperimentConfig:
        """Apply the Strategy to the Accepted_Setup, set the time budget, and start the search from the
        accepted result's best configuration of each learner family still in the candidate's list.

        A starting configuration outside the candidate's search space (for example after smaller trees)
        is left out by the engine, with a warning.
        """
        changed = get_strategy(proposal.spec.strategy_id).apply(accepted, proposal.spec.params)
        families = set(changed.estimator_list) if changed.estimator_list else None
        starting_points = {
            family: dict(config)
            for family, config in accepted_result.best_config_per_estimator.items()
            if config and (families is None or family in families)
        }
        return replace(changed, time_budget=int(budget_seconds), starting_points=starting_points or None)
