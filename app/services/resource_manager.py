"""Resource_Manager: spends a Session's time budget and decides when to stop (Requirement 11).

Time is counted by measured wall-clock time (an injected clock, so tests need no real waiting),
never by the budgets handed out. Each step is planned to end by a deadline: the first step by the
end of the Session's budget, a later step by "now + its search budget + the reserve". The reserve
is 1.2 x the Assessment and other time of the previous step, kept back for the next Assessment and
the final steps. A step's overrun is the time it ran past its deadline. Because every deadline
lies inside the budget, a Session ends at most one step's overrun late.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable

from app.services.search_summary import SearchSummary
from app.services.strategy_engine import Proposal

RESERVE_FACTOR = 1.2
MORE_SEARCH_TIME = "more_search_time"
KEPT = ("keep", "keep_simpler")


@dataclass(frozen=True)
class SessionBudget:
    total_seconds: float
    max_experiments: int
    n_jobs: int = -1
    min_experiment_seconds: int = 5  # the runner's smallest allowed time budget
    first_share: float = 0.35  # allowed range 0.30 to 0.40
    patience: int = 3  # discards in a row before stopping

    def __post_init__(self) -> None:
        if not self.total_seconds >= 10:
            raise ValueError("A Session needs a budget of at least 10 seconds.")
        if self.max_experiments < 2:
            raise ValueError("A Session needs at least 2 Experiments.")
        if not 0.30 <= self.first_share <= 0.40:
            raise ValueError("The first Experiment's share must be between 0.30 and 0.40.")
        if self.patience < 1 or self.min_experiment_seconds < 1 or self.n_jobs == 0:
            raise ValueError("patience and min_experiment_seconds must be at least 1, and n_jobs must not be 0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepCost:
    step: int
    budget_seconds: int
    search_wall_seconds: float
    assessment_seconds: float
    other_seconds: float  # retrain, records, critic and judge time
    overrun_seconds: float  # wall time of the step beyond its deadline
    utility: float | None  # validation gain of the step / its seconds; None for the first step
    verdict: str | None = None  # None for the first step

    @property
    def wall_seconds(self) -> float:
        return self.search_wall_seconds + self.assessment_seconds + self.other_seconds

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "wall_seconds": self.wall_seconds}


class StopReason(str, Enum):
    TIME = "time"
    EXPERIMENTS = "experiments"
    NO_PROPOSALS = "no_proposals"
    PATIENCE = "patience"
    USER = "user"
    ERROR = "error"


class ResourceManager:
    def __init__(self, budget: SessionBudget, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self.clock = clock
        self._start = clock()
        self._ledger: list[StepCost] = []
        self._deadline: float | None = None  # of the step whose budget was handed out last
        self._accepted_budget: int | None = None
        self._discards_in_a_row = 0
        self._user_stop = False
        self._error: str | None = None

    # ------------------------------------------------------------------ time
    def elapsed(self) -> float:
        return self.clock() - self._start

    def reserve(self) -> float:
        """Time kept back for the next Assessment and the final steps, from the previous step."""
        if not self._ledger:
            return 0.0
        last = self._ledger[-1]
        return RESERVE_FACTOR * (last.assessment_seconds + last.other_seconds)

    def time_left(self) -> float:
        """R = total - elapsed - reserve."""
        return self.budget.total_seconds - self.elapsed() - self.reserve()

    @property
    def n_jobs(self) -> int:
        """The CPU limit for every search and refit of the Session."""
        return self.budget.n_jobs

    # --------------------------------------------------------------- budgets
    def first_budget(self) -> int:
        """A fixed share of the total time, at least the minimum. The step must end by the end of the budget."""
        if self._ledger:
            raise RuntimeError("The first budget is given only once, before any step.")
        b = self.budget
        seconds = max(b.min_experiment_seconds, math.floor(b.first_share * b.total_seconds))
        self._deadline = self._start + b.total_seconds
        return seconds

    def next_budget(self, proposal: Proposal | None, accepted_search: SearchSummary | None,
                    wasted_budget: bool) -> int | StopReason:
        """The search budget of the next Experiment, or the reason to stop instead."""
        b = self.budget
        left, experiments_left = self.time_left(), b.max_experiments - len(self._ledger)
        if left < b.min_experiment_seconds:
            return StopReason.TIME
        if experiments_left <= 0:
            return StopReason.EXPERIMENTS
        seconds = max(b.min_experiment_seconds, math.floor(left / experiments_left))
        if proposal is not None and proposal.spec.strategy_id == MORE_SEARCH_TIME and self._accepted_budget:
            seconds = max(seconds, min(2 * self._accepted_budget, math.floor(left)))
        if wasted_budget and accepted_search is not None:
            cap = max(b.min_experiment_seconds, math.ceil(1.5 * accepted_search.time_to_best))
            seconds = min(seconds, cap)
        self._deadline = self.clock() + seconds + self.reserve()
        return seconds

    # ----------------------------------------------------------------- steps
    def record(
        self,
        *,
        budget_seconds: int,
        search_wall_seconds: float,
        assessment_seconds: float,
        other_seconds: float,
        gain: float | None,
        verdict: str | None,
    ) -> StepCost:
        """Record a finished step, measured by the caller, and return its cost.

        ``gain``: the change in the accepted validation score (the Judge's mean difference for a kept
        candidate, 0 for a discarded one, None for the first step). ``verdict``: None for the first step,
        "discard" also for a candidate that failed.
        """
        if self._deadline is None:
            raise RuntimeError("Ask for a budget before recording a step.")
        overrun = max(0.0, self.clock() - self._deadline)
        cost = StepCost(
            step=len(self._ledger) + 1,
            budget_seconds=int(budget_seconds),
            search_wall_seconds=float(search_wall_seconds),
            assessment_seconds=float(assessment_seconds),
            other_seconds=float(other_seconds),
            overrun_seconds=overrun,
            utility=None,
            verdict=verdict,
        )
        if gain is not None and cost.wall_seconds > 0:
            cost.utility = float(gain) / cost.wall_seconds
        self._ledger.append(cost)
        self._deadline = None
        if verdict is None or verdict in KEPT:
            self._accepted_budget = cost.budget_seconds
            self._discards_in_a_row = 0
        else:
            self._discards_in_a_row += 1
        return cost

    def request_stop(self) -> None:
        """Stop after the current step (the user asked)."""
        self._user_stop = True

    def report_error(self, message: str) -> None:
        self._error = message

    def check_stop(self, proposals_left: int) -> StopReason | None:
        """The reason to stop now, checked in the order: user, error, time, experiments, no Proposals, patience."""
        b = self.budget
        if self._user_stop:
            return StopReason.USER
        if self._error is not None:
            return StopReason.ERROR
        if self.time_left() < b.min_experiment_seconds:
            return StopReason.TIME
        if len(self._ledger) >= b.max_experiments:
            return StopReason.EXPERIMENTS
        if proposals_left <= 0:
            return StopReason.NO_PROPOSALS
        if self._discards_in_a_row >= b.patience:
            return StopReason.PATIENCE
        return None

    @property
    def error(self) -> str | None:
        return self._error

    def ledger(self) -> list[StepCost]:
        return list(self._ledger)


class EqualShareManager(ResourceManager):
    """The Session without a Resource_Manager (ablation, Requirement 20.1).

    Every search gets the same share, total time / maximum Experiments (at least the minimum, at most the
    time left); nothing is kept back for Assessments, a wasted budget does not shorten the next search,
    and more search time gets no larger budget. The Session stops only at the time and Experiment
    limits, besides Stop, an error, or no Proposal left (there is no patience rule).
    """

    def share(self) -> int:
        b = self.budget
        return max(b.min_experiment_seconds, math.floor(b.total_seconds / b.max_experiments))

    def reserve(self) -> float:
        return 0.0

    def first_budget(self) -> int:
        if self._ledger:
            raise RuntimeError("The first budget is given only once, before any step.")
        self._deadline = self._start + self.budget.total_seconds
        return self.share()

    def next_budget(self, proposal: Proposal | None, accepted_search: SearchSummary | None,
                    wasted_budget: bool) -> int | StopReason:
        b = self.budget
        left = self.time_left()
        if left < b.min_experiment_seconds:
            return StopReason.TIME
        if b.max_experiments - len(self._ledger) <= 0:
            return StopReason.EXPERIMENTS
        seconds = max(b.min_experiment_seconds, min(self.share(), math.floor(left)))
        self._deadline = self.clock() + seconds
        return seconds

    def check_stop(self, proposals_left: int) -> StopReason | None:
        b = self.budget
        if self._user_stop:
            return StopReason.USER
        if self._error is not None:
            return StopReason.ERROR
        if self.time_left() < b.min_experiment_seconds:
            return StopReason.TIME
        if len(self._ledger) >= b.max_experiments:
            return StopReason.EXPERIMENTS
        if proposals_left <= 0:
            return StopReason.NO_PROPOSALS
        return None
