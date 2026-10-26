"""Tests for the Resource_Manager (task 9.1 of the spec), with a fake clock: no real waiting."""

import numpy as np
import pytest

from app.services.critic import DiagnosisKind, Severity
from app.services.resource_manager import ResourceManager, SessionBudget, StopReason
from app.services.search_summary import SearchSummary
from app.services.strategies import StrategySpec
from app.services.strategy_engine import Proposal


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def manager(total=300, experiments=8, **kwargs):
    clock = FakeClock()
    return ResourceManager(SessionBudget(total_seconds=total, max_experiments=experiments, **kwargs), clock), clock


def run_step(rm, clock, budget, search, assessment, other, gain=None, verdict=None):
    """Spend the step's time on the fake clock, then record it."""
    clock.advance(search + assessment + other)
    return rm.record(budget_seconds=budget, search_wall_seconds=search, assessment_seconds=assessment,
                     other_seconds=other, gain=gain, verdict=verdict)


def proposal(strategy_id="drop_columns"):
    return Proposal(StrategySpec(strategy_id, {}), (DiagnosisKind.OVERFITTING,), Severity.MODERATE, "cheap", "same",
                    "reason", (5, 0))


def search(time_to_best):
    return SearchSummary(search_seconds=100.0, time_to_best=time_to_best, share_after_best=0.9,
                         gain_last_quarter=None, n_trials=10, best_per_family={}, seconds_per_family={})


# --------------------------------------------------------------------------- budgets
@pytest.mark.parametrize("total, share, expected", [(300, 0.35, 105), (10, 0.35, 5), (300, 0.30, 90),
                                                    (300, 0.40, 120), (17, 0.35, 5)])
def test_first_budget_is_the_share_and_never_below_5_seconds(total, share, expected):
    rm, _ = manager(total, first_share=share)
    assert rm.first_budget() == expected


def test_later_budgets_come_from_measured_time_and_the_reserve():
    rm, clock = manager(300, 8)
    assert rm.reserve() == 0.0
    run_step(rm, clock, rm.first_budget(), search=110, assessment=8, other=2)
    assert rm.reserve() == pytest.approx(12.0)  # 1.2 x (8 + 2)
    assert rm.time_left() == pytest.approx(300 - 120 - 12)
    assert rm.next_budget(proposal(), None, wasted_budget=False) == 24  # floor(168 / 7)


def test_budgets_follow_the_time_really_used_not_the_time_handed_out():
    rm, clock = manager(300, 8)
    run_step(rm, clock, rm.first_budget(), search=190, assessment=8, other=2)  # the search ran far past 105 s
    assert rm.next_budget(proposal(), None, wasted_budget=False) == 12  # floor((300 - 200 - 12) / 7)


def test_wasted_budget_caps_the_next_search():
    rm, clock = manager(300, 8)
    run_step(rm, clock, rm.first_budget(), search=110, assessment=8, other=2)
    assert rm.next_budget(proposal(), search(4.0), wasted_budget=True) == 6  # ceil(1.5 x 4)
    assert rm.next_budget(proposal(), search(2.0), wasted_budget=True) == 5  # never below 5
    assert rm.next_budget(proposal(), search(40.0), wasted_budget=True) == 24  # the cap is above the normal budget
    assert rm.next_budget(proposal(), search(4.0), wasted_budget=False) == 24


def test_more_search_time_asks_for_twice_the_accepted_budget_capped_by_the_time_left():
    rm, clock = manager(300, 8)
    run_step(rm, clock, rm.first_budget(), search=110, assessment=8, other=2)  # accepted budget 105
    assert rm.next_budget(proposal("more_search_time"), None, wasted_budget=False) == 168  # min(210, 168)
    rm2, clock2 = manager(1000, 8)
    run_step(rm2, clock2, rm2.first_budget(), search=360, assessment=8, other=2)  # accepted budget 350
    assert rm2.next_budget(proposal("more_search_time"), None, wasted_budget=False) == 618  # min(700, 618)


def test_the_accepted_budget_follows_kept_steps_only():
    rm, clock = manager(1000, 8)
    run_step(rm, clock, rm.first_budget(), search=350, assessment=5, other=1)  # accepted 350
    run_step(rm, clock, rm.next_budget(proposal(), None, False), search=90, assessment=5, other=1,
             gain=0.0, verdict="discard")
    budget = rm.next_budget(proposal("more_search_time"), None, False)
    assert budget == min(700, int(rm.time_left()))  # still twice the first step's 350 s


# --------------------------------------------------------------------------- stops
def test_next_budget_stops_on_time_and_on_experiments():
    rm, clock = manager(100, 8)
    run_step(rm, clock, rm.first_budget(), search=88, assessment=4, other=1)
    assert rm.time_left() < 5 and rm.next_budget(proposal(), None, False) is StopReason.TIME
    rm, clock = manager(300, 2)
    run_step(rm, clock, rm.first_budget(), search=106, assessment=4, other=1)
    run_step(rm, clock, rm.next_budget(proposal(), None, False), search=20, assessment=4, other=1, gain=0.0,
             verdict="discard")
    assert rm.next_budget(proposal(), None, False) is StopReason.EXPERIMENTS


def test_patience_counts_discards_in_a_row():
    rm, clock = manager(1000, 10, patience=3)
    run_step(rm, clock, rm.first_budget(), search=10, assessment=1, other=1)
    for verdict in ("discard", "discard", "keep", "discard", "discard"):
        run_step(rm, clock, rm.next_budget(proposal(), None, False), 10, 1, 1, gain=0.0, verdict=verdict)
        assert rm.check_stop(proposals_left=5) is None
    run_step(rm, clock, rm.next_budget(proposal(), None, False), 10, 1, 1, gain=0.0, verdict="discard")
    assert rm.check_stop(proposals_left=5) is StopReason.PATIENCE


def test_stop_reasons_are_checked_in_order():
    """With every stop condition true at once, removing them one by one shows the order:
    user, error, time, experiments, no Proposals, patience."""

    def state(user, error, time_out, experiments_used):
        rm, clock = manager(100, 4 if experiments_used else 10, patience=1)
        run_step(rm, clock, rm.first_budget(), search=30, assessment=1, other=1)
        run_step(rm, clock, rm.next_budget(proposal(), None, False), 5, 1, 1, gain=0.0, verdict="discard")  # patience
        if experiments_used:
            for _ in range(2):
                run_step(rm, clock, rm.next_budget(proposal(), None, False), 5, 1, 1, gain=0.0, verdict="discard")
        if time_out:
            clock.advance(100)
        if user:
            rm.request_stop()
        if error:
            rm.report_error("refit failed")
        return rm

    assert state(True, True, True, True).check_stop(0) is StopReason.USER
    assert state(False, True, True, True).check_stop(0) is StopReason.ERROR
    assert state(False, False, True, True).check_stop(0) is StopReason.TIME
    assert state(False, False, False, True).check_stop(0) is StopReason.EXPERIMENTS
    assert state(False, False, False, False).check_stop(0) is StopReason.NO_PROPOSALS
    assert state(False, False, False, False).check_stop(1) is StopReason.PATIENCE
    assert state(False, True, False, False).error == "refit failed"


# --------------------------------------------------------------------------- costs and overruns
def test_ledger_records_every_part_of_each_step():
    rm, clock = manager(300, 8)
    first = run_step(rm, clock, rm.first_budget(), search=110, assessment=8, other=2)
    second = run_step(rm, clock, rm.next_budget(proposal(), None, False), search=26, assessment=8, other=2,
                      gain=0.018, verdict="keep")
    third = run_step(rm, clock, rm.next_budget(proposal(), None, False), search=20, assessment=8, other=2,
                     gain=0.0, verdict="discard")
    assert [c.step for c in rm.ledger()] == [1, 2, 3]
    assert (first.budget_seconds, first.utility, first.verdict, first.overrun_seconds) == (105, None, None, 0.0)
    assert second.utility == pytest.approx(0.018 / 36) and third.utility == 0.0
    record = second.to_dict()
    assert record["wall_seconds"] == 36 and record["assessment_seconds"] == 8 and record["verdict"] == "keep"


def test_overruns_are_measured_against_each_steps_deadline():
    rm, clock = manager(300, 8)
    first = run_step(rm, clock, rm.first_budget(), search=110, assessment=8, other=2)  # ends at 120, deadline 300
    assert first.overrun_seconds == 0.0
    budget = rm.next_budget(proposal(), None, False)  # 24 s, reserve 12 s: deadline at 120 + 36
    late = run_step(rm, clock, budget, search=24 + 7, assessment=8, other=2)  # 3 s past the reserve as well
    assert late.overrun_seconds == pytest.approx(5.0)  # ran for 41 s against 36 s planned
    rm2, clock2 = manager(100, 8)
    slow_first = run_step(rm2, clock2, rm2.first_budget(), search=95, assessment=12, other=1)  # ends at 108
    assert slow_first.overrun_seconds == pytest.approx(8.0)


def test_recording_needs_a_budget_first_and_the_first_budget_is_given_once():
    rm, clock = manager()
    with pytest.raises(RuntimeError):
        rm.record(budget_seconds=5, search_wall_seconds=5, assessment_seconds=1, other_seconds=1, gain=None,
                  verdict=None)
    run_step(rm, clock, rm.first_budget(), 105, 1, 1)
    with pytest.raises(RuntimeError):
        rm.first_budget()


@pytest.mark.parametrize("seed", range(20))
def test_a_session_never_ends_later_than_the_budget_plus_one_steps_overrun(seed):
    """Simulated Sessions with random search overruns and Assessment times, run until the manager stops them."""
    rng = np.random.default_rng(seed)
    total = float(rng.choice([60, 120, 300, 600]))
    rm, clock = manager(total, int(rng.integers(2, 12)), patience=10)
    budget = rm.first_budget()
    verdict = None
    while True:
        search_wall = budget + rng.exponential(2.0)  # FLAML retrains after the search
        assessment, other = rng.uniform(0.5, 6.0), rng.uniform(0.1, 1.0)
        run_step(rm, clock, budget, search_wall, assessment, other, gain=None if verdict is None else 0.0,
                 verdict=verdict)
        verdict = "discard"
        if rm.check_stop(proposals_left=5) is not None:
            break
        budget = rm.next_budget(proposal(), None, wasted_budget=False)
        if isinstance(budget, StopReason):
            break
    last = rm.ledger()[-1]
    assert rm.elapsed() <= total + last.overrun_seconds + 1e-9
    assert all(c.budget_seconds >= 5 for c in rm.ledger())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"total_seconds": 9, "max_experiments": 5}, "10 seconds"),
        ({"total_seconds": 60, "max_experiments": 1}, "2 Experiments"),
        ({"total_seconds": 60, "max_experiments": 5, "first_share": 0.5}, "0.30 and 0.40"),
        ({"total_seconds": 60, "max_experiments": 5, "patience": 0}, "patience"),
        ({"total_seconds": 60, "max_experiments": 5, "n_jobs": 0}, "n_jobs"),
    ],
)
def test_invalid_budgets_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SessionBudget(**kwargs)
