"""Tests for the ablation switches (spec task 22.1, Requirement 20), with stand-in engines and a fake clock.

The memory switch is shown in tests/test_selectors.py (a whole Session with and without the memory).
"""

from app.services.automl_runner import ExperimentConfig
from app.services.resource_manager import EqualShareManager, ResourceManager, SessionBudget, StopReason
from app.services.task_detector import TaskType
from tests.test_autopilot import FakeClock, run_session


def changes(record):
    return [(s["proposal"] or {}).get("key") for s in record["steps"][1:]]


# --------------------------------------------------------------------------- no Critic
def test_without_the_critic_the_changes_are_random_valid_ones_that_answer_nothing(tmp_path):
    shifts = {1: 0.0, 2: 0.0, 3: 0.0}
    plain = run_session(tmp_path / "plain", experiments=4, shifts=shifts)[1]
    result, record, engine, *_ = run_session(tmp_path / "off", experiments=4, shifts=shifts, use_critic=False)
    again = run_session(tmp_path / "again", experiments=4, shifts=shifts, use_critic=False)[1]
    assert record["settings"]["use_critic"] is False and result.status == "completed"
    assert all(step["diagnoses"] == [] for step in record["steps"])  # the Critic never ran
    for step in record["steps"][1:]:
        assert step["proposal"]["answers"] == [] and "no Critic" in step["proposal"]["reason"]
        assert step["selection"] == {"selector": "random (no Critic)", "rank": 0} and step["status"] == "completed"
    assert len(set(changes(record))) == len(changes(record)) == 3  # no change is tried twice
    assert changes(record) != changes(plain) and changes(record) == changes(again)  # random, but fixed by the seed
    assert plain["steps"][0]["diagnoses"]  # the same data gives Diagnoses when the Critic is on


# --------------------------------------------------------------------------- no Resource_Manager
def test_without_the_resource_manager_every_search_gets_an_equal_share(tmp_path):
    shifts = {1: -0.05, 2: -0.05, 3: -0.05, 4: -0.05, 5: -0.05}  # every candidate is worse: all discarded
    managed = run_session(tmp_path / "on", total=600, experiments=6, shifts=shifts)[1]
    equal = run_session(tmp_path / "off", total=600, experiments=6, shifts=shifts, use_resource_manager=False)[1]
    assert equal["settings"]["use_resource_manager"] is False
    assert managed["steps"][0]["cost"]["budget_seconds"] == 210  # 35% of 600 s
    # 600 / 6 = 100 s each; each step also takes 2 s of Assessment, so the sixth gets the 90 s that are left.
    assert [s["cost"]["budget_seconds"] for s in equal["steps"]] == [100, 100, 100, 100, 100, 90]
    assert managed["stop_reason"] == "patience" and len(managed["steps"]) == 4  # three discards in a row
    assert equal["stop_reason"] in ("time", "experiments") and len(equal["steps"]) == 6  # no patience rule


def test_the_equal_share_manager_on_its_own():
    clock = FakeClock()
    budget = SessionBudget(total_seconds=100, max_experiments=4)
    manager = EqualShareManager(budget, clock)
    assert manager.first_budget() == 25 and manager.reserve() == 0.0
    clock.advance(30)
    manager.record(budget_seconds=25, search_wall_seconds=25, assessment_seconds=5, other_seconds=0,
                   gain=None, verdict=None)
    assert manager.next_budget(None, None, wasted_budget=True) == 25  # a wasted budget does not shorten it
    clock.advance(25)
    manager.record(budget_seconds=25, search_wall_seconds=25, assessment_seconds=0, other_seconds=0,
                   gain=0.0, verdict="discard")
    for _ in range(3):  # discards in a row do not stop it
        assert manager.check_stop(proposals_left=2) is None
    clock.advance(40)  # 95 s used: exactly 5 s left, still enough
    assert manager.check_stop(proposals_left=2) is None
    clock.advance(1)  # 4 s left: under the 5 s minimum
    assert manager.check_stop(proposals_left=2) is StopReason.TIME
    # The full Resource_Manager would have kept back time and stopped on patience instead.
    assert ResourceManager(budget, FakeClock()).first_budget() == 35


# --------------------------------------------------------------------------- random order and the records
def test_random_order_changes_the_choices_and_every_switch_is_recorded(tmp_path):
    shifts = {1: 0.0, 2: 0.0}
    rules = run_session(tmp_path / "rules", experiments=3, shifts=shifts)[1]
    orders = {}
    for seed in (1, 2, 3):
        base = ExperimentConfig(target_column="target", task_type=TaskType.BINARY, primary_metric="f1",
                                dataset_name="synthetic", random_seed=seed)
        record = run_session(tmp_path / f"random{seed}", experiments=3, shifts=shifts, base=base, selector="random")[1]
        orders[seed] = changes(record)
        assert record["steps"][1]["selection"]["selector"] == "random"
        settings = record["settings"]
        assert (settings["selector"], settings["use_memory"], settings["use_critic"],
                settings["use_resource_manager"]) == ("random", False, True, True)
    assert any(order != changes(rules) for order in orders.values())
