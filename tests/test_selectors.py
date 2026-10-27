"""Tests for the selectors of the Strategy_Engine (spec task 21.1): rules, random, thompson and ucb1."""

import math

import numpy as np
import pytest

from app.services.automl_runner import ExperimentConfig
from app.services.critic import Diagnosis, DiagnosisKind, Severity
from app.services.experiment_memory import ExperimentMemory
from app.services.profiler import DatasetProfiler
from app.services.strategies import StrategySpec
from app.services.strategy_engine import (
    SELECTORS, Proposal, RandomSelector, RulesSelector, SelectionContext, ThompsonSelector, UCB1Selector, arm_of,
    propose,
)
from app.services.task_detector import TaskType
from research.synthetic import make_synthetic
from tests.test_autopilot import run_session
from tests.test_experiment_memory import record


def proposal(strategy, kind, severity="high", cost="cheap", row=0, columns=None):
    params = {"columns": columns} if columns else {}
    return Proposal(spec=StrategySpec(strategy, params), answers=(DiagnosisKind(kind),), severity=Severity(severity),
                    cost_class=cost, complexity="simpler", reason=f"{strategy}. Answers: {kind}.",
                    rule_position=(row, 0), answered=((kind, severity),))


DROP = proposal("drop_columns", "high_cardinality", row=2, columns=["code"])  # rule rank 0
WEIGHTS = proposal("class_weight_balanced", "class_imbalance", row=8)  # rule rank 1
TREES = proposal("regularised_search_space", "overfitting", severity="moderate", row=5)  # rule rank 2
PROPOSALS = [TREES, WEIGHTS, DROP]  # any order: every selector starts from the rule order


def memory_with(tmp_path, entries):
    """entries: (fingerprint, strategy, kind, verdict, diff, cost, count)."""
    memory = ExperimentMemory(tmp_path / "memory.jsonl")
    n = 0
    for fingerprint, strategy, kind, verdict, diff, cost, count in entries:
        for _ in range(count):
            memory.add(record(n, fingerprint=fingerprint, strategy=strategy, verdict=verdict, diff=diff, cost=cost,
                              kinds=(kind,)))
            n += 1
    return memory


def context(memory, seed=0, fingerprint="NEW"):
    return SelectionContext(memory=memory, meta_features={"x": 0.0}, task_type=TaskType.BINARY,
                            fingerprint=fingerprint, rng=np.random.default_rng(seed))


def keys(proposals):
    return [p.spec.strategy_id for p in proposals]


# --------------------------------------------------------------------------- rules and random
def test_the_rules_order_and_the_arm_of_a_proposal():
    assert keys(RulesSelector().rank(PROPOSALS)) == ["drop_columns", "class_weight_balanced", "regularised_search_space"]
    both = Proposal(StrategySpec("drop_columns", {"columns": ["code"]}),
                    (DiagnosisKind.OVERFITTING, DiagnosisKind.HIGH_CARDINALITY), Severity.HIGH, "cheap", "simpler",
                    "r", (2, 0), (("overfitting", "moderate"), ("high_cardinality", "high")))
    assert arm_of(both) == ("drop_columns", "high_cardinality")  # the most severe Diagnosis answered
    assert set(SELECTORS) == {"rules", "random", "thompson", "ucb1"}


def test_random_shuffles_only_the_applicable_proposals_and_follows_the_seed():
    data = make_synthetic("regression", 300, [("high_cardinality", 2.0), ("missing_values", 0.3)], seed=1,
                          fresh_rows=10)
    profile = DatasetProfiler().profile(data.frame, "target", task_type=TaskType.REGRESSION)
    diagnoses = [
        Diagnosis(DiagnosisKind.HIGH_CARDINALITY, Severity.HIGH, ("code",), {}, "m", "v"),
        Diagnosis(DiagnosisKind.CLASS_IMBALANCE, Severity.HIGH, (), {}, "m", "v"),  # class weights: not for regression
        Diagnosis(DiagnosisKind.OVERFITTING, Severity.MODERATE, (), {}, "m", "v"),
    ]
    applicable = propose(diagnoses, ExperimentConfig(target_column="target"), profile, TaskType.REGRESSION, set())
    assert "class_weight_balanced" not in keys(applicable) and len(applicable) >= 4
    first = RandomSelector().rank(applicable, context(None, seed=3))
    assert sorted(p.spec.key() for p in first) == sorted(p.spec.key() for p in applicable)  # the same Proposals
    assert first == RandomSelector().rank(list(reversed(applicable)), context(None, seed=3))  # fixed by the seed
    orders = {tuple(p.spec.key() for p in RandomSelector().rank(applicable, context(None, seed=s))) for s in range(10)}
    assert len(orders) > 1


# --------------------------------------------------------------------------- learned selectors
@pytest.mark.parametrize("selector", [ThompsonSelector, UCB1Selector])
def test_a_strategy_often_kept_on_similar_datasets_moves_up(tmp_path, selector):
    memory = memory_with(tmp_path, [
        ("A", "class_weight_balanced", "class_imbalance", "keep", 0.05, 10.0, 3),
        ("B", "class_weight_balanced", "class_imbalance", "keep", 0.04, 10.0, 3),
        ("A", "drop_columns", "high_cardinality", "discard", 0.0, 10.0, 3),
        ("B", "drop_columns", "high_cardinality", "discard", -0.01, 10.0, 3),
    ])
    chooser = selector()
    ranked = chooser.rank(PROPOSALS, context(memory))
    assert keys(ranked) == ["class_weight_balanced", "drop_columns", "regularised_search_space"]
    weights, drop = chooser.explain(WEIGHTS), chooser.explain(DROP)
    assert weights["observations"] == 6 and weights["rank"] == 0 and weights["rule_rank"] == 1
    assert drop["score"] < weights["score"] and chooser.explain(TREES)["score"] is None  # no record: no score
    assert keys(selector().rank(PROPOSALS, context(None))) == keys(RulesSelector().rank(PROPOSALS))  # no memory


@pytest.mark.parametrize("selector", [ThompsonSelector, UCB1Selector])
def test_too_few_observations_keep_the_rule_order(tmp_path, selector):
    two = memory_with(tmp_path / "two", [("A", "class_weight_balanced", "class_imbalance", "keep", 0.05, 10.0, 2),
                                         ("A", "drop_columns", "high_cardinality", "discard", 0.0, 10.0, 2)])
    assert keys(selector().rank(PROPOSALS, context(two))) == keys(RulesSelector().rank(PROPOSALS))
    # Only the places of the known arms are re-filled: drop_columns (unknown) keeps the first place.
    known = memory_with(tmp_path / "known", [
        ("A", "class_weight_balanced", "class_imbalance", "discard", 0.0, 10.0, 4),
        ("A", "regularised_search_space", "overfitting", "keep", 0.05, 10.0, 4),
    ])
    assert keys(selector().rank(PROPOSALS, context(known))) == ["drop_columns", "regularised_search_space",
                                                                "class_weight_balanced"]


def test_the_current_dataset_is_left_out(tmp_path):
    memory = memory_with(tmp_path, [("SELF", "class_weight_balanced", "class_imbalance", "keep", 0.05, 10.0, 6),
                                    ("SELF", "drop_columns", "high_cardinality", "discard", 0.0, 10.0, 6)])
    ranked = ThompsonSelector().rank(PROPOSALS, context(memory, fingerprint="SELF"))
    assert keys(ranked) == keys(RulesSelector().rank(PROPOSALS))


def test_the_same_memory_and_seed_give_the_same_choices(tmp_path):
    memory = memory_with(tmp_path, [
        ("A", "class_weight_balanced", "class_imbalance", "keep", 0.02, 10.0, 2),
        ("A", "class_weight_balanced", "class_imbalance", "discard", 0.0, 10.0, 2),
        ("B", "drop_columns", "high_cardinality", "keep", 0.02, 10.0, 2),
        ("B", "drop_columns", "high_cardinality", "discard", 0.0, 10.0, 2),
    ])

    def scores(seed):
        chooser = ThompsonSelector()
        order = keys(chooser.rank(PROPOSALS, context(memory, seed=seed)))
        return order, [chooser.explain(p)["score"] for p in PROPOSALS]

    assert scores(7) == scores(7)
    assert len({tuple(scores(seed)[1]) for seed in range(5)}) == 5  # the sampled keep probabilities differ


def test_ucb1_score_worked_by_hand_and_ties_go_to_the_cheaper_arm(tmp_path):
    memory = memory_with(tmp_path, [("A", "class_weight_balanced", "class_imbalance", "keep", 0.05, 20.0, 3),
                                    ("A", "smoother_search_metric", "class_imbalance", "keep", 0.05, 5.0, 3)])
    smoother = proposal("smoother_search_metric", "class_imbalance", cost="normal", row=8)
    chooser = UCB1Selector()
    ranked = chooser.rank([WEIGHTS, smoother], context(memory))
    expected = 1.0 + math.sqrt(2 * math.log(6) / 3)  # keep rate 1, N = 6 records of the kind, n = 3
    assert chooser.explain(WEIGHTS)["score"] == pytest.approx(expected) == chooser.explain(smoother)["score"]
    assert keys(ranked) == ["smoother_search_metric", "class_weight_balanced"]  # equal scores: 5 s before 20 s


# --------------------------------------------------------------------------- inside a Session
def test_a_session_with_memory_uses_the_learned_choice(tmp_path):
    def prefill(root):
        memory = ExperimentMemory(root / "experiments" / "memory" / "memory.jsonl")
        for n in range(4):  # on another dataset, class weights helped and dropping columns did not
            memory.add(record(n, fingerprint="OTHER", strategy="class_weight_balanced", verdict="keep", diff=0.05,
                              kinds=("class_imbalance",)))
            memory.add(record(10 + n, fingerprint="OTHER", strategy="drop_columns", verdict="discard", diff=0.0,
                              kinds=("high_cardinality",)))

    first_change = {}
    for name, selector, use_memory in [("rules", "rules", False), ("thompson_off", "thompson", False),
                                       ("thompson_on", "thompson", True)]:
        prefill(tmp_path / name)
        record_ = run_session(tmp_path / name, experiments=3, selector=selector, use_memory=use_memory)[1]
        first_change[name] = record_["steps"][1]["proposal"]["spec"]["strategy_id"]
        selection = record_["steps"][1]["selection"]
        assert selection["selector"] == selector and record_["settings"]["use_memory"] is use_memory
    assert first_change["rules"] == first_change["thompson_off"] == "drop_columns"
    assert first_change["thompson_on"] == "class_weight_balanced"
