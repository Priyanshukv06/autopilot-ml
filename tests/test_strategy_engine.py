"""Tests for the rule-based Strategy_Engine, thin version (task 7.1 of the spec)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.automl_runner import ExperimentConfig
from app.services.critic import Diagnosis, DiagnosisKind, Severity
from app.services.profiler import DatasetProfiler
from app.services.strategies import CATALOGUE, StrategySpec
from app.services.strategy_engine import Proposal, RulesSelector, propose
from app.services.task_detector import TaskType

ROOT = Path(__file__).resolve().parents[1]
K, S = DiagnosisKind, Severity
TITANIC_CONFIG = ExperimentConfig(target_column="Survived", excluded_columns=["PassengerId", "Name"])


def diagnosis(kind, severity, columns=()):
    return Diagnosis(kind, severity, tuple(columns), {}, "message", "test")


def drop(column):
    return StrategySpec("drop_columns", {"columns": [column]}).key()


def encode(column):
    return StrategySpec("frequency_encode", {"columns": [column]}).key()


def group(column):
    return StrategySpec("group_rare_categories", {"columns": [column]}).key()


SMALLER = StrategySpec("regularised_search_space", {}).key()
WEIGHTS = StrategySpec("class_weight_balanced", {}).key()
SMOOTHER = StrategySpec("smoother_search_metric", {}).key()


@pytest.fixture(scope="module")
def titanic_profile():
    df = pd.read_csv(ROOT / "data" / "sample" / "titanic.csv")
    return DatasetProfiler().profile(df, "Survived", task_type=TaskType.BINARY)


@pytest.fixture(scope="module")
def titanic_diagnoses():
    return [diagnosis(K.OVERFITTING, S.MODERATE), diagnosis(K.HIGH_CARDINALITY, S.MODERATE, ["Cabin"]),
            diagnosis(K.HIGH_CARDINALITY, S.HIGH, ["Ticket"])]


def test_titanic_proposals_in_rule_order(titanic_profile, titanic_diagnoses):
    proposals = propose(titanic_diagnoses, TITANIC_CONFIG, titanic_profile, TaskType.BINARY, tried=set())
    # High first (everything for Ticket), then moderate (Cabin, then smaller trees); in each group, the
    # high-cardinality row's order: drop, frequency encoding, grouping.
    assert [p.spec.key() for p in proposals] == [
        drop("Ticket"), encode("Ticket"), group("Ticket"), drop("Cabin"), encode("Cabin"), group("Cabin"), SMALLER,
    ]
    ticket, cabin, smaller = proposals[0], proposals[3], proposals[6]
    # Dropping Ticket answers both Diagnoses that ask for it, with the higher severity.
    assert ticket.answers == (K.OVERFITTING, K.HIGH_CARDINALITY) and ticket.severity is S.HIGH
    assert ticket.rule_position == (2, 0) and ticket.cost_class == "cheap" and ticket.complexity == "simpler"
    assert "Stop using the column 'Ticket'" in ticket.reason and "high cardinality of 'Ticket' (high)" in ticket.reason
    assert cabin.severity is S.MODERATE and smaller.answers == (K.OVERFITTING,) and smaller.rule_position == (5, 0)
    json.dumps([p.to_dict() for p in proposals])  # JSON-safe for the records


def test_tried_changes_are_not_proposed_again(titanic_profile, titanic_diagnoses):
    proposals = propose(titanic_diagnoses, TITANIC_CONFIG, titanic_profile, TaskType.BINARY,
                        tried={drop("Ticket"), SMALLER})
    assert [p.spec.key() for p in proposals] == [encode("Ticket"), group("Ticket"), drop("Cabin"), encode("Cabin"),
                                                 group("Cabin")]


def test_changes_that_conflict_with_the_accepted_setup_are_left_out(titanic_profile, titanic_diagnoses):
    # The Accepted_Setup already has the smaller trees and no longer uses Ticket.
    accepted = CATALOGUE["regularised_search_space"].apply(
        ExperimentConfig(target_column="Survived", excluded_columns=["PassengerId", "Name", "Ticket"]), {}
    )
    proposals = propose(titanic_diagnoses, accepted, titanic_profile, TaskType.BINARY, tried=set())
    assert [p.spec.key() for p in proposals] == [drop("Cabin"), encode("Cabin"), group("Cabin")]


def test_class_imbalance_gets_class_weights_for_classification_only():
    df = pd.DataFrame({"x": np.arange(40.0), "y": [0] * 30 + [1] * 10})
    p = DatasetProfiler().profile(df, "y")
    imbalance = [diagnosis(K.CLASS_IMBALANCE, S.MODERATE, ["y"])]
    config = ExperimentConfig(target_column="y")
    binary = propose(imbalance, config, p, TaskType.BINARY, tried=set())
    assert [pr.spec.key() for pr in binary] == [WEIGHTS, SMOOTHER] and binary[0].answers == (K.CLASS_IMBALANCE,)
    assert propose(imbalance, config, p, TaskType.REGRESSION, tried=set()) == []
    weighted = ExperimentConfig(target_column="y", class_weight="balanced")
    assert [pr.spec.key() for pr in propose(imbalance, weighted, p, TaskType.BINARY, tried=set())] == [SMOOTHER]


def test_ranking_severity_before_table_order(titanic_profile):
    # class_imbalance comes late in the table (row 8) but is high; Cabin's row comes first but is moderate.
    diagnoses = [diagnosis(K.HIGH_CARDINALITY, S.MODERATE, ["Cabin"]), diagnosis(K.CLASS_IMBALANCE, S.HIGH, ["Survived"])]
    proposals = propose(diagnoses, TITANIC_CONFIG, titanic_profile, TaskType.BINARY, tried=set())
    # The high Diagnosis's Proposals come first even though the smoother metric costs more than dropping Cabin.
    assert [p.spec.key() for p in proposals] == [WEIGHTS, SMOOTHER, drop("Cabin"), encode("Cabin"), group("Cabin")]


def test_rules_selector_orders_by_severity_cost_position_then_key():
    def proposal(key, severity, cost, position):
        spec = StrategySpec(key, {})
        return Proposal(spec, (K.OVERFITTING,), severity, cost, "same", "reason", position)

    proposals = [
        proposal("e", S.MODERATE, "cheap", (5, 0)),
        proposal("d", S.MODERATE, "cheap", (2, 0)),
        proposal("c", S.MODERATE, "normal", (0, 0)),
        proposal("b", S.HIGH, "expensive", (9, 0)),
        proposal("f", S.MODERATE, "cheap", (2, 0)),  # ties with d on everything but the key
        proposal("a", S.LOW, "cheap", (0, 0)),
    ]
    ranked = RulesSelector().rank(proposals)
    assert [p.spec.strategy_id for p in ranked] == ["b", "d", "f", "e", "c", "a"]
    assert RulesSelector().rank(list(reversed(proposals))) == ranked  # the input order does not matter


def test_nothing_to_propose_gives_an_empty_list(titanic_profile):
    assert propose([], TITANIC_CONFIG, titanic_profile, TaskType.BINARY, tried=set()) == []
    # wasted_budget has no row: the Resource_Manager answers it by shortening the next search.
    assert propose([diagnosis(K.WASTED_BUDGET, S.HIGH)], TITANIC_CONFIG, titanic_profile, TaskType.BINARY,
                   tried=set()) == []
    # Every Proposal already tried: nothing left.
    every = [drop("Ticket"), encode("Ticket"), group("Ticket"), drop("Cabin"), encode("Cabin"), group("Cabin"), SMALLER]
    assert propose([diagnosis(K.OVERFITTING, S.MODERATE), diagnosis(K.HIGH_CARDINALITY, S.HIGH, ["Ticket"]),
                    diagnosis(K.HIGH_CARDINALITY, S.MODERATE, ["Cabin"])], TITANIC_CONFIG, titanic_profile,
                   TaskType.BINARY, tried=set(every)) == []


def test_proposals_are_repeatable(titanic_profile, titanic_diagnoses):
    first = propose(titanic_diagnoses, TITANIC_CONFIG, titanic_profile, TaskType.BINARY, tried=set())
    again = propose(list(titanic_diagnoses), TITANIC_CONFIG, titanic_profile, TaskType.BINARY, tried=set())
    assert [p.to_dict() for p in first] == [p.to_dict() for p in again]
