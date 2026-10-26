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


# =========================================================================== the full rule table (task 14.1)
@pytest.fixture(scope="module")
def mixed():
    """A text column with empty cells, numeric columns with and without empty cells, and a 0/1 target."""
    df = pd.DataFrame({
        "code": [f"c{i % 9}" if i % 5 else None for i in range(40)],
        "age": [np.nan if i % 3 == 0 else float(i) for i in range(40)],
        "a": np.arange(40, dtype=float),
        "b": np.arange(40, dtype=float) * 2,
        "row_id": np.arange(40),
        "flag": ["on"] * 40,
        "y": [0, 1] * 20,
    })
    return DatasetProfiler().profile(df, "y", task_type=TaskType.BINARY)


def keys(diagnoses, profile, config=None, task_type=TaskType.BINARY, tried=()):
    config = config or ExperimentConfig(target_column="y")
    return [p.spec.key() for p in propose(diagnoses, config, profile, task_type, set(tried))]


def spec(strategy_id, **params):
    return StrategySpec(strategy_id, params).key()


def test_identifier_and_constant_columns_are_dropped(mixed):
    assert keys([diagnosis(K.IDENTIFIER_COLUMN, S.HIGH, ["row_id"])], mixed) == [drop("row_id")]
    assert keys([diagnosis(K.CONSTANT_COLUMN, S.LOW, ["flag"])], mixed) == [drop("flag")]


def test_missing_values_row(mixed):
    indicator = spec("missing_indicator", columns=["age"])
    assert keys([diagnosis(K.MISSING_VALUES, S.HIGH, ["age"])], mixed) == [drop("age"), indicator]
    assert keys([diagnosis(K.MISSING_VALUES, S.MODERATE, ["age"])], mixed) == [indicator]  # drop only if high
    # A text column gets no indicator, so a moderate case has nothing to try.
    assert keys([diagnosis(K.MISSING_VALUES, S.HIGH, ["code"])], mixed) == [drop("code")]
    assert keys([diagnosis(K.MISSING_VALUES, S.MODERATE, ["code"])], mixed) == []


def test_redundant_pair_drops_the_later_column(mixed):
    assert keys([diagnosis(K.REDUNDANT_FEATURES, S.HIGH, ["a", "b"])], mixed) == [drop("b")]


def test_underfitting_and_budget_limited_share_one_more_search_time_proposal(mixed):
    proposals = propose([diagnosis(K.UNDERFITTING, S.MODERATE), diagnosis(K.BUDGET_LIMITED, S.HIGH)],
                        ExperimentConfig(target_column="y"), mixed, TaskType.BINARY, set())
    assert [p.spec.key() for p in proposals] == [spec("more_search_time")]
    assert proposals[0].answers == (K.UNDERFITTING, K.BUDGET_LIMITED) and proposals[0].severity is S.HIGH
    assert proposals[0].cost_class == "expensive" and proposals[0].rule_position == (6, 0)


def test_unreliable_validation_gets_the_smoother_metric_for_classification_only(mixed):
    assert keys([diagnosis(K.UNRELIABLE_VALIDATION, S.MODERATE)], mixed) == [SMOOTHER]
    assert keys([diagnosis(K.UNRELIABLE_VALIDATION, S.MODERATE)], mixed, task_type=TaskType.REGRESSION) == []


def test_weak_families_are_left_out_in_the_order_the_search_tried_them(mixed):
    weak = Diagnosis(K.WEAK_MODEL_FAMILIES, S.LOW, (), {"families": "lrl1,sgd",
                                                        "tried_families": "lgbm,sgd,xgboost,rf,lrl1"}, "m", "test")
    assert keys([weak], mixed) == [spec("restrict_model_families", keep=["lgbm", "xgboost", "rf"])]
    explicit = ExperimentConfig(target_column="y", estimator_list=["xgboost", "sgd", "lgbm"])
    assert keys([weak], mixed, config=explicit) == [spec("restrict_model_families", keep=["xgboost", "lgbm"])]
    only_weak = ExperimentConfig(target_column="y", estimator_list=["sgd", "lrl1"])
    assert keys([weak], mixed, config=only_weak) == []  # nothing would be left


def test_a_treated_column_is_not_encoded_again(mixed):
    encoded = CATALOGUE["frequency_encode"].apply(ExperimentConfig(target_column="y"), {"columns": ["code"]})
    assert keys([diagnosis(K.HIGH_CARDINALITY, S.HIGH, ["code"])], mixed, config=encoded) == [drop("code")]


def test_costs_order_proposals_of_the_same_severity(mixed):
    diagnoses = [diagnosis(K.UNDERFITTING, S.MODERATE), diagnosis(K.CLASS_IMBALANCE, S.MODERATE, ["y"]),
                 diagnosis(K.IDENTIFIER_COLUMN, S.MODERATE, ["row_id"])]
    # cheap (drop row_id, class weights) before normal (smoother metric) before expensive (more search time)
    assert keys(diagnoses, mixed) == [drop("row_id"), WEIGHTS, SMOOTHER, spec("more_search_time")]
