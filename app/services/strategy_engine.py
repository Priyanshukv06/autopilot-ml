"""Strategy_Engine: turns Diagnoses into ranked Proposals with a written rule table (Requirement 8).

The rule table maps each diagnosis kind to candidate Strategies, in order. ``propose`` builds one
Proposal per change (one per affected column), drops changes already tried in the Session, changes
that do not apply to the Accepted_Setup, and repeats, then ranks them with the ``rules`` selector:
severity, then cost class, then the position in the table, then the change's key.

Thin version (task 7): rows for high_cardinality, overfitting and class_imbalance. Task 14 adds the
other rows. Positions follow the full table in design.md, so they do not change when rows are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from app.services.automl_runner import ExperimentConfig
from app.services.critic import Diagnosis, DiagnosisKind, Severity
from app.services.profiler import DatasetProfile
from app.services.strategies import StrategySpec, get_strategy
from app.services.task_detector import TaskType

# Rows of the rule table, in the order of design.md section 8.
TABLE_ORDER: tuple[DiagnosisKind, ...] = (
    DiagnosisKind.IDENTIFIER_COLUMN,
    DiagnosisKind.CONSTANT_COLUMN,
    DiagnosisKind.HIGH_CARDINALITY,
    DiagnosisKind.MISSING_VALUES,
    DiagnosisKind.REDUNDANT_FEATURES,
    DiagnosisKind.OVERFITTING,
    DiagnosisKind.UNDERFITTING,
    DiagnosisKind.BUDGET_LIMITED,
    DiagnosisKind.CLASS_IMBALANCE,
    DiagnosisKind.UNRELIABLE_VALIDATION,
    DiagnosisKind.WEAK_MODEL_FAMILIES,
    DiagnosisKind.WASTED_BUDGET,
)
SEVERITY_ORDER = {Severity.HIGH: 0, Severity.MODERATE: 1, Severity.LOW: 2}
COST_ORDER = {"cheap": 0, "normal": 1, "expensive": 2}

Candidate = Callable[[Diagnosis, list[Diagnosis]], list[StrategySpec]]


def _drop_each_column(diagnosis: Diagnosis, _: list[Diagnosis]) -> list[StrategySpec]:
    return [StrategySpec("drop_columns", {"columns": [column]}) for column in diagnosis.columns]


def _drop_high_cardinality_columns(_: Diagnosis, diagnoses: list[Diagnosis]) -> list[StrategySpec]:
    columns = sorted({c for d in diagnoses if d.kind is DiagnosisKind.HIGH_CARDINALITY for c in d.columns})
    return [StrategySpec("drop_columns", {"columns": [column]}) for column in columns]


def _single(strategy_id: str) -> Candidate:
    return lambda _diagnosis, _diagnoses: [StrategySpec(strategy_id, {})]


# Candidate Strategies for each diagnosis kind, in order. A kind without a row gets no Proposal.
RULES: dict[DiagnosisKind, tuple[Candidate, ...]] = {
    DiagnosisKind.HIGH_CARDINALITY: (_drop_each_column,),
    DiagnosisKind.OVERFITTING: (_single("regularised_search_space"), _drop_high_cardinality_columns),
    DiagnosisKind.CLASS_IMBALANCE: (_single("class_weight_balanced"),),
}


@dataclass(frozen=True)
class Proposal:
    spec: StrategySpec
    answers: tuple[DiagnosisKind, ...]  # the kinds of the Diagnoses it answers, in the Critic's order
    severity: Severity  # the highest severity among them
    cost_class: str
    complexity: str
    reason: str
    rule_position: tuple[int, int]  # (row in the rule table, candidate in the row); the earliest if several

    def ranking_key(self) -> tuple[int, int, tuple[int, int], str]:
        return (SEVERITY_ORDER[self.severity], COST_ORDER[self.cost_class], self.rule_position, self.spec.key())

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "key": self.spec.key(),
            "answers": [kind.value for kind in self.answers],
            "severity": self.severity.value,
            "cost_class": self.cost_class,
            "complexity": self.complexity,
            "reason": self.reason,
            "rule_position": list(self.rule_position),
        }


class StrategySelector(Protocol):
    name: str

    def rank(self, proposals: list[Proposal], context: Any = None) -> list[Proposal]: ...


class RulesSelector:
    """The written ranking: severity, cost class, position in the rule table, key."""

    name = "rules"

    def rank(self, proposals: list[Proposal], context: Any = None) -> list[Proposal]:
        return sorted(proposals, key=Proposal.ranking_key)


def propose(
    diagnoses: list[Diagnosis],
    config: ExperimentConfig,
    profile: DatasetProfile,
    task_type: TaskType,
    tried: set[str],
) -> list[Proposal]:
    """Proposals for the Accepted_Setup ``config``, ranked by the rules selector.

    ``tried`` holds the keys of the changes already tried in the Session. A change that several
    Diagnoses ask for becomes one Proposal that answers all of them.
    """
    found: dict[str, dict[str, Any]] = {}
    for diagnosis in diagnoses:
        if diagnosis.kind not in RULES:
            continue
        row = TABLE_ORDER.index(diagnosis.kind)
        for index, candidate in enumerate(RULES[diagnosis.kind]):
            for spec in candidate(diagnosis, diagnoses):
                key = spec.key()
                if key in tried:
                    continue
                strategy = get_strategy(spec.strategy_id)
                if key not in found:
                    if not strategy.applicable(config, profile, task_type, spec.params):
                        continue
                    found[key] = {"spec": spec, "diagnoses": [], "position": (row, index)}
                entry = found[key]
                if diagnosis not in entry["diagnoses"]:
                    entry["diagnoses"].append(diagnosis)
                entry["position"] = min(entry["position"], (row, index))

    proposals = []
    for entry in found.values():
        spec, answered = entry["spec"], entry["diagnoses"]
        strategy = get_strategy(spec.strategy_id)
        severity = min((d.severity for d in answered), key=SEVERITY_ORDER.__getitem__)
        proposals.append(
            Proposal(
                spec=spec,
                answers=tuple(dict.fromkeys(d.kind for d in answered)),
                severity=severity,
                cost_class=strategy.cost_class,
                complexity=strategy.complexity,
                reason=f"{strategy.describe(spec.params)}. Answers: {'; '.join(_answer_text(d) for d in answered)}.",
                rule_position=entry["position"],
            )
        )
    return RulesSelector().rank(proposals)


def _answer_text(diagnosis: Diagnosis) -> str:
    name = diagnosis.kind.value.replace("_", " ")
    columns = f" of {', '.join(repr(c) for c in diagnosis.columns)}" if diagnosis.columns else ""
    return f"{name}{columns} ({diagnosis.severity.value})"
