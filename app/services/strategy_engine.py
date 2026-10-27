"""Strategy_Engine: turns Diagnoses into ranked Proposals with a written rule table (Requirement 8).

The rule table maps each diagnosis kind to candidate Strategies, in order. ``propose`` builds one
Proposal per change (one per affected column), drops changes already tried in the Session, changes
that do not apply to the Accepted_Setup, and repeats, then ranks them with the ``rules`` order:
severity, then cost class, then the position in the table, then the change's key.

Selectors (Requirement 19) re-rank them: ``rules`` keeps that order, ``random`` shuffles it, and the
learned ``thompson`` and ``ucb1`` use what the Experiment_Memory saw on the nearest past datasets for
each arm, a pair of Strategy and diagnosis kind.

The rule table is the full table of design.md (the thin version of task 7 had the rows for
high_cardinality, overfitting and class_imbalance; task 14 added the others). Positions count its
rows, so they did not change when rows were added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from app.services.automl_runner import ExperimentConfig
from app.services.critic import Diagnosis, DiagnosisKind, Severity
from app.services.experiment_memory import ExperimentMemory, StrategyStats
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

Candidate = Callable[[Diagnosis, list[Diagnosis], ExperimentConfig], list[StrategySpec]]


def _each_column(strategy_id: str) -> Candidate:
    """One Proposal per column of the Diagnosis."""
    return lambda diagnosis, _diagnoses, _config: [
        StrategySpec(strategy_id, {"columns": [column]}) for column in diagnosis.columns
    ]


def _drop_if_high(diagnosis: Diagnosis, _diagnoses: list[Diagnosis], _config: ExperimentConfig) -> list[StrategySpec]:
    if diagnosis.severity is not Severity.HIGH:
        return []
    return [StrategySpec("drop_columns", {"columns": [column]}) for column in diagnosis.columns]


def _drop_later_column(diagnosis: Diagnosis, _diagnoses: list[Diagnosis], _config: ExperimentConfig) -> list[StrategySpec]:
    """For a redundant pair (in data order), drop the second column."""
    return [StrategySpec("drop_columns", {"columns": [diagnosis.columns[-1]]})]


def _drop_high_cardinality_columns(_diagnosis: Diagnosis, diagnoses: list[Diagnosis],
                                   _config: ExperimentConfig) -> list[StrategySpec]:
    columns = sorted({c for d in diagnoses if d.kind is DiagnosisKind.HIGH_CARDINALITY for c in d.columns})
    return [StrategySpec("drop_columns", {"columns": [column]}) for column in columns]


def _keep_strong_families(diagnosis: Diagnosis, _diagnoses: list[Diagnosis],
                          config: ExperimentConfig) -> list[StrategySpec]:
    """Keep the families of the Setup (or, if it uses FLAML's own list, the families the search tried, in the
    order it first tried them) without the weak ones."""
    weak = set(str(diagnosis.evidence.get("families", "")).split(","))
    tried = [f for f in str(diagnosis.evidence.get("tried_families", "")).split(",") if f]
    keep = [f for f in (config.estimator_list or tried) if f not in weak]
    return [StrategySpec("restrict_model_families", {"keep": keep})] if keep else []


def _single(strategy_id: str) -> Candidate:
    return lambda _diagnosis, _diagnoses, _config: [StrategySpec(strategy_id, {})]


# Candidate Strategies for each diagnosis kind, in order. A kind without a row gets no Proposal
# (wasted_budget is answered by the Resource_Manager, which shortens the next search).
RULES: dict[DiagnosisKind, tuple[Candidate, ...]] = {
    DiagnosisKind.IDENTIFIER_COLUMN: (_each_column("drop_columns"),),
    DiagnosisKind.CONSTANT_COLUMN: (_each_column("drop_columns"),),
    DiagnosisKind.HIGH_CARDINALITY: (_each_column("drop_columns"), _each_column("frequency_encode"),
                                     _each_column("group_rare_categories")),
    DiagnosisKind.MISSING_VALUES: (_drop_if_high, _each_column("missing_indicator")),
    DiagnosisKind.REDUNDANT_FEATURES: (_drop_later_column,),
    DiagnosisKind.OVERFITTING: (_single("regularised_search_space"), _drop_high_cardinality_columns),
    DiagnosisKind.UNDERFITTING: (_single("more_search_time"),),
    DiagnosisKind.BUDGET_LIMITED: (_single("more_search_time"),),
    DiagnosisKind.CLASS_IMBALANCE: (_single("class_weight_balanced"), _single("smoother_search_metric")),
    DiagnosisKind.UNRELIABLE_VALIDATION: (_single("smoother_search_metric"),),
    DiagnosisKind.WEAK_MODEL_FAMILIES: (_keep_strong_families,),
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
    # (kind, severity) of each Diagnosis it answers, in the Critic's order (for the Experiment_Memory)
    answered: tuple[tuple[str, str], ...] = ()

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
            "answered": [{"kind": kind, "severity": severity} for kind, severity in self.answered],
        }


MIN_OBSERVATIONS = 3  # an arm with fewer observations keeps its rule-order position (Requirement 19.4)
NEAREST_DATASETS = 5
MIN_COST_SECONDS = 1.0  # floor of an arm's mean cost, so a very cheap arm cannot divide by almost zero


@dataclass
class SelectionContext:
    """What the selectors read (Requirement 19). ``memory`` None means: no memory (every arm unknown)."""

    memory: ExperimentMemory | None
    meta_features: dict[str, Any]  # of the Training_Part
    task_type: TaskType
    fingerprint: str | None  # the current dataset; its own records are left out, so only other datasets count
    rng: np.random.Generator  # seeded from the Session seed (Requirement 19.5)
    k: int = NEAREST_DATASETS


class StrategySelector(Protocol):
    name: str

    def rank(self, proposals: list[Proposal], context: SelectionContext | None = None) -> list[Proposal]: ...

    def explain(self, proposal: Proposal) -> dict[str, Any] | None: ...


def arm_of(proposal: Proposal) -> tuple[str, str]:
    """A Proposal's arm: its Strategy and the kind of the most severe Diagnosis it answers (the first in the
    Critic's order if several are equally severe)."""
    if proposal.answered:
        kind, _ = min(proposal.answered, key=lambda answer: SEVERITY_ORDER[Severity(answer[1])])
        return proposal.spec.strategy_id, kind
    return proposal.spec.strategy_id, proposal.answers[0].value if proposal.answers else "none"


class RulesSelector:
    """The written ranking: severity, cost class, position in the rule table, key."""

    name = "rules"

    def __init__(self) -> None:
        self._last: dict[str, dict[str, Any]] = {}

    def rank(self, proposals: list[Proposal], context: SelectionContext | None = None) -> list[Proposal]:
        ranked = sorted(proposals, key=Proposal.ranking_key)
        self._last = {p.spec.key(): {"selector": self.name, "rule_rank": i, "rank": i} for i, p in enumerate(ranked)}
        return ranked

    def explain(self, proposal: Proposal) -> dict[str, Any] | None:
        """How the last ``rank`` placed this Proposal (for the Session record)."""
        return self._last.get(proposal.spec.key())


class RandomSelector(RulesSelector):
    """All applicable Proposals in a random order, drawn with the Session's generator."""

    name = "random"

    def rank(self, proposals: list[Proposal], context: SelectionContext | None = None) -> list[Proposal]:
        ordered = sorted(proposals, key=Proposal.ranking_key)
        rng = context.rng if context is not None else np.random.default_rng(0)
        ranked = [ordered[i] for i in rng.permutation(len(ordered))]
        self._last = {p.spec.key(): {"selector": self.name, "rule_rank": ordered.index(p), "rank": i}
                      for i, p in enumerate(ranked)}
        return ranked


class _MemorySelector(RulesSelector):
    """Ranks by a score from the memory's arm statistics on the nearest past datasets of the same task.

    The Proposals are first put in rule order. The places of the Proposals whose arm has at least
    MIN_OBSERVATIONS observations are then filled by those Proposals, best score first; the others keep
    their places. Without a memory every arm has 0 observations, so the result is the rule order.
    """

    def rank(self, proposals: list[Proposal], context: SelectionContext | None = None) -> list[Proposal]:
        ordered = sorted(proposals, key=Proposal.ranking_key)
        stats = self._stats(context)
        info: dict[str, dict[str, Any]] = {}
        known: list[tuple[tuple[float, float, int], int]] = []  # (sort key, rule position)
        for position, proposal in enumerate(ordered):
            arm = arm_of(proposal)
            arm_stats = stats.get(arm)
            observations = arm_stats.observations if arm_stats else 0
            score = None
            if arm_stats is not None and observations >= MIN_OBSERVATIONS:
                score = self._score(arm_stats, stats, context)
                known.append(((-score, self._tie(arm_stats), position), position))
            info[proposal.spec.key()] = {"selector": self.name, "arm": list(arm), "observations": observations,
                                         "score": score, "rule_rank": position}
        slots = sorted(position for _, position in known)
        ranked = list(ordered)
        for slot, (_, position) in zip(slots, sorted(known)):
            ranked[slot] = ordered[position]
        for rank, proposal in enumerate(ranked):
            info[proposal.spec.key()]["rank"] = rank
        self._last = info
        return ranked

    @staticmethod
    def _stats(context: SelectionContext | None) -> dict[tuple[str, str], StrategyStats]:
        if context is None or context.memory is None:
            return {}
        rows = context.memory.strategy_stats(context.meta_features, context.task_type.value, k=context.k,
                                             exclude_fingerprint=context.fingerprint)
        return {(s.strategy_id, s.diagnosis_kind): s for s in rows}

    def _score(self, arm: StrategyStats, stats: dict[tuple[str, str], StrategyStats],
               context: SelectionContext) -> float:
        raise NotImplementedError

    def _tie(self, arm: StrategyStats) -> float:
        return 0.0  # equal scores keep the rule order


class ThompsonSelector(_MemorySelector):
    """Thompson sampling (Thompson, 1933): a keep probability drawn from Beta(1 + weighted keeps,
    1 + weighted discards), times the mean gain when kept, divided by the mean cost: an expected
    validation gain per second (Requirement 19.3)."""

    name = "thompson"

    def _score(self, arm: StrategyStats, stats: dict[tuple[str, str], StrategyStats],
               context: SelectionContext) -> float:
        keep_probability = float(context.rng.beta(1.0 + arm.weighted_keeps, 1.0 + arm.weighted_discards))
        gain = arm.mean_gain_when_kept if arm.mean_gain_when_kept is not None else 0.0
        return keep_probability * gain / max(arm.mean_cost_seconds, MIN_COST_SECONDS)


class UCB1Selector(_MemorySelector):
    """UCB1 (Auer, Cesa-Bianchi and Fischer, 2002): keep rate + sqrt(2 ln N / n), with n the arm's
    observations and N those of all arms of the same diagnosis kind; equal scores go to the cheaper arm."""

    name = "ucb1"

    def _score(self, arm: StrategyStats, stats: dict[tuple[str, str], StrategyStats],
               context: SelectionContext) -> float:
        total = sum(s.observations for s in stats.values() if s.diagnosis_kind == arm.diagnosis_kind)
        return arm.keep_rate + math.sqrt(2.0 * math.log(total) / arm.observations)

    def _tie(self, arm: StrategyStats) -> float:
        return arm.mean_cost_seconds


SELECTORS: dict[str, Callable[[], StrategySelector]] = {
    "rules": RulesSelector, "random": RandomSelector, "thompson": ThompsonSelector, "ucb1": UCB1Selector,
}


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
            for spec in candidate(diagnosis, diagnoses, config):
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

    proposals: list[Proposal] = []
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
                answered=tuple((d.kind.value, d.severity.value) for d in answered),
            )
        )
    return sorted(proposals, key=Proposal.ranking_key)


def random_proposals(
    config: ExperimentConfig,
    feature_columns: list[str],
    families: list[str],
    profile: DatasetProfile,
    task_type: TaskType,
    tried: set[str],
    rng: np.random.Generator,
) -> list[Proposal]:
    """The Session without the Critic (ablation, Requirement 20.1): for every Strategy of the catalogue, one
    change with random valid parameters, kept if it applies to ``config`` and was not tried; in a random order.

    ``feature_columns`` are the Accepted_Setup's columns and ``families`` the learner families its search
    tried. The Proposals answer no Diagnosis.
    """
    from app.services.strategies import CATALOGUE

    numeric = set(profile.numerical_features)
    text = [c for c in feature_columns if c not in numeric]
    with_missing = [c for c in feature_columns if c in numeric and profile.missing_ratio.get(c, 0.0) > 0]

    def pick(columns: list[str]) -> list[str]:
        return [columns[int(rng.integers(len(columns)))]] if columns else []

    candidates: list[StrategySpec] = []
    for strategy_id in CATALOGUE:
        if strategy_id == "drop_columns":
            columns = pick(list(feature_columns))
            params: dict[str, Any] | None = {"columns": columns} if columns else None
        elif strategy_id in ("frequency_encode", "group_rare_categories"):
            columns = pick(text)
            params = {"columns": columns} if columns else None
        elif strategy_id == "missing_indicator":
            columns = pick(with_missing)
            params = {"columns": columns} if columns else None
        elif strategy_id == "restrict_model_families":
            if len(families) >= 2:
                left_out = families[int(rng.integers(len(families)))]
                params = {"keep": [f for f in families if f != left_out]}
            else:
                params = None
        else:
            params = {}
        if params is not None:
            candidates.append(StrategySpec(strategy_id, params))

    proposals = []
    for index, spec in enumerate(candidates):
        strategy = get_strategy(spec.strategy_id)
        if spec.key() in tried or not strategy.applicable(config, profile, task_type, spec.params):
            continue
        proposals.append(Proposal(
            spec=spec, answers=(), severity=Severity.LOW, cost_class=strategy.cost_class,
            complexity=strategy.complexity, reason=f"{strategy.describe(spec.params)}. Chosen at random (no Critic).",
            rule_position=(len(TABLE_ORDER), index), answered=(),
        ))
    return [proposals[i] for i in rng.permutation(len(proposals))]


def _answer_text(diagnosis: Diagnosis) -> str:
    name = diagnosis.kind.value.replace("_", " ")
    columns = f" of {', '.join(repr(c) for c in diagnosis.columns)}" if diagnosis.columns else ""
    return f"{name}{columns} ({diagnosis.severity.value})"
