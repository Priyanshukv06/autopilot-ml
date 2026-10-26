"""Judge study (spec task 17, Requirement 10): how often does the Judge keep harmful changes, and how
often does it keep helpful ones?

Pairs of Setups on synthetic binary datasets (``research/synthetic.py``): A is the plain Setup, B
changes one thing. The true difference is B's F1 minus A's F1 on the fresh sample of 20,000 rows,
each measured with the model FLAML trained on all 1,000 rows of the dataset. The Judge only sees the
Assessments of the two configurations on the dataset's own rows.

Decisions, fixed before running (design.md, task 17 notes):

- Pair kinds (``PAIRS``): changes that are likely harmful (dropping one or two informative
  columns), likely about neutral (class weights on balanced classes, dropping a near-copy of a
  column) and likely helpful (dropping a random high-cardinality column, class weights for F1 on
  imbalanced classes, dropping 20 random columns, smaller trees on noisy labels). The true
  difference is measured, so a pair counts by what really happened, not by its kind.
- Tuning seeds 0 to 5 and reporting seeds 100 to 105: 7 pairs per seed, 84 pairs in all.
- Each Setup: one FLAML search (default learners, F1, at most 25 trials, 60 s cap, all cores),
  then an Assessment of its best configuration on 5 x 5 folds. The 1- and 3-repeat Judges use the
  first 5 and 15 of these folds, which are exactly the folds a Session with 1 or 3 repeats would make.
- For alpha in {0.05, 0.10, 0.20} and repeats in {1, 3, 5}: the Judge's verdict (margin 0.005,
  minimum gain 0).
- False keep: verdict keep or keep_simpler when the true difference is below -0.005 (the margin).
  Power: verdict keep when the true difference is above 0.01.
- Defaults: on the tuning seeds, the setting with the highest power among those with a false-keep
  rate of at most 0.10; ties go to fewer repeats (cheaper), then to the smaller alpha. If no setting
  meets the limit, the one with the lowest false-keep rate. The rates are then reported on the
  reporting seeds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from app.services.assessor import Assessment, Assessor
from app.services.automl_runner import AutoMLEngine, ExperimentConfig, FLAMLEngine, to_jsonable
from app.services.evaluator import Evaluator
from app.services.judge import Judge, JudgeSettings
from app.services.splits import make_session_folds
from app.services.strategies import REGULARISED_LIMITS
from app.services.task_detector import TaskType
from research.synthetic import TARGET, make_synthetic

ALPHAS = (0.05, 0.10, 0.20)
REPEATS = (1, 3, 5)
N_SPLITS = 5
MARGIN, MIN_GAIN = 0.005, 0.0
HARMFUL_BELOW, HELPFUL_ABOVE = -MARGIN, 0.01
MAX_FALSE_KEEP = 0.10
TUNING_SEEDS = (0, 1, 2, 3, 4, 5)
REPORTING_SEEDS = (100, 101, 102, 103, 104, 105)
N_ROWS, FRESH_ROWS = 1000, 20_000
MAX_ITER, TIME_BUDGET = 25, 60
METRIC = "f1"

Change = Callable[[ExperimentConfig, Any], ExperimentConfig]


@dataclass(frozen=True)
class PairKind:
    problems: tuple[tuple[str, float], ...]
    change: Change  # builds Setup B from Setup A and the dataset
    simpler: bool  # whether B counts as simpler for the Judge


def _drop(columns: Callable[[Any], list[str]]) -> Change:
    return lambda config, data: replace(config, excluded_columns=[*config.excluded_columns, *columns(data)])


PAIRS: dict[str, PairKind] = {
    "drop_one_informative": PairKind((), _drop(lambda data: ["x0"]), simpler=True),
    "drop_two_informative": PairKind((), _drop(lambda data: ["x0", "x1"]), simpler=True),
    "drop_noise_code": PairKind((("high_cardinality", 2.0),), _drop(lambda data: ["code"]), simpler=True),
    "weights_on_imbalance": PairKind((("class_imbalance", 5.0),),
                                     lambda config, data: replace(config, class_weight="balanced"), simpler=False),
    "weights_on_balance": PairKind((), lambda config, data: replace(config, class_weight="balanced"), simpler=False),
    "drop_useless": PairKind((("useless_features", 20.0),),
                             _drop(lambda data: list(data.problem("useless_features").columns)), simpler=True),
    "smaller_trees_noisy": PairKind((("label_noise", 0.2),),
                                    lambda config, data: replace(config, search_space_limits=REGULARISED_LIMITS),
                                    simpler=True),
}


@dataclass(frozen=True)
class StudyPair:
    seed: int
    kind: str
    generator_seed: int

    @property
    def pair_id(self) -> str:
        return f"s{self.seed}-{self.kind}"

    @property
    def split(self) -> str:
        return "tuning" if self.seed in TUNING_SEEDS else "reporting"


def study_pairs(seeds: Iterable[int]) -> list[StudyPair]:
    return [StudyPair(seed, kind, seed * 1000 + 500 + index) for seed in seeds for index, kind in enumerate(PAIRS)]


def _search_and_assess(engine: AutoMLEngine, data: Any, config: ExperimentConfig, folds: Any, seed: int, n_jobs: int,
                       work_dir: Path, max_iter: int, time_budget: int) -> tuple[Any, Assessment, float]:
    X = data.frame.drop(columns=[TARGET, *config.excluded_columns])
    y = data.frame[TARGET]
    weights = compute_sample_weight("balanced", y) if config.class_weight == "balanced" else None
    result = engine.fit(X, y, task_type=TaskType.BINARY, metric=METRIC, time_budget=time_budget, seed=seed,
                        n_jobs=n_jobs, estimator_list=None, work_dir=work_dir, max_iter=max_iter,
                        search_space_limits=config.search_space_limits or None, sample_weight=weights)
    stand_in = SimpleNamespace(setup_key=str(config.excluded_columns), experiment_id="study",
                               best_estimator=result.best_estimator, best_config=result.best_config)
    assessment = Assessor(engine, folds, TaskType.BINARY, METRIC, n_jobs, seed).assess(X, y, config, stand_in)
    fresh = data.fresh.drop(columns=[TARGET, *config.excluded_columns])
    fresh_f1 = Evaluator(TaskType.BINARY, METRIC).evaluate(result.model, fresh, data.fresh[TARGET],
                                                           result.classes).primary_score
    return result, assessment, fresh_f1


def first_folds(assessment: Assessment, n_folds: int, label: str) -> Assessment:
    """The Assessment on its first ``n_folds`` folds only (the folds of a Session with fewer repeats)."""
    return replace(assessment, folds=assessment.folds[:n_folds], folds_fingerprint=f"{assessment.folds_fingerprint}-{label}")


def run_pair(pair: StudyPair, *, work_dir: Path, engine: AutoMLEngine | None = None, max_iter: int = MAX_ITER,
             time_budget: int = TIME_BUDGET, n_jobs: int = -1, fresh_rows: int = FRESH_ROWS) -> list[dict[str, Any]]:
    """Run one pair; one row per (repeats, alpha)."""
    engine = engine or FLAMLEngine()
    started = time.perf_counter()
    kind = PAIRS[pair.kind]
    data = make_synthetic("binary", N_ROWS, list(kind.problems), seed=pair.generator_seed, fresh_rows=fresh_rows)
    folds, _ = make_session_folds(data.frame[TARGET], TaskType.BINARY, N_SPLITS, max(REPEATS), seed=pair.seed + 1)
    setup_a = ExperimentConfig(target_column=TARGET, task_type=TaskType.BINARY, primary_metric=METRIC)
    setup_b = kind.change(setup_a, data)
    work_dir.mkdir(parents=True, exist_ok=True)
    result_a, assessment_a, fresh_a = _search_and_assess(engine, data, setup_a, folds, pair.seed, n_jobs,
                                                         work_dir / "a", max_iter, time_budget)
    result_b, assessment_b, fresh_b = _search_and_assess(engine, data, setup_b, folds, pair.seed, n_jobs,
                                                         work_dir / "b", max_iter, time_budget)
    true_difference = fresh_b - fresh_a
    common = {
        "pair_id": pair.pair_id, "split": pair.split, "seed": pair.seed, "kind": pair.kind,
        "generator_seed": pair.generator_seed, "simpler": kind.simpler,
        "estimator_a": result_a.best_estimator, "estimator_b": result_b.best_estimator,
        "fresh_f1_a": fresh_a, "fresh_f1_b": fresh_b, "true_difference": true_difference,
        "failed": assessment_a.failed or assessment_b.failed,
        "seconds": round(time.perf_counter() - started, 2),
    }
    rows = []
    for repeats in REPEATS:
        n_folds = N_SPLITS * repeats
        a, b = first_folds(assessment_a, n_folds, f"r{repeats}"), first_folds(assessment_b, n_folds, f"r{repeats}")
        for alpha in ALPHAS:
            row = {**common, "repeats": repeats, "alpha": alpha}
            if not common["failed"]:
                judge = Judge(JudgeSettings(alpha=alpha, min_gain=MIN_GAIN, non_inferiority_margin=MARGIN),
                              test_train_ratio=folds.test_train_ratio)
                decision = judge.compare(a, b, candidate_is_simpler=kind.simpler)
                row.update({"verdict": decision.verdict, "mean_difference": decision.mean_difference,
                            "std_difference": decision.std_difference, "p_better": decision.p_better,
                            "p_worse": decision.p_worse, "wins": decision.wins, "losses": decision.losses})
            rows.append(to_jsonable(row))
    return rows


# --------------------------------------------------------------------------- analysis
def rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """False-keep rate and power for every (repeats, alpha) in ``rows``."""
    out = []
    for repeats in REPEATS:
        for alpha in ALPHAS:
            part = [r for r in rows if r["repeats"] == repeats and r["alpha"] == alpha and not r["failed"]]
            harmful = [r for r in part if r["true_difference"] < HARMFUL_BELOW]
            helpful = [r for r in part if r["true_difference"] > HELPFUL_ABOVE]
            false_keeps = sum(r["verdict"] in ("keep", "keep_simpler") for r in harmful)
            true_keeps = sum(r["verdict"] == "keep" for r in helpful)
            out.append({
                "repeats": repeats, "alpha": alpha, "pairs": len(part),
                "harmful": len(harmful), "false_keeps": false_keeps,
                "false_keep_rate": false_keeps / len(harmful) if harmful else None,
                "helpful": len(helpful), "true_keeps": true_keeps,
                "power": true_keeps / len(helpful) if helpful else None,
                "keeps_any": sum(r["verdict"] in ("keep", "keep_simpler") for r in part),
            })
    return out


def choose(tuning_rates: list[dict[str, Any]]) -> dict[str, Any]:
    """The default setting: highest power with a false-keep rate of at most MAX_FALSE_KEEP (ties: fewer
    repeats, then smaller alpha); if none qualifies, the lowest false-keep rate."""
    def false_keep(r: dict[str, Any]) -> float:
        return r["false_keep_rate"] if r["false_keep_rate"] is not None else 0.0

    allowed = [r for r in tuning_rates if false_keep(r) <= MAX_FALSE_KEEP]
    if allowed:
        return max(allowed, key=lambda r: (r["power"] or 0.0, -r["repeats"], -r["alpha"]))
    return min(tuning_rates, key=lambda r: (false_keep(r), r["repeats"], r["alpha"]))


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tuning = rates([r for r in rows if r["split"] == "tuning"])
    reporting = rates([r for r in rows if r["split"] == "reporting"])
    chosen = choose(tuning)
    reported = next(r for r in reporting if r["repeats"] == chosen["repeats"] and r["alpha"] == chosen["alpha"])
    return {"tuning": tuning, "reporting": reporting, "chosen": chosen, "chosen_on_reporting": reported}
