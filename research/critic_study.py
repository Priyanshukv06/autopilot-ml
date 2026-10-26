"""Critic study (spec task 16, Requirement 6): how often is the Critic right?

Synthetic binary datasets (``research/synthetic.py``) with one planted problem each, at three
strengths, plus clean datasets. For each dataset: one short FLAML search on all its rows, an
Assessment of the best configuration on 3 x 5 folds, the profile and the Critic. The truth comes
from the planted problem (data-level kinds) or from the fresh sample of 20,000 rows that the
generator makes alongside each dataset (model-level kinds).

Decisions, fixed before running (design.md, task 16 notes):

- Planted problems and strengths: see ``STRENGTHS``; 3 clean datasets per seed. Binary task,
  1,000 rows per dataset. Tuning seeds 0 to 4, reporting seeds 100 to 104, so the reported
  precision and recall come from datasets the tuning never saw.
- Search: FLAML's default learners, F1, at most 25 trials (``max_iter``) and 60 s, all CPU cores.
- Truth for data-level kinds: the planted problem (class imbalance, identifier column, constant
  column, high cardinality, missing values; duplicated columns for redundant_features).
- Truth for model-level kinds, from the model FLAML trained on all rows, scored on those rows and
  on the fresh sample (F1): overfitting if training minus fresh F1 is at least 0.10; underfitting
  if that gap is below 0.03 and the fresh skill over the trivial baseline is below 0.50;
  unreliable_validation if the folds' mean validation F1 is 0.05 or more away from the fresh F1.
- The Critic is run twice per dataset: with permissive "probe" thresholds, so that every rule
  reports its evidence numbers, and its inputs are saved, so that any thresholds can be applied
  afterwards (``predict``) and the real Critic can be replayed on them.
- The budget kinds (budget_limited, wasted_budget, weak_model_families) have no truth in
  Requirement 6.3 and are not measured.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from app.services.assessor import Assessment, Assessor
from app.services.automl_runner import AutoMLEngine, ExperimentConfig, FLAMLEngine, to_jsonable
from app.services.critic import Critic, CriticThresholds, DiagnosisKind
from app.services.evaluator import Evaluator
from app.services.profiler import DatasetProfile, DatasetProfiler
from app.services.search_summary import SearchSummary, summarize_search
from app.services.splits import make_session_folds
from app.services.task_detector import TaskType
from research.synthetic import TARGET, make_synthetic

# Strengths (design choices): the weakest level of each data-level problem is still a real problem,
# because every planted problem counts as one that the Critic should find.
STRENGTHS: dict[str, tuple[float, ...]] = {
    "class_imbalance": (3.0, 5.0, 10.0),  # largest / smallest class
    "label_noise": (0.1, 0.2, 0.3),  # share of changed labels (no Critic kind; affects the model-level truth)
    "useless_features": (5.0, 20.0, 50.0),  # random columns (no Critic kind)
    "identifier_column": (0.95, 0.99, 1.0),  # share of distinct IDs
    "constant_column": (0.98, 0.995, 1.0),  # share of the main value
    "high_cardinality": (0.2, 0.5, 2.0),  # categories per row
    "missing_values": (0.3, 0.5, 0.8),  # share of empty cells in 2 columns
    "duplicated_columns": (0.95, 0.98, 0.995),  # correlation of 2 noisy copies with their sources
}
# The starting thresholds (version v0-start, design.md section 6). Tuning always starts from these,
# so the analysis gives the same result after the tuned version has been written to the file.
V0_START: dict[str, float] = {
    "overfitting.moderate_gap": 0.10, "overfitting.high_gap": 0.20,
    "underfitting.max_gap": 0.03, "underfitting.moderate_skill": 0.50, "underfitting.high_skill": 0.25,
    "budget_limited.moderate_gain": 0.005, "budget_limited.high_gain": 0.02,
    "wasted_budget.moderate_share": 0.50, "wasted_budget.high_share": 0.75,
    "class_imbalance.moderate_ratio": 3.0, "class_imbalance.high_ratio": 9.0,
    "class_imbalance.moderate_recall": 0.5, "class_imbalance.high_recall": 0.3,
    "class_imbalance.min_ratio_for_recall": 1.5,
    "missing_values.moderate_share": 0.40, "missing_values.high_share": 0.70,
    "high_cardinality.moderate_distinct": 50.0, "high_cardinality.moderate_unseen": 0.30,
    "high_cardinality.high_unseen": 0.60,
    "constant_column.near_constant_share": 0.99,
    "redundant_features.moderate_correlation": 0.95, "redundant_features.high_correlation": 0.99,
    "unreliable_validation.moderate_std": 0.05, "unreliable_validation.high_std": 0.08,
    "weak_model_families.min_gap": 0.10, "weak_model_families.min_time_share": 0.05,
}
N_CLEAN = 3
TUNING_SEEDS = (0, 1, 2, 3, 4)
REPORTING_SEEDS = (100, 101, 102, 103, 104)
N_ROWS, FRESH_ROWS = 1000, 20_000
MAX_ITER, TIME_BUDGET = 25, 60
METRIC = "f1"

# Planted problem -> the Critic kind that should find it.
DATA_KINDS = {
    "class_imbalance": "class_imbalance",
    "identifier_column": "identifier_column",
    "constant_column": "constant_column",
    "high_cardinality": "high_cardinality",
    "missing_values": "missing_values",
    "duplicated_columns": "redundant_features",
}
MODEL_KINDS = ("overfitting", "underfitting", "unreliable_validation")
STUDIED_KINDS = (*MODEL_KINDS, *DATA_KINDS.values())
TRUE_OVERFIT_GAP, TRUE_UNDERFIT_GAP, TRUE_UNDERFIT_SKILL, TRUE_UNRELIABLE_ERROR = 0.10, 0.03, 0.50, 0.05

# Thresholds at which every rule fires, so that each Diagnosis reports its evidence numbers.
PROBE_THRESHOLDS = CriticThresholds.from_mapping({
    "version": "probe",
    "overfitting": {"moderate_gap": -10, "high_gap": 10},
    "underfitting": {"max_gap": 10, "moderate_skill": 10, "high_skill": -10},
    "budget_limited": {"moderate_gain": -10, "high_gain": 10},
    "wasted_budget": {"moderate_share": -10, "high_share": 10},
    "class_imbalance": {"moderate_ratio": 0, "high_ratio": 100, "moderate_recall": 2, "high_recall": -1,
                        "min_ratio_for_recall": 0},
    "missing_values": {"moderate_share": -1, "high_share": 2},
    "high_cardinality": {"moderate_distinct": -1, "moderate_unseen": -1, "high_unseen": 2},
    "constant_column": {"near_constant_share": 0},
    "redundant_features": {"moderate_correlation": 0, "high_correlation": 2},
    "unreliable_validation": {"moderate_std": -1, "high_std": 10},
    "weak_model_families": {"min_gap": -10, "min_time_share": -1},
})


@dataclass(frozen=True)
class StudyCase:
    seed: int  # study seed (tuning or reporting)
    problem: str  # a planted problem kind, or "clean"
    strength: float | None
    replicate: int  # numbers the clean datasets of one seed
    generator_seed: int

    @property
    def case_id(self) -> str:
        strength = "" if self.strength is None else f"{self.strength:g}"
        return f"s{self.seed}-{self.problem}{strength}-r{self.replicate}"

    @property
    def split(self) -> str:
        return "tuning" if self.seed in TUNING_SEEDS else "reporting"


def study_cases(seeds: Iterable[int]) -> list[StudyCase]:
    cases = []
    for seed in seeds:
        index = 0
        for problem, strengths in STRENGTHS.items():
            for strength in strengths:
                cases.append(StudyCase(seed, problem, strength, 0, seed * 1000 + index))
                index += 1
        for replicate in range(N_CLEAN):
            cases.append(StudyCase(seed, "clean", None, replicate, seed * 1000 + index))
            index += 1
    return cases


def _probe_evidence(profile: DatasetProfile, assessment: Assessment, search: SearchSummary | None) -> dict[str, Any]:
    """Every studied rule's evidence numbers, from the Critic itself run with the probe thresholds."""
    diagnoses = Critic(PROBE_THRESHOLDS).diagnose(
        profile=profile, config=ExperimentConfig(target_column=TARGET), assessment=assessment, search=search,
        task_type=TaskType.BINARY, primary_metric=METRIC)
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for d in diagnoses:
        by_kind.setdefault(d.kind.value, []).append(dict(d.evidence))

    def first(kind: str, key: str) -> float | None:
        values = [e[key] for e in by_kind.get(kind, []) if key in e]
        return float(values[0]) if values else None

    def largest(kind: str, key: str, absolute: bool = False) -> float | None:
        values = [abs(float(e[key])) if absolute else float(e[key]) for e in by_kind.get(kind, []) if key in e]
        return max(values) if values else None

    return {
        "ev_gap": first("overfitting", "gap"),
        "ev_skill": first("underfitting", "skill"),
        "ev_std": first("unreliable_validation", "std"),
        "ev_imbalance_ratio": first("class_imbalance", "imbalance_ratio"),
        "ev_minority_recall": first("class_imbalance", "minority_recall"),
        "ev_max_missing_share": largest("missing_values", "missing_share"),
        "ev_max_distinct": largest("high_cardinality", "distinct_values"),
        "ev_max_unseen": largest("high_cardinality", "unseen_share"),
        "ev_max_main_share": largest("constant_column", "main_value_share"),
        "ev_single_value": any(float(e.get("distinct_values", 2)) <= 1 for e in by_kind.get("constant_column", [])),
        "ev_max_correlation": largest("redundant_features", "correlation", absolute=True),
        "ev_identifier": bool(by_kind.get("identifier_column")),
    }


def run_case(case: StudyCase, *, work_dir: Path, engine: AutoMLEngine | None = None, max_iter: int = MAX_ITER,
             time_budget: int = TIME_BUDGET, n_jobs: int = -1, fresh_rows: int = FRESH_ROWS) -> tuple[dict, dict]:
    """Run one dataset. Returns (row for the results CSV, saved Critic inputs)."""
    engine = engine or FLAMLEngine()
    started = time.perf_counter()
    problems = [] if case.problem == "clean" else [(case.problem, case.strength)]
    data = make_synthetic("binary", N_ROWS, problems, seed=case.generator_seed, fresh_rows=fresh_rows)
    X, y = data.frame.drop(columns=TARGET), data.frame[TARGET]
    work_dir.mkdir(parents=True, exist_ok=True)
    result = engine.fit(X, y, task_type=TaskType.BINARY, metric=METRIC, time_budget=time_budget, seed=case.seed,
                        n_jobs=n_jobs, estimator_list=None, work_dir=work_dir, max_iter=max_iter)
    folds, _ = make_session_folds(y, TaskType.BINARY, seed=case.seed + 1)
    stand_in = SimpleNamespace(setup_key=case.case_id, experiment_id=case.case_id,
                               best_estimator=result.best_estimator, best_config=result.best_config)
    assessment = Assessor(engine, folds, TaskType.BINARY, METRIC, n_jobs, case.seed).assess(
        X, y, ExperimentConfig(target_column=TARGET), stand_in)
    profile = DatasetProfiler().profile(data.frame, TARGET, data.name, TaskType.BINARY)
    search = summarize_search(result.search_history, METRIC) if result.search_history else None

    # The truth for the model-level kinds, from the fresh sample.
    evaluator = Evaluator(TaskType.BINARY, METRIC)
    fresh_X, fresh_y = data.fresh.drop(columns=TARGET), data.fresh[TARGET]
    train_f1 = evaluator.evaluate(result.model, X, y, result.classes).primary_score
    fresh_f1 = evaluator.evaluate(result.model, fresh_X, fresh_y, result.classes).primary_score
    baseline_f1 = evaluator.baseline_metrics(y, fresh_y)[METRIC]
    validation = assessment.values(METRIC, "validation")
    cv_f1 = float(np.nanmean(validation)) if np.isfinite(validation).any() else None
    true_gap = train_f1 - fresh_f1
    fresh_skill = (fresh_f1 - baseline_f1) / (1 - baseline_f1) if baseline_f1 < 1 else None

    row: dict[str, Any] = {
        "case_id": case.case_id, "split": case.split, "seed": case.seed, "problem": case.problem,
        "strength": case.strength, "generator_seed": case.generator_seed, "dataset": data.name,
        "best_estimator": result.best_estimator, "n_trials": len(result.search_history),
        "train_f1": train_f1, "fresh_f1": fresh_f1, "baseline_fresh_f1": baseline_f1, "cv_f1": cv_f1,
        "true_gap": true_gap, "fresh_skill": fresh_skill,
        "assessment_failed": assessment.failed, "seconds": round(time.perf_counter() - started, 2),
    }
    if not assessment.failed:
        row.update(_probe_evidence(profile, assessment, search))
    for kind in STUDIED_KINDS:
        row[f"truth_{kind}"] = truth(case.problem, kind, row)
    inputs = {"case_id": case.case_id, "profile": to_jsonable(profile.to_dict()), "assessment": assessment.to_dict(),
              "search": search.to_dict() if search else None}
    return to_jsonable(row), inputs


def truth(problem: str, kind: str, row: dict[str, Any]) -> bool:
    """Whether a Diagnosis of this kind is correct for this dataset (definitions in the module docstring)."""
    if kind in DATA_KINDS.values():
        return DATA_KINDS.get(problem) == kind
    if kind == "overfitting":
        return row["true_gap"] >= TRUE_OVERFIT_GAP
    if kind == "underfitting":
        return (row["true_gap"] < TRUE_UNDERFIT_GAP and row["fresh_skill"] is not None
                and row["fresh_skill"] < TRUE_UNDERFIT_SKILL)
    if kind == "unreliable_validation":
        return row["cv_f1"] is not None and abs(row["cv_f1"] - row["fresh_f1"]) >= TRUE_UNRELIABLE_ERROR
    raise ValueError(f"Kind '{kind}' is not studied.")


# --------------------------------------------------------------------------- applying thresholds
def _is(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def predict(row: dict[str, Any], kind: str, t: dict[str, float]) -> bool:
    """The moderate-level detection rule of the Critic for this kind, applied to the probe evidence.

    ``t`` maps "section.name" to a threshold, as in ``CriticThresholds.values``. ``replay`` checks that
    this gives the same answers as the real Critic.
    """
    if row.get("assessment_failed"):
        return False
    if kind == "overfitting":
        return _is(row["ev_gap"]) and row["ev_gap"] >= t["overfitting.moderate_gap"]
    if kind == "underfitting":
        return (_is(row["ev_skill"]) and _is(row["ev_gap"]) and row["ev_gap"] < t["underfitting.max_gap"]
                and row["ev_skill"] < t["underfitting.moderate_skill"])
    if kind == "unreliable_validation":
        return _is(row["ev_std"]) and row["ev_std"] >= t["unreliable_validation.moderate_std"]
    if kind == "class_imbalance":
        ratio, recall = row["ev_imbalance_ratio"], row["ev_minority_recall"]
        if not _is(ratio):
            return False
        by_recall = (_is(recall) and ratio >= t["class_imbalance.min_ratio_for_recall"]
                     and recall < t["class_imbalance.moderate_recall"])
        return ratio >= t["class_imbalance.moderate_ratio"] or by_recall
    if kind == "missing_values":
        return _is(row["ev_max_missing_share"]) and row["ev_max_missing_share"] >= t["missing_values.moderate_share"]
    if kind == "high_cardinality":
        return ((_is(row["ev_max_distinct"]) and row["ev_max_distinct"] > t["high_cardinality.moderate_distinct"])
                or (_is(row["ev_max_unseen"]) and row["ev_max_unseen"] >= t["high_cardinality.moderate_unseen"]))
    if kind == "constant_column":
        return bool(row.get("ev_single_value")) or (
            _is(row["ev_max_main_share"]) and row["ev_max_main_share"] >= t["constant_column.near_constant_share"])
    if kind == "redundant_features":
        return (_is(row["ev_max_correlation"])
                and row["ev_max_correlation"] >= t["redundant_features.moderate_correlation"])
    if kind == "identifier_column":
        return bool(row["ev_identifier"])
    raise ValueError(f"Kind '{kind}' is not studied.")


def scores(rows: list[dict[str, Any]], kind: str, t: dict[str, float]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        predicted, actual = predict(row, kind, t), bool(row[f"truth_{kind}"])
        tp += predicted and actual
        fp += predicted and not actual
        fn += actual and not predicted
        tn += not predicted and not actual
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"kind": kind, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


# Detection thresholds that the study tunes, with the grid searched for each.
TUNED: dict[str, tuple[str, np.ndarray]] = {
    "overfitting": ("overfitting.moderate_gap", np.round(np.arange(0.02, 0.301, 0.01), 3)),
    "underfitting": ("underfitting.moderate_skill", np.round(np.arange(0.20, 0.901, 0.05), 3)),
    "unreliable_validation": ("unreliable_validation.moderate_std", np.round(np.arange(0.01, 0.1001, 0.005), 4)),
    "class_imbalance": ("class_imbalance.moderate_ratio", np.round(np.arange(1.5, 10.01, 0.5), 2)),
    "missing_values": ("missing_values.moderate_share", np.round(np.arange(0.05, 0.901, 0.05), 3)),
    "high_cardinality": ("high_cardinality.moderate_unseen", np.round(np.arange(0.05, 0.801, 0.05), 3)),
    "constant_column": ("constant_column.near_constant_share", np.round(np.arange(0.80, 1.0001, 0.01), 3)),
    "redundant_features": ("redundant_features.moderate_correlation", np.round(np.arange(0.90, 0.9951, 0.005), 4)),
}
# For these, a lower value means a stronger finding, so the "high" threshold sits below the moderate one.
_LOWER_IS_STRONGER = {"underfitting.moderate_skill": "underfitting.high_skill"}
_HIGHER_PAIRS = {
    "overfitting.moderate_gap": "overfitting.high_gap",
    "unreliable_validation.moderate_std": "unreliable_validation.high_std",
    "class_imbalance.moderate_ratio": "class_imbalance.high_ratio",
    "missing_values.moderate_share": "missing_values.high_share",
    "high_cardinality.moderate_unseen": "high_cardinality.high_unseen",
    "redundant_features.moderate_correlation": "redundant_features.high_correlation",
}


def tune(rows: list[dict[str, Any]], start: CriticThresholds) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """For each tuned kind, the grid value with the best F1 on ``rows`` (ties: the value nearest the
    starting one). Returns the full set of thresholds and every grid point's scores."""
    values = dict(start.values)
    grid_rows = []
    for kind, (key, grid) in TUNED.items():
        best = None
        for value in grid:
            trial = {**values, key: float(value)}
            result = {"threshold": key, "value": float(value), **scores(rows, kind, trial)}
            grid_rows.append(result)
            rank = (result["f1"], -abs(float(value) - start.values[key]))
            if best is None or rank > best[0]:
                best = (rank, float(value))
        values[key] = best[1]
        if key in _HIGHER_PAIRS and values[_HIGHER_PAIRS[key]] < values[key]:
            values[_HIGHER_PAIRS[key]] = values[key]  # keep "high" at least as strict as "moderate"
        if key in _LOWER_IS_STRONGER and values[_LOWER_IS_STRONGER[key]] > values[key]:
            values[_LOWER_IS_STRONGER[key]] = values[key]
    return values, grid_rows


def replay(inputs: dict[str, Any], thresholds: CriticThresholds) -> set[str]:
    """The kinds the real Critic finds on saved inputs with the given thresholds."""
    profile = DatasetProfile.from_dict(inputs["profile"])
    assessment = Assessment.from_dict(inputs["assessment"])
    search = SearchSummary(**inputs["search"]) if inputs["search"] else None
    found = Critic(thresholds).diagnose(profile=profile, config=ExperimentConfig(target_column=TARGET),
                                        assessment=assessment, search=search, task_type=TaskType.BINARY,
                                        primary_metric=METRIC)
    return {d.kind.value for d in found if d.kind.value in STUDIED_KINDS}


def analyse(rows: list[dict[str, Any]], start: CriticThresholds) -> dict[str, Any]:
    """Tune on the tuning rows; score the starting and the tuned thresholds on the reporting rows."""
    tuning = [r for r in rows if r["split"] == "tuning"]
    reporting = [r for r in rows if r["split"] == "reporting"]
    tuned, grid = tune(tuning, start)
    summary = []
    for kind in STUDIED_KINDS:
        for label, values in (("start", start.values), ("tuned", tuned)):
            for split, part in (("tuning", tuning), ("reporting", reporting)):
                summary.append({"thresholds": label, "split": split, **scores(part, kind, values)})
    return {"tuned": tuned, "grid": grid, "summary": summary}


_SECTION_NOTES = {
    "overfitting": "gap = mean over the folds of (training score - validation score)",
    "underfitting": "Only when gap < max_gap. skill = (validation - baseline) / (1 - baseline), means over the folds",
    "budget_limited": "Improvement of the search's best score in the last quarter of the search time (bounded metrics only)",
    "wasted_budget": "Share of the search time after the best configuration was found",
    "class_imbalance": "ratio = largest / smallest class; recall = mean validation recall of the smallest class",
    "missing_values": "Share of empty cells of a used column in the Training_Part",
    "high_cardinality": "distinct = values of a used text column (more than this is moderate); unseen = mean share of "
                        "validation rows with a value not seen in the fold's training rows",
    "constant_column": "A used column with one value, or with the same value in at least this share of rows (low)",
    "redundant_features": "Absolute Pearson correlation of two used numeric columns in the Training_Part",
    "unreliable_validation": "Standard deviation of the validation score over the folds",
    "weak_model_families": "A family at least min_gap behind the best family that used at least min_time_share "
                           "of the search time",
}


def thresholds_toml(values: dict[str, float], version: str, note: str) -> str:
    """The text of a thresholds file with these values (sections in the Critic's table order)."""
    lines = ["# Thresholds of the Critic (app/services/critic.py), read with Python's tomllib.",
             *(f"# {line}" for line in note.splitlines()),
             "# Gaps, skills, gains and spreads are in units of the diagnostic metric (F1, accuracy, ROC-AUC or R2).",
             f'version = "{version}"']
    for section, comment in _SECTION_NOTES.items():
        lines += ["", f"[{section}]", f"# {comment}"]
        for key in sorted(k for k in values if k.startswith(section + ".")):
            value = values[key]
            text = f"{value:g}" if float(value).is_integer() and "distinct" in key else repr(float(value))
            lines.append(f"{key.split('.', 1)[1]} = {text}")
    return "\n".join(lines) + "\n"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
            for row in frame.to_dict("records")]


def read_inputs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    inputs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            inputs[data["case_id"]] = data
    return inputs


KIND_OF = {kind.value: kind for kind in DiagnosisKind}
