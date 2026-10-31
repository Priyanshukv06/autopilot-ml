"""Judge agreement study (spec task 27, Requirement 25): does the cheap Judge decide like a strict comparison?

The Judge compares two Setups by refitting the configuration FLAML chose for each on the Session's 15 folds
(decision D2). The strict comparison instead searches again: for each of 5 inner splits of the Session's
Training_Part, a fresh FLAML search for each Setup on the inner training rows, scored on the inner
validation rows. Its 5 differences are judged with the same rules (``Judge.compare``: alpha 0.20, margin
0.005, minimum gain 0, the corrected resampled t-test with 0.2 / 0.8 = 0.25 validation rows per training row).

Fixed on 27 October 2026, before any strict comparison ran:

- Pairs: every step with a Verdict of the 30 Sessions that built the benchmark's frozen memory
  (``research/memory_build.py``: the 10 development datasets, seeds 1 to 3, random order), so the pairs are
  decisions the Judge really made, on development data only. At most 2 pairs per dataset (8 until
  31 October 2026, lowered before any strict comparison ran, to keep the run to about an hour), drawn at
  random with seed 2026 when a dataset has more, so that no dataset dominates; pairs run seed by seed, then
  dataset by dataset and step by step.
- Power: the command line waits before each pair while the laptop is on battery; a pair that ends on battery
  is written with status ``excluded`` (its searches may have been cut short by standby) and runs again.
- Each pair: the Accepted_Setup and the Candidate_Setup as the Session recorded them (columns, Data_Steps,
  class weights, metrics, learners, limits); the Session's Training_Part, rebuilt from its seed and checked
  against the Session's split fingerprint; the outer Test_Part is never read.
- Inner splits: 5 random splits of the Training_Part into 80% training and 20% validation rows (stratified
  for classification), seeds = Session seed + 100 + i. Searches: at most 25 trials with a 60 s cap (as in the
  judge study), the Session seed, all CPU cores, no warm start, MLflow off. A Setup's 5 searches are reused by
  the later pairs of the same Session that compare against it.
- Reported: agreement and Cohen's kappa on the three verdicts and on accepted (keep or keep_simpler) against
  discarded; the cost of each method per pair (strict: the 10 searches' seconds; Judge: the two Assessments'
  seconds from the Session's record) and their ratio.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from app.core.config import Settings, build_settings
from app.services.assessor import Assessment, FoldScore
from app.services.automl_runner import AutoMLEngine, AutoMLRunner, ExperimentConfig
from app.services.autopilot import CODE_ROOT, code_version
from app.services.evaluator import METRICS
from app.services.judge import Judge, JudgeDecision, JudgeSettings
from app.services.splits import OuterSplit, make_outer_split
from app.services.task_detector import TaskType
from research import benchmark, memory_build
from research.datasets import datasets

MAX_PER_DATASET = 2  # 8 until 31 October 2026; lowered before any strict comparison ran, to keep the run short
SELECTION_SEED = 2026
INNER_SPLITS = 5
INNER_TEST_SIZE = 0.2
INNER_SEED_OFFSET = 100
MAX_ITER, TIME_BUDGET = 25, 60
VERDICTS = ("keep", "keep_simpler", "discard")
ACCEPTED = ("keep", "keep_simpler")
SETUP_FIELDS = ("excluded_columns", "estimator_list", "data_steps", "class_weight", "search_metric",
                "search_space_limits")


@dataclass(frozen=True)
class StudyPair:
    pair_id: str  # "<session>/<candidate experiment>"
    dataset: str
    seed: int  # the Session's seed
    session_id: str
    step: int
    split_fingerprint: str  # the Session's outer split
    accepted_experiment: str
    candidate_experiment: str
    accepted_setup: dict[str, Any]
    candidate_setup: dict[str, Any]
    strategy: str
    params: dict[str, Any]
    simpler: bool
    judge: dict[str, Any]  # the Session's decision
    judge_seconds: float  # the two Assessments' seconds


def folders(project_root: Path) -> dict[str, Any]:
    base = build_settings(Path(project_root))
    settings = replace(base, experiments_dir=base.experiments_dir / "studies" / "judge_agreement",
                       models_dir=base.models_dir / "studies" / "judge_agreement")
    results = base.project_root / "results" / "studies"
    return {"settings": settings, "pairs": results / "judge_agreement_pairs.csv",
            "summary": results / "judge_agreement_summary.csv",
            "confusion": results / "judge_agreement_confusion.csv"}


# --------------------------------------------------------------------------- the pairs
def study_pairs(project_root: Path, max_per_dataset: int = MAX_PER_DATASET,
                seed: int = SELECTION_SEED) -> list[StudyPair]:
    """The judged steps of the finished memory-build Sessions, at most ``max_per_dataset`` per dataset."""
    where = memory_build.folders(Path(project_root))
    rows = [r for r in benchmark.latest(benchmark.read_cells(where["rows"])).values()
            if r.get("status") == "completed" and r.get("run_id")]
    order = [e.name for e in datasets("development")]
    by_dataset: dict[str, list[StudyPair]] = {}
    for row in sorted(rows, key=lambda r: (int(r["seed"]), order.index(r["dataset"]))):
        by_dataset.setdefault(row["dataset"], []).extend(
            pairs_of_session(where["settings"].experiments_dir, row["run_id"], row["dataset"], int(row["seed"])))
    rng = np.random.default_rng(seed)
    chosen: list[StudyPair] = []
    for name in order:  # one generator, datasets in the registry's order
        found = by_dataset.get(name, [])
        if len(found) > max_per_dataset:
            keep = set(rng.choice(len(found), size=max_per_dataset, replace=False).tolist())
            found = [p for i, p in enumerate(found) if i in keep]
        chosen.extend(found)
    return sorted(chosen, key=lambda p: (p.seed, order.index(p.dataset), p.session_id, p.step))


def pairs_of_session(experiments_dir: Path, session_id: str, dataset: str, seed: int) -> list[StudyPair]:
    """Every step of a Session with a Verdict, as a pair (the Setup it was compared with, the candidate)."""
    record = json.loads((experiments_dir / session_id / "session.json").read_text(encoding="utf-8"))
    steps = {s["experiment_id"]: s for s in record["steps"] if s.get("experiment_id")}
    pairs = []
    for step in record["steps"]:
        decision = step.get("decision")
        if step["index"] == 0 or not decision or step.get("status") == "failed":
            continue
        parent = steps.get(step.get("parent_experiment_id"))
        if parent is None or not parent.get("assessment"):
            continue
        spec = step["proposal"]["spec"]
        pairs.append(StudyPair(
            pair_id=f"{session_id}/{step['experiment_id']}", dataset=dataset, seed=seed, session_id=session_id,
            step=int(step["index"]), split_fingerprint=record["splits"]["outer"],
            accepted_experiment=parent["experiment_id"], candidate_experiment=step["experiment_id"],
            accepted_setup=_setup(experiments_dir, parent["experiment_id"]),
            candidate_setup=_setup(experiments_dir, step["experiment_id"]),
            strategy=spec["strategy_id"], params=dict(spec.get("params") or {}),
            simpler=step["proposal"].get("complexity") == "simpler",
            judge={k: decision.get(k) for k in ("verdict", "mean_difference", "std_difference", "n_folds",
                                                "p_better", "p_worse", "metric")},
            judge_seconds=float(parent["assessment"]["seconds"]) + float(step["assessment"]["seconds"]),
        ))
    return pairs


def _setup(experiments_dir: Path, experiment_id: str) -> dict[str, Any]:
    return json.loads((experiments_dir / experiment_id / "summary.json").read_text(encoding="utf-8"))["setup"]


# --------------------------------------------------------------------------- the strict comparison
def training_part(dataset: benchmark.Dataset, setup: dict[str, Any], seed: int, settings: Settings,
                  expected_fingerprint: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """The Session's Training_Part (labelled rows, new index 0..n-1) and its target."""
    config = ExperimentConfig(target_column=dataset.target, task_type=dataset.task_type,
                              primary_metric=setup.get("primary_metric"), random_seed=seed,
                              test_size=setup.get("test_size", 0.2), dataset_name=dataset.name)
    _, y, task_type, _, _ = AutoMLRunner(settings=settings, enable_mlflow=False).prepare(dataset.frame, config)
    split, _ = make_outer_split(y, task_type, config.test_size, seed)
    if expected_fingerprint is not None and split.fingerprint() != expected_fingerprint:
        raise ValueError("The rebuilt Training_Part is not the Session's (split fingerprints differ).")
    frame = dataset.frame.loc[y.index].iloc[split.train_positions].reset_index(drop=True)
    return frame, y.iloc[split.train_positions].reset_index(drop=True)


def inner_splits(y_train: pd.Series, task_type: TaskType, seed: int, n: int = INNER_SPLITS) -> list[OuterSplit]:
    return [make_outer_split(y_train, task_type, INNER_TEST_SIZE, seed + INNER_SEED_OFFSET + i)[0] for i in range(n)]


def config_for(setup: dict[str, Any], dataset: benchmark.Dataset, seed: int, n_jobs: int) -> ExperimentConfig:
    return ExperimentConfig(target_column=dataset.target, task_type=dataset.task_type,
                            primary_metric=setup["primary_metric"], time_budget=TIME_BUDGET, random_seed=seed,
                            test_size=INNER_TEST_SIZE, dataset_name=f"{dataset.name}-agreement", n_jobs=n_jobs,
                            max_iter=MAX_ITER, **{k: setup.get(k) if k != "excluded_columns" else
                                                  list(setup.get(k) or []) for k in SETUP_FIELDS})


def search_side(frame: pd.DataFrame, config: ExperimentConfig, splits: Sequence[OuterSplit], settings: Settings,
                engine: AutoMLEngine | None = None) -> list[dict[str, Any]]:
    """One search per inner split, scored on that split's validation rows."""
    runner = AutoMLRunner(settings=settings, engine=engine, enable_mlflow=False)
    out = []
    for split in splits:
        result = runner.run_experiment(frame, config, split=split, compute_importance=False)
        out.append({"experiment_id": result.experiment_id, "seconds": float(result.total_time),
                    "metrics": {k: (None if v is None else float(v)) for k, v in result.test_metrics.items()},
                    "model": result.best_model_name})
    return out


def strict_decision(accepted: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]], metric: str,
                    simpler: bool, splits_key: str) -> JudgeDecision:
    """The Judge's rules on the inner splits: each split is one 'fold' of a stand-in Assessment."""
    def assessment(side: Sequence[dict[str, Any]], name: str) -> Assessment:
        folds = [FoldScore(repeat=0, fold=i, train_metrics={}, validation_metrics=dict(s["metrics"]),
                           class_recall=None, unseen_share={}, baseline_validation={}, fit_seconds=s["seconds"])
                 for i, s in enumerate(side)]
        return Assessment(setup_key=name, experiment_id=name, estimator="search", config={}, primary_metric=metric,
                          folds_fingerprint=splits_key, folds=folds)

    judge = Judge(JudgeSettings(), test_train_ratio=INNER_TEST_SIZE / (1 - INNER_TEST_SIZE))
    return judge.compare(assessment(accepted, "accepted"), assessment(candidate, "candidate"), simpler)


def run_pair(pair: StudyPair, dataset: benchmark.Dataset, settings: Settings, cache: dict[tuple[str, str], Any],
             *, n_jobs: int = -1, engine: AutoMLEngine | None = None) -> dict[str, Any]:
    """The strict comparison of one pair, as one row (status ``failed`` with the error if it could not run)."""
    started_at = _now()
    row: dict[str, Any] = {
        "pair_id": pair.pair_id, "dataset": pair.dataset, "seed": pair.seed, "session_id": pair.session_id,
        "step": pair.step, "strategy": pair.strategy, "params": json.dumps(pair.params, sort_keys=True),
        "simpler": pair.simpler, "metric": pair.judge.get("metric"), "judge_verdict": pair.judge.get("verdict"),
        "judge_mean_difference": pair.judge.get("mean_difference"), "judge_p_better": pair.judge.get("p_better"),
        "judge_p_worse": pair.judge.get("p_worse"), "judge_seconds": pair.judge_seconds,
    }
    try:
        frame, y = training_part(dataset, pair.candidate_setup, pair.seed, settings, pair.split_fingerprint)
        splits = inner_splits(y, dataset.task_type, pair.seed)
        key = hashlib.sha256("".join(s.fingerprint() for s in splits).encode()).hexdigest()
        sides = []
        for name, setup in (("accepted", pair.accepted_setup), ("candidate", pair.candidate_setup)):
            cache_key = (pair.session_id, json.dumps(setup, sort_keys=True))
            if cache_key not in cache:
                cache[cache_key] = search_side(frame, config_for(setup, dataset, pair.seed, n_jobs), splits, settings,
                                               engine)
            sides.append(cache[cache_key])
        decision = strict_decision(sides[0], sides[1], row["metric"], pair.simpler, key)
        strict_seconds = sum(s["seconds"] for side in sides for s in side)
        sign = 1.0 if METRICS[row["metric"]].greater_is_better else -1.0  # positive: the candidate is better
        differences = []
        for a, c in zip(*sides):
            va, vc = a["metrics"].get(row["metric"]), c["metrics"].get(row["metric"])
            differences.append(None if va is None or vc is None else round(sign * (vc - va), 6))
        row.update({
            "status": "completed", "strict_verdict": decision.verdict,
            "strict_mean_difference": decision.mean_difference, "strict_p_better": decision.p_better,
            "strict_p_worse": decision.p_worse, "strict_differences": json.dumps(differences),
            "strict_seconds": round(strict_seconds, 2),
            "cost_ratio": round(strict_seconds / pair.judge_seconds, 3) if pair.judge_seconds > 0 else None,
            "accepted_searches": json.dumps([s["experiment_id"] for s in sides[0]]),
            "candidate_searches": json.dumps([s["experiment_id"] for s in sides[1]]), "error": None,
        })
    except Exception as exc:  # recorded, so the study goes on
        row.update({"status": "failed", "strict_verdict": None, "error": f"{type(exc).__name__}: {exc}"})
    version = code_version(CODE_ROOT)
    row.update({"git_commit": (version.get("git_commit") or "")[:7] or None, "git_dirty": version.get("git_dirty"),
                "started_at": started_at, "finished_at": _now()})
    return {column: row.get(column) for column in COLUMNS}


COLUMNS = ("pair_id", "dataset", "seed", "session_id", "step", "strategy", "params", "simpler", "metric",
           "judge_verdict", "judge_mean_difference", "judge_p_better", "judge_p_worse", "judge_seconds", "status",
           "strict_verdict", "strict_mean_difference", "strict_p_better", "strict_p_worse", "strict_differences",
           "strict_seconds", "cost_ratio", "accepted_searches", "candidate_searches", "error", "git_commit",
           "git_dirty", "started_at", "finished_at")


def run(pairs: Sequence[StudyPair], settings: Settings, rows_path: Path, *, n_jobs: int = -1,
        load: Callable[[str], benchmark.Dataset] = benchmark.load_dataset, engine: AutoMLEngine | None = None,
        limit: int = 0, report: Callable[[dict[str, Any], int, int], None] | None = None,
        power: Callable[[], bool | None] | None = None, sleep: Callable[[float], None] = time.sleep,
        on_wait: Callable[[], None] | None = None) -> list[dict[str, Any]]:
    """Run the pairs that have no usable row yet, one row each, appended as each finishes. With ``power``
    (for example ``benchmark.on_battery``) each pair waits while the laptop is on battery, and a pair that ends
    on battery is written as ``excluded``, its reused searches are dropped, and it runs again next time."""
    seen = {r["pair_id"] for r in read_rows(rows_path) if r.get("status") != "excluded"}
    todo = []
    for pair in pairs:  # each pair once, also when it is listed twice
        if pair.pair_id not in seen:
            seen.add(pair.pair_id)
            todo.append(pair)
    todo = todo[: limit or None]
    settings.experiments_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, benchmark.Dataset] = {}
    cache: dict[tuple[str, str], Any] = {}
    written = []
    for number, pair in enumerate(todo, start=1):
        if pair.dataset not in loaded:
            loaded[pair.dataset] = load(pair.dataset)
        if power is not None:
            benchmark.wait_for_power(power, sleep, on_wait)
        row = run_pair(pair, loaded[pair.dataset], settings, cache, n_jobs=n_jobs, engine=engine)
        if power is not None and power() is True:
            row.update({"status": "excluded", "error": "the laptop was on battery at the end of the pair"})
            cache.clear()  # searches made during this pair may have been cut short
        _append(rows_path, row)
        written.append(row)
        if report is not None:
            report(row, number, len(todo))
    return written


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path, dtype={c: str for c in ("pair_id", "session_id", "git_commit", "params")})
    return [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()} for r in frame.to_dict("records")]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    text = pd.DataFrame([row], columns=list(COLUMNS)).to_csv(index=False, header=header, lineterminator="\n")
    with path.open("a", encoding="utf-8", newline="") as file:
        file.write(text)


# --------------------------------------------------------------------------- agreement
def cohen_kappa(first: Sequence[str], second: Sequence[str], labels: Sequence[str]) -> float | None:
    """Cohen's kappa = (p_o - p_e) / (1 - p_e); None when chance agreement is 1 (one label only)."""
    n = len(first)
    if n == 0 or n != len(second):
        raise ValueError("Need two equally long, non-empty lists of labels.")
    observed = sum(a == b for a, b in zip(first, second)) / n
    expected = sum((list(first).count(label) / n) * (list(second).count(label) / n) for label in labels)
    return None if expected >= 1 else (observed - expected) / (1 - expected)


def agreement(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Agreement and Cohen's kappa on the verdicts and on accepted against discarded, the confusion table,
    and the costs, over the completed pairs."""
    rows = list(rows)
    done = [r for r in rows if r.get("status") == "completed" and r.get("strict_verdict")]
    judge = [r["judge_verdict"] for r in done]
    strict = [r["strict_verdict"] for r in done]
    accepted = lambda verdicts: ["accepted" if v in ACCEPTED else "discarded" for v in verdicts]  # noqa: E731
    confusion = pd.DataFrame(0, index=pd.Index(VERDICTS, name="judge"), columns=pd.Index(VERDICTS, name="strict"))
    for a, b in zip(judge, strict):
        confusion.loc[a, b] += 1
    judge_s = sum(float(r["judge_seconds"]) for r in done)
    strict_s = sum(float(r["strict_seconds"]) for r in done)
    ratios = [float(r["cost_ratio"]) for r in done if r.get("cost_ratio") is not None]
    n = len(done)
    return {
        "pairs": n, "failed": sum(1 for r in rows if r.get("status") == "failed"),
        "excluded": sum(1 for r in rows if r.get("status") == "excluded"),
        "agreement": sum(a == b for a, b in zip(judge, strict)) / n if n else None,
        "kappa": cohen_kappa(judge, strict, VERDICTS) if n else None,
        "accepted_agreement": sum(a == b for a, b in zip(accepted(judge), accepted(strict))) / n if n else None,
        "accepted_kappa": cohen_kappa(accepted(judge), accepted(strict), ("accepted", "discarded")) if n else None,
        "judge_seconds": round(judge_s, 1), "strict_seconds": round(strict_s, 1),
        "cost_ratio_total": round(strict_s / judge_s, 2) if judge_s else None,
        "cost_ratio_median": float(np.median(ratios)) if ratios else None, "confusion": confusion,
    }


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


__all__ = ["COLUMNS", "StudyPair", "agreement", "cohen_kappa", "config_for", "folders", "inner_splits",
           "pairs_of_session", "read_rows", "run", "run_pair", "strict_decision", "study_pairs", "training_part"]
