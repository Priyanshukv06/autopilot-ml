"""Tests for the autonomous Session (task 11.1 of the spec).

The engine and the Assessor are recording stand-ins and time comes from a fake clock, so a whole
Session runs in about a second with no FLAML search. The stand-in engine moves the clock on by
the search budget it is given; the stand-in Assessor moves it on by 2 seconds and returns scripted
fold scores, so each test chooses the Judge's verdicts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression

from app.core.config import build_settings
from app.services.assessor import Assessment, FoldScore
from app.services.automl_runner import EngineResult, ExperimentConfig, ExperimentError
from app.services.autopilot import AutopilotSession, SessionSettings
from app.services.critic import Critic, CriticThresholds
from app.services.judge import Judge, JudgeSettings
from app.services.profiler import DatasetProfile
from app.services.resource_manager import SessionBudget, StopReason
from app.services.search_summary import SearchSummary
from app.services.splits import make_outer_split
from app.services.task_detector import TaskType
from research.synthetic import make_synthetic

# 300 rows: 8 numeric features, a text column 'code' with about 600 levels (high cardinality) and a 0/1 target
# with 4 rows of class 0 for each row of class 1. The first Setup gets three Proposals: drop 'code', smaller
# trees and class weights.
DATA = make_synthetic("binary", 300, [("class_imbalance", 4), ("high_cardinality", 2)], seed=0, fresh_rows=10).frame
SPLIT, _ = make_outer_split(DATA["target"], TaskType.BINARY, 0.2, 42)
TRAIN_ROWS, TEST_ROWS = set(DATA.index[SPLIT.train_positions]), set(DATA.index[SPLIT.test_positions])
BASE_SCORES = 0.70 + 0.01 * np.sin(np.arange(15))  # validation F1 of the first Setup on the 15 folds


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class NumericModel:
    """Logistic regression on the numeric columns (picklable, so the runner can save it)."""

    def __init__(self, X, y):
        self.columns = [c for c in X.columns if is_numeric_dtype(X[c])]
        self.model = LogisticRegression(max_iter=1000).fit(X[self.columns], y)
        self.classes_ = self.model.classes_

    def fit(self, X, y):  # permutation importance needs an object with fit
        return self

    def predict(self, X):
        return self.model.predict(X[self.columns])

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.columns])


class RecordingEngine:
    """Uses its whole time budget, or at most ``search_cap`` seconds (a search that converged early)."""

    name = "recording"

    def __init__(self, clock, fail_on=(), search_cap=None):
        self.clock, self.fail_on, self.search_cap, self.fits = clock, set(fail_on), search_cap, []

    def fit(self, X_train, y_train, **kwargs):
        call = len(self.fits)
        self.fits.append({"index": set(X_train.index), "columns": list(X_train.columns), **kwargs})
        budget = kwargs["time_budget"]
        self.clock.advance(budget if self.search_cap is None else min(budget, self.search_cap))
        if call in self.fail_on:
            raise RuntimeError(f"search {call} exploded")
        history = [{"trial": 0, "estimator": "lgbm", "model_name": "LightGBM", "validation_score": 0.7,
                    "trial_time": 1.0, "wall_clock_time": float(kwargs["time_budget"]), "sample_size": len(X_train),
                    "config": {"num_leaves": 8}}]
        model = NumericModel(X_train, y_train)
        return EngineResult(model=model, engine_name=self.name, engine_version="test", best_estimator="lgbm",
                            best_model_name="LightGBM", best_config={"num_leaves": 8}, best_validation_score=0.7,
                            validation_method="cv", training_time=0.0, search_history=history, leaderboard=[],
                            native_importance=None, classes=list(model.classes_),
                            best_config_per_estimator={"lgbm": {"num_leaves": 8}})


class ScriptedAssessor:
    """Records the rows it is given and returns fold scores shifted by ``shifts[call]``."""

    def __init__(self, engine, folds, task_type, metric, n_jobs, seed, *, clock, shifts, gap, fail_on):
        self.folds, self.primary_metric = folds, metric
        self.clock, self.shifts, self.gap, self.fail_on = clock, shifts, gap, set(fail_on)
        self.calls = []

    def assess(self, X_train, y_train, config, result):
        call = len(self.calls)
        self.calls.append({"index": set(X_train.index), "labels": set(y_train.index), "columns": list(X_train.columns),
                           "experiment_id": result.experiment_id})
        self.clock.advance(2.0)
        if call in self.fail_on:
            raise RuntimeError(f"assessment {call} exploded")
        validation = BASE_SCORES + np.asarray(self.shifts.get(call, 0.0))  # one shift, or one per fold
        text_columns = [c for c in X_train.columns if not is_numeric_dtype(X_train[c])]
        folds = [
            FoldScore(index // 5, index % 5, {"f1": v + self.gap}, {"f1": v}, {"0": 0.8, "1": 0.8},
                      {c: 0.7 for c in text_columns}, {"f1": 0.0}, 0.1)
            for index, v in enumerate(validation)
        ]
        return Assessment(result.setup_key, result.experiment_id, result.best_estimator, dict(result.best_config),
                          self.primary_metric, self.folds.fingerprint(), folds, 2.0)


def run_session(tmp_path, *, total=1000, experiments=6, patience=3, shifts=None, gap=0.15, engine_fails=(),
                assessor_fails=(), stop=None, data=DATA, base=None, search_cap=None):
    clock = FakeClock()
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    engine = RecordingEngine(clock, engine_fails, search_cap)
    assessors = []

    def factory(*args):
        assessors.append(ScriptedAssessor(*args, clock=clock, shifts=shifts or {}, gap=gap, fail_on=assessor_fails))
        return assessors[-1]

    session = AutopilotSession(settings=settings, engine=engine, clock=clock, enable_mlflow=False,
                               assessor_factory=factory)
    plan = SessionSettings(
        base=base or ExperimentConfig(target_column="target", task_type=TaskType.BINARY, primary_metric="f1",
                                      dataset_name="synthetic"),
        budget=SessionBudget(total_seconds=total, max_experiments=experiments, patience=patience),
        n_repeats=3,  # the scripted Assessor scores 15 folds (BASE_SCORES)
    )
    result = session.run(data, plan, stop_requested=stop)
    record = json.loads((settings.experiments_dir / result.session_id / "session.json").read_text(encoding="utf-8"))
    return result, record, engine, (assessors[0] if assessors else None), settings


def verdicts(record):
    return [(s["status"], (s["decision"] or {}).get("verdict")) for s in record["steps"]]


# --------------------------------------------------------------------------- Information_Barrier
def test_no_test_row_reaches_the_engine_or_the_assessor(tmp_path, monkeypatch):
    seen_profiles = []
    original = Critic.diagnose

    def spy(self, **kwargs):
        seen_profiles.append(kwargs["profile"].rows)
        return original(self, **kwargs)

    monkeypatch.setattr(Critic, "diagnose", spy)
    result, record, engine, assessor, settings = run_session(tmp_path, shifts={1: 0.0, 2: 0.0, 3: -0.05})

    assert record["splits"]["outer"] == SPLIT.fingerprint()  # the split these tests compare with
    assert len(engine.fits) >= 3 and len(assessor.calls) == len(engine.fits)
    for fit in engine.fits:
        assert fit["index"] <= TRAIN_ROWS and not fit["index"] & TEST_ROWS
    for call in assessor.calls:
        assert call["index"] == call["labels"] == TRAIN_ROWS  # the whole Training_Part, nothing else
    assert seen_profiles and set(seen_profiles) == {len(TRAIN_ROWS)}  # the Critic sees the Training_Part profile
    train_profile = json.loads((settings.experiments_dir / result.session_id / "train_profile.json").read_text("utf-8"))
    assert train_profile["rows"] == len(TRAIN_ROWS)

    # No Experiment of the Session has any Test_Part number: their Decision_Views and records are clean.
    for step in record["steps"]:
        folder = settings.experiments_dir / step["experiment_id"]
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        assert summary["test_evaluated"] is False and summary["session_id"] == result.session_id
        assert summary["test_score"] is None and summary["test_metrics"] == {} and summary["baseline_metrics"] == {}
        assert summary["feature_importance"] == [] and not (folder / "predictions.csv").exists()
    # The Test_Part is used once, at the end.
    assert record["final_evaluation"]["final"]["test_score"] is not None


def test_a_session_records_every_step(tmp_path):
    result, record, engine, assessor, settings = run_session(tmp_path, shifts={1: 0.0, 2: 0.0, 3: -0.05})
    assert record["status"] == "completed" and result.status == "completed"
    assert record["first_setup_exclusions"] == []  # the synthetic data has no ID or constant column
    first = record["steps"][0]
    assert first["proposal"] is None and first["decision"] is None and first["accepted"]
    assert [d["kind"] for d in first["diagnoses"]] == ["overfitting", "class_imbalance", "high_cardinality"]
    step1 = record["steps"][1]
    assert step1["proposal"]["spec"] == {"strategy_id": "drop_columns", "params": {"columns": ["code"]}}
    assert step1["parent_experiment_id"] == first["experiment_id"] and step1["accepted"]
    assert "code" not in engine.fits[1]["columns"] and engine.fits[1]["starting_points"] == {"lgbm": {"num_leaves": 8}}
    for step in record["steps"]:
        files = {p.name for p in (settings.experiments_dir / step["experiment_id"]).iterdir()}
        assert "assessment.json" in files and ("decision.json" in files) == (step["index"] > 0)
    assert record["code_version"]["flaml"] and record["settings"]["thresholds_version"] == "v1-critic-study"


# --------------------------------------------------------------------------- stops
@pytest.mark.parametrize(
    "kwargs, reason, n_steps",
    [
        ({"total": 20, "experiments": 10}, StopReason.TIME, 2),  # 7 s + 2 s, then 5 s + 2 s; 1.6 s left
        # The last Experiment gets all the time left, so only a search that ends early leaves time over.
        ({"experiments": 2, "search_cap": 10}, StopReason.EXPERIMENTS, 2),
        ({"patience": 1, "shifts": {1: -0.05}}, StopReason.PATIENCE, 2),
        ({"stop": lambda: True}, StopReason.USER, 1),
    ],
)
def test_each_stop_reason_ends_the_loop(tmp_path, kwargs, reason, n_steps):
    result, record, *_ = run_session(tmp_path, **kwargs)
    assert result.stop_reason is reason and record["stop_reason"] == reason.value
    assert len(record["steps"]) == n_steps and record["status"] == "completed"
    assert record["final_evaluation"] is not None  # also after a stop request


def test_no_proposals_ends_the_loop(tmp_path):
    # Balanced classes, no text column and no overfitting gap: nothing to diagnose.
    clean = make_synthetic("binary", 300, [], seed=0, fresh_rows=10).frame
    result, record, *_ = run_session(tmp_path, gap=0.0, data=clean)
    assert record["steps"][0]["diagnoses"] == [] and result.stop_reason is StopReason.NO_PROPOSALS
    assert len(record["steps"]) == 1


def test_a_stop_request_still_runs_the_final_evaluation(tmp_path):
    result, record, *_ = run_session(tmp_path, stop=lambda: True)
    final = record["final_evaluation"]
    assert final["first"]["experiment_id"] == final["final"]["experiment_id"] == result.first_experiment_id
    assert final["final"]["test_score"] is not None and final["final"]["feature_importance"]
    assert final["n_test"] == len(TEST_ROWS)


# --------------------------------------------------------------------------- failures
def test_a_failing_candidate_is_discarded_and_the_session_continues(tmp_path):
    # Step 1's search fails; step 2's Assessment (the Assessor's second call) fails; step 3 works.
    result, record, engine, assessor, settings = run_session(tmp_path, experiments=4, engine_fails={1},
                                                              assessor_fails={1}, shifts={2: 0.05})
    failed_search, failed_assessment, ok = record["steps"][1:4]
    assert failed_search["status"] == "failed" and not failed_search["accepted"]
    assert "search 1 exploded" in failed_search["error"] and failed_search["experiment_id"] == "EXP-002"
    assert (settings.experiments_dir / "EXP-002" / "error.json").exists()
    assert failed_assessment["status"] == "failed" and "assessment 1 exploded" in failed_assessment["error"]
    assert failed_assessment["experiment_id"] == "EXP-003"
    assert ok["status"] == "completed" and ok["decision"]["verdict"] == "keep" and ok["accepted"]
    tried = [s["proposal"]["key"] for s in record["steps"][1:]]
    assert len(set(tried)) == 3  # each failure was marked as tried, and the next Proposal was taken
    assert [s["cost"]["verdict"] for s in record["steps"]] == [None, "discard", "discard", "keep"]
    assert result.final_experiment_id == ok["experiment_id"]


def test_a_failing_first_experiment_stops_the_session_with_an_error_record(tmp_path):
    result, record, engine, assessor, settings = run_session(tmp_path, engine_fails={0})
    assert result.status == "failed" and result.stop_reason is StopReason.ERROR
    assert record["status"] == "failed" and record["stop_reason"] == "error" and "search 0 exploded" in record["error"]
    assert record["final_evaluation"] is None and len(record["steps"]) == 1
    assert record["steps"][0]["status"] == "failed" and record["steps"][0]["experiment_id"] == "EXP-001"
    assert (settings.experiments_dir / "EXP-001" / "error.json").exists()


def test_bad_settings_are_refused_before_anything_is_written(tmp_path):
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    session = AutopilotSession(settings=settings, engine=RecordingEngine(FakeClock()), enable_mlflow=False)
    base = ExperimentConfig(target_column="target")
    for plan, message in [
        (SessionSettings(base, SessionBudget(60, 3), selector="thompson"), "Unknown selector"),
        (SessionSettings(base, SessionBudget(60, 3), thresholds_path=tmp_path / "missing.toml"), "not found"),
        (SessionSettings(ExperimentConfig(target_column="nope"), SessionBudget(60, 3)), "not found"),
    ]:
        with pytest.raises(ExperimentError, match=message):
            session.run(DATA, plan)
    assert not any(settings.experiments_dir.iterdir())


# --------------------------------------------------------------------------- replay
def test_diagnoses_and_verdicts_replay_from_the_records(tmp_path):
    # Step 1 drops 'code' (equal scores: keep_simpler); step 2 tries smaller trees (better, with spread
    # over the folds, so the t-test is used: keep); step 3 tries class weights (worse: discard); step 4
    # tries the smoother search metric (equal scores, not simpler: discard). Then nothing is left.
    shifts = {1: 0.0, 2: 0.03 + 0.01 * np.cos(np.arange(15)), 3: -0.02}
    result, record, engine, assessor, settings = run_session(tmp_path, experiments=6, shifts=shifts)
    folder = settings.experiments_dir
    profile = DatasetProfile.from_dict(
        json.loads((folder / result.session_id / "train_profile.json").read_text(encoding="utf-8")))
    critic = Critic(CriticThresholds.load())
    judge_settings = JudgeSettings(**record["settings"]["judge"])
    judge = Judge(judge_settings, test_train_ratio=1 / (record["settings"]["n_splits"] - 1))
    base = record["settings"]["base"]

    def load(experiment_id, name):
        return json.loads((folder / experiment_id / name).read_text(encoding="utf-8"))

    def diagnose(experiment_id):
        summary = load(experiment_id, "summary.json")
        config = ExperimentConfig(target_column=base["target_column"], excluded_columns=summary["excluded_columns"])
        search = SearchSummary(**summary["search_summary"]) if summary["search_summary"] else None
        return [d.to_dict() for d in critic.diagnose(
            profile=profile, config=config, assessment=Assessment.from_dict(load(experiment_id, "assessment.json")),
            search=search, task_type=TaskType(base["task_type"]), primary_metric=base["primary_metric"])]

    checked = 0
    for step in record["steps"]:
        parent = step["parent_experiment_id"] or step["experiment_id"]
        assert diagnose(parent) == step["diagnoses"]  # the Diagnoses the step's Proposal answered
        if step["decision"] is None:
            continue
        accepted = Assessment.from_dict(load(parent, "assessment.json"))
        candidate = Assessment.from_dict(load(step["experiment_id"], "assessment.json"))
        replayed = judge.compare(accepted, candidate, step["proposal"]["complexity"] == "simpler").to_dict()
        assert replayed == step["decision"] == load(step["experiment_id"], "decision.json")
        checked += 1
    assert checked == 4 and verdicts(record)[1:] == [("completed", "keep_simpler"), ("completed", "keep"),
                                                     ("completed", "discard"), ("completed", "discard")]
    assert record["steps"][4]["proposal"]["spec"]["strategy_id"] == "smoother_search_metric"
    assert engine.fits[4]["metric"] == "log_loss"  # the search optimised log loss; the Judge compared F1
    assert record["steps"][2]["decision"]["p_better"] is not None  # the t-test was used
    assert result.stop_reason is StopReason.NO_PROPOSALS  # every Proposal was tried
