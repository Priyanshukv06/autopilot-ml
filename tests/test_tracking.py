"""Tests for records and tracking of Sessions (task 10.1 of the spec)."""

import json
import os
import shutil
import threading
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

from app.core.config import build_settings
from app.services import experiment_tracker
from app.services.automl_runner import AutoMLRunner, EngineResult, ExperimentConfig
from app.services.experiment_tracker import ExperimentTracker, write_json_atomic
from app.services.task_detector import TaskType

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path):
    s = build_settings(tmp_path)
    s.ensure_directories()
    return s


@pytest.fixture
def tracker(settings):
    return ExperimentTracker.from_settings(settings)


def leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.endswith(".tmp"))


# --------------------------------------------------------------------------- session ids
def test_session_ids_never_clash(tracker, settings):
    assert [tracker.allocate_session_id() for _ in range(3)] == ["SES-001", "SES-002", "SES-003"]
    (settings.experiments_dir / "SES-007").mkdir()  # a folder made by hand, or by another program
    assert tracker.allocate_session_id() == "SES-008"
    assert tracker.allocate_experiment_id() == "EXP-001"  # experiments are counted separately


def test_session_ids_are_unique_under_concurrent_allocation(tracker):
    ids, lock = [], threading.Lock()

    def allocate():
        new_id = tracker.allocate_session_id()
        with lock:
            ids.append(new_id)

    threads = [threading.Thread(target=allocate) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(ids) == len(set(ids)) == 16


# --------------------------------------------------------------------------- safe writing
def test_session_json_stays_readable_if_writing_is_interrupted(tracker, settings, monkeypatch):
    tracker.save_session("SES-001", {"session_id": "SES-001", "steps": [{"index": 0}]})
    path = settings.experiments_dir / "SES-001" / "session.json"

    def crash(*args, **kwargs):
        raise OSError("disk full")

    for target in ("fsync", "replace"):  # a crash while writing the new file, or just before renaming it
        with monkeypatch.context() as patch:
            patch.setattr(experiment_tracker.os, target, crash)
            with pytest.raises(OSError, match="disk full"):
                tracker.save_session("SES-001", {"session_id": "SES-001", "steps": [{"index": 0}, {"index": 1}]})
        assert json.loads(path.read_text(encoding="utf-8"))["steps"] == [{"index": 0}]  # the earlier steps
        assert leftovers(path.parent) == []
    tracker.save_session("SES-001", {"session_id": "SES-001", "steps": [{"index": 0}, {"index": 1}]})
    assert len(tracker.load_session("SES-001")["steps"]) == 2


def test_a_file_held_open_for_a_moment_is_retried(tmp_path, monkeypatch):
    real_replace, calls = os.replace, []

    def busy_twice(source, target):
        calls.append(target)
        if len(calls) <= 2:
            raise PermissionError("in use by another program")
        real_replace(source, target)

    monkeypatch.setattr(experiment_tracker.os, "replace", busy_twice)
    monkeypatch.setattr(experiment_tracker.time, "sleep", lambda seconds: None)
    write_json_atomic(tmp_path / "record.json", {"a": 1})
    assert json.loads((tmp_path / "record.json").read_text(encoding="utf-8")) == {"a": 1} and len(calls) == 3


def test_step_files(tracker, settings):
    for name in ("assessment.json", "diagnoses.json", "decision.json"):
        tracker.save_step_file("EXP-004", name, {"name": name})
    folder = settings.experiments_dir / "EXP-004"
    assert json.loads((folder / "decision.json").read_text(encoding="utf-8")) == {"name": "decision.json"}
    with pytest.raises(ValueError, match="Unknown step file"):
        tracker.save_step_file("EXP-004", "summary.json", {})


# --------------------------------------------------------------------------- Phase 1 records
def test_phase1_records_still_load_and_appear_in_the_history(tracker, settings):
    for experiment in ("EXP-001", "EXP-002"):
        saved = ROOT / "experiments" / experiment / "summary.json"
        if not saved.exists():
            pytest.skip("EXP-001 and EXP-002 results are not in this copy of the project")
        (settings.experiments_dir / experiment).mkdir()
        shutil.copy(saved, settings.experiments_dir / experiment / "summary.json")
    tracker.save_record("EXP-003", {"experiment_id": "EXP-003", "schema_version": 2, "session_id": "SES-001",
                                    "test_evaluated": False, "test_score": None})
    records = tracker.load_records()
    assert [r["schema_version"] for r in records] == [1, 1, 2]
    assert records[0]["session_id"] is None and records[0]["test_evaluated"] is True
    history = tracker.history_frame()
    assert history["Experiment"].tolist() == ["EXP-001", "EXP-002", "EXP-003"]
    assert history["Session"].isna().tolist() == [True, True, False] and history["Session"].iloc[2] == "SES-001"
    assert history["Test score"].iloc[0] == pytest.approx(0.6726, abs=1e-4)  # EXP-001's test F1


# --------------------------------------------------------------------------- session history
def session_record(session_id, first_validation, final_validation):
    return {
        "schema_version": 1, "session_id": session_id, "status": "completed", "created_at": "2026-10-26T12:00:00",
        "settings": {"base": {"dataset_name": "titanic", "target_column": "Survived", "primary_metric": "f1"}},
        "steps": [
            {"index": 0, "accepted": True, "assessment": {"validation_mean": first_validation}},
            {"index": 1, "accepted": True, "assessment": {"validation_mean": final_validation}},
            {"index": 2, "accepted": False, "assessment": {"validation_mean": 0.5}},
        ],
        "stop_reason": "patience",
        "final_evaluation": {"first": {"test_metrics": {"f1": 0.70}}, "final": {"test_metrics": {"f1": 0.74}}},
        "total_seconds": 123.4,
    }


def test_session_history_lists_first_and_final_scores(tracker, settings):
    tracker.save_session("SES-001", session_record("SES-001", 0.75, 0.78))
    (settings.experiments_dir / "SES-002").mkdir()
    (settings.experiments_dir / "SES-002" / "session.json").write_text("{not json", encoding="utf-8")
    history = tracker.session_history_frame()
    assert len(history) == 1  # the unreadable record is skipped
    row = history.iloc[0]
    assert (row["Session"], row["Dataset"], row["Metric"], row["Status"]) == ("SES-001", "titanic", "f1", "completed")
    assert (row["First validation"], row["Final validation"]) == (0.75, 0.78)  # the last accepted step
    assert (row["First test"], row["Final test"], row["Experiments"]) == (0.70, 0.74, 3)
    assert (row["Time (s)"], row["Stop reason"]) == (123.4, "patience")


# --------------------------------------------------------------------------- MLflow parent and child runs
class LogisticEngine:
    name = "sklearn-logistic"

    def fit(self, X_train, y_train, **kwargs):
        model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
        return EngineResult(model=model, engine_name=self.name, engine_version="test", best_estimator="logreg",
                            best_model_name="Logistic Regression", best_config={"C": 1.0}, best_validation_score=None,
                            validation_method="none", training_time=0.0, search_history=[], leaderboard=[],
                            native_importance=None, classes=list(model.classes_))


def test_child_runs_point_to_their_parent_run(settings):
    df = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "diagnosis"})
    runner = AutoMLRunner(settings=settings, engine=LogisticEngine(), enable_mlflow=True)
    parent = runner.tracker.start_session_run("SES-001", {"total_seconds": 60}, {"dataset": "breast_cancer"})
    config = ExperimentConfig(target_column="diagnosis", task_type=TaskType.BINARY, time_budget=10)
    child = runner.run_experiment(df, config, evaluate_test=False, session_id="SES-001")
    alone = runner.run_experiment(df, config, compute_importance=False)  # not in a Session
    runner.tracker.finish_session_run("SES-001", {"final_validation": 0.9}, [])

    client = runner.tracker.client
    child_run = client.get_run(child.mlflow_run_id)
    assert child_run.data.tags["mlflow.parentRunId"] == parent
    assert child_run.data.tags["session_id"] == "SES-001" and child_run.data.tags["phase"] == "session"
    assert "mlflow.parentRunId" not in client.get_run(alone.mlflow_run_id).data.tags
    parent_run = client.get_run(parent)
    assert parent_run.info.run_name == "SES-001" and parent_run.info.status == "FINISHED"
    assert parent_run.data.metrics["final_validation"] == 0.9 and parent_run.data.params["total_seconds"] == "60"
