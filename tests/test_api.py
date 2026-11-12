"""Tests for the local HTTP API (optional task 35, Requirement 32.1), with FastAPI's test client."""

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.core.config import build_settings
from app.services.autopilot import AutopilotSession
from tests.test_autopilot import DATA, FakeClock, RecordingEngine, ScriptedAssessor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    DATA.to_csv(settings.sample_data_dir / "synthetic.csv", index=False)  # data the stand-in engine can fit
    (tmp_path / "outside.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return settings


def fake_session(settings, hold=None):
    """A Session with the stand-in engine and Assessor of tests/test_autopilot.py; ``hold`` keeps its first
    search waiting until set, so a test can see it running."""
    clock = FakeClock()

    class Engine(RecordingEngine):
        def fit(self, X_train, y_train, **kwargs):
            if hold is not None:
                hold.wait(10)
            return super().fit(X_train, y_train, **kwargs)

    def assessor(*args):
        return ScriptedAssessor(*args, clock=clock, shifts={}, gap=0.15, fail_on=())

    return AutopilotSession(settings=settings, engine=Engine(clock), clock=clock, enable_mlflow=False,
                            assessor_factory=assessor)


def wait_until_done(client, session_id, limit=60.0):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if client.get("/health").json()["running"] is None:
            return client.get(f"/sessions/{session_id}").json()
        time.sleep(0.2)
    raise AssertionError("the Session did not finish")


REQUEST = {"dataset": "data/sample/synthetic.csv", "target": "target", "seconds": 600, "max_experiments": 3}


def test_a_session_is_started_and_its_record_returned(project):
    client = TestClient(create_app(project, session_factory=fake_session))
    assert client.get("/health").json() == {"status": "ok", "version": "0.1.0", "running": None}
    answer = client.post("/sessions", json=REQUEST)
    assert answer.status_code == 202 and answer.json()["session_id"] == "SES-001"
    record = wait_until_done(client, "SES-001")
    assert record["status"] == "completed" and record["settings"]["base"]["target_column"] == "target"
    listed = client.get("/sessions").json()
    assert [(s["session_id"], s["dataset"], s["status"]) for s in listed] == [("SES-001", "synthetic", "completed")]
    assert record == json.loads((project.experiments_dir / "SES-001" / "session.json").read_text("utf-8"))


def test_only_csv_files_inside_data_are_read_and_bad_requests_are_refused(project):
    client = TestClient(create_app(project, session_factory=fake_session))
    for dataset in ("outside.csv", "data/../outside.csv", "data/sample/nothing.csv", "app/api.py"):
        assert client.post("/sessions", json={**REQUEST, "dataset": dataset}).status_code in (400, 404)
    assert client.post("/sessions", json={**REQUEST, "seconds": 1}).status_code == 422  # below 10 s
    assert client.post("/sessions", json={**REQUEST, "selector": "magic"}).status_code == 400
    wrong = client.post("/sessions", json={**REQUEST, "target": "nope"})
    assert wrong.status_code == 400 and "did not start" in wrong.json()["detail"]
    assert client.get("/sessions/SES-999").status_code == 404
    assert client.get("/sessions/..%2F..%2Fsecret").status_code == 404
    assert not any(project.experiments_dir.iterdir())  # nothing was started


def test_one_session_at_a_time_and_stop(project):
    hold = threading.Event()
    client = TestClient(create_app(project, session_factory=lambda s: fake_session(s, hold)))
    started = client.post("/sessions", json={**REQUEST, "max_experiments": 10}).json()["session_id"]
    assert client.get("/health").json()["running"] == started
    assert client.post("/sessions", json=REQUEST).status_code == 409
    assert client.post(f"/sessions/{started}/stop").json()["stopping"] is True
    hold.set()
    record = wait_until_done(client, started)
    assert record["stop_reason"] == "user" and record["status"] == "completed"
    assert client.post(f"/sessions/{started}/stop").status_code == 409  # no longer running
