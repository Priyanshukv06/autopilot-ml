"""Dashboard tests (spec task 18.1) with Streamlit's AppTest, in a temporary project folder.

The searches are real FLAML searches with a fixed number of trials, so a Session takes seconds and does
not depend on the speed of the machine. MLflow is switched off here; tracking has its own tests.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app.core.config as config_module
from app.components import autopilot_viewer, experiment_viewer
from app.core.config import build_settings
from app.services.automl_runner import AutoMLRunner, FLAMLEngine
from app.services.autopilot import AutopilotSession

ROOT = Path(__file__).resolve().parents[1]
MAIN = str(ROOT / "app" / "main.py")
TRIALS = 3
pytestmark = [
    pytest.mark.slow,
    # FLAML warns on every prediction for Titanic's unseen tickets; the app drops these (automl_runner).
    pytest.mark.filterwarnings("ignore:Column .* contains values unseen at fit time:UserWarning"),
]


class FixedTrialsEngine(FLAMLEngine):
    """FLAML that stops after TRIALS trials, whatever its time budget."""

    def fit(self, *args, **kwargs):
        kwargs["max_iter"] = TRIALS
        return super().fit(*args, **kwargs)


@pytest.fixture
def project(tmp_path, monkeypatch):
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    shutil.copy(ROOT / "data" / "sample" / "titanic.csv", settings.sample_data_dir / "titanic.csv")
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.setattr(autopilot_viewer, "new_session", lambda s: AutopilotSession(
        settings=s, engine=FixedTrialsEngine(), enable_mlflow=False))
    monkeypatch.setattr(experiment_viewer, "AutoMLRunner", lambda settings: AutoMLRunner(
        settings=settings, engine=FixedTrialsEngine(), enable_mlflow=False))
    return settings


def open_app(mode: str) -> AppTest:
    at = AppTest.from_file(MAIN, default_timeout=120)
    at.run()
    at.radio(key="data_source").set_value("Sample dataset").run()
    at.radio(key="mode").set_value(mode).run()
    assert not at.exception
    return at


def start_session(at: AppTest, max_experiments: int) -> None:
    at.number_input(key="auto_max_experiments").set_value(max_experiments)
    at.selectbox(key="auto_cores").set_value(2)
    at.selectbox(key="auto_repeats").set_value(1)  # 5 folds per Assessment keep the test short
    at.run()
    at.button(key="auto_start").click().run()
    assert not at.exception


def wait_for_session(at: AppTest, limit: float = 240.0) -> autopilot_viewer.FinishedSession:
    deadline = time.monotonic() + limit
    while autopilot_viewer.ACTIVE_SESSION_KEY in at.session_state and time.monotonic() < deadline:
        time.sleep(0.5)
        at.run()
    assert autopilot_viewer.ACTIVE_SESSION_KEY not in at.session_state, "the Session did not end in time"
    assert not at.exception
    return at.session_state[autopilot_viewer.LAST_SESSION_KEY]


def texts(elements) -> str:
    return "\n".join(str(element.value) for element in elements)


def timeline(at: AppTest):
    return next(d.value for d in at.dataframe if "Verdict" in d.value.columns and "Strategy" in d.value.columns)


def test_the_sidebar_marks_the_built_modules(project):
    at = AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    sidebar = texts(at.sidebar.markdown)
    assert "✅ Autonomous mode in this dashboard" in sidebar and "✅ Experiment memory" in sidebar
    assert "✅ Learned strategy choice" in sidebar
    planned = sidebar.split("Planned, not implemented", 1)[1]
    assert "Autonomous mode" not in planned and "Learned" not in planned and "Benchmark" in planned
    assert at.radio(key="mode").value == "Single experiment"  # the default mode is unchanged


def test_the_single_experiment_mode_works_as_before(project):
    at = open_app("Single experiment")
    headers = [h.value for h in at.header]
    assert "3 · Experiment Configuration" in headers and "4 · Run AutoML" in headers
    next(b for b in at.button if b.label == "Run Experiment").click().run()
    assert not at.exception
    assert experiment_viewer.LAST_RESULT_KEY in at.session_state
    assert {"5 · Results", "6 · Explainability", "7 · Experiment History"} <= {h.value for h in at.header}
    assert "3 · Session settings" not in [h.value for h in at.header]


def test_an_autonomous_run_shows_its_steps_and_the_final_comparison(project):
    at = open_app("Autonomous run")
    assert "3 · Session settings" in [h.value for h in at.header]
    start_session(at, max_experiments=3)
    finished = wait_for_session(at)
    record = finished.record
    assert finished.error is None and record["status"] == "completed"
    assert 2 <= len(record["steps"]) <= 3 and record["stop_reason"] in ("experiments", "no_proposals", "patience")

    steps = timeline(at)
    assert list(steps["Step"]) == [s["index"] for s in record["steps"]] and steps["Verdict"].iloc[0] == "first"
    assert list(steps["Verdict"].iloc[1:]) == [(s["decision"] or {}).get("verdict") or "failed"
                                               for s in record["steps"][1:]]
    labels = [m.label for m in at.metric]
    assert any(label.startswith(f"First Setup · {record['first_experiment_id']}") for label in labels)
    assert any(label.startswith(f"Final Setup · {record['final_experiment_id']}") for label in labels)
    assert "Not used for decisions" in texts(at.caption)
    assert "5 · Experiment memory for this dataset" in [h.value for h in at.header]
    judged = [s for s in record["steps"][1:] if s["decision"]]
    memory = [d.value for d in at.dataframe if "Cost (s)" in d.value.columns]
    assert len(memory) == (1 if judged else 0)
    if judged:
        assert len(memory[0]) == len(judged)  # one memory record per Verdict, shown for this dataset
    assert (project.experiments_dir / record["session_id"] / "session.json").exists()


def test_a_failed_session_is_shown_as_an_error(project, monkeypatch):
    class FailingEngine(FixedTrialsEngine):
        def fit(self, *args, **kwargs):
            raise RuntimeError("search exploded")

    monkeypatch.setattr(autopilot_viewer, "new_session", lambda s: AutopilotSession(
        settings=s, engine=FailingEngine(), enable_mlflow=False))
    at = open_app("Autonomous run")
    start_session(at, max_experiments=3)
    finished = wait_for_session(at)
    assert finished.record["status"] == "failed" and finished.record["stop_reason"] == "error"
    assert "search exploded" in texts(at.error)
    assert list(timeline(at)["Verdict"]) == ["failed"]
    assert at.button(key="auto_start")  # a new Session can be started


def test_stop_ends_the_session_with_the_final_evaluation(project):
    at = open_app("Autonomous run")
    start_session(at, max_experiments=10)
    at.button(key="auto_stop").click().run()
    assert not at.exception
    finished = wait_for_session(at)
    record = finished.record
    assert record["status"] == "completed" and record["stop_reason"] == "user"
    assert len(record["steps"]) < 10
    final = record["final_evaluation"]
    assert final["first"]["test_score"] is not None and final["final"]["test_score"] is not None
    assert "Stop was pressed" in texts(at.success)
