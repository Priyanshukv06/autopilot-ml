"""Tests for the local language-model summaries (optional task 34, Requirement 31), with a stand-in server."""

import json
from pathlib import Path

import pytest

from app.services import narrator
from app.services.narrator import NarrationError, build_prompt, narrate, save, session_facts, unsupported_numbers
from tests.test_autopilot import run_session

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    return run_session(tmp_path_factory.mktemp("session"), shifts={1: 0.0, 2: 0.03, 3: -0.05})[1]


class Server:
    """Answers like Ollama's /api/generate and remembers what it was asked."""

    def __init__(self, text):
        self.text, self.calls = text, []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        return {"model": payload["model"], "response": self.text, "done": True}


def test_the_prompt_holds_facts_from_the_record_only(record):
    facts = session_facts(record)
    assert facts["dataset"] == "synthetic" and facts["target"] == "target" and facts["metric"] == "f1"
    assert facts["stop_reason"] == record["stop_reason"] and len(facts["steps"]) == len(record["steps"])
    assert facts["steps"][1]["verdict"] == record["steps"][1]["decision"]["verdict"]
    assert facts["final_model_test_score"] == round(
        record["final_evaluation"]["final"]["test_metrics"]["f1"], 4)
    prompt = build_prompt(facts)
    assert "Use only the facts below" in prompt and json.dumps(facts, indent=1, sort_keys=True) in prompt
    assert "x0" not in prompt  # no data rows or column values beyond the record's own words


def test_a_summary_is_labelled_checked_for_numbers_and_saved_apart(record, tmp_path):
    score = session_facts(record)["final_model_test_score"]
    server = Server(f"The final model scored {score:.2f} on the test rows, against 0.123456 invented here.")
    narration = narrate(record, post=server)
    url, payload, _ = server.calls[0]
    assert url == "http://localhost:11434/api/generate" and payload["stream"] is False
    assert payload["model"] == narrator.DEFAULT_MODEL and payload["options"] == {"temperature": 0, "seed": 0}
    assert narration.label.startswith("Generated text, written by a local language model (qwen2.5:7b-instruct)")
    assert narration.unsupported_numbers == ["0.123456"]  # the score is in the facts, the other number is not
    path = save(narration, tmp_path)
    assert path.name == "summary_generated.json" and json.loads(path.read_text("utf-8"))["text"] == narration.text


def test_numbers_match_after_rounding_and_as_percentages():
    facts = {"a": 0.7826, "b": -0.0125, "n": 712, "steps": [{"x": 3}]}
    text = "It scored 0.78 (78%), lost 0.0125, used 712 rows in 3 steps, and 0.9 and 55% are made up."
    assert unsupported_numbers(text, facts) == ["0.9", "55%"]


def test_only_a_local_server_is_used_and_errors_are_clear(record):
    with pytest.raises(NarrationError, match="Only a model server on this machine"):
        narrate(record, url="https://api.example.com", post=Server("x"))
    with pytest.raises(NarrationError, match="no text"):
        narrate(record, post=Server(""))
    with pytest.raises(NarrationError, match="did not answer"):
        narrate(record, url="http://127.0.0.1:9", timeout=2)  # nothing listens on port 9


def test_no_decision_reads_the_summary():
    for module in ("autopilot", "critic", "strategy_engine", "judge", "resource_manager", "experiment_planner",
                   "experiment_memory", "assessor"):
        source = (ROOT / "app" / "services" / f"{module}.py").read_text(encoding="utf-8")
        assert "narrator" not in source and narrator.SUMMARY_FILE not in source
