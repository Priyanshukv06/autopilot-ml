"""Plain-language Session summaries by a local language model (optional task 34, Requirement 31).

The model runs on this machine through Ollama's HTTP API (``/api/generate``); only a server on this
machine is accepted, so no record leaves it and nothing is paid for. The prompt holds facts taken
from the Session record only (``session_facts``), never data rows. The text is labelled as generated,
is saved in its own file (``summary_generated.json``) and is never read by the Session loop, the
Critic, the Strategy_Engine or the Judge (Requirement 31.2). Numbers in the text that are not in the
facts are listed next to it, because a language model can invent them.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_URL = "http://localhost:11434"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
SUMMARY_FILE = "summary_generated.json"
LABEL = ("Generated text, written by a local language model ({model}) from the Session record only. It was "
         "not checked by a person and no decision used it.")
INSTRUCTIONS = """You explain the result of an automated machine-learning Session to a student.
Write 120 to 180 words of plain English in two or three short paragraphs, without headings, lists or emojis.
Say what data and target were used, what the Session tried and kept and why it stopped, and how the final
model compares with the first one on the test rows. Use only the facts below. Do not add numbers that are not
in the facts. If a fact is missing, do not guess it.

Facts (JSON):
"""

OUTCOMES = {"keep": "kept, because it was better", "keep_simpler": "kept, because it was simpler and not worse",
            "discard": "not kept"}

Post = Callable[[str, dict[str, Any], float], dict[str, Any]]


class NarrationError(RuntimeError):
    """The summary could not be written (no local server, an error from it, or a non-local address)."""


@dataclass
class Narration:
    text: str
    model: str
    label: str
    created_at: str
    seconds: float
    facts_sha256: str
    unsupported_numbers: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- facts
def _round(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def session_facts(record: dict[str, Any]) -> dict[str, Any]:
    """What the text may say: settings, steps and results, from ``session.json`` only (no data rows)."""
    base = record.get("settings", {}).get("base", {})
    budget = record.get("settings", {}).get("budget", {})
    steps = []
    for step in record.get("steps", []):
        proposal = step.get("proposal") or {}
        decision = step.get("decision") or {}
        assessment = step.get("assessment") or {}
        verdict = decision.get("verdict")
        steps.append({
            "step": step.get("index"),
            "change": (proposal.get("reason") or "first Setup").split(". Answers:")[0],
            "status": step.get("status"),
            # in words, because a small model misread the verdict codes (12 November 2026)
            "outcome": OUTCOMES.get(verdict, "the starting point" if not step.get("index") else "failed"),
            "verdict": verdict,
            "mean_difference": _round(decision.get("mean_difference")),
            "validation_score": _round(assessment.get("validation_mean")),
        })
    first_diagnoses = [{"kind": d.get("kind"), "columns": d.get("columns"), "severity": d.get("severity")}
                       for d in (record.get("steps") or [{}])[0].get("diagnoses", [])]
    final = record.get("final_evaluation") or {}
    metric = base.get("primary_metric")

    def test_score(part: str) -> Any:
        return _round(((final.get(part) or {}).get("test_metrics") or {}).get(metric))

    first_score, final_score = test_score("first"), test_score("final")
    return {
        "session_id": record.get("session_id"),
        "dataset": base.get("dataset_name"),
        "target": base.get("target_column"),
        "task": base.get("task_type"),
        "metric": metric,
        "time_budget_seconds": budget.get("total_seconds"),
        "max_experiments": budget.get("max_experiments"),
        "training_rows": record.get("splits", {}).get("n_train"),
        "test_rows": record.get("splits", {}).get("n_test"),
        "columns_left_out_at_start": record.get("first_setup_exclusions") or [],
        "first_diagnoses": first_diagnoses,
        "steps": steps,
        "changes_kept": [s["change"] for s in steps if s["verdict"] in OUTCOMES and s["verdict"] != "discard"],
        "stop_reason": record.get("stop_reason"),
        "first_model_test_score": first_score,
        "final_model_test_score": final_score,
        "final_minus_first_test_score": _round(final_score - first_score)
        if isinstance(first_score, float) and isinstance(final_score, float) else None,
        "seconds_used": _round(record.get("total_seconds")),
    }


def build_prompt(facts: dict[str, Any]) -> str:
    return INSTRUCTIONS + json.dumps(facts, indent=1, sort_keys=True)


# --------------------------------------------------------------------------- number check
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?:\s?%)?")


def _numbers_in(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return [float(m.replace(",", ".").rstrip("%").strip()) for m in _NUMBER.findall(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers_in(v)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _numbers_in(v)]
    return []


def unsupported_numbers(text: str, facts: dict[str, Any]) -> list[str]:
    """Numbers in ``text`` that no fact gives, after rounding to the text's decimals (a percentage also
    counts as a share, and a difference may be given without its sign)."""
    known = [abs(k) for k in _numbers_in(facts)]
    missing = []
    for token in _NUMBER.findall(text):
        clean = token.replace(",", ".").replace(" ", "")
        percent = clean.endswith("%")
        digits = clean.rstrip("%")
        number = abs(float(digits))
        tolerance = 0.5 * 10 ** -(len(digits.split(".")[1]) if "." in digits else 0) + 1e-9
        targets = [(number, tolerance)] + ([(number / 100, tolerance / 100)] if percent else [])
        if not any(abs(k - value) <= tol for k in known for value, tol in targets):
            missing.append(token.strip())
    return list(dict.fromkeys(missing))


# --------------------------------------------------------------------------- the model
def _check_local(url: str) -> None:
    host = urlparse(url).hostname
    if host not in LOCAL_HOSTS:
        raise NarrationError(f"Only a model server on this machine is allowed ({', '.join(LOCAL_HOSTS)}), not {host!r}.")


def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (local address, checked)
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise NarrationError(f"The local model server answered {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NarrationError(f"The local model server at {url} did not answer ({exc}). Is Ollama running?") from exc


def narrate(record: dict[str, Any], *, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL, timeout: float = 300.0,
            post: Post | None = None) -> Narration:
    """Ask the local model for a summary of the Session record (temperature 0 and a fixed seed)."""
    _check_local(url)
    facts = session_facts(record)
    payload = {"model": model, "prompt": build_prompt(facts), "stream": False,
               "options": {"temperature": 0, "seed": 0}}
    started = time.perf_counter()
    answer = (post or _post)(url.rstrip("/") + "/api/generate", payload, timeout)
    text = str(answer.get("response") or "").strip()
    if not text:
        raise NarrationError(f"The local model gave no text: {str(answer)[:200]}")
    digest = hashlib.sha256(json.dumps(facts, sort_keys=True).encode("utf-8")).hexdigest()
    return Narration(text=text, model=model, label=LABEL.format(model=model),
                     created_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                     seconds=round(time.perf_counter() - started, 2), facts_sha256=digest,
                     unsupported_numbers=unsupported_numbers(text, facts), facts=facts)


def save(narration: Narration, session_dir: Path) -> Path:
    """Write the summary to its own file in the Session folder (the record itself is not changed)."""
    path = Path(session_dir) / SUMMARY_FILE
    path.write_text(json.dumps(narration.to_dict(), indent=2), encoding="utf-8")
    return path
