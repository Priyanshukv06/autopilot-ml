"""Autonomous mode of the dashboard (Requirement 15): Session settings, Start and Stop, and the Session view.

The Session view shows a timeline of the steps, the Accepted_Setup's validation score per step and the
Diagnoses with their evidence. Once the Session has ended it also shows the first and the final model
on the test part, labelled "not used for decisions". The memory section lists the Experiment_Memory
records of the current dataset and target.

The Session runs in a background thread, like the single-experiment run. The thread never calls
Streamlit: it only writes the latest progress message and session record into a shared state under a
lock, and the page reads a copy. While the Session runs, a fragment redraws the view every second, so
the page stays usable and Stop can be pressed. Stop sets the flag the Session reads before it chooses
the next step: the Session ends after the current step and still scores both models on the test part.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.dataset_profiler import TargetSelection
from app.components.dataset_uploader import LoadedDataset
from app.core.config import Settings
from app.core.logging_config import get_logger
from app.services.automl_runner import ExperimentConfig
from app.services.autopilot import SELECTORS, AutopilotSession, SessionSettings
from app.services.evaluator import METRICS, PRIMARY_METRICS, metric_display_name, metric_warnings
from app.services.experiment_memory import ExperimentMemory, dataset_fingerprint
from app.services.judge import JudgeSettings
from app.services.resource_manager import SessionBudget
from app.services.task_detector import TaskType

logger = get_logger(__name__)

ACTIVE_SESSION_KEY = "active_session"
LAST_SESSION_KEY = "last_session"
REFRESH_SECONDS = 1.0
ALPHAS = (0.05, 0.10, 0.20)  # the values compared by the judge study
REPEATS = (1, 3, 5)
STOP_REASONS = {
    "time": "the time budget was used up",
    "experiments": "the maximum number of Experiments was reached",
    "no_proposals": "no Strategy was left to try",
    "patience": "{patience} candidates in a row were discarded",
    "user": "Stop was pressed",
    "error": "of an error",
}
_MARKERS = {"keep": "circle", "keep_simpler": "diamond", "discard": "x", "failed": "x-open"}


def new_session(settings: Settings) -> AutopilotSession:
    """The Session the dashboard runs. Tests replace it with one whose searches have a fixed number of trials."""
    return AutopilotSession(settings=settings)


# --------------------------------------------------------------------- section 3
def render_session_settings(settings: Settings, selection: TargetSelection, dataset: LoadedDataset) -> SessionSettings | None:
    """Budget inputs and advanced settings; None (with the reason shown) if they do not make a valid Session."""
    task = selection.task_type
    c1, c2, c3 = st.columns(3)
    minutes = c1.number_input(
        "Total time (minutes)", min_value=0.5, max_value=120.0, value=5.0, step=0.5, key="auto_minutes",
        help="Wall-clock budget of the whole Session: every search, every Assessment and the final test. "
        "The last step can end a little after it.",
    )
    max_experiments = c2.number_input(
        "Maximum Experiments", min_value=2, max_value=50, value=8, step=1, key="auto_max_experiments",
        help="Including the first Experiment.",
    )
    cpu_count = os.cpu_count() or 1
    cores = c3.selectbox(
        "CPU cores", [-1, *range(1, cpu_count + 1)], key="auto_cores",
        format_func=lambda n: f"All ({cpu_count})" if n == -1 else str(n),
        help="Used by every search and every refit of the Session.",
    )
    c4, c5, c6 = st.columns(3)
    metric = c4.selectbox(
        "Primary metric", list(PRIMARY_METRICS[task]), format_func=lambda m: metric_display_name(m, task),
        key=f"auto_metric::{task.value}", help="The metric every search optimises and every decision compares.",
    )
    seed = c5.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=settings.default_random_seed,
                           step=1, key="auto_seed")
    test_size = c6.slider(
        "Test split", min_value=0.1, max_value=0.4, value=min(max(float(settings.default_test_size), 0.1), 0.4),
        step=0.05, key="auto_test_size",
        help="Share of rows held out. No decision uses them; they are scored once, after the Session ends.",
    )
    with st.expander("Advanced settings"):
        a1, a2, a3 = st.columns(3)
        default_repeats = next(f.default for f in fields(SessionSettings) if f.name == "n_repeats")
        n_repeats = a1.selectbox(
            "Fold repeats", sorted({*REPEATS, default_repeats}), key="auto_repeats",
            index=sorted({*REPEATS, default_repeats}).index(default_repeats),
            help="Every Setup is refitted on 5 folds of the training part, repeated this many times with "
            "different shuffles. More repeats give the Judge more evidence and take more time.",
        )
        default_alpha = JudgeSettings().alpha
        alphas = sorted({*ALPHAS, default_alpha})
        alpha = a2.selectbox(
            "Judge alpha", alphas, index=alphas.index(default_alpha), format_func=lambda a: f"{a:.2f}",
            key="auto_alpha",
            help="Significance level of the Judge's corrected t-test. A larger alpha keeps more changes, "
            "including more harmful ones.",
        )
        selector = a3.selectbox(
            "Strategy choice", list(SELECTORS), key="auto_selector",
            help="rules: the fixed rule order. random: a random order. thompson and ucb1: learned from what the "
            "experiment memory saw on similar datasets (needs the memory switched on; with too little "
            "experience a change keeps its rule-order place).",
        )
        use_memory = st.checkbox(
            "Use the experiment memory", value=False, key="auto_use_memory",
            help="Let the Strategy choice read the memory records of other datasets.",
        )
        no_group = "(none)"
        group = st.selectbox(
            "Group column", [no_group, *[c for c in dataset.df.columns if c != selection.target]], key="auto_group",
            help="For data where new rows come from new groups (for example several rows per patient or per "
            "store): rows with the same value stay together, in the test part and in every fold, and the "
            "column is not used as a feature.",
        )
    for warning in metric_warnings(task, metric, selection.profile.class_balance):
        st.warning(warning)

    try:
        budget = SessionBudget(total_seconds=float(minutes) * 60, max_experiments=int(max_experiments),
                               n_jobs=int(cores))
    except ValueError as exc:
        st.error(str(exc))
        return None
    base = ExperimentConfig(
        target_column=selection.target, task_type=task, primary_metric=metric, random_seed=int(seed),
        test_size=float(test_size), excluded_columns=list(selection.excluded_columns), dataset_name=dataset.name,
    )
    return SessionSettings(base=base, budget=budget, n_repeats=int(n_repeats),
                           judge=replace(JudgeSettings(), alpha=float(alpha)), selector=selector,
                           use_memory=bool(use_memory), group_column=None if group == no_group else str(group))


# --------------------------------------------------------------------- section 4
@dataclass
class _ActiveSession:
    """A running Session. Lives in session state so it survives Streamlit reruns."""

    plan: SessionSettings
    started: float
    thread: threading.Thread | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    live: dict[str, Any] = field(default_factory=dict)  # stage, step, message and the latest session record
    outcome: dict[str, Any] = field(default_factory=dict)  # "result" or "error"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.live)


@dataclass(frozen=True)
class FinishedSession:
    record: dict[str, Any] | None  # the last session.json, None if the Session did not start
    error: str | None


def _start_session(df: pd.DataFrame, plan: SessionSettings, settings: Settings) -> _ActiveSession:
    active = _ActiveSession(plan=plan, started=time.monotonic())

    def on_progress(stage: str, step: int, message: str) -> None:  # runs in the worker thread
        with active.lock:
            active.live.update(stage=stage, step=step, message=message)

    def on_update(record: dict[str, Any]) -> None:  # runs in the worker thread
        with active.lock:
            active.live["record"] = record

    def worker() -> None:
        try:
            active.outcome["result"] = new_session(settings).run(
                df, plan, progress=on_progress, stop_requested=active.stop.is_set, on_update=on_update)
        except Exception as exc:  # shown on the page; the details are in the log
            logger.exception("The autonomous Session failed")
            active.outcome["error"] = exc

    active.thread = threading.Thread(target=worker, name="autopilot-session", daemon=True)
    active.thread.start()
    return active


def _finish(active: _ActiveSession) -> FinishedSession:
    active.thread.join()
    record = active.snapshot().get("record")
    error = active.outcome.get("error")
    message = None
    if error is not None:  # settings and data errors come before the record exists
        prefix = "The Session did not start" if record is None else "The Session stopped with an error"
        message = f"{prefix}: {error}"
    return FinishedSession(record=record, error=message)


def render_session_run(df: pd.DataFrame | None, plan: SessionSettings | None, settings: Settings) -> None:
    """Start button, then the live view while a Session runs, or the last finished Session."""
    active: _ActiveSession | None = st.session_state.get(ACTIVE_SESSION_KEY)
    if active is not None and not active.thread.is_alive():
        st.session_state[LAST_SESSION_KEY] = _finish(active)
        st.session_state.pop(ACTIVE_SESSION_KEY, None)
        active = None

    if active is not None:
        _live_view(settings)
        return
    if df is None or plan is None:
        st.info("Choose a dataset and a target above to start an autonomous run.")
    else:
        budget = plan.budget
        st.caption(
            f"The Session gets {budget.total_seconds / 60:g} minutes and at most {budget.max_experiments} "
            f"Experiments on '{plan.base.dataset_name}' → '{plan.base.target_column}'. It decides every step on "
            "the training part and scores the test part once, at the end."
        )
        if st.button("Start autonomous run", type="primary", width="stretch", key="auto_start"):
            st.session_state[ACTIVE_SESSION_KEY] = _start_session(df.copy(), plan, settings)
            st.session_state.pop(LAST_SESSION_KEY, None)
            st.rerun()
    finished: FinishedSession | None = st.session_state.get(LAST_SESSION_KEY)
    if finished is not None:
        _render_finished(finished, settings)


@st.fragment(run_every=REFRESH_SECONDS)
def _live_view(settings: Settings) -> None:
    active: _ActiveSession | None = st.session_state.get(ACTIVE_SESSION_KEY)
    if active is None:
        return
    if not active.thread.is_alive():
        st.rerun()  # the whole page, which shows the finished Session and the Start button again
    snapshot = active.snapshot()
    total = active.plan.budget.total_seconds
    elapsed = time.monotonic() - active.started
    text = f"{elapsed:.0f} s of {total:.0f} s"
    if elapsed > total:
        text += ", finishing the current step"
    st.progress(min(elapsed / total, 1.0), text=text)
    left, right = st.columns([4, 1])
    left.markdown(f"**Now:** {snapshot.get('message', 'Starting the Session')}")
    if not active.stop.is_set() and right.button("Stop", width="stretch", key="auto_stop",
                                                 help="End the Session after the current step."):
        active.stop.set()
    if active.stop.is_set():
        st.info("Stop requested. The Session ends after the current step, then scores the first and the final "
                "model on the test part.")
    record = snapshot.get("record")
    if record is not None:
        _render_steps(record, settings)


def _render_finished(finished: FinishedSession, settings: Settings) -> None:
    record = finished.record
    if finished.error is not None:
        st.error(finished.error)
    if record is None:
        if finished.error is None:
            st.error("The Session ended without a record.")
        return
    if record["status"] != "completed":
        if record.get("error"):
            st.error(f"{record['session_id']} failed: {record['error']}")
        elif finished.error is None:
            st.error(f"{record['session_id']} ended with the status '{record['status']}'.")
        _render_steps(record, settings)
        return
    budget = record["settings"]["budget"]
    st.success(f"{record['session_id']} finished: {_count(len(record['steps']), 'Experiment', 'Experiments')} in "
               f"{record['total_seconds']:.0f} s. It stopped because {stop_reason_text(record)}.")
    _render_final_comparison(record, settings)
    st.markdown("#### How the Session got there")
    _render_steps(record, settings)
    st.caption(f"Record: {settings.experiments_dir / record['session_id'] / 'session.json'} · "
               f"budget {budget['total_seconds']:.0f} s, at most {budget['max_experiments']} Experiments")


# --------------------------------------------------------------------- the Session view
def stop_reason_text(record: dict[str, Any]) -> str:
    reason = record.get("stop_reason") or ""
    patience = record.get("settings", {}).get("budget", {}).get("patience")
    return STOP_REASONS.get(reason, reason).format(patience=patience)


def split_reason(reason: str) -> tuple[str, str]:
    """A Proposal's reason as (the change, the Diagnoses it answers)."""
    change, _, answers = reason.partition(". Answers: ")
    return change.rstrip("."), answers.rstrip(".")


def verdict_of(step: dict[str, Any]) -> str:
    if step["status"] == "failed":
        return "failed"
    if step["index"] == 0:
        return "first"
    return (step.get("decision") or {}).get("verdict") or "n/a"


def timeline_frame(steps: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per step: what was tried, what it answered, the Judge's numbers and the time used."""
    rows = []
    for step in steps:
        proposal = step.get("proposal")
        decision = step.get("decision") or {}
        change, answers = split_reason(proposal["reason"]) if proposal else ("first Setup", "")
        rows.append({
            "Step": step["index"],
            "Experiment": step.get("experiment_id") or "-",
            "Diagnoses answered": answers or (", ".join(proposal.get("answers", [])) if proposal else "-"),
            "Strategy": proposal["spec"]["strategy_id"] if proposal else "-",
            "Change": change,
            "Mean difference": decision.get("mean_difference"),
            "Spread (std)": decision.get("std_difference"),
            "p(better)": decision.get("p_better"),
            "p(worse)": decision.get("p_worse"),
            "Verdict": verdict_of(step),
            "Validation score": (step.get("assessment") or {}).get("validation_mean"),
            "Time (s)": (step.get("cost") or {}).get("wall_seconds"),
        })
    return pd.DataFrame(rows)


def score_frame(steps: list[dict[str, Any]]) -> pd.DataFrame:
    """Per step: the Accepted_Setup's validation score after the step, and the candidate's score."""
    rows, accepted = [], None
    for step in steps:
        score = (step.get("assessment") or {}).get("validation_mean")
        if step["status"] == "completed" and (step["index"] == 0 or step.get("accepted")):
            accepted = score
        rows.append({"step": step["index"], "accepted": accepted,
                     "candidate": score if step["index"] > 0 else None, "verdict": verdict_of(step)})
    return pd.DataFrame(rows)


def accepted_diagnoses(steps: list[dict[str, Any]], experiments_dir: Path) -> list[tuple[int, str, list[dict[str, Any]]]]:
    """(step, experiment id, Diagnoses) of every Setup that was accepted, in order.

    A step's ``diagnoses`` belong to the Setup it tried to improve, so an accepted Setup's Diagnoses are
    in the next step that started from it; for the last one they are read from its ``diagnoses.json``.
    """
    out = []
    for step in steps:
        if step["status"] != "completed" or not (step["index"] == 0 or step.get("accepted")):
            continue
        experiment_id = step["experiment_id"]
        if step["index"] == 0:
            diagnoses = step.get("diagnoses") or []
        else:
            later = next((s for s in steps if s.get("parent_experiment_id") == experiment_id), None)
            diagnoses = later["diagnoses"] if later else _read_json(experiments_dir / experiment_id / "diagnoses.json") or []
        out.append((step["index"], experiment_id, diagnoses))
    return out


def _render_steps(record: dict[str, Any], settings: Settings) -> None:
    steps = record.get("steps") or []
    base = record["settings"]["base"]
    task = TaskType(base["task_type"])
    metric_name = metric_display_name(base["primary_metric"], task)
    n_folds = record.get("splits", {}).get("n_folds")
    if not steps:
        st.caption("The first Experiment is running.")
        return

    number = st.column_config.NumberColumn(format="%.4f")
    st.dataframe(
        timeline_frame(steps), hide_index=True, width="stretch",
        column_config={"Mean difference": number, "Spread (std)": number, "p(better)": number, "p(worse)": number,
                       "Validation score": number, "Time (s)": st.column_config.NumberColumn(format="%.1f")},
    )
    st.caption(
        f"Validation score: {metric_name} of the step's Setup, mean over {n_folds} folds of the training part. "
        "Mean difference: candidate minus Accepted_Setup on the same folds, with its standard deviation. "
        "p(better) and p(worse): one-sided corrected resampled t-test. keep: significantly better; "
        "keep_simpler: simpler and not significantly worse; discard: neither."
    )

    frame = score_frame(steps)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["step"], y=frame["accepted"], mode="lines+markers", name="Accepted Setup",
                             line_shape="hv"))
    candidates = frame.dropna(subset=["candidate"])
    for verdict, part in candidates.groupby("verdict", sort=False):
        fig.add_trace(go.Scatter(x=part["step"], y=part["candidate"], mode="markers", name=f"Candidate: {verdict}",
                                 marker=dict(size=10, symbol=_MARKERS.get(verdict, "circle"))))
    fig.update_layout(height=320, margin=dict(t=40, b=10), title=f"Validation {metric_name} per step",
                      xaxis_title="Step", yaxis_title=metric_name)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Diagnoses of the accepted Setups")
    st.caption(f"Each card is one Diagnosis of the Critic (thresholds {record['settings']['thresholds_version']}) "
               "with the numbers it is based on, all from the training part. Newest Setup first.")
    accepted = accepted_diagnoses(steps, settings.experiments_dir)
    for position, (index, experiment_id, diagnoses) in enumerate(reversed(accepted)):
        label = f"Step {index} · {experiment_id}: {_count(len(diagnoses), 'Diagnosis', 'Diagnoses')}"
        with st.expander(label, expanded=position == 0):
            if not diagnoses:
                st.caption("The Critic found no problem in this Setup.")
            for diagnosis in diagnoses:
                _render_diagnosis(diagnosis)


def _render_diagnosis(diagnosis: dict[str, Any]) -> None:
    with st.container(border=True):
        columns = diagnosis.get("columns") or []
        where = f" · columns: {', '.join(columns)}" if columns else ""
        title = str(diagnosis["kind"]).replace("_", " ").capitalize()
        st.markdown(f"**{title}** · severity: {diagnosis['severity']}{where}")
        st.write(diagnosis["message"])
        evidence = diagnosis.get("evidence") or {}
        if evidence:
            st.dataframe(pd.DataFrame({"Evidence": list(evidence), "Value": [_value(v) for v in evidence.values()]}),
                         hide_index=True)


def _render_final_comparison(record: dict[str, Any], settings: Settings) -> None:
    final = record.get("final_evaluation")
    if not final:
        return
    base = record["settings"]["base"]
    task = TaskType(base["task_type"])
    metric = base["primary_metric"]
    metric_name = metric_display_name(metric, task)
    better_up = METRICS[metric].greater_is_better if metric in METRICS else True
    first, last = final["first"], final["final"]
    kept = [split_reason(s["proposal"]["reason"])[0] for s in record["steps"]
            if s["index"] > 0 and s["status"] == "completed" and s.get("accepted")]

    st.markdown("#### First and final Setup on the test part")
    st.caption("Not used for decisions. Every decision above was made on the training part; the test part "
               f"({final['n_test']} rows) is scored once, after the Session ended.")
    left, right = st.columns(2)
    exclusions = record.get("first_setup_exclusions") or []
    with left.container(border=True):
        st.metric(f"First Setup · {first['experiment_id']} · test {metric_name}", _score(first["test_score"]))
        start = "The Session's first Setup"
        start += f", without {', '.join(exclusions)}." if exclusions else "."
        st.caption(f"{_model_text(settings, first['experiment_id'])} {start}")
    with right.container(border=True):
        delta = None
        if first["test_score"] is not None and last["test_score"] is not None:
            delta = f"{last['test_score'] - first['test_score']:+.4f} vs first"
        st.metric(f"Final Setup · {last['experiment_id']} · test {metric_name}", _score(last["test_score"]),
                  delta=delta, delta_color="normal" if better_up else "inverse")
        changes = f"Changes kept: {'; '.join(kept)}." if kept else "No change was kept, so it is the first Setup."
        st.caption(f"{_model_text(settings, last['experiment_id'])} {changes}")

    baseline = final.get("baseline_metrics") or {}
    names = [m for m in first.get("test_metrics", {}) if first["test_metrics"].get(m) is not None]
    table = pd.DataFrame({
        "Metric": [metric_display_name(m, task) for m in names],
        "First": [first["test_metrics"].get(m) for m in names],
        "Final": [last.get("test_metrics", {}).get(m) for m in names],
        "Trivial baseline": [baseline.get(m) for m in names],
    })
    number = st.column_config.NumberColumn(format="%.4f")
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"First": number, "Final": number, "Trivial baseline": number})
    if baseline.get("strategy"):
        st.caption(f"Trivial baseline: {baseline['strategy']}.")
    notes = list(dict.fromkeys([*first.get("notes", []), *last.get("notes", [])]))
    if notes:
        with st.expander(f"Notes on the test part ({len(notes)})"):
            for note in notes:
                st.markdown(f"- {note}")


def _model_text(settings: Settings, experiment_id: str) -> str:
    summary = _read_json(settings.experiments_dir / experiment_id / "summary.json") or {}
    return f"Model: {summary.get('best_model_name') or 'not recorded'}."


# --------------------------------------------------------------------- section 5
def render_memory_records(settings: Settings, df: pd.DataFrame, target: str) -> None:
    memory = ExperimentMemory(settings.experiments_dir / "memory" / "memory.jsonl", read_only=True)
    fingerprint = dataset_fingerprint(df, target)
    records = [r for r in memory.records() if r.dataset_fingerprint == fingerprint]
    if memory.skipped_lines:
        st.warning(f"{memory.skipped_lines} damaged lines of {memory.path} were skipped.")
    if not records:
        st.info("No memory records for this dataset and target yet. A record is written after every Verdict "
                "of an autonomous run.")
        return
    table = pd.DataFrame([{
        "Session": r.session_id,
        "Experiment": r.experiment_id,
        "Strategy": r.strategy.get("strategy_id"),
        "Settings": json.dumps(r.strategy.get("params") or {}, sort_keys=True),
        "Diagnoses answered": ", ".join(f"{d['kind']} ({d['severity']})" for d in r.diagnoses) or "-",
        "Verdict": r.verdict,
        "Mean difference": r.mean_difference,
        "Spread (std)": r.std_difference,
        "Folds": r.n_folds,
        "Cost (s)": r.cost_seconds,
        "Written": r.created_at,
    } for r in reversed(records)])
    number = st.column_config.NumberColumn(format="%.4f")
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Mean difference": number, "Spread (std)": number,
                                "Cost (s)": st.column_config.NumberColumn(format="%.1f")})
    sessions = len({r.session_id for r in records})
    st.caption(f"{_count(len(records), 'record', 'records')} from {_count(sessions, 'Session', 'Sessions')}, "
               "newest first. The learned Strategy choices (thompson, ucb1) read the records of other, similar "
               "datasets when the memory is switched on in the advanced settings.")


# --------------------------------------------------------------------- helpers
def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
