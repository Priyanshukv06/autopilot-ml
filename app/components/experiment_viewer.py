"""Sections 3, 4 and 7: experiment configuration, run with progress, experiment history."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.dataset_profiler import TargetSelection
from app.components.dataset_uploader import LoadedDataset
from app.core.config import Settings
from app.core.logging_config import get_logger
from app.services.automl_runner import AutoMLRunner, ExperimentConfig, ExperimentError, ExperimentResult
from app.services.evaluator import PRIMARY_METRICS, metric_display_name, metric_warnings
from app.services.experiment_tracker import ExperimentTracker

logger = get_logger(__name__)

ACTIVE_RUN_KEY = "active_run"
LAST_RESULT_KEY = "last_result"
_POLL_SECONDS = 0.5


# --------------------------------------------------------------------- section 3
def render_experiment_config(settings: Settings, selection: TargetSelection, dataset: LoadedDataset) -> ExperimentConfig:
    task = selection.task_type
    c1, c2, c3, c4 = st.columns(4)
    time_budget = c1.number_input(
        "Time budget (seconds)",
        min_value=settings.min_time_budget,
        max_value=settings.max_time_budget,
        value=settings.default_time_budget,
        step=10,
        help="Wall-clock budget for the FLAML search. Retraining the best model can add a few seconds.",
    )
    metric = c2.selectbox(
        "Primary metric",
        list(PRIMARY_METRICS[task]),
        format_func=lambda m: metric_display_name(m, task),
        key=f"metric::{task.value}",
        help="The metric FLAML optimises and the headline score of the experiment.",
    )
    seed = c3.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=settings.default_random_seed, step=1)
    test_size = c4.slider(
        "Test split",
        min_value=0.1,
        max_value=0.4,
        value=min(max(float(settings.default_test_size), 0.1), 0.4),
        step=0.05,
        help="Share of rows held out and never seen during the search.",
    )
    for warning in metric_warnings(task, metric, selection.profile.class_balance):
        st.warning(warning)

    return ExperimentConfig(
        target_column=selection.target,
        task_type=task,
        primary_metric=metric,
        time_budget=int(time_budget),
        random_seed=int(seed),
        test_size=float(test_size),
        excluded_columns=list(selection.excluded_columns),
        dataset_name=dataset.name,
        n_jobs=settings.n_jobs,
    )


# --------------------------------------------------------------------- section 4
@dataclass
class _ActiveRun:
    """A running experiment. Lives in session state so it survives Streamlit reruns."""

    thread: threading.Thread
    config: ExperimentConfig
    lock: threading.Lock
    state: dict[str, Any]
    outcome: dict[str, Any] = field(default_factory=dict)


def _start_run(df: pd.DataFrame, config: ExperimentConfig, settings: Settings) -> _ActiveRun:
    lock = threading.Lock()
    state: dict[str, Any] = {
        "stage": "starting", "fraction": 0.0, "message": "Starting experiment", "since": time.monotonic(),
    }
    outcome: dict[str, Any] = {}

    def on_progress(stage: str, fraction: float, message: str) -> None:
        # Runs in the worker thread: only record state, never call Streamlit here.
        with lock:
            state.update(stage=stage, fraction=fraction, message=message, since=time.monotonic())

    def worker() -> None:
        try:
            outcome["result"] = AutoMLRunner(settings=settings).run_experiment(df, config, on_progress)
        except Exception as exc:
            logger.exception("Experiment failed")
            outcome["error"] = exc

    thread = threading.Thread(target=worker, name="autopilot-experiment", daemon=True)
    active = _ActiveRun(thread=thread, config=config, lock=lock, state=state, outcome=outcome)
    thread.start()
    return active


def _wait_for_run(active: _ActiveRun) -> None:
    budget = active.config.time_budget
    progress = st.progress(0.0, text="Starting experiment...")
    while active.thread.is_alive():
        with active.lock:
            snapshot = dict(active.state)
        fraction, text = snapshot["fraction"], snapshot["message"]
        if snapshot["stage"] == "searching":
            elapsed = time.monotonic() - snapshot["since"]
            fraction = 0.05 + 0.80 * min(elapsed / budget, 1.0)
            text = f"{text} · {min(elapsed, budget):.0f}s / {budget}s"
            if elapsed > budget:
                text += " · retraining the best model"
        progress.progress(min(fraction, 0.99), text=text)
        time.sleep(_POLL_SECONDS)
    active.thread.join()
    st.session_state.pop(ACTIVE_RUN_KEY, None)

    error = active.outcome.get("error")
    if error is not None:
        progress.empty()
        if isinstance(error, (ExperimentError, ValueError)):
            st.error(f"Experiment failed: {error}")
        else:
            st.error(f"Unexpected error: {error}")
            with st.expander("Error details"):
                st.exception(error)
        return

    result: ExperimentResult = active.outcome["result"]
    st.session_state[LAST_RESULT_KEY] = result
    progress.progress(1.0, text=f"{result.experiment_id} finished in {result.total_time:.1f}s")
    st.success(f"{result.experiment_id} completed · best model: {result.best_model_name}")


def render_run_section(df: pd.DataFrame, config: ExperimentConfig, settings: Settings) -> ExperimentResult | None:
    active: _ActiveRun | None = st.session_state.get(ACTIVE_RUN_KEY)
    running = active is not None and active.thread.is_alive()

    st.caption(
        f"FLAML will search for up to {config.time_budget}s on '{config.dataset_name}' → "
        f"'{config.target_column}' ({config.task_type.display_name}), optimising "
        f"{metric_display_name(config.primary_metric, config.task_type)}."
    )
    clicked = st.button("Run Experiment", type="primary", width="stretch", disabled=running)
    if clicked and not running:
        active = _start_run(df.copy(), config, settings)
        st.session_state[ACTIVE_RUN_KEY] = active
    if active is not None:
        if running and not clicked:
            st.info(f"An experiment on '{active.config.dataset_name}' is still running; progress resumes below.")
        _wait_for_run(active)
    return st.session_state.get(LAST_RESULT_KEY)


# --------------------------------------------------------------------- section 7
def render_experiment_history(settings: Settings) -> None:
    tracker = ExperimentTracker.from_settings(settings)
    history = tracker.history_frame()
    if history.empty:
        st.info("No experiments yet. Every run is recorded here and in MLflow.")
        return

    history = history.sort_values("Experiment", ascending=False, ignore_index=True)
    score_format = st.column_config.NumberColumn(format="%.4f")
    st.dataframe(
        history,
        hide_index=True,
        width="stretch",
        column_config={
            "Test score": score_format,
            "Validation score": score_format,
            "Train score": score_format,
            "Training time (s)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "Test score = primary metric on the held-out split. Validation score = FLAML's internal "
        "cross-validation / holdout score. Scores are only comparable within the same dataset and metric."
    )

    chart_data = history.dropna(subset=["Test score"]).sort_values("Experiment")
    if len(chart_data) > 1:
        fig = px.line(
            chart_data, x="Experiment", y="Test score", color="Dataset", line_dash="Metric", markers=True,
            hover_data=["Model", "Metric", "Budget (s)", "Training time (s)"],
            title="Test score per experiment",
        )
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Open the full run details in MLflow"):
        st.code(
            f'mlflow ui --backend-store-uri "{settings.mlflow_tracking_uri}" --port 5000 --workers 1',
            language="bash",
        )
        st.caption(f"Then open http://127.0.0.1:5000 · local index: {settings.experiments_dir}")
