"""AutoPilot-ML Streamlit dashboard: the single-experiment mode (Phase 1) and the autonomous mode.

Run from the project root:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file as a script; make the `app` package importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app import __version__  # noqa: E402
from app.components.autopilot_viewer import (  # noqa: E402
    render_memory_records,
    render_session_run,
    render_session_settings,
)
from app.components.dataset_profiler import (  # noqa: E402
    TargetSelection,
    render_dataset_profile,
    render_target_selection,
)
from app.components.dataset_uploader import LoadedDataset, render_dataset_uploader  # noqa: E402
from app.components.experiment_viewer import (  # noqa: E402
    LAST_RESULT_KEY,
    render_experiment_config,
    render_experiment_history,
    render_run_section,
)
from app.components.results_viewer import render_explainability, render_results  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402

SINGLE, AUTONOMOUS = "Single experiment", "Autonomous run"


def render_sidebar(settings: Settings) -> str:
    """The mode switch and the module statuses; returns the chosen mode."""
    with st.sidebar:
        st.markdown("## AutoPilot-ML")
        st.caption(f"v{__version__}")
        mode = st.radio("Mode", [SINGLE, AUTONOMOUS], key="mode",
                        help="Single experiment: one FLAML search, scored on the test part (the Phase 1 baseline). "
                        "Autonomous run: a Session that diagnoses, changes and judges Setups on its own.")
        st.markdown("**Implemented (current baseline)**")
        st.markdown(
            "- ✅ Dataset profiler\n- ✅ Task detection\n- ✅ FLAML AutoML search\n"
            "- ✅ Test-set evaluation + trivial baselines\n- ✅ Permutation feature importance\n"
            "- ✅ MLflow tracking (local)\n- ✅ Experiment history"
        )
        st.markdown("**Autonomous layer (Autonomous run mode, or `scripts/run_autopilot.py`)**")
        st.markdown(
            "- ✅ Experiment planner and session loop\n"
            "- ✅ Critic (12 diagnoses with thresholds from the critic study, plus unused words)\n"
            "- ✅ Strategy engine (12 strategies)\n- ✅ Judge (corrected t-test)\n- ✅ Resource manager\n"
            "- ✅ Experiment memory\n"
            "- ✅ Learned strategy choice from the memory (Thompson sampling, UCB1)\n"
            "- ✅ Optional strategies: SMOTE, target encoding, word features\n"
            "- ✅ Autonomous mode in this dashboard"
        )
        st.markdown("**Research tools (command line, `scripts/`)**")
        st.markdown(
            "- ✅ Benchmark against FLAML alone, AutoGluon and a manual baseline, with ablations\n"
            "- ✅ Critic, Judge and judge agreement studies"
        )
        st.markdown("**Planned, not implemented**")
        st.markdown("- 🕒 Optional extras: group folds, SHAP, target suggestion and leakage check, local "
                    "summaries, an API and Docker")
        st.divider()
        st.caption("Runs fully locally: no paid APIs, no cloud services.")
        st.caption(f"MLflow store: `{settings.mlflow_tracking_uri}`")
    return mode


def main() -> None:
    st.set_page_config(page_title="AutoPilot-ML", layout="wide")
    settings = get_settings()
    setup_logging(settings.log_level)
    mode = render_sidebar(settings)

    st.title("AutoPilot-ML")
    if mode == SINGLE:
        st.caption("Single experiment: one FLAML search on the training split, scored on the held-out test split.")
    else:
        st.caption("Autonomous run: a Session of Experiments. Every change is judged on the training part; "
                   "the test part is scored once, at the end.")

    st.header("1 · Dataset")
    dataset = render_dataset_uploader(settings)
    selection = None
    if dataset is not None:
        profile = render_dataset_profile(dataset)
        st.header("2 · Target")
        selection = render_target_selection(dataset, profile)
    else:
        st.info("Upload a CSV file or choose a bundled sample dataset to begin.")

    if mode == AUTONOMOUS:
        render_autonomous_mode(settings, dataset, selection)
    else:
        render_single_mode(settings, dataset, selection)


def render_single_mode(settings: Settings, dataset: LoadedDataset | None, selection: TargetSelection | None) -> None:
    if selection is not None:
        st.header("3 · Experiment Configuration")
        config = render_experiment_config(settings, selection, dataset)
        st.header("4 · Run AutoML")
        render_run_section(dataset.df, config, settings)

    result = st.session_state.get(LAST_RESULT_KEY)
    if result is not None:
        st.header("5 · Results")
        render_results(result)
        st.header("6 · Explainability")
        render_explainability(result)

    st.header("7 · Experiment History")
    render_experiment_history(settings)


def render_autonomous_mode(settings: Settings, dataset: LoadedDataset | None,
                           selection: TargetSelection | None) -> None:
    plan = None
    if selection is not None:
        st.header("3 · Session settings")
        plan = render_session_settings(settings, selection, dataset)
    st.header("4 · Autonomous run")
    render_session_run(dataset.df if selection is not None else None, plan, settings)
    if selection is not None:
        st.header("5 · Experiment memory for this dataset")
        render_memory_records(settings, dataset.df, selection.target)


if __name__ == "__main__":
    main()
