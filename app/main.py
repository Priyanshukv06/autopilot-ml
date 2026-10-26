"""AutoPilot-ML Streamlit dashboard (Phase 1 baseline).

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
from app.components.dataset_profiler import render_dataset_profile, render_target_selection  # noqa: E402
from app.components.dataset_uploader import render_dataset_uploader  # noqa: E402
from app.components.experiment_viewer import (  # noqa: E402
    LAST_RESULT_KEY,
    render_experiment_config,
    render_experiment_history,
    render_run_section,
)
from app.components.results_viewer import render_explainability, render_results  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402


def render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.markdown("## AutoPilot-ML")
        st.caption(f"Phase 1 baseline · v{__version__}")
        st.markdown("**Implemented (current baseline)**")
        st.markdown(
            "- ✅ Dataset profiler\n- ✅ Task detection\n- ✅ FLAML AutoML search\n"
            "- ✅ Test-set evaluation + trivial baselines\n- ✅ Permutation feature importance\n"
            "- ✅ MLflow tracking (local)\n- ✅ Experiment history"
        )
        st.markdown("**Autonomous layer, first version (command line: `scripts/run_autopilot.py`)**")
        st.markdown(
            "- ✅ Experiment planner and session loop\n- ✅ Critic (3 diagnoses)\n"
            "- ✅ Strategy engine (3 strategies)\n- ✅ Judge (corrected t-test)\n- ✅ Resource manager"
        )
        st.markdown("**Planned, not implemented**")
        st.markdown(
            "- 🕒 Autonomous mode in this dashboard\n- 🕒 Full critic and strategies\n- 🕒 Experiment memory"
        )
        st.divider()
        st.caption("Runs fully locally: no paid APIs, no cloud services.")
        st.caption(f"MLflow store: `{settings.mlflow_tracking_uri}`")


def main() -> None:
    st.set_page_config(page_title="AutoPilot-ML", layout="wide")
    settings = get_settings()
    setup_logging(settings.log_level)
    render_sidebar(settings)

    st.title("AutoPilot-ML")
    st.caption("Autonomous Machine Learning Experimentation & Self-Optimization Framework · Phase 1 baseline")

    st.header("1 · Dataset")
    dataset = render_dataset_uploader(settings)
    if dataset is not None:
        profile = render_dataset_profile(dataset)

        st.header("2 · Target")
        selection = render_target_selection(dataset, profile)

        if selection is not None:
            st.header("3 · Experiment Configuration")
            config = render_experiment_config(settings, selection, dataset)

            st.header("4 · Run AutoML")
            render_run_section(dataset.df, config, settings)
    else:
        st.info("Upload a CSV file or choose a bundled sample dataset to begin.")

    result = st.session_state.get(LAST_RESULT_KEY)
    if result is not None:
        st.header("5 · Results")
        render_results(result)
        st.header("6 · Explainability")
        render_explainability(result)

    st.header("7 · Experiment History")
    render_experiment_history(settings)


if __name__ == "__main__":
    main()
