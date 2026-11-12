"""Section 1 (profile) and Section 2 (target selection + task detection)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.dataset_uploader import LoadedDataset
from app.services.profiler import LEAKAGE_REPORT_SCORE, DatasetProfile, DatasetProfiler
from app.services.task_detector import TaskDetectionError, TaskDetectionResult, TaskDetector, TaskType


@dataclass
class TargetSelection:
    target: str
    task_type: TaskType
    detection: TaskDetectionResult
    profile: DatasetProfile  # profile computed with the target and task
    excluded_columns: list[str]
    warnings: list[str]


@st.cache_data(show_spinner="Profiling dataset...")
def compute_profile(
    df: pd.DataFrame, dataset_name: str, target: str | None = None, task_value: str | None = None
) -> DatasetProfile:
    task_type = TaskType(task_value) if task_value else None
    # with a target and a task, also the single-column check, so a possible leak is shown before any run
    return DatasetProfiler().profile(df, target_column=target, dataset_name=dataset_name, task_type=task_type,
                                     check_leakage=task_type is not None)


@st.cache_data(show_spinner=False)
def compute_target_candidates(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([asdict(c) for c in TaskDetector().target_candidates(df)])


# --------------------------------------------------------------------------- profile
def render_dataset_profile(dataset: LoadedDataset) -> DatasetProfile:
    df = dataset.df
    profile = compute_profile(df, dataset.name)

    cards = st.columns(6)
    cards[0].metric("Rows", f"{profile.rows:,}", border=True)
    cards[1].metric("Columns", profile.columns, border=True)
    cards[2].metric(
        "Missing values", f"{profile.total_missing:,}", border=True,
        help=f"{profile.total_missing_ratio:.1%} of all cells",
    )
    cards[3].metric("Duplicate rows", f"{profile.duplicate_rows:,}", border=True)
    cards[4].metric("Numerical features", len(profile.numerical_features), border=True)
    cards[5].metric("Categorical features", len(profile.categorical_features), border=True)

    tab_columns, tab_preview, tab_quality, tab_targets = st.tabs(
        ["Columns", "Preview", f"Data quality ({len(profile.warnings)})", "Target candidates"]
    )
    with tab_columns:
        st.dataframe(profile.column_table(), hide_index=True, width="stretch")
        missing = {name: ratio for name, ratio in profile.missing_ratio.items() if ratio > 0}
        if missing:
            fig = px.bar(
                x=list(missing), y=[100 * r for r in missing.values()],
                labels={"x": "Column", "y": "Missing (%)"}, title="Missing values by column",
            )
            fig.update_layout(height=320, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Numerical: {', '.join(profile.numerical_features) or '—'}  \n"
            f"Categorical: {', '.join(profile.categorical_features) or '—'}"
        )
    with tab_preview:
        st.dataframe(df.head(100), width="stretch")
        st.caption(f"First {min(100, len(df))} of {len(df):,} rows · {profile.memory_mb:.2f} MB in memory")
    with tab_quality:
        if profile.warnings:
            for warning in profile.warnings:
                st.warning(warning, icon="⚠️")
        else:
            st.success("No data-quality issues detected by the baseline checks.")
        st.caption("Rule-based checks. In the autonomous mode the Critic turns many of these problems into "
                   "Diagnoses and the Strategy_Engine proposes fixes.")
    with tab_targets:
        st.dataframe(
            compute_target_candidates(df),
            hide_index=True,
            width="stretch",
            column_config={
                "missing_ratio": st.column_config.NumberColumn("missing", format="percent"),
                "inferred_task": "task if selected",
            },
        )
        suggestions = TaskDetector().suggest_targets(df)
        if suggestions:
            st.markdown("Likely targets: " + "; ".join(s.text() for s in suggestions) + ".")
        st.caption("Suggestions from simple signals (the name, the position, the values); check them and choose "
                   "the target column below.")
    return profile


# --------------------------------------------------------------------------- target
def render_target_selection(dataset: LoadedDataset, profile: DatasetProfile) -> TargetSelection | None:
    df = dataset.df
    detector = TaskDetector()
    columns = list(df.columns)
    default = dataset.suggested_target if dataset.suggested_target in columns else detector.suggest_target(df)

    left, right = st.columns(2)
    target = left.selectbox(
        "Target column", columns, index=columns.index(default), key=f"target::{dataset.key}",
        help="The column the model should predict.",
    )
    try:
        detection = detector.detect(df[target])
    except TaskDetectionError as exc:
        st.error(f"'{target}' cannot be used as a target: {exc}")
        return None

    with right:
        st.markdown("Detected problem")
        st.markdown(f"#### {detection.task_type.display_name}")
        st.caption(f"Confidence: {detection.confidence} · {detection.reason}")

    task_options = list(TaskType)
    task_type = left.selectbox(
        "Problem type (override if needed)",
        task_options,
        index=task_options.index(detection.task_type),
        format_func=lambda t: t.display_name,
        key=f"task::{dataset.key}::{target}",
    )
    try:
        warnings = detector.validate(df[target], task_type)
    except TaskDetectionError as exc:
        st.error(str(exc))
        return None
    if task_type is not detection.task_type:
        st.info(f"Task overridden: {detection.task_type.display_name} → {task_type.display_name}")
    for warning in warnings:
        st.warning(warning)

    target_profile = compute_profile(df, dataset.name, target, task_type.value)
    _render_target_distribution(df[target], target_profile, task_type)
    metric = (target_profile.single_column_metric or "score").replace("_", " ")
    for column, score in target_profile.single_column_scores.items():
        if score >= LEAKAGE_REPORT_SCORE:  # possible leakage (optional task 33): a warning, nothing is changed
            st.warning(f"'{column}' alone predicts '{target}' almost perfectly ({metric} {score:.3f} in "
                       "cross-validation on all labelled rows). If its value is only known after the outcome, "
                       "exclude it below.", icon="⚠️")

    options = [c for c in columns if c != target]
    default_excluded = [c for c in profile.suggested_exclusions() if c in options]
    excluded = st.multiselect(
        "Columns to exclude from training",
        options,
        default=default_excluded,
        key=f"exclude::{dataset.key}::{target}",
        help="Identifier-like and constant columns are pre-selected because they cannot generalise. "
        "High-cardinality columns are only flagged (see the Data quality tab).",
    )
    if len(excluded) == len(options):
        st.error("At least one feature column must remain.")
        return None
    return TargetSelection(target, task_type, detection, target_profile, list(excluded), warnings)


def _render_target_distribution(target: pd.Series, profile: DatasetProfile, task_type: TaskType) -> None:
    if task_type.is_classification and profile.class_distribution:
        dist = pd.DataFrame(
            {"class": list(profile.class_distribution), "count": list(profile.class_distribution.values())}
        )
        fig = px.bar(dist, x="class", y="count", text="count", title=f"Class distribution of '{target.name}'")
        fig.update_xaxes(type="category")
        fig.update_layout(height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        balance = profile.class_balance
        if balance is not None:
            message = (
                f"Majority class '{balance.majority_class}' = {balance.majority_ratio:.1%}, minority class "
                f"'{balance.minority_class}' = {balance.minority_ratio:.1%} (ratio {balance.imbalance_ratio:.2f}:1, "
                f"imbalance: {balance.severity})."
            )
            if balance.severity in ("moderate", "severe"):
                st.warning(message + " Accuracy can be misleading on this target.")
            else:
                st.caption(message)
    elif task_type is TaskType.REGRESSION:
        fig = px.histogram(target.dropna(), nbins=50, title=f"Distribution of '{target.name}'")
        fig.update_layout(height=300, margin=dict(t=40, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")
        if profile.target_summary:
            s = profile.target_summary
            st.caption(
                f"mean {s['mean']:.4g} · std {s['std']:.4g} · min {s['min']:.4g} · "
                f"median {s['50%']:.4g} · max {s['max']:.4g} · skew {s['skew']:.2f}"
            )
