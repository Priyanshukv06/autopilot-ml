"""Sections 5 and 6: results, leaderboard, search progress and explainability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from app.services.automl_runner import ExperimentResult
from app.services.evaluator import METRICS, metric_display_name
from app.services.task_detector import TaskType


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


# --------------------------------------------------------------------- section 5
def render_results(result: ExperimentResult) -> None:
    task = result.task_type
    metric_name = metric_display_name(result.primary_metric, task)
    st.subheader(f"{result.experiment_id} · {result.dataset_name} → {result.target_column} · {task.display_name}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", result.best_model_name, border=True, help=f"FLAML estimator key: {result.best_estimator}")
    c2.metric(
        f"Best score (test {metric_name})", _fmt(result.test_score), border=True,
        help=f"Primary metric on the held-out test set ({result.n_test} rows).",
    )
    c3.metric(
        f"Validation {metric_name}", _fmt(result.best_validation_score), border=True,
        help=f"FLAML's internal {result.validation_method} score of the best configuration on the training split.",
    )
    c4.metric(
        "Training time", f"{result.training_time:.1f} s", border=True,
        help=f"{result.n_trials} trials searched · {result.total_time:.1f}s total incl. evaluation and logging.",
    )

    if result.warnings:
        important = any(("misleading" in w or "imbalance" in w or "failed" in w) for w in result.warnings)
        with st.expander(f"Notes and warnings ({len(result.warnings)})", expanded=important):
            for warning in result.warnings:
                st.markdown(f"- {warning}")

    _render_metric_cards(result)
    _render_train_vs_test(result)
    _render_leaderboard(result, metric_name)
    _render_search_progress(result, metric_name)


def _render_metric_cards(result: ExperimentResult) -> None:
    task = result.task_type
    st.markdown("#### Test-set metrics")
    st.caption(
        f"Held-out {result.n_test} rows ({result.test_size:.0%} of the data), never used during the search. "
        f"Deltas compare with a trivial baseline that would {result.baseline_metrics.get('strategy', 'n/a')}."
    )
    keys = [k for k, v in result.test_metrics.items() if v is not None]
    for start in range(0, len(keys), 4):
        cols = st.columns(4)
        for col, key in zip(cols, keys[start : start + 4]):
            value = result.test_metrics[key]
            base = result.baseline_metrics.get(key)
            spec = METRICS[key]
            delta = None if base is None else f"{value - base:+.4f} vs baseline"
            col.metric(
                metric_display_name(key, task), _fmt(value), delta=delta, border=True,
                delta_color="normal" if spec.greater_is_better else "inverse",
            )


def _render_train_vs_test(result: ExperimentResult) -> None:
    task = result.task_type
    rows = []
    for key, test_value in result.test_metrics.items():
        train_value = result.train_metrics.get(key)
        if test_value is None or train_value is None:
            continue
        rows.append(
            {
                "Metric": metric_display_name(key, task),
                "Train": train_value,
                "Test": test_value,
                "Gap (train − test)": train_value - test_value,
            }
        )
    if not rows:
        return
    st.markdown("#### Train vs test")
    number = st.column_config.NumberColumn(format="%.4f")
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={"Train": number, "Test": number, "Gap (train − test)": number},
    )
    st.caption(
        "A large train/test gap can indicate overfitting. This mode only shows it; in the autonomous mode "
        "the Critic diagnoses it on the training part and the Strategy_Engine tries a fix."
    )


def _render_leaderboard(result: ExperimentResult, metric_name: str) -> None:
    st.markdown("#### Leaderboard")
    board = pd.DataFrame(result.leaderboard)
    if board.empty:
        st.info("No leaderboard information available.")
        return
    table = pd.DataFrame(
        {
            "Rank": board["rank"],
            "Model": board["model_name"],
            f"Best validation {metric_name}": board["best_validation_score"],
            "Trials": board["trials"],
        }
    )
    left, right = st.columns([3, 2])
    left.dataframe(
        table, hide_index=True, width="stretch",
        column_config={f"Best validation {metric_name}": st.column_config.NumberColumn(format="%.4f")},
    )
    evaluated = board.dropna(subset=["best_validation_score"])
    if not evaluated.empty:
        fig = px.bar(
            evaluated.iloc[::-1], x="best_validation_score", y="model_name", orientation="h",
            labels={"best_validation_score": f"Validation {metric_name}", "model_name": ""},
        )
        fig.update_layout(height=280, margin=dict(t=10, b=10))
        right.plotly_chart(fig, width="stretch")
    not_run = board[board["best_validation_score"].isna()]["model_name"].tolist()
    note = f" Not evaluated within the budget: {', '.join(not_run)}." if not_run else ""
    st.caption(
        f"FLAML's internal {result.validation_method} scores on the training split. Only the best model is "
        f"retrained and scored on the test set.{note}"
    )


def _render_search_progress(result: ExperimentResult, metric_name: str) -> None:
    history = pd.DataFrame(result.search_history)
    if history.empty or "validation_score" not in history:
        return
    history = history.dropna(subset=["validation_score"]).sort_values("wall_clock_time")
    if history.empty:
        return
    st.markdown("#### Search progress")
    greater = METRICS[result.primary_metric].greater_is_better
    scores = history["validation_score"]
    history["best_so_far"] = scores.cummax() if greater else scores.cummin()

    fig = px.scatter(
        history, x="wall_clock_time", y="validation_score", color="model_name", opacity=0.6,
        hover_data=["trial", "sample_size"],
        labels={"wall_clock_time": "Search time (s)", "validation_score": f"Validation {metric_name}",
                "model_name": "Model"},
    )
    fig.add_trace(
        go.Scatter(
            x=history["wall_clock_time"], y=history["best_so_far"], mode="lines", name="Best so far",
            line=dict(color="#111111", width=2, shape="hv"),
        )
    )
    # Keep the y-axis readable when a few early trials are extremely poor.
    lo, hi = (scores.quantile(0.05), scores.max()) if greater else (scores.min(), scores.quantile(0.95))
    pad = (hi - lo) * 0.1 or 0.01
    fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_layout(height=360, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Each point is one FLAML trial ({len(history)} shown). The line is the best score found so far.")


# --------------------------------------------------------------------- section 6
def render_explainability(result: ExperimentResult) -> None:
    task = result.task_type
    evaluation = result.test_evaluation
    names = ["Feature importance"]
    if evaluation is not None:
        names.append("Confusion matrix" if task.is_classification else "Errors and residuals")
        if task is TaskType.BINARY and evaluation.y_proba is not None:
            names.append("ROC curve")
    names.append("Best configuration")
    tabs = dict(zip(names, st.tabs(names)))

    with tabs["Feature importance"]:
        _render_feature_importance(result)
    if "Confusion matrix" in tabs:
        with tabs["Confusion matrix"]:
            _render_confusion_matrix(result)
    if "Errors and residuals" in tabs:
        with tabs["Errors and residuals"]:
            _render_residuals(result)
    if "ROC curve" in tabs:
        with tabs["ROC curve"]:
            _render_roc_curve(result)
    with tabs["Best configuration"]:
        st.markdown(f"**{result.best_model_name}** (`{result.best_estimator}`) · {result.engine} {result.engine_version}")
        st.json(result.best_config)
        st.caption(f"Model bundle saved to {result.model_path}")


def _render_feature_importance(result: ExperimentResult) -> None:
    metric_name = metric_display_name(result.primary_metric, result.task_type)
    importance = pd.DataFrame(result.feature_importance)
    if importance.empty:
        st.info("Feature importance is not available for this experiment.")
    else:
        top = importance.head(20).iloc[::-1]
        fig = px.bar(
            top, x="importance", y="feature", orientation="h", error_x="std",
            labels={"importance": f"Drop in {metric_name} when the column is shuffled", "feature": ""},
            title="Permutation importance (held-out test set)",
        )
        fig.update_layout(height=max(300, 26 * len(top) + 80), margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Model-agnostic: each input column is shuffled 5 times on the test set and the average drop in the "
            "primary metric is reported (error bar = std). Values near zero mean the model does not rely on it."
        )
    if result.native_feature_importance:
        with st.expander("Model-native importance (reported by the trained estimator)"):
            native = pd.DataFrame(result.native_feature_importance).head(20).iloc[::-1]
            fig = px.bar(
                native, x="importance_share", y="feature", orientation="h",
                labels={"importance_share": "Share of total importance", "feature": ""},
            )
            fig.update_layout(height=max(300, 26 * len(native) + 60), margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Computed by the estimator on FLAML's internally transformed features; units differ by model type.")
    st.caption("Planned: SHAP-based global and per-prediction explanations.")


def _render_confusion_matrix(result: ExperimentResult) -> None:
    evaluation = result.test_evaluation
    if evaluation is None or evaluation.confusion_matrix is None:
        st.info("Confusion matrix not available.")
        return
    labels = evaluation.labels or []
    fig = px.imshow(
        np.array(evaluation.confusion_matrix), x=labels, y=labels, text_auto=True,
        color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    left, right = st.columns([3, 2])
    left.plotly_chart(fig, width="stretch")
    if evaluation.per_class:
        report = pd.DataFrame(evaluation.per_class).T.rename_axis("class").reset_index()
        number = st.column_config.NumberColumn(format="%.3f")
        right.dataframe(
            report, hide_index=True, width="stretch",
            column_config={"precision": number, "recall": number, "f1-score": number},
        )
        right.caption("Per-class precision, recall and F1 on the test set.")


def _render_roc_curve(result: ExperimentResult) -> None:
    evaluation = result.test_evaluation
    positive = evaluation.positive_label
    y_true = np.array([str(v) == positive for v in evaluation.y_true], dtype=int)
    fpr, tpr, _ = roc_curve(y_true, evaluation.y_proba[:, 1])
    auc = result.test_metrics.get("roc_auc")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC = {_fmt(auc, 3)})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="gray")))
    fig.update_layout(
        xaxis_title="False positive rate", yaxis_title="True positive rate", height=420, margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Positive class: '{positive}'.")


def _render_residuals(result: ExperimentResult) -> None:
    frame = result.test_evaluation.predictions_frame()
    left, right = st.columns(2)
    lo, hi = float(min(frame["y_true"].min(), frame["y_pred"].min())), float(max(frame["y_true"].max(), frame["y_pred"].max()))
    fig = px.scatter(frame, x="y_true", y="y_pred", opacity=0.5, labels={"y_true": "Actual", "y_pred": "Predicted"},
                     title="Predicted vs actual")
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(height=400, margin=dict(t=40, b=10))
    left.plotly_chart(fig, width="stretch")

    fig = px.scatter(frame, x="y_pred", y="residual", opacity=0.5,
                     labels={"y_pred": "Predicted", "residual": "Residual (actual − predicted)"},
                     title="Residuals vs predicted")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400, margin=dict(t=40, b=10))
    right.plotly_chart(fig, width="stretch")

    fig = px.histogram(frame, x="residual", nbins=50, title="Residual distribution")
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
