"""Draw charts of one experiment's FLAML search from its saved trial history.

Reads experiments/<ID>/search_history.csv and summary.json (read-only) and writes three PNG files:

    <id>_search_timeline.png        every trial over time, with the best score so far
    <id>_time_per_learner.png       how much search time each learner family got
    <id>_best_learner_settings.png  how the winning family's size settings grew, trial by trial

Example, from the project root:

    python scripts/plot_search_history.py --experiment EXP-001 --out docs/imp/images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from app.services.evaluator import METRICS, metric_display_name  # noqa: E402
from app.services.task_detector import TaskType  # noqa: E402

# Settings that decide how big (and how slow) a tree model is.
SIZE_SETTINGS = ("n_estimators", "max_leaves", "num_leaves", "max_depth")
SIZE_LABELS = {
    "n_estimators": "number of trees",
    "max_leaves": "max leaves per tree",
    "num_leaves": "leaves per tree",
    "max_depth": "max depth",
}
VALIDATION_NAMES = {"cv": "5-fold cross-validation", "holdout": "holdout"}


def load(experiment_dir: Path) -> tuple[pd.DataFrame, dict]:
    history = pd.read_csv(experiment_dir / "search_history.csv")
    summary = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
    history["config"] = history["config"].apply(json.loads)
    return history.sort_values("trial", ignore_index=True), summary


def best_so_far(scores: pd.Series, greater_is_better: bool) -> pd.Series:
    return scores.cummax() if greater_is_better else scores.cummin()


def plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def plot_timeline(history: pd.DataFrame, summary: dict, path: Path) -> None:
    metric, task = summary["primary_metric"], TaskType(summary["task_type"])
    greater = METRICS[metric].greater_is_better
    scored = history.dropna(subset=["validation_score"]).sort_values("wall_clock_time", ignore_index=True)
    best = best_so_far(scored["validation_score"], greater)
    order = scored.groupby("model_name")["trial"].min().sort_values().index  # order FLAML first tried them

    fig, ax = plt.subplots(figsize=(10, 5.4))
    for model_name in order:
        group = scored[scored["model_name"] == model_name]
        ax.scatter(group["wall_clock_time"], group["validation_score"], s=16, alpha=0.65,
                   label=f"{model_name} ({plural(len(group), 'trial')})")
    ax.step(scored["wall_clock_time"], best, where="post", color="black", linewidth=2, label="best so far")
    top = scored.loc[best.idxmax() if greater else best.idxmin()]
    ax.annotate(f"best {top['validation_score']:.3f} at {top['wall_clock_time']:.1f} s",
                xy=(top["wall_clock_time"], top["validation_score"]),
                xytext=(0.55, 0.12), textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "black"})
    ax.set_xlabel("Seconds since the search started")
    method = VALIDATION_NAMES.get(summary["validation_method"], summary["validation_method"])
    ax.set_ylabel(f"Validation {metric_display_name(metric, task)} ({method})")
    ax.set_title(f"{summary['experiment_id']}: all {len(history)} trials FLAML ran "
                 f"({summary['dataset_name']}, {summary['time_budget']} s budget, seed {summary['random_seed']})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_time_per_learner(history: pd.DataFrame, summary: dict, path: Path) -> None:
    greater = METRICS[summary["primary_metric"]].greater_is_better
    agg = history.groupby("model_name").agg(
        seconds=("trial_time", "sum"),
        trials=("trial", "count"),
        best=("validation_score", "max" if greater else "min"),
    ).sort_values("seconds")
    total = agg["seconds"].sum()
    colours = ["#d62728" if name == summary["best_model_name"] else "#7f7f7f" for name in agg.index]

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.barh(agg.index, agg["seconds"], color=colours)
    for i, (name, row) in enumerate(agg.iterrows()):
        ax.text(row["seconds"] + total * 0.01, i,
                f"{row['seconds']:.1f} s, {plural(int(row['trials']), 'trial')}, best {row['best']:.3f}",
                va="center", fontsize=8)
    ax.set_xlim(0, agg["seconds"].max() * 1.45)
    ax.set_xlabel("Seconds spent training this family's trials")
    ax.set_title(f"{summary['experiment_id']}: where FLAML spent its search time "
                 f"({total:.1f} s of trials; winner in red)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_best_learner_settings(history: pd.DataFrame, summary: dict, path: Path) -> None:
    metric, task = summary["primary_metric"], TaskType(summary["task_type"])
    greater = METRICS[metric].greater_is_better
    rows = history[history["estimator"] == summary["best_estimator"]].sort_values("trial", ignore_index=True)
    sizes = [key for key in SIZE_SETTINGS if key in rows["config"].iloc[0]]
    x = range(1, len(rows) + 1)

    fig, (upper, lower) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    for key in sizes:
        upper.plot(x, [config.get(key) for config in rows["config"]], marker="o", markersize=3,
                   linewidth=1, label=SIZE_LABELS[key])
    upper.set_yscale("log")
    upper.set_ylabel("Value (log scale)")
    upper.legend(fontsize=8, loc="upper left")
    upper.grid(alpha=0.3)
    upper.set_title(f"{summary['experiment_id']}: {rows['model_name'].iloc[0]}'s {len(rows)} trials, "
                    "starting from the cheapest settings")

    scores = rows["validation_score"]
    best = best_so_far(scores, greater)
    improved = best.ne(best.shift())
    lower.scatter(x, scores, s=14, alpha=0.6, color="#1f77b4", label="trial score")
    lower.step(x, best, where="post", color="black", linewidth=2, label="best so far")
    lower.scatter([i + 1 for i in rows.index[improved]], scores[improved], s=40, color="#d62728",
                  zorder=3, label="new best")
    lower.set_xlabel(f"{rows['model_name'].iloc[0]} trial number")
    lower.set_ylabel(f"Validation {metric_display_name(metric, task)}")
    lower.legend(fontsize=8, loc="lower right")
    lower.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="EXP-001", help="experiment ID, for example EXP-001")
    parser.add_argument("--experiments-dir", default=str(ROOT / "experiments"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "imp" / "images"), help="folder for the PNG files")
    args = parser.parse_args()

    history, summary = load(Path(args.experiments_dir) / args.experiment)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = args.experiment.lower().replace("-", "")
    outputs = {
        "search_timeline": plot_timeline,
        "time_per_learner": plot_time_per_learner,
        "best_learner_settings": plot_best_learner_settings,
    }
    for name, draw in outputs.items():
        path = out / f"{prefix}_{name}.png"
        draw(history, summary, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
