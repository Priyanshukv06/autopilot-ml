"""Analysis of the Phase 3 benchmark (spec task 26, Requirement 24): tables, tests and charts from the rows
of ``results/benchmark/benchmark_cells.csv``.

Fixed on 27 October 2026, before any benchmark cell had run:

- Rows: the last row of each cell; only completed cells are scored, and failed cells are listed apart.
- Blocks: a comparison of some systems at one budget uses only the (dataset, seed) pairs where every one of
  them completed, so each system is scored on the same rows; a dataset's score is the mean over those seeds.
  Scores are signed so that higher is better (``stat_tests.signed``).
- Families, each with the Friedman test, the Nemenyi critical difference (alpha 0.05) and Wilcoxon tests with
  Holm's correction over the family's pairs: ``main`` at 120 s and at 300 s (the 5 main systems; pairs: the
  full system against each of the other four, and the rule-order Session against FLAML alone), and
  ``ablations`` at 300 s (the full system and the four systems that each lack one part; pairs: the full system
  against each of them).
- Time-to-target: over the main systems, which ran at both budgets; the target is 99% of the best score any
  main system reached at either budget on that dataset and seed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from research import benchmark
from research import stat_tests as st

CONTROL = "autopilot_full"
FAMILIES: dict[str, dict[str, Any]] = {
    "main": {"systems": benchmark.MAIN_SYSTEMS,
             "pairs": [(CONTROL, "manual"), (CONTROL, "flaml"), (CONTROL, "autogluon"), (CONTROL, "autopilot_rules"),
                       ("autopilot_rules", "flaml")]},
    "ablations": {"systems": (CONTROL, "autopilot_rules", *benchmark.ABLATIONS),
                  "pairs": [(CONTROL, "autopilot_rules"), *[(CONTROL, a) for a in benchmark.ABLATIONS]]},
}
ALPHA = 0.05
TARGET_FRACTION = 0.99


# --------------------------------------------------------------------------- the rows
def scored(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """The last row of each completed cell, with ``signed`` (higher is better)."""
    last = [r for r in benchmark.latest(rows).values() if r.get("status") == "completed" and r.get("test_score") is not None]
    columns = ["cell_id", "system", "dataset", "metric", "budget_seconds", "seed", "test_score", "wall_seconds",
               "n_experiments", "signed"]
    if not last:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(last)
    frame["signed"] = [st.signed(s, m) for s, m in zip(frame["test_score"], frame["metric"])]
    return frame[columns]


def failures(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    failed = [r for r in benchmark.latest(rows).values() if r.get("status") != "completed"]
    return pd.DataFrame([{k: r.get(k) for k in ("cell_id", "system", "dataset", "budget_seconds", "seed", "error")}
                         for r in failed], columns=["cell_id", "system", "dataset", "budget_seconds", "seed", "error"])


def blocks(frame: pd.DataFrame, budget: int, systems: Sequence[str]) -> pd.DataFrame:
    """Signed scores, one row per (dataset, seed) where every system completed, one column per system."""
    part = frame[(frame["budget_seconds"] == budget) & frame["system"].isin(systems)]
    if part.empty:
        return pd.DataFrame(columns=list(systems))
    table = part.pivot_table(index=["dataset", "seed"], columns="system", values="signed", aggfunc="first")
    return table.reindex(columns=list(systems)).dropna()


def dataset_means(table: pd.DataFrame) -> pd.DataFrame:
    return table.groupby(level="dataset").mean() if len(table) else table


# --------------------------------------------------------------------------- the comparisons
def compare_family(frame: pd.DataFrame, family: str, budget: int) -> dict[str, Any]:
    """Friedman, Nemenyi and the family's pairs at one budget, on the datasets' means over complete blocks."""
    spec = FAMILIES[family]
    systems = list(spec["systems"])
    table = blocks(frame, budget, systems)
    means = dataset_means(table)
    n, k = len(means), len(systems)
    result: dict[str, Any] = {"family": family, "budget_seconds": budget, "systems": systems, "datasets": n,
                              "blocks": len(table), "friedman": None, "cd": None, "cliques": [], "pairs": [],
                              "block_counts": []}
    if n >= 2:
        result["friedman"] = st.friedman(means)
        result["cd"] = st.nemenyi_cd(k, n, ALPHA)
        result["cliques"] = st.cliques(result["friedman"].average_ranks, result["cd"])
    if n >= 1:
        result["pairs"] = st.compare_pairs(means, spec["pairs"])
        for a, b in spec["pairs"]:  # descriptive only: seeds of one dataset are not independent
            d = np.round((table[a] - table[b]).to_numpy(dtype=float), 12)
            result["block_counts"].append({"system_a": a, "system_b": b, "block_wins": int(np.sum(d > st.TIE)),
                                           "block_ties": int(np.sum(np.abs(d) <= st.TIE)),
                                           "block_losses": int(np.sum(d < -st.TIE))})
    return result


def family_tables(results: Sequence[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """The comparisons as three tables: tests, average ranks and pairs."""
    tests, rank_rows, pair_rows = [], [], []
    for r in results:
        f = r["friedman"]
        tests.append({"family": r["family"], "budget_seconds": r["budget_seconds"], "systems": len(r["systems"]),
                      "datasets": r["datasets"], "blocks": r["blocks"],
                      "friedman_chi2": f.statistic if f else None, "friedman_p": f.p_value if f else None,
                      "iman_davenport_f": f.iman_davenport if f else None,
                      "iman_davenport_p": f.iman_davenport_p if f else None, "nemenyi_cd": r["cd"],
                      "cliques": "; ".join(" = ".join(c) for c in r["cliques"])})
        for system, rank in (f.average_ranks.items() if f else []):
            rank_rows.append({"family": r["family"], "budget_seconds": r["budget_seconds"], "system": system,
                              "average_rank": rank})
        counts = {(c["system_a"], c["system_b"]): c for c in r["block_counts"]}
        for p in r["pairs"]:
            c = counts[(p.system_a, p.system_b)]
            pair_rows.append({"family": r["family"], "budget_seconds": r["budget_seconds"], "system_a": p.system_a,
                              "system_b": p.system_b, "datasets": p.n, "wins": p.wins, "ties": p.ties,
                              "losses": p.losses, "median_difference": p.median_difference, "ci_low": p.ci_low,
                              "ci_high": p.ci_high, "wilcoxon_statistic": p.statistic, "p_value": p.p_value,
                              "p_holm": p.p_holm, "rank_biserial": p.rank_biserial, "block_wins": c["block_wins"],
                              "block_ties": c["block_ties"], "block_losses": c["block_losses"]})
    return {"tests": pd.DataFrame(tests), "ranks": pd.DataFrame(rank_rows), "pairs": pd.DataFrame(pair_rows)}


def per_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Per budget, dataset and system: seeds completed, and the mean and spread of the test score (in the
    metric's own direction) and of the wall time."""
    if frame.empty:
        return pd.DataFrame(columns=["budget_seconds", "dataset", "metric", "system", "seeds", "mean_test_score",
                                     "sd_test_score", "mean_wall_seconds"])
    grouped = frame.groupby(["budget_seconds", "dataset", "metric", "system"], sort=True)
    out = grouped.agg(seeds=("seed", "nunique"), mean_test_score=("test_score", "mean"),
                      sd_test_score=("test_score", "std"), mean_wall_seconds=("wall_seconds", "mean")).reset_index()
    out["order"] = [benchmark._order(s) for s in out["system"]]
    return out.sort_values(["budget_seconds", "dataset", "order"]).drop(columns="order").reset_index(drop=True)


def median_time(times: Sequence[float]) -> float:
    """The smallest budget by which at least half of the runs reached the target, with a run that never
    reached it counted as later than every budget (inf when fewer than half reached it)."""
    ordered = sorted(math.inf if t is None or (isinstance(t, float) and math.isnan(t)) else float(t) for t in times)
    return ordered[math.ceil(len(ordered) / 2) - 1] if ordered else math.nan


def targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-to-target of the main systems per dataset and seed, and a summary per system."""
    main = frame[frame["system"].isin(benchmark.MAIN_SYSTEMS)].rename(columns={"signed": "score"})
    detail = st.time_to_target(main[["system", "dataset", "seed", "budget_seconds", "score"]], TARGET_FRACTION)
    budgets = sorted(main["budget_seconds"].unique()) if len(main) else []
    rows = []
    for system, group in detail.groupby("system", sort=False):
        row = {"system": system, "pairs": len(group), "never": int((~group["reached"]).sum()),
               "median_time_to_target": median_time(group["time_to_target"].tolist())}
        for budget in budgets:
            row[f"reached_by_{int(budget)}s"] = int((group["time_to_target"] <= budget).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if len(summary):
        summary["order"] = [benchmark._order(s) for s in summary["system"]]
        summary = summary.sort_values("order").drop(columns="order").reset_index(drop=True)
    return detail, summary


def target_pairs(detail: pd.DataFrame, pairs: Sequence[tuple[str, str]] = tuple(FAMILIES["main"]["pairs"])) -> pd.DataFrame:
    """For each pair of systems, over the (dataset, seed) pairs where both ran: how often a reached the target
    at an earlier budget than b, at the same one (or neither reached it), or later; never counts as later than
    every budget. A description, not a test: with two budgets most pairs tie."""
    times = detail.assign(t=detail["time_to_target"].fillna(math.inf)).pivot_table(
        index=["dataset", "seed"], columns="system", values="t", aggfunc="first")
    rows = []
    for a, b in pairs:
        if a not in times or b not in times:
            continue
        both = times[[a, b]].dropna()
        rows.append({"system_a": a, "system_b": b, "pairs": len(both), "a_earlier": int((both[a] < both[b]).sum()),
                     "same": int((both[a] == both[b]).sum()), "a_later": int((both[a] > both[b]).sum())})
    return pd.DataFrame(rows, columns=["system_a", "system_b", "pairs", "a_earlier", "same", "a_later"])


def analyse(rows: Sequence[dict[str, Any]], budgets: Iterable[int] = benchmark.BUDGETS,
            ablation_budget: int = benchmark.ABLATION_BUDGET) -> dict[str, Any]:
    """Every table of the analysis, and the comparisons for the charts."""
    frame = scored(rows)
    comparisons = [compare_family(frame, "main", b) for b in sorted(set(budgets))]
    comparisons.append(compare_family(frame, "ablations", ablation_budget))
    detail, summary = targets(frame)
    return {"frame": frame, "comparisons": comparisons, **family_tables(comparisons), "per_dataset": per_dataset(frame),
            "time_to_target": detail, "time_to_target_summary": summary, "time_to_target_pairs": target_pairs(detail),
            "failures": failures(rows), "timing": pd.DataFrame(benchmark.timing(rows))}


def write(result: dict[str, Any], folder: Path) -> list[Path]:
    """The tables as CSV files and the charts as PNG files in ``folder`` (charts in ``folder/figures``)."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("tests", "ranks", "pairs", "per_dataset", "time_to_target", "time_to_target_summary",
                 "time_to_target_pairs", "failures", "timing"):
        path = folder / f"analysis_{name}.csv"
        result[name].to_csv(path, index=False)
        paths.append(path)
    figures = folder / "figures"
    for r in result["comparisons"]:
        if r["friedman"] is not None:
            path = figures / f"cd_{r['family']}_{r['budget_seconds']}s.png"
            draw_cd(r["friedman"].average_ranks, r["cd"], r["cliques"],
                    f"{r['family']}, {r['budget_seconds']} s: {r['datasets']} datasets, CD = {r['cd']:.2f}", path)
            paths.append(path)
    if not result["per_dataset"].empty:
        path = figures / "score_against_budget.png"
        draw_score_against_budget(result["per_dataset"], path)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- charts
def draw_cd(average_ranks: dict[str, float], cd: float, cliques: Sequence[Sequence[str]], title: str,
            path: Path) -> None:
    """A critical-difference diagram in the style of Demšar (2006): the rank axis (1, the best, on the left),
    each system at its average rank, the CD as a bar, and thick lines joining systems not shown to differ."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(average_ranks, key=average_ranks.get)
    k = len(names)
    left, right = names[: math.ceil(k / 2)], names[math.ceil(k / 2):]
    height = max(len(left), len(right)) * 0.35 + 0.6 + 0.25 * len(cliques)
    fig, ax = plt.subplots(figsize=(8, 1.6 + height))
    ax.set_xlim(0.2, k + 0.8)
    ax.set_ylim(-(height + 0.2), 1.0)
    ax.axis("off")
    ax.hlines(0, 1, k, color="black", linewidth=1)
    for tick in range(1, k + 1):
        ax.vlines(tick, 0, 0.08, color="black", linewidth=1)
        ax.text(tick, 0.15, str(tick), ha="center", va="bottom", fontsize=9)
    ax.hlines(0.6, 1, 1 + cd, color="black", linewidth=2)
    ax.vlines([1, 1 + cd], 0.53, 0.67, color="black", linewidth=1)
    ax.text(1 + cd / 2, 0.7, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)
    for i, name in enumerate(left):
        y = -0.35 * (i + 1) - 0.25 * len(cliques)
        ax.plot([average_ranks[name], average_ranks[name], 0.9], [0, y, y], color="black", linewidth=0.8)
        ax.text(0.85, y, f"{name} ({average_ranks[name]:.2f})", ha="right", va="center", fontsize=9)
    for i, name in enumerate(reversed(right)):
        y = -0.35 * (i + 1) - 0.25 * len(cliques)
        ax.plot([average_ranks[name], average_ranks[name], k + 0.1], [0, y, y], color="black", linewidth=0.8)
        ax.text(k + 0.15, y, f"({average_ranks[name]:.2f}) {name}", ha="left", va="center", fontsize=9)
    for i, group in enumerate(cliques):
        y = -0.12 - 0.25 * i
        ranks_in = [average_ranks[n] for n in group]
        ax.hlines(y, min(ranks_in) - 0.03, max(ranks_in) + 0.03, color="black", linewidth=3)
    ax.set_title(title, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_score_against_budget(table: pd.DataFrame, path: Path) -> None:
    """One panel per dataset: each main system's mean test score at each budget."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = table[table["system"].isin(benchmark.MAIN_SYSTEMS)]
    names = sorted(main["dataset"].unique())
    columns = min(3, len(names)) or 1
    rows = math.ceil(len(names) / columns) or 1
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.2 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        part = main[main["dataset"] == name]
        for system in benchmark.MAIN_SYSTEMS:
            line = part[part["system"] == system].sort_values("budget_seconds")
            if len(line):
                ax.plot(line["budget_seconds"], line["mean_test_score"], marker="o", label=system)
        ax.set_title(f"{name} ({part['metric'].iloc[0]})", fontsize=10)
        ax.set_xticks(sorted(part["budget_seconds"].unique()))
        ax.set_xlabel("total budget (s)", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


__all__ = ["CONTROL", "FAMILIES", "analyse", "blocks", "compare_family", "draw_cd", "draw_score_against_budget",
           "failures", "per_dataset", "scored", "targets", "write"]
