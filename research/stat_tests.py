"""Statistics for the Phase 3 benchmark (spec task 25, Requirement 24), following Demšar (2006).

The file is called ``stat_tests.py``, not ``statistics.py`` as design.md first said: the AutoGluon worker
runs as a script in ``research/``, so ``research/`` comes first on its import path, and a ``statistics.py``
there would hide Python's own ``statistics`` module and break AutoGluon's imports.

Systems are compared per budget over datasets: a dataset's score for a system is the mean over its seeds,
so each dataset is one independent observation (Requirement 24.1). Scores are first signed so that higher
is better (``signed``: error metrics are negated), and every function below assumes that.

- ``friedman``: the Friedman test on the ranks of the systems within each dataset
  (``scipy.stats.friedmanchisquare``, with its correction for ties), the average ranks, and the Iman and
  Davenport F statistic that Demšar recommends as less conservative.
- ``nemenyi_cd``: the Nemenyi critical difference CD = q_alpha * sqrt(k (k + 1) / (6 N)), where q_alpha is
  the studentized range quantile for k groups and infinite degrees of freedom divided by sqrt(2)
  (Demšar's Table 5). ``cliques`` lists the groups of systems whose average ranks differ by less than CD,
  the bars of a critical-difference diagram.
- ``compare_pairs``: for each pair of systems, the Wilcoxon signed-rank test over datasets
  (``scipy.stats.wilcoxon``, two-sided, zero differences dropped), Holm's correction over the pairs of one
  call, the matched-pairs rank-biserial correlation as effect size (Kerby, 2014), and a percentile bootstrap
  interval of the median difference (Requirement 24.2).
- ``time_to_target``: for each dataset and seed, the target is ``fraction`` (0.99) of the best score that
  any of the given systems reached at any budget; a system's time-to-target is the smallest budget at which
  it reached the target (Requirement 24.3).

With N datasets the Wilcoxon test cannot give a two-sided p-value below 2 / 2^N (0.031 for N = 6), so with
few datasets the effect sizes and intervals carry most of the information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from app.services.evaluator import METRICS

TIE = 1e-12  # differences at most this large count as ties


def signed(score: float | None, metric: str) -> float | None:
    """The score with its sign set so that higher is better (error metrics are negated)."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    return float(score) if METRICS[metric].greater_is_better else -float(score)


# --------------------------------------------------------------------------- Friedman and Nemenyi
def ranks(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank of each system (column) within each dataset (row): 1 is the best, tied systems share the mean rank."""
    values = scores.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("Every system needs a score on every dataset.")
    return pd.DataFrame(np.vstack([stats.rankdata(-row) for row in values]), index=scores.index,
                        columns=scores.columns)


@dataclass(frozen=True)
class FriedmanResult:
    n_datasets: int
    n_systems: int
    average_ranks: dict[str, float]  # best (lowest) first
    statistic: float  # chi-square, corrected for ties
    p_value: float
    iman_davenport: float  # F with (k - 1) and (k - 1)(N - 1) degrees of freedom
    iman_davenport_p: float


def friedman(scores: pd.DataFrame) -> FriedmanResult:
    """The Friedman test over the datasets (rows) for the systems (columns); at least 3 systems and 2 datasets."""
    n, k = scores.shape
    if k < 3 or n < 2:
        raise ValueError("The Friedman test needs at least 3 systems and 2 datasets.")
    rank_table = ranks(scores)
    average = rank_table.mean(axis=0).sort_values()
    values = scores.to_numpy(dtype=float)
    if np.all(np.abs(values - values[:, :1]) <= TIE):  # every system equal on every dataset
        chi2, p = 0.0, 1.0
    else:
        chi2, p = stats.friedmanchisquare(*values.T)
    f, f_p = iman_davenport(float(chi2), n, k)
    return FriedmanResult(n, k, {name: float(r) for name, r in average.items()}, float(chi2), float(p), f, f_p)


def friedman_statistic(average_ranks: Sequence[float], n: int) -> float:
    """Friedman's chi-square from the average ranks of k systems over n datasets, without the correction for
    ties (Demšar, 2006, section 3.2.2)."""
    k = len(average_ranks)
    return 12 * n / (k * (k + 1)) * (sum(r * r for r in average_ranks) - k * (k + 1) ** 2 / 4)


def iman_davenport(chi2: float, n: int, k: int) -> tuple[float, float]:
    """Iman and Davenport's F = (N - 1) chi2 / (N (k - 1) - chi2) and its p-value."""
    denominator = n * (k - 1) - chi2
    if denominator <= 0:  # every dataset ranks the systems the same way
        return math.inf, 0.0
    f = (n - 1) * chi2 / denominator
    return float(f), float(stats.f.sf(f, k - 1, (k - 1) * (n - 1)))


def nemenyi_q(k: int, alpha: float = 0.05) -> float:
    """The Nemenyi test's critical value for k systems (Demšar's Table 5a)."""
    if k < 2:
        raise ValueError("The Nemenyi test needs at least 2 systems.")
    return float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2))


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference of average ranks for k systems over n datasets."""
    return nemenyi_q(k, alpha) * math.sqrt(k * (k + 1) / (6 * n))


def cliques(average_ranks: dict[str, float], cd: float) -> list[tuple[str, ...]]:
    """The largest runs of systems (in rank order) whose average ranks differ by less than ``cd``, leaving out
    runs inside another run and single systems: the bars of a critical-difference diagram."""
    names = sorted(average_ranks, key=average_ranks.get)
    runs = []
    for start in range(len(names)):
        end = start
        while end + 1 < len(names) and average_ranks[names[end + 1]] - average_ranks[names[start]] < cd:
            end += 1
        if end > start and not any(s <= start and end <= e for s, e in runs):
            runs.append((start, end))
    return [tuple(names[s:e + 1]) for s, e in runs]


# --------------------------------------------------------------------------- pairs of systems
@dataclass(frozen=True)
class PairComparison:
    system_a: str
    system_b: str
    n: int  # datasets
    wins: int  # datasets where a is better
    ties: int
    losses: int
    median_difference: float  # median over datasets of a - b (positive: a is better)
    ci_low: float  # bootstrap interval of the median difference
    ci_high: float
    statistic: float | None  # Wilcoxon's smaller rank sum; None when every difference is zero
    p_value: float
    p_holm: float
    rank_biserial: float  # from -1 (b always better) to 1 (a always better)


def compare_pairs(scores: pd.DataFrame, pairs: Iterable[tuple[str, str]], confidence: float = 0.95,
                  n_resamples: int = 10_000, seed: int = 0) -> list[PairComparison]:
    """Wilcoxon signed-rank tests for the pairs (columns of ``scores``, rows are datasets), with Holm's
    correction over these pairs, rank-biserial correlations and bootstrap intervals."""
    pairs = list(pairs)
    raw = []
    for a, b in pairs:
        d = _rounded((scores[a] - scores[b]).to_numpy(dtype=float))
        if np.isnan(d).any():
            raise ValueError(f"{a} and {b} need scores on the same datasets.")
        nonzero = d[np.abs(d) > TIE]
        if len(nonzero) == 0:
            statistic, p = None, 1.0
        else:
            test = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
            statistic, p = float(test.statistic), float(test.pvalue)
        low, high = bootstrap_median_interval(d, confidence, n_resamples, seed)
        raw.append((a, b, d, statistic, p, low, high))
    adjusted = holm([p for *_, p, _, _ in raw])
    return [PairComparison(a, b, len(d), int(np.sum(d > TIE)), int(np.sum(np.abs(d) <= TIE)), int(np.sum(d < -TIE)),
                           float(np.median(d)), low, high, statistic, p, p_adj, rank_biserial(d))
            for (a, b, d, statistic, p, low, high), p_adj in zip(raw, adjusted)]


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjusted p-values, in the order given."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted, running = [0.0] * m, 0.0
    for step, i in enumerate(order):
        running = max(running, min(1.0, (m - step) * p_values[i]))
        adjusted[i] = running
    return adjusted


def rank_biserial(differences: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation (W+ - W-) / (W+ + W-) over the non-zero differences, ranked by
    size with tied sizes sharing the mean rank (Kerby, 2014); 0 when every difference is zero."""
    d = _rounded(np.asarray(differences, dtype=float))
    d = d[np.abs(d) > TIE]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    plus, minus = r[d > 0].sum(), r[d < 0].sum()
    return float((plus - minus) / (plus + minus))


def _rounded(differences: np.ndarray) -> np.ndarray:
    """Differences rounded to 12 decimals, so that differences equal up to floating-point noise (0.9 - 0.85
    and 0.95 - 0.9) share a rank instead of being ranked by the noise."""
    return np.round(differences, 12)


def bootstrap_median_interval(values: Sequence[float], confidence: float = 0.95, n_resamples: int = 10_000,
                              seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval of the median: resample the values with replacement ``n_resamples`` times
    (generator seeded with ``seed``) and take the middle ``confidence`` share of the medians."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        raise ValueError("No values.")
    rng = np.random.default_rng(seed)
    medians = np.median(v[rng.integers(0, len(v), size=(n_resamples, len(v)))], axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(medians, [tail, 100 - tail])
    return float(low), float(high)


# --------------------------------------------------------------------------- time-to-target
def target_of(best: float, fraction: float = 0.99) -> float:
    """``fraction`` of the best score; for a negative best (an R2 below 0, or a negated error) the target is
    the same distance below it, so the best score always reaches its own target."""
    return best - (1 - fraction) * abs(best)


def time_to_target(scores: pd.DataFrame, fraction: float = 0.99) -> pd.DataFrame:
    """``scores``: one row per finished run with columns system, dataset, seed, budget_seconds and score
    (signed, higher is better). For each dataset and seed, the target comes from the best score of any system
    at any budget; each system's time-to-target is the smallest budget at which its score reached the target
    (NaN if it never did). One row per system, dataset and seed."""
    needed = {"system", "dataset", "seed", "budget_seconds", "score"}
    if missing := needed - set(scores.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = scores.dropna(subset=["score"])
    rows = []
    for (dataset, seed), group in frame.groupby(["dataset", "seed"], sort=True):
        best = float(group["score"].max())
        target = target_of(best, fraction)
        for system, runs in group.groupby("system", sort=True):
            reached = runs.loc[runs["score"] >= target, "budget_seconds"]
            rows.append({"system": system, "dataset": dataset, "seed": seed, "best_score": best, "target": target,
                         "time_to_target": float(reached.min()) if len(reached) else math.nan,
                         "reached": bool(len(reached))})
    return pd.DataFrame(rows, columns=["system", "dataset", "seed", "best_score", "target", "time_to_target",
                                       "reached"])


__all__ = ["FriedmanResult", "PairComparison", "bootstrap_median_interval", "cliques", "compare_pairs", "friedman",
           "friedman_statistic", "holm", "iman_davenport", "nemenyi_cd", "nemenyi_q", "rank_biserial", "ranks",
           "signed", "target_of", "time_to_target"]
