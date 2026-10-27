"""Tests for the benchmark statistics (spec task 25.1, Requirement 24.5), against reference values from
SciPy's documentation, from Demšar (2006), or worked out by hand."""

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from research import stat_tests as st

# SciPy's wilcoxon example (Darwin's plant heights, from Fisher's "Design of Experiments"):
# statistic 24, exact two-sided p-value 0.041259765625.
DARWIN = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]
# Demšar (2006), Table 5(a): two-tailed Nemenyi critical values for k = 2 to 10 systems.
Q_05 = [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]
Q_10 = [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920]


def test_signed_scores_make_higher_better():
    assert st.signed(0.8, "f1") == 0.8 and st.signed(0.8, "r2") == 0.8
    assert st.signed(2.5, "rmse") == -2.5 and st.signed(0.3, "log_loss") == -0.3
    assert st.signed(None, "f1") is None and st.signed(float("nan"), "mae") is None


# --------------------------------------------------------------------------- Friedman and Nemenyi
def test_ranks_within_each_dataset():
    table = pd.DataFrame({"a": [0.9, 0.7], "b": [0.9, 0.8], "c": [0.8, 0.9]}, index=["d1", "d2"])
    assert st.ranks(table).to_numpy().tolist() == [[1.5, 1.5, 3.0], [3.0, 2.0, 1.0]]  # ties share the mean rank
    with pytest.raises(ValueError, match="every dataset"):
        st.ranks(table.assign(c=[0.8, np.nan]))


def test_the_friedman_statistic_of_demsars_example():
    # Section 3.2.4: four classifiers on 14 datasets with average ranks 3.143, 2.000, 2.893 and 1.964;
    # chi2_F = 9.28 and F_F = 3.69 with 3 and 39 degrees of freedom.
    chi2 = st.friedman_statistic([3.143, 2.000, 2.893, 1.964], 14)
    f, p = st.iman_davenport(chi2, 14, 4)
    assert chi2 == pytest.approx(9.28, abs=0.005) and f == pytest.approx(3.69, abs=0.005)
    assert p == pytest.approx(stats.f.sf(f, 3, 39)) and p < 0.05


def test_friedman_on_a_table_worked_by_hand():
    table = pd.DataFrame({"a": [0.9, 0.8, 0.7, 0.95], "b": [0.85, 0.82, 0.6, 0.9], "c": [0.7, 0.75, 0.65, 0.8]})
    result = st.friedman(table)
    # ranks a: 1, 2, 1, 1; b: 2, 1, 3, 2; c: 3, 3, 2, 3 -> means 1.25, 2.0, 2.75;
    # chi2 = 12 * 4 / (3 * 4) * (1.25^2 + 2^2 + 2.75^2 - 3 * 4^2 / 4) = 4.5
    assert result.average_ranks == {"a": 1.25, "b": 2.0, "c": 2.75}
    assert result.statistic == pytest.approx(4.5) and result.p_value == pytest.approx(math.exp(-4.5 / 2))
    assert result.statistic == pytest.approx(st.friedman_statistic([1.25, 2.0, 2.75], 4))  # no ties
    assert result.iman_davenport == pytest.approx(3 * 4.5 / (4 * 2 - 4.5))
    equal = st.friedman(pd.DataFrame({"a": [0.5, 0.6], "b": [0.5, 0.6], "c": [0.5, 0.6]}))
    assert equal.statistic == 0.0 and equal.p_value == 1.0
    with pytest.raises(ValueError, match="3 systems"):
        st.friedman(table[["a", "b"]])


def test_nemenyi_critical_values_and_difference_match_demsar():
    for k, (q05, q10) in enumerate(zip(Q_05, Q_10), start=2):
        assert st.nemenyi_q(k, 0.05) == pytest.approx(q05, abs=1e-3)
        assert st.nemenyi_q(k, 0.10) == pytest.approx(q10, abs=1e-3)
    # Section 3.2.4: CD = 2.569 * sqrt(4 * 5 / (6 * 14)) = 1.25 at alpha 0.05, and 1.12 at 0.10.
    assert st.nemenyi_cd(4, 14) == pytest.approx(1.25, abs=0.005)
    assert st.nemenyi_cd(4, 14, 0.10) == pytest.approx(1.12, abs=0.005)
    with pytest.raises(ValueError):
        st.nemenyi_q(1)


def test_cliques_are_the_bars_of_a_critical_difference_diagram():
    average = {"A": 1.0, "B": 1.5, "C": 2.4, "D": 3.9}
    assert st.cliques(average, 1.0) == [("A", "B"), ("B", "C")]
    assert st.cliques(average, 3.0) == [("A", "B", "C", "D")]
    assert st.cliques(average, 0.4) == []
    assert st.cliques({"A": 1.0, "B": 2.0}, 1.0) == []  # a difference equal to CD is significant


# --------------------------------------------------------------------------- pairs
def test_the_wilcoxon_test_of_scipys_example():
    table = pd.DataFrame({"a": DARWIN, "b": [0] * len(DARWIN)}, dtype=float)
    [pair] = st.compare_pairs(table, [("a", "b")], n_resamples=2_000)
    assert pair.statistic == 24.0 and pair.p_value == pytest.approx(0.041259765625)
    # the same p-value from all 2^15 sign patterns: the share with a rank sum at most 24, times 2
    r = stats.rankdata(np.abs(DARWIN))
    count = sum(1 for signs in itertools.product((0, 1), repeat=15) if sum(x for x, s in zip(r, signs) if s) <= 24)
    assert 2 * count / 2 ** 15 == pytest.approx(pair.p_value)
    assert (pair.n, pair.wins, pair.ties, pair.losses) == (15, 13, 0, 2)
    assert pair.median_difference == 24.0 and pair.p_holm == pair.p_value  # one pair: nothing to correct
    assert pair.rank_biserial == pytest.approx((96 - 24) / 120)  # W+ = 120 - 24


def test_holm_adjusted_p_values_worked_by_hand():
    # sorted 0.005, 0.01, 0.03, 0.04 -> 4 x 0.005 = 0.02, 3 x 0.01 = 0.03, 2 x 0.03 = 0.06, max(0.06, 0.04) = 0.06
    assert st.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert st.holm([0.6, 0.9]) == pytest.approx([1.0, 1.0]) and st.holm([]) == []


def test_pairs_with_zero_differences_and_holm_over_the_pairs():
    table = pd.DataFrame({"a": [0.9, 0.8, 0.7, 0.95, 0.6, 0.85], "b": [0.85, 0.8, 0.75, 0.9, 0.5, 0.8],
                          "c": [0.9, 0.8, 0.7, 0.95, 0.6, 0.85]})
    ab, ac = st.compare_pairs(table, [("a", "b"), ("a", "c")], n_resamples=2_000)
    assert (ab.wins, ab.ties, ab.losses) == (4, 1, 1)  # the zero difference is a tie and is left out of the test
    expected = stats.wilcoxon([0.05, -0.05, 0.05, 0.1, 0.05], zero_method="wilcox")
    assert ab.p_value == pytest.approx(expected.pvalue) and ab.statistic == pytest.approx(expected.statistic)
    # the four differences of 0.05 differ in floating point (0.9 - 0.85 is 0.05000000000000004) but share a rank
    assert ab.rank_biserial == pytest.approx((2.5 * 3 + 5 - 2.5) / 15)
    assert ac.statistic is None and ac.p_value == 1.0 and ac.rank_biserial == 0.0 and ac.median_difference == 0.0
    assert (ab.p_holm, ac.p_holm) == pytest.approx((min(1.0, 2 * ab.p_value), 1.0))
    with pytest.raises(ValueError, match="same datasets"):
        st.compare_pairs(table.assign(b=[0.1, None, 0.2, 0.3, 0.4, 0.5]), [("a", "b")])


def test_the_smallest_wilcoxon_p_value_with_six_datasets():
    table = pd.DataFrame({"a": [0.9, 0.8, 0.7, 0.95, 0.6, 0.85], "b": [0.8, 0.7, 0.6, 0.85, 0.5, 0.75]})
    [pair] = st.compare_pairs(table, [("a", "b")], n_resamples=1_000)
    assert pair.p_value == pytest.approx(2 / 2 ** 6) and pair.rank_biserial == 1.0  # a better on all six


def test_rank_biserial_correlation():
    assert st.rank_biserial([1, 2, 3]) == 1.0 and st.rank_biserial([-1, -2]) == -1.0
    assert st.rank_biserial([0, 0]) == 0.0
    # sizes 1, 2, 2, 3 -> ranks 1, 2.5, 2.5, 4; W+ = 1 + 2.5 + 4 = 7.5, W- = 2.5
    assert st.rank_biserial([1, -2, 2, 3]) == pytest.approx((7.5 - 2.5) / 10)


def test_bootstrap_intervals_of_the_median():
    # For 5 values, a resampled median is 1 when at least 3 of the 5 draws are 1: probability
    # 10 * 0.2^3 * 0.8^2 + 5 * 0.2^4 * 0.8 + 0.2^5 = 0.058, above 0.025, so the 95% interval is (1, 5).
    assert st.bootstrap_median_interval([1, 2, 3, 4, 5], seed=1) == (1.0, 5.0)
    assert st.bootstrap_median_interval([0.2] * 6) == (0.2, 0.2)
    low, high = st.bootstrap_median_interval(DARWIN, n_resamples=5_000, seed=3)
    assert low <= 24.0 <= high and min(DARWIN) <= low and high <= max(DARWIN)
    assert st.bootstrap_median_interval(DARWIN, n_resamples=5_000, seed=3) == (low, high)  # repeatable
    with pytest.raises(ValueError):
        st.bootstrap_median_interval([])


# --------------------------------------------------------------------------- time-to-target
def test_time_to_target_worked_by_hand():
    runs = pd.DataFrame([
        ("A", "d", 1, 120, 0.80), ("A", "d", 1, 300, 0.90),
        ("B", "d", 1, 120, 0.895), ("B", "d", 1, 300, 0.85),
        ("C", "d", 1, 120, 0.50), ("C", "d", 1, 300, None),
        ("A", "e", 1, 120, -0.504), ("B", "e", 1, 120, -0.50), ("C", "e", 1, 120, -0.6),  # a negated error
    ], columns=["system", "dataset", "seed", "budget_seconds", "score"])
    result = st.time_to_target(runs).set_index(["dataset", "system"])
    assert result.loc[("d", "A"), "target"] == pytest.approx(0.891)  # 99% of the best, 0.90
    assert result.loc[("d", "A"), "time_to_target"] == 300 and result.loc[("d", "B"), "time_to_target"] == 120
    assert math.isnan(result.loc[("d", "C"), "time_to_target"]) and not result.loc[("d", "C"), "reached"]
    assert result.loc[("e", "B"), "target"] == pytest.approx(-0.505)  # 1% of the best's size below it
    assert result.loc[("e", "A"), "time_to_target"] == 120 and not result.loc[("e", "C"), "reached"]
    assert st.target_of(0.0) == 0.0
    with pytest.raises(ValueError, match="Missing columns"):
        st.time_to_target(runs.drop(columns=["seed"]))
