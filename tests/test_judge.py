"""Tests for the Judge (task 8.1 of the spec), with hand-made Assessments."""

import math

import numpy as np
import pytest

from app.services.assessor import Assessment, FoldScore
from app.services.judge import Judge, JudgeError, JudgeSettings, corrected_resampled_t

ACCEPTED = [0.70, 0.72, 0.68, 0.71, 0.69]
WORKED = [0.02, 0.01, 0.03, -0.01, 0.02]  # the worked example in design.md and the report notes


def assessment(scores, metric="f1", fingerprint="folds-1", failed=False):
    folds = [FoldScore(0, j, {metric: s}, {metric: s}, None, {}, {}, 0.1) for j, s in enumerate(scores)]
    return Assessment("key", "EXP-001", "lgbm", {}, metric, fingerprint, folds, 1.0, failed=failed)


def compare(differences, simpler=False, metric="f1", settings=None, accepted=ACCEPTED):
    candidate = [a + d for a, d in zip(accepted, differences)]
    judge = Judge(settings, test_train_ratio=0.25)
    return judge.compare(assessment(accepted, metric), assessment(candidate, metric), candidate_is_simpler=simpler)


# --------------------------------------------------------------------------- the statistic
def test_corrected_t_and_p_values_against_hand_computation():
    # mean = 0.07 / 5 = 0.014; variance = sum((d - 0.014)^2) / 4 = 0.00092 / 4 = 0.00023
    # t = 0.014 / sqrt((1/5 + 1/4) * 0.00023) = 0.014 / sqrt(0.0001035) = 1.37612
    t, df = corrected_resampled_t(np.array(WORKED), 0.25)
    assert df == 4 and t == pytest.approx(0.014 / math.sqrt(0.45 * 0.00023), rel=1e-9)
    assert t == pytest.approx(1.37612, abs=1e-5)

    decision = compare(WORKED)
    assert decision.mean_difference == pytest.approx(0.014)
    assert decision.std_difference == pytest.approx(math.sqrt(0.00023))
    assert decision.t_better == pytest.approx(1.37612, abs=1e-5)
    assert decision.p_better == pytest.approx(0.1204, abs=1e-4)  # Student's t, 4 degrees of freedom
    assert decision.p_worse == pytest.approx(1 - decision.p_better)
    # Without the correction, t = 0.014 / sqrt(0.00023 / 5) = 2.064 and p = 0.054: it would be kept.
    assert decision.verdict == "discard"


def test_corrected_t_refuses_unusable_input():
    with pytest.raises(ValueError, match="at least 2"):
        corrected_resampled_t(np.array([0.1]), 0.25)
    with pytest.raises(ValueError, match="equal"):
        corrected_resampled_t(np.array([0.1, 0.1, 0.1]), 0.25)
    with pytest.raises(ValueError, match="positive"):
        corrected_resampled_t(np.array([0.1, 0.2]), 0.0)


# --------------------------------------------------------------------------- verdicts
def test_clearly_better_is_kept():
    decision = compare([0.05, 0.04, 0.06, 0.05, 0.05])
    assert decision.verdict == "keep" and decision.p_better < 0.10
    assert (decision.wins, decision.losses, decision.ties, decision.n_folds) == (5, 0, 0, 5)


def test_not_clearly_better_but_simpler_is_kept_as_simpler():
    assert compare(WORKED, simpler=True).verdict == "keep_simpler"
    slightly_worse = [-0.004, 0.002, -0.006, 0.001, -0.003]  # mean -0.002, within the 0.005 margin
    assert compare(slightly_worse, simpler=True).verdict == "keep_simpler"
    assert compare(slightly_worse, simpler=False).verdict == "discard"


def test_simpler_but_clearly_worse_is_discarded():
    assert compare([-0.05, -0.04, -0.06, -0.05, -0.05], simpler=True).verdict == "discard"  # below the margin
    # Within the margin on average, but consistently worse: p(worse) is small, so not kept either.
    consistently_worse = [-0.0040, -0.0041, -0.0039, -0.0040, -0.0040]
    decision = compare(consistently_worse, simpler=True)
    assert decision.p_worse < 0.10 and decision.verdict == "discard"


def test_minimum_gain():
    small_but_sure = [0.015, 0.016, 0.014, 0.015, 0.015]
    assert compare(small_but_sure).verdict == "keep"
    assert compare(small_but_sure, settings=JudgeSettings(min_gain=0.02)).verdict == "discard"


def test_error_metrics_have_the_sign_flipped():
    accepted = [3.0, 3.2, 2.9, 3.1, 3.0]  # RMSE: lower is better
    better = compare([-0.30, -0.25, -0.35, -0.30, -0.28], metric="rmse", accepted=accepted)
    assert better.verdict == "keep" and better.mean_difference == pytest.approx(0.296)
    worse = compare([0.30, 0.25, 0.35, 0.30, 0.28], metric="rmse", accepted=accepted)
    assert worse.verdict == "discard" and worse.mean_difference == pytest.approx(-0.296) and worse.losses == 5


def test_wins_losses_and_ties():
    decision = compare([0.01, -0.01, 0.0, 0.0, 0.02])
    assert (decision.wins, decision.losses, decision.ties) == (2, 1, 2)


# --------------------------------------------------------------------------- equal differences
@pytest.mark.parametrize(
    "difference, simpler, verdict",
    [(0.01, False, "keep"), (0.0, False, "discard"), (0.0, True, "keep_simpler"), (-0.004, True, "keep_simpler"),
     (-0.01, True, "discard")],
)
def test_all_differences_equal_decides_on_the_mean_without_dividing_by_zero(difference, simpler, verdict):
    decision = compare([difference] * 5, simpler=simpler)
    assert decision.verdict == verdict
    assert decision.t_better is None and decision.p_better is None and decision.p_worse is None
    assert "the decision uses the mean alone" in decision.reason


def test_identical_assessments_are_ties():
    judge = Judge(test_train_ratio=0.25)
    decision = judge.compare(assessment(ACCEPTED), assessment(ACCEPTED), candidate_is_simpler=False)
    assert decision.verdict == "discard" and decision.ties == 5 and decision.mean_difference == 0.0


# --------------------------------------------------------------------------- records and errors
def test_decision_record_holds_every_number_and_setting():
    record = compare(WORKED).to_dict()
    assert record["settings"] == {"alpha": 0.10, "min_gain": 0.0, "non_inferiority_margin": 0.005}
    assert record["test_train_ratio"] == 0.25 and record["n_folds"] == 5 and record["metric"] == "f1"
    assert {"mean_difference", "std_difference", "wins", "losses", "ties", "t_better", "p_better", "p_worse",
            "verdict", "reason"} <= set(record)
    assert "p(better) = 0.120" in record["reason"] and record["reason"].endswith(".")


def test_folds_without_scores_are_left_out():
    accepted, candidate = assessment(ACCEPTED), assessment([a + d for a, d in zip(ACCEPTED, WORKED)])
    candidate.folds[4].validation_metrics["f1"] = None
    decision = Judge(test_train_ratio=0.25).compare(accepted, candidate, candidate_is_simpler=False)
    assert decision.n_folds == 4 and "1 without scores left out" in decision.reason


@pytest.mark.parametrize(
    "candidate, message",
    [
        (assessment(ACCEPTED, fingerprint="folds-2"), "different folds"),
        (assessment(ACCEPTED[:4]), "different folds"),
        (assessment(ACCEPTED, metric="accuracy"), "different metrics"),
        (assessment(ACCEPTED, failed=True), "failed or empty"),
        (assessment([None] * 5), "Fewer than 2 folds"),
    ],
)
def test_assessments_that_cannot_be_compared_raise(candidate, message):
    with pytest.raises(JudgeError, match=message):
        Judge(test_train_ratio=0.25).compare(assessment(ACCEPTED), candidate, candidate_is_simpler=False)


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.0}, {"min_gain": -0.1},
                                    {"non_inferiority_margin": -0.01}])
def test_invalid_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        JudgeSettings(**kwargs)
    with pytest.raises(ValueError):
        Judge(test_train_ratio=0.0)
