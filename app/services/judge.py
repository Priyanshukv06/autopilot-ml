"""Judge: decides whether a Candidate_Setup replaces the Accepted_Setup (Requirement 9).

The Judge compares two Assessments on the same Session_Folds, fold by fold, on the Primary_Metric,
with the corrected resampled t-test (Nadeau and Bengio, 2003; Bouckaert and Frank, 2004). The
correction matters because the folds overlap: an ordinary paired t-test over a Session's 15 folds
would be far too confident.

Decision rules (design.md section 9), with d = candidate minus accepted on each fold, signed so that
positive is better:

- keep: mean(d) > min_gain and p_better <= alpha;
- keep_simpler: the candidate is simpler, mean(d) >= -margin and p_worse > alpha;
- discard: otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

from app.services.assessor import REPEAT_TOLERANCE, Assessment
from app.services.evaluator import METRICS

Verdict = Literal["keep", "keep_simpler", "discard"]


class JudgeError(ValueError):
    """The two Assessments cannot be compared."""


@dataclass(frozen=True)
class JudgeSettings:
    alpha: float = 0.20  # chosen by the judge study (task 17); 0.10 before it
    min_gain: float = 0.0
    non_inferiority_margin: float = 0.005

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        if self.min_gain < 0 or self.non_inferiority_margin < 0:
            raise ValueError("min_gain and non_inferiority_margin must not be negative.")


@dataclass(frozen=True)
class JudgeDecision:
    verdict: Verdict
    metric: str
    mean_difference: float  # candidate minus accepted, higher is better
    std_difference: float
    n_folds: int
    wins: int
    losses: int
    ties: int
    t_better: float | None
    p_better: float | None
    p_worse: float | None
    settings: JudgeSettings
    test_train_ratio: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["settings"] = asdict(self.settings)
        return data


def corrected_resampled_t(differences: np.ndarray, test_train_ratio: float) -> tuple[float, int]:
    """t = mean(d) / sqrt((1/J + test_train_ratio) * var(d, ddof=1)), with J = len(d) and df = J - 1.

    For k-fold cross-validation ``test_train_ratio`` is 1 / (k - 1).
    """
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Need at least 2 finite paired differences.")
    if test_train_ratio <= 0:
        raise ValueError("test_train_ratio must be positive.")
    # Compare the values, not the variance: the mean of equal values can carry rounding noise,
    # which would give a tiny non-zero variance and a meaningless, huge t.
    if np.ptp(d) == 0:
        raise ValueError("All differences are equal, so the t statistic is not defined.")
    return float(d.mean() / np.sqrt((1 / len(d) + test_train_ratio) * d.var(ddof=1))), len(d) - 1


class Judge:
    def __init__(self, settings: JudgeSettings | None = None, *, test_train_ratio: float) -> None:
        """``test_train_ratio``: validation rows per training row in one fold (``SessionFolds.test_train_ratio``)."""
        if test_train_ratio <= 0:
            raise ValueError("test_train_ratio must be positive.")
        self.settings = settings or JudgeSettings()
        self.test_train_ratio = test_train_ratio

    def compare(self, accepted: Assessment, candidate: Assessment, candidate_is_simpler: bool) -> JudgeDecision:
        metric = self._check(accepted, candidate)
        sign = 1.0 if METRICS[metric].greater_is_better else -1.0
        raw = sign * (candidate.values(metric, "validation") - accepted.values(metric, "validation"))
        d = raw[np.isfinite(raw)]  # a fold where either score is missing cannot be compared
        if len(d) < 2:
            raise JudgeError(f"Fewer than 2 folds have {metric} scores for both Setups.")

        s = self.settings
        mean, std = float(d.mean()), float(d.std(ddof=1))
        wins, losses = int((d > REPEAT_TOLERANCE).sum()), int((d < -REPEAT_TOLERANCE).sum())
        ties = len(d) - wins - losses
        if std <= REPEAT_TOLERANCE:  # all differences equal: decide on the mean alone (rule 6)
            t = p_better = p_worse = None
            keep = mean > s.min_gain
            keep_simpler = candidate_is_simpler and mean >= -s.non_inferiority_margin
        else:
            t, df = corrected_resampled_t(d, self.test_train_ratio)
            p_better, p_worse = float(stats.t.sf(t, df)), float(stats.t.cdf(t, df))
            keep = mean > s.min_gain and p_better <= s.alpha
            keep_simpler = candidate_is_simpler and mean >= -s.non_inferiority_margin and p_worse > s.alpha
        verdict: Verdict = "keep" if keep else "keep_simpler" if keep_simpler else "discard"

        decision = JudgeDecision(
            verdict=verdict, metric=metric, mean_difference=mean, std_difference=std, n_folds=len(d), wins=wins,
            losses=losses, ties=ties, t_better=t, p_better=p_better, p_worse=p_worse, settings=s,
            test_train_ratio=self.test_train_ratio, reason="",
        )
        return _with_reason(decision, candidate_is_simpler, skipped=len(raw) - len(d))

    @staticmethod
    def _check(accepted: Assessment, candidate: Assessment) -> str:
        for name, assessment in (("accepted", accepted), ("candidate", candidate)):
            if assessment.failed or not assessment.folds:
                raise JudgeError(f"The {name} Assessment is failed or empty.")
        if accepted.folds_fingerprint != candidate.folds_fingerprint or len(accepted.folds) != len(candidate.folds):
            raise JudgeError("The two Assessments were made on different folds.")
        if accepted.primary_metric != candidate.primary_metric:
            raise JudgeError(f"The two Assessments use different metrics ({accepted.primary_metric} and "
                             f"{candidate.primary_metric}).")
        if accepted.primary_metric not in METRICS:
            raise JudgeError(f"Unknown metric '{accepted.primary_metric}'.")
        return accepted.primary_metric


def _with_reason(decision: JudgeDecision, simpler: bool, skipped: int) -> JudgeDecision:
    s, d = decision.settings, decision
    folds = f"{d.n_folds} folds" + (f", {skipped} without scores left out" if skipped else "")
    text = (f"Candidate minus accepted {d.metric}: mean {d.mean_difference:+.4f} (standard deviation "
            f"{d.std_difference:.4f}); better on {d.wins}, worse on {d.losses}, equal on {d.ties} of {folds}. ")
    if d.p_better is None:
        text += "All differences are equal, so the decision uses the mean alone. "
    else:
        text += f"Corrected t = {d.t_better:.2f}: p(better) = {d.p_better:.3f}, p(worse) = {d.p_worse:.3f}. "
    if d.verdict == "keep":
        text += f"Keep: better by more than {s.min_gain:g}" + (
            f" and p(better) is at most alpha {s.alpha:g}." if d.p_better is not None else ".")
    elif d.verdict == "keep_simpler":
        text += (f"Keep the simpler Setup: not worse by more than the margin {s.non_inferiority_margin:g}"
                 + (f", and p(worse) is above alpha {s.alpha:g}." if d.p_worse is not None else "."))
    else:
        text += "Discard: " + ("not clearly better" + (", and not shown to be as good as the accepted Setup."
                                                         if simpler else " (and not simpler)."))
    return JudgeDecision(**{**decision.__dict__, "reason": text})
