"""Summary of one FLAML search, read from the trial history the runner saves.

The Critic uses it to judge how the time budget was spent (budget_limited, wasted_budget and
weak_model_families). Times are FLAML's ``wall_clock_time``: seconds from the start of the
search to the end of each trial.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.services.evaluator import METRICS


@dataclass(frozen=True)
class SearchSummary:
    search_seconds: float  # wall time of the last trial
    time_to_best: float  # wall time when the best trial finished (the first one, if tied)
    share_after_best: float  # (search_seconds - time_to_best) / search_seconds
    # Improvement of the best score in the last quarter of the search time: best at the end
    # minus best at 75% of it, signed so that positive is better also for error metrics.
    # None when no trial finished in the first 75% of the time.
    gain_last_quarter: float | None
    n_trials: int
    best_per_family: dict[str, float]  # best validation score of each learner family
    seconds_per_family: dict[str, float]  # sum of the trial times of each learner family

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_search(history: list[dict[str, Any]], metric: str) -> SearchSummary:
    """Summarise a trial history (``ExperimentResult.search_history``).

    ``metric`` is the metric of the history's validation scores, which is the Search_Metric
    (``ExperimentResult.search_metric``). Trials without a score count for time but not for the best.
    """
    if not history:
        raise ValueError("The search history is empty.")
    sign = 1.0 if METRICS[metric].greater_is_better else -1.0
    trials = sorted(history, key=lambda t: (float(t.get("wall_clock_time") or 0.0), int(t.get("trial", 0))))
    search_seconds = float(trials[-1].get("wall_clock_time") or 0.0)

    scored = [t for t in trials if t.get("validation_score") is not None]
    if not scored:
        raise ValueError("No trial in the search history has a validation score.")
    best = max(scored, key=lambda t: sign * float(t["validation_score"]))  # max keeps the first of ties
    time_to_best = float(best.get("wall_clock_time") or 0.0)

    cutoff = 0.75 * search_seconds
    early = [sign * float(t["validation_score"]) for t in scored if float(t.get("wall_clock_time") or 0.0) <= cutoff]
    gain = sign * float(best["validation_score"]) - max(early) if early else None

    best_per_family: dict[str, float] = {}
    seconds_per_family: dict[str, float] = {}
    for trial in trials:
        family = str(trial.get("estimator"))
        seconds_per_family[family] = seconds_per_family.get(family, 0.0) + float(trial.get("trial_time") or 0.0)
        score = trial.get("validation_score")
        if score is not None and (family not in best_per_family or sign * score > sign * best_per_family[family]):
            best_per_family[family] = float(score)

    return SearchSummary(
        search_seconds=search_seconds,
        time_to_best=time_to_best,
        share_after_best=(search_seconds - time_to_best) / search_seconds if search_seconds > 0 else 0.0,
        gain_last_quarter=gain,
        n_trials=len(trials),
        best_per_family=best_per_family,
        seconds_per_family=seconds_per_family,
    )
