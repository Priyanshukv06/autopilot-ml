"""Tests for the search summary (task 5 of the spec), including the real EXP-001 and EXP-002 histories."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from app.services.search_summary import summarize_search

ROOT = Path(__file__).resolve().parents[1]


def trial(n, family, score, wall, seconds=1.0):
    return {"trial": n, "estimator": family, "validation_score": score, "wall_clock_time": wall, "trial_time": seconds}


def saved_history(experiment: str) -> tuple[list[dict], str]:
    folder = ROOT / "experiments" / experiment
    if not (folder / "search_history.csv").exists():
        pytest.skip(f"{experiment} results are not in this copy of the project")
    rows = pd.read_csv(folder / "search_history.csv").to_dict("records")
    rows = [{k: None if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()} for row in rows]
    return rows, json.loads((folder / "summary.json").read_text(encoding="utf-8"))["primary_metric"]


def test_summary_of_a_small_history():
    history = [
        trial(0, "lgbm", 0.70, 1.0), trial(1, "xgboost", 0.75, 3.0, 2.0), trial(2, "lgbm", None, 4.0),
        trial(3, "lgbm", 0.80, 9.0, 0.5), trial(4, "xgboost", 0.80, 10.0),
    ]
    summary = summarize_search(history, "f1")
    assert summary.search_seconds == 10.0 and summary.n_trials == 5
    assert summary.time_to_best == 9.0  # the first of the tied best trials
    assert summary.share_after_best == pytest.approx(0.1)
    assert summary.gain_last_quarter == pytest.approx(0.05)  # 0.80 at the end against 0.75 at 7.5 s
    assert summary.best_per_family == {"lgbm": 0.80, "xgboost": 0.80}
    assert summary.seconds_per_family == {"lgbm": 2.5, "xgboost": 3.0}


def test_error_metrics_count_lower_as_better():
    history = [trial(0, "lgbm", 5.0, 1.0), trial(1, "lgbm", 3.0, 8.0), trial(2, "rf", 4.0, 10.0)]
    summary = summarize_search(history, "rmse")
    assert summary.time_to_best == 8.0 and summary.best_per_family == {"lgbm": 3.0, "rf": 4.0}
    assert summary.gain_last_quarter == pytest.approx(2.0)  # 5.0 at 7.5 s, 3.0 at the end: positive is better


def test_unusable_histories_are_rejected():
    with pytest.raises(ValueError):
        summarize_search([], "f1")
    with pytest.raises(ValueError):
        summarize_search([trial(0, "lgbm", None, 1.0)], "f1")


@pytest.mark.parametrize(
    "experiment, seconds, best_at, share_after_best, trials",
    [("EXP-001", 60.04, 42.9, 0.286, 290), ("EXP-002", 59.99, 23.16, 0.614, 501)],
)
def test_saved_phase1_searches(experiment, seconds, best_at, share_after_best, trials):
    history, metric = saved_history(experiment)
    summary = summarize_search(history, metric)
    assert summary.search_seconds == pytest.approx(seconds, abs=0.01)
    assert summary.time_to_best == pytest.approx(best_at, abs=0.01)
    assert summary.share_after_best == pytest.approx(share_after_best, abs=0.001)
    assert summary.gain_last_quarter == 0.0  # nothing better was found in the last 15 s
    assert summary.n_trials == trials
