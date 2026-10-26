"""Phase 2 measurement (task 19): the cells, the pairing of runs and the summaries, on hand-made rows."""

import pytest

from research import phase2


def row(system, dataset, seed, test, *, first=None, metric="f1", seconds=10.0, experiments=1, tried="", kept=""):
    return {"system": system, "dataset": dataset, "seed": seed, "budget_seconds": 300, "metric": metric,
            "test_score": test, "first_test_score": first, "wall_seconds": seconds, "n_experiments": experiments,
            "stop_reason": "time" if system == "session" else None, "kept": kept, "tried": tried,
            "model": "LightGBM", "n_trials": 5, "run_id": f"{system}-{dataset}-{seed}"}


def test_cells_put_each_session_before_its_flaml_run():
    cells = phase2.cells(["titanic", "penguins"], [1, 2], 300)
    assert len(cells) == 8 and len({c.cell_id for c in cells}) == 8
    assert [(c.system, c.dataset, c.seed) for c in cells[:2]] == [("session", "titanic", 1), ("flaml", "titanic", 1)]


def test_pairs_sign_the_difference_so_that_positive_favours_the_session():
    rows = [
        row("session", "titanic", 1, 0.75, first=0.70), row("flaml", "titanic", 1, 0.72),
        row("session", "california_housing", 1, 0.80, first=0.81, metric="r2"),
        row("flaml", "california_housing", 1, 0.82, metric="r2"),
        row("session", "titanic", 2, 0.60),  # no FLAML run yet: no pair
    ]
    pairs = {(p["dataset"], p["seed"]): p for p in phase2.pairs(rows)}
    assert set(pairs) == {("titanic", 1), ("california_housing", 1)}
    assert pairs[("titanic", 1)]["difference"] == pytest.approx(0.03)
    assert pairs[("titanic", 1)]["loop_gain"] == pytest.approx(0.05)
    assert pairs[("california_housing", 1)]["difference"] == pytest.approx(-0.02)
    rmse = [row("session", "d", 1, 3.0, metric="rmse"), row("flaml", "d", 1, 3.5, metric="rmse")]
    assert phase2.pairs(rmse)[0]["difference"] == pytest.approx(0.5)  # lower RMSE is better


def test_summary_and_strategy_counts():
    rows = [
        row("session", "titanic", 1, 0.75, tried="drop_columns(columns=Cabin): keep_simpler; smoother_search_metric: discard"),
        row("flaml", "titanic", 1, 0.72),
        row("session", "titanic", 2, 0.70, tried="drop_columns(columns=Ticket): keep"), row("flaml", "titanic", 2, 0.70),
        row("session", "titanic", 3, 0.68, tried=""), row("flaml", "titanic", 3, 0.71),
    ]
    summary = {s["dataset"]: s for s in phase2.summary(phase2.pairs(rows))}
    assert (summary["titanic"]["session_wins"], summary["titanic"]["ties"], summary["titanic"]["flaml_wins"]) == (1, 1, 1)
    assert summary["all"]["pairs"] == 3 and summary["titanic"]["mean_difference"] == pytest.approx(0.0)
    counts = {(c["dataset"], c["strategy"]): c for c in phase2.strategy_counts(rows)}
    assert counts[("titanic", "drop_columns")]["tried"] == 2
    assert counts[("titanic", "drop_columns")]["keep"] == 1 and counts[("titanic", "drop_columns")]["keep_simpler"] == 1
    assert counts[("titanic", "smoother_search_metric")]["discard"] == 1
