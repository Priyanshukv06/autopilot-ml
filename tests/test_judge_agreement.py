"""Tests for the judge agreement study (spec task 27): the pairs taken from Session records, the strict
verdict with the Judge's rules, Cohen's kappa against scikit-learn, the agreement summary, and one strict
comparison with a fast stand-in search (marked slow)."""

import json

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from app.core.config import build_settings
from app.services.automl_runner import FLAMLEngine
from research import judge_agreement as ja
from research import systems
from tests.test_benchmark import TOYS

SETUP = {"target_column": "target", "task_type": "binary", "primary_metric": "f1", "random_seed": 1,
         "test_size": 0.2, "excluded_columns": [], "estimator_list": None, "data_steps": [], "class_weight": None,
         "search_metric": None, "search_space_limits": {}}


def side(scores, metric="f1"):
    return [{"experiment_id": f"EXP-{i:03d}", "seconds": 2.0, "metrics": {metric: s}, "model": "m"}
            for i, s in enumerate(scores)]


# --------------------------------------------------------------------------- kappa and agreement
def test_cohen_kappa_matches_scikit_learn():
    cases = [(["keep", "discard", "discard", "keep_simpler", "discard"],
              ["keep", "discard", "keep_simpler", "keep_simpler", "discard"]),
             (["a", "a", "b", "b"], ["a", "b", "a", "b"]), (["a", "b", "a", "b", "a"], ["a", "b", "a", "b", "a"])]
    for first, second in cases:
        labels = sorted(set(first) | set(second))
        assert ja.cohen_kappa(first, second, labels) == pytest.approx(cohen_kappa_score(first, second))
    # by hand: p_o = 3/4, p_e = (3/4)(2/4) + (1/4)(2/4) = 1/2, kappa = (3/4 - 1/2) / (1/2) = 0.5
    assert ja.cohen_kappa(["a", "a", "a", "b"], ["a", "a", "b", "b"], ["a", "b"]) == pytest.approx(0.5)
    assert ja.cohen_kappa(["a", "a"], ["a", "a"], ["a", "b"]) is None  # chance agreement is 1
    with pytest.raises(ValueError):
        ja.cohen_kappa([], [], ["a"])


def test_the_agreement_summary():
    rows = [{"status": "completed", "judge_verdict": j, "strict_verdict": s, "judge_seconds": 10.0,
             "strict_seconds": 100.0, "cost_ratio": 10.0}
            for j, s in [("keep", "keep"), ("keep_simpler", "keep"), ("discard", "discard"), ("discard", "keep_simpler")]]
    rows.append({"status": "failed", "judge_verdict": "keep", "strict_verdict": None})
    result = ja.agreement(rows)
    assert result["pairs"] == 4 and result["failed"] == 1 and result["agreement"] == 0.5
    assert result["accepted_agreement"] == 0.75  # only the last pair disagrees on accepted against discarded
    assert result["kappa"] == pytest.approx(cohen_kappa_score(["keep", "keep_simpler", "discard", "discard"],
                                                              ["keep", "keep", "discard", "keep_simpler"]))
    assert result["confusion"].loc["keep_simpler", "keep"] == 1 and int(result["confusion"].to_numpy().sum()) == 4
    assert result["cost_ratio_total"] == 10.0 and result["cost_ratio_median"] == 10.0


# --------------------------------------------------------------------------- the strict verdict
def test_the_strict_verdict_uses_the_judges_rules():
    accepted = side([0.70, 0.72, 0.69, 0.71, 0.70])
    better = side([0.75, 0.78, 0.74, 0.77, 0.75])
    same = side([0.70, 0.72, 0.69, 0.71, 0.70])
    worse = side([0.60, 0.65, 0.58, 0.64, 0.61])
    assert ja.strict_decision(accepted, better, "f1", False, "k").verdict == "keep"
    assert ja.strict_decision(accepted, same, "f1", True, "k").verdict == "keep_simpler"  # equal and simpler
    assert ja.strict_decision(accepted, same, "f1", False, "k").verdict == "discard"
    assert ja.strict_decision(accepted, worse, "f1", True, "k").verdict == "discard"
    decision = ja.strict_decision(accepted, better, "f1", False, "k")
    assert decision.n_folds == 5 and decision.test_train_ratio == pytest.approx(0.25)
    assert decision.mean_difference == pytest.approx(0.054)
    rmse = ja.strict_decision(side([2.0, 2.1, 1.9, 2.0, 2.2], "rmse"), side([1.5, 1.6, 1.5, 1.4, 1.6], "rmse"),
                              "rmse", False, "k")
    assert rmse.verdict == "keep" and rmse.mean_difference > 0  # lower error is better


# --------------------------------------------------------------------------- pairs from Session records
def test_pairs_come_from_judged_steps_of_a_session(tmp_path):
    session = tmp_path / "SES-001"
    session.mkdir()
    steps = [
        {"index": 0, "experiment_id": "EXP-001", "parent_experiment_id": None, "proposal": None, "decision": None,
         "assessment": {"seconds": 5.0}, "status": "completed"},
        {"index": 1, "experiment_id": "EXP-002", "parent_experiment_id": "EXP-001", "status": "completed",
         "proposal": {"spec": {"strategy_id": "drop_columns", "params": {"columns": ["x0"]}}, "complexity": "simpler"},
         "decision": {"verdict": "keep_simpler", "mean_difference": -0.001, "metric": "f1"},
         "assessment": {"seconds": 4.0}},
        {"index": 2, "experiment_id": "EXP-003", "parent_experiment_id": "EXP-002", "status": "failed",
         "proposal": {"spec": {"strategy_id": "class_weight_balanced", "params": {}}}, "decision": None,
         "assessment": None},
        {"index": 3, "experiment_id": "EXP-004", "parent_experiment_id": "EXP-002", "status": "completed",
         "proposal": {"spec": {"strategy_id": "regularised_search_space", "params": {}}, "complexity": "same"},
         "decision": {"verdict": "discard", "mean_difference": -0.02, "metric": "f1"}, "assessment": {"seconds": 6.0}},
    ]
    (session / "session.json").write_text(json.dumps({"splits": {"outer": "abc"}, "steps": steps}), encoding="utf-8")
    for n, excluded in ((1, []), (2, ["x0"]), (4, ["x0"])):
        folder = tmp_path / f"EXP-{n:03d}"
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps({"setup": {**SETUP, "excluded_columns": excluded}}),
                                             encoding="utf-8")
    pairs = ja.pairs_of_session(tmp_path, "SES-001", "toy_a", 1)
    assert [p.pair_id for p in pairs] == ["SES-001/EXP-002", "SES-001/EXP-004"]  # the failed step is left out
    first, second = pairs
    assert first.accepted_experiment == "EXP-001" and first.simpler and first.judge["verdict"] == "keep_simpler"
    assert first.judge_seconds == 9.0 and first.candidate_setup["excluded_columns"] == ["x0"]
    assert second.accepted_experiment == "EXP-002" and not second.simpler and second.judge_seconds == 10.0
    assert second.strategy == "regularised_search_space" and second.split_fingerprint == "abc"


def test_the_training_part_is_the_sessions_and_is_checked(tmp_path):
    settings = build_settings(tmp_path)
    frame, y = ja.training_part(TOYS["toy_a"], SETUP, 1, settings)
    assert len(frame) == 160 and list(frame.index) == list(range(160)) and (frame["target"] == y).all()
    with pytest.raises(ValueError, match="fingerprints differ"):
        ja.training_part(TOYS["toy_a"], SETUP, 1, settings, expected_fingerprint="not the session's")
    splits = ja.inner_splits(y, TOYS["toy_a"].task_type, 1)
    assert len(splits) == 5 and len({s.fingerprint() for s in splits}) == 5
    assert all(len(s.test_positions) == 32 for s in splits) and {s.seed for s in splits} == set(range(101, 106))
    config = ja.config_for({**SETUP, "excluded_columns": ["x0"], "class_weight": "balanced"}, TOYS["toy_a"], 1, 2)
    assert config.max_iter == 25 and config.time_budget == 60 and config.excluded_columns == ["x0"]
    assert config.class_weight == "balanced" and config.n_jobs == 2 and config.starting_points is None


# --------------------------------------------------------------------------- one strict comparison
class TwoTrials(FLAMLEngine):
    def fit(self, *args, **kwargs):
        kwargs["max_iter"] = 2
        return super().fit(*args, **kwargs)


@pytest.mark.slow
def test_one_strict_comparison_end_to_end(tmp_path):
    settings = ja.folders(tmp_path)["settings"]
    toy = TOYS["toy_a"]
    # the Session's split fingerprint: the outer split of the whole dataset with the Session's seed
    outer = systems.prepare(systems.Problem("toy_a", toy.frame, "target", toy.task_type, 1, 30),
                            build_settings(tmp_path)).split.fingerprint()
    pair = ja.StudyPair("SES-001/EXP-002", "toy_a", 1, "SES-001", 1, outer, "EXP-001", "EXP-002", dict(SETUP),
                        {**SETUP, "excluded_columns": ["x0"]}, "drop_columns", {"columns": ["x0"]}, True,
                        {"verdict": "keep_simpler", "metric": "f1"}, 8.0)
    rows_path = ja.folders(tmp_path)["pairs"]
    rows = ja.run([pair, pair], settings, rows_path, n_jobs=1, load=TOYS.__getitem__, engine=TwoTrials())
    assert len(rows) == 1 and rows[0]["status"] == "completed", rows[0]["error"]  # the same pair once
    row = rows[0]
    assert row["strict_verdict"] in ja.VERDICTS and len(json.loads(row["strict_differences"])) == 5
    # cost_ratio is rounded to 3 decimals from the unrounded seconds, strict_seconds to 2 decimals
    assert len(json.loads(row["accepted_searches"])) == 5 and row["cost_ratio"] == pytest.approx(
        row["strict_seconds"] / 8.0, abs=0.002)
    assert ja.run([pair], settings, rows_path, load=TOYS.__getitem__, engine=TwoTrials()) == []  # restart
    bad = ja.StudyPair(**{**pair.__dict__, "pair_id": "SES-001/EXP-009", "split_fingerprint": "other"})
    [failed] = ja.run([bad], settings, rows_path, load=TOYS.__getitem__, engine=TwoTrials())
    assert failed["status"] == "failed" and "fingerprints differ" in failed["error"]
    assert np.isfinite(row["strict_seconds"]) and len(ja.read_rows(rows_path)) == 2


def test_a_pair_that_ends_on_battery_is_excluded_and_runs_again(tmp_path, monkeypatch):
    def fake_pair(pair, dataset, settings, cache, **kwargs):
        cache[("SES-001", "accepted")] = ["searches"]
        return {column: None for column in ja.COLUMNS} | {
            "pair_id": pair.pair_id, "status": "completed", "judge_verdict": "keep", "strict_verdict": "keep",
            "judge_seconds": 8.0, "strict_seconds": 80.0, "cost_ratio": 10.0}

    monkeypatch.setattr(ja, "run_pair", fake_pair)
    pair = ja.StudyPair("SES-001/EXP-002", "toy_a", 1, "SES-001", 1, "f", "EXP-001", "EXP-002", dict(SETUP),
                        dict(SETUP), "drop_columns", {}, True, {"verdict": "keep", "metric": "f1"}, 8.0)
    settings, rows_path = ja.folders(tmp_path)["settings"], ja.folders(tmp_path)["pairs"]
    answers, naps = iter([True, False, True]), []  # wait once, start on AC, end on battery
    [row] = ja.run([pair], settings, rows_path, load=TOYS.__getitem__, power=lambda: next(answers), sleep=naps.append)
    assert naps == [30] and row["status"] == "excluded" and "battery" in row["error"]
    [again] = ja.run([pair], settings, rows_path, load=TOYS.__getitem__, power=lambda: False)
    assert again["status"] == "completed" and len(ja.read_rows(rows_path)) == 2
    assert ja.run([pair], settings, rows_path, load=TOYS.__getitem__, power=lambda: False) == []
    summary = ja.agreement(ja.read_rows(rows_path))
    assert (summary["pairs"], summary["failed"], summary["excluded"]) == (1, 0, 1)
