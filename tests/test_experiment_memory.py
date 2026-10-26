"""Tests for the Experiment_Memory, version 1 (task 15.1 of the spec)."""

import json
import logging
import math

import numpy as np
import pandas as pd
import pytest

from app.services.experiment_memory import ExperimentMemory, MemoryRecord, StrategyStats, dataset_fingerprint


def record(n, fingerprint="A", task="binary", meta=None, strategy="drop_columns", verdict="keep", diff=0.02,
           cost=10.0, kinds=("overfitting",)):
    return MemoryRecord(
        record_id=f"SES-001/EXP-{n:03d}", created_at="2026-10-26T12:00:00", session_id="SES-001",
        experiment_id=f"EXP-{n:03d}", dataset_name=f"data-{fingerprint}", dataset_fingerprint=fingerprint,
        task_type=task, primary_metric="f1", meta_features=meta if meta is not None else {"x": 0.0},
        diagnoses=[{"kind": k, "severity": "moderate"} for k in kinds],
        strategy={"strategy_id": strategy, "params": {}}, verdict=verdict, mean_difference=diff,
        std_difference=0.01, n_folds=15, cost_seconds=cost, utility=diff / cost, engine="flaml==2.7.0",
    )


@pytest.fixture
def memory(tmp_path):
    return ExperimentMemory(tmp_path / "memory" / "memory.jsonl")


# --------------------------------------------------------------------------- storage
def test_records_survive_a_restart_and_keep_their_order(memory, tmp_path):
    for n in range(3):
        memory.add(record(n))
    reopened = ExperimentMemory(tmp_path / "memory" / "memory.jsonl")
    assert [r.experiment_id for r in reopened.records()] == ["EXP-000", "EXP-001", "EXP-002"]
    assert reopened.records()[1] == record(1)
    lines = (tmp_path / "memory" / "memory.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["schema_version"] == 1  # one JSON object per line


def test_adding_only_appends(memory):
    memory.add(record(0))
    before = memory.path.read_bytes()
    memory.add(record(1))
    assert memory.path.read_bytes().startswith(before)  # earlier lines are never rewritten


def test_damaged_lines_are_skipped_with_a_warning(memory, caplog):
    memory.add(record(0))
    with memory.path.open("a", encoding="utf-8") as file:
        file.write("this is not json\n")
        file.write(json.dumps({"record_id": "incomplete"}) + "\n")  # valid JSON, missing fields
        file.write("\n")
    memory.add(record(1))
    with memory.path.open("a", encoding="utf-8") as file:
        file.write('{"record_id": "SES-001/EXP-9", "created_at": ')  # a write cut off by a crash
    with caplog.at_level(logging.WARNING):
        records = memory.records()
    assert [r.experiment_id for r in records] == ["EXP-000", "EXP-001"]
    assert memory.skipped_lines == 3 and caplog.text.count("Skipping damaged line") == 3


def test_later_schema_fields_are_ignored(memory):
    data = record(0).to_dict()
    data["field_from_a_later_version"] = 42
    memory.path.parent.mkdir(parents=True)
    memory.path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    assert memory.records() == [record(0)]


def test_read_only_memory_and_invalid_numbers_are_refused(memory, tmp_path):
    memory.add(record(0))
    with pytest.raises(PermissionError):
        ExperimentMemory(memory.path, read_only=True).add(record(1))
    with pytest.raises(ValueError):
        memory.add(record(2, meta={"x": float("nan")}))
    assert len(memory.records()) == 1


def test_snapshot_hash(memory):
    empty = memory.snapshot_hash()
    memory.add(record(0))
    first = memory.snapshot_hash()
    assert first != empty and memory.snapshot_hash() == first
    memory.add(record(1))
    assert memory.snapshot_hash() != first


def test_dataset_fingerprint():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"], "y": [0, 1, 0]})
    assert dataset_fingerprint(df, "y") == dataset_fingerprint(df.copy(), "y")
    assert dataset_fingerprint(df, "y") == dataset_fingerprint(df.iloc[::-1], "y")  # row order does not matter
    assert len({dataset_fingerprint(df, "y"), dataset_fingerprint(df, "a"), dataset_fingerprint(df.head(2), "y"),
                dataset_fingerprint(df.assign(a=df["a"].astype(int)), "y"),
                dataset_fingerprint(df.rename(columns={"b": "c"}), "y")}) == 5


# --------------------------------------------------------------------------- similarity
def test_nearest_datasets_in_the_right_order(memory):
    # x differs between datasets; y is the same for all, so it carries no information and is left out.
    for n, (fingerprint, x) in enumerate([("A", 0.0), ("B", 1.0), ("C", 3.0), ("A", 0.0)]):
        memory.add(record(n, fingerprint=fingerprint, meta={"x": x, "y": 7.0}))
    spread = np.std([0.0, 1.0, 3.0])  # over the 3 datasets (A's two records are averaged first)
    nearest = memory.similar({"x": 0.9, "y": 100.0}, "binary")
    assert [fp for fp, _ in nearest] == ["B", "A", "C"]
    expected = {"A": 0.9 / spread, "B": 0.1 / spread, "C": 2.1 / spread}
    for fingerprint, distance in nearest:
        assert distance == pytest.approx(expected[fingerprint])
    assert [fp for fp, _ in memory.similar({"x": 0.9}, "binary", k=2)] == ["B", "A"]


def test_records_average_and_only_shared_features_count(memory):
    memory.add(record(0, fingerprint="A", meta={"x": 0.0, "z": 0.0}))
    memory.add(record(1, fingerprint="A", meta={"x": 2.0, "z": 10.0}))  # A's average: x 1, z 5
    memory.add(record(2, fingerprint="B", meta={"x": 3.0}))  # B has no z
    memory.add(record(3, fingerprint="C", meta={"x": 5.0, "z": 1.0}))
    query = {"x": 3.0, "z": 5.0}
    x_spread, z_spread = np.std([1.0, 3.0, 5.0]), np.std([5.0, 1.0])
    distances = dict(memory.similar(query, "binary"))
    assert distances["A"] == pytest.approx(math.hypot((1.0 - 3.0) / x_spread, 0.0))
    assert distances["B"] == pytest.approx(0.0)  # only x is shared, and it is equal
    assert distances["C"] == pytest.approx(math.hypot((5.0 - 3.0) / x_spread, (1.0 - 5.0) / z_spread))


def test_other_task_types_and_an_excluded_dataset_are_left_out(memory):
    memory.add(record(0, fingerprint="A", meta={"x": 0.0}))
    memory.add(record(1, fingerprint="B", meta={"x": 1.0}))
    memory.add(record(2, fingerprint="R", task="regression", meta={"x": 0.5}))
    assert [fp for fp, _ in memory.similar({"x": 0.0}, "binary")] == ["A", "B"]
    assert [fp for fp, _ in memory.similar({"x": 0.0}, "binary", exclude_fingerprint="A")] == ["B"]
    assert [fp for fp, _ in memory.similar({"x": 0.0}, "regression")] == ["R"]
    assert memory.similar({"x": 0.0}, "multiclass") == []


# --------------------------------------------------------------------------- statistics
def test_strategy_statistics_on_a_small_hand_made_memory(memory):
    # Two datasets with x 0 and 2 (spread 1): the query x = 0 is at distance 0 from A and 2 from B.
    memory.add(record(0, "A", meta={"x": 0.0}, strategy="drop_columns", verdict="keep", diff=0.04, cost=10,
                      kinds=("overfitting", "high_cardinality")))
    memory.add(record(1, "A", meta={"x": 0.0}, strategy="drop_columns", verdict="discard", diff=-0.01, cost=20,
                      kinds=("high_cardinality",)))
    memory.add(record(2, "B", meta={"x": 2.0}, strategy="drop_columns", verdict="keep_simpler", diff=0.0, cost=30,
                      kinds=("high_cardinality",)))
    memory.add(record(3, "B", meta={"x": 2.0}, strategy="class_weight_balanced", verdict="discard", diff=-0.02,
                      cost=5, kinds=()))
    memory.add(record(4, "X", task="regression", strategy="drop_columns", verdict="keep"))  # other task: ignored
    stats = {(s.strategy_id, s.diagnosis_kind): s for s in memory.strategy_stats({"x": 0.0}, "binary")}
    assert list(stats) == [("class_weight_balanced", "none"), ("drop_columns", "high_cardinality"),
                           ("drop_columns", "overfitting")]
    weight_a, weight_b = 1.0, 1.0 / 3.0  # 1 / (1 + distance)
    hc = stats[("drop_columns", "high_cardinality")]
    assert hc.observations == 3
    assert hc.weighted_keeps == pytest.approx(weight_a + weight_b) and hc.weighted_discards == pytest.approx(weight_a)
    assert hc.keep_rate == pytest.approx((weight_a + weight_b) / (2 * weight_a + weight_b))
    assert hc.mean_gain_when_kept == pytest.approx(0.02) and hc.mean_cost_seconds == pytest.approx(20.0)
    over = stats[("drop_columns", "overfitting")]
    assert (over.observations, over.keep_rate, over.mean_gain_when_kept) == (1, 1.0, 0.04)
    weights = stats[("class_weight_balanced", "none")]
    assert weights.keep_rate == 0.0 and weights.mean_gain_when_kept is None
    assert weights.weighted_discards == pytest.approx(weight_b)
    assert StrategyStats(**{k: v for k, v in hc.to_dict().items() if k != "keep_rate"}) == hc
    # Leaving A out keeps only B's records.
    only_b = {(s.strategy_id, s.diagnosis_kind): s for s in memory.strategy_stats({"x": 0.0}, "binary",
                                                                                   exclude_fingerprint="A")}
    assert only_b[("drop_columns", "high_cardinality")].observations == 1


def test_an_empty_memory(memory):
    assert memory.records() == [] and memory.similar({"x": 1.0}, "binary") == []
    assert memory.strategy_stats({"x": 1.0}, "binary") == []


# --------------------------------------------------------------------------- the Session writes it
def test_a_session_adds_one_record_per_verdict_and_none_for_failures(tmp_path):
    from tests.test_autopilot import run_session

    result, record_, engine, assessor, settings = run_session(tmp_path, experiments=4, engine_fails={1},
                                                              shifts={2: 0.05})
    memory = ExperimentMemory(settings.experiments_dir / "memory" / "memory.jsonl")
    records = memory.records()
    judged = [s for s in record_["steps"] if s["decision"]]
    assert [r.experiment_id for r in records] == [s["experiment_id"] for s in judged]  # the failed step has none
    first = records[0]
    step = judged[0]
    assert first.session_id == result.session_id and first.verdict == step["decision"]["verdict"]
    assert first.strategy == step["proposal"]["spec"] and first.diagnoses == step["proposal"]["answered"]
    assert first.mean_difference == step["decision"]["mean_difference"] and first.n_folds == 15
    assert first.cost_seconds == pytest.approx(step["cost"]["wall_seconds"], abs=1e-3)
    assert first.meta_features["n_rows"] == 240.0  # the Training_Part


def test_the_memory_does_not_influence_decisions(tmp_path):
    from tests.test_autopilot import run_session

    shifts = {1: 0.0, 2: 0.05, 3: -0.02}
    plain = run_session(tmp_path / "plain", experiments=5, shifts=shifts)[1]
    prefilled_root = tmp_path / "prefilled"
    memory = ExperimentMemory(prefilled_root / "experiments" / "memory" / "memory.jsonl")
    for n in range(20):  # a memory that says class weights always help
        memory.add(record(n, strategy="class_weight_balanced", verdict="keep", diff=0.1, kinds=("class_imbalance",)))
    prefilled = run_session(prefilled_root, experiments=5, shifts=shifts)[1]

    def steps(session):
        return [((s["proposal"] or {}).get("key"), (s["decision"] or {}).get("verdict")) for s in session["steps"]]

    assert steps(prefilled) == steps(plain)
