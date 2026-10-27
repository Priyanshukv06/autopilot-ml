"""Tests for the benchmark harness (spec task 24.1): the grid and the smoke setting, every cell once, restart,
interrupted and failed cells, the memory hash and its guards, the rows file checks, and progress. Stand-in
systems on small synthetic datasets, so no search runs; one real cell uses the manual LightGBM system."""

import dataclasses
import json

import pytest

from app.services.experiment_memory import ExperimentMemory, dataset_fingerprint
from app.services.task_detector import TaskType
from research import benchmark
from research.datasets import datasets
from research.synthetic import make_synthetic
from research.systems import LIGHTGBM_DEFAULTS, ManualSystem, Problem, SystemResult, prepare
from tests.test_experiment_memory import record

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def toy(name, seed, rows=200):
    data = make_synthetic("binary", rows, [], seed=seed, fresh_rows=10)
    return benchmark.Dataset(name, data.frame, "target", TaskType.BINARY)


TOYS = {"toy_a": toy("toy_a", 0), "toy_b": toy("toy_b", 1)}


class StandIn:
    """A system that returns a fixed result at once; it records each call with the cell's ID."""

    def __init__(self, name, calls, status="completed", on_run=None):
        self.name, self.calls, self.status, self.on_run = name, calls, status, on_run

    def describe(self):
        return {"stand_in": self.name}

    def run(self, problem, prepared, settings):
        self.calls.append(f"{self.name}-{problem.dataset}-{problem.budget_seconds}s-s{problem.seed}")
        if self.on_run is not None:
            self.on_run(len(self.calls))
        if self.status == "failed":
            return SystemResult(self.name, "failed", None, {}, 0.01, 0, None, error="RuntimeError: stand-in failure")
        return SystemResult(self.name, "completed", 0.75, {"f1": 0.75, "accuracy": 0.8}, 0.01, 2, "StandIn",
                            {"session_id": "SES-007", "stop_reason": "time", "first_test_score": 0.7,
                             "kept": ["drop_columns", "class_weight_balanced"]})


@pytest.fixture
def where(tmp_path):
    return benchmark.folders(tmp_path)


def run(cells, where, calls, factory=None, **kwargs):
    settings, rows_path = where
    make = factory or (lambda name: StandIn(name, calls))
    return benchmark.run(cells, settings, rows_path, make=make, load=TOYS.__getitem__, **kwargs)


SMALL = benchmark.grid(["toy_a", "toy_b"], seeds=[1, 2], budgets=[20, 40], systems=["s1", "s2"], ablations=["a1"],
                       ablation_budget=40)  # 2 seeds x 2 datasets x (2 cells at 20 s + 3 at 40 s) = 20 cells


# --------------------------------------------------------------------------- the grid
def test_the_default_grid_and_the_smoke_setting():
    cells = benchmark.grid([e.name for e in datasets("evaluation")])
    assert len(cells) == 390 and len({c.cell_id for c in cells}) == 390
    main = [c for c in cells if c.system in benchmark.MAIN_SYSTEMS]
    ablations = [c for c in cells if c.system in benchmark.ABLATIONS]
    assert len(main) == 300 and {c.budget_seconds for c in main} == {120, 300}
    assert len(ablations) == 90 and {c.budget_seconds for c in ablations} == {300}
    assert [c.seed for c in cells] == sorted(c.seed for c in cells)  # seed by seed
    assert cells[0].cell_id == "manual-pc4-120s-s1"

    smoke = benchmark.smoke_grid()
    assert [c.system for c in smoke] == [*benchmark.MAIN_SYSTEMS, *benchmark.ABLATIONS]
    assert {(c.dataset, c.budget_seconds, c.seed) for c in smoke} == {("titanic", 120, 1)}
    assert {c.system for c in SMALL if c.budget_seconds == 20} == {"s1", "s2"} and len(SMALL) == 20


def test_the_smoke_setting_runs_its_cells_in_its_own_folders(tmp_path):
    settings, rows_path = benchmark.folders(tmp_path, smoke=True)
    root = tmp_path.resolve()
    assert settings.experiments_dir == root / "experiments" / "benchmark-smoke"
    assert settings.models_dir == root / "models" / "benchmark-smoke"
    assert rows_path == root / "results" / "benchmark" / "smoke" / "benchmark_cells.csv"
    full_settings, full_rows = benchmark.folders(tmp_path)
    assert full_settings.experiments_dir == root / "experiments" / "benchmark"
    assert full_rows == root / "results" / "benchmark" / "benchmark_cells.csv"

    calls = []
    cells = benchmark.smoke_grid(datasets=["toy_a"], budgets=[40, 20])
    rows = run(cells, (settings, rows_path), calls)
    assert calls == [f"{name}-toy_a-20s-s1" for name in [*benchmark.MAIN_SYSTEMS, *benchmark.ABLATIONS]]
    assert [r["cell_id"] for r in benchmark.read_cells(rows_path)] == calls and len(rows) == 8
    assert settings.experiments_dir.is_dir() and not full_rows.exists()


# --------------------------------------------------------------------------- once, restart, interruptions
def test_every_cell_runs_once_and_a_restart_skips_finished_cells(where):
    calls = []
    rows = run(SMALL, where, calls)
    assert calls == [c.cell_id for c in SMALL] and len(rows) == 20
    on_disk = benchmark.read_cells(where[1])
    assert [r["cell_id"] for r in on_disk] == calls

    row = on_disk[0]
    assert row["system"] == "s1" and row["dataset"] == "toy_a" and row["budget_seconds"] == 20 and row["seed"] == 1
    assert row["status"] == "completed" and row["test_score"] == 0.75 and row["test_f1"] == 0.75
    assert row["test_accuracy"] == 0.8 and row["test_r2"] is None and row["error"] is None
    assert row["n_experiments"] == 2 and row["stop_reason"] == "time" and row["first_test_score"] == 0.7
    assert row["kept"] == "drop_columns; class_weight_balanced" and row["run_id"] == "SES-007"
    problem = Problem("toy_a", TOYS["toy_a"].frame, "target", TaskType.BINARY, 1, 20)
    prepared = prepare(problem, where[0])
    assert row["split"] == prepared.split.fingerprint()
    assert row["n_train"] == 160 and row["n_test"] == 40 and json.loads(row["excluded_columns"]) == []
    assert json.loads(row["system_settings"]) == {"stand_in": "s1"} and row["n_jobs"] == -1 and row["cpu_count"] >= 1
    assert row["memory_hash"] == EMPTY_SHA256 and row["memory_records"] == 0
    assert row["task_type"] == "binary" and row["metric"] == "f1" and row["finished_at"] >= row["started_at"]

    again = []
    assert run(SMALL, where, again) == [] and again == []
    assert len(benchmark.read_cells(where[1])) == 20


def test_an_interrupted_cell_has_no_row_and_runs_again(where):
    def stop_at_third(count):
        if count == 3:
            raise KeyboardInterrupt

    calls = []
    with pytest.raises(KeyboardInterrupt):
        run(SMALL, where, calls, factory=lambda name: StandIn(name, calls, on_run=stop_at_third))
    assert [r["cell_id"] for r in benchmark.read_cells(where[1])] == [c.cell_id for c in SMALL[:2]]

    second = []
    run(SMALL, where, second)
    assert second == [c.cell_id for c in SMALL[2:]]
    assert [r["cell_id"] for r in benchmark.read_cells(where[1])] == [c.cell_id for c in SMALL]


def test_failed_cells_are_rows_and_can_be_run_again(where):
    broken = benchmark.Dataset("broken", TOYS["toy_a"].frame, "no_such_column", TaskType.BINARY)
    loads = {**TOYS, "broken": broken}
    cells = [benchmark.Cell("s1", "toy_a", 20, 1), benchmark.Cell("s2", "toy_a", 20, 1),
             benchmark.Cell("s1", "broken", 20, 1)]
    settings, rows_path = where
    calls = []
    failing = lambda name: StandIn(name, calls, status="failed" if name == "s2" else "completed")  # noqa: E731
    benchmark.run(cells, settings, rows_path, make=failing, load=loads.__getitem__)
    first = benchmark.read_cells(rows_path)
    assert [r["status"] for r in first] == ["completed", "failed", "failed"]
    assert first[1]["error"] == "RuntimeError: stand-in failure" and first[1]["test_score"] is None
    assert "no_such_column" in first[2]["error"] and first[2]["split"] is None  # the split failed: no system ran
    assert calls == ["s1-toy_a-20s-s1", "s2-toy_a-20s-s1"]

    assert benchmark.run(cells, settings, rows_path, make=failing, load=loads.__getitem__) == []
    calls.clear()
    fixed = lambda name: StandIn(name, calls)  # noqa: E731
    retried = benchmark.run(cells, settings, rows_path, make=fixed, load=loads.__getitem__, retry_failed=True)
    assert calls == ["s2-toy_a-20s-s1"] and [r["cell_id"] for r in retried] == ["s2-toy_a-20s-s1", "s1-broken-20s-s1"]
    rows = benchmark.read_cells(rows_path)
    assert len(rows) == 5  # earlier rows are never edited
    latest = benchmark.latest(rows)
    assert latest["s2-toy_a-20s-s1"]["status"] == "completed" and latest["s1-broken-20s-s1"]["status"] == "failed"


# --------------------------------------------------------------------------- the frozen memory
def test_every_row_has_the_memory_hash_and_another_memory_is_refused(tmp_path, where):
    path = tmp_path / "frozen.jsonl"
    ExperimentMemory(path).add(record(0))
    calls = []
    run(SMALL[:3], where, calls, memory_path=path)
    rows = benchmark.read_cells(where[1])
    expected = ExperimentMemory(path).snapshot_hash()
    assert {r["memory_hash"] for r in rows} == {expected} and {r["memory_records"] for r in rows} == {1}
    assert rows[0]["memory_file"].endswith("frozen.jsonl")

    ExperimentMemory(path).add(record(1))
    with pytest.raises(ValueError, match="another memory"):
        run(SMALL, where, calls, memory_path=path)
    assert len(calls) == 3

    def change_memory(count):
        ExperimentMemory(path).add(record(10 + count))

    fresh = benchmark.folders(tmp_path / "second")
    with pytest.raises(RuntimeError, match="changed during the run"):
        run(SMALL, fresh, calls, factory=lambda name: StandIn(name, calls, on_run=change_memory), memory_path=path)
    assert benchmark.read_cells(fresh[1]) == []  # the cell during which it changed has no row


def test_the_memory_must_come_from_development_datasets(tmp_path):
    development, evaluation = toy("dev_set", 1), toy("eval_set", 2, rows=210)  # fingerprints differ by the rows
    path = tmp_path / "memory.jsonl"
    memory = ExperimentMemory(path)
    memory.add(dataclasses.replace(record(0), dataset_name="dev_set"))  # by name
    memory.add(dataclasses.replace(record(1), dataset_name="renamed",
                                   dataset_fingerprint=dataset_fingerprint(development.frame, "target")))
    memory.add(dataclasses.replace(record(2), dataset_name="eval_set"))
    memory.add(dataclasses.replace(record(3), dataset_name="other",
                                   dataset_fingerprint=dataset_fingerprint(evaluation.frame, "target")))
    memory.add(dataclasses.replace(record(4), dataset_name="my_upload"))
    from_evaluation, unknown = benchmark.memory_sources(path, [development], [evaluation])
    assert from_evaluation == ["SES-001/EXP-002", "SES-001/EXP-003"] and unknown == ["SES-001/EXP-004"]


def test_the_checks_before_a_run():
    grid = benchmark.grid(["titanic", "no_such_data"], seeds=[1], budgets=[120], systems=["manual", "autopilot_full"],
                          ablations=[])
    problems = benchmark.preflight(grid, None, require_memory=True, evaluation_only=True)
    assert any(p.startswith("titanic is a development dataset") for p in problems)
    assert any(p.startswith("no_such_data is not in research/datasets.json") for p in problems)
    assert any(p.startswith("autopilot_full read the frozen Experiment_Memory") for p in problems)
    smoke = benchmark.smoke_grid(systems=["manual", "autopilot_full"], ablations=[])
    assert benchmark.preflight(smoke, None, require_memory=False, evaluation_only=False) == []


# --------------------------------------------------------------------------- the rows file
def test_the_rows_file_is_checked_and_text_stays_text(where, tmp_path):
    calls = []
    run(SMALL[:2], where, calls)
    rows_path = where[1]
    data = rows_path.read_bytes()
    rows_path.write_bytes(data[:-1])  # a last line without its end, as after a crash while writing
    with pytest.raises(ValueError, match="incomplete"):
        run(SMALL, where, calls)
    other = tmp_path / "other.csv"
    other.write_text("cell_id,system\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="other columns"):
        benchmark.run(SMALL, where[0], other, make=lambda name: StandIn(name, calls), load=TOYS.__getitem__)
    assert len(calls) == 2

    row = {**benchmark.read_cells(where[1])[0], "git_commit": "1e84091", "memory_hash": "0123456789"}
    copy = tmp_path / "copy.csv"
    benchmark.append_row(copy, row)
    read = benchmark.read_cells(copy)[0]
    assert read["git_commit"] == "1e84091" and read["memory_hash"] == "0123456789" and read["seed"] == 1


# --------------------------------------------------------------------------- progress and a real cell
def test_the_time_left_and_the_timing_table():
    rows = [
        {"cell_id": "a", "system": "s1", "budget_seconds": 20, "wall_seconds": 25.0, "status": "completed"},
        {"cell_id": "b", "system": "s1", "budget_seconds": 20, "wall_seconds": 35.0, "status": "completed"},
        {"cell_id": "c", "system": "s2", "budget_seconds": 20, "wall_seconds": 5.0, "status": "failed"},
    ]
    left = [benchmark.Cell("s1", "x", 20, 3), benchmark.Cell("s2", "x", 20, 3), benchmark.Cell("s1", "x", 40, 3)]
    assert benchmark.estimate_seconds(left, rows) == 30.0 + 5.0 + 40.0  # unknown budget: the budget itself
    table = {r["system"]: r for r in benchmark.timing(rows)}
    assert table["s1"] == {"system": "s1", "budget_seconds": 20, "cells": 2, "completed": 2, "failed": 0,
                           "mean_wall_seconds": 30.0, "max_wall_seconds": 35.0, "mean_seconds_above_budget": 10.0}
    assert table["s2"]["failed"] == 1 and table["s2"]["completed"] == 0


def test_a_real_cell_with_the_manual_system(where):
    settings, rows_path = where
    benchmark.run([benchmark.Cell("manual", "toy_a", 30, 1)], settings, rows_path, make=lambda name: ManualSystem(),
                  load=TOYS.__getitem__)
    row = benchmark.read_cells(rows_path)[0]
    assert row["status"] == "completed" and row["test_score"] == row["test_f1"] and row["model"].startswith("LightGBM")
    assert all(row[f"test_{m}"] is not None for m in ("accuracy", "balanced_accuracy", "f1", "roc_auc", "log_loss"))
    assert all(row[f"test_{m}"] is None for m in ("mae", "mse", "rmse", "r2"))
    assert json.loads(row["system_settings"])["config"] == LIGHTGBM_DEFAULTS
