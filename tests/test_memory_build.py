"""Tests for the benchmark memory build (spec task 26): Sessions on development datasets only, each once,
the growing memory in the rows, and a frozen copy with only the records of finished Sessions."""

import dataclasses

import pytest

from app.services.experiment_memory import ExperimentMemory
from research import benchmark, memory_build
from research.systems import AutopilotSystem, SystemResult
from tests.test_benchmark import TOYS
from tests.test_experiment_memory import record

LOAD = {"titanic": TOYS["toy_a"], "penguins": TOYS["toy_b"]}.__getitem__


class RecordingSession:
    """Stands in for a Session: adds one memory record per run, then returns a result naming its Session."""

    def __init__(self, memory_path, counter, status="completed"):
        self.memory_path, self.counter, self.status = memory_path, counter, status
        self.name = memory_build.SYSTEM

    def run(self, problem, prepared, settings):
        self.counter.append(problem.dataset)
        session = f"SES-{len(self.counter):03d}"
        ExperimentMemory(self.memory_path).add(dataclasses.replace(record(len(self.counter)), session_id=session,
                                                                   dataset_name=problem.dataset))
        if self.status == "failed":
            return SystemResult(self.name, "failed", None, {}, 0.01, 1, None, error="RuntimeError: stand-in")
        return SystemResult(self.name, "completed", 0.7, {"f1": 0.7}, 0.01, 2, None,
                            {"session_id": session, "stop_reason": "time", "kept": []})


def test_the_build_settings_and_its_sessions():
    cells = memory_build.cells()
    assert len(cells) == 30 and {c.budget_seconds for c in cells} == {300} and [c.seed for c in cells[:10]] == [1] * 10
    assert len({c.dataset for c in cells}) == 10 and cells[0].cell_id == "memory_build-titanic-300s-s1"
    system = memory_build.make(memory_build.folders(".")["memory"])
    assert isinstance(system, AutopilotSystem) and system.options == {"selector": "random", "use_memory": False}
    assert system.read_only is False and system.describe()["memory_read_only"] is False


def test_each_session_runs_once_and_the_rows_show_the_growing_memory(tmp_path):
    where = memory_build.folders(tmp_path)
    cells = memory_build.cells(["titanic", "penguins"], seeds=[1, 2], seconds=20)
    runs = []
    make = lambda name: RecordingSession(where["memory"], runs)  # noqa: E731
    memory_build.build(cells, where["settings"], where["rows"], where["memory"], make_system=make, load=LOAD)
    rows = benchmark.read_cells(where["rows"])
    assert runs == ["titanic", "penguins", "titanic", "penguins"]
    assert [r["memory_records"] for r in rows] == [0, 1, 2, 3]  # each row: the memory its Session found
    assert rows[0]["memory_hash"] == ExperimentMemory(tmp_path / "none.jsonl").snapshot_hash()
    assert memory_build.build(cells, where["settings"], where["rows"], where["memory"], make_system=make,
                              load=LOAD) == [] and len(runs) == 4  # a restart runs nothing again

    with pytest.raises(ValueError, match="development datasets only"):
        memory_build.build(memory_build.cells(["pc4"], seeds=[1]), where["settings"], where["rows"],
                           where["memory"], make_system=make, load=LOAD)


def test_the_frozen_memory_keeps_only_finished_sessions_and_is_never_rewritten(tmp_path):
    where = memory_build.folders(tmp_path)
    runs = []
    finished = memory_build.cells(["titanic"], seeds=[1], seconds=20)
    memory_build.build(finished, where["settings"], where["rows"], where["memory"],
                       make_system=lambda name: RecordingSession(where["memory"], runs), load=LOAD)
    failing = memory_build.cells(["penguins"], seeds=[1], seconds=20)
    memory_build.build(failing, where["settings"], where["rows"], where["memory"],
                       make_system=lambda name: RecordingSession(where["memory"], runs, "failed"), load=LOAD)
    ExperimentMemory(where["memory"]).add(dataclasses.replace(record(9), session_id="SES-009"))  # interrupted
    assert len(ExperimentMemory(where["memory"]).records()) == 3

    result = memory_build.freeze(where["memory"], benchmark.read_cells(where["rows"]), where["frozen"])
    frozen = ExperimentMemory(where["frozen"]).records()
    assert [r.session_id for r in frozen] == ["SES-001"] and frozen[0].dataset_name == "titanic"
    assert result == {"sessions": 1, "records": 1, "left_out": 2,
                      "sha256": ExperimentMemory(where["frozen"]).snapshot_hash()}
    with pytest.raises(FileExistsError, match="never rewritten"):
        memory_build.freeze(where["memory"], benchmark.read_cells(where["rows"]), where["frozen"])


def test_the_summary_counts_records_per_arm(tmp_path):
    path = tmp_path / "memory.jsonl"
    memory = ExperimentMemory(path)
    memory.add(record(0, kinds=("overfitting", "high_cardinality"), verdict="keep", diff=0.02, cost=10.0))
    memory.add(record(1, kinds=("overfitting",), verdict="discard", diff=-0.01, cost=20.0))
    table = memory_build.summary(path).set_index(["strategy", "diagnosis_kind"])
    assert table.loc[("drop_columns", "overfitting"), "records"] == 2
    assert table.loc[("drop_columns", "overfitting"), "keeps"] == 1
    assert table.loc[("drop_columns", "overfitting"), "mean_cost_seconds"] == 15.0
    assert table.loc[("drop_columns", "high_cardinality"), "records"] == 1
    assert memory_build.summary(tmp_path / "empty.jsonl").empty
