"""Tests for the comparison systems (spec task 23.1): the shared split and columns, the manual and FLAML
systems on a small frame, the AutoGluon bridge with a stand-in worker process, and the same metrics on the
same rows for every system."""

import json
import sys
import textwrap

import numpy as np
import pytest

from app.core.config import build_settings
from app.services.automl_runner import FLAMLEngine
from app.services.evaluator import Evaluator
from app.services.splits import make_outer_split
from app.services.task_detector import TaskType
from research import systems
from research.synthetic import make_synthetic

CLASSIFICATION_KEYS = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"}


class FixedTrialsEngine(FLAMLEngine):
    """FLAML that stops after 2 trials, so a search takes about a second; records fixed configurations."""

    def __init__(self):
        super().__init__()
        self.configs = []

    def fit(self, *args, **kwargs):
        kwargs["max_iter"] = 2
        return super().fit(*args, **kwargs)

    def fit_config(self, *args, **kwargs):
        self.configs.append(dict(kwargs["config"]))
        return super().fit_config(*args, **kwargs)


@pytest.fixture
def setting(tmp_path):
    settings = build_settings(tmp_path / "project")
    settings.ensure_directories()
    return settings


def problem(task="binary", seed=0, budget=10):
    data = make_synthetic(task, 300, [("identifier_column", 1.0)], seed=seed, fresh_rows=10)
    return systems.Problem("synthetic", data.frame, "target", TaskType(task), seed=seed, budget_seconds=budget, n_jobs=1)


def stand_in_worker(tmp_path, name, body):
    """A worker script for the current Python: it records the settings it got, then runs ``body``."""
    seen = tmp_path / f"{name}_settings.json"
    path = tmp_path / f"{name}.py"
    header = textwrap.dedent(f"""\
        import json, sys, time
        from pathlib import Path
        import pandas as pd
        folder = Path(sys.argv[1])
        settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        train, test = pd.read_csv(folder / "train.csv"), pd.read_csv(folder / "test.csv")
        Path({str(seen)!r}).write_text(json.dumps(settings), encoding="utf-8")
    """)
    path.write_text(header + textwrap.dedent(body), encoding="utf-8")
    return path, seen


MAJORITY = """\
labels = sorted(train[settings["label"]].unique())
majority = train[settings["label"]].mode()[0]
out = pd.DataFrame({"prediction": [majority] * len(test)})
for label in labels:
    out[f"proba::{label}"] = [1.0 if label == majority else 0.0] * len(test)
out.to_csv(folder / "predictions.csv", index=False)
(folder / "result.json").write_text(json.dumps({"best_model": "Majority", "models": ["Majority"]}))
"""
MEAN = """\
out = pd.DataFrame({"prediction": [train[settings["label"]].mean()] * len(test)})
out.to_csv(folder / "predictions.csv", index=False)
(folder / "result.json").write_text(json.dumps({"best_model": "Mean", "models": ["Mean"]}))
"""


# --------------------------------------------------------------------------- the shared split
def test_every_system_gets_the_phase1_split_and_the_first_setup_columns(setting):
    p = problem()
    prepared = systems.prepare(p, setting)
    reference, _ = make_outer_split(p.frame["target"], TaskType.BINARY, 0.2, p.seed)
    assert prepared.split.fingerprint() == reference.fingerprint()
    assert prepared.excluded_columns == ["record_id"]  # the planted identifier column
    assert "record_id" not in prepared.X_train and "target" not in prepared.X_train
    assert len(prepared.X_train) == 240 and len(prepared.X_test) == 60
    assert list(prepared.y_test.index) == list(p.frame.index[reference.test_positions])


# --------------------------------------------------------------------------- manual and FLAML
def test_the_manual_system_is_lightgbm_with_library_defaults(setting):
    p = problem()
    engine = FixedTrialsEngine()
    result = systems.ManualSystem(engine).run(p, systems.prepare(p, setting), setting)
    assert result.status == "completed" and result.n_experiments == 1 and result.error is None
    assert engine.configs == [systems.LIGHTGBM_DEFAULTS]
    assert set(result.test_metrics) == CLASSIFICATION_KEYS and result.test_score == result.test_metrics["f1"]


def test_the_flaml_system_searches_the_shared_training_rows(setting):
    p = problem()
    prepared = systems.prepare(p, setting)
    result = systems.FlamlSystem(FixedTrialsEngine()).run(p, prepared, setting)
    assert result.status == "completed" and result.details["n_trials"] <= 2
    summary = json.loads((setting.experiments_dir / result.details["experiment_id"] / "summary.json").read_text())
    assert summary["n_test"] == len(prepared.y_test) and summary["excluded_columns"] == ["record_id"]
    assert set(result.test_metrics) == CLASSIFICATION_KEYS


def test_a_failing_system_returns_a_failed_result(setting):
    class Broken(FLAMLEngine):
        def fit_config(self, *args, **kwargs):
            raise RuntimeError("no model today")

    p = problem()
    result = systems.ManualSystem(Broken()).run(p, systems.prepare(p, setting), setting)
    assert result.status == "failed" and "no model today" in result.error and result.test_score is None


# --------------------------------------------------------------------------- the AutoGluon bridge
def test_the_autogluon_bridge_scores_the_workers_predictions_with_our_evaluator(tmp_path, setting):
    p = problem()
    prepared = systems.prepare(p, setting)
    worker, seen = stand_in_worker(tmp_path, "majority", MAJORITY)
    result = systems.AutoGluonSystem(python=sys.executable, worker=worker).run(p, prepared, setting)
    assert result.status == "completed", result.error
    majority, classes = prepared.y_train.mode()[0], sorted(prepared.y_train.unique())
    expected = Evaluator(TaskType.BINARY, "f1").compute_metrics(
        prepared.y_test, np.full(len(prepared.y_test), majority),
        np.array([[1.0 if c == majority else 0.0 for c in classes]] * len(prepared.y_test)), classes)
    assert result.test_metrics == pytest.approx(expected, nan_ok=True)
    assert result.model == "AutoGluon Majority"
    settings = json.loads(seen.read_text())
    assert settings["eval_metric"] == "f1" and settings["positive_class"] == classes[-1]
    assert settings["problem_type"] == "binary" and settings["num_cpus"] == 1 and settings["time_limit"] == 10


def test_the_autogluon_bridge_for_regression_and_its_failures(tmp_path, setting):
    p = problem(task="regression")
    prepared = systems.prepare(p, setting)
    worker, seen = stand_in_worker(tmp_path, "mean", MEAN)
    result = systems.AutoGluonSystem(python=sys.executable, worker=worker).run(p, prepared, setting)
    assert result.status == "completed" and set(result.test_metrics) == {"mae", "mse", "rmse", "r2"}
    settings = json.loads(seen.read_text())
    assert settings["eval_metric"] == "r2" and settings["positive_class"] is None
    assert settings["presets"] == "medium_quality"

    crash, _ = stand_in_worker(tmp_path, "crash", "raise SystemExit('worker exploded')\n")
    failed = systems.AutoGluonSystem(python=sys.executable, worker=crash).run(p, prepared, setting)
    assert failed.status == "failed" and "worker exploded" in failed.error

    slow, _ = stand_in_worker(tmp_path, "slow", "time.sleep(30)\n")
    quick = systems.Problem(p.dataset, p.frame, p.target, p.task_type, p.seed, budget_seconds=1, n_jobs=1)
    timed_out = systems.AutoGluonSystem(python=sys.executable, worker=slow, extra_seconds=1).run(quick, prepared,
                                                                                              setting)
    assert timed_out.status == "failed" and "Timeout" in timed_out.error


# --------------------------------------------------------------------------- every system, the same rows
@pytest.mark.slow
def test_all_systems_report_the_same_metrics_on_the_same_rows(tmp_path, setting):
    p = problem(budget=30)
    prepared = systems.prepare(p, setting)
    engine = FixedTrialsEngine()
    worker, _ = stand_in_worker(tmp_path, "majority", MAJORITY)
    runs = {
        "manual": systems.ManualSystem(engine),
        "flaml": systems.FlamlSystem(engine),
        "autogluon": systems.AutoGluonSystem(python=sys.executable, worker=worker),
        "autopilot_rules": systems.AutopilotSystem("autopilot_rules", None, engine, max_experiments=2),
    }
    results = {name: system.run(p, prepared, setting) for name, system in runs.items()}
    assert all(r.status == "completed" for r in results.values()), {n: r.error for n, r in results.items()}
    assert all(set(r.test_metrics) == CLASSIFICATION_KEYS for r in results.values())
    session = json.loads((setting.experiments_dir / results["autopilot_rules"].details["session_id"]
                          / "session.json").read_text())
    assert session["final_evaluation"]["n_test"] == len(prepared.y_test)
    assert session["splits"]["outer"] == prepared.split.fingerprint()
    assert session["first_setup_exclusions"] == prepared.excluded_columns
