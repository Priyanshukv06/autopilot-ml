"""Tests for target suggestion and the possible_leakage warning (optional task 33, Requirement 30)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.critic import Critic, CriticThresholds, DiagnosisKind, Severity, ThresholdsError
from app.services.profiler import DatasetProfiler, single_column_scores
from app.services.strategy_engine import propose
from app.services.task_detector import TaskDetector, TaskType
from app.services.automl_runner import ExperimentConfig
from tests.test_autopilot import DATA, run_session
from tests.test_critic import THRESHOLDS, assessment, fold

ROOT = Path(__file__).resolve().parents[1]


def leaky(n=300, seed=0, regression=False):
    """Two honest columns and two planted leaks: a numeric copy and a text copy of the target."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    if regression:
        target = 3 * x + rng.normal(size=n)
        return pd.DataFrame({"x": x, "z": rng.normal(size=n), "after": 2 * target + 0.001 * rng.normal(size=n),
                             "target": target})
    target = np.where(x + rng.normal(size=n) > 0, "yes", "no")
    return pd.DataFrame({"x": x, "city": rng.choice(["a", "b", "c"], n), "after": (target == "yes") + 0.01 * x,
                         "status": np.where(target == "yes", "closed", "open"), "target": target})


# --------------------------------------------------------------------------- target suggestion
def test_targets_are_suggested_with_reasons():
    titanic = pd.read_csv(ROOT / "data" / "sample" / "titanic.csv")
    suggestions = TaskDetector().suggest_targets(titanic)
    assert suggestions[0].column == "Survived" and "a 0/1 column" in suggestions[0].reasons
    assert len(suggestions) == 3 and all(s.reasons for s in suggestions)
    assert "PassengerId" not in [s.column for s in suggestions] and "Name" not in [s.column for s in suggestions]
    df = pd.DataFrame({"income_class": ["low", "high"] * 10, "age": np.arange(20.0), "score": np.arange(20) % 3})
    first = TaskDetector().suggest_targets(df)[0]
    assert first.column == "income_class" and "its name contains 'class'" in first.reasons
    assert first.text().startswith("'income_class' (its name contains 'class'")
    assert TaskDetector().suggest_target(df) == "income_class"


# --------------------------------------------------------------------------- the single-column check
@pytest.mark.parametrize("regression", [False, True])
def test_planted_leaks_score_almost_perfectly_and_honest_columns_do_not(regression):
    df = leaky(regression=regression)
    task = TaskType.REGRESSION if regression else TaskType.BINARY
    scores, metric = single_column_scores(df, "target", task)
    assert metric == ("r2" if regression else "balanced_accuracy")
    assert scores["after"] >= 0.99 and scores["x"] < 0.99
    if not regression:
        assert scores["status"] == 1.0 and scores["city"] < 0.7  # the text copy, and a random text column
    profile = DatasetProfiler().profile(df, "target", task_type=task, check_leakage=True)
    assert profile.single_column_scores == scores and profile.single_column_metric == metric
    assert any("'after' alone predicts the target almost perfectly" in w for w in profile.warnings)
    plain = DatasetProfiler().profile(df, "target", task_type=task)  # not asked for: not computed
    assert plain.single_column_scores == {} and plain.single_column_metric is None


# --------------------------------------------------------------------------- the Critic
def diagnose(profile, config, thresholds=THRESHOLDS):
    return Critic(CriticThresholds.from_mapping(thresholds)).diagnose(
        profile=profile, config=config, assessment=assessment([fold(0.8, 0.8)] * 3), search=None,
        task_type=TaskType.BINARY, primary_metric="f1")


def test_possible_leakage_only_warns():
    profile = DatasetProfiler().profile(leaky(), "target", task_type=TaskType.BINARY, check_leakage=True)
    config = ExperimentConfig(target_column="target", excluded_columns=["status"])
    found = [d for d in diagnose(profile, config) if d.kind is DiagnosisKind.POSSIBLE_LEAKAGE]
    assert [d.columns for d in found] == [("after",)]  # 'status' is excluded, so it cannot leak
    assert found[0].severity is Severity.HIGH and found[0].evidence["metric"] == "balanced_accuracy"
    assert found[0].evidence["min_score"] == 0.99 and "nothing is changed automatically" in found[0].message
    assert propose(found, config, profile, TaskType.BINARY, set()) == []  # no Strategy answers it
    strict = {**THRESHOLDS, "possible_leakage": {"min_score": 1.0}}
    assert [d.columns for d in diagnose(profile, config, strict) if d.kind is DiagnosisKind.POSSIBLE_LEAKAGE] in \
        ([], [("after",)])  # the file may set the threshold
    with pytest.raises(ThresholdsError, match="min_score"):
        CriticThresholds.from_mapping({**THRESHOLDS, "possible_leakage": {"min_score": 1.5}})
    assert CriticThresholds.load()["possible_leakage.min_score"] == 0.99  # the frozen file uses the default


def test_a_session_warns_about_a_leak_and_never_acts_on_it(tmp_path):
    data = DATA.assign(after=lambda d: (d["target"] == d["target"].max()).astype(float))
    result, record, *_ = run_session(tmp_path, data=data, shifts={1: 0.0, 2: 0.0, 3: -0.05})
    first = record["steps"][0]
    leaks = [d for d in first["diagnoses"] if d["kind"] == "possible_leakage"]
    assert [d["columns"] for d in leaks] == [["after"]]
    for step in record["steps"][1:]:  # the warning never becomes a change
        spec = (step["proposal"] or {}).get("spec") or {}
        assert "after" not in json.dumps(spec) and "possible_leakage" not in (step["proposal"] or {}).get("answers", [])
    profile = json.loads((tmp_path / "experiments" / result.session_id / "train_profile.json").read_text("utf-8"))
    assert profile["single_column_scores"]["after"] == 1.0  # computed on the Training_Part
