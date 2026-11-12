"""Smoke test of the critic study (task 16): a few small cases with a fast stand-in engine."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from app.services.automl_runner import EngineResult
from app.services.critic import OPTIONAL_THRESHOLDS, CriticThresholds
from research import critic_study as cs


class LogisticEngine:
    """Logistic regression on the numeric columns; no search."""

    name = "logistic"

    def fit(self, X, y, **kwargs):
        model = Numeric(X, y)
        history = [{"trial": 0, "estimator": "logreg", "validation_score": 0.7, "trial_time": 0.1,
                    "wall_clock_time": 0.1, "config": {}}]
        return EngineResult(model=model, engine_name=self.name, engine_version="test", best_estimator="logreg",
                            best_model_name="Logistic", best_config={}, best_validation_score=0.7,
                            validation_method="none", training_time=0.0, search_history=history, leaderboard=[],
                            native_importance=None, classes=list(model.classes_))

    def fit_config(self, X, y, **kwargs):
        return Numeric(X, y)


class Numeric:
    def __init__(self, X, y):
        self.columns = [c for c in X.columns if X[c].dtype.kind in "fi"]
        self.model = LogisticRegression(max_iter=500).fit(X[self.columns].fillna(0), y)
        self.classes_ = self.model.classes_

    def predict(self, X):
        return self.model.predict(X[self.columns].fillna(0))

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.columns].fillna(0))


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    cases = [c for c in cs.study_cases([0]) if (c.problem, c.strength) in {
        ("clean", None), ("missing_values", 0.8), ("constant_column", 1.0), ("duplicated_columns", 0.995)}
        and c.replicate == 0]
    rows, inputs = [], {}
    for case in cases:
        row, saved = cs.run_case(case, work_dir=tmp_path_factory.mktemp(case.case_id), engine=LogisticEngine(),
                                 n_jobs=1, fresh_rows=500)
        rows.append(row)
        inputs[row["case_id"]] = saved
    return rows, inputs


def test_cases_cover_every_problem_and_strength_on_separate_seeds():
    tuning, reporting = cs.study_cases(cs.TUNING_SEEDS), cs.study_cases(cs.REPORTING_SEEDS)
    per_seed = sum(len(s) for s in cs.STRENGTHS.values()) + cs.N_CLEAN
    assert len(tuning) == len(reporting) == per_seed * 5
    assert {c.split for c in tuning} == {"tuning"} and {c.split for c in reporting} == {"reporting"}
    assert len({c.generator_seed for c in tuning + reporting}) == len(tuning) + len(reporting)  # all datasets differ
    assert len({c.case_id for c in tuning + reporting}) == len(tuning) + len(reporting)


@pytest.mark.slow
def test_rows_have_the_truth_and_the_evidence(results):
    rows, _ = results
    by_problem = {r["problem"]: r for r in rows}
    assert by_problem["missing_values"]["truth_missing_values"] and not by_problem["clean"]["truth_missing_values"]
    assert by_problem["missing_values"]["ev_max_missing_share"] == pytest.approx(0.8)
    assert by_problem["constant_column"]["ev_single_value"] is True
    assert by_problem["duplicated_columns"]["ev_max_correlation"] > 0.99
    assert by_problem["clean"]["ev_max_correlation"] is None  # no pair reaches the profile's 0.90 level
    for row in rows:
        assert row["true_gap"] == pytest.approx(row["train_f1"] - row["fresh_f1"])
        assert 0 <= row["fresh_f1"] <= 1 and not row["assessment_failed"]


@pytest.mark.slow
def test_the_study_rule_agrees_with_the_real_critic(results):
    rows, inputs = results
    for values in (cs.V0_START, {**cs.V0_START, "missing_values.moderate_share": 0.9,
                                 "missing_values.high_share": 0.95, "overfitting.moderate_gap": 0.0}):
        thresholds = CriticThresholds.from_mapping({"version": "t", **_tables(values)})
        for row in rows:
            expected = {kind for kind in cs.STUDIED_KINDS if cs.predict(row, kind, thresholds.values)}
            assert cs.replay(inputs[row["case_id"]], thresholds) == expected, row["case_id"]


@pytest.mark.slow
def test_tuning_and_scores(results):
    rows, _ = results
    start = CriticThresholds.from_mapping({"version": "v0-start", **_tables(cs.V0_START)})
    missing = cs.scores(rows, "missing_values", start.values)
    assert (missing["tp"], missing["fp"], missing["fn"]) == (1, 0, 0) and missing["precision"] == 1.0
    tuned, grid = cs.tune(rows, start)
    assert set(tuned) == set(start.values)
    CriticThresholds.from_mapping({"version": "tuned", **_tables(tuned)})  # a valid set of thresholds
    assert {g["threshold"] for g in grid} == {key for key, _ in cs.TUNED.values()}
    result = cs.analyse([{**r, "split": "tuning" if i % 2 else "reporting"} for i, r in enumerate(rows)], start)
    assert {s["kind"] for s in result["summary"]} == set(cs.STUDIED_KINDS)


def test_the_thresholds_file_text_loads(tmp_path):
    path = tmp_path / "thresholds.toml"
    path.write_text(cs.thresholds_toml(cs.V0_START, "v-test", "a note\nover two lines"), encoding="utf-8")
    loaded = CriticThresholds.load(path)
    # thresholds of rules added after the study keep their defaults (optional task 33)
    assert loaded.version == "v-test" and loaded.values == {**cs.V0_START, **OPTIONAL_THRESHOLDS}


def _tables(values):
    tables = {}
    for key, value in values.items():
        section, name = key.split(".", 1)
        tables.setdefault(section, {})[name] = value
    return tables
