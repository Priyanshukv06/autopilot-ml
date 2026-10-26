"""Tests for the thin Critic (task 6.1 of the spec), with hand-made profiles and Assessments (no FLAML).

Most tests use thresholds that are exact in binary floating point (0.125, 0.25, ...), so that a
value exactly at a threshold can be tested.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.assessor import Assessment, FoldScore
from app.services.automl_runner import ExperimentConfig
from app.services.critic import (
    Critic,
    CriticThresholds,
    DiagnosisKind,
    Severity,
    ThresholdsError,
    diagnostic_metric,
)
from app.services.profiler import DatasetProfiler
from app.services.task_detector import TaskType

THRESHOLDS = {
    "version": "test-1",
    "overfitting": {"moderate_gap": 0.125, "high_gap": 0.25},
    # The rules of the full version, set so that the thin-rule tests below do not trigger them by accident.
    "underfitting": {"max_gap": 0.03, "moderate_skill": 0.25, "high_skill": 0.125},
    "budget_limited": {"moderate_gain": 0.005, "high_gain": 0.02},
    "wasted_budget": {"moderate_share": 0.5, "high_share": 0.75},
    "class_imbalance": {"moderate_ratio": 3, "high_ratio": 9, "moderate_recall": 0.5, "high_recall": 0.25,
                        "min_ratio_for_recall": 1.5},
    "missing_values": {"moderate_share": 0.4, "high_share": 0.7},
    "high_cardinality": {"moderate_distinct": 50, "moderate_unseen": 0.25, "high_unseen": 0.5},
    "constant_column": {"near_constant_share": 0.99},
    "redundant_features": {"moderate_correlation": 0.95, "high_correlation": 0.99},
    "unreliable_validation": {"moderate_std": 0.05, "high_std": 0.08},
    "weak_model_families": {"min_gap": 0.1, "min_time_share": 0.05},
}
CRITIC = Critic(CriticThresholds.from_mapping(THRESHOLDS))
CONFIG = ExperimentConfig(target_column="target")


def frame(classes=None, n_codes=3, rows=None, regression=False):
    """A Training_Part with a numeric column, a text column 'code' and a target with given class counts."""
    if regression:
        target = np.linspace(0, 1, rows or 100)
    else:
        target = np.concatenate([[label] * count for label, count in (classes or {"a": 50, "b": 50}).items()])
    n = len(target)
    return pd.DataFrame({"x": np.arange(n, dtype=float), "code": [f"c{i % n_codes}" for i in range(n)],
                         "target": target})


def profile_of(df, task_type=TaskType.BINARY):
    return DatasetProfiler().profile(df, "target", task_type=task_type)


def fold(train=0.5, valid=0.5, metric="f1", recall=None, unseen=None, baseline=0.0):
    return FoldScore(repeat=0, fold=0, train_metrics={metric: train}, validation_metrics={metric: valid},
                     class_recall=recall, unseen_share=unseen or {},
                     baseline_validation={} if baseline is None else {metric: baseline}, fit_seconds=0.1)


def assessment(folds, metric="f1"):
    return Assessment(setup_key="key", experiment_id="EXP-001", estimator="lgbm", config={}, primary_metric=metric,
                      folds_fingerprint="fp", folds=list(folds), seconds=1.0)


def diagnose(profile, folds, task_type=TaskType.BINARY, metric="f1", config=CONFIG, critic=CRITIC, search=None):
    return critic.diagnose(profile=profile, config=config, assessment=assessment(folds, metric), search=search,
                           task_type=task_type, primary_metric=metric)


def only(diagnoses, kind):
    found = [d for d in diagnoses if d.kind is kind]
    assert len(found) <= 1
    return found[0] if found else None


# --------------------------------------------------------------------------- overfitting
@pytest.mark.parametrize(
    "train, valid, severity",
    [(0.75, 0.5, Severity.HIGH), (0.875, 0.75, Severity.MODERATE), (0.875, 0.7505, None), (0.5, 0.5, None)],
)
def test_overfitting_levels(train, valid, severity):
    found = only(diagnose(profile_of(frame()), [fold(train, valid)] * 15), DiagnosisKind.OVERFITTING)
    assert (found.severity if found else None) is severity
    if found:
        assert found.columns == ()
        assert found.evidence == {"metric": "f1", "gap": train - valid, "train_mean": train,
                                  "validation_mean": valid, "folds": 15}


def test_overfitting_gap_is_the_mean_over_folds_and_skips_missing_scores():
    folds = [fold(0.5, 0.5), fold(0.75, 0.5), fold(0.75, None)]  # gaps 0 and 0.25; the last fold has no score
    found = only(diagnose(profile_of(frame()), folds), DiagnosisKind.OVERFITTING)
    assert found.severity is Severity.MODERATE
    assert found.evidence["gap"] == 0.125 and found.evidence["folds"] == 2


# --------------------------------------------------------------------------- class imbalance
@pytest.mark.parametrize(
    "counts, recall, severity",
    [
        ({"a": 900, "b": 100}, 0.9, Severity.HIGH),  # ratio 9
        ({"a": 300, "b": 100}, 0.9, Severity.MODERATE),  # ratio 3
        ({"a": 290, "b": 100}, 0.9, None),  # ratio 2.9
        ({"a": 200, "b": 100}, 0.4, Severity.MODERATE),  # ratio 2, recall below 0.5
        ({"a": 200, "b": 100}, 0.2, Severity.HIGH),  # ratio 2, recall below 0.25
        ({"a": 200, "b": 100}, 0.5, None),  # recall exactly 0.5 is not below it
        ({"a": 140, "b": 100}, 0.2, None),  # ratio 1.4: below the ratio where recall counts
    ],
)
def test_class_imbalance_levels(counts, recall, severity):
    folds = [fold(recall={"a": 0.95, "b": recall})] * 15
    found = only(diagnose(profile_of(frame(counts)), folds), DiagnosisKind.CLASS_IMBALANCE)
    assert (found.severity if found else None) is severity
    if found:
        assert found.columns == ("target",)
        assert found.evidence["imbalance_ratio"] == pytest.approx(counts["a"] / counts["b"])
        assert found.evidence["minority_class"] == "b"
        assert found.evidence["minority_share"] == pytest.approx(counts["b"] / sum(counts.values()))
        assert found.evidence["minority_recall"] == pytest.approx(recall) and found.evidence["folds"] == 15


def test_minority_recall_is_the_mean_over_folds():
    folds = [fold(recall={"a": 1.0, "b": 0.0}), fold(recall={"a": 1.0, "b": 0.5})]  # mean 0.25
    found = only(diagnose(profile_of(frame({"a": 200, "b": 100})), folds), DiagnosisKind.CLASS_IMBALANCE)
    assert found.evidence["minority_recall"] == 0.25 and found.severity is Severity.MODERATE


def test_multiclass_imbalance_uses_the_smallest_class():
    counts = {"a": 500, "b": 300, "c": 50}  # ratio 10
    folds = [fold(recall={"a": 0.9, "b": 0.8, "c": 0.6})] * 5
    found = only(diagnose(profile_of(frame(counts), TaskType.MULTICLASS), folds, TaskType.MULTICLASS),
                 DiagnosisKind.CLASS_IMBALANCE)
    assert found.severity is Severity.HIGH and found.evidence["minority_class"] == "c"
    assert found.evidence["minority_recall"] == pytest.approx(0.6)


def test_regression_has_no_class_imbalance_rule():
    df = frame(regression=True)
    found = diagnose(profile_of(df, TaskType.REGRESSION), [fold(0.9, 0.9, metric="r2")] * 5, TaskType.REGRESSION, "r2")
    assert found == []


# --------------------------------------------------------------------------- high cardinality
@pytest.mark.parametrize(
    "n_codes, unseen, severity",
    [
        (3, 0.5, Severity.HIGH),
        (3, 0.25, Severity.MODERATE),
        (51, 0.0, Severity.MODERATE),  # more than 50 values
        (50, 0.2, None),  # exactly 50 values and few unseen
        (3, 0.2, None),
    ],
)
def test_high_cardinality_levels(n_codes, unseen, severity):
    folds = [fold(unseen={"code": unseen})] * 15
    found = only(diagnose(profile_of(frame(n_codes=n_codes)), folds), DiagnosisKind.HIGH_CARDINALITY)
    assert (found.severity if found else None) is severity
    if found:
        assert found.columns == ("code",)
        assert found.evidence == {"distinct_values": n_codes, "unseen_share": unseen, "folds": 15}


def test_high_cardinality_skips_excluded_and_numeric_columns():
    df = frame(n_codes=80)  # 'x' is numeric with 100 different values
    folds = [fold(unseen={"code": 0.9})] * 3
    assert only(diagnose(profile_of(df), folds), DiagnosisKind.HIGH_CARDINALITY).columns == ("code",)
    excluded = ExperimentConfig(target_column="target", excluded_columns=["code"])
    assert only(diagnose(profile_of(df), folds, config=excluded), DiagnosisKind.HIGH_CARDINALITY) is None


# --------------------------------------------------------------------------- whole output
def test_output_order_is_fixed_and_every_diagnosis_has_the_version():
    df = frame({"a": 900, "b": 100}, n_codes=80)
    df["b_code"] = df["code"]
    df["a_code"] = df["code"]
    folds = [fold(0.75, 0.5, recall={"a": 1.0, "b": 0.9}, unseen={"code": 0.9, "a_code": 0.9, "b_code": 0.9})] * 15
    first = diagnose(profile_of(df), folds)
    assert [(d.kind, d.columns) for d in first] == [
        (DiagnosisKind.OVERFITTING, ()),
        (DiagnosisKind.CLASS_IMBALANCE, ("target",)),
        (DiagnosisKind.HIGH_CARDINALITY, ("a_code",)),
        (DiagnosisKind.HIGH_CARDINALITY, ("b_code",)),
        (DiagnosisKind.HIGH_CARDINALITY, ("code",)),
    ]
    assert diagnose(profile_of(df), folds) == first  # the same inputs give the same Diagnoses
    assert {d.thresholds_version for d in first} == {"test-1"}
    record = first[0].to_dict()
    assert record["kind"] == "overfitting" and record["severity"] == "high" and record["message"]


def test_nothing_applies_gives_an_empty_list():
    assert diagnose(profile_of(frame()), [fold(0.8, 0.75, recall={"a": 0.8, "b": 0.8}, unseen={"code": 0.0})] * 15) == []


def test_a_failed_assessment_is_refused():
    failed = assessment([fold()])
    failed.failed = True
    with pytest.raises(ValueError, match="completed Assessment"):
        CRITIC.diagnose(profile=profile_of(frame()), config=CONFIG, assessment=failed, search=None,
                        task_type=TaskType.BINARY, primary_metric="f1")


@pytest.mark.parametrize(
    "task_type, primary, expected",
    [
        (TaskType.BINARY, "f1", "f1"), (TaskType.BINARY, "roc_auc", "roc_auc"), (TaskType.BINARY, "log_loss", "f1"),
        (TaskType.MULTICLASS, "accuracy", "accuracy"), (TaskType.MULTICLASS, "log_loss", "f1"),
        (TaskType.REGRESSION, "r2", "r2"), (TaskType.REGRESSION, "rmse", "r2"), (TaskType.REGRESSION, "mae", "r2"),
    ],
)
def test_diagnostic_metric(task_type, primary, expected):
    assert diagnostic_metric(task_type, primary) == expected


def test_error_metric_primary_uses_r2_for_the_gap():
    folds = [FoldScore(0, 0, {"rmse": 1.0, "r2": 0.875}, {"rmse": 2.0, "r2": 0.5}, None, {}, {}, 0.1)] * 3
    found = only(diagnose(profile_of(frame(regression=True), TaskType.REGRESSION), folds, TaskType.REGRESSION,
                          "rmse"), DiagnosisKind.OVERFITTING)
    assert found.evidence["metric"] == "r2" and found.evidence["gap"] == 0.375 and found.severity is Severity.HIGH


# --------------------------------------------------------------------------- thresholds file
def test_default_thresholds_file_has_the_design_values():
    thresholds = CriticThresholds.load()
    assert thresholds.version == "v0-start"
    assert thresholds.values == {
        "overfitting.moderate_gap": 0.10, "overfitting.high_gap": 0.20,
        "underfitting.max_gap": 0.03, "underfitting.moderate_skill": 0.50, "underfitting.high_skill": 0.25,
        "budget_limited.moderate_gain": 0.005, "budget_limited.high_gain": 0.02,
        "wasted_budget.moderate_share": 0.50, "wasted_budget.high_share": 0.75,
        "class_imbalance.moderate_ratio": 3.0, "class_imbalance.high_ratio": 9.0,
        "class_imbalance.moderate_recall": 0.5, "class_imbalance.high_recall": 0.3,
        "class_imbalance.min_ratio_for_recall": 1.5,
        "missing_values.moderate_share": 0.40, "missing_values.high_share": 0.70,
        "high_cardinality.moderate_distinct": 50.0, "high_cardinality.moderate_unseen": 0.30,
        "high_cardinality.high_unseen": 0.60,
        "constant_column.near_constant_share": 0.99,
        "redundant_features.moderate_correlation": 0.95, "redundant_features.high_correlation": 0.99,
        "unreliable_validation.moderate_std": 0.05, "unreliable_validation.high_std": 0.08,
        "weak_model_families.min_gap": 0.10, "weak_model_families.min_time_share": 0.05,
    }


def test_missing_thresholds_file_names_the_file(tmp_path):
    path = tmp_path / "nothing.toml"
    with pytest.raises(ThresholdsError, match="not found") as error:
        CriticThresholds.load(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize(
    "text, message",
    [
        ("version = \n", "cannot be read"),  # not valid TOML
        ('[overfitting]\nmoderate_gap = 0.1\n', "'version'"),
        ('version = "x"\n[overfitting]\nmoderate_gap = 0.1\n', "missing thresholds"),
        ('version = "x"\nnumber = 3\n', "must be a table"),
    ],
)
def test_invalid_thresholds_files_are_refused(tmp_path, text, message):
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ThresholdsError, match=message):
        CriticThresholds.load(path)


@pytest.mark.parametrize(
    "change, message",
    [
        (("overfitting", "high_gapp", 0.2), "unknown thresholds"),  # a typing mistake
        (("overfitting", "high_gap", "high"), "must be a number"),
        (("overfitting", "high_gap", True), "must be a number"),
        (("overfitting", "high_gap", 0.05), "must not be above"),  # high below moderate
        (("class_imbalance", "high_recall", 0.6), "must not be above"),  # for recall, high is the lower one
    ],
)
def test_invalid_threshold_values_are_refused(change, message):
    section, name, value = change
    data = {key: dict(table) if isinstance(table, dict) else table for key, table in THRESHOLDS.items()}
    data[section][name] = value
    with pytest.raises(ThresholdsError, match=message):
        CriticThresholds.from_mapping(data)


# =========================================================================== the full Critic (task 13.1)
TASKS = [(TaskType.BINARY, "f1"), (TaskType.MULTICLASS, "f1"), (TaskType.REGRESSION, "r2")]


def task_frame(task_type, **columns):
    """100 rows: a numeric column, the given extra columns and a target for the task type."""
    if task_type is TaskType.REGRESSION:
        df = frame(regression=True)
    elif task_type is TaskType.MULTICLASS:
        df = frame({"a": 34, "b": 33, "c": 33})
    else:
        df = frame()
    for name, values in columns.items():
        df[name] = values
    return df


def summary(gain=None, share=0.0, best=None, seconds=None):
    from app.services.search_summary import SearchSummary

    best = best or {"lgbm": 0.8}
    return SearchSummary(search_seconds=60.0, time_to_best=60.0 * (1 - share), share_after_best=share,
                         gain_last_quarter=gain, n_trials=50, best_per_family=best,
                         seconds_per_family=seconds or {family: 1.0 for family in best})


@pytest.mark.parametrize("task_type, metric", TASKS)
@pytest.mark.parametrize(
    "valid, gap, baseline, severity",
    [
        (0.2, 0.0, 0.0, Severity.MODERATE),  # skill 0.2: below 0.25
        (0.1, 0.0, 0.0, Severity.HIGH),  # skill 0.1: below 0.125
        (0.25, 0.0, 0.0, None),  # skill exactly 0.25 is not below it
        (0.2, 0.015625, 0.0, Severity.MODERATE),  # a small gap is allowed
        (0.2, 0.0625, 0.0, None),  # the training score is clearly higher: not underfitting
        (0.6, 0.0, 0.5, Severity.MODERATE),  # skill (0.6 - 0.5) / 0.5 = 0.2
        (0.625, 0.0, 0.5, None),  # skill exactly 0.25
    ],
)
def test_underfitting_levels(task_type, metric, valid, gap, baseline, severity):
    folds = [fold(valid + gap, valid, metric=metric, baseline=baseline)] * 15
    found = only(diagnose(profile_of(task_frame(task_type), task_type), folds, task_type, metric),
                 DiagnosisKind.UNDERFITTING)
    assert (found.severity if found else None) is severity
    if found:
        assert found.evidence["skill"] == pytest.approx((valid - baseline) / (1 - baseline))
        assert found.evidence["baseline_mean"] == baseline and found.evidence["metric"] == metric


def test_underfitting_with_roc_auc_uses_a_constant_predictor_as_the_baseline():
    folds = [fold(0.55, 0.55, metric="roc_auc", baseline=None)] * 5  # the baseline has no ROC-AUC
    found = only(diagnose(profile_of(frame()), folds, metric="roc_auc"), DiagnosisKind.UNDERFITTING)
    assert found.severity is Severity.HIGH and found.evidence["baseline_mean"] == 0.5
    assert found.evidence["skill"] == pytest.approx(0.1)


@pytest.mark.parametrize("task_type, metric", [(TaskType.BINARY, "f1"), (TaskType.MULTICLASS, "accuracy"),
                                               (TaskType.REGRESSION, "r2")])
@pytest.mark.parametrize("gain, severity", [(0.02, Severity.HIGH), (0.005, Severity.MODERATE), (0.004, None),
                                            (None, None)])
def test_budget_limited_levels(task_type, metric, gain, severity):
    found = only(diagnose(profile_of(task_frame(task_type), task_type), [fold(0.8, 0.8, metric=metric)] * 3,
                          task_type, metric, search=summary(gain=gain)), DiagnosisKind.BUDGET_LIMITED)
    assert (found.severity if found else None) is severity
    if found:
        assert found.evidence["gain_last_quarter"] == gain and found.evidence["metric"] == metric


def test_budget_rules_skip_unbounded_search_metrics():
    rmse = frame(regression=True)
    folds = [fold(0.8, 0.8, metric="r2")] * 3
    best = {"lgbm": 1.0, "sgd": 5.0}
    search = summary(gain=0.5, best=best, seconds={"lgbm": 10.0, "sgd": 10.0})
    found = diagnose(profile_of(rmse, TaskType.REGRESSION), folds, TaskType.REGRESSION, "rmse", search=search)
    assert only(found, DiagnosisKind.BUDGET_LIMITED) is None and only(found, DiagnosisKind.WEAK_MODEL_FAMILIES) is None
    log_loss = ExperimentConfig(target_column="target", search_metric="log_loss")
    found = diagnose(profile_of(frame()), [fold(0.8, 0.8)] * 3, config=log_loss, search=summary(gain=0.5))
    assert only(found, DiagnosisKind.BUDGET_LIMITED) is None


@pytest.mark.parametrize("share, severity", [(0.75, Severity.HIGH), (0.5, Severity.MODERATE), (0.49, None)])
def test_wasted_budget_levels(share, severity):
    found = only(diagnose(profile_of(frame()), [fold(0.8, 0.8)] * 3, search=summary(share=share)),
                 DiagnosisKind.WASTED_BUDGET)
    assert (found.severity if found else None) is severity
    if found:
        assert found.evidence["share_after_best"] == share and found.evidence["time_to_best"] == pytest.approx(
            60 * (1 - share))


@pytest.mark.parametrize("task_type, metric", TASKS)
@pytest.mark.parametrize("empty, severity", [(70, Severity.HIGH), (40, Severity.MODERATE), (39, None)])
def test_missing_values_levels(task_type, metric, empty, severity):
    values = np.where(np.arange(100) < empty, np.nan, 1.0 + np.arange(100))
    df = task_frame(task_type, age=values)
    found = only(diagnose(profile_of(df, task_type), [fold(0.8, 0.8, metric=metric)] * 3, task_type, metric),
                 DiagnosisKind.MISSING_VALUES)
    assert (found.severity if found else None) is severity
    if found:
        assert found.columns == ("age",) and found.evidence["missing_share"] == empty / 100
    excluded = ExperimentConfig(target_column="target", excluded_columns=["age"])
    assert only(diagnose(profile_of(df, task_type), [fold(0.8, 0.8, metric=metric)] * 3, task_type, metric,
                         config=excluded), DiagnosisKind.MISSING_VALUES) is None


def test_identifier_column_is_always_high():
    df = frame()
    df["row_id"] = np.arange(len(df))
    found = only(diagnose(profile_of(df), [fold(0.8, 0.8)] * 3), DiagnosisKind.IDENTIFIER_COLUMN)
    assert found.severity is Severity.HIGH and found.columns == ("row_id",)
    assert found.evidence == {"distinct_values": 100, "rows": 100}
    excluded = ExperimentConfig(target_column="target", excluded_columns=["row_id"])
    assert only(diagnose(profile_of(df), [fold(0.8, 0.8)] * 3, config=excluded), DiagnosisKind.IDENTIFIER_COLUMN) is None


@pytest.mark.parametrize("same, found_expected", [(100, True), (99, True), (98, False)])
def test_constant_and_near_constant_columns_are_low(same, found_expected):
    df = frame()
    df["flag"] = np.where(np.arange(100) < same, "yes", "no")
    found = only(diagnose(profile_of(df), [fold(0.8, 0.8)] * 3), DiagnosisKind.CONSTANT_COLUMN)
    assert (found is not None) is found_expected
    if found:
        assert found.severity is Severity.LOW and found.columns == ("flag",)
        assert found.evidence["main_value_share"] == same / 100
        assert found.evidence["distinct_values"] == (1 if same == 100 else 2)


@pytest.mark.parametrize("task_type, metric", TASKS)
def test_redundant_features_levels(task_type, metric):
    rng = np.random.default_rng(0)
    base = rng.normal(size=100)
    copies = {"same": base * 2 + 1, "opposite": -base, "close": base + 0.2 * rng.normal(size=100),
              "loose": base + 0.6 * rng.normal(size=100)}
    df = task_frame(task_type, base=base, **copies)
    found = [d for d in diagnose(profile_of(df, task_type), [fold(0.8, 0.8, metric=metric)] * 3, task_type, metric)
             if d.kind is DiagnosisKind.REDUNDANT_FEATURES]
    by_pair = {d.columns: d for d in found}
    for name in ("same", "opposite", "close", "loose"):
        r = float(np.corrcoef(base, copies[name])[0, 1])
        expected = Severity.HIGH if abs(r) >= 0.99 else Severity.MODERATE if abs(r) >= 0.95 else None
        found_pair = by_pair.get(("base", name))
        assert (found_pair.severity if found_pair else None) is expected, (name, r)
        if found_pair:
            assert found_pair.evidence["correlation"] == pytest.approx(r)
    assert by_pair[("base", "same")].severity is Severity.HIGH and ("base", "loose") not in by_pair
    assert by_pair[("base", "close")].severity is Severity.MODERATE  # r = 0.98 for this seed
    # A pair is diagnosed only while both columns are used.
    excluded = ExperimentConfig(target_column="target", excluded_columns=["same"])
    kept = diagnose(profile_of(df, task_type), [fold(0.8, 0.8, metric=metric)] * 3, task_type, metric, config=excluded)
    assert not any("same" in d.columns for d in kept if d.kind is DiagnosisKind.REDUNDANT_FEATURES)


@pytest.mark.parametrize("task_type, metric", TASKS)
@pytest.mark.parametrize("values, severity", [((0.4, 0.6), Severity.HIGH), ((0.5, 0.6), Severity.MODERATE),
                                              ((0.5, 0.55), None)])
def test_unreliable_validation_levels(task_type, metric, values, severity):
    folds = [fold(v, v, metric=metric) for v in values]
    found = only(diagnose(profile_of(task_frame(task_type), task_type), folds, task_type, metric),
                 DiagnosisKind.UNRELIABLE_VALIDATION)
    assert (found.severity if found else None) is severity
    if found:
        assert found.evidence["std"] == pytest.approx(float(np.std(values, ddof=1)))


@pytest.mark.parametrize("task_type, metric", [(TaskType.BINARY, "f1"), (TaskType.MULTICLASS, "accuracy"),
                                               (TaskType.REGRESSION, "r2")])
def test_weak_model_families(task_type, metric):
    best = {"xgboost": 0.80, "sgd": 0.60, "rf": 0.75, "lrl1": 0.65}
    seconds = {"xgboost": 50.0, "sgd": 5.0, "rf": 10.0, "lrl1": 1.0}  # sgd 7.6% of the time, lrl1 1.5%
    found = only(diagnose(profile_of(task_frame(task_type), task_type), [fold(0.8, 0.8, metric=metric)] * 3,
                          task_type, metric, search=summary(best=best, seconds=seconds)),
                 DiagnosisKind.WEAK_MODEL_FAMILIES)
    assert found.severity is Severity.LOW and found.columns == ()
    assert found.evidence["families"] == "sgd" and found.evidence["best_family"] == "xgboost"
    assert found.evidence["gap.sgd"] == pytest.approx(0.2) and found.evidence["time_share.sgd"] == pytest.approx(5 / 66)
    # Not behind enough, or too little time: no Diagnosis.
    close = {"xgboost": 0.80, "sgd": 0.71}
    assert only(diagnose(profile_of(frame()), [fold(0.8, 0.8)] * 3, search=summary(best=close)),
                DiagnosisKind.WEAK_MODEL_FAMILIES) is None


def test_data_steps_change_which_columns_are_diagnosed():
    from app.services.transforms import DataStepSpec

    df = frame(n_codes=80)
    df["age"] = np.where(np.arange(100) < 80, np.nan, 1.0)
    folds = [fold(0.8, 0.8, unseen={"code": 0.9})] * 3
    plain = diagnose(profile_of(df), folds)
    assert only(plain, DiagnosisKind.HIGH_CARDINALITY) and only(plain, DiagnosisKind.MISSING_VALUES)
    steps = ExperimentConfig(target_column="target", data_steps=[DataStepSpec("frequency_encode", ("code",)),
                                                                 DataStepSpec("drop_columns", ("age",))])
    treated = diagnose(profile_of(df), folds, config=steps)
    assert only(treated, DiagnosisKind.HIGH_CARDINALITY) is None  # already encoded
    assert only(treated, DiagnosisKind.MISSING_VALUES) is None  # dropped by a Data_Step


def test_every_kind_in_table_order():
    df = frame({"a": 900, "b": 100}, n_codes=80)
    n = len(df)
    df["age"] = np.where(np.arange(n) < 0.8 * n, np.nan, 1.0)
    df["row_id"] = np.arange(n)
    df["flag"] = "same"
    df["twin"] = df["x"] * 3
    folds = [fold(0.75 + 0.1 * (i % 2), 0.5 + 0.1 * (i % 2), recall={"a": 1.0, "b": 0.9}, unseen={"code": 0.9})
             for i in range(6)]
    search = summary(gain=0.03, share=0.8, best={"xgboost": 0.8, "sgd": 0.5}, seconds={"xgboost": 10.0, "sgd": 10.0})
    # Some kinds come more than once (age is also constant apart from its empty cells; x, row_id and twin
    # are three correlated pairs), but always together, in the order of the table.
    kinds = list(dict.fromkeys(d.kind for d in diagnose(profile_of(df), folds, search=search)))
    assert kinds == [
        DiagnosisKind.OVERFITTING, DiagnosisKind.BUDGET_LIMITED, DiagnosisKind.WASTED_BUDGET,
        DiagnosisKind.CLASS_IMBALANCE, DiagnosisKind.MISSING_VALUES, DiagnosisKind.HIGH_CARDINALITY,
        DiagnosisKind.IDENTIFIER_COLUMN, DiagnosisKind.CONSTANT_COLUMN, DiagnosisKind.REDUNDANT_FEATURES,
        DiagnosisKind.UNRELIABLE_VALIDATION, DiagnosisKind.WEAK_MODEL_FAMILIES,
    ]
    assert set(DiagnosisKind) - set(kinds) == {DiagnosisKind.UNDERFITTING}  # it cannot come with overfitting
