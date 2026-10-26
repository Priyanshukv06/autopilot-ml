import json

import numpy as np
import pandas as pd
import pytest

from app.services.profiler import DatasetProfile, DatasetProfiler, analyze_class_balance
from app.services.task_detector import TaskType


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "customer_id": np.arange(n),  # identifier-like
            "age": rng.integers(18, 80, n).astype(float),
            "income": rng.normal(50_000, 10_000, n),  # float, all unique, NOT an identifier
            "city": rng.choice(["A", "B", "C"], n),
            "code": [f"c{i % 60}" for i in range(n)],  # high-cardinality categorical
            "constant": ["x"] * n,
            "target": ["yes"] * 70 + ["no"] * 30,
        }
    )
    df.loc[:49, "age"] = np.nan  # 50% missing
    df.loc[:4, "city"] = np.nan
    return df


def test_basic_shape_and_feature_types(sample_df):
    profile = DatasetProfiler().profile(sample_df, target_column="target", dataset_name="demo")
    assert profile.dataset_name == "demo"
    assert profile.rows == 100
    assert profile.columns == 7
    assert profile.numerical_features == ["customer_id", "age", "income"]
    assert profile.categorical_features == ["city", "code", "constant"]
    assert "target" not in profile.numerical_features + profile.categorical_features


def test_missing_values_and_duplicates(sample_df):
    profile = DatasetProfiler().profile(sample_df)
    assert profile.missing_values["age"] == 50
    assert profile.missing_values["city"] == 5
    assert profile.missing_ratio["age"] == pytest.approx(0.5)
    assert profile.total_missing == 55
    assert profile.duplicate_rows == 0
    assert profile.high_missing_columns == ["age"]

    small = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "z"]})
    assert DatasetProfiler().profile(small).duplicate_rows == 1


def test_column_quality_flags(sample_df):
    profile = DatasetProfiler().profile(sample_df, target_column="target")
    assert profile.id_like_columns == ["customer_id"]
    assert profile.constant_columns == ["constant"]
    assert profile.high_cardinality_columns == ["code"]
    assert profile.unique_counts["city"] == 3
    assert profile.suggested_exclusions() == ["customer_id", "constant"]
    assert any("identifier" in w for w in profile.warnings)


def test_class_distribution_and_balance(sample_df):
    profile = DatasetProfiler().profile(sample_df, target_column="target", task_type=TaskType.BINARY)
    assert profile.class_distribution == {"yes": 70, "no": 30}
    balance = profile.class_balance
    assert balance.majority_class == "yes"
    assert balance.majority_ratio == pytest.approx(0.7)
    assert balance.imbalance_ratio == pytest.approx(70 / 30)
    assert balance.severity == "mild"


@pytest.mark.parametrize(
    ("majority", "minority", "severity"),
    [(50, 50, "none"), (60, 40, "mild"), (75, 25, "moderate"), (92, 8, "severe")],
)
def test_class_balance_severity(majority, minority, severity):
    target = pd.Series(["a"] * majority + ["b"] * minority)
    assert analyze_class_balance(target).severity == severity


def test_regression_target_summary():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=200), "price": rng.lognormal(size=200)})
    profile = DatasetProfiler().profile(df, target_column="price", task_type=TaskType.REGRESSION)
    assert profile.class_distribution is None
    assert profile.target_summary["min"] > 0
    assert profile.target_summary["skew"] > 0  # log-normal is right-skewed


def test_profile_is_json_serialisable_and_has_meta_features(sample_df):
    profile = DatasetProfiler().profile(sample_df, target_column="target")
    json.dumps(profile.to_dict())  # must not raise
    meta = profile.meta_features()
    assert meta["n_rows"] == 100
    assert meta["n_numerical"] == 3
    assert meta["n_classes"] == 2
    assert meta["imbalance_ratio"] == pytest.approx(70 / 30)
    table = profile.column_table()
    assert list(table["column"]) == list(sample_df.columns)


def test_invalid_inputs():
    profiler = DatasetProfiler()
    with pytest.raises(TypeError):
        profiler.profile([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        profiler.profile(pd.DataFrame({"a": [1, 2]}), target_column="missing")


def test_main_value_share_near_constant_columns_and_correlated_pairs():
    rng = np.random.default_rng(3)
    n = 200
    base = rng.normal(size=n)
    df = pd.DataFrame({
        "base": base,
        "copy": 2 * base + 0.01 * rng.normal(size=n),  # r close to 1
        "loose": base + 2 * rng.normal(size=n),  # r about 0.45: not reported
        "flag": ["on"] * 196 + ["off"] * 4,  # 98% one value: near-constant
        "half": [None] * 120 + ["x"] * 80,  # empty cells count as one value: main share 0.6
        "target": base + rng.normal(size=n),  # strongly related to base, but the target is left out
    })
    profile = DatasetProfiler().profile(df, target_column="target", task_type=TaskType.REGRESSION)
    assert profile.main_value_share["flag"] == pytest.approx(0.98)
    assert profile.main_value_share["half"] == pytest.approx(0.6)
    assert profile.near_constant_columns == ["flag"]
    assert [pair["columns"] for pair in profile.correlated_pairs] == [["base", "copy"]]
    assert profile.correlated_pairs[0]["correlation"] == pytest.approx(np.corrcoef(base, df["copy"])[0, 1])
    assert any("nearly constant" in w for w in profile.warnings) and any("highly correlated" in w for w in
                                                                          profile.warnings)


def test_profiles_round_trip_and_old_profiles_load():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "x"], "y": [0, 1, 0, 1]})
    profile = DatasetProfiler().profile(df, target_column="y", task_type=TaskType.BINARY)
    rebuilt = DatasetProfile.from_dict(json.loads(json.dumps(profile.to_dict())))
    assert rebuilt == profile
    old = profile.to_dict()  # a profile saved before Phase 2 has none of the new fields
    for key in ("main_value_share", "near_constant_columns", "correlated_pairs"):
        old.pop(key)
    loaded = DatasetProfile.from_dict(old)
    assert loaded.correlated_pairs == [] and loaded.main_value_share == {} and loaded.near_constant_columns == []
