import json

import numpy as np
import pandas as pd
import pytest

from app.services.profiler import DatasetProfiler, analyze_class_balance
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
