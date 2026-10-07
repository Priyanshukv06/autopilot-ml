"""Dataset profiling, independent of any UI.

``DatasetProfiler.profile`` turns a DataFrame into a structured, JSON-serialisable
``DatasetProfile``. The profile is shown in the UI, logged as an MLflow artifact
and its ``meta_features()`` are the dataset descriptors that the planned
experiment-memory module will use to compare datasets.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_float_dtype, is_numeric_dtype

from app.services.task_detector import TaskType

_ID_NAME_PATTERN = re.compile(r"(^id$|[_\s-]id$|Id$|ID$)")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str  # "numerical" | "categorical" | "datetime"
    missing: int
    missing_ratio: float
    unique: int
    flags: list[str] = field(default_factory=list)
    sample_values: list[str] = field(default_factory=list)


@dataclass
class ClassBalance:
    majority_class: str
    majority_ratio: float
    minority_class: str
    minority_ratio: float
    imbalance_ratio: float  # majority count / minority count
    severity: str  # "none" | "mild" | "moderate" | "severe"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_class_balance(target: pd.Series) -> ClassBalance | None:
    """Summarise class balance of a classification target (None if < 2 classes)."""
    counts = target.dropna().value_counts()
    if len(counts) < 2:
        return None
    total = int(counts.sum())
    majority_label, majority_count = counts.index[0], int(counts.iloc[0])
    minority_label, minority_count = counts.index[-1], int(counts.iloc[-1])
    ratio = majority_count / max(minority_count, 1)
    if ratio >= 9:
        severity = "severe"
    elif ratio >= 3:
        severity = "moderate"
    elif ratio >= 1.5:
        severity = "mild"
    else:
        severity = "none"
    return ClassBalance(
        majority_class=str(majority_label),
        majority_ratio=majority_count / total,
        minority_class=str(minority_label),
        minority_ratio=minority_count / total,
        imbalance_ratio=float(ratio),
        severity=severity,
    )


@dataclass
class DatasetProfile:
    dataset_name: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    numerical_features: list[str]
    categorical_features: list[str]
    datetime_features: list[str]
    missing_values: dict[str, int]
    missing_ratio: dict[str, float]
    total_missing: int
    total_missing_ratio: float
    duplicate_rows: int
    unique_counts: dict[str, int]
    constant_columns: list[str]
    id_like_columns: list[str]
    high_cardinality_columns: list[str]
    high_missing_columns: list[str]
    memory_mb: float
    column_details: list[ColumnProfile]
    target_column: str | None = None
    class_distribution: dict[str, int] | None = None
    class_balance: ClassBalance | None = None
    target_summary: dict[str, float] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (used for MLflow artifacts)."""
        return asdict(self)

    def column_table(self) -> pd.DataFrame:
        """One row per column, convenient for display."""
        return pd.DataFrame(
            [
                {
                    "column": c.name,
                    "dtype": c.dtype,
                    "kind": c.kind,
                    "missing": c.missing,
                    "missing_%": round(100 * c.missing_ratio, 2),
                    "unique": c.unique,
                    "flags": ", ".join(c.flags),
                    "examples": ", ".join(c.sample_values),
                }
                for c in self.column_details
            ]
        )

    def suggested_exclusions(self) -> list[str]:
        """Columns that carry no generalisable signal (identifiers, constants)."""
        excluded = [c for c in self.id_like_columns + self.constant_columns if c != self.target_column]
        return list(dict.fromkeys(excluded))

    def meta_features(self) -> dict[str, float]:
        """Compact numeric dataset descriptors (input for the planned experiment memory)."""
        n_features = len(self.numerical_features) + len(self.categorical_features) + len(self.datetime_features)
        meta = {
            "n_rows": float(self.rows),
            "n_columns": float(self.columns),
            "n_numerical": float(len(self.numerical_features)),
            "n_categorical": float(len(self.categorical_features)),
            "numerical_ratio": len(self.numerical_features) / n_features if n_features else 0.0,
            "total_missing_ratio": self.total_missing_ratio,
            "max_missing_ratio": max(self.missing_ratio.values(), default=0.0),
            "duplicate_ratio": self.duplicate_rows / self.rows if self.rows else 0.0,
            "n_id_like": float(len(self.id_like_columns)),
            "n_high_cardinality": float(len(self.high_cardinality_columns)),
            "n_constant": float(len(self.constant_columns)),
        }
        if self.class_distribution is not None:
            meta["n_classes"] = float(len(self.class_distribution))
        if self.class_balance is not None:
            meta["imbalance_ratio"] = self.class_balance.imbalance_ratio
            meta["majority_ratio"] = self.class_balance.majority_ratio
        if self.target_summary is not None and "skew" in self.target_summary:
            meta["target_skew"] = self.target_summary["skew"]
        return meta


class DatasetProfiler:
    """Compute a structured profile of a tabular dataset."""

    def __init__(
        self,
        high_missing_threshold: float = 0.4,
        high_cardinality_threshold: int = 50,
        max_distribution_classes: int = 50,
    ) -> None:
        self.high_missing_threshold = high_missing_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.max_distribution_classes = max_distribution_classes

    def profile(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        dataset_name: str = "dataset",
        task_type: TaskType | None = None,
    ) -> DatasetProfile:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("profile() expects a pandas DataFrame.")
        if df.shape[1] == 0:
            raise ValueError("The dataset has no columns.")
        if target_column is not None and target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        n_rows = len(df)
        details: list[ColumnProfile] = []
        numerical, categorical, datetimes = [], [], []
        constant, id_like, high_card, high_missing = [], [], [], []
        warnings: list[str] = []

        for column in df.columns:
            series = df[column]
            col_profile = self._profile_column(series, n_rows)
            details.append(col_profile)

            if column != target_column:
                {"numerical": numerical, "categorical": categorical, "datetime": datetimes}[col_profile.kind].append(
                    column
                )
            if "constant" in col_profile.flags:
                constant.append(column)
                warnings.append(f"'{column}' is constant and carries no information.")
            if "id_like" in col_profile.flags and column != target_column:
                id_like.append(column)
                warnings.append(f"'{column}' looks like an identifier (unique per row); consider excluding it.")
            if "high_cardinality" in col_profile.flags:
                high_card.append(column)
                warnings.append(f"'{column}' is a high-cardinality categorical column ({col_profile.unique} values).")
            if "high_missing" in col_profile.flags:
                high_missing.append(column)
                warnings.append(f"'{column}' has {col_profile.missing_ratio:.0%} missing values.")

        duplicate_rows = self._count_duplicates(df)
        if duplicate_rows:
            warnings.append(f"{duplicate_rows} duplicate rows ({duplicate_rows / max(n_rows, 1):.1%}).")

        missing_values = {c.name: c.missing for c in details}
        total_missing = int(sum(missing_values.values()))
        total_cells = n_rows * df.shape[1]

        profile = DatasetProfile(
            dataset_name=dataset_name,
            rows=n_rows,
            columns=int(df.shape[1]),
            column_names=[str(c) for c in df.columns],
            dtypes={c.name: c.dtype for c in details},
            numerical_features=numerical,
            categorical_features=categorical,
            datetime_features=datetimes,
            missing_values=missing_values,
            missing_ratio={c.name: c.missing_ratio for c in details},
            total_missing=total_missing,
            total_missing_ratio=total_missing / total_cells if total_cells else 0.0,
            duplicate_rows=duplicate_rows,
            unique_counts={c.name: c.unique for c in details},
            constant_columns=constant,
            id_like_columns=id_like,
            high_cardinality_columns=high_card,
            high_missing_columns=high_missing,
            memory_mb=float(df.memory_usage(deep=True).sum() / 1024**2),
            column_details=details,
            target_column=target_column,
            warnings=warnings,
        )
        if target_column is not None:
            self._profile_target(profile, df[target_column], task_type)
        return profile

    # ----------------------------------------------------------------- helpers
    def _profile_column(self, series: pd.Series, n_rows: int) -> ColumnProfile:
        missing = int(series.isna().sum())
        missing_ratio = missing / n_rows if n_rows else 0.0
        non_null = series.dropna()
        unique = int(non_null.nunique())

        if is_datetime64_any_dtype(series):
            kind = "datetime"
        elif is_numeric_dtype(series) and not is_bool_dtype(series):
            kind = "numerical"
        else:
            kind = "categorical"

        flags: list[str] = []
        if unique <= 1:
            flags.append("constant")
        else:
            all_unique = unique == len(non_null) == n_rows
            name_says_id = bool(_ID_NAME_PATTERN.search(str(series.name)))
            if (all_unique and not is_float_dtype(series) and kind != "datetime") or (
                name_says_id and n_rows and unique / n_rows > 0.95
            ):
                flags.append("id_like")
            elif kind == "categorical" and unique > self.high_cardinality_threshold:
                flags.append("high_cardinality")
        if missing_ratio >= self.high_missing_threshold:
            flags.append("high_missing")

        samples = [str(v) for v in non_null.drop_duplicates().head(3).tolist()]
        return ColumnProfile(
            name=str(series.name),
            dtype=str(series.dtype),
            kind=kind,
            missing=missing,
            missing_ratio=missing_ratio,
            unique=unique,
            flags=flags,
            sample_values=samples,
        )

    @staticmethod
    def _count_duplicates(df: pd.DataFrame) -> int:
        try:
            return int(df.duplicated().sum())
        except TypeError:  # unhashable cell values
            return 0

    def _profile_target(self, profile: DatasetProfile, target: pd.Series, task_type: TaskType | None) -> None:
        n_missing = int(target.isna().sum())
        if n_missing:
            profile.warnings.append(f"Target '{target.name}' has {n_missing} missing values; those rows are dropped.")

        y = target.dropna()
        numeric = is_numeric_dtype(y) and not is_bool_dtype(y)
        if numeric and task_type in (None, TaskType.REGRESSION) and len(y) > 1:
            desc = y.astype(float).describe()
            profile.target_summary = {
                "mean": float(desc["mean"]),
                "std": float(desc["std"]),
                "min": float(desc["min"]),
                "25%": float(desc["25%"]),
                "50%": float(desc["50%"]),
                "75%": float(desc["75%"]),
                "max": float(desc["max"]),
                "skew": float(np.nan_to_num(y.astype(float).skew())),
            }

        is_classification = task_type.is_classification if task_type is not None else not numeric
        if task_type is None and numeric:
            is_classification = y.nunique() <= 20
        if is_classification and y.nunique() <= self.max_distribution_classes:
            counts = y.value_counts()
            profile.class_distribution = {str(k): int(v) for k, v in counts.items()}
            profile.class_balance = analyze_class_balance(y)
            balance = profile.class_balance
            if balance is not None and balance.severity in ("moderate", "severe"):
                profile.warnings.append(
                    f"Class imbalance ({balance.severity}): majority class '{balance.majority_class}' "
                    f"is {balance.majority_ratio:.0%} of rows."
                )
