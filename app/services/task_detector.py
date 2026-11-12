"""Problem-type detection for tabular targets.

Supported in the baseline: binary classification, multiclass classification and
regression. The user can always override the detected task; ``validate`` checks
that an override is actually feasible for the chosen target.

Target suggestion (optional task 33, Requirement 30.1): ``suggest_targets`` ranks the columns
that can be a target by a few transparent signals and gives the reasons; ``suggest_target`` takes
the first of them (before task 33: a target-like name, else the last column, which is still the
rule when no column can be a target). The user always confirms or changes the choice.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        return self in (TaskType.BINARY, TaskType.MULTICLASS)

    @property
    def display_name(self) -> str:
        return {
            TaskType.BINARY: "Binary Classification",
            TaskType.MULTICLASS: "Multiclass Classification",
            TaskType.REGRESSION: "Regression",
        }[self]


class TaskDetectionError(ValueError):
    """Raised when a column cannot be used as a target for the requested task."""


@dataclass
class TaskDetectionResult:
    task_type: TaskType
    confidence: str  # "high" | "medium" | "low"
    reason: str
    n_unique: int
    n_classes: int | None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["task_type"] = self.task_type.value
        return data


@dataclass
class TargetCandidate:
    """How suitable a column is as a prediction target (shown in the UI)."""

    column: str
    eligible: bool
    inferred_task: str | None
    n_unique: int
    missing_ratio: float
    note: str


# Generic (domain-agnostic) names that often denote the label column.
_TARGET_NAME_HINTS = ("target", "label", "class", "y", "outcome", "response")
# Points of each signal of suggest_targets (design choices of task 33, not tuned):
_NAME_POINTS, _NAME_WORD_POINTS, _LAST_POINTS, _FLAG_POINTS, _COMPLETE_POINTS = 3.0, 2.0, 1.0, 1.0, 0.5


@dataclass
class TargetSuggestion:
    """A column that may be the target, with the signals that point to it."""

    column: str
    score: float
    task: str  # the task it would give
    reasons: list[str]

    def text(self) -> str:
        return f"'{self.column}' ({'; '.join(self.reasons)})"


class TaskDetector:
    """Infer the ML task from the target column using simple, explainable rules."""

    def __init__(self, max_classes: int = 20, max_unique_ratio: float = 0.05) -> None:
        self.max_classes = max_classes
        self.max_unique_ratio = max_unique_ratio

    # ------------------------------------------------------------------ detect
    def detect(self, target: pd.Series) -> TaskDetectionResult:
        y = target.dropna()
        if y.empty:
            raise TaskDetectionError(f"Target '{target.name}' has no non-missing values.")
        n_unique = int(y.nunique())
        if n_unique < 2:
            raise TaskDetectionError(
                f"Target '{target.name}' has a single unique value; there is nothing to predict."
            )

        if is_bool_dtype(y) or not is_numeric_dtype(y):
            if n_unique == 2:
                return TaskDetectionResult(
                    TaskType.BINARY, "high", "Non-numeric target with exactly 2 classes.", n_unique, 2
                )
            confidence = "high" if n_unique <= self.max_classes else "low"
            reason = f"Non-numeric target with {n_unique} classes."
            if confidence == "low":
                reason += " That is a lot of classes; check that this column is really a class label."
            return TaskDetectionResult(TaskType.MULTICLASS, confidence, reason, n_unique, n_unique)

        if n_unique == 2:
            return TaskDetectionResult(
                TaskType.BINARY, "high", "Numeric target with exactly 2 distinct values.", n_unique, 2
            )

        values = y.to_numpy(dtype=float)
        integer_valued = bool(np.all(np.isclose(values, np.round(values))))
        unique_ratio = n_unique / len(y)

        if integer_valued and n_unique <= self.max_classes:
            confidence = "medium" if unique_ratio <= self.max_unique_ratio else "low"
            return TaskDetectionResult(
                TaskType.MULTICLASS,
                confidence,
                f"Integer target with only {n_unique} distinct values (treated as class codes). "
                "Override to regression if it is an ordinal or count quantity.",
                n_unique,
                n_unique,
            )
        if integer_valued:
            return TaskDetectionResult(
                TaskType.REGRESSION,
                "medium",
                f"Integer-valued target with {n_unique} distinct values (count-like quantity).",
                n_unique,
                None,
            )
        return TaskDetectionResult(
            TaskType.REGRESSION, "high", f"Continuous numeric target ({n_unique} distinct values).", n_unique, None
        )

    # ---------------------------------------------------------------- validate
    def validate(self, target: pd.Series, task_type: TaskType) -> list[str]:
        """Check that ``task_type`` is feasible for ``target``.

        Returns a list of non-blocking warnings; raises ``TaskDetectionError`` when
        the combination is impossible.
        """
        warnings: list[str] = []
        n_missing = int(target.isna().sum())
        y = target.dropna()
        if y.empty:
            raise TaskDetectionError(f"Target '{target.name}' has no non-missing values.")
        if n_missing:
            warnings.append(f"{n_missing} rows with a missing target value will be dropped.")
        n_unique = int(y.nunique())
        if n_unique < 2:
            raise TaskDetectionError(f"Target '{target.name}' has a single unique value.")

        if task_type is TaskType.REGRESSION:
            if is_bool_dtype(y) or not is_numeric_dtype(y):
                raise TaskDetectionError("Regression requires a numeric target column.")
            if n_unique <= 2:
                warnings.append("The target has only 2 distinct values; classification is usually more appropriate.")
        elif task_type is TaskType.BINARY:
            if n_unique != 2:
                raise TaskDetectionError(
                    f"Binary classification needs exactly 2 classes, but the target has {n_unique}."
                )
        elif task_type is TaskType.MULTICLASS:
            if n_unique < 3:
                raise TaskDetectionError(
                    "Multiclass classification needs at least 3 classes; use binary classification instead."
                )
            if n_unique > self.max_classes:
                warnings.append(
                    f"The target has {n_unique} classes; classification with this many classes may be "
                    "slow and unreliable. Is this really a categorical target?"
                )

        if task_type.is_classification:
            counts = y.value_counts()
            rare = counts[counts < 2]
            if not rare.empty:
                warnings.append(
                    f"{len(rare)} class(es) have fewer than 2 samples; the train/test split cannot be stratified."
                )
        return warnings

    # ------------------------------------------------------ target candidates
    def target_candidates(self, df: pd.DataFrame) -> list[TargetCandidate]:
        """Describe how suitable every column is as a target (for the UI)."""
        n_rows = len(df)
        candidates: list[TargetCandidate] = []
        for column in df.columns:
            series = df[column]
            non_null = series.dropna()
            n_unique = int(non_null.nunique())
            missing_ratio = float(series.isna().mean()) if n_rows else 0.0
            numeric = is_numeric_dtype(series) and not is_bool_dtype(series)

            if n_unique < 2:
                candidates.append(TargetCandidate(column, False, None, n_unique, missing_ratio, "Constant column"))
                continue
            if n_unique == len(non_null) == n_rows and not pd.api.types.is_float_dtype(series):
                candidates.append(
                    TargetCandidate(column, False, None, n_unique, missing_ratio, "Identifier-like (all values unique)")
                )
                continue
            if not numeric and n_unique > self.max_classes * 5:
                candidates.append(
                    TargetCandidate(column, False, None, n_unique, missing_ratio, "High-cardinality text column")
                )
                continue

            detection = self.detect(series)
            note = detection.reason
            if missing_ratio > 0:
                note += f" {missing_ratio:.0%} of target values are missing."
            candidates.append(
                TargetCandidate(column, True, detection.task_type.value, n_unique, missing_ratio, note)
            )
        return candidates

    def suggest_targets(self, df: pd.DataFrame, limit: int = 3) -> list[TargetSuggestion]:
        """The columns most likely to be the target, best first, each with its reasons.

        Only columns that can be a target count (``target_candidates``). Signals: a generic target name
        (target, label, class, y, outcome, response) 3 points, or such a word inside the name 2; the last
        column 1; a 0/1 or true/false column 1; no empty cells 0.5. Ties go to the later column.
        """
        eligible = [c for c in self.target_candidates(df) if c.eligible]
        position = {column: i for i, column in enumerate(df.columns)}
        suggestions = []
        for candidate in eligible:
            column, reasons, score = candidate.column, [], 0.0
            name = str(column).strip().lower()
            words = set(re.split(r"[^0-9a-z]+", re.sub(r"([a-z])([A-Z])", r"\1_\2", str(column)).lower()))
            if name in _TARGET_NAME_HINTS:
                score += _NAME_POINTS
                reasons.append("named like a target")
            elif words & set(_TARGET_NAME_HINTS):
                score += _NAME_WORD_POINTS
                reasons.append(f"its name contains '{sorted(words & set(_TARGET_NAME_HINTS))[0]}'")
            if position[column] == len(df.columns) - 1:
                score += _LAST_POINTS
                reasons.append("the last column")
            values = set(df[column].dropna().unique().tolist())
            if values in ({0, 1}, {True, False}):
                score += _FLAG_POINTS
                reasons.append("a 0/1 column")
            if candidate.missing_ratio == 0:
                score += _COMPLETE_POINTS
                reasons.append("no empty cells")
            reasons.append(f"gives {candidate.inferred_task} with {candidate.n_unique} values")
            suggestions.append(TargetSuggestion(str(column), score, str(candidate.inferred_task), reasons))
        suggestions.sort(key=lambda s: (-s.score, -position[s.column]))
        return suggestions[:limit]

    def suggest_target(self, df: pd.DataFrame) -> str | None:
        """The default target selection: the best of ``suggest_targets``; if no column can be a target, the
        first column with a generic target name, otherwise the last column. The user always confirms or
        changes this choice."""
        if len(df.columns) == 0:
            return None
        suggestions = self.suggest_targets(df, limit=1)
        if suggestions:
            return suggestions[0].column
        for column in df.columns:
            if str(column).strip().lower() in _TARGET_NAME_HINTS:
                return column
        return df.columns[-1]
