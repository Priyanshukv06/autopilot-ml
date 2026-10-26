"""Data_Steps: changes to the data that are learned from training rows only (Requirement 3).

Each step is fitted on the rows it is given and then applied unchanged to other rows, so it never
reads rows it was not fitted on. Steps are described by JSON-safe ``DataStepSpec`` objects, so a
Setup with Data_Steps can be saved, compared and repeated. ``PipelineModel`` puts a fitted
``DataPipeline`` in front of the engine's model, so evaluation, permutation importance and saved
models all work on the original columns.

| Step | Learned from the training rows | Applied to other rows |
|---|---|---|
| drop_columns | nothing | the columns are removed |
| frequency_encode | share of training rows of each value | the column becomes that share; an unseen value becomes 0; empty cells stay empty |
| group_rare_categories | values with at least ``min_share`` of the training rows (1% by default) | other values become "__other__"; empty cells stay empty |
| missing_indicator | nothing | adds ``<column>__missing`` (1 if empty, else 0) for each listed numeric column |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

StepKind = Literal["drop_columns", "frequency_encode", "group_rare_categories", "missing_indicator"]
STEP_KINDS: tuple[str, ...] = ("drop_columns", "frequency_encode", "group_rare_categories", "missing_indicator")
OTHER = "__other__"
MISSING_SUFFIX = "__missing"
DEFAULT_MIN_SHARE = 0.01
_PARAMS = {"drop_columns": set(), "frequency_encode": set(), "group_rare_categories": {"min_share"},
           "missing_indicator": set()}


class DataStepError(ValueError):
    """A Data_Step cannot be fitted or applied to the given rows."""


@dataclass(frozen=True)
class DataStepSpec:
    kind: StepKind
    columns: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)  # for example {"min_share": 0.01}

    def __post_init__(self) -> None:
        if self.kind not in STEP_KINDS:
            raise DataStepError(f"Unknown Data_Step '{self.kind}'. Known: {', '.join(STEP_KINDS)}.")
        columns = tuple(self.columns)
        if not columns or not all(isinstance(c, str) and c for c in columns) or len(set(columns)) != len(columns):
            raise DataStepError(f"{self.kind} needs one or more distinct column names, got {self.columns!r}.")
        object.__setattr__(self, "columns", columns)
        unknown = set(self.params) - _PARAMS[self.kind]
        if unknown:
            raise DataStepError(f"{self.kind} does not take the parameters {sorted(unknown)}.")
        share = self.params.get("min_share", DEFAULT_MIN_SHARE)
        if self.kind == "group_rare_categories" and (isinstance(share, bool) or not 0 < share < 1):
            raise DataStepError("min_share must be a number between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "columns": list(self.columns), "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataStepSpec:
        return cls(kind=data["kind"], columns=tuple(data["columns"]), params=dict(data.get("params") or {}))

    def describe(self) -> str:
        names = ", ".join(f"'{c}'" for c in self.columns)
        if self.kind == "drop_columns":
            return f"drop {names}"
        if self.kind == "frequency_encode":
            return f"replace {names} by the share of training rows with each value"
        if self.kind == "group_rare_categories":
            share = self.params.get("min_share", DEFAULT_MIN_SHARE)
            return f"group the values of {names} seen in fewer than {share:.0%} of training rows into '{OTHER}'"
        return f"add a 0/1 column for empty cells of {names}"


def as_specs(steps: Iterable[Any]) -> list[DataStepSpec]:
    """Data_Steps as ``DataStepSpec`` objects (they may also be given as their ``to_dict()`` form)."""
    return [step if isinstance(step, DataStepSpec) else DataStepSpec.from_dict(step) for step in steps]


def _is_numeric(series: pd.Series) -> bool:
    return is_numeric_dtype(series) and not is_bool_dtype(series)


class DataPipeline:
    """Data_Steps applied in order; each step is fitted on the output of the steps before it."""

    def __init__(self, steps: Sequence[DataStepSpec | dict[str, Any]]) -> None:
        self.steps = as_specs(steps)
        self.state: list[dict[str, Any]] = []
        self.input_columns: list[str] | None = None

    @property
    def fitted(self) -> bool:
        return self.input_columns is not None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DataPipeline:
        """Learn every step from ``X`` (``y`` is not used: no step looks at the target)."""
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        self.state, self.input_columns = [], [str(c) for c in X.columns]
        frame = X
        for step in self.steps:
            state = self._fit_step(step, frame)
            self.state.append(state)
            frame = self._apply_step(step, state, frame)
        return frame

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise DataStepError("The Data_Steps must be fitted before they are applied.")
        frame = X
        for step, state in zip(self.steps, self.state):
            frame = self._apply_step(step, state, frame)
        return frame

    # ----------------------------------------------------------------- steps
    @staticmethod
    def _check_columns(step: DataStepSpec, frame: pd.DataFrame) -> None:
        missing = [c for c in step.columns if c not in frame.columns]
        if missing:
            raise DataStepError(f"{step.kind}: the columns {missing} are not in the data.")

    def _fit_step(self, step: DataStepSpec, frame: pd.DataFrame) -> dict[str, Any]:
        self._check_columns(step, frame)
        if step.kind == "frequency_encode":
            shares = {}
            for column in step.columns:
                values = frame[column].dropna()
                shares[column] = (values.value_counts() / len(frame)).to_dict() if len(frame) else {}
            return {"shares": shares}
        if step.kind == "group_rare_categories":
            min_share = step.params.get("min_share", DEFAULT_MIN_SHARE)
            kept = {}
            for column in step.columns:
                if _is_numeric(frame[column]):
                    raise DataStepError(f"group_rare_categories: '{column}' is numeric; it needs a text column.")
                shares = frame[column].dropna().value_counts() / max(len(frame), 1)
                kept[column] = sorted(str(v) for v in shares[shares >= min_share].index)
            return {"kept": kept}
        if step.kind == "missing_indicator":
            for column in step.columns:
                if not _is_numeric(frame[column]):
                    raise DataStepError(f"missing_indicator: '{column}' is not numeric.")
                if column + MISSING_SUFFIX in frame.columns:
                    raise DataStepError(f"missing_indicator: the column '{column}{MISSING_SUFFIX}' already exists.")
            return {}
        return {}  # drop_columns learns nothing

    def _apply_step(self, step: DataStepSpec, state: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(step, frame)
        if step.kind == "drop_columns":
            return frame.drop(columns=list(step.columns))
        frame = frame.copy()
        if step.kind == "frequency_encode":
            for column in step.columns:
                shares = state["shares"][column]
                values = frame[column]
                encoded = values.map(lambda v: shares.get(v, 0.0) if pd.notna(v) else np.nan)
                frame[column] = encoded.astype(float)
        elif step.kind == "group_rare_categories":
            for column in step.columns:
                kept = set(state["kept"][column])
                values = frame[column].astype(object)
                frame[column] = values.map(lambda v: v if pd.isna(v) or str(v) in kept else OTHER)
        elif step.kind == "missing_indicator":
            for column in step.columns:
                frame[column + MISSING_SUFFIX] = frame[column].isna().astype(np.int64)
        return frame


class PipelineModel:
    """A fitted DataPipeline followed by the engine's model, used on the original columns.

    Exposes ``predict``, ``predict_proba`` and ``classes_``, so the Evaluator, permutation
    importance and saved bundles work on raw rows exactly as for a model without Data_Steps.
    """

    def __init__(self, pipeline: DataPipeline, model: Any) -> None:
        if not pipeline.fitted:
            raise DataStepError("PipelineModel needs a fitted DataPipeline.")
        self.pipeline = pipeline
        self.model = model

    @property
    def classes_(self) -> Any:
        return getattr(self.model, "classes_", None)

    def fit(self, X: pd.DataFrame, y: Any = None) -> PipelineModel:
        """Present only because permutation importance asks for it; the model is already fitted."""
        raise NotImplementedError("PipelineModel wraps an already fitted model and is not refitted.")

    def predict(self, X: pd.DataFrame) -> Any:
        return self.model.predict(self.pipeline.transform(X))

    def predict_proba(self, X: pd.DataFrame) -> Any:
        return self.model.predict_proba(self.pipeline.transform(X))

    def __getattr__(self, name: str) -> Any:
        # Model details (best_config, feature_importances_, ...) come from the wrapped model. Guarded,
        # because unpickling looks up attributes before __init__ has run.
        if name in ("pipeline", "model") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.model, name)
