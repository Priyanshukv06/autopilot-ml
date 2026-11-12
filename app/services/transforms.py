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
| target_encode (optional task 30) | for each value, the smoothed mean target (the share of each class) of the training rows with it | the column becomes that encoding; an unseen value gets the training mean; empty cells stay empty |
| token_features (optional task 30) | the frequent words of the column in the training rows | the column is replaced by one 0/1 column ``<column>__tok_<word>`` per word |

target_encode is the only step that reads the target. On the rows it is fitted on, ``fit_transform``
returns cross-fitted encodings: the rows are split into 5 parts and each part is encoded with the
encodings learned from the other 4, so no row's encoding uses its own label (Requirement 27.1).
``transform`` applies the encodings learned from all fitted rows to other rows. So for a pipeline
with target_encode, ``fit(X, y).transform(X)`` is not ``fit_transform(X, y)``: the runner and the
Assessor train on ``fit_transform`` and score other rows with ``transform``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.model_selection import KFold

StepKind = Literal["drop_columns", "frequency_encode", "group_rare_categories", "missing_indicator",
                   "target_encode", "token_features"]
STEP_KINDS: tuple[str, ...] = ("drop_columns", "frequency_encode", "group_rare_categories", "missing_indicator",
                               "target_encode", "token_features")
# Steps that replace a text column by numbers; a column gets at most one of them.
ENCODING_KINDS: tuple[str, ...] = ("frequency_encode", "group_rare_categories", "target_encode", "token_features")
OTHER = "__other__"
MISSING_SUFFIX = "__missing"
TOKEN_INFIX = "__tok_"
TARGET_INFIX = "__te_"
DEFAULT_MIN_SHARE = 0.01
# Design choices of optional task 30, set before any run (not tuned):
DEFAULT_SMOOTHING = 10.0  # target_encode: weight, in rows, of the overall mean in each value's encoding
TARGET_ENCODE_FOLDS = 5  # target_encode: parts of the cross-fitting
TARGET_ENCODE_SEED = 0
DEFAULT_TOKEN_MIN_SHARE = 0.02  # token_features: a word must be in at least 2% of the rows ...
MAX_TOKENS = 20  # ... and at most the 20 most frequent words of a column are kept
_TOKEN = re.compile(r"[^\W\d_]+|\d+")  # a run of letters or a run of digits
_PARAMS = {"drop_columns": set(), "frequency_encode": set(), "group_rare_categories": {"min_share"},
           "missing_indicator": set(), "target_encode": {"smoothing"}, "token_features": {"min_share"}}
_TASKS = ("binary", "multiclass", "regression")


def tokenize(value: Any) -> list[str]:
    """The words of a cell: runs of letters and runs of digits of its lower-case text ("Braund, Mr. Owen"
    gives braund, mr, owen; "C85" gives c, 85)."""
    return _TOKEN.findall(str(value).lower())


def frequent_tokens(values: pd.Series, min_share: float = DEFAULT_TOKEN_MIN_SHARE,
                    max_tokens: int = MAX_TOKENS) -> list[str]:
    """Words found in at least ``min_share`` of the rows (empty cells count in the total), in at least two
    different values and not in every row; the ``max_tokens`` most frequent, ties by the word."""
    n = len(values)
    if n == 0:
        return []
    rows: dict[str, int] = defaultdict(int)
    distinct: dict[str, int] = defaultdict(int)
    for value, count in values.dropna().astype(str).value_counts().items():
        for token in set(tokenize(value)):
            rows[token] += int(count)
            distinct[token] += 1
    kept = [(token, count / n) for token, count in rows.items() if distinct[token] >= 2 and min_share <= count / n < 1]
    kept.sort(key=lambda pair: (-pair[1], pair[0]))
    return [token for token, _ in kept[:max_tokens]]


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
        if self.kind in ("group_rare_categories", "token_features") and (
                isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 < share < 1):
            raise DataStepError("min_share must be a number between 0 and 1.")
        smoothing = self.params.get("smoothing", DEFAULT_SMOOTHING)
        if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)) or not 0 <= smoothing < np.inf:
            raise DataStepError("smoothing must be a number of 0 or more.")

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
        if self.kind == "target_encode":
            return f"replace {names} by the mean target of each value in the training rows (cross-fitted)"
        if self.kind == "token_features":
            share = self.params.get("min_share", DEFAULT_TOKEN_MIN_SHARE)
            return f"replace {names} by 0/1 columns for the words found in at least {share:.0%} of training rows"
        return f"add a 0/1 column for empty cells of {names}"


def as_specs(steps: Iterable[Any]) -> list[DataStepSpec]:
    """Data_Steps as ``DataStepSpec`` objects (they may also be given as their ``to_dict()`` form)."""
    return [step if isinstance(step, DataStepSpec) else DataStepSpec.from_dict(step) for step in steps]


def _is_numeric(series: pd.Series) -> bool:
    return is_numeric_dtype(series) and not is_bool_dtype(series)


class DataPipeline:
    """Data_Steps applied in order; each step is fitted on the output of the steps before it.

    ``task_type`` ("binary", "multiclass" or "regression", or a ``TaskType``) is needed only by target_encode.
    """

    def __init__(self, steps: Sequence[DataStepSpec | dict[str, Any]], task_type: Any = None) -> None:
        self.steps = as_specs(steps)
        self.task_type = None if task_type is None else str(getattr(task_type, "value", task_type))
        self.state: list[dict[str, Any]] = []
        self.input_columns: list[str] | None = None

    @property
    def fitted(self) -> bool:
        return self.input_columns is not None

    @property
    def learns_columns(self) -> bool:
        """True if some output columns are learned from the rows (the words of token_features, the classes of
        a multiclass target_encode), so a fit on no rows cannot show them."""
        return any(step.kind == "token_features" or (step.kind == "target_encode" and self.task_type == "multiclass")
                   for step in self.steps)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DataPipeline:
        """Learn every step from ``X`` (and ``y``, which only target_encode reads)."""
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Learn every step from ``X`` and return the rows as the model is trained on them: for
        target_encode, the cross-fitted encodings (see the module notes)."""
        if y is not None and len(y) != len(X):
            raise DataStepError(f"The target has {len(y)} values for {len(X)} rows.")
        self.state, self.input_columns = [], [str(c) for c in X.columns]
        frame = X
        for step in self.steps:
            if step.kind == "target_encode":
                state, frame = self._fit_target_encode(step, frame, y)
            else:
                state = self._fit_step(step, frame)
                frame = self._apply_step(step, state, frame)
            self.state.append(state)
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
        if step.kind == "token_features":
            min_share = step.params.get("min_share", DEFAULT_TOKEN_MIN_SHARE)
            tokens = {}
            for column in step.columns:
                if _is_numeric(frame[column]):
                    raise DataStepError(f"token_features: '{column}' is numeric; it needs a text column.")
                tokens[column] = frequent_tokens(frame[column], min_share)
                clash = [f"{column}{TOKEN_INFIX}{t}" for t in tokens[column]
                         if f"{column}{TOKEN_INFIX}{t}" in frame.columns]
                if clash:
                    raise DataStepError(f"token_features: the columns {clash} already exist.")
            return {"tokens": tokens}
        return {}  # drop_columns learns nothing

    def _fit_target_encode(self, step: DataStepSpec, frame: pd.DataFrame,
                           y: pd.Series | None) -> tuple[dict[str, Any], pd.DataFrame]:
        """Learn the encodings from all rows (for other rows) and return the cross-fitted encodings of these rows."""
        self._check_columns(step, frame)
        if y is None:
            raise DataStepError("target_encode needs the target values of the rows it is fitted on.")
        if self.task_type not in _TASKS:
            raise DataStepError(f"target_encode needs the task type ({', '.join(_TASKS)}), got {self.task_type!r}.")
        smoothing = float(step.params.get("smoothing", DEFAULT_SMOOTHING))
        targets, labels = _target_matrix(np.asarray(y, dtype=float if self.task_type == "regression" else object),
                                         self.task_type)
        n = len(frame)
        parts = list(KFold(min(TARGET_ENCODE_FOLDS, n), shuffle=True, random_state=TARGET_ENCODE_SEED)
                     .split(np.zeros((n, 1)))) if n >= 2 else []
        state: dict[str, Any] = {"tables": {}, "priors": {}, "names": {}}
        out = frame.copy()
        for column in step.columns:
            if _is_numeric(frame[column]):
                raise DataStepError(f"target_encode: '{column}' is numeric; it needs a text column.")
            names = _target_names(column, labels, self.task_type)
            clash = [name for name in names if name != column and name in frame.columns]
            if clash:
                raise DataStepError(f"target_encode: the columns {clash} already exist.")
            values = frame[column].to_numpy(dtype=object)
            crossed = np.full((n, targets.shape[1]), np.nan)  # one row alone has no other rows to learn from
            for learn, encoded in parts:  # each part is encoded from the other parts only
                crossed[encoded] = _encode(values[encoded], *_encodings(values[learn], targets[learn], smoothing))
            state["tables"][column], state["priors"][column] = _encodings(values, targets, smoothing)
            state["names"][column] = names
            out = _put(out, column, names, crossed)
        return state, out

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
        elif step.kind == "target_encode":
            for column in step.columns:
                encoded = _encode(frame[column].to_numpy(dtype=object), state["tables"][column],
                                  state["priors"][column])
                frame = _put(frame, column, state["names"][column], encoded)
        elif step.kind == "token_features":
            for column in step.columns:
                words = [set(tokenize(v)) if pd.notna(v) else set() for v in frame[column].to_numpy(dtype=object)]
                new = pd.DataFrame({f"{column}{TOKEN_INFIX}{t}": np.array([t in w for w in words], dtype=np.int64)
                                    for t in state["tokens"][column]}, index=frame.index)
                frame = pd.concat([frame.drop(columns=[column]), new], axis=1)
        return frame


# ----------------------------------------------------------------- target_encode helpers
def _target_matrix(y: np.ndarray, task: str) -> tuple[np.ndarray, list[Any]]:
    """The targets as numbers, one column per encoded quantity: the value (regression), 1 for the positive
    class, the last sorted label as in FLAML (binary), or 1 for each class (multiclass)."""
    if task == "regression":
        return y.reshape(-1, 1), []
    labels = list(np.unique(y)) if len(y) else []
    if task == "binary":
        positive = labels[-1:]
        return (np.asarray(y == positive[0], dtype=float).reshape(-1, 1) if positive
                else np.zeros((len(y), 1))), positive
    return (np.column_stack([np.asarray(y == label, dtype=float) for label in labels]) if labels
            else np.zeros((len(y), 0))), labels


def _target_names(column: str, labels: list[Any], task: str) -> list[str]:
    """The output columns: the column itself, or for multiclass one column per class, named after the class
    (letters and digits only, because some learners refuse other characters) or, if those clash, its position."""
    if task != "multiclass":
        return [column]
    names = [re.sub(r"[^0-9A-Za-z]+", "_", str(label)).strip("_") for label in labels]
    if len(set(names)) != len(names) or not all(names):
        names = [str(i) for i in range(len(labels))]
    return [f"{column}{TARGET_INFIX}{name}" for name in names]


def _encodings(values: np.ndarray, targets: np.ndarray, smoothing: float) -> tuple[dict[Any, np.ndarray], np.ndarray]:
    """For each value of these rows, (sum of its targets + smoothing x mean) / (its rows + smoothing); and the
    mean of all these rows, for values not seen."""
    prior = targets.mean(axis=0) if len(targets) else np.full(targets.shape[1], np.nan)
    present = ~pd.isna(values)
    if not present.any():
        return {}, prior
    groups = pd.DataFrame(targets[present]).groupby(values[present])
    sums = groups.sum()
    counts = groups.size().reindex(sums.index).to_numpy()
    encoded = (sums.to_numpy() + smoothing * prior) / (counts[:, None] + smoothing)
    return {key: encoded[i] for i, key in enumerate(sums.index)}, prior


def _encode(values: np.ndarray, table: dict[Any, np.ndarray], prior: np.ndarray) -> np.ndarray:
    out = np.tile(np.asarray(prior, dtype=float), (len(values), 1))
    if table:
        positions = pd.Index(list(table), dtype=object).get_indexer(values)
        known = positions >= 0
        out[known] = np.vstack(list(table.values()))[positions[known]]
    out[pd.isna(values)] = np.nan  # empty cells stay empty, as in frequency_encode
    return out


def _put(frame: pd.DataFrame, column: str, names: list[str], matrix: np.ndarray) -> pd.DataFrame:
    """Replace ``column`` by the columns ``names`` (in place if it keeps its name, else at the end)."""
    if names == [column]:
        frame = frame.copy()
        frame[column] = matrix[:, 0].astype(float)
        return frame
    new = pd.DataFrame(matrix, columns=names, index=frame.index, dtype=float)
    return pd.concat([frame.drop(columns=[column]), new], axis=1)


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
