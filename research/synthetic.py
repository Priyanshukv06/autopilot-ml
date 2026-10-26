"""Synthetic datasets with planted problems, for the component studies (spec task 4).

Each dataset comes with a fresh sample from the same generator, so the true score of a model
can be measured on rows it never saw (Requirement 6.1).

How a dataset is made:

1. Base rows for the dataset and the fresh sample together, from one call to scikit-learn's
   ``make_classification`` (all 8 features informative, no label flips) or ``make_regression``
   (plus Gaussian noise, so the best possible R2 is 0.9). One call means one generator: the same
   class centres or coefficients for both parts.
2. The rows are split into the dataset and the fresh sample (stratified for classification).
3. The problems are planted in each part by the same process, so the fresh sample has the same
   columns and the same kind of damage. Counts are exact in each part: with ``missing_values``
   at 0.2, exactly 20% of the cells of each affected column are empty.

Each problem kind uses its own random stream, derived from the seed, so adding one problem does
not change the draws of another. Apart from class_imbalance, which changes the base rows, a
dataset with problems has the same base rows as the clean dataset of the same seed. Problems are
planted in the order of ``PROBLEM_KINDS``, whatever the order of ``problems``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from app.services.task_detector import TaskType

TARGET = "target"
N_FEATURES = 8  # informative base features x0 .. x7
BASE_COLUMNS = tuple(f"x{i}" for i in range(N_FEATURES))
N_CLASSES = {"binary": 2, "multiclass": 3}
REGRESSION_NOISE = 1 / 3  # noise sd as a share of the signal sd: the best possible R2 is 0.9
MISSING_COLUMNS = 2  # base features that get missing cells
DUPLICATED_COLUMNS = 2  # base features that get a noisy copy
IDENTIFIER_COLUMN, CONSTANT_COLUMN, HIGH_CARDINALITY_COLUMN = "record_id", "flag", "code"
MIN_ROWS, MIN_FRESH_ROWS = 20, 10

# Every kind and the meaning of its strength. Problems are planted in this order.
PROBLEM_KINDS: dict[str, str] = {
    # Label noise moves rows between classes, so with both problems the measured ratio is lower
    # (ratio 4 with 10% noise measured 2.85 on 1,000 rows).
    "class_imbalance": "ratio of the largest to the smallest class before label noise, above 1 (classification)",
    "label_noise": "share of rows whose label is changed, between 0 and 0.5",
    "duplicated_columns": f"correlation of each of {DUPLICATED_COLUMNS} noisy copies with its source, up to 1",
    "useless_features": "number of added random numeric columns, a whole number of 1 or more",
    "identifier_column": "share of distinct values in an integer ID column, up to 1 (1: every row unique)",
    "constant_column": "share of rows with the column's main value, above 0.5 (1: constant)",
    # With L = strength x rows categories drawn uniformly, a fold's validation row (training on 80%
    # of the rows) has an unseen category with probability (1 - 1/L)^(0.8 rows), about exp(-0.8 / s):
    # 0.20 at 0.5, 0.45 at 1, 0.67 at 2, 0.85 at 5.
    "high_cardinality": "categories per row of the dataset, above 0 and up to 10; unrelated to the target",
    "missing_values": f"share of empty cells in each of {MISSING_COLUMNS} base features, between 0 and 1",
}
_KIND_INDEX = {kind: index for index, kind in enumerate(PROBLEM_KINDS)}
_SHARED, _FRAME, _FRESH = 0, 1, 2  # random streams: choices shared by both parts, then one per part


@dataclass(frozen=True)
class PlantedProblem:
    """One problem planted in a synthetic dataset.

    ``columns``: the target for class_imbalance and label_noise; the added column(s) for
    useless_features, identifier_column, constant_column and high_cardinality; the affected base
    features for missing_values; each source column followed by its copy for duplicated_columns.
    """

    kind: str
    strength: float
    columns: tuple[str, ...]


@dataclass
class SyntheticDataset:
    name: str
    frame: pd.DataFrame
    target: str
    task_type: TaskType
    planted: tuple[PlantedProblem, ...]
    fresh: pd.DataFrame  # more rows from the same generator, never used for training

    def problem(self, kind: str) -> PlantedProblem | None:
        return next((p for p in self.planted if p.kind == kind), None)


def make_synthetic(
    task: Literal["binary", "multiclass", "regression"],
    n_rows: int,
    problems: Sequence[tuple[str, float]],
    seed: int,
    fresh_rows: int = 20_000,
) -> SyntheticDataset:
    """Make a dataset with the given problems planted at the given strengths (see PROBLEM_KINDS).

    The same arguments always give the same data. ``problems=()`` gives a clean dataset.
    """
    strengths = _check(task, n_rows, problems, seed, fresh_rows)
    X, y = _base_rows(task, n_rows + fresh_rows, seed, strengths.get("class_imbalance"))
    positions = np.arange(len(y))
    frame_positions, fresh_positions = train_test_split(
        positions, train_size=n_rows, test_size=fresh_rows, random_state=seed,
        stratify=y if task != "regression" else None,
    )
    parts = [_Part(X, y, frame_positions, _FRAME), _Part(X, y, fresh_positions, _FRESH)]
    planted = [
        _plant(kind, strengths[kind], task, X, parts, seed, n_rows) for kind in PROBLEM_KINDS if kind in strengths
    ]
    label = "-".join(f"{kind}{strengths[kind]:g}" for kind in PROBLEM_KINDS if kind in strengths) or "clean"
    return SyntheticDataset(
        name=f"synthetic-{task}-{label}-seed{seed}",
        frame=parts[0].to_frame(),
        target=TARGET,
        task_type=TaskType(task),
        planted=tuple(planted),
        fresh=parts[1].to_frame(),
    )


# --------------------------------------------------------------------------- internals
class _Part:
    """The columns of the dataset or of the fresh sample while problems are planted."""

    def __init__(self, X: np.ndarray, y: np.ndarray, rows: np.ndarray, stream: int) -> None:
        self.columns: dict[str, np.ndarray] = {name: X[rows, i].copy() for i, name in enumerate(BASE_COLUMNS)}
        self.y = y[rows].copy()
        self.stream = stream
        self.n = len(rows)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({**self.columns, TARGET: self.y})


def _rng(seed: int, stream: int, kind: str) -> np.random.Generator:
    return np.random.default_rng([seed, stream, _KIND_INDEX[kind]])


def _count(share: float, n: int) -> int:
    return int(round(share * n))


def _check(task: str, n_rows: int, problems: Sequence[tuple[str, float]], seed: int, fresh_rows: int) -> dict:
    if task not in ("binary", "multiclass", "regression"):
        raise ValueError(f"Unknown task '{task}'. Use binary, multiclass or regression.")
    if n_rows < MIN_ROWS or fresh_rows < MIN_FRESH_ROWS:
        raise ValueError(f"Use at least {MIN_ROWS} rows and {MIN_FRESH_ROWS} fresh rows.")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("The seed must be a whole number of 0 or more.")
    strengths: dict[str, float] = {}
    for kind, strength in problems:
        if kind not in PROBLEM_KINDS:
            raise ValueError(f"Unknown problem '{kind}'. Known: {', '.join(PROBLEM_KINDS)}.")
        if kind in strengths:
            raise ValueError(f"Problem '{kind}' is listed twice.")
        strengths[kind] = float(strength)

    for kind, s in strengths.items():
        valid = {
            "class_imbalance": s > 1 and task != "regression",
            "label_noise": 0 < s < 0.5,
            "duplicated_columns": 0 < s <= 1,
            "useless_features": s >= 1 and s.is_integer(),
            "identifier_column": 0 < s <= 1,
            "constant_column": 0.5 < s <= 1,
            "high_cardinality": 0 < s <= 10,
            "missing_values": 0 < s < 1,
        }[kind]
        if not valid:
            raise ValueError(f"Strength {s:g} is not valid for '{kind}' on a {task} task: {PROBLEM_KINDS[kind]}.")
    if "class_imbalance" in strengths:
        smallest = n_rows * min(_class_weights(N_CLASSES[task], strengths["class_imbalance"]))
        if smallest < 2:
            raise ValueError("Too few rows for this imbalance: the smallest class would have fewer than 2 rows.")
    return strengths


def _class_weights(n_classes: int, ratio: float) -> list[float]:
    """Class shares falling geometrically from ``ratio`` to 1, so largest / smallest = ratio."""
    raw = ratio ** (np.arange(n_classes - 1, -1, -1) / (n_classes - 1))
    return list(raw / raw.sum())


def _base_rows(task: str, n_total: int, seed: int, imbalance: float | None) -> tuple[np.ndarray, np.ndarray]:
    if task == "regression":
        X, y = make_regression(
            n_samples=n_total, n_features=N_FEATURES, n_informative=N_FEATURES, noise=0.0, random_state=seed
        )
        noise = np.random.default_rng([seed, _SHARED, len(PROBLEM_KINDS)]).standard_normal(n_total)
        return X, y + REGRESSION_NOISE * y.std() * noise
    n_classes = N_CLASSES[task]
    weights = _class_weights(n_classes, imbalance) if imbalance is not None else None
    X, y = make_classification(
        n_samples=n_total, n_features=N_FEATURES, n_informative=N_FEATURES, n_redundant=0, n_repeated=0,
        n_classes=n_classes, weights=weights, flip_y=0.0, random_state=seed,
    )
    return X, y


def _plant(
    kind: str, s: float, task: str, X: np.ndarray, parts: list[_Part], seed: int, n_rows: int
) -> PlantedProblem:
    shared = _rng(seed, _SHARED, kind)
    if kind == "class_imbalance":  # already in the base rows
        return PlantedProblem(kind, s, (TARGET,))

    if kind == "label_noise":
        for part in parts:
            rng = _rng(seed, part.stream, kind)
            rows = rng.choice(part.n, size=_count(s, part.n), replace=False)
            if task == "regression":  # each chosen row takes the target of another chosen row
                order = rng.permutation(rows)
                part.y[order] = part.y[np.roll(order, 1)]
            else:  # each chosen row gets a different class, chosen at random
                n_classes = N_CLASSES[task]
                part.y[rows] = (part.y[rows] + rng.integers(1, n_classes, size=len(rows))) % n_classes
        return PlantedProblem(kind, s, (TARGET,))

    if kind == "duplicated_columns":
        # copy = s * z + sqrt(1 - s^2) * noise on the standardised source z, so corr(copy, source) = s.
        sources = sorted(int(i) for i in shared.choice(N_FEATURES, size=DUPLICATED_COLUMNS, replace=False))
        names = _new_names(parts[0], DUPLICATED_COLUMNS)
        scale = [(X[:, i].mean(), X[:, i].std()) for i in sources]  # one scaling for both parts
        for part in parts:
            noise = _rng(seed, part.stream, kind).standard_normal((DUPLICATED_COLUMNS, part.n))
            for j, (source, name) in enumerate(zip(sources, names)):
                mean, sd = scale[j]
                z = (part.columns[BASE_COLUMNS[source]] - mean) / sd
                part.columns[name] = (s * z + np.sqrt(1 - s * s) * noise[j]) * sd + mean
        pairs = [column for source, name in zip(sources, names) for column in (BASE_COLUMNS[source], name)]
        return PlantedProblem(kind, s, tuple(pairs))

    if kind == "useless_features":
        names = _new_names(parts[0], int(s))
        for part in parts:
            values = _rng(seed, part.stream, kind).standard_normal((len(names), part.n))
            part.columns.update(zip(names, values))
        return PlantedProblem(kind, s, tuple(names))

    if kind == "identifier_column":
        # Each part gets its own IDs (new rows, new IDs); a share s of each part's rows are distinct.
        distinct = [max(1, _count(s, part.n)) for part in parts]
        pool = 100_000 + shared.choice(10 * sum(distinct), size=sum(distinct), replace=False)
        pools = [pool[: distinct[0]], pool[distinct[0]:]]
        for part, ids in zip(parts, pools):
            rng = _rng(seed, part.stream, kind)
            values = np.concatenate([ids, rng.choice(ids, size=part.n - len(ids))])
            part.columns[IDENTIFIER_COLUMN] = rng.permutation(values).astype(np.int64)
        return PlantedProblem(kind, s, (IDENTIFIER_COLUMN,))

    if kind == "constant_column":
        for part in parts:
            values = np.ones(part.n, dtype=np.int64)
            values[_rng(seed, part.stream, kind).choice(part.n, size=_count(1 - s, part.n), replace=False)] = 0
            part.columns[CONSTANT_COLUMN] = values
        return PlantedProblem(kind, s, (CONSTANT_COLUMN,))

    if kind == "high_cardinality":
        # Every row draws a category uniformly from the same set, in both parts.
        n_levels = max(2, _count(s, n_rows))
        width = len(str(n_levels - 1))
        levels = np.array([f"C{i:0{width}d}" for i in range(n_levels)], dtype=object)
        for part in parts:
            part.columns[HIGH_CARDINALITY_COLUMN] = levels[_rng(seed, part.stream, kind).integers(0, n_levels, part.n)]
        return PlantedProblem(kind, s, (HIGH_CARDINALITY_COLUMN,))

    if kind == "missing_values":
        names = sorted(str(c) for c in shared.choice(BASE_COLUMNS, size=MISSING_COLUMNS, replace=False))
        for part in parts:
            rng = _rng(seed, part.stream, kind)
            for name in names:
                values = part.columns[name].copy()
                values[rng.choice(part.n, size=_count(s, part.n), replace=False)] = np.nan
                part.columns[name] = values
        return PlantedProblem(kind, s, tuple(names))

    raise AssertionError(f"unhandled problem kind {kind}")  # _check rejects unknown kinds


def _new_names(part: _Part, count: int) -> list[str]:
    """The next free feature names x8, x9, ... (planted numeric columns look like base features)."""
    used = sum(1 for name in part.columns if name.startswith("x"))
    return [f"x{used + i}" for i in range(count)]
