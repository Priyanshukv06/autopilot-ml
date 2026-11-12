"""Fixed data splits for experiments and Sessions.

``make_outer_split`` makes the training and test parts. It picks exactly the same rows as the
Phase 1 runner, which called ``train_test_split`` on the data itself: the function chooses rows
from their positions and the class labels only, so giving it positions gives the same partition.

``make_session_folds`` makes the fixed folds of the training part that every Assessment of a
Session uses, so that two setups are always compared on the same rows.

With ``groups`` (optional task 31, Requirement 28), every group stays on one side: the test part is
made of whole groups (``test_size`` is then the share of groups), and in every Session_Fold each
group is either all training rows or all validation rows. Without groups nothing changes.

Positions refer to the labelled rows (rows with a target value) in their original order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    train_test_split,
)

from app.services.task_detector import TaskType


class SplitError(ValueError):
    """Raised when the data cannot be split as requested."""


def _hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(np.ascontiguousarray(part, dtype=np.int64).tobytes())
        else:
            digest.update(repr(part).encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()


@dataclass(frozen=True)
class OuterSplit:
    """The training part and the test part, as positions in the labelled rows."""

    train_positions: np.ndarray
    test_positions: np.ndarray
    seed: int
    test_size: float
    stratified: bool
    grouped: bool = False  # optional task 31: whole groups on each side

    @property
    def n_train(self) -> int:
        return len(self.train_positions)

    @property
    def n_test(self) -> int:
        return len(self.test_positions)

    def fingerprint(self) -> str:
        grouped = ("grouped",) if self.grouped else ()  # splits without groups keep their fingerprints
        return _hash("outer", self.seed, self.test_size, self.stratified, *grouped, self.train_positions,
                     self.test_positions)

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint(),
            "seed": self.seed,
            "test_size": self.test_size,
            "stratified": self.stratified,
            "n_train": self.n_train,
            "n_test": self.n_test,
            **({"grouped": True} if self.grouped else {}),
        }


def group_codes(groups: Any, n_rows: int) -> np.ndarray:
    """The groups of the rows as integer codes, in row order. Empty cells and a wrong length are refused."""
    values = pd.Series(np.asarray(groups, dtype=object))
    if len(values) != n_rows:
        raise SplitError(f"There are {len(values)} group values for {n_rows} rows.")
    empty = int(values.isna().sum())
    if empty:
        raise SplitError(f"The grouping column has {empty} empty cells among the labelled rows; fill them first.")
    return pd.factorize(values)[0].astype(np.int64)


def _grouped_outer_split(y: pd.Series, task_type: TaskType, test_size: float, seed: int,
                         groups: Any) -> tuple[OuterSplit, list[str]]:
    codes = group_codes(groups, len(y))
    n_groups = len(np.unique(codes))
    if n_groups < 2:
        raise SplitError("A grouped split needs at least 2 groups.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(np.zeros((len(y), 1)), groups=codes))
    warnings = [f"Grouped split: {len(np.unique(codes[test]))} of {n_groups} groups ({len(test)} of {len(y)} rows) "
                "form the test part, which is not stratified."]
    if task_type.is_classification:
        labels = np.asarray(y)
        missing = sorted(set(labels[train]) ^ set(labels[test]), key=str)
        if missing:
            warnings.append(f"Classes found on only one side of the grouped split: {', '.join(map(str, missing))}.")
    split = OuterSplit(train_positions=np.asarray(train, dtype=np.int64), test_positions=np.asarray(test, dtype=np.int64),
                       seed=seed, test_size=test_size, stratified=False, grouped=True)
    return split, warnings


def make_outer_split(y: pd.Series, task_type: TaskType, test_size: float, seed: int,
                     groups: Any = None) -> tuple[OuterSplit, list[str]]:
    """Split the labelled rows into a training part and a test part, as in Phase 1.

    Classification targets are stratified when every class has at least 2 rows. With ``groups`` (one
    value per labelled row), the test part is made of whole groups instead. Returns the split and any
    warnings.
    """
    if groups is not None:
        return _grouped_outer_split(y, task_type, test_size, seed, groups)
    warnings: list[str] = []
    stratify = None
    if task_type.is_classification:
        if y.value_counts().min() >= 2:
            stratify = y
        else:
            warnings.append("Stratified split disabled because some classes have fewer than 2 samples.")
    positions = np.arange(len(y))
    try:
        train_positions, test_positions = train_test_split(
            positions, test_size=test_size, random_state=seed, stratify=stratify
        )
    except ValueError as exc:
        raise SplitError(f"Could not split the data: {exc}") from exc
    split = OuterSplit(
        train_positions=np.asarray(train_positions, dtype=np.int64),
        test_positions=np.asarray(test_positions, dtype=np.int64),
        seed=seed,
        test_size=test_size,
        stratified=stratify is not None,
    )
    return split, warnings


@dataclass(frozen=True)
class SessionFolds:
    """Fixed (training rows, validation rows) pairs inside the training part, as positions in it."""

    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    n_splits: int
    n_repeats: int
    seed: int
    stratified: bool
    grouped: bool = False  # optional task 31: each group inside one fold

    def __len__(self) -> int:
        return len(self.folds)

    @property
    def test_train_ratio(self) -> float:
        """Validation rows per training row in one fold: 1 / (n_splits - 1). Used by the Judge. With groups the
        folds differ a little in size; the Judge keeps this nominal ratio."""
        return 1.0 / (self.n_splits - 1)

    def fingerprint(self) -> str:
        parts: list[object] = ["folds", self.n_splits, self.n_repeats, self.seed, self.stratified]
        if self.grouped:  # folds without groups keep their fingerprints
            parts.append("grouped")
        for train, validation in self.folds:
            parts.extend([train, validation])
        return _hash(*parts)

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint(),
            "n_splits": self.n_splits,
            "n_repeats": self.n_repeats,
            "seed": self.seed,
            "stratified": self.stratified,
            "n_folds": len(self.folds),
            **({"grouped": True} if self.grouped else {}),
        }


def _grouped_folds(labels: np.ndarray, codes: np.ndarray, n_splits: int, n_repeats: int, seed: int,
                   stratified: bool) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Each repeat shuffles the groups with its own seed (seed, seed + 1, ...); a group is never split."""
    positions = np.arange(len(labels))
    folds = []
    for repeat in range(n_repeats):
        splitter = (StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + repeat) if stratified
                    else GroupKFold(n_splits, shuffle=True, random_state=seed + repeat))
        folds.extend((np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64))
                     for train, validation in splitter.split(positions, labels, codes))
    return tuple(folds)


def make_session_folds(
    y_train: pd.Series, task_type: TaskType, n_splits: int = 5, n_repeats: int = 3, seed: int = 0,
    groups: Any = None,
) -> tuple[SessionFolds, list[str]]:
    """Repeated k-fold splits of the training part: stratified for classification, plain otherwise.

    Falls back to plain folds, with a warning, if a class has fewer rows than ``n_splits``. With ``groups``
    (one value per training row), each group stays inside one fold (StratifiedGroupKFold or GroupKFold).
    """
    if n_splits < 2:
        raise SplitError("At least 2 folds are needed.")
    if n_repeats < 1:
        raise SplitError("At least 1 repeat is needed.")
    if len(y_train) < n_splits:
        raise SplitError(f"{n_splits}-fold splits need at least {n_splits} training rows (found {len(y_train)}).")

    warnings: list[str] = []
    stratified = task_type.is_classification
    if stratified and y_train.value_counts().min() < n_splits:
        warnings.append(
            f"Some classes have fewer than {n_splits} training rows, so the session folds are not stratified."
        )
        stratified = False

    if groups is not None:
        codes = group_codes(groups, len(y_train))
        n_groups = len(np.unique(codes))
        if n_groups < n_splits:
            raise SplitError(f"{n_splits}-fold splits need at least {n_splits} groups in the training part "
                             f"(found {n_groups}).")
        folds = _grouped_folds(np.asarray(y_train), codes, n_splits, n_repeats, seed, stratified)
        warnings.append(f"Grouped folds: {n_groups} groups in the training part, each inside one fold.")
        return SessionFolds(folds=folds, n_splits=n_splits, n_repeats=n_repeats, seed=seed, stratified=stratified,
                            grouped=True), warnings

    splitter_class = RepeatedStratifiedKFold if stratified else RepeatedKFold
    splitter = splitter_class(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    positions = np.arange(len(y_train))
    labels = np.asarray(y_train)
    folds = tuple(
        (np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64))
        for train, validation in splitter.split(positions, labels)
    )
    return SessionFolds(folds=folds, n_splits=n_splits, n_repeats=n_repeats, seed=seed, stratified=stratified), warnings
