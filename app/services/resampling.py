"""Resamplers for imbalanced classification (optional Requirement 26).

FLAML's ``resampler`` option calls ``fit_resample`` on the training rows of each search fold and on the
rows of the final model, never on validation rows; the Assessor's refits (``FLAMLEngine.fit_config``)
apply it to the training rows of each Session_Fold only. FLAML first prepares the data: text columns
become pandas ``category`` columns and empty numeric cells are filled (checked on Titanic, 1 November
2026), so plain SMOTE cannot be used as it is. ``FoldSafeSmote`` picks the variant of imbalanced-learn
that fits the columns it is given.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

RESAMPLERS = ("smote",)
DEFAULT_NEIGHBOURS = 5


class FoldSafeSmote(BaseEstimator):
    """SMOTE that works on the rows FLAML hands a resampler.

    - Numeric columns only: ``SMOTE``. Numeric and categorical columns: ``SMOTENC``. Categorical only:
      ``SMOTEN``. A column counts as categorical if it is not numeric (or is boolean).
    - Every class smaller than the largest is oversampled to the largest class (imbalanced-learn's
      ``"auto"``). The number of neighbours is lowered to the smallest of those classes minus 1; a class
      with fewer than 2 rows cannot be oversampled, so the rows are then returned unchanged.
    - Empty numeric cells (FLAML fills them, other callers may not) are filled with the column median of
      the given rows for finding neighbours only; the original rows come back unchanged, first and in
      their order, followed by the new rows.
    """

    def __init__(self, k_neighbors: int = DEFAULT_NEIGHBOURS, random_state: int | None = None) -> None:
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X: Any, y: Any) -> tuple[Any, Any]:
        from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC

        frame = pd.DataFrame(X).copy()
        labels = pd.Series(np.asarray(y), index=frame.index)
        counts = labels.value_counts()
        if len(counts) < 2:
            return X, y
        smaller = counts[counts < counts.max()]
        if smaller.empty:  # already balanced
            return X, y
        k = min(int(self.k_neighbors), int(smaller.min()) - 1)
        if k < 1:
            return X, y

        categorical = [c for c in frame.columns
                       if not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])]
        numeric = [c for c in frame.columns if c not in categorical]
        filled = frame.copy()
        for column in numeric:
            if filled[column].isna().any():
                median = filled[column].median()
                filled[column] = filled[column].fillna(0.0 if pd.isna(median) else median)
        for column in categorical:
            filled[column] = filled[column].astype(object).where(filled[column].notna(), "__missing__").astype(str)

        if not categorical:
            sampler = SMOTE(k_neighbors=k, random_state=self.random_state)
        elif not numeric:
            sampler = SMOTEN(k_neighbors=k, random_state=self.random_state)
        else:
            sampler = SMOTENC(categorical_features=[frame.columns.get_loc(c) for c in categorical], k_neighbors=k,
                              random_state=self.random_state)
        new_X, new_y = sampler.fit_resample(filled, labels.to_numpy())
        new_X = pd.DataFrame(new_X, columns=frame.columns)
        added = new_X.iloc[len(frame):].reset_index(drop=True)
        for column in categorical:  # the new rows' values, in the original column type
            values = added[column].where(added[column] != "__missing__", np.nan)
            if isinstance(frame[column].dtype, pd.CategoricalDtype):
                added[column] = pd.Categorical(values, categories=frame[column].cat.categories)
            else:
                added[column] = values.astype(frame[column].dtype)
        for column in numeric:
            added[column] = added[column].astype(frame[column].dtype if pd.api.types.is_float_dtype(frame[column])
                                                 else float)
        out_X = pd.concat([frame.reset_index(drop=True), added], ignore_index=True)
        out_y = np.concatenate([labels.to_numpy(), np.asarray(new_y)[len(frame):]])
        if isinstance(y, pd.Series):
            return out_X, pd.Series(out_y, name=y.name)
        return out_X, out_y


def make_resampler(name: str | None, seed: int) -> FoldSafeSmote | None:
    """The resampler object for a Setup's ``resampler`` name (None: no resampling)."""
    if name is None:
        return None
    if name == "smote":
        return FoldSafeSmote(random_state=seed)
    raise ValueError(f"Unknown resampler '{name}'. Known: {', '.join(RESAMPLERS)}.")


__all__ = ["FoldSafeSmote", "RESAMPLERS", "make_resampler"]
