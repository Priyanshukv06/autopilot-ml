"""SHAP explanations of a Session's final model (optional task 32, Requirement 29).

SHAP (Lundberg and Lee, 2017) splits each prediction into one contribution per input column, so that
the contributions add up to the prediction minus the average prediction. The explanation is computed on
Training_Part rows only (Requirement 29.1): the Test_Part stays for the Final_Evaluation.

The columns explained are the model's own input columns: the Data_Steps' output (for example
``Name__tok_mr``) after FLAML's preparation of the rows. Tree models (LightGBM, XGBoost, random forest,
extra trees) use shap's exact ``TreeExplainer``; any other model, or a tree model that ``TreeExplainer``
refuses, uses the model-agnostic ``PermutationExplainer`` on the class probabilities (or the predicted
value), against a background of Training_Part rows.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.services.transforms import PipelineModel

MAX_ROWS = 200  # Training_Part rows explained (a seeded sample when there are more)
# The model-agnostic explainer calls the model thousands of times per row: on Titanic (12 November 2026),
# 200 rows against 50 background rows took 28.5 s for L1 logistic regression and 169 s for k-nearest
# neighbours, so it explains the first 50 sampled rows against 25 background rows.
PERMUTATION_ROWS = 50
BACKGROUND_ROWS = 25
TOP_COLUMNS = 15


class ExplanationError(ValueError):
    """The model or the rows cannot be explained."""


@dataclass
class ShapExplanation:
    method: str  # "tree" or "permutation"
    estimator: str  # class name of the explained model
    units: str  # what the values add up to, for example "log-odds" or "probability"
    outputs: list[str]  # one per explained output: the class for classification, "prediction" for regression
    columns: list[str]  # the model's input columns
    values: np.ndarray  # rows x columns x outputs
    base_values: np.ndarray  # outputs: the average prediction the contributions start from
    data: pd.DataFrame  # the explained rows, as the model sees them
    row_labels: list[Any]  # labels of the explained Training_Part rows
    seconds: float

    def importance(self, output: int | None = None) -> pd.DataFrame:
        """Mean |SHAP value| per column over the explained rows (averaged over the outputs if ``output`` is None)."""
        values = np.abs(self.values if output is None else self.values[:, :, [output]]).mean(axis=(0, 2))
        frame = pd.DataFrame({"column": self.columns, "mean_abs_shap": values})
        return frame.sort_values(["mean_abs_shap", "column"], ascending=[False, True], ignore_index=True)

    def row(self, position: int, output: int = 0) -> pd.DataFrame:
        """The contributions to one explained row's prediction, largest first."""
        frame = pd.DataFrame({"column": self.columns, "value": [_text(v) for v in self.data.iloc[position]],
                              "shap": self.values[position, :, output]})
        return frame.reindex(frame["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    def summary(self, top: int = TOP_COLUMNS) -> dict[str, Any]:
        """JSON-safe summary, for the records."""
        importance = self.importance().head(top)
        return {"method": self.method, "estimator": self.estimator, "units": self.units, "outputs": self.outputs,
                "rows": len(self.row_labels), "columns": len(self.columns), "seconds": round(self.seconds, 2),
                "base_values": [float(v) for v in self.base_values],
                "importance": [{"column": c, "mean_abs_shap": float(v)}
                               for c, v in zip(importance["column"], importance["mean_abs_shap"])]}


def _text(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def model_input(model: Any, rows: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """The fitted estimator and ``rows`` as it sees them: through the Data_Steps (``PipelineModel``), then
    FLAML's preparation (its data transformer and the learner's own preprocessing)."""
    if isinstance(model, PipelineModel):
        rows, model = model.pipeline.transform(rows), model.model
    trained = getattr(model, "_trained_estimator", None)
    if trained is not None and hasattr(model, "_state"):  # a FLAML AutoML object
        rows = trained._preprocess(model._state.task.preprocess(rows, model._transformer))
        model = trained._model
    if not isinstance(rows, pd.DataFrame):
        rows = pd.DataFrame(rows)
    return model, rows


def explain(model: Any, X: pd.DataFrame, task_type: str, classes: list[Any] | None = None, *, seed: int = 0,
            max_rows: int = MAX_ROWS, background_rows: int = BACKGROUND_ROWS) -> ShapExplanation:
    """SHAP values of ``model`` for Training_Part rows ``X`` (original columns), at most ``max_rows`` of them."""
    import shap  # imported here: it takes a few seconds and only this feature needs it

    if len(X) == 0:
        raise ExplanationError("There are no rows to explain.")
    started = time.perf_counter()
    rows = X.sample(n=max_rows, random_state=seed) if len(X) > max_rows else X
    estimator, data = model_input(model, rows)
    classes = list(classes) if classes is not None else list(getattr(estimator, "classes_", []))
    classification = task_type in ("binary", "multiclass")
    outputs = ([str(classes[-1])] if task_type == "binary" else [str(c) for c in classes]) if classification \
        else ["prediction"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # shap and the learners warn about feature names and future changes
        try:
            explanation = shap.TreeExplainer(estimator)(data, check_additivity=False)
            method, units = "tree", _tree_units(estimator, classification)
        except Exception:  # not a tree model shap knows, or one it cannot read
            rows, data = rows.iloc[:PERMUTATION_ROWS], data.iloc[:PERMUTATION_ROWS]
            explanation = _permutation(shap, estimator, data, X, model, classification, seed, background_rows)
            method, units = "permutation", "probability" if classification else "predicted value"
    values = np.asarray(explanation.values, dtype=float)
    base = np.atleast_1d(np.asarray(explanation.base_values, dtype=float))
    if values.ndim == 2:  # one output: regression, or the positive class of a binary model
        values = values[:, :, None]
        base = base.reshape(len(data), -1)[0] if base.size == len(data) else base[:1]
    else:
        base = base.reshape(len(data), -1)[0] if base.size == len(data) * values.shape[2] else base
        if task_type == "binary":  # both classes given: keep the positive one
            values, base = values[:, :, -1:], base[-1:]
    return ShapExplanation(method=method, estimator=type(estimator).__name__, units=units, outputs=outputs,
                           columns=[str(c) for c in data.columns], values=values, base_values=base, data=data,
                           row_labels=list(rows.index), seconds=time.perf_counter() - started)


def explain_session(record: dict[str, Any], dataframe: pd.DataFrame, settings: Any, *, seed: int | None = None,
                    max_rows: int = MAX_ROWS) -> ShapExplanation:
    """Explain the final model of a finished Session on its Training_Part, rebuilt from ``dataframe`` with
    the Session's split settings; refused if the rebuilt split is not the Session's (other data)."""
    from app.services.automl_runner import AutoMLRunner, ExperimentConfig
    from app.services.splits import make_outer_split
    from app.services.task_detector import TaskType

    experiment_id = record.get("final_experiment_id")
    if not experiment_id:
        raise ExplanationError("The Session has no final model.")
    path = settings.models_dir / experiment_id / "model.pkl"
    if not path.exists():
        raise ExplanationError(f"The model file of {experiment_id} is missing: {path}")
    bundle = AutoMLRunner.load_model(path)  # a file this application wrote (pickle)
    base = record["settings"]["base"]
    config = ExperimentConfig(target_column=base["target_column"], task_type=TaskType(base["task_type"]),
                              primary_metric=base["primary_metric"], test_size=base["test_size"],
                              random_seed=base["random_seed"], excluded_columns=list(base["excluded_columns"]))
    _, y, task_type, _, _ = AutoMLRunner(settings=settings, enable_mlflow=False).prepare(dataframe, config)
    group = record["settings"].get("group_column")
    labelled = dataframe.loc[y.index]
    split, _ = make_outer_split(y, task_type, config.test_size, config.random_seed,
                                groups=labelled[group] if group else None)
    if split.fingerprint() != record["splits"]["outer"]:
        raise ExplanationError("These rows are not the Session's data: the rebuilt split differs from the recorded one.")
    X_train = labelled.iloc[split.train_positions][list(bundle["feature_columns"])]
    return explain(bundle["model"], X_train, bundle["task_type"], bundle.get("classes"),
                   seed=base["random_seed"] if seed is None else seed, max_rows=max_rows)


def _tree_units(estimator: Any, classification: bool) -> str:
    if not classification:
        return "predicted value"
    name = type(estimator).__name__
    return "probability" if name.startswith(("RandomForest", "ExtraTrees")) else "log-odds"


def _permutation(shap: Any, estimator: Any, data: pd.DataFrame, X: pd.DataFrame, model: Any,
                 classification: bool, seed: int, background_rows: int) -> Any:
    """The model-agnostic explainer on numbers: text categories become their codes, and are turned back
    into categories before the estimator sees them."""
    categories = {c: data[c].cat.categories for c in data.columns if isinstance(data[c].dtype, pd.CategoricalDtype)}
    dtypes = data.dtypes

    def numbers(frame: pd.DataFrame) -> np.ndarray:
        frame = frame.copy()
        for column in categories:
            frame[column] = frame[column].cat.codes.replace(-1, np.nan)
        return frame.astype(float).to_numpy()

    def frame_of(matrix: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(matrix, columns=data.columns)
        for column, dtype in dtypes.items():
            if column in categories:
                codes = np.where(np.isnan(frame[column]), -1, np.rint(frame[column])).astype(int)
                frame[column] = pd.Categorical.from_codes(codes, categories=categories[column])
            elif not pd.api.types.is_float_dtype(dtype):
                frame[column] = frame[column].fillna(0).astype(dtype) if pd.api.types.is_integer_dtype(dtype) \
                    else frame[column].astype(dtype)
        return frame

    predict = estimator.predict_proba if classification else estimator.predict
    background_source = X.sample(n=background_rows, random_state=seed + 1) if len(X) > background_rows else X
    background = numbers(model_input(model, background_source)[1])
    explainer = shap.PermutationExplainer(lambda matrix: predict(frame_of(matrix)),
                                          shap.maskers.Independent(background, max_samples=background_rows),
                                          seed=seed)
    return explainer(numbers(data), silent=True)
