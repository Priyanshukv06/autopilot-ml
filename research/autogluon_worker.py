"""AutoGluon worker (spec task 23): runs in its own conda environment (decision D4), called by
``research/systems.py`` as a separate process. It must not import the ``app`` package.

Input folder: ``train.csv`` (features and target), ``test.csv`` (features only) and ``settings.json``
(label, problem_type, eval_metric, positive_class, time_limit, num_cpus, presets). Output, in the same
folder: ``predictions.csv`` (one ``prediction`` column, and one probability column per class for
classification, in sorted class order) and ``result.json`` (AutoGluon version, best model, models
trained, fit seconds). The caller scores the predictions with its own Evaluator, so every system is
scored the same way.

    <autogluon python> research/autogluon_worker.py <folder>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main(folder: Path) -> None:
    import pandas as pd
    from autogluon.tabular import TabularPredictor, __version__

    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    train = pd.read_csv(folder / "train.csv")
    test = pd.read_csv(folder / "test.csv")
    label = settings["label"]
    predictor = TabularPredictor(
        label=label, problem_type=settings["problem_type"], eval_metric=settings["eval_metric"],
        path=str(folder / "model"), verbosity=0,
        **({"positive_class": settings["positive_class"]} if settings.get("positive_class") is not None else {}),
    )
    started = time.perf_counter()
    predictor.fit(train, time_limit=settings["time_limit"], presets=settings.get("presets", "medium_quality"),
                  num_cpus=settings.get("num_cpus", "auto"), num_gpus=0)
    fit_seconds = time.perf_counter() - started

    out = pd.DataFrame({"prediction": predictor.predict(test)})
    if settings["problem_type"] in ("binary", "multiclass"):
        proba = predictor.predict_proba(test)
        for column in sorted(proba.columns, key=str):
            out[f"proba::{column}"] = proba[column].to_numpy()
    out.to_csv(folder / "predictions.csv", index=False)
    leaderboard = predictor.leaderboard(silent=True)
    (folder / "result.json").write_text(json.dumps({
        "autogluon_version": __version__,
        "best_model": predictor.model_best,
        "models": [str(m) for m in leaderboard["model"]],
        "fit_seconds": round(fit_seconds, 2),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
