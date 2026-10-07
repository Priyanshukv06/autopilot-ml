"""Experiment tracking: local MLflow (SQLite backend) plus a lightweight JSON index.

* MLflow stores params, metrics and artifacts for every run (browse with ``mlflow ui``).
* ``experiments/EXP-###/summary.json`` is a small, engine-independent record of each
  experiment. It drives the UI history table today and is the raw material for the
  planned experiment-memory module.

This module knows nothing about FLAML; it only receives plain dictionaries and files.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import Settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_EXP_DIR_PATTERN = re.compile(r"EXP-(\d+)")
_INVALID_KEY_CHARS = re.compile(r"[^\w\-./: ]")
_MAX_PARAM_LENGTH = 500
_BATCH_SIZE = 100


def _clean_key(key: str) -> str:
    return _INVALID_KEY_CHARS.sub("_", str(key))


def _is_finite_number(value: Any) -> bool:
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class ExperimentTracker:
    """Persist experiment results to MLflow and to a local JSON index."""

    def __init__(self, tracking_uri: str, experiment_name: str, artifact_uri: str, experiments_dir: Path) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_uri = artifact_uri
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._mlflow_experiment_id: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "ExperimentTracker":
        return cls(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            artifact_uri=settings.mlflow_artifact_uri,
            experiments_dir=settings.experiments_dir,
        )

    # ------------------------------------------------------------ local index
    def _existing_numbers(self) -> list[int]:
        return [
            int(match.group(1))
            for path in self.experiments_dir.iterdir()
            if path.is_dir() and (match := _EXP_DIR_PATTERN.fullmatch(path.name))
        ]

    def next_experiment_id(self) -> str:
        """Peek at the next free id (does not reserve it)."""
        return f"EXP-{max(self._existing_numbers(), default=0) + 1:03d}"

    def allocate_experiment_id(self) -> str:
        """Reserve the next id by creating its directory atomically (safe for concurrent runs)."""
        number = max(self._existing_numbers(), default=0) + 1
        while True:
            experiment_id = f"EXP-{number:03d}"
            try:
                (self.experiments_dir / experiment_id).mkdir(parents=True, exist_ok=False)
                return experiment_id
            except FileExistsError:
                number += 1

    def experiment_dir(self, experiment_id: str) -> Path:
        path = self.experiments_dir / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_record(self, experiment_id: str, record: dict[str, Any]) -> Path:
        path = self.experiment_dir(experiment_id) / "summary.json"
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return path

    def load_records(self) -> list[dict[str, Any]]:
        records = []
        for summary in sorted(self.experiments_dir.glob("EXP-*/summary.json")):
            try:
                records.append(json.loads(summary.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable experiment record %s: %s", summary, exc)
        return records

    def history_frame(self) -> pd.DataFrame:
        """Tabular experiment history (one row per completed experiment)."""
        rows = []
        for r in self.load_records():
            rows.append(
                {
                    "Experiment": r.get("experiment_id"),
                    "Dataset": r.get("dataset_name"),
                    "Target": r.get("target_column"),
                    "Task": r.get("task_type"),
                    "Model": r.get("best_model_name"),
                    "Metric": r.get("primary_metric"),
                    "Test score": r.get("test_score"),
                    "Validation score": r.get("best_validation_score"),
                    "Train score": r.get("train_score"),
                    "Budget (s)": r.get("time_budget"),
                    "Training time (s)": r.get("training_time"),
                    "Trials": r.get("n_trials"),
                    "Created": r.get("created_at"),
                    "MLflow run": r.get("mlflow_run_id"),
                }
            )
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------- MLflow
    @property
    def client(self):
        if self._client is None:
            # MLflow's SQLAlchemy store emits a harmless deprecation warning on first use,
            # and recent MLflow versions print an "agent hint" banner on import.
            warnings.filterwarnings("ignore", message=".*noload.*", category=DeprecationWarning)
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            from mlflow.tracking import MlflowClient

            # MLflow resets its own logger level on import; keep it quiet.
            logging.getLogger("mlflow").setLevel(logging.WARNING)
            self._client = MlflowClient(tracking_uri=self.tracking_uri)
        return self._client

    def _get_mlflow_experiment_id(self) -> str:
        if self._mlflow_experiment_id is not None:
            return self._mlflow_experiment_id
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = self.client.create_experiment(self.experiment_name, artifact_location=self.artifact_uri)
        else:
            experiment_id = experiment.experiment_id
            if experiment.lifecycle_stage == "deleted":
                self.client.restore_experiment(experiment_id)
        self._mlflow_experiment_id = experiment_id
        return experiment_id

    def log_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        tags: dict[str, str],
        artifacts: list[tuple[Path, str | None]],
    ) -> str:
        """Create a finished MLflow run; returns its run id."""
        from mlflow.entities import Metric, Param

        run = self.client.create_run(
            self._get_mlflow_experiment_id(), run_name=run_name, tags={k: str(v) for k, v in tags.items()}
        )
        run_id = run.info.run_id
        try:
            param_entities = [
                Param(_clean_key(k), str(v)[:_MAX_PARAM_LENGTH]) for k, v in params.items() if v is not None
            ]
            timestamp = int(time.time() * 1000)
            metric_entities = [
                Metric(_clean_key(k), float(v), timestamp, 0) for k, v in metrics.items() if _is_finite_number(v)
            ]
            for start in range(0, len(param_entities), _BATCH_SIZE):
                self.client.log_batch(run_id, params=param_entities[start : start + _BATCH_SIZE])
            for start in range(0, len(metric_entities), _BATCH_SIZE):
                self.client.log_batch(run_id, metrics=metric_entities[start : start + _BATCH_SIZE])
            for local_path, artifact_dir in artifacts:
                if Path(local_path).exists():
                    self.client.log_artifact(run_id, str(local_path), artifact_path=artifact_dir)
            self.client.set_terminated(run_id, "FINISHED")
        except Exception:
            self.client.set_terminated(run_id, "FAILED")
            raise
        logger.info("Logged MLflow run %s (%s)", run_id, run_name)
        return run_id
