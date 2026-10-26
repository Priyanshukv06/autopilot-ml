"""Experiment tracking: local MLflow (SQLite backend) plus a lightweight JSON index.

* MLflow stores params, metrics and artifacts for every run (browse with ``mlflow ui``). A
  Session gets one parent run, with one child run per Experiment.
* ``experiments/EXP-###/summary.json`` is a small, engine-independent record of each
  experiment, and ``experiments/SES-###/session.json`` the record of a Session. They drive
  the UI history tables and are the raw material for the experiment memory.

JSON records are written to a temporary file and then renamed over the old one, so a crash while
writing never leaves a half-written record. This module knows nothing about FLAML; it only
receives plain dictionaries and files.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import Settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_EXP_DIR_PATTERN = re.compile(r"EXP-(\d+)")
_SES_DIR_PATTERN = re.compile(r"SES-(\d+)")
_INVALID_KEY_CHARS = re.compile(r"[^\w\-./: ]")
_MAX_PARAM_LENGTH = 500
_BATCH_SIZE = 100
# Records written before Phase 2 have no schema_version; they are version 1, and the fields
# added since then get these values.
_PHASE1_DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "session_id": None,
    "parent_experiment_id": None,
    "proposal": None,
    "test_evaluated": True,
}
_STEP_FILES = ("assessment.json", "diagnoses.json", "decision.json")


def write_json_atomic(path: Path, payload: Any) -> Path:
    """Write JSON to a temporary file in the same folder, then rename it over ``path``.

    A crash while writing leaves the old file as it was. On Windows the rename can fail for a moment
    while another program reads the file, so it is retried a few times.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=str)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


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
        self._session_runs: dict[str, str] = {}  # session id -> MLflow parent run id

    @classmethod
    def from_settings(cls, settings: Settings) -> "ExperimentTracker":
        return cls(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            artifact_uri=settings.mlflow_artifact_uri,
            experiments_dir=settings.experiments_dir,
        )

    # ------------------------------------------------------------ local index
    def _existing_numbers(self, pattern: re.Pattern[str] = _EXP_DIR_PATTERN) -> list[int]:
        return [
            int(match.group(1))
            for path in self.experiments_dir.iterdir()
            if path.is_dir() and (match := pattern.fullmatch(path.name))
        ]

    def _allocate(self, prefix: str, pattern: re.Pattern[str]) -> str:
        """Reserve the next id by creating its directory atomically (safe for concurrent runs)."""
        number = max(self._existing_numbers(pattern), default=0) + 1
        while True:
            new_id = f"{prefix}-{number:03d}"
            try:
                (self.experiments_dir / new_id).mkdir(parents=True, exist_ok=False)
                return new_id
            except FileExistsError:
                number += 1

    def next_experiment_id(self) -> str:
        """Peek at the next free id (does not reserve it)."""
        return f"EXP-{max(self._existing_numbers(), default=0) + 1:03d}"

    def allocate_experiment_id(self) -> str:
        """Reserve the next experiment id (EXP-###) by creating its directory."""
        return self._allocate("EXP", _EXP_DIR_PATTERN)

    def allocate_session_id(self) -> str:
        """Reserve the next session id (SES-###) by creating its directory, so two Sessions never share one."""
        return self._allocate("SES", _SES_DIR_PATTERN)

    def experiment_dir(self, experiment_id: str) -> Path:
        path = self.experiments_dir / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def session_dir(self, session_id: str) -> Path:
        path = self.experiments_dir / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_record(self, experiment_id: str, record: dict[str, Any]) -> Path:
        return write_json_atomic(self.experiment_dir(experiment_id) / "summary.json", record)

    def save_step_file(self, experiment_id: str, name: str, payload: Any) -> Path:
        """Write ``assessment.json``, ``diagnoses.json`` or ``decision.json`` of an Experiment made in a Session."""
        if name not in _STEP_FILES:
            raise ValueError(f"Unknown step file '{name}'. Use one of {', '.join(_STEP_FILES)}.")
        return write_json_atomic(self.experiment_dir(experiment_id) / name, payload)

    def save_session(self, session_id: str, record: dict[str, Any]) -> Path:
        """Write ``session.json``; called after every step, so it always holds the finished steps."""
        return write_json_atomic(self.session_dir(session_id) / "session.json", record)

    def load_session(self, session_id: str) -> dict[str, Any]:
        return json.loads((self.experiments_dir / session_id / "session.json").read_text(encoding="utf-8"))

    def load_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for path in sorted(self.experiments_dir.glob("SES-*/session.json")):
            try:
                sessions.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable session record %s: %s", path, exc)
        return sessions

    def load_records(self) -> list[dict[str, Any]]:
        """Every experiment record; Phase 1 records are read as schema version 1."""
        records = []
        for summary in sorted(self.experiments_dir.glob("EXP-*/summary.json")):
            try:
                record = json.loads(summary.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable experiment record %s: %s", summary, exc)
                continue
            records.append({**_PHASE1_DEFAULTS, **record})
        return records

    def history_frame(self) -> pd.DataFrame:
        """Tabular experiment history (one row per completed experiment)."""
        rows = []
        for r in self.load_records():
            rows.append(
                {
                    "Experiment": r.get("experiment_id"),
                    "Session": r.get("session_id"),
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

    def session_history_frame(self) -> pd.DataFrame:
        """One row per Session: first and final validation scores, their test scores from the
        Final_Evaluation, the number of Experiments, the time used and the stop reason."""
        rows = []
        for s in self.load_sessions():
            base = (s.get("settings") or {}).get("base") or {}
            metric = base.get("primary_metric")
            steps = s.get("steps") or []
            accepted = [step for step in steps if step.get("accepted")]
            final_eval = s.get("final_evaluation") or {}

            def validation(step: dict[str, Any] | None) -> float | None:
                return ((step or {}).get("assessment") or {}).get("validation_mean")

            def test(part: str) -> float | None:
                return ((final_eval.get(part) or {}).get("test_metrics") or {}).get(metric)

            rows.append(
                {
                    "Session": s.get("session_id"),
                    "Dataset": base.get("dataset_name"),
                    "Target": base.get("target_column"),
                    "Metric": metric,
                    "Status": s.get("status"),
                    "First validation": validation(steps[0] if steps else None),
                    "Final validation": validation(accepted[-1] if accepted else None),
                    "First test": test("first"),
                    "Final test": test("final"),
                    "Experiments": len(steps),
                    "Time (s)": s.get("total_seconds"),
                    "Stop reason": s.get("stop_reason"),
                    "Created": s.get("created_at"),
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

    def _log_contents(self, run_id: str, params: dict[str, Any], metrics: dict[str, Any],
                      artifacts: list[tuple[Path, str | None]]) -> None:
        from mlflow.entities import Metric, Param

        param_entities = [Param(_clean_key(k), str(v)[:_MAX_PARAM_LENGTH]) for k, v in params.items() if v is not None]
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

    def log_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        tags: dict[str, str],
        artifacts: list[tuple[Path, str | None]],
        parent_run_id: str | None = None,
    ) -> str:
        """Create a finished MLflow run; returns its run id. With ``parent_run_id`` it is a child run."""
        run_tags = {k: str(v) for k, v in tags.items()}
        if parent_run_id:
            run_tags["mlflow.parentRunId"] = parent_run_id  # MLflow shows the run nested under its parent
        run = self.client.create_run(self._get_mlflow_experiment_id(), run_name=run_name, tags=run_tags)
        run_id = run.info.run_id
        try:
            self._log_contents(run_id, params, metrics, artifacts)
            self.client.set_terminated(run_id, "FINISHED")
        except Exception:
            self.client.set_terminated(run_id, "FAILED")
            raise
        logger.info("Logged MLflow run %s (%s)", run_id, run_name)
        return run_id

    def start_session_run(self, session_id: str, params: dict[str, Any], tags: dict[str, str]) -> str:
        """Open the MLflow parent run of a Session (run name = session id); its Experiments become child runs."""
        run_tags = {"session_id": session_id, "project": "autopilot-ml", "phase": "session",
                    **{k: str(v) for k, v in tags.items()}}
        run = self.client.create_run(self._get_mlflow_experiment_id(), run_name=session_id, tags=run_tags)
        run_id = run.info.run_id
        try:
            self._log_contents(run_id, params, {}, [])
        except Exception:
            self.client.set_terminated(run_id, "FAILED")
            raise
        self._session_runs[session_id] = run_id
        return run_id

    def session_run_id(self, session_id: str | None) -> str | None:
        """The parent run of a Session started by this tracker, if any."""
        return self._session_runs.get(session_id) if session_id else None

    def finish_session_run(self, session_id: str, metrics: dict[str, Any], artifacts: list[tuple[Path, str | None]],
                           failed: bool = False) -> None:
        run_id = self._session_runs.get(session_id)
        if run_id is None:
            return
        try:
            self._log_contents(run_id, {}, metrics, artifacts)
        finally:
            self.client.set_terminated(run_id, "FAILED" if failed else "FINISHED")
