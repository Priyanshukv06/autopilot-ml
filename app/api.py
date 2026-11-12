"""Local HTTP API (optional task 35, Requirement 32): start Sessions and read their records.

    python scripts/run_api.py            # http://127.0.0.1:8000, documentation at /docs

Endpoints: ``GET /health``, ``GET /sessions``, ``GET /sessions/{id}``, ``POST /sessions`` (start one Session
in a background thread; one at a time), ``POST /sessions/{id}/stop``. A dataset is a CSV file inside the
project's ``data/`` folder, so the service never reads other files. There is no login: the service is for
this machine only and listens on 127.0.0.1 (in the Docker image, the host port is published on 127.0.0.1).
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import __version__
from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.services.automl_runner import ExperimentConfig
from app.services.autopilot import SELECTORS, AutopilotSession, SessionSettings
from app.services.resource_manager import SessionBudget
from app.services.task_detector import TaskType

logger = get_logger(__name__)
SESSION_ID = re.compile(r"SES-\d{3,}")
START_TIMEOUT = 60.0  # seconds to wait for a Session's ID (the data checks come before it)


class SessionRequest(BaseModel):
    dataset: str = Field(description="CSV file inside the project's data/ folder, e.g. data/sample/titanic.csv")
    target: str
    task: TaskType | None = None
    metric: str | None = None
    seconds: float = Field(120, ge=10, le=24 * 3600, description="total time budget")
    max_experiments: int = Field(8, ge=2, le=100)
    cores: int = Field(2, ge=1, le=max(1, os.cpu_count() or 1))
    seed: int = 42
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    exclude: list[str] = Field(default_factory=list)
    selector: str = "rules"
    use_memory: bool = False
    group_column: str | None = None


@dataclass
class _Running:
    session_id: str | None = None
    thread: threading.Thread | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    started: threading.Event = field(default_factory=threading.Event)
    error: str | None = None


def dataset_path(settings: Settings, name: str) -> Path:
    """The CSV file ``name`` (relative to the project) if it is inside ``data/``; otherwise HTTP 400."""
    path = (settings.project_root / name).resolve()
    if not path.is_relative_to(settings.data_dir.resolve()) or path.suffix.lower() != ".csv":
        raise HTTPException(400, "The dataset must be a .csv file inside the project's data/ folder.")
    if not path.is_file():
        raise HTTPException(404, f"No dataset at {name}.")
    return path


def create_app(settings: Settings | None = None,
               session_factory: Callable[[Settings], AutopilotSession] | None = None) -> FastAPI:
    settings = settings or get_settings()
    factory = session_factory or (lambda s: AutopilotSession(settings=s))
    api = FastAPI(title="AutoPilot-ML", version=__version__,
                  description="Start autonomous Sessions on local CSV files and read their records.")
    running: dict[str, _Running] = {}
    lock = threading.Lock()

    def record_path(session_id: str) -> Path:
        if not SESSION_ID.fullmatch(session_id):
            raise HTTPException(404, f"Not a session ID: {session_id}")
        return settings.experiments_dir / session_id / "session.json"

    def busy() -> _Running | None:
        with lock:
            return next((r for r in running.values() if r.thread is not None and r.thread.is_alive()), None)

    @api.get("/health")
    def health() -> dict[str, Any]:
        active = busy()
        return {"status": "ok", "version": __version__, "running": active.session_id if active else None}

    @api.get("/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        rows = []
        for path in sorted(settings.experiments_dir.glob("SES-*/session.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            base = record.get("settings", {}).get("base", {})
            rows.append({"session_id": record.get("session_id"), "status": record.get("status"),
                         "created_at": record.get("created_at"), "dataset": base.get("dataset_name"),
                         "target": base.get("target_column"), "stop_reason": record.get("stop_reason"),
                         "steps": len(record.get("steps") or [])})
        return rows

    @api.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        path = record_path(session_id)
        if not path.is_file():
            raise HTTPException(404, f"No record for {session_id}.")
        return json.loads(path.read_text(encoding="utf-8"))

    @api.post("/sessions", status_code=202)
    def start_session(request: SessionRequest) -> dict[str, Any]:
        if request.selector not in SELECTORS:
            raise HTTPException(400, f"Unknown selector '{request.selector}'. Known: {', '.join(SELECTORS)}.")
        path = dataset_path(settings, request.dataset)  # a bad request is refused first, running or not
        if busy() is not None:
            raise HTTPException(409, "A Session is already running; stop it or wait for it to finish.")
        frame = pd.read_csv(path)
        plan = SessionSettings(
            base=ExperimentConfig(target_column=request.target, task_type=request.task, primary_metric=request.metric,
                                  random_seed=request.seed, test_size=request.test_size,
                                  excluded_columns=list(request.exclude), dataset_name=Path(request.dataset).stem),
            budget=SessionBudget(total_seconds=request.seconds, max_experiments=request.max_experiments,
                                 n_jobs=request.cores),
            selector=request.selector, use_memory=request.use_memory, group_column=request.group_column,
        )
        job = _Running()

        def on_update(record: dict[str, Any]) -> None:
            if job.session_id is None:
                job.session_id = record["session_id"]
                job.started.set()

        def work() -> None:
            try:
                factory(settings).run(frame, plan, stop_requested=job.stop.is_set, on_update=on_update)
            except Exception as exc:  # reported to the caller; the details are in the log
                logger.exception("A Session started through the API failed")
                job.error = str(exc)
            finally:
                job.started.set()

        job.thread = threading.Thread(target=work, name="api-session", daemon=True)
        with lock:
            running[str(id(job))] = job
        job.thread.start()
        job.started.wait(START_TIMEOUT)
        if job.session_id is None:
            raise HTTPException(400, f"The Session did not start: {job.error or 'no record after 60 s'}")
        return {"session_id": job.session_id, "status": "running",
                "record": f"/sessions/{job.session_id}", "stop": f"/sessions/{job.session_id}/stop"}

    @api.post("/sessions/{session_id}/stop")
    def stop_session(session_id: str) -> dict[str, Any]:
        record_path(session_id)
        with lock:
            job = next((r for r in running.values() if r.session_id == session_id), None)
        if job is None or job.thread is None or not job.thread.is_alive():
            raise HTTPException(409, f"{session_id} is not running.")
        job.stop.set()
        return {"session_id": session_id, "stopping": True,
                "note": "The Session ends after its current step, then scores the first and final model."}

    return api


def __getattr__(name: str) -> Any:
    """``uvicorn app.api:app``: the app with the default settings, made when it is first asked for, so that
    importing this module (as the tests do) creates nothing."""
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
