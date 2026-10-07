"""Central configuration for AutoPilot-ML.

Values are read from environment variables (optionally from a local ``.env`` file)
with safe local defaults. Nothing here requires network access or a paid service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

ENV_PREFIX = "AUTOPILOT_"


def _env(name: str) -> str | None:
    """Return a stripped environment value, treating empty strings as unset."""
    value = os.getenv(ENV_PREFIX + name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    project_root: Path
    data_dir: Path
    sample_data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    experiments_dir: Path
    models_dir: Path
    mlruns_dir: Path

    mlflow_tracking_uri: str
    mlflow_artifact_uri: str
    mlflow_experiment_name: str

    default_time_budget: int
    min_time_budget: int
    max_time_budget: int
    default_random_seed: int
    default_test_size: float
    n_jobs: int
    max_upload_mb: int
    log_level: str

    def ensure_directories(self) -> None:
        """Create the local working directories if they do not exist yet."""
        for path in (
            self.sample_data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.experiments_dir,
            self.models_dir,
            self.mlruns_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def build_settings(project_root: Path | None = None) -> Settings:
    """Build settings for a given project root (used directly by tests)."""
    root = (project_root or PROJECT_ROOT).resolve()
    data_dir = root / "data"
    mlruns_dir = root / "mlruns"

    tracking_uri = _env("MLFLOW_TRACKING_URI") or f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"
    artifact_uri = (mlruns_dir / "artifacts").as_uri()

    min_budget, max_budget = 5, 3600
    default_budget = min(max(_env_int("DEFAULT_TIME_BUDGET", 60), min_budget), max_budget)

    return Settings(
        project_root=root,
        data_dir=data_dir,
        sample_data_dir=data_dir / "sample",
        raw_data_dir=data_dir / "raw",
        processed_data_dir=data_dir / "processed",
        experiments_dir=root / "experiments",
        models_dir=root / "models",
        mlruns_dir=mlruns_dir,
        mlflow_tracking_uri=tracking_uri,
        mlflow_artifact_uri=artifact_uri,
        mlflow_experiment_name=_env("MLFLOW_EXPERIMENT_NAME") or "autopilot-ml-baseline",
        default_time_budget=default_budget,
        min_time_budget=min_budget,
        max_time_budget=max_budget,
        default_random_seed=_env_int("DEFAULT_SEED", 42),
        default_test_size=_env_float("DEFAULT_TEST_SIZE", 0.2),
        n_jobs=_env_int("N_JOBS", -1),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 200),
        log_level=(_env("LOG_LEVEL") or "INFO").upper(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (loads ``.env`` once if present)."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = build_settings()
    settings.ensure_directories()
    return settings
