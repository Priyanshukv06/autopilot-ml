"""Logging setup shared by the UI, services and tests."""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
APP_LOGGER_NAME = "app"

# Third-party loggers that are very chatty at INFO level.
_NOISY_LOGGERS = ("flaml", "mlflow", "alembic", "urllib3", "matplotlib", "lightgbm")

_configured = False


def setup_logging(level: str | int = "INFO") -> None:
    """Configure the ``app`` logger hierarchy once (safe to call on every Streamlit rerun)."""
    global _configured
    if _configured:
        return

    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    app_logger.addHandler(handler)
    # Do not propagate to the root logger: Streamlit configures its own handlers.
    app_logger.propagate = False

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the ``app`` hierarchy (e.g. ``app.services.profiler``)."""
    if not name.startswith(APP_LOGGER_NAME):
        name = f"{APP_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
