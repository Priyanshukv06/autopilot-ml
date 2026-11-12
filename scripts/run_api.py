"""Run the local HTTP API (optional task 35).

    python scripts/run_api.py [--host 127.0.0.1] [--port 8000]

Documentation of the endpoints at http://127.0.0.1:8000/docs. The service has no login, so it listens on
this machine only by default; only change ``--host`` inside a container whose port is published on
127.0.0.1 (see the Dockerfile).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    uvicorn.run("app.api:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
