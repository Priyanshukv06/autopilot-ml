"""Write a plain-language summary of a finished Session with a local language model (optional task 34).

    python scripts/summarise_session.py SES-001 [--model qwen2.5:7b-instruct] [--url http://localhost:11434]

Needs Ollama running on this machine with the model pulled (``ollama pull qwen2.5:7b-instruct``). The
summary is generated text: it is saved to ``experiments/SES-###/summary_generated.json`` with its label,
the facts it was given and the numbers in it that are not in those facts. No decision reads it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import build_settings  # noqa: E402
from app.services.narrator import DEFAULT_MODEL, DEFAULT_URL, NarrationError, narrate, save  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_id", help="for example SES-001")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_URL, help="the local Ollama server")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args(argv)

    folder = build_settings(Path(args.project_root)).experiments_dir / args.session_id
    record_path = folder / "session.json"
    if not record_path.exists():
        print(f"No record at {record_path}", file=sys.stderr)
        return 2
    try:
        narration = narrate(json.loads(record_path.read_text(encoding="utf-8")), model=args.model, url=args.url)
    except NarrationError as exc:
        print(f"No summary: {exc}", file=sys.stderr)
        return 2
    print(narration.label)
    print()
    print(narration.text)
    print()
    if narration.unsupported_numbers:
        print(f"Numbers not found in the record: {', '.join(narration.unsupported_numbers)}")
    print(f"{narration.seconds:.1f} s; written: {save(narration, folder)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
