"""Explain the final model of a finished Session with SHAP, on its Training_Part (optional task 32).

    python scripts/explain_session.py SES-001 --dataset data/sample/titanic.csv

Writes ``experiments/SES-###/shap_summary.json`` (mean |SHAP value| of the top columns, the method and
the rows used) and prints the top columns. The dataset must be the one the Session ran on: the split
is rebuilt from the recorded settings and checked against the recorded fingerprint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.core.config import build_settings  # noqa: E402
from app.services.explainer import MAX_ROWS, ExplanationError, explain_session  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_id", help="for example SES-001")
    parser.add_argument("--dataset", required=True, help="the CSV file the Session ran on")
    parser.add_argument("--rows", type=int, default=MAX_ROWS, help="Training_Part rows to explain at most")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args(argv)

    settings = build_settings(Path(args.project_root))
    record_path = settings.experiments_dir / args.session_id / "session.json"
    if not record_path.exists():
        print(f"No record at {record_path}", file=sys.stderr)
        return 2
    record = json.loads(record_path.read_text(encoding="utf-8"))
    dataset = Path(args.dataset)
    try:
        explanation = explain_session(record, pd.read_csv(dataset if dataset.is_absolute() else ROOT / dataset),
                                      settings, max_rows=args.rows)
    except (ExplanationError, ValueError) as exc:
        print(f"Not explained: {exc}", file=sys.stderr)
        return 2
    summary = {"session_id": args.session_id, "experiment_id": record["final_experiment_id"],
               **explanation.summary()}
    out = record_path.with_name("shap_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{explanation.method} explainer for {explanation.estimator}, {len(explanation.row_labels)} training rows, "
          f"{explanation.seconds:.1f} s; values in {explanation.units}")
    for item in summary["importance"]:
        print(f"  {item['column']:30s} {item['mean_abs_shap']:.4f}")
    print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
