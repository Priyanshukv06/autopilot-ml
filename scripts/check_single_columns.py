"""Run the single-column leakage check and the target suggestion on every registered dataset (task 33).

    python scripts/check_single_columns.py [--out results/leakage_check.csv]

For each dataset of ``research/datasets.json``: the Training_Part of the seed-42 outer split (test share
0.2), the best single-column score and how many columns reach 0.99, the seconds taken, and where the
registered target comes in ``suggest_targets``. Nothing is written to ``experiments/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.services.critic import OPTIONAL_THRESHOLDS  # noqa: E402
from app.services.profiler import single_column_scores  # noqa: E402
from app.services.splits import make_outer_split  # noqa: E402
from app.services.task_detector import TaskDetector, TaskType  # noqa: E402

SEED, TEST_SIZE = 42, 0.2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "results" / "leakage_check.csv"))
    args = parser.parse_args(argv)
    threshold = OPTIONAL_THRESHOLDS["possible_leakage.min_score"]
    registry = json.loads((ROOT / "research" / "datasets.json").read_text(encoding="utf-8"))
    rows = []
    for entry in registry["datasets"]:
        df = pd.read_csv(ROOT / entry["file"])
        target, task = entry["target"], TaskType(entry["task"])
        y = df[target].dropna()
        split, _ = make_outer_split(y, task, TEST_SIZE, SEED)
        train = df.loc[y.index].iloc[split.train_positions]
        started = time.perf_counter()
        scores, metric = single_column_scores(train, target, task)
        seconds = time.perf_counter() - started
        best = max(scores, key=scores.get)
        ranking = [s.column for s in TaskDetector().suggest_targets(df, limit=len(df.columns))]
        rows.append({"dataset": entry["name"], "split": entry["split"], "task": task.value, "rows": len(train),
                     "columns": len(scores), "metric": metric, "best_column": best,
                     "best_score": round(scores[best], 4), "columns_at_threshold": sum(s >= threshold
                                                                                      for s in scores.values()),
                     "threshold": threshold, "seconds": round(seconds, 2), "target": target,
                     "suggested": ranking[0] if ranking else None,
                     "target_rank": ranking.index(target) + 1 if target in ranking else None})
        print(f"{entry['name']:24s} best {best} {scores[best]:.4f} ({metric}), {seconds:.2f} s, "
              f"target suggested at rank {rows[-1]['target_rank']}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
