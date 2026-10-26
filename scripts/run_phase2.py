"""Phase 2 measurement (spec task 19): Sessions against FLAML alone with the same total budget.

From the project root:

    python scripts/run_phase2.py run        # all runs; a restart skips finished runs
    python scripts/run_phase2.py analyse    # pairs, summary and Strategy counts

The records of every run (experiments/, models/, mlruns/ and the memory) go to the project folders
by default, because they are the raw results. Use --project-root with a temporary folder for a trial.
The CSV files go to <project root>/results/phase2/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.core.config import build_settings  # noqa: E402
from research import phase2  # noqa: E402
from research.critic_study import read_rows  # noqa: E402


def run(args: argparse.Namespace, results: Path) -> None:
    settings = build_settings(Path(args.project_root))
    settings.ensure_directories()
    results.mkdir(parents=True, exist_ok=True)
    runs_path = results / "phase2_runs.csv"
    rows = read_rows(runs_path)
    done = {row["cell_id"]: row for row in rows}
    todo = [c for c in phase2.cells(args.datasets, args.seeds, args.seconds) if c.cell_id not in done]
    todo = todo[: args.limit or None]
    print(f"{len(done)} runs already done, {len(todo)} to run", flush=True)
    for number, cell in enumerate(todo, start=1):
        if cell.system == "session":
            row = phase2.run_session(cell, settings, n_jobs=args.cores, enable_mlflow=not args.no_mlflow)
        else:
            session = done.get(phase2.Cell("session", cell.dataset, cell.seed, cell.seconds).cell_id)
            excluded = json.loads(session["excluded_columns"]) if session else []
            row = phase2.run_flaml(cell, settings, excluded, n_jobs=args.cores, enable_mlflow=not args.no_mlflow)
        pd.DataFrame([row]).to_csv(runs_path, mode="a", header=not runs_path.exists(), index=False)
        done[cell.cell_id] = row
        score = "n/a" if row["test_score"] is None else f"{row['test_score']:.4f}"
        print(f"[{number}/{len(todo)}] {cell.cell_id}: test {row['metric']} {score}, {row['n_experiments']} "
              f"Experiments, {row['wall_seconds']:.0f} s, kept: {row['kept'] or '-'}", flush=True)


def analyse(results: Path) -> None:
    rows = read_rows(results / "phase2_runs.csv")
    if not rows:
        sys.exit("No results yet: run 'run' first.")
    pair_rows = phase2.pairs(rows)
    pd.DataFrame(pair_rows).to_csv(results / "phase2_pairs.csv", index=False)
    summary = pd.DataFrame(phase2.summary(pair_rows))
    summary.to_csv(results / "phase2_summary.csv", index=False)
    strategies = pd.DataFrame(phase2.strategy_counts(rows))
    strategies.to_csv(results / "phase2_strategies.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(pair_rows)[["dataset", "seed", "session_test", "flaml_test", "difference", "loop_gain",
                                   "session_seconds", "flaml_seconds", "session_experiments", "stop_reason"]]
          .round(4).to_string(index=False))
    print()
    print(summary.round(4).to_string(index=False))
    print()
    print(strategies.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["run", "analyse"])
    parser.add_argument("--seconds", type=int, default=phase2.SECONDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(phase2.SEEDS))
    parser.add_argument("--datasets", nargs="+", default=list(phase2.DATASETS), choices=list(phase2.DATASETS))
    parser.add_argument("--limit", type=int, default=0, help="run at most this many new runs")
    parser.add_argument("--cores", type=int, default=-1)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--project-root", default=str(ROOT), help="folder for the records and results/")
    args = parser.parse_args()
    results = Path(args.project_root) / "results" / "phase2"
    run(args, results) if args.action == "run" else analyse(results)


if __name__ == "__main__":
    main()
