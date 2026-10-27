"""Build and freeze the Experiment_Memory of the Phase 3 benchmark (spec task 26), from development datasets only.

From the project root:

    python scripts/build_memory.py run       # 30 Sessions: 10 development datasets x seeds 1-3, 300 s each
    python scripts/build_memory.py status    # Sessions done, records so far, time left
    python scripts/build_memory.py freeze    # copy the finished Sessions' records to the frozen memory

The Sessions' records go to experiments/benchmark-memory/ and models/benchmark-memory/, one row per
finished Session to results/benchmark/memory_sessions.csv, and the frozen memory to
results/benchmark/frozen_memory.jsonl (never rewritten) with a summary per arm. A restart skips finished
Sessions. The benchmark then runs with --memory results/benchmark/frozen_memory.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.experiment_memory import ExperimentMemory  # noqa: E402
from research import benchmark, memory_build  # noqa: E402
from scripts.run_benchmark import keep_awake  # noqa: E402


def run(args: argparse.Namespace, where: dict) -> None:
    cells = memory_build.cells(args.datasets, args.seeds, args.seconds)
    todo = benchmark.pending(cells, benchmark.read_cells(where["rows"]))[: args.limit or None]
    print(f"{len(cells)} Sessions, {len(cells) - len(benchmark.pending(cells, benchmark.read_cells(where['rows'])))} "
          f"done, {len(todo)} to run now; memory {where['memory']}", flush=True)
    keep_awake()
    memory_build.build(todo, where["settings"], where["rows"], where["memory"], n_jobs=args.cores, report=report)


def report(row: dict, number: int, total: int) -> None:
    score = "n/a" if row["test_score"] is None else f"{row['test_score']:.4f}"
    outcome = row["status"] if row["status"] == "completed" else f"FAILED ({str(row['error'])[:150]})"
    print(f"[{number}/{total}] {row['dataset']} seed {row['seed']}: {outcome}, test {row['metric']} {score}, "
          f"{row['n_experiments']} Experiment(s), {row['wall_seconds']:.0f} s, kept: {row['kept'] or '-'}; "
          f"{row['memory_records']} records before it", flush=True)


def status(args: argparse.Namespace, where: dict) -> None:
    cells = memory_build.cells(args.datasets, args.seeds, args.seconds)
    rows = benchmark.read_cells(where["rows"])
    todo = benchmark.pending(cells, rows)
    failed = [r for r in benchmark.latest(rows).values() if r["status"] != "completed"]
    records = ExperimentMemory(where["memory"], read_only=True).records()
    print(f"{len(cells) - len(todo)} of {len(cells)} Sessions done ({len(failed)} failed), {len(records)} records, "
          f"about {benchmark.estimate_seconds(todo, rows) / 3600:.1f} h left (estimate)")
    for row in failed:
        print(f"  failed: {row['cell_id']}: {str(row['error'])[:200]}")


def freeze(args: argparse.Namespace, where: dict) -> None:
    cells = memory_build.cells(args.datasets, args.seeds, args.seconds)
    rows = benchmark.read_cells(where["rows"])
    if todo := benchmark.pending(cells, rows):
        sys.exit(f"Not frozen: {len(todo)} Sessions have not run yet (run 'run' first).")
    result = memory_build.freeze(where["memory"], rows, where["frozen"])
    memory_build.summary(where["frozen"]).to_csv(where["summary"], index=False)
    print(f"Frozen {result['records']} records of {result['sessions']} Sessions ({result['left_out']} records of "
          f"unfinished Sessions left out) to {where['frozen']}\nSHA-256 {result['sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["run", "status", "freeze"])
    parser.add_argument("--datasets", nargs="+", help="default: the 10 development datasets")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(memory_build.SEEDS))
    parser.add_argument("--seconds", type=int, default=memory_build.SECONDS)
    parser.add_argument("--cores", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0, help="run at most this many new Sessions")
    parser.add_argument("--project-root", default=str(ROOT), help="folder for the records and results/")
    args = parser.parse_args()
    where = memory_build.folders(Path(args.project_root))
    {"run": run, "status": status, "freeze": freeze}[args.action](args, where)


if __name__ == "__main__":
    main()
