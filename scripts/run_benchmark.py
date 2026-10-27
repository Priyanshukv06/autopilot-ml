"""Phase 3 benchmark (spec task 24): every system on every evaluation dataset, budget and seed.

From the project root:

    python scripts/run_benchmark.py run --memory FILE    # the whole grid; a restart skips finished cells
    python scripts/run_benchmark.py run --smoke          # every system once: titanic, seed 1, 120 s
    python scripts/run_benchmark.py status [--smoke]     # cells done, failures, time per system, time left
    python scripts/run_benchmark.py analyse [--smoke]    # tables and charts in results/benchmark/

FILE is the frozen Experiment_Memory, built from development datasets only. The systems' records go to
experiments/benchmark/ and models/benchmark/ (benchmark-smoke/ for the smoke setting), and one row per
finished cell to results/benchmark/benchmark_cells.csv (results/benchmark/smoke/). Use --project-root with
a temporary folder for a trial. While it runs, the script asks Windows not to sleep; that request ends
with the process.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from research import benchmark  # noqa: E402
from research.datasets import datasets  # noqa: E402


def cells(args: argparse.Namespace) -> list[benchmark.Cell]:
    if args.smoke:
        return benchmark.smoke_grid(args.datasets or [benchmark.SMOKE_DATASET], args.budgets, args.systems,
                                    args.ablations)
    names = args.datasets or [e.name for e in datasets("evaluation")]
    return benchmark.grid(names, args.seeds, args.budgets, args.systems, args.ablations, args.ablation_budget)


def run(args: argparse.Namespace) -> None:
    grid = cells(args)
    memory = Path(args.memory).resolve() if args.memory else None
    problems = benchmark.preflight(grid, memory, require_memory=not args.smoke, evaluation_only=not args.smoke)
    if problems:
        sys.exit("Not started:\n- " + "\n- ".join(problems))
    settings, rows_path = benchmark.folders(Path(args.project_root), smoke=args.smoke)
    todo = benchmark.pending(grid, benchmark.read_cells(rows_path), args.retry_failed)
    print(f"{len(grid)} cells in the grid, {len(grid) - len(todo)} already done, {len(todo)} to run; rows go to "
          f"{rows_path}", flush=True)
    keep_awake()
    benchmark.run(grid, settings, rows_path, memory_path=memory, n_jobs=args.cores, retry_failed=args.retry_failed,
                  limit=args.limit, report=report)


def report(row: dict, number: int, total: int, seconds_left: float) -> None:
    score = "n/a" if row["test_score"] is None else f"{row['test_score']:.4f}"
    outcome = row["status"] if row["status"] == "completed" else f"FAILED ({str(row['error'])[:150]})"
    print(f"[{number}/{total}] {row['cell_id']}: {outcome}, test {row['metric']} {score}, "
          f"{row['n_experiments']} Experiment(s), {row['wall_seconds']:.0f} s; "
          f"about {seconds_left / 3600:.1f} h left (estimate)", flush=True)


def status(args: argparse.Namespace) -> None:
    grid = cells(args)
    _, rows_path = benchmark.folders(Path(args.project_root), smoke=args.smoke)
    rows = benchmark.read_cells(rows_path)
    last = benchmark.latest(rows)
    in_grid = [last[c.cell_id] for c in grid if c.cell_id in last]
    failed = [r for r in in_grid if r["status"] != "completed"]
    todo = benchmark.pending(grid, rows)
    print(f"{rows_path}: {len(in_grid)} of {len(grid)} cells done ({len(failed)} failed), {len(todo)} to run, "
          f"about {benchmark.estimate_seconds(todo, rows) / 3600:.1f} h left (estimate)")
    hashes = sorted({r["memory_hash"] for r in rows if r.get("memory_hash")})
    print(f"memory: {', '.join(h[:12] for h in hashes) or 'none yet'}")
    for row in failed:
        print(f"  failed: {row['cell_id']}: {str(row['error'])[:200]}")
    table = benchmark.timing(in_grid)
    if table:
        pd.set_option("display.width", 200)
        print(pd.DataFrame(table).to_string(index=False))


def analyse(args: argparse.Namespace) -> None:
    from research import benchmark_analysis

    _, rows_path = benchmark.folders(Path(args.project_root), smoke=args.smoke)
    rows = benchmark.read_cells(rows_path)
    if not rows:
        sys.exit(f"No rows in {rows_path} yet: run 'run' first.")
    result = benchmark_analysis.analyse(rows, args.budgets, args.ablation_budget)
    paths = benchmark_analysis.write(result, rows_path.parent)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for name in ("tests", "ranks", "pairs", "time_to_target_summary", "failures"):
        table = result[name]
        print(f"\n{name}:\n" + (table.round(4).to_string(index=False) if len(table) else "(none)"))
    print(f"\n{len(paths)} files written to {rows_path.parent}")


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (SetThreadExecutionState; it ends with the process)."""
    if os.name == "nt":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["run", "status", "analyse"])
    parser.add_argument("--smoke", action="store_true", help="every system once on one small development dataset")
    parser.add_argument("--memory", help="the frozen Experiment_Memory file (needed for the grid)")
    parser.add_argument("--datasets", nargs="+", help="default: the evaluation datasets (smoke: titanic)")
    parser.add_argument("--systems", nargs="*", default=list(benchmark.MAIN_SYSTEMS))
    parser.add_argument("--ablations", nargs="*", default=list(benchmark.ABLATIONS))
    parser.add_argument("--budgets", type=int, nargs="+", default=list(benchmark.BUDGETS))
    parser.add_argument("--ablation-budget", type=int, default=benchmark.ABLATION_BUDGET)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(benchmark.SEEDS))
    parser.add_argument("--cores", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0, help="run at most this many new cells")
    parser.add_argument("--retry-failed", action="store_true", help="run cells whose last row failed again")
    parser.add_argument("--project-root", default=str(ROOT), help="folder for the records and results/")
    args = parser.parse_args()
    {"run": run, "status": status, "analyse": analyse}[args.action](args)


if __name__ == "__main__":
    main()
