"""Compare two experiment setups over several random splits, paired by seed.

One run on one split is not enough to tell whether a change really helps, because the
score depends on which rows end up in the test set. This script runs setup A and setup B
on the same splits (same seed) and reports the per-seed differences.

Example, from the project root:

    python scripts/compare_seeds.py --dataset data/sample/titanic.csv --target Survived ^
        --exclude-a PassengerId,Name --exclude-b PassengerId,Name,Ticket,Cabin ^
        --seeds 1 2 3 4 5 --budget 30

Runs go to a temporary folder with MLflow switched off, so the dashboard history is not touched.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.core.config import build_settings  # noqa: E402
from app.services.automl_runner import AutoMLRunner  # noqa: E402


def parse_columns(value: str) -> list[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="data/sample/titanic.csv")
    parser.add_argument("--target", default="Survived")
    parser.add_argument("--task", default=None, help="binary, multiclass or regression (default: detect)")
    parser.add_argument("--metric", default=None, help="primary metric (default: task default)")
    parser.add_argument("--exclude-a", default="PassengerId,Name", help="columns excluded in setup A")
    parser.add_argument("--exclude-b", default="PassengerId,Name,Ticket,Cabin", help="columns excluded in setup B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--budget", type=int, default=30, help="FLAML time budget per run, in seconds")
    args = parser.parse_args()

    df = pd.read_csv(ROOT / args.dataset if not Path(args.dataset).is_absolute() else args.dataset)
    settings = build_settings(Path(tempfile.mkdtemp(prefix="autopilot_seeds_")))
    settings.ensure_directories()
    runner = AutoMLRunner(settings=settings, enable_mlflow=False)
    setups = {"A": parse_columns(args.exclude_a), "B": parse_columns(args.exclude_b)}

    rows = []
    for seed in args.seeds:
        for name, excluded in setups.items():
            result = runner.run(
                df, args.target, task=args.task, primary_metric=args.metric, time_budget=args.budget,
                random_seed=seed, excluded_columns=excluded, dataset_name=Path(args.dataset).stem,
            )
            rows.append(
                {
                    "seed": seed,
                    "setup": name,
                    "test": result.test_score,
                    "train": result.train_score,
                    "validation": result.best_validation_score,
                    "trials": result.n_trials,
                }
            )
            print(f"seed {seed} setup {name}: test {result.primary_metric} = {result.test_score:.4f}, "
                  f"train = {result.train_score:.4f}, trials = {result.n_trials}", flush=True)

    results = pd.DataFrame(rows)
    wide = results.pivot(index="seed", columns="setup")
    diff = pd.DataFrame({col: wide[(col, "B")] - wide[(col, "A")] for col in ["test", "train", "validation", "trials"]})

    print(f"\nSetup A excludes: {setups['A']}\nSetup B excludes: {setups['B']}")
    print("\nMean per setup:")
    print(results.groupby("setup")[["test", "train", "validation", "trials"]].mean().round(4).to_string())
    print("\nDifference B - A per seed:")
    print(diff.round(4).to_string())
    print(f"\nMean test difference: {diff['test'].mean():+.4f}")
    print(f"B better on test in {(diff['test'] > 0).sum()} of {len(diff)} seeds")
    print(f"B ran more trials in {(diff['trials'] > 0).sum()} of {len(diff)} seeds")


if __name__ == "__main__":
    main()
