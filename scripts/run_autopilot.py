"""Run one autonomous Session on a CSV file and print every step.

Example, from the project root:

    python scripts/run_autopilot.py --dataset data/sample/titanic.csv --target Survived ^
        --seconds 300 --max-experiments 8 --cores -1 --seed 42

By default the records go to the project's experiments/, models/ and mlruns/ folders, so they
are kept and appear in the dashboard. Use --project-root with a temporary folder for a trial run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.core.config import build_settings  # noqa: E402
from app.services.automl_runner import ExperimentConfig, ExperimentError  # noqa: E402
from app.services.autopilot import AutopilotSession, SessionSettings  # noqa: E402
from app.services.resource_manager import SessionBudget  # noqa: E402
from app.services.task_detector import TaskType  # noqa: E402


def parse_columns(value: str) -> list[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="data/sample/titanic.csv", help="CSV file")
    parser.add_argument("--target", default="Survived")
    parser.add_argument("--task", default=None, help="binary, multiclass or regression (default: detect)")
    parser.add_argument("--metric", default=None, help="primary metric (default: task default)")
    parser.add_argument("--exclude", default="", help="columns to leave out, separated by commas")
    parser.add_argument("--seconds", type=float, default=300, help="total time budget of the Session")
    parser.add_argument("--max-experiments", type=int, default=8)
    parser.add_argument("--cores", type=int, default=-1, help="CPU cores for every search and refit (-1: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selector", default="rules", choices=["rules", "random", "thompson", "ucb1"],
                        help="how the next change is chosen (thompson and ucb1 learn from the memory)")
    parser.add_argument("--use-memory", action="store_true", help="let the selector read the experiment memory")
    parser.add_argument("--no-critic", action="store_true", help="ablation: random valid changes, no Diagnoses")
    parser.add_argument("--no-resource-manager", action="store_true",
                        help="ablation: equal search budgets; only the time and Experiment limits stop")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    parser.add_argument("--project-root", default=str(ROOT), help="folder for experiments/, models/ and mlruns/")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    df = pd.read_csv(dataset if dataset.is_absolute() else ROOT / dataset)
    settings = build_settings(Path(args.project_root))
    settings.ensure_directories()
    try:
        session = SessionSettings(
            base=ExperimentConfig(
                target_column=args.target, task_type=TaskType(args.task) if args.task else None,
                primary_metric=args.metric, random_seed=args.seed, test_size=args.test_size,
                excluded_columns=parse_columns(args.exclude), dataset_name=dataset.stem,
            ),
            budget=SessionBudget(total_seconds=args.seconds, max_experiments=args.max_experiments, n_jobs=args.cores),
            selector=args.selector, use_memory=args.use_memory, use_critic=not args.no_critic,
            use_resource_manager=not args.no_resource_manager,
        )
        result = AutopilotSession(settings=settings, enable_mlflow=not args.no_mlflow).run(
            df, session, progress=lambda stage, step, message: print(f"[{stage}] {message}", flush=True))
    except (ExperimentError, ValueError) as exc:
        print(f"Session not started: {exc}", file=sys.stderr)
        return 2

    print(f"\n{result.session_id}: {result.status}, stopped because: {result.stop_reason.value}, "
          f"{result.total_seconds:.1f} s in total")
    if result.status != "completed":
        print(f"Error: {result.error}")
        return 1
    print("\nStep  Experiment  Change                                      Verdict        Mean diff  p(better)  Validation")
    for step in result.steps:
        change = step.proposal["reason"].split(". Answers:")[0] if step.proposal else "first Setup"
        decision = step.decision or {}
        verdict = "first" if step.index == 0 else (decision.get("verdict") or step.status)
        print(f"{step.index:>4}  {step.experiment_id or '-':<10}  {change[:42]:<42}  {verdict:<13}  "
              f"{fmt(decision.get('mean_difference')):>9}  {fmt(decision.get('p_better'), 3):>9}  "
              f"{fmt((step.assessment or {}).get('validation_mean'))}")
    final = result.final_evaluation
    metric = result.settings["base"]["primary_metric"]
    print(f"\nFinal_Evaluation on {final['n_test']} test rows ({metric}):")
    print(f"  first model {final['first']['experiment_id']}: {fmt(final['first']['test_score'])}")
    print(f"  final model {final['final']['experiment_id']}: {fmt(final['final']['test_score'])}")
    print(f"\nRecord: {settings.experiments_dir / result.session_id / 'session.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
