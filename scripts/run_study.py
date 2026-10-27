"""Run a component study and analyse it.

Critic study (spec task 16), from the project root:

    python scripts/run_study.py critic run           # all cases; restarts skip finished cases
    python scripts/run_study.py critic analyse       # tune on the tuning seeds, score the reporting seeds
    python scripts/run_study.py critic analyse --freeze v1-critic-study   # also write the thresholds file

Raw results: results/studies/critic_study_cases.csv (one row per dataset, written as each one
finishes) and experiments/studies/critic_study_inputs.jsonl (the saved Critic inputs, for replay).

Judge study (spec task 17):

    python scripts/run_study.py judge run            # all pairs; restarts skip finished pairs
    python scripts/run_study.py judge analyse        # rates per (repeats, alpha), the chosen defaults

Raw results: results/studies/judge_study_pairs.csv (one row per pair, number of repeats and alpha).

Judge agreement study (spec task 27; needs the memory build's Sessions, scripts/build_memory.py):

    python scripts/run_study.py agreement run        # the strict comparison of each pair; restarts skip done pairs
    python scripts/run_study.py agreement analyse    # agreement, Cohen's kappa and the cost ratio

Raw results: results/studies/judge_agreement_pairs.csv (one row per pair); the searches' records in
experiments/studies/judge_agreement/ and models/studies/judge_agreement/.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.services.critic import DEFAULT_THRESHOLDS_PATH, CriticThresholds  # noqa: E402
from research import critic_study as cs  # noqa: E402
from research import judge_agreement as ja  # noqa: E402
from research import judge_study as js  # noqa: E402

RESULTS = ROOT / "results" / "studies"
INPUTS = ROOT / "experiments" / "studies"


def critic_run(args: argparse.Namespace) -> None:
    cases_path, inputs_path = RESULTS / "critic_study_cases.csv", INPUTS / "critic_study_inputs.jsonl"
    RESULTS.mkdir(parents=True, exist_ok=True)
    INPUTS.mkdir(parents=True, exist_ok=True)
    done = {row["case_id"] for row in cs.read_rows(cases_path)}
    seeds = {"tuning": cs.TUNING_SEEDS, "reporting": cs.REPORTING_SEEDS,
             "all": (*cs.TUNING_SEEDS, *cs.REPORTING_SEEDS)}[args.seeds]
    cases = [c for c in cs.study_cases(seeds) if c.case_id not in done][: args.limit or None]
    print(f"{len(done)} cases already done, {len(cases)} to run", flush=True)
    started = time.perf_counter()
    for number, case in enumerate(cases, start=1):
        with tempfile.TemporaryDirectory(prefix="critic_study_") as work_dir:
            row, inputs = cs.run_case(case, work_dir=Path(work_dir), n_jobs=args.cores)
        frame = pd.DataFrame([row])
        frame.to_csv(cases_path, mode="a", header=not cases_path.exists(), index=False)
        with inputs_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(inputs) + "\n")
        elapsed = time.perf_counter() - started
        print(f"[{number}/{len(cases)}] {case.case_id}: {row['best_estimator']}, fresh F1 {row['fresh_f1']:.3f}, "
              f"{row['seconds']:.1f} s ({elapsed / 60:.1f} min so far)", flush=True)


def critic_analyse(args: argparse.Namespace) -> None:
    rows = cs.read_rows(RESULTS / "critic_study_cases.csv")
    if not rows:
        sys.exit("No results yet: run 'critic run' first.")
    start = CriticThresholds.from_mapping({"version": "v0-start", **_tables(cs.V0_START)})
    result = cs.analyse(rows, start)
    pd.DataFrame(result["grid"]).to_csv(RESULTS / "critic_study_grid.csv", index=False)
    summary = pd.DataFrame(result["summary"])
    summary.to_csv(RESULTS / "critic_study_summary.csv", index=False)
    print(f"{len(rows)} cases: {sum(r['split'] == 'tuning' for r in rows)} tuning, "
          f"{sum(r['split'] == 'reporting' for r in rows)} reporting")
    print("\nTuned thresholds (changed ones):")
    for key, value in result["tuned"].items():
        if value != start.values[key]:
            print(f"  {key}: {start.values[key]:g} -> {value:g}")
    table = summary[summary["split"] == "reporting"].pivot(index="kind", columns="thresholds",
                                                           values=["precision", "recall", "tp", "fp", "fn"])
    print("\nReporting seeds (not used for tuning):")
    print(table.round(3).to_string())

    # Replay: the real Critic with the tuned thresholds must agree with the study's rule on every case.
    inputs = cs.read_inputs(INPUTS / "critic_study_inputs.jsonl")
    tuned = CriticThresholds.from_mapping({"version": "check", **_tables(result["tuned"])})
    mismatches = 0
    for row in rows:
        if row["case_id"] not in inputs or row.get("assessment_failed"):
            continue
        found = cs.replay(inputs[row["case_id"]], tuned)
        expected = {kind for kind in cs.STUDIED_KINDS if cs.predict(row, kind, tuned.values)}
        mismatches += found != expected
    print(f"\nReplay of the real Critic on {len(inputs)} saved cases: {mismatches} disagreements with the study's rule")
    if args.freeze:
        note = (f"Version {args.freeze}: tuned by the critic study (task 16) on the tuning seeds "
                f"{list(cs.TUNING_SEEDS)};\nprecision and recall on the reporting seeds {list(cs.REPORTING_SEEDS)} "
                "are in results/studies/critic_study_summary.csv.")
        DEFAULT_THRESHOLDS_PATH.write_text(cs.thresholds_toml(result["tuned"], args.freeze, note), encoding="utf-8")
        CriticThresholds.load()  # the written file must load
        print(f"Wrote {DEFAULT_THRESHOLDS_PATH} with version {args.freeze}")


def judge_run(args: argparse.Namespace) -> None:
    pairs_path = RESULTS / "judge_study_pairs.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = {row["pair_id"] for row in cs.read_rows(pairs_path)}
    seeds = {"tuning": js.TUNING_SEEDS, "reporting": js.REPORTING_SEEDS,
             "all": (*js.TUNING_SEEDS, *js.REPORTING_SEEDS)}[args.seeds]
    pairs = [p for p in js.study_pairs(seeds) if p.pair_id not in done][: args.limit or None]
    print(f"{len(done)} pairs already done, {len(pairs)} to run", flush=True)
    started = time.perf_counter()
    for number, pair in enumerate(pairs, start=1):
        with tempfile.TemporaryDirectory(prefix="judge_study_") as work_dir:
            rows = js.run_pair(pair, work_dir=Path(work_dir), n_jobs=args.cores)
        pd.DataFrame(rows).to_csv(pairs_path, mode="a", header=not pairs_path.exists(), index=False)
        default = next(r for r in rows if r["repeats"] == 3 and r["alpha"] == 0.10)
        elapsed = time.perf_counter() - started
        print(f"[{number}/{len(pairs)}] {pair.pair_id}: true difference {default['true_difference']:+.4f}, "
              f"verdict {default.get('verdict')} ({rows[0]['seconds']:.0f} s, {elapsed / 60:.1f} min so far)",
              flush=True)


def judge_analyse(args: argparse.Namespace) -> None:
    rows = cs.read_rows(RESULTS / "judge_study_pairs.csv")
    if not rows:
        sys.exit("No results yet: run 'judge run' first.")
    result = js.summary(rows)
    table = pd.DataFrame([{**r, "split": "tuning"} for r in result["tuning"]]
                         + [{**r, "split": "reporting"} for r in result["reporting"]])
    table.to_csv(RESULTS / "judge_study_summary.csv", index=False)
    pairs = {r["pair_id"]: r for r in rows}
    print(f"{len(pairs)} pairs: {sum(r['split'] == 'tuning' for r in pairs.values())} tuning, "
          f"{sum(r['split'] == 'reporting' for r in pairs.values())} reporting")
    print(table.round(3).to_string(index=False))
    chosen, reported = result["chosen"], result["chosen_on_reporting"]
    print(f"\nChosen on the tuning seeds: repeats {chosen['repeats']}, alpha {chosen['alpha']} "
          f"(false-keep rate {chosen['false_keep_rate']}, power {chosen['power']})")
    print(f"Same setting on the reporting seeds: false-keep rate {reported['false_keep_rate']}, "
          f"power {reported['power']} ({reported['harmful']} harmful and {reported['helpful']} helpful pairs)")


def agreement_run(args: argparse.Namespace) -> None:
    from scripts.run_benchmark import keep_awake

    where = ja.folders(ROOT)
    pairs = ja.study_pairs(ROOT)
    done = {r["pair_id"] for r in ja.read_rows(where["pairs"])}
    print(f"{len(pairs)} pairs, {sum(p.pair_id in done for p in pairs)} already done", flush=True)
    keep_awake()
    started = time.perf_counter()

    def report(row: dict, number: int, total: int) -> None:
        elapsed = time.perf_counter() - started
        outcome = (f"Judge {row['judge_verdict']}, strict {row['strict_verdict']}, {row['strict_seconds']:.0f} s"
                   if row["status"] == "completed" else f"FAILED ({str(row['error'])[:150]})")
        print(f"[{number}/{total}] {row['pair_id']} ({row['dataset']}, {row['strategy']}): {outcome} "
              f"({elapsed / 60:.1f} min so far)", flush=True)

    ja.run(pairs, where["settings"], where["pairs"], n_jobs=args.cores, limit=args.limit, report=report)


def agreement_analyse(args: argparse.Namespace) -> None:
    where = ja.folders(ROOT)
    rows = ja.read_rows(where["pairs"])
    if not rows:
        sys.exit("No results yet: run 'agreement run' first.")
    groups = [("all", rows)] + [(f"dataset={d}", [r for r in rows if r["dataset"] == d])
                                for d in sorted({r["dataset"] for r in rows})]
    groups += [(f"judge={v}", [r for r in rows if r["judge_verdict"] == v]) for v in ja.VERDICTS]
    table = pd.DataFrame([{"group": name, **{k: v for k, v in ja.agreement(part).items() if k != "confusion"}}
                          for name, part in groups if part])
    table.to_csv(where["summary"], index=False)
    confusion = ja.agreement(rows)["confusion"]
    confusion.to_csv(where["confusion"])
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    print("\nRows: Judge's verdict; columns: strict verdict\n" + confusion.to_string())


def _tables(values: dict[str, float]) -> dict[str, dict[str, float]]:
    tables: dict[str, dict[str, float]] = {}
    for key, value in values.items():
        section, name = key.split(".", 1)
        tables.setdefault(section, {})[name] = value
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", choices=["critic", "judge", "agreement"])
    parser.add_argument("action", choices=["run", "analyse"])
    parser.add_argument("--seeds", choices=["tuning", "reporting", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="run at most this many new cases")
    parser.add_argument("--cores", type=int, default=-1)
    parser.add_argument("--freeze", default="", help="write the tuned thresholds under this version label")
    args = parser.parse_args()
    actions = {("critic", "run"): critic_run, ("critic", "analyse"): critic_analyse,
               ("judge", "run"): judge_run, ("judge", "analyse"): judge_analyse,
               ("agreement", "run"): agreement_run, ("agreement", "analyse"): agreement_analyse}
    actions[(args.study, args.action)](args)


if __name__ == "__main__":
    main()
