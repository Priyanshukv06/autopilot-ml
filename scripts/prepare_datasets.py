"""Choose, download and check the development and evaluation datasets (spec task 20).

From the project root:

    python scripts/prepare_datasets.py select     # pool -> draw -> download -> checks -> research/datasets.json
    python scripts/prepare_datasets.py download   # download the files listed in datasets.json and check them

``select`` is run once, before any evaluation run; its pool is saved in results/datasets/openml_pool.csv.
The CSV files go to data/raw/ (not in git); ``download`` recreates them from the OpenML IDs and versions.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from research import datasets as ds  # noqa: E402


def select() -> None:
    pool = ds.fetch_pool()
    results = ROOT / "results" / "datasets"
    results.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{**asdict(c), "task": c.task, "eligible": ds.eligible(c), "group": ds.group_of(c)}
                          for c in pool])
    frame.to_csv(results / "openml_pool.csv", index=False)
    development, evaluation = ds.select(pool)
    print(f"Pool: {len(pool)} datasets, {int(frame['eligible'].sum())} laptop-sized and not excluded")
    entries = ds.sample_entries()
    for split, chosen in (("development", development), ("evaluation", evaluation)):
        for candidate in chosen:
            data, details = ds.download(candidate)
            entries.append(ds.entry_for(candidate, split, data, details))
            print(f"{split:<12} {ds.group_of(candidate):<18} {candidate.openml_id:>6} {candidate.name}")
    ds.write_registry(entries)
    print(f"Wrote {ds.REGISTRY_PATH}")


def download() -> None:
    for entry in ds.registry().values():
        if entry.source == "sample":
            continue
        candidate = ds.Candidate(entry.openml_id, entry.name, entry.version, entry.source, entry.rows,
                                 entry.columns, 0, 0, 0, None)
        frame, details = ds.download(candidate)
        stats = ds.describe(frame, entry.target, entry.task)
        same = int(details["version"]) == entry.version and all(
            stats[key] == getattr(entry, key) for key in ("rows", "columns", "missing_share", "categorical_columns"))
        print(f"{entry.name}: {'ok' if same else 'DIFFERENT FROM datasets.json'} {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["select", "download"])
    args = parser.parse_args()
    select() if args.action == "select" else download()


if __name__ == "__main__":
    main()
