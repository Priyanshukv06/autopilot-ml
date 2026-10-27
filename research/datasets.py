"""Dataset registry (spec task 20, Requirement 22): the development and evaluation datasets.

The two lists are fixed in ``research/datasets.json``, committed before any evaluation run. Development
datasets build the Experiment_Memory and may be looked at freely; evaluation datasets are only used by
the Phase 3 benchmark. No dataset is in both lists.

How the OpenML datasets were chosen (fixed before downloading any of them):

1. Pool: every dataset of two curated OpenML suites, OpenML-CC18 (classification, study 99) and
   OpenML-CTR23 (regression, study 353), with 500 to 10,000 rows and at most 100 feature columns
   (laptop-sized for budgets of 2 and 5 minutes). Left out: the two versions of the Wisconsin breast
   cancer data (the ``breast_cancer`` sample is one of them).
2. Each candidate falls into at most one of six groups, two per task type, so that both lists cover
   imbalance, missing values and categorical columns (Requirement 22.2):
   binary imbalanced (largest class at least 3 times the smallest); binary with missing values or
   categorical columns (and balanced); multiclass without missing values or categorical columns and
   balanced; multiclass with missing values, categorical columns or imbalance; regression without
   missing values or categorical columns; regression with missing values or categorical columns.
   Balanced binary datasets with only numeric columns fall into no group.
3. In each group the candidates are sorted by OpenML ID and shuffled with one random generator seeded
   with SEED, groups in the order above. The first becomes an evaluation dataset, the second a
   development dataset. With the four samples, that gives 10 development and 6 evaluation datasets.

Each OpenML dataset is downloaded once with scikit-learn's ``fetch_openml`` (its cache is in
``data/raw/openml_cache``) and saved as ``data/raw/<name>.csv``, so the rest of the system reads it like
any uploaded CSV file (codes of categorical columns that look like numbers become numbers).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from app.services.task_detector import TaskDetector

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).with_name("datasets.json")
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "sample"
API = "https://www.openml.org/api/v1/json"

SUITES = {99: "OpenML-CC18", 353: "OpenML-CTR23"}
MIN_ROWS, MAX_ROWS, MAX_FEATURES = 500, 10_000, 100
IMBALANCED = 3.0
SEED = 2026
EXCLUDED = {
    15: "breast-w: Wisconsin breast cancer data, the source of the breast_cancer sample",
    1510: "wdbc: the same data as the breast_cancer sample",
}
GROUPS = (
    "binary_imbalanced",
    "binary_messy",
    "multiclass_clean",
    "multiclass_messy",
    "regression_clean",
    "regression_messy",
)
SAMPLES = (  # the four bundled datasets, all development
    ("titanic", "titanic.csv", "Survived", "binary"),
    ("breast_cancer", "breast_cancer.csv", "diagnosis", "binary"),
    ("penguins", "penguins.csv", "species", "multiclass"),
    ("california_housing", "california_housing.csv", "MedHouseVal", "regression"),
)

Task = Literal["binary", "multiclass", "regression"]


@dataclass(frozen=True)
class Candidate:
    """One suite dataset with the OpenML qualities the selection uses."""

    openml_id: int
    name: str
    version: int
    suite: str
    rows: int
    features: int  # feature columns, without the target
    classes: int  # 0 for regression
    missing_values: int
    categorical_features: int  # nominal feature columns, without the target
    imbalance_ratio: float | None  # largest / smallest class; None for regression

    @property
    def task(self) -> Task:
        return "regression" if self.classes == 0 else "binary" if self.classes == 2 else "multiclass"


@dataclass(frozen=True)
class DatasetEntry:
    """One dataset of ``datasets.json``."""

    name: str
    split: Literal["development", "evaluation"]
    source: str  # "sample" or the suite name
    openml_id: int | None
    version: int | None
    target: str
    task: Task  # OpenML's task; the benchmark passes it to every system
    detected_task: Task  # what the app's task detector would choose from the CSV file
    licence: str
    file: str  # relative to the project root
    group: str | None
    url: str | None
    rows: int
    columns: int  # without the target
    missing_share: float  # share of empty feature cells
    categorical_columns: int  # text or category feature columns in the CSV file
    imbalance_ratio: float | None


# --------------------------------------------------------------------------- selection
def group_of(c: Candidate) -> str | None:
    """The selection group of a candidate, or None (see the module docstring)."""
    messy = c.missing_values > 0 or c.categorical_features > 0
    imbalanced = c.imbalance_ratio is not None and c.imbalance_ratio >= IMBALANCED
    if c.task == "binary":
        return "binary_imbalanced" if imbalanced else ("binary_messy" if messy else None)
    if c.task == "multiclass":
        return "multiclass_messy" if messy or imbalanced else "multiclass_clean"
    return "regression_messy" if messy else "regression_clean"


def eligible(c: Candidate) -> bool:
    return MIN_ROWS <= c.rows <= MAX_ROWS and c.features <= MAX_FEATURES and c.openml_id not in EXCLUDED


def select(pool: Iterable[Candidate], seed: int = SEED) -> tuple[list[Candidate], list[Candidate]]:
    """(development, evaluation): one of each per group, drawn as described in the module docstring."""
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[Candidate]] = {g: [] for g in GROUPS}
    for c in pool:
        group = group_of(c)
        if eligible(c) and group is not None:
            by_group[group].append(c)
    development, evaluation = [], []
    for group in GROUPS:
        candidates = sorted(by_group[group], key=lambda c: c.openml_id)
        if len(candidates) < 2:
            raise ValueError(f"Group {group} has {len(candidates)} candidates; 2 are needed.")
        order = rng.permutation(len(candidates))
        evaluation.append(candidates[order[0]])
        development.append(candidates[order[1]])
    return development, evaluation


def fetch_pool() -> list[Candidate]:
    """Every dataset of the two suites with its OpenML qualities (needs the internet)."""
    pool = []
    for study, suite in SUITES.items():
        ids = _get(f"study/{study}")["study"]["data"]["data_id"]
        for entry in _get("data/list/data_id/" + ",".join(map(str, ids)))["data"]["dataset"]:
            q = {item["name"]: item["value"] for item in entry.get("quality", [])}
            classes = int(float(q.get("NumberOfClasses") or 0))
            symbolic = int(float(q.get("NumberOfSymbolicFeatures") or 0))
            majority, minority = float(q.get("MajorityClassSize") or 0), float(q.get("MinorityClassSize") or 0)
            pool.append(Candidate(
                openml_id=int(entry["did"]), name=entry["name"], version=int(entry["version"]), suite=suite,
                rows=int(float(q["NumberOfInstances"])), features=int(float(q["NumberOfFeatures"])) - 1,
                classes=classes, missing_values=int(float(q.get("NumberOfMissingValues") or 0)),
                categorical_features=symbolic - (1 if classes else 0),
                imbalance_ratio=majority / minority if classes and minority > 0 else None,
            ))
    return sorted(pool, key=lambda c: c.openml_id)


# --------------------------------------------------------------------------- download and checks
def download(c: Candidate, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download one dataset (once; scikit-learn caches it) and save it as ``<raw_dir>/<name>.csv``."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=c.openml_id, as_frame=True, data_home=str(raw_dir / "openml_cache"))
    details = dict(bunch.details)
    target = details["default_target_attribute"]
    if "," in target:
        raise ValueError(f"{c.name} has several default targets: {target}")
    frame = bunch.frame
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_dir / f"{c.name}.csv", index=False)
    return pd.read_csv(raw_dir / f"{c.name}.csv"), details


def describe(frame: pd.DataFrame, target: str, task: Task) -> dict[str, Any]:
    """Size and properties of a dataset as the app reads it (from its CSV file), for the given task;
    ``detected_task`` is what the app's task detector would choose by itself."""
    features = frame.drop(columns=[target])
    counts = frame[target].dropna().value_counts()
    return {
        "detected_task": TaskDetector().detect(frame[target]).task_type.value,
        "rows": int(len(frame)), "columns": int(features.shape[1]),
        "missing_share": round(float(features.isna().to_numpy().mean()), 4),
        "categorical_columns": int(sum(not pd.api.types.is_numeric_dtype(features[c]) for c in features)),
        "imbalance_ratio": round(float(counts.max() / counts.min()), 3) if task != "regression" else None,
    }


def entry_for(c: Candidate, split: str, frame: pd.DataFrame, details: dict[str, Any]) -> DatasetEntry:
    target = details["default_target_attribute"]
    return DatasetEntry(
        name=c.name, split=split, source=c.suite, openml_id=c.openml_id, version=int(details["version"]),
        target=target, task=c.task, licence=str(details.get("licence") or "not given"),
        file=f"data/raw/{c.name}.csv", group=group_of(c), url=f"https://www.openml.org/d/{c.openml_id}",
        **describe(frame, target, c.task),
    )


def sample_entries() -> list[DatasetEntry]:
    entries = []
    for name, file_name, target, task in SAMPLES:
        entries.append(DatasetEntry(
            name=name, split="development", source="sample", openml_id=None, version=None, target=target,
            task=task, licence="see README.md (Sample datasets)", file=f"data/sample/{file_name}", group=None,
            url=None, **describe(pd.read_csv(SAMPLE_DIR / file_name), target, task),
        ))
    return entries


# --------------------------------------------------------------------------- the registry
def write_registry(entries: list[DatasetEntry], path: Path = REGISTRY_PATH) -> None:
    payload = {
        "description": "Development and evaluation datasets of AutoPilot-ML (spec task 20). Fixed before any "
                       "evaluation run; how they were chosen is in research/datasets.py.",
        "selection": {"suites": {str(k): v for k, v in SUITES.items()}, "rows": [MIN_ROWS, MAX_ROWS],
                      "max_features": MAX_FEATURES, "imbalanced_ratio": IMBALANCED, "seed": SEED,
                      "groups": list(GROUPS), "excluded": {str(k): v for k, v in EXCLUDED.items()}},
        "datasets": [asdict(e) for e in entries],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def registry(path: Path = REGISTRY_PATH) -> dict[str, DatasetEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {d["name"]: DatasetEntry(**d) for d in data["datasets"]}


def datasets(split: str, path: Path = REGISTRY_PATH) -> list[DatasetEntry]:
    return [e for e in registry(path).values() if e.split == split]


def load(entry: DatasetEntry, root: Path = ROOT) -> pd.DataFrame:
    """The dataset as a frame, read from its CSV file (run ``prepare_datasets.py download`` first)."""
    path = root / entry.file
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run python scripts/prepare_datasets.py download")
    return pd.read_csv(path)


def _get(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{API}/{path}", timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))
