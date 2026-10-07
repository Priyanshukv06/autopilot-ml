"""Section 1 (input): CSV upload or bundled sample dataset."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.config import Settings


@dataclass(frozen=True)
class SampleDataset:
    file_name: str
    title: str
    task: str
    target: str


# Small public datasets bundled in data/sample (sources are listed in README.md).
SAMPLE_DATASETS: dict[str, SampleDataset] = {
    "titanic": SampleDataset("titanic.csv", "Titanic passenger survival", "Binary classification", "Survived"),
    "breast_cancer": SampleDataset(
        "breast_cancer.csv", "Breast cancer diagnosis (Wisconsin)", "Binary classification", "diagnosis"
    ),
    "penguins": SampleDataset("penguins.csv", "Palmer penguins species", "Multiclass classification", "species"),
    "california_housing": SampleDataset(
        "california_housing.csv", "California housing prices", "Regression", "MedHouseVal"
    ),
}


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    name: str
    source: str  # "upload" | "sample"
    suggested_target: str | None = None

    @property
    def key(self) -> str:
        """Identifier used to scope widget state to this dataset."""
        return f"{self.name}:{self.df.shape[0]}x{self.df.shape[1]}"


class DatasetLoadError(ValueError):
    """Raised when a file cannot be turned into a usable DataFrame."""


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Parse CSV bytes with encoding and delimiter fallbacks (no Streamlit dependency)."""
    if not data or not data.strip():
        raise DatasetLoadError("The file is empty.")
    df: pd.DataFrame | None = None
    for encoding in ("utf-8-sig", "latin-1"):  # latin-1 accepts any byte sequence
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding)
            if df.shape[1] == 1 and ";" in str(df.columns[0]):
                df = pd.read_csv(io.BytesIO(data), encoding=encoding, sep=";")
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DatasetLoadError(f"Could not parse the CSV: {exc}") from exc
    if df is None:
        raise DatasetLoadError("Unsupported text encoding.")

    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise DatasetLoadError("The CSV has a header but no data rows.")
    if df.shape[1] < 2:
        raise DatasetLoadError("At least 2 columns are required (features and a target).")
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise DatasetLoadError(f"Duplicate column names: {duplicated}")
    return df


@st.cache_data(show_spinner="Reading CSV...")
def _load_csv(data: bytes) -> pd.DataFrame:
    return read_csv_bytes(data)


def render_dataset_uploader(settings: Settings) -> LoadedDataset | None:
    source = st.radio("Data source", ["Upload CSV", "Sample dataset"], horizontal=True, key="data_source")

    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            max_upload_size=settings.max_upload_mb,
            help="A tabular CSV file with a header row. One column will be chosen as the target.",
        )
        if uploaded is None:
            return None
        data, name, suggested, origin = uploaded.getvalue(), Path(uploaded.name).stem, None, "upload"
    else:
        available = {
            key: sample
            for key, sample in SAMPLE_DATASETS.items()
            if (settings.sample_data_dir / sample.file_name).exists()
        }
        if not available:
            st.warning(f"No sample datasets found in {settings.sample_data_dir}.")
            return None
        key = st.selectbox(
            "Sample dataset",
            list(available),
            format_func=lambda k: f"{available[k].title} · {available[k].task}",
            key="sample_choice",
        )
        sample = available[key]
        data = (settings.sample_data_dir / sample.file_name).read_bytes()
        name, suggested, origin = key, sample.target, "sample"

    try:
        df = _load_csv(data)
    except DatasetLoadError as exc:
        st.error(f"Could not load the dataset: {exc}")
        return None
    return LoadedDataset(df=df, name=name, source=origin, suggested_target=suggested)
