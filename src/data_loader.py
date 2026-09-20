"""Reading and validating the churn data.

Two different things get read here. The raw event log, which is large and
stays out of version control, and the engineered feature table the app
actually uses.

Deliberately free of any Streamlit import, so every function is testable.
"""

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "churn"

# The only columns the feature extractor touches. Reading just these keeps
# 17.5 million rows in memory, and keeps names, locations and user agents
# out of everything downstream.
EVENT_COLUMNS = [
    "userId",
    "ts",
    "time",
    "registration",
    "gender",
    "level",
    "page",
    "sessionId",
]

READERS = {".parquet": pd.read_parquet, ".csv": pd.read_csv}

GENDER_LABELS = {1: "Male", 0: "Female"}
LEVEL_LABELS = {1: "Paid", 0: "Free"}


def read_table(path, columns=None) -> pd.DataFrame:
    """Read a table, picking the reader from the file extension.

    Parquet carries its own schema, so types survive the round trip. CSV is
    supported too, mostly so tests can write small fixtures cheaply.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No data file at {path}")
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Expected one of {sorted(READERS)}"
        )
    if columns is not None and path.suffix.lower() == ".parquet":
        return reader(path, columns=columns)
    frame = reader(path)
    return frame[columns] if columns is not None else frame


def validate_columns(frame: pd.DataFrame, required) -> None:
    """Raise if any required column is missing, naming what was found."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Found: {list(frame.columns)}"
        )


def load_event_log(path, columns=None) -> pd.DataFrame:
    """Read the raw event log and check it has what the extractor needs."""
    columns = list(EVENT_COLUMNS if columns is None else columns)
    frame = read_table(path, columns=columns)
    validate_columns(frame, columns)
    return frame


def coerce_label(series: pd.Series) -> pd.Series:
    """Turn a 0/1 label into booleans, rejecting anything else."""
    if series.dtype == bool:
        return series
    unexpected = set(pd.unique(series.dropna())) - {0, 1}
    if unexpected:
        raise ValueError(
            f"Expected a 0/1 label, found: {sorted(str(v) for v in unexpected)}"
        )
    return series.astype(bool)


def add_readable_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn the 0/1 profile features into words, for the UI.

    The extractor encodes gender and level as integers because the model
    needs numbers. A filter dropdown reading Male and Paid is far clearer
    than one reading 1 and 1.
    """
    frame = frame.copy()
    if "gender" in frame.columns:
        frame["gender_label"] = frame["gender"].map(GENDER_LABELS)
    if "level" in frame.columns:
        frame["level_label"] = frame["level"].map(LEVEL_LABELS)
    return frame


def load_feature_table(path, target_column: str = "label") -> pd.DataFrame:
    """Read the engineered per-user table the app runs on.

    The extractor writes the target as 'label'; everything downstream reads
    'churn', so it is renamed once here rather than in twenty places.
    """
    frame = read_table(path)
    validate_columns(frame, [target_column])
    if target_column != TARGET_COLUMN:
        frame = frame.rename(columns={target_column: TARGET_COLUMN})
    frame[TARGET_COLUMN] = coerce_label(frame[TARGET_COLUMN])
    return add_readable_labels(frame)