"""Filtering and summarising the feature table.

Also free of Streamlit, for the same reason.
"""

import pandas as pd

TARGET_COLUMN = "churn"


def _require_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise KeyError(f"No column named '{column}'. Available: {list(frame.columns)}")


def filter_by_category(frame: pd.DataFrame, column: str, values) -> pd.DataFrame:
    """Keep rows whose column value is in values.

    An empty list means no filter, which is what a cleared multiselect in
    the UI should do.
    """
    _require_column(frame, column)
    values = list(values)
    if not values:
        return frame.copy()
    return frame[frame[column].isin(values)].copy()


def filter_by_numeric_range(
    frame: pd.DataFrame, column: str, minimum=None, maximum=None
) -> pd.DataFrame:
    """Keep rows where column sits between minimum and maximum, both inclusive."""
    _require_column(frame, column)
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"minimum {minimum} is greater than maximum {maximum}")

    mask = pd.Series(True, index=frame.index)
    if minimum is not None:
        mask &= frame[column] >= minimum
    if maximum is not None:
        mask &= frame[column] <= maximum
    return frame[mask].copy()


def churn_rate(frame: pd.DataFrame) -> float:
    """Share of users who churned, between 0 and 1.

    An empty frame returns 0.0 rather than NaN, which keeps the UI from
    displaying nan% when a filter matches nothing.
    """
    _require_column(frame, TARGET_COLUMN)
    if len(frame) == 0:
        return 0.0
    return float(frame[TARGET_COLUMN].mean())


def churn_rate_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate and user count per value of column, worst first."""
    _require_column(frame, column)
    _require_column(frame, TARGET_COLUMN)
    return (
        frame.groupby(column)[TARGET_COLUMN]
        .agg(churn_rate="mean", users="size")
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )