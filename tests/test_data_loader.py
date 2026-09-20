from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    EVENT_COLUMNS,
    add_readable_labels,
    coerce_label,
    load_event_log,
    load_feature_table,
    read_table,
    validate_columns,
)

FEATURES = Path("data/features.parquet")


def test_read_table_reads_parquet(tmp_path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(path)
    assert read_table(path)["a"].tolist() == [1, 2]


def test_read_table_reads_csv(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("a\n1\n2\n")
    assert read_table(path)["a"].tolist() == [1, 2]


def test_read_table_can_take_a_column_subset(tmp_path):
    path = tmp_path / "wide.parquet"
    pd.DataFrame({"a": [1], "b": [2], "c": [3]}).to_parquet(path)
    assert list(read_table(path, columns=["a", "c"]).columns) == ["a", "c"]


def test_read_table_rejects_an_unsupported_extension(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_table(path)


def test_read_table_raises_on_a_missing_file():
    with pytest.raises(FileNotFoundError):
        read_table("data/does_not_exist.parquet")


def test_validate_columns_names_what_is_missing():
    frame = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match=r"Missing required columns \['b'\]"):
        validate_columns(frame, ["a", "b"])


def test_load_event_log_reads_only_the_needed_columns(tmp_path, event_log):
    path = tmp_path / "log.parquet"
    extra = event_log.assign(firstName="Ada", location="Paris")
    extra.to_parquet(path)
    loaded = load_event_log(path)
    assert list(loaded.columns) == EVENT_COLUMNS
    assert "firstName" not in loaded.columns


def test_load_event_log_raises_when_a_needed_column_is_absent(tmp_path, event_log):
    path = tmp_path / "broken.parquet"
    event_log.drop(columns=["page"]).to_parquet(path)
    with pytest.raises((ValueError, KeyError)):
        load_event_log(path)


@pytest.mark.parametrize("raw", [[1, 0, 1], [True, False, True]])
def test_coerce_label_accepts_ints_and_bools(raw):
    assert coerce_label(pd.Series(raw)).tolist() == [True, False, True]


def test_coerce_label_rejects_anything_else():
    with pytest.raises(ValueError, match="Expected a 0/1 label"):
        coerce_label(pd.Series([0, 1, 7]))


def test_add_readable_labels_turns_codes_into_words():
    frame = pd.DataFrame({"gender": [1, 0], "level": [0, 1]})
    result = add_readable_labels(frame)
    assert result["gender_label"].tolist() == ["Male", "Female"]
    assert result["level_label"].tolist() == ["Free", "Paid"]


def test_load_feature_table_renames_the_target(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame({"userId": ["a"], "gender": [1], "level": [1], "label": [1]}).to_parquet(path)
    frame = load_feature_table(path)
    assert "churn" in frame.columns
    assert "label" not in frame.columns
    assert frame["churn"].dtype == bool


@pytest.mark.skipif(not FEATURES.exists(), reason="features.parquet not built yet")
def test_load_feature_table_reads_the_real_table():
    frame = load_feature_table(FEATURES)
    assert len(frame) > 0
    assert frame["churn"].dtype == bool
    assert {"gender_label", "level_label"} <= set(frame.columns)