import pytest

from src.filters import (
    churn_rate,
    churn_rate_by,
    filter_by_category,
    filter_by_numeric_range,
)


def test_filter_by_category_keeps_only_the_wanted_values(feature_frame):
    result = filter_by_category(feature_frame, "level_label", ["Paid"])
    assert len(result) == 3
    assert set(result["level_label"]) == {"Paid"}


def test_filter_by_category_with_no_values_is_a_no_op(feature_frame):
    assert len(filter_by_category(feature_frame, "level_label", [])) == len(feature_frame)


def test_filter_by_category_can_return_nothing(feature_frame):
    assert len(filter_by_category(feature_frame, "level_label", ["Platinum"])) == 0


def test_filter_by_category_rejects_an_unknown_column(feature_frame):
    with pytest.raises(KeyError):
        filter_by_category(feature_frame, "not_a_column", ["x"])


def test_filter_by_numeric_range_includes_both_bounds(feature_frame):
    result = filter_by_numeric_range(feature_frame, "active_days", minimum=4, maximum=10)
    assert sorted(result["active_days"].tolist()) == [4, 7, 10]


def test_filter_by_numeric_range_accepts_an_open_end(feature_frame):
    assert len(filter_by_numeric_range(feature_frame, "active_days", minimum=10)) == 3


def test_filter_by_numeric_range_rejects_an_inverted_range(feature_frame):
    with pytest.raises(ValueError, match="greater than"):
        filter_by_numeric_range(feature_frame, "active_days", minimum=20, maximum=5)


def test_churn_rate_counts_the_churners(feature_frame):
    assert churn_rate(feature_frame) == pytest.approx(0.5)


def test_churn_rate_of_an_empty_frame_is_zero(feature_frame):
    assert churn_rate(feature_frame.iloc[0:0]) == 0.0


def test_churn_rate_by_groups_and_sorts_worst_first(feature_frame):
    result = churn_rate_by(feature_frame, "level_label")
    assert result.iloc[0]["level_label"] == "Paid"
    assert result.iloc[0]["churn_rate"] == pytest.approx(2 / 3)
    assert result["users"].sum() == len(feature_frame)


def test_filters_do_not_mutate_their_input(feature_frame):
    before = feature_frame.copy()
    filter_by_category(feature_frame, "level_label", ["Paid"])
    filter_by_numeric_range(feature_frame, "active_days", minimum=5)
    assert feature_frame.equals(before)