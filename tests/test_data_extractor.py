"""The windowing and labelling logic, which is where churn is defined."""

import pandas as pd

from src.data_extractor import DataExtractor
from tests.conftest import OBS_DAYS, OBS_START, PRED_DAYS, PRED_START


def test_observation_window_keeps_only_events_inside_it(event_log):
    obs = DataExtractor().get_observation_window_df(event_log, OBS_START, OBS_DAYS)
    start = pd.Timestamp(OBS_START)
    assert obs["time"].min() >= start
    assert obs["time"].max() < start + pd.Timedelta(days=OBS_DAYS)


def test_observation_window_drops_users_who_cancel_inside_it(event_log):
    obs = DataExtractor().get_observation_window_df(event_log, OBS_START, OBS_DAYS)
    users = set(obs["userId"])
    assert "u_cancels_in_obs" not in users
    assert users == {"u_stays", "u_cancels_in_pred"}


def test_prediction_window_labels_only_the_user_who_cancels(event_log):
    users = pd.Index(["u_stays", "u_cancels_in_pred"], name="userId")
    labels = DataExtractor().get_prediction_window_label(
        event_log, PRED_START, PRED_DAYS, users
    )
    assert labels["u_cancels_in_pred"] == 1
    assert labels["u_stays"] == 0


def test_get_train_data_returns_one_row_per_surviving_user(event_log):
    train = DataExtractor().get_train_data(
        event_log, OBS_START, OBS_DAYS, PRED_START, PRED_DAYS
    )
    assert set(train.index) == {"u_stays", "u_cancels_in_pred"}
    assert train.loc["u_cancels_in_pred", "label"] == 1
    assert train.loc["u_stays", "label"] == 0


def test_get_train_data_leaves_no_missing_values(event_log):
    train = DataExtractor().get_train_data(
        event_log, OBS_START, OBS_DAYS, PRED_START, PRED_DAYS
    )
    assert train.isna().sum().sum() == 0


def test_total_events_counts_only_the_observation_window(event_log):
    train = DataExtractor().get_train_data(
        event_log, OBS_START, OBS_DAYS, PRED_START, PRED_DAYS
    )
    # u_stays has four events inside the window and one after it.
    assert train.loc["u_stays", "total_events"] == 4
    assert train.loc["u_cancels_in_pred", "total_events"] == 2


def test_profile_features_are_encoded_as_integers(event_log):
    train = DataExtractor().get_train_data(
        event_log, OBS_START, OBS_DAYS, PRED_START, PRED_DAYS
    )
    assert train.loc["u_stays", "gender"] == 1          # M
    assert train.loc["u_cancels_in_pred", "gender"] == 0  # F
    assert train.loc["u_stays", "level"] == 1           # paid
    assert train.loc["u_cancels_in_pred", "level"] == 0  # free