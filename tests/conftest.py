"""Fixtures for the test suite.

The event log here is synthetic and tiny, with three users whose fates are
known by construction. That is what makes it possible to assert that the
sliding windows and the churn labelling are correct, rather than merely
that they run.
"""

import pandas as pd
import pytest

OBS_START = "2018-10-01"
OBS_DAYS = 10
PRED_START = "2018-10-11"
PRED_DAYS = 5


def _event(user, when, page, session=1, gender="M", level="paid",
           registration="2018-08-01"):
    """One row of the event log, with ts kept consistent with time."""
    moment = pd.to_datetime(when)
    return {
        "userId": user,
        "time": moment,
        "ts": int(moment.value // 10**6),
        "registration": pd.to_datetime(registration),
        "gender": gender,
        "level": level,
        "page": page,
        "sessionId": session,
    }


@pytest.fixture
def event_log() -> pd.DataFrame:
    """Three users: one stays, one leaves during observation, one after."""
    rows = [
        # Active all through the observation window, never cancels.
        _event("u_stays", "2018-10-01 10:00", "NextSong", session=1),
        _event("u_stays", "2018-10-01 10:05", "NextSong", session=1),
        _event("u_stays", "2018-10-04 18:00", "Thumbs Up", session=2),
        _event("u_stays", "2018-10-09 21:00", "NextSong", session=3),
        _event("u_stays", "2018-10-12 09:00", "NextSong", session=4),

        # Cancels inside the observation window, so must be dropped entirely.
        _event("u_cancels_in_obs", "2018-10-02 11:00", "NextSong", session=5),
        _event("u_cancels_in_obs", "2018-10-05 11:30",
               "Cancellation Confirmation", session=5),

        # Survives observation, cancels in the prediction window: label 1.
        _event("u_cancels_in_pred", "2018-10-03 08:00", "NextSong",
               session=6, gender="F", level="free"),
        _event("u_cancels_in_pred", "2018-10-08 08:30", "Roll Advert",
               session=6, gender="F", level="free"),
        _event("u_cancels_in_pred", "2018-10-12 12:00",
               "Cancellation Confirmation", session=7, gender="F", level="free"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """A small engineered table, shaped like what build_features writes."""
    return pd.DataFrame(
        {
            "userId": ["a", "b", "c", "d", "e", "f"],
            "gender_label": ["Male", "Male", "Female", "Female", "Male", "Female"],
            "level_label": ["Paid", "Free", "Paid", "Free", "Free", "Paid"],
            "total_events": [120, 40, 300, 15, 80, 500],
            "active_days": [10, 4, 18, 2, 7, 20],
            "churn": [True, False, True, False, True, False],
        }
    )