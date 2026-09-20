"""Feature engineering for the music-streaming churn challenge.

The raw data is an event log: one row per user action (playing a song, giving a
thumbs up, seeing an advert, cancelling the subscription, ...). To turn this into
a supervised learning problem we slide two windows over the timeline:

- an **observation window**, from which we build one feature vector per user, and
- a **prediction window**, immediately after it, in which we check whether the
  user churned (triggered a ``Cancellation Confirmation``).

Users who already churn *inside* the observation window are dropped, so every
training example is a user who is still active at the moment we make the
prediction. Sliding the pair of windows to several positions in time gives us
several snapshots of the same population and multiplies the amount of training
data (see ``2_dataset_creation.ipynb``).

The class is deliberately stateless: every method takes the event log and the
window definition explicitly, which keeps the feature logic easy to test and to
reuse for both the training and the test sets.
"""

import pandas as pd
import numpy as np


class DataExtractor():

    def get_observation_window_df(self, df, start_date, window_length):
        """
        Returns all rows in the observation window:
        from start_date to start_date + obs_days
        and removes users who churn inside that window.

        Parameters
        ----------
        df : DataFrame
            Full event log
        start_date : Timestamp
            Start of window
        window_length : int
            Length of window in days

        Returns
        -------
        DataFrame
            Filtered observation window:
            - only events inside the time range
            - excludes users who churn during this observation window
        """
        start_date = pd.to_datetime(start_date)
        end_date = start_date + pd.Timedelta(days=window_length)

        obs_df = df[(df['time'] >= start_date) & (df['time'] < end_date)].copy()

        # Remove users who churned inside the observation window
        churned_in_obs = obs_df[obs_df['page'] == 'Cancellation Confirmation']['userId']

        # Filter out churned users
        obs_df = obs_df[~obs_df['userId'].isin(churned_in_obs)].copy()

        return obs_df

    def get_observation_features(self, df, end_date):
        '''
        Creates features from logs appearing in an observation window
        '''

        df = df.copy()

        # --------------------------
        # TIME PROCESSING
        # --------------------------
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        df['date'] = df['datetime'].dt.date
        df['registration'] = pd.to_datetime(df['registration'])

        df['days_since_registration'] = (end_date - df['registration']).dt.days

        user_group = df.groupby('userId')

        # --------------------------
        # A. USER PROFILE
        # --------------------------
        gender = (user_group['gender'].first() == 'M').astype(int)
        level = (user_group['level'].last() == 'paid').astype(int)
        days_reg = user_group['days_since_registration'].first()

        # --------------------------
        # B. CORE ACTIVITY
        # --------------------------
        total_events = user_group.size()
        nextsong_count = user_group['page'].apply(lambda x: (x == 'NextSong').sum())
        num_sessions = user_group['sessionId'].nunique()
        active_days = user_group['date'].nunique()

        # Normalized activity
        events_per_day = total_events / active_days.replace(0, 1)
        songs_per_day = nextsong_count / active_days.replace(0, 1)
        sessions_per_day = num_sessions / active_days.replace(0, 1)

        # --------------------------
        # C. ENGAGEMENT / FRUSTRATION
        # --------------------------
        thumbs_up = user_group['page'].apply(lambda x: (x == 'Thumbs Up').sum())
        thumbs_down = user_group['page'].apply(lambda x: (x == 'Thumbs Down').sum())
        adverts = user_group['page'].apply(lambda x: (x == 'Roll Advert').sum())
        errors = user_group['page'].apply(lambda x: (x == 'Error').sum())
        logouts = user_group['page'].apply(lambda x: (x == 'Logout').sum())
        downgrades = user_group['page'].apply(lambda x: (x == 'Submit Downgrade').sum())

        # Normalized frustration features
        songs_safe = nextsong_count.replace(0, 1)
        sessions_safe = num_sessions.replace(0, 1)
        events_safe = total_events.replace(0, 1)

        thumbsdown_per_song = thumbs_down / songs_safe
        thumbsup_per_song = thumbs_up / songs_safe
        errors_per_session = errors / sessions_safe
        adverts_per_session = adverts / sessions_safe

        # --------------------------
        # D. TEMPORAL (RECENCY + SESSIONS)
        # --------------------------
        last_event_time = user_group['datetime'].max()
        recency = (end_date - last_event_time).dt.total_seconds() / 86400
        log_recency = np.log1p(recency)

        # session durations
        sessions = df.groupby(['userId', 'sessionId'])['datetime'].agg(['min', 'max'])
        sessions['duration'] = (sessions['max'] - sessions['min']).dt.total_seconds()

        session_stats = sessions.groupby('userId')['duration'].agg(['mean', 'std']).fillna(0)

        short_sessions = sessions.groupby('userId')['duration'].apply(lambda x: (x < 120).sum())
        song_counts_per_session = df.groupby(['userId', 'sessionId'])['page'].apply(lambda x: (x == 'NextSong').sum())
        stalled_sessions = song_counts_per_session.groupby('userId').apply(lambda x: (x == 0).sum())

        stalled_ratio = stalled_sessions / sessions_safe
        short_ratio = short_sessions / sessions_safe

        # --------------------------
        # E. LAST 7 DAYS (VERY IMPORTANT)
        # --------------------------
        cutoff_7 = end_date - pd.Timedelta(days=7)
        df_7 = df[df['datetime'] >= cutoff_7]
        user_7 = df_7.groupby('userId')

        activity_7 = user_7.size().reindex(user_group.groups.keys(), fill_value=0)
        songs_7 = user_7['page'].apply(lambda x: (x == 'NextSong').sum()).reindex(user_group.groups.keys(), fill_value=0)

        # Trend indicators (key for logistic regression)
        activity_7_ratio = activity_7 / total_events.replace(0, 1)
        songs_7_ratio = songs_7 / nextsong_count.replace(0, 1)

        # --------------------------
        # F. DAILY VARIATION
        # --------------------------
        daily_events = df.groupby(['userId', 'date']).size().unstack(fill_value=0)
        events_std = daily_events.std(axis=1).fillna(0)
        events_cv = events_std / (daily_events.mean(axis=1).replace(0, 1))

        # --------------------------
        # FINAL FEATURE TABLE
        # --------------------------
        features = pd.DataFrame({
            # Profile
            'gender': gender,
            'level': level,
            'days_since_registration': days_reg,

            # Basic Activity
            'total_events': total_events,
            'nextsong_count': nextsong_count,
            'num_sessions': num_sessions,
            'active_days': active_days,

            # Normalized Activity
            'events_per_day': events_per_day,
            'songs_per_day': songs_per_day,
            'sessions_per_day': sessions_per_day,

            # Engagement / Frustration normalized
            'thumbsdown_per_song': thumbsdown_per_song,
            'thumbsup_per_song': thumbsup_per_song,
            'errors_per_session': errors_per_session,
            'adverts_per_session': adverts_per_session,

            # Temporal
            'recency': recency,
            'log_recency': log_recency,
            'session_mean': session_stats['mean'],
            'session_std': session_stats['std'],
            'short_session_ratio': short_ratio,
            'stalled_session_ratio': stalled_ratio,

            # Last 7 days trend
            'activity_last_7_days': activity_7,
            'song_count_last_7_days': songs_7,
            'activity_7d_ratio': activity_7_ratio,
            'songs_7d_ratio': songs_7_ratio,

            # Stability / volatility
            'events_std': events_std,
            'events_cv': events_cv,
        })

        return features

    def get_prediction_window_label(self, df, pred_start, pred_length, users_in_obs):
        """
        Assigns churn labels for users based on events in the prediction window.

        Parameters
        ----------
        df : DataFrame
            Full event log.
        pred_start : str or datetime
            Start of the prediction window.
        pred_length : int
            Number of days in the prediction window.
        users_in_obs : array-like
            User IDs that survived the observation window (no churn in obs).

        Returns
        -------
        Series
            Index: userId
            Values: 1 = churn in prediction window, 0 = no churn
        """
        pred_start = pd.to_datetime(pred_start)
        pred_end = pred_start + pd.Timedelta(days=pred_length)

        # Filter prediction window rows
        pred_df = df[(df['time'] >= pred_start) & (df['time'] < pred_end)]

        # Find users who churned in the prediction window
        churn_users = pred_df.loc[
            pred_df['page'] == 'Cancellation Confirmation',
            'userId'
        ].unique()

        # Build label series
        labels = pd.Series(
            data=[1 if user in churn_users else 0 for user in users_in_obs],
            index=users_in_obs,
            name="label"
        )

        return labels

    def get_train_data(self, df, obs_start, obs_length, pred_start, pred_length):
        '''
        Extracts a dataset from df by specifying the observation window and prediction window
        '''
        # extract rows within the observation window for customers who did not churn
        obs = self.get_observation_window_df(df, obs_start, obs_length)

        # aggregate features per user in observation window
        obs_start = pd.to_datetime(obs_start)
        end_date = obs_start + pd.Timedelta(days=obs_length)
        obs_features = self.get_observation_features(obs, end_date)

        # get labels for each user in the observation window
        users_in_obs = obs_features.index
        labels = self.get_prediction_window_label(df, pred_start, pred_length, users_in_obs)

        # concatenate features and label to get a dataset for this particular obs window and pred window
        train = obs_features.copy()
        train["label"] = labels

        train = train.fillna(0)

        return train
