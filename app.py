"""Streamlit front end for the churn explorer.

Reads the engineered feature table, never the raw event log. All data logic
lives in src/ so it can be unit tested; this file is presentation only.
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.data_loader import load_feature_table
from src.filters import (
    churn_rate,
    churn_rate_by,
    filter_by_category,
    filter_by_numeric_range,
)

FEATURES_PATH = "data/features.parquet"
NOT_FEATURES = {"userId", "churn", "window", "gender", "level",
                "gender_label", "level_label"}

st.set_page_config(page_title="Churn explorer", page_icon="📉", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_feature_table(FEATURES_PATH)


data = get_data()
numeric_features = sorted(
    column for column in data.columns
    if column not in NOT_FEATURES and pd.api.types.is_numeric_dtype(data[column])
)

st.title("Music streaming churn explorer")
st.caption(
    f"{len(data):,} user snapshots from a {len(data['window'].unique())}-window "
    f"sweep of the event log. A snapshot is one user observed for 20 days; "
    f"churn means they cancelled in the 10 days that followed."
)

st.sidebar.header("Filters")
segment_columns = ["gender_label", "level_label"]

filtered = data
for column in segment_columns:
    options = sorted(data[column].dropna().unique().tolist())
    chosen = st.sidebar.multiselect(column.replace("_label", "").title(), options)
    filtered = filter_by_category(filtered, column, chosen)

chosen_feature = st.sidebar.selectbox("Numeric filter", numeric_features)
low = float(data[chosen_feature].min())
high = float(data[chosen_feature].max())
bounds = st.sidebar.slider(
    chosen_feature.replace("_", " "), low, high, (low, high)
)
filtered = filter_by_numeric_range(
    filtered, chosen_feature, minimum=bounds[0], maximum=bounds[1]
)

left, middle, right = st.columns(3)
left.metric("Users in view", f"{len(filtered):,}")
middle.metric("Churn rate in view", f"{churn_rate(filtered):.2%}")
right.metric("Overall churn rate", f"{churn_rate(data):.2%}")

if len(filtered) == 0:
    st.warning("No users match these filters. Widen them in the sidebar.")
    st.stop()

st.subheader("Churn rate by segment")
segment = st.selectbox("Break down by", segment_columns)
breakdown = churn_rate_by(filtered, segment)
st.altair_chart(
    alt.Chart(breakdown)
    .mark_bar()
    .encode(
        x=alt.X("churn_rate:Q", title="Churn rate", axis=alt.Axis(format="%")),
        y=alt.Y(f"{segment}:N", sort="-x", title=None),
        tooltip=[segment, "churn_rate", "users"],
    )
    .properties(height=max(120, 46 * len(breakdown))),
    use_container_width=True,
)

st.subheader("How churners differ")
compare = st.selectbox("Compare the distribution of", numeric_features,
                       key="compare")
plot_frame = filtered[[compare, "churn"]].copy()
plot_frame["group"] = plot_frame["churn"].map({True: "Churned", False: "Stayed"})
st.altair_chart(
    alt.Chart(plot_frame)
    .mark_bar(opacity=0.65)
    .encode(
        x=alt.X(f"{compare}:Q", bin=alt.Bin(maxbins=40), title=compare.replace("_", " ")),
        y=alt.Y("count():Q", stack=None, title="Users"),
        color=alt.Color("group:N", title=None),
    )
    .properties(height=280),
    use_container_width=True,
)

with st.expander("Show the filtered data"):
    st.dataframe(filtered, use_container_width=True)