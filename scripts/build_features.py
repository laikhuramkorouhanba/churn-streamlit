"""Turn the raw event log into the per-user feature table the app reads.

Run once, commit the output. The raw log is 567 MB and stays out of the
repository; features.parquet is small enough to live in it, which is what
lets the app and the container start without the log at all.

    python -m scripts.build_features

Use the module form. Running the file directly puts scripts/ on the import
path instead of the project root, and the src imports below will fail.
"""

import argparse
import time

import pandas as pd

from src.data_extractor import DataExtractor
from src.data_loader import load_event_log

RAW_PATH = "data/train.parquet"
OUTPUT_PATH = "data/features.parquet"

# (observation start, observation days, prediction start, prediction days)
WINDOWS = [
    ("2018-10-01", 20, "2018-10-29", 10),
    ("2018-10-20", 20, "2018-11-10", 10),
]


def build(raw_path=RAW_PATH, output_path=OUTPUT_PATH, windows=WINDOWS, sample_users=None):
    started = time.time()
    print(f"Reading {raw_path} ...")
    log = load_event_log(raw_path)
    print(f"  {len(log):,} events, {log['userId'].nunique():,} users")

    if sample_users:
        keep = log["userId"].drop_duplicates().head(sample_users)
        log = log[log["userId"].isin(keep)]
        print(f"  sampled down to {len(log):,} events for {sample_users:,} users")

    extractor = DataExtractor()
    frames = []
    for number, (obs_start, obs_days, pred_start, pred_days) in enumerate(windows, 1):
        print(f"Window {number}: observe {obs_start} for {obs_days}d, "
              f"predict from {pred_start} for {pred_days}d")
        frame = extractor.get_train_data(log, obs_start, obs_days, pred_start, pred_days)
        frame = frame.reset_index()          # userId stops being the index
        frame["window"] = number
        print(f"  {len(frame):,} users, churn rate {frame['label'].mean():.2%}")
        frames.append(frame)

    features = pd.concat(frames, ignore_index=True)
    features.to_parquet(output_path, index=False)
    print(f"Wrote {output_path}: {features.shape[0]:,} rows x {features.shape[1]} columns")
    print(f"Overall churn rate {features['label'].mean():.2%}")
    print(f"Done in {time.time() - started:.0f}s")
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-users", type=int, default=None,
                        help="Build from the first N users only, for a faster run")
    args = parser.parse_args()
    build(sample_users=args.sample_users)