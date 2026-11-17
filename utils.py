"""Shared helpers: loading, column detection, feature building, evaluation."""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config

# Columns holding other energy outputs (e.g. wind power) would leak information
# when we predict solar power, so they are dropped from the features.
LEAK_KEYWORDS = ("power", "generation", "output", "energy", "production")


# ----------------------------------------------------------------- loading
def load_data(csv_file=None, data_dir=None):
    data_dir = data_dir or config.DATA_DIR
    csv_file = csv_file or config.CSV_FILE
    if csv_file:
        path = csv_file if os.path.exists(csv_file) else os.path.join(data_dir, csv_file)
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(
                f"No CSV found in '{data_dir}/'. Download the Kaggle dataset and "
                "place the CSV file(s) there (see data/README.md)."
            )
        path = files[0]
        if len(files) > 1:
            print(f"[info] {len(files)} CSV files found, using '{path}'. "
                  "Set CSV_FILE in config.py to choose another.")
    print(f"[info] Loading {path}")
    return pd.read_csv(path)


# --------------------------------------------------------- column detection
def detect_time_col(df):
    if config.TIME_COL:
        return config.TIME_COL
    for c in df.columns:
        if any(k in c.lower() for k in ("time", "date", "stamp")):
            return c
    return None


def detect_target_col(df, time_col=None):
    if config.TARGET_COL:
        return config.TARGET_COL
    numeric = [c for c in df.select_dtypes("number").columns if c != time_col]
    lower = {c: c.lower() for c in numeric}
    for rule in (
        lambda n: "solar" in n and any(k in n for k in ("power", "generation", "output", "energy")),
        lambda n: "power" in n,
        lambda n: "generation" in n or "output" in n or "energy" in n,
        lambda n: "solar" in n or "pv" in n,
    ):
        hits = [c for c, n in lower.items() if rule(n)]
        if hits:
            return hits[0]
    raise ValueError(
        "Could not auto-detect the target column. Set TARGET_COL in config.py. "
        f"Available numeric columns: {numeric}"
    )


# -------------------------------------------------------------- features
def make_time_features(ts):
    """Calendar features (plus cyclical encodings) from a datetime Series."""
    ts = pd.to_datetime(pd.Series(ts).reset_index(drop=True))
    f = pd.DataFrame(index=ts.index)
    f["hour"] = ts.dt.hour
    f["month"] = ts.dt.month
    f["dayofyear"] = ts.dt.dayofyear
    f["dayofweek"] = ts.dt.dayofweek
    f["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24)
    f["doy_sin"] = np.sin(2 * np.pi * f["dayofyear"] / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * f["dayofyear"] / 365.25)
    return f


def prepare_data(df):
    """Return X, y, info. Rows are sorted chronologically when a time column exists."""
    df = df.copy()
    time_col = detect_time_col(df)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    target = detect_target_col(df, time_col)
    df = df.dropna(subset=[target]).reset_index(drop=True)

    weather_cols = []
    for c in df.select_dtypes("number").columns:
        if c in (target, time_col):
            continue
        if any(k in c.lower() for k in LEAK_KEYWORDS):
            continue
        weather_cols.append(c)

    X = df[weather_cols].copy()
    if time_col is not None:
        X = pd.concat([X, make_time_features(df[time_col])], axis=1)
    y = df[target]

    info = {
        "target": target,
        "time_col": time_col,
        "weather_cols": weather_cols,
        "feature_names": list(X.columns),
        "dropped_cols": [c for c in df.columns
                         if c not in weather_cols + [target, time_col]],
        "time_index": df[time_col] if time_col is not None else None,
    }
    return X, y, info


def time_split(X, y, test_size=None):
    """Chronological split: no shuffling, so the model never sees the future."""
    test_size = test_size or config.TEST_SIZE
    cut = int(len(X) * (1 - test_size))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


# ------------------------------------------------------------ evaluation
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }
