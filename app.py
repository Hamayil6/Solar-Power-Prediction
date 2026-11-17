"""Streamlit app: predict solar power generation from weather conditions.

Run:  streamlit run app.py   (after `python train.py`)
"""
import datetime as dt
import os

import joblib
import pandas as pd
import streamlit as st

import config
from utils import make_time_features

st.set_page_config(page_title="Solar Power Prediction", page_icon="☀️", layout="wide")
st.title("☀️ Solar Power Generation Prediction")
st.caption("Predict solar power output from weather conditions.")

if not os.path.exists(config.MODEL_PATH):
    st.error("Model not found. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def load_bundle():
    return joblib.load(config.MODEL_PATH)


b = load_bundle()
stats = b["feature_stats"]

with st.sidebar:
    st.header("Weather inputs")
    values = {}
    for col in b["weather_cols"]:
        lo, med, hi = float(stats.loc[col, "min"]), float(stats.loc[col, "median"]), float(stats.loc[col, "max"])
        values[col] = st.slider(col, lo, hi, med) if hi > lo else med
    when = None
    if b["time_col"]:
        st.header("Date and time")
        d = st.date_input("Date", dt.date.today())
        h = st.slider("Hour of day", 0, 23, 12)
        when = dt.datetime.combine(d, dt.time(h))

if st.button("Predict", type="primary"):
    row = pd.DataFrame([values])
    if when is not None:
        row = pd.concat([row, make_time_features([when])], axis=1)
    row = row[b["feature_names"]]
    pred = max(float(b["model"].predict(row)[0]), 0.0)
    st.metric(f"Predicted {b['target']}", f"{pred:,.3f}")

st.subheader(f"Model: {b['model_name']}")
m = b["metrics"]
c1, c2, c3 = st.columns(3)
c1.metric("RMSE", f"{m['RMSE']:.3f}")
c2.metric("MAE", f"{m['MAE']:.3f}")
c3.metric("R²", f"{m['R2']:.3f}")

csv = f"{config.RESULTS_DIR}/model_comparison.csv"
if os.path.exists(csv):
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv(csv).round(4))
for img in ("actual_vs_predicted", "feature_importance"):
    p = f"{config.RESULTS_DIR}/{img}.png"
    if os.path.exists(p):
        st.image(p)
