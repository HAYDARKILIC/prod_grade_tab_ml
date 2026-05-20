"""Streamlit demo for interactive inference against a saved model.

Usage::

    streamlit run scripts/streamlit_demo.py -- --model artifacts/xgb.pkl
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GradientForge — Inference Demo", layout="wide")
st.title("GradientForge — interactive inference")

model_path = st.sidebar.text_input("Model path", value="artifacts/model.pkl")
if not Path(model_path).is_file():
    st.warning("Provide a path to a pickled trained model.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

uploaded = st.file_uploader("Upload a CSV with the same columns used at train time", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview", df.head())
    try:
        preds = model.predict(df.to_numpy())
        st.write("Predictions", pd.DataFrame({"prediction": np.asarray(preds).ravel()}))
    except Exception as e:
        st.error(f"Inference failed: {e}")
