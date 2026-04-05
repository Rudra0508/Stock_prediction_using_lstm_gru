import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📈 Stock Market Prediction (LSTM + GRU)")

# ---------------- STOCK SELECTION ---------------- #
stock = st.selectbox(
    "Select Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
)

# ---------------- FETCH DATA ---------------- #
@st.cache_data
def load_data(stock):
    df = yf.download(stock, period="6mo", interval="1d")
    df = df.reset_index()
    return df

df = load_data(stock)

st.subheader("📊 Historical Data")
st.line_chart(df["Close"])

# ---------------- PREPARE INPUT ---------------- #
if len(df) < 60:
    st.error("Not enough data")
else:
    last_60_days = df[["Open", "High", "Low", "Close", "Volume"]].tail(60)

    # Convert to numpy
    input_data = last_60_days.values

    # ---------------- PREDICT ---------------- #
    if st.button("🔮 Predict"):
        pipeline = PredictPipeline()

        result = pipeline.predict(input_data)

        st.subheader("📊 Prediction Results")

        st.write(f"### 📅 Next Day Price (Scaled): {result['next_day_price']:.4f}")

        st.write("### 📅 Next 5 Days Forecast (Scaled):")
        st.write(result["next_week_prices"])

        st.write(f"### 📈 Trend: {result['trend']}")