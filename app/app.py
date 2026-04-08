import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.pipeline.predict_pipeline import PredictPipeline
from src.components.data_ingestion import DataIngestionConfig

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Stock Prediction AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Prediction System (LSTM + GRU Hybrid)")
st.markdown("Real-time predictions using deep learning ensemble model")

# -----------------------------
# Load Pipeline
# -----------------------------
@st.cache_resource
def load_pipeline():
    return PredictPipeline()

pipeline = load_pipeline()

# -----------------------------
# Sidebar
# -----------------------------
config = DataIngestionConfig()
ticker = st.sidebar.selectbox("Select Stock", config.tickers)

# -----------------------------
# Buttons
# -----------------------------
col1, col2 = st.sidebar.columns(2)

predict_clicked = col1.button("🔮 Predict")
refresh_clicked = col2.button("🔄 Refresh")

# -----------------------------
# Prediction Logic
# -----------------------------
if predict_clicked or refresh_clicked:

    with st.spinner("Fetching data and running model..."):

        try:
            if refresh_clicked:
                result = pipeline.refresh_and_predict(ticker)
            else:
                result = pipeline.predict(ticker)

            # -----------------------------
            # Display Results
            # -----------------------------
            st.subheader(f"📊 Predictions for {ticker}")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Last Close",
                f"₹{result.last_close}"
            )

            col2.metric(
                "Next Day Prediction",
                f"₹{result.next_day_price}",
                delta=f"{result.next_day_price - result.last_close:.2f}"
            )

            col3.metric(
                "Next Week Prediction",
                f"₹{result.next_week_price}",
                delta=f"{result.next_week_price - result.last_close:.2f}"
            )

            # Trend
            st.markdown("### 📈 Trend Prediction")

            trend_color = "green" if result.trend == "Bullish" else "red"

            st.markdown(
                f"<h3 style='color:{trend_color}'>{result.trend} ({result.trend_confidence*100:.2f}%)</h3>",
                unsafe_allow_html=True
            )

            st.caption(f"Prediction Time: {result.prediction_timestamp}")

            # -----------------------------
            # Graph
            # -----------------------------
            st.markdown("### 📉 Price Chart")

            df = pipeline.live_fetcher.fetch(ticker)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode='lines',
                name='Close Price'
            ))

            # Add prediction point
            fig.add_trace(go.Scatter(
                x=[df["Date"].iloc[-1]],
                y=[result.next_day_price],
                mode='markers',
                name='Next Day Prediction',
                marker=dict(size=10)
            ))

            fig.update_layout(
                template="plotly_dark",
                height=500,
                xaxis_title="Date",
                yaxis_title="Price"
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")