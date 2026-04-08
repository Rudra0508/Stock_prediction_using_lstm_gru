import os
import sys
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime, timedelta

import yfinance as yf
import tensorflow as tf

from utils.logger import logging
from utils.exception import CustomException
from src.components.data_transformation import (
    FeatureEngineer,
    FEATURE_COLUMNS,
    DataTransformationConfig,
)


@dataclass
class PredictPipelineConfig:
    model_dir: str = os.path.join("artifacts", "models")
    scaler_dir: str = os.path.join("artifacts", "scaler")
    sequence_length: int = 60
    refresh_lookback_days: int = 120


@dataclass
class PredictionResult:
    ticker: str
    next_day_price: float
    next_week_price: float
    trend: str
    trend_confidence: float
    last_close: float
    prediction_timestamp: str


class ModelLoader:

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._cache: Dict[str, tf.keras.Model] = {}

    def _model_path(self, ticker: str, task: str) -> str:
        safe_ticker = ticker.replace(".", "_")
        return os.path.join(self.model_dir, safe_ticker, f"model_{task}.keras")

    def load(self, ticker: str, task: str) -> tf.keras.Model:
        key = f"{ticker}_{task}"
        if key not in self._cache:
            path = self._model_path(ticker, task)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: {path}. Run train pipeline first.")
            self._cache[key] = tf.keras.models.load_model(path)
            logging.info(f"Loaded model: {path}")
        return self._cache[key]

    def clear_cache(self):
        self._cache.clear()
        logging.info("Model cache cleared")


class ScalerLoader:

    def __init__(self, scaler_dir: str):
        self.scaler_dir = scaler_dir
        self._cache: Dict[str, object] = {}

    def load(self, ticker: str):
        if ticker not in self._cache:
            path = os.path.join(self.scaler_dir, f"scaler_{ticker}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Scaler not found: {path}. Run train pipeline first.")
            self._cache[ticker] = joblib.load(path)
            logging.info(f"Loaded scaler: {path}")
        return self._cache[ticker]


class LiveDataFetcher:

    def __init__(self, lookback_days: int = 120):
        self.lookback_days = lookback_days

    def fetch(self, ticker: str) -> pd.DataFrame:
        logging.info(f"Fetching live data for: {ticker}")
        try:
            end = datetime.today()
            start = end - timedelta(days=self.lookback_days)

            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                raise ValueError(f"No live data returned for {ticker}")

            df = df.reset_index()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]

            df.columns = [str(c).strip() for c in df.columns]
            df["Ticker"] = ticker
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

            logging.info(f"Fetched {len(df)} live rows for {ticker}")
            return df

        except Exception as e:
            raise CustomException(e, sys)


class FeatureBuilder:

    @staticmethod
    
    def build(df: pd.DataFrame) -> pd.DataFrame:
        
        df = FeatureEngineer.add_sma(df)
        df = FeatureEngineer.add_ema(df)
        df = FeatureEngineer.add_macd(df)
        df = FeatureEngineer.add_rsi(df)
        df = FeatureEngineer.add_roc(df)
        df = FeatureEngineer.add_bollinger_bands(df)
        df = FeatureEngineer.add_atr(df)
        df = FeatureEngineer.add_obv(df)
        df = FeatureEngineer.add_mfi(df)
        return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


class PredictPipeline:

    def __init__(self, config: PredictPipelineConfig = PredictPipelineConfig()):
        self.config = config
        self.model_loader = ModelLoader(config.model_dir)
        self.scaler_loader = ScalerLoader(config.scaler_dir)
        self.live_fetcher = LiveDataFetcher(config.refresh_lookback_days)

    def _build_sequence(self, df: pd.DataFrame, ticker: str) -> Optional[np.ndarray]:
        scaler = self.scaler_loader.load(ticker)
        features = df[FEATURE_COLUMNS].values
        scaled = scaler.transform(features)

        seq_len = self.config.sequence_length
        if len(scaled) <= seq_len:
            logging.warning(f"Not enough rows for {ticker}: need {seq_len}, got {len(scaled)}")
            return None

        sequence = scaled[-seq_len:]
        return sequence.reshape(1, seq_len, len(FEATURE_COLUMNS)).astype(np.float32)

    def predict(self, ticker: str) -> PredictionResult:
        logging.info(f"Running prediction for: {ticker}")
        try:
            df = self.live_fetcher.fetch(ticker)
            df = FeatureBuilder.build(df)

# Safety AFTER feature engineering
            df = df.reindex(columns=df.columns.union(FEATURE_COLUMNS), fill_value=0)

            last_close = float(df["Close"].iloc[-1])
            X = self._build_sequence(df, ticker)

            if X is None:
                raise ValueError(f"Insufficient data to build sequence for {ticker}")

            next_day_raw = float(
                self.model_loader.load(ticker, "next_day").predict(X, verbose=0).flatten()[0]
            )
            next_week_raw = float(
                self.model_loader.load(ticker, "next_week").predict(X, verbose=0).flatten()[0]
            )
            trend_prob = float(
                self.model_loader.load(ticker, "trend").predict(X, verbose=0).flatten()[0]
            )

            next_day_price = next_day_raw
            next_week_price = next_week_raw 

            trend = "Bullish" if trend_prob >= 0.5 else "Bearish"

            result = PredictionResult(
                ticker=ticker,
                next_day_price=round(next_day_price, 2),
                next_week_price=round(next_week_price, 2),
                trend=trend,
                trend_confidence=round(trend_prob if trend == "Bullish" else 1 - trend_prob, 4),
                last_close=round(last_close, 2),
                prediction_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            logging.info(
                f"{ticker} → NextDay: {result.next_day_price} | "
                f"NextWeek: {result.next_week_price} | "
                f"Trend: {result.trend} ({result.trend_confidence:.2%})"
            )
            return result

        except Exception as e:
            raise CustomException(e, sys)

    def predict_batch(self, tickers: list) -> Dict[str, PredictionResult]:
        logging.info(f"Batch prediction for {len(tickers)} tickers")
        results = {}
        failed = []
        for ticker in tickers:
            try:
                results[ticker] = self.predict(ticker)
            except Exception as e:
                logging.error(f"Prediction failed for {ticker}: {str(e)}")
                failed.append(ticker)
        if failed:
            logging.warning(f"Failed tickers in batch: {failed}")
        return results

    def refresh_and_predict(self, ticker: str) -> PredictionResult:
        logging.info(f"Refresh-and-predict triggered for: {ticker}")
        self.model_loader.clear_cache()
        return self.predict(ticker)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestionConfig

    tickers = DataIngestionConfig().tickers
    pipeline = PredictPipeline()
    results = pipeline.predict_batch(tickers[:3])

    for ticker, result in results.items():
        print(f"\n{'='*40}")
        print(f"Ticker            : {result.ticker}")
        print(f"Last Close        : ₹{result.last_close}")
        print(f"Next Day Price    : ₹{result.next_day_price}")
        print(f"Next Week Price   : ₹{result.next_week_price}")
        print(f"Trend             : {result.trend} ({result.trend_confidence:.2%} confidence)")
        print(f"Timestamp         : {result.prediction_timestamp}")