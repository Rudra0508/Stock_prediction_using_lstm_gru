import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler
import joblib

from utils.logger import logging
from utils.exception import CustomException


@dataclass
class DataTransformationConfig:
    processed_data_path: str = os.path.join("artifacts", "data", "processed_data.csv")
    scaler_dir: str = os.path.join("artifacts", "scaler")
    sequence_length: int = 60
    forecast_horizon_week: int = 5
    trend_threshold: float = 0.0


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_20", "SMA_50",
    "EMA_10", "EMA_20",
    "MACD", "MACD_Signal", "MACD_Hist",
    "RSI",
    "ROC_10",
    "BB_Upper", "BB_Lower", "BB_Width",
    "ATR",
    "OBV",
    "MFI",
]

TARGET_NEXT_DAY = "Target_NextDay"
TARGET_NEXT_WEEK = "Target_NextWeek"
TARGET_TREND = "Target_Trend"


class FeatureEngineer:

    @staticmethod
    def add_sma(df: pd.DataFrame) -> pd.DataFrame:
        for window in [10, 20, 50]:
            df[f"SMA_{window}"] = df.groupby("Ticker")["Close"].transform(
                lambda x: x.rolling(window=window, min_periods=window).mean()
            )
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame) -> pd.DataFrame:
        for span in [10, 20]:
            df[f"EMA_{span}"] = df.groupby("Ticker")["Close"].transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        def compute_macd(group):
            ema12 = group.ewm(span=12, adjust=False).mean()
            ema26 = group.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            return macd, signal, hist

        results = df.groupby("Ticker")["Close"].apply(
            lambda x: pd.DataFrame(
                {"MACD": compute_macd(x)[0], "MACD_Signal": compute_macd(x)[1], "MACD_Hist": compute_macd(x)[2]},
                index=x.index,
            )
        )
        df = df.join(results.droplevel(0))
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        def compute_rsi(series):
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
            avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            return 100 - (100 / (1 + rs))

        df["RSI"] = df.groupby("Ticker")["Close"].transform(compute_rsi)
        return df

    @staticmethod
    def add_roc(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        df[f"ROC_{period}"] = df.groupby("Ticker")["Close"].transform(
            lambda x: x.pct_change(periods=period) * 100
        )
        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        def compute_bb(group):
            sma = group.rolling(window=window, min_periods=window).mean()
            std = group.rolling(window=window, min_periods=window).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            width = (upper - lower) / (sma + 1e-10)
            return upper, lower, width

        def apply_bb(group):
            upper, lower, width = compute_bb(group)
            return pd.DataFrame(
                {"BB_Upper": upper, "BB_Lower": lower, "BB_Width": width},
                index=group.index,
            )

        results = df.groupby("Ticker")["Close"].apply(apply_bb)
        df = df.join(results.droplevel(0))
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        def compute_atr(group):
            high = group["High"]
            low = group["Low"]
            close = group["Close"]
            prev_close = close.shift(1)
            tr = pd.concat(
                [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
            ).max(axis=1)
            return tr.ewm(com=period - 1, min_periods=period).mean()

        df["ATR"] = df.groupby("Ticker").apply(compute_atr).droplevel(0).sort_index()
        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        def compute_obv(group):
            direction = np.sign(group["Close"].diff().fillna(0))
            return (direction * group["Volume"]).cumsum()

        df["OBV"] = df.groupby("Ticker").apply(compute_obv).droplevel(0).sort_index()
        return df

    @staticmethod
    def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        def compute_mfi(group):
            typical_price = (group["High"] + group["Low"] + group["Close"]) / 3
            raw_money_flow = typical_price * group["Volume"]
            direction = typical_price.diff()
            pos_flow = raw_money_flow.where(direction > 0, 0)
            neg_flow = raw_money_flow.where(direction < 0, 0)
            pos_sum = pos_flow.rolling(window=period, min_periods=period).sum()
            neg_sum = neg_flow.rolling(window=period, min_periods=period).sum()
            mfr = pos_sum / (neg_sum + 1e-10)
            return 100 - (100 / (1 + mfr))

        df["MFI"] = df.groupby("Ticker").apply(compute_mfi).droplevel(0).sort_index()
        return df


class TargetEngineer:

    @staticmethod
    def add_targets(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0) -> pd.DataFrame:
        df[TARGET_NEXT_DAY] = df.groupby("Ticker")["Close"].transform(
            lambda x: x.shift(-1)
        )
        df[TARGET_NEXT_WEEK] = df.groupby("Ticker")["Close"].transform(
            lambda x: x.shift(-horizon)
        )
        df[TARGET_TREND] = (
            (df[TARGET_NEXT_DAY] - df["Close"]) / (df["Close"] + 1e-10) > threshold
        ).astype(int)
        return df


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.target_engineer = TargetEngineer()

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying feature engineering")
        try:
            df = FeatureEngineer.add_sma(df)
            df = FeatureEngineer.add_ema(df)
            df = FeatureEngineer.add_macd(df)
            df = FeatureEngineer.add_rsi(df)
            df = FeatureEngineer.add_roc(df)
            df = FeatureEngineer.add_bollinger_bands(df)
            df = FeatureEngineer.add_atr(df)
            df = FeatureEngineer.add_obv(df)
            df = FeatureEngineer.add_mfi(df)
            logging.info(f"Features engineered: {len(FEATURE_COLUMNS)} feature columns")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Adding target columns")
        try:
            df = TargetEngineer.add_targets(
                df,
                horizon=self.config.forecast_horizon_week,
                threshold=self.config.trend_threshold,
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def _drop_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        all_required = FEATURE_COLUMNS + [TARGET_NEXT_DAY, TARGET_NEXT_WEEK, TARGET_TREND]
        df = df.dropna(subset=all_required).reset_index(drop=True)
        logging.info(f"Dropped NaN rows: {before} → {len(df)}")
        return df

    def _fit_and_save_scalers(self, df: pd.DataFrame) -> dict:
        logging.info("Fitting RobustScalers per ticker")
        os.makedirs(self.config.scaler_dir, exist_ok=True)
        scalers = {}
        try:
            for ticker in df["Ticker"].unique():
                ticker_df = df[df["Ticker"] == ticker][FEATURE_COLUMNS]
                scaler = RobustScaler()
                scaler.fit(ticker_df)
                scaler_path = os.path.join(self.config.scaler_dir, f"scaler_{ticker}.pkl")
                joblib.dump(scaler, scaler_path)
                scalers[ticker] = scaler
                logging.info(f"Scaler saved: {scaler_path}")
            return scalers
        except Exception as e:
            raise CustomException(e, sys)

    def _scale_features(self, df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
        logging.info("Scaling features using fitted scalers")
        try:
            scaled_parts = []
            for ticker, scaler in scalers.items():
                mask = df["Ticker"] == ticker
                ticker_df = df[mask].copy()
                ticker_df[FEATURE_COLUMNS] = scaler.transform(ticker_df[FEATURE_COLUMNS])
                scaled_parts.append(ticker_df)
            return pd.concat(scaled_parts).sort_values(["Ticker", "Date"]).reset_index(drop=True)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_data_path: str) -> str:
        logging.info("Starting data transformation pipeline")
        try:
            df = pd.read_csv(raw_data_path, parse_dates=["Date"])
            logging.info(f"Loaded raw data: {df.shape}")

            df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

            df = self._apply_feature_engineering(df)
            df = self._add_targets(df)
            df = self._drop_nulls(df)

            scalers = self._fit_and_save_scalers(df)
            df = self._scale_features(df, scalers)

            os.makedirs(os.path.dirname(self.config.processed_data_path), exist_ok=True)
            df.to_csv(self.config.processed_data_path, index=False)
            logging.info(f"Processed data saved to: {self.config.processed_data_path}")
            logging.info(
                f"Transformation complete | Shape: {df.shape} | "
                f"Tickers: {df['Ticker'].nunique()} | Features: {len(FEATURE_COLUMNS)}"
            )
            return self.config.processed_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformation = DataTransformation()
    path = transformation.initiate_data_transformation(
        os.path.join("artifacts", "data", "raw_data.csv")
    )
    print(f"Processed data saved at: {path}")