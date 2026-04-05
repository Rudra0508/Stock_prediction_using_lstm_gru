import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    raw_data_path: str = os.path.join("artifacts", "data", "raw_data.csv")
    processed_data_path: str = os.path.join("artifacts", "data", "processed_data.npy")
    scaler_path: str = os.path.join("artifacts", "scaler", "scaler.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def create_sequences(self, data, lookback=60, forecast_horizon=5):
        X, y_day, y_week, y_trend = [], [], [], []

        for i in range(lookback, len(data) - forecast_horizon):
            X.append(data[i - lookback:i])

            # Next day prediction
            y_day.append(data[i][3])  # Close price

            # Next week prediction (5 days)
            y_week.append(data[i:i + forecast_horizon, 3])

            # Trend (classification)
            trend = 1 if data[i][3] > data[i - 1][3] else 0
            y_trend.append(trend)

        return np.array(X), np.array(y_day), np.array(y_week), np.array(y_trend)

    def initiate_data_transformation(self):
        logging.info("Starting data transformation")

        try:
            # Load data
            df = pd.read_csv(self.config.raw_data_path)

            logging.info("Raw data loaded successfully")

            
            # ---------------- CLEANING ---------------- #

            # Remove duplicates
            df = df.drop_duplicates()

            # Handle missing values
            df = df.dropna()

            # Convert Date properly
            df["Date"] = pd.to_datetime(df["Date"])

            # Sort again (important for time series)
            df = df.sort_values("Date")

            # Convert numeric columns safely
            features = ["Open", "High", "Low", "Close", "Volume"]

            for col in features:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows where conversion failed
            df = df.dropna()

            logging.info("Data cleaning completed")

            # Select features
            print(df.columns)
            df = df[["Date", "Open", "High", "Low", "Close", "Volume", "stock"]]
            features = ["Open", "High", "Low", "Close", "Volume"]
            data = df[features].astype(float).values

            # Scaling
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            logging.info("Data scaling completed")

            # Create sequences
            X, y_day, y_week, y_trend = self.create_sequences(scaled_data)

            logging.info(f"Sequences created: {X.shape}")

            # Train-test split
            X_train, X_test, y_day_train, y_day_test = train_test_split(X, y_day, test_size=0.2, shuffle=False)

            _, _, y_week_train, y_week_test = train_test_split(X, y_week, test_size=0.2, shuffle=False)

            _, _, y_trend_train, y_trend_test = train_test_split(X, y_trend, test_size=0.2, shuffle=False)

            # Save processed data
            os.makedirs(os.path.dirname(self.config.processed_data_path), exist_ok=True)

            np.save(self.config.processed_data_path, {
                "X_train": X_train,
                "X_test": X_test,
                "y_day_train": y_day_train,
                "y_day_test": y_day_test,
                "y_week_train": y_week_train,
                "y_week_test": y_week_test,
                "y_trend_train": y_trend_train,
                "y_trend_test": y_trend_test,
            })

            logging.info("Processed data saved")

            # Save scaler
            os.makedirs(os.path.dirname(self.config.scaler_path), exist_ok=True)
            joblib.dump(scaler, self.config.scaler_path)

            logging.info("Scaler saved")

            return self.config.processed_data_path

        except Exception as e:
            logging.error("Error in data transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    print(obj.initiate_data_transformation())