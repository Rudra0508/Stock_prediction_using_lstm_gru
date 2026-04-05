import os
import sys
import numpy as np
import joblib
import tensorflow as tf

from src.logger import logging
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        self.model_dir = os.path.join("artifacts", "models")
        self.scaler_path = os.path.join("artifacts", "scaler", "scaler.pkl")

    def load_models(self):
        models = {
            "lstm_day": tf.keras.models.load_model(os.path.join(self.model_dir, "lstm_next_day.h5"), compile=False),
            "gru_day": tf.keras.models.load_model(os.path.join(self.model_dir, "gru_next_day.h5"), compile=False),

            "lstm_week": tf.keras.models.load_model(os.path.join(self.model_dir, "lstm_next_week.h5"), compile=False),
            "gru_week": tf.keras.models.load_model(os.path.join(self.model_dir, "gru_next_week.h5"), compile=False),

            "lstm_trend": tf.keras.models.load_model(os.path.join(self.model_dir, "lstm_trend.h5"), compile=False),
            "gru_trend": tf.keras.models.load_model(os.path.join(self.model_dir, "gru_trend.h5"), compile=False),
        }
        return models

    def predict(self, input_data):
        try:
            logging.info("Starting prediction")

            # Load scaler
            scaler = joblib.load(self.scaler_path)

            # Scale input
            scaled_input = scaler.transform(input_data)

            # Reshape for model (1 sample, 60 timesteps, 5 features)
            scaled_input = scaled_input.reshape(1, 60, 5)

            # Load models
            models = self.load_models()

            # Predictions
            next_day = models["lstm_day"].predict(scaled_input)[0][0]
            next_week = models["lstm_week"].predict(scaled_input)[0]
            trend = models["lstm_trend"].predict(scaled_input)[0][0]

            # Convert trend
            trend_label = "Bullish 📈" if trend > 0.5 else "Bearish 📉"

            return {
                "next_day_price": float(next_day),
                "next_week_prices": next_week.tolist(),
                "trend": trend_label
            }

        except Exception as e:
            raise CustomException(e, sys)


# ---------------- TEST RUN ---------------- #
if __name__ == "__main__":
    pipeline = PredictPipeline()

    # Dummy input (60 days, 5 features)
    data = np.random.rand(60, 5)

    result = pipeline.predict(data)

    print("\n🔮 Prediction Output:")
    print(result)