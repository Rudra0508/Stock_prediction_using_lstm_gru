import os
import sys
import numpy as np
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

from src.logger import logging
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    processed_data_path: str = os.path.join("artifacts", "data", "processed_data.npy")
    model_dir: str = os.path.join("artifacts", "models")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def load_data(self):
        data = np.load(self.config.processed_data_path, allow_pickle=True).item()

        return (
            data["X_train"], data["X_test"],
            data["y_day_train"], data["y_day_test"],
            data["y_week_train"], data["y_week_test"],
            data["y_trend_train"], data["y_trend_test"]
        )

    # ---------------- LSTM MODEL ---------------- #
    def build_lstm(self, output_size, task="regression"):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(60, 5)),
            Dropout(0.2),
            LSTM(32),
            Dense(32, activation='relu')
        ])

        if task == "classification":
            model.add(Dense(output_size, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(Dense(output_size))
            model.compile(optimizer='adam', loss='mse')

        return model

    # ---------------- GRU MODEL ---------------- #
    def build_gru(self, output_size, task="regression"):
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(60, 5)),
            Dropout(0.2),
            GRU(32),
            Dense(32, activation='relu')
        ])

        if task == "classification":
            model.add(Dense(output_size, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(Dense(output_size))
            model.compile(optimizer='adam', loss='mse')

        return model

    def initiate_model_training(self):
        logging.info("Starting model training")

        try:
            X_train, X_test, y_day_train, y_day_test, \
            y_week_train, y_week_test, \
            y_trend_train, y_trend_test = self.load_data()

            os.makedirs(self.config.model_dir, exist_ok=True)

            # -------- NEXT DAY (REGRESSION) -------- #
            lstm_day = self.build_lstm(1, task="regression")
            lstm_day.fit(X_train, y_day_train, epochs=5, batch_size=32)
            lstm_day.save(os.path.join(self.config.model_dir, "lstm_next_day.h5"))

            gru_day = self.build_gru(1, task="regression")
            gru_day.fit(X_train, y_day_train, epochs=5, batch_size=32)
            gru_day.save(os.path.join(self.config.model_dir, "gru_next_day.h5"))

            # -------- NEXT WEEK (REGRESSION) -------- #
            lstm_week = self.build_lstm(5, task="regression")
            lstm_week.fit(X_train, y_week_train, epochs=5, batch_size=32)
            lstm_week.save(os.path.join(self.config.model_dir, "lstm_next_week.h5"))

            gru_week = self.build_gru(5, task="regression")
            gru_week.fit(X_train, y_week_train, epochs=5, batch_size=32)
            gru_week.save(os.path.join(self.config.model_dir, "gru_next_week.h5"))

            # -------- TREND (CLASSIFICATION) -------- #
            lstm_trend = self.build_lstm(1, task="classification")
            lstm_trend.fit(X_train, y_trend_train, epochs=5, batch_size=32)
            lstm_trend.save(os.path.join(self.config.model_dir, "lstm_trend.h5"))

            gru_trend = self.build_gru(1, task="classification")
            gru_trend.fit(X_train, y_trend_train, epochs=5, batch_size=32)
            gru_trend.save(os.path.join(self.config.model_dir, "gru_trend.h5"))

            logging.info("All models trained and saved successfully")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = ModelTrainer()
    obj.initiate_model_training()