import os
import sys
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass, field
from typing import Tuple, Dict

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings("ignore")

from utils.logger import logging
from utils.exception import CustomException
from src.components.data_transformation import FEATURE_COLUMNS, TARGET_NEXT_DAY, TARGET_NEXT_WEEK, TARGET_TREND


@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join("artifacts", "models")
    sequence_length: int = 60
    forecast_horizon_week: int = 5
    lstm_units: Tuple[int, int] = (128, 64)
    gru_units: Tuple[int, int] = (128, 64)
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 40
    patience: int = 5
    validation_split: float = 0.1
    train_split: float = 0.8


class SequenceBuilder:

    @staticmethod
    def build_sequences(features, targets, sequence_length):
        X, y = [], []
        for i in range(sequence_length, len(features)):
            X.append(features[i - sequence_length: i])
            y.append(targets[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def train_test_split_sequential(X, y, train_split):
        split_idx = int(len(X) * train_split)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


class BaseModelBuilder:

    @staticmethod
    def build_lstm(input_shape, units, dropout, name):
        inputs = Input(shape=input_shape, name=f"{name}_input")
        x = LSTM(units[0], return_sequences=True, name=f"{name}_lstm1")(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = LSTM(units[1], return_sequences=False, name=f"{name}_lstm2")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        outputs = Dense(32, activation="relu", name=f"{name}_dense")(x)
        return Model(inputs, outputs, name=name)

    @staticmethod
    def build_gru(input_shape, units, dropout, name):
        inputs = Input(shape=input_shape, name=f"{name}_input")
        x = GRU(units[0], return_sequences=True, name=f"{name}_gru1")(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = GRU(units[1], return_sequences=False, name=f"{name}_gru2")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        outputs = Dense(32, activation="relu", name=f"{name}_dense")(x)
        return Model(inputs, outputs, name=name)


class HybridEnsembleBuilder:

    @staticmethod
    def build_regression_model(input_shape, lstm_units, gru_units, dropout, task_name, learning_rate):
        lstm_base = BaseModelBuilder.build_lstm(input_shape, lstm_units, dropout, f"lstm_{task_name}")
        gru_base = BaseModelBuilder.build_gru(input_shape, gru_units, dropout, f"gru_{task_name}")

        shared_input = Input(shape=input_shape)
        lstm_out = lstm_base(shared_input)
        gru_out = gru_base(shared_input)

        merged = Concatenate()([lstm_out, gru_out])
        x = Dense(64, activation="relu")(merged)
        x = Dropout(dropout)(x)
        x = Dense(32, activation="relu")(x)
        output = Dense(1, activation="linear")(x)

        model = Model(inputs=shared_input, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),  # FIXED
            loss="mse",
            metrics=["mae"]
        )
        return model

    @staticmethod
    def build_classification_model(input_shape, lstm_units, gru_units, dropout, learning_rate):
        lstm_base = BaseModelBuilder.build_lstm(input_shape, lstm_units, dropout, "lstm_trend")
        gru_base = BaseModelBuilder.build_gru(input_shape, gru_units, dropout, "gru_trend")

        shared_input = Input(shape=input_shape)
        lstm_out = lstm_base(shared_input)
        gru_out = gru_base(shared_input)

        merged = Concatenate()([lstm_out, gru_out])
        x = Dense(64, activation="relu")(merged)
        x = Dropout(dropout)(x)
        x = Dense(32, activation="relu")(x)
        output = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=shared_input, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),  # FIXED
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model


class ModelTrainer:

    def __init__(self, config=ModelTrainerConfig()):
        self.config = config

    def _get_callbacks(self, path):
        return [
            EarlyStopping(monitor="val_loss", patience=self.config.patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
            ModelCheckpoint(path, save_best_only=True),
        ]

    def _train_single_task(self, model, X_train, y_train, X_val, y_val, path, task_name):
        logging.info(f"Training {task_name}")

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=self._get_callbacks(path),
            verbose=1   # FIXED
        )

        return model

    def train_for_ticker(self, df, ticker):
        logging.info(f"Training {ticker}")

        df = df[df["Ticker"] == ticker].sort_values("Date")

        X = df[FEATURE_COLUMNS].values
        y_nd = df[TARGET_NEXT_DAY].values
        y_nw = df[TARGET_NEXT_WEEK].values
        y_tr = df[TARGET_TREND].values

        X_nd, y_nd = SequenceBuilder.build_sequences(X, y_nd, self.config.sequence_length)
        X_nw, y_nw = SequenceBuilder.build_sequences(X, y_nw, self.config.sequence_length)
        X_tr, y_tr = SequenceBuilder.build_sequences(X, y_tr, self.config.sequence_length)

        split = self.config.train_split

        # Next Day
        X_trn, X_val, y_trn, y_val = SequenceBuilder.train_test_split_sequential(X_nd, y_nd, split)
        model_nd = HybridEnsembleBuilder.build_regression_model(
            (X_nd.shape[1], X_nd.shape[2]),
            self.config.lstm_units,
            self.config.gru_units,
            self.config.dropout_rate,
            "next_day",
            self.config.learning_rate
        )
        self._train_single_task(model_nd, X_trn, y_trn, X_val, y_val, "model_next_day.keras", "NextDay")

        # Next Week
        X_trn, X_val, y_trn, y_val = SequenceBuilder.train_test_split_sequential(X_nw, y_nw, split)
        model_nw = HybridEnsembleBuilder.build_regression_model(
            (X_nw.shape[1], X_nw.shape[2]),
            self.config.lstm_units,
            self.config.gru_units,
            self.config.dropout_rate,
            "next_week",
            self.config.learning_rate
        )
        self._train_single_task(model_nw, X_trn, y_trn, X_val, y_val, "model_next_week.keras", "NextWeek")

        # Trend
        X_trn, X_val, y_trn, y_val = SequenceBuilder.train_test_split_sequential(X_tr, y_tr, split)
        model_tr = HybridEnsembleBuilder.build_classification_model(
            (X_tr.shape[1], X_tr.shape[2]),
            self.config.lstm_units,
            self.config.gru_units,
            self.config.dropout_rate,
            self.config.learning_rate
        )
        self._train_single_task(model_tr, X_trn, y_trn, X_val, y_val, "model_trend.keras", "Trend")

        logging.info(f"Finished {ticker}")

    def initiate_model_training(self, processed_data_path: str):
            logging.info("Starting model training pipeline")
            try:
                df = pd.read_csv(processed_data_path, parse_dates=["Date"])
                logging.info(f"Loaded processed data: {df.shape}")

                tickers = df["Ticker"].unique().tolist()
                logging.info(f"Tickers to train: {tickers}")

                os.makedirs(self.config.model_dir, exist_ok=True)

                for ticker in tickers:
                    self.train_for_ticker(df, ticker)

                logging.info("Model training complete for all tickers")

            except Exception as e:
                raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training(
        os.path.join("artifacts", "data", "processed_data.csv")
        )