import os
import sys
import pandas as pd
import yfinance as yf
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data", "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")

        try:
            # Create directory if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Start with ONE stock (scalable to multiple later)
            stocks = ["RELIANCE.NS"]

            all_data = []

            logging.info(f"Fetching data for stocks: {stocks}")

            for stock in stocks:
                logging.info(f"Downloading data for {stock}")

                df = yf.download(stock, period="10y", interval="1d")

                if df.empty:
                    logging.warning(f"No data found for {stock}")
                    continue

                df.reset_index(inplace=True)

                # Add stock column (VERY IMPORTANT)
                df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
                df["stock"] = stock

                all_data.append(df)

            if len(all_data) == 0:
                raise Exception("No data fetched for any stock")

            final_df = pd.concat(all_data, ignore_index=True)

            logging.info(f"Total records fetched: {len(final_df)}")

            # Save to CSV
            final_df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info(f"Data saved at {self.ingestion_config.raw_data_path}")

            return self.ingestion_config.raw_data_path

        except Exception as e:
            logging.error("Exception occurred in data ingestion")
            raise CustomException(e, sys)


# For testing
if __name__ == "__main__":
    obj = DataIngestion()
    print(obj.initiate_data_ingestion())