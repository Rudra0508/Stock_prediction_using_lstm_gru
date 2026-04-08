import os
import time
import hashlib
import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.exception import DataIngestionException, InsufficientDataException
from src.utils.config_loader import AppConfig, load_config


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


class DataIngestion:
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger(
            self.__class__.__name__,
            log_dir=config.paths.logs_dir,
        )
        os.makedirs(config.paths.raw_data_dir, exist_ok=True)

    def _raw_path(self, ticker: str) -> str:
        safe = ticker.replace(".", "_")
        return os.path.join(self.config.paths.raw_data_dir, f"{safe}.csv")

    def _checksum(self, df: pd.DataFrame) -> str:
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    def _is_cached(self, ticker: str, df: pd.DataFrame) -> bool:
        path = self._raw_path(ticker)
        if not os.path.exists(path):
            return False
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
        return self._checksum(existing) == self._checksum(df)

    def _fetch_single(self, ticker: str) -> pd.DataFrame:
        cfg = self.config.stocks
        ing = self.config.ingestion

        for attempt in range(1, ing.max_retries + 1):
            try:
                self.logger.debug(f"Fetching {ticker} (attempt {attempt})")
                df = yf.download(
                    ticker,
                    start=cfg.start_date,
                    end=cfg.end_date,
                    interval=cfg.interval,
                    auto_adjust=True,
                    progress=False,
                )

                if df.empty:
                    raise DataIngestionException(f"Empty response for {ticker}")

                if df.columns.nlevels > 1:
                    df.columns = df.columns.get_level_values(0)

                missing = REQUIRED_COLUMNS - set(df.columns)
                if missing:
                    raise DataIngestionException(
                        f"Missing columns {missing} for {ticker}"
                    )

                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"
                df = df[sorted(REQUIRED_COLUMNS)]
                df = df.dropna(how="all")

                if len(df) < ing.min_rows_threshold:
                    raise InsufficientDataException(
                        f"{ticker} has only {len(df)} rows (min={ing.min_rows_threshold})"
                    )

                return df

            except (DataIngestionException, InsufficientDataException):
                raise

            except Exception as e:
                self.logger.warning(f"Attempt {attempt} failed for {ticker}: {e}")
                if attempt < ing.max_retries:
                    time.sleep(ing.retry_delay_seconds)
                else:
                    raise DataIngestionException(
                        f"All {ing.max_retries} attempts failed for {ticker}", e
                    )

    def _save(self, ticker: str, df: pd.DataFrame) -> None:
        path = self._raw_path(ticker)
        df.to_csv(path)
        self.logger.info(f"Saved {ticker} → {path} ({len(df)} rows)")

    def ingest_ticker(self, ticker: str, force: bool = False) -> Optional[pd.DataFrame]:
        path = self._raw_path(ticker)

        if not force and os.path.exists(path):
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            self.logger.info(f"Cache hit for {ticker} ({len(existing)} rows). Skipping fetch.")
            return existing

        try:
            df = self._fetch_single(ticker)
            self._save(ticker, df)
            return df

        except (DataIngestionException, InsufficientDataException) as e:
            self.logger.error(str(e))
            return None

    def ingest_all(self, force: bool = False) -> dict[str, pd.DataFrame]:
        results = {}
        tickers = self.config.stocks.tickers

        self.logger.info(f"Starting ingestion for {len(tickers)} stocks.")
        for ticker in tickers:
            df = self.ingest_ticker(ticker, force=force)
            if df is not None:
                results[ticker] = df
            else:
                self.logger.warning(f"Skipping {ticker} due to ingestion failure.")

        self.logger.info(
            f"Ingestion complete. Success: {len(results)}/{len(tickers)}"
        )
        return results

    def build_combined_dataset(
        self, stock_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        frames = []
        for ticker, df in stock_data.items():
            temp = df.copy()
            temp["ticker"] = ticker
            temp = temp.reset_index()
            frames.append(temp)

        if not frames:
            raise DataIngestionException("No data available to build combined dataset.")

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["ticker", "Date"]).reset_index(drop=True)

        out_path = self.config.paths.combined_data_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        combined.to_csv(out_path, index=False)

        self.logger.info(
            f"Combined dataset saved → {out_path} "
            f"({len(combined)} rows, {combined['ticker'].nunique()} stocks)"
        )
        return combined

    def run(self, force: bool = False) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        stock_data = self.ingest_all(force=force)
        combined = self.build_combined_dataset(stock_data)
        return stock_data, combined


def run_ingestion(config_path: str = "config/config.yaml", force: bool = False):
    config = load_config(config_path)
    ingestion = DataIngestion(config)
    stock_data, combined = ingestion.run(force=force)
    return stock_data, combined


if __name__ == "__main__":
    stock_data, combined = run_ingestion()
    print(f"\nStocks fetched: {list(stock_data.keys())}")
    print(f"Combined shape: {combined.shape}")
    print(combined.head())
