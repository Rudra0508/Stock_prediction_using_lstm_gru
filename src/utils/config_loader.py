import os
import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class StockConfig:
    tickers: List[str]
    start_date: str
    end_date: str
    interval: str


@dataclass
class PathConfig:
    raw_data_dir: str
    combined_data_path: str
    logs_dir: str


@dataclass
class IngestionConfig:
    max_retries: int
    retry_delay_seconds: int
    min_rows_threshold: int


@dataclass
class AppConfig:
    stocks: StockConfig
    paths: PathConfig
    ingestion: IngestionConfig


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    stocks = StockConfig(**raw["stocks"])
    paths = PathConfig(**raw["paths"])
    ingestion = IngestionConfig(**raw["ingestion"])

    return AppConfig(stocks=stocks, paths=paths, ingestion=ingestion)