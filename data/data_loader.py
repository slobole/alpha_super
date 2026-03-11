# data_loader.py

import logging
from abc import ABC, abstractmethod
import datetime
import yfinance as yf
import pandas_datareader.data as web
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class DataLoader(ABC):
    @abstractmethod
    def load_data(self, symbol: str, start: datetime.datetime, end: datetime.datetime):
        pass


class StooqDataLoader(DataLoader):
    def load_data(self, symbol: str, start: datetime.datetime, end: datetime.datetime):
        logger.debug(f"Attempting to load Stooq data for {symbol} from {start.date()} to {end.date()}")
        try:
            data = web.DataReader(symbol, 'stooq', start, end)
            logger.info(f"Successfully loaded Stooq data for {symbol}")
            return data.sort_index(ascending=True)
        except Exception as e:
            logger.error(f"Error loading data from Stooq for {symbol}: {e}")
            return None


class YahooDataLoader(DataLoader):
    def load_data(self, symbol: str, start: datetime.datetime, end: datetime.datetime):
        logger.debug(f"Attempting to load Yahoo Finance data for {symbol} from {start.date()} to {end.date()}")
        try:
            data = yf.download(symbol, auto_adjust=True, start=start, end=end)
            if isinstance(data.columns, pd.MultiIndex):
                data = data.xs(symbol, axis=1, level='Ticker')
            logger.info(f"Successfully loaded Yahoo data for {symbol}")
            return data.sort_index(ascending=True)
        except Exception as e:
            logger.error(f"Error loading data from Yahoo for {symbol}: {e}")
            return None


if __name__ == "__main__":
    stooq_loader = StooqDataLoader()
    yahoo_loader = YahooDataLoader()

    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2023, 10, 1)
    symbol = "AAPL"

    stooq_data = stooq_loader.load_data(symbol, start_date, end_date)
    if stooq_data is not None:
        logger.info(f"Stooq Data Sample:\n{stooq_data.head()}")

    yahoo_data = yahoo_loader.load_data(symbol, start_date, end_date)
    if yahoo_data is not None:
        logger.info(f"Yahoo Data Sample:\n{yahoo_data.head()}")
