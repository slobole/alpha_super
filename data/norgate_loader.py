import pandas as pd
import norgatedata
from tqdm.auto import tqdm
from typing import Tuple, List


def build_index_constituent_matrix(indexname: str = 'S&P 500') -> Tuple[List[str], pd.DataFrame]:
    """
    Builds a survivorship-bias-free universe matrix for backtesting.
    Loads directly from Norgate every time.
    """
    symbols = norgatedata.watchlist_symbols(f'{indexname} Current & Past')
    calendar = norgatedata.price_timeseries('$SPX', timeseriesformat='pandas-dataframe').index
    last_trading_day = calendar[-1]
    universe_df = []

    for symbol in tqdm(symbols, desc='building universe'):
        idx = norgatedata.index_constituent_timeseries(symbol, indexname, timeseriesformat="pandas-dataframe")
        if idx['Index Constituent'].sum() > 0:
            idx = idx.rename(columns={'Index Constituent': symbol})
            idx = idx.loc[idx[symbol] == 1]
            if last_trading_day != idx.index[-1]:
                idx = idx.iloc[:-5]
            universe_df.append(idx)

    universe_df = pd.concat(universe_df, axis=1).fillna(0).astype(int).sort_index()
    return symbols, universe_df


def load_raw_prices(symbols: List[str], benchmarks: List[str], start_date: str = '1998-01-01', end_date: str = None) -> pd.DataFrame:
    """
    Downloads raw OHLCV data from Norgate for all symbols and benchmarks.
    No strategy-specific features — just raw price data with MultiIndex columns.
    """
    pricing_data = []

    for symbol in tqdm(symbols + benchmarks, desc='loading prices'):
        if symbol in benchmarks:
            adjs = norgatedata.StockPriceAdjustmentType.TOTALRETURN
        else:
            adjs = norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL

        p = norgatedata.price_timeseries(
            symbol,
            stock_price_adjustment_setting=adjs,
            padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
            start_date=start_date,
            end_date=end_date,
            timeseriesformat='pandas-dataframe',
        )

        if len(p) == 0:
            continue

        p.columns = pd.MultiIndex.from_tuples([(symbol, c) for c in p.columns])
        pricing_data.append(p)

    return pd.concat(pricing_data, axis=1).sort_index()
