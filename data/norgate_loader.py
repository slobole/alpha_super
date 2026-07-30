from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from tqdm.auto import tqdm

from data import norgate_snapshot_store
from data.norgate_snapshot_store import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    TOTALRETURN_ADJUSTMENT_STR,
    build_data_source_metadata_dict,
    is_snapshot_mode_enabled_bool,
    load_latest_snapshot_session_label_ts,
    use_norgate_data_profile,
)


# *** CRITICAL*** Norgate's TOTALRETURN price adjustment only affects stocks
# and ETFs (dividend back-adjustment). An index symbol such as $SPX carries no
# dividends, so requesting TOTALRETURN on it silently returns the PRICE index —
# the genuinely total-return index is a separate data symbol. Strategies that
# benchmark against an index should load the mapped TR data symbol below and
# keep the familiar label via ``Strategy._benchmark_data_symbol_map_dict``.
INDEX_TOTALRETURN_DATA_SYMBOL_MAP_DICT: dict[str, str] = {"$SPX": "$SPXTR"}


class _NorgateDataProxy:
    def __getattr__(self, name_str: str):
        return getattr(_load_direct_norgate_module(), name_str)


norgatedata = _NorgateDataProxy()


def _load_direct_norgate_module():
    try:
        import norgatedata as norgatedata_module
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError(
            "Direct Norgate mode requires the norgatedata package and local updater. "
            "Set ALPHA_USE_NORGATE_SNAPSHOT_BOOL=true on VPS machines."
        ) from exc
    return norgatedata_module


def _direct_adjustment_type_obj(adjustment_str: str):
    norgatedata_module = _load_direct_norgate_module()
    normalized_adjustment_str = norgate_snapshot_store.normalize_adjustment_str(adjustment_str)
    if normalized_adjustment_str == TOTALRETURN_ADJUSTMENT_STR:
        return norgatedata_module.StockPriceAdjustmentType.TOTALRETURN
    return norgatedata_module.StockPriceAdjustmentType.CAPITALSPECIAL


def load_price_timeseries(
    symbol_str: str,
    *,
    adjustment_str: str = CAPITALSPECIAL_ADJUSTMENT_STR,
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    data_profile_str: str | None = None,
) -> pd.DataFrame:
    if is_snapshot_mode_enabled_bool():
        return norgate_snapshot_store.load_price_timeseries_df(
            symbol_str,
            adjustment_str,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            data_profile_str=data_profile_str,
        )

    norgatedata_module = _load_direct_norgate_module()
    return norgatedata_module.price_timeseries(
        symbol_str,
        stock_price_adjustment_setting=_direct_adjustment_type_obj(adjustment_str),
        padding_setting=norgatedata_module.PaddingType.ALLMARKETDAYS,
        start_date=start_date_str,
        end_date=end_date_str,
        timeseriesformat="pandas-dataframe",
    )


def build_index_constituent_matrix(indexname: str = "S&P 500") -> Tuple[List[str], pd.DataFrame]:
    """
    Builds a survivorship-bias-free universe matrix for backtesting.
    Snapshot mode preserves the same return shape as direct Norgate mode.
    """
    if is_snapshot_mode_enabled_bool():
        return norgate_snapshot_store.load_index_constituent_matrix_df(indexname)

    norgatedata_module = _load_direct_norgate_module()
    symbols = norgatedata_module.watchlist_symbols(f"{indexname} Current & Past")
    calendar = norgatedata_module.price_timeseries("$SPX", timeseriesformat="pandas-dataframe").index
    last_trading_day = calendar[-1]
    universe_df = []

    for symbol in tqdm(symbols, desc="building universe"):
        idx = norgatedata_module.index_constituent_timeseries(
            symbol,
            indexname,
            timeseriesformat="pandas-dataframe",
        )
        if idx["Index Constituent"].sum() > 0:
            idx = idx.rename(columns={"Index Constituent": symbol})
            idx = idx.loc[idx[symbol] == 1]
            if last_trading_day != idx.index[-1]:
                idx = idx.iloc[:-5]
            universe_df.append(idx)

    universe_df = pd.concat(universe_df, axis=1).fillna(0).astype(int).sort_index()
    return symbols, universe_df


def load_raw_prices(
    symbols: List[str],
    benchmarks: List[str],
    start_date: str = "1998-01-01",
    end_date: str = None,
) -> pd.DataFrame:
    """
    Load raw OHLCV data from Norgate or from a validated snapshot.
    The output stays a MultiIndex column DataFrame: (symbol, field).
    """
    benchmark_symbol_set = {str(symbol_str) for symbol_str in benchmarks}
    adjustment_by_symbol_dict = {
        str(symbol_str): (
            TOTALRETURN_ADJUSTMENT_STR
            if str(symbol_str) in benchmark_symbol_set
            else CAPITALSPECIAL_ADJUSTMENT_STR
        )
        for symbol_str in symbols + benchmarks
    }
    if is_snapshot_mode_enabled_bool():
        pricing_data_df = norgate_snapshot_store.load_raw_prices_df(
            symbols=symbols,
            benchmarks=benchmarks,
            start_date_str=start_date,
            end_date_str=end_date,
        )
        pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = (
            adjustment_by_symbol_dict
        )
        return pricing_data_df

    pricing_data = []

    for symbol in tqdm(symbols + benchmarks, desc="loading prices"):
        if symbol in benchmarks:
            adjustment_str = TOTALRETURN_ADJUSTMENT_STR
        else:
            adjustment_str = CAPITALSPECIAL_ADJUSTMENT_STR

        price_df = load_price_timeseries(
            symbol,
            adjustment_str=adjustment_str,
            start_date_str=start_date,
            end_date_str=end_date,
        )

        if len(price_df) == 0:
            continue

        price_df.columns = pd.MultiIndex.from_tuples([(symbol, column_str) for column_str in price_df.columns])
        pricing_data.append(price_df)

    pricing_data_df = pd.concat(pricing_data, axis=1).sort_index()
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = (
        adjustment_by_symbol_dict
    )
    return pricing_data_df
