"""
Defense First VIX Tranche with BTAL and UPRO fallback.

Practical baseline variant:
- Defensive assets: GLD, TLT, UUP, DBC, BTAL
- Fallback risk-on asset: UPRO
- Fallback risk-off state: CASH
- Monthly rebalancing at the first trading day of each month
- Defensive signal: average annualized daily 21d, 63d, 126d, 252d returns
- Absolute momentum filter: keep only assets beating T-bills
- Rank weights: [5, 4, 3, 2, 1] / 15 to top momentum survivors
- VIX tranche logic applies only to leftover fallback weight
- Benchmark: $SPX

Data Requirements:
- Norgate Data subscription
- DTB3 historical rates CSV
- Internet access for FRED VIXCLS data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import norgatedata
import numpy as np
import pandas as pd
from IPython.display import display
from pandas_datareader import data as web

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.strategy_taa_defense_first import (
    DefenseFirstConfig,
    DefenseFirstStrategy,
    load_cash_return_ser,
    load_execution_price_df,
    load_signal_close_df,
    map_month_end_weights_to_rebalance_open_df,
)
from strategies.strategy_taa_defense_first_upro import filter_valid_rebalance_weight_df


@dataclass(frozen=True)
class DefenseFirstPracticalVixConfig(DefenseFirstConfig):
    signal_asset_list: tuple[str, ...] = ("SPY", "GLD", "UUP", "TLT", "DBC", "BTAL")
    momentum_lookback_day_vec: tuple[int, ...] = (21, 63, 126, 252)
    vol_lookback_day_vec: tuple[int, ...] = (10, 15, 20)


PRACTICAL_BTAL_VIX_UPRO_CONFIG = DefenseFirstPracticalVixConfig(
    defensive_asset_list=("GLD", "UUP", "TLT", "DBC", "BTAL"),
    fallback_asset="UPRO",
    rank_weight_vec=(5.0 / 15.0, 4.0 / 15.0, 3.0 / 15.0, 2.0 / 15.0, 1.0 / 15.0),
)



def load_single_signal_close_ser(
    symbol_str: str,
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.Series:
    """
    Load a single TOTALRETURN close series for signal calculations.
    """
    price_df = norgatedata.price_timeseries(
        symbol_str,
        stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.TOTALRETURN,
        padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
        start_date=start_date_str,
        end_date=end_date_str,
        timeseriesformat="pandas-dataframe",
    )
    if len(price_df) == 0:
        raise RuntimeError(f"No signal close data was loaded for symbol {symbol_str}.")

    price_close_ser = price_df["Close"].sort_index()
    price_close_ser.name = symbol_str
    return price_close_ser



def load_vix_close_ser(start_date_str: str, end_date_str: str | None) -> pd.Series:
    """
    Load VIX close data from FRED.
    """
    vix_df = web.DataReader("VIXCLS", "fred", start=start_date_str, end=end_date_str)
    vix_close_ser = pd.to_numeric(vix_df["VIXCLS"], errors="coerce")
    vix_close_ser.name = "VIXCLS"
    return vix_close_ser.sort_index()



def compute_practical_btal_vix_month_end_weight_df(
    signal_close_df: pd.DataFrame,
    spy_close_ser: pd.Series,
    cash_return_ser: pd.Series,
    vix_close_ser: pd.Series,
    config: DefenseFirstPracticalVixConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Compute practical BTAL month-end weights with a VIX tranche overlay.

    Returns:
    - momentum_score_df
    - realized_volatility_df
    - vix_close_aligned_ser
    - safe_count_ser
    - month_end_weight_df
    """
    max_lookback_day_int = int(max(config.momentum_lookback_day_vec))
    if len(signal_close_df) <= max_lookback_day_int:
        raise ValueError("signal_close_df does not contain enough rows for the requested momentum lookbacks.")

    annualized_component_df_list: list[pd.DataFrame] = []
    for lookback_day_int in config.momentum_lookback_day_vec:
        # *** CRITICAL*** This shift uses only information available on or
        # before decision date t. No future bars are referenced.
        lagged_close_df = signal_close_df.shift(lookback_day_int)
        annualized_component_df = (signal_close_df / lagged_close_df) ** (252.0 / float(lookback_day_int)) - 1.0
        annualized_component_df_list.append(annualized_component_df)

    momentum_score_daily_df = sum(annualized_component_df_list) / float(len(annualized_component_df_list))
    momentum_score_df = momentum_score_daily_df.resample("ME").last()
    monthly_cash_return_ser = cash_return_ser.resample("ME").last()

    # *** CRITICAL*** SPY realized volatility is computed from daily SPY returns
    # using only prices available on or before date t.
    spy_return_ser = spy_close_ser.pct_change(fill_method=None)
    realized_volatility_df = pd.DataFrame(index=spy_return_ser.index)
    for lookback_day_int in config.vol_lookback_day_vec:
        realized_volatility_df[f"rv_{lookback_day_int}d"] = (
            spy_return_ser.rolling(lookback_day_int, min_periods=lookback_day_int).std() * np.sqrt(252.0) * 100.0
        )

    vix_close_aligned_ser = vix_close_ser.reindex(realized_volatility_df.index).ffill()
    vix_close_aligned_ser.name = "VIXCLS"

    safe_mask_df = pd.DataFrame(index=realized_volatility_df.index)
    for lookback_day_int in config.vol_lookback_day_vec:
        # *** CRITICAL*** Compare lagged RV to lagged VIX so the month-end signal
        # uses only information available before the next rebalance.
        safe_mask_df[f"safe_{lookback_day_int}d"] = (
            realized_volatility_df[f"rv_{lookback_day_int}d"].shift(1) < vix_close_aligned_ser.shift(1)
        )

    safe_count_ser = safe_mask_df.sum(axis=1).rename("safe_count")
    monthly_safe_count_ser = safe_count_ser.resample("ME").last()

    combined_df = pd.concat([momentum_score_df, monthly_cash_return_ser, monthly_safe_count_ser], axis=1).dropna()

    tradeable_asset_list = list(config.tradeable_asset_list)
    month_end_weight_df = pd.DataFrame(0.0, index=combined_df.index, columns=tradeable_asset_list, dtype=float)
    fallback_scale_map: dict[int, float] = {0: 0.0, 1: 1.0 / 3.0, 2: 2.0 / 3.0, 3: 1.0}

    for decision_date, row_ser in combined_df.iterrows():
        defensive_score_ser = row_ser[list(config.defensive_asset_list)].astype(float)
        eligible_asset_ser = defensive_score_ser[defensive_score_ser > float(row_ser["cash_return"])].sort_values(ascending=False)

        target_weight_ser = pd.Series(0.0, index=tradeable_asset_list, dtype=float)
        assigned_count_int = len(eligible_asset_ser)
        if assigned_count_int > 0:
            rank_weight_vec = np.array(config.rank_weight_vec[:assigned_count_int], dtype=float)
            target_weight_ser.loc[eligible_asset_ser.index] = rank_weight_vec

        unassigned_weight_float = 1.0 - float(target_weight_ser.sum())
        safe_count_int = int(row_ser["safe_count"])
        fallback_weight_float = unassigned_weight_float * fallback_scale_map[safe_count_int]

        if fallback_weight_float > 0.0:
            target_weight_ser.loc[config.fallback_asset] = fallback_weight_float

        if (target_weight_ser < -1e-12).any():
            raise ValueError(f"Target weights must be non-negative on {decision_date}.")
        if target_weight_ser.sum() > 1.0 + 1e-12:
            raise ValueError(f"Target weights must sum to <= 1.0 on {decision_date}.")

        month_end_weight_df.loc[decision_date] = target_weight_ser

    return momentum_score_df, realized_volatility_df, vix_close_aligned_ser, safe_count_ser, month_end_weight_df



def get_defense_first_practical_btal_vix_data(
    config: DefenseFirstPracticalVixConfig = PRACTICAL_BTAL_VIX_UPRO_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Load signal data, execution data, VIX diagnostics, and rebalance weights for
    the practical BTAL + UPRO + VIX tranche strategy.
    """
    signal_close_df = load_signal_close_df(
        symbol_list=config.defensive_asset_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    spy_close_ser = load_single_signal_close_ser(
        symbol_str="SPY",
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    vix_close_ser = load_vix_close_ser(config.start_date_str, config.end_date_str)
    execution_price_df = load_execution_price_df(
        tradeable_asset_list=config.tradeable_asset_list,
        benchmark_list=config.benchmark_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    cash_return_ser = load_cash_return_ser(config.dtb3_csv_path_str)
    momentum_score_df, realized_volatility_df, vix_close_aligned_ser, safe_count_ser, month_end_weight_df = compute_practical_btal_vix_month_end_weight_df(
        signal_close_df=signal_close_df,
        spy_close_ser=spy_close_ser,
        cash_return_ser=cash_return_ser,
        vix_close_ser=vix_close_ser,
        config=config,
    )
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(month_end_weight_df, execution_price_df.index)
    rebalance_weight_df = filter_valid_rebalance_weight_df(rebalance_weight_df, execution_price_df)

    return (
        execution_price_df,
        momentum_score_df,
        realized_volatility_df,
        vix_close_aligned_ser,
        safe_count_ser,
        month_end_weight_df,
        rebalance_weight_df,
    )


if __name__ == "__main__":
    config = PRACTICAL_BTAL_VIX_UPRO_CONFIG

    (
        execution_price_df,
        momentum_score_df,
        realized_volatility_df,
        vix_close_ser,
        safe_count_ser,
        month_end_weight_df,
        rebalance_weight_df,
    ) = get_defense_first_practical_btal_vix_data(config)

    strategy = DefenseFirstStrategy(
        name="DefenseFirstPracticalBTALVixUPROStrategy",
        benchmarks=config.benchmark_list,
        rebalance_weight_df=rebalance_weight_df,
        tradeable_asset_list=config.tradeable_asset_list,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.show_taa_weights_report = True
    strategy.daily_target_weights = rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()

    calendar_idx = execution_price_df.index[execution_price_df.index >= rebalance_weight_df.index[0]]
    run_daily(strategy, execution_price_df, calendar_idx)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(f"Fallback asset: {config.fallback_asset}")
    print(f"Defensive asset list: {config.defensive_asset_list}")
    print(f"Rank weight vector: {config.rank_weight_vec}")
    print(f"Momentum lookback day vector: {config.momentum_lookback_day_vec}")
    print(f"Vol lookback day vector: {config.vol_lookback_day_vec}")
    print("Signal formula: annualized daily 21/63/126/252-day momentum")

    print("First momentum scores:")
    display(momentum_score_df.dropna().head())

    print("First realized volatility rows:")
    display(realized_volatility_df.dropna().head())

    print("First VIX rows:")
    display(vix_close_ser.dropna().head())

    print("First safe-count rows:")
    display(safe_count_ser.dropna().head())

    print("First month-end decisions after VIX tranche:")
    display(month_end_weight_df.head())

    print("First valid rebalance opens:")
    display(rebalance_weight_df.head())

    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
