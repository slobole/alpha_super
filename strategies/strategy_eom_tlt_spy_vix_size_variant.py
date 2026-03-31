"""
TLT / SPY end-of-month relative-strength variant with VIX-sized SPY exposure.

TL;DR: Keep the original monthly `SPY` versus `TLT` signal and trade map, but
whenever the strategy would otherwise go long `SPY`, scale only that `SPY`
weight by a capped inverse-VIX multiplier.

Core formulas
-------------
Let the ordered trading dates in month m be:

    D_m = {d_{m,1}, d_{m,2}, ..., d_{m,N_m}}

Define:

    L = 15   # signal window
    H = 5    # late-month hold window
    K = 5    # next-month pair hold window

First-15 trading-day relative-strength signal:

    r_spy_m^{(15)}
        = Close_SPY_{d_{m,L}} / Open_SPY_{d_{m,1}} - 1

    r_tlt_m^{(15)}
        = Close_TLT_{d_{m,L}} / Open_TLT_{d_{m,1}} - 1

    rel_m
        = r_spy_m^{(15)} - r_tlt_m^{(15)}

For any entry bar e where the original strategy would go long `SPY`, define the
latest prior-close VIX decision bar:

    d_vix(e) = max { d in trading_calendar : d < e }

    vix_e = Close_VIX_{d_vix(e)}

SPY size multiplier:

    raw_scale_e
        = vix_ref / vix_e

    spy_weight_scale_e
        = clip(raw_scale_e, scale_floor, scale_cap)

with defaults:

    vix_ref     = 20.0
    scale_floor = 0.50
    scale_cap   = 1.50

Applied weights:

    w_spy_reversal_e
        = w_reversal_base * spy_weight_scale_e

    w_spy_pair_e
        = w_pair_base * spy_weight_scale_e

    w_tlt_pair_e
        = -w_pair_base

Default base weights:

    w_reversal_base = 1.0
    w_pair_base     = 0.5

Data-source note
----------------
The user referred to `VIX.CLOSE`. This research file uses Yahoo data, so the
VIX proxy is Yahoo's `^VIX` close series.

Execution-model note
--------------------
This file is a research path, not a `run_daily()` engine strategy, because the
requested exits occur at:

    Close_{month_end}
    Close_{next_month_day_5}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from strategies.strategy_eom_trend_ibit import (
    attach_symbol_level,
    load_yahoo_ohlcv_df,
)
from strategies.strategy_eom_tlt_spy_variant import (
    EomTltSpyVariantResearchStrategy,
    build_daily_target_weight_df,
    get_completed_month_period_set,
    run_variant_research_backtest,
)
from alpha.engine.strategy import Strategy
from alpha.engine.report import save_results


@dataclass(frozen=True)
class EomTltSpyVixSizeVariantConfig:
    trade_symbol_list: tuple[str, ...] = ("SPY", "TLT")
    vix_symbol_str: str = "^VIX"
    benchmark_list: tuple[str, ...] = ("SPY",)
    signal_day_count_int: int = 15
    eom_hold_day_count_int: int = 5
    bom_pair_hold_day_count_int: int = 5
    reversal_weight_float: float = 1.0
    pair_abs_weight_float: float = 0.5
    vix_reference_float: float = 20.0
    spy_weight_scale_floor_float: float = 0.50
    spy_weight_scale_cap_float: float = 1.50
    start_date_str: str = "2003-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if set(self.trade_symbol_list) != {"SPY", "TLT"}:
            raise ValueError("trade_symbol_list must contain exactly SPY and TLT.")
        if self.vix_symbol_str in set(self.trade_symbol_list):
            raise ValueError("vix_symbol_str must differ from SPY and TLT.")
        if self.signal_day_count_int <= 0:
            raise ValueError("signal_day_count_int must be positive.")
        if self.eom_hold_day_count_int <= 0:
            raise ValueError("eom_hold_day_count_int must be positive.")
        if self.bom_pair_hold_day_count_int <= 0:
            raise ValueError("bom_pair_hold_day_count_int must be positive.")
        if self.reversal_weight_float <= 0.0:
            raise ValueError("reversal_weight_float must be positive.")
        if self.pair_abs_weight_float <= 0.0:
            raise ValueError("pair_abs_weight_float must be positive.")
        if not np.isfinite(self.vix_reference_float) or self.vix_reference_float <= 0.0:
            raise ValueError("vix_reference_float must be positive and finite.")
        if not np.isfinite(self.spy_weight_scale_floor_float) or self.spy_weight_scale_floor_float <= 0.0:
            raise ValueError("spy_weight_scale_floor_float must be positive and finite.")
        if not np.isfinite(self.spy_weight_scale_cap_float) or self.spy_weight_scale_cap_float <= 0.0:
            raise ValueError("spy_weight_scale_cap_float must be positive and finite.")
        if self.spy_weight_scale_floor_float > self.spy_weight_scale_cap_float:
            raise ValueError("spy_weight_scale_floor_float must be <= spy_weight_scale_cap_float.")

    @property
    def data_symbol_list(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*self.trade_symbol_list, self.vix_symbol_str, *self.benchmark_list]))

    @property
    def tradeable_symbol_list(self) -> tuple[str, ...]:
        return self.trade_symbol_list


DEFAULT_CONFIG = EomTltSpyVixSizeVariantConfig()


def load_pricing_data_df(
    config: EomTltSpyVixSizeVariantConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    price_df_list: list[pd.DataFrame] = []

    for symbol_str in config.data_symbol_list:
        price_df = load_yahoo_ohlcv_df(
            symbol_str=symbol_str,
            start_date_str=config.start_date_str,
            end_date_str=config.end_date_str,
        )
        if len(price_df) == 0:
            raise RuntimeError(f"{symbol_str} returned no data.")
        price_df_list.append(attach_symbol_level(price_df=price_df, symbol_str=symbol_str))

    pricing_data_df = pd.concat(price_df_list, axis=1).sort_index()
    return pricing_data_df


def compute_spy_weight_scale_float(
    vix_close_float: float,
    config: EomTltSpyVixSizeVariantConfig = DEFAULT_CONFIG,
) -> float:
    if not np.isfinite(vix_close_float) or vix_close_float <= 0.0:
        return 1.0

    raw_spy_weight_scale_float = float(config.vix_reference_float / vix_close_float)
    clipped_spy_weight_scale_float = float(
        np.clip(
            raw_spy_weight_scale_float,
            float(config.spy_weight_scale_floor_float),
            float(config.spy_weight_scale_cap_float),
        )
    )
    return clipped_spy_weight_scale_float


def build_month_signal_df(
    open_price_df: pd.DataFrame,
    close_price_df: pd.DataFrame,
    config: EomTltSpyVixSizeVariantConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    trading_index = pd.DatetimeIndex(close_price_df.index).sort_values()
    month_group_map = pd.Series(trading_index, index=trading_index).groupby(trading_index.to_period("M")).groups
    completed_month_period_set = get_completed_month_period_set(trading_index)

    month_signal_row_list: list[dict[str, object]] = []
    for month_period, month_date_index in month_group_map.items():
        month_trading_index = pd.DatetimeIndex(month_date_index).sort_values()
        month_observation_count_int = len(month_trading_index)
        if month_observation_count_int < config.signal_day_count_int:
            continue

        month_start_bar_ts = pd.Timestamp(month_trading_index[0])
        signal_end_bar_ts = pd.Timestamp(month_trading_index[config.signal_day_count_int - 1])

        spy_first15_return_float = float(
            close_price_df.loc[signal_end_bar_ts, "SPY"] / open_price_df.loc[month_start_bar_ts, "SPY"] - 1.0
        )
        tlt_first15_return_float = float(
            close_price_df.loc[signal_end_bar_ts, "TLT"] / open_price_df.loc[month_start_bar_ts, "TLT"] - 1.0
        )
        rel_15_return_float = float(spy_first15_return_float - tlt_first15_return_float)
        spy_outperformed_bool = bool(rel_15_return_float > 0.0)
        tlt_outperformed_bool = bool(rel_15_return_float < 0.0)

        reversal_asset_str = None
        reversal_entry_bar_ts = pd.NaT
        reversal_exit_bar_ts = pd.NaT
        reversal_signed_weight_float = np.nan
        reversal_vix_decision_bar_ts = pd.NaT
        reversal_vix_close_float = np.nan
        reversal_spy_weight_scale_float = np.nan

        # *** CRITICAL*** The reversal leg must enter strictly after the
        # first-15-day signal window. If N_m < L + H, the month is skipped.
        if (
            month_period in completed_month_period_set
            and month_observation_count_int >= config.signal_day_count_int + config.eom_hold_day_count_int
        ):
            reversal_entry_bar_ts = pd.Timestamp(month_trading_index[month_observation_count_int - config.eom_hold_day_count_int])
            reversal_exit_bar_ts = pd.Timestamp(month_trading_index[-1])
            if reversal_entry_bar_ts > signal_end_bar_ts:
                if spy_outperformed_bool:
                    reversal_asset_str = "TLT"
                    reversal_signed_weight_float = float(config.reversal_weight_float)
                elif tlt_outperformed_bool:
                    reversal_asset_str = "SPY"
                    # *** CRITICAL*** The SPY size must use the latest close
                    # strictly before the reversal entry open, never the same
                    # bar as the open fill.
                    reversal_vix_decision_bar_ts = pd.Timestamp(
                        month_trading_index[month_observation_count_int - config.eom_hold_day_count_int - 1]
                    )
                    reversal_vix_close_float = float(
                        close_price_df.loc[reversal_vix_decision_bar_ts, config.vix_symbol_str]
                    )
                    reversal_spy_weight_scale_float = compute_spy_weight_scale_float(
                        vix_close_float=reversal_vix_close_float,
                        config=config,
                    )
                    reversal_signed_weight_float = float(
                        config.reversal_weight_float * reversal_spy_weight_scale_float
                    )
                else:
                    reversal_entry_bar_ts = pd.NaT
                    reversal_exit_bar_ts = pd.NaT

        pair_entry_bar_ts = pd.NaT
        pair_exit_bar_ts = pd.NaT
        pair_long_signed_weight_float = np.nan
        pair_vix_decision_bar_ts = pd.NaT
        pair_vix_close_float = np.nan
        pair_spy_weight_scale_float = np.nan
        next_month_period = month_period + 1

        # *** CRITICAL*** The early-next-month pair leg uses only the prior
        # month's completed first-15-day signal. The SPY sizing uses the last
        # fully observed close before the next-month open entry.
        if spy_outperformed_bool and next_month_period in month_group_map:
            next_month_trading_index = pd.DatetimeIndex(month_group_map[next_month_period]).sort_values()
            if len(next_month_trading_index) >= config.bom_pair_hold_day_count_int:
                pair_entry_bar_ts = pd.Timestamp(next_month_trading_index[0])
                pair_exit_bar_ts = pd.Timestamp(next_month_trading_index[config.bom_pair_hold_day_count_int - 1])
                pair_vix_decision_bar_ts = pd.Timestamp(month_trading_index[-1])
                pair_vix_close_float = float(close_price_df.loc[pair_vix_decision_bar_ts, config.vix_symbol_str])
                pair_spy_weight_scale_float = compute_spy_weight_scale_float(
                    vix_close_float=pair_vix_close_float,
                    config=config,
                )
                pair_long_signed_weight_float = float(
                    config.pair_abs_weight_float * pair_spy_weight_scale_float
                )

        month_signal_row_list.append(
            {
                "signal_month_period": month_period,
                "month_observation_count_int": int(month_observation_count_int),
                "month_start_bar_ts": month_start_bar_ts,
                "signal_end_bar_ts": signal_end_bar_ts,
                "spy_first15_return_float": spy_first15_return_float,
                "tlt_first15_return_float": tlt_first15_return_float,
                "rel_15_return_float": rel_15_return_float,
                "spy_outperformed_bool": spy_outperformed_bool,
                "tlt_outperformed_bool": tlt_outperformed_bool,
                "reversal_asset_str": reversal_asset_str,
                "reversal_entry_bar_ts": reversal_entry_bar_ts,
                "reversal_exit_bar_ts": reversal_exit_bar_ts,
                "reversal_signed_weight_float": reversal_signed_weight_float,
                "reversal_vix_decision_bar_ts": reversal_vix_decision_bar_ts,
                "reversal_vix_close_float": reversal_vix_close_float,
                "reversal_spy_weight_scale_float": reversal_spy_weight_scale_float,
                "pair_entry_bar_ts": pair_entry_bar_ts,
                "pair_exit_bar_ts": pair_exit_bar_ts,
                "pair_long_signed_weight_float": pair_long_signed_weight_float,
                "pair_vix_decision_bar_ts": pair_vix_decision_bar_ts,
                "pair_vix_close_float": pair_vix_close_float,
                "pair_spy_weight_scale_float": pair_spy_weight_scale_float,
            }
        )

    month_signal_df = pd.DataFrame(month_signal_row_list)
    if len(month_signal_df) == 0:
        return pd.DataFrame(
            columns=[
                "signal_month_period",
                "month_observation_count_int",
                "month_start_bar_ts",
                "signal_end_bar_ts",
                "spy_first15_return_float",
                "tlt_first15_return_float",
                "rel_15_return_float",
                "spy_outperformed_bool",
                "tlt_outperformed_bool",
                "reversal_asset_str",
                "reversal_entry_bar_ts",
                "reversal_exit_bar_ts",
                "reversal_signed_weight_float",
                "reversal_vix_decision_bar_ts",
                "reversal_vix_close_float",
                "reversal_spy_weight_scale_float",
                "pair_entry_bar_ts",
                "pair_exit_bar_ts",
                "pair_long_signed_weight_float",
                "pair_vix_decision_bar_ts",
                "pair_vix_close_float",
                "pair_spy_weight_scale_float",
            ]
        )

    month_signal_df["signal_month_period_str"] = month_signal_df["signal_month_period"].astype(str)
    month_signal_df = month_signal_df.set_index("signal_month_period_str", drop=True)
    return month_signal_df


def build_trade_leg_plan_df(
    month_signal_df: pd.DataFrame,
    config: EomTltSpyVixSizeVariantConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    trade_leg_row_list: list[dict[str, object]] = []
    trade_id_int = 0

    for _, signal_row_ser in month_signal_df.iterrows():
        reversal_asset_str = signal_row_ser["reversal_asset_str"]
        reversal_entry_bar_ts = signal_row_ser["reversal_entry_bar_ts"]
        reversal_exit_bar_ts = signal_row_ser["reversal_exit_bar_ts"]
        reversal_signed_weight_float = signal_row_ser["reversal_signed_weight_float"]
        pair_entry_bar_ts = signal_row_ser["pair_entry_bar_ts"]
        pair_exit_bar_ts = signal_row_ser["pair_exit_bar_ts"]
        pair_long_signed_weight_float = signal_row_ser["pair_long_signed_weight_float"]

        if (
            pd.notna(reversal_entry_bar_ts)
            and pd.notna(reversal_exit_bar_ts)
            and reversal_asset_str is not None
            and np.isfinite(float(reversal_signed_weight_float))
        ):
            trade_id_int += 1
            trade_leg_row_list.append(
                {
                    "trade_id_int": trade_id_int,
                    "leg_type_str": "reversal",
                    "signal_month_period_str": signal_row_ser.name,
                    "asset_str": str(reversal_asset_str),
                    "signed_weight_float": float(reversal_signed_weight_float),
                    "entry_bar_ts": pd.Timestamp(reversal_entry_bar_ts),
                    "exit_bar_ts": pd.Timestamp(reversal_exit_bar_ts),
                    "rel_15_return_float": float(signal_row_ser["rel_15_return_float"]),
                }
            )

        if (
            pd.notna(pair_entry_bar_ts)
            and pd.notna(pair_exit_bar_ts)
            and np.isfinite(float(pair_long_signed_weight_float))
        ):
            trade_id_int += 1
            trade_leg_row_list.append(
                {
                    "trade_id_int": trade_id_int,
                    "leg_type_str": "pair_long_spy",
                    "signal_month_period_str": signal_row_ser.name,
                    "asset_str": "SPY",
                    "signed_weight_float": float(pair_long_signed_weight_float),
                    "entry_bar_ts": pd.Timestamp(pair_entry_bar_ts),
                    "exit_bar_ts": pd.Timestamp(pair_exit_bar_ts),
                    "rel_15_return_float": float(signal_row_ser["rel_15_return_float"]),
                }
            )
            trade_id_int += 1
            trade_leg_row_list.append(
                {
                    "trade_id_int": trade_id_int,
                    "leg_type_str": "pair_short_tlt",
                    "signal_month_period_str": signal_row_ser.name,
                    "asset_str": "TLT",
                    "signed_weight_float": float(-config.pair_abs_weight_float),
                    "entry_bar_ts": pd.Timestamp(pair_entry_bar_ts),
                    "exit_bar_ts": pd.Timestamp(pair_exit_bar_ts),
                    "rel_15_return_float": float(signal_row_ser["rel_15_return_float"]),
                }
            )

    trade_leg_plan_df = pd.DataFrame(trade_leg_row_list)
    if len(trade_leg_plan_df) == 0:
        return pd.DataFrame(
            columns=[
                "trade_id_int",
                "leg_type_str",
                "signal_month_period_str",
                "asset_str",
                "signed_weight_float",
                "entry_bar_ts",
                "exit_bar_ts",
                "rel_15_return_float",
            ]
        )

    trade_leg_plan_df = trade_leg_plan_df.sort_values(["entry_bar_ts", "trade_id_int"]).set_index("trade_id_int", drop=True)
    trade_leg_plan_df.index.name = "trade_id_int"
    return trade_leg_plan_df


class EomTltSpyVixSizeVariantResearchStrategy(EomTltSpyVariantResearchStrategy):
    """
    Research-only container for the VIX-sized SPY execution variant.
    """


if __name__ == "__main__":
    config = DEFAULT_CONFIG

    pricing_data_df = load_pricing_data_df(config=config)
    open_price_df = pricing_data_df.xs("Open", axis=1, level=1)[list(config.data_symbol_list)].astype(float)
    close_price_df = pricing_data_df.xs("Close", axis=1, level=1)[list(config.data_symbol_list)].astype(float)

    month_signal_df = build_month_signal_df(
        open_price_df=open_price_df,
        close_price_df=close_price_df,
        config=config,
    )
    trade_leg_plan_df = build_trade_leg_plan_df(
        month_signal_df=month_signal_df,
        config=config,
    )
    daily_target_weight_df = build_daily_target_weight_df(
        trading_index=pricing_data_df.index,
        trade_leg_plan_df=trade_leg_plan_df,
        asset_list=config.tradeable_symbol_list,
    )

    strategy = EomTltSpyVixSizeVariantResearchStrategy(
        name="strategy_eom_tlt_spy_vix_size_variant_research",
        benchmarks=config.benchmark_list,
        trade_leg_plan_df=trade_leg_plan_df,
        daily_target_weight_df=daily_target_weight_df,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )
    run_variant_research_backtest(
        strategy=strategy,
        pricing_data_df=pricing_data_df,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("Month signal preview:")
    display(month_signal_df.head())
    print("Trade leg preview:")
    display(trade_leg_plan_df.head())
    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
