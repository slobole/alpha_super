"""
TLT / SPY signal, UPRO execution, VIX-sized end-of-month relative-strength variant.

TL;DR: Keep the original monthly `SPY` versus `TLT` signal and trade map, but
whenever the strategy would otherwise go long `SPY`, execute with `UPRO`
instead. Only that long-equity leg is scaled by a capped inverse-VIX rule.

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

UPRO size multiplier:

    raw_scale_e
        = vix_ref / vix_e

    upro_weight_scale_e
        = clip(raw_scale_e, scale_floor, scale_cap)

with defaults:

    vix_ref     = 20.0
    scale_floor = 0.50
    scale_cap   = 1.50

Applied weights:

    w_upro_reversal_e
        = w_reversal_base * upro_weight_scale_e

    w_upro_pair_e
        = w_pair_base * upro_weight_scale_e

    w_tlt_pair_e
        = -w_pair_base

Default base weights:

    w_reversal_base = 1.0
    w_pair_base     = 0.5

Instrument-availability note
----------------------------
`UPRO` starts on 2009-06-25, so the default backtest starts there rather than
pretending earlier `UPRO` history exists.

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

from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from strategies.strategy_eom_trend_ibit import (
    attach_symbol_level,
    build_results_df,
    compute_commission_float,
    load_yahoo_ohlcv_df,
)


@dataclass(frozen=True)
class EomTltSpyUproVixSizeVariantConfig:
    signal_symbol_str: str = "SPY"
    defense_symbol_str: str = "TLT"
    long_symbol_str: str = "UPRO"
    vix_symbol_str: str = "^VIX"
    benchmark_list: tuple[str, ...] = ("SPY", "UPRO")
    signal_day_count_int: int = 15
    eom_hold_day_count_int: int = 5
    bom_pair_hold_day_count_int: int = 5
    reversal_weight_float: float = 1.0
    pair_abs_weight_float: float = 0.5
    vix_reference_float: float = 20.0
    long_weight_scale_floor_float: float = 0.50
    long_weight_scale_cap_float: float = 1.50
    start_date_str: str = "2009-06-25"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        symbol_str_list = [
            self.signal_symbol_str,
            self.defense_symbol_str,
            self.long_symbol_str,
            self.vix_symbol_str,
        ]
        if any(len(symbol_str) == 0 for symbol_str in symbol_str_list):
            raise ValueError("All symbol strings must be non-empty.")
        if len(set(symbol_str_list)) != len(symbol_str_list):
            raise ValueError("signal, defense, long, and VIX symbols must all differ.")
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
        if not np.isfinite(self.long_weight_scale_floor_float) or self.long_weight_scale_floor_float <= 0.0:
            raise ValueError("long_weight_scale_floor_float must be positive and finite.")
        if not np.isfinite(self.long_weight_scale_cap_float) or self.long_weight_scale_cap_float <= 0.0:
            raise ValueError("long_weight_scale_cap_float must be positive and finite.")
        if self.long_weight_scale_floor_float > self.long_weight_scale_cap_float:
            raise ValueError("long_weight_scale_floor_float must be <= long_weight_scale_cap_float.")

    @property
    def data_symbol_list(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [
                    self.signal_symbol_str,
                    self.defense_symbol_str,
                    self.long_symbol_str,
                    self.vix_symbol_str,
                    *self.benchmark_list,
                ]
            )
        )

    @property
    def tradeable_symbol_list(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([self.long_symbol_str, self.defense_symbol_str]))


DEFAULT_CONFIG = EomTltSpyUproVixSizeVariantConfig()


def load_pricing_data_df(
    config: EomTltSpyUproVixSizeVariantConfig = DEFAULT_CONFIG,
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


def get_completed_month_period_set(trading_index: pd.DatetimeIndex) -> set[pd.Period]:
    if len(trading_index) == 0:
        return set()

    completed_month_period_set = set(trading_index.to_period("M").unique().tolist())
    last_available_ts = pd.Timestamp(trading_index[-1])
    expected_business_month_end_ts = pd.Timestamp(last_available_ts + pd.offsets.BMonthEnd(0))
    if expected_business_month_end_ts.normalize() != last_available_ts.normalize():
        completed_month_period_set.discard(last_available_ts.to_period("M"))
    return completed_month_period_set


def compute_long_weight_scale_float(
    vix_close_float: float,
    config: EomTltSpyUproVixSizeVariantConfig = DEFAULT_CONFIG,
) -> float:
    if not np.isfinite(vix_close_float) or vix_close_float <= 0.0:
        return 1.0

    raw_long_weight_scale_float = float(config.vix_reference_float / vix_close_float)
    clipped_long_weight_scale_float = float(
        np.clip(
            raw_long_weight_scale_float,
            float(config.long_weight_scale_floor_float),
            float(config.long_weight_scale_cap_float),
        )
    )
    return clipped_long_weight_scale_float


def build_month_signal_df(
    open_price_df: pd.DataFrame,
    close_price_df: pd.DataFrame,
    config: EomTltSpyUproVixSizeVariantConfig = DEFAULT_CONFIG,
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
            close_price_df.loc[signal_end_bar_ts, config.signal_symbol_str]
            / open_price_df.loc[month_start_bar_ts, config.signal_symbol_str]
            - 1.0
        )
        tlt_first15_return_float = float(
            close_price_df.loc[signal_end_bar_ts, config.defense_symbol_str]
            / open_price_df.loc[month_start_bar_ts, config.defense_symbol_str]
            - 1.0
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
        reversal_long_weight_scale_float = np.nan

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
                    reversal_asset_str = config.defense_symbol_str
                    reversal_signed_weight_float = float(config.reversal_weight_float)
                elif tlt_outperformed_bool:
                    reversal_asset_str = config.long_symbol_str
                    # *** CRITICAL*** The UPRO size must use the latest close
                    # strictly before the reversal entry open, never the same
                    # bar as the open fill.
                    reversal_vix_decision_bar_ts = pd.Timestamp(
                        month_trading_index[month_observation_count_int - config.eom_hold_day_count_int - 1]
                    )
                    reversal_vix_close_float = float(
                        close_price_df.loc[reversal_vix_decision_bar_ts, config.vix_symbol_str]
                    )
                    reversal_long_weight_scale_float = compute_long_weight_scale_float(
                        vix_close_float=reversal_vix_close_float,
                        config=config,
                    )
                    reversal_signed_weight_float = float(
                        config.reversal_weight_float * reversal_long_weight_scale_float
                    )
                else:
                    reversal_entry_bar_ts = pd.NaT
                    reversal_exit_bar_ts = pd.NaT

        pair_entry_bar_ts = pd.NaT
        pair_exit_bar_ts = pd.NaT
        pair_long_signed_weight_float = np.nan
        pair_vix_decision_bar_ts = pd.NaT
        pair_vix_close_float = np.nan
        pair_long_asset_str: str | None = None
        pair_long_weight_scale_float = np.nan
        next_month_period = month_period + 1

        # *** CRITICAL*** The early-next-month pair leg uses only the prior
        # month's completed first-15-day signal. The UPRO sizing uses the last
        # fully observed close before the next-month open entry.
        if spy_outperformed_bool and next_month_period in month_group_map:
            next_month_trading_index = pd.DatetimeIndex(month_group_map[next_month_period]).sort_values()
            if len(next_month_trading_index) >= config.bom_pair_hold_day_count_int:
                pair_entry_bar_ts = pd.Timestamp(next_month_trading_index[0])
                pair_exit_bar_ts = pd.Timestamp(next_month_trading_index[config.bom_pair_hold_day_count_int - 1])
                pair_vix_decision_bar_ts = pd.Timestamp(month_trading_index[-1])
                pair_vix_close_float = float(close_price_df.loc[pair_vix_decision_bar_ts, config.vix_symbol_str])
                pair_long_weight_scale_float = compute_long_weight_scale_float(
                    vix_close_float=pair_vix_close_float,
                    config=config,
                )
                pair_long_asset_str = config.long_symbol_str
                pair_long_signed_weight_float = float(
                    config.pair_abs_weight_float * pair_long_weight_scale_float
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
                "reversal_long_weight_scale_float": reversal_long_weight_scale_float,
                "pair_long_asset_str": pair_long_asset_str,
                "pair_entry_bar_ts": pair_entry_bar_ts,
                "pair_exit_bar_ts": pair_exit_bar_ts,
                "pair_long_signed_weight_float": pair_long_signed_weight_float,
                "pair_vix_decision_bar_ts": pair_vix_decision_bar_ts,
                "pair_vix_close_float": pair_vix_close_float,
                "pair_long_weight_scale_float": pair_long_weight_scale_float,
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
                "reversal_long_weight_scale_float",
                "pair_long_asset_str",
                "pair_entry_bar_ts",
                "pair_exit_bar_ts",
                "pair_long_signed_weight_float",
                "pair_vix_decision_bar_ts",
                "pair_vix_close_float",
                "pair_long_weight_scale_float",
            ]
        )

    month_signal_df["signal_month_period_str"] = month_signal_df["signal_month_period"].astype(str)
    month_signal_df = month_signal_df.set_index("signal_month_period_str", drop=True)
    return month_signal_df


def build_trade_leg_plan_df(
    month_signal_df: pd.DataFrame,
    config: EomTltSpyUproVixSizeVariantConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    trade_leg_row_list: list[dict[str, object]] = []
    trade_id_int = 0

    for _, signal_row_ser in month_signal_df.iterrows():
        reversal_asset_str = signal_row_ser["reversal_asset_str"]
        reversal_entry_bar_ts = signal_row_ser["reversal_entry_bar_ts"]
        reversal_exit_bar_ts = signal_row_ser["reversal_exit_bar_ts"]
        reversal_signed_weight_float = signal_row_ser["reversal_signed_weight_float"]
        pair_long_asset_str = signal_row_ser["pair_long_asset_str"]
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
            and pair_long_asset_str is not None
            and np.isfinite(float(pair_long_signed_weight_float))
        ):
            trade_id_int += 1
            trade_leg_row_list.append(
                {
                    "trade_id_int": trade_id_int,
                    "leg_type_str": "pair_long_upro",
                    "signal_month_period_str": signal_row_ser.name,
                    "asset_str": str(pair_long_asset_str),
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
                    "asset_str": config.defense_symbol_str,
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


def build_daily_target_weight_df(
    trading_index: pd.DatetimeIndex,
    trade_leg_plan_df: pd.DataFrame,
    asset_list: Sequence[str],
) -> pd.DataFrame:
    target_weight_df = pd.DataFrame(0.0, index=pd.DatetimeIndex(trading_index), columns=list(asset_list), dtype=float)
    for _, trade_leg_row_ser in trade_leg_plan_df.iterrows():
        target_weight_df.loc[
            pd.Timestamp(trade_leg_row_ser["entry_bar_ts"]):pd.Timestamp(trade_leg_row_ser["exit_bar_ts"]),
            str(trade_leg_row_ser["asset_str"]),
        ] += float(trade_leg_row_ser["signed_weight_float"])
    return target_weight_df


class EomTltSpyUproVixSizeVariantResearchStrategy(Strategy):
    """
    Research-only container for the TLT / SPY signal / UPRO execution / VIX-sized variant.
    """

    enable_signal_audit = False
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        tradeable_asset_list: Sequence[str],
        trade_leg_plan_df: pd.DataFrame,
        daily_target_weight_df: pd.DataFrame,
        capital_base: float = 100_000.0,
        slippage: float = 0.0001,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        self.trade_leg_plan_df = trade_leg_plan_df.copy()
        self.daily_target_weight_df = daily_target_weight_df.copy()
        self.tradeable_asset_list = list(tradeable_asset_list)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        return


def run_variant_research_backtest(
    strategy: EomTltSpyUproVixSizeVariantResearchStrategy,
    pricing_data_df: pd.DataFrame,
) -> EomTltSpyUproVixSizeVariantResearchStrategy:
    open_price_df = pricing_data_df.xs("Open", axis=1, level=1)[strategy.tradeable_asset_list].astype(float)
    close_price_df = pricing_data_df.xs("Close", axis=1, level=1)[strategy.tradeable_asset_list].astype(float)
    trading_index = pd.DatetimeIndex(pricing_data_df.index)

    entry_plan_map = {
        pd.Timestamp(entry_bar_ts): trade_leg_sub_df.copy()
        for entry_bar_ts, trade_leg_sub_df in strategy.trade_leg_plan_df.groupby("entry_bar_ts", sort=True)
    }
    exit_plan_map = {
        pd.Timestamp(exit_bar_ts): trade_leg_sub_df.copy()
        for exit_bar_ts, trade_leg_sub_df in strategy.trade_leg_plan_df.groupby("exit_bar_ts", sort=True)
    }

    cash_value_float = float(strategy._capital_base)
    position_share_map: dict[str, int] = {asset_str: 0 for asset_str in strategy.tradeable_asset_list}
    active_trade_share_map: dict[int, int] = {}
    transaction_row_list: list[dict[str, object]] = []
    portfolio_value_map: dict[pd.Timestamp, float] = {}
    cash_value_map: dict[pd.Timestamp, float] = {}
    total_value_map: dict[pd.Timestamp, float] = {}

    for bar_ts in trading_index:
        bar_ts = pd.Timestamp(bar_ts)

        if bar_ts in entry_plan_map:
            pre_entry_equity_float = float(
                cash_value_float + sum(
                    position_share_map[asset_str] * float(open_price_df.loc[bar_ts, asset_str])
                    for asset_str in strategy.tradeable_asset_list
                )
            )
            for trade_id_int, trade_leg_row_ser in entry_plan_map[bar_ts].iterrows():
                asset_str = str(trade_leg_row_ser["asset_str"])
                signed_weight_float = float(trade_leg_row_ser["signed_weight_float"])
                direction_sign_int = 1 if signed_weight_float > 0.0 else -1
                open_price_float = float(open_price_df.loc[bar_ts, asset_str])
                entry_fill_price_float = float(open_price_float * (1.0 + direction_sign_int * float(strategy._slippage)))
                notional_value_float = float(pre_entry_equity_float * abs(signed_weight_float))
                share_count_int = int(notional_value_float / entry_fill_price_float) * direction_sign_int
                entry_commission_float = compute_commission_float(
                    share_count_int=share_count_int,
                    commission_per_share_float=float(strategy._commission_per_share),
                    commission_minimum_float=float(strategy._commission_minimum),
                )
                if share_count_int == 0:
                    continue

                cash_value_float -= float(share_count_int * entry_fill_price_float)
                cash_value_float -= float(entry_commission_float)
                position_share_map[asset_str] += int(share_count_int)
                active_trade_share_map[int(trade_id_int)] = int(share_count_int)

                transaction_row_list.append(
                    {
                        "trade_id": int(trade_id_int),
                        "bar": bar_ts,
                        "asset": asset_str,
                        "amount": int(share_count_int),
                        "price": float(entry_fill_price_float),
                        "total_value": float(share_count_int * entry_fill_price_float),
                        "order_id": len(transaction_row_list) + 1,
                        "commission": float(entry_commission_float),
                    }
                )

        if bar_ts in exit_plan_map:
            for trade_id_int, trade_leg_row_ser in exit_plan_map[bar_ts].iterrows():
                asset_str = str(trade_leg_row_ser["asset_str"])
                if int(trade_id_int) not in active_trade_share_map:
                    raise RuntimeError(f"Missing active trade shares for trade_id {trade_id_int} on {bar_ts}.")

                active_share_count_int = int(active_trade_share_map.pop(int(trade_id_int)))
                exit_amount_int = -active_share_count_int
                direction_sign_int = 1 if exit_amount_int > 0 else -1
                close_price_float = float(close_price_df.loc[bar_ts, asset_str])
                exit_fill_price_float = float(close_price_float * (1.0 + direction_sign_int * float(strategy._slippage)))
                exit_commission_float = compute_commission_float(
                    share_count_int=exit_amount_int,
                    commission_per_share_float=float(strategy._commission_per_share),
                    commission_minimum_float=float(strategy._commission_minimum),
                )

                cash_value_float -= float(exit_amount_int * exit_fill_price_float)
                cash_value_float -= float(exit_commission_float)
                position_share_map[asset_str] += int(exit_amount_int)

                transaction_row_list.append(
                    {
                        "trade_id": int(trade_id_int),
                        "bar": bar_ts,
                        "asset": asset_str,
                        "amount": int(exit_amount_int),
                        "price": float(exit_fill_price_float),
                        "total_value": float(exit_amount_int * exit_fill_price_float),
                        "order_id": len(transaction_row_list) + 1,
                        "commission": float(exit_commission_float),
                    }
                )

        portfolio_value_float = float(
            sum(position_share_map[asset_str] * float(close_price_df.loc[bar_ts, asset_str]) for asset_str in strategy.tradeable_asset_list)
        )
        total_value_float = float(cash_value_float + portfolio_value_float)
        portfolio_value_map[bar_ts] = portfolio_value_float
        cash_value_map[bar_ts] = float(cash_value_float)
        total_value_map[bar_ts] = total_value_float

    portfolio_value_ser = pd.Series(portfolio_value_map, dtype=float).sort_index()
    cash_ser = pd.Series(cash_value_map, dtype=float).sort_index()
    total_value_ser = pd.Series(total_value_map, dtype=float).sort_index()

    benchmark_equity_map: dict[str, pd.Series] = {}
    for benchmark_str in strategy._benchmarks:
        benchmark_close_ser = pricing_data_df[(benchmark_str, "Close")].astype(float)
        benchmark_start_close_float = float(benchmark_close_ser.iloc[0])
        benchmark_equity_map[benchmark_str] = benchmark_close_ser / benchmark_start_close_float * float(strategy._capital_base)

    strategy.results = build_results_df(
        total_value_ser=total_value_ser,
        portfolio_value_ser=portfolio_value_ser,
        cash_ser=cash_ser,
        benchmark_equity_map=benchmark_equity_map,
    )
    strategy._transactions = pd.DataFrame(
        transaction_row_list,
        columns=["trade_id", "bar", "asset", "amount", "price", "total_value", "order_id", "commission"],
    )
    strategy._position_amount_map = {}
    strategy.cash = float(cash_ser.iloc[-1])
    strategy.portfolio_value = float(portfolio_value_ser.iloc[-1])
    strategy.total_value = float(total_value_ser.iloc[-1])
    strategy.summarize()
    return strategy


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

    strategy = EomTltSpyUproVixSizeVariantResearchStrategy(
        name="strategy_eom_tlt_spy_upro_vix_size_variant_research",
        benchmarks=config.benchmark_list,
        tradeable_asset_list=config.tradeable_symbol_list,
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
