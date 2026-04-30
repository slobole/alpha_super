"""
TLT / SPY signal, SSO / UBT execution, end-of-month relative-strength variant.

TL;DR: This research strategy keeps the monthly signal comparison on `SPY`
versus `TLT`, but replaces every traded `TLT` leg with traded `UBT`.

Core formulas
-------------
Let the ordered trading dates in month m be:

    D_m = {d_{m,1}, d_{m,2}, ..., d_{m,N_m}}

Define:

    L = 15   # signal window
    H = 5    # late-month hold window
    K = 5    # next-month pair hold window

First-15 trading-day returns:

    r_spy_m^{(15)}
        = Close_SPY_{d_{m,L}} / Open_SPY_{d_{m,1}} - 1

    r_tlt_m^{(15)}
        = Close_TLT_{d_{m,L}} / Open_TLT_{d_{m,1}} - 1

Relative signal:

    rel_m = r_spy_m^{(15)} - r_tlt_m^{(15)}

Execution mapping:

    equity_trade_asset = SSO
    defense_trade_asset = UBT

Late-month reversal leg in month m:

    if rel_m > 0:
        long UBT from Open_{d_{m,N_m-H+1}} to Close_{d_{m,N_m}}

    if rel_m < 0:
        long SSO from Open_{d_{m,N_m-H+1}} to Close_{d_{m,N_m}}

Early-next-month pair leg in month m+1:

    if rel_m > 0:
        long  SSO weight +w_pair
        short UBT weight -w_pair
        from Open_{d_{m+1,1}} to Close_{d_{m+1,K}}

Default sizing assumption
-------------------------
The pair leg uses:

    w_pair = 0.50

so the next-month pair is:

    +50% SSO
    -50% UBT

with gross exposure near 100%, not 200%.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
from IPython.display import display

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.report import save_results
from strategies.eom_tlt_vs_spy.strategy_eom_tlt_spy_sso_variant import (
    EomTltSpySsoVariantConfig,
    EomTltSpySsoVariantResearchStrategy,
    build_daily_target_weight_df,
    get_completed_month_period_set,
    run_variant_research_backtest,
)
from strategies.eom_tlt_vs_spy.strategy_eom_trend_ibit import (
    attach_symbol_level,
    load_yahoo_ohlcv_df,
)


@dataclass(frozen=True)
class EomTltSpySsoUbtVariantConfig(EomTltSpySsoVariantConfig):
    defense_trade_symbol_str: str = "UBT"

    def __post_init__(self):
        super().__post_init__()
        if len(self.defense_trade_symbol_str) == 0:
            raise ValueError("defense_trade_symbol_str must be non-empty.")
        if self.long_symbol_str == self.defense_trade_symbol_str:
            raise ValueError("long_symbol_str and defense_trade_symbol_str must differ.")

    @property
    def data_symbol_list(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [
                    self.signal_symbol_str,
                    self.defense_symbol_str,
                    self.long_symbol_str,
                    self.defense_trade_symbol_str,
                    *self.benchmark_list,
                ]
            )
        )

    @property
    def tradeable_symbol_list(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([self.long_symbol_str, self.defense_trade_symbol_str]))


DEFAULT_CONFIG = EomTltSpySsoUbtVariantConfig()


class EomTltSpySsoUbtVariantResearchStrategy(EomTltSpySsoVariantResearchStrategy):
    """
    Research-only container for the TLT / SPY-signal / SSO-UBT execution variant.
    """


def load_pricing_data_df(
    config: EomTltSpySsoUbtVariantConfig = DEFAULT_CONFIG,
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
    required_price_column_list = list(
        dict.fromkeys(
            (symbol_str, field_str)
            for symbol_str in config.data_symbol_list
            for field_str in ("Open", "Close")
        )
    )
    # *** CRITICAL*** UBT starts later than TLT/SPY. Dropping rows without
    # complete Open/Close data prevents synthetic pre-inception traded UBT legs.
    pricing_data_df = pricing_data_df.dropna(subset=required_price_column_list, how="any")
    if len(pricing_data_df) == 0:
        raise RuntimeError("No common Open/Close history across SPY, TLT, SSO, UBT, and benchmarks.")
    return pricing_data_df


def build_month_signal_df(
    open_price_df: pd.DataFrame,
    close_price_df: pd.DataFrame,
    config: EomTltSpySsoUbtVariantConfig = DEFAULT_CONFIG,
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

        reversal_entry_bar_ts = pd.NaT
        reversal_exit_bar_ts = pd.NaT
        reversal_asset_str = None

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
                    reversal_asset_str = config.defense_trade_symbol_str
                elif tlt_outperformed_bool:
                    reversal_asset_str = config.long_symbol_str
                else:
                    reversal_entry_bar_ts = pd.NaT
                    reversal_exit_bar_ts = pd.NaT

        pair_entry_bar_ts = pd.NaT
        pair_exit_bar_ts = pd.NaT
        next_month_period = month_period + 1

        # *** CRITICAL*** The early-next-month pair leg uses only the prior
        # month's completed first-15-day signal. It requires K observed trading
        # days in month m+1, but month m+1 does not need to be fully complete.
        if spy_outperformed_bool and next_month_period in month_group_map:
            next_month_trading_index = pd.DatetimeIndex(month_group_map[next_month_period]).sort_values()
            if len(next_month_trading_index) >= config.bom_pair_hold_day_count_int:
                pair_entry_bar_ts = pd.Timestamp(next_month_trading_index[0])
                pair_exit_bar_ts = pd.Timestamp(next_month_trading_index[config.bom_pair_hold_day_count_int - 1])

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
                "pair_entry_bar_ts": pair_entry_bar_ts,
                "pair_exit_bar_ts": pair_exit_bar_ts,
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
                "pair_entry_bar_ts",
                "pair_exit_bar_ts",
            ]
        )

    month_signal_df["signal_month_period_str"] = month_signal_df["signal_month_period"].astype(str)
    month_signal_df = month_signal_df.set_index("signal_month_period_str", drop=True)
    return month_signal_df


def build_trade_leg_plan_df(
    month_signal_df: pd.DataFrame,
    config: EomTltSpySsoUbtVariantConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    trade_leg_row_list: list[dict[str, object]] = []
    trade_id_int = 0

    for _, signal_row_ser in month_signal_df.iterrows():
        reversal_asset_str = signal_row_ser["reversal_asset_str"]
        reversal_entry_bar_ts = signal_row_ser["reversal_entry_bar_ts"]
        reversal_exit_bar_ts = signal_row_ser["reversal_exit_bar_ts"]
        pair_entry_bar_ts = signal_row_ser["pair_entry_bar_ts"]
        pair_exit_bar_ts = signal_row_ser["pair_exit_bar_ts"]

        if pd.notna(reversal_entry_bar_ts) and pd.notna(reversal_exit_bar_ts) and reversal_asset_str is not None:
            trade_id_int += 1
            trade_leg_row_list.append(
                {
                    "trade_id_int": trade_id_int,
                    "leg_type_str": "reversal",
                    "signal_month_period_str": signal_row_ser.name,
                    "asset_str": str(reversal_asset_str),
                    "signed_weight_float": float(config.reversal_weight_float),
                    "entry_bar_ts": pd.Timestamp(reversal_entry_bar_ts),
                    "exit_bar_ts": pd.Timestamp(reversal_exit_bar_ts),
                    "rel_15_return_float": float(signal_row_ser["rel_15_return_float"]),
                }
            )

        if pd.notna(pair_entry_bar_ts) and pd.notna(pair_exit_bar_ts):
            trade_id_int += 1
            trade_leg_row_list.append(
                {
                    "trade_id_int": trade_id_int,
                    "leg_type_str": "pair_long_sso",
                    "signal_month_period_str": signal_row_ser.name,
                    "asset_str": config.long_symbol_str,
                    "signed_weight_float": float(config.pair_abs_weight_float),
                    "entry_bar_ts": pd.Timestamp(pair_entry_bar_ts),
                    "exit_bar_ts": pd.Timestamp(pair_exit_bar_ts),
                    "rel_15_return_float": float(signal_row_ser["rel_15_return_float"]),
                }
            )
            trade_id_int += 1
            trade_leg_row_list.append(
                {
                    "trade_id_int": trade_id_int,
                    "leg_type_str": "pair_short_ubt",
                    "signal_month_period_str": signal_row_ser.name,
                    "asset_str": config.defense_trade_symbol_str,
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

    strategy = EomTltSpySsoUbtVariantResearchStrategy(
        name="strategy_eom_tlt_spy_sso_ubt_variant_research",
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
