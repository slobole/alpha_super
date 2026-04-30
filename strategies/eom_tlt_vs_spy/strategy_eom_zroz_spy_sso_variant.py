"""
ZROZ / SPY log-return signal, SSO execution, end-of-month relative-strength variant.

TL;DR: This research strategy mirrors `strategy_eom_tlt_spy_sso_variant.py`,
but replaces the bond side with `ZROZ` in both the signal and traded legs.

Core formulas
-------------
Let the ordered trading dates in month m be:

    D_m = {d_{m,1}, d_{m,2}, ..., d_{m,N_m}}

Define:

    L = 15   # signal window
    H = 5    # late-month hold window
    K = 5    # next-month pair hold window

First-15 trading-day log returns:

    g_spy_m^{(15)}
        = log(Close_SPY_{d_{m,L}} / Open_SPY_{d_{m,1}})

    g_zroz_m^{(15)}
        = log(Close_ZROZ_{d_{m,L}} / Open_ZROZ_{d_{m,1}})

Relative signal:

    rel_m = g_spy_m^{(15)} - g_zroz_m^{(15)}

Optional no-trade zone:

    trade_signal_m =
        +1 if rel_m > threshold
        -1 if rel_m < -threshold
         0 otherwise

Late-month reversal leg in month m:

    if trade_signal_m = +1:
        long ZROZ from Open_{d_{m,N_m-H+1}} to Close_{d_{m,N_m}}

    if trade_signal_m = -1:
        long SSO from Open_{d_{m,N_m-H+1}} to Close_{d_{m,N_m}}

Early-next-month pair leg in month m+1:

    if trade_signal_m = +1:
        long  SSO  weight +w_pair
        short ZROZ weight -w_pair
        from Open_{d_{m+1,1}} to Close_{d_{m+1,K}}

Default sizing assumption
-------------------------
The pair leg uses:

    w_pair = 0.50

so the next-month pair is:

    +50% SSO
    -50% ZROZ

with gross exposure near 100%, not 200%.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
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
class EomZrozSpySsoVariantConfig(EomTltSpySsoVariantConfig):
    defense_symbol_str: str = "ZROZ"
    signal_deadband_float: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.signal_deadband_float < 0.0:
            raise ValueError("signal_deadband_float must be non-negative.")


DEFAULT_CONFIG = EomZrozSpySsoVariantConfig()


class EomZrozSpySsoVariantResearchStrategy(EomTltSpySsoVariantResearchStrategy):
    """
    Research-only container for the ZROZ / SPY-signal / SSO-execution variant.
    """


class EomZrozSpySsoVariantExecutionTimingStrategy(EomZrozSpySsoVariantResearchStrategy):
    """
    Market-order adapter for ExecutionTimingAnalysis.

    Formula:

        entry_intent_t = planned entry_bar_ts
        exit_intent_t  = planned exit_bar_ts

        entry_fill = entry_intent_t + entry_lag at entry_price_field
        exit_fill  = exit_intent_t  + exit_lag  at exit_price_field

    The existing research runner keeps its custom open-entry / close-exit
    path. This adapter only exposes equivalent order intent for the matrix.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._entry_plan_map = {
            pd.Timestamp(entry_bar_ts): trade_leg_sub_df.copy()
            for entry_bar_ts, trade_leg_sub_df in self.trade_leg_plan_df.groupby("entry_bar_ts", sort=True)
        }
        self._exit_plan_map = {
            pd.Timestamp(exit_bar_ts): trade_leg_sub_df.copy()
            for exit_bar_ts, trade_leg_sub_df in self.trade_leg_plan_df.groupby("exit_bar_ts", sort=True)
        }

    def _pre_entry_open_equity_float(self, open_price_ser: pd.Series) -> float:
        position_ser = self.get_positions()
        active_position_ser = position_ser[position_ser != 0.0].astype(float)
        if len(active_position_ser) == 0:
            return float(self.cash)

        active_open_price_ser = open_price_ser.reindex(active_position_ser.index).astype(float)
        if active_open_price_ser.isna().any():
            missing_asset_list = active_open_price_ser[active_open_price_ser.isna()].index.astype(str).tolist()
            raise RuntimeError(f"Missing open prices for active EOM timing positions: {missing_asset_list}")

        return float(self.cash + (active_position_ser * active_open_price_ser).sum())

    def _submit_entry_orders_for_bar(self, bar_ts: pd.Timestamp, open_price_ser: pd.Series) -> None:
        if bar_ts not in self._entry_plan_map:
            return

        # *** CRITICAL*** Existing custom EOM research sizing uses equity at
        # the planned entry open and the planned entry open price:
        # shares_i = floor(V_open_t * |w_i| / entry_price_i).
        # Keeping share orders here preserves that baseline when the matrix
        # default is planned-open entry.
        pre_entry_equity_float = self._pre_entry_open_equity_float(open_price_ser=open_price_ser)
        for trade_id_int, trade_leg_row_ser in self._entry_plan_map[bar_ts].iterrows():
            asset_str = str(trade_leg_row_ser["asset_str"])
            signed_weight_float = float(trade_leg_row_ser["signed_weight_float"])
            if np.isclose(signed_weight_float, 0.0, atol=1e-12):
                continue
            if asset_str not in open_price_ser.index:
                raise RuntimeError(f"Missing open price for planned EOM entry asset {asset_str}.")

            raw_open_price_float = float(open_price_ser.loc[asset_str])
            if not np.isfinite(raw_open_price_float) or raw_open_price_float <= 0.0:
                continue

            direction_sign_int = 1 if signed_weight_float > 0.0 else -1
            entry_sizing_price_float = float(
                raw_open_price_float * (1.0 + direction_sign_int * float(self._slippage))
            )
            notional_value_float = float(pre_entry_equity_float * abs(signed_weight_float))
            share_count_int = int(notional_value_float / entry_sizing_price_float) * direction_sign_int
            if share_count_int == 0:
                continue

            self.order(asset_str, share_count_int, trade_id=int(trade_id_int))

    def _submit_exit_orders_for_bar(self, bar_ts: pd.Timestamp) -> None:
        if bar_ts not in self._exit_plan_map:
            return

        for trade_id_int, trade_leg_row_ser in self._exit_plan_map[bar_ts].iterrows():
            asset_str = str(trade_leg_row_ser["asset_str"])
            open_trade_amount_ser = self._get_open_trade_amount_ser(asset_str=asset_str)
            trade_id_lookup_int = int(trade_id_int)
            if trade_id_lookup_int not in open_trade_amount_ser.index:
                continue

            open_share_count_float = float(open_trade_amount_ser.loc[trade_id_lookup_int])
            if np.isclose(open_share_count_float, 0.0, atol=1e-12):
                continue

            self.order(asset_str, -open_share_count_float, trade_id=trade_id_lookup_int)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None:
            return

        bar_ts = pd.Timestamp(self.current_bar)

        # *** CRITICAL*** For same timestamp fills, exits are emitted before
        # entries so the matrix does not silently let an entry use capital from
        # an exit that has not filled yet.
        self._submit_exit_orders_for_bar(bar_ts=bar_ts)
        self._submit_entry_orders_for_bar(bar_ts=bar_ts, open_price_ser=open_price_ser)


def load_pricing_data_df(
    config: EomZrozSpySsoVariantConfig = DEFAULT_CONFIG,
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
    # *** CRITICAL*** ZROZ starts later than SPY/SSO. Dropping rows without
    # complete Open/Close data prevents synthetic pre-inception bond signals.
    pricing_data_df = pricing_data_df.dropna(subset=required_price_column_list, how="any")
    if len(pricing_data_df) == 0:
        raise RuntimeError("No common Open/Close history across SPY, ZROZ, SSO, and benchmarks.")
    return pricing_data_df


def build_month_signal_df(
    open_price_df: pd.DataFrame,
    close_price_df: pd.DataFrame,
    config: EomZrozSpySsoVariantConfig = DEFAULT_CONFIG,
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

        spy_first15_log_return_float = float(
            np.log(
                close_price_df.loc[signal_end_bar_ts, config.signal_symbol_str]
                / open_price_df.loc[month_start_bar_ts, config.signal_symbol_str]
            )
        )
        zroz_first15_log_return_float = float(
            np.log(
                close_price_df.loc[signal_end_bar_ts, config.defense_symbol_str]
                / open_price_df.loc[month_start_bar_ts, config.defense_symbol_str]
            )
        )
        rel_15_log_return_float = float(spy_first15_log_return_float - zroz_first15_log_return_float)
        spy_outperformed_bool = bool(rel_15_log_return_float > config.signal_deadband_float)
        zroz_outperformed_bool = bool(rel_15_log_return_float < -config.signal_deadband_float)

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
                    reversal_asset_str = config.defense_symbol_str
                elif zroz_outperformed_bool:
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
                "spy_first15_log_return_float": spy_first15_log_return_float,
                "zroz_first15_log_return_float": zroz_first15_log_return_float,
                "rel_15_log_return_float": rel_15_log_return_float,
                "signal_deadband_float": float(config.signal_deadband_float),
                "spy_outperformed_bool": spy_outperformed_bool,
                "zroz_outperformed_bool": zroz_outperformed_bool,
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
                "spy_first15_log_return_float",
                "zroz_first15_log_return_float",
                "rel_15_log_return_float",
                "signal_deadband_float",
                "spy_outperformed_bool",
                "zroz_outperformed_bool",
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
    config: EomZrozSpySsoVariantConfig = DEFAULT_CONFIG,
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
                    "rel_15_log_return_float": float(signal_row_ser["rel_15_log_return_float"]),
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
                    "rel_15_log_return_float": float(signal_row_ser["rel_15_log_return_float"]),
                }
            )
            trade_id_int += 1
            trade_leg_row_list.append(
                {
                    "trade_id_int": trade_id_int,
                    "leg_type_str": "pair_short_zroz",
                    "signal_month_period_str": signal_row_ser.name,
                    "asset_str": config.defense_symbol_str,
                    "signed_weight_float": float(-config.pair_abs_weight_float),
                    "entry_bar_ts": pd.Timestamp(pair_entry_bar_ts),
                    "exit_bar_ts": pd.Timestamp(pair_exit_bar_ts),
                    "rel_15_log_return_float": float(signal_row_ser["rel_15_log_return_float"]),
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
                "rel_15_log_return_float",
            ]
        )

    trade_leg_plan_df = trade_leg_plan_df.sort_values(["entry_bar_ts", "trade_id_int"]).set_index("trade_id_int", drop=True)
    trade_leg_plan_df.index.name = "trade_id_int"
    return trade_leg_plan_df


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    """
    Build inputs for ExecutionTimingAnalysis.

    Formula:

        planned_entry_t = entry_bar_ts
        planned_exit_t  = exit_bar_ts

        entry_fill = planned_entry_t + entry_lag at entry_price_field
        exit_fill  = planned_exit_t  + exit_lag  at exit_price_field
    """
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

    def strategy_factory_fn():
        return EomZrozSpySsoVariantExecutionTimingStrategy(
            name="strategy_eom_zroz_spy_sso_variant_research",
            benchmarks=config.benchmark_list,
            tradeable_asset_list=config.tradeable_symbol_list,
            trade_leg_plan_df=trade_leg_plan_df.copy(),
            daily_target_weight_df=daily_target_weight_df.copy(),
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pd.DatetimeIndex(pricing_data_df.index),
        "order_generation_mode_str": "vanilla_current_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": ("same_open", "same_close_moc"),
        "exit_timing_str_tuple": ("same_open", "same_close_moc"),
        "default_entry_timing_str": "same_open",
        "default_exit_timing_str": "same_close_moc",
    }


def run_execution_timing_analysis(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
):
    from alpha.engine.execution_timing import ExecutionTimingAnalysis

    strategy_input_dict = build_execution_timing_analysis_inputs()
    timing_result_obj = ExecutionTimingAnalysis(
        strategy_factory_fn=strategy_input_dict["strategy_factory_fn"],
        pricing_data_df=strategy_input_dict["pricing_data_df"],
        calendar_idx=strategy_input_dict["calendar_idx"],
        output_dir_str=output_dir_str,
        save_output_bool=save_results_bool,
        entry_timing_str_tuple=strategy_input_dict["entry_timing_str_tuple"],
        exit_timing_str_tuple=strategy_input_dict["exit_timing_str_tuple"],
        order_generation_mode_str=strategy_input_dict["order_generation_mode_str"],
        risk_model_str=strategy_input_dict["risk_model_str"],
        default_entry_timing_str=strategy_input_dict["default_entry_timing_str"],
        default_exit_timing_str=strategy_input_dict["default_exit_timing_str"],
    ).run()

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(timing_result_obj.metric_df.to_string(index=False))
        if timing_result_obj.output_dir_path is not None:
            print(f"\nSaved execution timing artifacts to: {timing_result_obj.output_dir_path.resolve()}")

    return timing_result_obj


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

    strategy = EomZrozSpySsoVariantResearchStrategy(
        name="strategy_eom_zroz_spy_sso_variant_research",
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
