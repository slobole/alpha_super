"""
ZROZ / SPY signal, SSO execution, L1-vol-normalized end-of-month variant.

TL;DR: This research strategy mirrors `strategy_eom_zroz_spy_sso_variant.py`,
but compares volatility-normalized log returns instead of raw simple returns.

Core formulas
-------------
Let the ordered trading dates in month m be:

    D_m = {d_{m,1}, d_{m,2}, ..., d_{m,N_m}}

Define:

    L = 15   # signal window
    H = 5    # late-month hold window
    K = 5    # next-month pair hold window
    W = 63   # trailing L1 volatility window

First-15 trading-day log returns:

    g_spy_m^{(15)}
        = log(Close_SPY_{d_{m,L}} / Open_SPY_{d_{m,1}})

    g_zroz_m^{(15)}
        = log(Close_ZROZ_{d_{m,L}} / Open_ZROZ_{d_{m,1}})

Trailing robust daily volatility estimate:

    l1_vol_asset_m
        = sqrt(pi / 2) * mean(abs(log(Close_asset_t / Close_asset_{t-1})))

where the mean uses the W daily returns ending before d_{m,1}.

Alternative month-to-date signal-window estimate:

    l1_vol_asset_mtd_m
        = sqrt(pi / 2) * mean(abs(u_asset_m))

where:

    u_asset_m = {
        log(Close_asset_{d_{m,1}} / Open_asset_{d_{m,1}}),
        log(Close_asset_{d_{m,2}} / Close_asset_{d_{m,1}}),
        ...,
        log(Close_asset_{d_{m,L}} / Close_asset_{d_{m,L-1}})
    }

Normalized scores:

    score_spy_m  = g_spy_m^{(15)}  / (l1_vol_SPY_m  * sqrt(L))
    score_zroz_m = g_zroz_m^{(15)} / (l1_vol_ZROZ_m * sqrt(L))
    rel_m        = score_spy_m - score_zroz_m

Trading rule:

    if rel_m > 0:
        long ZROZ during the last H trading days of month m
        then long SSO / short ZROZ during the first K trading days of m+1

    if rel_m < 0:
        long SSO during the last H trading days of month m
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
from strategies.eom_tlt_vs_spy.strategy_eom_zroz_spy_sso_variant import (
    EomZrozSpySsoVariantConfig,
    EomZrozSpySsoVariantResearchStrategy,
    build_daily_target_weight_df,
    get_completed_month_period_set,
    load_pricing_data_df,
    run_variant_research_backtest,
)


@dataclass(frozen=True)
class EomZrozSpySsoL1NormVariantConfig(EomZrozSpySsoVariantConfig):
    vol_window_day_count_int: int = 63
    vol_window_mode_str: str = "trailing_63"
    min_l1_daily_vol_float: float = 1e-8

    def __post_init__(self):
        super().__post_init__()
        if self.vol_window_day_count_int <= 0:
            raise ValueError("vol_window_day_count_int must be positive.")
        if self.vol_window_mode_str not in {"trailing_63", "month_to_date_signal"}:
            raise ValueError("vol_window_mode_str must be 'trailing_63' or 'month_to_date_signal'.")
        if self.min_l1_daily_vol_float <= 0.0:
            raise ValueError("min_l1_daily_vol_float must be positive.")


DEFAULT_CONFIG = EomZrozSpySsoL1NormVariantConfig()


class EomZrozSpySsoL1NormVariantResearchStrategy(EomZrozSpySsoVariantResearchStrategy):
    """
    Research-only container for the L1-normalized ZROZ / SPY-signal / SSO-execution variant.
    """


def build_month_signal_df(
    open_price_df: pd.DataFrame,
    close_price_df: pd.DataFrame,
    config: EomZrozSpySsoL1NormVariantConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    trading_index = pd.DatetimeIndex(close_price_df.index).sort_values()
    month_group_map = pd.Series(trading_index, index=trading_index).groupby(trading_index.to_period("M")).groups
    completed_month_period_set = get_completed_month_period_set(trading_index)
    signal_symbol_list = [config.signal_symbol_str, config.defense_symbol_str]

    signal_close_price_df = close_price_df[signal_symbol_list].astype(float)
    # *** CRITICAL*** This shift creates trailing daily log returns. Each row t
    # uses Close_t / Close_{t-1}; later code only samples windows ending before
    # the current month's first trading day to avoid look-ahead leakage.
    daily_log_return_df = np.log(signal_close_price_df / signal_close_price_df.shift(1))

    month_signal_row_list: list[dict[str, object]] = []
    for month_period, month_date_index in month_group_map.items():
        month_trading_index = pd.DatetimeIndex(month_date_index).sort_values()
        month_observation_count_int = len(month_trading_index)
        if month_observation_count_int < config.signal_day_count_int:
            continue

        month_start_bar_ts = pd.Timestamp(month_trading_index[0])
        signal_end_bar_ts = pd.Timestamp(month_trading_index[config.signal_day_count_int - 1])

        if config.vol_window_mode_str == "trailing_63":
            month_start_pos_int = int(trading_index.get_loc(month_start_bar_ts))
            vol_window_end_pos_int = month_start_pos_int - 1
            if vol_window_end_pos_int + 1 < config.vol_window_day_count_int:
                continue

            vol_window_start_pos_int = vol_window_end_pos_int - config.vol_window_day_count_int + 1
            # *** CRITICAL*** The volatility window ends before month_start_bar_ts,
            # so the normalizer is known before the first-15-day signal period.
            vol_window_index = trading_index[vol_window_start_pos_int : vol_window_end_pos_int + 1]
            vol_window_log_return_df = daily_log_return_df.loc[vol_window_index, signal_symbol_list]
        else:
            signal_window_index = pd.DatetimeIndex(month_trading_index[: config.signal_day_count_int])
            signal_window_close_price_df = close_price_df.loc[signal_window_index, signal_symbol_list].astype(float)
            signal_window_open_price_ser = open_price_df.loc[signal_window_index[0], signal_symbol_list].astype(float)
            first_day_log_return_ser = np.log(signal_window_close_price_df.iloc[0] / signal_window_open_price_ser)
            first_day_log_return_df = pd.DataFrame([first_day_log_return_ser], index=[signal_window_index[0]])
            # *** CRITICAL*** Month-to-date signal-window volatility uses only
            # day 1 open-to-close and day 2..L close-to-close returns. This
            # includes the signal window but not later reversal/pair returns.
            later_day_log_return_df = np.log(
                signal_window_close_price_df.iloc[1:] / signal_window_close_price_df.shift(1).iloc[1:]
            )
            vol_window_log_return_df = pd.concat([first_day_log_return_df, later_day_log_return_df], axis=0)

        if vol_window_log_return_df.isna().any().any():
            continue

        l1_to_sigma_scale_float = float(np.sqrt(np.pi / 2.0))
        spy_l1_daily_vol_float = float(
            l1_to_sigma_scale_float * vol_window_log_return_df[config.signal_symbol_str].abs().mean()
        )
        zroz_l1_daily_vol_float = float(
            l1_to_sigma_scale_float * vol_window_log_return_df[config.defense_symbol_str].abs().mean()
        )
        if (
            spy_l1_daily_vol_float <= config.min_l1_daily_vol_float
            or zroz_l1_daily_vol_float <= config.min_l1_daily_vol_float
        ):
            continue

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
        signal_horizon_scale_float = float(np.sqrt(config.signal_day_count_int))
        spy_l1_score_float = float(
            spy_first15_log_return_float / (spy_l1_daily_vol_float * signal_horizon_scale_float)
        )
        zroz_l1_score_float = float(
            zroz_first15_log_return_float / (zroz_l1_daily_vol_float * signal_horizon_scale_float)
        )
        rel_l1_score_float = float(spy_l1_score_float - zroz_l1_score_float)
        spy_outperformed_bool = bool(rel_l1_score_float > 0.0)
        zroz_outperformed_bool = bool(rel_l1_score_float < 0.0)

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
        # month's completed normalized signal. It requires K observed trading
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
                "spy_l1_daily_vol_float": spy_l1_daily_vol_float,
                "zroz_l1_daily_vol_float": zroz_l1_daily_vol_float,
                "vol_window_mode_str": config.vol_window_mode_str,
                "spy_l1_score_float": spy_l1_score_float,
                "zroz_l1_score_float": zroz_l1_score_float,
                "rel_l1_score_float": rel_l1_score_float,
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
                "spy_l1_daily_vol_float",
                "zroz_l1_daily_vol_float",
                "vol_window_mode_str",
                "spy_l1_score_float",
                "zroz_l1_score_float",
                "rel_l1_score_float",
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
    config: EomZrozSpySsoL1NormVariantConfig = DEFAULT_CONFIG,
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
                    "rel_l1_score_float": float(signal_row_ser["rel_l1_score_float"]),
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
                    "rel_l1_score_float": float(signal_row_ser["rel_l1_score_float"]),
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
                    "rel_l1_score_float": float(signal_row_ser["rel_l1_score_float"]),
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
                "rel_l1_score_float",
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

    strategy = EomZrozSpySsoL1NormVariantResearchStrategy(
        name="strategy_eom_zroz_spy_sso_l1_norm_variant_research",
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
