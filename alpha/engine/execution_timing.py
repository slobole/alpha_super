"""
Execution timing matrix analysis for daily market-order strategies.

TL;DR: This module replays a strategy's existing order intent under alternate
entry/exit fill timings without changing the Vanilla backtest contract.

For a signal formed on bar t:

    entry_fill = signal_bar_t + entry_lag at entry_price_field
    exit_fill  = signal_bar_t + exit_lag  at exit_price_field

For each realized daily equity observation:

    r_t = V_t / V_{t-1} - 1

Annualized return follows the existing summary metric:

    AnnReturn = (V_T / V_0)^(252 / N) - 1

Tail risk uses:

    VaR_5 = Quantile(r_t, 0.05)
    CVaR_5 = Mean(r_t | r_t <= VaR_5)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import html
import json

import numpy as np
import pandas as pd

from alpha.engine.order import MarketOrder, Order
from alpha.engine.strategy import Strategy


SUPPORTED_TIMING_MODE_TUPLE: tuple[str, ...] = (
    "same_open",
    "same_close_moc",
    "next_open",
    "next_close",
)

DEFAULT_SIGNAL_CLOSE_TIMING_MODE_TUPLE: tuple[str, ...] = (
    "same_close_moc",
    "next_open",
    "next_close",
)

DEFAULT_TAA_REBALANCE_TIMING_MODE_TUPLE: tuple[str, ...] = (
    "same_open",
    "same_close_moc",
)

SUPPORTED_ORDER_GENERATION_MODE_TUPLE: tuple[str, ...] = (
    "signal_bar",
    "vanilla_current_bar",
)

SUPPORTED_RISK_MODEL_TUPLE: tuple[str, ...] = (
    "daily_ohlc_signal",
    "taa_rebalance",
)


@dataclass(frozen=True)
class ExecutionTimingRule:
    timing_str: str
    bar_offset_int: int
    price_field_str: str
    fill_phase_str: str


@dataclass(frozen=True)
class ScheduledOrder:
    order_obj: Order
    order_kind_str: str
    sequence_int: int
    signal_bar_ts: pd.Timestamp
    fill_bar_ts: pd.Timestamp
    fill_price_field_str: str
    fill_phase_str: str
    sizing_price_float: float
    sizing_portfolio_value_float: float


@dataclass
class ExecutionTimingAnalysisResult:
    strategy_name_str: str
    metric_df: pd.DataFrame
    ann_return_matrix_df: pd.DataFrame
    cvar_5_matrix_df: pd.DataFrame
    sharpe_matrix_df: pd.DataFrame
    max_drawdown_matrix_df: pd.DataFrame
    strategy_map: dict[tuple[str, str], Strategy]
    order_generation_mode_str: str = "signal_bar"
    risk_model_str: str = "daily_ohlc_signal"
    default_entry_timing_str: str = "next_open"
    default_exit_timing_str: str = "next_open"
    output_dir_path: Path | None = None


def get_execution_timing_rule(timing_str: str) -> ExecutionTimingRule:
    if timing_str == "same_open":
        return ExecutionTimingRule(
            timing_str=timing_str,
            bar_offset_int=0,
            price_field_str="Open",
            fill_phase_str="open",
        )
    if timing_str == "same_close_moc":
        return ExecutionTimingRule(
            timing_str=timing_str,
            bar_offset_int=0,
            price_field_str="Close",
            fill_phase_str="close",
        )
    if timing_str == "next_open":
        return ExecutionTimingRule(
            timing_str=timing_str,
            bar_offset_int=1,
            price_field_str="Open",
            fill_phase_str="open",
        )
    if timing_str == "next_close":
        return ExecutionTimingRule(
            timing_str=timing_str,
            bar_offset_int=1,
            price_field_str="Close",
            fill_phase_str="close",
        )
    raise ValueError(
        f"Unsupported timing_str '{timing_str}'. "
        f"Supported modes: {SUPPORTED_TIMING_MODE_TUPLE}"
    )


def compute_cvar_5_pct_float(daily_return_ser: pd.Series) -> float:
    """
    Compute daily CVaR 5% in percent units.

    Formula:

        VaR_5 = Quantile(r_t, 0.05)
        CVaR_5 = Mean(r_t | r_t <= VaR_5)
    """
    clean_return_ser = pd.Series(daily_return_ser, dtype=float).dropna()
    if len(clean_return_ser) == 0:
        return np.nan

    var_5_float = float(clean_return_ser.quantile(0.05))
    tail_return_ser = clean_return_ser[clean_return_ser <= var_5_float]
    if len(tail_return_ser) == 0:
        return np.nan
    return float(tail_return_ser.mean() * 100.0)


def _timing_rank_int(timing_str: str) -> int:
    timing_rule = get_execution_timing_rule(timing_str)
    phase_rank_int = 0 if timing_rule.fill_phase_str == "open" else 1
    return int(timing_rule.bar_offset_int * 2 + phase_rank_int)


def _default_timing_mode_tuple(order_generation_mode_str: str) -> tuple[str, ...]:
    if order_generation_mode_str == "vanilla_current_bar":
        return DEFAULT_TAA_REBALANCE_TIMING_MODE_TUPLE
    return DEFAULT_SIGNAL_CLOSE_TIMING_MODE_TUPLE


def _default_entry_exit_timing_tuple(order_generation_mode_str: str) -> tuple[str, str]:
    if order_generation_mode_str == "vanilla_current_bar":
        return "same_open", "same_open"
    return "next_open", "next_open"


def _timing_display_label_str(timing_str: str, order_generation_mode_str: str) -> str:
    if order_generation_mode_str == "vanilla_current_bar":
        label_map = {
            "same_open": "T+1 Open",
            "same_close_moc": "T+1 Close (MOC)",
            "next_open": "T+2 Open",
            "next_close": "T+2 Close",
        }
        return label_map[timing_str]

    label_map = {
        "same_open": "T Open (Diagnostic)",
        "same_close_moc": "T Close (Biased/MOC)",
        "next_open": "T+1 Open",
        "next_close": "T+1 Close (MOC)",
    }
    return label_map[timing_str]


def _daily_ohlc_signal_timing_risk_tuple(
    entry_timing_str: str,
    exit_timing_str: str,
) -> tuple[str, str]:
    timing_set = {entry_timing_str, exit_timing_str}
    if "same_open" in timing_set:
        return (
            "Diagnostic Only",
            "Daily OHLC signals use signal-day data, so filling at the same day's open is "
            "lookahead unless a separate pre-open signal exists.",
        )
    if "same_close_moc" in timing_set:
        return (
            "MOC Assumption",
            "Daily-bar close fill is acceptable as a research approximation, "
            "but live-clean only if the signal can be known before broker MOC cutoff.",
        )
    return ("Clean", "No obvious timing bias under the daily close-known-after-close assumption.")


def _taa_rebalance_timing_risk_tuple(
    entry_timing_str: str,
    exit_timing_str: str,
    order_generation_mode_str: str,
) -> tuple[str, str]:
    timing_set = {entry_timing_str, exit_timing_str}
    if order_generation_mode_str == "signal_bar" and "same_close_moc" in timing_set:
        return (
            "Biased MOC",
            "TAA signal uses the final decision-bar close, so filling at that same close "
            "is a biased research diagnostic unless the order can be known before cutoff.",
        )
    if "same_close_moc" in timing_set:
        return (
            "MOC Assumption",
            "TAA rebalance weights are known before the rebalance day only if the prior "
            "month-end data is finalized. Daily close fills are live-clean only if the "
            "broker MOC order can be submitted before cutoff.",
        )
    if "next_close" in timing_set:
        return (
            "MOC Assumption",
            "Rebalance-day close fills require broker MOC order support and cutoff-aware "
            "order submission.",
        )
    if _timing_rank_int(entry_timing_str) < _timing_rank_int(exit_timing_str):
        return (
            "Funding Assumption",
            "Entry fills before exit in this rebalance cell. Live trading may require "
            "temporary cash, margin, or explicit sell-before-buy order staging.",
        )
    return (
        "Clean",
        "TAA rebalance intent is generated from prior-close/month-end information. "
        "Same-open fills model the baseline OPG-style rebalance.",
    )


def _timing_risk_tuple(
    entry_timing_str: str,
    exit_timing_str: str,
    risk_model_str: str,
    order_generation_mode_str: str,
) -> tuple[str, str]:
    if risk_model_str == "daily_ohlc_signal":
        return _daily_ohlc_signal_timing_risk_tuple(entry_timing_str, exit_timing_str)
    if risk_model_str == "taa_rebalance":
        return _taa_rebalance_timing_risk_tuple(
            entry_timing_str,
            exit_timing_str,
            order_generation_mode_str,
        )
    raise ValueError(
        f"Unsupported risk_model_str '{risk_model_str}'. "
        f"Supported models: {SUPPORTED_RISK_MODEL_TUPLE}"
    )


def _first_available_previous_bar_ts(
    full_index: pd.DatetimeIndex,
    first_calendar_bar_ts: pd.Timestamp,
) -> pd.Timestamp | None:
    first_pos_int = int(full_index.get_loc(first_calendar_bar_ts))
    if first_pos_int == 0:
        return None
    return pd.Timestamp(full_index[first_pos_int - 1])


def _previous_bar_ts(full_index: pd.DatetimeIndex, bar_ts: pd.Timestamp) -> pd.Timestamp | None:
    current_pos_int = int(full_index.get_loc(bar_ts))
    previous_pos_int = current_pos_int - 1
    if previous_pos_int < 0:
        return None
    return pd.Timestamp(full_index[previous_pos_int])


def _next_bar_ts(full_index: pd.DatetimeIndex, bar_ts: pd.Timestamp) -> pd.Timestamp | None:
    current_pos_int = int(full_index.get_loc(bar_ts))
    next_pos_int = current_pos_int + 1
    if next_pos_int >= len(full_index):
        return None
    return pd.Timestamp(full_index[next_pos_int])


def _coerce_calendar_idx(
    pricing_data_df: pd.DataFrame,
    calendar_idx: pd.DatetimeIndex | None,
) -> pd.DatetimeIndex:
    if calendar_idx is None:
        return pd.DatetimeIndex(pricing_data_df.index)

    resolved_calendar_idx = pd.DatetimeIndex(calendar_idx)
    missing_bar_idx = resolved_calendar_idx.difference(pd.DatetimeIndex(pricing_data_df.index))
    if len(missing_bar_idx) > 0:
        missing_bar_list = [str(bar_ts.date()) for bar_ts in missing_bar_idx[:5]]
        raise ValueError(f"calendar_idx contains bars absent from pricing_data_df: {missing_bar_list}")
    return resolved_calendar_idx


def _open_price_ser(signal_data_df: pd.DataFrame, bar_ts: pd.Timestamp) -> pd.Series:
    open_price_ser = signal_data_df.loc[bar_ts, (slice(None), "Open")]
    open_price_ser.index = open_price_ser.index.get_level_values(0)
    return open_price_ser.astype(float)


def _close_price_ser(signal_data_df: pd.DataFrame, bar_ts: pd.Timestamp) -> pd.Series:
    close_price_ser = signal_data_df.loc[bar_ts, (slice(None), "Close")]
    close_price_ser.index = close_price_ser.index.get_level_values(0)
    return close_price_ser.astype(float)


def _active_portfolio_value_float(strategy_obj: Strategy, close_price_ser: pd.Series) -> float:
    position_ser = strategy_obj.get_positions()
    active_position_ser = position_ser[position_ser != 0.0]
    if len(active_position_ser) == 0:
        return 0.0

    active_close_price_ser = close_price_ser.reindex(active_position_ser.index).astype(float)
    if active_close_price_ser.isna().any():
        missing_asset_list = active_close_price_ser[active_close_price_ser.isna()].index.astype(str).tolist()
        raise RuntimeError(f"Missing close prices for active positions: {missing_asset_list}")

    return float((active_position_ser.astype(float) * active_close_price_ser).sum())


def _liquidate_missing_close_positions(
    strategy_obj: Strategy,
    signal_data_df: pd.DataFrame,
    bar_ts: pd.Timestamp,
    close_price_ser: pd.Series,
) -> None:
    position_ser = strategy_obj.get_positions()
    active_position_ser = position_ser[position_ser != 0.0]
    if len(active_position_ser) == 0:
        return

    active_close_price_ser = close_price_ser.reindex(active_position_ser.index).astype(float)
    missing_asset_list = active_close_price_ser[active_close_price_ser.isna()].index.astype(str).tolist()
    for asset_str in missing_asset_list:
        close_key_tuple = (asset_str, "Close")
        if close_key_tuple not in signal_data_df.columns:
            raise RuntimeError(f"Missing close history for active asset {asset_str}.")

        # *** CRITICAL*** Missing-close liquidation uses the latest available
        # close strictly before the current bar. Using a later close would leak
        # future information into the forced exit.
        prior_close_ser = signal_data_df.loc[:bar_ts, close_key_tuple].dropna()
        prior_close_ser = prior_close_ser[prior_close_ser.index < bar_ts]
        if len(prior_close_ser) == 0:
            raise RuntimeError(f"No prior close is available to liquidate {asset_str}.")

        liquidation_price_float = float(prior_close_ser.iloc[-1])
        open_trade_amount_ser = strategy_obj._get_open_trade_amount_ser(asset_str=asset_str)
        if len(open_trade_amount_ser) == 0:
            raise RuntimeError(f"Found active position in {asset_str} without open trade amounts.")

        for trade_id_obj, open_amount_float in open_trade_amount_ser.items():
            liquidation_amount_float = -float(open_amount_float)
            commission_float = float(strategy_obj._compute_commission(liquidation_amount_float))
            transaction_value_float = float(liquidation_amount_float * liquidation_price_float)
            strategy_obj.add_transaction(
                trade_id_obj,
                bar_ts,
                asset_str,
                liquidation_amount_float,
                liquidation_price_float,
                transaction_value_float,
                order_id=-1,
                commission=commission_float,
            )
            strategy_obj.cash -= transaction_value_float
            strategy_obj.cash -= commission_float


def _reset_strategy_state(strategy_obj: Strategy) -> None:
    strategy_obj.cash = float(strategy_obj._capital_base)
    strategy_obj.portfolio_value = 0.0
    strategy_obj.total_value = float(strategy_obj._capital_base)
    strategy_obj.results = strategy_obj.initialize_results()
    strategy_obj._orders = []
    strategy_obj._transactions = strategy_obj.initialize_transactions()
    strategy_obj._position_amount_map = {}
    strategy_obj._trades = None
    strategy_obj._open_trades = None
    strategy_obj._drawdowns = None
    strategy_obj.summary = None
    strategy_obj.summary_trades = None
    strategy_obj.current_bar = None
    strategy_obj.previous_bar = None
    strategy_obj._daily_return_history_list = []
    strategy_obj._portfolio_value_history_list = []
    strategy_obj._total_value_history_list = []
    strategy_obj.realized_weight_df = pd.DataFrame(dtype=float)
    strategy_obj._realized_weight_snapshot_row_dict_list = []
    strategy_obj._latest_close_price_ser = pd.Series(dtype=float)


def _sizing_price_float(
    signal_data_df: pd.DataFrame,
    order_obj: Order,
    signal_bar_ts: pd.Timestamp,
    fallback_price_float: float,
) -> float:
    close_key_tuple = (order_obj.asset, "Close")
    if close_key_tuple not in signal_data_df.columns:
        return float(fallback_price_float)

    # *** CRITICAL*** Value/percent order sizing is anchored to the signal
    # bar close. Moving this to the fill bar would silently change the tested
    # strategy sizing semantics.
    close_price_float = float(signal_data_df.loc[signal_bar_ts, close_key_tuple])
    if np.isfinite(close_price_float) and close_price_float > 0.0:
        return close_price_float
    return float(fallback_price_float)


def _classify_order_kind_str(position_float: float, amount_float: float) -> str:
    if np.isclose(amount_float, 0.0, atol=1e-12):
        return "flat"
    if np.isclose(position_float, 0.0, atol=1e-12):
        return "entry"
    if np.sign(position_float) == np.sign(amount_float):
        return "entry"
    return "exit"


def _max_position_count_int(strategy_obj: Strategy) -> int | None:
    if hasattr(strategy_obj, "max_positions"):
        return int(getattr(strategy_obj, "max_positions"))
    if hasattr(strategy_obj, "max_positions_int"):
        return int(getattr(strategy_obj, "max_positions_int"))
    return None


def _entry_capacity_int(strategy_obj: Strategy) -> int | None:
    max_position_count_int = _max_position_count_int(strategy_obj)
    if max_position_count_int is None:
        return None

    position_ser = strategy_obj.get_positions()
    active_long_count_int = int((position_ser.astype(float) > 0.0).sum())
    return max(0, max_position_count_int - active_long_count_int)


def _filter_entry_orders_for_delayed_exits(
    order_info_list: list[ScheduledOrder],
    entry_timing_str: str,
    exit_timing_str: str,
    pre_signal_entry_capacity_int: int | None,
) -> list[ScheduledOrder]:
    if pre_signal_entry_capacity_int is None:
        return order_info_list
    if _timing_rank_int(entry_timing_str) >= _timing_rank_int(exit_timing_str):
        return order_info_list

    kept_order_info_list: list[ScheduledOrder] = []
    used_entry_count_int = 0
    for order_info in order_info_list:
        if order_info.order_kind_str != "entry":
            kept_order_info_list.append(order_info)
            continue

        # *** CRITICAL*** When entries fill before delayed exits, exit signals
        # do not free slots yet. This prevents hidden temporary leverage from
        # same-close entries against next-open exits.
        if used_entry_count_int < pre_signal_entry_capacity_int:
            kept_order_info_list.append(order_info)
            used_entry_count_int += 1

    return kept_order_info_list


def _resolve_fill_bar_ts(
    full_index: pd.DatetimeIndex,
    signal_bar_ts: pd.Timestamp,
    timing_rule: ExecutionTimingRule,
) -> pd.Timestamp | None:
    # *** CRITICAL*** This is the signal-to-fill bar mapping under test.
    # Changing this mapping changes execution timing and therefore the
    # quantitative meaning of every matrix cell.
    if timing_rule.bar_offset_int == 0:
        return pd.Timestamp(signal_bar_ts)
    return _next_bar_ts(full_index, signal_bar_ts)


def _scheduled_order_sort_key_tuple(scheduled_order_obj: ScheduledOrder) -> tuple[int, int]:
    kind_rank_int = 0 if scheduled_order_obj.order_kind_str == "exit" else 1
    return (kind_rank_int, scheduled_order_obj.sequence_int)


def _execute_scheduled_order(
    strategy_obj: Strategy,
    signal_data_df: pd.DataFrame,
    scheduled_order_obj: ScheduledOrder,
) -> None:
    order_obj = scheduled_order_obj.order_obj
    fill_key_tuple = (order_obj.asset, scheduled_order_obj.fill_price_field_str)
    if fill_key_tuple not in signal_data_df.columns:
        raise RuntimeError(f"Missing fill price column {fill_key_tuple}.")

    current_position_float = float(strategy_obj.get_position(order_obj.asset))
    amount_float = float(
        order_obj.amount_in_shares(
            scheduled_order_obj.sizing_price_float,
            scheduled_order_obj.sizing_portfolio_value_float,
            current_position_float,
        )
    )
    if np.isclose(amount_float, 0.0, atol=1e-12):
        return

    raw_fill_price_float = float(signal_data_df.loc[scheduled_order_obj.fill_bar_ts, fill_key_tuple])
    if not np.isfinite(raw_fill_price_float) or raw_fill_price_float <= 0.0:
        # *** CRITICAL*** Match Vanilla process_orders(): if a new entry has
        # no tradable fill price on its scheduled bar, cancel the order instead
        # of inventing a price. Active positions still fail loud because a
        # missing executable exit can materially change path risk.
        if (
            scheduled_order_obj.order_kind_str == "entry"
            and np.isclose(current_position_float, 0.0, atol=1e-12)
        ):
            return
        raise RuntimeError(
            f"Invalid {scheduled_order_obj.fill_price_field_str} fill price for "
            f"{order_obj.asset} on {scheduled_order_obj.fill_bar_ts.date()}."
        )

    # *** CRITICAL*** same_open and same_close_moc intentionally allow biased
    # same-bar fills for research diagnostics. The result table labels those
    # cells so they are not mistaken for live-clean timing.
    penalty_float = 1.0 + float(np.sign(amount_float)) * float(strategy_obj._slippage)
    fill_price_float = float(raw_fill_price_float * penalty_float)
    commission_float = float(strategy_obj._compute_commission(amount_float))
    transaction_value_float = float(fill_price_float * amount_float)

    strategy_obj.add_transaction(
        order_obj.trade_id,
        scheduled_order_obj.fill_bar_ts,
        order_obj.asset,
        amount_float,
        fill_price_float,
        transaction_value_float,
        order_obj.id,
        commission_float,
    )
    strategy_obj.cash -= transaction_value_float
    strategy_obj.cash -= commission_float


def _process_scheduled_order_list(
    strategy_obj: Strategy,
    signal_data_df: pd.DataFrame,
    scheduled_order_list: list[ScheduledOrder],
) -> None:
    for scheduled_order_obj in sorted(scheduled_order_list, key=_scheduled_order_sort_key_tuple):
        _execute_scheduled_order(
            strategy_obj=strategy_obj,
            signal_data_df=signal_data_df,
            scheduled_order_obj=scheduled_order_obj,
        )


def _build_results_df(
    strategy_obj: Strategy,
    pricing_data_df: pd.DataFrame,
    calendar_idx: pd.DatetimeIndex,
    portfolio_value_map: dict[pd.Timestamp, float],
    cash_value_map: dict[pd.Timestamp, float],
    total_value_map: dict[pd.Timestamp, float],
) -> pd.DataFrame:
    total_value_ser = pd.Series(total_value_map, dtype=float).reindex(calendar_idx)
    portfolio_value_ser = pd.Series(portfolio_value_map, dtype=float).reindex(calendar_idx)
    cash_ser = pd.Series(cash_value_map, dtype=float).reindex(calendar_idx)

    results_df = pd.DataFrame(index=calendar_idx)
    results_df["portfolio_value"] = portfolio_value_ser.astype(float)
    results_df["cash"] = cash_ser.astype(float)
    results_df["total_value"] = total_value_ser.astype(float)

    daily_return_ser = total_value_ser.pct_change(fill_method=None).fillna(0.0)
    results_df["daily_returns"] = daily_return_ser.astype(float)
    results_df["total_returns"] = total_value_ser / float(strategy_obj._capital_base) - 1.0

    elapsed_day_count_ser = pd.Series(
        np.arange(1, len(calendar_idx) + 1, dtype=float),
        index=calendar_idx,
        dtype=float,
    )
    results_df["annualized_returns"] = (
        (total_value_ser / float(strategy_obj._capital_base)) ** (252.0 / elapsed_day_count_ser)
        - 1.0
    )
    results_df["annualized_volatility"] = (
        daily_return_ser.expanding(min_periods=2).std(ddof=1) * np.sqrt(252.0)
    )
    sharpe_ser = pd.Series(np.nan, index=calendar_idx, dtype=float)
    active_return_mask_ser = (portfolio_value_ser.astype(float) != 0.0) | (daily_return_ser != 0.0)
    for end_idx_int in range(len(calendar_idx)):
        sample_return_ser = daily_return_ser.iloc[: end_idx_int + 1].loc[
            active_return_mask_ser.iloc[: end_idx_int + 1]
        ]
        if len(sample_return_ser) < 2:
            continue
        sample_std_float = float(sample_return_ser.std(ddof=1))
        if sample_std_float > 0.0 and np.isfinite(sample_std_float):
            sharpe_ser.iloc[end_idx_int] = float(
                sample_return_ser.mean() / sample_std_float * np.sqrt(252.0)
            )
    results_df["sharpe_ratio"] = sharpe_ser

    drawdown_ser = total_value_ser / total_value_ser.cummax() - 1.0
    results_df["drawdown"] = drawdown_ser.astype(float)
    results_df["max_drawdown"] = drawdown_ser.cummin().astype(float)

    for benchmark_str in strategy_obj._benchmarks:
        benchmark_close_ser = pricing_data_df.loc[calendar_idx, (benchmark_str, "Close")].astype(float)
        benchmark_value_ser = (
            benchmark_close_ser / float(benchmark_close_ser.iloc[0]) * float(strategy_obj._capital_base)
        )
        benchmark_drawdown_ser = benchmark_value_ser / benchmark_value_ser.cummax() - 1.0
        results_df[benchmark_str] = benchmark_value_ser.astype(float)
        results_df[f"{benchmark_str}_drawdown"] = benchmark_drawdown_ser.astype(float)
        results_df[f"{benchmark_str}_max_drawdown"] = benchmark_drawdown_ser.cummin().astype(float)

    return results_df


def _build_metric_row_dict(
    strategy_obj: Strategy,
    entry_timing_str: str,
    exit_timing_str: str,
    risk_model_str: str,
    order_generation_mode_str: str,
    default_entry_timing_str: str,
    default_exit_timing_str: str,
) -> dict[str, object]:
    risk_label_str, _risk_note_str = _timing_risk_tuple(
        entry_timing_str,
        exit_timing_str,
        risk_model_str,
        order_generation_mode_str,
    )
    summary_ser = strategy_obj.summary["Strategy"]
    daily_return_ser = strategy_obj.results["total_value"].astype(float).pct_change(fill_method=None).dropna()
    default_cell_bool = (
        entry_timing_str == default_entry_timing_str
        and exit_timing_str == default_exit_timing_str
    )

    return {
        "Scenario": "Default" if default_cell_bool else "",
        "Entry Timing": _timing_display_label_str(entry_timing_str, order_generation_mode_str),
        "Exit Timing": _timing_display_label_str(exit_timing_str, order_generation_mode_str),
        "Ann. Return [%]": float(summary_ser.loc["Return (Ann.) [%]"]),
        "Max Drawdown [%]": float(summary_ser.loc["Max. Drawdown [%]"]),
        "Sharpe": float(summary_ser.loc["Sharpe Ratio"]),
        "CVaR 5% [%]": compute_cvar_5_pct_float(daily_return_ser),
        "Risk Label": risk_label_str,
    }


def _pivot_metric_df(
    metric_df: pd.DataFrame,
    metric_name_str: str,
    entry_timing_str_tuple: Sequence[str],
    exit_timing_str_tuple: Sequence[str],
    order_generation_mode_str: str,
) -> pd.DataFrame:
    entry_label_list = [
        _timing_display_label_str(str(timing_str), order_generation_mode_str)
        for timing_str in entry_timing_str_tuple
    ]
    exit_label_list = [
        _timing_display_label_str(str(timing_str), order_generation_mode_str)
        for timing_str in exit_timing_str_tuple
    ]
    return metric_df.pivot(
        index="Entry Timing",
        columns="Exit Timing",
        values=metric_name_str,
    ).reindex(index=entry_label_list, columns=exit_label_list)


def _format_metric_table_html(metric_df: pd.DataFrame) -> str:
    if metric_df is None or len(metric_df) == 0:
        return "<p>No execution timing cells were evaluated.</p>"

    header_html_str = "".join(f"<th>{html.escape(str(column_str))}</th>" for column_str in metric_df.columns)
    row_html_list: list[str] = []
    for _, row_ser in metric_df.iterrows():
        cell_html_list: list[str] = []
        row_class_str = ' class="default-row"' if str(row_ser.get("Scenario", "")) == "Default" else ""
        for column_str, cell_value_obj in row_ser.items():
            if isinstance(cell_value_obj, (float, np.floating)):
                cell_text_str = "" if np.isnan(cell_value_obj) else f"{cell_value_obj:.2f}"
            else:
                cell_text_str = str(cell_value_obj)
            class_str = ""
            if column_str == "Risk Label":
                normalized_label_str = cell_text_str.lower().replace(" ", "-")
                class_str = f' class="risk-{html.escape(normalized_label_str)}"'
            cell_html_list.append(f"<td{class_str}>{html.escape(cell_text_str)}</td>")
        row_html_list.append(f"<tr{row_class_str}>{''.join(cell_html_list)}</tr>")

    return f"<table><thead><tr>{header_html_str}</tr></thead><tbody>{''.join(row_html_list)}</tbody></table>"


def save_execution_timing_results(
    execution_timing_result_obj: ExecutionTimingAnalysisResult,
    output_dir_str: str = "results",
) -> Path:
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = (
        Path(output_dir_str)
        / "execution_timing"
        / execution_timing_result_obj.strategy_name_str
        / timestamp_str
    )
    output_path.mkdir(parents=True, exist_ok=True)

    execution_timing_result_obj.metric_df.to_csv(output_path / "execution_timing_metrics.csv", index=False)
    execution_timing_result_obj.ann_return_matrix_df.to_csv(output_path / "ann_return_matrix.csv")
    execution_timing_result_obj.cvar_5_matrix_df.to_csv(output_path / "cvar_5_matrix.csv")
    execution_timing_result_obj.sharpe_matrix_df.to_csv(output_path / "sharpe_matrix.csv")
    execution_timing_result_obj.max_drawdown_matrix_df.to_csv(output_path / "max_drawdown_matrix.csv")

    metadata_dict = {
        "artifact_type": "execution_timing_analysis",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_name_str": execution_timing_result_obj.strategy_name_str,
        "order_generation_mode_str": execution_timing_result_obj.order_generation_mode_str,
        "risk_model_str": execution_timing_result_obj.risk_model_str,
        "default_entry_timing": _timing_display_label_str(
            execution_timing_result_obj.default_entry_timing_str,
            execution_timing_result_obj.order_generation_mode_str,
        ),
        "default_exit_timing": _timing_display_label_str(
            execution_timing_result_obj.default_exit_timing_str,
            execution_timing_result_obj.order_generation_mode_str,
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2),
        encoding="utf-8",
    )

    html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(execution_timing_result_obj.strategy_name_str)} Execution Timing Analysis</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #17202a; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #d7dde5; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f3f6fa; }}
.note {{ color: #52606d; max-width: 960px; line-height: 1.45; }}
.default-row {{ background: #eef7f1; }}
.default-row td:first-child {{ color: #0f6b3d; font-weight: 700; }}
.risk-clean {{ color: #0f6b3d; font-weight: 700; }}
.risk-moc-assumption {{ color: #9a5b00; font-weight: 700; }}
.risk-biased-moc {{ color: #9b1c31; font-weight: 700; }}
.risk-diagnostic-only {{ color: #9b1c31; font-weight: 700; }}
.risk-funding-assumption {{ color: #7a4b00; font-weight: 700; }}
</style>
</head>
<body>
<h1>{html.escape(execution_timing_result_obj.strategy_name_str)} Execution Timing Analysis</h1>
<p class="note">
This report is a research diagnostic. The default execution path is highlighted. MOC and funding labels mark cells
that need explicit live execution handling before they should be treated as live-clean.
</p>
{_format_metric_table_html(execution_timing_result_obj.metric_df)}
</body>
</html>"""
    (output_path / "report.html").write_text(html_str, encoding="utf-8")
    return output_path


class ExecutionTimingAnalysis:
    """
    Run an entry/exit fill timing matrix for a strategy factory.

    The strategy factory must return a fresh strategy instance. V1 intentionally
    supports DV2-style market orders only; limit/stop orders fail loud.
    """

    def __init__(
        self,
        strategy_factory_fn: Callable[[], Strategy],
        pricing_data_df: pd.DataFrame,
        calendar_idx: pd.DatetimeIndex | None = None,
        entry_timing_str_tuple: Sequence[str] | None = None,
        exit_timing_str_tuple: Sequence[str] | None = None,
        output_dir_str: str = "results",
        save_output_bool: bool = True,
        show_signal_progress_bool: bool = False,
        audit_override_bool: bool | None = False,
        audit_sample_size_int: int | None = None,
        order_generation_mode_str: str = "signal_bar",
        risk_model_str: str = "daily_ohlc_signal",
        default_entry_timing_str: str | None = None,
        default_exit_timing_str: str | None = None,
    ):
        self.strategy_factory_fn = strategy_factory_fn
        self.pricing_data_df = pricing_data_df.sort_index().copy()
        self.calendar_idx = _coerce_calendar_idx(self.pricing_data_df, calendar_idx)
        self.output_dir_str = str(output_dir_str)
        self.save_output_bool = bool(save_output_bool)
        self.show_signal_progress_bool = bool(show_signal_progress_bool)
        self.audit_override_bool = audit_override_bool
        self.audit_sample_size_int = audit_sample_size_int
        self.order_generation_mode_str = str(order_generation_mode_str)
        self.risk_model_str = str(risk_model_str)

        default_timing_mode_tuple = _default_timing_mode_tuple(self.order_generation_mode_str)
        self.entry_timing_str_tuple = (
            tuple(entry_timing_str_tuple)
            if entry_timing_str_tuple is not None
            else default_timing_mode_tuple
        )
        self.exit_timing_str_tuple = (
            tuple(exit_timing_str_tuple)
            if exit_timing_str_tuple is not None
            else default_timing_mode_tuple
        )

        inferred_default_entry_str, inferred_default_exit_str = _default_entry_exit_timing_tuple(
            self.order_generation_mode_str
        )
        self.default_entry_timing_str = (
            inferred_default_entry_str
            if default_entry_timing_str is None
            else str(default_entry_timing_str)
        )
        self.default_exit_timing_str = (
            inferred_default_exit_str
            if default_exit_timing_str is None
            else str(default_exit_timing_str)
        )

        for timing_str in (
            self.entry_timing_str_tuple
            + self.exit_timing_str_tuple
            + (self.default_entry_timing_str, self.default_exit_timing_str)
        ):
            get_execution_timing_rule(str(timing_str))
        if self.order_generation_mode_str not in SUPPORTED_ORDER_GENERATION_MODE_TUPLE:
            raise ValueError(
                f"Unsupported order_generation_mode_str '{self.order_generation_mode_str}'. "
                f"Supported modes: {SUPPORTED_ORDER_GENERATION_MODE_TUPLE}"
            )
        if self.risk_model_str not in SUPPORTED_RISK_MODEL_TUPLE:
            raise ValueError(
                f"Unsupported risk_model_str '{self.risk_model_str}'. "
                f"Supported models: {SUPPORTED_RISK_MODEL_TUPLE}"
            )

    def run(self) -> ExecutionTimingAnalysisResult:
        base_strategy_obj = self.strategy_factory_fn()
        base_strategy_obj.show_signal_progress_bool = self.show_signal_progress_bool
        base_strategy_obj.show_audit_progress_bool = self.show_signal_progress_bool
        signal_data_df = base_strategy_obj.compute_signals(self.pricing_data_df)

        audit_enabled_bool = (
            base_strategy_obj.enable_signal_audit
            if self.audit_override_bool is None
            else bool(self.audit_override_bool)
        )
        if audit_enabled_bool:
            base_strategy_obj.audit_signals(
                self.pricing_data_df,
                signal_data_df,
                sample_size=self.audit_sample_size_int,
            )

        metric_row_list: list[dict[str, object]] = []
        strategy_map: dict[tuple[str, str], Strategy] = {}

        for entry_timing_str in self.entry_timing_str_tuple:
            for exit_timing_str in self.exit_timing_str_tuple:
                strategy_obj = self.strategy_factory_fn()
                _reset_strategy_state(strategy_obj)
                completed_strategy_obj = self._run_single_cell(
                    strategy_obj=strategy_obj,
                    signal_data_df=signal_data_df,
                    entry_timing_str=str(entry_timing_str),
                    exit_timing_str=str(exit_timing_str),
                )
                strategy_map[(str(entry_timing_str), str(exit_timing_str))] = completed_strategy_obj
                metric_row_list.append(
                    _build_metric_row_dict(
                        strategy_obj=completed_strategy_obj,
                        entry_timing_str=str(entry_timing_str),
                        exit_timing_str=str(exit_timing_str),
                        risk_model_str=self.risk_model_str,
                        order_generation_mode_str=self.order_generation_mode_str,
                        default_entry_timing_str=self.default_entry_timing_str,
                        default_exit_timing_str=self.default_exit_timing_str,
                    )
                )

        metric_df = pd.DataFrame(metric_row_list)
        if "Scenario" in metric_df.columns:
            metric_df = metric_df.sort_values(
                by="Scenario",
                key=lambda scenario_ser: scenario_ser.ne("Default").astype(int),
                kind="stable",
            ).reset_index(drop=True)
        result_obj = ExecutionTimingAnalysisResult(
            strategy_name_str=str(base_strategy_obj.name),
            metric_df=metric_df,
            ann_return_matrix_df=_pivot_metric_df(
                metric_df,
                "Ann. Return [%]",
                self.entry_timing_str_tuple,
                self.exit_timing_str_tuple,
                self.order_generation_mode_str,
            ),
            cvar_5_matrix_df=_pivot_metric_df(
                metric_df,
                "CVaR 5% [%]",
                self.entry_timing_str_tuple,
                self.exit_timing_str_tuple,
                self.order_generation_mode_str,
            ),
            sharpe_matrix_df=_pivot_metric_df(
                metric_df,
                "Sharpe",
                self.entry_timing_str_tuple,
                self.exit_timing_str_tuple,
                self.order_generation_mode_str,
            ),
            max_drawdown_matrix_df=_pivot_metric_df(
                metric_df,
                "Max Drawdown [%]",
                self.entry_timing_str_tuple,
                self.exit_timing_str_tuple,
                self.order_generation_mode_str,
            ),
            strategy_map=strategy_map,
            order_generation_mode_str=self.order_generation_mode_str,
            risk_model_str=self.risk_model_str,
            default_entry_timing_str=self.default_entry_timing_str,
            default_exit_timing_str=self.default_exit_timing_str,
        )

        if self.save_output_bool:
            result_obj.output_dir_path = save_execution_timing_results(
                result_obj,
                output_dir_str=self.output_dir_str,
            )

        return result_obj

    def _run_single_cell(
        self,
        strategy_obj: Strategy,
        signal_data_df: pd.DataFrame,
        entry_timing_str: str,
        exit_timing_str: str,
    ) -> Strategy:
        full_index = pd.DatetimeIndex(signal_data_df.index)
        calendar_idx = self.calendar_idx
        if len(calendar_idx) == 0:
            raise ValueError("calendar_idx must contain at least one bar.")

        first_calendar_bar_ts = pd.Timestamp(calendar_idx[0])
        previous_start_bar_ts = _first_available_previous_bar_ts(full_index, first_calendar_bar_ts)
        engine_bar_list = []
        if previous_start_bar_ts is not None:
            engine_bar_list.append(previous_start_bar_ts)
        engine_bar_list.extend([pd.Timestamp(bar_ts) for bar_ts in calendar_idx])
        engine_bar_idx = pd.DatetimeIndex(engine_bar_list).drop_duplicates()
        calendar_bar_set = set(pd.Timestamp(bar_ts) for bar_ts in calendar_idx)

        if self.order_generation_mode_str == "vanilla_current_bar":
            signal_bar_set = set(pd.Timestamp(bar_ts) for bar_ts in calendar_idx)
        else:
            signal_bar_set = set(pd.Timestamp(bar_ts) for bar_ts in calendar_idx[:-1])
            if previous_start_bar_ts is not None:
                signal_bar_set.add(previous_start_bar_ts)

        open_schedule_map: dict[pd.Timestamp, list[ScheduledOrder]] = {}
        close_schedule_map: dict[pd.Timestamp, list[ScheduledOrder]] = {}
        portfolio_value_map: dict[pd.Timestamp, float] = {}
        cash_value_map: dict[pd.Timestamp, float] = {}
        total_value_map: dict[pd.Timestamp, float] = {}

        for bar_ts in engine_bar_idx:
            bar_ts = pd.Timestamp(bar_ts)
            strategy_obj.current_bar = bar_ts

            if self.order_generation_mode_str == "vanilla_current_bar":
                previous_bar_ts = _previous_bar_ts(full_index, bar_ts)
                if previous_bar_ts is not None:
                    # *** CRITICAL*** Current-bar rebalance strategies size
                    # from the prior close before the rebalance open is known.
                    # Marking to the same-day close here would leak information
                    # into percent/value target order sizing.
                    previous_close_price_ser = _close_price_ser(signal_data_df, previous_bar_ts)
                    _liquidate_missing_close_positions(
                        strategy_obj=strategy_obj,
                        signal_data_df=signal_data_df,
                        bar_ts=previous_bar_ts,
                        close_price_ser=previous_close_price_ser,
                    )
                    pre_open_portfolio_value_float = _active_portfolio_value_float(
                        strategy_obj,
                        previous_close_price_ser,
                    )
                    pre_open_total_value_float = float(
                        strategy_obj.cash + pre_open_portfolio_value_float
                    )
                    strategy_obj.portfolio_value = pre_open_portfolio_value_float
                    strategy_obj.total_value = pre_open_total_value_float
                    strategy_obj._total_value_history_list = [pre_open_total_value_float]

                if bar_ts in signal_bar_set:
                    self._generate_and_schedule_orders(
                        strategy_obj=strategy_obj,
                        signal_data_df=signal_data_df,
                        full_index=full_index,
                        signal_bar_ts=bar_ts,
                        entry_timing_str=entry_timing_str,
                        exit_timing_str=exit_timing_str,
                        open_schedule_map=open_schedule_map,
                        close_schedule_map=close_schedule_map,
                    )

                close_price_ser = _close_price_ser(signal_data_df, bar_ts)
                _liquidate_missing_close_positions(
                    strategy_obj=strategy_obj,
                    signal_data_df=signal_data_df,
                    bar_ts=bar_ts,
                    close_price_ser=close_price_ser,
                )

                _process_scheduled_order_list(
                    strategy_obj=strategy_obj,
                    signal_data_df=signal_data_df,
                    scheduled_order_list=open_schedule_map.pop(bar_ts, []),
                )

                close_price_ser = _close_price_ser(signal_data_df, bar_ts)
                _liquidate_missing_close_positions(
                    strategy_obj=strategy_obj,
                    signal_data_df=signal_data_df,
                    bar_ts=bar_ts,
                    close_price_ser=close_price_ser,
                )

                _process_scheduled_order_list(
                    strategy_obj=strategy_obj,
                    signal_data_df=signal_data_df,
                    scheduled_order_list=close_schedule_map.pop(bar_ts, []),
                )

                close_price_ser = _close_price_ser(signal_data_df, bar_ts)
                _liquidate_missing_close_positions(
                    strategy_obj=strategy_obj,
                    signal_data_df=signal_data_df,
                    bar_ts=bar_ts,
                    close_price_ser=close_price_ser,
                )
                portfolio_value_float = _active_portfolio_value_float(strategy_obj, close_price_ser)
                total_value_float = float(strategy_obj.cash + portfolio_value_float)
                strategy_obj.portfolio_value = portfolio_value_float
                strategy_obj.total_value = total_value_float

                if bar_ts in calendar_bar_set:
                    portfolio_value_map[bar_ts] = portfolio_value_float
                    cash_value_map[bar_ts] = float(strategy_obj.cash)
                    total_value_map[bar_ts] = total_value_float

                continue

            close_price_ser = _close_price_ser(signal_data_df, bar_ts)
            _liquidate_missing_close_positions(
                strategy_obj=strategy_obj,
                signal_data_df=signal_data_df,
                bar_ts=bar_ts,
                close_price_ser=close_price_ser,
            )

            _process_scheduled_order_list(
                strategy_obj=strategy_obj,
                signal_data_df=signal_data_df,
                scheduled_order_list=open_schedule_map.pop(bar_ts, []),
            )

            close_price_ser = _close_price_ser(signal_data_df, bar_ts)
            _liquidate_missing_close_positions(
                strategy_obj=strategy_obj,
                signal_data_df=signal_data_df,
                bar_ts=bar_ts,
                close_price_ser=close_price_ser,
            )
            pre_signal_portfolio_value_float = _active_portfolio_value_float(strategy_obj, close_price_ser)
            pre_signal_total_value_float = float(strategy_obj.cash + pre_signal_portfolio_value_float)
            strategy_obj.portfolio_value = pre_signal_portfolio_value_float
            strategy_obj.total_value = pre_signal_total_value_float
            strategy_obj._total_value_history_list = [pre_signal_total_value_float]

            if bar_ts in signal_bar_set:
                self._generate_and_schedule_orders(
                    strategy_obj=strategy_obj,
                    signal_data_df=signal_data_df,
                    full_index=full_index,
                    signal_bar_ts=bar_ts,
                    entry_timing_str=entry_timing_str,
                    exit_timing_str=exit_timing_str,
                    open_schedule_map=open_schedule_map,
                    close_schedule_map=close_schedule_map,
                )

            _process_scheduled_order_list(
                strategy_obj=strategy_obj,
                signal_data_df=signal_data_df,
                scheduled_order_list=close_schedule_map.pop(bar_ts, []),
            )

            close_price_ser = _close_price_ser(signal_data_df, bar_ts)
            _liquidate_missing_close_positions(
                strategy_obj=strategy_obj,
                signal_data_df=signal_data_df,
                bar_ts=bar_ts,
                close_price_ser=close_price_ser,
            )
            portfolio_value_float = _active_portfolio_value_float(strategy_obj, close_price_ser)
            total_value_float = float(strategy_obj.cash + portfolio_value_float)
            strategy_obj.portfolio_value = portfolio_value_float
            strategy_obj.total_value = total_value_float

            if bar_ts in calendar_bar_set:
                portfolio_value_map[bar_ts] = portfolio_value_float
                cash_value_map[bar_ts] = float(strategy_obj.cash)
                total_value_map[bar_ts] = total_value_float

        strategy_obj.results = _build_results_df(
            strategy_obj=strategy_obj,
            pricing_data_df=self.pricing_data_df,
            calendar_idx=calendar_idx,
            portfolio_value_map=portfolio_value_map,
            cash_value_map=cash_value_map,
            total_value_map=total_value_map,
        )
        strategy_obj._latest_close_price_ser = _close_price_ser(signal_data_df, pd.Timestamp(calendar_idx[-1]))
        strategy_obj.current_bar = pd.Timestamp(calendar_idx[-1])
        strategy_obj.cash = float(cash_value_map[pd.Timestamp(calendar_idx[-1])])
        strategy_obj.portfolio_value = float(portfolio_value_map[pd.Timestamp(calendar_idx[-1])])
        strategy_obj.total_value = float(total_value_map[pd.Timestamp(calendar_idx[-1])])
        strategy_obj.summarize()
        return strategy_obj

    def _generate_and_schedule_orders(
        self,
        strategy_obj: Strategy,
        signal_data_df: pd.DataFrame,
        full_index: pd.DatetimeIndex,
        signal_bar_ts: pd.Timestamp,
        entry_timing_str: str,
        exit_timing_str: str,
        open_schedule_map: dict[pd.Timestamp, list[ScheduledOrder]],
        close_schedule_map: dict[pd.Timestamp, list[ScheduledOrder]],
    ) -> None:
        pre_signal_entry_capacity_int = _entry_capacity_int(strategy_obj)
        if self.order_generation_mode_str == "vanilla_current_bar":
            previous_bar_ts = _previous_bar_ts(full_index, signal_bar_ts)
            if previous_bar_ts is None:
                return

            # *** CRITICAL*** In vanilla-current-bar mode, the order intent is
            # created on current_bar using data only through previous_bar. This
            # preserves TAA rebalance semantics: prior month-end/close signal,
            # current open execution opportunity.
            strategy_obj.previous_bar = previous_bar_ts
            strategy_obj.current_bar = signal_bar_ts
            data_df = signal_data_df.loc[:previous_bar_ts]
            close_row_ser = signal_data_df.loc[previous_bar_ts]
            sizing_bar_ts = previous_bar_ts
        else:
            strategy_obj.previous_bar = signal_bar_ts
            strategy_obj.current_bar = signal_bar_ts
            data_df = signal_data_df.loc[:signal_bar_ts]
            close_row_ser = signal_data_df.loc[signal_bar_ts]
            sizing_bar_ts = signal_bar_ts

        open_price_ser = _open_price_ser(signal_data_df, signal_bar_ts)

        strategy_obj.clear_orders()
        strategy_obj.iterate(data_df, close_row_ser, open_price_ser)
        order_list = list(strategy_obj.get_orders())
        strategy_obj.clear_orders()

        order_info_list: list[ScheduledOrder] = []
        for sequence_int, order_obj in enumerate(order_list):
            if not isinstance(order_obj, MarketOrder):
                raise ValueError(
                    "ExecutionTimingAnalysis v1 supports market orders only. "
                    f"Found {order_obj.__class__.__name__} for {order_obj.asset}."
                )

            fallback_open_price_float = float(open_price_ser.loc[order_obj.asset])
            sizing_price_float = _sizing_price_float(
                signal_data_df=signal_data_df,
                order_obj=order_obj,
                signal_bar_ts=sizing_bar_ts,
                fallback_price_float=fallback_open_price_float,
            )
            current_position_float = float(strategy_obj.get_position(order_obj.asset))
            amount_float = float(
                order_obj.amount_in_shares(
                    sizing_price_float,
                    float(strategy_obj.total_value),
                    current_position_float,
                )
            )
            order_kind_str = _classify_order_kind_str(current_position_float, amount_float)
            if order_kind_str == "flat":
                continue

            timing_rule = get_execution_timing_rule(
                entry_timing_str if order_kind_str == "entry" else exit_timing_str
            )
            fill_bar_ts = _resolve_fill_bar_ts(full_index, signal_bar_ts, timing_rule)
            if fill_bar_ts is None:
                continue

            order_info_list.append(
                ScheduledOrder(
                    order_obj=order_obj,
                    order_kind_str=order_kind_str,
                    sequence_int=sequence_int,
                    signal_bar_ts=signal_bar_ts,
                    fill_bar_ts=fill_bar_ts,
                    fill_price_field_str=timing_rule.price_field_str,
                    fill_phase_str=timing_rule.fill_phase_str,
                    sizing_price_float=float(sizing_price_float),
                    sizing_portfolio_value_float=float(strategy_obj.total_value),
                )
            )

        order_info_list = _filter_entry_orders_for_delayed_exits(
            order_info_list=order_info_list,
            entry_timing_str=entry_timing_str,
            exit_timing_str=exit_timing_str,
            pre_signal_entry_capacity_int=pre_signal_entry_capacity_int,
            )

        immediate_open_order_info_list: list[ScheduledOrder] = []
        for order_info in order_info_list:
            if order_info.fill_bar_ts == signal_bar_ts and order_info.fill_phase_str == "open":
                immediate_open_order_info_list.append(order_info)
                continue

            target_schedule_map = (
                open_schedule_map if order_info.fill_phase_str == "open" else close_schedule_map
            )
            target_schedule_map.setdefault(order_info.fill_bar_ts, []).append(order_info)

        _process_scheduled_order_list(
            strategy_obj=strategy_obj,
            signal_data_df=signal_data_df,
            scheduled_order_list=immediate_open_order_info_list,
        )


__all__ = [
    "ExecutionTimingAnalysis",
    "ExecutionTimingAnalysisResult",
    "DEFAULT_SIGNAL_CLOSE_TIMING_MODE_TUPLE",
    "DEFAULT_TAA_REBALANCE_TIMING_MODE_TUPLE",
    "SUPPORTED_ORDER_GENERATION_MODE_TUPLE",
    "SUPPORTED_RISK_MODEL_TUPLE",
    "SUPPORTED_TIMING_MODE_TUPLE",
    "compute_cvar_5_pct_float",
    "get_execution_timing_rule",
    "save_execution_timing_results",
]
