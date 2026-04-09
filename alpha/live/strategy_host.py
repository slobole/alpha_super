"""
SUMMARY: bridge research strategy to live frozen plans

FrozenOrderPlan=f(data load,state seed,signal timing,order extraction)


Here are the important functions:

_seed_strategy_state
Loads the saved live pod state into the strategy object before running it.
Why it exists: so live runs continue from real positions/cash/state instead of starting fresh.

_extract_strategy_state_dict
Pulls the strategy’s internal memory back out after the decision is made.
Why it exists: so the next live run can continue from the updated state.

_build_broker_order_type_str
Converts internal order/execution mode into a broker-facing order type like MOO or MOC.
Why it exists: so the live layer knows how the order should be sent.

_build_order_intent_list
Converts the strategy’s pending orders into FrozenOrderIntent objects.
Why it exists: so the raw strategy decision becomes a clean, auditable list of planned trades.

_build_frozen_order_plan
Wraps signal time, submit time, execution time, strategy state, and intents into one FrozenOrderPlan.
Why it exists: so decision and execution are separated cleanly.

_build_dv2_order_plan
Runs the DV2 research logic in live-hosted form and produces a frozen plan.
Why it exists: this is the DV2-specific adapter from research code to live flow.

_build_taa_btal_tqqq_vix_cash_order_plan
Runs the monthly TAA variant in live-hosted form and produces a frozen plan.
Why it exists: this is the TAA-specific adapter.

_build_atr_normalized_ndx_order_plan
Runs the monthly ATR-normalized NDX strategy in live-hosted form and produces a frozen plan.
Why it exists: this is the momentum-specific adapter.

build_order_plan_for_release
Main entrypoint that chooses the correct strategy adapter based on the manifest.
Why it exists: so the runner can call one generic function, while the host routes to the correct strategy implementation.

Shortest possible version:

state in
run research strategy safely
convert orders to intents
freeze one plan
save updated strategy state

"""


from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from importlib import import_module
from typing import Any

import pandas as pd

from alpha.engine.order import LimitOrder, MarketOrder, StopOrder
from alpha.engine.strategy import Strategy
from alpha.live import scheduler_utils
from alpha.live.models import DecisionPlan, FrozenOrderIntent, FrozenOrderPlan, LiveRelease, PodState


def _seed_strategy_state(strategy_obj: Strategy, pod_state_obj: PodState | None) -> None:
    if pod_state_obj is None:
        return

    strategy_obj._position_amount_map = {
        asset_str: float(amount_float)
        for asset_str, amount_float in pod_state_obj.position_amount_map.items()
    }
    strategy_obj._total_value_history_list = [float(pod_state_obj.total_value_float)]
    strategy_obj.cash = float(pod_state_obj.cash_float)

    strategy_state_dict = dict(pod_state_obj.strategy_state_dict)
    if "trade_id_int" in strategy_state_dict and hasattr(strategy_obj, "trade_id_int"):
        strategy_obj.trade_id_int = int(strategy_state_dict["trade_id_int"])
    if "trade_id_int" in strategy_state_dict and hasattr(strategy_obj, "trade_id"):
        strategy_obj.trade_id = int(strategy_state_dict["trade_id_int"])
    if "current_trade_map" in strategy_state_dict:
        current_trade_map_dict = {
            str(asset_str): int(trade_id_int)
            for asset_str, trade_id_int in strategy_state_dict["current_trade_map"].items()
        }
        if hasattr(strategy_obj, "current_trade_map"):
            strategy_obj.current_trade_map = defaultdict(lambda: -1, current_trade_map_dict)
        if hasattr(strategy_obj, "current_trade"):
            strategy_obj.current_trade = defaultdict(lambda: -1, current_trade_map_dict)


def _extract_strategy_state_dict(strategy_obj: Strategy) -> dict[str, Any]:
    strategy_state_dict: dict[str, Any] = {}
    if hasattr(strategy_obj, "trade_id_int"):
        strategy_state_dict["trade_id_int"] = int(strategy_obj.trade_id_int)
    elif hasattr(strategy_obj, "trade_id"):
        strategy_state_dict["trade_id_int"] = int(strategy_obj.trade_id)

    if hasattr(strategy_obj, "current_trade_map"):
        strategy_state_dict["current_trade_map"] = {
            str(asset_str): int(trade_id_int)
            for asset_str, trade_id_int in dict(strategy_obj.current_trade_map).items()
        }
    elif hasattr(strategy_obj, "current_trade"):
        strategy_state_dict["current_trade_map"] = {
            str(asset_str): int(trade_id_int)
            for asset_str, trade_id_int in dict(strategy_obj.current_trade).items()
        }
    return strategy_state_dict


def _build_broker_order_type_str(order_obj, execution_policy_str: str) -> str:
    if isinstance(order_obj, LimitOrder):
        return "LIMIT"
    if isinstance(order_obj, StopOrder):
        return "STOP"
    if execution_policy_str == "next_open_moo":
        return "MOO"
    if execution_policy_str == "same_day_moc":
        return "MOC"
    if execution_policy_str == "next_month_first_open":
        return "MOO"
    if isinstance(order_obj, MarketOrder):
        return "MARKET"
    raise ValueError(f"Unsupported order/execution combination for {type(order_obj).__name__}.")


def _build_order_intent_list(
    strategy_obj: Strategy,
    close_row_ser: pd.Series,
    execution_policy_str: str,
) -> list[FrozenOrderIntent]:
    order_intent_list: list[FrozenOrderIntent] = []
    portfolio_value_float = float(strategy_obj.previous_total_value)

    for order_obj in strategy_obj.get_orders():
        close_price_key = (order_obj.asset, "Close")
        if close_price_key not in close_row_ser.index:
            raise RuntimeError(
                f"Missing close price for asset {order_obj.asset} while freezing live order intents."
            )
        sizing_reference_price_float = float(close_row_ser[close_price_key])
        order_intent_list.append(
            FrozenOrderIntent(
                asset_str=str(order_obj.asset),
                order_class_str=type(order_obj).__name__,
                unit_str=str(order_obj.unit),
                amount_float=float(order_obj.amount),
                target_bool=bool(order_obj.target),
                trade_id_int=order_obj.trade_id,
                broker_order_type_str=_build_broker_order_type_str(order_obj, execution_policy_str),
                sizing_reference_price_float=sizing_reference_price_float,
                portfolio_value_float=portfolio_value_float,
            )
        )
    return order_intent_list


def _build_frozen_order_plan(
    release_obj: LiveRelease,
    signal_date_ts: datetime,
    close_row_ser: pd.Series,
    strategy_obj: Strategy,
    snapshot_metadata_dict: dict[str, Any] | None = None,
) -> FrozenOrderPlan:
    signal_timestamp_ts = scheduler_utils.build_signal_timestamp_ts(
        signal_date_ts=signal_date_ts,
        release_obj=release_obj,
    )
    submission_timestamp_ts = scheduler_utils.build_submission_timestamp_ts(
        signal_date_ts=signal_date_ts,
        release_obj=release_obj,
    )
    target_execution_timestamp_ts = scheduler_utils.build_target_execution_timestamp_ts(
        signal_date_ts=signal_date_ts,
        release_obj=release_obj,
    )
    order_intent_list = _build_order_intent_list(
        strategy_obj=strategy_obj,
        close_row_ser=close_row_ser,
        execution_policy_str=release_obj.execution_policy_str,
    )
    return FrozenOrderPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=signal_timestamp_ts,
        submission_timestamp_ts=submission_timestamp_ts,
        target_execution_timestamp_ts=target_execution_timestamp_ts,
        execution_policy_str=release_obj.execution_policy_str,
        snapshot_metadata_dict=snapshot_metadata_dict or {},
        strategy_state_dict=_extract_strategy_state_dict(strategy_obj),
        order_intent_list=order_intent_list,
    )


def _build_decision_plan_from_orders(
    release_obj: LiveRelease,
    signal_date_ts: datetime,
    strategy_obj: Strategy,
    snapshot_metadata_dict: dict[str, Any] | None = None,
) -> DecisionPlan:
    signal_timestamp_ts = scheduler_utils.build_signal_timestamp_ts(
        signal_date_ts=signal_date_ts,
        release_obj=release_obj,
    )
    submission_timestamp_ts = scheduler_utils.build_submission_timestamp_ts(
        signal_date_ts=signal_date_ts,
        release_obj=release_obj,
    )
    target_execution_timestamp_ts = scheduler_utils.build_target_execution_timestamp_ts(
        signal_date_ts=signal_date_ts,
        release_obj=release_obj,
    )

    decision_base_position_map = {
        str(asset_str): float(amount_float)
        for asset_str, amount_float in strategy_obj.get_positions().items()
        if abs(float(amount_float)) > 1e-9
    }
    previous_total_value_float = float(strategy_obj.previous_total_value)
    entry_target_weight_map_dict: dict[str, float] = {}
    exit_asset_set: set[str] = set()
    entry_priority_list: list[str] = []

    for order_obj in strategy_obj.get_orders():
        if bool(order_obj.target) and str(order_obj.unit) == "value" and abs(float(order_obj.amount)) <= 1e-9:
            exit_asset_set.add(str(order_obj.asset))
            continue
        if (not bool(order_obj.target)) and str(order_obj.unit) == "value" and float(order_obj.amount) > 0.0:
            entry_target_weight_map_dict[str(order_obj.asset)] = (
                float(order_obj.amount) / previous_total_value_float
            )
            entry_priority_list.append(str(order_obj.asset))
            continue
        raise NotImplementedError(
            "incremental_entry_exit_book currently supports only value entries "
            "and target-zero exits."
        )

    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=signal_timestamp_ts,
        submission_timestamp_ts=submission_timestamp_ts,
        target_execution_timestamp_ts=target_execution_timestamp_ts,
        execution_policy_str=release_obj.execution_policy_str,
        decision_base_position_map=decision_base_position_map,
        snapshot_metadata_dict=snapshot_metadata_dict or {},
        strategy_state_dict=_extract_strategy_state_dict(strategy_obj),
        decision_book_type_str="incremental_entry_exit_book",
        entry_target_weight_map_dict=entry_target_weight_map_dict,
        target_weight_map=entry_target_weight_map_dict,
        exit_asset_set=exit_asset_set,
        entry_priority_list=entry_priority_list,
        cash_reserve_weight_float=0.0,
        preserve_untouched_positions_bool=True,
    )


def _build_full_target_weight_decision_plan(
    release_obj: LiveRelease,
    signal_date_ts: datetime,
    decision_base_position_map_dict: dict[str, float],
    full_target_weight_map_dict: dict[str, float],
    cash_reserve_weight_float: float,
    strategy_state_dict: dict[str, Any],
    snapshot_metadata_dict: dict[str, Any] | None = None,
) -> DecisionPlan:
    signal_timestamp_ts = scheduler_utils.build_signal_timestamp_ts(
        signal_date_ts=signal_date_ts,
        release_obj=release_obj,
    )
    submission_timestamp_ts = scheduler_utils.build_submission_timestamp_ts(
        signal_date_ts=signal_date_ts,
        release_obj=release_obj,
    )
    target_execution_timestamp_ts = scheduler_utils.build_target_execution_timestamp_ts(
        signal_date_ts=signal_date_ts,
        release_obj=release_obj,
    )

    full_target_weight_map_dict = {
        str(asset_str): float(target_weight_float)
        for asset_str, target_weight_float in full_target_weight_map_dict.items()
        if abs(float(target_weight_float)) > 1e-12
    }
    total_target_weight_float = sum(full_target_weight_map_dict.values())
    if total_target_weight_float + float(cash_reserve_weight_float) > 1.0 + 1e-9:
        raise ValueError(
            "full_target_weight_book weights plus cash reserve must satisfy "
            "sum(weights) + cash_reserve <= 1."
        )

    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=signal_timestamp_ts,
        submission_timestamp_ts=submission_timestamp_ts,
        target_execution_timestamp_ts=target_execution_timestamp_ts,
        execution_policy_str=release_obj.execution_policy_str,
        decision_base_position_map=decision_base_position_map_dict,
        snapshot_metadata_dict=snapshot_metadata_dict or {},
        strategy_state_dict=strategy_state_dict,
        decision_book_type_str="full_target_weight_book",
        full_target_weight_map_dict=full_target_weight_map_dict,
        target_weight_map=full_target_weight_map_dict,
        cash_reserve_weight_float=float(cash_reserve_weight_float),
        preserve_untouched_positions_bool=False,
        rebalance_omitted_assets_to_zero_bool=True,
    )


def _build_dv2_order_plan(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState | None,
) -> FrozenOrderPlan:
    dv2_module = import_module("strategies.dv2.strategy_mr_dv2")
    benchmark_list = release_obj.params_dict.get("benchmark_list_str", ["$SPX"])
    start_date_str = str(release_obj.params_dict.get("start_date_str", "1998-01-01"))
    indexname_str = str(release_obj.params_dict.get("indexname_str", "S&P 500"))

    _, universe_df = dv2_module.build_index_constituent_matrix(indexname=indexname_str)
    pricing_data_df = dv2_module.get_prices(
        universe_df.columns.tolist(),
        benchmark_list,
        start_date=start_date_str,
        end_date=pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
    )
    if len(pricing_data_df.index) == 0:
        raise RuntimeError("DV2 live host loaded no pricing data.")

    strategy_obj = dv2_module.DVO2Strategy(
        name=release_obj.pod_id_str,
        benchmarks=list(benchmark_list),
        capital_base=float(release_obj.params_dict.get("capital_base_float", 100_000.0)),
        slippage=float(release_obj.params_dict.get("slippage_float", 0.0001)),
        commission_per_share=float(release_obj.params_dict.get("commission_per_share_float", 0.005)),
        commission_minimum=float(release_obj.params_dict.get("commission_minimum_float", 1.0)),
    )
    strategy_obj.max_positions = int(release_obj.params_dict.get("max_positions_int", 10))
    strategy_obj.trade_id = 0
    strategy_obj.current_trade = defaultdict(lambda: -1)
    strategy_obj.universe_df = universe_df.loc[universe_df.index.isin(pricing_data_df.index)].copy()
    _seed_strategy_state(strategy_obj, pod_state_obj)

    full_signal_df = strategy_obj.compute_signals(pricing_data_df.copy())
    signal_date_ts = pd.Timestamp(pricing_data_df.index[-1]).to_pydatetime()
    strategy_obj.previous_bar = pd.Timestamp(signal_date_ts)
    # *** CRITICAL*** The live host must map the signal session to the true next tradable session on the pod calendar.
    strategy_obj.current_bar = pd.Timestamp(
        scheduler_utils.next_business_day_timestamp_ts(
            signal_date_ts,
            session_calendar_id_str=release_obj.session_calendar_id_str,
        ).date()
    )
    close_row_ser = full_signal_df.loc[pd.Timestamp(signal_date_ts)]
    current_data_df = full_signal_df.loc[: pd.Timestamp(signal_date_ts)]
    strategy_obj.iterate(
        current_data_df,
        close_row_ser,
        pd.Series(dtype=float),
    )

    return _build_frozen_order_plan(
        release_obj=release_obj,
        signal_date_ts=signal_date_ts,
        close_row_ser=close_row_ser,
        strategy_obj=strategy_obj,
        snapshot_metadata_dict={"strategy_family_str": "dv2"},
    )


def _build_dv2_decision_plan(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState | None,
) -> DecisionPlan:
    dv2_module = import_module("strategies.dv2.strategy_mr_dv2")
    benchmark_list = release_obj.params_dict.get("benchmark_list_str", ["$SPX"])
    start_date_str = str(release_obj.params_dict.get("start_date_str", "1998-01-01"))
    indexname_str = str(release_obj.params_dict.get("indexname_str", "S&P 500"))

    _, universe_df = dv2_module.build_index_constituent_matrix(indexname=indexname_str)
    pricing_data_df = dv2_module.get_prices(
        universe_df.columns.tolist(),
        benchmark_list,
        start_date=start_date_str,
        end_date=pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
    )
    if len(pricing_data_df.index) == 0:
        raise RuntimeError("DV2 live host loaded no pricing data.")

    strategy_obj = dv2_module.DVO2Strategy(
        name=release_obj.pod_id_str,
        benchmarks=list(benchmark_list),
        capital_base=float(release_obj.params_dict.get("capital_base_float", 100_000.0)),
        slippage=float(release_obj.params_dict.get("slippage_float", 0.0001)),
        commission_per_share=float(release_obj.params_dict.get("commission_per_share_float", 0.005)),
        commission_minimum=float(release_obj.params_dict.get("commission_minimum_float", 1.0)),
    )
    strategy_obj.max_positions = int(release_obj.params_dict.get("max_positions_int", 10))
    strategy_obj.trade_id = 0
    strategy_obj.current_trade = defaultdict(lambda: -1)
    strategy_obj.universe_df = universe_df.loc[universe_df.index.isin(pricing_data_df.index)].copy()
    _seed_strategy_state(strategy_obj, pod_state_obj)

    full_signal_df = strategy_obj.compute_signals(pricing_data_df.copy())
    signal_date_ts = pd.Timestamp(pricing_data_df.index[-1]).to_pydatetime()
    strategy_obj.previous_bar = pd.Timestamp(signal_date_ts)
    # *** CRITICAL*** The live host must map the signal session to the true next tradable session on the pod calendar.
    strategy_obj.current_bar = pd.Timestamp(
        scheduler_utils.next_business_day_timestamp_ts(
            signal_date_ts,
            session_calendar_id_str=release_obj.session_calendar_id_str,
        ).date()
    )
    close_row_ser = full_signal_df.loc[pd.Timestamp(signal_date_ts)]
    current_data_df = full_signal_df.loc[: pd.Timestamp(signal_date_ts)]
    strategy_obj.iterate(
        current_data_df,
        close_row_ser,
        pd.Series(dtype=float),
    )

    return _build_decision_plan_from_orders(
        release_obj=release_obj,
        signal_date_ts=signal_date_ts,
        strategy_obj=strategy_obj,
        snapshot_metadata_dict={"strategy_family_str": "dv2"},
    )


def _build_qpi_ibs_rsi_exit_decision_plan(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState | None,
) -> DecisionPlan:
    qpi_module = import_module("strategies.qpi.strategy_mr_qpi_ibs_rsi_exit")
    benchmark_list = release_obj.params_dict.get("benchmark_list_str", ["$SPX"])
    start_date_str = str(release_obj.params_dict.get("start_date_str", "1998-01-01"))
    indexname_str = str(release_obj.params_dict.get("indexname_str", "S&P 500"))

    _, universe_df = qpi_module.build_index_constituent_matrix(indexname=indexname_str)
    pricing_data_df = qpi_module.get_prices(
        universe_df.columns.tolist(),
        benchmark_list,
        start_date_str=start_date_str,
        end_date_str=pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
    )
    if len(pricing_data_df.index) == 0:
        raise RuntimeError("QPI IBS RSI exit live host loaded no pricing data.")

    strategy_obj = qpi_module.QPIIbsRsiExitStrategy(
        name=release_obj.pod_id_str,
        benchmarks=list(benchmark_list),
        capital_base=float(release_obj.params_dict.get("capital_base_float", 100_000.0)),
        slippage=float(release_obj.params_dict.get("slippage_float", 0.0001)),
        commission_per_share=float(release_obj.params_dict.get("commission_per_share_float", 0.005)),
        commission_minimum=float(release_obj.params_dict.get("commission_minimum_float", 1.0)),
        max_positions_int=int(release_obj.params_dict.get("max_positions_int", 10)),
        qpi_threshold_float=float(release_obj.params_dict.get("qpi_threshold_float", 30.0)),
        sma_window_int=int(release_obj.params_dict.get("sma_window_int", 200)),
        qpi_window_int=int(release_obj.params_dict.get("qpi_window_int", 3)),
        qpi_lookback_years_int=int(release_obj.params_dict.get("qpi_lookback_years_int", 5)),
        return_lookback_days_int=int(release_obj.params_dict.get("return_lookback_days_int", 3)),
        max_entry_ibs_float=float(release_obj.params_dict.get("max_entry_ibs_float", 0.1)),
        exit_ibs_threshold_float=float(release_obj.params_dict.get("exit_ibs_threshold_float", 0.90)),
        rsi_window_int=int(release_obj.params_dict.get("rsi_window_int", 2)),
        exit_rsi2_threshold_float=float(release_obj.params_dict.get("exit_rsi2_threshold_float", 90.0)),
    )
    strategy_obj.universe_df = universe_df.loc[universe_df.index.isin(pricing_data_df.index)].copy()
    _seed_strategy_state(strategy_obj, pod_state_obj)

    full_signal_df = strategy_obj.compute_signals(pricing_data_df.copy())
    signal_date_ts = pd.Timestamp(pricing_data_df.index[-1]).to_pydatetime()
    strategy_obj.previous_bar = pd.Timestamp(signal_date_ts)
    # *** CRITICAL*** The live host must map the signal session to the true next tradable session on the pod calendar.
    strategy_obj.current_bar = pd.Timestamp(
        scheduler_utils.next_business_day_timestamp_ts(
            signal_date_ts,
            session_calendar_id_str=release_obj.session_calendar_id_str,
        ).date()
    )
    close_row_ser = full_signal_df.loc[pd.Timestamp(signal_date_ts)]
    current_data_df = full_signal_df.loc[: pd.Timestamp(signal_date_ts)]
    strategy_obj.iterate(
        current_data_df,
        close_row_ser,
        pd.Series(dtype=float),
    )

    return _build_decision_plan_from_orders(
        release_obj=release_obj,
        signal_date_ts=signal_date_ts,
        strategy_obj=strategy_obj,
        snapshot_metadata_dict={"strategy_family_str": "qpi_ibs_rsi_exit"},
    )


def _build_taa_btal_tqqq_vix_cash_decision_plan(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState | None,
) -> DecisionPlan:
    base_taa_module = import_module("strategies.taa_df.strategy_taa_df")
    variant_module = import_module("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash")
    vix_overlay_module = import_module("strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils")

    config_obj = replace(
        variant_module.DEFAULT_CONFIG,
        end_date_str=pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
    )
    execution_price_df, _, base_month_end_weight_df, _ = base_taa_module.get_defense_first_data(config_obj)
    _, month_end_vrp_signal_df = vix_overlay_module._load_vrp_overlay_signal_frames(config_obj)
    month_end_weight_df, month_end_vrp_diagnostic_df = (
        vix_overlay_module.apply_vrp_cash_gate_to_month_end_weight_df(
            base_month_end_weight_df=base_month_end_weight_df,
            month_end_vrp_signal_df=month_end_vrp_signal_df,
            config=config_obj,
        )
    )
    if len(month_end_weight_df.index) == 0:
        raise RuntimeError("TAA live host produced no month-end weights.")

    signal_date_ts = pd.Timestamp(month_end_weight_df.index[-1]).to_pydatetime()
    # *** CRITICAL*** Month-end rebalance dates must snap to the true first tradable session of the next month.
    execution_date_ts = scheduler_utils.first_business_day_of_next_month_timestamp_ts(
        signal_date_ts,
        session_calendar_id_str=release_obj.session_calendar_id_str,
    )
    rebalance_weight_df = pd.DataFrame(
        [month_end_weight_df.loc[pd.Timestamp(signal_date_ts)].copy()],
        index=pd.DatetimeIndex([pd.Timestamp(execution_date_ts.date())], name="rebalance_date"),
    )

    strategy_obj = base_taa_module.DefenseFirstStrategy(
        name=release_obj.pod_id_str,
        benchmarks=config_obj.benchmark_list,
        rebalance_weight_df=rebalance_weight_df,
        tradeable_asset_list=config_obj.tradeable_asset_list,
        capital_base=float(release_obj.params_dict.get("capital_base_float", 100_000.0)),
        slippage=float(release_obj.params_dict.get("slippage_float", 0.0001)),
        commission_per_share=float(release_obj.params_dict.get("commission_per_share_float", 0.005)),
        commission_minimum=float(release_obj.params_dict.get("commission_minimum_float", 1.0)),
    )
    _seed_strategy_state(strategy_obj, pod_state_obj)

    close_row_ser = execution_price_df.loc[pd.Timestamp(signal_date_ts)]
    strategy_obj.previous_bar = pd.Timestamp(signal_date_ts)
    strategy_obj.current_bar = pd.Timestamp(execution_date_ts.date())
    strategy_obj.iterate(
        execution_price_df.loc[: pd.Timestamp(signal_date_ts)],
        close_row_ser,
        pd.Series(dtype=float),
    )

    full_target_weight_map_dict = {
        str(asset_str): float(target_weight_float)
        for asset_str, target_weight_float in rebalance_weight_df.loc[pd.Timestamp(execution_date_ts.date())].items()
        if abs(float(target_weight_float)) > 1e-12
    }
    cash_reserve_weight_float = max(0.0, 1.0 - sum(full_target_weight_map_dict.values()))
    latest_diagnostic_ser = month_end_vrp_diagnostic_df.loc[pd.Timestamp(signal_date_ts)]
    decision_base_position_map_dict = {
        str(asset_str): float(amount_float)
        for asset_str, amount_float in strategy_obj.get_positions().items()
        if abs(float(amount_float)) > 1e-9
    }
    return _build_full_target_weight_decision_plan(
        release_obj=release_obj,
        signal_date_ts=signal_date_ts,
        decision_base_position_map_dict=decision_base_position_map_dict,
        full_target_weight_map_dict=full_target_weight_map_dict,
        cash_reserve_weight_float=cash_reserve_weight_float,
        strategy_state_dict=_extract_strategy_state_dict(strategy_obj),
        snapshot_metadata_dict={
            "strategy_family_str": "taa_df_btal_fallback_tqqq_vix_cash",
            "cash_weight_float": float(latest_diagnostic_ser["cash_weight"]),
        },
    )


def _build_atr_normalized_ndx_decision_plan(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState | None,
) -> DecisionPlan:
    atr_module = import_module("strategies.momentum.strategy_mo_atr_normalized_ndx")
    config_obj = replace(
        atr_module.DEFAULT_CONFIG,
        end_date_str=pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
        max_positions_int=int(
            release_obj.params_dict.get("max_positions_int", atr_module.DEFAULT_CONFIG.max_positions_int)
        ),
    )
    pricing_data_df, universe_df, _ = atr_module.get_atr_normalized_ndx_data(config_obj)
    tradeable_symbol_list = [
        symbol_str
        for symbol_str in pricing_data_df.columns.get_level_values(0).unique()
        if symbol_str != config_obj.regime_symbol_str
    ]
    price_close_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Close")]
            for symbol_str in tradeable_symbol_list
        },
        index=pricing_data_df.index,
    ).astype(float)
    price_high_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "High")]
            for symbol_str in tradeable_symbol_list
        },
        index=pricing_data_df.index,
    ).astype(float)
    price_low_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Low")]
            for symbol_str in tradeable_symbol_list
        },
        index=pricing_data_df.index,
    ).astype(float)
    regime_close_ser = pricing_data_df[(config_obj.regime_symbol_str, "Close")].astype(float)

    (
        monthly_decision_close_df,
        _monthly_roc_df,
        _atr_decision_df,
        _stock_trend_pass_df,
        _regime_sma_ser,
        _regime_pass_ser,
        _risk_adj_score_df,
    ) = atr_module.compute_atr_normalized_signal_tables(
        price_close_df=price_close_df,
        price_high_df=price_high_df,
        price_low_df=price_low_df,
        regime_close_ser=regime_close_ser,
        config=config_obj,
    )
    if len(monthly_decision_close_df.index) == 0:
        raise RuntimeError("ATR-normalized NDX live host produced no valid monthly decision dates.")

    signal_date_ts = pd.Timestamp(monthly_decision_close_df.index[-1]).to_pydatetime()
    # *** CRITICAL*** The monthly decision date must map to the true next tradable session on the release calendar.
    execution_date_ts = scheduler_utils.next_business_day_timestamp_ts(
        signal_date_ts,
        session_calendar_id_str=release_obj.session_calendar_id_str,
    )
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": [pd.Timestamp(signal_date_ts)]},
        index=pd.DatetimeIndex([pd.Timestamp(execution_date_ts.date())], name="execution_date_ts"),
    )

    strategy_obj = atr_module.AtrNormalizedNdxStrategy(
        name=release_obj.pod_id_str,
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=float(release_obj.params_dict.get("capital_base_float", config_obj.capital_base_float)),
        slippage=float(release_obj.params_dict.get("slippage_float", config_obj.slippage_float)),
        commission_per_share=float(
            release_obj.params_dict.get("commission_per_share_float", config_obj.commission_per_share_float)
        ),
        commission_minimum=float(
            release_obj.params_dict.get("commission_minimum_float", config_obj.commission_minimum_float)
        ),
        lookback_month_int=int(release_obj.params_dict.get("lookback_month_int", config_obj.lookback_month_int)),
        index_trend_window_int=int(
            release_obj.params_dict.get("index_trend_window_int", config_obj.index_trend_window_int)
        ),
        stock_trend_window_int=int(
            release_obj.params_dict.get("stock_trend_window_int", config_obj.stock_trend_window_int)
        ),
        max_positions_int=int(release_obj.params_dict.get("max_positions_int", config_obj.max_positions_int)),
    )
    strategy_obj.universe_df = universe_df
    _seed_strategy_state(strategy_obj, pod_state_obj)

    full_signal_df = strategy_obj.compute_signals(pricing_data_df.copy())
    close_row_ser = full_signal_df.loc[pd.Timestamp(signal_date_ts)]
    target_weight_ser = strategy_obj.get_target_weight_ser(close_row_ser=close_row_ser)
    strategy_obj.previous_bar = pd.Timestamp(signal_date_ts)
    strategy_obj.current_bar = pd.Timestamp(execution_date_ts.date())
    strategy_obj.iterate(
        full_signal_df.loc[: pd.Timestamp(signal_date_ts)],
        close_row_ser,
        pd.Series(dtype=float),
    )

    full_target_weight_map_dict = {
        str(asset_str): float(target_weight_float)
        for asset_str, target_weight_float in target_weight_ser.items()
        if abs(float(target_weight_float)) > 1e-12
    }
    cash_reserve_weight_float = max(0.0, 1.0 - sum(full_target_weight_map_dict.values()))
    decision_base_position_map_dict = {
        str(asset_str): float(amount_float)
        for asset_str, amount_float in strategy_obj.get_positions().items()
        if abs(float(amount_float)) > 1e-9
    }
    return _build_full_target_weight_decision_plan(
        release_obj=release_obj,
        signal_date_ts=signal_date_ts,
        decision_base_position_map_dict=decision_base_position_map_dict,
        full_target_weight_map_dict=full_target_weight_map_dict,
        cash_reserve_weight_float=cash_reserve_weight_float,
        strategy_state_dict=_extract_strategy_state_dict(strategy_obj),
        snapshot_metadata_dict={"strategy_family_str": "atr_normalized_ndx"},
    )


def _build_taa_btal_tqqq_vix_cash_order_plan(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState | None,
) -> FrozenOrderPlan:
    base_taa_module = import_module("strategies.taa_df.strategy_taa_df")
    variant_module = import_module("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash")
    vix_overlay_module = import_module("strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils")

    config_obj = replace(
        variant_module.DEFAULT_CONFIG,
        end_date_str=pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
    )
    execution_price_df, _, base_month_end_weight_df, _ = base_taa_module.get_defense_first_data(config_obj)
    _, month_end_vrp_signal_df = vix_overlay_module._load_vrp_overlay_signal_frames(config_obj)
    month_end_weight_df, month_end_vrp_diagnostic_df = (
        vix_overlay_module.apply_vrp_cash_gate_to_month_end_weight_df(
            base_month_end_weight_df=base_month_end_weight_df,
            month_end_vrp_signal_df=month_end_vrp_signal_df,
            config=config_obj,
        )
    )
    if len(month_end_weight_df.index) == 0:
        raise RuntimeError("TAA live host produced no month-end weights.")

    signal_date_ts = pd.Timestamp(month_end_weight_df.index[-1]).to_pydatetime()
    # *** CRITICAL*** Month-end rebalance dates must snap to the true first tradable session of the next month.
    execution_date_ts = scheduler_utils.first_business_day_of_next_month_timestamp_ts(
        signal_date_ts,
        session_calendar_id_str=release_obj.session_calendar_id_str,
    )
    rebalance_weight_df = pd.DataFrame(
        [month_end_weight_df.loc[pd.Timestamp(signal_date_ts)].copy()],
        index=pd.DatetimeIndex([pd.Timestamp(execution_date_ts.date())], name="rebalance_date"),
    )

    strategy_obj = base_taa_module.DefenseFirstStrategy(
        name=release_obj.pod_id_str,
        benchmarks=config_obj.benchmark_list,
        rebalance_weight_df=rebalance_weight_df,
        tradeable_asset_list=config_obj.tradeable_asset_list,
        capital_base=float(release_obj.params_dict.get("capital_base_float", 100_000.0)),
        slippage=float(release_obj.params_dict.get("slippage_float", 0.0001)),
        commission_per_share=float(release_obj.params_dict.get("commission_per_share_float", 0.005)),
        commission_minimum=float(release_obj.params_dict.get("commission_minimum_float", 1.0)),
    )
    _seed_strategy_state(strategy_obj, pod_state_obj)

    close_row_ser = execution_price_df.loc[pd.Timestamp(signal_date_ts)]
    strategy_obj.previous_bar = pd.Timestamp(signal_date_ts)
    strategy_obj.current_bar = pd.Timestamp(execution_date_ts.date())
    strategy_obj.iterate(
        execution_price_df.loc[: pd.Timestamp(signal_date_ts)],
        close_row_ser,
        pd.Series(dtype=float),
    )

    latest_diagnostic_ser = month_end_vrp_diagnostic_df.loc[pd.Timestamp(signal_date_ts)]
    return _build_frozen_order_plan(
        release_obj=release_obj,
        signal_date_ts=signal_date_ts,
        close_row_ser=close_row_ser,
        strategy_obj=strategy_obj,
        snapshot_metadata_dict={
            "strategy_family_str": "taa_df_btal_fallback_tqqq_vix_cash",
            "cash_weight_float": float(latest_diagnostic_ser["cash_weight"]),
        },
    )


def _build_atr_normalized_ndx_order_plan(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState | None,
) -> FrozenOrderPlan:
    atr_module = import_module("strategies.momentum.strategy_mo_atr_normalized_ndx")
    config_obj = replace(
        atr_module.DEFAULT_CONFIG,
        end_date_str=pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
        max_positions_int=int(release_obj.params_dict.get("max_positions_int", atr_module.DEFAULT_CONFIG.max_positions_int)),
    )
    pricing_data_df, universe_df, _ = atr_module.get_atr_normalized_ndx_data(config_obj)
    tradeable_symbol_list = [
        symbol_str
        for symbol_str in pricing_data_df.columns.get_level_values(0).unique()
        if symbol_str != config_obj.regime_symbol_str
    ]
    price_close_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Close")]
            for symbol_str in tradeable_symbol_list
        },
        index=pricing_data_df.index,
    ).astype(float)
    price_high_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "High")]
            for symbol_str in tradeable_symbol_list
        },
        index=pricing_data_df.index,
    ).astype(float)
    price_low_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Low")]
            for symbol_str in tradeable_symbol_list
        },
        index=pricing_data_df.index,
    ).astype(float)
    regime_close_ser = pricing_data_df[(config_obj.regime_symbol_str, "Close")].astype(float)

    (
        monthly_decision_close_df,
        _monthly_roc_df,
        _atr_decision_df,
        _stock_trend_pass_df,
        _regime_sma_ser,
        _regime_pass_ser,
        _risk_adj_score_df,
    ) = atr_module.compute_atr_normalized_signal_tables(
        price_close_df=price_close_df,
        price_high_df=price_high_df,
        price_low_df=price_low_df,
        regime_close_ser=regime_close_ser,
        config=config_obj,
    )
    if len(monthly_decision_close_df.index) == 0:
        raise RuntimeError("ATR-normalized NDX live host produced no valid monthly decision dates.")

    signal_date_ts = pd.Timestamp(monthly_decision_close_df.index[-1]).to_pydatetime()
    # *** CRITICAL*** The monthly decision date must map to the true next tradable session on the release calendar.
    execution_date_ts = scheduler_utils.next_business_day_timestamp_ts(
        signal_date_ts,
        session_calendar_id_str=release_obj.session_calendar_id_str,
    )
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": [pd.Timestamp(signal_date_ts)]},
        index=pd.DatetimeIndex([pd.Timestamp(execution_date_ts.date())], name="execution_date_ts"),
    )

    strategy_obj = atr_module.AtrNormalizedNdxStrategy(
        name=release_obj.pod_id_str,
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=float(release_obj.params_dict.get("capital_base_float", config_obj.capital_base_float)),
        slippage=float(release_obj.params_dict.get("slippage_float", config_obj.slippage_float)),
        commission_per_share=float(
            release_obj.params_dict.get("commission_per_share_float", config_obj.commission_per_share_float)
        ),
        commission_minimum=float(
            release_obj.params_dict.get("commission_minimum_float", config_obj.commission_minimum_float)
        ),
        lookback_month_int=int(release_obj.params_dict.get("lookback_month_int", config_obj.lookback_month_int)),
        index_trend_window_int=int(
            release_obj.params_dict.get("index_trend_window_int", config_obj.index_trend_window_int)
        ),
        stock_trend_window_int=int(
            release_obj.params_dict.get("stock_trend_window_int", config_obj.stock_trend_window_int)
        ),
        max_positions_int=int(release_obj.params_dict.get("max_positions_int", config_obj.max_positions_int)),
    )
    strategy_obj.universe_df = universe_df
    _seed_strategy_state(strategy_obj, pod_state_obj)

    full_signal_df = strategy_obj.compute_signals(pricing_data_df.copy())
    close_row_ser = full_signal_df.loc[pd.Timestamp(signal_date_ts)]
    strategy_obj.previous_bar = pd.Timestamp(signal_date_ts)
    strategy_obj.current_bar = pd.Timestamp(execution_date_ts.date())
    strategy_obj.iterate(
        full_signal_df.loc[: pd.Timestamp(signal_date_ts)],
        close_row_ser,
        pd.Series(dtype=float),
    )

    return _build_frozen_order_plan(
        release_obj=release_obj,
        signal_date_ts=signal_date_ts,
        close_row_ser=close_row_ser,
        strategy_obj=strategy_obj,
        snapshot_metadata_dict={"strategy_family_str": "atr_normalized_ndx"},
    )


def build_order_plan_for_release(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState | None,
) -> FrozenOrderPlan:
    if (
        release_obj.execution_policy_str == "same_day_moc"
        and "intraday" in release_obj.data_profile_str
    ):
        raise NotImplementedError(
            "same_day_moc release plumbing exists in v1, but real intraday snapshot adapters are deferred."
        )

    if release_obj.strategy_import_str == "strategies.dv2.strategy_mr_dv2:DVO2Strategy":
        return _build_dv2_order_plan(release_obj, as_of_ts, pod_state_obj)
    if release_obj.strategy_import_str == "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash":
        return _build_taa_btal_tqqq_vix_cash_order_plan(release_obj, as_of_ts, pod_state_obj)
    if release_obj.strategy_import_str == "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy":
        return _build_atr_normalized_ndx_order_plan(release_obj, as_of_ts, pod_state_obj)

    raise ValueError(
        f"Unsupported strategy_import_str '{release_obj.strategy_import_str}' for v1 live hosting."
    )


def build_decision_plan_for_release(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState | None,
) -> DecisionPlan:
    if release_obj.strategy_import_str == "strategies.dv2.strategy_mr_dv2:DVO2Strategy":
        return _build_dv2_decision_plan(release_obj, as_of_ts, pod_state_obj)
    if release_obj.strategy_import_str == "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy":
        return _build_qpi_ibs_rsi_exit_decision_plan(release_obj, as_of_ts, pod_state_obj)
    if release_obj.strategy_import_str == "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash":
        return _build_taa_btal_tqqq_vix_cash_decision_plan(release_obj, as_of_ts, pod_state_obj)
    if release_obj.strategy_import_str == "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy":
        return _build_atr_normalized_ndx_decision_plan(release_obj, as_of_ts, pod_state_obj)
    raise NotImplementedError(
        "V2 broker-truth execution currently supports the configured decision-book families only. "
        f"Unsupported strategy_import_str '{release_obj.strategy_import_str}'."
    )
