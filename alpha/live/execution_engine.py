from __future__ import annotations

import math

from alpha.live.models import (
    BrokerOrderRequest,
    BrokerSnapshot,
    DecisionPlan,
    LivePriceSnapshot,
    LiveRelease,
    VPlan,
    VPlanRow,
)


def _broker_order_type_from_execution_policy_str(execution_policy_str: str) -> str:
    if execution_policy_str in ("next_open_moo", "next_month_first_open"):
        return "MOO"
    if execution_policy_str == "same_day_moc":
        return "MOC"
    raise ValueError(f"Unsupported execution_policy_str '{execution_policy_str}'.")


def get_touched_asset_list_for_decision_plan(
    decision_plan_obj: DecisionPlan,
    broker_position_map_dict: dict[str, float] | None = None,
) -> list[str]:
    return decision_plan_obj.get_execution_touched_asset_list(
        broker_position_map_dict=broker_position_map_dict,
    )


def _build_live_reference_source_map_dict(
    live_price_snapshot_obj: LivePriceSnapshot,
) -> dict[str, str]:
    if len(live_price_snapshot_obj.asset_reference_source_map_dict) > 0:
        return {
            str(asset_str): str(source_str)
            for asset_str, source_str in live_price_snapshot_obj.asset_reference_source_map_dict.items()
        }
    return {
        str(asset_str): str(live_price_snapshot_obj.price_source_str)
        for asset_str in live_price_snapshot_obj.asset_reference_price_map
    }


def _build_incremental_entry_exit_vplan(
    release_obj: LiveRelease,
    decision_plan_obj: DecisionPlan,
    broker_snapshot_obj: BrokerSnapshot,
    live_price_snapshot_obj: LivePriceSnapshot,
) -> VPlan:
    touched_asset_list = get_touched_asset_list_for_decision_plan(decision_plan_obj)
    missing_asset_list = sorted(
        asset_str
        for asset_str in touched_asset_list
        if asset_str not in live_price_snapshot_obj.asset_reference_price_map
    )
    if len(missing_asset_list) > 0:
        raise ValueError(
            "Missing live reference prices for assets: "
            f"{missing_asset_list}."
        )

    pod_budget_float = float(broker_snapshot_obj.net_liq_float) * float(release_obj.pod_budget_fraction_float)
    target_share_map: dict[str, float] = {}
    order_delta_map: dict[str, float] = {}
    vplan_row_list: list[VPlanRow] = []
    live_reference_source_map_dict = _build_live_reference_source_map_dict(live_price_snapshot_obj)
    broker_order_type_str = _broker_order_type_from_execution_policy_str(decision_plan_obj.execution_policy_str)

    for asset_str in touched_asset_list:
        current_share_float = float(broker_snapshot_obj.position_amount_map.get(asset_str, 0.0))
        live_reference_price_float = float(live_price_snapshot_obj.asset_reference_price_map[asset_str])
        live_reference_source_str = str(
            live_reference_source_map_dict.get(
                asset_str,
                live_price_snapshot_obj.price_source_str,
            )
        )
        if live_reference_price_float <= 0.0:
            raise ValueError(
                f"Live reference price must be positive for asset '{asset_str}'."
            )

        if asset_str in decision_plan_obj.exit_asset_set:
            target_share_float = 0.0
        else:
            target_weight_float = float(decision_plan_obj.entry_target_weight_map_dict.get(asset_str, 0.0))
            # TargetDollar_i = EntryWeight_i * PodBudget
            # TargetShares_i = floor(TargetDollar_i / P_i^{live_ref})
            target_share_float = float(
                math.floor((target_weight_float * pod_budget_float) / live_reference_price_float)
            )

        estimated_target_notional_float = target_share_float * live_reference_price_float
        order_delta_share_float = target_share_float - current_share_float
        target_share_map[asset_str] = target_share_float
        order_delta_map[asset_str] = order_delta_share_float
        vplan_row_list.append(
            VPlanRow(
                asset_str=asset_str,
                current_share_float=current_share_float,
                target_share_float=target_share_float,
                order_delta_share_float=order_delta_share_float,
                live_reference_price_float=live_reference_price_float,
                estimated_target_notional_float=estimated_target_notional_float,
                broker_order_type_str=broker_order_type_str,
                live_reference_source_str=live_reference_source_str,
            )
        )

    return VPlan(
        release_id_str=decision_plan_obj.release_id_str,
        user_id_str=decision_plan_obj.user_id_str,
        pod_id_str=decision_plan_obj.pod_id_str,
        account_route_str=decision_plan_obj.account_route_str,
        decision_plan_id_int=int(decision_plan_obj.decision_plan_id_int or 0),
        signal_timestamp_ts=decision_plan_obj.signal_timestamp_ts,
        submission_timestamp_ts=decision_plan_obj.submission_timestamp_ts,
        target_execution_timestamp_ts=decision_plan_obj.target_execution_timestamp_ts,
        execution_policy_str=decision_plan_obj.execution_policy_str,
        broker_snapshot_timestamp_ts=broker_snapshot_obj.snapshot_timestamp_ts,
        live_reference_snapshot_timestamp_ts=live_price_snapshot_obj.snapshot_timestamp_ts,
        live_price_source_str=live_price_snapshot_obj.price_source_str,
        net_liq_float=float(broker_snapshot_obj.net_liq_float),
        available_funds_float=broker_snapshot_obj.available_funds_float,
        excess_liquidity_float=broker_snapshot_obj.excess_liquidity_float,
        pod_budget_fraction_float=float(release_obj.pod_budget_fraction_float),
        pod_budget_float=pod_budget_float,
        current_broker_position_map={
            asset_str: float(amount_float)
            for asset_str, amount_float in broker_snapshot_obj.position_amount_map.items()
        },
        live_reference_price_map={
            asset_str: float(price_float)
            for asset_str, price_float in live_price_snapshot_obj.asset_reference_price_map.items()
        },
        target_share_map=target_share_map,
        order_delta_map=order_delta_map,
        vplan_row_list=vplan_row_list,
        live_reference_source_map_dict=live_reference_source_map_dict,
        submission_key_str=f"vplan:{decision_plan_obj.decision_plan_id_int}",
    )


def _build_full_target_weight_vplan(
    release_obj: LiveRelease,
    decision_plan_obj: DecisionPlan,
    broker_snapshot_obj: BrokerSnapshot,
    live_price_snapshot_obj: LivePriceSnapshot,
) -> VPlan:
    touched_asset_list = get_touched_asset_list_for_decision_plan(
        decision_plan_obj,
        broker_position_map_dict=broker_snapshot_obj.position_amount_map,
    )
    missing_asset_list = sorted(
        asset_str
        for asset_str in touched_asset_list
        if asset_str not in live_price_snapshot_obj.asset_reference_price_map
    )
    if len(missing_asset_list) > 0:
        raise ValueError(
            "Missing live reference prices for assets: "
            f"{missing_asset_list}."
        )

    pod_budget_float = float(broker_snapshot_obj.net_liq_float) * float(release_obj.pod_budget_fraction_float)
    target_share_map: dict[str, float] = {}
    order_delta_map: dict[str, float] = {}
    vplan_row_list: list[VPlanRow] = []
    live_reference_source_map_dict = _build_live_reference_source_map_dict(live_price_snapshot_obj)
    broker_order_type_str = _broker_order_type_from_execution_policy_str(decision_plan_obj.execution_policy_str)

    for asset_str in touched_asset_list:
        current_share_float = float(broker_snapshot_obj.position_amount_map.get(asset_str, 0.0))
        live_reference_price_float = float(live_price_snapshot_obj.asset_reference_price_map[asset_str])
        live_reference_source_str = str(
            live_reference_source_map_dict.get(
                asset_str,
                live_price_snapshot_obj.price_source_str,
            )
        )
        if live_reference_price_float <= 0.0:
            raise ValueError(
                f"Live reference price must be positive for asset '{asset_str}'."
            )

        target_weight_float = float(decision_plan_obj.full_target_weight_map_dict.get(asset_str, 0.0))
        # TargetDollar_i = FullTargetWeight_i * PodBudget
        # TargetShares_i = floor(TargetDollar_i / P_i^{live_ref})
        target_share_float = float(
            math.floor((target_weight_float * pod_budget_float) / live_reference_price_float)
        )
        estimated_target_notional_float = target_share_float * live_reference_price_float
        order_delta_share_float = target_share_float - current_share_float
        target_share_map[asset_str] = target_share_float
        order_delta_map[asset_str] = order_delta_share_float
        vplan_row_list.append(
            VPlanRow(
                asset_str=asset_str,
                current_share_float=current_share_float,
                target_share_float=target_share_float,
                order_delta_share_float=order_delta_share_float,
                live_reference_price_float=live_reference_price_float,
                estimated_target_notional_float=estimated_target_notional_float,
                broker_order_type_str=broker_order_type_str,
                live_reference_source_str=live_reference_source_str,
            )
        )

    return VPlan(
        release_id_str=decision_plan_obj.release_id_str,
        user_id_str=decision_plan_obj.user_id_str,
        pod_id_str=decision_plan_obj.pod_id_str,
        account_route_str=decision_plan_obj.account_route_str,
        decision_plan_id_int=int(decision_plan_obj.decision_plan_id_int or 0),
        signal_timestamp_ts=decision_plan_obj.signal_timestamp_ts,
        submission_timestamp_ts=decision_plan_obj.submission_timestamp_ts,
        target_execution_timestamp_ts=decision_plan_obj.target_execution_timestamp_ts,
        execution_policy_str=decision_plan_obj.execution_policy_str,
        broker_snapshot_timestamp_ts=broker_snapshot_obj.snapshot_timestamp_ts,
        live_reference_snapshot_timestamp_ts=live_price_snapshot_obj.snapshot_timestamp_ts,
        live_price_source_str=live_price_snapshot_obj.price_source_str,
        net_liq_float=float(broker_snapshot_obj.net_liq_float),
        available_funds_float=broker_snapshot_obj.available_funds_float,
        excess_liquidity_float=broker_snapshot_obj.excess_liquidity_float,
        pod_budget_fraction_float=float(release_obj.pod_budget_fraction_float),
        pod_budget_float=pod_budget_float,
        current_broker_position_map={
            asset_str: float(amount_float)
            for asset_str, amount_float in broker_snapshot_obj.position_amount_map.items()
        },
        live_reference_price_map={
            asset_str: float(price_float)
            for asset_str, price_float in live_price_snapshot_obj.asset_reference_price_map.items()
        },
        target_share_map=target_share_map,
        order_delta_map=order_delta_map,
        vplan_row_list=vplan_row_list,
        live_reference_source_map_dict=live_reference_source_map_dict,
        submission_key_str=f"vplan:{decision_plan_obj.decision_plan_id_int}",
    )


def build_vplan(
    release_obj: LiveRelease,
    decision_plan_obj: DecisionPlan,
    broker_snapshot_obj: BrokerSnapshot,
    live_price_snapshot_obj: LivePriceSnapshot,
) -> VPlan:
    if decision_plan_obj.decision_book_type_str == "incremental_entry_exit_book":
        return _build_incremental_entry_exit_vplan(
            release_obj=release_obj,
            decision_plan_obj=decision_plan_obj,
            broker_snapshot_obj=broker_snapshot_obj,
            live_price_snapshot_obj=live_price_snapshot_obj,
        )
    if decision_plan_obj.decision_book_type_str == "full_target_weight_book":
        return _build_full_target_weight_vplan(
            release_obj=release_obj,
            decision_plan_obj=decision_plan_obj,
            broker_snapshot_obj=broker_snapshot_obj,
            live_price_snapshot_obj=live_price_snapshot_obj,
        )
    raise ValueError(
        f"Unsupported decision_book_type_str '{decision_plan_obj.decision_book_type_str}'."
    )


def build_broker_order_request_list_from_vplan(vplan_obj: VPlan) -> list[BrokerOrderRequest]:
    broker_order_request_list: list[BrokerOrderRequest] = []
    submit_batch_key_str = str(
        vplan_obj.submission_key_str or f"vplan:{vplan_obj.decision_plan_id_int}"
    )
    for request_idx_int, vplan_row_obj in enumerate(vplan_obj.vplan_row_list, start=1):
        if abs(vplan_row_obj.order_delta_share_float) <= 1e-9:
            continue
        order_request_key_str = (
            f"{submit_batch_key_str}:{vplan_row_obj.asset_str}:{request_idx_int}"
        )
        broker_order_request_list.append(
            BrokerOrderRequest(
                decision_plan_id_int=int(vplan_obj.decision_plan_id_int),
                vplan_id_int=int(vplan_obj.vplan_id_int or 0),
                release_id_str=vplan_obj.release_id_str,
                pod_id_str=vplan_obj.pod_id_str,
                account_route_str=vplan_obj.account_route_str,
                submission_key_str=submit_batch_key_str,
                order_request_key_str=order_request_key_str,
                asset_str=vplan_row_obj.asset_str,
                broker_order_type_str=vplan_row_obj.broker_order_type_str,
                order_class_str="MarketOrder",
                unit_str="shares",
                amount_float=float(vplan_row_obj.order_delta_share_float),
                target_bool=False,
                trade_id_int=None,
                sizing_reference_price_float=float(vplan_row_obj.live_reference_price_float),
                portfolio_value_float=float(vplan_obj.pod_budget_float),
            )
        )
    return broker_order_request_list
