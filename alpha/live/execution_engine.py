from __future__ import annotations

import math
from dataclasses import replace

from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR, validated_core5_target_share_dict

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
    if execution_policy_str == "next_open_market":
        return "MKT"
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


def validate_mr_capsule_execution_contract(
    release_obj: LiveRelease,
    decision_plan_obj: DecisionPlan,
    broker_snapshot_obj: BrokerSnapshot,
    live_price_snapshot_obj: LivePriceSnapshot | None = None,
) -> float | None:
    """Validate frozen Close-T sizing and holdings; cash postings may change."""
    if not release_obj.strategy_import_str.startswith("strategies.mr_capsule."):
        if decision_plan_obj.target_share_map_dict:
            raise ValueError("Explicit target shares are supported only for MR capsule releases.")
        return None
    for field_str in ("release_id_str", "user_id_str", "pod_id_str", "account_route_str"):
        if getattr(release_obj, field_str) != getattr(decision_plan_obj, field_str):
            raise ValueError(f"MR capsule decision identity mismatch: {field_str}.")
    if (
        broker_snapshot_obj.account_route_str != release_obj.account_route_str
        or (
            live_price_snapshot_obj is not None
            and live_price_snapshot_obj.account_route_str != release_obj.account_route_str
        )
    ):
        raise ValueError("MR capsule broker and quote account routes must match the release.")
    if (
        decision_plan_obj.execution_policy_str != "next_open_moo"
        or release_obj.execution_policy_str != "next_open_moo"
        or decision_plan_obj.decision_book_type_str != "incremental_entry_exit_book"
        or not decision_plan_obj.preserve_untouched_positions_bool
        or decision_plan_obj.rebalance_omitted_assets_to_zero_bool
    ):
        raise ValueError("MR capsule requires an incremental next_open_moo decision.")
    if float(release_obj.pod_budget_fraction_float) != 1.0:
        raise ValueError("MR capsule requires a dedicated full-account pod budget of 1.0.")
    metadata_dict = decision_plan_obj.snapshot_metadata_dict
    if (
        metadata_dict.get("sizing_contract_str") != "mr_capsule_close_targets_v1"
        or metadata_dict.get("strategy_import_str") != release_obj.strategy_import_str
    ):
        raise ValueError("MR capsule decision must prove its Close-T sizing contract and strategy identity.")
    parking_mode_str = str(metadata_dict.get("mr_capsule_parking_mode_str", ""))
    allowed_parking_map_dict = {"cash": set(), "bil": {"BIL"}, "spmo": {"BIL", "SPMO"}}
    if (
        parking_mode_str not in allowed_parking_map_dict
        or not release_obj.strategy_import_str.endswith(f"_{parking_mode_str}")
    ):
        raise ValueError("MR capsule parking mode must match the named strategy.")
    allowed_parking_set = allowed_parking_map_dict[parking_mode_str]
    if set(decision_plan_obj.target_share_map_dict) - allowed_parking_set:
        raise ValueError("MR capsule explicit targets contain an asset outside the parking mode.")
    if set(decision_plan_obj.entry_target_weight_map_dict) & {"BIL", "SPMO"}:
        raise ValueError("MR capsule parking orders must use explicit share targets.")
    touched_parking_set = set(decision_plan_obj.exit_asset_set) & {"BIL", "SPMO"}
    if touched_parking_set - allowed_parking_set:
        raise ValueError("MR capsule parking exits contain an asset outside the parking mode.")
    try:
        decision_nav_float = float(metadata_dict["decision_nav_float"])
        decision_cash_float = float(metadata_dict["decision_cash_float"])
    except (KeyError, TypeError, ValueError) as exception_obj:
        raise ValueError("MR capsule requires numeric decision NAV and cash.") from exception_obj
    if (
        not math.isfinite(decision_nav_float)
        or decision_nav_float <= 0.0
        or not math.isfinite(decision_cash_float)
        or not math.isfinite(float(broker_snapshot_obj.net_liq_float))
        or float(broker_snapshot_obj.net_liq_float) <= 0.0
    ):
        raise ValueError("MR capsule requires finite cash and positive finite decision/broker NAV.")
    # Cash postings do not invalidate frozen intent. Keep broker cash truthful
    # without resizing Close-T entry dollars or ETF share targets.
    if not math.isfinite(float(broker_snapshot_obj.cash_float)):
        raise ValueError("MR capsule requires finite broker cash.")
    position_map_list: list[dict[str, float]] = []
    for position_dict in (decision_plan_obj.decision_base_position_map, broker_snapshot_obj.position_amount_map):
        normalized_position_dict: dict[str, float] = {}
        for asset_str, share_float in position_dict.items():
            share_float = float(share_float)
            if not math.isfinite(share_float) or share_float < 0.0 or not share_float.is_integer():
                raise ValueError("MR capsule positions must be finite, nonnegative whole shares.")
            if share_float > 0.0:
                normalized_position_dict[str(asset_str)] = share_float
        if (set(normalized_position_dict) & {"BIL", "SPMO"}) - allowed_parking_set:
            raise ValueError("MR capsule holdings contain parking outside the named mode.")
        position_map_list.append(normalized_position_dict)
    if position_map_list[0] != position_map_list[1]:
        raise ValueError("MR capsule broker positions changed after the decision; rebuild from trusted state.")
    if broker_snapshot_obj.open_order_id_list:
        raise ValueError("MR capsule account has outstanding broker orders.")
    return decision_nav_float


def _build_incremental_entry_exit_vplan(
    release_obj: LiveRelease,
    decision_plan_obj: DecisionPlan,
    broker_snapshot_obj: BrokerSnapshot,
    live_price_snapshot_obj: LivePriceSnapshot,
) -> VPlan:
    capsule_nav_float = validate_mr_capsule_execution_contract(
        release_obj, decision_plan_obj, broker_snapshot_obj, live_price_snapshot_obj
    )
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
    entry_sizing_value_float = pod_budget_float if capsule_nav_float is None else capsule_nav_float
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
        if not math.isfinite(live_reference_price_float) or live_reference_price_float <= 0.0:
            raise ValueError(
                f"Live reference price must be finite and positive for asset '{asset_str}'."
            )

        if asset_str in decision_plan_obj.exit_asset_set:
            target_share_float = 0.0
        elif asset_str in decision_plan_obj.target_share_map_dict:
            # *** CRITICAL *** The ETF whole-share target was frozen at Close_T.
            # TargetShares_i = N_i,T; OrderDelta_i = N_i,T - BrokerShares_i,submit.
            # A pre-submit quote values that target; it must never resize it.
            target_share_float = float(decision_plan_obj.target_share_map_dict[asset_str])
        else:
            target_weight_float = float(decision_plan_obj.entry_target_weight_map_dict.get(asset_str, 0.0))
            # *** CRITICAL *** Capsule EntryDollar_i = EntryWeight_i * NAV_Close_T:
            # preserve the stock order_value amount alongside fixed parking shares.
            # Existing strategies retain EntryWeight_i * BrokerNAV_submit * BudgetFraction.
            # TargetShares_i = floor(TargetDollar_i / P_i^{live_ref})
            target_share_float = float(
                math.floor((target_weight_float * entry_sizing_value_float) / live_reference_price_float)
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
    fixed_target_share_dict = (
        validated_core5_target_share_dict(decision_plan_obj, release_obj, broker_snapshot_obj.position_amount_map)
        if release_obj.strategy_import_str == CORE5_STRATEGY_IMPORT_STR else None
    )
    touched_asset_list = get_touched_asset_list_for_decision_plan(
        decision_plan_obj,
        broker_position_map_dict=broker_snapshot_obj.position_amount_map,
    )
    if fixed_target_share_dict is not None:
        touched_asset_list = sorted(
            asset_str for asset_str, amount_float in fixed_target_share_dict.items()
            if amount_float != 0.0 or broker_snapshot_obj.position_amount_map.get(asset_str, 0.0) != 0.0
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

    pod_budget_float = (
        float(decision_plan_obj.snapshot_metadata_dict["sizing_close_nav_float"])
        if fixed_target_share_dict is not None else
        float(broker_snapshot_obj.net_liq_float) * float(release_obj.pod_budget_fraction_float)
    )
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
        if not math.isfinite(live_reference_price_float) or live_reference_price_float <= 0.0:
            raise ValueError(
                f"Live reference price must be positive for asset '{asset_str}'."
            )

        target_weight_float = float(decision_plan_obj.full_target_weight_map_dict.get(asset_str, 0.0))
        # Existing books: TargetShares_i = floor(Weight_i * PodBudget / live_ref).
        # *** CRITICAL*** CORE5 uses already frozen trunc(NAV_Close_T*w/Close_T)
        # shares; the current quote only estimates execution notional.
        target_share_float = float(
            math.floor((target_weight_float * pod_budget_float) / live_reference_price_float)
        ) if fixed_target_share_dict is None else fixed_target_share_dict[asset_str]
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
        if fixed_target_share_dict is not None and current_share_float * target_share_float < 0.0:
            # Preserve the research close-old/open-new legs and their distinct
            # request IDs/commissions. Aggregate maps above keep the final book.
            final_row_obj = vplan_row_list.pop()
            vplan_row_list.extend([
                replace(final_row_obj, target_share_float=0.0, order_delta_share_float=-current_share_float, estimated_target_notional_float=0.0),
                replace(final_row_obj, current_share_float=0.0, order_delta_share_float=target_share_float),
            ])

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
    if (
        release_obj.strategy_import_str.startswith("strategies.mr_capsule.")
        and decision_plan_obj.decision_book_type_str != "incremental_entry_exit_book"
    ):
        raise ValueError("MR capsule requires an incremental next_open_moo decision.")
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
