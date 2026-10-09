"""CORE5/capsule daily settlement from fresh holdings and all-client open orders.

Fills are reporting records. They never determine whether this cycle can close.
"""
from dataclasses import asdict, dataclass, field
import json
import math

from alpha.live import scheduler_utils
from alpha.live.core5_adapter import CORE5_CONTRACT_STR, CORE5_STRATEGY_IMPORT_STR
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerOrderRequest, BrokerSnapshot
from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE


DAILY_TERMINAL_STATUS_SET = {"completed", "completed_with_exceptions"}
QUANTITY_TOLERANCE_FLOAT = 1e-9


@dataclass(frozen=True)
class DailyReconcileResult:
    status_str: str
    broker_snapshot_obj: BrokerSnapshot
    exception_list: list[dict] = field(default_factory=list)
    completion_request_list: list[BrokerOrderRequest] = field(default_factory=list)
    reporting_result_list: list = field(default_factory=list)


def is_daily_reconcile_release_bool(release_obj):
    return release_obj.strategy_import_str in (*MR_CAPSULE_STRATEGY_IMPORT_TUPLE, CORE5_STRATEGY_IMPORT_STR)


def ensure_daily_reconcile_schema(connection_obj):
    from alpha.live.daily_notifications import ensure_daily_alert_schema

    ensure_daily_alert_schema(connection_obj)
    connection_obj.execute("""CREATE TABLE IF NOT EXISTS daily_completion_request (
        decision_plan_id_int INTEGER NOT NULL, asset_str TEXT NOT NULL,
        vplan_id_int INTEGER NOT NULL, pod_id_str TEXT NOT NULL, account_route_str TEXT NOT NULL,
        order_request_key_str TEXT NOT NULL, request_json_str TEXT NOT NULL,
        claimed_timestamp_str TEXT NOT NULL, send_error_str TEXT,
        PRIMARY KEY(decision_plan_id_int, asset_str))""")
    if connection_obj.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='mr_capsule_execution_request'").fetchone():
        # An upgrade/restart must not allow another sale for an already claimed
        # asset. Preserve the historical table and its original request verbatim.
        connection_obj.execute("""INSERT OR IGNORE INTO daily_completion_request
            (decision_plan_id_int,asset_str,vplan_id_int,pod_id_str,account_route_str,
             order_request_key_str,request_json_str,claimed_timestamp_str)
            SELECT v.decision_plan_id_int,r.asset_str,r.vplan_id_int,v.pod_id_str,v.account_route_str,
                r.order_request_key_str,r.request_json_str,r.claimed_timestamp_str
            FROM mr_capsule_execution_request r JOIN vplan v ON v.vplan_id_int=r.vplan_id_int
            WHERE r.request_kind_str='recovery'""")


def _validate_identity(release_obj, decision_plan_obj, vplan_obj):
    if not is_daily_reconcile_release_bool(release_obj):
        raise ValueError("Daily reconciliation is limited to CORE5 and MR capsule.")
    for cycle_obj in (decision_plan_obj, vplan_obj):
        if cycle_obj is not None and any(getattr(cycle_obj, field_str) != getattr(release_obj, field_str)
                for field_str in ("pod_id_str", "account_route_str", "release_id_str", "user_id_str")):
            raise ValueError("Daily reconciliation account and cycle identity differ.")
    if decision_plan_obj.decision_plan_id_int is None or (vplan_obj is not None
            and vplan_obj.decision_plan_id_int != decision_plan_obj.decision_plan_id_int):
        raise ValueError("Daily reconciliation requires the saved matching decision.")


def _validate_snapshot(daily_snapshot_obj, release_obj, as_of_ts):
    snapshot_obj = daily_snapshot_obj.broker_snapshot_obj
    timestamp_list = [as_of_ts, daily_snapshot_obj.refresh_started_timestamp_ts,
        daily_snapshot_obj.refreshed_timestamp_ts, snapshot_obj.snapshot_timestamp_ts]
    if (daily_snapshot_obj.complete_bool is not True
            or any(timestamp_ts.tzinfo is None or timestamp_ts.utcoffset() is None for timestamp_ts in timestamp_list)
            or not as_of_ts <= daily_snapshot_obj.refresh_started_timestamp_ts <= snapshot_obj.snapshot_timestamp_ts
                <= daily_snapshot_obj.refreshed_timestamp_ts
            or snapshot_obj.account_route_str != release_obj.account_route_str
            or not isinstance(snapshot_obj.position_amount_map, dict)
            or not all(math.isfinite(float(value_float)) for value_float in (
                snapshot_obj.cash_float, snapshot_obj.net_liq_float, snapshot_obj.total_value_float,
                *snapshot_obj.position_amount_map.values()))
            or snapshot_obj.net_liq_float <= 0):
        raise ValueError("Daily reconciliation requires complete, fresh broker holdings and open orders.")
    if not isinstance(daily_snapshot_obj.open_order_row_list, list) or any(
            not isinstance(row_dict, dict) or row_dict.get("account_route_str") != release_obj.account_route_str
            or not isinstance(row_dict.get("asset_str"), str) or not row_dict["asset_str"].strip()
            for row_dict in daily_snapshot_obj.open_order_row_list):
        raise ValueError("Daily open orders must identify their account and symbol.")
    return snapshot_obj


def _refresh(broker_adapter_obj, release_obj, as_of_ts):
    daily_snapshot_obj = broker_adapter_obj.get_daily_execution_snapshot(release_obj.account_route_str)
    _validate_snapshot(daily_snapshot_obj, release_obj, as_of_ts)
    return daily_snapshot_obj


def load_daily_owned_order_ref_set(state_store_obj, release_obj, decision_plan_obj, open_order_row_list):
    """Attribute exact refs through this cycle; an older close cannot cancel a newer day."""
    open_ref_set = {row_dict.get("order_ref_str") for row_dict in open_order_row_list if row_dict.get("order_ref_str")}
    owned_ref_set = set()
    target_timestamp_str = decision_plan_obj.target_execution_timestamp_ts.isoformat()
    with state_store_obj._connect() as connection_obj:
        for table_str in ("vplan_broker_order", "vplan_broker_ack", "daily_completion_request"):
            row_list = connection_obj.execute(f"""SELECT r.order_request_key_str FROM {table_str} r
                JOIN vplan v ON v.vplan_id_int=r.vplan_id_int
                WHERE v.pod_id_str=? AND v.account_route_str=? AND r.account_route_str=?
                    AND julianday(v.target_execution_timestamp_str)<=julianday(?)""",
                (release_obj.pod_id_str, release_obj.account_route_str, release_obj.account_route_str, target_timestamp_str)).fetchall()
            owned_ref_set.update(row_obj["order_request_key_str"] for row_obj in row_list if row_obj["order_request_key_str"])
        if connection_obj.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='mr_capsule_execution_request'").fetchone():
            row_list = connection_obj.execute("""SELECT r.order_request_key_str FROM mr_capsule_execution_request r
                JOIN vplan v ON v.vplan_id_int=r.vplan_id_int WHERE v.pod_id_str=? AND v.account_route_str=?
                    AND julianday(v.target_execution_timestamp_str)<=julianday(?)""",
                (release_obj.pod_id_str, release_obj.account_route_str, target_timestamp_str)).fetchall()
            owned_ref_set.update(row_obj["order_request_key_str"] for row_obj in row_list)
        plan_row_list = connection_obj.execute("""SELECT vplan_id_int,decision_plan_id_int,submission_key_str FROM vplan
            WHERE pod_id_str=? AND account_route_str=? AND julianday(target_execution_timestamp_str)<=julianday(?)""",
            (release_obj.pod_id_str, release_obj.account_route_str, target_timestamp_str)).fetchall()
    for row_obj in plan_row_list:
        prefix_str = str(row_obj["submission_key_str"] or f"vplan:{row_obj['decision_plan_id_int']}") + ":"
        if any(ref_str.startswith(prefix_str) for ref_str in open_ref_set):
            plan_obj = state_store_obj.get_vplan_by_id(row_obj["vplan_id_int"])
            owned_ref_set.update(request_obj.order_request_key_str
                for request_obj in build_broker_order_request_list_from_vplan(plan_obj))
    return owned_ref_set


def claim_daily_completion_request(state_store_obj, release_obj, decision_plan_obj, vplan_obj, request_obj, as_of_ts):
    """One committed attempt per decision/asset, before any broker send."""
    _validate_identity(release_obj, decision_plan_obj, vplan_obj)
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("BEGIN IMMEDIATE")
        row_obj = connection_obj.execute("""SELECT v.status_str,v.account_route_str,v.decision_plan_id_int,
            d.status_str AS decision_status_str FROM vplan v JOIN decision_plan d
            ON d.decision_plan_id_int=v.decision_plan_id_int WHERE v.vplan_id_int=?""", (vplan_obj.vplan_id_int,)).fetchone()
        if (row_obj is None or row_obj["status_str"] not in {"submitted", "submitting"}
                or row_obj["decision_status_str"] not in {"vplan_ready", "submitted"}
                or row_obj["decision_plan_id_int"] != decision_plan_obj.decision_plan_id_int):
            return False
        if (request_obj.account_route_str != row_obj["account_route_str"] or request_obj.amount_float >= 0
                or request_obj.broker_order_type_str != "MKT"):
            raise ValueError("Daily completion is one market sell in the matching account.")
        cursor_obj = connection_obj.execute("""INSERT OR IGNORE INTO daily_completion_request
            (decision_plan_id_int,asset_str,vplan_id_int,pod_id_str,account_route_str,order_request_key_str,
             request_json_str,claimed_timestamp_str) VALUES (?,?,?,?,?,?,?,?)""",
            (decision_plan_obj.decision_plan_id_int, request_obj.asset_str, vplan_obj.vplan_id_int,
                release_obj.pod_id_str, release_obj.account_route_str, request_obj.order_request_key_str,
                json.dumps(asdict(request_obj), sort_keys=True, allow_nan=False), as_of_ts.isoformat()))
        return cursor_obj.rowcount == 1


def _exception_list(decision_plan_obj, vplan_obj, snapshot_obj, *, vplan_sent_bool, funding_buys_dropped_bool):
    exception_list = []
    if vplan_obj is None or not vplan_sent_bool:
        # Unsized weight intentions are reported as such. Do not invent a quote,
        # resize the missed decision, or claim an exact unsent share quantity.
        weight_dict = (decision_plan_obj.full_target_weight_map_dict if decision_plan_obj.decision_book_type_str == "full_target_weight_book"
            else decision_plan_obj.entry_target_weight_map_dict)
        share_dict = dict(decision_plan_obj.target_share_map_dict)
        if decision_plan_obj.snapshot_metadata_dict.get("sizing_contract_str") == CORE5_CONTRACT_STR:
            share_dict.update(decision_plan_obj.snapshot_metadata_dict["fixed_target_share_map_dict"])
        share_dict.update({asset_str: 0.0 for asset_str in decision_plan_obj.exit_asset_set})
        if decision_plan_obj.rebalance_omitted_assets_to_zero_bool:
            share_dict.update({asset_str: 0.0 for asset_str in decision_plan_obj.decision_base_position_map if asset_str not in weight_dict})
        for asset_str in sorted(set(weight_dict) | set(share_dict)):
            actual_float = float(snapshot_obj.position_amount_map.get(asset_str, 0.0))
            expected_float = share_dict.get(asset_str)
            residual_float = None if expected_float is None else float(expected_float) - actual_float
            if residual_float is not None and abs(residual_float) <= QUANTITY_TOLERANCE_FLOAT:
                continue
            exception_list.append({"asset_str": asset_str,
                "quantity_float": None if residual_float is None else abs(residual_float),
                "side_str": ("BUY" if residual_float > 0 else "SELL") if residual_float is not None
                    else ("BUY" if weight_dict[asset_str] > 0 else "SELL"),
                "reason_str": "decision_intent_not_dispatched", "expected_share_float": expected_float,
                "actual_share_float": actual_float, "target_weight_float": weight_dict.get(asset_str),
                "intent_str": "target_weight" if expected_float is None else "target_shares"})
        intended_asset_set = set(weight_dict) | set(share_dict)
        for asset_str in sorted((set(snapshot_obj.position_amount_map) | set(decision_plan_obj.decision_base_position_map)) - intended_asset_set):
            expected_float = (0.0 if decision_plan_obj.rebalance_omitted_assets_to_zero_bool else
                float(decision_plan_obj.decision_base_position_map.get(asset_str, 0.0)))
            actual_float = float(snapshot_obj.position_amount_map.get(asset_str, 0.0))
            residual_float = expected_float - actual_float
            if abs(residual_float) > QUANTITY_TOLERANCE_FLOAT:
                exception_list.append({"asset_str": asset_str, "quantity_float": abs(residual_float),
                    "side_str": "BUY" if residual_float > 0 else "SELL", "reason_str": "actual_holding_differs_from_decision",
                    "expected_share_float": expected_float, "actual_share_float": actual_float,
                    "residual_amount_float": residual_float})
        return exception_list
    expected_dict = {**vplan_obj.current_broker_position_map, **vplan_obj.target_share_map}
    for asset_str in sorted(set(expected_dict) | set(snapshot_obj.position_amount_map)):
        expected_float = float(expected_dict.get(asset_str, 0.0))
        actual_float = float(snapshot_obj.position_amount_map.get(asset_str, 0.0))
        residual_float = expected_float - actual_float
        if abs(residual_float) <= QUANTITY_TOLERANCE_FLOAT:
            continue
        reason_str = "actual_holding_differs_from_vplan"
        if funding_buys_dropped_bool and (residual_float > 0 or asset_str == "BIL"):
            reason_str = "funding_buys_dropped" if asset_str != "BIL" else "bil_sale_withheld_after_funding_buys_dropped"
        exception_list.append({"asset_str": asset_str, "quantity_float": abs(residual_float),
            "side_str": "BUY" if residual_float > 0 else "SELL", "reason_str": reason_str,
            "expected_share_float": expected_float, "actual_share_float": actual_float,
            "residual_amount_float": residual_float})
    return exception_list


def reconcile_daily_cycle(state_store_obj, broker_adapter_obj, release_obj, decision_plan_obj, as_of_ts,
        *, vplan_obj=None, vplan_sent_bool=True, funding_buys_dropped_bool=False):
    """Poll within the session; after its close return actual holdings for commit.

    The caller commits final holdings, strategy memory, status and one exception
    alert atomically. Broker failures propagate and therefore cannot close a day.
    """
    _validate_identity(release_obj, decision_plan_obj, vplan_obj)
    target_ts = decision_plan_obj.target_execution_timestamp_ts
    session_date_obj = scheduler_utils.to_market_timestamp_ts(target_ts, release_obj.session_calendar_id_str).date()
    close_ts = scheduler_utils.get_session_close_timestamp_ts(session_date_obj, release_obj.session_calendar_id_str)
    daily_snapshot_obj = _refresh(broker_adapter_obj, release_obj, as_of_ts)
    completion_request_list, reporting_result_list = [], []
    if as_of_ts >= close_ts:
        owned_ref_set = load_daily_owned_order_ref_set(state_store_obj, release_obj, decision_plan_obj,
            daily_snapshot_obj.open_order_row_list)
        if any(row_dict.get("order_ref_str") in owned_ref_set for row_dict in daily_snapshot_obj.open_order_row_list):
            owned_order_row_list = [row_dict for row_dict in daily_snapshot_obj.open_order_row_list
                if row_dict.get("order_ref_str") in owned_ref_set]
            try:
                daily_snapshot_obj = broker_adapter_obj.cancel_daily_owned_orders(release_obj.account_route_str,
                    owned_ref_set, session_close_timestamp_ts=close_ts)
                _validate_snapshot(daily_snapshot_obj, release_obj, as_of_ts)
                if any(row_dict.get("order_ref_str") in owned_ref_set for row_dict in daily_snapshot_obj.open_order_row_list):
                    owned_order_row_list = [row_dict for row_dict in daily_snapshot_obj.open_order_row_list
                        if row_dict.get("order_ref_str") in owned_ref_set]
                    raise RuntimeError("Daily close is waiting for confirmation that our open orders are cancelled.")
            except Exception as error_obj:
                from alpha.live.daily_notifications import enqueue_daily_cycle_overdue

                enqueue_daily_cycle_overdue(state_store_obj, release_obj, decision_plan_obj, as_of_ts,
                    error_str=str(error_obj), owned_order_row_list=owned_order_row_list)
                raise
        snapshot_obj = daily_snapshot_obj.broker_snapshot_obj
        exception_list = _exception_list(decision_plan_obj, vplan_obj, snapshot_obj,
            vplan_sent_bool=vplan_sent_bool, funding_buys_dropped_bool=funding_buys_dropped_bool)
        return DailyReconcileResult("completed_with_exceptions" if exception_list else "completed", snapshot_obj, exception_list)
    if vplan_obj is not None and vplan_sent_bool and target_ts < as_of_ts:
        for asset_str, target_float in sorted(vplan_obj.target_share_map.items()):
            if (asset_str == "BIL" and (funding_buys_dropped_bool or target_float < 0)) or (
                    asset_str != "BIL" and abs(target_float) > QUANTITY_TOLERANCE_FLOAT):
                continue
            # Refresh before EACH asset: an earlier send/manual trade can change
            # holdings. Every open order for this symbol blocks its completion.
            daily_snapshot_obj = _refresh(broker_adapter_obj, release_obj, as_of_ts)
            snapshot_obj = daily_snapshot_obj.broker_snapshot_obj
            if not target_ts < snapshot_obj.snapshot_timestamp_ts < close_ts:
                break
            if any(row_dict["asset_str"] == asset_str for row_dict in daily_snapshot_obj.open_order_row_list):
                continue
            # Q_sell = max(actual broker shares - frozen target shares, 0).
            remainder_float = max(float(snapshot_obj.position_amount_map.get(asset_str, 0.0)) - float(target_float), 0.0)
            if remainder_float <= QUANTITY_TOLERANCE_FLOAT:
                continue
            submission_key_str = str(vplan_obj.submission_key_str or f"vplan:{vplan_obj.decision_plan_id_int}")
            request_obj = BrokerOrderRequest(release_id_str=release_obj.release_id_str, pod_id_str=release_obj.pod_id_str,
                account_route_str=release_obj.account_route_str, submission_key_str=submission_key_str,
                order_request_key_str=f"{submission_key_str}:daily-completion:{asset_str}", asset_str=asset_str,
                broker_order_type_str="MKT", order_class_str="MarketOrder", unit_str="shares",
                amount_float=-remainder_float, target_bool=False, trade_id_int=None,
                sizing_reference_price_float=float(vplan_obj.live_reference_price_map.get(asset_str, 0.0)),
                portfolio_value_float=float(vplan_obj.pod_budget_float), decision_plan_id_int=decision_plan_obj.decision_plan_id_int,
                vplan_id_int=vplan_obj.vplan_id_int, execution_deadline_timestamp_str=close_ts.isoformat())
            if not claim_daily_completion_request(state_store_obj, release_obj, decision_plan_obj, vplan_obj,
                    request_obj, snapshot_obj.snapshot_timestamp_ts):
                continue
            completion_request_list.append(request_obj)
            try:
                reporting_result_list.append(broker_adapter_obj.submit_order_request_list(
                    account_route_str=release_obj.account_route_str, broker_order_request_list=[request_obj],
                    submitted_timestamp_ts=snapshot_obj.snapshot_timestamp_ts))
            except Exception as error_obj:
                with state_store_obj._connect() as connection_obj:
                    connection_obj.execute("UPDATE daily_completion_request SET send_error_str=? WHERE decision_plan_id_int=? AND asset_str=?",
                        (str(error_obj) or type(error_obj).__name__, decision_plan_obj.decision_plan_id_int, asset_str))
                raise
        if completion_request_list:
            daily_snapshot_obj = _refresh(broker_adapter_obj, release_obj, as_of_ts)
    return DailyReconcileResult("pending", daily_snapshot_obj.broker_snapshot_obj,
        _exception_list(decision_plan_obj, vplan_obj, daily_snapshot_obj.broker_snapshot_obj,
            vplan_sent_bool=vplan_sent_bool, funding_buys_dropped_bool=funding_buys_dropped_bool),
        completion_request_list, reporting_result_list)
