"""V4 order panel: "Complete a missing trade" and "New order" (manual ticket, MOC/MKT/LMT).

Two steps, both JSON POSTs with the action token and a same-origin check:

* check: one fresh read-only broker query (positions + all open orders). Applies the
  stops and returns the numbers for the panel plus a one-use nonce (120 s).
* send: consumes the nonce, re-checks every rule and the broker once more, then sends
  one manual ticket through alpha.live.manual_order (the same path as Dashboard V3).

Production sends orders by default (``--disable-orders`` turns the panel off). Every other V4 tool stays copy-only.
Rules: docs/plans/MOC_COMPLETION_HANDOFF.md §5-§8b.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import getpass
import math
import os
import secrets
from urllib.parse import urlsplit

from flask import jsonify, request

from alpha.live.dashboard import DashboardActionInFlightError, DashboardPodJobGate
from alpha.live.dashboard_authorization import (
    operator_execution_context,
    read_confirmation_context_dict,
    validate_current_confirmation,
)
from alpha.live.dashboard_v3.actions import ConfirmationStore, validate_action_request
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.manual_order import (
    MANUAL_ORDER_CONFIRMATION_TEXT_STR,
    _normalize_asset_str,
    build_manual_order_ticket_obj,
    manual_order_session_window_dict,
    submit_manual_order_ticket_dict,
)
from alpha.live.order_clerk import IBKRGatewayBrokerAdapter


ACTION_NAME_STR = "manual_order"
ORDER_TYPE_TUPLE = ("MOC", "MKT", "LMT")
REASON_MAX_LEN_INT = 200


class OrderStop(ValueError):
    """A rule stopped the ticket. The message is one line for the panel."""


def default_broker_factory(release_obj):
    return IBKRGatewayBrokerAdapter(
        host_str=release_obj.broker_host_str,
        port_int=int(release_obj.broker_port_int),
        client_id_int=int(release_obj.broker_client_id_int),
        timeout_seconds_float=float(release_obj.broker_timeout_seconds_float),
    )


def default_operator_id_str():
    return os.environ.get("ALPHA_OPERATOR_ID") or getpass.getuser() or "dashboard_v4"


def _et(timestamp_ts, format_str="%H:%M"):
    return timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ).strftime(format_str)


def _share_str(value_float):
    value_float = float(value_float)
    return str(int(value_float)) if value_float.is_integer() else f"{value_float:g}"


def order_line_str(side_str, quantity_int, asset_str, type_str, limit_price_float=None):
    when_str = {"MOC": "at close", "MKT": "now"}.get(type_str) or f"at {limit_price_float:g}"
    return f"{side_str} {quantity_int} {asset_str} {when_str}"


class OrderTicketService:
    def __init__(self, *, target_fn, missing_fn, log_path_fn, target_loader_fn, enabled_bool=False,
                 broker_factory_fn=None, sender_fn=None, now_fn=None, operator_id_str=None,
                 confirmation_fn=None):
        self.target_fn = target_fn                  # pod_id -> verified LIVE target (raises ValueError)
        self.missing_fn = missing_fn                # target -> load_missing_trades_dict(...) result
        self.log_path_fn = log_path_fn              # () -> event log path
        self.target_loader_fn = target_loader_fn    # pod_id -> current target (authorization re-read)
        self.enabled_bool = bool(enabled_bool)
        self.broker_factory_fn = broker_factory_fn or default_broker_factory
        self.sender_fn = sender_fn or submit_manual_order_ticket_dict
        self.now_fn = now_fn or (lambda: datetime.now(UTC))
        self.operator_id_str = operator_id_str or default_operator_id_str()
        self.confirmation_fn = confirmation_fn or read_confirmation_context_dict
        self.confirmation_store_obj = ConfirmationStore(ttl_seconds_float=120)
        self.pod_gate_obj = DashboardPodJobGate()
        self._action_token_str = secrets.token_urlsafe(32) if self.enabled_bool else ""

    @property
    def action_token_str(self):
        return self._action_token_str

    # ------------------------------------------------------------------ helpers
    def _window_dict(self, release_obj, now_ts):
        window_dict = manual_order_session_window_dict(release_obj.session_calendar_id_str, now_ts)
        if not window_dict["session_bool"] or now_ts >= window_dict["close_ts"]:
            raise OrderStop("Market closed · next open " + _et(window_dict["next_open_ts"], "%a %H:%M"))
        return window_dict

    def _fresh_tuple(self, release_obj, asset_str):
        try:
            adapter_obj = self.broker_factory_fn(release_obj)
            snapshot_obj = adapter_obj.get_daily_execution_snapshot(release_obj.account_route_str)
            held_float = float(snapshot_obj.broker_snapshot_obj.position_amount_map.get(asset_str, 0.0))
            open_list = [row_dict for row_dict in snapshot_obj.open_order_row_list
                if row_dict.get("asset_str") == asset_str and float(row_dict.get("remaining_amount_float") or 0) > 0]
        except Exception as exception_obj:
            raise OrderStop("Broker check failed · nothing sent") from exception_obj
        if not math.isfinite(held_float):
            raise OrderStop("Broker check failed · nothing sent")
        if open_list:
            row_dict = open_list[0]
            raise OrderStop(f"Blocked · {asset_str} order already open ({row_dict.get('side_str')} "
                f"{_share_str(row_dict.get('remaining_amount_float'))}) · wait or cancel in TWS")
        return adapter_obj, held_float

    def _missing_row(self, target_obj, asset_str):
        missing_dict = self.missing_fn(target_obj)
        if missing_dict.get("status_str") != "ok":
            raise OrderStop("No pre-fill · use New order")
        row_dict = next((item_dict for item_dict in missing_dict["row_list"] if item_dict["asset_str"] == asset_str), None)
        if row_dict is None:
            raise OrderStop("Nothing to complete · broker holds the plan")
        if row_dict["state_str"] == "sent":
            raise OrderStop("Already sent · " + row_dict["text_str"].removeprefix("Sent · "))
        if row_dict["state_str"] != "missing":
            raise OrderStop("No pre-fill · use New order")
        return missing_dict, row_dict

    # ------------------------------------------------------------------ check
    def check_dict(self, pod_id_str, body_dict):
        kind_str = body_dict.get("kind_str")
        if kind_str not in {"complete", "new"}:
            raise OrderStop("Unknown ticket")
        target_obj = self.target_fn(pod_id_str)
        release_obj = target_obj.release_obj
        now_ts = self.now_fn()
        try:
            asset_str = _normalize_asset_str(body_dict.get("asset_str"))
        except ValueError as exception_obj:
            raise OrderStop("Symbol must be a simple US stock or ETF symbol") from exception_obj
        window_dict = self._window_dict(release_obj, now_ts)
        plan_float, vplan_id_int = None, None
        if kind_str == "complete":
            missing_dict, row_dict = self._missing_row(target_obj, asset_str)
            plan_float, vplan_id_int = row_dict["target_share_float"], missing_dict["vplan_id_int"]
        else:
            plan_obj = (self.missing_fn(target_obj).get("target_share_map") or {}).get(asset_str)
            plan_float = None if plan_obj is None else float(plan_obj)
        _, held_float = self._fresh_tuple(release_obj, asset_str)
        checked_ts = self.now_fn()
        result_dict = {"kind_str": kind_str, "asset_str": asset_str, "held_str": _share_str(held_float),
            "plan_str": "—" if plan_float is None else _share_str(plan_float), "open_orders_str": "0",
            "checked_str": _et(checked_ts, "%H:%M:%S")}
        context_dict = {"pod_id_str": pod_id_str, "action_name_str": ACTION_NAME_STR, "kind_str": kind_str,
            "asset_str": asset_str, "held_float": held_float, "plan_float": plan_float, "vplan_id_int": vplan_id_int,
            "confirmation_dict": self.confirmation_fn(target_obj)}
        if kind_str == "complete":
            gap_float = float(plan_float) - held_float
            quantity_int = int(math.floor(abs(gap_float) + 1e-9))
            if quantity_int < 1:
                raise OrderStop(f"Nothing to complete · broker holds {_share_str(held_float)} = plan")
            side_str = "BUY" if gap_float > 0 else "SELL"
            context_dict.update(side_str=side_str, max_quantity_int=quantity_int)
            result_dict.update(side_str=side_str, missing_str=f"{side_str} {quantity_int}", max_quantity_int=quantity_int,
                changed_bool=quantity_int != row_dict["quantity_int"] or side_str != row_dict["side_str"],
                page_str=f"{row_dict['side_str']} {row_dict['quantity_int']}", vplan_id_int=vplan_id_int)
        else:
            context_dict.update(side_str=None, max_quantity_int=None)
            result_dict.update(max_sell_int=int(math.floor(max(held_float, 0.0) + 1e-9)))
        moc_bool = checked_ts < window_dict["moc_cutoff_ts"]
        result_dict.update(moc_bool=moc_bool, default_type_str="MOC" if moc_bool else "MKT",
            cutoff_str=_et(window_dict["moc_cutoff_ts"]), close_str=_et(window_dict["close_ts"]),
            minutes_left_int=max(0, int((window_dict["moc_cutoff_ts"] - checked_ts).total_seconds() // 60)))
        result_dict["confirmation_nonce_str"] = self.confirmation_store_obj.issue(context_dict)
        result_dict["expires_in_seconds_int"] = 120
        return result_dict

    # ------------------------------------------------------------------ send
    def send_dict(self, pod_id_str, body_dict):
        try:
            expected_dict = self.confirmation_store_obj.consume(body_dict.get("confirmation_nonce_str"), pod_id_str, ACTION_NAME_STR)
        except ValueError as exception_obj:
            raise OrderStop("Expired · check again") from exception_obj
        target_obj = self.target_fn(pod_id_str)
        release_obj = target_obj.release_obj
        asset_str = expected_dict["asset_str"]
        type_str = str(body_dict.get("broker_order_type_str") or "").upper()
        if type_str not in ORDER_TYPE_TUPLE:
            raise OrderStop("Choose At close, Now or Limit")
        quantity_obj = body_dict.get("quantity_int")
        if isinstance(quantity_obj, bool) or not isinstance(quantity_obj, int) or quantity_obj < 1:
            raise OrderStop("Shares must be a whole number above 0")
        if expected_dict["kind_str"] == "complete":
            side_str = expected_dict["side_str"]
            if quantity_obj > expected_dict["max_quantity_int"]:
                raise OrderStop(f"Max {expected_dict['max_quantity_int']} = what is missing")
            reason_str = f"Completion · VPlan #{expected_dict['vplan_id_int']} · {asset_str}"
        else:
            side_str = str(body_dict.get("side_str") or "").upper()
            if side_str not in {"BUY", "SELL"}:
                raise OrderStop("Choose BUY or SELL")
            reason_str = " ".join(str(body_dict.get("reason_str") or "").split())[:REASON_MAX_LEN_INT]
            if not reason_str:
                raise OrderStop("Reason is required")
        if side_str == "SELL" and quantity_obj > expected_dict["held_float"] + 1e-9:
            raise OrderStop(f"Blocked · {asset_str} holds {_share_str(expected_dict['held_float'])} · this pod does not go short")
        limit_float = None
        if type_str == "LMT":
            limit_obj = body_dict.get("limit_price_float")
            if isinstance(limit_obj, bool) or not isinstance(limit_obj, (int, float)) or not math.isfinite(limit_obj) or limit_obj <= 0:
                raise OrderStop("Limit price must be above 0")
            limit_float = float(limit_obj)
        now_ts = self.now_fn()
        window_dict = self._window_dict(release_obj, now_ts)
        if type_str == "MOC" and now_ts >= window_dict["moc_cutoff_ts"]:
            raise OrderStop(f"MOC closed at {_et(window_dict['moc_cutoff_ts'])} · choose Now")
        adapter_obj, held_float = self._fresh_tuple(release_obj, asset_str)
        if abs(held_float - expected_dict["held_float"]) > 1e-9:
            raise OrderStop("Broker changed since the check · check again")
        ticket_dict = {"asset_str": asset_str, "side_str": side_str, "broker_order_type_str": type_str,
            "quantity_int": quantity_obj, "limit_price_float": limit_float, "time_in_force_str": "DAY",
            "operator_id_str": self.operator_id_str, "reason_str": reason_str,
            "confirmation_text_str": MANUAL_ORDER_CONFIRMATION_TEXT_STR}
        if expected_dict["kind_str"] == "complete":
            ticket_dict.update(completion_vplan_id_int=expected_dict["vplan_id_int"],
                completion_target_share_float=float(expected_dict["plan_float"]),
                completion_fresh_position_float=held_float)
        line_str = order_line_str(side_str, quantity_obj, asset_str, type_str, limit_float)
        authorized_dict = {**expected_dict["confirmation_dict"], "expires_at_float": expected_dict["expires_at_float"],
            "action_name_str": ACTION_NAME_STR}
        try:
            # Same validation the sender repeats; refusing here means nothing reached the broker.
            build_manual_order_ticket_obj(release_obj=release_obj, request_body_dict=ticket_dict,
                submitted_timestamp_ts=self.now_fn())
        except ValueError as exception_obj:
            raise OrderStop(str(exception_obj).rstrip(".")) from exception_obj
        confirmed_target_obj = replace(target_obj, operator_confirmation_dict=authorized_dict)
        try:
            current_target_obj = self.target_loader_fn(pod_id_str)
            if current_target_obj is None:
                raise ValueError("Target is no longer enabled.")
            validate_current_confirmation(current_target_obj, authorized_dict)
        except ValueError as exception_obj:
            raise OrderStop("Pod state changed since the check · check again") from exception_obj
        try:
            self.pod_gate_obj.acquire(pod_id_str)
        except DashboardActionInFlightError as exception_obj:
            raise OrderStop("Another order is being sent for this pod") from exception_obj
        try:
            with operator_execution_context(confirmed_target_obj, self.target_loader_fn, ACTION_NAME_STR):
                result_dict = self.sender_fn(release_obj=release_obj, request_body_dict=ticket_dict,
                    submitted_timestamp_ts=self.now_fn(), broker_adapter_obj=adapter_obj, log_path_str=self.log_path_fn())
        except Exception:
            # Past validation the order may have reached the broker. Never claim it did not.
            return {"status_str": "unknown", "line_str": "Outcome unknown · check TWS before resending"}
        finally:
            self.pod_gate_obj.release(pod_id_str)
        order_id_str = ", ".join(result_dict.get("broker_order_id_list") or []) or "—"
        if result_dict.get("submit_ack_status_str") == "complete":
            return {"status_str": "sent", "line_str": f"Sent · {line_str} · order {order_id_str}"}
        return {"status_str": "unknown", "line_str": f"Sent, broker ACK not confirmed · {line_str} · check TWS before resending"}


def _json_body_dict(allowed_set):
    if request.args or request.mimetype != "application/json":
        raise ValueError("Use a JSON body without query parameters.")
    if request.content_length is not None and request.content_length > 8192:
        raise ValueError("Request body is too large.")
    body_dict = request.get_json(silent=False)
    if not isinstance(body_dict, dict) or set(body_dict) - allowed_set or body_dict.get("confirmed_bool") is not True:
        raise ValueError("Unexpected fields or missing explicit confirmation.")
    return body_dict


CHECK_FIELD_SET = frozenset({"confirmed_bool", "kind_str", "asset_str"})
SEND_FIELD_SET = frozenset({"confirmed_bool", "confirmation_nonce_str", "broker_order_type_str", "quantity_int",
    "limit_price_float", "side_str", "reason_str"})


def register_order_routes(app_obj, service_obj):
    def guarded(allowed_set):
        if not service_obj.enabled_bool:
            return None, (jsonify(error="orders_disabled", message="Orders are off in this dashboard."), 403)
        try:
            body_dict = _json_body_dict(allowed_set)
        except Exception:
            return None, (jsonify(error="invalid_body", message="Invalid request."), 400)
        rejection_tuple = validate_action_request(request.headers, body_dict, service_obj.action_token_str)
        origin_tuple = urlsplit(request.host_url)
        for key_str in ("Origin", "Referer"):
            if request.headers.get(key_str):
                source_tuple = urlsplit(request.headers[key_str])
                if (source_tuple.scheme, source_tuple.netloc.lower()) != (origin_tuple.scheme, origin_tuple.netloc.lower()):
                    rejection_tuple = (403, "origin_rejected", "A same-origin request is required.")
        if rejection_tuple:
            return None, (jsonify(error=rejection_tuple[1], message=rejection_tuple[2]), rejection_tuple[0])
        return body_dict, None

    @app_obj.post("/api/pods/<pod_id_str>/order/check", endpoint="order_check")
    def order_check(pod_id_str):
        body_dict, error_obj = guarded(CHECK_FIELD_SET)
        if error_obj is not None:
            return error_obj
        try:
            return jsonify(status_str="ok", **service_obj.check_dict(pod_id_str, body_dict))
        except OrderStop as exception_obj:
            return jsonify(status_str="stopped", line_str=str(exception_obj)), 200
        except ValueError:
            return jsonify(status_str="stopped", line_str="Pod not verified · refresh the page"), 200

    @app_obj.post("/api/pods/<pod_id_str>/order/send", endpoint="order_send")
    def order_send(pod_id_str):
        body_dict, error_obj = guarded(SEND_FIELD_SET)
        if error_obj is not None:
            return error_obj
        try:
            return jsonify(service_obj.send_dict(pod_id_str, body_dict))
        except OrderStop as exception_obj:
            return jsonify(status_str="stopped", line_str=str(exception_obj)), 200
        except ValueError:
            return jsonify(status_str="stopped", line_str="Pod not verified · refresh the page"), 200
