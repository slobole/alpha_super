from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re
from typing import Any, Callable
import uuid

from alpha.live.logging_utils import DEFAULT_LOG_PATH_STR, log_event
from alpha.live.models import BrokerOrderRequest, LiveRelease, SubmitBatchResult
from alpha.live.order_clerk import (
    BrokerAdapter,
    IBKRGatewayBrokerAdapter,
    validate_account_route_matches_mode,
)


MANUAL_ORDER_CONFIRMATION_TEXT_STR = "SUBMIT MANUAL ORDER"
SUPPORTED_MANUAL_ORDER_TYPE_TUPLE = ("MKT", "LMT")
SUPPORTED_MANUAL_SIDE_TUPLE = ("BUY", "SELL")
SUPPORTED_MANUAL_TIF_TUPLE = ("DAY",)
_ASSET_PATTERN_OBJ = re.compile(r"^[A-Z][A-Z0-9.\-]{0,15}$")


BrokerAdapterFactory = Callable[[str, int, int, float], BrokerAdapter]


@dataclass(frozen=True)
class ManualOrderTicket:
    ticket_id_str: str
    release_id_str: str
    pod_id_str: str
    mode_str: str
    account_route_str: str
    operator_id_str: str
    asset_str: str
    side_str: str
    broker_order_type_str: str
    quantity_int: int
    time_in_force_str: str
    reason_str: str
    submitted_timestamp_ts: datetime
    submission_key_str: str
    order_request_key_str: str
    limit_price_float: float | None = None

    @property
    def signed_quantity_float(self) -> float:
        side_multiplier_float = 1.0 if self.side_str == "BUY" else -1.0
        return side_multiplier_float * float(self.quantity_int)


def submit_manual_order_ticket_dict(
    *,
    release_obj: LiveRelease,
    request_body_dict: dict[str, Any],
    submitted_timestamp_ts: datetime | None = None,
    broker_adapter_obj: BrokerAdapter | None = None,
    adapter_factory_func: BrokerAdapterFactory | None = None,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
) -> dict[str, Any]:
    submitted_timestamp_ts = submitted_timestamp_ts or datetime.now(UTC)
    manual_order_ticket_obj = build_manual_order_ticket_obj(
        release_obj=release_obj,
        request_body_dict=request_body_dict,
        submitted_timestamp_ts=submitted_timestamp_ts,
    )
    broker_order_request_obj = build_manual_broker_order_request_obj(
        manual_order_ticket_obj
    )
    request_payload_dict = _manual_order_log_payload_dict(
        manual_order_ticket_obj,
        extra_payload_dict={"reason_code_str": "manual_order_submit_requested"},
    )
    log_event(
        "manual_order_submit_requested",
        request_payload_dict,
        log_path_str=log_path_str,
    )

    if broker_adapter_obj is None:
        adapter_factory_func = adapter_factory_func or _default_adapter_factory
        broker_adapter_obj = adapter_factory_func(
            release_obj.broker_host_str,
            int(release_obj.broker_port_int),
            int(release_obj.broker_client_id_int),
            float(release_obj.broker_timeout_seconds_float),
        )

    try:
        submit_batch_result_obj = broker_adapter_obj.submit_order_request_list(
            account_route_str=manual_order_ticket_obj.account_route_str,
            broker_order_request_list=[broker_order_request_obj],
            submitted_timestamp_ts=submitted_timestamp_ts,
        )
    except Exception as exception_obj:
        log_event(
            "manual_order_submit_failed",
            _manual_order_log_payload_dict(
                manual_order_ticket_obj,
                extra_payload_dict={
                    "severity_str": "critical",
                    "reason_code_str": "manual_order_submit_failed",
                    "error_str": str(exception_obj),
                },
            ),
            log_path_str=log_path_str,
        )
        raise

    result_dict = _manual_order_submit_result_dict(
        manual_order_ticket_obj,
        submit_batch_result_obj,
    )
    log_event(
        "manual_order_submit_completed",
        _manual_order_log_payload_dict(
            manual_order_ticket_obj,
            extra_payload_dict={
                "reason_code_str": "manual_order_submit_completed",
                "submit_ack_status_str": result_dict["submit_ack_status_str"],
                "broker_order_id_list": result_dict["broker_order_id_list"],
                "broker_order_ack_count_int": result_dict["broker_order_ack_count_int"],
                "missing_ack_asset_list": result_dict["missing_ack_asset_list"],
            },
        ),
        log_path_str=log_path_str,
    )
    return result_dict


def build_manual_order_ticket_obj(
    *,
    release_obj: LiveRelease,
    request_body_dict: dict[str, Any],
    submitted_timestamp_ts: datetime,
) -> ManualOrderTicket:
    if release_obj.mode_str not in {"paper", "live"}:
        raise ValueError(
            "Manual broker tickets are supported only for paper/live IBKR pods."
        )
    validate_account_route_matches_mode(release_obj.mode_str, release_obj.account_route_str)

    asset_str = _normalize_asset_str(request_body_dict.get("asset_str"))
    side_str = _normalize_side_str(request_body_dict.get("side_str"))
    broker_order_type_str = _normalize_order_type_str(
        request_body_dict.get("broker_order_type_str")
    )
    quantity_int = _parse_positive_quantity_int(request_body_dict.get("quantity_int"))
    time_in_force_str = _normalize_tif_str(request_body_dict.get("time_in_force_str", "DAY"))
    operator_id_str = _require_nonempty_text_str(
        request_body_dict.get("operator_id_str"),
        field_name_str="operator_id_str",
    )
    reason_str = _require_nonempty_text_str(
        request_body_dict.get("reason_str"),
        field_name_str="reason_str",
    )
    confirmation_text_str = str(request_body_dict.get("confirmation_text_str", "")).strip()
    if confirmation_text_str != MANUAL_ORDER_CONFIRMATION_TEXT_STR:
        raise ValueError(
            "confirmation_text_str must match "
            f"{MANUAL_ORDER_CONFIRMATION_TEXT_STR!r}."
        )

    limit_price_float = _parse_limit_price_float(
        request_body_dict.get("limit_price_float"),
        broker_order_type_str=broker_order_type_str,
    )
    ticket_id_str = uuid.uuid4().hex
    submission_key_str = f"manual:{release_obj.pod_id_str}:{ticket_id_str}"
    order_request_key_str = f"{submission_key_str}:{asset_str}:1"
    return ManualOrderTicket(
        ticket_id_str=ticket_id_str,
        release_id_str=release_obj.release_id_str,
        pod_id_str=release_obj.pod_id_str,
        mode_str=release_obj.mode_str,
        account_route_str=release_obj.account_route_str,
        operator_id_str=operator_id_str,
        asset_str=asset_str,
        side_str=side_str,
        broker_order_type_str=broker_order_type_str,
        quantity_int=quantity_int,
        time_in_force_str=time_in_force_str,
        reason_str=reason_str,
        submitted_timestamp_ts=submitted_timestamp_ts,
        submission_key_str=submission_key_str,
        order_request_key_str=order_request_key_str,
        limit_price_float=limit_price_float,
    )


def build_manual_broker_order_request_obj(
    manual_order_ticket_obj: ManualOrderTicket,
) -> BrokerOrderRequest:
    reference_price_float = (
        float(manual_order_ticket_obj.limit_price_float)
        if manual_order_ticket_obj.limit_price_float is not None
        else 1.0
    )
    return BrokerOrderRequest(
        release_id_str=manual_order_ticket_obj.release_id_str,
        pod_id_str=manual_order_ticket_obj.pod_id_str,
        account_route_str=manual_order_ticket_obj.account_route_str,
        submission_key_str=manual_order_ticket_obj.submission_key_str,
        order_request_key_str=manual_order_ticket_obj.order_request_key_str,
        asset_str=manual_order_ticket_obj.asset_str,
        broker_order_type_str=manual_order_ticket_obj.broker_order_type_str,
        order_class_str="ManualBrokerTicket",
        unit_str="shares",
        amount_float=manual_order_ticket_obj.signed_quantity_float,
        target_bool=False,
        trade_id_int=None,
        sizing_reference_price_float=reference_price_float,
        portfolio_value_float=0.0,
        decision_plan_id_int=None,
        vplan_id_int=None,
        limit_price_float=manual_order_ticket_obj.limit_price_float,
    )


def _default_adapter_factory(
    broker_host_str: str,
    broker_port_int: int,
    broker_client_id_int: int,
    broker_timeout_seconds_float: float,
) -> BrokerAdapter:
    return IBKRGatewayBrokerAdapter(
        host_str=broker_host_str,
        port_int=broker_port_int,
        client_id_int=broker_client_id_int,
        timeout_seconds_float=broker_timeout_seconds_float,
    )


def _normalize_asset_str(asset_obj: Any) -> str:
    asset_str = str(asset_obj or "").strip().upper()
    if not _ASSET_PATTERN_OBJ.fullmatch(asset_str):
        raise ValueError("asset_str must be a simple US stock/ETF symbol.")
    return asset_str


def _normalize_side_str(side_obj: Any) -> str:
    side_str = str(side_obj or "").strip().upper()
    if side_str not in SUPPORTED_MANUAL_SIDE_TUPLE:
        raise ValueError(f"side_str must be one of {SUPPORTED_MANUAL_SIDE_TUPLE}.")
    return side_str


def _normalize_order_type_str(order_type_obj: Any) -> str:
    order_type_str = str(order_type_obj or "").strip().upper()
    if order_type_str == "MARKET":
        order_type_str = "MKT"
    if order_type_str == "LIMIT":
        order_type_str = "LMT"
    if order_type_str not in SUPPORTED_MANUAL_ORDER_TYPE_TUPLE:
        raise ValueError(
            f"broker_order_type_str must be one of {SUPPORTED_MANUAL_ORDER_TYPE_TUPLE}."
        )
    return order_type_str


def _normalize_tif_str(time_in_force_obj: Any) -> str:
    time_in_force_str = str(time_in_force_obj or "DAY").strip().upper()
    if time_in_force_str not in SUPPORTED_MANUAL_TIF_TUPLE:
        raise ValueError(f"time_in_force_str must be one of {SUPPORTED_MANUAL_TIF_TUPLE}.")
    return time_in_force_str


def _parse_positive_quantity_int(quantity_obj: Any) -> int:
    quantity_text_str = str(quantity_obj or "").strip()
    if not quantity_text_str.isdigit():
        raise ValueError("quantity_int must be a positive integer share quantity.")
    quantity_int = int(quantity_text_str)
    if quantity_int <= 0:
        raise ValueError("quantity_int must be greater than zero.")
    return quantity_int


def _parse_limit_price_float(
    limit_price_obj: Any,
    *,
    broker_order_type_str: str,
) -> float | None:
    limit_price_text_str = str(limit_price_obj or "").strip()
    if broker_order_type_str == "MKT":
        return None
    if limit_price_text_str == "":
        raise ValueError("limit_price_float is required for LMT orders.")
    try:
        limit_price_float = float(limit_price_text_str)
    except ValueError as exception_obj:
        raise ValueError("limit_price_float must be a positive number.") from exception_obj
    if limit_price_float <= 0.0:
        raise ValueError("limit_price_float must be greater than zero.")
    return limit_price_float


def _require_nonempty_text_str(value_obj: Any, *, field_name_str: str) -> str:
    value_str = str(value_obj or "").strip()
    if value_str == "":
        raise ValueError(f"{field_name_str} is required.")
    return value_str


def _manual_order_submit_result_dict(
    manual_order_ticket_obj: ManualOrderTicket,
    submit_batch_result_obj: SubmitBatchResult,
) -> dict[str, Any]:
    broker_order_id_list = [
        str(broker_order_record_obj.broker_order_id_str)
        for broker_order_record_obj in submit_batch_result_obj.broker_order_record_list
    ]
    return {
        "schema_version_str": "manual_order_ticket.v1",
        "ticket_id_str": manual_order_ticket_obj.ticket_id_str,
        "status_str": "submitted",
        "mode_str": manual_order_ticket_obj.mode_str,
        "pod_id_str": manual_order_ticket_obj.pod_id_str,
        "account_route_str": manual_order_ticket_obj.account_route_str,
        "operator_id_str": manual_order_ticket_obj.operator_id_str,
        "asset_str": manual_order_ticket_obj.asset_str,
        "side_str": manual_order_ticket_obj.side_str,
        "broker_order_type_str": manual_order_ticket_obj.broker_order_type_str,
        "quantity_int": manual_order_ticket_obj.quantity_int,
        "limit_price_float": manual_order_ticket_obj.limit_price_float,
        "time_in_force_str": manual_order_ticket_obj.time_in_force_str,
        "reason_str": manual_order_ticket_obj.reason_str,
        "submission_key_str": manual_order_ticket_obj.submission_key_str,
        "order_request_key_str": manual_order_ticket_obj.order_request_key_str,
        "submitted_timestamp_str": manual_order_ticket_obj.submitted_timestamp_ts.isoformat(),
        "submit_ack_status_str": submit_batch_result_obj.submit_ack_status_str,
        "ack_coverage_ratio_float": float(submit_batch_result_obj.ack_coverage_ratio_float),
        "missing_ack_asset_list": list(submit_batch_result_obj.missing_ack_asset_list),
        "broker_order_id_list": broker_order_id_list,
        "broker_order_record_count_int": len(submit_batch_result_obj.broker_order_record_list),
        "broker_order_event_count_int": len(submit_batch_result_obj.broker_order_event_list),
        "broker_order_ack_count_int": len(submit_batch_result_obj.broker_order_ack_list),
    }


def _manual_order_log_payload_dict(
    manual_order_ticket_obj: ManualOrderTicket,
    *,
    extra_payload_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload_dict: dict[str, Any] = {
        "severity_str": "warning",
        "source_str": "manual_broker_ticket",
        "ticket_id_str": manual_order_ticket_obj.ticket_id_str,
        "release_id_str": manual_order_ticket_obj.release_id_str,
        "pod_id_str": manual_order_ticket_obj.pod_id_str,
        "mode_str": manual_order_ticket_obj.mode_str,
        "account_route_str": manual_order_ticket_obj.account_route_str,
        "operator_id_str": manual_order_ticket_obj.operator_id_str,
        "asset_str": manual_order_ticket_obj.asset_str,
        "side_str": manual_order_ticket_obj.side_str,
        "broker_order_type_str": manual_order_ticket_obj.broker_order_type_str,
        "quantity_int": manual_order_ticket_obj.quantity_int,
        "limit_price_float": manual_order_ticket_obj.limit_price_float,
        "time_in_force_str": manual_order_ticket_obj.time_in_force_str,
        "reason_str": manual_order_ticket_obj.reason_str,
        "submission_key_str": manual_order_ticket_obj.submission_key_str,
        "order_request_key_str": manual_order_ticket_obj.order_request_key_str,
        "submitted_timestamp_str": manual_order_ticket_obj.submitted_timestamp_ts.isoformat(),
    }
    payload_dict.update(extra_payload_dict or {})
    return payload_dict
