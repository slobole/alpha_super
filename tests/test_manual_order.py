from __future__ import annotations

from datetime import UTC, datetime
import json

import pytest

from alpha.live.manual_order import (
    MANUAL_ORDER_CONFIRMATION_TEXT_STR,
    submit_manual_order_ticket_dict,
)
from alpha.live.models import (
    BrokerOrderAck,
    BrokerOrderFill,
    BrokerOrderRecord,
    BrokerOrderRequest,
    BrokerSnapshot,
    LivePriceSnapshot,
    LiveRelease,
    SessionOpenPrice,
    SubmitBatchResult,
)
from alpha.live.order_clerk import BrokerAdapter


class RecordingSubmitOnlyBrokerAdapter(BrokerAdapter):
    def __init__(self) -> None:
        self.submitted_request_list: list[BrokerOrderRequest] = []

    def get_visible_account_route_set(self) -> set[str] | None:
        raise AssertionError("manual order submit must not read visible accounts")

    def get_session_mode_str(self, account_route_str: str) -> str | None:
        raise AssertionError("manual order submit must not read session mode")

    def is_session_ready(self, account_route_str: str) -> bool:
        raise AssertionError("manual order submit must not probe session readiness")

    def get_account_snapshot(self, account_route_str: str) -> BrokerSnapshot:
        raise AssertionError("manual order submit must not read broker snapshot")

    def get_live_price_snapshot(
        self,
        account_route_str: str,
        asset_str_list: list[str],
        execution_policy_str: str | None = None,
    ) -> LivePriceSnapshot:
        raise AssertionError("manual order submit must not read live prices")

    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> SubmitBatchResult:
        self.submitted_request_list.extend(broker_order_request_list)
        request_obj = broker_order_request_list[0]
        broker_order_id_str = "manual_order_1"
        return SubmitBatchResult(
            broker_order_record_list=[
                BrokerOrderRecord(
                    broker_order_id_str=broker_order_id_str,
                    decision_plan_id_int=None,
                    vplan_id_int=None,
                    account_route_str=account_route_str,
                    asset_str=request_obj.asset_str,
                    order_request_key_str=request_obj.order_request_key_str,
                    broker_order_type_str=request_obj.broker_order_type_str,
                    unit_str=request_obj.unit_str,
                    amount_float=request_obj.amount_float,
                    filled_amount_float=0.0,
                    status_str="Submitted",
                    submitted_timestamp_ts=submitted_timestamp_ts,
                    submission_key_str=request_obj.submission_key_str,
                    raw_payload_dict={},
                )
            ],
            broker_order_ack_list=[
                BrokerOrderAck(
                    decision_plan_id_int=None,
                    vplan_id_int=None,
                    account_route_str=account_route_str,
                    order_request_key_str=request_obj.order_request_key_str,
                    asset_str=request_obj.asset_str,
                    broker_order_type_str=request_obj.broker_order_type_str,
                    local_submit_ack_bool=True,
                    broker_response_ack_bool=True,
                    ack_status_str="broker_acked",
                    ack_source_str="recording_adapter",
                    broker_order_id_str=broker_order_id_str,
                    response_timestamp_ts=submitted_timestamp_ts,
                    raw_payload_dict={},
                )
            ],
            ack_coverage_ratio_float=1.0,
            missing_ack_asset_list=[],
            submit_ack_status_str="complete",
        )

    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> list[BrokerOrderFill]:
        raise AssertionError("manual order submit must not read fills")

    def get_recent_order_state_snapshot(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        submission_key_str: str | None = None,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> tuple[list[BrokerOrderRecord], list, list[BrokerOrderFill]]:
        raise AssertionError("manual order submit must not read order state")

    def get_session_open_price_list(
        self,
        account_route_str: str,
        asset_str_list: list[str],
        session_open_timestamp_ts: datetime,
        session_calendar_id_str: str,
    ) -> list[SessionOpenPrice]:
        raise AssertionError("manual order submit must not read session opens")


def _release_obj(mode_str: str = "paper", account_route_str: str = "DU123") -> LiveRelease:
    return LiveRelease(
        release_id_str="release_manual",
        user_id_str="user",
        pod_id_str="pod_manual",
        account_route_str=account_route_str,
        strategy_import_str="strategies.demo:Demo",
        mode_str=mode_str,
        session_calendar_id_str="XNYS",
        signal_clock_str="manual",
        execution_policy_str="manual",
        data_profile_str="manual",
        params_dict={},
        risk_profile_str="manual",
        enabled_bool=True,
        source_path_str="manual.yaml",
    )


def _body_dict(**override_dict) -> dict[str, str]:
    body_dict = {
        "asset_str": "aapl",
        "side_str": "BUY",
        "broker_order_type_str": "MARKET",
        "quantity_int": "12",
        "time_in_force_str": "DAY",
        "operator_id_str": "ops",
        "reason_str": "missed monthly trade",
        "confirmation_text_str": MANUAL_ORDER_CONFIRMATION_TEXT_STR,
    }
    body_dict.update(override_dict)
    return body_dict


def test_manual_market_order_submits_without_broker_truth_reads(tmp_path) -> None:
    broker_adapter_obj = RecordingSubmitOnlyBrokerAdapter()
    log_path_obj = tmp_path / "events.jsonl"

    result_dict = submit_manual_order_ticket_dict(
        release_obj=_release_obj(),
        request_body_dict=_body_dict(),
        submitted_timestamp_ts=datetime(2026, 6, 15, 14, 0, tzinfo=UTC),
        broker_adapter_obj=broker_adapter_obj,
        log_path_str=str(log_path_obj),
    )

    request_obj = broker_adapter_obj.submitted_request_list[0]
    assert request_obj.asset_str == "AAPL"
    assert request_obj.broker_order_type_str == "MKT"
    assert request_obj.amount_float == 12.0
    assert request_obj.limit_price_float is None
    assert request_obj.order_class_str == "ManualBrokerTicket"
    assert result_dict["submit_ack_status_str"] == "complete"
    assert result_dict["broker_order_id_list"] == ["manual_order_1"]

    event_dict_list = [
        json.loads(line_str)
        for line_str in log_path_obj.read_text(encoding="utf-8").splitlines()
    ]
    assert [event_dict["event_name_str"] for event_dict in event_dict_list] == [
        "manual_order_submit_requested",
        "manual_order_submit_completed",
    ]
    assert event_dict_list[0]["asset_str"] == "AAPL"
    assert event_dict_list[0]["pod_id_str"] == "pod_manual"


def test_manual_limit_order_sets_limit_price_and_signed_sell_quantity(tmp_path) -> None:
    broker_adapter_obj = RecordingSubmitOnlyBrokerAdapter()

    submit_manual_order_ticket_dict(
        release_obj=_release_obj(),
        request_body_dict=_body_dict(
            side_str="SELL",
            broker_order_type_str="LIMIT",
            quantity_int="3",
            limit_price_float="123.45",
        ),
        submitted_timestamp_ts=datetime(2026, 6, 15, 14, 5, tzinfo=UTC),
        broker_adapter_obj=broker_adapter_obj,
        log_path_str=str(tmp_path / "events.jsonl"),
    )

    request_obj = broker_adapter_obj.submitted_request_list[0]
    assert request_obj.broker_order_type_str == "LMT"
    assert request_obj.amount_float == -3.0
    assert request_obj.limit_price_float == 123.45
    assert request_obj.sizing_reference_price_float == 123.45


def test_manual_limit_order_requires_limit_price() -> None:
    with pytest.raises(ValueError, match="limit_price_float is required"):
        submit_manual_order_ticket_dict(
            release_obj=_release_obj(),
            request_body_dict=_body_dict(
                broker_order_type_str="LIMIT",
                limit_price_float="",
            ),
            submitted_timestamp_ts=datetime(2026, 6, 15, 14, 10, tzinfo=UTC),
            broker_adapter_obj=RecordingSubmitOnlyBrokerAdapter(),
        )
