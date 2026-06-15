from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from alpha.live.dashboard_v3.app import create_app
from alpha.live.manual_order import MANUAL_ORDER_CONFIRMATION_TEXT_STR


class ManualOrderProvider:
    def __init__(self) -> None:
        self.action_token_str = "token_123"
        self.submitted_body_dict: dict[str, Any] | None = None
        self.target_obj = SimpleNamespace(release_obj=SimpleNamespace(pod_id_str="pod_manual"))
        self.row_dict = {
            "pod_id_str": "pod_manual",
            "mode_str": "paper",
            "account_route_str": "DU123",
            "health_str": "green",
            "required_action_dict": {},
            "debug_summary_dict": {},
        }

    def get_summary_dict(self) -> dict[str, Any]:
        return {
            "pod_row_dict_list": [self.row_dict],
            "alert_dict_list": [],
            "alert_summary_dict": {},
            "mode_list": ["paper"],
        }

    def get_pod_detail_dict(self, pod_id_str: str) -> dict[str, Any]:
        raise AssertionError("manual order routes should not load pod detail")

    def get_pod_event_dict_list(
        self, pod_id_str: str, limit_int: int = 80
    ) -> list[dict[str, Any]]:
        return []

    def get_pod_trace_event_dict_list(
        self, pod_id_str: str, limit_int: int = 80
    ) -> list[dict[str, Any]]:
        return []

    def get_action_token_str(self) -> str:
        return self.action_token_str

    def get_target_for_pod(self, pod_id_str: str):
        if pod_id_str != "pod_manual":
            return None
        return self.target_obj

    def submit_manual_order_dict(
        self,
        target_obj,
        request_body_dict: dict[str, Any],
    ) -> dict[str, Any]:
        del target_obj
        self.submitted_body_dict = dict(request_body_dict)
        return {
            "ticket_id_str": "ticket_1",
            "side_str": "BUY",
            "quantity_int": 10,
            "asset_str": "AAPL",
            "broker_order_type_str": "MKT",
            "limit_price_float": None,
            "account_route_str": "DU123",
            "submit_ack_status_str": "complete",
            "broker_order_ack_count_int": 1,
            "broker_order_id_list": ["order_1"],
        }


def _action_headers_dict() -> dict[str, str]:
    return {
        "Host": "localhost",
        "Origin": "http://localhost",
        "HX-Request": "true",
        "X-Alpha-Action-Token": "token_123",
    }


def test_manual_order_ticket_fragment_renders_form_fields() -> None:
    provider_obj = ManualOrderProvider()
    app_obj = create_app(data_provider_obj=provider_obj)
    client_obj = app_obj.test_client()

    response_obj = client_obj.get("/fragments/manual-order-ticket/pod_manual")
    response_text_str = response_obj.get_data(as_text=True)

    assert response_obj.status_code == 200
    assert "Manual Broker Ticket" in response_text_str
    assert 'name="asset_str"' in response_text_str
    assert 'name="broker_order_type_str"' in response_text_str
    assert 'name="limit_price_float"' in response_text_str
    assert MANUAL_ORDER_CONFIRMATION_TEXT_STR in response_text_str


def test_manual_order_submit_route_validates_action_and_calls_provider() -> None:
    provider_obj = ManualOrderProvider()
    app_obj = create_app(data_provider_obj=provider_obj)
    client_obj = app_obj.test_client()

    response_obj = client_obj.post(
        "/api/pods/pod_manual/manual-order",
        headers=_action_headers_dict(),
        data={
            "confirmed_bool": "true",
            "asset_str": "AAPL",
            "side_str": "BUY",
            "broker_order_type_str": "MKT",
            "quantity_int": "10",
            "operator_id_str": "ops",
            "reason_str": "missed trade",
            "confirmation_text_str": MANUAL_ORDER_CONFIRMATION_TEXT_STR,
        },
    )
    response_text_str = response_obj.get_data(as_text=True)

    assert response_obj.status_code == 200
    assert provider_obj.submitted_body_dict is not None
    assert provider_obj.submitted_body_dict["asset_str"] == "AAPL"
    assert "Manual order submitted" in response_text_str
    assert "order_1" in response_text_str
