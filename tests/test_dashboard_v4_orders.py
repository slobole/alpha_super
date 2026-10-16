"""V4 order panel: MOC in the manual ticket, Missing trades, and the check/send service."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
import json
import re
from types import SimpleNamespace

import pytest

from alpha.live.daily_broker import DailyExecutionSnapshot
from alpha.live.dashboard import DashboardPodTarget
from alpha.live.dashboard_v4 import missing_trades as missing_module
from alpha.live.dashboard_v4 import orders as orders_module
from alpha.live.dashboard_v4.missing_trades import load_missing_trades_dict, plan_fresh_bool, read_completion_event_list
from alpha.live.dashboard_v4.orders import OrderStop, OrderTicketService
from alpha.live.manual_order import (
    MANUAL_ORDER_CONFIRMATION_TEXT_STR,
    build_manual_broker_order_request_obj,
    build_manual_order_ticket_obj,
    manual_order_session_window_dict,
    submit_manual_order_ticket_dict,
)
from alpha.live.models import BrokerSnapshot
from test_manual_order import RecordingSubmitOnlyBrokerAdapter, _body_dict, _release_obj


# 2026-10-16 is a Friday; 2026-11-27 (day after Thanksgiving) closes at 13:00 ET.
FRI_1402_TS = datetime(2026, 10, 16, 18, 2, tzinfo=UTC)
FRI_1546_TS = datetime(2026, 10, 16, 19, 46, tzinfo=UTC)
FRI_1601_TS = datetime(2026, 10, 16, 20, 1, tzinfo=UTC)
FRI_0800_TS = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
SAT_TS = datetime(2026, 10, 17, 15, 0, tzinfo=UTC)
HALF_1240_TS = datetime(2026, 11, 27, 17, 40, tzinfo=UTC)
HALF_1246_TS = datetime(2026, 11, 27, 17, 46, tzinfo=UTC)


def _live_release():
    return _release_obj(mode_str="live", account_route_str="U1234567")


# ---------------------------------------------------------------- backend: MOC
@pytest.mark.parametrize("as_of_ts, ok_bool", [
    (FRI_1402_TS, True), (FRI_0800_TS, True), (HALF_1240_TS, True),
    (FRI_1546_TS, False), (HALF_1246_TS, False), (SAT_TS, False)])
def test_moc_window(as_of_ts, ok_bool):
    body_dict = _body_dict(broker_order_type_str="MOC")
    if ok_bool:
        ticket_obj = build_manual_order_ticket_obj(release_obj=_live_release(), request_body_dict=body_dict,
            submitted_timestamp_ts=as_of_ts)
        assert ticket_obj.broker_order_type_str == "MOC" and ticket_obj.limit_price_float is None
    else:
        with pytest.raises(ValueError, match="MOC"):
            build_manual_order_ticket_obj(release_obj=_live_release(), request_body_dict=body_dict,
                submitted_timestamp_ts=as_of_ts)


def test_session_window_half_day_and_next_open():
    window_dict = manual_order_session_window_dict("XNYS", HALF_1240_TS)
    assert window_dict["moc_cutoff_ts"].strftime("%H:%M") == "12:45"
    assert manual_order_session_window_dict("XNYS", SAT_TS)["next_open_ts"].strftime("%Y-%m-%d %H:%M") == "2026-10-19 09:30"


def test_moc_alias_ignores_limit_and_reaches_adapter_as_moc():
    ticket_obj = build_manual_order_ticket_obj(release_obj=_live_release(),
        request_body_dict=_body_dict(broker_order_type_str="MARKET_ON_CLOSE", limit_price_float="12.5"),
        submitted_timestamp_ts=FRI_1402_TS)
    request_obj = build_manual_broker_order_request_obj(ticket_obj)
    assert request_obj.broker_order_type_str == "MOC" and request_obj.limit_price_float is None


def test_mkt_and_lmt_keep_their_behaviour_on_any_day():
    ticket_obj = build_manual_order_ticket_obj(release_obj=_live_release(), request_body_dict=_body_dict(),
        submitted_timestamp_ts=SAT_TS)
    assert ticket_obj.broker_order_type_str == "MKT"


def test_completion_reference_is_logged_and_key_stays_manual(tmp_path):
    log_path_obj = tmp_path / "events.jsonl"
    result_dict = submit_manual_order_ticket_dict(release_obj=_live_release(),
        request_body_dict=_body_dict(broker_order_type_str="MOC", completion_vplan_id_int=412,
            completion_target_share_float=300, completion_fresh_position_float=0.0),
        submitted_timestamp_ts=FRI_1402_TS, broker_adapter_obj=RecordingSubmitOnlyBrokerAdapter(),
        log_path_str=str(log_path_obj))
    assert result_dict["submission_key_str"].startswith("manual:pod_manual:")
    assert result_dict["completion_vplan_id_int"] == 412
    record_list = [json.loads(line_str) for line_str in log_path_obj.read_text(encoding="utf-8").splitlines()]
    assert [record_dict["event_name_str"] for record_dict in record_list] == [
        "manual_order_submit_requested", "manual_order_submit_completed"]
    assert all(record_dict["completion_vplan_id_int"] == 412 and record_dict["completion_target_share_float"] == 300.0
        for record_dict in record_list)


def test_completion_reference_rejects_bad_values():
    with pytest.raises(ValueError):
        build_manual_order_ticket_obj(release_obj=_live_release(), request_body_dict=_body_dict(
            completion_vplan_id_int=True, completion_target_share_float=1, completion_fresh_position_float=0),
            submitted_timestamp_ts=FRI_1402_TS)
    with pytest.raises(ValueError):
        build_manual_order_ticket_obj(release_obj=_live_release(), request_body_dict=_body_dict(
            completion_vplan_id_int=4, completion_target_share_float=float("nan"), completion_fresh_position_float=0),
            submitted_timestamp_ts=FRI_1402_TS)


# ---------------------------------------------------------------- Missing trades model
def _vplan(**override_dict):
    vplan_dict = dict(pod_id_str="pod_manual", account_route_str="U1234567", decision_plan_id_int=7, vplan_id_int=412,
        status_str="submitted", target_execution_timestamp_ts=datetime(2026, 10, 16, 13, 30, tzinfo=UTC),
        target_share_map={"APX": 300.0, "MDLZ": 0.0, "NVDA": 18.0, "BIL": -1.0},
        order_delta_map={"APX": 300.0, "MDLZ": -41.0, "NVDA": 0.0})
    vplan_dict.update(override_dict)
    return SimpleNamespace(**vplan_dict)


def _snapshot(position_map, at_ts=datetime(2026, 10, 16, 13, 45, tzinfo=UTC)):
    return BrokerSnapshot(account_route_str="U1234567", snapshot_timestamp_ts=at_ts, cash_float=0.0,
        total_value_float=0.0, position_amount_map=position_map)


@pytest.fixture
def fake_store(monkeypatch):
    state_dict = {"vplan": _vplan(), "decision": SimpleNamespace(decision_plan_id_int=7),
        "snapshot": _snapshot({"MDLZ": 41.0, "NVDA": 18.0}), "why": ({"APX": "broker cancelled"}, {"NVDA", "APX"})}

    class FakeStore:
        def __init__(self, db_path_str):
            self.connection_obj = SimpleNamespace(close=lambda: None)

        def get_latest_vplan_for_pod(self, pod_id_str):
            return state_dict["vplan"]

        def get_latest_decision_plan_for_pod(self, pod_id_str):
            return state_dict["decision"]

        def get_latest_broker_snapshot_for_account(self, account_route_str):
            return state_dict["snapshot"]

    monkeypatch.setattr(missing_module, "_ReadOnlyConfirmationStore", FakeStore)
    monkeypatch.setattr(missing_module, "_why_map", lambda db_path_str, vplan_id_int: state_dict["why"])
    return state_dict


def _target():
    return DashboardPodTarget(_live_release(), "synthetic.sqlite3", False)


def _load(as_of_ts=FRI_1402_TS, *, shared_bool=False, event_list=()):
    return load_missing_trades_dict(_target(), as_of_ts=as_of_ts, shared_account_bool=shared_bool,
        completion_event_list=list(event_list))


def test_missing_rows_and_reasons(fake_store):
    result_dict = _load()
    assert result_dict["status_str"] == "ok"
    assert [(row["asset_str"], row["side_str"], row["quantity_int"], row["text_str"]) for row in result_dict["row_list"]] == [
        ("APX", "BUY", 300, "Missing · broker cancelled"), ("MDLZ", "SELL", 41, "Missing · no broker ACK")]


@pytest.mark.parametrize("change_dict, reason_str", [
    ({"vplan": _vplan(status_str="ready")}, "Plan not sent"),
    ({"vplan": _vplan(status_str="blocked")}, "Plan not sent"),
    ({"decision": SimpleNamespace(decision_plan_id_int=8)}, "A newer decision supersedes this plan"),
    ({"snapshot": _snapshot({}, at_ts=datetime(2026, 10, 16, 13, 20, tzinfo=UTC))}, "Waiting for broker positions after the plan's execution time"),
    ({"snapshot": None}, "Waiting for broker positions after the plan's execution time"),
    ({"vplan": None}, "No order plan")])
def test_missing_hidden_rules(fake_store, change_dict, reason_str):
    fake_store.update(change_dict)
    result_dict = _load()
    assert (result_dict["status_str"], result_dict["reason_str"], result_dict["row_list"]) == ("hidden", reason_str, [])


def test_shared_account_shows_nothing(fake_store):
    assert _load(shared_bool=True)["status_str"] == "hidden"


def test_old_plan_rows_have_no_prefill(fake_store):
    fake_store["vplan"] = _vplan(target_execution_timestamp_ts=datetime(2026, 10, 1, 13, 30, tzinfo=UTC))
    assert {row["state_str"] for row in _load()["row_list"]} == {"off"}
    assert plan_fresh_bool(datetime(2026, 10, 15, 13, 30, tzinfo=UTC), FRI_1402_TS, "XNYS")
    assert not plan_fresh_bool(datetime(2026, 10, 14, 13, 30, tzinfo=UTC), FRI_1402_TS, "XNYS")


def test_sent_and_closed_rows_from_events(fake_store):
    event_list = [{"event_name_str": "manual_order_submit_completed", "pod_id_str": "pod_manual", "asset_str": "APX",
        "completion_vplan_id_int": 412, "event_timestamp_str": "2026-10-16T18:03:00+00:00", "broker_order_type_str": "MOC",
        "side_str": "BUY", "quantity_int": 300, "submit_ack_status_str": "complete", "broker_order_id_list": ["1184220455"]}]
    row_dict = _load(event_list=event_list)["row_list"][0]
    assert (row_dict["state_str"], row_dict["text_str"]) == ("sent", "Sent · MOC · waits for close · order 1184220455")
    fake_store["snapshot"] = _snapshot({"APX": 300.0, "MDLZ": 41.0, "NVDA": 18.0}, at_ts=datetime(2026, 10, 16, 20, 10, tzinfo=UTC))
    row_dict = _load(as_of_ts=datetime(2026, 10, 16, 20, 15, tzinfo=UTC), event_list=event_list)["row_list"][0]
    assert (row_dict["asset_str"], row_dict["state_str"], row_dict["text_str"]) == ("APX", "closed", "Closed by manual MOC · 300")


def test_read_completion_event_list_tail(tmp_path):
    path_obj = tmp_path / "events.jsonl"
    path_obj.write_text("\n".join([
        json.dumps({"event_name_str": "manual_order_submit_completed", "pod_id_str": "pod_manual", "asset_str": "APX"}),
        json.dumps({"event_name_str": "manual_order_submit_completed", "pod_id_str": "other", "asset_str": "APX"}),
        json.dumps({"event_name_str": "build_vplan_created", "pod_id_str": "pod_manual"}), "{broken"]), encoding="utf-8")
    assert [event_dict["asset_str"] for event_dict in read_completion_event_list(str(path_obj), "pod_manual")] == ["APX"]
    assert read_completion_event_list(str(tmp_path / "missing.jsonl"), "pod_manual") == []


# ---------------------------------------------------------------- order service
class FakeBroker:
    def __init__(self, position_map=None, open_list=None, fail_bool=False):
        self.position_map, self.open_list, self.fail_bool = dict(position_map or {}), list(open_list or []), fail_bool

    def get_daily_execution_snapshot(self, account_route_str):
        if self.fail_bool:
            raise ConnectionError("down")
        now_ts = datetime.now(UTC)
        return DailyExecutionSnapshot(_snapshot(self.position_map, at_ts=now_ts), self.open_list, now_ts, now_ts)


def _missing_ok(rows=None):
    return {"status_str": "ok", "vplan_id_int": 412, "target_share_map": {"APX": 300.0, "NVDA": 18.0},
        "row_list": rows if rows is not None else [{"asset_str": "APX", "side_str": "BUY", "quantity_int": 300,
            "target_share_float": 300.0, "state_str": "missing", "text_str": "Missing · broker cancelled"}]}


@pytest.fixture
def service_factory(monkeypatch):
    monkeypatch.setattr(orders_module, "validate_current_confirmation", lambda target_obj, expected_dict: None)
    monkeypatch.setattr(orders_module, "operator_execution_context", lambda *arg_tuple: nullcontext())

    def build(*, broker=None, missing=None, now_ts=FRI_1402_TS, sender=None):
        sent_list = []

        def record_sender(**keyword_dict):
            sent_list.append(keyword_dict)
            return {"submit_ack_status_str": "complete", "broker_order_id_list": ["555"]}
        target_obj = _target()
        service_obj = OrderTicketService(target_fn=lambda pod_id_str: target_obj,
            missing_fn=lambda target: missing or _missing_ok(), log_path_fn=lambda: "events.jsonl",
            target_loader_fn=lambda pod_id_str: target_obj, enabled_bool=True,
            broker_factory_fn=lambda release_obj: broker or FakeBroker(), sender_fn=sender or record_sender,
            now_fn=lambda: now_ts, operator_id_str="ops", confirmation_fn=lambda target: {"state_hash_str": "x"})
        return service_obj, sent_list
    return build


def test_complete_check_then_send_moc(service_factory):
    service_obj, sent_list = service_factory()
    check_dict = service_obj.check_dict("pod_manual", {"kind_str": "complete", "asset_str": "apx"})
    assert (check_dict["missing_str"], check_dict["default_type_str"], check_dict["cutoff_str"], check_dict["held_str"]) == (
        "BUY 300", "MOC", "15:45", "0")
    result_dict = service_obj.send_dict("pod_manual", {"confirmation_nonce_str": check_dict["confirmation_nonce_str"],
        "broker_order_type_str": "MOC", "quantity_int": 300})
    assert result_dict == {"status_str": "sent", "line_str": "Sent · BUY 300 APX at close · order 555"}
    ticket_dict = sent_list[0]["request_body_dict"]
    assert (ticket_dict["broker_order_type_str"], ticket_dict["side_str"], ticket_dict["completion_vplan_id_int"],
        ticket_dict["confirmation_text_str"], ticket_dict["reason_str"]) == (
        "MOC", "BUY", 412, MANUAL_ORDER_CONFIRMATION_TEXT_STR, "Completion · VPlan #412 · APX")
    with pytest.raises(OrderStop, match="Expired"):
        service_obj.send_dict("pod_manual", {"confirmation_nonce_str": check_dict["confirmation_nonce_str"],
            "broker_order_type_str": "MOC", "quantity_int": 300})


@pytest.mark.parametrize("broker, message_str", [
    (FakeBroker(open_list=[{"asset_str": "APX", "side_str": "BUY", "remaining_amount_float": 300.0}]),
        "Blocked · APX order already open (BUY 300) · wait or cancel in TWS"),
    (FakeBroker(position_map={"APX": 300.0}), "Nothing to complete · broker holds 300 = plan"),
    (FakeBroker(fail_bool=True), "Broker check failed · nothing sent")])
def test_complete_check_stops(service_factory, broker, message_str):
    service_obj, _ = service_factory(broker=broker)
    with pytest.raises(OrderStop) as error_obj:
        service_obj.check_dict("pod_manual", {"kind_str": "complete", "asset_str": "APX"})
    assert str(error_obj.value) == message_str


def test_fresh_broker_number_wins(service_factory):
    service_obj, _ = service_factory(broker=FakeBroker(position_map={"APX": 120.0}))
    check_dict = service_obj.check_dict("pod_manual", {"kind_str": "complete", "asset_str": "APX"})
    assert (check_dict["missing_str"], check_dict["max_quantity_int"], check_dict["changed_bool"], check_dict["page_str"]) == (
        "BUY 180", 180, True, "BUY 300")
    with pytest.raises(OrderStop, match="Max 180"):
        service_obj.send_dict("pod_manual", {"confirmation_nonce_str": check_dict["confirmation_nonce_str"],
            "broker_order_type_str": "MOC", "quantity_int": 181})


def test_market_closed_and_moc_cutoff(service_factory):
    service_obj, _ = service_factory(now_ts=FRI_1601_TS)
    with pytest.raises(OrderStop, match="Market closed · next open Mon 09:30"):
        service_obj.check_dict("pod_manual", {"kind_str": "complete", "asset_str": "APX"})
    service_obj, _ = service_factory(now_ts=FRI_1546_TS)
    check_dict = service_obj.check_dict("pod_manual", {"kind_str": "complete", "asset_str": "APX"})
    assert (check_dict["moc_bool"], check_dict["default_type_str"]) == (False, "MKT")
    with pytest.raises(OrderStop, match="MOC closed at 15:45"):
        service_obj.send_dict("pod_manual", {"confirmation_nonce_str": check_dict["confirmation_nonce_str"],
            "broker_order_type_str": "MOC", "quantity_int": 300})


def test_new_order_sell_cannot_go_short_and_needs_reason(service_factory):
    service_obj, sent_list = service_factory(broker=FakeBroker(position_map={"NVDA": 18.0}))
    check_dict = service_obj.check_dict("pod_manual", {"kind_str": "new", "asset_str": "nvda"})
    assert (check_dict["held_str"], check_dict["plan_str"], check_dict["max_sell_int"]) == ("18", "18", 18)
    nonce_str = check_dict["confirmation_nonce_str"]
    with pytest.raises(OrderStop, match="does not go short"):
        service_obj.send_dict("pod_manual", {"confirmation_nonce_str": nonce_str, "broker_order_type_str": "MOC",
            "quantity_int": 19, "side_str": "SELL", "reason_str": "exit"})
    nonce_str = service_obj.check_dict("pod_manual", {"kind_str": "new", "asset_str": "NVDA"})["confirmation_nonce_str"]
    with pytest.raises(OrderStop, match="Reason is required"):
        service_obj.send_dict("pod_manual", {"confirmation_nonce_str": nonce_str, "broker_order_type_str": "MOC",
            "quantity_int": 5, "side_str": "SELL", "reason_str": "   "})
    nonce_str = service_obj.check_dict("pod_manual", {"kind_str": "new", "asset_str": "NVDA"})["confirmation_nonce_str"]
    result_dict = service_obj.send_dict("pod_manual", {"confirmation_nonce_str": nonce_str, "broker_order_type_str": "LMT",
        "limit_price_float": 41.2, "quantity_int": 18, "side_str": "SELL", "reason_str": "Exit before earnings"})
    assert result_dict["line_str"] == "Sent · SELL 18 NVDA at 41.2 · order 555"
    assert "completion_vplan_id_int" not in sent_list[0]["request_body_dict"]


def test_send_refuses_when_broker_changed(service_factory):
    broker_obj = FakeBroker()
    service_obj, sent_list = service_factory(broker=broker_obj)
    nonce_str = service_obj.check_dict("pod_manual", {"kind_str": "complete", "asset_str": "APX"})["confirmation_nonce_str"]
    broker_obj.open_list = [{"asset_str": "APX", "side_str": "BUY", "remaining_amount_float": 300.0}]
    with pytest.raises(OrderStop, match="already open"):
        service_obj.send_dict("pod_manual", {"confirmation_nonce_str": nonce_str, "broker_order_type_str": "MKT", "quantity_int": 300})
    assert sent_list == []


def test_sender_failure_is_outcome_unknown(service_factory):
    def failing_sender(**keyword_dict):
        raise TimeoutError("socket")
    service_obj, _ = service_factory(sender=failing_sender)
    nonce_str = service_obj.check_dict("pod_manual", {"kind_str": "complete", "asset_str": "APX"})["confirmation_nonce_str"]
    assert service_obj.send_dict("pod_manual", {"confirmation_nonce_str": nonce_str, "broker_order_type_str": "MKT",
        "quantity_int": 300}) == {"status_str": "unknown", "line_str": "Outcome unknown · check TWS before resending"}


def test_already_sent_row_is_not_completed_again(service_factory):
    service_obj, _ = service_factory(missing=_missing_ok([{"asset_str": "APX", "side_str": "BUY", "quantity_int": 300,
        "target_share_float": 300.0, "state_str": "sent", "text_str": "Sent · MOC · waits for close · order 9"}]))
    with pytest.raises(OrderStop, match="Already sent · MOC · waits for close · order 9"):
        service_obj.check_dict("pod_manual", {"kind_str": "complete", "asset_str": "APX"})


# ---------------------------------------------------------------- routes (demo app, synthetic broker)
@pytest.fixture
def demo_client():
    from alpha.live.dashboard_v4.demo import create_demo_app
    app_obj = create_demo_app(demo_tools_bool=True)
    return app_obj.test_client()


def _token(client_obj):
    html_str = client_obj.get("/pods/demo_1_1").get_data(as_text=True)
    return re.search(r'data-token="([^"]+)"', html_str).group(1), html_str


def test_demo_pod_page_shows_missing_trade_and_buttons(demo_client):
    _, html_str = _token(demo_client)
    assert "Missing trades" in html_str and "Missing · no broker ACK" in html_str
    assert 'data-kind="complete" data-asset="MSFT"' in html_str and 'data-kind="new"' in html_str
    assert 'data-order-panel="new"' in html_str and 'data-order-panel="complete"' in html_str and 'hx-preserve="true"' in html_str


def test_demo_routes_check_send_and_guards(demo_client):
    token_str, _ = _token(demo_client)
    header_dict = {"X-Alpha-Action-Token": token_str, "Content-Type": "application/json", "Origin": "http://localhost"}

    def post(path_str, body_dict, headers=header_dict):
        response_obj = demo_client.post(path_str, data=json.dumps({"confirmed_bool": True, **body_dict}), headers=headers)
        return response_obj.status_code, response_obj.get_json()
    status_int, check_dict = post("/api/pods/demo_1_1/order/check", {"kind_str": "complete", "asset_str": "MSFT"})
    assert status_int == 200 and check_dict["missing_str"] == "BUY 17"
    status_int, send_dict = post("/api/pods/demo_1_1/order/send", {"confirmation_nonce_str": check_dict["confirmation_nonce_str"],
        "broker_order_type_str": "MOC", "quantity_int": 17})
    assert send_dict["status_str"] == "sent" and send_dict["line_str"].startswith("Sent · BUY 17 MSFT at close · order DEMO-")
    html_str = demo_client.get("/pods/demo_1_1").get_data(as_text=True)
    assert "Sent · MOC · waits for close" in html_str
    assert post("/api/pods/demo_1_1/order/check", {"kind_str": "new", "asset_str": "MSFT"})[1]["line_str"].startswith("Blocked · MSFT order already open")
    assert post("/api/pods/demo_1_1/order/check", {"kind_str": "new", "asset_str": "AAPL"},
        headers={"Content-Type": "application/json", "Origin": "http://localhost"})[0] == 403
    assert post("/api/pods/demo_1_1/order/check", {"kind_str": "new", "asset_str": "AAPL", "extra": 1})[0] == 400


def test_activity_sentence_for_manual_orders():
    from alpha.live.dashboard_v4.activity import _event_row_dict
    event_dict = {"timestamp_str": "2026-10-16T18:03:00+00:00", "event_type_str": "manual_order_submit_completed",
        "pod_id_str": "pod_manual", "level_str": "WARNING", "payload_dict": {"broker_order_type_str": "MOC", "side_str": "BUY",
            "quantity_int": 300, "asset_str": "APX", "completion_vplan_id_int": 412}}
    assert _event_row_dict(event_dict, {}, [])["title_str"] == "Manual MOC order sent: BUY 300 APX (completion of VPlan #412)."
    event_dict["payload_dict"].pop("completion_vplan_id_int")
    event_dict["event_type_str"] = "manual_order_submit_failed"
    assert _event_row_dict(event_dict, {}, [])["title_str"] == "Manual MOC order failed: BUY 300 APX."


def test_overview_lists_missing_trades(demo_client):
    html_str = demo_client.get("/").get_data(as_text=True)
    assert "1 trade missing" in html_str and "MSFT BUY 17" in html_str


def test_production_stays_read_only_without_flag():
    from alpha.live.dashboard_v4.app import create_app
    from alpha.live.dashboard_v4.demo import build_demo_workspace_tuple
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    app_obj = create_app(provider_obj, workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj),
        operations_workspace_fn=lambda: workspace_dict)
    response_obj = app_obj.test_client().post("/api/pods/demo_1_1/order/check", json={"confirmed_bool": True})
    assert response_obj.status_code == 403
    with pytest.raises(ValueError, match="synthetic broker"):
        create_app(provider_obj, demo_bool=True, orders_enabled_bool=True)
