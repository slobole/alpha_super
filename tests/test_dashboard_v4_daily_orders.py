"""Daily pods: same-day exit completions in the fill proof and the Orders tab, plus --demo-daily.

Synthetic demo state only, written through the production writers; no broker or real DB.
"""
from contextlib import closing
from datetime import timedelta
import re
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.daily_demo import CURRENT_TARGET_TS, DEMO_DAILY_POD_TUPLE, DEMO_NOW_TS, DemoDailyStore
from alpha.live.dashboard_v4.evidence import load_cycle_evidence_dict
from alpha.live.dashboard_v4.pod import build_daily_completion_order_row_list
from alpha.live.dashboard_v4.pod_data import load_pod_cycles_dict
from alpha.live.dashboard_v4.pod_demo import _DemoStateStore
from alpha.live.models import BrokerOrderAck, BrokerOrderFill, BrokerOrderRecord


COMPLETION_SUBMIT_TS = CURRENT_TARGET_TS + timedelta(minutes=6)


@pytest.fixture(autouse=True)
def healthy_host_disk(monkeypatch):
    monkeypatch.setattr("alpha.live.dashboard_v3.health.shutil.disk_usage",
        lambda path_str: SimpleNamespace(total=100, used=50, free=50))


@pytest.fixture()
def trading_case():
    """The demo CORE5 pod trading today, with one claimed DBC exit completion (MKT SELL 25)."""
    store_obj = DemoDailyStore()
    try:
        yield store_obj, store_obj.get_target_for_pod("demo_daily_trading")
    finally:
        store_obj.close()


def _record_completion(store_obj, target_obj, *, amount_float=-25.0, order_type_str="MKT"):
    """What the daily close saves right after sending a completion: order, ACK and fill."""
    with closing(sqlite3.connect(target_obj.db_path_str)) as connection_obj:
        vplan_id_int, decision_id_int, submission_key_str = connection_obj.execute(
            "SELECT vplan_id_int, decision_plan_id_int, submission_key_str FROM vplan ORDER BY vplan_id_int DESC LIMIT 1").fetchone()
    field_dict = {"broker_order_id_str": "DEMO-C-1", "decision_plan_id_int": decision_id_int, "vplan_id_int": vplan_id_int,
        "account_route_str": target_obj.release_obj.account_route_str, "asset_str": "DBC",
        "order_request_key_str": f"{submission_key_str}:daily-completion:DBC"}
    with closing(_DemoStateStore(target_obj.db_path_str)) as writer_obj:
        writer_obj.upsert_vplan_broker_order_record_list([BrokerOrderRecord(**field_dict, broker_order_type_str=order_type_str,
            unit_str="shares", amount_float=amount_float, filled_amount_float=abs(amount_float), remaining_amount_float=0.0,
            avg_fill_price_float=23.05, status_str="Filled", submitted_timestamp_ts=COMPLETION_SUBMIT_TS,
            last_status_timestamp_ts=COMPLETION_SUBMIT_TS + timedelta(seconds=4), submission_key_str=submission_key_str)])
        writer_obj.upsert_vplan_broker_ack_list([BrokerOrderAck(**field_dict, broker_order_type_str=order_type_str,
            local_submit_ack_bool=True, broker_response_ack_bool=True, ack_status_str="broker_acked",
            ack_source_str="open_order", response_timestamp_ts=COMPLETION_SUBMIT_TS + timedelta(seconds=1))])
        writer_obj.upsert_vplan_fill_list([BrokerOrderFill(broker_order_id_str="DEMO-C-1", decision_plan_id_int=decision_id_int,
            vplan_id_int=vplan_id_int, account_route_str=target_obj.release_obj.account_route_str, asset_str="DBC",
            fill_amount_float=amount_float, fill_price_float=23.05, fill_timestamp_ts=COMPLETION_SUBMIT_TS + timedelta(seconds=4),
            raw_payload_dict={"exec_id_str": "DEMO-exec-C-1"})])
    store_obj._freeze_timestamps(target_obj.db_path_str)  # Writers stamp wall time; pin it to the demo clock.


def _source_and_daily(store_obj, target_obj):
    source_dict = load_pod_cycles_dict(target_obj, as_of_ts=DEMO_NOW_TS)
    assert source_dict["status_str"] == "ok"
    source_dict["cycle_evidence_dict"] = load_cycle_evidence_dict(target_obj, source_dict["pod_row_dict"], as_of_ts=DEMO_NOW_TS)
    daily_dict = store_obj.get_daily_cycle_dict({"pod_id_str": target_obj.release_obj.pod_id_str,
        "account_route_str": target_obj.release_obj.account_route_str}, as_of_ts=DEMO_NOW_TS)
    return source_dict, daily_dict


def _cells(row_list):
    return [row_dict["cell_list"] for row_dict in row_list]


def test_a_claim_without_a_broker_record_is_not_confirmed(trading_case):
    store_obj, target_obj = trading_case
    source_dict, daily_dict = _source_and_daily(store_obj, target_obj)
    assert [(item_dict["asset_str"], item_dict["amount_float"]) for item_dict in daily_dict["completion_request_list"]] == [("DBC", -25.0)]
    assert source_dict["cycle_evidence_dict"]["state_str"] == "complete"
    row_list = build_daily_completion_order_row_list(source_dict, daily_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)
    assert _cells(row_list) == [["DBC", "SELL 25 · same-day MKT", "Not confirmed", "—", "—", "—", "—", "—"]]
    assert row_list[0]["state_str"] == ""


def test_a_saved_completion_shows_its_ack_and_fill_and_keeps_the_opening_proof(trading_case):
    store_obj, target_obj = trading_case
    _record_completion(store_obj, target_obj)
    source_dict, daily_dict = _source_and_daily(store_obj, target_obj)
    evidence_dict = source_dict["cycle_evidence_dict"]
    # The opening batch's proof is unchanged; the completion is proven on its own.
    assert (evidence_dict["state_str"], evidence_dict["filled_order_count_int"], evidence_dict["order_count_int"]) == ("complete", 5, 5)
    completion_list = [item_dict for item_dict in evidence_dict["order_list"] if item_dict.get("completion_bool")]
    assert [(item_dict["asset_str"], item_dict["filled_share_float"]) for item_dict in completion_list] == [("DBC", -25.0)]
    row_list = build_daily_completion_order_row_list(source_dict, daily_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)
    assert _cells(row_list) == [["DBC", "SELL 25 · same-day MKT", "Acked", "open_order", "09:36:01", "25", "23.05", "DEMO-C-1"]]
    # A stale page never claims a fresh ACK.
    stale_row_list = build_daily_completion_order_row_list(source_dict, daily_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=False)
    assert _cells(stale_row_list)[0][2] == "Unknown"


@pytest.mark.parametrize("amount_float,order_type_str", [(25.0, "MKT"), (-25.0, "MOO")])
def test_a_completion_that_is_not_an_exit_sale_makes_the_fill_proof_unknown(trading_case, amount_float, order_type_str):
    store_obj, target_obj = trading_case
    _record_completion(store_obj, target_obj, amount_float=amount_float, order_type_str=order_type_str)
    source_dict, _ = _source_and_daily(store_obj, target_obj)
    assert source_dict["cycle_evidence_dict"]["state_str"] == "unknown"
    assert source_dict["cycle_evidence_dict"]["reason_str"] == "A saved same-day completion order is not a valid exit sale."


def test_a_failed_completion_send_is_named(trading_case):
    store_obj, target_obj = trading_case
    with closing(sqlite3.connect(target_obj.db_path_str)) as connection_obj, connection_obj:
        connection_obj.execute("UPDATE daily_completion_request SET send_error_str='DEMO send failed'")
    source_dict, daily_dict = _source_and_daily(store_obj, target_obj)
    row_list = build_daily_completion_order_row_list(source_dict, daily_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)
    assert _cells(row_list)[0][2] == "Send failed" and row_list[0]["state_str"] == "fail"
    # The saved error text itself is never shown.
    assert "DEMO send failed" not in str(row_list)


def _page_text(client_obj, path_str):
    response_obj = client_obj.get(path_str)
    assert response_obj.status_code == 200, path_str
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", response_obj.get_data(as_text=True)))


def test_demo_daily_adds_the_daily_pods_beside_the_monthly_ones():
    from alpha.live.dashboard_v4.demo import create_demo_app

    client_obj = create_demo_app(demo_daily_bool=True).test_client()
    overview_str = _page_text(client_obj, "/")
    assert all(name_str in overview_str for name_str in ("DVO2", "CORE5 · Trading", "Capsule · Last close exceptions",
        "CORE5 · Not sent", "Capsule · Unconfirmed", "CORE5 · Open after close"))
    # Daily demo pods have no account report: the money tiles keep the monthly scope.
    assert "Client value $96,602.02" in overview_str
    for pod_id_str, *_ in DEMO_DAILY_POD_TUPLE:
        _page_text(client_obj, f"/pods/{pod_id_str}")
    orders_str = _page_text(client_obj, "/pods/demo_daily_trading?tab=orders")
    assert "SELL 25 · same-day MKT" in orders_str and "Not confirmed" in orders_str
    assert "Check the broker; it is never resent." in orders_str
    # The default demo has no daily pods.
    assert create_demo_app().test_client().get("/pods/demo_daily_trading").status_code == 404


def test_demo_daily_requires_demo(monkeypatch):
    from alpha.live.dashboard_v4 import __main__ as main_module

    monkeypatch.setattr(sys, "argv", ["dashboard_v4", "--demo-daily"])
    with pytest.raises(SystemExit):
        main_module.main()


def test_demo_daily_system_health_counts_the_undelivered_daily_alerts():
    from alpha.live.dashboard_v4.demo import create_demo_app

    # The capsule's close-with-exceptions alert and the overdue pod's alert; the demo never sends.
    system_str = _page_text(create_demo_app(demo_daily_bool=True).test_client(), "/system")
    assert "Daily alerts 2 daily alerts not delivered" in system_str
    assert "Daily alerts" not in _page_text(create_demo_app().test_client(), "/system")
