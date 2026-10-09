"""Legacy supplemental fills retain reporting lineage after the daily upgrade."""
from dataclasses import asdict, replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from alpha.live import runner
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerOrderFill, BrokerOrderRecord, SessionOpenPrice
from test_live_daily_reconcile import CLOSE_TS, daily_case


def _legacy_request(store_obj, plan_obj, kind_str="manual", mutation_dict=None):
    request_obj = replace(build_broker_order_request_list_from_vplan(plan_obj)[0],
        asset_str="MSFT", amount_float=-3.0, broker_order_type_str="MKT",
        order_request_key_str=f"{plan_obj.submission_key_str}:manual:manual-77")
    request_dict = {**asdict(request_obj), **(mutation_dict or {})}
    with store_obj._connect() as connection_obj:
        connection_obj.execute("""CREATE TABLE IF NOT EXISTS mr_capsule_execution_request (
            vplan_id_int INTEGER, order_request_key_str TEXT, asset_str TEXT, request_kind_str TEXT,
            request_json_str TEXT, claimed_timestamp_str TEXT, PRIMARY KEY(vplan_id_int,order_request_key_str))""")
        connection_obj.execute("INSERT INTO mr_capsule_execution_request VALUES (?,?,?,?,?,?)",
            (plan_obj.vplan_id_int, request_obj.order_request_key_str, "MSFT", kind_str,
                json.dumps(request_dict), (CLOSE_TS - timedelta(hours=1)).isoformat()))
    return request_obj


def _record(release_obj, decision_obj, plan_obj, request_key_str):
    return BrokerOrderRecord("manual-77", decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        release_obj.account_route_str, "MSFT", request_key_str, "MKT", "shares", -3.0, -3.0,
        "Filled", CLOSE_TS - timedelta(hours=1), remaining_amount_float=0.0, avg_fill_price_float=101.0)


def _fill(release_obj, decision_obj, plan_obj):
    return BrokerOrderFill("manual-77", decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        release_obj.account_route_str, "MSFT", -3.0, 101.0, CLOSE_TS - timedelta(hours=1))


def _open_price_list(release_obj):
    return [SessionOpenPrice("2026-10-05", release_obj.account_route_str, "MSFT", 100.0, "fixture_open", CLOSE_TS)]


@pytest.mark.parametrize("kind_str", ["manual", "recovery"])
def test_saved_supplemental_id_keeps_late_label_during_sparse_refresh(daily_case, tmp_path, kind_str):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    request_obj = _legacy_request(store_obj, plan_obj, kind_str)
    record_obj = _record(release_obj, decision_obj, plan_obj, request_obj.order_request_key_str)
    store_obj.upsert_vplan_broker_order_record_list([record_obj])
    broker_obj = SimpleNamespace(get_session_open_price_list=lambda **_kwarg_dict: _open_price_list(release_obj))
    runner._persist_daily_execution_report(store_obj, broker_obj, release_obj, decision_obj, plan_obj,
        [replace(record_obj, order_request_key_str=None)], [], [_fill(release_obj, decision_obj, plan_obj)],
        CLOSE_TS, str(tmp_path / "report.log"))
    fill_dict, = store_obj.get_fill_row_dict_list_for_vplan(plan_obj.vplan_id_int)
    with store_obj._connect() as connection_obj:
        payload_str, = connection_obj.execute("SELECT raw_payload_json_str FROM vplan_fill WHERE vplan_id_int=?",
            (plan_obj.vplan_id_int,)).fetchone()
    assert fill_dict["official_open_price_float"] is None
    assert fill_dict["open_price_source_str"] == "late_execution"
    assert json.loads(payload_str)["execution_phase_str"] == "late_execution"
    assert store_obj.get_broker_order_row_dict_list_for_vplan(plan_obj.vplan_id_int)[0]["order_request_key_str"] == request_obj.order_request_key_str


def test_adoption_commit_before_broker_record_fetches_exact_id_and_retains_late_label(daily_case, tmp_path):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    request_obj = _legacy_request(store_obj, plan_obj)
    broker_obj.as_of_ts = CLOSE_TS
    broker_obj.position_dict.update(plan_obj.target_share_map)
    def reporting_fn(**kwarg_dict):
        assert kwarg_dict["account_route_str"] == release_obj.account_route_str
        assert kwarg_dict["allowed_broker_order_id_set"] == {"manual-77"}
        return [_record(release_obj, decision_obj, plan_obj, None)], [], [_fill(release_obj, decision_obj, plan_obj)]
    broker_obj.get_recent_order_state_snapshot = reporting_fn
    broker_obj.get_session_open_price_list = lambda **_kwarg_dict: _open_price_list(release_obj)
    assert not store_obj.get_broker_order_row_dict_list_for_vplan(plan_obj.vplan_id_int)
    result_int = runner._reconcile_daily_cycles(store_obj, SimpleNamespace(get_adapter=lambda _release_obj: broker_obj),
        CLOSE_TS, "paper", None, str(tmp_path / "report.log"), False, str(tmp_path / "traces"))
    assert result_int == 1
    assert store_obj.get_broker_order_row_dict_list_for_vplan(plan_obj.vplan_id_int)[0]["order_request_key_str"] == request_obj.order_request_key_str
    assert store_obj.get_fill_row_dict_list_for_vplan(plan_obj.vplan_id_int)[0]["open_price_source_str"] == "late_execution"
    assert broker_obj.sent_request_list == []


@pytest.mark.parametrize("field_str", ["account_route_str", "pod_id_str", "decision_plan_id_int", "vplan_id_int"])
def test_conflicting_legacy_request_identity_cannot_attribute_reporting(daily_case, tmp_path, field_str):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    _legacy_request(store_obj, plan_obj, mutation_dict={field_str: "wrong"})
    with pytest.raises(ValueError, match="conflicting cycle identity"):
        runner._persist_daily_execution_report(store_obj, SimpleNamespace(), release_obj, decision_obj,
            plan_obj, [], [], [_fill(release_obj, decision_obj, plan_obj)], CLOSE_TS, str(tmp_path / "report.log"))
    assert store_obj.get_fill_row_dict_list_for_vplan(plan_obj.vplan_id_int) == []
