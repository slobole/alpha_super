"""Production-format saved evidence: no broker and no runtime DB writes."""

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from alpha.live.dashboard import DashboardPodTarget
from alpha.live.dashboard_v4.evidence import load_cycle_evidence_dict
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerOrderFill, BrokerOrderRecord, DecisionPlan, LiveRelease, VPlan, VPlanRow
from alpha.live.state_store_v2 import LiveStateStore


NOW_TS = datetime(2026, 9, 18, 14, 0, tzinfo=UTC)
SUBMIT_TS = NOW_TS - timedelta(minutes=36)
FILL_TS = NOW_TS - timedelta(minutes=30)


def build_fixture_tuple(tmp_path, *, amount_list=None, filled_fraction_float=1.0):
    """Write only pytest's temporary directory using the real persisted models."""
    if amount_list is None:
        amount_list = [10.0, -8.0]
    release_obj = LiveRelease(
        release_id_str="release", user_id_str="owner", pod_id_str="pod", account_route_str="U111",
        strategy_import_str="strategies.example:Example", mode_str="live", session_calendar_id_str="XNYS",
        signal_clock_str="eod_snapshot_ready", execution_policy_str="next_open_moo", data_profile_str="test",
        params_dict={}, risk_profile_str="standard", enabled_bool=True, source_path_str="test.yaml",
    )
    target_obj = DashboardPodTarget(release_obj, str(tmp_path / "pod.sqlite3"), False)
    store_obj = LiveStateStore(target_obj.db_path_str)
    store_obj.upsert_release(release_obj)
    decision_obj = store_obj.insert_decision_plan(DecisionPlan(
        release_id_str="release", user_id_str="owner", pod_id_str="pod", account_route_str="U111",
        signal_timestamp_ts=NOW_TS - timedelta(days=1), submission_timestamp_ts=SUBMIT_TS,
        target_execution_timestamp_ts=FILL_TS, execution_policy_str="next_open_moo",
        decision_base_position_map={}, snapshot_metadata_dict={}, strategy_state_dict={}, status_str="completed",
    ))
    plan_row_list = [VPlanRow("SPY", 0.0, amount_float, amount_float, 100.0, abs(amount_float) * 100, "MOO") for amount_float in amount_list]
    plan_obj = store_obj.insert_vplan(VPlan(
        release_id_str="release", user_id_str="owner", pod_id_str="pod", account_route_str="U111",
        decision_plan_id_int=decision_obj.decision_plan_id_int, signal_timestamp_ts=decision_obj.signal_timestamp_ts,
        submission_timestamp_ts=SUBMIT_TS, target_execution_timestamp_ts=FILL_TS, execution_policy_str="next_open_moo",
        broker_snapshot_timestamp_ts=SUBMIT_TS, live_reference_snapshot_timestamp_ts=SUBMIT_TS,
        live_price_source_str="ibkr", net_liq_float=10000.0, available_funds_float=10000.0,
        excess_liquidity_float=10000.0, pod_budget_fraction_float=1.0, pod_budget_float=10000.0,
        current_broker_position_map={}, live_reference_price_map={"SPY": 100.0}, target_share_map={"SPY": sum(amount_list)},
        order_delta_map={"SPY": sum(amount_list)}, vplan_row_list=plan_row_list, status_str="completed",
    ))
    request_list = build_broker_order_request_list_from_vplan(plan_obj)
    for index_int, request_obj in enumerate(request_list):
        order_id_str = f"order-{index_int}"
        filled_float = request_obj.amount_float * filled_fraction_float
        store_obj.upsert_vplan_broker_order_record_list([BrokerOrderRecord(
            broker_order_id_str=order_id_str, decision_plan_id_int=decision_obj.decision_plan_id_int,
            vplan_id_int=plan_obj.vplan_id_int, account_route_str="U111", asset_str="SPY",
            order_request_key_str=request_obj.order_request_key_str, broker_order_type_str="MOO", unit_str="shares",
            amount_float=request_obj.amount_float, filled_amount_float=abs(filled_float),
            remaining_amount_float=abs(request_obj.amount_float - filled_float), status_str="Filled" if filled_fraction_float == 1 else "Submitted",
            submitted_timestamp_ts=SUBMIT_TS, last_status_timestamp_ts=FILL_TS, submission_key_str=request_obj.submission_key_str,
        )])
        if filled_float:
            store_obj.upsert_vplan_fill_list([BrokerOrderFill(
                broker_order_id_str=order_id_str, decision_plan_id_int=decision_obj.decision_plan_id_int,
                vplan_id_int=plan_obj.vplan_id_int, account_route_str="U111", asset_str="SPY",
                fill_amount_float=filled_float, fill_price_float=100.0, fill_timestamp_ts=FILL_TS,
                raw_payload_dict={"exec_id_str": f"exec-{index_int}"},
            )])
    with sqlite3.connect(target_obj.db_path_str) as connection_obj:
        for table_str in ("decision_plan", "vplan"):
            connection_obj.execute(f"UPDATE {table_str} SET created_timestamp_str = ?, updated_timestamp_str = ?", (SUBMIT_TS.isoformat(), FILL_TS.isoformat()))
    row_dict = {
        "mode_str": "live", "pod_id_str": "pod", "account_route_str": "U111", "latest_vplan_id_int": plan_obj.vplan_id_int,
        "latest_vplan_decision_plan_id_int": decision_obj.decision_plan_id_int, "latest_decision_plan_id_int": decision_obj.decision_plan_id_int,
        "latest_vplan_status_str": "completed", "broker_order_count_int": len(request_list),
        "fill_count_int": len(request_list) if filled_fraction_float else 0,
    }
    return target_obj, row_dict


def update_db(target_obj, sql_str, params_tuple=()):
    with sqlite3.connect(target_obj.db_path_str) as connection_obj:
        connection_obj.execute(sql_str, params_tuple)


def test_complete_signed_fills_for_same_asset_exit_and_entry_are_independent(tmp_path):
    target_obj, row_dict = build_fixture_tuple(tmp_path)
    before_bytes = (tmp_path / "pod.sqlite3").read_bytes()
    before_ns_int = (tmp_path / "pod.sqlite3").stat().st_mtime_ns
    evidence_dict = load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)
    assert evidence_dict["state_str"] == "complete"
    assert evidence_dict["filled_order_count_int"] == evidence_dict["order_count_int"] == 2
    assert evidence_dict["actual_fill_timestamp_str"] == FILL_TS.isoformat()
    assert [order_dict["filled_share_float"] for order_dict in evidence_dict["order_list"]] == [10.0, -8.0]
    assert (tmp_path / "pod.sqlite3").read_bytes() == before_bytes
    assert (tmp_path / "pod.sqlite3").stat().st_mtime_ns == before_ns_int


@pytest.mark.parametrize("field_str,value_obj", [("missing_ack_count_int", 1), ("latest_submit_ack_status_str", "missing_critical")])
def test_no_order_evidence_rejects_ack_failure(tmp_path, field_str, value_obj):
    target_obj, row_dict = build_fixture_tuple(tmp_path, amount_list=[0.0])
    row_dict[field_str] = value_obj
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"


def test_no_order_evidence_rejects_saved_ack_row(tmp_path):
    target_obj, row_dict = build_fixture_tuple(tmp_path, amount_list=[0.0])
    update_db(target_obj, """INSERT INTO vplan_broker_ack
        (vplan_id_int, account_route_str, order_request_key_str, asset_str, broker_order_type_str,
         local_submit_ack_bool, broker_response_ack_bool, ack_status_str, ack_source_str, raw_payload_json_str)
        VALUES (?, 'U111', 'unexpected', 'SPY', 'MOO', 1, 0, 'missing_critical', 'test', '{}')""",
        (row_dict["latest_vplan_id_int"],))
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"


@pytest.mark.parametrize("field_str,value_obj", [("missing_ack_count_int", 1), ("submit_ack_status_str", "missing_critical")])
def test_no_order_evidence_rejects_saved_header_ack_failure(tmp_path, field_str, value_obj):
    target_obj, row_dict = build_fixture_tuple(tmp_path, amount_list=[0.0])
    row_dict.update(missing_ack_count_int=0, latest_submit_ack_status_str="complete")
    update_db(target_obj, f"UPDATE vplan SET {field_str} = ?", (value_obj,))
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"


def test_saved_quantities_reach_overview_through_live_provider(tmp_path, monkeypatch):
    from alpha.live.dashboard_v4.data import LiveDataProvider
    from alpha.live.dashboard_v4.demo import build_demo_workspace_tuple
    from alpha.live.dashboard_v4.overview import build_overview_dict

    target_obj, saved_row_dict = build_fixture_tuple(tmp_path)
    workspace_dict, snapshot_obj, _ = build_demo_workspace_tuple()
    account_dict = workspace_dict["operations_account_list"][0]
    account_dict.update(pod_id="pod", account_route="U111")
    workspace_dict["operations_account_list"] = [account_dict]
    summary_dict = workspace_dict["summary_dict"]
    row_dict = summary_dict["pod_row_dict_list"][0]
    row_dict.update(saved_row_dict)
    row_dict.pop("cycle_evidence_dict")
    row_dict.update(as_of_timestamp_str=NOW_TS.isoformat(), latest_decision_plan_status_str="completed",
        latest_vplan_submission_timestamp_str=SUBMIT_TS.isoformat(), latest_vplan_target_execution_timestamp_str=FILL_TS.isoformat(),
        latest_reconciliation_timestamp_str=(FILL_TS + timedelta(minutes=6)).isoformat(),
        broker_ack_count_int=2, missing_ack_count_int=0)
    row_dict["eod_snapshot_dict"].update(expected_market_date_str="2026-09-18", expected_due_timestamp_str="2026-09-18T20:10:00+00:00")
    summary_dict.update(as_of_timestamp_str=NOW_TS.isoformat(), pod_row_dict_list=[row_dict])
    provider_obj = LiveDataProvider()
    monkeypatch.setattr(provider_obj, "get_target_for_pod", lambda pod_id_str: target_obj)
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_financial_overview_dict", lambda *args, **kwargs: {})
    before_bytes = (tmp_path / "pod.sqlite3").read_bytes()
    view_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=NOW_TS)
    assert view_dict["pod_list"][0]["pill_str"] == "On track"
    assert view_dict["pod_list"][0]["now_detail_str"] == "2 of 2 filled"
    assert view_dict["verdict_str"] == "No action needed."
    assert (tmp_path / "pod.sqlite3").read_bytes() == before_bytes


def test_complete_fill_does_not_require_reconciliation_or_completed_plan(tmp_path):
    target_obj, row_dict = build_fixture_tuple(tmp_path)
    row_dict["latest_vplan_status_str"] = "submitted"
    update_db(target_obj, "UPDATE vplan SET status_str='submitted'")
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "complete"


@pytest.mark.parametrize("filled_fraction_float", [0.0, 0.5])
def test_completed_or_reconciled_plan_never_proves_partial_fill(tmp_path, filled_fraction_float):
    target_obj, row_dict = build_fixture_tuple(tmp_path, filled_fraction_float=filled_fraction_float)
    row_dict["latest_reconciliation_status_str"] = "passed"
    evidence_dict = load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)
    assert evidence_dict["state_str"] == "partial"
    assert evidence_dict["actual_fill_timestamp_str"] is None


@pytest.mark.parametrize("sql_str,params_tuple", [
    ("UPDATE vplan SET pod_id_str='other'", ()),
    ("UPDATE decision_plan SET account_route_str='U999'", ()),
    ("UPDATE live_release SET mode_str='paper'", ()),
    ("UPDATE vplan SET status_str='submitted'", ()),
    ("UPDATE vplan SET updated_timestamp_str=?", ((NOW_TS + timedelta(seconds=1)).isoformat(),)),
    ("UPDATE vplan_broker_order SET account_route_str='U999'", ()),
    ("UPDATE vplan_broker_order SET decision_plan_id_int=999", ()),
    ("UPDATE vplan_broker_order SET unit_str='value'", ()),
    ("ALTER TABLE vplan_broker_order RENAME COLUMN unit_str TO legacy_unit_str", ()),
    ("UPDATE vplan_broker_order SET amount_float=99", ()),
    ("UPDATE vplan_broker_order SET order_request_key_str='wrong'", ()),
    ("UPDATE vplan_broker_order SET last_status_timestamp_str=?", ((NOW_TS + timedelta(seconds=1)).isoformat(),)),
    ("UPDATE vplan_fill SET account_route_str='U999'", ()),
    ("UPDATE vplan_fill SET decision_plan_id_int=999", ()),
    ("UPDATE vplan_fill SET broker_order_id_str='orphan'", ()),
    ("UPDATE vplan_fill SET asset_str='QQQ'", ()),
    ("UPDATE vplan_fill SET fill_amount_float=-fill_amount_float", ()),
    ("UPDATE vplan_fill SET fill_amount_float=fill_amount_float*2", ()),
    ("UPDATE vplan_fill SET fill_timestamp_str=?", ((NOW_TS + timedelta(seconds=1)).isoformat(),)),
    ("UPDATE vplan_fill SET fill_timestamp_str=?", ((SUBMIT_TS - timedelta(seconds=1)).isoformat(),)),
    ("UPDATE vplan_fill SET raw_payload_json_str='[]'", ()),
    ("UPDATE vplan_fill SET raw_payload_json_str='{\"exec_id_str\":\"repeated\"}'", ()),
    ("UPDATE vplan SET order_delta_json_str='null'", ()),
    ("UPDATE vplan SET order_delta_json_str='{\"SPY\":NaN}'", ()),
    ("UPDATE vplan SET order_delta_json_str='{\"SPY\":2,\"QQQ\":10}'", ()),
    ("UPDATE vplan SET order_delta_json_str='{}'", ()),
    ("DROP TABLE vplan_fill", ()),
])
def test_invalid_or_mismatched_evidence_stays_unknown(tmp_path, sql_str, params_tuple):
    target_obj, row_dict = build_fixture_tuple(tmp_path)
    update_db(target_obj, sql_str, params_tuple)
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"


def test_missing_or_changed_order_count_cannot_hide_unfilled_request(tmp_path):
    target_obj, row_dict = build_fixture_tuple(tmp_path)
    update_db(target_obj, "DELETE FROM vplan_broker_order WHERE broker_order_id_str='order-1'")
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"
    row_dict["broker_order_count_int"] = 1
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"


def test_nonlive_and_wrong_summary_identity_are_rejected_before_open(tmp_path, monkeypatch):
    target_obj, row_dict = build_fixture_tuple(tmp_path)
    monkeypatch.setattr("alpha.live.dashboard_v4.evidence.sqlite3.connect", lambda *arg_list, **kwarg_dict: pytest.fail("Should not open any database"))
    for mode_str in ("paper", "incubation"):
        wrong_target_obj = replace(target_obj, release_obj=replace(target_obj.release_obj, mode_str=mode_str))
        assert load_cycle_evidence_dict(wrong_target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"
    assert load_cycle_evidence_dict(target_obj, {**row_dict, "account_route_str": "U999"}, as_of_ts=NOW_TS)["state_str"] == "unknown"


def test_missing_file_is_not_created(tmp_path):
    target_obj, row_dict = build_fixture_tuple(tmp_path)
    missing_path_obj = tmp_path / "missing.sqlite3"
    target_obj = replace(target_obj, db_path_str=str(missing_path_obj))
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"
    assert not missing_path_obj.exists()


@pytest.mark.parametrize("amount_list", [[], [0.0]])
def test_no_orders_requires_explicit_zero_intent_and_no_orders_or_fills(tmp_path, amount_list):
    target_obj, row_dict = build_fixture_tuple(tmp_path, amount_list=amount_list)
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "no_orders"
    update_db(target_obj, "UPDATE vplan SET order_delta_json_str='{\"SPY\":1}'")
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"


def test_net_zero_does_not_skip_two_real_legs(tmp_path):
    target_obj, row_dict = build_fixture_tuple(tmp_path, amount_list=[10.0, -10.0], filled_fraction_float=0.5)
    evidence_dict = load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)
    assert evidence_dict["state_str"] == "partial"
    assert evidence_dict["order_count_int"] == 2


def test_zero_plan_with_persisted_orders_is_not_no_orders(tmp_path):
    target_obj, row_dict = build_fixture_tuple(tmp_path)
    update_db(target_obj, "UPDATE vplan_row SET order_delta_share_float=0")
    update_db(target_obj, "UPDATE vplan SET order_delta_json_str='{}'")
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"


def test_multiple_executions_sum_once_per_order(tmp_path):
    target_obj, row_dict = build_fixture_tuple(tmp_path, amount_list=[10.0], filled_fraction_float=0.5)
    with sqlite3.connect(target_obj.db_path_str) as connection_obj:
        connection_obj.execute(
            "INSERT INTO vplan_fill (broker_order_id_str, decision_plan_id_int, vplan_id_int, account_route_str, asset_str, fill_amount_float, fill_price_float, fill_timestamp_str, raw_payload_json_str) "
            "VALUES ('order-0', 1, 1, 'U111', 'SPY', 5, 100, ?, '{\"exec_id_str\":\"exec-second\"}')",
            ((FILL_TS + timedelta(seconds=1)).isoformat(),),
        )
    row_dict["fill_count_int"] = 2
    evidence_dict = load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)
    assert evidence_dict["state_str"] == "complete"
    assert evidence_dict["order_list"][0]["filled_share_float"] == 10.0
    assert evidence_dict["actual_fill_timestamp_str"] == (FILL_TS + timedelta(seconds=1)).isoformat()
    update_db(target_obj, "UPDATE vplan_fill SET raw_payload_json_str='{\"exec_id_str\":\"exec-0\"}'")
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"


def test_retry_or_duplicate_request_never_counts_as_two_completed_requests(tmp_path):
    target_obj, row_dict = build_fixture_tuple(tmp_path)
    update_db(target_obj, "UPDATE vplan_broker_order SET order_request_key_str='vplan:1:SPY:1'")
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"


def test_duplicate_order_id_in_legacy_store_is_ambiguous(tmp_path):
    target_obj, row_dict = build_fixture_tuple(tmp_path)
    update_db(target_obj, "DROP INDEX vplan_broker_order_unique_order_idx")
    update_db(target_obj, "UPDATE vplan_broker_order SET broker_order_id_str='reused'")
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"


def test_reader_uses_readonly_uri_and_one_read_transaction(tmp_path, monkeypatch):
    target_obj, row_dict = build_fixture_tuple(tmp_path)
    connect_fn = sqlite3.connect
    statement_list = []

    def connect_readonly_fn(uri_str, **option_dict):
        assert uri_str.endswith("?mode=ro")
        assert option_dict["uri"] is True
        connection_obj = connect_fn(uri_str, **option_dict)
        connection_obj.set_trace_callback(statement_list.append)
        return connection_obj

    monkeypatch.setattr("alpha.live.dashboard_v4.evidence.sqlite3.connect", connect_readonly_fn)
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "complete"
    assert statement_list[0] == "BEGIN"
    assert all(statement_str.startswith("SELECT") for statement_str in statement_list[1:])


@pytest.mark.parametrize("delta_json_str", ["", "null", "[]", '{"SPY":NaN}', '{"SPY":Infinity}', '{"SPY":false}'])
def test_no_orders_rejects_unverifiable_aggregate_intent(tmp_path, delta_json_str):
    target_obj, row_dict = build_fixture_tuple(tmp_path, amount_list=[])
    update_db(target_obj, "UPDATE vplan SET order_delta_json_str=?", (delta_json_str,))
    assert load_cycle_evidence_dict(target_obj, row_dict, as_of_ts=NOW_TS)["state_str"] == "unknown"
