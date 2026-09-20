"""The preview exercises real read-only adapters over owned synthetic state."""

from contextlib import closing
from copy import deepcopy
from datetime import timedelta
import gc
import json
from pathlib import Path
import sqlite3
import weakref

import pytest

from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple, create_demo_app
from alpha.live.dashboard_v4.pod_demo import DemoPodStore


@pytest.fixture
def demo_fixture_tuple():
    result_tuple = build_demo_workspace_tuple()
    try:
        yield result_tuple
    finally:
        result_tuple[2].close()


def _files_dict(provider_obj):
    return {path_obj.name: (path_obj.read_bytes(), path_obj.stat().st_mtime_ns)
            for path_obj in provider_obj.pod_store_obj.directory_path_obj.iterdir()}


def test_every_demo_cycle_is_real_scoped_saved_evidence(demo_fixture_tuple):
    workspace_dict, _, provider_obj = demo_fixture_tuple
    target_list = provider_obj.get_target_list()
    assert len(target_list) == 4
    assert len({target_obj.db_path_str for target_obj in target_list}) == 4
    for row_dict in workspace_dict["summary_dict"]["pod_row_dict_list"]:
        target_obj = provider_obj.get_target_for_pod(row_dict["pod_id_str"])
        assert Path(target_obj.db_path_str).parent == provider_obj.pod_store_obj.directory_path_obj
        assert target_obj.release_obj.source_path_str == "DEMO-only"
        assert target_obj.release_obj.auto_submit_enabled_bool is False
        for cycle_int in (1, 2):
            source_dict = provider_obj.get_pod_cycles_dict(row_dict["pod_id_str"], as_of_ts=DEMO_NOW_TS, vplan_id_int=cycle_int)
            assert source_dict["status_str"] == "ok"
            assert source_dict["selected_cycle_dict"]["vplan_id_int"] == cycle_int
            assert [cycle_dict["vplan_id_int"] for cycle_dict in source_dict["cycle_list"]] == [2, 1]
            assert len(source_dict["order_list"]) == len(source_dict["ack_list"]) == 3
            assert source_dict["cycle_evidence_dict"]["state_str"] == ("partial" if row_dict["strategy_name_str"] == "QPI" and cycle_int == 2 else "complete")
            assert source_dict["event_list"]
            assert all(item_dict["vplan_id_int"] == cycle_int and item_dict["decision_plan_id_int"] == cycle_int
                       and item_dict["account_route_str"] == row_dict["account_route_str"]
                       and item_dict["event_timestamp_str"] for item_dict in source_dict["event_list"])
            assert all("event_type_str" not in item_dict for item_dict in source_dict["event_list"])
            assert "raw_payload_json_str" not in str(source_dict)


def test_polls_call_both_production_readers_without_rebuilding_or_writing(demo_fixture_tuple, monkeypatch):
    _, _, provider_obj = demo_fixture_tuple
    from alpha.live.dashboard_v4 import pod_demo

    before_dict = _files_dict(provider_obj)
    calls_dict = {"cycles": 0, "fills": 0}
    cycle_fn, evidence_fn = pod_demo.load_pod_cycles_dict, pod_demo.load_cycle_evidence_dict

    def cycle_reader(*args, **kwargs):
        calls_dict["cycles"] += 1
        return cycle_fn(*args, **kwargs)

    def fill_reader(*args, **kwargs):
        calls_dict["fills"] += 1
        return evidence_fn(*args, **kwargs)

    monkeypatch.setattr(pod_demo, "load_pod_cycles_dict", cycle_reader)
    monkeypatch.setattr(pod_demo, "load_cycle_evidence_dict", fill_reader)
    monkeypatch.setattr(pod_demo._DemoStateStore, "__init__", lambda *args, **kwargs: pytest.fail("Writer created during a read"))
    for _ in range(3):
        for target_obj in provider_obj.get_target_list():
            assert provider_obj.get_pod_cycles_dict(target_obj.release_obj.pod_id_str, as_of_ts=DEMO_NOW_TS)["status_str"] == "ok"
    assert calls_dict == {"cycles": 12, "fills": 12}
    assert _files_dict(provider_obj) == before_dict


def test_saved_fill_removal_changes_demo_result(demo_fixture_tuple):
    workspace_dict, _, provider_obj = demo_fixture_tuple
    pod_id_str = workspace_dict["operations_account_list"][0]["pod_id"]
    target_obj = provider_obj.get_target_for_pod(pod_id_str)
    with closing(sqlite3.connect(target_obj.db_path_str)) as connection_obj, connection_obj:
        connection_obj.execute("DELETE FROM vplan_fill WHERE fill_record_id_int=(SELECT MAX(fill_record_id_int) FROM vplan_fill WHERE vplan_id_int=2)")
    source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert source_dict["cycle_evidence_dict"]["state_str"] == "partial"
    assert source_dict["cycle_evidence_dict"]["filled_order_count_int"] == 2
    assert source_dict["cycle_evidence_dict"]["actual_fill_timestamp_str"] is None


def test_missing_ack_is_a_real_unacknowledged_record(demo_fixture_tuple):
    workspace_dict, _, provider_obj = demo_fixture_tuple
    pod_id_str = workspace_dict["operations_account_list"][1]["pod_id"]
    source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert source_dict["pod_row_dict"]["broker_ack_count_int"] == 2
    assert source_dict["pod_row_dict"]["missing_ack_count_int"] == 1
    assert len(source_dict["ack_list"]) == 3
    missing_list = [item_dict for item_dict in source_dict["ack_list"] if item_dict["ack_status_str"] == "missing_critical"]
    assert len(missing_list) == 1
    assert missing_list[0]["broker_response_ack_bool"] == 0
    assert source_dict["reconciliation_dict"] == {}


def test_summary_holdings_and_times_match_actual_saved_pod_state(demo_fixture_tuple):
    workspace_dict, _, provider_obj = demo_fixture_tuple

    for row_dict in workspace_dict["summary_dict"]["pod_row_dict_list"]:
        target_obj = provider_obj.get_target_for_pod(row_dict["pod_id_str"])
        with closing(sqlite3.connect(Path(target_obj.db_path_str).as_uri() + "?mode=ro", uri=True)) as connection_obj:
            position_json_str, timestamp_str = connection_obj.execute("SELECT position_json_str,updated_timestamp_str FROM pod_state WHERE pod_id_str=?", (row_dict["pod_id_str"],)).fetchone()
        assert json.loads(position_json_str) == {position_dict["asset_str"]: position_dict["share_float"] for position_dict in row_dict["position_exposure_dict_list"]}
        assert timestamp_str == row_dict["latest_pod_state_timestamp_str"]
    assert {row_dict["asset_str"] for row_dict in workspace_dict["summary_dict"]["pod_row_dict_list"][0]["position_exposure_dict_list"]} == {"AMD", "CRM", "SGOV"}


def test_cycle_cash_matches_before_after_positions_and_latest_state(demo_fixture_tuple):
    provider_obj = demo_fixture_tuple[2]
    for target_obj in provider_obj.get_target_list():
        with closing(sqlite3.connect(Path(target_obj.db_path_str).as_uri() + "?mode=ro", uri=True)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            for plan_obj in connection_obj.execute("SELECT * FROM vplan"):
                price_dict = json.loads(plan_obj["live_reference_price_json_str"])
                before_dict = json.loads(plan_obj["current_broker_position_json_str"])
                after_dict = json.loads(plan_obj["target_share_json_str"])
                expected_before_float = plan_obj["net_liq_float"] - sum(amount_float * price_dict[asset_str] for asset_str, amount_float in before_dict.items())
                expected_after_float = plan_obj["net_liq_float"] - sum(amount_float * price_dict[asset_str] for asset_str, amount_float in after_dict.items())
                assert plan_obj["available_funds_float"] == pytest.approx(expected_before_float)
                reconcile_obj = connection_obj.execute("SELECT * FROM vplan_reconciliation_snapshot WHERE vplan_id_int=?", (plan_obj["vplan_id_int"],)).fetchone()
                if reconcile_obj is not None:
                    assert reconcile_obj["model_cash_float"] == pytest.approx(expected_after_float)
                    assert reconcile_obj["broker_cash_float"] == pytest.approx(expected_after_float)
            latest_obj = connection_obj.execute("SELECT * FROM pod_state").fetchone()
            latest_position_dict = json.loads(latest_obj["position_json_str"])
            assert latest_obj["cash_float"] + sum(amount_float * price_dict[asset_str] for asset_str, amount_float in latest_position_dict.items()) == pytest.approx(latest_obj["total_value_float"])


def test_real_identity_rejected_before_creating_a_temp_directory(demo_fixture_tuple, monkeypatch):
    row_list = deepcopy(demo_fixture_tuple[0]["summary_dict"]["pod_row_dict_list"])
    row_list[0]["account_route_str"] = "U12345"
    monkeypatch.setattr("alpha.live.dashboard_v4.pod_demo.TemporaryDirectory", lambda **kwargs: pytest.fail("Created a directory for non-demo identity"))
    with pytest.raises(ValueError, match="Synthetic DEMO"):
        DemoPodStore(row_list, as_of_ts=DEMO_NOW_TS)


def test_unknown_identity_or_cycle_never_opens_another_source(demo_fixture_tuple, monkeypatch):
    provider_obj = demo_fixture_tuple[2]
    monkeypatch.setattr("alpha.live.dashboard_v4.pod_demo.load_pod_cycles_dict", lambda *args, **kwargs: pytest.fail("Unscoped source read"))
    assert provider_obj.get_pod_cycles_dict("real_pod", as_of_ts=DEMO_NOW_TS)["status_str"] == "not_found"
    assert provider_obj.get_target_for_pod("real_pod") is None


def test_fixture_setup_and_reads_never_connect_to_network(monkeypatch):
    import socket

    monkeypatch.setattr(socket.socket, "connect", lambda *args, **kwargs: pytest.fail("Network connection attempted"))
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: pytest.fail("Network connection attempted"))
    _, _, provider_obj = build_demo_workspace_tuple()
    try:
        for target_obj in provider_obj.get_target_list():
            assert provider_obj.get_pod_cycles_dict(target_obj.release_obj.pod_id_str, as_of_ts=DEMO_NOW_TS)["status_str"] == "ok"
    finally:
        provider_obj.close()


def test_provider_lifetime_cleans_owned_directory_without_gc_locked_connections():
    _, _, provider_obj = build_demo_workspace_tuple()
    directory_path_obj = provider_obj.pod_store_obj.directory_path_obj
    provider_ref = weakref.ref(provider_obj)
    del provider_obj
    gc.collect()
    assert provider_ref() is None
    assert not directory_path_obj.exists()


def test_setup_failure_closes_real_writer_connections_and_cleans_directory(demo_fixture_tuple, monkeypatch):
    from alpha.live.dashboard_v4 import pod_demo

    directory_list = []
    temporary_fn, seed_fn = pod_demo.TemporaryDirectory, DemoPodStore._seed_cycle

    def temporary_directory(**kwargs):
        temporary_obj = temporary_fn(**kwargs)
        directory_list.append(Path(temporary_obj.name))
        return temporary_obj

    def failing_seed(*args, **kwargs):
        seed_fn(*args, **kwargs)
        raise RuntimeError("Synthetic setup failure")

    monkeypatch.setattr(pod_demo, "TemporaryDirectory", temporary_directory)
    monkeypatch.setattr(DemoPodStore, "_seed_cycle", staticmethod(failing_seed))
    with pytest.raises(RuntimeError, match="Synthetic setup failure"):
        DemoPodStore(deepcopy(demo_fixture_tuple[2].row_list), as_of_ts=DEMO_NOW_TS)
    assert len(directory_list) == 1
    assert not directory_list[0].exists()


def test_explicit_close_is_idempotent_and_cannot_recreate_state():
    _, _, provider_obj = build_demo_workspace_tuple()
    directory_path_obj = provider_obj.pod_store_obj.directory_path_obj
    pod_id_str = provider_obj.get_target_list()[0].release_obj.pod_id_str
    provider_obj.close()
    provider_obj.close()
    assert not directory_path_obj.exists()
    assert provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)["status_str"] == "unknown"
    assert not directory_path_obj.exists()


def test_demo_app_clock_advances_but_saved_artifact_times_are_fixed(monkeypatch):
    from alpha.live.dashboard_v4 import app as app_module

    option_dict = {}
    monotonic_list = [100.0]
    monkeypatch.setattr("alpha.live.dashboard_v4.demo.monotonic", lambda: monotonic_list[0])

    def capture_app(provider_obj, **kwargs):
        option_dict.update(provider_obj=provider_obj, **kwargs)
        return object()

    monkeypatch.setattr(app_module, "create_app", capture_app)
    create_demo_app()
    try:
        first_dict, _ = option_dict["workspace_snapshot_fn"]()
        monotonic_list[0] += 31
        second_dict, _ = option_dict["workspace_snapshot_fn"]()
        assert option_dict["now_fn"]() == DEMO_NOW_TS + timedelta(seconds=31)
        assert second_dict["summary_dict"]["as_of_timestamp_str"] != first_dict["summary_dict"]["as_of_timestamp_str"]
        for first_row_dict, second_row_dict in zip(first_dict["summary_dict"]["pod_row_dict_list"], second_dict["summary_dict"]["pod_row_dict_list"]):
            assert second_row_dict["as_of_timestamp_str"] != first_row_dict["as_of_timestamp_str"]
            assert second_row_dict["latest_pod_state_timestamp_str"] == first_row_dict["latest_pod_state_timestamp_str"]
            assert second_row_dict["latest_vplan_target_execution_timestamp_str"] == first_row_dict["latest_vplan_target_execution_timestamp_str"]
    finally:
        option_dict["provider_obj"].close()
