"""Selected cycles keep their own saved evidence and never write live state."""

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from alpha.live.dashboard_v4.cycle import build_cycle_view_dict
from alpha.live.dashboard_v4.evidence import load_cycle_evidence_dict
from alpha.live.dashboard_v4.pod_data import load_pod_cycles_dict
from test_dashboard_v4_evidence import FILL_TS, NOW_TS, SUBMIT_TS, build_fixture_tuple, update_db


def _decision_only_int(target_obj, *, day_offset_int=1, **changes_dict):
    with sqlite3.connect(target_obj.db_path_str) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        decision_dict = dict(connection_obj.execute("SELECT * FROM decision_plan ORDER BY decision_plan_id_int LIMIT 1").fetchone())
        decision_dict.pop("decision_plan_id_int")
        for field_str in ("signal_timestamp_str", "submission_timestamp_str", "target_execution_timestamp_str"):
            decision_dict[field_str] = (datetime.fromisoformat(decision_dict[field_str]) + timedelta(days=day_offset_int)).isoformat()
        decision_dict.update(changes_dict)
        field_str = ",".join(decision_dict)
        placeholder_str = ",".join("?" for _ in decision_dict)
        return connection_obj.execute(f"INSERT INTO decision_plan ({field_str}) VALUES ({placeholder_str})", tuple(decision_dict.values())).lastrowid


def _reconcile(target_obj, *, vplan_id_int=1, decision_id_int=1, status_str="passed", stage_str="post_execution", timestamp_ts=None):
    with sqlite3.connect(target_obj.db_path_str) as connection_obj:
        connection_obj.execute("""INSERT INTO vplan_reconciliation_snapshot
            (pod_id_str, decision_plan_id_int, vplan_id_int, stage_str, status_str,
             mismatch_json_str, model_position_json_str, broker_position_json_str,
             model_cash_float, broker_cash_float, created_timestamp_str)
             VALUES ('pod',?,?,?,?, '{}','{"SPY":2}', '{"SPY":2}', 9800,9800,?)""",
            (decision_id_int, vplan_id_int, stage_str, status_str, (timestamp_ts or (FILL_TS + timedelta(minutes=6))).isoformat()))


def _eod(target_obj, timestamp_ts, *, account_str="U111", source_str="broker"):
    with sqlite3.connect(target_obj.db_path_str) as connection_obj:
        connection_obj.execute("""INSERT INTO pod_state_history
            (pod_id_str,user_id_str,account_route_str,position_json_str,cash_float,total_value_float,
             strategy_state_json_str,snapshot_stage_str,snapshot_source_str,updated_timestamp_str,recorded_timestamp_str)
             VALUES ('pod','owner',?,'{}',10000,10000,'{}','eod',?,?,?)""",
            (account_str, source_str, timestamp_ts.isoformat(), timestamp_ts.isoformat()))


def _upgrade_target_obj(target_obj):
    with sqlite3.connect(target_obj.db_path_str) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        release_dict = dict(connection_obj.execute("SELECT * FROM live_release WHERE release_id_str='release'").fetchone())
        release_dict.update(release_id_str="release-v2", data_profile_str="new-profile", session_calendar_id_str="XTSE")
        field_str = ",".join(release_dict)
        placeholder_str = ",".join("?" for _ in release_dict)
        connection_obj.execute(f"INSERT INTO live_release ({field_str}) VALUES ({placeholder_str})", tuple(release_dict.values()))
    return replace(target_obj, release_obj=replace(target_obj.release_obj,
        release_id_str="release-v2", data_profile_str="new-profile", session_calendar_id_str="XTSE"))


def test_selected_complete_cycle_is_read_only_and_keeps_same_asset_legs(tmp_path):
    target_obj, row_dict = build_fixture_tuple(tmp_path)
    _reconcile(target_obj)
    update_db(target_obj, "UPDATE decision_plan SET snapshot_metadata_json_str=?", (json.dumps({"norgate_data_profile_str": "test", "norgate_snapshot_date_str": "2026-09-17", "secret_path_str": "C:/private"}),))
    db_path_obj = tmp_path / "pod.sqlite3"
    before_bytes, before_mtime_int = db_path_obj.read_bytes(), db_path_obj.stat().st_mtime_ns
    history_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS)
    assert history_dict["status_str"] == "ok"
    assert history_dict["selected_cycle_dict"]["cycle_key_str"] == "vplan:1"
    assert history_dict["selected_cycle_dict"]["current_bool"] is True
    assert [leg_dict["order_delta_share_float"] for leg_dict in history_dict["plan_row_list"]] == [10, -8]
    assert [leg_dict["order_request_key_str"] for leg_dict in history_dict["plan_row_list"]] == ["vplan:1:SPY:1", "vplan:1:SPY:2"]
    assert history_dict["reconciliation_dict"]["broker_position_map_dict"] == {"SPY": 2}
    assert "secret_path_str" not in history_dict["decision_dict"]["snapshot_metadata_dict"]
    assert "raw_payload_json_str" not in json.dumps(history_dict)
    assert history_dict["file_list"] == []
    projected_row_dict = history_dict["pod_row_dict"]
    projected_row_dict["cycle_evidence_dict"] = load_cycle_evidence_dict(target_obj, projected_row_dict, as_of_ts=NOW_TS)
    assert projected_row_dict["cycle_evidence_dict"]["state_str"] == "complete"
    cycle_dict = build_cycle_view_dict(projected_row_dict, now_ts=NOW_TS)
    assert cycle_dict["step_dict_list"][4]["state_str"] == "Done"
    assert cycle_dict["step_dict_list"][5]["state_str"] == "Done"
    assert db_path_obj.read_bytes() == before_bytes
    assert db_path_obj.stat().st_mtime_ns == before_mtime_int


def test_decision_only_and_older_unresolved_remain_separate(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, "UPDATE vplan SET status_str='submitted'")
    newer_id_int = _decision_only_int(target_obj, status_str="planned")
    later_ts = NOW_TS + timedelta(days=1)
    latest_dict = load_pod_cycles_dict(target_obj, as_of_ts=later_ts)
    assert latest_dict["status_str"] == "ok"
    assert [cycle_dict["cycle_key_str"] for cycle_dict in latest_dict["cycle_list"]] == [f"decision:{newer_id_int}", "vplan:1"]
    assert latest_dict["vplan_dict"] == {}
    assert latest_dict["order_list"] == []
    assert latest_dict["pod_row_dict"]["latest_vplan_id_int"] is None
    old_dict = load_pod_cycles_dict(target_obj, as_of_ts=later_ts, vplan_id_int=1)
    assert old_dict["status_str"] == "ok"
    assert old_dict["selected_cycle_dict"]["current_bool"] is False
    assert old_dict["pod_row_dict"]["latest_decision_plan_id_int"] == 1
    assert old_dict["pod_row_dict"]["latest_vplan_decision_plan_id_int"] == 1
    assert len(old_dict["order_list"]) == 2


@pytest.mark.parametrize("selector_dict", [{"vplan_id_int": 999}, {"decision_plan_id_int": 999}, {"decision_plan_id_int": 2, "vplan_id_int": 1}, {"vplan_id_int": "1"}, {"decision_plan_id_int": -1}, {"decision_plan_id_int": True}])
def test_invalid_cycle_selection_never_falls_back_to_latest(tmp_path, selector_dict):
    target_obj, _ = build_fixture_tuple(tmp_path)
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS, **selector_dict)
    assert result_dict["status_str"] == ("not_found" if any(type(value_obj) is int and value_obj > 0 for value_obj in selector_dict.values()) else "unknown")
    assert result_dict["selected_cycle_dict"] is None
    assert not result_dict["pod_row_dict"]


@pytest.mark.parametrize("sql_str", [
    "UPDATE live_release SET mode_str='paper'",
    "UPDATE live_release SET account_route_str='other'",
    "UPDATE vplan SET account_route_str='other'",
    "UPDATE vplan_broker_order SET account_route_str='other'",
    "UPDATE vplan_fill SET decision_plan_id_int=999",
    "UPDATE decision_plan SET snapshot_metadata_json_str='[]'",
    "UPDATE vplan SET order_delta_json_str='{\"SPY\":NaN}'",
    "UPDATE vplan_row SET live_reference_price_float='bad'",
    "UPDATE vplan_fill SET fill_price_float='bad'",
    "DROP TABLE vplan_broker_ack",
])
def test_bad_identity_or_saved_evidence_is_unavailable(tmp_path, sql_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, sql_str)
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS)
    assert result_dict["status_str"] == "unknown"
    assert result_dict["order_list"] == result_dict["fill_list"] == []


def test_target_non_live_rejected_before_any_database_open(tmp_path, monkeypatch):
    target_obj, _ = build_fixture_tuple(tmp_path)
    target_obj = replace(target_obj, release_obj=replace(target_obj.release_obj, mode_str="paper"))
    monkeypatch.setattr("alpha.live.dashboard_v4.pod_data.sqlite3.connect", lambda *args, **kwargs: pytest.fail("Opened non-LIVE state"))
    assert load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS)["status_str"] == "unknown"


def test_missing_database_does_not_create_file(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    target_obj = replace(target_obj, db_path_str=str(tmp_path / "not-created.sqlite3"))
    assert load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS)["status_str"] == "unknown"
    assert not (tmp_path / "not-created.sqlite3").exists()


@pytest.mark.parametrize("table_str,field_str", [("decision_plan", "updated_timestamp_str"), ("vplan", "updated_timestamp_str"), ("vplan_fill", "fill_timestamp_str"), ("vplan_broker_order", "last_status_timestamp_str")])
def test_future_saved_observation_cannot_be_green(tmp_path, table_str, field_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, f"UPDATE {table_str} SET {field_str}=?", ((NOW_TS + timedelta(seconds=1)).isoformat(),))
    assert load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS)["status_str"] == "unknown"


def test_reconcile_is_exact_cycle_and_post_execution_only(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj, status_str="blocked")
    _reconcile(target_obj, vplan_id_int=999)
    _reconcile(target_obj, decision_id_int=999)
    _reconcile(target_obj, stage_str="pre_vplan")
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS)
    assert result_dict["status_str"] == "ok"
    assert result_dict["reconciliation_dict"]["status_str"] == "blocked"
    assert result_dict["pod_row_dict"]["latest_reconciliation_status_str"] == "blocked"


@pytest.mark.parametrize("reconcile_ts", [FILL_TS - timedelta(seconds=1), FILL_TS + timedelta(days=1)])
def test_reconcile_before_execution_or_on_another_session_cannot_supply_after_positions(tmp_path, reconcile_ts):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj, timestamp_ts=reconcile_ts)
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS + timedelta(days=1))
    assert result_dict["status_str"] == "ok"
    assert result_dict["reconciliation_dict"] == {}
    assert "latest_reconciliation_status_str" not in result_dict["pod_row_dict"]


def test_selected_vplan_owns_cycle_timing_and_eod_session(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    changed_target_ts = FILL_TS.replace(hour=20)
    update_db(target_obj, "UPDATE vplan SET target_execution_timestamp_str=?, execution_policy_str='same_day_moc'", (changed_target_ts.isoformat(),))
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS)
    assert result_dict["status_str"] == "ok"
    assert result_dict["selected_cycle_dict"]["target_execution_timestamp_str"] == changed_target_ts.isoformat()
    assert result_dict["selected_cycle_dict"]["execution_policy_str"] == "same_day_moc"


def test_missing_historical_data_does_not_hide_other_proven_steps(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj)
    history_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS)
    row_dict = history_dict["pod_row_dict"]
    assert row_dict["norgate_snapshot_status_dict"] == {"status_str": "unknown"}
    cycle_dict = build_cycle_view_dict(row_dict, now_ts=NOW_TS)
    assert cycle_dict["step_dict_list"][0]["state_str"] == "Unknown"
    assert cycle_dict["step_dict_list"][1]["state_str"] == "Done"
    assert cycle_dict["step_dict_list"][5]["state_str"] == "Done"


def test_eod_never_borrows_other_session_account_or_early_observation(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    good_ts = NOW_TS.replace(hour=20, minute=10)
    _eod(target_obj, good_ts - timedelta(days=1))
    _eod(target_obj, good_ts, account_str="other")
    _eod(target_obj, good_ts, source_str="pod_state")
    _eod(target_obj, good_ts - timedelta(seconds=1))
    before_dict = load_pod_cycles_dict(target_obj, as_of_ts=good_ts + timedelta(minutes=1))
    assert before_dict["status_str"] == "ok"
    assert before_dict["eod_dict"]["same_session_bool"] is False
    _eod(target_obj, good_ts)
    after_dict = load_pod_cycles_dict(target_obj, as_of_ts=good_ts + timedelta(minutes=1))
    assert after_dict["eod_dict"]["status_str"] == "completed"
    assert after_dict["eod_dict"]["latest_timestamp_str"] == good_ts.isoformat()


def test_bounded_history_allows_explicit_older_cycle(tmp_path, monkeypatch):
    target_obj, _ = build_fixture_tuple(tmp_path)
    monkeypatch.setattr("alpha.live.dashboard_v4.pod_data.HISTORY_LIMIT_INT", 2)
    _decision_only_int(target_obj, day_offset_int=1)
    _decision_only_int(target_obj, day_offset_int=2)
    _decision_only_int(target_obj, day_offset_int=3)
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS + timedelta(days=4), vplan_id_int=1)
    assert result_dict["status_str"] == "ok"
    assert result_dict["history_truncated_bool"] is True
    assert [cycle_dict["decision_plan_id_int"] for cycle_dict in result_dict["cycle_list"]] == [4, 3, 1]
    assert result_dict["selected_cycle_dict"]["vplan_id_int"] == 1


def test_detail_over_limit_fails_closed_instead_of_truncating_to_false_full_fill(tmp_path, monkeypatch):
    target_obj, _ = build_fixture_tuple(tmp_path)
    monkeypatch.setattr("alpha.live.dashboard_v4.pod_data.DETAIL_LIMIT_INT", 1)
    assert load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS)["status_str"] == "unknown"


def test_older_unresolved_cycle_stays_in_bounded_picker(tmp_path, monkeypatch):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, "UPDATE vplan SET status_str='submitted'")
    monkeypatch.setattr("alpha.live.dashboard_v4.pod_data.HISTORY_LIMIT_INT", 2)
    for offset_int in (1, 2, 3):
        _decision_only_int(target_obj, day_offset_int=offset_int)
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS + timedelta(days=4))
    assert result_dict["status_str"] == "ok"
    assert [cycle_dict["decision_plan_id_int"] for cycle_dict in result_dict["cycle_list"]] == [4, 3, 1]
    assert result_dict["cycle_list"][-1]["unresolved_bool"] is True
    assert result_dict["selected_cycle_dict"]["decision_plan_id_int"] == 4


def test_empty_saved_history_returns_empty_without_writes(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, "DELETE FROM decision_plan")
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS)
    assert result_dict["status_str"] == "empty"
    assert result_dict["selected_cycle_dict"] is None


def test_eod_uses_actual_early_close_for_selected_session(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    target_ts = NOW_TS.replace(month=11, day=27, hour=14, minute=30)
    due_ts = target_ts.replace(hour=18, minute=10)
    for table_str in ("decision_plan", "vplan"):
        update_db(target_obj, f"UPDATE {table_str} SET target_execution_timestamp_str=?", (target_ts.isoformat(),))
    _eod(target_obj, due_ts)
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=due_ts + timedelta(minutes=1))
    assert result_dict["status_str"] == "ok"
    assert datetime.fromisoformat(result_dict["eod_dict"]["expected_due_timestamp_str"]) == due_ts
    assert result_dict["eod_dict"]["status_str"] == "completed"


def test_release_upgrade_keeps_old_live_cycle_and_selected_provenance(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, "UPDATE decision_plan SET snapshot_metadata_json_str=?", (json.dumps({"norgate_data_profile_str": "test", "norgate_snapshot_date_str": "2026-09-17"}),))
    target_obj = _upgrade_target_obj(target_obj)
    latest_id_int = _decision_only_int(target_obj, release_id_str="release-v2", status_str="planned")
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS + timedelta(days=1), vplan_id_int=1)
    assert result_dict["status_str"] == "ok"
    assert [cycle_dict["decision_plan_id_int"] for cycle_dict in result_dict["cycle_list"]] == [latest_id_int, 1]
    assert result_dict["selected_cycle_dict"]["current_bool"] is False
    assert result_dict["selected_release_dict"] == {
        "release_id_str": "release", "user_id_str": "owner", "pod_id_str": "pod", "account_route_str": "U111",
        "mode_str": "live", "session_calendar_id_str": "XNYS", "data_profile_str": "test",
        "signal_clock_str": "eod_snapshot_ready", "execution_policy_str": "next_open_moo",
    }
    assert result_dict["pod_row_dict"]["release_id_str"] == "release"
    assert result_dict["pod_row_dict"]["norgate_snapshot_status_dict"]["status_str"] == "ready"
    evidence_target_obj = replace(target_obj, release_obj=replace(target_obj.release_obj, **result_dict["selected_release_dict"]))
    assert load_cycle_evidence_dict(evidence_target_obj, result_dict["pod_row_dict"], as_of_ts=NOW_TS + timedelta(days=1))["state_str"] == "complete"


@pytest.mark.parametrize("field_str,value_str", [("mode_str", "paper"), ("mode_str", "incubation"), ("account_route_str", "other"), ("user_id_str", "other"), ("pod_id_str", "other")])
def test_nonlive_or_foreign_historical_release_never_enters_picker(tmp_path, field_str, value_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    target_obj = _upgrade_target_obj(target_obj)
    newest_id_int = _decision_only_int(target_obj, release_id_str="release-v2", status_str="planned")
    update_db(target_obj, f"UPDATE live_release SET {field_str}=? WHERE release_id_str='release'", (value_str,))
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS + timedelta(days=1))
    assert result_dict["status_str"] == "ok"
    assert [cycle_dict["decision_plan_id_int"] for cycle_dict in result_dict["cycle_list"]] == [newest_id_int]
    assert load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS + timedelta(days=1), vplan_id_int=1)["status_str"] == "not_found"


def test_old_release_does_not_bypass_invalid_current_anchor(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    target_obj = _upgrade_target_obj(target_obj)
    update_db(target_obj, "UPDATE live_release SET mode_str='paper' WHERE release_id_str='release-v2'")
    assert load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS, vplan_id_int=1)["status_str"] == "unknown"


def test_selected_old_release_owns_early_close_calendar(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    target_obj = _upgrade_target_obj(target_obj)
    target_ts = NOW_TS.replace(month=11, day=27, hour=14, minute=30)
    due_ts = target_ts.replace(hour=18, minute=10)
    for table_str in ("decision_plan", "vplan"):
        update_db(target_obj, f"UPDATE {table_str} SET target_execution_timestamp_str=?", (target_ts.isoformat(),))
    _eod(target_obj, due_ts)
    result_dict = load_pod_cycles_dict(target_obj, as_of_ts=due_ts + timedelta(minutes=1), vplan_id_int=1)
    assert result_dict["status_str"] == "ok"
    assert result_dict["selected_release_dict"]["session_calendar_id_str"] == "XNYS"
    assert datetime.fromisoformat(result_dict["eod_dict"]["expected_due_timestamp_str"]) == due_ts
    assert result_dict["eod_dict"]["status_str"] == "completed"


def test_old_release_cycle_future_observation_stays_unavailable(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    target_obj = _upgrade_target_obj(target_obj)
    update_db(target_obj, "UPDATE vplan SET updated_timestamp_str=?", ((NOW_TS + timedelta(seconds=1)).isoformat(),))
    assert load_pod_cycles_dict(target_obj, as_of_ts=NOW_TS, vplan_id_int=1)["status_str"] == "unknown"
