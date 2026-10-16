"""Doctor must diagnose the existing daily execution contract without trading."""
from contextlib import closing
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import pytest

from alpha.live import doctor, runner
from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
from alpha.live.daily_reconcile import DAILY_STRATEGY_IMPORT_TUPLE
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.runner import BrokerAdapterResolver
import test_live_core5_adapter as core5_fixture
import test_live_core5_wiring as wiring_fixture
import test_live_doctor as doctor_fixture


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def _market_timestamp(date_str, time_str):
    return datetime.fromisoformat(date_str + "T" + time_str).replace(tzinfo=MARKET_TIMEZONE_OBJ)


def _write_cycle(db_path, release_obj, *, date_str="2026-10-05", status_str="submitted",
        dispatch_dict=None, vplan_bool=True, target_str=None, vplan_decision_id_int=1):
    target_str = target_str or _market_timestamp(date_str, "09:30:00").isoformat()
    metadata_dict = {} if dispatch_dict is None else {"opening_dispatch_result_dict": dispatch_dict}
    with closing(sqlite3.connect(db_path)) as connection_obj:
        connection_obj.execute("""CREATE TABLE decision_plan (decision_plan_id_int INTEGER,
            pod_id_str TEXT,status_str TEXT,signal_timestamp_str TEXT,submission_timestamp_str TEXT,
            target_execution_timestamp_str TEXT,execution_policy_str TEXT,snapshot_metadata_json_str TEXT)""")
        connection_obj.execute("""CREATE TABLE vplan (vplan_id_int INTEGER,decision_plan_id_int INTEGER,
            pod_id_str TEXT,status_str TEXT,signal_timestamp_str TEXT,submission_timestamp_str TEXT,
            target_execution_timestamp_str TEXT,execution_policy_str TEXT)""")
        connection_obj.execute("INSERT INTO decision_plan VALUES (1,?,?,?,?,?,?,?)", (
            release_obj.pod_id_str, status_str, target_str, target_str, target_str,
            release_obj.execution_policy_str, json.dumps(metadata_dict)))
        if vplan_bool:
            connection_obj.execute("INSERT INTO vplan VALUES (1,?,?,?,?,?,?,?)", (
                vplan_decision_id_int, release_obj.pod_id_str,
                "ready" if status_str in {"planned", "vplan_ready"} else status_str,
                target_str, target_str, target_str, release_obj.execution_policy_str))
        connection_obj.commit()


@pytest.mark.parametrize("strategy_import_str", DAILY_STRATEGY_IMPORT_TUPLE)
@pytest.mark.parametrize("date_str,time_str,status_str,expected_status_str,expected_reason_str", [
    ("2026-10-05", "09:30:00", "submitted", "WAIT", "daily_cycle_in_progress"),
    ("2026-10-05", "10:00:00", "submitted", "WAIT", "daily_cycle_in_progress"),
    ("2026-10-05", "10:00:00", "submitting", "WAIT", "daily_cycle_submission_uncertain"),
    ("2026-10-05", "16:00:00", "submitted", "WAIT", "daily_cycle_awaiting_close_reconciliation"),
    ("2026-10-05", "16:59:59", "submitted", "WAIT", "daily_cycle_awaiting_close_reconciliation"),
    ("2026-10-05", "17:00:00", "submitted", "BLOCK", "daily_cycle_overdue"),
    ("2026-11-27", "12:59:59", "submitted", "WAIT", "daily_cycle_in_progress"),
    ("2026-11-27", "13:00:00", "submitted", "WAIT", "daily_cycle_awaiting_close_reconciliation"),
    ("2026-11-27", "14:00:00", "submitted", "BLOCK", "daily_cycle_overdue"),
    ("2026-11-02", "10:00:00", "submitted", "WAIT", "daily_cycle_in_progress"),
])
def test_daily_lifecycle_uses_exchange_close_and_never_certifies_pending_cycle(tmp_path,
        strategy_import_str, date_str, time_str, status_str, expected_status_str, expected_reason_str):
    release_obj = replace(core5_fixture.release_obj.__wrapped__(), strategy_import_str=strategy_import_str)
    db_path = tmp_path / "daily.sqlite3"
    _write_cycle(db_path, release_obj, date_str=date_str, status_str=status_str)
    before_bytes = db_path.read_bytes()
    lifecycle_dict = doctor.inspect_persisted_lifecycle_read_only_dict(
        db_path_str=str(db_path), pod_id_str=release_obj.pod_id_str,
        as_of_ts=_market_timestamp(date_str, time_str), release_obj=release_obj)
    component_dict = doctor._persisted_lifecycle_result_dict(lifecycle_dict)
    assert component_dict["status_str"] == expected_status_str
    assert component_dict["reason_code_str"] == expected_reason_str
    assert lifecycle_dict["active_lifecycle_bool"]
    assert not lifecycle_dict["expired_lifecycle_bool"]
    assert db_path.read_bytes() == before_bytes


@pytest.mark.parametrize("status_str,time_str,dispatch_dict", [
    ("planned", "09:28:00", None),
    ("vplan_ready", "10:00:00", None),
    ("blocked", "10:00:00", None),
    ("expired", "10:00:00", None),
    ("submitted", "10:00:00", {"reason_code_str": "opening_dispatch_incomplete",
        "attempted_request_key_list": [], "never_dispatched_request_key_list": ["unsent"]}),
    ("vplan_ready", "10:00:00", {"reason_code_str": "dispatch_retry_pending",
        "attempted_request_key_list": []}),
])
def test_daily_unsent_or_never_dispatched_cycle_stays_blocked(tmp_path, status_str, time_str, dispatch_dict):
    release_obj = core5_fixture.release_obj.__wrapped__()
    db_path = tmp_path / "unsent.sqlite3"
    _write_cycle(db_path, release_obj, status_str=status_str, dispatch_dict=dispatch_dict)
    lifecycle_dict = doctor.inspect_persisted_lifecycle_read_only_dict(db_path_str=str(db_path),
        pod_id_str=release_obj.pod_id_str, as_of_ts=_market_timestamp("2026-10-05", time_str),
        release_obj=release_obj)
    assert lifecycle_dict["active_lifecycle_bool"]
    assert doctor._persisted_lifecycle_result_dict(lifecycle_dict)["status_str"] == "BLOCK"
    assert lifecycle_dict["reason_code_str"] == "persisted_lifecycle_expired"


@pytest.mark.parametrize("dispatch_dict", [
    {"reason_code_str": "opening_dispatch_incomplete", "attempted_request_key_list": ["attempted"],
        "never_dispatched_request_key_list": ["unsent"]},
    {"reason_code_str": "submission_uncertain", "attempted_request_key_list": ["attempted"]},
])
def test_partially_attempted_daily_batch_waits_without_replacement(tmp_path, monkeypatch, dispatch_dict):
    release_obj = core5_fixture.release_obj.__wrapped__()
    release_path = wiring_fixture._write_release_yaml(tmp_path, release_obj)
    db_path = tmp_path / "partial.sqlite3"
    _write_cycle(db_path, release_obj, dispatch_dict=dispatch_dict)
    before_bytes = db_path.read_bytes()
    monkeypatch.setattr(doctor.scheduler_utils, "evaluate_build_gate_dict",
        lambda *_args: pytest.fail("An active daily cycle must not preview a replacement"))
    report_dict = doctor.compute_doctor_verdict(releases_root_path_str=str(release_path),
        env_mode_str="paper", as_of_ts=_market_timestamp("2026-10-05", "10:00:00"),
        pod_id_str=release_obj.pod_id_str, db_path_str=str(db_path), trace_enabled_bool=False,
        **doctor_fixture._doctor_config_kwargs(tmp_path, release_path))
    assert report_dict["overall_verdict_str"] == "WAIT"
    assert report_dict["decision_plan_dict"] == {}
    assert report_dict["vplan_preview_dict"] == {}
    assert db_path.read_bytes() == before_bytes


@pytest.mark.parametrize("status_str", ["completed", "completed_with_exceptions"])
def test_completed_daily_cycle_is_not_overdue(tmp_path, status_str):
    release_obj = core5_fixture.release_obj.__wrapped__()
    db_path = tmp_path / "completed.sqlite3"
    _write_cycle(db_path, release_obj, status_str=status_str)
    lifecycle_dict = doctor.inspect_persisted_lifecycle_read_only_dict(db_path_str=str(db_path),
        pod_id_str=release_obj.pod_id_str, as_of_ts=_market_timestamp("2026-10-05", "18:00:00"),
        release_obj=release_obj)
    assert not lifecycle_dict["active_lifecycle_bool"]
    assert doctor._persisted_lifecycle_result_dict(lifecycle_dict)["status_str"] == "PASS"


@pytest.mark.parametrize("strategy_import_str,policy_str", [
    ("strategies.dv2.strategy_mr_dv2:DVO2Strategy", "next_open_moo"),
    ("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash", "next_month_first_open"),
    ("strategies.momentum.strategy_mo_atr_normalized_ndx", "next_month_first_open"),
])
def test_legacy_and_monthly_lifecycle_expiry_is_unchanged(tmp_path, strategy_import_str, policy_str):
    release_obj = replace(core5_fixture.release_obj.__wrapped__(),
        strategy_import_str=strategy_import_str, execution_policy_str=policy_str)
    db_path = tmp_path / "legacy.sqlite3"
    _write_cycle(db_path, release_obj)
    lifecycle_dict = doctor.inspect_persisted_lifecycle_read_only_dict(db_path_str=str(db_path),
        pod_id_str=release_obj.pod_id_str, as_of_ts=_market_timestamp("2026-10-05", "10:00:00"),
        release_obj=release_obj)
    assert doctor._persisted_lifecycle_result_dict(lifecycle_dict)["reason_code_str"] == "persisted_lifecycle_expired"


@pytest.mark.parametrize("target_str,vplan_decision_id_int", [("invalid", 1), ("2026-10-05T09:30:00", 1)])
def test_invalid_daily_cycle_evidence_fails_closed(tmp_path, target_str, vplan_decision_id_int):
    release_obj = core5_fixture.release_obj.__wrapped__()
    db_path = tmp_path / "invalid.sqlite3"
    _write_cycle(db_path, release_obj, target_str=target_str, vplan_decision_id_int=vplan_decision_id_int)
    lifecycle_dict = doctor.inspect_persisted_lifecycle_read_only_dict(db_path_str=str(db_path),
        pod_id_str=release_obj.pod_id_str, as_of_ts=_market_timestamp("2026-10-05", "10:00:00"),
        release_obj=release_obj)
    assert doctor._persisted_lifecycle_result_dict(lifecycle_dict)["status_str"] == "BLOCK"


@pytest.fixture
def core5_case(tmp_path, monkeypatch):
    release_obj = core5_fixture.release_obj.__wrapped__()
    pricing_df = core5_fixture.price_df.__wrapped__()
    core5_fixture._controlled_signals(monkeypatch, pricing_df)
    state_obj = core5_fixture._state(release_obj, "2026-09-11", cash_float=1_000_000)
    decision_obj = core5_fixture._build(release_obj, pricing_df, "2026-09-11", state_obj)
    release_path = wiring_fixture._write_release_yaml(tmp_path, release_obj)
    store_obj, broker_obj = core5_fixture._store_and_broker(tmp_path, release_obj, state_obj)
    monkeypatch.setattr(doctor.scheduler_utils, "evaluate_build_gate_dict",
        lambda *_args: {"due_bool": True, "reason_code_str": "snapshot_ready"})
    monkeypatch.setattr(doctor.strategy_host, "build_decision_plan_for_release", lambda **_kwargs: decision_obj)
    monkeypatch.delenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", raising=False)
    monkeypatch.delenv("NORGATE_SNAPSHOT_ROOT", raising=False)
    def fail_quote(*_args, **_kwargs):
        pytest.fail("CORE5 must not require or fall back to live quotes")
    monkeypatch.setattr(broker_obj, "get_live_price_snapshot", fail_quote)
    return release_obj, state_obj, decision_obj, release_path, store_obj, broker_obj


def _core5_doctor_report(core5_case, tmp_path, as_of_ts=None):
    release_obj, state_obj, _decision_obj, release_path, _store_obj, broker_obj = core5_case
    return doctor.compute_doctor_verdict(releases_root_path_str=str(release_path), env_mode_str="paper",
        as_of_ts=as_of_ts or _decision_obj.submission_timestamp_ts, pod_id_str=release_obj.pod_id_str,
        db_path_str=str(tmp_path / "not_existing.sqlite3"), pod_state_loader_fn=lambda *_args: state_obj,
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=broker_obj),
        norgate_snapshot_sync_checker_fn=lambda **_kwargs: {"status_str": "ready", "reason_code_str": "local_snapshot_ready"},
        trace_enabled_bool=False, **doctor_fixture._doctor_config_kwargs(tmp_path, release_path))


@pytest.mark.parametrize("broker_nav_float", [800_000.0, 1_200_000.0])
def test_core5_doctor_matches_runner_without_quotes_and_keeps_db_read_only(core5_case, tmp_path, monkeypatch, broker_nav_float):
    release_obj, _state_obj, decision_obj, _release_path, store_obj, broker_obj = core5_case
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(
        snapshot_obj, net_liq_float=broker_nav_float, total_value_float=broker_nav_float)
    captured_vplan_list = []
    real_builder_fn = doctor.build_vplan
    def capture_builder(**kwargs_dict):
        vplan_obj = real_builder_fn(**kwargs_dict)
        captured_vplan_list.append(vplan_obj)
        return vplan_obj
    monkeypatch.setattr(doctor, "build_vplan", capture_builder)
    db_path = tmp_path / "core5.sqlite3"
    before_bytes = db_path.read_bytes()
    report_dict = _core5_doctor_report(core5_case, tmp_path)
    assert report_dict["overall_verdict_str"] == "PASS"
    assert db_path.read_bytes() == before_bytes
    assert not broker_obj.submitted_order_request_list
    preview_obj = captured_vplan_list[0]
    store_obj.insert_decision_plan(decision_obj)
    runner_dict = runner.build_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        "paper", log_path_str=str(tmp_path / "runner.log"), trace_enabled_bool=False)
    assert runner_dict["created_vplan_count_int"] == 1
    actual_obj = store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
    assert preview_obj.target_share_map == actual_obj.target_share_map
    frozen_target_dict = decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    assert all(preview_obj.target_share_map.get(asset_str, 0.0) == frozen_target_dict.get(asset_str, 0.0)
        for asset_str in set(preview_obj.target_share_map) | set(frozen_target_dict))
    assert preview_obj.live_reference_snapshot_timestamp_ts == decision_obj.signal_timestamp_ts
    assert preview_obj.live_reference_price_map == decision_obj.snapshot_metadata_dict["sizing_close_price_map_dict"]
    assert set(preview_obj.live_reference_source_map_dict.values()) == {"core5.frozen_close_t"}
    assert all(row_obj.live_reference_source_str == "core5.frozen_close_t" for row_obj in preview_obj.vplan_row_list)
    assert preview_obj.order_delta_map == actual_obj.order_delta_map
    assert preview_obj.live_price_source_str == actual_obj.live_price_source_str == "core5.frozen_close_t"
    def economic_legs(vplan_obj):
        return [(item_obj.asset_str, item_obj.amount_float, item_obj.unit_str)
            for item_obj in build_broker_order_request_list_from_vplan(vplan_obj)]
    assert economic_legs(preview_obj) == economic_legs(actual_obj)
    assert not broker_obj.submitted_order_request_list


@pytest.mark.parametrize("mutation_str", ["missing_prices", "zero_price", "nan_price", "wrong_target"])
def test_core5_invalid_frozen_book_still_blocks_without_quote_fallback(core5_case, tmp_path, mutation_str):
    decision_obj = core5_case[2]
    metadata_dict = decision_obj.snapshot_metadata_dict
    if mutation_str == "missing_prices":
        metadata_dict.pop("sizing_close_price_map_dict")
    elif mutation_str == "wrong_target":
        metadata_dict["fixed_target_share_map_dict"]["SPY"] += 1
    else:
        metadata_dict["sizing_close_price_map_dict"]["SPY"] = 0.0 if mutation_str == "zero_price" else float("nan")
    report_dict = _core5_doctor_report(core5_case, tmp_path)
    assert report_dict["overall_verdict_str"] == "BLOCK"
    assert report_dict["vplan_preview_dict"] == {}
    assert not core5_case[5].submitted_order_request_list


@pytest.mark.parametrize("time_str,expected_status_str", [("09:24:00", "WAIT"), ("09:28:00", "BLOCK")])
def test_new_daily_decision_ignores_completed_previous_cycle_vplan(tmp_path, time_str, expected_status_str):
    release_obj = core5_fixture.release_obj.__wrapped__()
    db_path = tmp_path / "rollover.sqlite3"
    _write_cycle(db_path, release_obj, status_str="planned", vplan_decision_id_int=0)
    prior_target_str = _market_timestamp("2026-10-02", "09:30:00").isoformat()
    with closing(sqlite3.connect(db_path)) as connection_obj:
        connection_obj.execute("UPDATE vplan SET status_str='completed',target_execution_timestamp_str=?", (prior_target_str,))
        connection_obj.execute("INSERT INTO decision_plan SELECT 0,pod_id_str,'completed',?,?,?,execution_policy_str,'{}' FROM decision_plan WHERE decision_plan_id_int=1",
            (prior_target_str, prior_target_str, prior_target_str))
        connection_obj.commit()
    lifecycle_dict = doctor.inspect_persisted_lifecycle_read_only_dict(db_path_str=str(db_path),
        pod_id_str=release_obj.pod_id_str, as_of_ts=_market_timestamp("2026-10-05", time_str),
        release_obj=release_obj)
    assert lifecycle_dict["latest_vplan_dict"] == {}
    assert doctor._persisted_lifecycle_result_dict(lifecycle_dict)["status_str"] == expected_status_str


@pytest.mark.parametrize("time_str,expected_status_str,expected_reason_str", [
    ("10:00:00", "WAIT", "daily_cycle_in_progress"),
    ("16:00:00", "WAIT", "daily_cycle_awaiting_close_reconciliation"),
    ("17:00:00", "BLOCK", "daily_cycle_overdue"),
])
def test_real_no_order_daily_decision_waits_for_close(tmp_path, monkeypatch, time_str, expected_status_str, expected_reason_str):
    release_obj = core5_fixture.release_obj.__wrapped__()
    pricing_df = core5_fixture.price_df.__wrapped__()
    _store_obj, decision_obj, _state_obj = core5_fixture._no_order_cycle(release_obj, pricing_df, monkeypatch, tmp_path)
    lifecycle_dict = doctor.inspect_persisted_lifecycle_read_only_dict(
        db_path_str=str(tmp_path / "core5.sqlite3"), pod_id_str=release_obj.pod_id_str,
        as_of_ts=_market_timestamp("2026-09-14", time_str), release_obj=release_obj)
    assert decision_obj.snapshot_metadata_dict["no_order_bool"]
    assert doctor._persisted_lifecycle_result_dict(lifecycle_dict)["status_str"] == expected_status_str
    assert lifecycle_dict["reason_code_str"] == expected_reason_str


@pytest.mark.parametrize("time_str,expected_status_str", [("09:27:59", "PASS"), ("09:28:00", "BLOCK"), ("09:29:00", "BLOCK")])
def test_fresh_daily_preview_matches_runner_opening_deadline(core5_case, tmp_path, time_str, expected_status_str):
    report_dict = _core5_doctor_report(core5_case, tmp_path, _market_timestamp("2026-09-14", time_str))
    assert report_dict["overall_verdict_str"] == expected_status_str
    if expected_status_str == "BLOCK":
        assert report_dict["component_result_dict_list"][-1]["reason_code_str"] == "submission_window_expired"
        assert report_dict["vplan_preview_dict"] == {}
    assert not core5_case[5].submitted_order_request_list


@pytest.mark.parametrize("mutation_str", ["open_orders", "holdings", "nonpositive_nav"])
def test_core5_existing_account_guards_still_block(core5_case, tmp_path, mutation_str):
    release_obj, _state_obj, _decision_obj, _release_path, _store_obj, broker_obj = core5_case
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    mutation_dict = (
        {"open_order_id_list": ["existing_order"]} if mutation_str == "open_orders" else
        {"position_amount_map": {"SPY": 1.0}} if mutation_str == "holdings" else {"net_liq_float": 0.0}
    )
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj, **mutation_dict)
    report_dict = _core5_doctor_report(core5_case, tmp_path)
    assert report_dict["overall_verdict_str"] == "BLOCK"
    assert report_dict["vplan_preview_dict"] == {}
    assert not broker_obj.submitted_order_request_list


@pytest.mark.parametrize("time_str,expected_status_str", [
    ("09:30:00", "WAIT"), ("09:31:00", "WAIT"), ("09:31:00.000001", "BLOCK")])
def test_legacy_market_order_grace_boundary_is_unchanged(tmp_path, time_str, expected_status_str):
    release_obj = replace(core5_fixture.release_obj.__wrapped__(),
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        execution_policy_str="next_open_market")
    db_path = tmp_path / "market.sqlite3"
    _write_cycle(db_path, release_obj)
    lifecycle_dict = doctor.inspect_persisted_lifecycle_read_only_dict(db_path_str=str(db_path),
        pod_id_str=release_obj.pod_id_str, as_of_ts=_market_timestamp("2026-10-05", time_str),
        release_obj=release_obj)
    assert doctor._persisted_lifecycle_result_dict(lifecycle_dict)["status_str"] == expected_status_str


@pytest.mark.parametrize("status_str", ["planned", "completed", "completed_with_exceptions"])
def test_older_active_daily_cycle_cannot_disappear_behind_newer_decision(tmp_path, monkeypatch, status_str):
    release_obj = core5_fixture.release_obj.__wrapped__()
    release_path = wiring_fixture._write_release_yaml(tmp_path, release_obj)
    db_path = tmp_path / "older_active.sqlite3"
    _write_cycle(db_path, release_obj, status_str=status_str, vplan_decision_id_int=0)
    with closing(sqlite3.connect(db_path)) as connection_obj:
        connection_obj.execute("UPDATE vplan SET status_str='submitted'")
        connection_obj.commit()
    before_bytes = db_path.read_bytes()
    monkeypatch.setattr(doctor.scheduler_utils, "evaluate_build_gate_dict",
        lambda *_args: pytest.fail("An older active cycle must prevent fresh preview"))
    report_dict = doctor.compute_doctor_verdict(releases_root_path_str=str(release_path),
        env_mode_str="paper", as_of_ts=_market_timestamp("2026-10-05", "10:00:00"),
        pod_id_str=release_obj.pod_id_str, db_path_str=str(db_path), trace_enabled_bool=False,
        **doctor_fixture._doctor_config_kwargs(tmp_path, release_path))
    assert report_dict["overall_verdict_str"] == "BLOCK"
    assert report_dict["component_result_dict_list"][-1]["reason_code_str"] == "persisted_prior_daily_cycle_pending"
    assert report_dict["decision_plan_dict"] == {}
    assert report_dict["vplan_preview_dict"] == {}
    assert db_path.read_bytes() == before_bytes
