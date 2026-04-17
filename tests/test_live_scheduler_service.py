from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from alpha.live.models import DecisionPlan, VPlan, VPlanRow
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.scheduler_service import get_scheduler_decision, run_once
from alpha.live.state_store_v2 import LiveStateStore


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def _write_manifest(
    root_path_obj: Path,
    auto_submit_enabled_bool: bool = True,
) -> None:
    releases_root_path_obj = root_path_obj / "releases" / "user_001"
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / "pod_test.yaml").write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: user_001.pod_test.daily.v2",
                "  user_id: user_001",
                "  pod_id: pod_test_01",
                "deployment:",
                "  mode: paper",
                "  enabled_bool: true",
                "broker:",
                "  account_route: DU1",
                "strategy:",
                "  strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy",
                "  data_profile_str: norgate_eod_sp500_pit",
                "  params: {}",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                "  signal_clock_str: eod_snapshot_ready",
                "  execution_policy_str: next_open_moo",
                "execution:",
                "  pod_budget_fraction_float: 0.5",
                f"  auto_submit_enabled_bool: {'true' if auto_submit_enabled_bool else 'false'}",
                "bootstrap:",
                "  initial_cash_float: 10000.0",
                "risk:",
                "  risk_profile_str: standard",
            ]
        ),
        encoding="utf-8",
    )


def _build_decision_plan_stub(release_obj, as_of_ts, pod_state_obj):
    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 23, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        decision_base_position_map={},
        snapshot_metadata_dict={"strategy_family_str": "stub"},
        strategy_state_dict={"trade_id_int": 1},
        decision_book_type_str="incremental_entry_exit_book",
        entry_target_weight_map_dict={"AAPL": 0.2},
        target_weight_map={"AAPL": 0.2},
        exit_asset_set=set(),
        entry_priority_list=["AAPL"],
        cash_reserve_weight_float=0.0,
        preserve_untouched_positions_bool=True,
    )


def _insert_planned_decision_plan(
    state_store_obj: LiveStateStore,
) -> DecisionPlan:
    return state_store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str="user_001.pod_test.daily.v2",
            user_id_str="user_001",
            pod_id_str="pod_test_01",
            account_route_str="DU1",
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 23, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            decision_base_position_map={},
            snapshot_metadata_dict={"strategy_family_str": "stub"},
            strategy_state_dict={},
            decision_book_type_str="incremental_entry_exit_book",
            entry_target_weight_map_dict={"AAPL": 0.2},
            target_weight_map={"AAPL": 0.2},
            exit_asset_set=set(),
            entry_priority_list=["AAPL"],
            cash_reserve_weight_float=0.0,
        )
    )


def _insert_submitted_vplan(
    state_store_obj: LiveStateStore,
) -> VPlan:
    decision_plan_obj = _insert_planned_decision_plan(state_store_obj)
    state_store_obj.mark_decision_plan_status(int(decision_plan_obj.decision_plan_id_int or 0), "submitted")
    return state_store_obj.insert_vplan(
        VPlan(
            release_id_str=decision_plan_obj.release_id_str,
            user_id_str=decision_plan_obj.user_id_str,
            pod_id_str=decision_plan_obj.pod_id_str,
            account_route_str=decision_plan_obj.account_route_str,
            decision_plan_id_int=int(decision_plan_obj.decision_plan_id_int or 0),
            signal_timestamp_ts=decision_plan_obj.signal_timestamp_ts,
            submission_timestamp_ts=decision_plan_obj.submission_timestamp_ts,
            target_execution_timestamp_ts=decision_plan_obj.target_execution_timestamp_ts,
            execution_policy_str=decision_plan_obj.execution_policy_str,
            broker_snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
            live_reference_snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
            live_price_source_str="stub",
            net_liq_float=10000.0,
            available_funds_float=8000.0,
            excess_liquidity_float=7000.0,
            pod_budget_fraction_float=0.5,
            pod_budget_float=5000.0,
            current_broker_position_map={},
            live_reference_price_map={"AAPL": 100.0},
            target_share_map={"AAPL": 10.0},
            order_delta_map={"AAPL": 10.0},
            vplan_row_list=[
                VPlanRow(
                    asset_str="AAPL",
                    current_share_float=0.0,
                    target_share_float=10.0,
                    order_delta_share_float=10.0,
                    live_reference_price_float=100.0,
                    estimated_target_notional_float=1000.0,
                    broker_order_type_str="MOO",
                )
            ],
            status_str="submitted",
        )
    )


def test_scheduler_decision_selects_build_now(tmp_path: Path, monkeypatch):
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    scheduler_decision_obj = get_scheduler_decision(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 22, 10, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )

    assert scheduler_decision_obj.due_now_bool is True
    assert scheduler_decision_obj.next_phase_str == "build_decision_plan"
    assert scheduler_decision_obj.reason_code_str == "ready_to_build_decision_plan"
    assert scheduler_decision_obj.related_pod_id_list == ["pod_test_01"]


def test_scheduler_decision_uses_submission_timestamp_in_utc(tmp_path: Path, monkeypatch):
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    _insert_planned_decision_plan(state_store_obj)
    scheduler_decision_obj = get_scheduler_decision(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 14, 0, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )

    assert scheduler_decision_obj.due_now_bool is False
    assert scheduler_decision_obj.next_phase_str == "build_vplan"
    assert scheduler_decision_obj.reason_code_str == "waiting_for_submission_window"
    assert scheduler_decision_obj.next_due_timestamp_ts == datetime(2024, 2, 1, 14, 23, 30, tzinfo=UTC)


def test_scheduler_decision_active_polls_for_submitted_vplan_before_reconcile(tmp_path: Path, monkeypatch):
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    _insert_submitted_vplan(state_store_obj)
    scheduler_decision_obj = get_scheduler_decision(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 14, 25, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )

    assert scheduler_decision_obj.due_now_bool is False
    assert scheduler_decision_obj.active_poll_bool is True
    assert scheduler_decision_obj.next_phase_str == "post_execution_reconcile"
    assert scheduler_decision_obj.reason_code_str == "waiting_for_post_execution_reconcile"
    assert scheduler_decision_obj.next_due_timestamp_ts == datetime(2024, 2, 1, 14, 35, tzinfo=UTC)


def test_scheduler_decision_reconcile_due_now_after_grace(tmp_path: Path, monkeypatch):
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    _insert_submitted_vplan(state_store_obj)
    scheduler_decision_obj = get_scheduler_decision(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 14, 36, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )

    assert scheduler_decision_obj.due_now_bool is True
    assert scheduler_decision_obj.next_phase_str == "post_execution_reconcile"
    assert scheduler_decision_obj.reason_code_str == "ready_to_reconcile"


def test_scheduler_run_once_invokes_tick_for_startup_catchup(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=True)
    monkeypatch.setattr("alpha.live.strategy_host.build_decision_plan_for_release", _build_decision_plan_stub)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    detail_dict = run_once(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 1, 31, 22, 10, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )

    assert detail_dict["tick_invoked_bool"] is True
    assert detail_dict["tick_detail_dict"]["created_decision_plan_count_int"] == 1
    assert detail_dict["post_tick_pod_status_dict_list"] == [
        {
            "release_id_str": "user_001.pod_test.daily.v2",
            "pod_id_str": "pod_test_01",
            "account_route_str": "DU1",
            "latest_decision_plan_status_str": "planned",
            "latest_vplan_status_str": None,
            "latest_signal_timestamp_str": "2024-01-31T16:00:00-05:00",
            "latest_submission_timestamp_str": "2024-02-01T09:23:30-05:00",
            "latest_broker_snapshot_timestamp_str": None,
            "next_action_str": "wait",
            "reason_code_str": "waiting_for_submission_window",
            "latest_fill_timestamp_str": None,
            "latest_broker_order_row_dict_list": [],
        }
    ]
    assert detail_dict["post_tick_execution_report_dict_list"] == []


def test_scheduler_run_once_cleans_up_stale_cycle(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=True)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    decision_plan_obj = state_store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str="user_001.pod_test.daily.v2",
            user_id_str="user_001",
            pod_id_str="pod_test_01",
            account_route_str="DU1",
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 23, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            decision_base_position_map={},
            snapshot_metadata_dict={"strategy_family_str": "stale_stub"},
            strategy_state_dict={},
            decision_book_type_str="incremental_entry_exit_book",
            entry_target_weight_map_dict={"AAPL": 0.2},
            target_weight_map={"AAPL": 0.2},
            exit_asset_set=set(),
            entry_priority_list=["AAPL"],
            cash_reserve_weight_float=0.0,
        )
    )
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )

    detail_dict = run_once(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 14, 40, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )
    roundtrip_decision_plan_obj = state_store_obj.get_decision_plan_by_id(int(decision_plan_obj.decision_plan_id_int or 0))

    assert detail_dict["tick_invoked_bool"] is True
    assert detail_dict["tick_detail_dict"]["expired_decision_plan_count_int"] == 1
    assert roundtrip_decision_plan_obj.status_str == "expired"
