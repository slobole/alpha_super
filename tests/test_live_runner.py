from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from alpha.live.models import DecisionPlan
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.runner import (
    build_decision_plans,
    build_vplans,
    get_execution_report_summary,
    get_status_summary,
    post_execution_reconcile,
    show_vplan_summary,
    submit_ready_vplans,
    tick,
)
from alpha.live.state_store_v2 import LiveStateStore


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def _write_manifest(
    root_path_obj: Path,
    auto_submit_enabled_bool: bool,
    pod_budget_fraction_float: float = 0.5,
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
                f"  pod_budget_fraction_float: {pod_budget_fraction_float}",
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
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
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


def _build_decision_plan_with_position_drift_stub(release_obj, as_of_ts, pod_state_obj):
    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        decision_base_position_map={"AAPL": 5.0},
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


def _build_full_target_weight_decision_plan_stub(release_obj, as_of_ts, pod_state_obj):
    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        decision_base_position_map={"AAPL": 4.0},
        snapshot_metadata_dict={"strategy_family_str": "full_target_stub"},
        strategy_state_dict={},
        decision_book_type_str="full_target_weight_book",
        full_target_weight_map_dict={"AAPL": 0.3, "TLT": 0.2},
        target_weight_map={"AAPL": 0.3, "TLT": 0.2},
        cash_reserve_weight_float=0.5,
        preserve_untouched_positions_bool=False,
        rebalance_omitted_assets_to_zero_bool=True,
    )


def test_live_runner_manual_vplan_flow(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
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

    build_detail_dict = build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    vplan_detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    show_vplan_detail_dict = show_vplan_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        pod_id_str="pod_test_01",
    )
    status_summary_dict = get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )

    assert build_detail_dict["created_decision_plan_count_int"] == 1
    assert vplan_detail_dict["created_vplan_count_int"] == 1
    assert show_vplan_detail_dict["vplan_dict_list"][0]["vplan_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "decision_base_share_float": 0.0,
            "current_share_float": 0.0,
            "drift_share_float": 0.0,
            "target_share_float": 10.0,
            "order_delta_share_float": 10.0,
            "live_reference_price_float": 100.0,
            "estimated_target_notional_float": 1000.0,
            "warning_bool": False,
        }
    ]
    assert show_vplan_detail_dict["vplan_dict_list"][0]["warning_row_dict_list"] == []
    assert status_summary_dict["pod_status_dict_list"][0]["next_action_str"] == "review_vplan"
    assert status_summary_dict["pod_status_dict_list"][0]["reason_code_str"] == "vplan_ready"


def test_build_vplan_warns_on_position_mismatch_but_continues(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr(
        "alpha.live.strategy_host.build_decision_plan_for_release",
        _build_decision_plan_with_position_drift_stub,
    )
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
        position_amount_map={"AAPL": 3.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    vplan_detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    show_vplan_detail_dict = show_vplan_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        pod_id_str="pod_test_01",
    )

    assert vplan_detail_dict["created_vplan_count_int"] == 1
    assert vplan_detail_dict["blocked_action_count_int"] == 0
    assert vplan_detail_dict["warning_count_map_dict"] == {"position_reconciliation_warning": 1}
    assert show_vplan_detail_dict["vplan_dict_list"][0]["vplan_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "decision_base_share_float": 5.0,
            "current_share_float": 3.0,
            "drift_share_float": -2.0,
            "target_share_float": 10.0,
            "order_delta_share_float": 7.0,
            "live_reference_price_float": 100.0,
            "estimated_target_notional_float": 1000.0,
            "warning_bool": True,
        }
    ]
    assert show_vplan_detail_dict["vplan_dict_list"][0]["warning_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "decision_base_share_float": 5.0,
            "current_share_float": 3.0,
            "drift_share_float": -2.0,
        }
    ]


def test_live_runner_auto_tick_builds_submits_and_reconciles(tmp_path: Path, monkeypatch):
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

    first_tick_detail_dict = tick(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )
    second_tick_detail_dict = tick(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )
    third_tick_detail_dict = tick(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )
    status_summary_dict = get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    execution_report_summary_dict = get_execution_report_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )

    assert first_tick_detail_dict["created_decision_plan_count_int"] == 1
    assert second_tick_detail_dict["created_vplan_count_int"] == 1
    assert second_tick_detail_dict["submitted_vplan_count_int"] == 1
    assert second_tick_detail_dict["completed_vplan_count_int"] == 0
    assert third_tick_detail_dict["completed_vplan_count_int"] == 1
    assert status_summary_dict["pod_status_dict_list"][0]["latest_decision_plan_status_str"] == "completed"
    assert status_summary_dict["pod_status_dict_list"][0]["latest_vplan_status_str"] == "completed"
    assert execution_report_summary_dict["execution_report_dict_list"][0]["fill_row_dict_list"][0]["asset_str"] == "AAPL"


def test_submit_vplan_explicit_manual_path(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
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

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_test_01")
    assert latest_vplan_obj is not None

    submit_detail_dict = submit_ready_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        manual_only_bool=False,
        vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
    )
    reconcile_detail_dict = post_execution_reconcile(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    assert submit_detail_dict["submitted_vplan_count_int"] == 1
    assert reconcile_detail_dict["completed_vplan_count_int"] == 1


def test_build_vplan_blocks_when_live_price_is_missing(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
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
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    vplan_detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )

    assert vplan_detail_dict["created_vplan_count_int"] == 0
    assert vplan_detail_dict["blocked_action_count_int"] == 1
    assert vplan_detail_dict["reason_count_map_dict"] == {"missing_live_price": 1}


def test_build_vplan_full_target_weight_book_rebalances_omitted_holdings(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live_full_target.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr(
        "alpha.live.strategy_host.build_decision_plan_for_release",
        _build_full_target_weight_decision_plan_stub,
    )
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
        position_amount_map={"AAPL": 4.0, "MSFT": 2.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0, "TLT": 50.0, "MSFT": 200.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    vplan_detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    show_vplan_detail_dict = show_vplan_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        pod_id_str="pod_test_01",
    )

    assert vplan_detail_dict["created_vplan_count_int"] == 1
    assert show_vplan_detail_dict["vplan_dict_list"][0]["vplan_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "decision_base_share_float": 4.0,
            "current_share_float": 4.0,
            "drift_share_float": 0.0,
            "target_share_float": 15.0,
            "order_delta_share_float": 11.0,
            "live_reference_price_float": 100.0,
            "estimated_target_notional_float": 1500.0,
            "warning_bool": False,
        },
        {
            "asset_str": "MSFT",
            "decision_base_share_float": 0.0,
            "current_share_float": 2.0,
            "drift_share_float": 2.0,
            "target_share_float": 0.0,
            "order_delta_share_float": -2.0,
            "live_reference_price_float": 200.0,
            "estimated_target_notional_float": 0.0,
            "warning_bool": True,
        },
        {
            "asset_str": "TLT",
            "decision_base_share_float": 0.0,
            "current_share_float": 0.0,
            "drift_share_float": 0.0,
            "target_share_float": 20.0,
            "order_delta_share_float": 20.0,
            "live_reference_price_float": 50.0,
            "estimated_target_notional_float": 1000.0,
            "warning_bool": False,
        },
    ]
