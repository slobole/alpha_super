from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from alpha.live.models import FrozenOrderIntent, FrozenOrderPlan
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.runner import (
    build_order_plans,
    execute_order_plans,
    get_execution_report_summary,
    get_status_summary,
    post_execution_reconcile,
    tick,
)
from alpha.live.state_store import LiveStateStore


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def test_live_runner_roundtrips_build_execute_and_reconcile(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / "pod_test.yaml").write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: user_001.pod_test.daily.v1",
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
                "bootstrap:",
                "  initial_cash_float: 10000.0",
                "risk:",
                "  risk_profile_str: standard",
            ]
        ),
        encoding="utf-8",
    )

    def build_plan_stub(release_obj, as_of_ts, pod_state_obj):
        return FrozenOrderPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            snapshot_metadata_dict={"strategy_family_str": "stub"},
            strategy_state_dict={"trade_id_int": 1},
            order_intent_list=[
                FrozenOrderIntent(
                    asset_str="AAPL",
                    order_class_str="MarketOrder",
                    unit_str="shares",
                    amount_float=5.0,
                    target_bool=False,
                    trade_id_int=1,
                    broker_order_type_str="MOO",
                    sizing_reference_price_float=100.0,
                    portfolio_value_float=10000.0,
                )
            ],
        )

    monkeypatch.setattr("alpha.live.strategy_host.build_order_plan_for_release", build_plan_stub)
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
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )

    build_detail_dict = build_order_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    execute_detail_dict = execute_order_plans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    reconcile_detail_dict = post_execution_reconcile(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    pod_state_obj = state_store_obj.get_pod_state("pod_test_01")

    assert build_detail_dict["created_plan_count_int"] == 1
    assert execute_detail_dict["submitted_plan_count_int"] == 1
    assert reconcile_detail_dict["completed_plan_count_int"] == 1
    assert pod_state_obj is not None
    assert pod_state_obj.position_amount_map["AAPL"] == 5.0


def test_live_runner_persists_fill_prices_and_raw_execution_report(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / "pod_test.yaml").write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: user_001.pod_test.daily.v1",
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
                "bootstrap:",
                "  initial_cash_float: 10000.0",
                "risk:",
                "  risk_profile_str: standard",
            ]
        ),
        encoding="utf-8",
    )

    def build_plan_stub(release_obj, as_of_ts, pod_state_obj):
        return FrozenOrderPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            snapshot_metadata_dict={"strategy_family_str": "stub"},
            strategy_state_dict={"trade_id_int": 1},
            order_intent_list=[
                FrozenOrderIntent(
                    asset_str="AAPL",
                    order_class_str="MarketOrder",
                    unit_str="shares",
                    amount_float=5.0,
                    target_bool=False,
                    trade_id_int=1,
                    broker_order_type_str="MOO",
                    sizing_reference_price_float=100.0,
                    portfolio_value_float=10000.0,
                )
            ],
        )

    monkeypatch.setattr("alpha.live.strategy_host.build_order_plan_for_release", build_plan_stub)
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
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.set_fill_price_multiplier("AAPL", 1.01)

    build_order_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    execute_order_plans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    post_execution_reconcile(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    latest_order_plan_obj = state_store_obj.get_latest_order_plan_for_pod("pod_test_01")
    assert latest_order_plan_obj is not None

    execution_quality_snapshot_obj = state_store_obj.get_execution_quality_snapshot_by_plan(
        int(latest_order_plan_obj.order_plan_id_int or 0)
    )
    assert execution_quality_snapshot_obj is not None
    assert execution_quality_snapshot_obj.reference_notional_float == 500.0
    assert execution_quality_snapshot_obj.actual_notional_float == 505.0
    assert execution_quality_snapshot_obj.slippage_cash_float == 5.0
    assert execution_quality_snapshot_obj.slippage_bps_float == 100.0

    status_summary_dict = get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 36, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    execution_report_summary_dict = get_execution_report_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 36, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )

    assert status_summary_dict["pod_status_dict_list"][0]["latest_fill_timestamp_str"] == "2024-02-01T09:20:00-05:00"
    assert "latest_slippage_cash_float" not in status_summary_dict["pod_status_dict_list"][0]
    assert "latest_slippage_bps_float" not in status_summary_dict["pod_status_dict_list"][0]
    assert execution_report_summary_dict["execution_report_dict_list"][0]["fill_count_int"] == 1
    assert execution_report_summary_dict["execution_report_dict_list"][0]["fill_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "fill_amount_float": 5.0,
            "fill_price_float": 101.0,
            "fill_timestamp_str": "2024-02-01T09:20:00-05:00",
        }
    ]


def test_execute_order_plans_blocks_broker_session_mode_mismatch(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / "pod_test.yaml").write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: user_001.pod_test.daily.v1",
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
                "bootstrap:",
                "  initial_cash_float: 10000.0",
                "risk:",
                "  risk_profile_str: standard",
            ]
        ),
        encoding="utf-8",
    )

    def build_plan_stub(release_obj, as_of_ts, pod_state_obj):
        return FrozenOrderPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            snapshot_metadata_dict={"strategy_family_str": "stub"},
            strategy_state_dict={},
            order_intent_list=[
                FrozenOrderIntent(
                    asset_str="AAPL",
                    order_class_str="MarketOrder",
                    unit_str="shares",
                    amount_float=5.0,
                    target_bool=False,
                    trade_id_int=1,
                    broker_order_type_str="MOO",
                    sizing_reference_price_float=100.0,
                    portfolio_value_float=10000.0,
                )
            ],
        )

    monkeypatch.setattr("alpha.live.strategy_host.build_order_plan_for_release", build_plan_stub)
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
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="live",
    )

    build_order_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )

    execute_detail_dict = execute_order_plans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    submitted_plan_list = state_store_obj.get_submitted_order_plan_list()

    assert execute_detail_dict["submitted_plan_count_int"] == 0
    assert execute_detail_dict["blocked_plan_count_int"] == 1
    assert submitted_plan_list == []


def test_tick_is_idempotent_for_repeated_scheduler_wakes(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / "pod_test.yaml").write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: user_001.pod_test.daily.v1",
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
                "bootstrap:",
                "  initial_cash_float: 10000.0",
                "risk:",
                "  risk_profile_str: standard",
            ]
        ),
        encoding="utf-8",
    )

    def build_plan_stub(release_obj, as_of_ts, pod_state_obj):
        return FrozenOrderPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            snapshot_metadata_dict={"strategy_family_str": "stub"},
            strategy_state_dict={"trade_id_int": 1},
            order_intent_list=[
                FrozenOrderIntent(
                    asset_str="AAPL",
                    order_class_str="MarketOrder",
                    unit_str="shares",
                    amount_float=5.0,
                    target_bool=False,
                    trade_id_int=1,
                    broker_order_type_str="MOO",
                    sizing_reference_price_float=100.0,
                    portfolio_value_float=10000.0,
                )
            ],
        )

    monkeypatch.setattr("alpha.live.strategy_host.build_order_plan_for_release", build_plan_stub)
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
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )

    build_order_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )

    first_tick_detail_dict = tick(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )
    second_tick_detail_dict = tick(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )

    assert first_tick_detail_dict["lease_acquired_bool"] is True
    assert first_tick_detail_dict["submitted_plan_count_int"] == 1
    assert second_tick_detail_dict["lease_acquired_bool"] is True
    assert second_tick_detail_dict["submitted_plan_count_int"] == 0


def test_status_summary_reports_next_action_and_reason(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / "pod_test.yaml").write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: user_001.pod_test.daily.v1",
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
                "bootstrap:",
                "  initial_cash_float: 10000.0",
                "risk:",
                "  risk_profile_str: standard",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-30"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    status_summary_dict = get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 12, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )

    assert status_summary_dict["enabled_pod_count_int"] == 1
    assert status_summary_dict["pod_status_dict_list"][0]["next_action_str"] == "wait"
    assert status_summary_dict["pod_status_dict_list"][0]["reason_code_str"] == "snapshot_window_expired"


def test_tick_writes_structured_event_log(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    log_path_str = str((tmp_path / "logs" / "live_events.jsonl").resolve())
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / "pod_test.yaml").write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: user_001.pod_test.daily.v1",
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
                "bootstrap:",
                "  initial_cash_float: 10000.0",
                "risk:",
                "  risk_profile_str: standard",
            ]
        ),
        encoding="utf-8",
    )

    def build_plan_stub(release_obj, as_of_ts, pod_state_obj):
        return FrozenOrderPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            snapshot_metadata_dict={"strategy_family_str": "stub"},
            strategy_state_dict={"trade_id_int": 1},
            order_intent_list=[
                FrozenOrderIntent(
                    asset_str="AAPL",
                    order_class_str="MarketOrder",
                    unit_str="shares",
                    amount_float=5.0,
                    target_bool=False,
                    trade_id_int=1,
                    broker_order_type_str="MOO",
                    sizing_reference_price_float=100.0,
                    portfolio_value_float=10000.0,
                )
            ],
        )

    monkeypatch.setattr("alpha.live.strategy_host.build_order_plan_for_release", build_plan_stub)
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
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )

    tick(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
        log_path_str=log_path_str,
    )
    tick(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
        log_path_str=log_path_str,
    )

    log_line_list = Path(log_path_str).read_text(encoding="utf-8").strip().splitlines()
    event_name_list = [json.loads(log_line_str)["event_name_str"] for log_line_str in log_line_list]

    assert "build_plan_created" in event_name_list
    assert "submit_plan_completed" in event_name_list
    assert "post_execution_reconcile_completed" in event_name_list
    assert "tick_completed" in event_name_list
