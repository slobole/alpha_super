from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import alpha.live.doctor as doctor_module
import alpha.live.runner as runner_module
from alpha.live.models import DecisionPlan, PodState
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.runner import BrokerAdapterResolver
from alpha.live.state_store_v2 import LiveStateStore


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def _write_release_manifest(
    releases_root_path_obj: Path,
    *,
    mode_str: str,
    pod_id_str: str,
    account_route_str: str,
    strategy_import_str: str,
    data_profile_str: str,
    signal_clock_str: str,
    execution_policy_str: str,
) -> None:
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / f"{pod_id_str}.yaml").write_text(
        "\n".join(
            [
                "identity:",
                f"  release_id: user_001.{pod_id_str}.v1",
                "  user_id: user_001",
                f"  pod_id: {pod_id_str}",
                "deployment:",
                f"  mode: {mode_str}",
                "  enabled_bool: true",
                "broker:",
                f"  account_route: {account_route_str}",
                "  host_str: 127.0.0.1",
                "  port_int: 7497",
                "  client_id_int: 31",
                "  timeout_seconds_float: 4.0",
                "strategy:",
                f"  strategy_import_str: {strategy_import_str}",
                f"  data_profile_str: {data_profile_str}",
                "  params: {}",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                f"  signal_clock_str: {signal_clock_str}",
                f"  execution_policy_str: {execution_policy_str}",
                "execution:",
                "  pod_budget_fraction_float: 0.5",
                "  auto_submit_enabled_bool: true",
                "bootstrap:",
                "  initial_cash_float: 100000.0",
                "risk:",
                "  risk_profile_str: standard",
            ]
        ),
        encoding="utf-8",
    )


def _write_live_taa_manifest(releases_root_path_obj: Path) -> None:
    _write_release_manifest(
        releases_root_path_obj,
        mode_str="live",
        pod_id_str="pod_taa_live_01",
        account_route_str="U1",
        strategy_import_str="strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
        data_profile_str="norgate_eod_etf_plus_vix_helper",
        signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open",
    )


def _write_paper_dv2_manifest(releases_root_path_obj: Path) -> None:
    _write_release_manifest(
        releases_root_path_obj,
        mode_str="paper",
        pod_id_str="pod_dv2_paper_01",
        account_route_str="DU1",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        data_profile_str="norgate_eod_sp500_pit",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
    )


def _build_taa_decision_plan_stub(release_obj, as_of_ts, pod_state_obj) -> DecisionPlan:
    del as_of_ts, pod_state_obj
    signal_timestamp_ts = datetime(2026, 5, 29, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ)
    submission_timestamp_ts = datetime(2026, 6, 1, 9, 23, 30, tzinfo=MARKET_TIMEZONE_OBJ)
    target_execution_timestamp_ts = datetime(2026, 6, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ)
    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=signal_timestamp_ts,
        submission_timestamp_ts=submission_timestamp_ts,
        target_execution_timestamp_ts=target_execution_timestamp_ts,
        execution_policy_str=release_obj.execution_policy_str,
        decision_base_position_map={},
        snapshot_metadata_dict={
            "strategy_family_str": "taa_df",
            "norgate_data_profile_str": release_obj.data_profile_str,
            "norgate_snapshot_date_str": "2026-05-29",
            "raw_month_end_label_str": "2026-05-31",
            "resolved_signal_session_date_str": "2026-05-29",
            "available_price_last_date_str": "2026-05-29",
            "timing_resolution_reason_str": (
                "calendar_month_end_label_resolved_to_last_tradable_session"
            ),
        },
        strategy_state_dict={},
        decision_book_type_str="full_target_weight_book",
        full_target_weight_map_dict={"GLD": 0.5, "TQQQ": 0.5},
        target_weight_map={"GLD": 0.5, "TQQQ": 0.5},
        preserve_untouched_positions_bool=False,
        rebalance_omitted_assets_to_zero_bool=True,
    )


def _build_paper_decision_plan_stub(release_obj, as_of_ts, pod_state_obj) -> DecisionPlan:
    del as_of_ts, pod_state_obj
    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 23, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str=release_obj.execution_policy_str,
        decision_base_position_map={},
        snapshot_metadata_dict={
            "strategy_family_str": "dv2",
            "norgate_data_profile_str": release_obj.data_profile_str,
        },
        strategy_state_dict={},
        decision_book_type_str="incremental_entry_exit_book",
        entry_target_weight_map_dict={"AAPL": 0.2},
        target_weight_map={"AAPL": 0.2},
        entry_priority_list=["AAPL"],
    )


def _clear_snapshot_env(monkeypatch) -> None:
    monkeypatch.delenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", raising=False)
    monkeypatch.delenv("NORGATE_SNAPSHOT_ROOT", raising=False)


def _doctor_config_kwargs(
    tmp_path: Path,
    releases_root_path_obj: Path,
    *,
    explicit_bool: bool = True,
) -> dict[str, object]:
    config_env_path_obj = tmp_path / "config.env"
    config_env_path_obj.write_text(
        f"NORGATE_RELEASES_ROOT={releases_root_path_obj}\n",
        encoding="utf-8",
    )
    return {
        "releases_root_explicit_bool": explicit_bool,
        "config_env_path_str": str(config_env_path_obj),
        "loaded_config_env_dict": {
            "NORGATE_RELEASES_ROOT": str(releases_root_path_obj),
        },
    }


def _table_count_map(db_path_obj: Path) -> dict[str, int]:
    with sqlite3.connect(db_path_obj) as connection_obj:
        table_name_list = [
            row_obj[0]
            for row_obj in connection_obj.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        return {
            table_name_str: int(
                connection_obj.execute(f"SELECT COUNT(*) FROM {table_name_str}").fetchone()[0]
            )
            for table_name_str in table_name_list
        }


def _stub_snapshot_sync_ready(**kwargs) -> dict[str, object]:
    return {
        "status_str": "ready",
        "reason_code_str": "local_snapshot_ready",
        "required_profile_list": ["norgate_eod_sp500_pit"],
    }


def test_doctor_blocks_when_config_env_is_missing(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        config_env_path_str=str(tmp_path / "missing_config.env"),
        loaded_config_env_dict={
            "NORGATE_RELEASES_ROOT": str(releases_root_path_obj),
        },
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "config_env_missing"


def test_doctor_blocks_when_norgate_releases_root_missing(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    config_env_path_obj = tmp_path / "config.env"
    config_env_path_obj.write_text("ALPHA_USE_NORGATE_SNAPSHOT_BOOL=false\n", encoding="utf-8")

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(tmp_path / "releases" / "user_001"),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        config_env_path_str=str(config_env_path_obj),
        loaded_config_env_dict={"ALPHA_USE_NORGATE_SNAPSHOT_BOOL": "false"},
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert (
        detail_dict["component_result_dict_list"][-1]["reason_code_str"]
        == "norgate_releases_root_missing"
    )


def test_doctor_blocks_when_env_release_root_path_is_missing(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    missing_root_path_obj = tmp_path / "missing_releases" / "user_001"
    config_env_path_obj = tmp_path / "config.env"
    config_env_path_obj.write_text(
        f"NORGATE_RELEASES_ROOT={missing_root_path_obj}\n",
        encoding="utf-8",
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=None,
        releases_root_explicit_bool=False,
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        config_env_path_str=str(config_env_path_obj),
        loaded_config_env_dict={
            "NORGATE_RELEASES_ROOT": str(missing_root_path_obj),
        },
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "release_root_path_missing"


def test_doctor_blocks_when_explicit_release_root_mismatches_config_env(
    tmp_path: Path,
    monkeypatch,
):
    _clear_snapshot_env(monkeypatch)
    env_releases_root_path_obj = tmp_path / "env_releases" / "user_001"
    cli_releases_root_path_obj = tmp_path / "cli_releases" / "user_001"
    _write_paper_dv2_manifest(env_releases_root_path_obj)
    _write_paper_dv2_manifest(cli_releases_root_path_obj)
    config_env_path_obj = tmp_path / "config.env"
    config_env_path_obj.write_text(
        f"NORGATE_RELEASES_ROOT={env_releases_root_path_obj}\n",
        encoding="utf-8",
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(cli_releases_root_path_obj),
        releases_root_explicit_bool=True,
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        config_env_path_str=str(config_env_path_obj),
        loaded_config_env_dict={
            "NORGATE_RELEASES_ROOT": str(env_releases_root_path_obj),
        },
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "release_root_mismatch"
    assert detail_dict["config_release_root_dict"]["release_root_match_bool"] is False


def test_doctor_resolves_relative_env_release_root_from_config_location(
    tmp_path: Path,
    monkeypatch,
):
    _clear_snapshot_env(monkeypatch)
    repo_path_obj = tmp_path / "repo"
    releases_root_path_obj = repo_path_obj / "alpha" / "live" / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    config_env_path_obj = repo_path_obj / "config.env"
    config_env_path_obj.write_text(
        "NORGATE_RELEASES_ROOT=alpha/live/releases/user_001\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": False,
            "reason_code_str": "snapshot_not_ready_for_session",
        },
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=None,
        releases_root_explicit_bool=False,
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        config_env_path_str=str(config_env_path_obj),
        loaded_config_env_dict={
            "NORGATE_RELEASES_ROOT": "alpha/live/releases/user_001",
        },
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
    )

    assert detail_dict["overall_verdict_str"] == "WAIT"
    config_release_root_dict = detail_dict["config_release_root_dict"]
    assert config_release_root_dict["release_root_source_str"] == "config_env"
    assert config_release_root_dict["release_root_match_bool"] is True
    assert config_release_root_dict["env_release_root_resolved_str"] == str(
        releases_root_path_obj.resolve()
    )
    assert config_release_root_dict["selected_release_source_path_str"].endswith(
        "pod_dv2_paper_01.yaml"
    )


def test_doctor_json_does_not_include_secret_config_values(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    config_env_path_obj = tmp_path / "config.env"
    config_env_path_obj.write_text(
        f"NORGATE_RELEASES_ROOT={releases_root_path_obj}\nNORGATE_API_TOKEN=super-secret-token\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": False,
            "reason_code_str": "snapshot_not_ready_for_session",
        },
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=None,
        releases_root_explicit_bool=False,
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        config_env_path_str=str(config_env_path_obj),
        loaded_config_env_dict={
            "NORGATE_RELEASES_ROOT": str(releases_root_path_obj),
            "NORGATE_API_TOKEN": "super-secret-token",
        },
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
    )

    assert detail_dict["overall_verdict_str"] == "WAIT"
    assert "super-secret-token" not in json.dumps(detail_dict)
    assert "NORGATE_API_TOKEN" in json.dumps(detail_dict)


def test_doctor_passes_live_taa_and_keeps_state_read_only(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_live_taa_manifest(releases_root_path_obj)
    state_store_obj = LiveStateStore(str(tmp_path / "pod.sqlite3"))
    before_count_map = _table_count_map(tmp_path / "pod.sqlite3")

    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2026-05-29"),
    )
    monkeypatch.setattr(
        doctor_module.strategy_host,
        "build_decision_plan_for_release",
        _build_taa_decision_plan_stub,
    )

    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        "U1",
        cash_float=100_000.0,
        total_value_float=100_000.0,
        session_mode_str="live",
        snapshot_timestamp_ts=datetime(2026, 6, 1, 13, 20, tzinfo=UTC),
    )
    broker_adapter_obj.seed_live_price_snapshot(
        "U1",
        {"GLD": 100.0, "TQQQ": 50.0},
        snapshot_timestamp_ts=datetime(2026, 6, 1, 13, 20, tzinfo=UTC),
    )
    resolver_obj = BrokerAdapterResolver(broker_adapter_obj=broker_adapter_obj)

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="live",
        as_of_ts=datetime(2026, 6, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_taa_live_01",
        state_store_obj=state_store_obj,
        broker_adapter_resolver_obj=resolver_obj,
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "PASS"
    assert detail_dict["feature_name_str"] == "doctor"
    manifest_qualification_dict = detail_dict["manifest_qualification_dict"]
    assert manifest_qualification_dict["qualified_bool"] is True
    assert manifest_qualification_dict["strategy_import_str"] == (
        "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash"
    )
    assert manifest_qualification_dict["account_route_placeholder_bool"] is False
    assert detail_dict["scheduler_gate_dict"]["reason_code_str"] == "carry_forward_snapshot_ready"
    metadata_dict = detail_dict["decision_plan_dict"]["snapshot_metadata_dict"]
    assert metadata_dict["raw_month_end_label_str"] == "2026-05-31"
    assert metadata_dict["resolved_signal_session_date_str"] == "2026-05-29"
    assert detail_dict["decision_plan_dict"]["signal_timestamp_str"] == "2026-05-29T16:00:00-04:00"
    assert detail_dict["decision_plan_dict"]["submission_timestamp_str"] == "2026-06-01T09:23:30-04:00"
    assert detail_dict["decision_plan_dict"]["target_execution_timestamp_str"] == "2026-06-01T09:30:00-04:00"
    assert detail_dict["broker_dict"]["expected_account_visible_bool"] is True
    assert detail_dict["vplan_preview_dict"]["broker_order_request_count_int"] == 2
    assert detail_dict["vplan_preview_dict"]["target_share_map"] == {"GLD": 250.0, "TQQQ": 500.0}
    assert detail_dict["vplan_preview_dict"]["order_delta_map"] == {"GLD": 250.0, "TQQQ": 500.0}
    order_type_set = {
        order_request_dict["broker_order_type_str"]
        for order_request_dict in detail_dict["vplan_preview_dict"]["broker_order_request_dict_list"]
    }
    assert order_type_set == {"MOO"}
    assert _table_count_map(tmp_path / "pod.sqlite3") == before_count_map


def test_doctor_blocks_unsupported_manifest_strategy(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_release_manifest(
        releases_root_path_obj,
        mode_str="paper",
        pod_id_str="pod_bad_strategy_01",
        account_route_str="DU1",
        strategy_import_str="strategies.unknown:Strategy",
        data_profile_str="norgate_eod_sp500_pit",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_bad_strategy_01",
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["manifest_qualification_dict"]["qualified_bool"] is False
    assert detail_dict["component_result_dict_list"][-1]["component_name_str"] == "manifest_qualification"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "manifest_error"
    assert "Unsupported strategy_import_str" in detail_dict["component_result_dict_list"][-1]["detail_str"]


def test_doctor_blocks_placeholder_manifest_account_route(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_release_manifest(
        releases_root_path_obj,
        mode_str="paper",
        pod_id_str="pod_placeholder_account_01",
        account_route_str="DU_TODO",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        data_profile_str="norgate_eod_sp500_pit",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_placeholder_account_01",
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["manifest_qualification_dict"]["qualified_bool"] is False
    assert detail_dict["component_result_dict_list"][-1]["component_name_str"] == "manifest_qualification"
    assert "real IBKR account/subaccount route" in detail_dict["component_result_dict_list"][-1]["detail_str"]


def test_doctor_wait_skips_decision_and_order_preview(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)

    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": False,
            "reason_code_str": "snapshot_not_ready_for_session",
            "latest_heartbeat_session_date_str": "2024-01-30",
        },
    )

    def fail_build(*args, **kwargs):
        raise AssertionError("DecisionPlan build must not run for WAIT doctor.")

    monkeypatch.setattr(
        doctor_module.strategy_host,
        "build_decision_plan_for_release",
        fail_build,
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 1, 31, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "WAIT"
    assert detail_dict["decision_plan_dict"] == {}
    assert detail_dict["vplan_preview_dict"] == {}


def test_doctor_waits_when_persisted_decision_plan_is_active(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    db_path_obj = tmp_path / "pod.sqlite3"
    state_store_obj = LiveStateStore(str(db_path_obj))
    state_store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str="user_001.pod_dv2_paper_01.v1",
            user_id_str="user_001",
            pod_id_str="pod_dv2_paper_01",
            account_route_str="DU1",
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 23, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            decision_base_position_map={},
            snapshot_metadata_dict={"norgate_data_profile_str": "norgate_eod_sp500_pit"},
            strategy_state_dict={},
            decision_book_type_str="incremental_entry_exit_book",
            entry_target_weight_map_dict={"AAPL": 0.2},
            target_weight_map={"AAPL": 0.2},
            entry_priority_list=["AAPL"],
        )
    )

    def fail_gate(*args, **kwargs):
        raise AssertionError("Scheduler gate must not run while persisted lifecycle is active.")

    monkeypatch.setattr(doctor_module.scheduler_utils, "evaluate_build_gate_dict", fail_gate)

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        db_path_str=str(db_path_obj),
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "WAIT"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "persisted_lifecycle_active"
    assert detail_dict["decision_plan_dict"] == {}
    assert detail_dict["vplan_preview_dict"] == {}


def test_doctor_blocks_when_persisted_decision_plan_is_expired(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    db_path_obj = tmp_path / "pod.sqlite3"
    state_store_obj = LiveStateStore(str(db_path_obj))
    state_store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str="user_001.pod_dv2_paper_01.v1",
            user_id_str="user_001",
            pod_id_str="pod_dv2_paper_01",
            account_route_str="DU1",
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 23, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            decision_base_position_map={},
            snapshot_metadata_dict={"norgate_data_profile_str": "norgate_eod_sp500_pit"},
            strategy_state_dict={},
            decision_book_type_str="incremental_entry_exit_book",
            entry_target_weight_map_dict={"AAPL": 0.2},
            target_weight_map={"AAPL": 0.2},
            entry_priority_list=["AAPL"],
        )
    )

    def fail_gate(*args, **kwargs):
        raise AssertionError("Scheduler gate must not run while persisted lifecycle is expired.")

    monkeypatch.setattr(doctor_module.scheduler_utils, "evaluate_build_gate_dict", fail_gate)

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 20, 0, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        db_path_str=str(db_path_obj),
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "persisted_lifecycle_expired"
    assert detail_dict["decision_plan_dict"] == {}
    assert detail_dict["vplan_preview_dict"] == {}


def test_doctor_blocks_when_account_is_not_visible(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": True,
            "reason_code_str": "snapshot_ready",
            "latest_heartbeat_session_date_str": "2024-01-31",
        },
    )
    monkeypatch.setattr(
        doctor_module.strategy_host,
        "build_decision_plan_for_release",
        _build_paper_decision_plan_stub,
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "account_not_visible"


def test_doctor_blocks_when_managed_accounts_are_unavailable(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": True,
            "reason_code_str": "snapshot_ready",
            "latest_heartbeat_session_date_str": "2024-01-31",
        },
    )
    monkeypatch.setattr(
        doctor_module.strategy_host,
        "build_decision_plan_for_release",
        _build_paper_decision_plan_stub,
    )

    class UnavailableAccountsBrokerAdapter(StubBrokerAdapter):
        def get_visible_account_route_set(self):
            return None

    broker_adapter_obj = UnavailableAccountsBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        "DU1",
        cash_float=100_000.0,
        total_value_float=100_000.0,
        session_mode_str="paper",
        snapshot_timestamp_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        norgate_snapshot_sync_checker_fn=_stub_snapshot_sync_ready,
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=broker_adapter_obj),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "broker_accounts_unavailable"
    assert detail_dict["broker_dict"]["visible_account_route_list"] is None


def test_doctor_blocks_decision_plan_build_error(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": True,
            "reason_code_str": "snapshot_ready",
            "latest_heartbeat_session_date_str": "2024-01-31",
        },
    )

    def raise_build_error(release_obj, as_of_ts, pod_state_obj):
        raise RuntimeError("TAA live DecisionPlan timing mismatch: test")

    monkeypatch.setattr(
        doctor_module.strategy_host,
        "build_decision_plan_for_release",
        raise_build_error,
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "decision_plan_build_error"
    assert "timing mismatch" in detail_dict["component_result_dict_list"][-1]["detail_str"]


def test_doctor_blocks_stale_decision_plan_before_broker_probe(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": True,
            "reason_code_str": "snapshot_ready",
            "latest_heartbeat_session_date_str": "2024-01-31",
        },
    )

    def build_stale_plan(release_obj, as_of_ts, pod_state_obj):
        return replace(
            _build_paper_decision_plan_stub(release_obj, as_of_ts, pod_state_obj),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        )

    monkeypatch.setattr(
        doctor_module.strategy_host,
        "build_decision_plan_for_release",
        build_stale_plan,
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 20, 0, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        norgate_snapshot_sync_checker_fn=_stub_snapshot_sync_ready,
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "submission_window_expired"
    assert detail_dict["broker_dict"] == {}
    assert detail_dict["vplan_preview_dict"] == {}


@pytest.mark.parametrize(
    ("snapshot_metadata_dict", "expected_reason_code_str"),
    [
        ({"strategy_family_str": "dv2"}, "decision_plan_norgate_provenance_missing"),
        (
            {
                "strategy_family_str": "dv2",
                "norgate_data_profile_str": "norgate_eod_ndx_pit",
            },
            "decision_plan_norgate_profile_mismatch",
        ),
    ],
)
def test_doctor_blocks_invalid_norgate_provenance_before_broker_probe(
    tmp_path: Path,
    monkeypatch,
    snapshot_metadata_dict: dict[str, object],
    expected_reason_code_str: str,
):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": True,
            "reason_code_str": "snapshot_ready",
            "latest_heartbeat_session_date_str": "2024-01-31",
        },
    )

    def build_bad_provenance_plan(release_obj, as_of_ts, pod_state_obj):
        return replace(
            _build_paper_decision_plan_stub(release_obj, as_of_ts, pod_state_obj),
            snapshot_metadata_dict=snapshot_metadata_dict,
        )

    monkeypatch.setattr(
        doctor_module.strategy_host,
        "build_decision_plan_for_release",
        build_bad_provenance_plan,
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        norgate_snapshot_sync_checker_fn=_stub_snapshot_sync_ready,
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == expected_reason_code_str
    assert detail_dict["broker_dict"] == {}
    assert detail_dict["vplan_preview_dict"] == {}


def test_doctor_blocks_when_norgate_snapshot_sync_blocks_live_build(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": True,
            "reason_code_str": "snapshot_ready",
            "latest_heartbeat_session_date_str": "2024-01-31",
        },
    )

    def fail_build(*args, **kwargs):
        raise AssertionError("DecisionPlan build must not run when snapshot sync blocks.")

    monkeypatch.setattr(
        doctor_module.strategy_host,
        "build_decision_plan_for_release",
        fail_build,
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        norgate_snapshot_sync_checker_fn=lambda **kwargs: {
            "status_str": "local_snapshot_only",
            "reason_code_str": "api_config_missing",
        },
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "api_config_missing"
    assert detail_dict["decision_plan_dict"] == {}


def test_doctor_blocks_snapshot_root_missing_for_truthy_env_alias(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "1")
    monkeypatch.delenv("NORGATE_SNAPSHOT_ROOT", raising=False)

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(tmp_path / "missing"),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=StubBrokerAdapter()),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "snapshot_root_missing"


def test_doctor_blocks_when_broker_snapshot_has_open_orders(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": True,
            "reason_code_str": "snapshot_ready",
            "latest_heartbeat_session_date_str": "2024-01-31",
        },
    )
    monkeypatch.setattr(
        doctor_module.strategy_host,
        "build_decision_plan_for_release",
        _build_paper_decision_plan_stub,
    )

    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        "DU1",
        cash_float=100_000.0,
        total_value_float=100_000.0,
        session_mode_str="paper",
        snapshot_timestamp_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
    )
    broker_adapter_obj._snapshot_map["DU1"] = replace(
        broker_adapter_obj.get_account_snapshot("DU1"),
        open_order_id_list=["101", "102"],
    )
    broker_adapter_obj.seed_live_price_snapshot(
        "DU1",
        {"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=broker_adapter_obj),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["broker_dict"]["open_order_id_list"] == ["101", "102"]
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "open_orders_present"
    assert detail_dict["vplan_preview_dict"] == {}


def test_doctor_blocks_missing_live_reference_price(tmp_path: Path, monkeypatch):
    _clear_snapshot_env(monkeypatch)
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": True,
            "reason_code_str": "snapshot_ready",
            "latest_heartbeat_session_date_str": "2024-01-31",
        },
    )
    monkeypatch.setattr(
        doctor_module.strategy_host,
        "build_decision_plan_for_release",
        _build_paper_decision_plan_stub,
    )

    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        "DU1",
        cash_float=100_000.0,
        total_value_float=100_000.0,
        session_mode_str="paper",
        snapshot_timestamp_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
    )
    broker_adapter_obj.seed_live_price_snapshot(
        "DU1",
        {},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
    )

    detail_dict = doctor_module.compute_doctor_verdict(
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        as_of_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
        pod_id_str="pod_dv2_paper_01",
        broker_adapter_resolver_obj=BrokerAdapterResolver(broker_adapter_obj=broker_adapter_obj),
        **_doctor_config_kwargs(tmp_path, releases_root_path_obj),
    )

    assert detail_dict["overall_verdict_str"] == "BLOCK"
    assert detail_dict["component_result_dict_list"][-1]["reason_code_str"] == "missing_live_price"
    assert detail_dict["component_result_dict_list"][-1]["missing_asset_list"] == ["AAPL"]


def _sample_cli_doctor_detail_dict(
    overall_verdict_str: str,
    *,
    command_name_str: str = "doctor",
) -> dict[str, object]:
    return {
        "feature_name_str": "doctor",
        "command_name_str": command_name_str,
        "overall_verdict_str": overall_verdict_str,
        "mode_str": "paper",
        "pod_id_str": "pod_dv2_paper_01",
        "as_of_timestamp_str": "2024-02-01T13:20:00+00:00",
        "component_result_dict_list": [
            {
                "component_name_str": "scheduler_gate",
                "status_str": overall_verdict_str,
                "reason_code_str": "snapshot_ready",
                "detail_str": "Scheduler/data gate is ready.",
            }
        ],
        "config_release_root_dict": {
            "config_env_path_str": "C:\\repo\\config.env",
            "config_env_exists_bool": True,
            "env_release_root_raw_str": "alpha/live/releases/user_001",
            "env_release_root_resolved_str": "C:\\repo\\alpha\\live\\releases\\user_001",
            "effective_release_root_resolved_str": "C:\\repo\\alpha\\live\\releases\\user_001",
            "release_root_source_str": "config_env",
            "release_root_match_bool": True,
            "selected_release_source_path_str": (
                "C:\\repo\\alpha\\live\\releases\\user_001\\pod_dv2_paper_01.yaml"
            ),
        },
        "manifest_qualification_dict": {
            "qualification_status_str": "PASS",
            "qualification_reason_code_str": "manifest_qualified",
            "qualified_bool": True,
            "source_path_str": (
                "C:\\repo\\alpha\\live\\releases\\user_001\\pod_dv2_paper_01.yaml"
            ),
            "release_id_str": "user_001.pod_dv2_paper_01.v1",
            "user_id_str": "user_001",
            "pod_id_str": "pod_dv2_paper_01",
            "enabled_bool": True,
            "mode_str": "paper",
            "account_route_str": "DU1",
            "account_route_placeholder_bool": False,
            "broker_host_str": "127.0.0.1",
            "broker_port_int": 7497,
            "broker_client_id_int": 31,
            "broker_timeout_seconds_float": 4.0,
            "strategy_import_str": "strategies.dv2.strategy_mr_dv2:DVO2Strategy",
            "data_profile_str": "norgate_eod_sp500_pit",
            "signal_clock_str": "eod_snapshot_ready",
            "execution_policy_str": "next_open_moo",
            "session_calendar_id_str": "XNYS",
            "risk_profile_str": "standard",
            "pod_budget_fraction_float": 0.5,
            "auto_submit_enabled_bool": True,
        },
        "release_dict": {
            "release_id_str": "user_001.pod_dv2_paper_01.v1",
            "source_path_str": (
                "C:\\repo\\alpha\\live\\releases\\user_001\\pod_dv2_paper_01.yaml"
            ),
            "pod_id_str": "pod_dv2_paper_01",
            "strategy_import_str": "strategies.dv2.strategy_mr_dv2:DVO2Strategy",
            "account_route_str": "DU1",
            "auto_submit_enabled_bool": True,
        },
        "scheduler_gate_dict": {
            "due_bool": True,
            "reason_code_str": "snapshot_ready",
            "latest_heartbeat_session_date_str": "2024-01-31",
        },
        "decision_plan_dict": {
            "signal_timestamp_str": "2024-01-31T16:00:00-05:00",
            "submission_timestamp_str": "2024-02-01T09:23:30-05:00",
            "target_execution_timestamp_str": "2024-02-01T09:30:00-05:00",
            "decision_book_type_str": "incremental_entry_exit_book",
            "target_weight_map": {"AAPL": 0.2},
            "snapshot_metadata_dict": {"strategy_family_str": "dv2"},
        },
        "broker_dict": {
            "broker_host_str": "127.0.0.1",
            "broker_port_int": 7497,
            "broker_client_id_int": 31,
            "visible_account_route_list": ["DU1"],
            "expected_account_visible_bool": True,
            "session_mode_str": "paper",
            "net_liq_float": 100000.0,
            "available_funds_float": 100000.0,
            "open_order_id_list": [],
        },
        "vplan_preview_dict": {
            "pod_budget_float": 50000.0,
            "broker_order_request_dict_list": [
                {
                    "asset_str": "AAPL",
                    "broker_order_type_str": "MOO",
                    "unit_str": "shares",
                    "amount_float": 100.0,
                    "target_bool": False,
                    "sizing_reference_price_float": 100.0,
                    "portfolio_value_float": 50000.0,
                    "order_request_key_str": "vplan:0:AAPL:1",
                }
            ],
        },
    }


def test_runner_doctor_json_outputs_structured_verdict(tmp_path: Path, monkeypatch, capsys):
    _clear_snapshot_env(monkeypatch)

    def fake_compute_doctor_verdict(**kwargs):
        assert kwargs["command_name_str"] == "doctor"
        return _sample_cli_doctor_detail_dict("PASS", command_name_str="doctor")

    monkeypatch.setattr(
        doctor_module,
        "compute_doctor_verdict",
        fake_compute_doctor_verdict,
    )

    return_code_int = runner_module.main(
        [
            "doctor",
            "--json",
            "--mode",
            "paper",
            "--releases-root",
            str(tmp_path / "missing_releases"),
            "--pod-id",
            "pod_dv2_paper_01",
            "--db-path",
            str(tmp_path / "missing.sqlite3"),
        ]
    )

    assert return_code_int == 0
    output_dict = json.loads(capsys.readouterr().out)
    assert output_dict["overall_verdict_str"] == "PASS"
    assert output_dict["feature_name_str"] == "doctor"
    assert output_dict["command_name_str"] == "doctor"
    assert output_dict["pod_id_str"] == "pod_dv2_paper_01"
    assert output_dict["manifest_qualification_dict"]["qualified_bool"] is True


def test_runner_rejects_removed_preflight_command(capsys):
    with pytest.raises(SystemExit) as exc_info:
        runner_module.main(["preflight", "--json"])

    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_runner_doctor_uses_config_release_root_when_cli_root_omitted(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    _clear_snapshot_env(monkeypatch)
    config_env_path_obj = tmp_path / "config.env"
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_paper_dv2_manifest(releases_root_path_obj)
    config_env_path_obj.write_text(
        f"NORGATE_RELEASES_ROOT={releases_root_path_obj}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner_module,
        "default_config_env_path_obj",
        lambda: config_env_path_obj,
    )
    monkeypatch.setattr(
        runner_module,
        "load_config_env_file",
        lambda override_existing_bool=True: {
            "NORGATE_RELEASES_ROOT": str(releases_root_path_obj),
        },
    )

    def fake_compute_doctor_verdict(**kwargs):
        assert kwargs["releases_root_path_str"] is None
        assert kwargs["releases_root_explicit_bool"] is False
        assert kwargs["config_env_path_str"] == str(config_env_path_obj.resolve())
        assert kwargs["loaded_config_env_dict"]["NORGATE_RELEASES_ROOT"] == str(releases_root_path_obj)
        return _sample_cli_doctor_detail_dict("PASS")

    monkeypatch.setattr(
        doctor_module,
        "compute_doctor_verdict",
        fake_compute_doctor_verdict,
    )

    return_code_int = runner_module.main(
        [
            "doctor",
            "--json",
            "--mode",
            "paper",
            "--pod-id",
            "pod_dv2_paper_01",
            "--db-path",
            str(tmp_path / "missing.sqlite3"),
        ]
    )

    assert return_code_int == 0
    assert json.loads(capsys.readouterr().out)["overall_verdict_str"] == "PASS"


def test_runner_doctor_without_pod_id_uses_selected_pod_db_read_only(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    _clear_snapshot_env(monkeypatch)
    config_env_path_obj = tmp_path / "config.env"
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    pod_state_root_path_obj = tmp_path / "state"
    pod_db_path_obj = pod_state_root_path_obj / "paper" / "pod_dv2_paper_01.sqlite3"
    pod_db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    _write_paper_dv2_manifest(releases_root_path_obj)
    config_env_path_obj.write_text(
        f"NORGATE_RELEASES_ROOT={releases_root_path_obj}\n",
        encoding="utf-8",
    )
    state_store_obj = LiveStateStore(str(pod_db_path_obj))
    state_store_obj.upsert_pod_state(
        PodState(
            pod_id_str="pod_dv2_paper_01",
            user_id_str="user_001",
            account_route_str="DU1",
            position_amount_map={},
            cash_float=100_000.0,
            total_value_float=100_000.0,
            strategy_state_dict={"loaded_from": "pod_db"},
            updated_timestamp_ts=datetime(2024, 2, 1, 12, 0, tzinfo=UTC),
        ),
        snapshot_stage_str="doctor_test",
        snapshot_source_str="pod_db",
    )
    before_bytes = pod_db_path_obj.read_bytes()
    before_count_map = _table_count_map(pod_db_path_obj)

    monkeypatch.setattr(runner_module, "DEFAULT_POD_STATE_ROOT_PATH_STR", str(pod_state_root_path_obj))
    monkeypatch.setattr(runner_module, "default_config_env_path_obj", lambda: config_env_path_obj)
    monkeypatch.setattr(
        runner_module,
        "load_config_env_file",
        lambda override_existing_bool=True: {
            "NORGATE_RELEASES_ROOT": str(releases_root_path_obj),
        },
    )
    monkeypatch.setattr(
        doctor_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {
            "due_bool": True,
            "reason_code_str": "snapshot_ready",
            "latest_heartbeat_session_date_str": "2024-01-31",
        },
    )

    def build_plan_asserts_pod_state(release_obj, as_of_ts, pod_state_obj):
        assert pod_state_obj.account_route_str == "DU1"
        assert pod_state_obj.cash_float == 100_000.0
        assert pod_state_obj.strategy_state_dict["loaded_from"] == "pod_db"
        return _build_paper_decision_plan_stub(release_obj, as_of_ts, pod_state_obj)

    monkeypatch.setattr(
        doctor_module.strategy_host,
        "build_decision_plan_for_release",
        build_plan_asserts_pod_state,
    )
    monkeypatch.setattr(
        doctor_module,
        "ensure_norgate_snapshots_for_live_tick",
        _stub_snapshot_sync_ready,
    )
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        "DU1",
        cash_float=100_000.0,
        total_value_float=100_000.0,
        session_mode_str="paper",
        snapshot_timestamp_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
    )
    broker_adapter_obj.seed_live_price_snapshot(
        "DU1",
        {"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 13, 20, tzinfo=UTC),
    )
    monkeypatch.setattr(
        runner_module,
        "BrokerAdapterResolver",
        lambda **kwargs: BrokerAdapterResolver(broker_adapter_obj=broker_adapter_obj),
    )

    return_code_int = runner_module.main(
        [
            "doctor",
            "--json",
            "--mode",
            "paper",
            "--as-of-ts",
            "2024-02-01T13:20:00+00:00",
        ]
    )

    output_dict = json.loads(capsys.readouterr().out)
    assert return_code_int == 0
    assert output_dict["overall_verdict_str"] == "PASS"
    assert output_dict["pod_id_str"] == "pod_dv2_paper_01"
    assert output_dict["persisted_lifecycle_dict"]["db_path_str"] == str(pod_db_path_obj)
    assert any(
        component_result_dict["reason_code_str"] == "pod_state_loaded"
        for component_result_dict in output_dict["component_result_dict_list"]
    )
    assert pod_db_path_obj.read_bytes() == before_bytes
    assert _table_count_map(pod_db_path_obj) == before_count_map


def test_runner_doctor_blocks_multiple_enabled_pods_without_pod_id(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    _clear_snapshot_env(monkeypatch)
    config_env_path_obj = tmp_path / "config.env"
    releases_root_path_obj = tmp_path / "releases" / "user_001"
    _write_release_manifest(
        releases_root_path_obj,
        mode_str="paper",
        pod_id_str="pod_dv2_a_01",
        account_route_str="DU1",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        data_profile_str="norgate_eod_sp500_pit",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
    )
    _write_release_manifest(
        releases_root_path_obj,
        mode_str="paper",
        pod_id_str="pod_dv2_b_01",
        account_route_str="DU2",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        data_profile_str="norgate_eod_sp500_pit",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
    )
    config_env_path_obj.write_text(
        f"NORGATE_RELEASES_ROOT={releases_root_path_obj}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner_module, "default_config_env_path_obj", lambda: config_env_path_obj)
    monkeypatch.setattr(
        runner_module,
        "load_config_env_file",
        lambda override_existing_bool=True: {
            "NORGATE_RELEASES_ROOT": str(releases_root_path_obj),
        },
    )

    def fail_build(*args, **kwargs):
        raise AssertionError("DecisionPlan build must not run for ambiguous doctor manifests.")

    monkeypatch.setattr(doctor_module.strategy_host, "build_decision_plan_for_release", fail_build)

    return_code_int = runner_module.main(
        [
            "doctor",
            "--json",
            "--mode",
            "paper",
        ]
    )

    output_dict = json.loads(capsys.readouterr().out)
    assert return_code_int == 1
    assert output_dict["overall_verdict_str"] == "BLOCK"
    assert output_dict["component_result_dict_list"][-1]["reason_code_str"] == "multiple_enabled_pods"
    assert output_dict["decision_plan_dict"] == {}
    assert output_dict["vplan_preview_dict"] == {}


def test_runner_doctor_human_output_includes_broker_and_order_preview(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    _clear_snapshot_env(monkeypatch)
    monkeypatch.setattr(
        doctor_module,
        "compute_doctor_verdict",
        lambda **kwargs: _sample_cli_doctor_detail_dict("PASS"),
    )

    return_code_int = runner_module.main(
        [
            "doctor",
            "--mode",
            "paper",
            "--releases-root",
            str(tmp_path / "missing_releases"),
            "--pod-id",
            "pod_dv2_paper_01",
            "--db-path",
            str(tmp_path / "missing.sqlite3"),
        ]
    )

    output_str = capsys.readouterr().out
    assert return_code_int == 0
    assert "Live Doctor" in output_str
    assert "VERDICT: PASS" in output_str
    assert "Manifest Qualification" in output_str
    assert "Qualified: True" in output_str
    assert "Strategy: strategies.dv2.strategy_mr_dv2:DVO2Strategy" in output_str
    assert "Reason: snapshot_ready" in output_str
    assert "VPS Config / Release Root" in output_str
    assert "Env/effective match: True" in output_str
    assert "Selected YAML:" in output_str
    assert "Visible accounts: ['DU1']" in output_str
    assert "Open orders: []" in output_str
    assert "Order count: 1" in output_str
    assert "AAPL BUY 100.0000 shares MOO" in output_str


def test_runner_doctor_block_returns_nonzero(tmp_path: Path, monkeypatch, capsys):
    _clear_snapshot_env(monkeypatch)
    monkeypatch.setattr(
        doctor_module,
        "compute_doctor_verdict",
        lambda **kwargs: _sample_cli_doctor_detail_dict("BLOCK"),
    )

    return_code_int = runner_module.main(
        [
            "doctor",
            "--json",
            "--mode",
            "paper",
            "--releases-root",
            str(tmp_path / "missing_releases"),
            "--pod-id",
            "pod_dv2_paper_01",
            "--db-path",
            str(tmp_path / "missing.sqlite3"),
        ]
    )

    assert return_code_int == 1
    output_dict = json.loads(capsys.readouterr().out)
    assert output_dict["overall_verdict_str"] == "BLOCK"


def test_runner_doctor_wait_returns_zero(tmp_path: Path, monkeypatch, capsys):
    _clear_snapshot_env(monkeypatch)
    monkeypatch.setattr(
        doctor_module,
        "compute_doctor_verdict",
        lambda **kwargs: _sample_cli_doctor_detail_dict("WAIT"),
    )

    return_code_int = runner_module.main(
        [
            "doctor",
            "--json",
            "--mode",
            "paper",
            "--releases-root",
            str(tmp_path / "missing_releases"),
            "--pod-id",
            "pod_dv2_paper_01",
            "--db-path",
            str(tmp_path / "missing.sqlite3"),
        ]
    )

    assert return_code_int == 0
    output_dict = json.loads(capsys.readouterr().out)
    assert output_dict["overall_verdict_str"] == "WAIT"


def test_runner_doctor_does_not_open_state_store_or_mutate_existing_db(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    _clear_snapshot_env(monkeypatch)
    db_path_obj = tmp_path / "live.sqlite3"
    LiveStateStore(str(db_path_obj))
    before_bytes = db_path_obj.read_bytes()
    before_count_map = _table_count_map(db_path_obj)

    def fake_compute_doctor_verdict(**kwargs):
        assert "state_store_obj" not in kwargs or kwargs["state_store_obj"] is None
        assert kwargs["db_path_str"] == str(db_path_obj)
        return _sample_cli_doctor_detail_dict("PASS")

    monkeypatch.setattr(
        doctor_module,
        "compute_doctor_verdict",
        fake_compute_doctor_verdict,
    )

    return_code_int = runner_module.main(
        [
            "doctor",
            "--json",
            "--mode",
            "paper",
            "--releases-root",
            str(tmp_path / "missing_releases"),
            "--pod-id",
            "pod_dv2_paper_01",
            "--db-path",
            str(db_path_obj),
        ]
    )

    assert return_code_int == 0
    assert json.loads(capsys.readouterr().out)["overall_verdict_str"] == "PASS"
    assert db_path_obj.read_bytes() == before_bytes
    assert _table_count_map(db_path_obj) == before_count_map


def test_runner_doctor_config_env_error_returns_structured_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        runner_module,
        "load_config_env_file",
        lambda override_existing_bool=True: (_ for _ in ()).throw(
            ValueError("Invalid config.env line 3: expected KEY=value.")
        ),
    )

    return_code_int = runner_module.main(
        [
            "doctor",
            "--json",
            "--mode",
            "paper",
            "--releases-root",
            str(tmp_path / "missing_releases"),
            "--pod-id",
            "pod_dv2_paper_01",
            "--db-path",
            str(tmp_path / "missing.sqlite3"),
        ]
    )

    assert return_code_int == 1
    output_dict = json.loads(capsys.readouterr().out)
    assert output_dict["overall_verdict_str"] == "BLOCK"
    assert output_dict["component_result_dict_list"][0]["reason_code_str"] == "config_env_error"
    assert output_dict["feature_name_str"] == "doctor"


def test_runner_doctor_incubation_returns_unsupported_block(tmp_path: Path, monkeypatch, capsys):
    _clear_snapshot_env(monkeypatch)

    return_code_int = runner_module.main(
        [
            "doctor",
            "--json",
            "--mode",
            "incubation",
            "--releases-root",
            str(tmp_path / "missing_releases"),
            "--pod-id",
            "pod_sim_01",
            "--db-path",
            str(tmp_path / "missing.sqlite3"),
        ]
    )

    assert return_code_int == 1
    output_dict = json.loads(capsys.readouterr().out)
    assert output_dict["overall_verdict_str"] == "BLOCK"
    assert (
        output_dict["component_result_dict_list"][-1]["reason_code_str"]
        == "incubation_doctor_unsupported"
    )


def test_runner_doctor_incubation_without_pod_or_db_returns_unsupported_block(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    _clear_snapshot_env(monkeypatch)
    config_env_path_obj = tmp_path / "config.env"
    config_env_path_obj.write_text(
        f"NORGATE_RELEASES_ROOT={tmp_path / 'releases'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner_module, "default_config_env_path_obj", lambda: config_env_path_obj)
    monkeypatch.setattr(
        runner_module,
        "load_config_env_file",
        lambda override_existing_bool=True: {
            "NORGATE_RELEASES_ROOT": str(tmp_path / "releases"),
        },
    )

    return_code_int = runner_module.main(
        [
            "doctor",
            "--json",
            "--mode",
            "incubation",
        ]
    )

    output_dict = json.loads(capsys.readouterr().out)
    assert return_code_int == 1
    assert output_dict["overall_verdict_str"] == "BLOCK"
    assert output_dict["command_name_str"] == "doctor"
    assert output_dict["component_result_dict_list"][-1]["reason_code_str"] == (
        "incubation_doctor_unsupported"
    )



