from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from alpha.live.models import (
    BrokerOrderAck,
    BrokerOrderFill,
    BrokerOrderRecord,
    BrokerSnapshot,
    DecisionPlan,
    ReconciliationResult,
    VPlan,
    VPlanRow,
)
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.release_manifest import load_release_list
from alpha.live.scheduler_service import (
    SchedulerDecision,
    _build_phase_failure_operator_message_spec_list,
    _build_phase_start_operator_message_spec_list,
    _build_phase_result_operator_message_spec_list,
    _build_stuck_operator_message_spec_list,
    _build_wait_operator_message_spec_list,
    _render_serve_tick_summary_str,
    _resolve_db_path_for_mode_str,
    _should_emit_wait_operator_message,
    get_scheduler_decision,
    main,
    run_once,
    serve,
)
from alpha.live import runner
from alpha.live.state_store_v2 import LiveStateStore


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def _write_manifest(
    root_path_obj: Path,
    auto_submit_enabled_bool: bool = True,
    account_route_str: str = "DU1",
    broker_host_str: str = "127.0.0.1",
    broker_port_int: int = 7497,
    broker_client_id_int: int = 31,
    broker_timeout_seconds_float: float = 4.0,
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
                f"  account_route: {account_route_str}",
                f"  host_str: {broker_host_str}",
                f"  port_int: {int(broker_port_int)}",
                f"  client_id_int: {int(broker_client_id_int)}",
                f"  timeout_seconds_float: {float(broker_timeout_seconds_float)}",
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


def _write_guardrail_manifest(
    root_path_obj: Path,
    *,
    user_id_str: str,
    pod_id_str: str,
    release_id_str: str,
    mode_str: str,
    enabled_bool: bool,
    account_route_str: str,
) -> None:
    releases_root_path_obj = root_path_obj / "releases" / user_id_str
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / f"{pod_id_str}.yaml").write_text(
        "\n".join(
            [
                "identity:",
                f"  release_id: {release_id_str}",
                f"  user_id: {user_id_str}",
                f"  pod_id: {pod_id_str}",
                "deployment:",
                f"  mode: {mode_str}",
                f"  enabled_bool: {'true' if enabled_bool else 'false'}",
                "broker:",
                f"  account_route: {account_route_str}",
                "  host_str: 127.0.0.1",
                "  port_int: 7497",
                "  client_id_int: 31",
                "  timeout_seconds_float: 4.0",
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
                "  auto_submit_enabled_bool: true",
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


def test_run_once_rejects_invalid_deployment_before_invoking_tick(tmp_path: Path, monkeypatch):
    _write_guardrail_manifest(
        tmp_path,
        user_id_str="user_001",
        pod_id_str="pod_a",
        release_id_str="user_001.pod_a.paper",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU1",
    )
    _write_guardrail_manifest(
        tmp_path,
        user_id_str="user_002",
        pod_id_str="pod_b",
        release_id_str="user_002.pod_b.paper",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU2",
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("tick should not run for invalid deployment")

    monkeypatch.setattr(runner, "tick", _fail_if_called)
    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))

    with pytest.raises(ValueError, match="one client per VPS/deployment"):
        run_once(
            state_store_obj=state_store_obj,
            broker_adapter_obj=StubBrokerAdapter(),
            as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
            releases_root_path_str=str(tmp_path / "releases"),
            env_mode_str="paper",
        )


def test_serve_rejects_invalid_deployment_before_starting_loop(tmp_path: Path):
    _write_guardrail_manifest(
        tmp_path,
        user_id_str="user_001",
        pod_id_str="pod_a",
        release_id_str="user_001.pod_a.paper",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU1",
    )
    _write_guardrail_manifest(
        tmp_path,
        user_id_str="user_002",
        pod_id_str="pod_b",
        release_id_str="user_002.pod_b.paper",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU2",
    )
    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))

    with pytest.raises(ValueError, match="one client per VPS/deployment"):
        serve(
            state_store_obj=state_store_obj,
            broker_adapter_obj=StubBrokerAdapter(),
            releases_root_path_str=str(tmp_path / "releases"),
            env_mode_str="paper",
            broker_host_str=None,
            broker_port_int=None,
            broker_client_id_int=None,
        )


def test_scheduler_service_uses_incubation_default_db_path():
    resolved_db_path_str = _resolve_db_path_for_mode_str(
        db_path_str=None,
        env_mode_str="incubation",
    )

    assert Path(resolved_db_path_str).name == "incubation_state.sqlite3"


def test_scheduler_service_uses_pod_scoped_default_db_path():
    resolved_db_path_str = _resolve_db_path_for_mode_str(
        db_path_str=None,
        env_mode_str="paper",
        pod_id_str="pod_dv2_01",
    )

    resolved_db_path_obj = Path(resolved_db_path_str)
    assert resolved_db_path_obj.name == "pod_dv2_01.sqlite3"
    assert resolved_db_path_obj.parent.name == "paper"


def test_scheduler_service_rejects_incubation_on_live_default_db_path():
    with pytest.raises(ValueError, match="Incubation must not use the paper/live default DB"):
        _resolve_db_path_for_mode_str(
            db_path_str=runner.DEFAULT_DB_PATH_STR,
            env_mode_str="incubation",
        )


def test_scheduler_eod_snapshot_command_accepts_all_modes(tmp_path: Path, monkeypatch, capsys):
    captured_mode_list: list[str] = []

    def _fake_eod_snapshot(**kwargs):
        captured_mode_list.append(str(kwargs["env_mode_str"]))
        return {
            "lease_acquired_bool": True,
            "eod_snapshot_count_int": 0,
            "skipped_snapshot_count_int": 0,
            "blocked_action_count_int": 0,
            "reason_count_map_dict": {},
        }

    monkeypatch.setattr(runner, "eod_snapshot", _fake_eod_snapshot)
    releases_root_path_obj = tmp_path / "releases"
    releases_root_path_obj.mkdir()

    for mode_str in ("incubation", "paper", "live"):
        main(
            [
                "eod_snapshot",
                "--mode",
                mode_str,
                "--db-path",
                str(tmp_path / f"{mode_str}.sqlite3"),
                "--releases-root",
                str(releases_root_path_obj),
            ]
        )

    capsys.readouterr()
    assert captured_mode_list == ["incubation", "paper", "live"]


def test_scheduler_run_once_passes_incubation_context_to_resolver(tmp_path: Path, monkeypatch):
    _write_guardrail_manifest(
        tmp_path,
        user_id_str="incubation_user",
        pod_id_str="pod_incubation_test",
        release_id_str="incubation_user.pod_incubation_test.incubation.v1",
        mode_str="incubation",
        enabled_bool=True,
        account_route_str="SIM_pod_incubation_test",
    )
    state_store_obj = LiveStateStore(str((tmp_path / "incubation.sqlite3").resolve()))
    decision_plan_obj = DecisionPlan(
        release_id_str="incubation_user.pod_incubation_test.incubation.v1",
        user_id_str="incubation_user",
        pod_id_str="pod_incubation_test",
        account_route_str="SIM_pod_incubation_test",
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 23, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        decision_base_position_map={},
        snapshot_metadata_dict={},
        strategy_state_dict={},
        decision_book_type_str="incremental_entry_exit_book",
        entry_target_weight_map_dict={"AAPL": 0.2},
        target_weight_map={"AAPL": 0.2},
        exit_asset_set=set(),
        entry_priority_list=["AAPL"],
        cash_reserve_weight_float=0.0,
    )
    state_store_obj.insert_decision_plan(decision_plan_obj)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    def _fake_tick(*, broker_adapter_resolver_obj, releases_root_path_str, **kwargs):
        release_obj = load_release_list(releases_root_path_str)[0]
        broker_adapter_obj = broker_adapter_resolver_obj.get_adapter(release_obj)
        assert broker_adapter_obj.get_session_mode_str("SIM_pod_incubation_test") == "incubation"
        return {
            "created_decision_plan_count_int": 0,
            "skipped_decision_plan_count_int": 0,
            "created_vplan_count_int": 0,
            "submitted_vplan_count_int": 0,
            "completed_vplan_count_int": 0,
            "blocked_action_count_int": 0,
            "expired_decision_plan_count_int": 0,
            "warning_count_map_dict": {},
            "reason_count_map_dict": {},
        }

    monkeypatch.setattr(runner, "tick", _fake_tick)

    detail_dict = run_once(
        state_store_obj=state_store_obj,
        broker_adapter_obj=None,
        as_of_ts=datetime(2024, 2, 1, 14, 24, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="incubation",
    )

    assert detail_dict["tick_invoked_bool"] is True


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


def _insert_submitting_vplan(
    state_store_obj: LiveStateStore,
) -> VPlan:
    latest_vplan_obj = _insert_submitted_vplan(state_store_obj)
    state_store_obj.mark_vplan_status(int(latest_vplan_obj.vplan_id_int or 0), "submitting")
    state_store_obj.mark_decision_plan_status(int(latest_vplan_obj.decision_plan_id_int), "expired")
    return state_store_obj.get_vplan_by_id(int(latest_vplan_obj.vplan_id_int or 0))


def _insert_ready_vplan(
    state_store_obj: LiveStateStore,
) -> VPlan:
    decision_plan_obj = _insert_planned_decision_plan(state_store_obj)
    state_store_obj.mark_decision_plan_status(int(decision_plan_obj.decision_plan_id_int or 0), "vplan_ready")
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
            status_str="ready",
        )
    )


def _seed_post_execution_reconcile_truth(
    state_store_obj: LiveStateStore,
    vplan_obj: VPlan,
    broker_share_float: float,
    latest_broker_order_status_str: str,
) -> None:
    state_store_obj.upsert_broker_snapshot_cache(
        BrokerSnapshot(
            account_route_str=vplan_obj.account_route_str,
            snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
            cash_float=10000.0,
            total_value_float=10000.0,
            net_liq_float=10000.0,
            position_amount_map={"AAPL": float(broker_share_float)},
        )
    )
    state_store_obj.upsert_vplan_broker_order_record_list(
        [
            BrokerOrderRecord(
                broker_order_id_str="stub_order_1",
                decision_plan_id_int=int(vplan_obj.decision_plan_id_int),
                vplan_id_int=int(vplan_obj.vplan_id_int or 0),
                account_route_str=vplan_obj.account_route_str,
                asset_str="AAPL",
                order_request_key_str="vplan:1:AAPL:1",
                broker_order_type_str="MOO",
                unit_str="shares",
                amount_float=10.0,
                filled_amount_float=float(broker_share_float),
                remaining_amount_float=max(10.0 - float(broker_share_float), 0.0),
                avg_fill_price_float=None,
                status_str=latest_broker_order_status_str,
                submitted_timestamp_ts=datetime(2024, 2, 1, 9, 23, 30, tzinfo=MARKET_TIMEZONE_OBJ),
                last_status_timestamp_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
                submission_key_str="vplan:1",
                raw_payload_dict={
                    "submission_key_str": "vplan:1",
                    "order_request_key_str": "vplan:1:AAPL:1",
                },
            )
        ]
    )
    state_store_obj.insert_vplan_reconciliation_snapshot(
        pod_id_str=vplan_obj.pod_id_str,
        decision_plan_id_int=int(vplan_obj.decision_plan_id_int),
        vplan_id_int=int(vplan_obj.vplan_id_int or 0),
        stage_str="post_execution",
        reconciliation_result_obj=ReconciliationResult(
            passed_bool=False,
            status_str="blocked",
            mismatch_dict={
                "AAPL": {
                    "model_share_float": 10.0,
                    "broker_share_float": float(broker_share_float),
                }
            },
            model_position_map={"AAPL": 10.0},
            broker_position_map={"AAPL": float(broker_share_float)},
            model_cash_float=10000.0,
            broker_cash_float=10000.0,
        ),
    )


def _seed_submit_progress_truth(
    state_store_obj: LiveStateStore,
    vplan_obj: VPlan,
    *,
    filled_amount_float: float,
    broker_response_ack_bool: bool,
    latest_broker_order_status_str: str,
) -> None:
    response_timestamp_ts = datetime(2024, 2, 1, 9, 23, 35, tzinfo=MARKET_TIMEZONE_OBJ)
    state_store_obj.upsert_vplan_broker_order_record_list(
        [
            BrokerOrderRecord(
                broker_order_id_str="stub_order_1",
                decision_plan_id_int=int(vplan_obj.decision_plan_id_int),
                vplan_id_int=int(vplan_obj.vplan_id_int or 0),
                account_route_str=vplan_obj.account_route_str,
                asset_str="AAPL",
                order_request_key_str="vplan:1:AAPL:1",
                broker_order_type_str="MOO",
                unit_str="shares",
                amount_float=10.0,
                filled_amount_float=float(filled_amount_float),
                remaining_amount_float=max(10.0 - float(abs(filled_amount_float)), 0.0),
                avg_fill_price_float=100.0 if abs(float(filled_amount_float)) > 0.0 else None,
                status_str=latest_broker_order_status_str,
                submitted_timestamp_ts=datetime(2024, 2, 1, 9, 23, 30, tzinfo=MARKET_TIMEZONE_OBJ),
                last_status_timestamp_ts=response_timestamp_ts,
                submission_key_str="vplan:1",
                raw_payload_dict={
                    "submission_key_str": "vplan:1",
                    "order_request_key_str": "vplan:1:AAPL:1",
                },
            )
        ]
    )
    state_store_obj.upsert_vplan_broker_ack_list(
        [
            BrokerOrderAck(
                decision_plan_id_int=int(vplan_obj.decision_plan_id_int),
                vplan_id_int=int(vplan_obj.vplan_id_int or 0),
                account_route_str=vplan_obj.account_route_str,
                order_request_key_str="vplan:1:AAPL:1",
                asset_str="AAPL",
                broker_order_type_str="MOO",
                local_submit_ack_bool=True,
                broker_response_ack_bool=bool(broker_response_ack_bool),
                ack_status_str="broker_acked" if broker_response_ack_bool else "missing_critical",
                ack_source_str="stub.state_snapshot" if broker_response_ack_bool else "missing",
                broker_order_id_str="stub_order_1" if broker_response_ack_bool else None,
                response_timestamp_ts=response_timestamp_ts if broker_response_ack_bool else None,
                raw_payload_dict={},
            )
        ]
    )
    state_store_obj.update_vplan_submit_ack_summary(
        vplan_id_int=int(vplan_obj.vplan_id_int or 0),
        submit_ack_status_str="complete" if broker_response_ack_bool else "missing_critical",
        ack_coverage_ratio_float=1.0 if broker_response_ack_bool else 0.0,
        missing_ack_count_int=0 if broker_response_ack_bool else 1,
        submit_ack_checked_timestamp_ts=response_timestamp_ts,
    )
    if abs(float(filled_amount_float)) > 0.0:
        state_store_obj.upsert_vplan_fill_list(
            [
                BrokerOrderFill(
                    broker_order_id_str="stub_order_1",
                    decision_plan_id_int=int(vplan_obj.decision_plan_id_int),
                    vplan_id_int=int(vplan_obj.vplan_id_int or 0),
                    account_route_str=vplan_obj.account_route_str,
                    asset_str="AAPL",
                    fill_amount_float=float(filled_amount_float),
                    fill_price_float=100.0,
                    fill_timestamp_ts=response_timestamp_ts,
                    raw_payload_dict={},
                )
            ]
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


def test_scheduler_decision_filters_to_requested_pod(tmp_path: Path, monkeypatch):
    _write_guardrail_manifest(
        tmp_path,
        user_id_str="user_001",
        pod_id_str="pod_a",
        release_id_str="user_001.pod_a.paper",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU1",
    )
    _write_guardrail_manifest(
        tmp_path,
        user_id_str="user_001",
        pod_id_str="pod_b",
        release_id_str="user_001.pod_b.paper",
        mode_str="paper",
        enabled_bool=True,
        account_route_str="DU2",
    )
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
        pod_id_str="pod_b",
    )

    assert scheduler_decision_obj.due_now_bool is True
    assert scheduler_decision_obj.next_phase_str == "build_decision_plan"
    assert scheduler_decision_obj.related_pod_id_list == ["pod_b"]
    assert [release_obj.pod_id_str for release_obj in state_store_obj.get_enabled_release_list()] == ["pod_b"]


def test_scheduler_decision_selects_eod_after_close_when_idle(tmp_path: Path, monkeypatch):
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-30"),
    )

    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    scheduler_decision_obj = get_scheduler_decision(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 21, 15, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )

    assert scheduler_decision_obj.due_now_bool is True
    assert scheduler_decision_obj.next_phase_str == "eod_snapshot"
    assert scheduler_decision_obj.reason_code_str == "eod_snapshot_due"
    assert scheduler_decision_obj.related_pod_id_list == ["pod_test_01"]


def test_scheduler_decision_keeps_reconcile_priority_over_eod(tmp_path: Path, monkeypatch):
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-30"),
    )

    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    _insert_submitted_vplan(state_store_obj)
    scheduler_decision_obj = get_scheduler_decision(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 21, 15, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )

    assert scheduler_decision_obj.due_now_bool is True
    assert scheduler_decision_obj.next_phase_str == "post_execution_reconcile"
    assert scheduler_decision_obj.reason_code_str == "ready_to_reconcile"


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


def test_scheduler_decision_reconciles_stale_submitting_vplan_after_grace(
    tmp_path: Path,
    monkeypatch,
):
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    _insert_submitting_vplan(state_store_obj)

    scheduler_decision_obj = get_scheduler_decision(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 14, 36, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )

    assert scheduler_decision_obj.due_now_bool is True
    assert scheduler_decision_obj.next_phase_str == "post_execution_reconcile"
    assert scheduler_decision_obj.reason_code_str == "ready_to_reconcile"


def test_scheduler_decision_parks_submitted_vplan_after_terminal_post_execution_reconcile(
    tmp_path: Path,
    monkeypatch,
):
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    latest_vplan_obj = _insert_submitted_vplan(state_store_obj)
    _seed_post_execution_reconcile_truth(
        state_store_obj=state_store_obj,
        vplan_obj=latest_vplan_obj,
        broker_share_float=4.0,
        latest_broker_order_status_str="Cancelled",
    )

    scheduler_decision_obj = get_scheduler_decision(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 14, 36, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )

    assert scheduler_decision_obj.due_now_bool is False
    assert scheduler_decision_obj.next_phase_str == "manual_review_pending"
    assert scheduler_decision_obj.reason_code_str == "execution_exception_parked"
    assert scheduler_decision_obj.related_pod_id_list == ["pod_test_01"]


def test_scheduler_decision_keeps_reconcile_for_nonterminal_post_execution_residual(
    tmp_path: Path,
    monkeypatch,
):
    _write_manifest(tmp_path)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    latest_vplan_obj = _insert_submitted_vplan(state_store_obj)
    _seed_post_execution_reconcile_truth(
        state_store_obj=state_store_obj,
        vplan_obj=latest_vplan_obj,
        broker_share_float=0.0,
        latest_broker_order_status_str="PendingSubmit",
    )

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
            "user_id_str": "user_001",
            "pod_id_str": "pod_test_01",
            "mode_str": "paper",
            "account_route_str": "DU1",
            "auto_submit_enabled_bool": True,
            "latest_decision_plan_status_str": "planned",
            "latest_vplan_status_str": None,
            "latest_signal_timestamp_str": "2024-01-31T16:00:00-05:00",
            "latest_submission_timestamp_str": "2024-02-01T09:23:30-05:00",
            "latest_broker_snapshot_timestamp_str": None,
            "next_action_str": "wait",
            "reason_code_str": "waiting_for_submission_window",
            "latest_fill_timestamp_str": None,
            "submit_ack_status_str": None,
            "ack_coverage_ratio_float": None,
            "missing_ack_count_int": 0,
            "missing_ack_row_dict_list": [],
            "latest_broker_order_row_dict_list": [],
            "broker_ack_row_dict_list": [],
            "exception_row_dict_list": [],
            "exception_count_int": 0,
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


def test_wait_operator_message_emits_on_state_change_and_fifteen_minute_heartbeat():
    as_of_ts = datetime(2024, 2, 1, 14, 0, tzinfo=UTC)
    current_signature_tup = ("paper", "build_vplan", "waiting_for_submission_window")

    assert _should_emit_wait_operator_message(
        last_printed_signature_tup=None,
        current_signature_tup=current_signature_tup,
        last_printed_timestamp_ts=None,
        as_of_ts=as_of_ts,
    )
    assert _should_emit_wait_operator_message(
        last_printed_signature_tup=("paper", "build_decision_plan", "snapshot_ready"),
        current_signature_tup=current_signature_tup,
        last_printed_timestamp_ts=as_of_ts,
        as_of_ts=as_of_ts + timedelta(minutes=1),
    )
    assert not _should_emit_wait_operator_message(
        last_printed_signature_tup=current_signature_tup,
        current_signature_tup=current_signature_tup,
        last_printed_timestamp_ts=as_of_ts,
        as_of_ts=as_of_ts + timedelta(minutes=14, seconds=59),
    )
    assert _should_emit_wait_operator_message(
        last_printed_signature_tup=current_signature_tup,
        current_signature_tup=current_signature_tup,
        last_printed_timestamp_ts=as_of_ts,
        as_of_ts=as_of_ts + timedelta(minutes=15),
    )
    assert not _should_emit_wait_operator_message(
        last_printed_signature_tup=current_signature_tup,
        current_signature_tup=current_signature_tup,
        last_printed_timestamp_ts=as_of_ts,
        as_of_ts=as_of_ts + timedelta(seconds=29),
        heartbeat_seconds_int=30,
    )
    assert _should_emit_wait_operator_message(
        last_printed_signature_tup=current_signature_tup,
        current_signature_tup=current_signature_tup,
        last_printed_timestamp_ts=as_of_ts,
        as_of_ts=as_of_ts + timedelta(seconds=30),
        heartbeat_seconds_int=30,
    )


def test_phase_result_operator_messages_include_submit_ack_and_fill_progress(tmp_path: Path):
    _write_manifest(tmp_path, auto_submit_enabled_bool=True)
    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    state_store_obj.upsert_release_list(load_release_list(str(tmp_path / "releases")))
    submitted_vplan_obj = _insert_submitted_vplan(state_store_obj)
    _seed_submit_progress_truth(
        state_store_obj=state_store_obj,
        vplan_obj=submitted_vplan_obj,
        filled_amount_float=0.0,
        broker_response_ack_bool=True,
        latest_broker_order_status_str="Submitted",
    )
    scheduler_decision_obj = SchedulerDecision(
        as_of_timestamp_ts=datetime(2024, 2, 1, 14, 24, tzinfo=UTC),
        env_mode_str="paper",
        due_now_bool=True,
        active_poll_bool=False,
        next_phase_str="submit_vplan",
        reason_code_str="vplan_ready",
        next_due_timestamp_ts=datetime(2024, 2, 1, 14, 24, tzinfo=UTC),
        related_pod_id_list=["pod_test_01"],
    )

    operator_message_spec_dict_list = _build_phase_result_operator_message_spec_list(
        state_store_obj=state_store_obj,
        scheduler_decision_obj=scheduler_decision_obj,
        tick_detail_dict={
            "submitted_vplan_count_int": 1,
            "warning_count_map_dict": {},
            "reason_count_map_dict": {},
        },
        pod_status_dict_list=runner.get_status_summary(
            state_store_obj=state_store_obj,
            as_of_ts=datetime(2024, 2, 1, 14, 24, tzinfo=UTC),
            releases_root_path_str=str(tmp_path / "releases"),
        )["pod_status_dict_list"],
        execution_report_dict_list=[],
        broker_adapter_resolver_obj=runner.BrokerAdapterResolver(
            broker_host_str="127.0.0.1",
            broker_port_int=7496,
            broker_client_id_int=31,
        ),
    )

    assert [
        str(operator_message_spec_dict["phase_action_str"])
        for operator_message_spec_dict in operator_message_spec_dict_list
    ] == [
        "broker_connect.ok",
        "submit_vplan.ok",
        "submit_ack.ok",
        "fill.none",
    ]


def test_serve_tick_summary_uses_human_tick_renderer():
    output_str = _render_serve_tick_summary_str(
        {
            "lease_acquired_bool": True,
            "created_decision_plan_count_int": 1,
            "skipped_decision_plan_count_int": 0,
            "expired_decision_plan_count_int": 0,
            "created_vplan_count_int": 1,
            "submitted_vplan_count_int": 0,
            "completed_vplan_count_int": 0,
            "blocked_action_count_int": 1,
            "warning_count_map_dict": {"missing_live_price": 1},
            "reason_count_map_dict": {"vplan_ready": 1},
        }
    )

    assert "Tick Result" in output_str
    assert "- Built 1 decision plan." in output_str
    assert "- Built 1 order plan." in output_str
    assert "- Submitted 0 order plans." in output_str
    assert "- Completed 0 reconciliations." in output_str
    assert "- Blocked actions: 1" in output_str
    assert "- Warnings: 1" in output_str
    assert "execution plan is ready to submit" in output_str
    assert "missing live reference price for at least one asset" in output_str
    assert "Next" in output_str
    assert "Raw Fields" not in output_str


def test_phase_start_operator_messages_use_release_broker_tuple(tmp_path: Path):
    _write_manifest(
        tmp_path,
        auto_submit_enabled_bool=True,
        broker_port_int=7498,
        broker_client_id_int=41,
    )
    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    state_store_obj.upsert_release_list(load_release_list(str(tmp_path / "releases")))
    scheduler_decision_obj = SchedulerDecision(
        as_of_timestamp_ts=datetime(2024, 2, 1, 14, 24, tzinfo=UTC),
        env_mode_str="paper",
        due_now_bool=True,
        active_poll_bool=False,
        next_phase_str="build_vplan",
        reason_code_str="ready_to_build_vplan",
        next_due_timestamp_ts=datetime(2024, 2, 1, 14, 24, tzinfo=UTC),
        related_pod_id_list=["pod_test_01"],
    )

    operator_message_spec_dict_list = _build_phase_start_operator_message_spec_list(
        state_store_obj=state_store_obj,
        scheduler_decision_obj=scheduler_decision_obj,
        broker_adapter_resolver_obj=runner.BrokerAdapterResolver(),
    )

    assert operator_message_spec_dict_list[0]["phase_action_str"] == "broker_connect.start"
    assert operator_message_spec_dict_list[0]["field_map_dict"]["broker"] == "127.0.0.1:7498"
    assert operator_message_spec_dict_list[0]["field_map_dict"]["client_id"] == 41


def test_wait_operator_messages_show_active_reconcile_progress(tmp_path: Path):
    _write_manifest(tmp_path, auto_submit_enabled_bool=True)
    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    state_store_obj.upsert_release_list(load_release_list(str(tmp_path / "releases")))
    submitted_vplan_obj = _insert_submitted_vplan(state_store_obj)
    _seed_submit_progress_truth(
        state_store_obj=state_store_obj,
        vplan_obj=submitted_vplan_obj,
        filled_amount_float=5.0,
        broker_response_ack_bool=True,
        latest_broker_order_status_str="Submitted",
    )

    scheduler_decision_obj = SchedulerDecision(
        as_of_timestamp_ts=datetime(2024, 2, 1, 14, 25, tzinfo=UTC),
        env_mode_str="paper",
        due_now_bool=False,
        active_poll_bool=True,
        next_phase_str="post_execution_reconcile",
        reason_code_str="waiting_for_post_execution_reconcile",
        next_due_timestamp_ts=datetime(2024, 2, 1, 14, 35, tzinfo=UTC),
        related_pod_id_list=["pod_test_01"],
    )
    pod_status_dict_list = runner.get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 14, 25, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
    )["pod_status_dict_list"]
    execution_report_dict_list = runner.get_execution_report_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 14, 25, tzinfo=UTC),
        releases_root_path_str=str(tmp_path / "releases"),
    )["execution_report_dict_list"]
    broker_order_snapshot_dict_list = [
        {
            "pod_id_str": "pod_test_01",
            "latest_vplan_id_int": int(submitted_vplan_obj.vplan_id_int or 0),
            "broker_order_row_dict_list": state_store_obj.get_broker_order_row_dict_list_for_vplan(
                int(submitted_vplan_obj.vplan_id_int or 0)
            ),
        }
    ]

    operator_message_spec_dict_list = _build_wait_operator_message_spec_list(
        scheduler_decision_obj,
        pod_status_dict_list=pod_status_dict_list,
        execution_report_dict_list=execution_report_dict_list,
        broker_order_snapshot_dict_list=broker_order_snapshot_dict_list,
    )

    assert [
        str(operator_message_spec_dict["phase_action_str"])
        for operator_message_spec_dict in operator_message_spec_dict_list
    ] == [
        "fill.partial",
        "reconcile.wait",
    ]
    assert operator_message_spec_dict_list[1]["field_map_dict"]["acked"] == "1/1"
    assert operator_message_spec_dict_list[1]["field_map_dict"]["fills"] == "0/1"
    assert operator_message_spec_dict_list[1]["field_map_dict"]["partial"] == 1


def test_phase_failure_operator_messages_include_broker_and_phase_failure(tmp_path: Path):
    _write_manifest(tmp_path, auto_submit_enabled_bool=True)
    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    state_store_obj.upsert_release_list(
        load_release_list(str(tmp_path / "releases"))
    )
    scheduler_decision_obj = SchedulerDecision(
        as_of_timestamp_ts=datetime(2024, 2, 1, 14, 25, tzinfo=UTC),
        env_mode_str="live",
        due_now_bool=True,
        active_poll_bool=False,
        next_phase_str="submit_vplan",
        reason_code_str="vplan_ready",
        next_due_timestamp_ts=datetime(2024, 2, 1, 14, 25, tzinfo=UTC),
        related_pod_id_list=["pod_test_01"],
    )

    operator_message_spec_dict_list = _build_phase_failure_operator_message_spec_list(
        state_store_obj=state_store_obj,
        scheduler_decision_obj=scheduler_decision_obj,
        as_of_ts=datetime(2024, 2, 1, 14, 25, tzinfo=UTC),
        error_str="[WinError 1225] The remote computer refused the network connection",
        error_retry_seconds_int=60,
        broker_adapter_resolver_obj=runner.BrokerAdapterResolver(
            broker_host_str="127.0.0.1",
            broker_port_int=7496,
            broker_client_id_int=31,
        ),
    )
    phase_action_str_list = [
        str(operator_message_spec_dict["phase_action_str"])
        for operator_message_spec_dict in operator_message_spec_dict_list
    ]

    assert phase_action_str_list == [
        "broker_connect.fail",
        "submit_vplan.fail",
        "cycle.fail",
    ]


def test_stuck_operator_messages_detect_submit_stall(tmp_path: Path):
    _write_manifest(tmp_path, auto_submit_enabled_bool=True)
    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    state_store_obj.upsert_release_list(
        load_release_list(str(tmp_path / "releases"))
    )

    ready_vplan_obj = _insert_ready_vplan(state_store_obj)
    assert state_store_obj.claim_vplan_for_submission(int(ready_vplan_obj.vplan_id_int or 0)) is True

    operator_message_spec_dict_list = _build_stuck_operator_message_spec_list(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 14, 36, tzinfo=UTC),
        reconcile_grace_seconds_int=300,
    )

    assert operator_message_spec_dict_list == [
        {
            "level_str": "CRITICAL",
            "phase_action_str": "submit_vplan.stuck",
            "timestamp_obj": datetime(2024, 2, 1, 14, 36, tzinfo=UTC),
            "field_map_dict": {
                "pod": "pod_test_01",
                "account": "DU1",
                "vplan": int(ready_vplan_obj.vplan_id_int or 0),
                "reason": "no broker orders recorded after submit claim",
            },
        }
    ]


def test_stuck_operator_messages_detect_reconcile_stall(tmp_path: Path):
    _write_manifest(tmp_path, auto_submit_enabled_bool=True)
    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    state_store_obj.upsert_release_list(
        load_release_list(str(tmp_path / "releases"))
    )

    submitted_vplan_obj = _insert_submitted_vplan(state_store_obj)

    operator_message_spec_dict_list = _build_stuck_operator_message_spec_list(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 14, 36, tzinfo=UTC),
        reconcile_grace_seconds_int=300,
    )

    assert operator_message_spec_dict_list == [
        {
            "level_str": "CRITICAL",
            "phase_action_str": "reconcile.stuck",
            "timestamp_obj": datetime(2024, 2, 1, 14, 36, tzinfo=UTC),
            "field_map_dict": {
                "pod": "pod_test_01",
                "account": "DU1",
                "vplan": int(submitted_vplan_obj.vplan_id_int or 0),
                "reason": "no post-execution reconcile snapshot recorded",
            },
        }
    ]
