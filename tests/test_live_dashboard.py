from __future__ import annotations

from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
from urllib.request import urlopen

import alpha.live.dashboard as dashboard_module
import alpha.live.norgate_snapshot_sync as norgate_sync_module
import alpha.live.runner as runner_module
from alpha.live.dashboard import (
    DASHBOARD_HTML_STR,
    DashboardApp,
    DashboardConfig,
    DashboardPodTarget,
    DiffJobManager,
    _build_alert_dict_list,
    _build_execution_report_from_vplan_dict,
    _build_lifecycle_step_dict_list,
    _build_required_action_dict,
    build_dashboard_summary_dict,
    build_pod_detail_dict,
    find_latest_diff_artifact_dict,
    load_dashboard_config,
    make_dashboard_handler_class,
    resolve_db_path_for_release_str,
)
from alpha.live.models import (
    BrokerOrderAck,
    BrokerOrderFill,
    BrokerOrderRecord,
    DecisionPlan,
    LiveRelease,
    PodState,
    ReconciliationResult,
    VPlan,
    VPlanRow,
)
from alpha.live.release_manifest import load_release_list
from alpha.live.state_store_v2 import LiveStateStore


AS_OF_TS = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)


def _write_release_manifest(
    releases_root_path_obj: Path,
    *,
    user_id_str: str,
    pod_id_str: str,
    mode_str: str,
    enabled_bool: bool = True,
    account_route_str: str | None = None,
) -> None:
    user_dir_path_obj = releases_root_path_obj / user_id_str
    user_dir_path_obj.mkdir(parents=True, exist_ok=True)
    default_route_str = f"SIM_{pod_id_str}" if mode_str == "incubation" else f"DU_{pod_id_str}"
    route_str = account_route_str or default_route_str
    (user_dir_path_obj / f"{pod_id_str}.yaml").write_text(
        "\n".join(
            [
                "identity:",
                f"  release_id: {user_id_str}.{pod_id_str}.{mode_str}.v1",
                f"  user_id: {user_id_str}",
                f"  pod_id: {pod_id_str}",
                "deployment:",
                f"  mode: {mode_str}",
                f"  enabled_bool: {'true' if enabled_bool else 'false'}",
                "broker:",
                f"  account_route: {route_str}",
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


def _write_config(config_path_obj: Path, override_map_dict: dict[str, str]) -> None:
    line_list = ["db_overrides:"]
    for pod_id_str, db_path_str in override_map_dict.items():
        line_list.extend(
            [
                f"  {pod_id_str}:",
                f"    db_path: {db_path_str}",
            ]
        )
    config_path_obj.write_text("\n".join(line_list) + "\n", encoding="utf-8")


def _build_release_obj(pod_id_str: str = "pod_job") -> LiveRelease:
    return LiveRelease(
        release_id_str=f"user_001.{pod_id_str}.paper.v1",
        user_id_str="user_001",
        pod_id_str=pod_id_str,
        account_route_str=f"DU_{pod_id_str}",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        mode_str="paper",
        session_calendar_id_str="XNYS",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
        data_profile_str="norgate_eod_sp500_pit",
        params_dict={},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="",
    )


def _seed_pod_state(db_path_obj: Path, release_obj: LiveRelease, total_value_float: float) -> None:
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    store_obj = LiveStateStore(str(db_path_obj))
    store_obj.upsert_pod_state(
        PodState(
            pod_id_str=release_obj.pod_id_str,
            user_id_str=release_obj.user_id_str,
            account_route_str=release_obj.account_route_str,
            position_amount_map={"AAPL": 3.0},
            cash_float=1000.0,
            total_value_float=total_value_float,
            strategy_state_dict={"trade_id_int": 7},
            updated_timestamp_ts=AS_OF_TS,
            snapshot_stage_str="post_execution",
            snapshot_source_str="broker",
        )
    )


def _seed_eod_pod_state(
    db_path_obj: Path,
    release_obj: LiveRelease,
    *,
    total_value_float: float,
    updated_timestamp_ts: datetime,
    cash_float: float = 1200.0,
    snapshot_source_str: str = "broker",
) -> None:
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    store_obj = LiveStateStore(str(db_path_obj))
    store_obj.upsert_pod_state(
        PodState(
            pod_id_str=release_obj.pod_id_str,
            user_id_str=release_obj.user_id_str,
            account_route_str=release_obj.account_route_str,
            position_amount_map={"AAPL": 5.0},
            cash_float=cash_float,
            total_value_float=total_value_float,
            strategy_state_dict={"trade_id_int": 8},
            updated_timestamp_ts=updated_timestamp_ts,
            snapshot_stage_str="eod",
            snapshot_source_str=snapshot_source_str,
        )
    )


def _seed_decision_vplan_and_broker_rows(db_path_obj: Path, release_obj: LiveRelease) -> None:
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    store_obj = LiveStateStore(str(db_path_obj))
    store_obj.upsert_release(release_obj)
    decision_plan_obj = store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=AS_OF_TS,
            submission_timestamp_ts=AS_OF_TS,
            target_execution_timestamp_ts=AS_OF_TS,
            execution_policy_str=release_obj.execution_policy_str,
            decision_base_position_map={"AAPL": 2.0},
            snapshot_metadata_dict={
                "strategy_family_str": "dashboard_test",
                "dtb3_download_status_str": "download_success",
                "dtb3_latest_observation_date_str": "2024-01-01",
                "dtb3_freshness_business_days_int": 1,
                "dtb3_source_name_str": "FRED",
                "dtb3_used_cache_bool": False,
            },
            strategy_state_dict={"trade_id_int": 9},
            decision_book_type_str="full_target_weight_book",
            full_target_weight_map_dict={"AAPL": 0.6, "TLT": 0.4},
            target_weight_map={"AAPL": 0.6, "TLT": 0.4},
            exit_asset_set=set(),
            entry_priority_list=[],
            cash_reserve_weight_float=0.0,
            preserve_untouched_positions_bool=False,
            rebalance_omitted_assets_to_zero_bool=True,
        )
    )
    vplan_obj = store_obj.insert_vplan(
        VPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            decision_plan_id_int=int(decision_plan_obj.decision_plan_id_int or 0),
            signal_timestamp_ts=AS_OF_TS,
            submission_timestamp_ts=AS_OF_TS,
            target_execution_timestamp_ts=AS_OF_TS,
            execution_policy_str=release_obj.execution_policy_str,
            broker_snapshot_timestamp_ts=AS_OF_TS,
            live_reference_snapshot_timestamp_ts=AS_OF_TS,
            live_price_source_str="test_price",
            net_liq_float=10000.0,
            available_funds_float=9000.0,
            excess_liquidity_float=8000.0,
            pod_budget_fraction_float=0.5,
            pod_budget_float=5000.0,
            current_broker_position_map={"AAPL": 2.0},
            live_reference_price_map={"AAPL": 100.0},
            target_share_map={"AAPL": 30.0},
            order_delta_map={"AAPL": 28.0},
            vplan_row_list=[
                VPlanRow(
                    asset_str="AAPL",
                    current_share_float=2.0,
                    target_share_float=30.0,
                    order_delta_share_float=28.0,
                    live_reference_price_float=100.0,
                    estimated_target_notional_float=3000.0,
                    broker_order_type_str="MOO",
                    live_reference_source_str="test.snapshot",
                )
            ],
            live_reference_source_map_dict={"AAPL": "test.snapshot"},
            submission_key_str=f"vplan:{decision_plan_obj.decision_plan_id_int}",
            status_str="submitted",
            submit_ack_status_str="complete",
            ack_coverage_ratio_float=1.0,
            missing_ack_count_int=0,
        )
    )
    store_obj.mark_decision_plan_status(int(decision_plan_obj.decision_plan_id_int or 0), "submitted")
    store_obj.upsert_vplan_broker_order_record_list(
        [
            BrokerOrderRecord(
                broker_order_id_str="order_1",
                decision_plan_id_int=decision_plan_obj.decision_plan_id_int,
                vplan_id_int=vplan_obj.vplan_id_int,
                account_route_str=release_obj.account_route_str,
                asset_str="AAPL",
                order_request_key_str="req_1",
                broker_order_type_str="MOO",
                unit_str="shares",
                amount_float=28.0,
                filled_amount_float=28.0,
                status_str="Filled",
                submitted_timestamp_ts=AS_OF_TS,
                raw_payload_dict={},
                remaining_amount_float=0.0,
                avg_fill_price_float=101.0,
                last_status_timestamp_ts=AS_OF_TS,
                submission_key_str=f"vplan:{decision_plan_obj.decision_plan_id_int}",
            )
        ]
    )
    store_obj.upsert_vplan_fill_list(
        [
            BrokerOrderFill(
                broker_order_id_str="order_1",
                decision_plan_id_int=decision_plan_obj.decision_plan_id_int,
                vplan_id_int=vplan_obj.vplan_id_int,
                account_route_str=release_obj.account_route_str,
                asset_str="AAPL",
                fill_amount_float=28.0,
                fill_price_float=101.0,
                fill_timestamp_ts=AS_OF_TS,
                raw_payload_dict={},
                official_open_price_float=100.5,
                open_price_source_str="test.open",
            )
        ]
    )
    store_obj.upsert_vplan_broker_ack_list(
        [
            BrokerOrderAck(
                decision_plan_id_int=decision_plan_obj.decision_plan_id_int,
                vplan_id_int=vplan_obj.vplan_id_int,
                account_route_str=release_obj.account_route_str,
                order_request_key_str="req_1",
                asset_str="AAPL",
                broker_order_type_str="MOO",
                local_submit_ack_bool=True,
                broker_response_ack_bool=True,
                ack_status_str="broker_acked",
                ack_source_str="test",
                broker_order_id_str="order_1",
                perm_id_int=123,
                response_timestamp_ts=AS_OF_TS,
                raw_payload_dict={},
            )
        ]
    )


def _mark_latest_vplan_completed_and_reconciled(db_path_obj: Path, release_obj: LiveRelease) -> None:
    store_obj = LiveStateStore(str(db_path_obj))
    latest_vplan_obj = store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
    assert latest_vplan_obj is not None
    store_obj.mark_vplan_status(int(latest_vplan_obj.vplan_id_int or 0), "completed")
    store_obj.mark_decision_plan_status(
        int(latest_vplan_obj.decision_plan_id_int),
        "completed",
    )
    store_obj.insert_vplan_reconciliation_snapshot(
        pod_id_str=release_obj.pod_id_str,
        decision_plan_id_int=int(latest_vplan_obj.decision_plan_id_int),
        vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
        stage_str="post_execution",
        reconciliation_result_obj=ReconciliationResult(
            passed_bool=True,
            status_str="passed",
            mismatch_dict={},
            model_position_map={"AAPL": 30.0},
            broker_position_map={"AAPL": 30.0},
            model_cash_float=7172.0,
            broker_cash_float=7172.0,
        ),
    )
    store_obj.upsert_pod_state(
        PodState(
            pod_id_str=release_obj.pod_id_str,
            user_id_str=release_obj.user_id_str,
            account_route_str=release_obj.account_route_str,
            position_amount_map={"AAPL": 30.0},
            cash_float=7172.0,
            total_value_float=10187.0,
            strategy_state_dict={"trade_id_int": 9},
            updated_timestamp_ts=AS_OF_TS,
        ),
        snapshot_stage_str="post_execution",
        snapshot_source_str="virtual_broker",
    )


def _seed_incubation_decision_only(db_path_obj: Path, release_obj: LiveRelease) -> None:
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    store_obj = LiveStateStore(str(db_path_obj))
    store_obj.upsert_release(release_obj)
    store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=AS_OF_TS,
            submission_timestamp_ts=AS_OF_TS,
            target_execution_timestamp_ts=AS_OF_TS,
            execution_policy_str=release_obj.execution_policy_str,
            decision_base_position_map={},
            snapshot_metadata_dict={},
            strategy_state_dict={},
            entry_target_weight_map_dict={"MSFT": 0.5},
            entry_priority_list=["MSFT"],
        )
    )
    store_obj.upsert_pod_state(
        PodState(
            pod_id_str=release_obj.pod_id_str,
            user_id_str=release_obj.user_id_str,
            account_route_str=release_obj.account_route_str,
            position_amount_map={},
            cash_float=10000.0,
            total_value_float=10000.0,
            strategy_state_dict={},
            updated_timestamp_ts=AS_OF_TS,
        ),
        snapshot_stage_str="post_execution",
        snapshot_source_str="virtual_broker",
    )


def _row_by_pod_id(summary_dict: dict, pod_id_str: str) -> dict:
    row_list = [
        row_dict
        for row_dict in summary_dict["pod_row_dict_list"]
        if row_dict["pod_id_str"] == pod_id_str
    ]
    assert len(row_list) == 1
    return row_list[0]


def _combined_env_by_mode(summary_dict: dict, mode_str: str) -> dict:
    environment_list = [
        environment_dict
        for environment_dict in summary_dict["combined_book_dict"]["environment_dict_list"]
        if environment_dict["mode_str"] == mode_str
    ]
    assert len(environment_list) == 1
    return environment_list[0]


def _combined_contribution_by_pod(environment_dict: dict, pod_id_str: str) -> dict:
    contribution_list = [
        contribution_dict
        for contribution_dict in environment_dict["contribution_dict_list"]
        if contribution_dict["pod_id_str"] == pod_id_str
    ]
    assert len(contribution_list) == 1
    return contribution_list[0]


def _wait_for_job_status(
    manager_obj: DiffJobManager,
    job_id_str: str,
    status_str: str,
    timeout_float: float = 2.0,
) -> dict:
    deadline_float = time.time() + timeout_float
    while time.time() < deadline_float:
        job_dict = manager_obj.get_job_dict(job_id_str)
        assert job_dict is not None
        if job_dict["status_str"] == status_str:
            return job_dict
        time.sleep(0.01)
    job_dict = manager_obj.get_job_dict(job_id_str)
    assert job_dict is not None
    raise AssertionError(f"job did not reach {status_str}: {job_dict}")


def _base_operator_row_dict(**override_dict) -> dict:
    row_dict = {
        "db_status_str": "ok",
        "pod_id_str": "pod_ok",
        "mode_str": "paper",
        "account_route_str": "DU_OK",
        "db_path_str": "state/pod.sqlite3",
        "error_str": None,
        "latest_decision_plan_status_str": "completed",
        "latest_vplan_status_str": "completed",
        "latest_vplan_id_int": 1,
        "latest_submit_ack_status_str": "complete",
        "latest_broker_snapshot_timestamp_str": AS_OF_TS.isoformat(),
        "latest_live_reference_snapshot_timestamp_str": AS_OF_TS.isoformat(),
        "latest_pod_state_timestamp_str": AS_OF_TS.isoformat(),
        "latest_event_timestamp_str": AS_OF_TS.isoformat(),
        "broker_order_count_int": 1,
        "broker_ack_count_int": 1,
        "fill_count_int": 1,
        "next_action_str": "wait",
        "reason_code_str": "no_due_work",
        "missing_ack_count_int": 0,
        "exception_count_int": 0,
        "latest_reconciliation_status_str": "passed",
        "latest_reconciliation_timestamp_str": AS_OF_TS.isoformat(),
        "latest_diff_status_str": "green",
        "latest_diff_timestamp_str": "20240102T000000Z",
        "latest_diff_open_issue_count_int": 0,
        "eod_snapshot_dict": {
            "status_str": "completed",
            "severity_str": "green",
            "latest_timestamp_str": AS_OF_TS.isoformat(),
            "latest_market_date_str": "2024-01-02",
            "recorded_timestamp_str": AS_OF_TS.isoformat(),
            "source_str": "broker",
            "equity_float": 10000.0,
            "cash_float": 1000.0,
            "position_count_int": 1,
            "expected_due_timestamp_str": AS_OF_TS.isoformat(),
            "expected_market_date_str": "2024-01-02",
            "same_session_bool": True,
            "unresolved_execution_bool": False,
            "detail_str": "Same-session EOD snapshot exists.",
        },
        "dtb3_download_status_str": "download_success",
        "dtb3_latest_observation_date_str": "2024-01-01",
        "dtb3_freshness_business_days_int": 1,
        "dtb3_source_name_str": "FRED",
        "dtb3_used_cache_bool": False,
    }
    row_dict.update(override_dict)
    return row_dict


def test_dashboard_config_loads_mapping_and_string_overrides(tmp_path: Path):
    config_path_obj = tmp_path / "dashboard_config.yaml"
    config_path_obj.write_text(
        "\n".join(
            [
                "db_overrides:",
                "  pod_string: state/shared.sqlite3",
                "  pod_mapping:",
                "    db_path: state/pod_mapping.sqlite3",
            ]
        ),
        encoding="utf-8",
    )

    config_obj = load_dashboard_config(str(config_path_obj))

    assert config_obj.db_override_map_dict == {
        "pod_string": "state/shared.sqlite3",
        "pod_mapping": "state/pod_mapping.sqlite3",
    }


def test_dashboard_required_action_mapping_covers_operator_states():
    case_list = [
        ("missing", {"db_status_str": "missing"}, "Setup DB", "gray"),
        ("empty", {"db_status_str": "empty"}, "No state yet", "gray"),
        ("db_error", {"db_status_str": "error", "error_str": "boom"}, "Manual review", "red"),
        (
            "decision_blocked",
            {"latest_decision_plan_status_str": "blocked"},
            "Manual review",
            "red",
        ),
        ("vplan_blocked", {"latest_vplan_status_str": "blocked"}, "Manual review", "red"),
        ("missing_ack", {"missing_ack_count_int": 1}, "Review broker ACK", "red"),
        ("reconcile_fail", {"exception_count_int": 1}, "Review reconcile", "red"),
        (
            "due_decision",
            {"next_action_str": "build_decision_plan", "reason_code_str": "ready_to_build_decision_plan"},
            "Build DecisionPlan",
            "yellow",
        ),
        (
            "ready_vplan_build",
            {"next_action_str": "build_vplan", "reason_code_str": "ready_to_build_vplan"},
            "Build VPlan",
            "yellow",
        ),
        (
            "waiting_submission",
            {"next_action_str": "wait", "reason_code_str": "waiting_for_submission_window"},
            "Wait submission window",
            "yellow",
        ),
        (
            "ready_submit",
            {"next_action_str": "submit_vplan", "reason_code_str": "vplan_ready"},
            "VPlan ready",
            "yellow",
        ),
        (
            "manual_review",
            {"next_action_str": "review_vplan", "reason_code_str": "vplan_ready"},
            "Review VPlan",
            "yellow",
        ),
        (
            "waiting_reconcile",
            {
                "next_action_str": "post_execution_reconcile",
                "reason_code_str": "waiting_for_post_execution_reconcile",
            },
            "Waiting reconcile",
            "yellow",
        ),
        ("idle", {}, "No action", "green"),
    ]

    for _case_name_str, override_dict, expected_label_str, expected_severity_str in case_list:
        action_dict = _build_required_action_dict(_base_operator_row_dict(**override_dict))
        assert action_dict["label_str"] == expected_label_str
        assert action_dict["severity_str"] == expected_severity_str


def test_dashboard_required_action_context_items_are_informational():
    submission_timestamp_str = "2024-01-03T14:20:00+00:00"
    target_execution_timestamp_str = "2024-01-03T14:30:00+00:00"
    action_dict = _build_required_action_dict(
        _base_operator_row_dict(
            next_action_str="wait",
            reason_code_str="waiting_for_submission_window",
            latest_decision_plan_submission_timestamp_str=submission_timestamp_str,
            latest_decision_plan_target_execution_timestamp_str=target_execution_timestamp_str,
            latest_broker_snapshot_timestamp_str=None,
            latest_event_timestamp_str=None,
            latest_reconciliation_timestamp_str=None,
        )
    )
    context_by_label_dict = {
        context_item_dict["label_str"]: context_item_dict
        for context_item_dict in action_dict["context_item_dict_list"]
    }

    assert action_dict["label_str"] == "Wait submission window"
    assert action_dict["reason_str"] == "waiting_for_submission_window"
    assert context_by_label_dict["Submission opens"]["value_str"] == submission_timestamp_str
    assert context_by_label_dict["Target execution"]["value_str"] == target_execution_timestamp_str
    assert context_by_label_dict["Broker snapshot"]["value_str"] == "unavailable"
    assert context_by_label_dict["Broker snapshot"]["severity_str"] == "gray"
    assert context_by_label_dict["Last event"]["value_str"] == "unavailable"
    assert context_by_label_dict["Last event"]["severity_str"] == "gray"
    assert context_by_label_dict["EOD"]["value_str"].startswith("completed / due ")


def test_dashboard_lifecycle_steps_cover_execution_evidence_and_diff():
    row_dict = _base_operator_row_dict(latest_diff_status_str="red", latest_diff_open_issue_count_int=2)
    step_dict_list = _build_lifecycle_step_dict_list(row_dict)

    step_by_key_dict = {
        step_dict["step_key_str"]: step_dict
        for step_dict in step_dict_list
    }

    assert [step_dict["step_key_str"] for step_dict in step_dict_list] == [
        "db",
        "decision",
        "vplan",
        "ack",
        "fill",
        "reconcile",
        "eod",
        "diff",
    ]
    assert step_by_key_dict["db"]["severity_str"] == "green"
    assert step_by_key_dict["decision"]["status_str"] == "completed"
    assert step_by_key_dict["vplan"]["status_str"] == "completed"
    assert step_by_key_dict["ack"]["severity_str"] == "green"
    assert step_by_key_dict["fill"]["status_str"] == "recorded"
    assert step_by_key_dict["reconcile"]["status_str"] == "passed"
    assert step_by_key_dict["eod"]["status_str"] == "completed"
    assert step_by_key_dict["diff"]["severity_str"] == "red"


def test_dashboard_alert_builder_sorts_and_suppresses_green_noise():
    green_row_dict = _base_operator_row_dict()
    green_row_dict["required_action_dict"] = _build_required_action_dict(green_row_dict)
    green_row_dict["data_freshness_dict"] = {
        "item_dict_list": [
            {
                "label_str": "Pod state",
                "value_str": AS_OF_TS.isoformat(),
                "severity_str": "green",
                "detail_str": "ok",
            }
        ]
    }
    red_row_dict = _base_operator_row_dict(
        pod_id_str="pod_red",
        missing_ack_count_int=2,
        latest_submit_ack_status_str="missing_critical",
    )
    red_row_dict["required_action_dict"] = _build_required_action_dict(red_row_dict)
    red_row_dict["data_freshness_dict"] = {"item_dict_list": []}
    yellow_row_dict = _base_operator_row_dict(
        pod_id_str="pod_yellow",
        next_action_str="build_vplan",
        reason_code_str="ready_to_build_vplan",
    )
    yellow_row_dict["required_action_dict"] = _build_required_action_dict(yellow_row_dict)
    yellow_row_dict["data_freshness_dict"] = {"item_dict_list": []}
    gray_row_dict = _base_operator_row_dict(
        pod_id_str="pod_gray",
        db_status_str="missing",
    )
    gray_row_dict["required_action_dict"] = _build_required_action_dict(gray_row_dict)
    gray_row_dict["data_freshness_dict"] = {"item_dict_list": []}

    alert_dict_list = _build_alert_dict_list(
        [green_row_dict, gray_row_dict, yellow_row_dict, red_row_dict]
    )

    assert [alert_dict["severity_str"] for alert_dict in alert_dict_list] == [
        "red",
        "yellow",
        "gray",
    ]
    assert [alert_dict["pod_id_str"] for alert_dict in alert_dict_list] == [
        "pod_red",
        "pod_yellow",
        "pod_gray",
    ]
    assert "pod_ok" not in {alert_dict["pod_id_str"] for alert_dict in alert_dict_list}


def test_dashboard_execution_report_separates_open_and_reference_slippage():
    report_dict = _build_execution_report_from_vplan_dict(
        {
            "pod_id_str": "pod_exec",
            "vplan_id_int": 7,
            "live_reference_price_json_str": json.dumps(
                {
                    "BUY_WORSE": 100.0,
                    "SELL_WORSE": 100.0,
                }
            ),
            "target_share_json_str": json.dumps(
                {
                    "BUY_WORSE": 1.0,
                    "SELL_WORSE": -1.0,
                }
            ),
            "current_broker_position_json_str": json.dumps(
                {
                    "BUY_WORSE": 0.0,
                    "SELL_WORSE": 0.0,
                }
            ),
            "order_delta_json_str": json.dumps(
                {
                    "BUY_WORSE": 1.0,
                    "SELL_WORSE": -1.0,
                }
            ),
            "vplan_row_dict_list": [
                {
                    "asset_str": "BUY_WORSE",
                    "current_share_float": 0.0,
                    "target_share_float": 1.0,
                    "order_delta_share_float": 1.0,
                    "live_reference_price_float": 100.0,
                },
                {
                    "asset_str": "SELL_WORSE",
                    "current_share_float": 0.0,
                    "target_share_float": -1.0,
                    "order_delta_share_float": -1.0,
                    "live_reference_price_float": 100.0,
                },
            ],
            "fill_row_dict_list": [
                {
                    "asset_str": "BUY_WORSE",
                    "fill_amount_float": 1.0,
                    "fill_price_float": 101.0,
                    "official_open_price_float": 100.0,
                    "open_price_source_str": "test.open",
                    "fill_timestamp_str": AS_OF_TS.isoformat(),
                },
                {
                    "asset_str": "SELL_WORSE",
                    "fill_amount_float": -1.0,
                    "fill_price_float": 99.0,
                    "official_open_price_float": None,
                    "open_price_source_str": None,
                    "fill_timestamp_str": AS_OF_TS.isoformat(),
                },
            ],
            "broker_order_row_dict_list": [],
            "broker_ack_row_dict_list": [],
        },
        {"BUY_WORSE": 1.0, "SELL_WORSE": -1.0},
    )

    execution_by_asset_dict = {
        row_dict["asset_str"]: row_dict
        for row_dict in report_dict["execution_row_dict_list"]
    }

    assert execution_by_asset_dict["BUY_WORSE"]["side_str"] == "buy"
    assert execution_by_asset_dict["BUY_WORSE"]["official_open_slippage_bps_float"] == 100.0
    assert execution_by_asset_dict["BUY_WORSE"]["vplan_reference_slippage_bps_float"] == 100.0
    assert execution_by_asset_dict["SELL_WORSE"]["side_str"] == "sell"
    assert execution_by_asset_dict["SELL_WORSE"]["official_open_slippage_bps_float"] is None
    assert execution_by_asset_dict["SELL_WORSE"]["vplan_reference_slippage_bps_float"] == 100.0
    assert report_dict["fill_with_official_open_count_int"] == 1
    assert report_dict["official_open_slippage_bps_float"] == 100.0
    assert report_dict["official_open_slippage_notional_float"] == 1.0
    assert report_dict["vplan_reference_slippage_bps_float"] == 100.0
    assert report_dict["vplan_reference_slippage_notional_float"] == 2.0
    assert report_dict["aggregate_official_open_slippage_notional_float"] == 1.0
    assert report_dict["aggregate_vplan_reference_slippage_notional_float"] == 2.0


def test_dashboard_discovers_enabled_pods_and_resolves_defaults_and_override(tmp_path: Path):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="paper_user",
        pod_id_str="pod_enabled_paper",
        mode_str="paper",
    )
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="incubation_user",
        pod_id_str="pod_enabled_incubation",
        mode_str="incubation",
    )
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="paper_user",
        pod_id_str="pod_disabled_paper",
        mode_str="paper",
        enabled_bool=False,
    )
    config_path_obj = tmp_path / "dashboard_config.yaml"
    override_db_path_str = str(tmp_path / "shared" / "live_state.sqlite3")
    _write_config(config_path_obj, {"pod_enabled_paper": override_db_path_str})

    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )
    target_map_dict = {target_obj.release_obj.pod_id_str: target_obj for target_obj in app_obj.get_target_list()}

    assert sorted(target_map_dict) == ["pod_enabled_incubation", "pod_enabled_paper"]
    assert target_map_dict["pod_enabled_paper"].db_path_str == override_db_path_str
    assert target_map_dict["pod_enabled_paper"].db_override_bool is True
    incubation_db_path_str = target_map_dict["pod_enabled_incubation"].db_path_str
    assert incubation_db_path_str.endswith(
        "alpha\\live\\state\\incubation\\pod_enabled_incubation.sqlite3"
    ) or incubation_db_path_str.endswith(
        "alpha/live/state/incubation/pod_enabled_incubation.sqlite3"
    )
    assert target_map_dict["pod_enabled_incubation"].db_override_bool is False

    release_obj = next(
        release_obj
        for release_obj in load_release_list(str(releases_root_path_obj))
        if release_obj.pod_id_str == "pod_enabled_paper"
    )
    default_db_path_str = resolve_db_path_for_release_str(release_obj, DashboardConfig())
    if release_obj.mode_str == "paper":
        assert default_db_path_str.endswith("alpha\\live\\state\\paper\\pod_enabled_paper.sqlite3") or default_db_path_str.endswith(
            "alpha/live/state/paper/pod_enabled_paper.sqlite3"
        )


def test_dashboard_incubation_rehearsal_keeps_sim_ledger_separate_from_paper_probe(tmp_path: Path):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="incubation_user",
        pod_id_str="pod_inc_done",
        mode_str="incubation",
    )
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="incubation_user",
        pod_id_str="pod_inc_wait",
        mode_str="incubation",
    )
    release_map_dict = {
        release_obj.pod_id_str: release_obj
        for release_obj in load_release_list(str(releases_root_path_obj))
    }
    done_db_path_obj = tmp_path / "state" / "pod_inc_done.sqlite3"
    wait_db_path_obj = tmp_path / "state" / "pod_inc_wait.sqlite3"
    _seed_decision_vplan_and_broker_rows(done_db_path_obj, release_map_dict["pod_inc_done"])
    _mark_latest_vplan_completed_and_reconciled(
        done_db_path_obj,
        release_map_dict["pod_inc_done"],
    )
    _seed_incubation_decision_only(wait_db_path_obj, release_map_dict["pod_inc_wait"])

    config_path_obj = tmp_path / "dashboard_config.yaml"
    _write_config(
        config_path_obj,
        {
            "pod_inc_done": str(done_db_path_obj),
            "pod_inc_wait": str(wait_db_path_obj),
        },
    )
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )

    summary_dict = build_dashboard_summary_dict(app_obj, as_of_ts=AS_OF_TS)
    done_row_dict = _row_by_pod_id(summary_dict, "pod_inc_done")
    wait_row_dict = _row_by_pod_id(summary_dict, "pod_inc_wait")

    assert done_row_dict["account_route_str"] == "SIM_pod_inc_done"
    assert wait_row_dict["account_route_str"] == "SIM_pod_inc_wait"
    assert done_row_dict["equity_float"] == 10187.0
    assert wait_row_dict["equity_float"] == 10000.0
    assert done_row_dict["rehearsal_status_dict"]["official_accounting_source_str"] == "incubation_sim_ledger"
    assert done_row_dict["rehearsal_status_dict"]["paper_probe_accounting_truth_bool"] is False
    assert done_row_dict["rehearsal_status_dict"]["paper_probe_status_str"] == "separate_probe_required"
    assert done_row_dict["rehearsal_status_dict"]["promotion_gate_status_str"] == "complete_one_cycle"
    assert done_row_dict["rehearsal_status_dict"]["completed_cycle_count_int"] == 1
    assert done_row_dict["rehearsal_status_dict"]["ibkr_reference_source_str"] == "test.snapshot"
    assert done_row_dict["rehearsal_status_dict"]["ibkr_open_price_source_str"] == "test.open"
    assert wait_row_dict["rehearsal_status_dict"]["promotion_gate_status_str"] == "incomplete"
    assert wait_row_dict["rehearsal_status_dict"]["last_cycle_status_str"] == "decision_only"

    detail_dict = build_pod_detail_dict(app_obj, "pod_inc_done", as_of_ts=AS_OF_TS)
    context_by_label_dict = {
        item_dict["label_str"]: item_dict
        for item_dict in detail_dict["required_action_dict"]["context_item_dict_list"]
    }
    assert context_by_label_dict["Rehearsal gate"]["value_str"] == "complete_one_cycle"
    assert context_by_label_dict["Paper probe"]["detail_str"] == (
        "Paper probe is evidence only; it does not count as SIM ledger P&L."
    )
    assert "Rehearsal" in {
        event_dict["source_str"]
        for event_dict in detail_dict["debug_story_dict"]["timeline_event_dict_list"]
    }


def test_dashboard_incubation_aggregates_default_pod_dbs_without_shared_state(
    tmp_path: Path,
    monkeypatch,
):
    releases_root_path_obj = tmp_path / "releases"
    state_root_path_obj = tmp_path / "state"
    old_shared_db_path_obj = tmp_path / "incubation_state.sqlite3"
    monkeypatch.setattr(
        runner_module,
        "DEFAULT_POD_STATE_ROOT_PATH_STR",
        str(state_root_path_obj),
    )
    monkeypatch.setattr(
        runner_module,
        "DEFAULT_INCUBATION_DB_PATH_STR",
        str(old_shared_db_path_obj),
    )
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="incubation_user",
        pod_id_str="pod_inc_a",
        mode_str="incubation",
    )
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="incubation_user",
        pod_id_str="pod_inc_b",
        mode_str="incubation",
    )
    release_map_dict = {
        release_obj.pod_id_str: release_obj
        for release_obj in load_release_list(str(releases_root_path_obj))
    }
    db_a_path_obj = state_root_path_obj / "incubation" / "pod_inc_a.sqlite3"
    db_b_path_obj = state_root_path_obj / "incubation" / "pod_inc_b.sqlite3"
    _seed_pod_state(old_shared_db_path_obj, release_map_dict["pod_inc_a"], 99999.0)
    _seed_pod_state(db_a_path_obj, release_map_dict["pod_inc_a"], 11111.0)
    _seed_pod_state(db_b_path_obj, release_map_dict["pod_inc_b"], 22222.0)
    config_path_obj = tmp_path / "dashboard_config.yaml"
    config_path_obj.write_text("db_overrides: {}\n", encoding="utf-8")

    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )

    target_map_dict = {
        target_obj.release_obj.pod_id_str: target_obj
        for target_obj in app_obj.get_target_list()
    }
    summary_dict = build_dashboard_summary_dict(app_obj, as_of_ts=AS_OF_TS)
    row_a_dict = _row_by_pod_id(summary_dict, "pod_inc_a")
    row_b_dict = _row_by_pod_id(summary_dict, "pod_inc_b")

    assert sorted(target_map_dict) == ["pod_inc_a", "pod_inc_b"]
    assert target_map_dict["pod_inc_a"].db_path_str == str(db_a_path_obj)
    assert target_map_dict["pod_inc_b"].db_path_str == str(db_b_path_obj)
    assert target_map_dict["pod_inc_a"].db_path_str != target_map_dict["pod_inc_b"].db_path_str
    assert row_a_dict["equity_float"] == 11111.0
    assert row_b_dict["equity_float"] == 22222.0
    assert row_a_dict["account_route_str"] == "SIM_pod_inc_a"
    assert row_b_dict["account_route_str"] == "SIM_pod_inc_b"


def test_dashboard_incubation_missing_per_pod_db_renders_safe_debug_output(
    tmp_path: Path,
    monkeypatch,
):
    releases_root_path_obj = tmp_path / "releases"
    state_root_path_obj = tmp_path / "state"
    monkeypatch.setattr(
        runner_module,
        "DEFAULT_POD_STATE_ROOT_PATH_STR",
        str(state_root_path_obj),
    )
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="incubation_user",
        pod_id_str="pod_inc_missing",
        mode_str="incubation",
    )
    config_path_obj = tmp_path / "dashboard_config.yaml"
    config_path_obj.write_text("db_overrides: {}\n", encoding="utf-8")

    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )

    summary_dict = build_dashboard_summary_dict(app_obj, as_of_ts=AS_OF_TS)
    row_dict = _row_by_pod_id(summary_dict, "pod_inc_missing")
    debug_summary_dict = row_dict["debug_summary_dict"]
    detail_dict = build_pod_detail_dict(app_obj, "pod_inc_missing", as_of_ts=AS_OF_TS)

    assert row_dict["mode_str"] == "incubation"
    assert row_dict["db_status_str"] == "missing"
    assert row_dict["db_path_str"] == str(
        state_root_path_obj / "incubation" / "pod_inc_missing.sqlite3"
    )
    assert debug_summary_dict["verdict_label_str"] == "Incubation DB missing"
    assert debug_summary_dict["primary_reason_str"] == (
        "Incubation SIM ledger DB has not been created yet."
    )
    assert debug_summary_dict["next_inspect_command_name_str"] == "status"
    assert (
        "Dashboard reads per-POD incubation state"
        in debug_summary_dict["primary_evidence_str"]
    )
    assert detail_dict["debug_story_dict"]["verdict_dict"]["next_inspect_command_name_str"] == "status"


def test_dashboard_summary_handles_missing_empty_shared_and_pod_db(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    releases_root_path_obj = tmp_path / "releases"
    for pod_id_str in ("pod_missing", "pod_empty", "pod_shared", "pod_specific"):
        _write_release_manifest(
            releases_root_path_obj,
            user_id_str="paper_user",
            pod_id_str=pod_id_str,
            mode_str="paper",
        )
    empty_db_path_obj = tmp_path / "state" / "empty.sqlite3"
    empty_db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(str(empty_db_path_obj)).close()
    shared_db_path_obj = tmp_path / "state" / "live_state.sqlite3"
    specific_db_path_obj = tmp_path / "state" / "paper" / "pod_specific.sqlite3"
    release_map_dict = {release_obj.pod_id_str: release_obj for release_obj in load_release_list(str(releases_root_path_obj))}
    _seed_pod_state(shared_db_path_obj, release_map_dict["pod_shared"], 12345.0)
    _seed_pod_state(specific_db_path_obj, release_map_dict["pod_specific"], 67890.0)
    config_path_obj = tmp_path / "dashboard_config.yaml"
    _write_config(
        config_path_obj,
        {
            "pod_empty": str(empty_db_path_obj),
            "pod_shared": str(shared_db_path_obj),
            "pod_specific": str(specific_db_path_obj),
        },
    )
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )
    summary_dict = build_dashboard_summary_dict(app_obj, as_of_ts=AS_OF_TS)

    assert _row_by_pod_id(summary_dict, "pod_missing")["db_status_str"] == "missing"
    assert _row_by_pod_id(summary_dict, "pod_missing")["health_str"] == "gray"
    assert _row_by_pod_id(summary_dict, "pod_missing")["required_action_dict"]["label_str"] == "Setup DB"
    assert _row_by_pod_id(summary_dict, "pod_missing")["debug_summary_dict"]["verdict_label_str"] == "DB missing"
    assert _row_by_pod_id(summary_dict, "pod_missing")["lifecycle_step_dict_list"][0]["status_str"] == "missing"
    assert _row_by_pod_id(summary_dict, "pod_missing")["eod_snapshot_dict"]["status_str"] == "not_applicable"
    assert _row_by_pod_id(summary_dict, "pod_empty")["db_status_str"] == "empty"
    assert _row_by_pod_id(summary_dict, "pod_empty")["health_str"] == "gray"
    assert _row_by_pod_id(summary_dict, "pod_empty")["required_action_dict"]["label_str"] == "No state yet"
    assert _row_by_pod_id(summary_dict, "pod_empty")["debug_summary_dict"]["verdict_label_str"] == "DB empty"
    assert _row_by_pod_id(summary_dict, "pod_empty")["eod_snapshot_dict"]["status_str"] == "not_applicable"
    assert _row_by_pod_id(summary_dict, "pod_shared")["db_status_str"] == "ok"
    assert _row_by_pod_id(summary_dict, "pod_shared")["equity_float"] == 12345.0
    assert _row_by_pod_id(summary_dict, "pod_shared")["position_count_int"] == 1
    assert "data_freshness_dict" in _row_by_pod_id(summary_dict, "pod_shared")
    norgate_item_dict = next(
        item_dict
        for item_dict in _row_by_pod_id(summary_dict, "pod_shared")["data_freshness_dict"]["item_dict_list"]
        if item_dict["label_str"] == "Norgate"
    )
    assert norgate_item_dict["severity_str"] == "green"
    assert _row_by_pod_id(summary_dict, "pod_shared")["data_freshness_dict"]["norgate_status_str"] == "direct"
    assert _row_by_pod_id(summary_dict, "pod_shared")["eod_snapshot_dict"]["status_str"] == "waiting"
    assert _row_by_pod_id(summary_dict, "pod_specific")["db_status_str"] == "ok"
    assert _row_by_pod_id(summary_dict, "pod_specific")["equity_float"] == 67890.0
    assert "alert_dict_list" in summary_dict
    assert "alert_summary_dict" in summary_dict
    assert summary_dict["alert_summary_dict"]["gray_count_int"] >= 2
    assert {
        (alert_dict["pod_id_str"], alert_dict["alert_type_str"], alert_dict["severity_str"])
        for alert_dict in summary_dict["alert_dict_list"]
    } >= {
        ("pod_missing", "db", "gray"),
        ("pod_empty", "db", "gray"),
    }


def test_dashboard_norgate_snapshot_status_includes_sync_debug_payload(tmp_path: Path, monkeypatch):
    release_obj = _build_release_obj("pod_sync")
    snapshot_root_path_obj = tmp_path / "snapshots"
    snapshot_root_path_obj.mkdir(parents=True)
    status_path_obj = snapshot_root_path_obj / ".client_sync_status.json"
    status_path_obj.write_text(
        json.dumps(
            {
                "status_str": "ready",
                "reason_code_str": "local_snapshot_ready",
                "required_profile_list": ["norgate_eod_sp500_pit"],
                "snapshot_date_by_profile_dict": {"norgate_eod_sp500_pit": "2026-05-19"},
                "manifest_hash_by_profile_dict": {"norgate_eod_sp500_pit": "abcdef1234567890"},
                "error_by_profile_dict": {},
                "gate_reason_by_release_id_dict": {
                    release_obj.release_id_str: "carry_forward_snapshot_ready"
                },
                "release_id_list": [release_obj.release_id_str],
                "pod_id_list": [release_obj.pod_id_str],
                "last_attempt_utc_str": None,
                "last_success_utc_str": "2026-05-20T07:19:04+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(snapshot_root_path_obj))
    monkeypatch.setattr(
        norgate_sync_module,
        "load_valid_snapshot_manifest",
        lambda profile_str: SimpleNamespace(
            snapshot_date_ts=datetime(2026, 5, 19, tzinfo=UTC),
            manifest_hash_str="abcdef1234567890",
        ),
    )
    monkeypatch.setattr(
        norgate_sync_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {"reason_code_str": "carry_forward_snapshot_ready"},
    )

    status_dict = norgate_sync_module.build_norgate_snapshot_status_dict(release_obj, AS_OF_TS)

    assert status_dict["sync_stage_label_str"] == "Local snapshot ready"
    assert status_dict["reason_code_str"] == "local_snapshot_ready"
    assert status_dict["profile_status_dict_list"] == [
        {
            "profile_str": "norgate_eod_sp500_pit",
            "snapshot_date_str": "2026-05-19",
            "manifest_hash_str": "abcdef1234567890",
            "manifest_hash_prefix_str": "abcdef123456",
            "error_str": None,
        }
    ]
    assert status_dict["release_gate_status_dict_list"] == [
        {
            "release_id_str": release_obj.release_id_str,
            "pod_id_str": release_obj.pod_id_str,
            "gate_reason_code_str": "carry_forward_snapshot_ready",
        }
    ]


def test_dashboard_norgate_release_gate_rows_are_scoped_to_selected_pod(tmp_path: Path, monkeypatch):
    release_obj = _build_release_obj("pod_selected")
    other_release_id_str = "user_001.pod_other.paper.v1"
    snapshot_root_path_obj = tmp_path / "snapshots"
    snapshot_root_path_obj.mkdir(parents=True)
    (snapshot_root_path_obj / ".client_sync_status.json").write_text(
        json.dumps(
            {
                "status_str": "ready",
                "reason_code_str": "local_snapshot_ready",
                "required_profile_list": ["norgate_eod_sp500_pit"],
                "snapshot_date_by_profile_dict": {"norgate_eod_sp500_pit": "2026-05-19"},
                "manifest_hash_by_profile_dict": {"norgate_eod_sp500_pit": "abcdef1234567890"},
                "gate_reason_by_release_id_dict": {
                    other_release_id_str: "carry_forward_snapshot_ready"
                },
                "release_id_list": [other_release_id_str],
                "pod_id_list": ["pod_other"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(snapshot_root_path_obj))
    monkeypatch.setattr(
        norgate_sync_module,
        "load_valid_snapshot_manifest",
        lambda profile_str: SimpleNamespace(
            snapshot_date_ts=datetime(2026, 5, 19, tzinfo=UTC),
            manifest_hash_str="abcdef1234567890",
        ),
    )
    monkeypatch.setattr(
        norgate_sync_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {"reason_code_str": "carry_forward_snapshot_ready"},
    )

    status_dict = norgate_sync_module.build_norgate_snapshot_status_dict(release_obj, AS_OF_TS)

    assert status_dict["release_gate_status_dict_list"] == [
        {
            "release_id_str": release_obj.release_id_str,
            "pod_id_str": release_obj.pod_id_str,
            "gate_reason_code_str": "carry_forward_snapshot_ready",
        }
    ]


def test_dashboard_norgate_ready_status_respects_stale_build_gate(tmp_path: Path, monkeypatch):
    release_obj = _build_release_obj("pod_sync")
    snapshot_root_path_obj = tmp_path / "snapshots"
    snapshot_root_path_obj.mkdir(parents=True)
    (snapshot_root_path_obj / ".client_sync_status.json").write_text(
        json.dumps(
            {
                "status_str": "ready",
                "reason_code_str": "local_snapshot_ready",
                "required_profile_list": ["norgate_eod_sp500_pit"],
                "snapshot_date_by_profile_dict": {"norgate_eod_sp500_pit": "2026-05-19"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(snapshot_root_path_obj))
    monkeypatch.setattr(
        norgate_sync_module,
        "load_valid_snapshot_manifest",
        lambda profile_str: SimpleNamespace(
            snapshot_date_ts=datetime(2026, 5, 19, tzinfo=UTC),
            manifest_hash_str="abcdef1234567890",
        ),
    )
    monkeypatch.setattr(
        norgate_sync_module.scheduler_utils,
        "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {"reason_code_str": "snapshot_window_expired"},
    )

    status_dict = norgate_sync_module.build_norgate_snapshot_status_dict(release_obj, AS_OF_TS)

    assert status_dict["status_str"] == "ready"
    assert status_dict["severity_str"] == "red"
    assert status_dict["sync_stage_label_str"] == "Snapshot window expired"
    assert status_dict["release_gate_status_dict_list"][0]["gate_reason_code_str"] == "snapshot_window_expired"


def test_dashboard_norgate_status_labels_waiting_and_failed_stages(tmp_path: Path, monkeypatch):
    release_obj = _build_release_obj("pod_sync")
    snapshot_root_path_obj = tmp_path / "snapshots"
    snapshot_root_path_obj.mkdir(parents=True)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(snapshot_root_path_obj))
    monkeypatch.setattr(
        norgate_sync_module,
        "load_valid_snapshot_manifest",
        lambda profile_str: (_ for _ in ()).throw(RuntimeError("manifest missing")),
    )

    (snapshot_root_path_obj / ".client_sync_status.json").write_text(
        json.dumps(
            {
                "status_str": "waiting",
                "reason_code_str": "api_config_missing",
                "required_profile_list": ["norgate_eod_sp500_pit"],
                "error_str": "Missing Norgate API config: NORGATE_API_URL.",
            }
        ),
        encoding="utf-8",
    )
    waiting_status_dict = norgate_sync_module.build_norgate_snapshot_status_dict(
        release_obj,
        AS_OF_TS,
    )
    assert waiting_status_dict["sync_stage_label_str"] == "API config missing"
    assert waiting_status_dict["last_error_str"] == "Missing Norgate API config: NORGATE_API_URL."

    (snapshot_root_path_obj / ".client_sync_status.json").write_text(
        json.dumps(
            {
                "status_str": "failed",
                "reason_code_str": "sync_failed",
                "required_profile_list": ["norgate_eod_sp500_pit"],
                "error_str": "HTTP 401",
            }
        ),
        encoding="utf-8",
    )
    failed_status_dict = norgate_sync_module.build_norgate_snapshot_status_dict(
        release_obj,
        AS_OF_TS,
    )
    assert failed_status_dict["sync_stage_label_str"] == "Sync failed"
    assert failed_status_dict["severity_str"] == "red"
    assert failed_status_dict["last_error_str"] == "HTTP 401"


def test_dashboard_norgate_direct_mode_has_compact_debug_payload(monkeypatch):
    release_obj = _build_release_obj("pod_direct")
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")

    status_dict = norgate_sync_module.build_norgate_snapshot_status_dict(release_obj, AS_OF_TS)

    assert status_dict["sync_stage_label_str"] == "Direct Norgate"
    assert status_dict["profile_status_dict_list"] == []
    assert status_dict["release_gate_status_dict_list"] == []


def test_dashboard_norgate_debug_story_marks_post_sync_data_load_failure():
    detail_dict = {
        "pod_row_dict": {
            "db_status_str": "ok",
            "norgate_snapshot_status_dict": {
                "data_source_mode_str": "snapshot",
                "status_str": "ready",
                "severity_str": "green",
                "sync_stage_label_str": "Local snapshot ready",
                "reason_code_str": "local_snapshot_ready",
                "build_gate_reason_code_str": "carry_forward_snapshot_ready",
                "last_sync_utc_str": "2026-05-20T07:19:04+00:00",
            },
            "eod_snapshot_dict": {"status_str": "not_applicable", "severity_str": "gray"},
        },
        "event_dict_list": [
            {
                "event_name_str": "scheduler_error_retry",
                "next_phase_str": "build_decision_plan",
                "event_timestamp_str": "2026-05-20T07:20:18+00:00",
                "error_str": "Snapshot has no rows for requested date range.",
            }
        ],
    }

    story_dict = dashboard_module._build_debug_story_dict(detail_dict)

    assert any(
        event_dict["label_str"] == "Post-sync data load failed"
        for event_dict in story_dict["timeline_event_dict_list"]
    )
    norgate_evidence_dict = next(
        item_dict
        for item_dict in story_dict["evidence_item_dict_list"]
        if item_dict["label_str"] == "Norgate sync"
    )
    assert norgate_evidence_dict["value_str"] == "ready"
    assert "stage=Local snapshot ready" in norgate_evidence_dict["detail_str"]


def test_dashboard_norgate_debug_story_ignores_pre_sync_data_load_failure():
    detail_dict = {
        "pod_row_dict": {
            "db_status_str": "ok",
            "norgate_snapshot_status_dict": {
                "data_source_mode_str": "snapshot",
                "status_str": "ready",
                "severity_str": "green",
                "sync_stage_label_str": "Local snapshot ready",
                "reason_code_str": "local_snapshot_ready",
                "build_gate_reason_code_str": "carry_forward_snapshot_ready",
                "last_sync_utc_str": "2026-05-20T07:19:04+00:00",
            },
            "eod_snapshot_dict": {"status_str": "not_applicable", "severity_str": "gray"},
        },
        "event_dict_list": [
            {
                "event_name_str": "scheduler_error_retry",
                "next_phase_str": "build_decision_plan",
                "event_timestamp_str": "2026-05-20T07:18:18+00:00",
                "error_str": "Old snapshot data-load error.",
            }
        ],
    }

    story_dict = dashboard_module._build_debug_story_dict(detail_dict)

    assert not any(
        event_dict["label_str"] == "Post-sync data load failed"
        for event_dict in story_dict["timeline_event_dict_list"]
    )


def test_dashboard_norgate_debug_story_uses_fresh_post_sync_data_load_failure():
    detail_dict = {
        "pod_row_dict": {
            "db_status_str": "ok",
            "norgate_snapshot_status_dict": {
                "data_source_mode_str": "snapshot",
                "status_str": "ready",
                "severity_str": "green",
                "sync_stage_label_str": "Local snapshot ready",
                "reason_code_str": "local_snapshot_ready",
                "build_gate_reason_code_str": "carry_forward_snapshot_ready",
                "last_sync_utc_str": "2026-05-20T07:19:04+00:00",
            },
            "eod_snapshot_dict": {"status_str": "not_applicable", "severity_str": "gray"},
        },
        "event_dict_list": [
            {
                "event_name_str": "scheduler_error_retry",
                "next_phase_str": "build_decision_plan",
                "event_timestamp_str": "2026-05-20T07:18:18+00:00",
                "error_str": "Old snapshot data-load error.",
            },
            {
                "event_name_str": "scheduler_error_retry",
                "next_phase_str": "build_decision_plan",
                "event_timestamp_str": "2026-05-20T07:20:18+00:00",
                "error_str": "Fresh snapshot data-load error.",
            },
        ],
    }

    story_dict = dashboard_module._build_debug_story_dict(detail_dict)
    failure_event_dict = next(
        event_dict
        for event_dict in story_dict["timeline_event_dict_list"]
        if event_dict["label_str"] == "Post-sync data load failed"
    )

    assert failure_event_dict["detail_str"] == "Fresh snapshot data-load error."


def test_dashboard_event_loader_matches_norgate_sync_pod_id_list(tmp_path: Path):
    log_path_obj = tmp_path / "events.jsonl"
    log_path_obj.write_text(
        json.dumps(
            {
                "event_name_str": "norgate_snapshot_sync_ready",
                "event_timestamp_str": "2026-05-20T07:19:04+00:00",
                "pod_id_list": ["pod_sync"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    event_dict_list = dashboard_module.load_recent_event_dict_list(
        log_path_str=str(log_path_obj),
        pod_id_str="pod_sync",
    )

    assert [event_dict["event_name_str"] for event_dict in event_dict_list] == [
        "norgate_snapshot_sync_ready"
    ]


def test_dashboard_html_contains_norgate_sync_panel():
    assert "Norgate Sync" in DASHBOARD_HTML_STR
    assert "renderNorgateSyncSection" in DASHBOARD_HTML_STR


def test_dashboard_main_loads_config_env_before_serving(monkeypatch, tmp_path: Path):
    call_dict = {}

    def _fake_load_config_env_file(*, override_existing_bool: bool):
        call_dict["override_existing_bool"] = override_existing_bool

    def _fake_serve_dashboard(**kwargs):
        call_dict["serve_kwargs"] = kwargs

    monkeypatch.setattr(dashboard_module, "load_config_env_file", _fake_load_config_env_file)
    monkeypatch.setattr(dashboard_module, "serve_dashboard", _fake_serve_dashboard)

    result_int = dashboard_module.main(
        [
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "8999",
            "--releases-root",
            str(tmp_path / "releases"),
            "--config",
            str(tmp_path / "dashboard.yaml"),
            "--results-root",
            str(tmp_path / "results"),
            "--event-log",
            str(tmp_path / "events.jsonl"),
        ]
    )

    assert result_int == 0
    assert call_dict["override_existing_bool"] is True
    assert call_dict["serve_kwargs"]["port_int"] == 8999


def test_dashboard_summary_does_not_use_diff_status_for_overall_health(tmp_path: Path):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="paper_user",
        pod_id_str="pod_diff_health",
        mode_str="paper",
    )
    release_obj = load_release_list(str(releases_root_path_obj))[0]
    db_path_obj = tmp_path / "state" / "pod_diff_health.sqlite3"
    _seed_pod_state(db_path_obj, release_obj, 12345.0)

    results_root_path_obj = tmp_path / "results"
    artifact_dir_path_obj = (
        results_root_path_obj
        / "live_reference_compare"
        / "paper"
        / "pod_diff_health"
        / "20240102T000000Z"
    )
    artifact_dir_path_obj.mkdir(parents=True)
    (artifact_dir_path_obj / "summary.json").write_text(
        json.dumps({"status_str": "red", "open_issue_count_int": 3}),
        encoding="utf-8",
    )

    config_path_obj = tmp_path / "dashboard_config.yaml"
    _write_config(config_path_obj, {"pod_diff_health": str(db_path_obj)})
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(results_root_path_obj),
    )

    summary_dict = build_dashboard_summary_dict(app_obj, as_of_ts=AS_OF_TS)
    row_dict = _row_by_pod_id(summary_dict, "pod_diff_health")

    assert row_dict["latest_diff_status_str"] == "red"
    assert row_dict["health_str"] != "red"
    assert row_dict["required_action_dict"]["severity_str"] != "red"
    assert row_dict["debug_summary_dict"]["severity_str"] == "red"
    assert row_dict["debug_summary_dict"]["verdict_label_str"] == "DIFF red"
    assert row_dict["debug_summary_dict"]["next_inspect_command_name_str"] == "compare_reference"
    assert row_dict["lifecycle_step_dict_list"][-1]["step_key_str"] == "diff"
    assert row_dict["lifecycle_step_dict_list"][-1]["severity_str"] == "red"
    assert {
        (alert_dict["alert_type_str"], alert_dict["severity_str"], alert_dict["pod_id_str"])
        for alert_dict in summary_dict["alert_dict_list"]
    } >= {("diff", "red", "pod_diff_health")}


def test_dashboard_eod_snapshot_states_and_alerts(tmp_path: Path):
    releases_root_path_obj = tmp_path / "releases"
    for pod_id_str in ("pod_eod_done", "pod_eod_missing", "pod_eod_blocked"):
        _write_release_manifest(
            releases_root_path_obj,
            user_id_str="paper_user",
            pod_id_str=pod_id_str,
            mode_str="paper",
        )
    release_map_dict = {
        release_obj.pod_id_str: release_obj
        for release_obj in load_release_list(str(releases_root_path_obj))
    }
    eod_timestamp_ts = datetime(2024, 1, 2, 21, 30, tzinfo=UTC)
    done_db_path_obj = tmp_path / "state" / "pod_eod_done.sqlite3"
    missing_db_path_obj = tmp_path / "state" / "pod_eod_missing.sqlite3"
    blocked_db_path_obj = tmp_path / "state" / "pod_eod_blocked.sqlite3"
    _seed_pod_state(done_db_path_obj, release_map_dict["pod_eod_done"], 10000.0)
    _seed_eod_pod_state(
        done_db_path_obj,
        release_map_dict["pod_eod_done"],
        total_value_float=10125.0,
        updated_timestamp_ts=eod_timestamp_ts,
    )
    _seed_pod_state(missing_db_path_obj, release_map_dict["pod_eod_missing"], 10000.0)
    _seed_pod_state(blocked_db_path_obj, release_map_dict["pod_eod_blocked"], 10000.0)
    _seed_decision_vplan_and_broker_rows(
        blocked_db_path_obj,
        release_map_dict["pod_eod_blocked"],
    )
    config_path_obj = tmp_path / "dashboard_config.yaml"
    _write_config(
        config_path_obj,
        {
            "pod_eod_done": str(done_db_path_obj),
            "pod_eod_missing": str(missing_db_path_obj),
            "pod_eod_blocked": str(blocked_db_path_obj),
        },
    )
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )

    summary_dict = build_dashboard_summary_dict(app_obj, as_of_ts=eod_timestamp_ts)
    done_eod_dict = _row_by_pod_id(summary_dict, "pod_eod_done")["eod_snapshot_dict"]
    missing_eod_dict = _row_by_pod_id(summary_dict, "pod_eod_missing")["eod_snapshot_dict"]
    blocked_eod_dict = _row_by_pod_id(summary_dict, "pod_eod_blocked")["eod_snapshot_dict"]

    assert done_eod_dict["status_str"] == "completed"
    assert done_eod_dict["severity_str"] == "green"
    assert done_eod_dict["latest_market_date_str"] == "2024-01-02"
    assert done_eod_dict["equity_float"] == 10125.0
    assert missing_eod_dict["status_str"] == "due_missing"
    assert missing_eod_dict["severity_str"] == "yellow"
    assert blocked_eod_dict["status_str"] == "blocked_by_execution"
    assert blocked_eod_dict["severity_str"] == "gray"

    alert_key_set = {
        (alert_dict["pod_id_str"], alert_dict["alert_type_str"], alert_dict["label_str"])
        for alert_dict in summary_dict["alert_dict_list"]
    }
    assert ("pod_eod_missing", "freshness", "EOD Snapshot freshness") in alert_key_set
    assert all(
        not (
            alert_dict["pod_id_str"] == "pod_eod_blocked"
            and alert_dict["label_str"] == "EOD Snapshot freshness"
        )
        for alert_dict in summary_dict["alert_dict_list"]
    )
    freshness_item_list = _row_by_pod_id(summary_dict, "pod_eod_missing")[
        "data_freshness_dict"
    ]["item_dict_list"]
    assert any(item_dict["label_str"] == "EOD Snapshot" for item_dict in freshness_item_list)


def test_dashboard_detail_pod_pnl_uses_eod_history_and_latest_same_day_snapshot(tmp_path: Path):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="paper_user",
        pod_id_str="pod_pnl",
        mode_str="paper",
    )
    release_obj = load_release_list(str(releases_root_path_obj))[0]
    db_path_obj = tmp_path / "state" / "pod_pnl.sqlite3"
    _seed_eod_pod_state(
        db_path_obj,
        release_obj,
        total_value_float=10000.0,
        cash_float=1000.0,
        updated_timestamp_ts=datetime(2024, 1, 2, 21, 30, tzinfo=UTC),
    )
    _seed_eod_pod_state(
        db_path_obj,
        release_obj,
        total_value_float=10050.0,
        cash_float=1010.0,
        updated_timestamp_ts=datetime(2024, 1, 3, 21, 30, tzinfo=UTC),
    )
    _seed_eod_pod_state(
        db_path_obj,
        release_obj,
        total_value_float=10075.0,
        cash_float=1025.0,
        updated_timestamp_ts=datetime(2024, 1, 3, 21, 35, tzinfo=UTC),
    )
    _seed_eod_pod_state(
        db_path_obj,
        release_obj,
        total_value_float=10025.0,
        cash_float=990.0,
        updated_timestamp_ts=datetime(2024, 1, 4, 21, 30, tzinfo=UTC),
    )
    config_path_obj = tmp_path / "dashboard_config.yaml"
    _write_config(config_path_obj, {"pod_pnl": str(db_path_obj)})
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )

    detail_dict = build_pod_detail_dict(
        app_obj,
        "pod_pnl",
        as_of_ts=datetime(2024, 1, 4, 22, 0, tzinfo=UTC),
    )
    pod_pnl_dict = detail_dict["pod_pnl_dict"]
    point_by_market_date_dict = {
        point_dict["market_date_str"]: point_dict
        for point_dict in pod_pnl_dict["equity_point_dict_list"]
    }

    assert pod_pnl_dict["status_str"] == "available"
    assert pod_pnl_dict["source_str"] == "pod_state_history.eod"
    assert pod_pnl_dict["point_count_int"] == 3
    assert pod_pnl_dict["latest_market_date_str"] == "2024-01-04"
    assert pod_pnl_dict["latest_equity_float"] == 10025.0
    assert pod_pnl_dict["previous_market_date_str"] == "2024-01-03"
    assert pod_pnl_dict["previous_equity_float"] == 10075.0
    assert pod_pnl_dict["daily_pnl_float"] == -50.0
    assert abs(pod_pnl_dict["daily_pnl_pct_float"] - (-50.0 / 10075.0)) < 1e-12
    assert pod_pnl_dict["since_start_pnl_float"] == 25.0
    assert abs(pod_pnl_dict["since_start_pnl_pct_float"] - (25.0 / 10000.0)) < 1e-12
    assert point_by_market_date_dict["2024-01-03"]["equity_float"] == 10075.0
    assert point_by_market_date_dict["2024-01-03"]["cash_float"] == 1025.0
    assert point_by_market_date_dict["2024-01-02"]["daily_pnl_float"] is None
    assert point_by_market_date_dict["2024-01-02"]["since_start_pnl_float"] == 0.0


def test_dashboard_combined_book_uses_strict_and_carry_forward_eod_series(tmp_path: Path):
    releases_root_path_obj = tmp_path / "releases"
    for pod_id_str in ("pod_alpha", "pod_beta"):
        _write_release_manifest(
            releases_root_path_obj,
            user_id_str="paper_user",
            pod_id_str=pod_id_str,
            mode_str="paper",
        )
    release_map_dict = {
        release_obj.pod_id_str: release_obj
        for release_obj in load_release_list(str(releases_root_path_obj))
    }
    alpha_db_path_obj = tmp_path / "state" / "pod_alpha.sqlite3"
    beta_db_path_obj = tmp_path / "state" / "pod_beta.sqlite3"
    _seed_eod_pod_state(
        alpha_db_path_obj,
        release_map_dict["pod_alpha"],
        total_value_float=10000.0,
        updated_timestamp_ts=datetime(2024, 1, 2, 21, 30, tzinfo=UTC),
    )
    _seed_eod_pod_state(
        alpha_db_path_obj,
        release_map_dict["pod_alpha"],
        total_value_float=10050.0,
        updated_timestamp_ts=datetime(2024, 1, 3, 21, 30, tzinfo=UTC),
    )
    _seed_eod_pod_state(
        alpha_db_path_obj,
        release_map_dict["pod_alpha"],
        total_value_float=10075.0,
        updated_timestamp_ts=datetime(2024, 1, 3, 21, 35, tzinfo=UTC),
    )
    _seed_eod_pod_state(
        alpha_db_path_obj,
        release_map_dict["pod_alpha"],
        total_value_float=10150.0,
        updated_timestamp_ts=datetime(2024, 1, 4, 21, 30, tzinfo=UTC),
    )
    _seed_eod_pod_state(
        beta_db_path_obj,
        release_map_dict["pod_beta"],
        total_value_float=20000.0,
        updated_timestamp_ts=datetime(2024, 1, 2, 21, 30, tzinfo=UTC),
    )
    _seed_eod_pod_state(
        beta_db_path_obj,
        release_map_dict["pod_beta"],
        total_value_float=20100.0,
        updated_timestamp_ts=datetime(2024, 1, 3, 21, 30, tzinfo=UTC),
    )
    config_path_obj = tmp_path / "dashboard_config.yaml"
    _write_config(
        config_path_obj,
        {
            "pod_alpha": str(alpha_db_path_obj),
            "pod_beta": str(beta_db_path_obj),
        },
    )
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )

    summary_dict = build_dashboard_summary_dict(
        app_obj,
        as_of_ts=datetime(2024, 1, 4, 22, 0, tzinfo=UTC),
    )
    paper_env_dict = _combined_env_by_mode(summary_dict, "paper")
    alpha_contribution_dict = _combined_contribution_by_pod(
        paper_env_dict,
        "pod_alpha",
    )
    beta_contribution_dict = _combined_contribution_by_pod(
        paper_env_dict,
        "pod_beta",
    )

    assert paper_env_dict["pod_count_int"] == 2
    assert paper_env_dict["status_str"] == "available_with_warnings"
    assert paper_env_dict["latest_common_market_date_str"] == "2024-01-03"
    assert paper_env_dict["strict_point_count_int"] == 2
    assert paper_env_dict["strict_latest_equity_float"] == 30175.0
    assert paper_env_dict["strict_daily_pnl_float"] == 175.0
    assert abs(paper_env_dict["strict_daily_pnl_pct_float"] - (175.0 / 30000.0)) < 1e-12
    assert paper_env_dict["latest_operational_market_date_str"] == "2024-01-04"
    assert paper_env_dict["carry_forward_latest_equity_float"] == 30250.0
    assert paper_env_dict["carry_forward_point_count_int"] == 3
    assert paper_env_dict["carry_forward_equity_point_dict_list"][-1]["stale_pod_count_int"] == 1
    assert paper_env_dict["carry_forward_equity_point_dict_list"][-1]["stale_pod_id_list"] == ["pod_beta"]
    assert {warning_dict["warning_type_str"] for warning_dict in paper_env_dict["warning_dict_list"]} >= {
        "stale_eod"
    }
    assert alpha_contribution_dict["latest_market_date_str"] == "2024-01-04"
    assert alpha_contribution_dict["strict_daily_pnl_float"] == 75.0
    assert alpha_contribution_dict["carry_forward_status_str"] == "current"
    assert len(alpha_contribution_dict["equity_point_dict_list"]) == 3
    assert beta_contribution_dict["latest_market_date_str"] == "2024-01-03"
    assert beta_contribution_dict["strict_daily_pnl_float"] == 100.0
    assert beta_contribution_dict["carry_forward_status_str"] == "carried_forward"
    assert beta_contribution_dict["freshness_warning_str"] == "Carried from 2024-01-03"
    assert len(beta_contribution_dict["equity_point_dict_list"]) == 2


def test_dashboard_combined_book_separates_modes_and_warns_without_failing(
    tmp_path: Path,
):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="live_user",
        pod_id_str="pod_live",
        mode_str="live",
        account_route_str="U_LIVE_TEST",
    )
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="paper_user",
        pod_id_str="pod_paper_a",
        mode_str="paper",
        account_route_str="DU_SHARED",
    )
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="paper_user",
        pod_id_str="pod_paper_b",
        mode_str="paper",
        account_route_str="DU_SHARED",
    )
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="paper_user",
        pod_id_str="pod_paper_missing",
        mode_str="paper",
    )
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="incubation_user",
        pod_id_str="pod_inc",
        mode_str="incubation",
    )
    release_map_dict = {
        release_obj.pod_id_str: release_obj
        for release_obj in load_release_list(str(releases_root_path_obj))
    }
    live_db_path_obj = tmp_path / "state" / "pod_live.sqlite3"
    paper_a_db_path_obj = tmp_path / "state" / "pod_paper_a.sqlite3"
    paper_b_db_path_obj = tmp_path / "state" / "pod_paper_b.sqlite3"
    inc_db_path_obj = tmp_path / "state" / "pod_inc.sqlite3"
    paper_b_db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(str(paper_b_db_path_obj)).close()
    _seed_eod_pod_state(
        live_db_path_obj,
        release_map_dict["pod_live"],
        total_value_float=50000.0,
        updated_timestamp_ts=datetime(2024, 1, 3, 21, 30, tzinfo=UTC),
    )
    _seed_eod_pod_state(
        paper_a_db_path_obj,
        release_map_dict["pod_paper_a"],
        total_value_float=10000.0,
        updated_timestamp_ts=datetime(2024, 1, 3, 21, 30, tzinfo=UTC),
    )
    _seed_eod_pod_state(
        inc_db_path_obj,
        release_map_dict["pod_inc"],
        total_value_float=75000.0,
        updated_timestamp_ts=datetime(2024, 1, 3, 21, 30, tzinfo=UTC),
        snapshot_source_str="virtual_broker",
    )
    config_path_obj = tmp_path / "dashboard_config.yaml"
    _write_config(
        config_path_obj,
        {
            "pod_live": str(live_db_path_obj),
            "pod_paper_a": str(paper_a_db_path_obj),
            "pod_paper_b": str(paper_b_db_path_obj),
            "pod_paper_missing": str(tmp_path / "state" / "pod_paper_missing.sqlite3"),
            "pod_inc": str(inc_db_path_obj),
        },
    )
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )

    summary_dict = build_dashboard_summary_dict(app_obj, as_of_ts=AS_OF_TS)
    live_env_dict = _combined_env_by_mode(summary_dict, "live")
    paper_env_dict = _combined_env_by_mode(summary_dict, "paper")
    incubation_env_dict = _combined_env_by_mode(summary_dict, "incubation")

    assert live_env_dict["pod_count_int"] == 1
    assert live_env_dict["strict_latest_equity_float"] == 50000.0
    assert paper_env_dict["pod_count_int"] == 3
    assert paper_env_dict["strict_point_count_int"] == 0
    assert paper_env_dict["carry_forward_latest_equity_float"] == 10000.0
    assert incubation_env_dict["pod_count_int"] == 1
    assert incubation_env_dict["strict_latest_equity_float"] == 75000.0
    assert {
        warning_dict["warning_type_str"]
        for warning_dict in paper_env_dict["warning_dict_list"]
    } >= {"duplicate_account_route", "empty_db", "missing_db"}
    assert _combined_contribution_by_pod(
        paper_env_dict,
        "pod_paper_b",
    )["freshness_warning_str"] == "DB empty"
    assert _combined_contribution_by_pod(
        paper_env_dict,
        "pod_paper_missing",
    )["freshness_warning_str"] == "DB missing"


def test_dashboard_detail_parses_decision_plan_and_preserves_execution_rows(tmp_path: Path):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="paper_user",
        pod_id_str="pod_detail",
        mode_str="paper",
    )
    release_obj = load_release_list(str(releases_root_path_obj))[0]
    db_path_obj = tmp_path / "state" / "pod_detail.sqlite3"
    _seed_pod_state(db_path_obj, release_obj, 43210.0)
    _seed_decision_vplan_and_broker_rows(db_path_obj, release_obj)
    config_path_obj = tmp_path / "dashboard_config.yaml"
    _write_config(config_path_obj, {"pod_detail": str(db_path_obj)})
    event_log_path_obj = tmp_path / "live_events.jsonl"
    event_log_path_obj.write_text(
        json.dumps(
            {
                "timestamp_str": AS_OF_TS.isoformat(),
                "pod_id_str": "pod_detail",
                "severity_str": "INFO",
                "event_name_str": "dashboard.test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
        event_log_path_str=str(event_log_path_obj),
    )

    detail_dict = build_pod_detail_dict(app_obj, "pod_detail", as_of_ts=AS_OF_TS)

    decision_dict = detail_dict["latest_decision_plan_dict"]
    assert decision_dict["decision_book_type_str"] == "full_target_weight_book"
    assert decision_dict["display_target_weight_map_dict"] == {"AAPL": 0.6, "TLT": 0.4}
    assert decision_dict["decision_base_position_map_dict"] == {"AAPL": 2.0}
    assert decision_dict["snapshot_metadata_dict"]["strategy_family_str"] == "dashboard_test"
    assert decision_dict["latest_vplan_status_str"] == "submitted"
    assert detail_dict["latest_vplan_dict"]["vplan_row_dict_list"][0]["asset_str"] == "AAPL"
    assert detail_dict["latest_vplan_dict"]["broker_order_row_dict_list"][0]["status_str"] == "Filled"
    assert detail_dict["latest_vplan_dict"]["fill_row_dict_list"][0]["fill_price_float"] == 101.0
    assert detail_dict["latest_vplan_dict"]["broker_ack_row_dict_list"][0]["ack_status_str"] == "broker_acked"
    execution_report_dict = detail_dict["latest_execution_report_dict"]
    execution_row_dict = execution_report_dict["execution_row_dict_list"][0]
    assert execution_report_dict["fill_with_official_open_count_int"] == 1
    assert execution_report_dict["residual_count_int"] == 1
    assert execution_report_dict["official_open_slippage_bps_float"] == 49.7512437811
    assert execution_report_dict["official_open_slippage_notional_float"] == 14.0
    assert execution_report_dict["vplan_reference_slippage_bps_float"] == 100.0
    assert execution_report_dict["vplan_reference_slippage_notional_float"] == 28.0
    assert execution_report_dict["aggregate_official_open_slippage_notional_float"] == 14.0
    assert execution_report_dict["aggregate_vplan_reference_slippage_notional_float"] == 28.0
    assert execution_row_dict["asset_str"] == "AAPL"
    assert execution_row_dict["side_str"] == "buy"
    assert execution_row_dict["official_open_slippage_bps_float"] == 49.7512437811
    assert execution_row_dict["vplan_reference_slippage_bps_float"] == 100.0
    assert detail_dict["required_action_dict"]["label_str"] == "Waiting reconcile"
    required_context_by_label_dict = {
        context_item_dict["label_str"]: context_item_dict
        for context_item_dict in detail_dict["required_action_dict"]["context_item_dict_list"]
    }
    assert required_context_by_label_dict["Broker snapshot"]["severity_str"] == "gray"
    assert required_context_by_label_dict["Last event"]["value_str"] == AS_OF_TS.isoformat()
    assert required_context_by_label_dict["EOD"]["label_str"] == "EOD"
    assert detail_dict["lifecycle_step_dict_list"][3]["step_key_str"] == "ack"
    assert detail_dict["lifecycle_step_dict_list"][3]["severity_str"] == "green"
    assert detail_dict["lifecycle_step_dict_list"][4]["status_str"] == "recorded"
    assert detail_dict["data_freshness_dict"]["latest_event_timestamp_str"] == AS_OF_TS.isoformat()
    assert detail_dict["data_freshness_dict"]["dtb3_freshness_business_days_int"] == 1
    debug_story_dict = detail_dict["debug_story_dict"]
    assert debug_story_dict["verdict_dict"]["verdict_label_str"] == "Waiting reconcile"
    assert debug_story_dict["verdict_dict"]["next_inspect_command_name_str"] == "show_vplan"
    source_set = {
        timeline_event_dict["source_str"]
        for timeline_event_dict in debug_story_dict["timeline_event_dict_list"]
    }
    assert {
        "DecisionPlan",
        "VPlan",
        "ACK",
        "Broker order",
        "Fill",
        "Reconcile",
        "EOD",
        "Live reference",
        "DIFF",
        "DTB3/FRED",
        "Event log",
    }.issubset(source_set)
    assert debug_story_dict["timeline_event_dict_list"][0]["source_str"] == "EOD"
    assert any(
        timeline_event_dict["timestamp_str"] == AS_OF_TS.isoformat()
        for timeline_event_dict in debug_story_dict["timeline_event_dict_list"]
    )
    assert any(
        blocker_dict["label_str"] == "Waiting reconcile"
        for blocker_dict in debug_story_dict["blocker_dict_list"]
    )
    assert [
        command_dict["command_name_str"]
        for command_dict in debug_story_dict["recommended_command_dict_list"]
    ] == ["show_vplan", "status", "show_decision_plan"]


def test_dashboard_summary_flags_missing_ack_and_reconcile_failure_actions(tmp_path: Path):
    releases_root_path_obj = tmp_path / "releases"
    for pod_id_str in ("pod_ack", "pod_reconcile"):
        _write_release_manifest(
            releases_root_path_obj,
            user_id_str="paper_user",
            pod_id_str=pod_id_str,
            mode_str="paper",
        )
    release_map_dict = {release_obj.pod_id_str: release_obj for release_obj in load_release_list(str(releases_root_path_obj))}
    ack_db_path_obj = tmp_path / "state" / "pod_ack.sqlite3"
    reconcile_db_path_obj = tmp_path / "state" / "pod_reconcile.sqlite3"
    _seed_pod_state(ack_db_path_obj, release_map_dict["pod_ack"], 10000.0)
    _seed_decision_vplan_and_broker_rows(ack_db_path_obj, release_map_dict["pod_ack"])
    _seed_pod_state(reconcile_db_path_obj, release_map_dict["pod_reconcile"], 10000.0)
    _seed_decision_vplan_and_broker_rows(reconcile_db_path_obj, release_map_dict["pod_reconcile"])

    with sqlite3.connect(str(ack_db_path_obj)) as connection_obj:
        connection_obj.execute(
            """
            UPDATE vplan
            SET missing_ack_count_int = 1,
                submit_ack_status_str = 'missing_critical'
            """
        )

    with sqlite3.connect(str(reconcile_db_path_obj)) as connection_obj:
        row_obj = connection_obj.execute(
            "SELECT decision_plan_id_int, vplan_id_int FROM vplan LIMIT 1"
        ).fetchone()
    LiveStateStore(str(reconcile_db_path_obj)).insert_vplan_reconciliation_snapshot(
        pod_id_str="pod_reconcile",
        decision_plan_id_int=int(row_obj[0]),
        vplan_id_int=int(row_obj[1]),
        stage_str="post_execution",
        reconciliation_result_obj=ReconciliationResult(
            passed_bool=False,
            status_str="failed",
            mismatch_dict={"AAPL": {"model": 30.0, "broker": 29.0}},
            model_position_map={"AAPL": 30.0},
            broker_position_map={"AAPL": 29.0},
            model_cash_float=100.0,
            broker_cash_float=100.0,
        ),
    )

    config_path_obj = tmp_path / "dashboard_config.yaml"
    _write_config(
        config_path_obj,
        {
            "pod_ack": str(ack_db_path_obj),
            "pod_reconcile": str(reconcile_db_path_obj),
        },
    )
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )
    artifact_dir_path_obj = (
        tmp_path
        / "results"
        / "live_reference_compare"
        / "paper"
        / "pod_reconcile"
        / "20240102T000000Z"
    )
    artifact_dir_path_obj.mkdir(parents=True)
    (artifact_dir_path_obj / "summary.json").write_text(
        json.dumps({"status_str": "red", "open_issue_count_int": 9}),
        encoding="utf-8",
    )

    summary_dict = build_dashboard_summary_dict(app_obj, as_of_ts=AS_OF_TS)
    ack_row_dict = _row_by_pod_id(summary_dict, "pod_ack")
    reconcile_row_dict = _row_by_pod_id(summary_dict, "pod_reconcile")

    assert ack_row_dict["required_action_dict"]["label_str"] == "Review broker ACK"
    assert ack_row_dict["required_action_dict"]["severity_str"] == "red"
    assert ack_row_dict["lifecycle_step_dict_list"][3]["severity_str"] == "red"
    assert reconcile_row_dict["required_action_dict"]["label_str"] == "Review reconcile"
    assert reconcile_row_dict["required_action_dict"]["severity_str"] == "red"
    assert reconcile_row_dict["lifecycle_step_dict_list"][5]["severity_str"] == "red"
    assert ack_row_dict["debug_summary_dict"]["verdict_label_str"] == "Broker ACK missing"
    assert ack_row_dict["debug_summary_dict"]["severity_str"] == "red"
    assert reconcile_row_dict["latest_diff_status_str"] == "red"
    assert reconcile_row_dict["debug_summary_dict"]["verdict_label_str"] == "Reconcile blocked"
    assert reconcile_row_dict["debug_summary_dict"]["primary_evidence_str"].startswith("reconcile_status=failed")
    alert_key_list = [
        (alert_dict["severity_str"], alert_dict["pod_id_str"], alert_dict["alert_type_str"])
        for alert_dict in summary_dict["alert_dict_list"]
    ]
    assert alert_key_list[0][0] == "red"
    assert ("red", "pod_ack", "broker_ack") in alert_key_list
    assert ("red", "pod_reconcile", "reconcile") in alert_key_list


def test_dashboard_detail_missing_db_returns_safe_empty_sections(tmp_path: Path):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="paper_user",
        pod_id_str="pod_missing_detail",
        mode_str="paper",
    )
    config_path_obj = tmp_path / "dashboard_config.yaml"
    _write_config(config_path_obj, {"pod_missing_detail": str(tmp_path / "missing.sqlite3")})
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )

    detail_dict = build_pod_detail_dict(app_obj, "pod_missing_detail", as_of_ts=AS_OF_TS)

    assert detail_dict["pod_row_dict"]["db_status_str"] == "missing"
    assert detail_dict["required_action_dict"]["label_str"] == "Setup DB"
    assert detail_dict["lifecycle_step_dict_list"][0]["status_str"] == "missing"
    assert detail_dict["data_freshness_dict"]["pod_state_updated_timestamp_str"] is None
    assert detail_dict["eod_snapshot_dict"]["status_str"] == "not_applicable"
    assert detail_dict["pod_pnl_dict"]["status_str"] == "unavailable"
    assert detail_dict["pod_pnl_dict"]["equity_point_dict_list"] == []
    assert detail_dict["latest_decision_plan_dict"] is None
    assert detail_dict["latest_vplan_dict"] is None
    assert detail_dict["latest_execution_report_dict"] is None
    assert detail_dict["debug_story_dict"]["verdict_dict"]["verdict_label_str"] == "DB missing"
    assert detail_dict["debug_story_dict"]["blocker_dict_list"][0]["inspect_command_name_str"] == "status"
    assert detail_dict["debug_story_dict"]["recommended_command_dict_list"][0]["command_name_str"] == "status"


def test_dashboard_html_uses_structured_operator_sections():
    assert "dashboard-section" in DASHBOARD_HTML_STR
    assert "dashboard-section-header" in DASHBOARD_HTML_STR
    assert "--section-accent" in DASHBOARD_HTML_STR
    assert "border-left: 5px solid var(--section-accent)" in DASHBOARD_HTML_STR
    assert ".dashboard-section-header h2::before" in DASHBOARD_HTML_STR
    assert ".console-section {" in DASHBOARD_HTML_STR
    assert ".attention-panel {" in DASHBOARD_HTML_STR
    assert ".detail-workspace {" in DASHBOARD_HTML_STR
    assert "Console Status" in DASHBOARD_HTML_STR
    assert "Alert Box" in DASHBOARD_HTML_STR
    assert "Combined Book" in DASHBOARD_HTML_STR
    assert "selectedCombinedBookEnvironmentList" in DASHBOARD_HTML_STR
    assert "renderCombinedBookAbsoluteCurve" in DASHBOARD_HTML_STR
    assert "renderCombinedBookIndexedCurve" in DASHBOARD_HTML_STR
    assert "Combined book equity ($)" in DASHBOARD_HTML_STR
    assert "Strategy comparison (indexed to 100)" in DASHBOARD_HTML_STR
    assert "bold line is Combined" in DASHBOARD_HTML_STR
    assert "indexLineToBase100" in DASHBOARD_HTML_STR
    assert 'data-combined-toggle-pod' in DASHBOARD_HTML_STR
    assert "Strict common-date equity" not in DASHBOARD_HTML_STR
    assert "Carry-forward operational equity" not in DASHBOARD_HTML_STR
    assert "Selected POD equity curves" not in DASHBOARD_HTML_STR
    assert "POD List" in DASHBOARD_HTML_STR
    assert "Selected POD / What We Learned" in DASHBOARD_HTML_STR
    assert "read-only / copy-only" in DASHBOARD_HTML_STR
    assert "detail-workspace" in DASHBOARD_HTML_STR
    assert "renderDecisionSection" in DASHBOARD_HTML_STR
    assert "Alert Inbox" in DASHBOARD_HTML_STR
    assert "alert-counts" in DASHBOARD_HTML_STR
    assert "alert-toggle-button" in DASHBOARD_HTML_STR
    assert "console-status-strip" in DASHBOARD_HTML_STR
    assert "Environment" in DASHBOARD_HTML_STR
    assert "Visible PODs" in DASHBOARD_HTML_STR
    assert "Needs Action" in DASHBOARD_HTML_STR
    assert "Last Refresh" in DASHBOARD_HTML_STR
    assert "Selected POD" in DASHBOARD_HTML_STR
    assert "renderConsoleStatusStrip" in DASHBOARD_HTML_STR
    assert "visibleOperatorRows" in DASHBOARD_HTML_STR
    assert "alertExpanded: false" in DASHBOARD_HTML_STR
    assert "Show audit alerts" in DASHBOARD_HTML_STR
    assert "Hide audit alerts" in DASHBOARD_HTML_STR
    assert "alertPanel.classList.toggle('collapsed'" in DASHBOARD_HTML_STR
    assert "position: sticky" in DASHBOARD_HTML_STR
    assert "renderAlertInbox" in DASHBOARD_HTML_STR
    assert "URLSearchParams" in DASHBOARD_HTML_STR
    assert "localStorage" in DASHBOARD_HTML_STR
    assert "Clear selection" in DASHBOARD_HTML_STR
    assert "clearSelectedPod" in DASHBOARD_HTML_STR
    assert "clearSelectedPodIfHidden" in DASHBOARD_HTML_STR
    assert "rowMatchesSelectedEnvironment" in DASHBOARD_HTML_STR
    assert "podId: null" in DASHBOARD_HTML_STR
    assert "window.localStorage.removeItem('dashboard.pod')" in DASHBOARD_HTML_STR
    assert "window.localStorage.setItem('dashboard.pod'" not in DASHBOARD_HTML_STR
    assert "window.localStorage.getItem('dashboard.pod')" not in DASHBOARD_HTML_STR
    assert "url.searchParams.delete('pod')" in DASHBOARD_HTML_STR
    assert "url.searchParams.set('pod'" not in DASHBOARD_HTML_STR
    assert "dashboard.tab." in DASHBOARD_HTML_STR
    assert "detail-tabs" in DASHBOARD_HTML_STR
    assert "data-detail-tab" in DASHBOARD_HTML_STR
    assert "Debug" in DASHBOARD_HTML_STR
    assert "renderDebugSection" in DASHBOARD_HTML_STR
    assert "Root Cause" in DASHBOARD_HTML_STR
    assert "Blocker Chain" in DASHBOARD_HTML_STR
    assert "Evidence Timeline" in DASHBOARD_HTML_STR
    assert "Model vs Broker Snapshot" in DASHBOARD_HTML_STR
    assert "Suggested Copy Commands" in DASHBOARD_HTML_STR
    assert "Attention Queue" in DASHBOARD_HTML_STR
    assert "Needs Action" in DASHBOARD_HTML_STR
    assert "Waiting / Parked" in DASHBOARD_HTML_STR
    assert "Missing / Stale Data" in DASHBOARD_HTML_STR
    assert "Current State" in DASHBOARD_HTML_STR
    assert "Latest Evidence" in DASHBOARD_HTML_STR
    assert "Next Inspect" in DASHBOARD_HTML_STR
    assert "Changed since last refresh" in DASHBOARD_HTML_STR
    assert ".attention-row.red {\n      border-left: 4px solid var(--red);" in DASHBOARD_HTML_STR
    assert ".attention-row.red { box-shadow" not in DASHBOARD_HTML_STR
    assert "buildAttentionQueueRows" in DASHBOARD_HTML_STR
    assert "operatorStateSentence" in DASHBOARD_HTML_STR
    assert "operatorReasonSentence" in DASHBOARD_HTML_STR
    assert "operatorEvidenceSentence" in DASHBOARD_HTML_STR
    assert "hasPodChangedSinceLastRefresh" in DASHBOARD_HTML_STR
    assert "operator-filter-chips" not in DASHBOARD_HTML_STR
    assert "data-operator-filter" not in DASHBOARD_HTML_STR
    assert "filter-chip" not in DASHBOARD_HTML_STR
    assert "operatorFilter" not in DASHBOARD_HTML_STR
    assert "environmentLabel" in DASHBOARD_HTML_STR
    assert "Incubation" in DASHBOARD_HTML_STR
    assert "No incubation PODs are currently enabled." in DASHBOARD_HTML_STR
    assert "Incubation SIM ledger DB has not been created yet." in DASHBOARD_HTML_STR
    assert "Incubation DB exists, but no rehearsal cycle has been recorded yet." in DASHBOARD_HTML_STR
    assert "Incubation is waiting for one complete SIM ledger rehearsal cycle." in DASHBOARD_HTML_STR
    assert "Incubation completed at least one SIM ledger rehearsal cycle." in DASHBOARD_HTML_STR
    assert "Paper probe is evidence only; it does not count as SIM ledger P&L." in DASHBOARD_HTML_STR
    assert "Dashboard reads per-POD incubation state." in DASHBOARD_HTML_STR
    assert "Reconcile is blocked." in DASHBOARD_HTML_STR
    assert "EOD is blocked because execution is unresolved." in DASHBOARD_HTML_STR
    assert "Reference DIFF is red." in DASHBOARD_HTML_STR
    assert "State DB is missing." in DASHBOARD_HTML_STR
    assert "Reconcile is waiting for broker truth." in DASHBOARD_HTML_STR
    assert "safeInspectCommandNameList" in DASHBOARD_HTML_STR
    assert "audit-detail-drawer" in DASHBOARD_HTML_STR
    assert "Show audit evidence timeline" in DASHBOARD_HTML_STR
    assert "Overview" in DASHBOARD_HTML_STR
    assert "PnL" in DASHBOARD_HTML_STR
    assert "renderPnlSection" in DASHBOARD_HTML_STR
    assert "renderPnlCurve" in DASHBOARD_HTML_STR
    assert "Daily PnL $" in DASHBOARD_HTML_STR
    assert "Since-start PnL %" in DASHBOARD_HTML_STR
    assert "Market date" in DASHBOARD_HTML_STR
    assert "pnl-curve" in DASHBOARD_HTML_STR
    assert "pnl-curve-plot" in DASHBOARD_HTML_STR
    assert "pnl-y-axis" in DASHBOARD_HTML_STR
    assert "pnl-x-axis" in DASHBOARD_HTML_STR
    assert "pnl-grid-line" in DASHBOARD_HTML_STR
    assert "pnl-point-layer" in DASHBOARD_HTML_STR
    assert "renderPnlTable" in DASHBOARD_HTML_STR
    assert "metricSigned" in DASHBOARD_HTML_STR
    assert "signedValueClass" in DASHBOARD_HTML_STR
    assert "Unified Rehearsal" in DASHBOARD_HTML_STR
    assert "renderRehearsalSection" in DASHBOARD_HTML_STR
    assert "Paper fills count as SIM P&L" in DASHBOARD_HTML_STR
    assert "Paper probe status" in DASHBOARD_HTML_STR
    assert ".signed-value.positive" in DASHBOARD_HTML_STR
    assert ".signed-value.negative" in DASHBOARD_HTML_STR
    assert "vector-effect: non-scaling-stroke" in DASHBOARD_HTML_STR
    assert "preserveAspectRatio=\"xMidYMid meet\"" in DASHBOARD_HTML_STR
    assert "preserveAspectRatio=\"none\"" not in DASHBOARD_HTML_STR
    assert "Only one EOD equity point is available" in DASHBOARD_HTML_STR
    assert "Decision" in DASHBOARD_HTML_STR
    assert "Execution" in DASHBOARD_HTML_STR
    assert "renderExecutionSection" in DASHBOARD_HTML_STR
    assert "Open slippage bps / $" in DASHBOARD_HTML_STR
    assert "Reference slippage bps / $" in DASHBOARD_HTML_STR
    assert "Open slippage bps" in DASHBOARD_HTML_STR
    assert "Reference slippage bps" in DASHBOARD_HTML_STR
    assert "fmtUnavailable" in DASHBOARD_HTML_STR
    assert "Freshness" in DASHBOARD_HTML_STR
    assert "Open Logger" in DASHBOARD_HTML_STR
    assert "logger-drawer" in DASHBOARD_HTML_STR
    assert "logger-filter" in DASHBOARD_HTML_STR
    assert "/events" in DASHBOARD_HTML_STR
    assert "<h3>Commands</h3>" in DASHBOARD_HTML_STR
    assert "command-drawer" in DASHBOARD_HTML_STR
    assert "show_decision_plan" in DASHBOARD_HTML_STR
    assert "compare_reference" in DASHBOARD_HTML_STR
    assert "renderLifecycleSection" in DASHBOARD_HTML_STR
    assert "Lifecycle Progress" in DASHBOARD_HTML_STR
    assert "Current" in DASHBOARD_HTML_STR
    assert "deriveCurrentLifecycleStepKey" in DASHBOARD_HTML_STR
    assert "isLifecycleActiveWaitingStep" in DASHBOARD_HTML_STR
    assert "['waiting', 'blocked_by_execution'].includes(status)" in DASHBOARD_HTML_STR
    assert "firstActiveWaitingStep" in DASHBOARD_HTML_STR
    assert "operatorLifecycleSentence" in DASHBOARD_HTML_STR
    assert "renderLifecycleRail" in DASHBOARD_HTML_STR
    assert "renderLifecycleArrow" in DASHBOARD_HTML_STR
    assert "renderReferenceCheckStep" in DASHBOARD_HTML_STR
    assert "lifecycle-step-card" in DASHBOARD_HTML_STR
    assert "lifecycle-arrow" in DASHBOARD_HTML_STR
    assert "lifecycle-reference-check" in DASHBOARD_HTML_STR
    assert "Reference Check" in DASHBOARD_HTML_STR
    assert "firstRedStep" in DASHBOARD_HTML_STR
    assert "firstYellowStep" in DASHBOARD_HTML_STR
    assert "Idle / no blocking lifecycle step" in DASHBOARD_HTML_STR
    assert "renderActionContextItems" in DASHBOARD_HTML_STR
    assert "action-context-grid" in DASHBOARD_HTML_STR
    assert "action-context-value" in DASHBOARD_HTML_STR
    assert ".action-context-item.green," in DASHBOARD_HTML_STR
    assert ".action-context-item .action-context-value { color: var(--text); }" in DASHBOARD_HTML_STR
    assert "America/New_York" in DASHBOARD_HTML_STR
    assert "DASHBOARD_TIME_ZONE_LABEL_STR = 'NYC'" in DASHBOARD_HTML_STR
    assert "DASHBOARD_COMPACT_TIMESTAMP_PATTERN" in DASHBOARD_HTML_STR
    assert "function formatTimestamp" in DASHBOARD_HTML_STR
    assert "function formatCompactTimestamp" in DASHBOARD_HTML_STR
    assert "formatTimestamp(new Date().toISOString())" in DASHBOARD_HTML_STR
    assert "new Date().toLocaleTimeString()" not in DASHBOARD_HTML_STR
    assert "summary-action" in DASHBOARD_HTML_STR
    assert "summary-action-label" in DASHBOARD_HTML_STR
    assert '<table class="pod-table">' in DASHBOARD_HTML_STR
    assert "renderEodSnapshotSection" in DASHBOARD_HTML_STR
    assert "<h3>EOD Snapshot</h3>" in DASHBOARD_HTML_STR
    assert "<h3>Data Freshness</h3>" in DASHBOARD_HTML_STR
    assert "<h3>Overview</h3>" in DASHBOARD_HTML_STR
    assert "<h3>Broker</h3>" in DASHBOARD_HTML_STR
    assert 'data-copy-command="eod_snapshot"' not in DASHBOARD_HTML_STR
    assert 'data-copy-command="tick"' not in DASHBOARD_HTML_STR
    assert 'data-copy-command="submit_vplan"' not in DASHBOARD_HTML_STR
    assert 'data-copy-command="post_execution_reconcile"' not in DASHBOARD_HTML_STR
    assert "<pre" not in DASHBOARD_HTML_STR
    assert "JSON.stringify" not in DASHBOARD_HTML_STR


def test_latest_diff_artifact_discovery_uses_newest_timestamp(tmp_path: Path):
    artifact_root_path_obj = tmp_path / "results" / "live_reference_compare" / "paper" / "pod_diff"
    old_dir_path_obj = artifact_root_path_obj / "20240101T000000Z"
    new_dir_path_obj = artifact_root_path_obj / "20240102T000000Z"
    old_dir_path_obj.mkdir(parents=True)
    new_dir_path_obj.mkdir(parents=True)
    (old_dir_path_obj / "summary.json").write_text('{"status_str": "green"}', encoding="utf-8")
    (new_dir_path_obj / "summary.json").write_text(
        json.dumps(
            {
                "status_str": "red",
                "equity_tracking_error_float": 12.5,
                "open_issue_count_int": 2,
            }
        ),
        encoding="utf-8",
    )
    (new_dir_path_obj / "index.html").write_text("<html></html>", encoding="utf-8")
    (new_dir_path_obj / "equity_compare.png").write_bytes(b"png")

    diff_dict = find_latest_diff_artifact_dict(
        results_root_path_str=str(tmp_path / "results"),
        mode_str="paper",
        pod_id_str="pod_diff",
    )

    assert diff_dict["status_str"] == "red"
    assert diff_dict["artifact_timestamp_str"] == "20240102T000000Z"
    assert diff_dict["equity_tracking_error_float"] == 12.5
    assert diff_dict["open_issue_count_int"] == 2
    assert diff_dict["html_url_str"] == "/artifacts/live_reference_compare/paper/pod_diff/20240102T000000Z/index.html"
    assert diff_dict["equity_png_url_str"] == "/artifacts/live_reference_compare/paper/pod_diff/20240102T000000Z/equity_compare.png"


def test_diff_job_manager_reports_running_succeeded_and_failed(tmp_path: Path):
    started_event_obj = threading.Event()
    finish_event_obj = threading.Event()
    release_obj = _build_release_obj()
    target_obj = DashboardPodTarget(
        release_obj=release_obj,
        db_path_str=str(tmp_path / "pod.sqlite3"),
        db_override_bool=True,
    )

    def fake_success_runner(
        pod_target_obj: DashboardPodTarget,
        releases_root_path_str: str,
        results_root_path_str: str,
        as_of_ts: datetime,
    ) -> dict:
        started_event_obj.set()
        assert pod_target_obj.release_obj.pod_id_str == "pod_job"
        assert releases_root_path_str == "releases-root"
        assert results_root_path_str == "results-root"
        finish_event_obj.wait(timeout=2.0)
        return {"status_str": "green", "as_of_timestamp_str": as_of_ts.isoformat()}

    manager_obj = DiffJobManager(fake_success_runner)
    job_obj = manager_obj.start_job(
        target_obj,
        releases_root_path_str="releases-root",
        results_root_path_str="results-root",
    )
    assert job_obj.to_dict()["status_str"] in {"queued", "running"}
    assert started_event_obj.wait(timeout=2.0)
    running_job_dict = _wait_for_job_status(manager_obj, job_obj.job_id_str, "running")
    assert running_job_dict["result_dict"] is None
    finish_event_obj.set()
    succeeded_job_dict = _wait_for_job_status(manager_obj, job_obj.job_id_str, "succeeded")
    assert succeeded_job_dict["result_dict"]["status_str"] == "green"

    def fake_failure_runner(
        pod_target_obj: DashboardPodTarget,
        releases_root_path_str: str,
        results_root_path_str: str,
        as_of_ts: datetime,
    ) -> dict:
        raise RuntimeError("diff exploded")

    failing_manager_obj = DiffJobManager(fake_failure_runner)
    failed_job_obj = failing_manager_obj.start_job(
        target_obj,
        releases_root_path_str="releases-root",
        results_root_path_str="results-root",
    )
    failed_job_dict = _wait_for_job_status(failing_manager_obj, failed_job_obj.job_id_str, "failed")
    assert failed_job_dict["error_str"] == "diff exploded"


def test_dashboard_http_get_pods_smoke(tmp_path: Path):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(
        releases_root_path_obj,
        user_id_str="paper_user",
        pod_id_str="pod_http",
        mode_str="paper",
    )
    release_obj = load_release_list(str(releases_root_path_obj))[0]
    db_path_obj = tmp_path / "state" / "pod_http.sqlite3"
    _seed_pod_state(db_path_obj, release_obj, 43210.0)
    config_path_obj = tmp_path / "dashboard_config.yaml"
    _write_config(config_path_obj, {"pod_http": str(db_path_obj)})
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
    )
    server_obj = ThreadingHTTPServer(("127.0.0.1", 0), make_dashboard_handler_class(app_obj))
    thread_obj = threading.Thread(target=server_obj.serve_forever, daemon=True)
    thread_obj.start()
    try:
        host_str, port_int = server_obj.server_address
        request_timeout_float = 6.0
        with urlopen(f"http://{host_str}:{port_int}/api/pods", timeout=request_timeout_float) as response_obj:
            payload_dict = json.loads(response_obj.read().decode("utf-8"))
        with urlopen(f"http://{host_str}:{port_int}/api/pods/pod_http", timeout=request_timeout_float) as response_obj:
            detail_payload_dict = json.loads(response_obj.read().decode("utf-8"))
        with urlopen(f"http://{host_str}:{port_int}/api/pods/pod_http/events", timeout=request_timeout_float) as response_obj:
            event_payload_dict = json.loads(response_obj.read().decode("utf-8"))
    finally:
        server_obj.shutdown()
        server_obj.server_close()
        thread_obj.join(timeout=2.0)

    row_dict = _row_by_pod_id(payload_dict, "pod_http")
    assert "alert_dict_list" in payload_dict
    assert "alert_summary_dict" in payload_dict
    assert "total_count_int" in payload_dict["alert_summary_dict"]
    assert row_dict["db_status_str"] == "ok"
    assert row_dict["equity_float"] == 43210.0
    assert "required_action_dict" in row_dict
    assert "debug_summary_dict" in row_dict
    assert "lifecycle_step_dict_list" in row_dict
    assert "data_freshness_dict" in row_dict
    assert "eod_snapshot_dict" in row_dict
    assert detail_payload_dict["pod_row_dict"]["pod_id_str"] == "pod_http"
    assert "debug_story_dict" in detail_payload_dict
    assert detail_payload_dict["debug_story_dict"]["verdict_dict"]["next_inspect_command_name_str"] in {
        "next_due",
        "status",
    }
    assert "eod_snapshot_dict" in detail_payload_dict
    assert detail_payload_dict["required_action_dict"]["label_str"] in {
        "Build DecisionPlan",
        "No action",
        "No state yet",
    }
    assert detail_payload_dict["data_freshness_dict"]["pod_state_updated_timestamp_str"] == AS_OF_TS.isoformat()
    assert detail_payload_dict["latest_decision_plan_dict"] is None
    assert event_payload_dict["pod_id_str"] == "pod_http"
    assert event_payload_dict["event_dict_list"] == []
