from __future__ import annotations

from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
import time
from urllib.request import urlopen

from alpha.live.dashboard import (
    DASHBOARD_HTML_STR,
    DashboardApp,
    DashboardConfig,
    DashboardPodTarget,
    DiffJobManager,
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
    VPlan,
    VPlanRow,
)
from alpha.live.release_manifest import load_release_list
from alpha.live.runner import DEFAULT_INCUBATION_DB_PATH_STR
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


def _row_by_pod_id(summary_dict: dict, pod_id_str: str) -> dict:
    row_list = [
        row_dict
        for row_dict in summary_dict["pod_row_dict_list"]
        if row_dict["pod_id_str"] == pod_id_str
    ]
    assert len(row_list) == 1
    return row_list[0]


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
    assert target_map_dict["pod_enabled_incubation"].db_path_str == DEFAULT_INCUBATION_DB_PATH_STR
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


def test_dashboard_summary_handles_missing_empty_shared_and_pod_db(tmp_path: Path):
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
    assert _row_by_pod_id(summary_dict, "pod_empty")["db_status_str"] == "empty"
    assert _row_by_pod_id(summary_dict, "pod_empty")["health_str"] == "gray"
    assert _row_by_pod_id(summary_dict, "pod_shared")["db_status_str"] == "ok"
    assert _row_by_pod_id(summary_dict, "pod_shared")["equity_float"] == 12345.0
    assert _row_by_pod_id(summary_dict, "pod_shared")["position_count_int"] == 1
    assert _row_by_pod_id(summary_dict, "pod_specific")["db_status_str"] == "ok"
    assert _row_by_pod_id(summary_dict, "pod_specific")["equity_float"] == 67890.0


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
    app_obj = DashboardApp(
        releases_root_path_str=str(releases_root_path_obj),
        config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"),
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
    assert detail_dict["latest_decision_plan_dict"] is None
    assert detail_dict["latest_vplan_dict"] is None
    assert detail_dict["latest_execution_report_dict"] is None


def test_dashboard_html_uses_structured_operator_sections():
    assert "detail-workspace" in DASHBOARD_HTML_STR
    assert "renderDecisionSection" in DASHBOARD_HTML_STR
    assert "Copy show_decision_plan command" in DASHBOARD_HTML_STR
    assert "<h3>Overview</h3>" in DASHBOARD_HTML_STR
    assert "<h3>Broker</h3>" in DASHBOARD_HTML_STR
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
        with urlopen(f"http://{host_str}:{port_int}/api/pods", timeout=2.0) as response_obj:
            payload_dict = json.loads(response_obj.read().decode("utf-8"))
        with urlopen(f"http://{host_str}:{port_int}/api/pods/pod_http", timeout=2.0) as response_obj:
            detail_payload_dict = json.loads(response_obj.read().decode("utf-8"))
    finally:
        server_obj.shutdown()
        server_obj.server_close()
        thread_obj.join(timeout=2.0)

    row_dict = _row_by_pod_id(payload_dict, "pod_http")
    assert row_dict["db_status_str"] == "ok"
    assert row_dict["equity_float"] == 43210.0
    assert detail_payload_dict["pod_row_dict"]["pod_id_str"] == "pod_http"
    assert detail_payload_dict["latest_decision_plan_dict"] is None
