"""Real operator entrypoints and diagnostic-reference contracts for DV2-IND."""
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
import json
import shutil

import pandas as pd

import pytest

from alpha.live import reference_compare, runner, scheduler_utils, scheduler_service, logging_utils
from alpha.live.dv2_ind_adapter import DV2_IND_STRATEGY_IMPORT_STR
from data import norgate_snapshot_store as snapshot_module
from data.dv2_ind_snapshot_contract import DV2_IND_PROFILE_STR, DV2_IND_ETF_TUPLE
from scripts import export_norgate_snapshot as export_module
from strategies.dv2 import strategy_mr_dv2_industry_etf as strategy_module
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.state_store_v2 import LiveStateStore
from daily_broker_fakes import install_daily_broker_stub
from scripts import doctor_norgate_client as doctor_module
from scripts.live_debug import ibkr_connectivity_probe as probe_module
from test_live_dv2_ind_wiring import industry_case, daily_case, TEMPLATE_PATH
from test_dashboard_v4_activity import _event_dict, _page_dict


def test_client_doctor_smokes_actual_dv2_snapshot_and_true_return_benchmark(industry_case):
    result_list = []
    doctor_module._check_profile_smoke(result_list, local_root_path_obj=industry_case.tmp_path_obj,
        profile_list=[DV2_IND_PROFILE_STR], printer_fn=lambda *_arg_tuple: None)
    assert len(result_list) == 20
    assert all(result_obj.status_str == "PASS" for result_obj in result_list), [
        result_obj.detail_str for result_obj in result_list if result_obj.status_str != "PASS"]


def test_daily_account_probe_accepts_dv2_without_connecting():
    parsed_args_obj = probe_module.parse_args(["--daily-pod-checks", "--release-manifest-path",
        str(TEMPLATE_PATH), "--client-id", "51"])
    release_obj = probe_module._daily_pod_release_obj(parsed_args_obj)
    assert release_obj.strategy_import_str == DV2_IND_STRATEGY_IMPORT_STR
    parsed_args_obj.client_id_int = release_obj.broker_client_id_int
    with pytest.raises(ValueError, match="client ID"):
        probe_module._daily_pod_release_obj(parsed_args_obj)


@pytest.mark.parametrize("event_str", [
    "dv2_ind_decision_blocked", "dv2_ind_vplan_blocked", "dv2_ind_pre_submit_account_changed",
    "dv2_ind_eod_source_untrusted", "dv2_ind_release_disabled",
])
def test_dv2_safety_blocks_are_visible_as_activity_alerts(event_str):
    row_dict, = _page_dict([_event_dict(event_str)])["row_list"]
    assert row_dict["type_str"] == "alerts"
    assert row_dict["state_str"] == "fail"
    assert not row_dict["child_list"]
    assert row_dict["code_str"] == event_str


def test_real_auto_reference_is_explicitly_diagnostic(industry_case):
    case_obj = industry_case
    end_date_str = _extend_next_snapshot(case_obj)
    reference_strategy_obj = reference_compare.run_auto_reference_strategy(
        release_obj=case_obj.release_obj,
        deployment_start_date_str=end_date_str,
        reference_end_date_str=end_date_str, deployment_initial_cash_float=200000.0,
        output_dir_path_obj=case_obj.tmp_path_obj/"reference")
    assert len(reference_strategy_obj.get_transactions()) == 10
    reference_maps_dict = reference_compare.build_reference_maps_dict(reference_strategy_obj)
    provenance_dict = reference_maps_dict.get("reference_provenance_dict", {})
    assert provenance_dict.get("scope_str") == "diagnostic_current_snapshot_reconstruction"
    assert provenance_dict.get("exact_share_parity_bool") is False
    assert provenance_dict.get("exact_pnl_parity_bool") is False
    assert provenance_dict.get("broker_dividend_cash_parity_bool") is False
    assert provenance_dict.get("snapshot_manifest_hash_str")
    assert provenance_dict.get("data_profile_str") == DV2_IND_PROFILE_STR
    artifact_dict = reference_compare.write_reference_compare_artifacts(
        output_dir_path_obj=case_obj.tmp_path_obj/"report", release_obj=case_obj.release_obj,
        compare_report_dict={"target_session_date_str": end_date_str}, reference_maps_dict=reference_maps_dict,
        live_history_row_dict_list=[], reference_strategy_pickle_path_str=None)
    summary_dict = json.loads(Path(artifact_dict["summary_json_path_str"]).read_text())
    assert summary_dict["reference_provenance_dict"]["exact_pnl_parity_bool"] is False
    html_str = Path(artifact_dict["html_path_str"]).read_text(encoding="utf-8")
    assert "DV2-IND diagnostic reference" in html_str
    assert "Close_T" in html_str and "dividend" in html_str



def _extend_next_snapshot(case_obj):
    next_session_ts = scheduler_utils.get_next_session_label_ts(case_obj.provider_obj.date_idx[-1], "XNYS")
    for symbol_str, price_df in case_obj.provider_obj.frame_dict.items():
        extended_price_df = price_df.copy()
        extended_price_df.loc[next_session_ts] = price_df.iloc[-1]
        extended_price_df.index.name = "Date"
        case_obj.provider_obj.frame_dict[symbol_str] = extended_price_df
    export_module.export_profile_snapshot(snapshot_root_str=str(case_obj.tmp_path_obj),
        profile_str=DV2_IND_PROFILE_STR, snapshot_date_str=next_session_ts.date().isoformat())
    return next_session_ts.date().isoformat()


def _write_enabled_release(case_obj, root_path_obj, auto_submit_bool=False):
    root_path_obj.mkdir()
    template_str = TEMPLATE_PATH.read_text().replace("enabled_bool: false", "enabled_bool: true", 1).replace("DU_YOUR_PAPER_ACCOUNT", "DU_TEST")
    if auto_submit_bool:
        template_str = template_str.replace("auto_submit_enabled_bool: false", "auto_submit_enabled_bool: true")
    (root_path_obj/"dv2.yaml").write_text(template_str, encoding="utf-8")


def test_scheduler_runs_two_complete_daily_sessions(industry_case, monkeypatch):
    case_obj = industry_case
    releases_root_path_obj = case_obj.tmp_path_obj/"your_user"
    _write_enabled_release(case_obj, releases_root_path_obj, auto_submit_bool=True)
    store_obj = LiveStateStore(str(case_obj.tmp_path_obj/"scheduler.sqlite3"))
    broker_obj = StubBrokerAdapter()
    install_daily_broker_stub(broker_obj)
    broker_obj.seed_account_snapshot(case_obj.release_obj.account_route_str, 100000, 100000, {},
        case_obj.as_of_ts, session_mode_str="paper")
    monkeypatch.setattr(scheduler_service, "ensure_norgate_snapshots_for_live_tick", lambda **_arg_dict: {"status_str": "ready"})
    monkeypatch.setattr(logging_utils, "DEFAULT_CRITICAL_LOG_PATH_STR", str(case_obj.tmp_path_obj/"critical.jsonl"))
    clock_list = [case_obj.as_of_ts]
    class SimulatedDatetime(datetime):
        @classmethod
        def now(cls, timezone_obj=None):
            return clock_list[0].astimezone(timezone_obj)
    monkeypatch.setattr(runner, "datetime", SimulatedDatetime)
    phase_list = []
    def step_fn(timestamp_ts):
        clock_list[0] = timestamp_ts
        detail_dict = scheduler_service.run_once(store_obj, broker_obj, timestamp_ts,
            str(releases_root_path_obj), "paper", log_path_str=str(case_obj.tmp_path_obj/"scheduler.log"), trace_enabled_bool=False)
        phase_list.append(detail_dict["next_phase_str"])
        return detail_dict
    step_fn(case_obj.as_of_ts)
    state_obj = store_obj.get_pod_state(case_obj.release_obj.pod_id_str)
    assert state_obj.snapshot_stage_str == "eod"
    step_fn(case_obj.as_of_ts+timedelta(seconds=1))
    first_decision_obj = store_obj.get_latest_decision_plan_for_pod(case_obj.release_obj.pod_id_str)
    assert first_decision_obj is not None and len(first_decision_obj.entry_priority_list) == 10
    broker_obj.seed_account_snapshot(case_obj.release_obj.account_route_str, 100000, 100000, {},
        first_decision_obj.submission_timestamp_ts, session_mode_str="paper")
    broker_obj.seed_live_price_snapshot(case_obj.release_obj.account_route_str,
        {asset_str: 125.0 for asset_str in DV2_IND_ETF_TUPLE}, first_decision_obj.submission_timestamp_ts)
    step_fn(first_decision_obj.submission_timestamp_ts)
    vplan_obj = store_obj.get_latest_vplan_for_pod(case_obj.release_obj.pod_id_str)
    assert vplan_obj.status_str == "submitted"
    actual_snapshot_obj = broker_obj.get_account_snapshot(case_obj.release_obj.account_route_str)
    close_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(vplan_obj.target_execution_timestamp_ts.date(), "XNYS")+timedelta(minutes=20)
    broker_obj.seed_account_snapshot(case_obj.release_obj.account_route_str, actual_snapshot_obj.cash_float,
        actual_snapshot_obj.net_liq_float, actual_snapshot_obj.position_amount_map, close_timestamp_ts, session_mode_str="paper")
    step_fn(close_timestamp_ts)
    assert store_obj.get_decision_plan_by_id(first_decision_obj.decision_plan_id_int).status_str == "completed"
    _extend_next_snapshot(case_obj)
    step_fn(close_timestamp_ts+timedelta(seconds=1))
    step_fn(close_timestamp_ts+timedelta(seconds=2))
    second_decision_obj = store_obj.get_latest_decision_plan_for_pod(case_obj.release_obj.pod_id_str)
    assert second_decision_obj.decision_plan_id_int != first_decision_obj.decision_plan_id_int
    assert second_decision_obj.strategy_state_dict["dv2_ind_strategy_import_str"] == DV2_IND_STRATEGY_IMPORT_STR
    assert phase_list == ["eod_snapshot", "build_decision_plan", "build_vplan", "post_execution_reconcile", "eod_snapshot", "build_decision_plan"]


def test_producer_service_sync_and_client_exact_dv2_prices(industry_case, monkeypatch):
    from scripts.serve_norgate_snapshot_api import NorgateSnapshotApiService
    from scripts import sync_norgate_snapshots_api as sync_module
    case_obj = industry_case
    releases_root_path_obj = case_obj.tmp_path_obj/"your_user"
    _write_enabled_release(case_obj, releases_root_path_obj)
    service_obj = NorgateSnapshotApiService(service_root_path_obj=case_obj.tmp_path_obj/"server", token_str="offline-test")
    snapshot_date_str = case_obj.provider_obj.date_idx[-1].date().isoformat()
    monkeypatch.setattr(export_module, "_latest_norgate_session_date_str", lambda: snapshot_date_str)
    def requirements_fn(**option_dict):
        assert option_dict["profile_list"] == [DV2_IND_PROFILE_STR]
        return service_obj.handle_requirements_dict(option_dict["client_id_str"], {"profile_list": option_dict["profile_list"]})
    def download_fn(**option_dict):
        part_list = option_dict["url_path_str"].strip("/").split("/")
        source_path_obj = service_obj.resolve_snapshot_file_path_obj(client_id_str=part_list[2], profile_str=part_list[4],
            snapshot_date_str=part_list[5], file_name_str=part_list[6])
        shutil.copyfile(source_path_obj, option_dict["output_path_obj"])
    monkeypatch.setattr(sync_module, "post_requirements_dict", requirements_fn)
    monkeypatch.setattr(sync_module, "_download_file", download_fn)
    client_root_path_obj = case_obj.tmp_path_obj/"client"
    promoted_path_list = sync_module.sync_required_snapshots(api_url_str="http://offline-test", token_str="offline-test",
        client_id_str="test_client", releases_root_path_str=str(releases_root_path_obj),
        local_root_path_str=str(client_root_path_obj), mode_str="paper")
    assert len(promoted_path_list) == 1
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(client_root_path_obj))
    with snapshot_module.use_norgate_data_profile(DV2_IND_PROFILE_STR):
        client_price_df = strategy_module.get_prices(list(DV2_IND_ETF_TUPLE), ["$SPX"], start_date="2009-01-01", end_date=snapshot_date_str)
    pd.testing.assert_frame_equal(case_obj.direct_price_df, client_price_df, check_exact=True)


def test_reference_rejects_mid_run_snapshot_mutation(industry_case, monkeypatch):
    case_obj = industry_case
    original_run_fn = strategy_module.run_variant
    @wraps(original_run_fn)
    def mutate_run_fn(**option_dict):
        strategy_obj = original_run_fn(**option_dict)
        manifest_path_obj = case_obj.snapshot_path_obj/"manifest.json"
        manifest_dict = json.loads(manifest_path_obj.read_text())
        manifest_dict["test_revision_str"] = "changed-during-reference"
        manifest_path_obj.write_text(json.dumps(manifest_dict), encoding="utf-8")
        return strategy_obj
    monkeypatch.setattr(strategy_module, "run_variant", mutate_run_fn)
    with pytest.raises(RuntimeError, match="reference snapshot changed"):
        reference_compare.run_auto_reference_strategy(release_obj=case_obj.release_obj,
            deployment_start_date_str=case_obj.provider_obj.date_idx[-4].date().isoformat(),
            reference_end_date_str=case_obj.provider_obj.date_idx[-1].date().isoformat(),
            deployment_initial_cash_float=100000, output_dir_path_obj=case_obj.tmp_path_obj/"reference")


def test_manual_reference_initial_capital_is_not_assumed():
    provenance_dict = reference_compare.get_dv2_ind_reference_provenance_dict({})
    assert provenance_dict["provenance_available_bool"] is False
    assert provenance_dict["reference_initial_capital_basis_str"] == "unverified_reference_initial_capital"
    assert provenance_dict["exact_pnl_parity_bool"] is False



def test_runner_compare_wrapper_discloses_frozen_nav_difference(daily_case):
    case_obj = daily_case
    _extend_next_snapshot(case_obj)
    releases_root_path_obj = case_obj.tmp_path_obj/"your_user"
    _write_enabled_release(case_obj, releases_root_path_obj)
    report_dict = runner.get_compare_reference_summary(case_obj.store_obj,
        as_of_ts=case_obj.vplan_obj.target_execution_timestamp_ts+timedelta(hours=7),
        releases_root_path_str=str(releases_root_path_obj), env_mode_str="paper",
        pod_id_str=case_obj.release_obj.pod_id_str, html_output_bool=True,
        output_dir_str=str(case_obj.tmp_path_obj/"compare"))["compare_report_dict_list"][0]
    assert case_obj.vplan_obj.pod_budget_float == 200000.0
    assert case_obj.decision_obj.snapshot_metadata_dict["decision_nav_float"] < 100000.0
    assert report_dict["reference_provenance_dict"]["scope_str"] == "diagnostic_current_snapshot_reconstruction"
    assert report_dict["reference_provenance_dict"]["reference_initial_capital_basis_str"] == "first_pre_submit_broker_budget"
    assert report_dict["reference_provenance_dict"]["exact_share_parity_bool"] is False
    assert report_dict["reference_provenance_dict"]["broker_dividend_cash_parity_bool"] is False
