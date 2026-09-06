import hashlib
import json

import pytest

from alpha.live.client_reporting import ClientReportingError, validate_client_registry_dict
from alpha.live.dashboard_v3.app import create_app
from alpha.live.dashboard_v3.client_comparison import saved_comparison_dict
from flask import render_template
from test_client_reporting import client_config_dict, nav_attributes_dict, snapshot_obj
from test_dashboard_operator_access import TEST_ACCESS_STR, ForbiddenProvider, auth_headers_dict


def summary_dict(**override_dict):
    return dict({"pod_id_str": "p1", "account_route_str": "U1", "mode_str": "live", "release_id_str": "historic-release",
        "deployment_start_date_str": "2026-09-01", "target_session_date_str": "2026-09-01",
        "deployment_initial_cash_float": 1000.0, "actual_equity_float": 1010.0, "backtest_equity_float": 1005.0,
        "actual_equity_source_str": "pod_state_history.eod", "actual_equity_basis_str": "eod_broker_netliq",
        "actual_equity_timestamp_str": "2026-09-01T20:10:00Z", "reference_accounting_contract_version_str": "price_return_ledger_v1",
        "reference_dividend_cash_ledger_mode_str": "disabled", "reference_strategy_pickle_path_str": "NEVER_EXPOSE.pkl",
        "equity_tracking_error_float": .005, "total_price_diff_notional_float": 0}, **override_dict)


def pinned_account_dict(tmp_path, saved_dict):
    path_obj = tmp_path / "summary.json"
    path_obj.write_text(json.dumps(saved_dict), encoding="utf-8")
    return {"pod_id": "p1", "account_route": "U1", "display_name": "Strategy A", "effective_from": "2026-09-01",
        "effective_to": "2026-09-01", "reference_summary_path": str(path_obj)}


def test_exact_identity_dates_expose_recorded_values_not_replay_or_return(tmp_path):
    account_dict = pinned_account_dict(tmp_path, summary_dict())
    result_dict = saved_comparison_dict(account_dict, from_date_str="2026-09-01", to_date_str="2026-09-04")
    assert result_dict["interval_str"] == "Exact effective account dates"
    assert result_dict["to_date_str"] == "2026-09-01"  # Retirement, not today's route.
    assert len(result_dict["recorded_value_list"]) == 3
    assert result_dict["status_str"] == "unverified"
    assert result_dict["replay_str"].startswith("Not proven")
    assert "NEVER_EXPOSE" not in json.dumps(result_dict)
    assert "equity_tracking_error_float" not in json.dumps(result_dict)
    assert "total_price_diff_notional_float" not in json.dumps(result_dict)
    assert result_dict["metadata_dict"]["release_id_str"] == "historic-release"
    assert result_dict["source_hash_str"] == hashlib.sha256((tmp_path / "summary.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("override_dict", [{"pod_id_str": "PRIVATE_OTHER"}, {"account_route_str": "PRIVATE_OTHER"},
    {"mode_str": "paper"}, {"account_id_str": "PRIVATE_OTHER"}, {"env_mode_str": "paper"}, {"pod_str": "PRIVATE_OTHER"}])
def test_wrong_identity_withholds_artifact_facts(tmp_path, override_dict):
    result_dict = saved_comparison_dict(pinned_account_dict(tmp_path, summary_dict(**override_dict)), from_date_str="2026-09-01", to_date_str="2026-09-01")
    assert result_dict["status_str"] == "unavailable"
    assert result_dict["recorded_value_list"] == [] and result_dict["metadata_dict"] == {}
    assert "PRIVATE_OTHER" not in json.dumps(result_dict)


def test_legacy_missing_account_is_metadata_only_not_current_route_inference(tmp_path):
    result_dict = saved_comparison_dict(pinned_account_dict(tmp_path, summary_dict(account_route_str=None)), from_date_str="2026-09-01", to_date_str="2026-09-01")
    assert result_dict["recorded_value_list"] == []
    assert result_dict["attribution_str"] == "Historical account identity missing"
    assert result_dict["artifact_from_date_str"] == "2026-09-01"
    assert "today" in " ".join(result_dict["issue_list"]).lower()


@pytest.mark.parametrize("override_dict", [{"deployment_start_date_str": "2026-08-01"}, {"target_session_date_str": "2026-09-02"}])
def test_overlapping_or_out_of_ownership_interval_never_clips_aggregates(tmp_path, override_dict):
    result_dict = saved_comparison_dict(pinned_account_dict(tmp_path, summary_dict(**override_dict)), from_date_str="2026-09-01", to_date_str="2026-09-01")
    assert result_dict["recorded_value_list"] == []
    assert result_dict["interval_str"].startswith("Different dates")


@pytest.mark.parametrize("bad_value_obj", [float("nan"), float("inf"), True, -5, "1005", 10 ** 500, None])
def test_invalid_values_are_unknown_not_zero(tmp_path, bad_value_obj):
    result_dict = saved_comparison_dict(pinned_account_dict(tmp_path, summary_dict(backtest_equity_float=bad_value_obj)), from_date_str="2026-09-01", to_date_str="2026-09-01")
    assert len(result_dict["recorded_value_list"]) == 2
    assert all(row_dict["value_float"] in (1000, 1010) for row_dict in result_dict["recorded_value_list"])


@pytest.mark.parametrize("saved_obj", [[], {"mode_str": "live"}, "PRIVATE", summary_dict(target_session_date_str="20260901")])
def test_malformed_summaries_do_not_raise_or_substitute(tmp_path, saved_obj):
    result_dict = saved_comparison_dict(pinned_account_dict(tmp_path, saved_obj), from_date_str="2026-09-01", to_date_str="2026-09-01")
    assert result_dict["status_str"] == "unavailable"
    assert result_dict["recorded_value_list"] == []


def test_missing_oversized_and_revised_source(tmp_path):
    account_dict = pinned_account_dict(tmp_path, summary_dict())
    first_dict = saved_comparison_dict(account_dict, from_date_str="2026-09-01", to_date_str="2026-09-01")
    (tmp_path / "summary.json").write_text(json.dumps(summary_dict(actual_equity_float=1011)), encoding="utf-8")
    second_dict = saved_comparison_dict(account_dict, from_date_str="2026-09-01", to_date_str="2026-09-01")
    assert first_dict["source_hash_str"] != second_dict["source_hash_str"]
    (tmp_path / "summary.json").write_bytes(b" " * 1_000_001)
    assert saved_comparison_dict(account_dict, from_date_str="2026-09-01", to_date_str="2026-09-01")["status_str"] == "unavailable"
    account_dict["reference_summary_path"] = str(tmp_path / "missing.json")
    assert saved_comparison_dict(account_dict, from_date_str="2026-09-01", to_date_str="2026-09-01")["status_str"] == "unavailable"


def test_pinned_path_is_server_config_only_and_not_in_investor_export(tmp_path):
    config_dict = client_config_dict()
    account_dict = pinned_account_dict(tmp_path, summary_dict(pod_id_str=config_dict["accounts"][0]["pod_id"], account_route_str=config_dict["accounts"][0]["account_route"]))
    config_dict["accounts"][0]["reference_summary_path"] = account_dict["reference_summary_path"]
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True, operator_access_token_str=TEST_ACCESS_STR,
        client_registry_dict={"schema_version": 1, "clients": [config_dict]}, client_reporting_snapshot_fn=lambda client_id_str: snapshot_obj([nav_attributes_dict()]))
    client_obj = app_obj.test_client()
    path_str = "/clients/sample/performance?from=2026-09-01&to=2026-09-01"
    html_str = client_obj.get(path_str, headers=auth_headers_dict()).get_data(as_text=True)
    assert "LIVE versus saved simulation" in html_str and "historic-release" in html_str
    assert "Not proven" in html_str and "NEVER_EXPOSE" not in html_str
    assert str(tmp_path) not in html_str
    assert client_obj.get(path_str + "&reference_summary_path=C:/secret.json", headers=auth_headers_dict()).status_code == 400
    report_html_str = client_obj.get("/clients/sample/report?from=2026-09-01&to=2026-09-01", headers=auth_headers_dict()).get_data(as_text=True)
    assert "LIVE versus saved simulation" not in report_html_str
    assert "historic-release" not in report_html_str


@pytest.mark.parametrize("path_obj", [None, {}, "", 5])
def test_registry_rejects_invalid_reference_pin(path_obj):
    config_dict = client_config_dict()
    config_dict["accounts"][0]["reference_summary_path"] = path_obj
    with pytest.raises(ClientReportingError, match="reference_summary_path"):
        validate_client_registry_dict({"schema_version": 1, "clients": [config_dict]})


@pytest.mark.parametrize("path_str", ["C:/PRIVATE_METADATA/value.csv", "C:\\PRIVATE_METADATA\\value.csv", "/private/value.csv", "C:PRIVATE_METADATA"])
def test_path_shaped_metadata_is_withheld(tmp_path, path_str):
    result_dict = saved_comparison_dict(pinned_account_dict(tmp_path, summary_dict(actual_equity_source_str=path_str)), from_date_str="2026-09-01", to_date_str="2026-09-01")
    assert result_dict["metadata_dict"]["actual_equity_source_str"] == "Path-like metadata withheld"
    assert path_str not in json.dumps(result_dict)


@pytest.mark.parametrize("timestamp_str", ["2026-09-01T20:10:00", "2027-09-01T20:10:00Z"])
def test_render_preserves_full_original_mark_time_without_guessing_zone(tmp_path, timestamp_str):
    result_dict = saved_comparison_dict(pinned_account_dict(tmp_path, summary_dict(actual_equity_timestamp_str=timestamp_str)), from_date_str="2026-09-01", to_date_str="2026-09-01")
    assert result_dict["recorded_value_list"] == []
    app_obj = create_app(ForbiddenProvider())
    with app_obj.test_request_context():
        html_str = render_template("_client_comparison.html", comparison_result_dict=result_dict)
    assert timestamp_str in html_str
    assert "09-01 16:10:00 ET" not in html_str
    assert "a missing offset remains ambiguous" in html_str


@pytest.mark.parametrize("override_dict", [{"actual_equity_timestamp_str": None}, {"actual_equity_timestamp_str": "2026-09-02T20:10:00Z"},
    {"actual_equity_source_str": "pod_state.latest", "actual_equity_basis_str": "latest_broker_netliq"},
    {"actual_equity_timestamp_str": "2026-09-01T00:00:00Z"}])
def test_wrong_or_unproven_actual_mark_withholds_recorded_values(tmp_path, override_dict):
    result_dict = saved_comparison_dict(pinned_account_dict(tmp_path, summary_dict(**override_dict)), from_date_str="2026-09-01", to_date_str="2026-09-01")
    assert result_dict["recorded_value_list"] == []
    assert "missing or mismatched" in result_dict["valuation_str"]


def test_reused_route_and_weekend_start_select_distinct_historical_pins(tmp_path):
    config_dict = client_config_dict()
    config_dict["mandate_start_date"] = "2026-08-01"
    original_dict = config_dict["accounts"][0]
    old_path_obj, new_path_obj = tmp_path / "old.json", tmp_path / "new.json"
    old_path_obj.write_text(json.dumps(summary_dict(pod_id_str=original_dict["pod_id"], account_route_str=original_dict["account_route"], release_id_str="old-period",
        deployment_start_date_str="2026-08-29", target_session_date_str="2026-08-31")), encoding="utf-8")
    new_path_obj.write_text(json.dumps(summary_dict(pod_id_str=original_dict["pod_id"], account_route_str=original_dict["account_route"], release_id_str="new-period")), encoding="utf-8")
    config_dict["accounts"] = [dict(original_dict, effective_from="2026-08-01", effective_to="2026-08-31", reference_summary_path=str(old_path_obj)),
        dict(original_dict, effective_from="2026-09-01", reference_summary_path=str(new_path_obj))]
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True, operator_access_token_str=TEST_ACCESS_STR,
        client_registry_dict={"schema_version": 1, "clients": [config_dict]}, client_reporting_snapshot_fn=lambda client_id_str: snapshot_obj([nav_attributes_dict()]))
    response_obj = app_obj.test_client().get("/clients/sample/performance?from=2026-08-29&to=2026-09-01", headers=auth_headers_dict())
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert "old-period" in html_str and "new-period" in html_str
    assert "Selected account interval 2026-08-29" in html_str
    assert str(tmp_path) not in html_str
