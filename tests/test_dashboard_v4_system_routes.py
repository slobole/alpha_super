"""System health routes keep read-only scope, refresh and export boundaries."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.overview import build_overview_dict


NOW_TS = datetime(2026, 9, 21, 15, tzinfo=UTC)
PRIVATE_STR = "private-token-C:/private/account-state.db"


def _forbidden(*args_tuple, **kwargs_dict):
    pytest.fail("System health reached a forbidden source before validation")


def _workspace_dict(*, age_int=0):
    return {"client_dict": {"accounts": [], "display_name": "Fixture"},
        "operations_account_list": [], "operations_error_str": None,
        "summary_dict": {"as_of_timestamp_str": (NOW_TS - timedelta(seconds=age_int)).isoformat(),
            "pod_row_dict_list": [], "raw_private_dict": {"token_str": PRIVATE_STR}}}


@pytest.fixture(autouse=True)
def healthy_host_disk(monkeypatch):
    monkeypatch.setattr("alpha.live.dashboard_v3.health.shutil.disk_usage",
        lambda path_str: SimpleNamespace(total=100, used=50, free=50))


@pytest.mark.parametrize("path_str", ["/system", "/system/refresh"])
@pytest.mark.parametrize("query_str", ["mode=paper", "pod=pod_live", "period=All", "x=1",
    "download=unknown", "download=", "download=status&download=status", "download=status&download=diagnostic"])
def test_invalid_query_is_rejected_before_any_source(path_str, query_str):
    provider_obj = SimpleNamespace(get_system_source_dict=_forbidden)
    app_obj = create_app(provider_obj, operations_workspace_fn=_forbidden, workspace_snapshot_fn=_forbidden)
    assert app_obj.test_client().get(path_str + "?" + query_str).status_code == 400


@pytest.mark.parametrize("download_str", ["status", "diagnostic"])
def test_refresh_cannot_be_used_as_a_download_endpoint(download_str):
    app_obj = create_app(SimpleNamespace(get_system_source_dict=_forbidden),
        operations_workspace_fn=_forbidden, workspace_snapshot_fn=_forbidden)
    assert app_obj.test_client().get("/system/refresh?download=" + download_str).status_code == 400


@pytest.mark.parametrize("path_str", ["/system", "/system/refresh", "/system?download=status"])
@pytest.mark.parametrize("method_str", ["POST", "PUT", "PATCH", "DELETE"])
def test_non_read_methods_are_blocked_before_sources(path_str, method_str):
    app_obj = create_app(SimpleNamespace(get_system_source_dict=_forbidden),
        operations_workspace_fn=_forbidden, workspace_snapshot_fn=_forbidden)
    response_obj = app_obj.test_client().open(path_str, method=method_str)
    assert response_obj.status_code == 403
    assert response_obj.get_json()["error"] == "read_only"


@pytest.mark.parametrize("path_str", ["/system", "/system/refresh", "/system?download=diagnostic"])
def test_remote_plain_http_is_blocked_before_sources(path_str):
    app_obj = create_app(SimpleNamespace(get_system_source_dict=_forbidden),
        operations_workspace_fn=_forbidden, workspace_snapshot_fn=_forbidden)
    response_obj = app_obj.test_client().get(path_str, environ_overrides={"REMOTE_ADDR": "192.0.2.3"})
    assert response_obj.status_code == 426


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.element_list = []

    def handle_starttag(self, tag_str, attribute_list):
        self.element_list.append((tag_str, dict(attribute_list)))


def _app_tuple(monkeypatch, *, age_int=0, source_fn=None):
    workspace_dict = _workspace_dict(age_int=age_int)
    call_list = []

    def operations_fn():
        call_list.append("operations")
        return deepcopy(workspace_dict)

    def system_source_fn(current_workspace_dict, *, as_of_ts):
        call_list.append("system")
        assert current_workspace_dict == workspace_dict
        assert as_of_ts >= NOW_TS
        if source_fn is not None:
            return {"scope_verified_bool": True, **source_fn(current_workspace_dict, as_of_ts=as_of_ts)}
        return {"scope_verified_bool": True, "checked_timestamp_str": as_of_ts.isoformat(),
            "raw_private_dict": {"token_str": PRIVATE_STR}}

    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_workspace_snapshot_tuple", _forbidden)
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_financial_overview_dict", _forbidden)
    provider_obj = SimpleNamespace(get_system_source_dict=system_source_fn)
    app_obj = create_app(provider_obj, operations_workspace_fn=operations_fn,
        workspace_snapshot_fn=_forbidden, now_fn=lambda: NOW_TS)
    return app_obj, call_list


@pytest.mark.parametrize("path_str,hx_bool", [("/system", False), ("/system", True), ("/system/refresh", False)])
def test_page_and_refresh_use_operations_only_and_preserve_shared_shell(monkeypatch, path_str, hx_bool):
    app_obj, call_list = _app_tuple(monkeypatch)
    response_obj = app_obj.test_client().get(path_str, headers={"HX-Request": "true"} if hx_bool else {})
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert call_list == ["operations", "system"]
    assert ("<!doctype html>" in html_str.lower()) is (path_str == "/system" and not hx_bool)
    page_obj = _PageParser()
    page_obj.feed(html_str)
    shell_dict, = [attribute_dict for _, attribute_dict in page_obj.element_list
        if attribute_dict.get("id") == "overview-shell"]
    assert shell_dict["hx-get"] == "/system/refresh"
    assert shell_dict["hx-trigger"] == "every 15s"
    assert shell_dict["hx-swap"] == "outerHTML"
    assert shell_dict["data-selection-scope"].startswith("system:")
    assert any("system-page" in attribute_dict.get("class", "").split()
        for _, attribute_dict in page_obj.element_list)
    active_link_list = [attribute_dict for tag_str, attribute_dict in page_obj.element_list
        if tag_str == "a" and attribute_dict.get("aria-current") == "page"]
    assert any(attribute_dict.get("href") == "/system" for attribute_dict in active_link_list)
    assert not any(attribute_dict.get("href") == "/" for attribute_dict in active_link_list)
    assert any(tag_str == "a" and attribute_dict.get("href") == "/system"
        and "syslight" in attribute_dict.get("class", "").split()
        for tag_str, attribute_dict in page_obj.element_list)
    assert not any(attribute_dict.get("href") in {"#", "/tools", "tools.html"}
        for _, attribute_dict in page_obj.element_list)
    for label_str in ("Runs all the time", "Runs on a schedule", "Data and space", "What should run"):
        assert label_str in html_str
    assert PRIVATE_STR not in html_str
    assert response_obj.headers["Cache-Control"] == "no-store"
    assert "script-src 'self'" in response_obj.headers["Content-Security-Policy"]
    assert response_obj.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize("download_str", ["status", "diagnostic"])
def test_json_downloads_contain_public_view_only(monkeypatch, download_str):
    app_obj, call_list = _app_tuple(monkeypatch)
    response_obj = app_obj.test_client().get("/system?download=" + download_str)
    assert response_obj.status_code == 200
    assert response_obj.is_json
    assert isinstance(response_obj.get_json(), dict)
    assert "attachment" in response_obj.headers["Content-Disposition"]
    assert ".json" in response_obj.headers["Content-Disposition"]
    text_str = response_obj.get_data(as_text=True)
    assert PRIVATE_STR not in text_str and "raw_private_dict" not in text_str
    assert call_list == ["operations", "system"]
    assert response_obj.headers["Cache-Control"] == "no-store"
    export_dict = response_obj.get_json()
    assert export_dict["schema_str"] == "dashboard_v4.system." + download_str + ".v1"
    assert export_dict["mode_str"] == "live"
    assert export_dict["as_of_timestamp_str"] == NOW_TS.isoformat()
    common_key_set = {"schema_str", "as_of_timestamp_str", "mode_str", "demo_bool", "state_str", "verdict_str"}
    if download_str == "status":
        assert set(export_dict) == common_key_set | {"check_list"}
        assert export_dict["check_list"]
        assert all(set(row_dict) == {"key_str", "label_str", "state_str", "now_str"}
            for row_dict in export_dict["check_list"])
    else:
        assert set(export_dict) == common_key_set | {"assessment_dict", "source_fresh_bool", "source_valid_ms_int"}
        assert {"group_list", "pod_list", "release_list"} <= export_dict["assessment_dict"].keys()


@pytest.mark.parametrize("age_int", [121, -1])
@pytest.mark.parametrize("path_str", ["/system", "/system/refresh"])
def test_stale_and_future_operations_cannot_render_fresh_header(monkeypatch, age_int, path_str):
    app_obj, _ = _app_tuple(monkeypatch, age_int=age_int)
    response_obj = app_obj.test_client().get(path_str)
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert 'data-source-valid-ms="0"' in html_str
    assert "System unknown" in html_str
    assert PRIVATE_STR not in html_str


def test_source_acquisition_time_cannot_renew_saved_operations(monkeypatch):
    clock_dict = {"now_ts": NOW_TS}

    def slow_source_fn(workspace_dict, *, as_of_ts):
        clock_dict["now_ts"] += timedelta(seconds=121)
        return {"scope_verified_bool": True, "checked_timestamp_str": clock_dict["now_ts"].isoformat()}

    app_obj = create_app(SimpleNamespace(get_system_source_dict=slow_source_fn),
        operations_workspace_fn=_workspace_dict, workspace_snapshot_fn=_forbidden,
        now_fn=lambda: clock_dict["now_ts"])
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_financial_overview_dict", _forbidden)
    response_obj = app_obj.test_client().get("/system?download=diagnostic")
    assert response_obj.status_code == 200
    export_dict = response_obj.get_json()
    assert export_dict["source_fresh_bool"] is False
    assert export_dict["source_valid_ms_int"] == 0
    assert export_dict["as_of_timestamp_str"] == clock_dict["now_ts"].isoformat()


def test_provider_without_override_uses_scoped_read_only_collector(monkeypatch):
    workspace_dict = _workspace_dict()
    provider_obj = object()
    call_list = []

    def source_fn(current_provider_obj, current_workspace_dict, *, as_of_ts, performance_db_path_str):
        assert current_provider_obj is provider_obj
        assert current_workspace_dict == workspace_dict
        assert as_of_ts == NOW_TS
        assert performance_db_path_str == "unopened_fixture.db"
        call_list.append("collector")
        return {"scope_verified_bool": True, "checked_timestamp_str": NOW_TS.isoformat()}

    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_system_source_dict", source_fn)
    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_workspace_snapshot_tuple", _forbidden)
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_financial_overview_dict", _forbidden)
    app_obj = create_app(provider_obj, operations_workspace_fn=lambda: workspace_dict,
        workspace_snapshot_fn=_forbidden, performance_db_path_str="unopened_fixture.db", now_fn=lambda: NOW_TS)
    assert app_obj.test_client().get("/system").status_code == 200
    assert call_list == ["collector"]


def test_long_release_names_are_escaped_and_identifiers_remain_selectable(monkeypatch):
    name_str = "Long portfolio strategy " * 8 + '<script>alert("name")</script> & Holdings'
    release_id_str = "release_" + "long_identity_" * 8

    def release_source_fn(workspace_dict, *, as_of_ts):
        return {"checked_timestamp_str": as_of_ts.isoformat(), "release_list": [{
            "pod_id_str": "pod_live", "name_str": name_str, "mode_str": "live", "enabled_bool": False,
            "release_id_str": release_id_str, "execution_policy_str": "next_month_first_open",
            "account_str": "U···771", "account_route_str": "U_PRIVATE_ACCOUNT",
            "private_trace_str": PRIVATE_STR}]}

    app_obj, _ = _app_tuple(monkeypatch, source_fn=release_source_fn)
    response_obj = app_obj.test_client().get("/system")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert '<script>alert("name")</script>' not in html_str
    assert "&lt;script&gt;" in html_str and "&amp; Holdings" in html_str
    assert release_id_str in html_str and "U···771" in html_str
    assert "U_PRIVATE_ACCOUNT" not in html_str and PRIVATE_STR not in html_str
    page_obj = _PageParser()
    page_obj.feed(html_str)
    key_list = [attribute_dict["data-selection-key"] for _, attribute_dict in page_obj.element_list
        if "data-selection-key" in attribute_dict]
    assert f"system:release:pod_live:{release_id_str}:id" in key_list
    assert len(key_list) == len(set(key_list))


def test_same_pod_release_versions_have_distinct_selection_keys(monkeypatch):
    def version_source_fn(workspace_dict, *, as_of_ts):
        return {"checked_timestamp_str": as_of_ts.isoformat(), "release_list": [{
            "pod_id_str": "pod_live", "name_str": "One strategy", "mode_str": "live", "enabled_bool": enabled_bool,
            "release_id_str": release_str, "execution_policy_str": "next_month_first_open", "account_str": "U···771"}
            for release_str, enabled_bool in (("release_current", True), ("release_prior", False))]}

    app_obj, _ = _app_tuple(monkeypatch, source_fn=version_source_fn)
    response_obj = app_obj.test_client().get("/system")
    assert response_obj.status_code == 200
    page_obj = _PageParser()
    page_obj.feed(response_obj.get_data(as_text=True))
    key_list = [attribute_dict["data-selection-key"] for _, attribute_dict in page_obj.element_list
        if "data-selection-key" in attribute_dict]
    assert len(key_list) == len(set(key_list))
    for release_str in ("release_current", "release_prior"):
        assert f"system:release:pod_live:{release_str}:id" in key_list
        assert f"system:release:pod_live:{release_str}:name" in key_list


def _healthy_system_tuple():
    workspace_dict = _workspace_dict()
    workspace_dict["operations_account_list"] = [{"pod_id": "pod_live", "account_route": "U123456",
        "display_name": "One strategy"}]
    workspace_dict["summary_dict"]["pod_row_dict_list"] = [{"pod_id_str": "pod_live", "account_route_str": "U123456",
        "mode_str": "live", "release_id_str": "release_live", "db_status_str": "ok", "as_of_timestamp_str": NOW_TS.isoformat(),
        "data_freshness_dict": {"item_dict_list": [{"label_str": label_str, "severity_str": "green",
            "value_str": "2026-09-18"} for label_str in ("Norgate", "Pod state", "EOD Snapshot")]}}]

    def scheduler_fn(pod_id_str, *, as_of_ts):
        assert pod_id_str == "pod_live"
        return {"state_str": "sleeping", "alive_bool": True, "checked_timestamp_str": as_of_ts.isoformat(),
            "last_seen_timestamp_str": NOW_TS.isoformat(),
            "promised_wake_timestamp_str": (NOW_TS + timedelta(minutes=30)).isoformat()}

    source_dict = {"scope_verified_bool": True, "checked_timestamp_str": NOW_TS.isoformat(), "release_list": [{
        "pod_id_str": "pod_live", "name_str": "One strategy", "mode_str": "live", "enabled_bool": True,
        "release_id_str": "release_live", "execution_policy_str": "next_month_first_open", "account_str": "U···456"}]}

    def source_fn(current_workspace_dict, *, as_of_ts):
        return deepcopy(source_dict)

    provider_obj = SimpleNamespace(get_system_source_dict=source_fn, get_scheduler_status_dict=scheduler_fn)
    return workspace_dict, provider_obj, source_dict


def test_auxiliary_failure_updates_header_page_and_export_consistently(monkeypatch):
    workspace_dict, provider_obj, source_dict = _healthy_system_tuple()
    source_dict["watchdog_dict"] = {"state_str": "error", "now_str": "Saved watchdog failure",
        "last_timestamp_str": NOW_TS.isoformat(), "expected_str": "Saved evidence"}
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_financial_overview_dict", _forbidden)
    assert build_overview_dict(workspace_dict, None, provider_obj, as_of_ts=NOW_TS,
        include_finance_bool=False)["system_dict"]["state_str"] == "done"
    app_obj = create_app(provider_obj, operations_workspace_fn=lambda: deepcopy(workspace_dict),
        workspace_snapshot_fn=_forbidden, now_fn=lambda: NOW_TS)
    client_obj = app_obj.test_client()
    for path_str in ("/system", "/system/refresh"):
        response_obj = client_obj.get(path_str)
        html_str = response_obj.get_data(as_text=True)
        assert response_obj.status_code == 200
        assert "data-status-label>System needs action</" in html_str
        assert "data-verdict>System needs action.</" in html_str
        assert "System OK" not in html_str
        assert "Saved watchdog failure" in html_str
    export_dict = client_obj.get("/system?download=status").get_json()
    assert export_dict["state_str"] == "fail"
    assert export_dict["verdict_str"] == "System needs action."


@pytest.mark.parametrize("delay_int,prior_failure_bool", [(31, False), (120, False), (121, False), (121, True)])
def test_overview_acquisition_consumes_remaining_source_lifetime(monkeypatch, delay_int, prior_failure_bool):
    workspace_dict, provider_obj, _ = _healthy_system_tuple()
    clock_dict = {"now_ts": NOW_TS}
    call_list = []
    original_scheduler_fn = provider_obj.get_scheduler_status_dict
    original_source_fn = provider_obj.get_system_source_dict

    def source_fn(current_workspace_dict, *, as_of_ts):
        call_list.append("source")
        return original_source_fn(current_workspace_dict, as_of_ts=as_of_ts)

    def slow_scheduler_fn(pod_id_str, *, as_of_ts):
        call_list.append("scheduler")
        scheduler_dict = original_scheduler_fn(pod_id_str, as_of_ts=as_of_ts)
        if prior_failure_bool:
            scheduler_dict["state_str"] = "error"
        clock_dict["now_ts"] += timedelta(seconds=delay_int)
        return scheduler_dict

    provider_obj.get_system_source_dict = source_fn
    provider_obj.get_scheduler_status_dict = slow_scheduler_fn
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_financial_overview_dict", _forbidden)
    app_obj = create_app(provider_obj, operations_workspace_fn=lambda: deepcopy(workspace_dict),
        workspace_snapshot_fn=_forbidden, now_fn=lambda: clock_dict["now_ts"])
    response_obj = app_obj.test_client().get("/system?download=diagnostic")
    assert response_obj.status_code == 200
    export_dict = response_obj.get_json()
    fresh_bool = delay_int < 120
    assert call_list == ["source", "scheduler"]
    assert export_dict["as_of_timestamp_str"] == (NOW_TS + timedelta(seconds=delay_int)).isoformat()
    assert export_dict["source_valid_ms_int"] == max(0, 120 - delay_int) * 1000
    assert export_dict["source_fresh_bool"] is fresh_bool
    pod_dict, = export_dict["assessment_dict"]["pod_list"]
    assert pod_dict["scheduler_state_str"] == ("done" if fresh_bool else "unk")
    if not fresh_bool:
        assert export_dict["state_str"] == "unk"
        assert export_dict["verdict_str"] == "Some system checks are unverified."


@pytest.mark.parametrize("checked_str,remaining_int", [
    ((NOW_TS - timedelta(seconds=31)).isoformat(), 89_000),
    ((NOW_TS - timedelta(seconds=120)).isoformat(), 0),
    ((NOW_TS - timedelta(seconds=121)).isoformat(), 0),
    ((NOW_TS + timedelta(seconds=1)).isoformat(), 0),
    (NOW_TS.replace(tzinfo=None).isoformat(), 0),
    ("invalid timestamp", 0),
])
def test_auxiliary_checked_time_caps_page_lifetime(monkeypatch, checked_str, remaining_int):
    workspace_dict, provider_obj, source_dict = _healthy_system_tuple()
    source_dict["checked_timestamp_str"] = checked_str
    source_dict["watchdog_dict"] = {"state_str": "error", "now_str": "Saved watchdog failure",
        "last_timestamp_str": NOW_TS.isoformat(), "expected_str": "Saved evidence"}
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_financial_overview_dict", _forbidden)
    app_obj = create_app(provider_obj, operations_workspace_fn=lambda: deepcopy(workspace_dict),
        workspace_snapshot_fn=_forbidden, now_fn=lambda: NOW_TS)
    response_obj = app_obj.test_client().get("/system?download=diagnostic")
    assert response_obj.status_code == 200
    export_dict = response_obj.get_json()
    assert export_dict["source_valid_ms_int"] == remaining_int
    assert export_dict["source_fresh_bool"] is bool(remaining_int)
    pod_dict, = export_dict["assessment_dict"]["pod_list"]
    assert pod_dict["scheduler_state_str"] == ("done" if remaining_int else "unk")
    watchdog_dict, = [row_dict for group_dict in export_dict["assessment_dict"]["group_list"]
        for row_dict in group_dict["row_list"] if row_dict["key_str"] == "watchdog"]
    assert watchdog_dict["state_str"] == ("fail" if remaining_int else "unk")
    assert export_dict["state_str"] == ("fail" if remaining_int else "unk")


@pytest.mark.parametrize("scope_case_str", ["rejected", "missing", "changed_release"])
def test_unverified_or_changed_source_scope_cannot_green_pod_operations(monkeypatch, scope_case_str):
    workspace_dict, provider_obj, source_dict = _healthy_system_tuple()
    if scope_case_str == "rejected":
        source_dict["scope_verified_bool"] = False
    elif scope_case_str == "missing":
        source_dict.pop("scope_verified_bool")
    else:
        source_dict["release_list"][0]["release_id_str"] = "another_release"
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_financial_overview_dict", _forbidden)
    assert build_overview_dict(workspace_dict, None, provider_obj, as_of_ts=NOW_TS,
        include_finance_bool=False)["system_dict"]["state_str"] == "done"
    app_obj = create_app(provider_obj, operations_workspace_fn=lambda: deepcopy(workspace_dict),
        workspace_snapshot_fn=_forbidden, now_fn=lambda: NOW_TS)
    response_obj = app_obj.test_client().get("/system?download=diagnostic")
    assert response_obj.status_code == 200
    assessment_dict = response_obj.get_json()["assessment_dict"]
    pod_dict, = assessment_dict["pod_list"]
    assert pod_dict["scheduler_state_str"] == "unk"
    assert pod_dict["scheduler_str"] == "Unknown"
    assert pod_dict["last_str"] == pod_dict["wake_str"] == "—"
    assert pod_dict["broker_str"] == pod_dict["data_str"] == pod_dict["eod_str"] == "—"
    scheduler_dict, = [row_dict for group_dict in assessment_dict["group_list"]
        for row_dict in group_dict["row_list"] if row_dict["key_str"] == "schedulers"]
    assert scheduler_dict["state_str"] == "unk"
    assert scheduler_dict["now_str"] == "0 of 1 alive"


def test_demo_source_is_independent_and_never_falls_back_to_local_services(monkeypatch):
    from alpha.live.dashboard_v4.system_demo import attach_demo_system

    workspace_dict = _workspace_dict()
    original_dict = deepcopy(workspace_dict)
    provider_obj = SimpleNamespace(get_target_list=lambda: [], row_list=[])
    attach_demo_system(provider_obj)
    first_dict = provider_obj.get_system_source_dict(workspace_dict, as_of_ts=NOW_TS)
    first_dict["watchdog_dict"]["now_str"] = "Mutated previous result"
    second_dict = provider_obj.get_system_source_dict(workspace_dict, as_of_ts=NOW_TS)
    assert second_dict["watchdog_dict"]["now_str"] != "Mutated previous result"
    assert workspace_dict == original_dict
    for method_str in ("load_system_source_dict", "load_operations_workspace_dict", "load_workspace_snapshot_tuple"):
        monkeypatch.setattr("alpha.live.dashboard_v4.app." + method_str, _forbidden)
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_financial_overview_dict", _forbidden)
    app_obj = create_app(provider_obj, demo_bool=True, operations_workspace_fn=lambda: deepcopy(workspace_dict),
        workspace_snapshot_fn=_forbidden, now_fn=lambda: NOW_TS)
    client_obj = app_obj.test_client()
    for path_str in ("/system", "/system/refresh", "/system?download=status", "/system?download=diagnostic"):
        response_obj = client_obj.get(path_str)
        assert response_obj.status_code == 200
        assert PRIVATE_STR not in response_obj.get_data(as_text=True)
    assert client_obj.get("/system?download=diagnostic").get_json()["demo_bool"] is True
