from datetime import UTC, datetime, timedelta

import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import create_demo_app


NOW_TS = datetime(2026, 9, 8, 15, tzinfo=UTC)


def _forbidden(*args_tuple, **kwargs_dict):
    pytest.fail("Activity reached a financial or production-demo source")


@pytest.mark.parametrize("path_str", ["/activity", "/activity/refresh"])
@pytest.mark.parametrize("query_str", ["days=1", "days=0", "days=91", "days=7&days=14", "days=all", "x=1", "mode=paper"])
def test_bad_queries_rejected_before_acquisition(path_str, query_str):
    app_obj = create_app(object(), operations_workspace_fn=_forbidden, workspace_snapshot_fn=_forbidden)
    assert app_obj.test_client().get(path_str + "?" + query_str).status_code == 400


@pytest.mark.parametrize("method_str", ["POST", "PUT", "DELETE", "PATCH"])
def test_no_write_route(method_str):
    app_obj = create_app(object(), operations_workspace_fn=_forbidden, workspace_snapshot_fn=_forbidden)
    assert app_obj.test_client().open("/activity", method=method_str).status_code == 403


def test_demo_activity_is_scoped_read_only_and_finance_free(monkeypatch):
    app_obj = create_demo_app()
    for method_str in ("load_workspace_snapshot_tuple", "load_operations_workspace_dict", "load_activity_source_dict"):
        monkeypatch.setattr("alpha.live.dashboard_v4.app." + method_str, _forbidden)
    client_obj = app_obj.test_client()
    for path_str in ("/activity", "/activity/refresh?days=14", "/activity?days=90", "/activity/body?days=14"):
        response_obj = client_obj.get(path_str)
        html_str = response_obj.get_data(as_text=True)
        assert response_obj.status_code == 200
        assert "Broker acknowledgement missing." in html_str
        assert "Scheduler error. Retry scheduled." in html_str
        assert "Trade sheet export requested." in html_str
        assert "Manual order requested." in html_str
        assert "Alert delivered." not in html_str
        assert "Open cycle completed." in html_str
        # The link names the next window, not the current one.
        next_str = {"/activity": "Load 14 days", "/activity/refresh?days=14": "Load 30 days",
            "/activity/body?days=14": "Load 30 days"}.get(path_str, "")
        assert (next_str in html_str) if next_str else "Load " not in html_str
        assert "action_token" not in html_str and "DEMO-owner" not in html_str
        assert "script-src 'self'" in response_obj.headers["Content-Security-Policy"]
    response_obj = client_obj.get("/activity")
    assert b"/static/activity.js" in response_obj.data
    assert b'aria-current="page"' in response_obj.data


def test_slow_read_does_not_renew_live_shell(monkeypatch):
    clock_dict = {"now_ts": NOW_TS}
    workspace_dict = {"client_dict": {"accounts": []}, "operations_account_list": [],
        "summary_dict": {"as_of_timestamp_str": NOW_TS.isoformat(), "pod_row_dict_list": []}, "operations_error_str": None}

    def slow_source(*args_tuple, **kwargs_dict):
        clock_dict["now_ts"] += timedelta(seconds=121)
        return {"event_list": [], "warning_list": [], "scope_key_str": "empty"}

    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_activity_source_dict", slow_source)
    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_workspace_snapshot_tuple", _forbidden)
    app_obj = create_app(object(), operations_workspace_fn=lambda: workspace_dict, now_fn=lambda: clock_dict["now_ts"])
    html_str = app_obj.test_client().get("/activity").get_data(as_text=True)
    assert 'data-source-valid-ms="0"' in html_str
    assert "System unknown" in html_str


def test_activity_shell_polls_status_only_and_the_timeline_refreshes_itself():
    client_obj = create_demo_app().test_client()
    html_str = client_obj.get("/activity").get_data(as_text=True)
    assert 'id="overview-shell" hx-get="/activity/status" hx-trigger="v4poll"' in html_str
    assert 'hx-swap="none"' in html_str and 'id="performance-status"' in html_str
    assert ('id="activity-body" data-activity-page data-own-refresh-ms="60000" hx-get="/activity/body?days=7"'
        ' hx-trigger="v4own" hx-target="this" hx-swap="outerHTML"') in html_str
    status_str = client_obj.get("/activity/status").get_data(as_text=True)
    assert status_str.count('hx-swap-oob="outerHTML"') >= 3 and "activity-body" not in status_str
    assert client_obj.get("/activity/status?days=7").status_code == 400
    body_str = client_obj.get("/activity/body?days=14").get_data(as_text=True)
    assert body_str.lstrip().startswith("<section") and 'id="overview-shell"' not in body_str
    assert 'hx-get="/activity/body?days=14"' in body_str


def test_pod_link_opens_activity_filtered_and_unknown_pod_is_rejected_before_scanning(monkeypatch):
    client_obj = create_demo_app().test_client()
    html_str = client_obj.get("/activity?pod=demo_1_1").get_data(as_text=True)
    assert 'data-initial-pod="demo_1_1"' in html_str
    assert 'hx-get="/activity/body?days=7&amp;pod=demo_1_1"' in html_str
    assert 'href="/activity?days=14&amp;pod=demo_1_1"' in html_str
    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_activity_source_dict", _forbidden)
    assert client_obj.get("/activity?pod=unknown_pod").status_code == 404
    assert client_obj.get("/activity?pod=a&pod=b").status_code == 400
    pod_str = client_obj.get("/pods/demo_1_1").get_data(as_text=True)
    assert 'href="/activity?pod=demo_1_1"' in pod_str
