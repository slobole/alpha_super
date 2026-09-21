"""Financial acquisition is unreachable from the lightweight status route."""

from datetime import UTC, datetime, timedelta

import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import create_demo_app


AS_OF_TS = datetime(2026, 9, 21, 14, tzinfo=UTC)


def _workspace_dict(*, timestamp_str=AS_OF_TS.isoformat()):
    return {"client_dict": {"accounts": [], "display_name": "Fixture"},
        "operations_account_list": [{"pod_id": "own_pod", "account_route": "U_PRIVATE", "display_name": "Owned pod"}],
        "summary_dict": {"as_of_timestamp_str": timestamp_str, "pod_row_dict_list": []},
        "operations_error_str": None}


def _forbidden_finance(*args_tuple, **kwargs_dict):
    pytest.fail("A status request reached full financial acquisition")


def test_explicit_status_callback_never_calls_financial_callback_or_default_loader(monkeypatch):
    call_list = []

    def operations_fn():
        call_list.append("operations")
        return _workspace_dict()

    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_workspace_snapshot_tuple", _forbidden_finance)
    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_operations_workspace_dict", _forbidden_finance)
    monkeypatch.setattr("alpha.live.dashboard_v4.app.build_performance_page_dict", _forbidden_finance)
    # The shared System light reads bounded service receipts, not financial history.
    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_system_source_dict", lambda *args_tuple, **kwargs_dict:
        {"scope_verified_bool": True, "checked_timestamp_str": AS_OF_TS.isoformat()})
    app_obj = create_app(object(), workspace_snapshot_fn=_forbidden_finance,
        operations_workspace_fn=operations_fn, now_fn=lambda: AS_OF_TS)
    response_obj = app_obj.test_client().get("/performance/status")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert call_list == ["operations"]
    assert html_str.count('hx-swap-oob="outerHTML"') == 4
    for element_str in ("performance-status", "operator-rail", "operator-header", "operator-mobile-header"):
        assert f'id="{element_str}"' in html_str
    assert "Owned pod" in html_str and "U_PRIVATE" not in html_str
    # This minimal fixture has no matching saved LIVE row/release. It may
    # render the owned navigation, but cannot assert fresh operating status.
    assert 'data-source-valid-ms="0"' in html_str
    assert 'id="overview-shell"' not in html_str and 'data-performance-dates' not in html_str
    assert 'Monthly return' not in html_str and 'Investor PDF' not in html_str
    assert response_obj.headers["Cache-Control"] == "no-store"


def test_missing_status_callback_uses_operations_loader_even_with_full_snapshot_injection(monkeypatch):
    call_list = []
    provider_obj = object()

    def operations_fn(actual_provider_obj, *, as_of_ts):
        assert actual_provider_obj is provider_obj and as_of_ts == AS_OF_TS
        call_list.append("loader")
        return _workspace_dict()

    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_operations_workspace_dict", operations_fn)
    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_workspace_snapshot_tuple", _forbidden_finance)
    app_obj = create_app(provider_obj, workspace_snapshot_fn=_forbidden_finance, now_fn=lambda: AS_OF_TS)
    assert app_obj.test_client().get("/performance/status").status_code == 200
    assert call_list == ["loader"]


@pytest.mark.parametrize("query_str", ["period=All", "level=pods", "download=csv", "expected=anything", "x=1&x=2"])
def test_status_query_is_rejected_before_any_acquisition(query_str):
    app_obj = create_app(object(), workspace_snapshot_fn=_forbidden_finance,
        operations_workspace_fn=_forbidden_finance, now_fn=lambda: AS_OF_TS)
    assert app_obj.test_client().get("/performance/status?" + query_str).status_code == 400


@pytest.mark.parametrize("method_str", ["POST", "PUT", "DELETE"])
def test_status_cannot_enable_actions(method_str):
    app_obj = create_app(object(), workspace_snapshot_fn=_forbidden_finance,
        operations_workspace_fn=_forbidden_finance, now_fn=lambda: AS_OF_TS)
    assert app_obj.test_client().open("/performance/status", method=method_str).status_code == 403


def test_status_uses_completion_clock_so_slow_acquisition_cannot_renew_evidence():
    clock_dict = {"now_ts": AS_OF_TS}

    def operations_fn():
        clock_dict["now_ts"] += timedelta(seconds=121)
        return _workspace_dict()

    app_obj = create_app(object(), workspace_snapshot_fn=_forbidden_finance,
        operations_workspace_fn=operations_fn, now_fn=lambda: clock_dict["now_ts"])
    response_obj = app_obj.test_client().get("/performance/status")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert 'data-source-valid-ms="0"' in html_str
    assert 'data-last-update="10:00:00"' in html_str
    assert "System unknown" in html_str


def test_demo_status_has_its_own_in_memory_source_without_local_fallback(monkeypatch):
    app_obj = create_demo_app()
    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_operations_workspace_dict", _forbidden_finance)
    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_workspace_snapshot_tuple", _forbidden_finance)
    monkeypatch.setattr("alpha.live.dashboard_v4.app.build_performance_page_dict", _forbidden_finance)
    response_obj = app_obj.test_client().get("/performance/status")
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert "DVO2" in html_str and "QPI" in html_str
    assert "sample data" in html_str
    assert 'data-source-valid-ms="0"' not in html_str
