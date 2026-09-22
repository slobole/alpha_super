"""All actions below use only in-memory synthetic targets and fake results."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json

from flask import Flask
import pytest

from alpha.live.dashboard_v4.tools_actions import ToolsActionService, register_tools_action_routes
from alpha.live.dashboard_v4.tools_execution import SyntheticToolsActionProvider


@dataclass(frozen=True)
class DemoRelease:
    release_id_str: str = "demo-release"
    pod_id_str: str = "demo-pod"
    user_id_str: str = "demo-owner"
    account_route_str: str = "U1234567"
    mode_str: str = "live"
    enabled_bool: bool = True


@dataclass(frozen=True)
class DemoTarget:
    release_obj: DemoRelease = DemoRelease()
    db_path_str: str = "synthetic.db"
    operator_confirmation_dict: dict | None = None


@pytest.fixture
def harness_tuple():
    target_list = [DemoTarget()]
    def target_fn(pod_id_str):
        target_obj = target_list[0]
        return target_obj if target_obj.release_obj.pod_id_str == pod_id_str else None
    provider_obj = SyntheticToolsActionProvider(target_fn)
    service_obj = ToolsActionService(provider_obj, target_scope_fn=target_fn, enabled_bool=True, demo_bool=True,
        now_fn=lambda: datetime(2026, 9, 22, 13, tzinfo=UTC))
    app_obj = Flask(__name__)
    register_tools_action_routes(app_obj, service_obj)
    headers_dict = {"Origin": "http://localhost", "X-Alpha-Action-Token": service_obj.action_token_str}
    return app_obj.test_client(), service_obj, provider_obj, target_list, headers_dict


def _preview_dict(harness_tuple, action_str="tick", **extra_dict):
    client_obj, _, _, _, headers_dict = harness_tuple
    response_obj = client_obj.post(f"/api/demo-tools/demo-pod/{action_str}/preview",
        json={"confirmed_bool": True, **extra_dict}, headers=headers_dict)
    assert response_obj.status_code == 200
    return response_obj.json


def _confirm_obj(harness_tuple, preview_dict, action_str="tick"):
    client_obj, _, _, _, headers_dict = harness_tuple
    return client_obj.post(f"/api/demo-tools/demo-pod/{action_str}/confirm", json={"confirmed_bool": True,
        "confirmation_nonce_str": preview_dict["confirmation_nonce_str"], "browser_confirmed_bool": True}, headers=headers_dict)


@pytest.mark.parametrize("route_str", ["tick/preview", "tick/confirm", "tick/cancel", "jobs/example"])
def test_disabled_before_provider_token_or_target(route_str):
    def blocked_fn(*argument_tuple):
        raise AssertionError("Disabled routes must not inspect a target.")
    service_obj = ToolsActionService(object(), target_scope_fn=blocked_fn)
    app_obj = Flask(__name__)
    register_tools_action_routes(app_obj, service_obj)
    client_obj = app_obj.test_client()
    response_obj = (client_obj.get if route_str.startswith("jobs/") else client_obj.post)("/api/demo-tools/demo-pod/" + route_str)
    assert response_obj.status_code == 403
    assert service_obj.action_token_str == ""
    assert service_obj._action_token_str == ""


def test_enabling_non_demo_or_non_synthetic_provider_is_rejected():
    with pytest.raises(ValueError, match="synthetic demo"):
        ToolsActionService(object(), target_scope_fn=lambda _: None, enabled_bool=True)
    with pytest.raises(ValueError, match="synthetic demo"):
        ToolsActionService(object(), target_scope_fn=lambda _: None, enabled_bool=True, demo_bool=True)


@pytest.mark.parametrize("action_str", ["tick", "submit_vplan", "post_execution_reconcile", "eod_snapshot", "compare_reference"])
def test_preview_confirm_poll_simulates_and_journals_only(harness_tuple, action_str):
    client_obj, service_obj, provider_obj, _, _ = harness_tuple
    preview_dict = _preview_dict(harness_tuple, action_str)
    assert preview_dict["expires_in_seconds_int"] == 120
    assert preview_dict["demo_bool"] is True
    assert "U1234567" not in json.dumps(preview_dict)
    assert not provider_obj.job_dict
    response_obj = _confirm_obj(harness_tuple, preview_dict, action_str)
    assert response_obj.status_code == 202
    assert response_obj.json["status_str"] == "succeeded"
    assert "No real command" in response_obj.json["message_str"]
    assert client_obj.get(response_obj.json["poll_url_str"]).json == response_obj.json
    assert len(provider_obj.journal_list) == 2
    assert [event_dict["payload_dict"]["status_str"] for event_dict in service_obj.demo_event_list] == ["simulated_requested", "simulated_succeeded"]
    assert service_obj.demo_event_list[0]["timestamp_str"] == "2026-09-22T13:00:00+00:00"
    assert _confirm_obj(harness_tuple, preview_dict, action_str).status_code == 409
    assert len(provider_obj.job_dict) == 1


@pytest.mark.parametrize("raw_str", [
    '{"confirmed_bool":true,"confirmed_bool":true}',
    '{"confirmed_bool":true,"command_str":"do-anything"}',
    '{"confirmed_bool":"true"}', '[]', 'null', '{',
    '{"confirmed_bool":true,"manual_order_dict":{"quantity_int":1,"quantity_int":2}}',
    '{"confirmed_bool":true,"quantity_int":NaN}',
])
def test_ambiguous_extra_nonfinite_and_nonobject_json_rejected(harness_tuple, raw_str):
    client_obj, _, provider_obj, _, headers_dict = harness_tuple
    response_obj = client_obj.post("/api/demo-tools/demo-pod/tick/preview", data=raw_str,
        content_type="application/json", headers=headers_dict)
    assert response_obj.status_code == 400
    assert provider_obj.job_dict == {}


@pytest.mark.parametrize("headers_change_dict", [{"Origin": "http://evil.invalid"}, {"Origin": "https://localhost"},
    {"Origin": "null"}, {"X-Alpha-Action-Token": "wrong"}, {"Referer": "http://evil.invalid/page"}])
def test_origin_scheme_referer_and_token_boundaries(harness_tuple, headers_change_dict):
    client_obj, _, _, _, headers_dict = harness_tuple
    response_obj = client_obj.post("/api/demo-tools/demo-pod/tick/preview", json={"confirmed_bool": True},
        headers={**headers_dict, **headers_change_dict})
    assert response_obj.status_code == 403


def test_no_origin_and_no_token_rejected(harness_tuple):
    client_obj, _, _, _, _ = harness_tuple
    assert client_obj.post("/api/demo-tools/demo-pod/tick/preview", json={"confirmed_bool": True}).status_code == 403


@pytest.mark.parametrize("suffix_str", ["?pod=other", "?pod=a&pod=b"])
def test_query_params_rejected(harness_tuple, suffix_str):
    client_obj, _, _, _, headers_dict = harness_tuple
    assert client_obj.post("/api/demo-tools/demo-pod/tick/preview" + suffix_str,
        json={"confirmed_bool": True}, headers=headers_dict).status_code == 400


def test_cancel_invalidates_preview(harness_tuple):
    client_obj, _, _, _, headers_dict = harness_tuple
    preview_dict = _preview_dict(harness_tuple)
    assert client_obj.post("/api/demo-tools/demo-pod/tick/cancel", json={"confirmed_bool": True}, headers=headers_dict).json == {"cancelled_bool": True}
    assert _confirm_obj(harness_tuple, preview_dict).status_code == 409


@pytest.mark.parametrize("confirmation_obj", [None, False, "true", 1])
def test_browser_confirmation_is_explicit_and_missing_flag_does_not_dispatch(harness_tuple, confirmation_obj):
    client_obj, _, provider_obj, _, headers_dict = harness_tuple
    preview_dict = _preview_dict(harness_tuple)
    body_dict = {"confirmed_bool": True, "confirmation_nonce_str": preview_dict["confirmation_nonce_str"]}
    if confirmation_obj is not None:
        body_dict["browser_confirmed_bool"] = confirmation_obj
    response_obj = client_obj.post("/api/demo-tools/demo-pod/tick/confirm", json=body_dict, headers=headers_dict)
    assert response_obj.status_code == 400
    assert response_obj.json["error"] == "browser_confirmation_required"
    assert not provider_obj.job_dict
    # A cancelled browser prompt did not submit or consume the server preview.
    assert _confirm_obj(harness_tuple, preview_dict).status_code == 202


def test_new_preview_invalidates_previous_even_different_action(harness_tuple):
    old_dict = _preview_dict(harness_tuple)
    new_dict = _preview_dict(harness_tuple, "eod_snapshot")
    assert _confirm_obj(harness_tuple, old_dict).status_code == 409
    assert _confirm_obj(harness_tuple, new_dict, "eod_snapshot").status_code == 202


def test_changed_release_or_target_rejects_and_consumes_nonce(harness_tuple):
    _, _, provider_obj, target_list, _ = harness_tuple
    preview_dict = _preview_dict(harness_tuple)
    target_list[0] = replace(target_list[0], release_obj=replace(target_list[0].release_obj, account_route_str="U7654321"))
    assert _confirm_obj(harness_tuple, preview_dict).status_code == 409
    assert not provider_obj.job_dict


@pytest.mark.parametrize("release_change_dict", [{"enabled_bool": False}, {"mode_str": "paper"}, {"mode_str": "incubation"}])
def test_nonlive_or_disabled_target_rejected(harness_tuple, release_change_dict):
    client_obj, _, _, target_list, headers_dict = harness_tuple
    target_list[0] = replace(target_list[0], release_obj=replace(target_list[0].release_obj, **release_change_dict))
    assert client_obj.post("/api/demo-tools/demo-pod/tick/preview", json={"confirmed_bool": True}, headers=headers_dict).status_code == 409


def test_expired_and_wrong_action_nonce_consumed(harness_tuple):
    _, service_obj, _, _, _ = harness_tuple
    preview_dict = _preview_dict(harness_tuple)
    service_obj.confirmation_store_obj._entry_dict[preview_dict["confirmation_nonce_str"]]["expires_at_float"] = 0
    assert _confirm_obj(harness_tuple, preview_dict).status_code == 409
    preview_dict = _preview_dict(harness_tuple)
    assert _confirm_obj(harness_tuple, preview_dict, "eod_snapshot").status_code == 409
    assert _confirm_obj(harness_tuple, preview_dict).status_code == 409


def test_job_poll_cannot_cross_pod_changed_release_or_database(harness_tuple):
    client_obj, _, _, target_list, _ = harness_tuple
    response_obj = _confirm_obj(harness_tuple, _preview_dict(harness_tuple))
    poll_str = response_obj.json["poll_url_str"]
    assert client_obj.get(poll_str.replace("demo-pod", "other-pod")).status_code == 404
    assert client_obj.get(poll_str + "?a=1").status_code == 400
    target_list[0] = replace(target_list[0], db_path_str="different.db")
    assert client_obj.get(poll_str).status_code == 404


def test_dispatch_exception_is_scrubbed_and_not_retried(harness_tuple, monkeypatch):
    _, _, provider_obj, _, _ = harness_tuple
    def fail_fn(*argument_tuple):
        raise RuntimeError("C:/secret/account/U1234567?token=SECRET")
    monkeypatch.setattr(provider_obj, "start_action_job", fail_fn)
    preview_dict = _preview_dict(harness_tuple)
    response_obj = _confirm_obj(harness_tuple, preview_dict)
    assert response_obj.status_code == 503
    assert response_obj.json["status_str"] == "unknown"
    assert "SECRET" not in response_obj.text and "U1234567" not in response_obj.text
    assert _confirm_obj(harness_tuple, preview_dict).status_code == 409


def test_raw_executor_result_never_exposed(harness_tuple, monkeypatch):
    client_obj, _, provider_obj, _, _ = harness_tuple
    def job_fn(action_name_str, target_obj):
        result_dict = {"job_id_str": "upstream", "pod_id_str": "demo-pod", "status_str": "succeeded",
            "result_dict": {"account": "U1234567", "secret": "SECRET", "path": "C:/private"}, "traceback_str": "SECRET"}
        provider_obj.job_dict["upstream"] = result_dict
        return result_dict
    monkeypatch.setattr(provider_obj, "start_action_job", job_fn)
    response_obj = _confirm_obj(harness_tuple, _preview_dict(harness_tuple))
    for response_text_str in (response_obj.text, client_obj.get(response_obj.json["poll_url_str"]).text):
        assert "SECRET" not in response_text_str and "U1234567" not in response_text_str and "C:/" not in response_text_str


def test_manual_ticket_is_normalized_in_preview_and_cannot_change_at_confirm(harness_tuple):
    client_obj, _, _, _, headers_dict = harness_tuple
    manual_dict = {"asset_str": "aapl", "side_str": "BUY", "broker_order_type_str": "LMT", "quantity_int": 2,
        "limit_price_float": 12.5, "time_in_force_str": "DAY", "operator_id_str": "demo", "reason_str": "Visual demo",
        "confirmation_text_str": "SUBMIT MANUAL ORDER"}
    preview_dict = _preview_dict(harness_tuple, "manual_order", manual_order_dict=manual_dict)
    assert "BUY 2 AAPL" in preview_dict["preview_line_list"]
    assert "Limit: 12.5" in preview_dict["preview_line_list"]
    changed_obj = client_obj.post("/api/demo-tools/demo-pod/manual_order/confirm", json={"confirmed_bool": True,
        "confirmation_nonce_str": preview_dict["confirmation_nonce_str"], "manual_order_dict": {**manual_dict, "quantity_int": 3}}, headers=headers_dict)
    assert changed_obj.status_code == 400
    assert _confirm_obj(harness_tuple, preview_dict, "manual_order").status_code == 202


@pytest.mark.parametrize("manual_change_dict", [{"quantity_int": 0}, {"quantity_int": -1}, {"quantity_int": 1.5},
    {"confirmation_text_str": "yes"}, {"time_in_force_str": "GTC"}, {"arbitrary_flag": True}])
def test_invalid_manual_ticket_rejected(harness_tuple, manual_change_dict):
    client_obj, _, provider_obj, _, headers_dict = harness_tuple
    manual_dict = {"asset_str": "AAPL", "side_str": "BUY", "broker_order_type_str": "MKT", "quantity_int": 2,
        "time_in_force_str": "DAY", "operator_id_str": "demo", "reason_str": "Visual demo", "confirmation_text_str": "SUBMIT MANUAL ORDER"}
    response_obj = client_obj.post("/api/demo-tools/demo-pod/manual_order/preview", json={"confirmed_bool": True,
        "manual_order_dict": {**manual_dict, **manual_change_dict}}, headers=headers_dict)
    assert response_obj.status_code == 409
    assert not provider_obj.job_dict


def test_production_flip_fails_closed_after_construction(harness_tuple):
    client_obj, service_obj, _, _, headers_dict = harness_tuple
    service_obj.demo_bool = False
    assert service_obj.action_token_str == ""
    assert client_obj.post("/api/demo-tools/demo-pod/tick/preview", json={"confirmed_bool": True}, headers=headers_dict).status_code == 403


def test_wrong_action_and_large_body_rejected(harness_tuple):
    client_obj, _, _, _, headers_dict = harness_tuple
    assert client_obj.post("/api/demo-tools/demo-pod/shell/preview", json={"confirmed_bool": True}, headers=headers_dict).status_code == 400
    assert client_obj.post("/api/demo-tools/demo-pod/tick/preview", data="x" * 16385,
        content_type="application/json", headers=headers_dict).status_code == 400


@pytest.mark.parametrize("price_str", ['"NaN"', '"Infinity"', '"-Infinity"', '1e309'])
def test_nonfinite_manual_limit_price_rejected(harness_tuple, price_str):
    client_obj, service_obj, provider_obj, _, headers_dict = harness_tuple
    manual_dict = {"asset_str": "AAPL", "side_str": "BUY", "broker_order_type_str": "LMT", "quantity_int": 2,
        "time_in_force_str": "DAY", "operator_id_str": "demo", "reason_str": "Visual demo", "confirmation_text_str": "SUBMIT MANUAL ORDER"}
    body_str = json.dumps({"confirmed_bool": True, "manual_order_dict": {**manual_dict, "limit_price_float": "REPLACE_PRICE"}})
    body_str = body_str.replace('"REPLACE_PRICE"', price_str)
    response_obj = client_obj.post("/api/demo-tools/demo-pod/manual_order/preview", data=body_str,
        content_type="application/json", headers=headers_dict)
    assert response_obj.status_code == 409
    assert not provider_obj.job_dict and not service_obj.confirmation_store_obj._entry_dict
