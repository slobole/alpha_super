"""Synthetic request and execution-boundary regressions. Never uses a broker."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
import json
import re
import sqlite3
import time
from pathlib import Path

import pytest
from werkzeug.datastructures import MultiDict

from alpha.live import dashboard, runner
from alpha.live.dashboard import DashboardPodTarget
from alpha.live.dashboard_authorization import (
    assert_authorized_release, assert_authorized_vplans, operator_execution_context,
    assert_authorized_release_list,
    read_confirmation_context_dict,
)
from alpha.live.dashboard_v3.actions import ConfirmationStore
from alpha.live.dashboard_v3.app import create_app
from alpha.live.dashboard_v3.operator_tools import build_command_catalog_list
from alpha.live.state_store_v2 import LiveStateStore
from test_dashboard_v3_routes import (
    ACTION_HEADERS_DICT, StubDataProvider, confirmed_body_str, preview_nonce_str,
    fixture_provider_obj, fixture_test_client_obj, fixture_journal_path_str,
)
from test_dashboard_v3_manual_order import ManualOrderProvider
from test_live_dashboard import _build_release_obj
from test_live_runner import _insert_ready_vplan_for_release


ACTION_URL_STR = "/api/pods/dv2_caspersky_live/actions/submit_vplan"


def test_approval_replay_never_dispatches_twice(test_client_obj, provider_obj):
    body_str = confirmed_body_str(test_client_obj)
    assert test_client_obj.post(ACTION_URL_STR, data=body_str, headers=ACTION_HEADERS_DICT).status_code == 202
    assert test_client_obj.post(ACTION_URL_STR, data=body_str, headers=ACTION_HEADERS_DICT).status_code == 409
    assert len(provider_obj.action_job_dict_list) == 1


def test_simultaneous_confirms_invoke_exactly_once(test_client_obj, provider_obj):
    body_str = confirmed_body_str(test_client_obj)
    def post_status_int(_index_int):
        with test_client_obj.application.test_client() as client_obj:
            return client_obj.post(ACTION_URL_STR, data=body_str, headers=ACTION_HEADERS_DICT).status_code
    with ThreadPoolExecutor(max_workers=2) as pool_obj:
        status_list = list(pool_obj.map(post_status_int, range(2)))
    assert sorted(status_list) == [202, 409]
    assert len(provider_obj.action_job_dict_list) == 1


@pytest.mark.parametrize("changed_field_str", ["mode_str", "account_route_str", "latest_vplan_id_int", "latest_decision_plan_id_int"])
def test_changed_scope_or_plan_rejects_before_dispatch(test_client_obj, provider_obj, changed_field_str):
    body_str = confirmed_body_str(test_client_obj)
    provider_obj.summary_dict["pod_row_dict_list"][0][changed_field_str] = "changed"
    assert test_client_obj.post(ACTION_URL_STR, data=body_str, headers=ACTION_HEADERS_DICT).status_code == 409
    assert provider_obj.action_job_dict_list == []


def test_render_timestamp_is_not_execution_state(test_client_obj, provider_obj):
    body_str = confirmed_body_str(test_client_obj)
    provider_obj.summary_dict["as_of_timestamp_str"] = "2099-01-01T00:00:00Z"
    assert test_client_obj.post(ACTION_URL_STR, data=body_str, headers=ACTION_HEADERS_DICT).status_code == 202


@pytest.mark.parametrize("target_url_str", [
    "/api/pods/qp_mr_live/actions/submit_vplan", "/api/pods/dv2_caspersky_live/actions/tick",
])
def test_cross_action_and_pod_confirmation_rejected(test_client_obj, provider_obj, target_url_str):
    body_str = confirmed_body_str(test_client_obj)
    assert test_client_obj.post(target_url_str, data=body_str, headers=ACTION_HEADERS_DICT).status_code == 409
    assert test_client_obj.post(ACTION_URL_STR, data=body_str, headers=ACTION_HEADERS_DICT).status_code == 409
    assert provider_obj.action_job_dict_list == []


@pytest.mark.parametrize("outcome_str", ["cancel", "expire", "new_preview"])
def test_cancel_expiry_and_replacement_invalidate(test_client_obj, provider_obj, monkeypatch, outcome_str):
    body_str = confirmed_body_str(test_client_obj)
    if outcome_str == "cancel":
        test_client_obj.get("/fragments/action-preview-cancel/dv2_caspersky_live")
    elif outcome_str == "new_preview":
        preview_nonce_str(test_client_obj)
    else:
        future_float = time.monotonic() + 121
        monkeypatch.setattr("alpha.live.dashboard_v3.actions.time.monotonic", lambda: future_float)
    assert test_client_obj.post(ACTION_URL_STR, data=body_str, headers=ACTION_HEADERS_DICT).status_code == 409
    assert provider_obj.action_job_dict_list == []


def test_dispatch_exception_cannot_restore_nonce(test_client_obj, provider_obj, monkeypatch):
    body_str = confirmed_body_str(test_client_obj)
    call_list = []
    def fail_fn(*arg_tuple):
        call_list.append(True)
        raise RuntimeError("ambiguous dispatch")
    monkeypatch.setattr(provider_obj, "start_action_job", fail_fn)
    assert test_client_obj.post(ACTION_URL_STR, data=body_str, headers=ACTION_HEADERS_DICT).status_code == 503
    assert test_client_obj.post(ACTION_URL_STR, data=body_str, headers=ACTION_HEADERS_DICT).status_code == 409
    assert call_list == [True]


def test_preview_does_not_dispatch_or_journal(test_client_obj, provider_obj, journal_path_str):
    from pathlib import Path
    preview_nonce_str(test_client_obj)
    assert not Path(journal_path_str).exists()
    assert provider_obj.action_job_dict_list == []


@pytest.mark.parametrize("body_str", ['[]', 'true', 'null', '42', '{"confirmed_bool":true,"confirmed_bool":false}', '{"confirmed_bool":true,"shell":"bad"}'])
def test_ambiguous_or_extra_json_rejected(test_client_obj, provider_obj, body_str):
    assert test_client_obj.post(ACTION_URL_STR, data=body_str, headers=ACTION_HEADERS_DICT).status_code == 400
    assert provider_obj.action_job_dict_list == []


def test_duplicate_form_rejected(test_client_obj):
    headers_dict = {key_str: value_str for key_str, value_str in ACTION_HEADERS_DICT.items() if key_str != "Content-Type"}
    response_obj = test_client_obj.post(ACTION_URL_STR,
        data=MultiDict([("confirmed_bool", "true"), ("confirmed_bool", "false")]), headers=headers_dict)
    assert response_obj.status_code == 400


def manual_preview_tuple():
    provider_obj = ManualOrderProvider()
    client_obj = create_app(provider_obj).test_client()
    body_dict = {"confirmed_bool": True, "asset_str": "aapl", "side_str": "buy", "broker_order_type_str": "LMT",
        "quantity_int": "10", "limit_price_float": "100.25", "time_in_force_str": "DAY",
        "operator_id_str": "ops", "reason_str": "test", "confirmation_text_str": "SUBMIT MANUAL ORDER"}
    headers_dict = {"Origin": "http://localhost", "X-Alpha-Action-Token": "token_123"}
    return provider_obj, client_obj, body_dict, headers_dict


@pytest.mark.parametrize("price_str", ["nan", "inf", "-inf", "1e999"])
@pytest.mark.parametrize("format_str", ["json", "data"])
def test_nonfinite_manual_limit_never_authorized(price_str, format_str):
    provider_obj, client_obj, body_dict, headers_dict = manual_preview_tuple()
    body_dict["limit_price_float"] = price_str
    if format_str == "data":
        body_dict["confirmed_bool"] = "true"
    response_obj = client_obj.post("/api/pods/pod_manual/manual-order-preview", headers=headers_dict, **{format_str: body_dict})
    assert response_obj.status_code == 400
    assert provider_obj.submitted_body_dict is None
    assert client_obj.application.config["confirmation_store_obj"]._entry_dict == {}


@pytest.mark.parametrize("field_str", ["asset_str", "side_str", "quantity_int", "limit_price_float", "broker_order_type_str", "time_in_force_str", "operator_id_str", "reason_str"])
def test_manual_final_request_cannot_change_frozen_fields(field_str):
    provider_obj, client_obj, body_dict, headers_dict = manual_preview_tuple()
    response_obj = client_obj.post("/api/pods/pod_manual/manual-order-preview", json=body_dict, headers=headers_dict)
    nonce_str = re.search(r'"confirmation_nonce_str": "([^"]+)"', response_obj.get_data(as_text=True)).group(1)
    response_obj = client_obj.post("/api/pods/pod_manual/manual-order", headers=headers_dict,
        json={"confirmed_bool": True, "confirmation_nonce_str": nonce_str, field_str: "changed"})
    assert response_obj.status_code == 400
    assert provider_obj.submitted_body_dict is None


def test_manual_post_dispatch_valueerror_is_uncertain_and_not_retried(monkeypatch):
    provider_obj, client_obj, body_dict, headers_dict = manual_preview_tuple()
    response_obj = client_obj.post("/api/pods/pod_manual/manual-order-preview", json=body_dict, headers=headers_dict)
    nonce_str = re.search(r'"confirmation_nonce_str": "([^"]+)"', response_obj.get_data(as_text=True)).group(1)
    call_list = []
    def fail_fn(*arg_tuple):
        call_list.append(True)
        raise ValueError("response lost after attempt")
    monkeypatch.setattr(provider_obj, "submit_manual_order_dict", fail_fn)
    confirm_dict = {"confirmed_bool": True, "confirmation_nonce_str": nonce_str}
    response_obj = client_obj.post("/api/pods/pod_manual/manual-order", json=confirm_dict, headers=headers_dict)
    assert response_obj.status_code == 503
    assert "Outcome unknown" in response_obj.get_json()["message_str"]
    assert client_obj.post("/api/pods/pod_manual/manual-order", json=confirm_dict, headers=headers_dict).status_code == 409
    assert call_list == [True]


@pytest.fixture
def execution_tuple(tmp_path):
    release_obj = _build_release_obj()
    db_path_obj = tmp_path / "pod.sqlite3"
    store_obj = LiveStateStore(str(db_path_obj))
    store_obj.upsert_release(release_obj)
    plan_obj = _insert_ready_vplan_for_release(store_obj, release_obj)
    target_obj = DashboardPodTarget(release_obj, str(db_path_obj), False)
    return target_obj, store_obj, plan_obj


def approved_target_obj(target_obj, action_str="submit_vplan"):
    context_dict = {**read_confirmation_context_dict(target_obj), "action_name_str": action_str,
        "expires_at_float": time.monotonic() + 120, "releases_root_path_str": "synthetic", "config_path_str": "synthetic"}
    return replace(target_obj, operator_confirmation_dict=context_dict)


def test_context_reader_is_read_only_and_missing_db_not_created(execution_tuple, tmp_path):
    target_obj, store_obj, plan_obj = execution_tuple
    before_bytes = open(target_obj.db_path_str, "rb").read()
    context_dict = read_confirmation_context_dict(target_obj)
    assert context_dict["vplan_id_int"] == plan_obj.vplan_id_int
    assert open(target_obj.db_path_str, "rb").read() == before_bytes
    missing_path_obj = tmp_path / "not-created.sqlite3"
    with pytest.raises(ValueError, match="evidence"):
        read_confirmation_context_dict(replace(target_obj, db_path_str=str(missing_path_obj)))
    assert not missing_path_obj.exists()


@pytest.mark.parametrize("table_str", ["decision_plan", "vplan"])
def test_foreign_account_plan_cannot_be_approved(execution_tuple, table_str):
    target_obj, store_obj, plan_obj = execution_tuple
    with store_obj._connect() as connection_obj:
        connection_obj.execute(f"UPDATE {table_str} SET account_route_str = ?", ("DU_FOREIGN",))
    with pytest.raises(ValueError, match="evidence"):
        read_confirmation_context_dict(target_obj)


@pytest.mark.parametrize("change_str", ["state", "account", "mode", "release", "db", "expiry", "action"])
def test_worker_revalidates_before_any_command(execution_tuple, monkeypatch, change_str):
    target_obj, store_obj, plan_obj = execution_tuple
    approved_obj = approved_target_obj(target_obj)
    current_obj = target_obj
    action_str = "submit_vplan"
    if change_str == "state":
        store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitted")
    elif change_str in {"account", "mode", "release"}:
        key_str = {"account": "account_route_str", "mode": "mode_str", "release": "release_id_str"}[change_str]
        current_obj = replace(target_obj, release_obj=replace(target_obj.release_obj, **{key_str: "changed"}))
    elif change_str == "db":
        current_obj = replace(target_obj, db_path_str="missing.sqlite3")
    elif change_str == "expiry":
        approved_obj.operator_confirmation_dict["expires_at_float"] = 0
    else:
        action_str = "tick"
    monkeypatch.setattr(dashboard.DashboardApp, "get_target_for_pod", lambda *_: current_obj)
    monkeypatch.setattr(dashboard, "LiveStateStore", lambda *_: pytest.fail("Worker reached a state writer"))
    with pytest.raises(ValueError):
        dashboard._run_dashboard_action_for_pod(approved_obj, action_str, "synthetic", "synthetic", "synthetic", datetime.now(UTC))


def test_release_reload_and_adapter_checks_precede_writes_or_connect(execution_tuple, monkeypatch):
    target_obj, store_obj, plan_obj = execution_tuple
    approved_obj = approved_target_obj(target_obj)
    changed_obj = replace(target_obj.release_obj, broker_port_int=1)
    monkeypatch.setattr(runner, "load_release_list", lambda *_: [changed_obj])
    monkeypatch.setattr(store_obj, "upsert_release_list", lambda *_: pytest.fail("Release drift reached write"))
    resolver_obj = runner.BrokerAdapterResolver(adapter_factory_func=lambda *_: pytest.fail("Adapter created"))
    with operator_execution_context(approved_obj, lambda _: target_obj, "submit_vplan"):
        with pytest.raises(ValueError, match="Release changed"):
            runner._load_release_list_and_sync("synthetic", store_obj, pod_id_str=target_obj.release_obj.pod_id_str)
        with pytest.raises(ValueError, match="Release changed"):
            resolver_obj.get_adapter(changed_obj)
    # Guard reset even after errors; scheduled/non-dashboard callers unchanged.
    assert_authorized_release(changed_obj)


@pytest.mark.parametrize("action_str", ["submit_vplan", "post_execution_reconcile", "tick"])
def test_selected_plan_account_always_bound_including_tick(execution_tuple, action_str):
    target_obj, store_obj, plan_obj = execution_tuple
    approved_obj = approved_target_obj(target_obj, action_str)
    with operator_execution_context(approved_obj, lambda _: target_obj, action_str):
        with pytest.raises(ValueError, match="Pod/account/release"):
            assert_authorized_vplans("submit_vplan", [replace(plan_obj, account_route_str="DU_FOREIGN")])


def test_direct_submit_pins_rows_but_tick_can_build_new_plan(execution_tuple):
    target_obj, store_obj, plan_obj = execution_tuple
    changed_obj = replace(plan_obj, order_delta_map={"AAPL": 999.0})
    for action_str in ("submit_vplan", "tick"):
        approved_obj = approved_target_obj(target_obj, action_str)
        with operator_execution_context(approved_obj, lambda _: target_obj, action_str):
            if action_str == "submit_vplan":
                with pytest.raises(ValueError, match="plan changed"):
                    assert_authorized_vplans("submit_vplan", [changed_obj])
            else:
                assert_authorized_vplans("submit_vplan", [changed_obj])


@pytest.mark.parametrize("action_str", ["submit_vplan", "post_execution_reconcile"])
def test_runner_changed_selected_plan_rejected_before_broker(execution_tuple, monkeypatch, action_str):
    target_obj, store_obj, plan_obj = execution_tuple
    if action_str == "post_execution_reconcile":
        store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitted")
    approved_obj = approved_target_obj(target_obj, action_str)
    with operator_execution_context(approved_obj, lambda _: target_obj, action_str):
        with store_obj._connect() as connection_obj:
            connection_obj.execute("UPDATE vplan SET order_delta_json_str = ?", ('{"AAPL":999}',))
        monkeypatch.setattr(runner.BrokerAdapterResolver, "get_adapter", lambda *_: pytest.fail("Broker reached"))
        with pytest.raises(ValueError, match="plan changed"):
            if action_str == "submit_vplan":
                runner.submit_ready_vplans(store_obj, None, datetime.now(UTC), "paper", False, pod_id_str=target_obj.release_obj.pod_id_str)
            else:
                runner.post_execution_reconcile(store_obj, None, datetime.now(UTC), "paper", log_path_str=str(Path(target_obj.db_path_str).with_suffix('.jsonl')), pod_id_str=target_obj.release_obj.pod_id_str)


def test_release_upgrade_allows_terminal_history_but_not_active_old_plan(execution_tuple):
    target_obj, store_obj, plan_obj = execution_tuple
    upgraded_obj = replace(target_obj, release_obj=replace(target_obj.release_obj, release_id_str="version2"))
    with pytest.raises(ValueError):
        read_confirmation_context_dict(upgraded_obj)
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "completed")
    store_obj.mark_decision_plan_status(plan_obj.decision_plan_id_int, "completed")
    context_dict = read_confirmation_context_dict(upgraded_obj)
    assert context_dict["release_id_str"] == "version2"
    assert context_dict["submit_hash_list"] == []
    assert context_dict["submitted_hash_list"] == []


def test_catalog_quotes_paths_and_has_no_shell_execution(execution_tuple):
    target_obj, _, _ = execution_tuple
    command_list = build_command_catalog_list(target_obj, "C:/owner's releases")
    assert len(command_list) == 6
    assert "'C:/owner''s releases'" in command_list[0]["command_str"]
    assert all("--pod-id" in item_dict["command_str"] and "--db-path" in item_dict["command_str"] for item_dict in command_list)
    assert "Writes job/release" in command_list[0]["effects_str"]


def test_readonly_blocks_new_manual_preview_and_catalog_before_provider(monkeypatch):
    provider_obj = StubDataProvider()
    monkeypatch.setattr(provider_obj, "get_target_for_pod", lambda *_: pytest.fail("Read-only reached target"))
    client_obj = create_app(provider_obj, read_only_bool=True).test_client()
    assert client_obj.get("/fragments/command-catalog/any").status_code == 403
    assert client_obj.post("/api/pods/any/manual-order-preview", json={}).status_code == 403


@pytest.mark.parametrize("argument_list,expected_bool", [([], True), (["--read-only"], True), (["--enable-actions"], False)])
def test_supported_cli_requires_explicit_action_opt_in(monkeypatch, argument_list, expected_bool):
    from types import SimpleNamespace
    from alpha.live.dashboard_v3 import __main__ as launcher_module
    captured_dict = {}
    def create_fn(**option_dict):
        captured_dict.update(option_dict)
        return SimpleNamespace(run=lambda **_: None)
    monkeypatch.setenv("ALPHA_OPS_OPERATOR_ACCESS_TOKEN_STR", "synthetic-credential-for-tests-only")
    monkeypatch.setattr("sys.argv", ["dashboard", "--skip-env-file", *argument_list])
    monkeypatch.setattr(launcher_module, "create_app", create_fn)
    assert launcher_module.main() == 0
    assert captured_dict["read_only_bool"] is expected_bool


def test_disabled_release_history_preserves_metadata_without_approval_bypass(execution_tuple, monkeypatch):
    target_obj, store_obj, _ = execution_tuple
    approved_obj = approved_target_obj(target_obj)
    historical_obj = replace(target_obj.release_obj, release_id_str="old-version", enabled_bool=False)
    release_list = [target_obj.release_obj, historical_obj]
    monkeypatch.setattr(runner, "load_release_list", lambda *_: release_list)
    written_list = []
    monkeypatch.setattr(store_obj, "upsert_release_list", lambda item_list: written_list.extend(item_list))
    with operator_execution_context(approved_obj, lambda _: target_obj, "submit_vplan"):
        assert runner._load_release_list_and_sync("synthetic", store_obj, pod_id_str=target_obj.release_obj.pod_id_str) == release_list
        assert written_list == release_list
        for invalid_list in ([historical_obj], [target_obj.release_obj, target_obj.release_obj],
            [replace(target_obj.release_obj, broker_port_int=1)],
            [target_obj.release_obj, replace(target_obj.release_obj, enabled_bool=False)]):
            with pytest.raises(ValueError):
                assert_authorized_release_list(invalid_list)
