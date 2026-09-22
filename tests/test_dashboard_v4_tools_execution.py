"""Synthetic execution must never fall through to production dependencies."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import sqlite3
import subprocess

import pytest

from alpha.live.dashboard_v4.tools_actions import ToolsActionService
from alpha.live.dashboard_v4.tools_execution import SyntheticToolsActionProvider


@dataclass(frozen=True)
class SyntheticRelease:
    release_id_str: str = "synthetic-release"
    pod_id_str: str = "synthetic-pod"
    user_id_str: str = "synthetic-owner"
    account_route_str: str = "U1234567"
    mode_str: str = "live"
    enabled_bool: bool = True


@dataclass(frozen=True)
class SyntheticTarget:
    release_obj: SyntheticRelease = SyntheticRelease()
    db_path_str: str = "never-open-this.sqlite"
    operator_confirmation_dict: dict | None = None


def _service_obj():
    target_obj = SyntheticTarget()
    scope_fn = lambda pod_id_str: target_obj if pod_id_str == "synthetic-pod" else None
    provider_obj = SyntheticToolsActionProvider(scope_fn)
    return ToolsActionService(provider_obj, target_scope_fn=scope_fn, enabled_bool=True, demo_bool=True)


def test_all_synthetic_tools_without_database_subprocess_executor_or_broker(monkeypatch):
    from alpha.live.dashboard import DashboardApp
    from alpha.live import manual_order
    from alpha.live.state_store_v2 import LiveStateStore

    def forbidden_fn(*argument_tuple, **keyword_dict):
        raise AssertionError("Synthetic Tools touched a production dependency")

    monkeypatch.setattr(sqlite3, "connect", forbidden_fn)
    monkeypatch.setattr(subprocess, "run", forbidden_fn)
    monkeypatch.setattr(subprocess, "Popen", forbidden_fn)
    monkeypatch.setattr(DashboardApp, "__post_init__", forbidden_fn)
    monkeypatch.setattr(LiveStateStore, "__init__", forbidden_fn)
    monkeypatch.setattr(manual_order, "submit_manual_order_ticket_dict", forbidden_fn)
    monkeypatch.setattr("alpha.live.dashboard_v3.journal.append_journal_entry", forbidden_fn)
    service_obj = _service_obj()
    for action_str in ("tick", "submit_vplan", "post_execution_reconcile", "eod_snapshot", "compare_reference", "manual_order"):
        body_dict = {"confirmed_bool": True}
        if action_str == "submit_vplan":
            body_dict["vplan_id_int"] = 41
        if action_str == "manual_order":
            body_dict["manual_order_dict"] = {"asset_str": "MSFT", "side_str": "SELL", "quantity_int": 2,
                "broker_order_type_str": "MKT", "time_in_force_str": "DAY", "operator_id_str": "demo",
                "reason_str": "Test simulation", "confirmation_text_str": "SUBMIT MANUAL ORDER"}
        preview_dict = service_obj.preview_dict("synthetic-pod", action_str, body_dict)
        result_dict, status_int = service_obj.confirm_tuple("synthetic-pod", action_str, preview_dict["confirmation_nonce_str"])
        assert status_int == 202
        assert result_dict["status_str"] == "succeeded"
        assert result_dict["demo_bool"] is True
    assert len(service_obj.demo_event_list) == 12


def test_duplicate_concurrent_confirmation_simulates_once():
    service_obj = _service_obj()
    preview_dict = service_obj.preview_dict("synthetic-pod", "tick", {"confirmed_bool": True})
    def confirm_fn():
        try:
            return service_obj.confirm_tuple("synthetic-pod", "tick", preview_dict["confirmation_nonce_str"])[1]
        except ValueError:
            return 409
    with ThreadPoolExecutor(max_workers=2) as pool_obj:
        status_list = list(pool_obj.map(lambda _: confirm_fn(), range(2)))
    assert sorted(status_list) == [202, 409]
    assert len(service_obj.provider_obj.job_dict) == 1


def test_synthetic_journal_and_job_storage_bounded():
    service_obj = _service_obj()
    for _index_int in range(210):
        preview_dict = service_obj.preview_dict("synthetic-pod", "tick", {"confirmed_bool": True})
        service_obj.confirm_tuple("synthetic-pod", "tick", preview_dict["confirmation_nonce_str"])
    assert len(service_obj.demo_event_list) == 400
    assert len(service_obj.provider_obj.journal_list) == 400
    assert len(service_obj._job_dict) == 200
    assert len(service_obj.provider_obj.job_dict) == 200


def test_provider_subclass_cannot_enable_real_execution_by_claiming_synthetic():
    class OtherProvider(SyntheticToolsActionProvider):
        pass
    with pytest.raises(ValueError, match="synthetic demo"):
        ToolsActionService(OtherProvider(lambda _: None), target_scope_fn=lambda _: None, enabled_bool=True, demo_bool=True)
