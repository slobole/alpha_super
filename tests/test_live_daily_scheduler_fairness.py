"""Daily reconciliation retries must not starve monthly pod close snapshots."""
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from alpha.live import runner, scheduler_service, scheduler_utils
from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
from alpha.live.models import DecisionPlan
from alpha.live.state_store_v2 import LiveStateStore
from test_live_reference_compare import _build_release, _insert_vplan
from test_live_scheduler_service import _seed_post_execution_reconcile_truth

MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
MONTHLY_STRATEGY_LIST = [
    "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy",
    "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
]


def _time(date_str, hour_int, minute_int=0):
    return datetime.fromisoformat(date_str).replace(hour=hour_int, minute=minute_int, tzinfo=MARKET_TIMEZONE_OBJ)


def _configure_scheduler(monkeypatch, release_list):
    monkeypatch.setattr(scheduler_service, "_load_release_list_and_sync", lambda *_arg_tuple, **_kwarg_dict: release_list)
    # Missing new data prevents a new decision but does not prevent a due account snapshot.
    monkeypatch.setattr(scheduler_utils, "is_snapshot_mode_enabled_bool", lambda: True)
    monkeypatch.setattr(scheduler_utils, "load_latest_snapshot_session_label_ts", lambda *_arg_tuple: None)


@pytest.mark.parametrize("daily_strategy_str", [CORE5_STRATEGY_IMPORT_STR,
    "strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil"])
@pytest.mark.parametrize("monthly_strategy_str", MONTHLY_STRATEGY_LIST)
def test_postclose_daily_retry_allows_other_monthly_eod(tmp_path, monkeypatch, daily_strategy_str, monthly_strategy_str):
    store_obj = LiveStateStore(str(tmp_path / "fairness.sqlite3"))
    daily_obj = _build_release(strategy_import_str=daily_strategy_str, pod_id_str="daily")
    monthly_obj = replace(_build_release(strategy_import_str=monthly_strategy_str, pod_id_str="monthly"),
        account_route_str="DU_MONTHLY", signal_clock_str="month_end_snapshot_ready", execution_policy_str="next_month_first_open")
    for release_obj in (daily_obj, monthly_obj):
        store_obj.upsert_release(release_obj)
    store_obj.insert_decision_plan(DecisionPlan(
        release_id_str=daily_obj.release_id_str, user_id_str=daily_obj.user_id_str,
        pod_id_str=daily_obj.pod_id_str, account_route_str=daily_obj.account_route_str,
        signal_timestamp_ts=_time("2024-01-30", 16), submission_timestamp_ts=_time("2024-01-31", 9, 24),
        target_execution_timestamp_ts=_time("2024-01-31", 9, 30), execution_policy_str="next_open_moo",
        decision_base_position_map={}, snapshot_metadata_dict={}, strategy_state_dict={}, status_str="blocked"))
    _configure_scheduler(monkeypatch, [monthly_obj])
    baseline_obj = scheduler_service.get_scheduler_decision(store_obj, _time("2024-01-31", 16, 10), "unused", "paper")
    assert baseline_obj.next_phase_str == "eod_snapshot"
    _configure_scheduler(monkeypatch, [daily_obj, monthly_obj])
    mixed_obj = scheduler_service.get_scheduler_decision(store_obj, _time("2024-01-31", 16, 10), "unused", "paper")
    assert mixed_obj.next_phase_str == "eod_snapshot"
    assert mixed_obj.related_pod_id_list == baseline_obj.related_pod_id_list == ["monthly"]
    # The same daily pod keeps polling, and cannot take an EOD snapshot before settlement.
    _configure_scheduler(monkeypatch, [daily_obj])
    for as_of_ts in (_time("2024-01-31", 12), _time("2024-01-31", 16, 10)):
        pending_obj = scheduler_service.get_scheduler_decision(store_obj, as_of_ts, "unused", "paper")
        assert pending_obj.next_phase_str == "post_execution_reconcile"
        assert pending_obj.due_now_bool and pending_obj.active_poll_bool


@pytest.mark.parametrize("monthly_strategy_str", MONTHLY_STRATEGY_LIST)
@pytest.mark.parametrize("order_status_str,parked_bool", [("Cancelled", True), ("Submitted", False)])
def test_monthly_unfilled_orders_keep_legacy_park_or_poll(tmp_path, monkeypatch, monthly_strategy_str, order_status_str, parked_bool):
    store_obj = LiveStateStore(str(tmp_path / "monthly.sqlite3"))
    release_obj = replace(_build_release(strategy_import_str=monthly_strategy_str),
        signal_clock_str="month_end_snapshot_ready", execution_policy_str="next_month_first_open")
    store_obj.upsert_release(release_obj)
    plan_obj = _insert_vplan(store_obj, release_obj, _time("2024-01-31", 16), _time("2024-02-01", 9, 30), 5000.0)
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitted")
    store_obj.mark_decision_plan_status(plan_obj.decision_plan_id_int, "submitted")
    _seed_post_execution_reconcile_truth(store_obj, plan_obj, 0.0, order_status_str)
    _configure_scheduler(monkeypatch, [release_obj])
    result_obj = scheduler_service.get_scheduler_decision(store_obj, _time("2024-02-01", 16, 10), "unused", "paper")
    assert runner.is_vplan_execution_exception_parked(store_obj, plan_obj) is parked_bool
    assert result_obj.next_phase_str == ("manual_review_pending" if parked_bool else "post_execution_reconcile")
    assert result_obj.due_now_bool is (not parked_bool)
    assert result_obj.reason_code_str == ("execution_exception_parked" if parked_bool else "ready_to_reconcile")
    assert store_obj.get_vplan_by_id(plan_obj.vplan_id_int).status_str == "submitted"
    with store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT COUNT(*) FROM mr_capsule_execution_alert").fetchone()[0] == 0
