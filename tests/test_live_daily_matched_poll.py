"""Daily pods: a clean broker read is repeated on the slow interval, not every active poll.

The scheduler decision is a pure function of the saved state. A process restart
sees the same saved read, and a failed read saves nothing newer, so both land in
the "due again" case below. Monthly pods keep the legacy formula untouched.
"""
from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from alpha.live import runner, scheduler_service, scheduler_utils
from alpha.live.core5_adapter import CORE5_CONTRACT_STR, CORE5_STRATEGY_IMPORT_STR
from alpha.live.dv2_ind_adapter import DV2_IND_STRATEGY_IMPORT_STR
from alpha.live.models import BrokerSnapshot, DecisionPlan
from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE
from alpha.live.scheduler_service import (
    SchedulerDecision,
    _build_related_pod_status_dict_list,
    _build_wait_operator_message_spec_list,
    get_scheduler_decision,
)
from alpha.live.state_store_v2 import LiveStateStore
from test_live_reference_compare import _build_release, _insert_vplan

MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
NDX_STRATEGY_STR = "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy"


def _time(date_str, hour_int, minute_int=0, second_int=0):
    return datetime.fromisoformat(date_str).replace(
        hour=hour_int, minute=minute_int, second=second_int, tzinfo=MARKET_TIMEZONE_OBJ)


def _configure_scheduler(monkeypatch, release_list):
    monkeypatch.setattr(scheduler_service, "_load_release_list_and_sync", lambda *_arg_tuple, **_kwarg_dict: release_list)
    monkeypatch.setattr(scheduler_utils, "is_snapshot_mode_enabled_bool", lambda: True)
    monkeypatch.setattr(scheduler_utils, "load_latest_snapshot_session_label_ts", lambda *_arg_tuple: None)


def _store_with_sent_plan(tmp_path, monkeypatch, *, strategy_str=CORE5_STRATEGY_IMPORT_STR,
        date_str="2024-01-31", signal_date_str="2024-01-30"):
    """A daily pod whose plan (AAPL 10 shares) was sent for the open of ``date_str``."""
    store_obj = LiveStateStore(str(tmp_path / "daily.sqlite3"))
    release_obj = _build_release(strategy_import_str=strategy_str, pod_id_str="daily")
    store_obj.upsert_release(release_obj)
    plan_obj = _insert_vplan(store_obj, release_obj, _time(signal_date_str, 16), _time(date_str, 9, 30), 5000.0)
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitted")
    store_obj.mark_decision_plan_status(plan_obj.decision_plan_id_int, "submitted")
    _configure_scheduler(monkeypatch, [release_obj])
    return store_obj, release_obj, plan_obj


def _read(store_obj, release_obj, read_ts, position_map=None, open_order_list=()):
    """Save what a daily broker read found at ``read_ts``."""
    store_obj.upsert_broker_snapshot_cache(BrokerSnapshot(
        account_route_str=release_obj.account_route_str, snapshot_timestamp_ts=read_ts,
        cash_float=4000.0, total_value_float=5000.0, net_liq_float=5000.0,
        position_amount_map={"AAPL": 10.0} if position_map is None else dict(position_map),
        open_order_id_list=list(open_order_list)))


@pytest.mark.parametrize("strategy_str", [CORE5_STRATEGY_IMPORT_STR, DV2_IND_STRATEGY_IMPORT_STR, *MR_CAPSULE_STRATEGY_IMPORT_TUPLE])
def test_clean_read_is_repeated_after_the_slow_interval(tmp_path, monkeypatch, strategy_str):
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch, strategy_str=strategy_str)
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35))

    decision_obj = get_scheduler_decision(store_obj, _time("2024-01-31", 9, 36), "unused", "paper")

    assert decision_obj.next_phase_str == "post_execution_reconcile"
    assert decision_obj.due_now_bool is False
    # The loop still wakes on the active poll; only the broker read waits.
    assert decision_obj.active_poll_bool is True
    assert decision_obj.reason_code_str == "daily_holdings_match"
    assert decision_obj.next_due_timestamp_ts == _time("2024-01-31", 9, 50)


def test_slow_interval_elapsed_reads_again_now(tmp_path, monkeypatch):
    # Also the restart case and the failed-read case: nothing newer is saved.
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35))

    decision_obj = get_scheduler_decision(store_obj, _time("2024-01-31", 9, 50), "unused", "paper")

    assert decision_obj.due_now_bool is True and decision_obj.active_poll_bool is True
    assert decision_obj.reason_code_str == "ready_to_reconcile"


@pytest.mark.parametrize("read_tuple,position_map,open_order_list,as_of_tuple,label_str", [
    ((9, 28), None, (), (9, 36), "the plan-build read before the open does not count"),
    ((9, 35), {"AAPL": 4.0}, (), (9, 36), "holdings differ: partial fill or manual sale"),
    ((9, 35), {"AAPL": 10.0, "MSFT": 3.0}, (), (9, 36), "an extra holding"),
    ((9, 35), None, ("1184220455",), (9, 36), "an order is open at the account"),
    ((15, 31), None, (), (15, 32), "the last half hour before the close"),
])
def test_only_a_clean_read_slows_the_poll(tmp_path, monkeypatch, read_tuple, position_map, open_order_list,
        as_of_tuple, label_str):
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    _read(store_obj, release_obj, _time("2024-01-31", *read_tuple), position_map, open_order_list)

    decision_obj = get_scheduler_decision(store_obj, _time("2024-01-31", *as_of_tuple), "unused", "paper")

    assert decision_obj.due_now_bool is True, label_str
    assert decision_obj.reason_code_str == "ready_to_reconcile", label_str


def test_shared_account_keeps_the_active_poll(tmp_path, monkeypatch):
    # Another enabled pod on the same account may have saved the read with a
    # this-client-only open-order list, so the read is never trusted to slow.
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    store_obj.upsert_release(replace(_build_release(strategy_import_str=NDX_STRATEGY_STR, pod_id_str="monthly"),
        signal_clock_str="month_end_snapshot_ready", execution_policy_str="next_month_first_open"))
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35))

    decision_obj = get_scheduler_decision(store_obj, _time("2024-01-31", 9, 36), "unused", "paper")

    assert decision_obj.due_now_bool is True and decision_obj.reason_code_str == "ready_to_reconcile"


def test_unreadable_saved_read_keeps_the_active_poll(tmp_path, monkeypatch):
    # Corrupt saved state must neither slow the poll nor stop the decision.
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "get_status_summary", lambda **_kwarg_dict: _pod_status_summary_dict(release_obj))
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35))
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE broker_snapshot_cache SET position_json_str='{bad' WHERE account_route_str=?",
            (release_obj.account_route_str,))

    decision_obj = get_scheduler_decision(store_obj, _time("2024-01-31", 9, 36), "unused", "paper")
    pod_status_dict_list, spec_list = _wait_spec_list(store_obj, release_obj, _time("2024-01-31", 9, 36), _time("2024-01-31", 9, 36))

    assert decision_obj.due_now_bool is True and decision_obj.reason_code_str == "ready_to_reconcile"
    assert pod_status_dict_list[0]["daily_check_dict"] == {}
    assert spec_list[0]["field_map_dict"]["holdings"] == "not read yet"


def test_future_dated_saved_read_keeps_the_active_poll(tmp_path, monkeypatch):
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    _read(store_obj, release_obj, _time("2024-01-31", 14))

    decision_obj = get_scheduler_decision(store_obj, _time("2024-01-31", 9, 36), "unused", "paper")

    assert decision_obj.due_now_bool is True and decision_obj.reason_code_str == "ready_to_reconcile"


def test_daily_wait_line_next_read_comes_from_the_read_itself(tmp_path, monkeypatch):
    # Right after a read the decision in hand predates it; the line must not say "next = now".
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "get_status_summary", lambda **_kwarg_dict: _pod_status_summary_dict(release_obj))
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35, 2))

    _pod_status_dict_list, spec_list = _wait_spec_list(store_obj, release_obj, _time("2024-01-31", 9, 35, 2), _time("2024-01-31", 9, 35, 2))
    assert spec_list[0]["field_map_dict"]["next"] == "2024-01-31 14:50:02 UTC"

    # Once the slow window has ended the next read is now, whatever the read says.
    _read(store_obj, release_obj, _time("2024-01-31", 15, 40))
    _pod_status_dict_list, spec_list = _wait_spec_list(store_obj, release_obj, _time("2024-01-31", 15, 41), _time("2024-01-31", 15, 41))
    assert spec_list[0]["field_map_dict"]["next"] == "2024-01-31 20:41:00 UTC"


def test_close_read_is_never_delayed_by_a_clean_read(tmp_path, monkeypatch):
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    _read(store_obj, release_obj, _time("2024-01-31", 15, 59, 30))

    decision_obj = get_scheduler_decision(store_obj, _time("2024-01-31", 16), "unused", "paper")

    assert decision_obj.due_now_bool is True and decision_obj.next_phase_str == "post_execution_reconcile"


@pytest.mark.parametrize("vplan_status_str", ["ready", "submitting", "parked"])
def test_unconfirmed_plan_keeps_the_active_poll(tmp_path, monkeypatch, vplan_status_str):
    store_obj, release_obj, plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, vplan_status_str)
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35))

    decision_obj = get_scheduler_decision(store_obj, _time("2024-01-31", 9, 36), "unused", "paper")

    assert decision_obj.due_now_bool is True


def test_early_close_session_stops_the_slow_interval_before_its_own_close(tmp_path, monkeypatch):
    # 2024-07-03 closes at 13:00 ET, so the slow interval ends at 12:30.
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(
        tmp_path, monkeypatch, date_str="2024-07-03", signal_date_str="2024-07-02")
    _read(store_obj, release_obj, _time("2024-07-03", 12, 20))

    assert get_scheduler_decision(store_obj, _time("2024-07-03", 12, 21), "unused", "paper").next_due_timestamp_ts == _time("2024-07-03", 12, 30)
    assert get_scheduler_decision(store_obj, _time("2024-07-03", 12, 31), "unused", "paper").due_now_bool is True


def test_no_order_day_without_a_plan_slows_when_holdings_equal_the_intent(tmp_path, monkeypatch):
    store_obj = LiveStateStore(str(tmp_path / "daily.sqlite3"))
    release_obj = _build_release(strategy_import_str=DV2_IND_STRATEGY_IMPORT_STR, pod_id_str="daily")
    store_obj.upsert_release(release_obj)
    store_obj.insert_decision_plan(DecisionPlan(
        release_id_str=release_obj.release_id_str, user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str, account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=_time("2024-01-30", 16), submission_timestamp_ts=_time("2024-01-31", 9, 28),
        target_execution_timestamp_ts=_time("2024-01-31", 9, 30), execution_policy_str="next_open_moo",
        decision_base_position_map={"XLE": 25.0}, snapshot_metadata_dict={}, strategy_state_dict={}))
    _configure_scheduler(monkeypatch, [release_obj])

    _read(store_obj, release_obj, _time("2024-01-31", 9, 35), {"XLE": 25.0})
    decision_obj = get_scheduler_decision(store_obj, _time("2024-01-31", 9, 36), "unused", "paper")
    assert decision_obj.due_now_bool is False
    assert decision_obj.reason_code_str == "daily_holdings_match"
    assert decision_obj.next_due_timestamp_ts == _time("2024-01-31", 9, 50)

    # Held positions are the intent here; "no trades" is not "cash".
    _read(store_obj, release_obj, _time("2024-01-31", 9, 50), {"XLE": 20.0})
    assert get_scheduler_decision(store_obj, _time("2024-01-31", 9, 51), "unused", "paper").due_now_bool is True


def test_nothing_attempted_dispatch_record_is_judged_by_the_decision_intent(tmp_path, monkeypatch):
    # The plan says AAPL 10 and the account holds AAPL 10, but the dispatch
    # record shows nothing was attempted: the close judges by the decision's
    # weight intent and records an exception, so the scheduler must not slow.
    import json

    store_obj, release_obj, plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str=? WHERE decision_plan_id_int=?",
            (json.dumps({"opening_dispatch_result_dict": {"reason_code_str": "opening_dispatch_incomplete",
                "attempted_request_key_list": [], "never_dispatched_request_key_list": ["vplan:1:AAPL:1"]}}),
                plan_obj.decision_plan_id_int))
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35))

    decision_obj = get_scheduler_decision(store_obj, _time("2024-01-31", 9, 36), "unused", "paper")

    assert decision_obj.due_now_bool is True and decision_obj.reason_code_str == "ready_to_reconcile"


def test_unjudgeable_read_keeps_the_active_poll(tmp_path, monkeypatch):
    # CORE5 metadata that names the sizing contract without its share map raises
    # inside the close's holdings test; the scheduler must not fail on it.
    store_obj = LiveStateStore(str(tmp_path / "daily.sqlite3"))
    release_obj = _build_release(strategy_import_str=CORE5_STRATEGY_IMPORT_STR, pod_id_str="daily")
    store_obj.upsert_release(release_obj)
    store_obj.insert_decision_plan(DecisionPlan(
        release_id_str=release_obj.release_id_str, user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str, account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=_time("2024-01-30", 16), submission_timestamp_ts=_time("2024-01-31", 9, 28),
        target_execution_timestamp_ts=_time("2024-01-31", 9, 30), execution_policy_str="next_open_moo",
        decision_base_position_map={}, snapshot_metadata_dict={"sizing_contract_str": CORE5_CONTRACT_STR},
        strategy_state_dict={}))
    _configure_scheduler(monkeypatch, [release_obj])
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35), {})

    decision_obj = get_scheduler_decision(store_obj, _time("2024-01-31", 9, 36), "unused", "paper")

    assert decision_obj.due_now_bool is True and decision_obj.reason_code_str == "ready_to_reconcile"


def test_monthly_pod_keeps_the_legacy_formula(tmp_path, monkeypatch):
    store_obj = LiveStateStore(str(tmp_path / "monthly.sqlite3"))
    release_obj = replace(_build_release(strategy_import_str=NDX_STRATEGY_STR, pod_id_str="monthly"),
        signal_clock_str="month_end_snapshot_ready", execution_policy_str="next_month_first_open")
    store_obj.upsert_release(release_obj)
    plan_obj = _insert_vplan(store_obj, release_obj, _time("2024-01-31", 16), _time("2024-02-01", 9, 30), 5000.0)
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitted")
    store_obj.mark_decision_plan_status(plan_obj.decision_plan_id_int, "submitted")
    _configure_scheduler(monkeypatch, [release_obj])
    _read(store_obj, release_obj, _time("2024-02-01", 9, 35))

    before_obj = get_scheduler_decision(store_obj, _time("2024-02-01", 9, 33), "unused", "paper")
    after_obj = get_scheduler_decision(store_obj, _time("2024-02-01", 9, 36), "unused", "paper")

    assert before_obj.reason_code_str == "waiting_for_post_execution_reconcile"
    assert before_obj.next_due_timestamp_ts == _time("2024-02-01", 9, 35)
    assert after_obj.due_now_bool is True and after_obj.reason_code_str == "ready_to_reconcile"


def test_session_read_schedule(tmp_path, monkeypatch):
    """One clean session: reads at 09:35 then every 15 minutes, 15:30, then the active poll to the close."""
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    read_list = []
    as_of_ts = _time("2024-01-31", 9, 30)
    while as_of_ts < _time("2024-01-31", 16):
        decision_obj = get_scheduler_decision(store_obj, as_of_ts, "unused", "paper")
        assert decision_obj.next_phase_str == "post_execution_reconcile" and decision_obj.active_poll_bool
        if decision_obj.due_now_bool:
            read_list.append(as_of_ts)
            _read(store_obj, release_obj, as_of_ts)
        as_of_ts += timedelta(seconds=30)

    slow_list = [_time("2024-01-31", 9, 35) + timedelta(minutes=15 * index_int) for index_int in range(24)]
    assert read_list[:25] == [*slow_list, _time("2024-01-31", 15, 30)]
    assert read_list[25:] == [_time("2024-01-31", 15, 30, 30) + timedelta(seconds=30 * index_int) for index_int in range(59)]
    assert len(read_list) == 84  # the active poll alone would read 770 times


def _pod_status_summary_dict(release_obj):
    return {"pod_status_dict_list": [{
        "release_id_str": release_obj.release_id_str, "pod_id_str": release_obj.pod_id_str,
        "account_route_str": release_obj.account_route_str, "next_action_str": "post_execution_reconcile",
        "latest_vplan_status_str": "submitted", "latest_submission_timestamp_str": None,
        "exception_count_int": 0, "broker_ack_row_dict_list": []}]}


def _wait_spec_list(store_obj, release_obj, as_of_ts, next_due_ts):
    pod_status_dict_list = _build_related_pod_status_dict_list(store_obj, as_of_ts, "unused", "paper", [release_obj.pod_id_str])
    decision_obj = SchedulerDecision(as_of_timestamp_ts=as_of_ts, env_mode_str="paper", due_now_bool=False,
        active_poll_bool=True, next_phase_str="post_execution_reconcile", reason_code_str="daily_holdings_match",
        next_due_timestamp_ts=next_due_ts, related_pod_id_list=[release_obj.pod_id_str])
    return pod_status_dict_list, _build_wait_operator_message_spec_list(decision_obj,
        pod_status_dict_list=pod_status_dict_list, execution_report_dict_list=[], broker_order_snapshot_dict_list=[])


def test_daily_wait_line_replaces_the_fill_lines(tmp_path, monkeypatch):
    store_obj, release_obj, plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "get_status_summary", lambda **_kwarg_dict: _pod_status_summary_dict(release_obj))
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35))

    pod_status_dict_list, spec_list = _wait_spec_list(store_obj, release_obj, _time("2024-01-31", 9, 36), _time("2024-01-31", 9, 50))

    daily_check_dict = dict(pod_status_dict_list[0]["daily_check_dict"])
    assert datetime.fromisoformat(daily_check_dict.pop("checked_timestamp_str")) == _time("2024-01-31", 9, 35)
    assert datetime.fromisoformat(daily_check_dict.pop("next_read_timestamp_str")) == _time("2024-01-31", 9, 50)
    assert daily_check_dict == {"open_order_count_int": 0, "missing_list": [], "match_bool": True}
    assert [spec_dict["phase_action_str"] for spec_dict in spec_list] == ["daily.wait"]
    assert spec_list[0]["level_str"] == "INFO"
    assert spec_list[0]["field_map_dict"] == {
        "pod": "daily", "account": release_obj.account_route_str, "vplan": None, "holdings": "match", "missing": None,
        "open_orders": 0, "checked": "2024-01-31 14:35:00 UTC", "next": "2024-01-31 14:50:00 UTC"}


def test_daily_wait_line_names_what_is_missing(tmp_path, monkeypatch):
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "get_status_summary", lambda **_kwarg_dict: _pod_status_summary_dict(release_obj))
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35), {"AAPL": 4.0}, ("1184220455",))

    pod_status_dict_list, spec_list = _wait_spec_list(store_obj, release_obj, _time("2024-01-31", 9, 36), _time("2024-01-31", 9, 36))

    assert pod_status_dict_list[0]["daily_check_dict"]["match_bool"] is False
    assert spec_list[0]["level_str"] == "WARN"
    field_dict = spec_list[0]["field_map_dict"]
    assert (field_dict["holdings"], field_dict["missing"], field_dict["open_orders"]) == ("1 missing", "AAPL BUY 6", 1)


def test_daily_wait_line_warns_on_an_open_order_with_matching_holdings(tmp_path, monkeypatch):
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "get_status_summary", lambda **_kwarg_dict: _pod_status_summary_dict(release_obj))
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35), None, ("1184220455",))

    pod_status_dict_list, spec_list = _wait_spec_list(store_obj, release_obj, _time("2024-01-31", 9, 36), _time("2024-01-31", 9, 36))

    assert pod_status_dict_list[0]["daily_check_dict"]["match_bool"] is False
    assert spec_list[0]["level_str"] == "WARN"
    field_dict = spec_list[0]["field_map_dict"]
    assert (field_dict["holdings"], field_dict["missing"], field_dict["open_orders"]) == ("match", None, 1)


def test_unsent_intent_is_named_without_a_quantity(tmp_path, monkeypatch):
    # A blocked cycle never built a plan: the weight intent has no share count.
    store_obj = LiveStateStore(str(tmp_path / "daily.sqlite3"))
    release_obj = _build_release(strategy_import_str=CORE5_STRATEGY_IMPORT_STR, pod_id_str="daily")
    store_obj.upsert_release(release_obj)
    store_obj.insert_decision_plan(DecisionPlan(
        release_id_str=release_obj.release_id_str, user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str, account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=_time("2024-01-30", 16), submission_timestamp_ts=_time("2024-01-31", 9, 28),
        target_execution_timestamp_ts=_time("2024-01-31", 9, 30), execution_policy_str="next_open_moo",
        decision_base_position_map={}, snapshot_metadata_dict={}, strategy_state_dict={},
        entry_target_weight_map_dict={"AAPL": 0.2}, entry_priority_list=["AAPL"], status_str="blocked"))
    _configure_scheduler(monkeypatch, [release_obj])
    monkeypatch.setattr(runner, "get_status_summary", lambda **_kwarg_dict: _pod_status_summary_dict(release_obj))
    _read(store_obj, release_obj, _time("2024-01-31", 9, 35), {})

    assert get_scheduler_decision(store_obj, _time("2024-01-31", 9, 36), "unused", "paper").due_now_bool is True
    pod_status_dict_list, spec_list = _wait_spec_list(store_obj, release_obj, _time("2024-01-31", 9, 36), _time("2024-01-31", 9, 36))
    assert pod_status_dict_list[0]["daily_check_dict"]["missing_list"] == ["AAPL BUY"]
    assert spec_list[0]["level_str"] == "WARN"
    assert (spec_list[0]["field_map_dict"]["holdings"], spec_list[0]["field_map_dict"]["missing"]) == ("1 missing", "AAPL BUY")


def test_daily_wait_line_before_the_first_read(tmp_path, monkeypatch):
    store_obj, release_obj, _plan_obj = _store_with_sent_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "get_status_summary", lambda **_kwarg_dict: _pod_status_summary_dict(release_obj))
    _read(store_obj, release_obj, _time("2024-01-31", 9, 28))

    pod_status_dict_list, spec_list = _wait_spec_list(store_obj, release_obj, _time("2024-01-31", 9, 31), _time("2024-01-31", 9, 35))

    assert pod_status_dict_list[0]["daily_check_dict"] == {}
    assert spec_list[0]["level_str"] == "INFO"
    field_dict = spec_list[0]["field_map_dict"]
    assert (field_dict["holdings"], field_dict["open_orders"], field_dict["checked"]) == ("not read yet", None, None)


def test_monthly_pod_status_has_no_daily_check(tmp_path, monkeypatch):
    store_obj = LiveStateStore(str(tmp_path / "monthly.sqlite3"))
    release_obj = replace(_build_release(strategy_import_str=NDX_STRATEGY_STR, pod_id_str="monthly"),
        signal_clock_str="month_end_snapshot_ready", execution_policy_str="next_month_first_open")
    store_obj.upsert_release(release_obj)
    monkeypatch.setattr(runner, "get_status_summary", lambda **_kwarg_dict: _pod_status_summary_dict(release_obj))

    pod_status_dict_list = _build_related_pod_status_dict_list(store_obj, _time("2024-02-01", 9, 36), "unused", "paper", ["monthly"])

    assert "daily_check_dict" not in pod_status_dict_list[0]
