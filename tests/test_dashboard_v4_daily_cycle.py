"""Daily pod status projection: every vocabulary row, precedence, and the monthly path untouched."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.cycle import build_cycle_view_dict
from alpha.live.dashboard_v4.daily_cycle import STATUS_STYLE_DICT
from alpha.live.dashboard_v4.daily_demo import CORE5_IMPORT_STR, DEMO_NOW_TS, DemoDailyStore
from alpha.live.dashboard_v4.demo import build_demo_workspace_tuple
from alpha.live.dashboard_v4.next_commands import attach_next_command_list
from alpha.live.dashboard_v4.overview import build_overview_dict
from alpha.live.dashboard_v4.pod import build_pod_page_dict


# Fri 10-02 signal session, Mon 10-05 09:30 ET open (13:30 UTC), 16:00 ET close.
SIGNAL_TS = datetime(2026, 10, 2, 20, 0, tzinfo=UTC)
TARGET_TS = datetime(2026, 10, 5, 13, 30, tzinfo=UTC)
CLOSE_TS = datetime(2026, 10, 5, 20, 0, tzinfo=UTC)
NEXT_CUTOFF_TS = datetime(2026, 10, 6, 13, 28, tzinfo=UTC)


def _daily_dict(*, decision_status_str="submitted", vplan_status_str="submitted", **override_dict):
    daily_dict = {
        "status_str": "ok", "reason_str": "", "as_of_timestamp_str": None, "current_bool": True,
        "decision_dict": {"decision_plan_id_int": 7, "release_id_str": "core5.v1", "status_str": decision_status_str,
            "signal_timestamp_str": SIGNAL_TS.isoformat(), "submission_timestamp_str": (TARGET_TS - timedelta(minutes=8)).isoformat(),
            "target_execution_timestamp_str": TARGET_TS.isoformat(), "signal_session_date_str": "2026-10-02",
            "target_session_date_str": "2026-10-05", "updated_timestamp_str": TARGET_TS.isoformat()},
        "vplan_dict": {} if vplan_status_str is None else {"vplan_id_int": 9, "status_str": vplan_status_str,
            "submission_key_str": "daily:7", "submission_timestamp_str": (TARGET_TS - timedelta(minutes=8)).isoformat(),
            "target_execution_timestamp_str": TARGET_TS.isoformat(), "missing_ack_count_int": 0,
            "submit_ack_status_str": "complete", "sent_order_count_int": 3, "ack_count_int": 3, "fill_count_int": 3},
        "opening_dispatch_dict": {}, "execution_result_dict": {},
        "completion_request_list": [], "pod_alert_list": [], "execution_alert_list": [], "post_close_report_dict": {},
        "open_previous_list": [], "open_previous_truncated_bool": False, "last_closed_dict": {},
        "signal_session_dict": {"session_date_str": "2026-10-02", "decision_cutoff_timestamp_str": (TARGET_TS - timedelta(minutes=2)).isoformat(),
            "decision_plan_id_int": 7, "decision_status_str": decision_status_str, "capsule_halt_bool": False},
        "session_close_timestamp_str": CLOSE_TS.isoformat(), "next_open_timestamp_str": NEXT_CUTOFF_TS.isoformat(),
    }
    daily_dict.update(override_dict)
    return daily_dict


def _owed_after_close(daily_dict):
    """After the 10-05 close the 10-05 decision is owed by the 10-06 cutoff."""
    daily_dict["signal_session_dict"] = {"session_date_str": "2026-10-05", "decision_cutoff_timestamp_str": NEXT_CUTOFF_TS.isoformat(),
        "decision_plan_id_int": None, "decision_status_str": None, "capsule_halt_bool": False}
    return daily_dict


def _row_dict(daily_dict, now_ts, *, data_ready_bool=True):
    row_dict = {"mode_str": "live", "pod_id_str": "core5", "account_route_str": "U1", "db_status_str": "ok",
        "as_of_timestamp_str": now_ts.isoformat(), "release_id_str": "core5.v1",
        "norgate_snapshot_status_dict": {"status_str": "ready", "snapshot_fresh_for_cycle_bool": True, "snapshot_date_str": "2026-10-02"}
            if data_ready_bool else {"status_str": "waiting", "snapshot_fresh_for_cycle_bool": False,
                "snapshot_stale_past_alert_deadline_bool": False, "snapshot_date_str": "2026-10-02"}}
    if daily_dict is not None:
        daily_dict["as_of_timestamp_str"] = now_ts.isoformat()
        row_dict["daily_cycle_dict"] = daily_dict
    return row_dict


def _view(daily_dict, now_ts, **option_dict):
    return build_cycle_view_dict(_row_dict(daily_dict, now_ts, **option_dict), now_ts=now_ts)


def _closed(status_str, exception_list=()):
    daily_dict = _daily_dict(decision_status_str=status_str, vplan_status_str=status_str)
    daily_dict["execution_result_dict"] = {"status_str": status_str, "completed_timestamp_str": (CLOSE_TS + timedelta(minutes=2)).isoformat(),
        "exception_count_int": len(exception_list), "exception_list": list(exception_list)}
    return daily_dict


def _exception(reason_str):
    return {"asset_str": "SPY", "side_str": "BUY", "quantity_float": 3.0, "reason_str": reason_str, "expected_share_float": 10.0,
        "actual_share_float": 7.0, "target_weight_float": None, "intent_str": None}


def _dispatch(reason_str, attempted_int, never_int, deadline_ts=TARGET_TS - timedelta(minutes=2)):
    return {"reason_code_str": reason_str, "severity_str": "critical", "observed_timestamp_str": (TARGET_TS - timedelta(minutes=3)).isoformat(),
        "dispatch_deadline_timestamp_str": deadline_ts.isoformat(), "attempted_count_int": attempted_int, "never_dispatched_count_int": never_int}


def _previous_open(close_ts):
    return [{"decision_plan_id_int": 5, "status_str": "submitted", "target_execution_timestamp_str": (close_ts - timedelta(hours=6, minutes=30)).isoformat(),
        "target_session_date_str": close_ts.date().isoformat(), "session_close_timestamp_str": close_ts.isoformat()}]


CASE_LIST = [
    # Row 1-3: terminal decisions (data still pending, so the owed decision is not yet current).
    ("closed", lambda: _closed("completed"), CLOSE_TS + timedelta(minutes=10), False, "closed", "Closed"),
    ("exceptions", lambda: _closed("completed_with_exceptions", [_exception("actual_holding_differs_from_vplan"),
        _exception("actual_holding_differs_from_decision")]), CLOSE_TS + timedelta(minutes=10), False, "closed_with_exceptions", "Closed with exceptions (2)"),
    ("exceptions_not_sent", lambda: _closed("completed_with_exceptions", [_exception("decision_intent_not_dispatched")]),
        CLOSE_TS + timedelta(minutes=10), False, "closed_with_exceptions", "Not sent · closed with exceptions (1)"),
    ("superseded", lambda: _daily_dict(decision_status_str="superseded", vplan_status_str=None, current_bool=False),
        TARGET_TS - timedelta(hours=1), True, "superseded", "Superseded"),
    # Row 4.
    ("open_after_close", lambda: _daily_dict(), CLOSE_TS + timedelta(minutes=61), True, "open_after_close", "Open after close"),
    # Row 5.
    ("unconfirmed_submitting", lambda: _daily_dict(decision_status_str="vplan_ready", vplan_status_str="submitting"),
        TARGET_TS + timedelta(minutes=6), True, "dispatch_unconfirmed", "Dispatch unconfirmed: check broker, do not resend"),
    ("unconfirmed_attempted", lambda: _daily_dict(opening_dispatch_dict=_dispatch("opening_dispatch_incomplete", 3, 0)),
        TARGET_TS + timedelta(minutes=1), True, "dispatch_unconfirmed", "Dispatch unconfirmed: check broker, do not resend"),
    # Row 6.
    ("partly_sent", lambda: _daily_dict(opening_dispatch_dict=_dispatch("opening_dispatch_incomplete", 2, 1)),
        TARGET_TS + timedelta(minutes=1), True, "partly_sent", "Partly sent (2 of 3)"),
    # Row 7.
    ("not_sent_incomplete", lambda: _daily_dict(opening_dispatch_dict=_dispatch("opening_dispatch_incomplete", 0, 3)),
        TARGET_TS + timedelta(minutes=1), True, "not_sent", "Not sent"),
    ("not_sent_no_plan", lambda: _daily_dict(decision_status_str="planned", vplan_status_str=None),
        TARGET_TS + timedelta(minutes=6), True, "not_sent", "Not sent"),
    ("not_sent_ready", lambda: _daily_dict(decision_status_str="vplan_ready", vplan_status_str="ready"),
        TARGET_TS + timedelta(minutes=6), True, "not_sent", "Not sent"),
    ("not_sent_blocked", lambda: _daily_dict(decision_status_str="blocked", vplan_status_str="blocked"),
        TARGET_TS - timedelta(minutes=20), True, "not_sent", "Not sent"),
    # Row 8.
    ("retrying", lambda: _daily_dict(decision_status_str="vplan_ready", vplan_status_str="ready",
        opening_dispatch_dict=_dispatch("dispatch_retry_pending", 0, 3)), TARGET_TS - timedelta(minutes=5), True, "retrying_send", "Retrying send"),
    # Row 9.
    ("no_decision_owed", lambda: _owed_after_close(_closed("completed")), NEXT_CUTOFF_TS, True, "no_decision", "No decision for 2026-10-05"),
    ("no_decision_blocked", lambda: _daily_dict(decision_status_str="blocked", vplan_status_str=None),
        TARGET_TS - timedelta(minutes=2), True, "no_decision", "No decision for 2026-10-02"),
    # Row 10-13.
    ("closing", lambda: _daily_dict(), CLOSE_TS + timedelta(minutes=30), True, "closing", "Closing"),
    ("trading", lambda: _daily_dict(), TARGET_TS + timedelta(minutes=20), True, "trading", "Trading"),
    ("trading_completion", lambda: _daily_dict(completion_request_list=[{"asset_str": "DBC", "vplan_id_int": 9,
        "claimed_timestamp_str": TARGET_TS.isoformat(), "send_failed_bool": False}]), TARGET_TS + timedelta(minutes=20), True,
        "trading", "Trading · completing exit DBC (MKT)"),
    ("ready_no_plan", lambda: _daily_dict(decision_status_str="planned", vplan_status_str=None),
        TARGET_TS - timedelta(hours=3), True, "ready", "Ready · sends 09:28"),
    ("ready_plan", lambda: _daily_dict(decision_status_str="vplan_ready", vplan_status_str="ready"),
        TARGET_TS - timedelta(minutes=10), True, "ready", "Ready · sends 09:28"),
    ("deciding", lambda: _owed_after_close(_closed("completed")), CLOSE_TS + timedelta(minutes=40), True, "deciding", "Deciding"),
]


@pytest.mark.parametrize("name_str,factory_fn,now_ts,data_ready_bool,key_str,label_str", CASE_LIST,
    ids=[case_tuple[0] for case_tuple in CASE_LIST])
def test_every_vocabulary_row(name_str, factory_fn, now_ts, data_ready_bool, key_str, label_str):
    cycle_dict = _view(factory_fn(), now_ts, data_ready_bool=data_ready_bool)
    status_dict = cycle_dict["daily_status_dict"]
    assert (status_dict["key_str"], cycle_dict["now_str"]) == (key_str, label_str)
    tone_str, pill_str = STATUS_STYLE_DICT[key_str]
    assert (cycle_dict["tone_str"], cycle_dict["pill_str"]) == (tone_str, pill_str)
    assert [step_dict["label_str"] for step_dict in cycle_dict["step_dict_list"]] == [
        "Data", "Decide", "Plan", "Submit", "Fill", "Reconcile", "EOD"]


def test_all_thirteen_rows_are_covered():
    assert {case_tuple[4] for case_tuple in CASE_LIST} >= {key_str for key_str in STATUS_STYLE_DICT
        if key_str not in {"sending", "unknown"}}


def test_previous_cycle_open_after_close_is_never_hidden_by_a_newer_close():
    daily_dict = _closed("completed")
    daily_dict["open_previous_list"] = _previous_open(CLOSE_TS - timedelta(days=3))
    cycle_dict = _view(daily_dict, CLOSE_TS + timedelta(minutes=10), data_ready_bool=False)
    assert cycle_dict["daily_status_dict"]["key_str"] == "open_after_close"
    assert "Decision #5 for 2026-10-02" in cycle_dict["now_detail_str"]


def test_closed_gives_way_to_the_owed_decision_once_data_is_ready():
    daily_dict = _owed_after_close(_closed("completed"))
    assert _view(deepcopy(daily_dict), CLOSE_TS + timedelta(minutes=40), data_ready_bool=False)["now_str"] == "Closed"
    assert _view(deepcopy(daily_dict), CLOSE_TS + timedelta(minutes=40))["now_str"] == "Deciding"


def test_open_after_close_outranks_dispatch_states():
    daily_dict = _daily_dict(opening_dispatch_dict=_dispatch("opening_dispatch_incomplete", 0, 3))
    assert _view(daily_dict, CLOSE_TS + timedelta(minutes=61))["daily_status_dict"]["key_str"] == "open_after_close"


def test_unconfirmed_outranks_partly_and_not_sent():
    daily_dict = _daily_dict(decision_status_str="vplan_ready", vplan_status_str="submitting",
        opening_dispatch_dict=_dispatch("opening_dispatch_incomplete", 0, 3))
    assert _view(daily_dict, TARGET_TS + timedelta(minutes=6))["daily_status_dict"]["key_str"] == "dispatch_unconfirmed"


def test_retry_after_its_deadline_is_not_working_anymore():
    daily_dict = _daily_dict(decision_status_str="vplan_ready", vplan_status_str="ready",
        opening_dispatch_dict=_dispatch("dispatch_retry_pending", 0, 3))
    assert _view(deepcopy(daily_dict), TARGET_TS - timedelta(minutes=1))["daily_status_dict"]["key_str"] == "ready"
    assert _view(deepcopy(daily_dict), TARGET_TS + timedelta(minutes=5))["daily_status_dict"]["key_str"] == "not_sent"


def test_completed_with_exceptions_is_never_waiting_to_submit():
    daily_dict = _closed("completed_with_exceptions", [_exception("actual_holding_differs_from_vplan")])
    for now_ts in (CLOSE_TS + timedelta(minutes=5), CLOSE_TS + timedelta(days=1)):
        cycle_dict = _view(deepcopy(daily_dict), now_ts, data_ready_bool=False)
        assert all("Waiting to submit" not in step_dict["fact_str"] for step_dict in cycle_dict["step_dict_list"])
        assert cycle_dict["pill_str"] == "Needs review" and cycle_dict["tone_str"] == "amber"
        assert cycle_dict["step_dict_list"][5]["fact_str"] == "Closed · 1 exception"
    # Without the daily facts the monthly path would treat this status as an unsent plan.
    monthly_row_dict = _row_dict(None, CLOSE_TS + timedelta(minutes=5))
    monthly_row_dict.update(latest_decision_plan_id_int=7, latest_decision_plan_status_str="completed_with_exceptions",
        latest_vplan_id_int=9, latest_vplan_decision_plan_id_int=7, latest_vplan_status_str="completed_with_exceptions",
        latest_vplan_submission_timestamp_str=(CLOSE_TS + timedelta(hours=1)).isoformat(),
        latest_vplan_target_execution_timestamp_str=(CLOSE_TS + timedelta(hours=2)).isoformat())
    monthly_dict = build_cycle_view_dict(monthly_row_dict, now_ts=CLOSE_TS + timedelta(minutes=5))
    assert monthly_dict["step_dict_list"][3]["fact_str"] == "Waiting to submit"


def test_unconfirmed_is_red_and_never_working():
    for daily_dict, now_ts in (
            (_daily_dict(decision_status_str="vplan_ready", vplan_status_str="submitting"), TARGET_TS + timedelta(minutes=6)),
            (_daily_dict(opening_dispatch_dict=_dispatch("opening_dispatch_incomplete", 3, 0)), TARGET_TS + timedelta(minutes=20))):
        cycle_dict = _view(daily_dict, now_ts)
        assert cycle_dict["tone_str"] == "red" and cycle_dict["pill_str"] == "Action needed"
        assert cycle_dict["pill_str"] != "Working"
        submit_dict = cycle_dict["step_dict_list"][3]
        assert submit_dict["state_str"] == "Failed" and "do not resend" in submit_dict["detail_str"]


def test_last_close_exceptions_sub_line_only_while_a_new_cycle_runs():
    daily_dict = _daily_dict(last_closed_dict={"decision_plan_id_int": 6, "status_str": "completed_with_exceptions",
        "target_session_date_str": "2026-10-02", "completed_timestamp_str": SIGNAL_TS.isoformat(), "exception_count_int": 2})
    cycle_dict = _view(deepcopy(daily_dict), TARGET_TS + timedelta(minutes=20))
    assert cycle_dict["now_str"] == "Trading"
    assert cycle_dict["now_detail_str"].endswith("Last close 2026-10-02: 2 exceptions")
    closed_dict = _closed("completed")
    closed_dict["last_closed_dict"] = daily_dict["last_closed_dict"]
    assert "Last close" not in _view(closed_dict, CLOSE_TS + timedelta(minutes=5), data_ready_bool=False)["now_detail_str"]


def test_reconcile_step_carries_the_close():
    assert [_view(_daily_dict(), now_ts)["step_dict_list"][5]["fact_str"] for now_ts in (
        TARGET_TS + timedelta(minutes=20), CLOSE_TS + timedelta(minutes=1), CLOSE_TS + timedelta(minutes=61))] == [
        "Closes 16:00", "Closing", "Open after close"]
    assert _view(_closed("completed"), CLOSE_TS + timedelta(minutes=5), data_ready_bool=False)["step_dict_list"][5]["fact_str"] == "Closed from holdings"


def test_submit_counts_and_unverified_daily_facts():
    assert _view(_daily_dict(), TARGET_TS + timedelta(minutes=20))["step_dict_list"][3]["fact_str"] == "3 sent · 3 ack"
    for daily_dict in ({"status_str": "unknown"}, _daily_dict(as_of_timestamp_str=None, status_str="unknown")):
        cycle_dict = _view(daily_dict, TARGET_TS)
        assert cycle_dict["pill_str"] == "Unknown"
        assert [step_dict["state_str"] for step_dict in cycle_dict["step_dict_list"][1:6]] == ["Unknown"] * 5


def test_a_cached_early_read_cannot_prove_a_miss():
    daily_dict = _daily_dict(decision_status_str="vplan_ready", vplan_status_str="ready")
    row_dict = _row_dict(daily_dict, TARGET_TS + timedelta(minutes=6))
    daily_dict["as_of_timestamp_str"] = (TARGET_TS + timedelta(minutes=4)).isoformat()
    assert build_cycle_view_dict(row_dict, now_ts=TARGET_TS + timedelta(minutes=6))["daily_status_dict"]["key_str"] == "ready"


def test_same_row_without_daily_facts_keeps_the_monthly_output():
    now_ts = TARGET_TS + timedelta(minutes=3)
    row_dict = _row_dict(None, now_ts)
    row_dict.update(strategy_import_str="strategies.example:Monthly", latest_decision_plan_id_int=7,
        latest_decision_plan_status_str="submitted", latest_vplan_id_int=9, latest_vplan_decision_plan_id_int=7,
        latest_vplan_status_str="submitted", latest_vplan_submission_timestamp_str=(TARGET_TS - timedelta(minutes=8)).isoformat(),
        latest_vplan_target_execution_timestamp_str=TARGET_TS.isoformat(), broker_order_count_int=3, broker_ack_count_int=3,
        missing_ack_count_int=0, latest_submit_ack_status_str="complete", fill_count_int=3,
        eod_snapshot_dict={"status_str": "waiting", "expected_due_timestamp_str": (CLOSE_TS + timedelta(minutes=10)).isoformat()})
    monthly_dict = build_cycle_view_dict(deepcopy(row_dict), now_ts=now_ts)
    assert "daily_status_dict" not in monthly_dict
    assert (monthly_dict["pill_str"], monthly_dict["now_str"]) == ("Working", "Filling")
    assert [step_dict["state_str"] for step_dict in monthly_dict["step_dict_list"]] == [
        "Done", "Done", "Done", "Done", "Now", "Planned", "Planned"]
    row_dict["daily_cycle_dict"] = _daily_dict()
    row_dict["daily_cycle_dict"]["as_of_timestamp_str"] = now_ts.isoformat()
    assert build_cycle_view_dict(row_dict, now_ts=now_ts)["daily_status_dict"]["key_str"] == "trading"


def test_stale_daily_row_stays_unknown_and_modes_are_live_by_default():
    row_dict = _row_dict(_daily_dict(), TARGET_TS)
    row_dict["source_stale_bool"] = True
    cycle_dict = build_cycle_view_dict(row_dict, now_ts=TARGET_TS)
    assert cycle_dict["pill_str"] == "Unknown" and "daily_status_dict" not in cycle_dict
    other_dict = _row_dict(_daily_dict(), TARGET_TS)
    other_dict["mode_str"] = "paper"
    with pytest.raises(ValueError, match="LIVE only"):
        build_cycle_view_dict(other_dict, now_ts=TARGET_TS)
    assert build_cycle_view_dict(other_dict, now_ts=TARGET_TS,
        allowed_mode_set=frozenset({"live", "paper"}))["daily_status_dict"]["key_str"] == "trading"


# Wiring: Overview and Pod page, behind the daily gate only.

@pytest.fixture(autouse=True)
def healthy_host_disk(monkeypatch):
    monkeypatch.setattr("alpha.live.dashboard_v3.health.shutil.disk_usage",
        lambda path_str: SimpleNamespace(total=100, used=50, free=50))


@pytest.fixture()
def daily_wiring_tuple():
    """The demo workspace, its first pod marked CORE5 and fed one demo daily store's facts."""
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    daily_store_obj = DemoDailyStore()
    selected_dict = {"pod_id_str": "demo_daily_not_sent"}

    def daily_cycle_dict(row_dict, *, as_of_ts):
        target_obj = daily_store_obj.get_target_for_pod(selected_dict["pod_id_str"])
        return daily_store_obj.get_daily_cycle_dict({"pod_id_str": selected_dict["pod_id_str"],
            "account_route_str": target_obj.release_obj.account_route_str}, as_of_ts=as_of_ts)

    provider_obj.get_daily_cycle_dict = daily_cycle_dict
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["strategy_import_str"] = CORE5_IMPORT_STR
    try:
        yield workspace_dict, snapshot_obj, provider_obj, selected_dict
    finally:
        daily_store_obj.close()
        provider_obj.close()


def _overview(workspace_dict, provider_obj):
    return build_overview_dict(workspace_dict, None, provider_obj, as_of_ts=DEMO_NOW_TS, include_finance_bool=False)


def test_overview_daily_red_state_is_an_attention_item_with_safe_commands(daily_wiring_tuple):
    workspace_dict, _, provider_obj, _ = daily_wiring_tuple
    overview_dict = _overview(workspace_dict, provider_obj)
    pod_dict = overview_dict["pod_list"][0]
    assert (pod_dict["state_str"], pod_dict["pill_str"], pod_dict["now_str"]) == ("fail", "Action needed", "Not sent")
    attention_dict = next(item_dict for item_dict in overview_dict["attention_list"] if item_dict["pod_id_str"] == pod_dict["pod_id_str"])
    assert attention_dict["title_str"] == "Not sent" and attention_dict["daily_status_key_str"] == "not_sent"
    attach_next_command_list([attention_dict], workspace_dict, provider_obj, demo_bool=True, as_of_ts=DEMO_NOW_TS)
    assert attention_dict["next_command_list"] and not any(
        command_dict["orders_bool"] for command_dict in attention_dict["next_command_list"])
    # The other (monthly) demo pods keep their monthly projection.
    assert all("daily_cycle_dict" not in row_dict for row_dict in workspace_dict["summary_dict"]["pod_row_dict_list"][1:])


def test_routine_daily_steps_are_not_amber_but_a_monthly_row_is(daily_wiring_tuple):
    workspace_dict, _, provider_obj, selected_dict = daily_wiring_tuple
    selected_dict["pod_id_str"] = "demo_daily_trading"
    for row_dict in workspace_dict["summary_dict"]["pod_row_dict_list"][:3]:
        row_dict.update(next_action_str="post_execution_reconcile",
            required_action_dict={"severity_str": "yellow", "label_str": "Build DecisionPlan", "reason_str": "routine"})
    overview_dict = _overview(workspace_dict, provider_obj)
    daily_pod_dict = overview_dict["pod_list"][0]
    assert (daily_pod_dict["state_str"], daily_pod_dict["now_str"]) == ("now", "Trading · completing exit DBC (MKT)")
    attention_pod_set = {item_dict["pod_id_str"] for item_dict in overview_dict["attention_list"]}
    assert daily_pod_dict["pod_id_str"] not in attention_pod_set
    assert overview_dict["pod_list"][2]["pod_id_str"] in attention_pod_set


def test_daily_row_keeps_its_status_unless_the_daily_plan_counts_a_missing_ack(daily_wiring_tuple):
    workspace_dict, _, provider_obj, selected_dict = daily_wiring_tuple
    selected_dict["pod_id_str"] = "demo_daily_unconfirmed"
    workspace_dict["summary_dict"]["pod_row_dict_list"][0].update(latest_submit_ack_status_str="missing_critical")
    pod_dict = _overview(workspace_dict, provider_obj)["pod_list"][0]
    assert pod_dict["now_str"] == "Dispatch unconfirmed: check broker, do not resend"


def test_daily_row_without_a_provider_reader_is_unknown_never_monthly(daily_wiring_tuple):
    workspace_dict, _, provider_obj, _ = daily_wiring_tuple
    del provider_obj.get_daily_cycle_dict
    pod_dict = _overview(workspace_dict, provider_obj)["pod_list"][0]
    assert pod_dict["pill_str"] == "Unknown" and pod_dict["now_str"] == "Status unknown"


def test_pod_page_issue_title_and_detail_come_from_the_projection(daily_wiring_tuple):
    workspace_dict, _, provider_obj, _ = daily_wiring_tuple
    overview_dict = _overview(workspace_dict, provider_obj)
    pod_id_str = overview_dict["pod_list"][0]["pod_id_str"]
    source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)
    source_dict["pod_row_dict"]["daily_cycle_dict"] = provider_obj.get_daily_cycle_dict({}, as_of_ts=DEMO_NOW_TS)
    page_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert page_dict["issue_bool"] and page_dict["issue_title_str"] == "Not sent"
    assert page_dict["issue_detail_str"].startswith("The opening batch never reached the broker.")
    assert page_dict["attention_dict"]["daily_status_key_str"] == "not_sent"
    assert [step_dict["fact_str"] for step_dict in page_dict["step_list"]][3] == "Not sent"


def _weight_exception():
    return {"asset_str": "AAPL", "side_str": "BUY", "quantity_float": 12.0, "reason_str": "decision_intent_not_dispatched",
        "expected_share_float": None, "actual_share_float": 0.0, "target_weight_float": 0.1, "intent_str": "target_weight"}


def test_exceptions_table_lists_each_saved_exception_with_readable_reason():
    from alpha.live.dashboard_v4.pod import build_daily_exceptions_table_dict
    daily_dict = _closed("completed_with_exceptions", [_exception("actual_holding_differs_from_vplan"), _weight_exception(),
        _exception("some_future_reason")])
    daily_dict["completion_request_list"] = [{"asset_str": "DBC", "vplan_id_int": 9, "claimed_timestamp_str": None, "send_failed_bool": False}]
    table_dict = build_daily_exceptions_table_dict(daily_dict)
    assert table_dict["column_list"] == ["Symbol", "Missing trade", "Reason", "Expected", "Actual"]
    assert [row_dict["cell_list"] for row_dict in table_dict["row_list"]] == [
        ["SPY", "BUY 3", "Not filled / holding differs from plan", "10", "7"],
        ["AAPL", "BUY 12", "Not sent", "10% weight", "0"],
        ["SPY", "BUY 3", "Other exception", "10", "7"]]
    assert table_dict["note_str"] == "Same-day exit completion (MKT): DBC."


def test_exceptions_table_says_when_rows_are_capped_and_explains_empty_states():
    from alpha.live.dashboard_v4.pod import build_daily_exceptions_table_dict
    capped_dict = _closed("completed_with_exceptions", [_exception("decision_intent_not_dispatched")])
    capped_dict["execution_result_dict"]["exception_count_int"] = 250
    assert build_daily_exceptions_table_dict(capped_dict)["note_str"] == "Showing 1 of 250 exceptions."
    assert build_daily_exceptions_table_dict(_closed("completed"))["empty_str"] == "No exceptions: the day closed as planned."
    assert build_daily_exceptions_table_dict(_daily_dict())["empty_str"] == "No close-of-day result yet for this cycle."
    assert build_daily_exceptions_table_dict({"status_str": "unknown"})["empty_str"] == "Close-of-day exceptions could not be read."


def test_pod_page_shows_the_exceptions_tab_for_daily_pods_only(daily_wiring_tuple):
    workspace_dict, _, provider_obj, _ = daily_wiring_tuple
    overview_dict = _overview(workspace_dict, provider_obj)
    daily_pod_str, monthly_pod_str = (overview_dict["pod_list"][index_int]["pod_id_str"] for index_int in (0, 1))
    source_dict = provider_obj.get_pod_cycles_dict(daily_pod_str, as_of_ts=DEMO_NOW_TS)
    daily_dict = _closed("completed_with_exceptions", [_exception("actual_holding_differs_from_vplan")])
    daily_dict["as_of_timestamp_str"] = DEMO_NOW_TS.isoformat()
    source_dict["pod_row_dict"]["daily_cycle_dict"] = daily_dict
    page_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=daily_pod_str, as_of_ts=DEMO_NOW_TS)
    tab_dict = next(item_dict for item_dict in page_dict["tab_list"] if item_dict["key_str"] == "exceptions")
    assert (tab_dict["label_str"], tab_dict["count_int"]) == ("Exceptions", 1)
    assert next(step_dict for step_dict in page_dict["step_list"] if step_dict["label_str"] == "Reconcile")["tab_str"] == "exceptions"
    selected_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=daily_pod_str, as_of_ts=DEMO_NOW_TS, tab_str="exceptions")
    assert selected_dict["table_dict"]["row_list"][0]["cell_list"][0] == "SPY"
    monthly_source_dict = provider_obj.get_pod_cycles_dict(monthly_pod_str, as_of_ts=DEMO_NOW_TS)
    monthly_dict = build_pod_page_dict(overview_dict, monthly_source_dict, {}, pod_id_str=monthly_pod_str,
        as_of_ts=DEMO_NOW_TS, tab_str="exceptions")
    assert "exceptions" not in {item_dict["key_str"] for item_dict in monthly_dict["tab_list"]}
    assert monthly_dict["tab_str"] != "exceptions"
