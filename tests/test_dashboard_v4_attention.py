"""Attention items: held Pods, manual-submit deadlines, scheduler error reasons."""

from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.overview import build_overview_dict
from alpha.live.dashboard_v4.scheduler_view import scheduler_hold_dict, scheduler_issue_dict
from test_dashboard_v4_cycle import pod_row_dict, _complete_evidence_dict


NOW_TS = datetime.fromisoformat("2026-09-18T13:32:00+00:00")


class _Provider:
    def __init__(self, scheduler_dict):
        self.scheduler_dict = scheduler_dict

    def get_scheduler_status_dict(self, pod_id_str, *, as_of_ts):
        return dict(self.scheduler_dict)


def _overview_dict(row_dict_list, scheduler_dict, monkeypatch, *, now_ts=NOW_TS):
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_health_rollup",
        lambda *args, **kwargs: SimpleNamespace(severity_str="green"))
    workspace_dict = {"client_dict": {"accounts": []}, "operations_account_list": [
        {"pod_id": row_dict["pod_id_str"], "account_route": row_dict["account_route_str"], "display_name": row_dict["pod_id_str"]}
        for row_dict in row_dict_list],
        "summary_dict": {"as_of_timestamp_str": now_ts.isoformat(), "pod_row_dict_list": row_dict_list}}
    return build_overview_dict(workspace_dict, None, _Provider(scheduler_dict), as_of_ts=now_ts, include_finance_bool=False)


def _healthy_row_dict(pod_row_dict, pod_id_str="test_daily", account_str="TEST_ACCOUNT"):
    row_dict = deepcopy(pod_row_dict)
    row_dict.update(pod_id_str=pod_id_str, account_route_str=account_str, as_of_timestamp_str=NOW_TS.isoformat(),
        next_action_str="wait", required_action_dict={"severity_str": "green", "label_str": "No action"},
        latest_decision_plan_status_str="completed", latest_vplan_status_str="completed",
        latest_reconciliation_timestamp_str="2026-09-18T13:31:00+00:00")
    row_dict["cycle_evidence_dict"] = _complete_evidence_dict(row_dict)
    return row_dict


def _alive_dict(state_str="sleeping", reason_str="", **extra_dict):
    return {"state_str": state_str, "alive_bool": True, "reason_code_str": reason_str,
        "last_seen_timestamp_str": (NOW_TS - timedelta(minutes=2)).isoformat(),
        "promised_wake_timestamp_str": (NOW_TS + timedelta(minutes=30)).isoformat(), **extra_dict}


@pytest.mark.parametrize("reason_str,phrase_str", [
    ("execution_exception_parked", "An execution problem is parked for your review."),
    ("mr_capsule_eod_snapshot_untrusted", "The saved end-of-day snapshot is not trusted."),
    ("some_new_reason", "The scheduler holds this Pod."),
])
def test_held_pod_becomes_an_amber_attention_item(pod_row_dict, monkeypatch, reason_str, phrase_str):
    overview_dict = _overview_dict([_healthy_row_dict(pod_row_dict)], _alive_dict("holding", reason_str), monkeypatch)
    item_dict, = overview_dict["attention_list"]
    assert item_dict["state_str"] == "late" and item_dict["title_str"] == "Waiting for you"
    assert item_dict["detail_str"] == phrase_str + " It will not retry by itself."
    assert item_dict["console_bool"] is True
    pod_dict, = overview_dict["pod_list"]
    assert (pod_dict["pill_str"], pod_dict["now_str"], pod_dict["next_str"]) == ("Needs review", "Waiting for you", "Review saved evidence")


def test_hold_requires_a_live_scheduler(pod_row_dict, monkeypatch):
    overview_dict = _overview_dict([_healthy_row_dict(pod_row_dict)],
        {**_alive_dict("holding", "execution_exception_parked"), "alive_bool": None}, monkeypatch)
    assert overview_dict["attention_list"] == []
    assert scheduler_hold_dict({"state_str": "sleeping", "alive_bool": True}) == {}


def test_manual_submit_shows_the_trade_time_and_turns_urgent(pod_row_dict, monkeypatch):
    row_dict = _healthy_row_dict(pod_row_dict)
    target_ts = NOW_TS + timedelta(minutes=42)
    row_dict.update(next_action_str="review_vplan", latest_vplan_status_str="ready", auto_submit_enabled_bool=False,
        latest_vplan_target_execution_timestamp_str=target_ts.isoformat(),
        required_action_dict={"severity_str": "yellow", "label_str": "Review VPlan",
            "reason_str": "Auto-submit is disabled; inspect the VPlan."})
    overview_dict = _overview_dict([row_dict], _alive_dict("holding", "manual_review_required"), monkeypatch)
    item_dict, = overview_dict["attention_list"]
    assert item_dict["title_str"] == "Review VPlan" and item_dict["state_str"] == "late"
    assert item_dict["deadline_str"] == "You submit · trade 10:14 ET · in 42 min"
    urgent_dict = _overview_dict([row_dict], _alive_dict(), monkeypatch, now_ts=target_ts - timedelta(minutes=9))
    assert urgent_dict["attention_list"][0]["state_str"] == "fail"
    assert urgent_dict["attention_list"][0]["deadline_urgent_bool"] is True
    # Still unsent after the planned trade: stays an action, never quieter.
    passed_dict = _overview_dict([row_dict], _alive_dict(), monkeypatch, now_ts=target_ts + timedelta(minutes=1))
    assert passed_dict["attention_list"][0]["state_str"] == "fail"
    assert passed_dict["attention_list"][0]["deadline_str"] == "You submit · trade time passed 10:14 ET"
    assert passed_dict["pod_list"][0]["pill_str"] == "Action needed"


def test_deadline_items_sort_first(pod_row_dict, monkeypatch):
    held_dict = _healthy_row_dict(pod_row_dict, "pod_a_held", "ACCOUNT_A")
    manual_dict = _healthy_row_dict(pod_row_dict, "pod_b_manual", "ACCOUNT_B")
    manual_dict.update(next_action_str="review_vplan", latest_vplan_status_str="ready",
        latest_vplan_target_execution_timestamp_str=(NOW_TS + timedelta(hours=3)).isoformat(),
        required_action_dict={"severity_str": "yellow", "label_str": "Review VPlan", "reason_str": "Inspect."})

    class _MixedProvider:
        def get_scheduler_status_dict(self, pod_id_str, *, as_of_ts):
            return _alive_dict("holding", "execution_exception_parked") if pod_id_str == "pod_a_held" else _alive_dict()

    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_health_rollup",
        lambda *args, **kwargs: SimpleNamespace(severity_str="green"))
    workspace_dict = {"client_dict": {"accounts": []}, "operations_account_list": [
        {"pod_id": row_dict["pod_id_str"], "account_route": row_dict["account_route_str"], "display_name": row_dict["pod_id_str"]}
        for row_dict in (held_dict, manual_dict)],
        "summary_dict": {"as_of_timestamp_str": NOW_TS.isoformat(), "pod_row_dict_list": [held_dict, manual_dict]}}
    overview_dict = build_overview_dict(workspace_dict, None, _MixedProvider(), as_of_ts=NOW_TS, include_finance_bool=False)
    assert [item_dict["pod_id_str"] for item_dict in overview_dict["attention_list"]] == ["pod_b_manual", "pod_a_held"]


def test_scheduler_error_shows_redacted_reason_and_console_link():
    issue_dict = scheduler_issue_dict({"state_str": "error", "alive_bool": True,
        "error_reason_str": "[WinError 1225] The remote computer refused the network connection",
        "promised_wake_timestamp_str": (NOW_TS + timedelta(seconds=60)).isoformat()}, now_ts=NOW_TS)
    assert issue_dict["detail_str"] == ("Reason: [WinError 1225] The remote computer refused the network connection."
        " Retry due 09:33:00 ET.")
    assert issue_dict["console_bool"] is True
    plain_dict = scheduler_issue_dict({"state_str": "error", "alive_bool": True}, now_ts=NOW_TS)
    assert plain_dict["detail_str"] == "The scheduler reported an error."


@pytest.mark.parametrize("error_str,secret_str,masked_str", [
    ("Positions differ for account\nU1234567", "U1234567", "U···567"),
    ("IBKR error 321\nDU1234567 not allowed", "DU1234567", "D···567"),
    ("auth failed\nBearer abcdef123456", "abcdef123456", "[redacted]"),
    ("colored\x1b[31mU7654321\x1b[0m done", "U7654321", "U···321"),
])
def test_multiline_error_reason_is_still_redacted(error_str, secret_str, masked_str):
    from alpha.live.dashboard_v4.scheduler_status import _error_reason_str
    reason_str = _error_reason_str(error_str)
    assert secret_str not in reason_str and masked_str in reason_str
    assert "\n" not in reason_str and "\x1b" not in reason_str
