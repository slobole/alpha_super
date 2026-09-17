"""Tests for dashboard_v3.notifications."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
import urllib.error

import pytest

from alpha.live.dashboard_v3.notifications import (
    NotificationState,
    NotificationStateStore,
    check_and_notify_for_red_transitions,
    post_discord_webhook_bool,
)


def _row_dict(pod_id_str: str, severity_str: str, mode_str: str = "live") -> dict[str, Any]:
    return {
        "pod_id_str": pod_id_str,
        "mode_str": mode_str,
        "health_str": severity_str,
        "required_action_dict": {
            "label_str": "Manual review",
            "severity_str": severity_str,
            "reason_str": f"stub reason for {pod_id_str}",
        },
        "debug_summary_dict": {
            "severity_str": severity_str,
            "primary_reason_str": f"primary reason for {pod_id_str}",
        },
    }


@pytest.fixture(name="state_store_obj")
def fixture_state_store_obj(tmp_path: Path) -> NotificationStateStore:
    return NotificationStateStore(state_path_str=str(tmp_path / "notification_state.json"))


class CapturingPoster:
    def __init__(self) -> None:
        self.calls_list: list[tuple[str, dict[str, Any]]] = []
        self.return_value_bool = True

    def __call__(self, webhook_url_str: str, payload_dict: dict[str, Any]) -> bool:
        self.calls_list.append((webhook_url_str, dict(payload_dict)))
        return self.return_value_bool


def test_state_store_round_trip(state_store_obj) -> None:
    initial_state_obj = state_store_obj.load_state()
    assert initial_state_obj.pod_severity_map_dict == {}
    state_store_obj.save_state(
        NotificationState(
            pod_severity_map_dict={"pod_a": "green"},
            last_updated_str="2026-05-21T16:00:00+00:00",
        )
    )
    reloaded_state_obj = state_store_obj.load_state()
    assert reloaded_state_obj.pod_severity_map_dict == {"pod_a": "green"}
    assert reloaded_state_obj.last_updated_str == "2026-05-21T16:00:00+00:00"


def test_first_pass_with_no_red_pods_fires_no_webhook(state_store_obj) -> None:
    poster_obj = CapturingPoster()
    summary_dict = {"pod_row_dict_list": [_row_dict("pod_a", "green")]}
    fired_list = check_and_notify_for_red_transitions(
        summary_dict,
        state_store_obj=state_store_obj,
        webhook_url_str="https://discord.example/hook",
        webhook_poster_fn=poster_obj,
    )
    assert fired_list == []
    assert poster_obj.calls_list == []
    # State is now persisted so subsequent transitions can be detected.
    assert state_store_obj.load_state().pod_severity_map_dict == {"pod_a": "green"}


def test_green_to_red_transition_fires_webhook(state_store_obj) -> None:
    # Prime the store with the pod already green.
    state_store_obj.save_state(NotificationState(pod_severity_map_dict={"pod_a": "green"}))
    poster_obj = CapturingPoster()
    summary_dict = {"pod_row_dict_list": [_row_dict("pod_a", "red")]}
    fired_list = check_and_notify_for_red_transitions(
        summary_dict,
        state_store_obj=state_store_obj,
        webhook_url_str="https://discord.example/hook",
        webhook_poster_fn=poster_obj,
    )
    assert len(fired_list) == 1
    assert fired_list[0].pod_id_str == "pod_a"
    assert fired_list[0].previous_severity_str == "green"
    assert fired_list[0].delivered_bool is True
    assert len(poster_obj.calls_list) == 1
    payload_dict = poster_obj.calls_list[0][1]
    assert "pod_a" in payload_dict["content"]
    assert "RED" in payload_dict["content"]


def test_red_to_red_does_not_refire(state_store_obj) -> None:
    state_store_obj.save_state(NotificationState(pod_severity_map_dict={"pod_a": "red"}))
    poster_obj = CapturingPoster()
    summary_dict = {"pod_row_dict_list": [_row_dict("pod_a", "red")]}
    fired_list = check_and_notify_for_red_transitions(
        summary_dict,
        state_store_obj=state_store_obj,
        webhook_url_str="https://discord.example/hook",
        webhook_poster_fn=poster_obj,
    )
    assert fired_list == []
    assert poster_obj.calls_list == []


def test_recovery_then_red_re_fires(state_store_obj) -> None:
    poster_obj = CapturingPoster()

    summary_red_dict = {"pod_row_dict_list": [_row_dict("pod_a", "red")]}
    summary_green_dict = {"pod_row_dict_list": [_row_dict("pod_a", "green")]}

    # 1st: green-to-red on first pass (previous is "" not red, so fires).
    state_store_obj.save_state(NotificationState(pod_severity_map_dict={"pod_a": "green"}))
    check_and_notify_for_red_transitions(
        summary_red_dict,
        state_store_obj=state_store_obj,
        webhook_url_str="https://discord.example/hook",
        webhook_poster_fn=poster_obj,
    )
    assert len(poster_obj.calls_list) == 1

    # 2nd: pod recovers — no fire, but state updates to green.
    check_and_notify_for_red_transitions(
        summary_green_dict,
        state_store_obj=state_store_obj,
        webhook_url_str="https://discord.example/hook",
        webhook_poster_fn=poster_obj,
    )
    assert len(poster_obj.calls_list) == 1

    # 3rd: red again — fires a fresh notification.
    check_and_notify_for_red_transitions(
        summary_red_dict,
        state_store_obj=state_store_obj,
        webhook_url_str="https://discord.example/hook",
        webhook_poster_fn=poster_obj,
    )
    assert len(poster_obj.calls_list) == 2


def test_missing_webhook_url_still_updates_state(state_store_obj) -> None:
    poster_obj = CapturingPoster()
    summary_dict = {"pod_row_dict_list": [_row_dict("pod_a", "red")]}
    fired_list = check_and_notify_for_red_transitions(
        summary_dict,
        state_store_obj=state_store_obj,
        webhook_url_str="",
        webhook_poster_fn=poster_obj,
    )
    # No webhook fired (delivered_bool=False), but the transition still
    # registered so re-configuring the URL later won't backfill alerts.
    assert poster_obj.calls_list == []
    assert len(fired_list) == 1
    assert fired_list[0].delivered_bool is False
    assert state_store_obj.load_state().pod_severity_map_dict == {"pod_a": "red"}


def test_webhook_failure_returns_delivered_false(state_store_obj) -> None:
    poster_obj = CapturingPoster()
    poster_obj.return_value_bool = False
    state_store_obj.save_state(NotificationState(pod_severity_map_dict={"pod_a": "green"}))
    summary_dict = {"pod_row_dict_list": [_row_dict("pod_a", "red")]}
    fired_list = check_and_notify_for_red_transitions(
        summary_dict,
        state_store_obj=state_store_obj,
        webhook_url_str="https://discord.example/hook",
        webhook_poster_fn=poster_obj,
    )
    assert fired_list[0].delivered_bool is False


def test_inspector_red_transition_fires_webhook(state_store_obj) -> None:
    state_store_obj.save_state(
        NotificationState(pod_severity_map_dict={"__inspector__": "green"})
    )
    poster_obj = CapturingPoster()
    summary_dict = {
        "pod_row_dict_list": [_row_dict("pod_a", "green")],
        "inspector_report_dict": {
            "overall_severity_str": "red",
            "overall_reason_str": "Inspector source summary is stale.",
            "mode_str": "live",
            "vps_id_str": "vps_01",
        },
    }

    fired_list = check_and_notify_for_red_transitions(
        summary_dict,
        state_store_obj=state_store_obj,
        webhook_url_str="https://discord.example/hook",
        webhook_poster_fn=poster_obj,
    )

    assert len(fired_list) == 1
    assert fired_list[0].pod_id_str == "__inspector__"
    assert fired_list[0].previous_severity_str == "green"
    assert len(poster_obj.calls_list) == 1
    payload_dict = poster_obj.calls_list[0][1]
    assert "INSPECTOR" in payload_dict["content"]
    assert "vps_01" in payload_dict["content"]


def _notification_summary_dict(notification_key_str: str, severity_str: str) -> dict[str, Any]:
    if notification_key_str != "__inspector__":
        return {"pod_row_dict_list": [_row_dict(notification_key_str, severity_str)]}
    return {
        "pod_row_dict_list": [],
        "inspector_report_dict": {
            "generated_at_utc_str": datetime.now(UTC).isoformat(),
            "overall_severity_str": severity_str,
            "overall_reason_str": "Inspector fixture reason.",
            "mode_str": "live",
            "vps_id_str": "vps_01",
        },
    }


@pytest.mark.parametrize("notification_key_str", ["pod_a", "__inspector__"])
@pytest.mark.parametrize("previous_severity_str", ["", "green"])
def test_failed_delivery_retries_across_restart_until_success(
    state_store_obj, notification_key_str, previous_severity_str
) -> None:
    state_store_obj.save_state(NotificationState(
        pod_severity_map_dict={notification_key_str: previous_severity_str}
    ))
    summary_dict = _notification_summary_dict(notification_key_str, "red")
    poster_obj = CapturingPoster()
    for delivery_bool, expected_attempt_count_int in [(False, 1), (False, 1), (True, 1), (True, 0)]:
        poster_obj.return_value_bool = delivery_bool
        # Every pass starts from persisted state, just like scheduled processes.
        restarted_store_obj = NotificationStateStore(state_store_obj.state_path_str)
        fired_list = check_and_notify_for_red_transitions(
            summary_dict, state_store_obj=restarted_store_obj,
            webhook_url_str="https://discord.example/hook", webhook_poster_fn=poster_obj,
        )
        assert len(fired_list) == expected_attempt_count_int
        assert restarted_store_obj.load_state().pod_severity_map_dict[notification_key_str] == "red"
        if fired_list:
            assert fired_list[0].delivered_bool is delivery_bool
            assert fired_list[0].previous_severity_str == (previous_severity_str or "unknown")
    assert len(poster_obj.calls_list) == 3


@pytest.mark.parametrize("notification_key_str", ["pod_a", "__inspector__"])
def test_recovery_discards_failed_alert_and_rearms_next_red(state_store_obj, notification_key_str) -> None:
    poster_obj = CapturingPoster()
    for severity_str, delivery_bool, expected_attempt_count_int in [
        ("red", False, 1), ("green", True, 0), ("green", True, 0), ("red", True, 1), ("red", True, 0),
    ]:
        poster_obj.return_value_bool = delivery_bool
        fired_list = check_and_notify_for_red_transitions(
            _notification_summary_dict(notification_key_str, severity_str),
            state_store_obj=NotificationStateStore(state_store_obj.state_path_str),
            webhook_url_str="https://discord.example/hook", webhook_poster_fn=poster_obj,
        )
        assert len(fired_list) == expected_attempt_count_int
    assert len(poster_obj.calls_list) == 2


@pytest.mark.parametrize("notification_key_str", ["pod_a", "__inspector__"])
def test_temporarily_missing_url_preserves_an_existing_failed_delivery(state_store_obj, notification_key_str) -> None:
    poster_obj = CapturingPoster()
    poster_obj.return_value_bool = False
    summary_dict = _notification_summary_dict(notification_key_str, "red")
    check_and_notify_for_red_transitions(
        summary_dict, state_store_obj=state_store_obj,
        webhook_url_str="https://discord.example/hook", webhook_poster_fn=poster_obj,
    )
    check_and_notify_for_red_transitions(
        summary_dict, state_store_obj=NotificationStateStore(state_store_obj.state_path_str),
        webhook_url_str="", webhook_poster_fn=poster_obj,
    )
    assert len(poster_obj.calls_list) == 1
    poster_obj.return_value_bool = True
    fired_list = check_and_notify_for_red_transitions(
        summary_dict, state_store_obj=NotificationStateStore(state_store_obj.state_path_str),
        webhook_url_str="https://discord.example/hook", webhook_poster_fn=poster_obj,
    )
    assert len(fired_list) == 1
    assert fired_list[0].delivered_bool is True
    assert len(poster_obj.calls_list) == 2


@pytest.mark.parametrize("notification_key_str", ["pod_a", "__inspector__"])
def test_unconfigured_webhook_does_not_backfill_old_red(state_store_obj, notification_key_str) -> None:
    poster_obj = CapturingPoster()
    summary_dict = _notification_summary_dict(notification_key_str, "red")
    for webhook_url_str in ["", "https://discord.example/hook"]:
        check_and_notify_for_red_transitions(
            summary_dict, state_store_obj=NotificationStateStore(state_store_obj.state_path_str),
            webhook_url_str=webhook_url_str, webhook_poster_fn=poster_obj,
        )
    assert poster_obj.calls_list == []


@pytest.mark.parametrize("notification_key_str", ["pod_a", "__inspector__"])
def test_legacy_red_state_is_not_replayed_on_upgrade(state_store_obj, notification_key_str) -> None:
    # Actual legacy JSON has no delivery/pending field.
    Path(state_store_obj.state_path_str).write_text(json.dumps({
        "pod_severity_map_dict": {notification_key_str: "red"}, "last_updated_str": "2026-09-16",
    }), encoding="utf-8")
    poster_obj = CapturingPoster()
    for severity_str, expected_attempt_count_int in [("red", 0), ("green", 0), ("red", 1)]:
        fired_list = check_and_notify_for_red_transitions(
            _notification_summary_dict(notification_key_str, severity_str),
            state_store_obj=NotificationStateStore(state_store_obj.state_path_str),
            webhook_url_str="https://discord.example/hook", webhook_poster_fn=poster_obj,
        )
        assert len(fired_list) == expected_attempt_count_int
    assert len(poster_obj.calls_list) == 1


def test_only_failed_members_of_a_batch_retry_with_current_reason(state_store_obj) -> None:
    summary_dict = _notification_summary_dict("__inspector__", "red")
    summary_dict["pod_row_dict_list"] = [_row_dict("pod_a", "red"), _row_dict("pod_b", "red")]
    payload_list = []

    def partial_poster_bool(webhook_url_str, payload_dict):
        payload_list.append(payload_dict)
        return "pod_a" in payload_dict["content"]

    check_and_notify_for_red_transitions(
        summary_dict, state_store_obj=state_store_obj,
        webhook_url_str="https://discord.example/hook", webhook_poster_fn=partial_poster_bool,
    )
    assert len(payload_list) == 3
    summary_dict["pod_row_dict_list"][1]["debug_summary_dict"]["primary_reason_str"] = "Updated failure evidence"
    poster_obj = CapturingPoster()
    fired_list = check_and_notify_for_red_transitions(
        summary_dict, state_store_obj=NotificationStateStore(state_store_obj.state_path_str),
        webhook_url_str="https://discord.example/hook", webhook_poster_fn=poster_obj,
    )
    assert {record_obj.pod_id_str for record_obj in fired_list} == {"pod_b", "__inspector__"}
    assert "Updated failure evidence" in poster_obj.calls_list[0][1]["content"]
    assert check_and_notify_for_red_transitions(
        summary_dict, state_store_obj=NotificationStateStore(state_store_obj.state_path_str),
        webhook_url_str="https://discord.example/hook", webhook_poster_fn=poster_obj,
    ) == []


def test_removed_pod_does_not_leave_a_retry_backlog(state_store_obj) -> None:
    poster_obj = CapturingPoster()
    poster_obj.return_value_bool = False
    check_and_notify_for_red_transitions(
        _notification_summary_dict("pod_a", "red"), state_store_obj=state_store_obj,
        webhook_url_str="https://discord.example/hook", webhook_poster_fn=poster_obj,
    )
    assert check_and_notify_for_red_transitions(
        {"pod_row_dict_list": []}, state_store_obj=NotificationStateStore(state_store_obj.state_path_str),
        webhook_url_str="https://discord.example/hook", webhook_poster_fn=poster_obj,
    ) == []
    assert state_store_obj.load_state().pending_red_previous_severity_map_dict == {}


@pytest.mark.parametrize("error_obj", [
    TimeoutError("fixture timeout"),
    urllib.error.URLError("fixture connection failure"),
    urllib.error.HTTPError("https://discord.example/hook", 429, "fixture rate limit", {}, None),
    urllib.error.HTTPError("https://discord.example/hook", 500, "fixture server error", {}, None),
])
def test_http_delivery_errors_return_failure(monkeypatch, error_obj) -> None:
    def failing_urlopen(request_obj, timeout):
        assert request_obj.method == "POST"
        assert timeout == 3.0
        raise error_obj

    monkeypatch.setattr("alpha.live.dashboard_v3.notifications.urllib.request.urlopen", failing_urlopen)
    assert post_discord_webhook_bool("https://discord.example/hook", {"content": "fixture"}) is False
