"""Tests for dashboard_v3.notifications."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alpha.live.dashboard_v3.notifications import (
    NotificationState,
    NotificationStateStore,
    check_and_notify_for_red_transitions,
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
