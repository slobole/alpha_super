"""Tests for the reverse-stream event log reader optimisation.

The polish branch rewrote ``_iter_event_dict_reverse`` /
``load_recent_event_dict_list`` / ``_latest_event_timestamp_str`` to read
the JSONL log from the *end* instead of the start. Behaviour must be
preserved exactly; only the time complexity changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alpha.live.dashboard import (
    _iter_event_dict_reverse,
    _latest_event_timestamp_str,
    load_recent_event_dict_list,
)


def _write_jsonl(path_obj: Path, event_dict_list: list[dict[str, Any]]) -> None:
    path_obj.write_text(
        "\n".join(json.dumps(event_dict) for event_dict in event_dict_list) + "\n",
        encoding="utf-8",
    )


def test_iter_reverse_yields_events_in_reverse_order(tmp_path: Path) -> None:
    path_obj = tmp_path / "events.jsonl"
    _write_jsonl(
        path_obj,
        [
            {"pod_id_str": "a", "timestamp_str": "T1"},
            {"pod_id_str": "a", "timestamp_str": "T2"},
            {"pod_id_str": "a", "timestamp_str": "T3"},
        ],
    )
    yielded_list = list(_iter_event_dict_reverse(path_obj))
    assert [event_dict["timestamp_str"] for event_dict in yielded_list] == ["T3", "T2", "T1"]


def test_iter_reverse_skips_malformed_lines(tmp_path: Path) -> None:
    path_obj = tmp_path / "events.jsonl"
    path_obj.write_text(
        '{"pod_id_str": "a", "timestamp_str": "T1"}\n'
        "not-json\n"
        '{"pod_id_str": "a", "timestamp_str": "T2"}\n'
        "\n"
        '{"pod_id_str": "a", "timestamp_str": "T3"}\n',
        encoding="utf-8",
    )
    yielded_list = list(_iter_event_dict_reverse(path_obj))
    timestamps_list = [event_dict["timestamp_str"] for event_dict in yielded_list]
    assert timestamps_list == ["T3", "T2", "T1"]


def test_iter_reverse_handles_many_lines_across_chunks(tmp_path: Path) -> None:
    path_obj = tmp_path / "events.jsonl"
    # 5000 events ≈ many reverse-read chunks. Each event is small.
    event_dict_list = [
        {"pod_id_str": "a", "timestamp_str": f"T{idx_int:06d}"}
        for idx_int in range(5000)
    ]
    _write_jsonl(path_obj, event_dict_list)
    yielded_list = list(_iter_event_dict_reverse(path_obj))
    assert len(yielded_list) == 5000
    assert yielded_list[0]["timestamp_str"] == "T005000".replace("T005000", "T004999")
    assert yielded_list[-1]["timestamp_str"] == "T000000"


def test_load_recent_event_dict_list_returns_chronological(tmp_path: Path) -> None:
    path_obj = tmp_path / "events.jsonl"
    _write_jsonl(
        path_obj,
        [
            {"pod_id_str": "a", "timestamp_str": "T1"},
            {"pod_id_str": "b", "timestamp_str": "T2"},
            {"pod_id_str": "a", "timestamp_str": "T3"},
            {"pod_id_str": "a", "timestamp_str": "T4"},
        ],
    )
    result_list = load_recent_event_dict_list(str(path_obj), "a", limit_int=10)
    assert [event_dict["timestamp_str"] for event_dict in result_list] == ["T1", "T3", "T4"]


def test_load_recent_event_dict_list_caps_at_limit_int(tmp_path: Path) -> None:
    path_obj = tmp_path / "events.jsonl"
    _write_jsonl(
        path_obj,
        [
            {"pod_id_str": "a", "timestamp_str": f"T{idx_int:02d}"}
            for idx_int in range(20)
        ],
    )
    result_list = load_recent_event_dict_list(str(path_obj), "a", limit_int=5)
    # The 5 most recent matching events in chronological order.
    assert [event_dict["timestamp_str"] for event_dict in result_list] == ["T15", "T16", "T17", "T18", "T19"]


def test_load_recent_event_dict_list_returns_empty_for_missing_file(tmp_path: Path) -> None:
    result_list = load_recent_event_dict_list(str(tmp_path / "nope.jsonl"), "a")
    assert result_list == []


def test_latest_event_timestamp_str_returns_most_recent_match(tmp_path: Path) -> None:
    path_obj = tmp_path / "events.jsonl"
    _write_jsonl(
        path_obj,
        [
            {"pod_id_str": "a", "timestamp_str": "T1"},
            {"pod_id_str": "a", "timestamp_str": "T2"},
            {"pod_id_str": "b", "timestamp_str": "T3"},  # later but different pod
        ],
    )
    assert _latest_event_timestamp_str(str(path_obj), "a") == "T2"


def test_latest_event_timestamp_str_returns_none_for_unknown_pod(tmp_path: Path) -> None:
    path_obj = tmp_path / "events.jsonl"
    _write_jsonl(
        path_obj,
        [{"pod_id_str": "a", "timestamp_str": "T1"}],
    )
    assert _latest_event_timestamp_str(str(path_obj), "b") is None


def test_latest_event_timestamp_str_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert _latest_event_timestamp_str(str(tmp_path / "nope.jsonl"), "a") is None


def test_latest_event_timestamp_str_handles_none_path() -> None:
    assert _latest_event_timestamp_str(None, "a") is None


def test_latest_event_timestamp_str_matches_related_pod_id_list(tmp_path: Path) -> None:
    path_obj = tmp_path / "events.jsonl"
    _write_jsonl(
        path_obj,
        [
            {"pod_id_str": "x", "related_pod_id_list": ["a", "z"], "timestamp_str": "T1"},
            {"pod_id_str": "y", "timestamp_str": "T2"},
        ],
    )
    # 'a' appears in the related-pod list of the first event but not the second.
    assert _latest_event_timestamp_str(str(path_obj), "a") == "T1"
