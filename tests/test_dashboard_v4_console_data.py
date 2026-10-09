"""Console tail of the operator log: bounded, pod-exact, redacted, never raises for I/O."""

from datetime import datetime, timedelta, timezone
import os
import re
import sys

import pytest

from alpha.live.dashboard_v4 import console_data
from alpha.live.dashboard_v4.console_data import (
    build_console_download_str,
    line_matches_pod_bool,
    open_shared_read_obj,
    parse_console_line_dict,
    read_console_tail_dict,
    redact_console_text_str,
)
from alpha.live.logging_utils import log_operator_message, render_operator_message_str


BASE_TS = datetime(2026, 9, 21, 17, 0, tzinfo=timezone.utc)


def _line_str(action_str="cycle.wait", *, pod_str="pod_a", level_str="INFO", seconds_int=0, **field_dict):
    field_map_dict = {"pod": pod_str, "account": "SIM_" + str(pod_str), **field_dict}
    return render_operator_message_str(level_str, action_str, BASE_TS + timedelta(seconds=seconds_int), field_map_dict)


def _write_lines(path_obj, line_list, *, mode_str="ab", ending_bytes=b"\n"):
    with open(path_obj, mode_str) as file_obj:
        for line_str in line_list:
            file_obj.write(line_str.encode("utf-8") + ending_bytes)


def _seq_list(result_dict):
    return [int(re.search(r"seq=(\d+)", line_dict["x"]).group(1)) for line_dict in result_dict["line_list"]
        if line_dict["l"] != "M"]


def _offset_int(cursor_str):
    return int(cursor_str.rsplit(".", 1)[1])


# ---------- pod token ----------

def test_pod_token_is_exact_and_may_end_the_line():
    line_a_str = _line_str(pod_str="pod_a")
    line_ab_str = _line_str(pod_str="pod_ab")
    assert line_matches_pod_bool(line_a_str, "pod_a")
    assert not line_matches_pod_bool(line_ab_str, "pod_a")
    assert line_matches_pod_bool(line_ab_str, "pod_ab")
    end_str = render_operator_message_str("INFO", "cycle.start", BASE_TS, {"mode": "live", "pod": "pod_a"})
    assert end_str.endswith(" pod=pod_a")
    assert line_matches_pod_bool(end_str, "pod_a")
    assert line_matches_pod_bool(end_str + "\r", "pod_a")
    assert not line_matches_pod_bool(end_str + "b", "pod_a")
    # A value that merely contains the token text is not the pod field.
    assert not line_matches_pod_bool(_line_str(pod_str="pod_b", reason="xpod=pod_a"), "pod_a")
    assert not line_matches_pod_bool(_line_str("norgate.sync.start", pod_str=None), "pod_a")


def test_all_pods_excludes_only_norgate_sync_skipped():
    skipped_str = render_operator_message_str("INFO", "norgate.sync.skipped", BASE_TS,
        {"status": "direct", "dates": "{}", "reason": "direct_norgate_mode"})
    start_str = render_operator_message_str("INFO", "norgate.sync.start", BASE_TS, {"status": "waiting"})
    assert not line_matches_pod_bool(skipped_str, None)
    assert line_matches_pod_bool(start_str, None)
    assert line_matches_pod_bool(_line_str(pod_str="pod_x"), None)
    assert line_matches_pod_bool("free text without a header", None)
    # A pod line that only mentions the action in its reason is still shown.
    assert line_matches_pod_bool(_line_str(reason="norgate.sync.skipped"), None)


# ---------- parsing ----------

def test_parse_header_levels_and_other_lines():
    for level_str, code_str in (("INFO", "I"), ("WARN", "W"), ("ERROR", "E"), ("CRITICAL", "C")):
        line_dict = parse_console_line_dict(_line_str("cycle.fail", level_str=level_str, reason="x y"))
        assert line_dict["t"] == "2026-09-21T17:00:00+00:00"
        assert line_dict["l"] == code_str
        assert line_dict["a"] == "cycle.fail"
        assert line_dict["x"] == "pod=pod_a account=SIM_pod_a reason=x y"
    assert parse_console_line_dict("Traceback (most recent call last):") == {
        "t": None, "l": "O", "a": "", "x": "Traceback (most recent call last):"}
    # Impossible dates and unknown levels are not trusted as a header.
    assert parse_console_line_dict("[2026-13-45 99:00:00 UTC] INFO cycle.wait pod=p")["l"] == "O"
    assert parse_console_line_dict("[2026-09-21 17:00:00 UTC] NOTICE cycle.wait pod=p")["l"] == "O"
    bare_dict = parse_console_line_dict(render_operator_message_str("INFO", "service.start", BASE_TS, {}))
    assert bare_dict == {"t": "2026-09-21T17:00:00+00:00", "l": "I", "a": "service.start", "x": ""}


def test_parse_crlf_and_carriage_return_progress():
    assert parse_console_line_dict(_line_str(reason="r") + "\r")["x"].endswith("reason=r")
    progress_dict = parse_console_line_dict("Downloading 10%\rDownloading 55%\rDownloading 100%\r\r")
    assert progress_dict == {"t": None, "l": "O", "a": "", "x": "Downloading 100%"}
    header_dict = parse_console_line_dict("garbage\r" + _line_str(reason="last"))
    assert header_dict["l"] == "I" and header_dict["x"].endswith("reason=last")


def test_long_line_is_truncated_with_suffix():
    long_str = _line_str(reason="r" * 5000)
    line_dict = parse_console_line_dict(long_str)
    assert len(line_dict["x"]) < console_data.LINE_CHARS_INT + 40
    prefix_str, suffix_str = line_dict["x"].split(" … [+")
    assert len(prefix_str) == console_data.LINE_CHARS_INT
    full_len_int = len(long_str.split(" cycle.wait ", 1)[1])
    assert suffix_str == f"{full_len_int - console_data.LINE_CHARS_INT} chars]"
    # A huge line is pre-cut before redaction; the cut token is dropped, never shown.
    huge_dict = parse_console_line_dict(("word " * 3000) + "https://discord.com/api/webhooks/1/" + "S" * 100000)
    assert huge_dict["x"].startswith("word word")
    assert "SSSS" not in huge_dict["x"] and "webhooks" not in huge_dict["x"]
    assert re.search(r" … \[\+\d+ chars\]\Z", huge_dict["x"])


# ---------- redaction ----------

@pytest.mark.parametrize("raw_str, expected_str, secret_str", [
    ("broker said U21192795 refused", "broker said U···795 refused", "21192795"),
    ("paper DU1234567 ok", "paper D···567 ok", "1234567"),
    ("paper DUK322077 ok", "paper D···077 ok", "322077"),
    ("account=U21192795 decision=none", "account=U···795 decision=none", "21192795"),
    ("account_route=U123456 x", "account_route=U···456 x", "123456"),
    ("account=SIM_pod_dv2_01 decision=none", "account=SIM_pod_dv2_01 decision=none", None),
    ("reason=failed token=abc next", "reason=failed token=[redacted] next", "abc"),
    ("Authorization: Bearer xyz", "Authorization: [redacted]", "xyz"),
    ("header Bearer xyz end", "header Bearer [redacted] end", "xyz"),
    ("post https://discord.com/api/webhooks/123/abcDEF failed", "post [webhook redacted] failed", "abcDEF"),
    ("get https://user:pw@host/x failed", "get https://[redacted]@host/x failed", "pw"),
    ("get https://host/x?sig=abc&key=def&X-Amz-Signature=ghi&page=2",
        "get https://host/x?sig=[redacted]&key=[redacted]&X-Amz-Signature=[redacted]&page=2", "ghi"),
    ("path C:\\Users\\Someone\\x\\y.py", "path C:\\Users\\…\\x\\y.py", "Someone"),
    ("path c:/users/Someone/x", "path c:/users/…/x", "Someone"),
    ("\x1b[31mERROR\x1b[0m text\x07\x00 and\ttab", "ERROR text and\ttab", "\x1b"),
])
def test_redaction_patterns(raw_str, expected_str, secret_str):
    redacted_str = redact_console_text_str(raw_str)
    assert redacted_str == expected_str
    if secret_str:
        assert secret_str not in redacted_str


def test_parse_redacts_account_in_remainder():
    line_dict = parse_console_line_dict(render_operator_message_str("ERROR", "cycle.fail", BASE_TS,
        {"pod": "pod_live", "account": "U21192795", "reason": "[WinError 1225] refused"}))
    assert line_dict["x"] == "pod=pod_live account=U···795 reason=[WinError 1225] refused"


# ---------- first load ----------

def test_first_load_returns_newest_matching_lines_and_cursor(tmp_path):
    log_path = tmp_path / "live_operator.log"
    line_list = []
    for index_int in range(700):
        line_list.append(_line_str(pod_str="pod_a", seq=index_int))
        line_list.append(_line_str(pod_str="pod_ab", seq=index_int))
    _write_lines(log_path, line_list)
    result_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    assert _seq_list(result_dict) == list(range(200, 700))
    assert result_dict["cursor_str"].startswith("v1.")
    assert _offset_int(result_dict["cursor_str"]) == log_path.stat().st_size
    assert result_dict["reset_bool"] is False and result_dict["more_bool"] is False
    assert result_dict["pending_bytes_int"] == 0
    assert result_dict["file_size_int"] == log_path.stat().st_size
    assert result_dict["last_write_utc_str"].endswith("+00:00")
    assert result_dict["source_label_str"] == "Operator log"
    assert all("pod=pod_a " in line_dict["x"] for line_dict in result_dict["line_list"])
    assert set(result_dict) == {"source_label_str", "cursor_str", "reset_bool", "more_bool", "gap_bytes_int",
        "pending_bytes_int", "file_size_int", "last_write_utc_str", "note_str", "line_list"}


def test_first_load_notes_when_no_lines_match(tmp_path, monkeypatch):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(pod_str="pod_b", seq=index_int) for index_int in range(50)])
    result_dict = read_console_tail_dict(str(log_path), "pod_a", "")
    assert result_dict["line_list"] == []
    assert result_dict["note_str"] == "No lines for this Pod in the operator log."
    monkeypatch.setattr(console_data, "FIRST_SCAN_BYTES_INT", 1024)
    limited_dict = read_console_tail_dict(str(log_path), "pod_a", "")
    assert limited_dict["note_str"].startswith("No lines for this Pod in the last ")
    assert limited_dict["note_str"].endswith(" MB of the operator log.")
    assert read_console_tail_dict(str(log_path), None, "")["note_str"] == ""


def test_first_load_stops_at_scan_limit_and_drops_partial_first_line(tmp_path, monkeypatch):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(pod_str="pod_a", seq=index_int) for index_int in range(300)])
    monkeypatch.setattr(console_data, "FIRST_SCAN_BYTES_INT", 10000)
    monkeypatch.setattr(console_data, "CHUNK_BYTES_INT", 4096)
    result_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    seq_list = _seq_list(result_dict)
    assert seq_list == list(range(seq_list[0], 300))
    line_bytes_int = len(_line_str(pod_str="pod_a", seq=100).encode()) + 1
    # Only complete lines inside the last 10,000 bytes are shown.
    assert len(seq_list) == 10000 // line_bytes_int
    assert result_dict["note_str"] == ""


def test_all_pods_first_load_skips_norgate_noise(tmp_path):
    log_path = tmp_path / "live_operator.log"
    noise_str = render_operator_message_str("INFO", "norgate.sync.skipped", BASE_TS,
        {"status": "direct", "reason": "direct_norgate_mode"})
    ready_str = render_operator_message_str("INFO", "norgate.sync.ready", BASE_TS, {"status": "ready", "seq": 1})
    _write_lines(log_path, [noise_str, _line_str(pod_str="pod_a", seq=0), noise_str, ready_str, noise_str])
    result_dict = read_console_tail_dict(str(log_path), None, None)
    assert [line_dict["a"] for line_dict in result_dict["line_list"]] == ["cycle.wait", "norgate.sync.ready"]


# ---------- partial lines, CRLF, BOM, encoding ----------

def test_partial_trailing_line_is_held_then_delivered_once(tmp_path):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(seq=0)])
    tail_bytes = _line_str(seq=1).encode()
    with open(log_path, "ab") as file_obj:
        file_obj.write(tail_bytes[:30])
    first_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    assert _seq_list(first_dict) == [0]
    assert first_dict["pending_bytes_int"] == 30
    hold_dict = read_console_tail_dict(str(log_path), "pod_a", first_dict["cursor_str"])
    assert hold_dict["line_list"] == [] and hold_dict["cursor_str"] == first_dict["cursor_str"]
    assert hold_dict["more_bool"] is False and hold_dict["pending_bytes_int"] == 30
    with open(log_path, "ab") as file_obj:
        file_obj.write(tail_bytes[30:] + b"\n")
    done_dict = read_console_tail_dict(str(log_path), "pod_a", hold_dict["cursor_str"])
    assert _seq_list(done_dict) == [1]
    again_dict = read_console_tail_dict(str(log_path), "pod_a", done_dict["cursor_str"])
    assert again_dict["line_list"] == [] and again_dict["cursor_str"] == done_dict["cursor_str"]


def test_whole_file_partial_line_is_held_at_zero(tmp_path):
    log_path = tmp_path / "live_operator.log"
    log_path.write_bytes(b"\xef\xbb\xbf" + _line_str(seq=7).encode())
    first_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    assert first_dict["line_list"] == [] and _offset_int(first_dict["cursor_str"]) == 0
    with open(log_path, "ab") as file_obj:
        file_obj.write(b"\r\n")
    poll_dict = read_console_tail_dict(str(log_path), "pod_a", first_dict["cursor_str"])
    assert _seq_list(poll_dict) == [7]
    assert poll_dict["line_list"][0]["t"] == "2026-09-21T17:00:00+00:00"


def test_crlf_bom_and_invalid_utf8(tmp_path):
    log_path = tmp_path / "live_operator.log"
    log_path.write_bytes(b"\xef\xbb\xbf" + _line_str(seq=0).encode() + b"\r\n"
        + _line_str(seq=1, reason="bad").encode() + b"\xff\xfe\r\n"
        + b"step 1\rstep 2\r" + _line_str(seq=2).encode() + b"\r\n")
    result_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    assert _seq_list(result_dict) == [0, 1, 2]
    assert [line_dict["l"] for line_dict in result_dict["line_list"]] == ["I", "I", "I"]
    assert result_dict["line_list"][1]["x"].endswith("reason=bad\ufffd\ufffd")
    assert all("\r" not in line_dict["x"] for line_dict in result_dict["line_list"])
    assert _offset_int(result_dict["cursor_str"]) == log_path.stat().st_size


def test_real_writer_output_is_read(tmp_path):
    """The writer appends in text mode (CRLF on Windows); read it back as-is."""
    log_path = tmp_path / "live_operator.log"
    for index_int in range(3):
        log_operator_message(level_str="ERROR", phase_action_str="cycle.fail", timestamp_obj=BASE_TS,
            field_map_dict={"pod": "pod_live", "account": "U21192795", "seq": index_int},
            operator_log_path_str=str(log_path))
    result_dict = read_console_tail_dict(str(log_path), "pod_live", None)
    assert _seq_list(result_dict) == [0, 1, 2]
    assert result_dict["line_list"][0] == {"t": "2026-09-21T17:00:00+00:00", "l": "E", "a": "cycle.fail",
        "x": "pod=pod_live account=U···795 seq=0"}


# ---------- forward polling ----------

def test_forward_poll_is_bounded_and_drains_without_loss(tmp_path):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(seq=-1)])
    first_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    filler_str = "f" * 300
    _write_lines(log_path, [_line_str(pod_str="pod_a" if index_int % 2 == 0 else "pod_b", seq=index_int,
        reason=filler_str) for index_int in range(2400)])
    pending_int = log_path.stat().st_size - _offset_int(first_dict["cursor_str"])
    assert console_data.POLL_BYTES_INT < pending_int <= console_data.JUMP_BYTES_INT
    cursor_str, seen_list, poll_count_int = first_dict["cursor_str"], [], 0
    while True:
        before_int = _offset_int(cursor_str)
        poll_dict = read_console_tail_dict(str(log_path), "pod_a", cursor_str)
        assert poll_dict["reset_bool"] is False
        assert _offset_int(poll_dict["cursor_str"]) - before_int <= console_data.POLL_BYTES_INT
        seen_list.extend(_seq_list(poll_dict))
        cursor_str = poll_dict["cursor_str"]
        poll_count_int += 1
        if not poll_dict["more_bool"]:
            break
    assert poll_count_int >= 3
    assert seen_list == list(range(0, 2400, 2))
    assert _offset_int(cursor_str) == log_path.stat().st_size


def test_line_longer_than_poll_window_shows_its_start_once(tmp_path, monkeypatch):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(seq=0)])
    first_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    _write_lines(log_path, [_line_str(seq=1, reason="z" * 1000), _line_str(seq=2)])
    monkeypatch.setattr(console_data, "POLL_BYTES_INT", 300)
    cursor_str, line_list = first_dict["cursor_str"], []
    for _poll_int in range(20):
        poll_dict = read_console_tail_dict(str(log_path), "pod_a", cursor_str)
        line_list.extend(poll_dict["line_list"])
        cursor_str = poll_dict["cursor_str"]
        if not poll_dict["more_bool"]:
            break
    assert [line_dict["l"] for line_dict in line_list] == ["I", "I"]
    assert _seq_list({"line_list": line_list}) == [1, 2]
    assert _offset_int(cursor_str) == log_path.stat().st_size


def test_response_is_capped_keeping_newest_lines(tmp_path):
    log_path = tmp_path / "live_operator.log"
    log_path.write_bytes(b"")
    first_dict = read_console_tail_dict(str(log_path), None, None)
    assert first_dict["note_str"] == "No lines in the operator log."
    _write_lines(log_path, [f"x seq={index_int}" for index_int in range(3000)])
    poll_dict = read_console_tail_dict(str(log_path), None, first_dict["cursor_str"])
    assert len(poll_dict["line_list"]) == console_data.RESPONSE_LINE_LIMIT_INT
    assert poll_dict["line_list"][0]["l"] == "M"
    assert poll_dict["line_list"][0]["x"] == "Skipped 2001 older lines. Use Download for more."
    assert _seq_list(poll_dict) == list(range(2001, 3000))


def test_large_backlog_jumps_to_tail(tmp_path):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(seq=-1)])
    first_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    _write_lines(log_path, [_line_str(seq=index_int, reason="f" * 400) for index_int in range(3000)])
    pending_int = log_path.stat().st_size - _offset_int(first_dict["cursor_str"])
    assert pending_int > console_data.JUMP_BYTES_INT
    jump_dict = read_console_tail_dict(str(log_path), "pod_a", first_dict["cursor_str"])
    assert jump_dict["reset_bool"] is True
    assert jump_dict["gap_bytes_int"] == pending_int
    assert jump_dict["line_list"][0] == {"t": None, "l": "M", "a": "",
        "x": f"Skipped {pending_int / (1024 * 1024):.1f} MB of older output. Use Download for more."}
    assert _seq_list(jump_dict) == list(range(2500, 3000))
    assert _offset_int(jump_dict["cursor_str"]) == log_path.stat().st_size


# ---------- cursor validation, rotation, missing file ----------

def test_rotation_resets_with_marker(tmp_path):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(seq=index_int) for index_int in range(5)])
    first_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    os.replace(log_path, str(log_path) + ".1")
    _write_lines(log_path, [_line_str(seq=index_int) for index_int in range(100, 103)])
    rotated_dict = read_console_tail_dict(str(log_path), "pod_a", first_dict["cursor_str"])
    assert rotated_dict["reset_bool"] is True
    assert rotated_dict["line_list"][0]["x"] == "Log rotated or replaced; showing the latest lines."
    assert _seq_list(rotated_dict) == [100, 101, 102]
    assert rotated_dict["cursor_str"] != first_dict["cursor_str"]


def test_truncated_file_resets_with_marker(tmp_path):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(seq=index_int) for index_int in range(20)])
    first_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    with open(log_path, "r+b") as file_obj:
        file_obj.truncate(0)
    _write_lines(log_path, [_line_str(seq=50)])
    shrunk_dict = read_console_tail_dict(str(log_path), "pod_a", first_dict["cursor_str"])
    assert shrunk_dict["reset_bool"] is True
    assert shrunk_dict["line_list"][0]["l"] == "M"
    assert _seq_list(shrunk_dict) == [50]


@pytest.mark.parametrize("cursor_obj", ["garbage", "v1.zz.10", "v1.abc.-1", "v2.abc.0", "v1.ABC.0",
    "v1.abc.01", "v1.abc.0 ", "v1..0", 12345, ["v1"]])
def test_malformed_cursor_resets_without_marker(tmp_path, cursor_obj):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(seq=index_int) for index_int in range(3)])
    result_dict = read_console_tail_dict(str(log_path), "pod_a", cursor_obj)
    assert result_dict["reset_bool"] is True
    assert _seq_list(result_dict) == [0, 1, 2]
    assert all(line_dict["l"] != "M" for line_dict in result_dict["line_list"])


def test_cursor_inside_a_line_resynchronises(tmp_path):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(seq=0, reason="secret-tail")])
    first_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    file_id_str = first_dict["cursor_str"].split(".")[1]
    _write_lines(log_path, [_line_str(seq=1)])
    poll_dict = read_console_tail_dict(str(log_path), "pod_a", f"v1.{file_id_str}.10")
    assert _seq_list(poll_dict) == [1]
    assert "secret-tail" not in str(poll_dict["line_list"])


def test_missing_file_never_raises(tmp_path):
    missing_str = str(tmp_path / "nope" / "live_operator.log")
    result_dict = read_console_tail_dict(missing_str, "pod_a", None)
    assert result_dict["note_str"] == "Operator log not found."
    assert result_dict["line_list"] == [] and result_dict["cursor_str"] == ""
    assert result_dict["reset_bool"] is False
    gone_dict = read_console_tail_dict(missing_str, "pod_a", "v1.abc.10")
    assert gone_dict["reset_bool"] is True and gone_dict["cursor_str"] == ""
    peek_dict = read_console_tail_dict(missing_str, "pod_a", "v1.abc.10", peek_bool=True)
    assert peek_dict["note_str"] == "Operator log not found." and peek_dict["line_list"] == []
    assert build_console_download_str(missing_str, "pod_a") == (
        "# Operator log · pod=pod_a · last 2 MB · redacted\n# Operator log not found.")


def test_unreadable_path_keeps_cursor(tmp_path):
    result_dict = read_console_tail_dict(str(tmp_path), "pod_a", "v1.abc.10")
    assert result_dict["note_str"] == "Operator log could not be read."
    assert result_dict["cursor_str"] == "v1.abc.10" and result_dict["line_list"] == []


def test_peek_reports_pending_without_content(tmp_path):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(seq=0)])
    first_dict = read_console_tail_dict(str(log_path), "pod_a", None)
    _write_lines(log_path, [_line_str(seq=1), _line_str(seq=2)])
    peek_dict = read_console_tail_dict(str(log_path), "pod_a", first_dict["cursor_str"], peek_bool=True)
    assert peek_dict["cursor_str"] == first_dict["cursor_str"]
    assert peek_dict["line_list"] == []
    assert peek_dict["file_size_int"] == log_path.stat().st_size
    assert peek_dict["pending_bytes_int"] == log_path.stat().st_size - _offset_int(first_dict["cursor_str"])
    assert peek_dict["last_write_utc_str"].endswith("+00:00")
    bad_peek_dict = read_console_tail_dict(str(log_path), "pod_a", "v1.abc.0", peek_bool=True)
    assert bad_peek_dict["pending_bytes_int"] == log_path.stat().st_size


# ---------- download ----------

def test_download_is_filtered_redacted_bounded_and_has_header(tmp_path):
    log_path = tmp_path / "live_operator.log"
    line_list = []
    for index_int in range(9000):
        line_list.append(render_operator_message_str("ERROR", "cycle.fail", BASE_TS, {"pod": "pod_a",
            "account": "U21192795", "seq": index_int, "reason": "post https://discord.com/api/webhooks/1/abc " + "f" * 200}))
        line_list.append(_line_str(pod_str="pod_ab", seq=index_int))
    _write_lines(log_path, line_list, ending_bytes=b"\r\n")
    with open(log_path, "ab") as file_obj:
        file_obj.write(b"[2026-09-21 17:00:00 UTC] INFO cycle.wait pod=pod_a seq=999999")
    assert log_path.stat().st_size > console_data.DOWNLOAD_BYTES_INT + 1024 * 1024
    download_str = build_console_download_str(str(log_path), "pod_a")
    header_str, *body_list = download_str.split("\n")
    assert header_str == "# Operator log · pod=pod_a · last 2 MB · redacted"
    assert len(download_str.encode("utf-8")) <= console_data.DOWNLOAD_BYTES_INT + 100
    assert body_list and all(" pod=pod_a " in line_str for line_str in body_list)
    assert all(line_str.startswith("[2026-09-21 17:00:00 UTC] ERROR cycle.fail") for line_str in body_list)
    assert "U21192795" not in download_str and "account=U···795" in body_list[0]
    assert "webhooks" not in download_str and "\r" not in download_str
    seq_list = [int(re.search(r"seq=(\d+)", line_str).group(1)) for line_str in body_list]
    assert seq_list == list(range(seq_list[0], 9000))
    all_str = build_console_download_str(str(log_path), None)
    assert all_str.split("\n")[0] == "# Operator log · pod=all · last 2 MB · redacted"
    assert " pod=pod_ab " in all_str


# ---------- Windows sharing ----------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows share-mode semantics")
def test_open_reader_does_not_block_rotation_rename(tmp_path):
    log_path = tmp_path / "live_operator.log"
    _write_lines(log_path, [_line_str(seq=0)])
    # Control: a plain open() lacks FILE_SHARE_DELETE and blocks the writer's rename.
    with open(log_path, "rb"):
        with pytest.raises(PermissionError):
            os.replace(log_path, str(log_path) + ".1")
    with open_shared_read_obj(str(log_path)) as file_obj:
        os.replace(log_path, str(log_path) + ".1")
        assert file_obj.read().startswith(b"[2026-09-21")
    assert not log_path.exists() and (tmp_path / "live_operator.log.1").exists()
    raw_path = tmp_path / "raw.log"
    raw_path.write_bytes(b"a\r\nb\x1ac\n")
    with open_shared_read_obj(str(raw_path)) as file_obj:
        assert file_obj.read() == b"a\r\nb\x1ac\n"  # binary: no CRLF or Ctrl-Z translation
    with pytest.raises(FileNotFoundError):
        open_shared_read_obj(str(log_path))
    with pytest.raises(PermissionError):
        open_shared_read_obj(str(tmp_path))


@pytest.mark.parametrize("text_str,secret_str", [
    ("account\nU1234567", "U1234567"), ("x\x1b[0mDU1234567", "DU1234567"),
    ("page\x0cU1234567", "U1234567"), ("auth\rBearer abcdef123456", "abcdef123456"),
])
def test_control_characters_cannot_glue_a_secret_past_redaction(text_str, secret_str):
    assert secret_str not in redact_console_text_str(text_str)
