"""Bounded, redacted Console tail of the shared operator log. Read-only.

Every request opens the log, reads a bounded window and closes it again. On
Windows the handle shares DELETE so the writer's rotation rename never fails.
"""

from datetime import datetime, timezone
import errno
import os
import re

from alpha.live.dashboard_v3.operator_tools import redact_diagnostic_value


FIRST_LINE_LIMIT_INT = 500
RESPONSE_LINE_LIMIT_INT = 1000
CHUNK_BYTES_INT = 256 * 1024
FIRST_SCAN_BYTES_INT = 8 * 1024 * 1024
POLL_BYTES_INT = 256 * 1024
JUMP_BYTES_INT = 1024 * 1024
LINE_CHARS_INT = 2048
DOWNLOAD_BYTES_INT = 2 * 1024 * 1024

SOURCE_LABEL_STR = "Operator log"
SKIPPED_ACTION_STR = "norgate.sync.skipped"
BOM_BYTES = b"\xef\xbb\xbf"
# Redaction input is pre-cut to this many characters so one huge line cannot
# make the regex pass expensive; only LINE_CHARS_INT are ever shown.
REDACT_CHARS_INT = 4 * LINE_CHARS_INT
LEVEL_CODE_DICT = {"DEBUG": "I", "INFO": "I", "WARN": "W", "WARNING": "W", "ERROR": "E",
    "CRITICAL": "C", "FATAL": "C"}

HEADER_PATTERN_OBJ = re.compile(
    r"\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) UTC\] ([A-Z]+) ([A-Za-z][A-Za-z0-9_.:-]{0,127})(?: (.*))?\Z",
    re.DOTALL)
CURSOR_PATTERN_OBJ = re.compile(r"v1\.([0-9a-f]{1,64})\.(0|[1-9][0-9]{0,18})\Z")
ANSI_PATTERN_OBJ = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]{0,512}(?:\x07|\x1b\\)|\x1b[@-Z\\-_]|\x9b[0-?]*[ -/]*[@-~]")
# C0/C1 controls except tab, plus bidi overrides that could visually reorder text.
CONTROL_PATTERN_OBJ = re.compile("[\x00-\x08\x0a-\x1f\x7f-\x9f‪-‮⁦-⁩]")
MULTI_SPACE_PATTERN_OBJ = re.compile(" {2,}")
USERINFO_PATTERN_OBJ = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]{0,15}://)[^\s/?#]+@")
WEBHOOK_PATTERN_OBJ = re.compile(r"(?i)https?://\S*/api/webhooks/\S+|https?://hooks\.slack\.com/services/\S+")
QUERY_SECRET_PATTERN_OBJ = re.compile(
    r"(?i)((?:^|[?&;\s])[\w.\-]*?(?:token|key|sig|signature|password))=[^\s&#]+")
ACCOUNT_FIELD_PATTERN_OBJ = re.compile(r"(?i)(?<![\w\-])(account(?:_route|_id)?=)([^\s,;&\"']+)")
# IBKR ids: U/DU/DF/F + digits; paper ids also appear as DU + one letter (DUK322077).
ACCOUNT_ID_PATTERN_OBJ = re.compile(r"(?<![A-Za-z0-9])(?:U|DU[A-Z]?|DF|F)\d{5,9}(?![A-Za-z0-9])")
USER_PATH_PATTERN_OBJ = re.compile(r"(?i)\b([a-z]:(?:\\\\|\\|/)users(?:\\\\|\\|/))[^\\/\s\"':*?<>|]+")

if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    import msvcrt

    GENERIC_READ_INT = 0x80000000
    SHARE_ALL_INT = 0x00000001 | 0x00000002 | 0x00000004  # READ | WRITE | DELETE
    OPEN_EXISTING_INT = 3
    FILE_ATTRIBUTE_NORMAL_INT = 0x80
    INVALID_HANDLE_INT = ctypes.c_void_p(-1).value
    # A private WinDLL instance so these prototypes never leak into ctypes.windll.
    KERNEL32_OBJ = ctypes.WinDLL("kernel32", use_last_error=True)
    CREATE_FILE_FUNC = KERNEL32_OBJ.CreateFileW
    CREATE_FILE_FUNC.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    CREATE_FILE_FUNC.restype = wintypes.HANDLE
    CLOSE_HANDLE_FUNC = KERNEL32_OBJ.CloseHandle
    CLOSE_HANDLE_FUNC.argtypes = (wintypes.HANDLE,)
    CLOSE_HANDLE_FUNC.restype = wintypes.BOOL


class ConsoleReadError(OSError):
    """The file changed under a bounded read (e.g. truncated mid-request)."""


def open_shared_read_obj(path_str):
    """Open for binary reading without ever blocking the writer's rename/delete."""
    if os.name != "nt":
        return open(path_str, "rb")
    handle_int = CREATE_FILE_FUNC(str(path_str), GENERIC_READ_INT, SHARE_ALL_INT, None,
        OPEN_EXISTING_INT, FILE_ATTRIBUTE_NORMAL_INT, None)
    if handle_int is None or handle_int == INVALID_HANDLE_INT:
        error_int = ctypes.get_last_error()
        if error_int in (2, 3):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path_str))
        if error_int in (5, 32):
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path_str))
        raise OSError(errno.EIO, f"CreateFileW failed with Windows error {error_int}", str(path_str))
    try:
        # Without O_TEXT the CRT descriptor is binary: no CRLF or Ctrl-Z translation.
        fd_int = msvcrt.open_osfhandle(handle_int, os.O_RDONLY)
    except OSError:
        CLOSE_HANDLE_FUNC(handle_int)
        raise
    try:
        return os.fdopen(fd_int, "rb")
    except Exception:
        os.close(fd_int)
        raise


def _mask_account_str(value_str):
    return value_str[:1] + "···" + value_str[-3:] if len(value_str) >= 4 else "···"


def _account_field_str(match_obj):
    value_str = match_obj.group(2)
    if value_str.startswith("SIM_"):
        return match_obj.group(0)
    return match_obj.group(1) + _mask_account_str(value_str)


def redact_console_text_str(text_str):
    """Server-side redaction for every Console line and the download."""
    # Controls go first, replaced by a space: an escape sequence or line break can
    # neither split a secret past the patterns nor glue an id onto the previous word.
    value_str = ANSI_PATTERN_OBJ.sub(" ", str(text_str))
    value_str = CONTROL_PATTERN_OBJ.sub(" ", value_str)
    value_str = MULTI_SPACE_PATTERN_OBJ.sub(" ", value_str).strip(" ")
    value_str = redact_diagnostic_value(value_str)
    value_str = USERINFO_PATTERN_OBJ.sub(r"\1[redacted]@", value_str)
    value_str = WEBHOOK_PATTERN_OBJ.sub("[webhook redacted]", value_str)
    value_str = QUERY_SECRET_PATTERN_OBJ.sub(r"\1=[redacted]", value_str)
    value_str = ACCOUNT_FIELD_PATTERN_OBJ.sub(_account_field_str, value_str)
    value_str = ACCOUNT_ID_PATTERN_OBJ.sub(lambda match_obj: _mask_account_str(match_obj.group(0)), value_str)
    return USER_PATH_PATTERN_OBJ.sub(r"\1…", value_str)


def _normalize_line_str(line_str):
    """Drop CR line endings; a progress line keeps only what the CMD window shows last."""
    line_str = str(line_str).rstrip("\r")
    if "\r" in line_str:
        line_str = line_str.rpartition("\r")[2]
    return line_str


def _action_str(line_str):
    match_obj = HEADER_PATTERN_OBJ.match(line_str)
    return match_obj.group(8) if match_obj else ""


def line_matches_pod_bool(line_str, pod_id_str):
    """Exact ' pod=<id>' token; None/'' means every line except norgate.sync.skipped."""
    line_str = _normalize_line_str(line_str)
    if not pod_id_str:
        return _action_str(line_str) != SKIPPED_ACTION_STR
    token_str = " pod=" + str(pod_id_str)
    start_int = 0
    while True:
        index_int = line_str.find(token_str, start_int)
        if index_int < 0:
            return False
        end_int = index_int + len(token_str)
        if end_int == len(line_str) or line_str[end_int] == " ":
            return True
        start_int = index_int + 1


def _display_text_str(raw_str):
    """Redact first, then cut to LINE_CHARS_INT so a cut can never expose a secret's tail."""
    dropped_int = 0
    if len(raw_str) > REDACT_CHARS_INT:
        kept_str = raw_str[:REDACT_CHARS_INT]
        # Drop the token cut in half; it could be a partial secret the patterns miss.
        kept_str = kept_str[:max(kept_str.rfind(" "), 0)]
        dropped_int = len(raw_str) - len(kept_str)
        raw_str = kept_str
    text_str = redact_console_text_str(raw_str)
    if len(text_str) > LINE_CHARS_INT or dropped_int:
        hidden_int = max(len(text_str) - LINE_CHARS_INT, 0) + dropped_int
        text_str = text_str[:LINE_CHARS_INT] + f" … [+{hidden_int} chars]"
    return text_str


def parse_console_line_dict(line_str):
    """{"t": ISO UTC or None, "l": I/W/E/C/O, "a": action, "x": redacted remainder}."""
    line_str = _normalize_line_str(line_str)
    match_obj = HEADER_PATTERN_OBJ.match(line_str)
    level_code_str = LEVEL_CODE_DICT.get(match_obj.group(7)) if match_obj else None
    timestamp_str = None
    if level_code_str is not None:
        try:
            timestamp_str = datetime(*(int(part_str) for part_str in match_obj.group(1, 2, 3, 4, 5, 6)),
                tzinfo=timezone.utc).isoformat()
        except ValueError:
            level_code_str = None
    if level_code_str is None:
        return {"t": None, "l": "O", "a": "", "x": _display_text_str(line_str)}
    return {"t": timestamp_str, "l": level_code_str, "a": match_obj.group(8),
        "x": _display_text_str(match_obj.group(9) or "")}


def _marker_dict(message_str):
    return {"t": None, "l": "M", "a": "", "x": message_str}


def _file_id_str(stat_obj):
    return format(stat_obj.st_ino or stat_obj.st_ctime_ns, "x")


def _cursor_str(stat_obj, offset_int):
    return f"v1.{_file_id_str(stat_obj)}.{offset_int}"


def _parse_cursor_tuple(cursor_str):
    """("none"|"bad"|"ok", file_id_hex, offset). Anything not strictly v1 is "bad"."""
    if cursor_str is None or cursor_str == "":
        return "none", "", 0
    if not isinstance(cursor_str, str):
        return "bad", "", 0
    match_obj = CURSOR_PATTERN_OBJ.match(cursor_str)
    if match_obj is None:
        return "bad", "", 0
    return "ok", match_obj.group(1), int(match_obj.group(2))


def _mtime_str(stat_obj):
    try:
        return datetime.fromtimestamp(stat_obj.st_mtime_ns / 1e9, timezone.utc).replace(microsecond=0).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _read_exact_bytes(file_obj, offset_int, length_int):
    file_obj.seek(offset_int)
    data_bytes = file_obj.read(length_int)
    if len(data_bytes) != length_int:
        raise ConsoleReadError("Operator log shrank during read")
    return data_bytes


def _decode_line_str(line_bytes, start_bool):
    """UTF-8 with replacement; the BOM is only meaningful at file offset 0."""
    if start_bool and line_bytes.startswith(BOM_BYTES):
        line_bytes = line_bytes[len(BOM_BYTES):]
    return _normalize_line_str(line_bytes.decode("utf-8", errors="replace"))


def _tail_line_tuple(file_obj, size_int, pod_id_str, *, line_limit_int, scan_bytes_int):
    """Scan backwards from EOF for complete matching lines.

    Returns (line_str_list oldest first, end_offset_int after the last complete
    line, whole_file_bool). A trailing line without its newline is never read.
    """
    position_int, scanned_int, carry_bytes, end_int = size_int, 0, b"", None
    newest_first_list = []
    while position_int > 0 and scanned_int < scan_bytes_int and len(newest_first_list) < line_limit_int:
        read_int = min(CHUNK_BYTES_INT, position_int, scan_bytes_int - scanned_int)
        position_int -= read_int
        scanned_int += read_int
        buffer_bytes = _read_exact_bytes(file_obj, position_int, read_int) + carry_bytes
        part_list = buffer_bytes.split(b"\n")
        if end_int is None:
            if len(part_list) == 1:
                carry_bytes = buffer_bytes
                continue
            # The text after the last newline is still being written: hold it back.
            end_int = position_int + len(buffer_bytes) - len(part_list[-1])
            part_list = part_list[:-1]
        carry_bytes, complete_list = part_list[0], part_list[1:]
        if position_int == 0:
            complete_list.insert(0, carry_bytes)
            carry_bytes = b""
        for index_int in range(len(complete_list) - 1, -1, -1):
            line_str = _decode_line_str(complete_list[index_int], position_int == 0 and index_int == 0)
            if line_matches_pod_bool(line_str, pod_id_str):
                newest_first_list.append(line_str)
                if len(newest_first_list) >= line_limit_int:
                    break
    if end_int is None:
        # No newline at all: a whole-file partial line is held at 0. Only a
        # pathological >8 MiB line resumes at EOF (the forward reader resyncs).
        end_int = 0 if position_int == 0 else size_int
    newest_first_list.reverse()
    return newest_first_list, end_int, position_int == 0


def _forward_line_tuple(file_obj, offset_int, size_int, pod_id_str):
    """Read at most POLL_BYTES_INT forward. Returns (line_str_list, new_offset_int, more_bool)."""
    pending_int = size_int - offset_int
    read_int = min(POLL_BYTES_INT, pending_int)
    if read_int <= 0:
        return [], offset_int, False
    more_bool = read_int < pending_int
    # A cursor must sit on a line start. If not (oversized line, file rewritten
    # in place), skip the fragment instead of showing a line's tail.
    mid_line_bool = offset_int > 0 and _read_exact_bytes(file_obj, offset_int - 1, 1) != b"\n"
    data_bytes = _read_exact_bytes(file_obj, offset_int, read_int)
    start_int = 0
    if mid_line_bool:
        newline_int = data_bytes.find(b"\n")
        if newline_int < 0:
            return [], offset_int + (read_int if more_bool else 0), more_bool
        start_int = newline_int + 1
    end_int = data_bytes.rfind(b"\n") + 1
    if end_int > start_int:
        raw_list = data_bytes[start_int:end_int - 1].split(b"\n")
        new_offset_int = offset_int + end_int
    elif more_bool and start_int == 0:
        # One line longer than the poll window: show its start once, then resync.
        raw_list, new_offset_int = [data_bytes], offset_int + read_int
    else:
        raw_list, new_offset_int = [], offset_int + start_int
    line_list = []
    for index_int, line_bytes in enumerate(raw_list):
        line_str = _decode_line_str(line_bytes, offset_int == 0 and index_int == 0 and start_int == 0)
        if line_matches_pod_bool(line_str, pod_id_str):
            line_list.append(line_str)
    return line_list, new_offset_int, more_bool


def _empty_result_dict(cursor_str):
    return {"source_label_str": SOURCE_LABEL_STR, "cursor_str": cursor_str, "reset_bool": False,
        "more_bool": False, "gap_bytes_int": 0, "pending_bytes_int": 0, "file_size_int": 0,
        "last_write_utc_str": "", "note_str": "", "line_list": []}


def _no_lines_note_str(pod_id_str, whole_file_bool):
    subject_str = "No lines for this Pod" if pod_id_str else "No lines"
    if whole_file_bool:
        return f"{subject_str} in the operator log."
    return f"{subject_str} in the last {FIRST_SCAN_BYTES_INT // (1024 * 1024)} MB of the operator log."


def read_console_tail_dict(log_path_str, pod_id_str, cursor_str, *, peek_bool=False):
    """One bounded Console response; never raises for I/O problems.

    An empty cursor is a first load and reset_bool=True means the same thing
    for a cursor the client already had: replace the buffer with line_list.
    Otherwise line_list is appended. pending_bytes_int counts bytes after the
    returned cursor (normally a partial line or unread output).
    """
    status_str, file_id_str, offset_int = _parse_cursor_tuple(cursor_str)
    request_cursor_str = cursor_str if isinstance(cursor_str, str) else ""
    result_dict = _empty_result_dict(request_cursor_str)
    if peek_bool:
        try:
            stat_obj = os.stat(log_path_str)
        except FileNotFoundError:
            result_dict["note_str"] = "Operator log not found."
            return result_dict
        except (OSError, ValueError):
            result_dict["note_str"] = "Operator log could not be read."
            return result_dict
        size_int = stat_obj.st_size
        fits_bool = status_str == "ok" and file_id_str == _file_id_str(stat_obj) and offset_int <= size_int
        result_dict.update(file_size_int=size_int, last_write_utc_str=_mtime_str(stat_obj),
            pending_bytes_int=size_int - offset_int if fits_bool else size_int)
        return result_dict
    marker_list = []
    try:
        with open_shared_read_obj(log_path_str) as file_obj:
            stat_obj = os.fstat(file_obj.fileno())
            size_int = stat_obj.st_size
            result_dict.update(file_size_int=size_int, last_write_utc_str=_mtime_str(stat_obj))
            if status_str == "bad":
                result_dict["reset_bool"] = True
            elif status_str == "ok" and (file_id_str != _file_id_str(stat_obj) or offset_int > size_int):
                result_dict["reset_bool"] = True
                marker_list.append(_marker_dict("Log rotated or replaced; showing the latest lines."))
            elif status_str == "ok" and size_int - offset_int > JUMP_BYTES_INT:
                gap_int = size_int - offset_int
                result_dict.update(reset_bool=True, gap_bytes_int=gap_int)
                marker_list.append(_marker_dict(
                    f"Skipped {gap_int / (1024 * 1024):.1f} MB of older output. Use Download for more."))
            if status_str == "ok" and not result_dict["reset_bool"]:
                line_str_list, end_int, more_bool = _forward_line_tuple(file_obj, offset_int, size_int, pod_id_str)
                result_dict["more_bool"] = more_bool
            else:
                line_str_list, end_int, whole_file_bool = _tail_line_tuple(file_obj, size_int, pod_id_str,
                    line_limit_int=FIRST_LINE_LIMIT_INT, scan_bytes_int=FIRST_SCAN_BYTES_INT)
                if not line_str_list:
                    result_dict["note_str"] = _no_lines_note_str(pod_id_str, whole_file_bool)
            result_dict.update(cursor_str=_cursor_str(stat_obj, end_int), pending_bytes_int=size_int - end_int)
    except FileNotFoundError:
        # The client's position refers to a file that is gone: it must start over.
        result_dict.update(cursor_str="", reset_bool=status_str != "none", note_str="Operator log not found.")
        return result_dict
    except (OSError, ValueError):
        # Transient (sharing, antivirus, shrink mid-read): keep the client's cursor.
        result_dict["note_str"] = "Operator log could not be read."
        return result_dict
    room_int = RESPONSE_LINE_LIMIT_INT - len(marker_list)
    if len(line_str_list) > room_int:
        skipped_int = len(line_str_list) - (room_int - 1)
        line_str_list = line_str_list[-(room_int - 1):]
        marker_list.append(_marker_dict(f"Skipped {skipped_int} older lines. Use Download for more."))
    result_dict["line_list"] = marker_list + [parse_console_line_dict(line_str) for line_str in line_str_list]
    return result_dict


def build_console_download_str(log_path_str, pod_id_str):
    """Last <= 2 MiB of complete, filtered, fully redacted lines with a header."""
    header_str = redact_console_text_str(
        f"# Operator log · pod={pod_id_str or 'all'} · last {DOWNLOAD_BYTES_INT // (1024 * 1024)} MB · redacted")
    try:
        with open_shared_read_obj(log_path_str) as file_obj:
            size_int = os.fstat(file_obj.fileno()).st_size
            start_int = max(size_int - DOWNLOAD_BYTES_INT, 0)
            # One byte before the window tells whether its first line is complete.
            read_from_int = max(start_int - 1, 0)
            data_bytes = _read_exact_bytes(file_obj, read_from_int, size_int - read_from_int)
    except FileNotFoundError:
        return header_str + "\n# Operator log not found."
    except (OSError, ValueError):
        return header_str + "\n# Operator log could not be read."
    part_list = data_bytes.split(b"\n")[:-1]  # text after the last newline is not complete yet
    if start_int > 0:
        part_list = part_list[1:]  # fragment before the window (empty if it starts on a line)
    line_list = [header_str]
    for index_int, line_bytes in enumerate(part_list):
        line_str = _decode_line_str(line_bytes, start_int == 0 and index_int == 0)
        if line_matches_pod_bool(line_str, pod_id_str):
            line_list.append(redact_console_text_str(line_str))
    return "\n".join(line_list)
