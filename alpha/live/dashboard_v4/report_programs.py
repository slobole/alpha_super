"""Programs for the monthly client report: the program each Pod belongs to.

One small JSON file next to the Live Ops event log, written only from the
report's Programs panel. Display only: nothing here touches orders, releases
or pod state. A Pod without a program shows under "Portfolio".
"""

import json
import os
from pathlib import Path
import re
import secrets
import threading
import time


PROGRAM_FILE_NAME_STR = "report_programs.json"
PROGRAM_BYTE_LIMIT_INT = 64 * 1024
PROGRAM_LIMIT_INT = 200
REQUEST_BYTE_LIMIT_INT = 8192
SUGGESTED_PROGRAM_TUPLE = ("Steady", "Growth")
DEFAULT_PROGRAM_STR = "Portfolio"
# A program name is short client-facing text: letters, digits, spaces and a few marks.
PROGRAM_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 &+.'/-]{0,29}")
REPLACE_ATTEMPT_INT = 5


def clean_program_str(value_obj):
    """A valid program name with single spaces, "" to clear a choice, or None when invalid.

    Case never makes a second program: a suggested name keeps its own spelling
    and "Portfolio" in any case means no program.
    """
    if not isinstance(value_obj, str):
        return None
    program_str = " ".join(value_obj.split())
    if not program_str or program_str.casefold() == DEFAULT_PROGRAM_STR.casefold():
        return ""
    if not PROGRAM_NAME_RE.fullmatch(program_str):
        return None
    return next((name_str for name_str in SUGGESTED_PROGRAM_TUPLE if name_str.casefold() == program_str.casefold()), program_str)


class ReportProgramStore:
    """File-backed for LIVE; in memory for the demo (its log paths are not real)."""

    def __init__(self, provider_obj, *, memory_bool=False):
        self._provider_obj = provider_obj
        self._memory_bool = memory_bool
        self._memory_dict = {}
        self._lock_obj = threading.Lock()
        self.action_token_str = secrets.token_urlsafe(32)

    def _path_obj(self):
        if self._memory_bool:
            return None
        for source_obj in (self._provider_obj, getattr(self._provider_obj, "app_obj", lambda: None)()):
            path_str = getattr(source_obj, "event_log_path_str", None) if source_obj is not None else None
            if isinstance(path_str, str) and path_str:
                return Path(path_str).parent / PROGRAM_FILE_NAME_STR
        return None

    def program_by_pod_dict(self):
        """{pod id: program name}; a missing, oversized, unreadable or foreign file is no choices."""
        path_obj = self._path_obj()
        if path_obj is None:
            return dict(self._memory_dict)
        try:
            if path_obj.stat().st_size > PROGRAM_BYTE_LIMIT_INT:
                return {}
            payload_dict = json.loads(path_obj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        program_dict = payload_dict.get("program_by_pod_dict") if isinstance(payload_dict, dict) else None
        if not isinstance(program_dict, dict):
            return {}
        clean_dict = {pod_id_str: clean_program_str(program_str) for pod_id_str, program_str in program_dict.items() if isinstance(pod_id_str, str)}
        return {pod_id_str: program_str for pod_id_str, program_str in clean_dict.items() if program_str}

    def save(self, update_dict):
        """Set each Pod's program; "" clears it. Pods not named keep their choice."""
        with self._lock_obj:
            program_dict = self.program_by_pod_dict()
            for pod_id_str, program_str in update_dict.items():
                program_dict.pop(pod_id_str, None)
                if program_str:
                    program_dict[pod_id_str] = program_str
            program_dict = dict(list(program_dict.items())[-PROGRAM_LIMIT_INT:])
            path_obj = self._path_obj()
            if path_obj is None:
                self._memory_dict = program_dict
                return
            temporary_path_obj = Path(str(path_obj) + ".tmp")
            temporary_path_obj.write_text(json.dumps({"program_by_pod_dict": program_dict}, indent=1, sort_keys=True), encoding="utf-8")
            # Windows refuses the swap while a reader holds the file open; that lasts milliseconds.
            for attempt_int in range(REPLACE_ATTEMPT_INT):
                try:
                    os.replace(temporary_path_obj, path_obj)
                    return
                except PermissionError:
                    if attempt_int == REPLACE_ATTEMPT_INT - 1:
                        temporary_path_obj.unlink(missing_ok=True)
                        raise
                    time.sleep(.05 * (attempt_int + 1))


def register_program_route(app_obj, store_obj, current_pod_set_fn):
    """POST the panel's choices: same-origin, server token, confirmation, and only Pods the report shows."""
    from urllib.parse import urlsplit

    from flask import jsonify, request

    from alpha.live.dashboard_v3.actions import validate_action_request

    @app_obj.post("/monthly-report/programs", endpoint="monthly_report_programs")
    def monthly_report_programs():
        # A declared, small body only: a chunked request has no length to check before parsing.
        if (request.args or request.mimetype != "application/json" or request.content_length is None
                or request.content_length > REQUEST_BYTE_LIMIT_INT):
            return jsonify(error="invalid_body", message="Invalid request."), 400
        body_dict = request.get_json(silent=True)
        if not isinstance(body_dict, dict) or set(body_dict) - {"confirmed_bool", "program_by_pod_dict"}:
            return jsonify(error="invalid_body", message="Invalid request."), 400
        rejection_tuple = validate_action_request(request.headers, body_dict, store_obj.action_token_str)
        origin_tuple = urlsplit(request.host_url)
        for header_str in ("Origin", "Referer"):
            if request.headers.get(header_str):
                source_tuple = urlsplit(request.headers[header_str])
                if (source_tuple.scheme, source_tuple.netloc.lower()) != (origin_tuple.scheme, origin_tuple.netloc.lower()):
                    rejection_tuple = (403, "origin_rejected", "A same-origin request is required.")
        if rejection_tuple:
            return jsonify(error=rejection_tuple[1], message=rejection_tuple[2]), rejection_tuple[0]
        update_dict = body_dict.get("program_by_pod_dict")
        if not isinstance(update_dict, dict) or not update_dict or len(update_dict) > PROGRAM_LIMIT_INT:
            return jsonify(error="invalid_body", message="Invalid request."), 400
        pod_id_set = current_pod_set_fn()
        if not pod_id_set:
            return jsonify(error="accounts_unreadable", message="The saved accounts could not be read. Refresh the page and try again."), 503
        if set(update_dict) - pod_id_set:
            return jsonify(error="not_current", message="A Pod is no longer in the report. Refresh the page."), 409
        clean_dict = {pod_id_str: clean_program_str(program_str) for pod_id_str, program_str in update_dict.items()}
        if any(program_str is None for program_str in clean_dict.values()):
            return jsonify(error="invalid_name", message="Use up to 30 letters, digits or spaces for a program name."), 400
        try:
            store_obj.save(clean_dict)
        except FileNotFoundError:
            return jsonify(error="folder_missing", message="The Live Ops log folder is missing, so programs cannot be saved."), 500
        except OSError:
            return jsonify(error="write_failed", message="Could not save. Try again."), 500
        return jsonify(status_str="ok")
