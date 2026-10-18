"""Operator marks for backtest differences already reviewed.

One small JSON file next to the backtest check report. A mark names one pod's
one decision (release, decision plan, session): that difference leaves the
Overview, and a new decision that differs shows again. The report, the Pod page
and Vs backtest keep every difference. Nothing here touches orders or pod state.
"""

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import secrets
import threading

from alpha.live.dashboard_v4.backtest_check_data import _report_path

REVIEW_FILE_NAME_STR = "backtest_reviewed.json"
REVIEW_LIMIT_INT = 200
REVIEW_BYTE_LIMIT_INT = 256 * 1024


class BacktestReviewStore:
    """File-backed for LIVE; in memory for the demo (its log paths are not real)."""

    def __init__(self, provider_obj, *, memory_bool=False):
        self._provider_obj = provider_obj
        self._memory_bool = memory_bool
        self._memory_list = []
        self._lock_obj = threading.Lock()
        self.action_token_str = secrets.token_urlsafe(32)

    def _path_obj(self):
        report_path_obj = None if self._memory_bool else _report_path(self._provider_obj)
        return report_path_obj.parent / REVIEW_FILE_NAME_STR if report_path_obj is not None else None

    def _entry_list(self):
        path_obj = self._path_obj()
        if path_obj is None:
            return list(self._memory_list)
        try:
            if path_obj.stat().st_size > REVIEW_BYTE_LIMIT_INT:
                return []
            payload_dict = json.loads(path_obj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        entry_list = payload_dict.get("reviewed_list") if isinstance(payload_dict, dict) else None
        return [entry_dict for entry_dict in entry_list or [] if isinstance(entry_dict, dict) and isinstance(entry_dict.get("key_str"), str)]

    def reviewed_key_set(self):
        return {entry_dict["key_str"] for entry_dict in self._entry_list()}

    def mark(self, key_str, *, pod_id_str, now_ts=None):
        entry_dict = {"key_str": key_str, "pod_id_str": pod_id_str,
                      "reviewed_at_utc_str": (now_ts or datetime.now(UTC)).astimezone(UTC).isoformat()}
        with self._lock_obj:
            entry_list = [item_dict for item_dict in self._entry_list() if item_dict["key_str"] != key_str]
            entry_list = (entry_list + [entry_dict])[-REVIEW_LIMIT_INT:]
            path_obj = self._path_obj()
            if path_obj is None:
                self._memory_list = entry_list
                return
            temporary_path_obj = Path(str(path_obj) + ".tmp")
            temporary_path_obj.write_text(json.dumps({"reviewed_list": entry_list}, indent=1), encoding="utf-8")
            os.replace(temporary_path_obj, path_obj)


def register_review_route(app_obj, store_obj, current_key_set_fn, *, now_fn):
    """POST one mark: same-origin, server token, explicit confirmation, and only a difference the report shows now."""
    from urllib.parse import urlsplit

    from flask import jsonify, request

    from alpha.live.dashboard_v3.actions import validate_action_request

    @app_obj.post("/api/pods/<pod_id_str>/backtest/reviewed", endpoint="backtest_reviewed")
    def backtest_reviewed(pod_id_str):
        if request.args or request.mimetype != "application/json" or (request.content_length or 0) > 2048:
            return jsonify(error="invalid_body", message="Invalid request."), 400
        body_dict = request.get_json(silent=True)
        if not isinstance(body_dict, dict) or set(body_dict) - {"confirmed_bool", "key_str"}:
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
        key_str = body_dict.get("key_str")
        if not isinstance(key_str, str) or key_str not in current_key_set_fn(pod_id_str):
            return jsonify(error="not_current", message="This difference is no longer in the report. Refresh the page."), 409
        try:
            store_obj.mark(key_str, pod_id_str=pod_id_str, now_ts=now_fn())
        except OSError:
            return jsonify(error="write_failed", message="Could not save. Try again."), 500
        return jsonify(status_str="ok")
