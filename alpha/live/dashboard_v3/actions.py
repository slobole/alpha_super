"""Server-side action request validation for Dashboard V3.

Same security model as V2: every action request must be JSON, originate
from the dashboard's own host (Origin/Referer match Host), present a
server-issued ``X-Alpha-Action-Token`` header, and explicitly contain
``"confirmed_bool": true`` in its body. Any missing piece returns an
error code; the view is the operator pressing the wrong button by
mistake, not a CSRF attacker on the open internet (the dashboard is
Tailscale-only) — but the belt-and-braces protections are cheap so we
keep them.
"""

from __future__ import annotations

import secrets
from urllib.parse import urlparse


ACTION_TOKEN_HEADER_STR = "X-Alpha-Action-Token"
SUPPORTED_ACTION_NAME_LIST = [
    "tick",
    "submit_vplan",
    "post_execution_reconcile",
    "eod_snapshot",
]


def request_header_matches_host_bool(header_value_str: str, host_str: str) -> bool:
    if host_str == "":
        return False
    parsed_obj = urlparse(header_value_str)
    return parsed_obj.netloc.lower() == host_str.lower()


def action_request_origin_valid_bool(request_headers_obj) -> bool:
    content_type_str = ""
    raw_content_type_str = request_headers_obj.get("Content-Type", "")
    if raw_content_type_str:
        content_type_str = raw_content_type_str.split(";")[0].strip().lower()
    if content_type_str != "application/json":
        return False
    host_str = request_headers_obj.get("Host", "")
    origin_seen_bool = False
    for header_name_str in ("Origin", "Referer"):
        header_value_str = request_headers_obj.get(header_name_str)
        if header_value_str:
            origin_seen_bool = True
            if not request_header_matches_host_bool(header_value_str, host_str):
                return False
    return origin_seen_bool


def action_token_valid_bool(request_token_str: str | None, expected_token_str: str) -> bool:
    return secrets.compare_digest(str(request_token_str or ""), str(expected_token_str))


def confirmation_present_bool(body_dict: dict | None) -> bool:
    return bool(body_dict and body_dict.get("confirmed_bool") is True)


def validate_action_request(
    request_headers_obj,
    body_dict: dict | None,
    expected_token_str: str,
) -> tuple[int, str, str] | None:
    """Returns ``(status_int, error_code_str, message_str)`` on rejection,
    or ``None`` if the request is acceptable."""
    if not action_request_origin_valid_bool(request_headers_obj):
        return (403, "origin_rejected", "Dashboard actions require a same-origin JSON POST.")
    request_token_str = request_headers_obj.get(ACTION_TOKEN_HEADER_STR, "")
    if not action_token_valid_bool(request_token_str, expected_token_str):
        return (403, "action_token_required", "Dashboard actions require a server-issued action token.")
    if not confirmation_present_bool(body_dict):
        return (400, "confirmation_required", "Dashboard actions require confirmed_bool=true.")
    return None
