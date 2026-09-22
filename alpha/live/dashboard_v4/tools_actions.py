"""Synthetic demo Tools actions with expiring one-use previews.

Production stays read-only. There is deliberately no real executor adapter or
production action flag in this module.
"""

from dataclasses import replace
from datetime import UTC, datetime
import json
import math
import secrets
import threading
from urllib.parse import urlsplit

from flask import jsonify, request, url_for

from alpha.live.dashboard_v3.actions import ConfirmationStore, SUPPORTED_ACTION_NAME_LIST, validate_action_request
from alpha.live.manual_order import MANUAL_ORDER_CONFIRMATION_TEXT_STR, build_manual_order_ticket_obj
from alpha.live.dashboard_v4.tools_execution import SyntheticToolsActionProvider, release_hash_str


class ToolsActionInFlightError(RuntimeError):
    pass


ACTION_NAME_SET = frozenset([*SUPPORTED_ACTION_NAME_LIST, "compare_reference", "manual_order"])
MANUAL_FIELD_SET = frozenset({"asset_str", "side_str", "broker_order_type_str", "quantity_int",
    "limit_price_float", "time_in_force_str", "operator_id_str", "reason_str", "confirmation_text_str"})
ACTION_DESCRIPTION_DICT = {
    "tick": "Run one scheduler tick. Due phases can create plans or submit orders under the existing release settings.",
    "submit_vplan": "Submit the saved eligible VPlan. This can send real orders to IBKR.",
    "post_execution_reconcile": "Read IBKR execution evidence and update saved positions and reconciliation.",
    "eod_snapshot": "Capture the eligible end-of-day account snapshot using the existing EOD rules.",
    "compare_reference": "Generate the existing reference comparison. This may update data caches and report files.",
    "manual_order": "Submit one manual broker ticket. This is separate from the strategy's target positions.",
}
STATUS_MESSAGE_DICT = {
    "queued": "Demo request accepted; waiting to start.", "running": "The simulation is running.",
    "succeeded": "Demo action completed. No real command or broker order was run.",
    "unknown": "Demo result unavailable. No production action was run.",
    "rejected": "Demo request rejected. No production action was run.",
}


class ToolsActionService:
    def __init__(self, provider_obj, *, target_scope_fn, enabled_bool=False, demo_bool=False,
                 now_fn=None):
        if enabled_bool and (not demo_bool or type(provider_obj) is not SyntheticToolsActionProvider):
            raise ValueError("Tools execution is available only in the synthetic demo.")
        self.provider_obj = provider_obj
        self.target_scope_fn = target_scope_fn
        self.enabled_bool = enabled_bool
        self.demo_bool = demo_bool
        self.now_fn = now_fn or (lambda: datetime.now(UTC))
        self.demo_event_list = []
        self.confirmation_store_obj = ConfirmationStore(ttl_seconds_float=120)
        self._action_token_str = ""
        self._job_dict = {}
        self._active_pod_set = set()
        self._lock_obj = threading.Lock()

    @property
    def action_token_str(self):
        if not self.allowed_bool:
            return ""
        with self._lock_obj:
            if not self._action_token_str:
                self._action_token_str = secrets.token_urlsafe(32)
            return self._action_token_str

    @property
    def allowed_bool(self):
        return self.enabled_bool and self.demo_bool and type(self.provider_obj) is SyntheticToolsActionProvider

    def _target_context_tuple(self, pod_id_str, action_name_str):
        if not self.allowed_bool:
            raise ValueError("Synthetic demo execution is disabled.")
        target_obj = self.target_scope_fn(pod_id_str)
        if (target_obj is None or target_obj.release_obj.pod_id_str != pod_id_str
                or target_obj.release_obj.mode_str != "live" or not target_obj.release_obj.enabled_bool):
            raise ValueError("Unknown enabled LIVE Pod.")
        context_dict = self.provider_obj.get_confirmation_context_dict(target_obj)
        return target_obj, {**context_dict, "action_name_str": action_name_str}

    def preview_dict(self, pod_id_str, action_name_str, body_dict):
        target_obj, context_dict = self._target_context_tuple(pod_id_str, action_name_str)
        preview_line_list = ["Demo only. No real command, broker request or trading state change.", ACTION_DESCRIPTION_DICT[action_name_str],
            "Pod: " + pod_id_str, "Account: …" + str(target_obj.release_obj.account_route_str)[-3:]]
        for key_str, label_str in (("decision_plan_id_int", "Decision"), ("vplan_id_int", "VPlan")):
            if isinstance(context_dict.get(key_str), int):
                preview_line_list.append(f"{label_str}: {context_dict[key_str]}")
        if action_name_str == "submit_vplan":
            vplan_id_int = body_dict.get("vplan_id_int")
            if type(vplan_id_int) is not int or not 1 <= vplan_id_int <= 2147483647:
                raise ValueError("A positive VPlan ID is required.")
            # This is the operator's simulated selection, not saved broker
            # evidence. Confirm takes it only from this one-use preview.
            context_dict["requested_vplan_id_int"] = vplan_id_int
            preview_line_list.append(f"Simulated VPlan ID: {vplan_id_int}")
        if action_name_str == "manual_order":
            manual_dict = body_dict.get("manual_order_dict")
            if not isinstance(manual_dict, dict) or set(manual_dict) - MANUAL_FIELD_SET:
                raise ValueError("Invalid manual ticket fields.")
            ticket_obj = build_manual_order_ticket_obj(release_obj=target_obj.release_obj,
                request_body_dict=manual_dict, submitted_timestamp_ts=datetime.now(UTC))
            if ticket_obj.limit_price_float is not None and (
                    not math.isfinite(ticket_obj.limit_price_float) or ticket_obj.limit_price_float <= 0):
                raise ValueError("A limit price must be finite and greater than zero.")
            # Store normalized values only. Confirm cannot replace the ticket.
            manual_dict = {key_str: getattr(ticket_obj, key_str) for key_str in MANUAL_FIELD_SET
                if key_str != "confirmation_text_str"}
            manual_dict["confirmation_text_str"] = MANUAL_ORDER_CONFIRMATION_TEXT_STR
            context_dict["manual_order_dict"] = manual_dict
            preview_line_list.extend([f"{ticket_obj.side_str} {ticket_obj.quantity_int} {ticket_obj.asset_str}",
                f"{ticket_obj.broker_order_type_str} · {ticket_obj.time_in_force_str}"])
            if ticket_obj.limit_price_float is not None:
                preview_line_list.append(f"Limit: {ticket_obj.limit_price_float:g}")
        nonce_str = self.confirmation_store_obj.issue(context_dict)
        return {"pod_id_str": pod_id_str, "action_name_str": action_name_str,
            "confirmation_nonce_str": nonce_str, "expires_in_seconds_int": 120,
            "preview_line_list": preview_line_list, "demo_bool": self.demo_bool}

    def _journal(self, job_dict, status_str):
        journal_dict = {"pod_id_str": job_dict["pod_id_str"], "mode_str": "live",
            "action_name_str": job_dict["action_name_str"], "job_id_str": job_dict["job_id_str"],
            "initial_status_str": status_str}
        self.provider_obj.journal_list.append(journal_dict)
        self.demo_event_list.append({"timestamp_str": self.now_fn().isoformat(),
            "pod_id_str": job_dict["pod_id_str"], "event_type_str": "operator_demo_action",
            "level_str": "INFO", "source_str": "Synthetic Tools demo",
            "payload_dict": {"action_name_str": job_dict["action_name_str"],
                "job_id_str": job_dict["job_id_str"], "status_str": "simulated_" + status_str}})
        del self.provider_obj.journal_list[:-400]
        del self.demo_event_list[:-400]

    def confirm_tuple(self, pod_id_str, action_name_str, nonce_str):
        expected_dict = self.confirmation_store_obj.consume(nonce_str, pod_id_str, action_name_str)
        target_obj, current_dict = self._target_context_tuple(pod_id_str, action_name_str)
        if any(value_obj != expected_dict.get(key_str) for key_str, value_obj in current_dict.items()):
            raise ValueError("Target or saved execution state changed. Open a new preview.")
        target_obj = replace(target_obj, operator_confirmation_dict=expected_dict)
        job_id_str = secrets.token_hex(16)
        job_dict = {"job_id_str": job_id_str, "pod_id_str": pod_id_str, "action_name_str": action_name_str,
            "status_str": "queued", "upstream_id_str": None, "context_dict": current_dict,
            "terminal_journal_bool": False}
        with self._lock_obj:
            if pod_id_str in self._active_pod_set:
                raise ToolsActionInFlightError("A Tools request is already running for this Pod.")
            self._active_pod_set.add(pod_id_str)
            # Keep request history bounded; active requests are never discarded.
            if len(self._job_dict) >= 200:
                for old_id_str, old_dict in list(self._job_dict.items()):
                    if old_dict["status_str"] not in {"queued", "running"}:
                        del self._job_dict[old_id_str]
                        break
                else:
                    self._active_pod_set.discard(pod_id_str)
                    raise ToolsActionInFlightError("Tools history is busy.")
            self._job_dict[job_id_str] = job_dict
        try:
            # Journal failure before dispatch must prevent an unrecorded action.
            self._journal(job_dict, "requested")
        except Exception:
            job_dict["status_str"] = "rejected"
            self._active_pod_set.discard(pod_id_str)
            return self.public_job_dict(job_dict), 503
        status_int = 202
        try:
            if action_name_str == "manual_order":
                result_dict = self.provider_obj.submit_manual_order_dict(target_obj, expected_dict["manual_order_dict"])
                job_dict["status_str"] = "succeeded" if result_dict.get("submit_ack_status_str") == "acknowledged" else "unknown"
            else:
                result_dict = (self.provider_obj.start_diff_job(target_obj) if action_name_str == "compare_reference"
                    else self.provider_obj.start_action_job(action_name_str, target_obj))
                if result_dict.get("pod_id_str") != pod_id_str or not result_dict.get("job_id_str"):
                    raise ValueError("Dispatch result does not identify the request.")
                job_dict["upstream_id_str"] = result_dict["job_id_str"]
                job_dict["status_str"] = self._status_str(result_dict)
        except ToolsActionInFlightError:
            job_dict["status_str"], status_int = "rejected", 409
        except Exception:
            job_dict["status_str"], status_int = "unknown", 503
        try:
            self._journal(job_dict, job_dict["status_str"])
            job_dict["terminal_journal_bool"] = job_dict["status_str"] not in {"queued", "running"}
        except Exception:
            job_dict["status_str"], status_int = "unknown", 503
        if job_dict["status_str"] not in {"queued", "running"}:
            self._active_pod_set.discard(pod_id_str)
        return self.public_job_dict(job_dict), status_int

    @staticmethod
    def _status_str(result_dict):
        # A failed runner may already have reached the broker. Never claim that
        # failure means no side effects, or that success proves a fill.
        status_str = result_dict.get("status_str")
        return status_str if status_str in {"queued", "running", "succeeded"} else "unknown"

    def job_dict(self, pod_id_str, job_id_str):
        if not self.allowed_bool:
            return None
        target_obj = self.target_scope_fn(pod_id_str)
        job_dict = self._job_dict.get(job_id_str)
        if job_dict is None or job_dict["pod_id_str"] != pod_id_str or target_obj is None:
            return None
        if (target_obj.release_obj.mode_str != "live" or not target_obj.release_obj.enabled_bool
                or target_obj.db_path_str != job_dict["context_dict"]["db_path_str"]
                or release_hash_str(target_obj) != job_dict["context_dict"]["release_hash_str"]):
            return None
        if job_dict["upstream_id_str"] and job_dict["status_str"] in {"queued", "running"}:
            result_dict = self.provider_obj.get_job_dict(job_dict["upstream_id_str"])
            if result_dict is None or result_dict.get("pod_id_str") != pod_id_str:
                job_dict["status_str"] = "unknown"
            else:
                job_dict["status_str"] = self._status_str(result_dict)
        if job_dict["status_str"] not in {"queued", "running"}:
            self._active_pod_set.discard(pod_id_str)
            if not job_dict["terminal_journal_bool"]:
                try:
                    self._journal(job_dict, job_dict["status_str"])
                    job_dict["terminal_journal_bool"] = True
                except Exception:
                    job_dict["status_str"] = "unknown"
        return self.public_job_dict(job_dict)

    def public_job_dict(self, job_dict):
        # Deliberate allowlist: never send executor errors, tracebacks, account
        # routes, local paths, raw broker payloads or arbitrary report output.
        return {key_str: job_dict[key_str] for key_str in ("job_id_str", "pod_id_str", "action_name_str", "status_str")} | {
            "message_str": "Simulated only · " + STATUS_MESSAGE_DICT[job_dict["status_str"]],
            "demo_bool": self.demo_bool}


def _body_dict(allowed_set):
    if request.args or request.mimetype != "application/json":
        raise ValueError("Use a JSON body without query parameters.")
    if request.content_length is not None and request.content_length > 16384:
        raise ValueError("Request body is too large.")
    def unique_dict(pair_list):
        result_dict = {}
        for key_str, value_obj in pair_list:
            if key_str in result_dict:
                raise ValueError("Duplicate request fields.")
            result_dict[key_str] = value_obj
        return result_dict
    def invalid_constant(value_str):
        raise ValueError("Non-finite request value.")
    raw_bytes = request.stream.read(16385)
    if len(raw_bytes) > 16384:
        raise ValueError("Request body is too large.")
    body_dict = json.loads(raw_bytes, object_pairs_hook=unique_dict, parse_constant=invalid_constant)
    if not isinstance(body_dict, dict) or set(body_dict) - allowed_set or body_dict.get("confirmed_bool") is not True:
        raise ValueError("Unexpected fields or missing explicit confirmation.")
    return body_dict


def register_tools_action_routes(app_obj, service_obj):
    def guarded_body(action_name_str, allowed_set):
        if not service_obj.allowed_bool:
            return None, (jsonify(error="actions_disabled", message="Tools execution is disabled."), 403)
        if action_name_str not in ACTION_NAME_SET:
            return None, (jsonify(error="unsupported_action", message="Unknown tool."), 400)
        try:
            body_dict = _body_dict(allowed_set)
        except (ValueError, UnicodeError):
            return None, (jsonify(error="invalid_body", message="Invalid or ambiguous request fields."), 400)
        rejection_tuple = validate_action_request(request.headers, body_dict, service_obj.action_token_str)
        # V3 compares hosts; additionally bind scheme to reject HTTP -> HTTPS.
        origin_tuple = urlsplit(request.host_url)
        for key_str in ("Origin", "Referer"):
            if request.headers.get(key_str):
                source_tuple = urlsplit(request.headers[key_str])
                if (source_tuple.scheme, source_tuple.netloc.lower()) != (origin_tuple.scheme, origin_tuple.netloc.lower()):
                    rejection_tuple = (403, "origin_rejected", "A same-origin request is required.")
        return (None, (jsonify(error=rejection_tuple[1], message=rejection_tuple[2]), rejection_tuple[0])) if rejection_tuple else (body_dict, None)

    def with_poll_url(result_dict):
        result_dict["poll_url_str"] = url_for("tools_action_job", pod_id_str=result_dict["pod_id_str"], job_id_str=result_dict["job_id_str"])
        return result_dict

    @app_obj.post("/api/demo-tools/<pod_id_str>/<action_name_str>/preview", endpoint="tools_action_preview")
    def preview(pod_id_str, action_name_str):
        allowed_set = ({"confirmed_bool", "manual_order_dict"} if action_name_str == "manual_order" else
            {"confirmed_bool", "vplan_id_int"} if action_name_str == "submit_vplan" else {"confirmed_bool"})
        body_dict, error_obj = guarded_body(action_name_str, allowed_set)
        if error_obj is not None:
            return error_obj
        try:
            return jsonify(service_obj.preview_dict(pod_id_str, action_name_str, body_dict))
        except Exception:
            return jsonify(error="preview_unavailable", message="The target or saved evidence could not be verified. Check the selected Pod and ticket fields."), 409

    @app_obj.post("/api/demo-tools/<pod_id_str>/<action_name_str>/confirm", endpoint="tools_action_confirm")
    def confirm(pod_id_str, action_name_str):
        body_dict, error_obj = guarded_body(action_name_str, {"confirmed_bool", "confirmation_nonce_str", "browser_confirmed_bool"})
        if error_obj is not None:
            return error_obj
        if body_dict.get("browser_confirmed_bool") is not True:
            return jsonify(error="browser_confirmation_required", message="Confirm the Are you sure prompt before running this action."), 400
        try:
            result_dict, status_int = service_obj.confirm_tuple(pod_id_str, action_name_str, body_dict.get("confirmation_nonce_str"))
            return jsonify(with_poll_url(result_dict)), status_int
        except ToolsActionInFlightError:
            return jsonify(error="action_in_flight", message="A request is already running for this Pod. Check its result before retrying."), 409
        except Exception:
            return jsonify(error="confirmation_rejected", message="Confirmation expired, was used, or the target changed. Open a new preview."), 409

    @app_obj.post("/api/demo-tools/<pod_id_str>/<action_name_str>/cancel", endpoint="tools_action_cancel")
    def cancel(pod_id_str, action_name_str):
        body_dict, error_obj = guarded_body(action_name_str, {"confirmed_bool"})
        if error_obj is not None:
            return error_obj
        try:
            target_obj = service_obj.target_scope_fn(pod_id_str)
            if target_obj is None or target_obj.release_obj.mode_str != "live" or not target_obj.release_obj.enabled_bool:
                raise ValueError("Unknown enabled LIVE Pod.")
            service_obj.confirmation_store_obj.cancel(pod_id_str)
        except Exception:
            return jsonify(error="target_unavailable", message="The selected Pod is unavailable."), 409
        return jsonify(cancelled_bool=True)

    @app_obj.get("/api/demo-tools/<pod_id_str>/jobs/<job_id_str>", endpoint="tools_action_job")
    def job(pod_id_str, job_id_str):
        if not service_obj.allowed_bool:
            return jsonify(error="actions_disabled", message="Tools execution is disabled."), 403
        if request.args:
            return jsonify(error="invalid_parameters", message="Unexpected query parameters."), 400
        try:
            result_dict = service_obj.job_dict(pod_id_str, job_id_str)
        except Exception:
            result_dict = None
        if result_dict is None:
            return jsonify(error="job_unavailable", message="Job unavailable for this Pod. Inspect saved evidence before retrying."), 404
        return jsonify(with_poll_url(result_dict))
