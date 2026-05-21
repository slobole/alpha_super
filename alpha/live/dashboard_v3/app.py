"""Flask application factory for Dashboard V3.

Phase 1 wires the read-only operator console: three mode pages
(``/live``, ``/paper``, ``/incubation``), an HTMX-polled top bar
verdict, an expandable per-pod detail panel rendered as a vertical
timeline of today's cycle, and a polled events tail per pod.

The factory accepts an injectable ``data_provider_obj`` so tests can
swap in a fixture instead of touching the live ``DashboardApp``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, url_for

from alpha.live.dashboard_v3.actions import (
    SUPPORTED_ACTION_NAME_LIST,
    validate_action_request,
)
from alpha.live.dashboard_v3.charts import (
    SUPPORTED_WINDOW_STR_LIST,
    build_equity_chart_dict,
)
from alpha.live.dashboard_v3.data import (
    DashboardActionInFlightError,
    DashboardDataProvider,
    get_pod_row_dict_by_id,
    get_pod_row_dict_list_for_mode,
)
from alpha.live.dashboard_v3.expected_pnl import (
    DEFAULT_EXPECTED_PNL_PATH_STR,
    build_tracking_comparison,
    load_expected_pnl_map,
)
from alpha.live.dashboard_v3.filters import FILTER_MAP_DICT
from alpha.live.dashboard_v3.health import build_health_rollup
from alpha.live.dashboard_v3.journal import (
    DEFAULT_JOURNAL_PATH_STR,
    append_journal_entry,
    read_journal_entry_dict_list,
)
from alpha.live.dashboard_v3.notifications import (
    DEFAULT_NOTIFICATION_STATE_PATH_STR,
    NotificationStateStore,
    check_and_notify_for_red_transitions,
    discord_webhook_url_from_env_str,
    post_discord_webhook_bool,
)
from alpha.live.dashboard_v3.schedule import build_schedule_entry_list
from alpha.live.dashboard_v3.verdict import resolve_top_bar_verdict


DASHBOARD_V3_VERSION_STR = "0.6.0-phase-6"
ALL_ACTION_NAME_LIST = ["compare_reference"] + list(SUPPORTED_ACTION_NAME_LIST)
ACTION_LABEL_DICT = {
    "compare_reference": "DIFF compare",
    "tick": "Tick scheduler",
    "submit_vplan": "Submit VPlan",
    "post_execution_reconcile": "Post-execution reconcile",
    "eod_snapshot": "EOD snapshot",
}
SUPPORTED_MODE_STR_LIST = ["live", "paper", "incubation"]
MODE_LABEL_DICT = {"live": "Live", "paper": "Paper", "incubation": "Incubation"}


class DataProviderProtocol(Protocol):
    def get_summary_dict(self) -> dict[str, Any]: ...
    def get_pod_detail_dict(self, pod_id_str: str) -> dict[str, Any]: ...
    def get_pod_event_dict_list(
        self, pod_id_str: str, limit_int: int = 80
    ) -> list[dict[str, Any]]: ...


def create_app(
    data_provider_obj: DataProviderProtocol | None = None,
    *,
    journal_path_str: str = DEFAULT_JOURNAL_PATH_STR,
    expected_pnl_path_str: str = DEFAULT_EXPECTED_PNL_PATH_STR,
    notification_state_path_str: str = DEFAULT_NOTIFICATION_STATE_PATH_STR,
    notification_webhook_url_str: str | None = None,
    notification_webhook_poster_fn=None,
) -> Flask:
    flask_app_obj = Flask(__name__)
    flask_app_obj.config["data_provider_obj"] = (
        data_provider_obj if data_provider_obj is not None else DashboardDataProvider()
    )
    flask_app_obj.config["journal_path_str"] = journal_path_str
    flask_app_obj.config["expected_pnl_path_str"] = expected_pnl_path_str
    flask_app_obj.config["notification_state_store_obj"] = NotificationStateStore(
        state_path_str=notification_state_path_str
    )
    flask_app_obj.config["notification_webhook_url_str"] = (
        notification_webhook_url_str
        if notification_webhook_url_str is not None
        else discord_webhook_url_from_env_str()
    )
    flask_app_obj.config["notification_webhook_poster_fn"] = (
        notification_webhook_poster_fn or post_discord_webhook_bool
    )

    for filter_name_str, filter_fn in FILTER_MAP_DICT.items():
        flask_app_obj.jinja_env.filters[filter_name_str] = filter_fn

    @flask_app_obj.context_processor
    def inject_globals_fn() -> dict[str, Any]:
        return {
            "version_str": DASHBOARD_V3_VERSION_STR,
            "server_time_str": _now_clock_str(),
        }

    # ── plain routes ─────────────────────────────────────────────────────

    @flask_app_obj.route("/healthz")
    def healthz_route_fn() -> tuple[str, int]:
        return (f"dashboard_v3 ok {DASHBOARD_V3_VERSION_STR}", 200)

    @flask_app_obj.route("/")
    def index_route_fn() -> Response:
        return redirect(url_for("mode_page_route_fn", mode_str="live"))

    @flask_app_obj.route("/<mode_str>")
    def mode_page_route_fn(mode_str: str):
        if mode_str not in SUPPORTED_MODE_STR_LIST:
            abort(404)
        provider_obj = flask_app_obj.config["data_provider_obj"]
        summary_dict = provider_obj.get_summary_dict()
        pod_row_dict_list = get_pod_row_dict_list_for_mode(summary_dict, mode_str)
        attention_row_list = [
            row_dict
            for row_dict in pod_row_dict_list
            if _is_attention_row_bool(row_dict)
        ]
        verdict_obj = resolve_top_bar_verdict(summary_dict)
        combined_book_env_dict = _find_combined_book_environment_dict(summary_dict, mode_str)
        combined_book_chart_dict = None
        if combined_book_env_dict is not None:
            chart_obj = build_equity_chart_dict(
                combined_book_env_dict.get("equity_point_dict_list")
                or combined_book_env_dict.get("carry_forward_equity_point_dict_list"),
                window_str="all",
            )
            if chart_obj.point_count_int > 0:
                combined_book_chart_dict = chart_obj.as_dict()
        return render_template(
            "mode_page.html",
            mode_str=mode_str,
            mode_label_str=MODE_LABEL_DICT[mode_str],
            pod_row_dict_list=pod_row_dict_list,
            attention_row_list=attention_row_list,
            verdict_dict=verdict_obj.as_dict(),
            as_of_clock_str=_now_clock_str(),
            combined_book_chart_dict=combined_book_chart_dict,
            combined_book_summary_dict=combined_book_env_dict or {},
        )

    # ── HTMX fragments ───────────────────────────────────────────────────

    @flask_app_obj.route("/fragments/top-bar")
    def top_bar_fragment_route_fn():
        provider_obj = flask_app_obj.config["data_provider_obj"]
        summary_dict = provider_obj.get_summary_dict()
        verdict_obj = resolve_top_bar_verdict(summary_dict)
        check_and_notify_for_red_transitions(
            summary_dict,
            state_store_obj=flask_app_obj.config["notification_state_store_obj"],
            webhook_url_str=flask_app_obj.config["notification_webhook_url_str"],
            webhook_poster_fn=flask_app_obj.config["notification_webhook_poster_fn"],
        )
        return render_template(
            "_top_bar_verdict.html",
            verdict_dict=verdict_obj.as_dict(),
            as_of_clock_str=_now_clock_str(),
        )

    @flask_app_obj.route("/fragments/pod-detail/<pod_id_str>")
    def pod_detail_fragment_route_fn(pod_id_str: str):
        provider_obj = flask_app_obj.config["data_provider_obj"]
        try:
            detail_dict = provider_obj.get_pod_detail_dict(pod_id_str)
        except KeyError:
            abort(404)
        expected_pnl_map_dict = load_expected_pnl_map(
            flask_app_obj.config["expected_pnl_path_str"]
        )
        pnl_dict = detail_dict.get("pod_pnl_dict") or {}
        tracking_obj = build_tracking_comparison(
            pod_id_str,
            pnl_dict.get("daily_pnl_pct_float"),
            expected_pnl_map_dict,
        )
        return render_template(
            "_pod_detail.html",
            detail_dict=detail_dict,
            tracking_dict=tracking_obj.as_dict(),
            as_of_clock_str=_now_clock_str(),
        )

    @flask_app_obj.route("/fragments/events-tail/<pod_id_str>")
    def events_tail_fragment_route_fn(pod_id_str: str):
        provider_obj = flask_app_obj.config["data_provider_obj"]
        event_dict_list = provider_obj.get_pod_event_dict_list(pod_id_str)
        return render_template(
            "_events_tail.html",
            event_dict_list=event_dict_list,
            pod_id_str=pod_id_str,
        )

    @flask_app_obj.route("/fragments/health-strip")
    def health_strip_fragment_route_fn():
        provider_obj = flask_app_obj.config["data_provider_obj"]
        summary_dict = provider_obj.get_summary_dict()
        health_obj = build_health_rollup(summary_dict)
        return render_template(
            "_health_strip.html",
            health_dict=health_obj.as_dict(),
        )

    @flask_app_obj.route("/fragments/schedule-strip")
    def schedule_strip_fragment_route_fn():
        provider_obj = flask_app_obj.config["data_provider_obj"]
        summary_dict = provider_obj.get_summary_dict()
        schedule_entry_obj_list = build_schedule_entry_list(summary_dict)
        return render_template(
            "_schedule_strip.html",
            schedule_entry_dict_list=[entry_obj.as_dict() for entry_obj in schedule_entry_obj_list],
        )

    @flask_app_obj.route("/fragments/equity-chart/<pod_id_str>")
    def equity_chart_fragment_route_fn(pod_id_str: str):
        window_str = request.args.get("window", "all")
        if window_str not in SUPPORTED_WINDOW_STR_LIST:
            window_str = "all"
        provider_obj = flask_app_obj.config["data_provider_obj"]
        try:
            detail_dict = provider_obj.get_pod_detail_dict(pod_id_str)
        except KeyError:
            abort(404)
        pnl_dict = detail_dict.get("pod_pnl_dict") or {}
        chart_obj = build_equity_chart_dict(
            pnl_dict.get("equity_point_dict_list"),
            window_str=window_str,
        )
        return render_template(
            "_equity_chart.html",
            chart_dict=chart_obj.as_dict(),
            window_selector_pod_id_str=pod_id_str,
        )

    # ── Phase 4: action preview + execution + journal ─────────────────

    @flask_app_obj.route("/api/action-token")
    def action_token_route_fn():
        provider_obj = flask_app_obj.config["data_provider_obj"]
        return jsonify({"action_token_str": provider_obj.get_action_token_str()})

    @flask_app_obj.route("/fragments/action-preview/<pod_id_str>/<action_name_str>")
    def action_preview_fragment_route_fn(pod_id_str: str, action_name_str: str):
        if action_name_str not in ALL_ACTION_NAME_LIST:
            abort(404)
        provider_obj = flask_app_obj.config["data_provider_obj"]
        summary_dict = provider_obj.get_summary_dict()
        row_dict = get_pod_row_dict_by_id(summary_dict, pod_id_str)
        if row_dict is None:
            abort(404)
        preview_line_str_list = _build_action_preview_line_str_list(
            action_name_str, row_dict
        )
        post_url_str = (
            f"/api/pods/{pod_id_str}/diff/run"
            if action_name_str == "compare_reference"
            else f"/api/pods/{pod_id_str}/actions/{action_name_str}"
        )
        return render_template(
            "_action_preview.html",
            row_dict=row_dict,
            action_name_str=action_name_str,
            action_label_str=ACTION_LABEL_DICT.get(action_name_str, action_name_str),
            preview_line_str_list=preview_line_str_list,
            post_url_str=post_url_str,
            action_token_str=provider_obj.get_action_token_str(),
        )

    @flask_app_obj.route("/fragments/action-preview-cancel/<pod_id_str>")
    def action_preview_cancel_route_fn(pod_id_str: str):
        return (
            '<div class="text-xs text-ink-500 italic">'
            "Click an action above to see a preview before confirming."
            "</div>"
        )

    @flask_app_obj.route("/api/pods/<pod_id_str>/diff/run", methods=["POST"])
    def diff_run_route_fn(pod_id_str: str):
        return _handle_action_post(pod_id_str, "compare_reference")

    @flask_app_obj.route(
        "/api/pods/<pod_id_str>/actions/<action_name_str>", methods=["POST"]
    )
    def action_run_route_fn(pod_id_str: str, action_name_str: str):
        return _handle_action_post(pod_id_str, action_name_str)

    @flask_app_obj.route("/api/jobs/<job_id_str>")
    def job_status_route_fn(job_id_str: str):
        provider_obj = flask_app_obj.config["data_provider_obj"]
        job_dict = provider_obj.get_job_dict(job_id_str)
        if job_dict is None:
            abort(404)
        # Detect whether this is an HTMX poll (wants HTML) or a JSON consumer.
        if request.headers.get("HX-Request"):
            return render_template("_job_badge.html", job_dict=job_dict)
        return jsonify(job_dict)

    @flask_app_obj.route("/journal")
    def journal_page_route_fn():
        journal_entry_dict_list = read_journal_entry_dict_list(
            journal_path_str=flask_app_obj.config["journal_path_str"],
        )
        return render_template(
            "journal.html",
            mode_str="journal",
            mode_label_str="Operator Journal",
            journal_entry_dict_list=journal_entry_dict_list,
            verdict_dict=resolve_top_bar_verdict(
                flask_app_obj.config["data_provider_obj"].get_summary_dict()
            ).as_dict(),
            as_of_clock_str=_now_clock_str(),
        )

    # ── helper that several action routes share ──────────────────────

    def _handle_action_post(pod_id_str: str, action_name_str: str):
        if action_name_str not in ALL_ACTION_NAME_LIST:
            return _json_error_fn(400, "unsupported_action", f"Unknown action {action_name_str!r}.")
        provider_obj = flask_app_obj.config["data_provider_obj"]
        rejection_obj = validate_action_request(
            request.headers,
            request.get_json(silent=True),
            provider_obj.get_action_token_str(),
        )
        if rejection_obj is not None:
            status_int, error_code_str, message_str = rejection_obj
            return _json_error_fn(status_int, error_code_str, message_str)
        target_obj = provider_obj.get_target_for_pod(pod_id_str)
        if target_obj is None:
            return _json_error_fn(404, "unknown_pod", f"Unknown enabled pod_id_str {pod_id_str!r}.")
        try:
            if action_name_str == "compare_reference":
                job_dict = provider_obj.start_diff_job(target_obj)
            else:
                job_dict = provider_obj.start_action_job(action_name_str, target_obj)
        except DashboardActionInFlightError as exception_obj:
            return _json_error_fn(409, "action_in_flight", str(exception_obj))
        append_journal_entry(
            pod_id_str=str(job_dict.get("pod_id_str") or pod_id_str),
            mode_str=str(job_dict.get("mode_str") or "?"),
            action_name_str=str(job_dict.get("action_name_str") or action_name_str),
            job_id_str=str(job_dict.get("job_id_str") or ""),
            initial_status_str=str(job_dict.get("status_str") or "queued"),
            journal_path_str=flask_app_obj.config["journal_path_str"],
        )
        if request.headers.get("HX-Request"):
            return render_template("_job_badge.html", job_dict=job_dict)
        return jsonify(job_dict), 202

    return flask_app_obj


# ── helpers ──────────────────────────────────────────────────────────────


def _now_clock_str() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")


def _is_attention_row_bool(row_dict: dict[str, Any]) -> bool:
    from alpha.live.dashboard_v3.data import _effective_severity_str

    return _effective_severity_str(row_dict) in ("red", "yellow")


def _find_combined_book_environment_dict(
    summary_dict: dict[str, Any], mode_str: str
) -> dict[str, Any] | None:
    combined_book_dict = summary_dict.get("combined_book_dict") or {}
    for environment_dict in combined_book_dict.get("environment_dict_list") or []:
        if environment_dict.get("mode_str") == mode_str:
            return environment_dict
    return None


def _build_action_preview_line_str_list(
    action_name_str: str, row_dict: dict[str, Any]
) -> list[str]:
    if action_name_str == "compare_reference":
        return [
            "Re-runs the offline reference DIFF against the latest live state.",
            "Read-only relative to live trading — no orders are sent.",
        ]
    if action_name_str == "tick":
        return [
            f"Manually advances the {row_dict.get('mode_str')}/{row_dict.get('pod_id_str')} pod scheduler.",
            "May build a new DecisionPlan if the data gate allows.",
        ]
    if action_name_str == "submit_vplan":
        latest_vplan_id = row_dict.get("latest_vplan_id_int") or "—"
        return [
            f"Submits VPlan #{latest_vplan_id} to the broker.",
            "Sends real orders against the configured IBKR account if the pod is LIVE.",
            "Reference orders only on PAPER/INCUBATION pods.",
        ]
    if action_name_str == "post_execution_reconcile":
        return [
            "Reconciles model positions against the broker after fills.",
            "Updates residuals + writes a reconciliation record to sqlite.",
        ]
    if action_name_str == "eod_snapshot":
        return [
            "Writes the EOD equity/cash/position snapshot for this pod.",
            "Used by the equity curve and combined-book rollup.",
        ]
    return [f"Action: {action_name_str}"]


def _json_error_fn(status_int: int, error_code_str: str, message_str: str):
    response_obj = jsonify({"error_code_str": error_code_str, "message_str": message_str})
    response_obj.status_code = status_int
    return response_obj
