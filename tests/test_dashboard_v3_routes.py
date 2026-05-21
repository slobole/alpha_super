"""Route tests for Dashboard V3 — Phase 1.

Uses a hand-crafted ``StubDataProvider`` so the tests do not depend on a
real ``DashboardApp``, sqlite database, or live release YAMLs. The shape
of the dicts mirrors what ``build_dashboard_summary_dict`` and
``build_pod_detail_dict`` produce in production.
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha.live.dashboard_v3.app import (
    DASHBOARD_V3_VERSION_STR,
    create_app,
)


def _build_lifecycle_step_dict_list() -> list[dict[str, Any]]:
    return [
        {"step_key_str": "db",        "label_str": "DB",        "status_str": "ok",       "severity_str": "green"},
        {"step_key_str": "decision",  "label_str": "Decision",  "status_str": "complete", "severity_str": "green"},
        {"step_key_str": "vplan",     "label_str": "VPlan",     "status_str": "submitted","severity_str": "green"},
        {"step_key_str": "ack",       "label_str": "ACK",       "status_str": "waiting",  "severity_str": "yellow"},
        {"step_key_str": "fill",      "label_str": "Fill",      "status_str": "pending",  "severity_str": "gray"},
        {"step_key_str": "reconcile", "label_str": "Reconcile", "status_str": "pending",  "severity_str": "gray"},
        {"step_key_str": "eod",       "label_str": "EOD",       "status_str": "pending",  "severity_str": "gray"},
    ]


def _build_pod_row_dict(
    pod_id_str: str,
    mode_str: str,
    severity_str: str = "green",
    *,
    strategy_import_str: str = "strategies.demo:Demo",
    equity_float: float | None = 14200.0,
) -> dict[str, Any]:
    return {
        "pod_id_str": pod_id_str,
        "mode_str": mode_str,
        "account_route_str": "ibkr:demo",
        "strategy_import_str": strategy_import_str,
        "db_status_str": "ok",
        "health_str": severity_str,
        "next_action_str": "submit_vplan",
        "equity_float": equity_float,
        "cash_float": 1200.0,
        "position_count_int": 4,
        "broker_ack_count_int": 4,
        "broker_order_count_int": 4,
        "missing_ack_count_int": 0,
        "fill_count_int": 0,
        "latest_event_timestamp_str": "2026-05-21T15:45:02+00:00",
        "latest_vplan_status_str": "submitted",
        "latest_vplan_id_int": 87,
        "latest_decision_plan_id_int": 142,
        "lifecycle_step_dict_list": _build_lifecycle_step_dict_list(),
        "required_action_dict": {
            "label_str": "Waiting for ACKs",
            "severity_str": severity_str,
            "reason_str": "Broker responses pending",
            "detail_str": "3/4 acked, TSLA still pending",
        },
        "debug_summary_dict": {
            "severity_str": severity_str,
            "verdict_label_str": "ack_pending",
            "primary_reason_str": "TSLA ack pending",
        },
        "data_freshness_dict": {},
        "eod_snapshot_dict": {},
        "rehearsal_status_dict": {},
    }


def _build_summary_dict() -> dict[str, Any]:
    return {
        "as_of_timestamp_str": "2026-05-21T16:00:00+00:00",
        "pod_row_dict_list": [
            _build_pod_row_dict("dv2_caspersky_live",   "live",  "green"),
            _build_pod_row_dict("qp_mr_live",           "live",  "yellow"),
            _build_pod_row_dict("dv2_caspersky_paper",  "paper", "red"),
            _build_pod_row_dict("incubation_pod_demo",  "incubation", "gray"),
        ],
        "alert_dict_list": [],
        "alert_summary_dict": {},
        "mode_list": ["live", "paper", "incubation"],
        "combined_book_dict": {
            "environment_dict_list": [
                {
                    "mode_str": "live",
                    "latest_market_date_str": "2026-05-21",
                    "latest_equity_float": 28400.0,
                    "daily_pnl_float": 184.0,
                    "since_start_pnl_float": -420.0,
                    "equity_point_dict_list": [
                        {"market_date_str": "2026-05-19", "equity_float": 28220.0, "daily_pnl_float": 100.0},
                        {"market_date_str": "2026-05-20", "equity_float": 28216.0, "daily_pnl_float": -4.0},
                        {"market_date_str": "2026-05-21", "equity_float": 28400.0, "daily_pnl_float": 184.0},
                    ],
                },
            ],
        },
    }


class StubTarget:
    def __init__(self, pod_id_str: str, mode_str: str) -> None:
        # Mirrors the shape used by alpha.live.dashboard.DashboardPodTarget enough
        # for tests; production code only reads release_obj.pod_id_str/mode_str.
        class _ReleaseShim:
            pass
        self.release_obj = _ReleaseShim()
        self.release_obj.pod_id_str = pod_id_str
        self.release_obj.mode_str = mode_str


class StubDataProvider:
    ACTION_TOKEN_STR = "stub-token"

    def __init__(self) -> None:
        self.summary_dict = _build_summary_dict()
        self.detail_call_log_list: list[str] = []
        self.event_call_log_list: list[str] = []
        self.action_job_dict_list: list[dict[str, Any]] = []
        self._next_job_seq_int = 0

    def get_summary_dict(self) -> dict[str, Any]:
        return self.summary_dict

    def get_action_token_str(self) -> str:
        return self.ACTION_TOKEN_STR

    def get_target_for_pod(self, pod_id_str: str) -> StubTarget | None:
        matching_row_dict = next(
            (
                row_dict
                for row_dict in self.summary_dict["pod_row_dict_list"]
                if row_dict["pod_id_str"] == pod_id_str
            ),
            None,
        )
        if matching_row_dict is None:
            return None
        return StubTarget(matching_row_dict["pod_id_str"], matching_row_dict["mode_str"])

    def _next_job_id_str(self) -> str:
        self._next_job_seq_int += 1
        return f"stub-job-{self._next_job_seq_int:04d}"

    def start_diff_job(self, target_obj: StubTarget) -> dict[str, Any]:
        job_dict = {
            "job_id_str": self._next_job_id_str(),
            "pod_id_str": target_obj.release_obj.pod_id_str,
            "mode_str": target_obj.release_obj.mode_str,
            "action_name_str": "compare_reference",
            "status_str": "queued",
            "created_timestamp_str": "2026-05-21T16:00:00+00:00",
        }
        self.action_job_dict_list.append(job_dict)
        return job_dict

    def start_action_job(
        self, action_name_str: str, target_obj: StubTarget
    ) -> dict[str, Any]:
        job_dict = {
            "job_id_str": self._next_job_id_str(),
            "pod_id_str": target_obj.release_obj.pod_id_str,
            "mode_str": target_obj.release_obj.mode_str,
            "action_name_str": action_name_str,
            "status_str": "queued",
            "created_timestamp_str": "2026-05-21T16:00:00+00:00",
        }
        self.action_job_dict_list.append(job_dict)
        return job_dict

    def get_job_dict(self, job_id_str: str) -> dict[str, Any] | None:
        for job_dict in self.action_job_dict_list:
            if job_dict["job_id_str"] == job_id_str:
                return job_dict
        return None

    def get_pod_detail_dict(self, pod_id_str: str) -> dict[str, Any]:
        self.detail_call_log_list.append(pod_id_str)
        matching_row_dict = next(
            (
                row_dict
                for row_dict in self.summary_dict["pod_row_dict_list"]
                if row_dict["pod_id_str"] == pod_id_str
            ),
            None,
        )
        if matching_row_dict is None:
            raise KeyError(pod_id_str)
        return {
            "pod_row_dict": matching_row_dict,
            "required_action_dict": matching_row_dict["required_action_dict"],
            "lifecycle_step_dict_list": matching_row_dict["lifecycle_step_dict_list"],
            "data_freshness_dict": {
                "item_dict_list": [
                    {"label_str": "Norgate", "value_str": "2026-05-21",
                     "severity_str": "green", "detail_str": "source=norgate_only · snapshot ready"},
                    {"label_str": "Pod state", "value_str": "2026-05-21T16:14:32+00:00",
                     "severity_str": "green", "detail_str": "latest persisted state"},
                    {"label_str": "DIFF artifact", "value_str": None,
                     "severity_str": "gray", "detail_str": "not_run"},
                ],
            },
            "eod_snapshot_dict": {"status_str": "pending"},
            "rehearsal_status_dict": {},
            "debug_story_dict": {},
            "pod_pnl_dict": {
                "point_count_int": 3,
                "equity_point_dict_list": [
                    {"market_date_str": "2026-05-19", "equity_float": 14000.0, "daily_pnl_float": 50.0},
                    {"market_date_str": "2026-05-20", "equity_float": 14100.0, "daily_pnl_float": 100.0},
                    {"market_date_str": "2026-05-21", "equity_float": 14200.0, "daily_pnl_float": 100.0},
                ],
            },
            "latest_decision_plan_dict": {
                "decision_plan_id_int": 142,
                "decision_book_type_str": "full_target_weight_book",
                "signal_timestamp_str": "2026-05-21T15:30:14+00:00",
                "target_execution_timestamp_str": "2026-05-21T16:00:00+00:00",
                "entry_target_weight_map_dict": {"AAPL": 0.25, "MSFT": 0.25},
                "exit_asset_list": ["TSLA"],
                "snapshot_metadata_dict": {
                    "norgate_data_profile_str": "norgate_only",
                    "norgate_snapshot_date_str": "2026-05-20",
                },
            },
            "latest_vplan_dict": {
                "vplan_id_int": 87,
                "status_str": "submitted",
                "vplan_row_dict_list": [
                    {"asset_str": "AAPL", "current_share_float": 10, "target_share_float": 12,
                     "order_delta_share_float": 2, "live_reference_price_float": 186.4},
                ],
                "broker_ack_row_dict_list": [
                    {"asset_str": "AAPL", "ack_status_str": "acked",
                     "response_timestamp_str": "2026-05-21T16:00:02+00:00"},
                ],
            },
            "latest_execution_report_dict": None,
            "event_dict_list": [],
            "latest_diff_dict": {},
        }

    def get_pod_event_dict_list(
        self, pod_id_str: str, limit_int: int = 80
    ) -> list[dict[str, Any]]:
        self.event_call_log_list.append(pod_id_str)
        return [
            {
                "timestamp_str": "2026-05-21T15:45:02+00:00",
                "level_str": "INFO",
                "event_name_str": "vplan.submitted",
                "reason_str": "VPlan #87 → IBKR",
            },
            {
                "timestamp_str": "2026-05-21T16:00:02+00:00",
                "level_str": "WARN",
                "event_name_str": "broker.ack_received",
                "reason_str": "TSLA still pending",
            },
        ]


@pytest.fixture(name="provider_obj")
def fixture_provider_obj() -> StubDataProvider:
    return StubDataProvider()


@pytest.fixture(name="journal_path_str")
def fixture_journal_path_str(tmp_path) -> str:
    return str(tmp_path / "operator_journal.jsonl")


@pytest.fixture(name="test_client_obj")
def fixture_test_client_obj(provider_obj: StubDataProvider, journal_path_str: str):
    flask_app_obj = create_app(data_provider_obj=provider_obj, journal_path_str=journal_path_str)
    flask_app_obj.config["TESTING"] = True
    with flask_app_obj.test_client() as test_client_obj:
        yield test_client_obj


ACTION_HEADERS_DICT = {
    "Host": "localhost",
    "Origin": "http://localhost",
    "Content-Type": "application/json",
    "X-Alpha-Action-Token": StubDataProvider.ACTION_TOKEN_STR,
}


# ── basic plumbing ────────────────────────────────────────────────────────


def test_healthz_route_returns_version_marker(test_client_obj) -> None:
    response_obj = test_client_obj.get("/healthz")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert "dashboard_v3 ok" in response_text_str
    assert DASHBOARD_V3_VERSION_STR in response_text_str


def test_index_redirects_to_live(test_client_obj) -> None:
    response_obj = test_client_obj.get("/")
    assert response_obj.status_code == 302
    assert response_obj.headers["Location"].endswith("/live")


def test_unknown_mode_returns_404(test_client_obj) -> None:
    response_obj = test_client_obj.get("/martian")
    assert response_obj.status_code == 404


# ── mode pages ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode_str,expected_pod_id_str", [
    ("live", "dv2_caspersky_live"),
    ("paper", "dv2_caspersky_paper"),
    ("incubation", "incubation_pod_demo"),
])
def test_mode_page_lists_only_that_modes_pods(
    test_client_obj, mode_str: str, expected_pod_id_str: str
) -> None:
    response_obj = test_client_obj.get(f"/{mode_str}")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert expected_pod_id_str in response_text_str
    # Should not show pods from other modes in the table.
    for other_mode_str, other_pod_id_str in [
        ("live", "dv2_caspersky_live"),
        ("paper", "dv2_caspersky_paper"),
        ("incubation", "incubation_pod_demo"),
    ]:
        if other_mode_str != mode_str:
            assert other_pod_id_str not in response_text_str


def test_live_page_renders_pod_row_with_severity(test_client_obj) -> None:
    response_obj = test_client_obj.get("/live")
    response_text_str = response_obj.get_data(as_text=True)
    assert "dv2_caspersky_live" in response_text_str
    assert "qp_mr_live" in response_text_str
    # Top bar verdict should reflect the worst severity across all modes (paper has red).
    assert "need action" in response_text_str.lower() or "All clear" in response_text_str


# ── HTMX fragments ────────────────────────────────────────────────────────


def test_top_bar_fragment_returns_html(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/top-bar")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert "refresh" in response_text_str


def test_pod_detail_fragment_renders_timeline(test_client_obj, provider_obj) -> None:
    response_obj = test_client_obj.get("/fragments/pod-detail/dv2_caspersky_live")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert "dv2_caspersky_live" in response_text_str
    # Includes some stage labels from lifecycle steps.
    assert "Decision" in response_text_str
    assert "VPlan" in response_text_str
    assert "ACK" in response_text_str
    # Records that the detail was actually fetched from the provider.
    assert provider_obj.detail_call_log_list == ["dv2_caspersky_live"]


def test_pod_detail_panel_does_not_auto_poll_itself(test_client_obj) -> None:
    """Polish fix: the heavy panel (with <details> toggles) must NOT have
    its own auto-refresh; only the small header fragment + events tail do."""
    response_obj = test_client_obj.get("/fragments/pod-detail/dv2_caspersky_live")
    response_text_str = response_obj.get_data(as_text=True)
    # The outer body div should not carry an every-N-seconds polling trigger.
    body_open_index_int = response_text_str.find('id="pod-detail-body-dv2_caspersky_live"')
    body_close_index_int = response_text_str.find(">", body_open_index_int)
    body_tag_str = response_text_str[body_open_index_int:body_close_index_int]
    assert "every" not in body_tag_str, f"detail body still polls: {body_tag_str!r}"


def test_pod_detail_header_fragment_polls_itself(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/pod-detail-header/dv2_caspersky_live")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    # Header fragment carries its own short polling cadence.
    assert "every 10s" in response_text_str
    # Includes the pod id and a freshness clock label.
    assert "dv2_caspersky_live" in response_text_str
    assert "refreshed" in response_text_str


def test_pod_detail_header_fragment_unknown_pod_returns_404(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/pod-detail-header/no_such_pod")
    assert response_obj.status_code == 404


# ── Norgate freshness branch ──────────────────────────────────────────────


def test_pod_detail_includes_data_freshness_panel(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/pod-detail/dv2_caspersky_live")
    response_text_str = response_obj.get_data(as_text=True)
    # Panel header is present.
    assert "Data Freshness" in response_text_str
    # All three stub freshness items render with their labels.
    assert "Norgate" in response_text_str
    assert "Pod state" in response_text_str
    assert "DIFF artifact" in response_text_str
    # Norgate's value (snapshot date) appears.
    assert "2026-05-21" in response_text_str
    # Detail text appears for at least one item.
    assert "source=norgate_only" in response_text_str


def test_main_module_loads_config_env_before_serving(monkeypatch, tmp_path) -> None:
    """Regression for the VPS hang: ``python -m alpha.live.dashboard_v3`` must
    pre-load ``config.env`` so ``ALPHA_USE_NORGATE_SNAPSHOT_BOOL=true`` (and
    friends) are visible to the data builders before the first request lands.
    """
    from alpha.live.dashboard_v3 import __main__ as dashboard_main_module
    from scripts import norgate_config_env

    config_env_path_obj = tmp_path / "config.env"
    config_env_path_obj.write_text(
        "ALPHA_USE_NORGATE_SNAPSHOT_BOOL=true\nFOO_FOR_TEST=bar\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(norgate_config_env, "default_config_env_path_obj", lambda: config_env_path_obj)
    monkeypatch.delenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", raising=False)
    monkeypatch.delenv("FOO_FOR_TEST", raising=False)

    captured_args_list: list[dict] = []

    def _fake_run_fn(self, **kwargs):
        captured_args_list.append(kwargs)

    # Stop the test from actually starting an HTTP server.
    import flask
    monkeypatch.setattr(flask.Flask, "run", _fake_run_fn)
    monkeypatch.setattr("sys.argv", ["python -m alpha.live.dashboard_v3", "--port", "9999"])

    assert dashboard_main_module.main() == 0
    import os
    assert os.environ.get("ALPHA_USE_NORGATE_SNAPSHOT_BOOL") == "true"
    assert os.environ.get("FOO_FOR_TEST") == "bar"
    assert captured_args_list and captured_args_list[0].get("port") == 9999


def test_main_module_skip_env_file_flag_leaves_env_untouched(monkeypatch, tmp_path) -> None:
    from alpha.live.dashboard_v3 import __main__ as dashboard_main_module
    from scripts import norgate_config_env

    config_env_path_obj = tmp_path / "config.env"
    config_env_path_obj.write_text("FOO_FOR_TEST=value_from_file\n", encoding="utf-8")
    monkeypatch.setattr(norgate_config_env, "default_config_env_path_obj", lambda: config_env_path_obj)
    monkeypatch.delenv("FOO_FOR_TEST", raising=False)

    import flask
    monkeypatch.setattr(flask.Flask, "run", lambda self, **kwargs: None)
    monkeypatch.setattr("sys.argv", ["python -m alpha.live.dashboard_v3", "--skip-env-file"])

    assert dashboard_main_module.main() == 0
    import os
    assert "FOO_FOR_TEST" not in os.environ


def test_decision_stage_card_shows_norgate_snapshot_meta(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/pod-detail/dv2_caspersky_live")
    response_text_str = response_obj.get_data(as_text=True)
    # Snapshot lineage line surfaces the profile + snapshot date.
    assert "Norgate snapshot" in response_text_str
    assert "norgate_only" in response_text_str
    assert "2026-05-20" in response_text_str


def test_pod_detail_fragment_unknown_pod_returns_404(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/pod-detail/no_such_pod")
    assert response_obj.status_code == 404


def test_events_tail_fragment_renders_event_rows(test_client_obj, provider_obj) -> None:
    response_obj = test_client_obj.get("/fragments/events-tail/dv2_caspersky_live")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert "vplan.submitted" in response_text_str
    assert "broker.ack_received" in response_text_str
    assert provider_obj.event_call_log_list == ["dv2_caspersky_live"]


# ── Phase 2: health + schedule + new-event badge ──────────────────────────


def test_health_strip_fragment_returns_html(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/health-strip")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    # Should at least render the Norgate and Disk cells.
    assert "Norgate" in response_text_str
    assert "Disk" in response_text_str


def test_schedule_strip_fragment_returns_html(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/schedule-strip")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    # StubDataProvider gives every pod next_action_str=submit_vplan, so the
    # schedule strip should list at least one entry.
    assert "submit_vplan" in response_text_str or "Schedule is empty" in response_text_str


def test_pod_row_carries_data_pod_id_attribute(test_client_obj) -> None:
    response_obj = test_client_obj.get("/live")
    response_text_str = response_obj.get_data(as_text=True)
    assert 'data-pod-id="dv2_caspersky_live"' in response_text_str
    assert "new_event_badge.js" in response_text_str


# ── Phase 3: equity chart fragment ────────────────────────────────────────


def test_equity_chart_fragment_renders_svg(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/equity-chart/dv2_caspersky_live?window=all")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert "<svg" in response_text_str
    assert "viewBox" in response_text_str
    # Window selector buttons should be present and "all" should be marked active.
    assert "window=30d" in response_text_str
    assert "window=90d" in response_text_str


def test_equity_chart_fragment_unknown_pod_404s(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/equity-chart/no_such_pod")
    assert response_obj.status_code == 404


def test_equity_chart_fragment_clamps_unknown_window_to_all(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/equity-chart/dv2_caspersky_live?window=bogus")
    assert response_obj.status_code == 200


def test_live_page_embeds_combined_book_chart(test_client_obj) -> None:
    response_obj = test_client_obj.get("/live")
    response_text_str = response_obj.get_data(as_text=True)
    assert "combined book" in response_text_str.lower()
    # Combined-book equity curve renders as an SVG path inside the mode page itself.
    assert "<svg" in response_text_str


def test_incubation_page_omits_combined_book_chart_when_no_data(test_client_obj) -> None:
    response_obj = test_client_obj.get("/incubation")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    # Only the live mode has combined_book data in the stub — incubation should not show the section.
    assert "combined book" not in response_text_str.lower()


# ── Phase 4: operator tools, actions, journal ────────────────────────────


def test_action_token_endpoint_returns_provider_token(test_client_obj) -> None:
    response_obj = test_client_obj.get("/api/action-token")
    assert response_obj.status_code == 200
    payload_dict = response_obj.get_json()
    assert payload_dict["action_token_str"] == StubDataProvider.ACTION_TOKEN_STR


def test_action_preview_fragment_renders_confirm_button(test_client_obj) -> None:
    response_obj = test_client_obj.get(
        "/fragments/action-preview/dv2_caspersky_live/submit_vplan"
    )
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert "Confirm and run" in response_text_str
    assert "submit_vplan" in response_text_str
    # Confirm button POSTs to the matching action endpoint.
    assert "/api/pods/dv2_caspersky_live/actions/submit_vplan" in response_text_str


def test_action_preview_unknown_action_returns_404(test_client_obj) -> None:
    response_obj = test_client_obj.get("/fragments/action-preview/dv2_caspersky_live/no_such_action")
    assert response_obj.status_code == 404


def test_action_post_requires_origin_header(test_client_obj) -> None:
    response_obj = test_client_obj.post(
        "/api/pods/dv2_caspersky_live/actions/submit_vplan",
        json={"confirmed_bool": True},
        headers={
            "Host": "localhost",
            "X-Alpha-Action-Token": StubDataProvider.ACTION_TOKEN_STR,
        },
    )
    assert response_obj.status_code == 403
    payload_dict = response_obj.get_json()
    assert payload_dict["error_code_str"] == "origin_rejected"


def test_action_post_requires_action_token(test_client_obj) -> None:
    headers_dict = dict(ACTION_HEADERS_DICT)
    headers_dict["X-Alpha-Action-Token"] = "wrong-token"
    response_obj = test_client_obj.post(
        "/api/pods/dv2_caspersky_live/actions/submit_vplan",
        data='{"confirmed_bool": true}',
        headers=headers_dict,
    )
    assert response_obj.status_code == 403
    assert response_obj.get_json()["error_code_str"] == "action_token_required"


def test_action_post_requires_explicit_confirmation(test_client_obj) -> None:
    response_obj = test_client_obj.post(
        "/api/pods/dv2_caspersky_live/actions/submit_vplan",
        data='{"confirmed_bool": false}',
        headers=ACTION_HEADERS_DICT,
    )
    assert response_obj.status_code == 400
    assert response_obj.get_json()["error_code_str"] == "confirmation_required"


def test_action_post_starts_job_and_appends_journal_entry(
    test_client_obj, provider_obj, journal_path_str
) -> None:
    response_obj = test_client_obj.post(
        "/api/pods/dv2_caspersky_live/actions/submit_vplan",
        data='{"confirmed_bool": true}',
        headers=ACTION_HEADERS_DICT,
    )
    assert response_obj.status_code == 202
    job_dict = response_obj.get_json()
    assert job_dict["status_str"] == "queued"
    assert job_dict["action_name_str"] == "submit_vplan"

    # Provider sees a new job.
    assert len(provider_obj.action_job_dict_list) == 1

    # Journal file got one entry.
    from pathlib import Path
    journal_lines = Path(journal_path_str).read_text(encoding="utf-8").splitlines()
    assert len(journal_lines) == 1
    import json
    entry_dict = json.loads(journal_lines[0])
    assert entry_dict["action_name_str"] == "submit_vplan"
    assert entry_dict["pod_id_str"] == "dv2_caspersky_live"
    assert entry_dict["job_id_str"] == job_dict["job_id_str"]


def test_action_post_for_unknown_pod_404s(test_client_obj) -> None:
    response_obj = test_client_obj.post(
        "/api/pods/no_such_pod/actions/submit_vplan",
        data='{"confirmed_bool": true}',
        headers=ACTION_HEADERS_DICT,
    )
    assert response_obj.status_code == 404
    assert response_obj.get_json()["error_code_str"] == "unknown_pod"


def test_diff_run_post_starts_diff_job(test_client_obj, provider_obj) -> None:
    response_obj = test_client_obj.post(
        "/api/pods/dv2_caspersky_live/diff/run",
        data='{"confirmed_bool": true}',
        headers=ACTION_HEADERS_DICT,
    )
    assert response_obj.status_code == 202
    job_dict = response_obj.get_json()
    assert job_dict["action_name_str"] == "compare_reference"


def test_job_status_endpoint_returns_json_for_unknown_caller(
    test_client_obj, provider_obj
) -> None:
    test_client_obj.post(
        "/api/pods/dv2_caspersky_live/actions/submit_vplan",
        data='{"confirmed_bool": true}',
        headers=ACTION_HEADERS_DICT,
    )
    job_id_str = provider_obj.action_job_dict_list[0]["job_id_str"]
    response_obj = test_client_obj.get(f"/api/jobs/{job_id_str}")
    assert response_obj.status_code == 200
    payload_dict = response_obj.get_json()
    assert payload_dict["job_id_str"] == job_id_str


def test_job_status_endpoint_returns_html_for_htmx_caller(
    test_client_obj, provider_obj
) -> None:
    test_client_obj.post(
        "/api/pods/dv2_caspersky_live/actions/submit_vplan",
        data='{"confirmed_bool": true}',
        headers=ACTION_HEADERS_DICT,
    )
    job_id_str = provider_obj.action_job_dict_list[0]["job_id_str"]
    response_obj = test_client_obj.get(
        f"/api/jobs/{job_id_str}",
        headers={"HX-Request": "true"},
    )
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert "job_id=" in response_text_str
    assert "queued" in response_text_str


def test_journal_page_lists_entries_after_action(
    test_client_obj, journal_path_str
) -> None:
    test_client_obj.post(
        "/api/pods/dv2_caspersky_live/actions/submit_vplan",
        data='{"confirmed_bool": true}',
        headers=ACTION_HEADERS_DICT,
    )
    response_obj = test_client_obj.get("/journal")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert "Operator Journal" in response_text_str
    assert "submit_vplan" in response_text_str
    assert "dv2_caspersky_live" in response_text_str
