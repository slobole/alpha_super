"""Dashboard V4 view of the nightly backtest check report: reading, scoping, views and routes."""

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4 import backtest_check_data as btd

NOW = datetime(2026, 12, 10, 23, 40, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def healthy_host_disk(monkeypatch):
    from alpha.live.dashboard_v3 import health
    monkeypatch.setattr(health.shutil, "disk_usage", lambda path: SimpleNamespace(total=100, used=50, free=50))
    btd._CACHE_DICT.clear()


def _pod(pod_id="pod_a", account="U1", release="r1", status="same", rare=None):
    return {"pod_id_str": pod_id, "account_route_str": account, "release_id_str": release, "session_str": "2026-12-10",
            "decision_dict": {"status_str": status, "session_str": "2026-12-10",
                              "row_list": [{"asset_str": "XLE", "live_str": "BUY · 10.0%", "backtest_str": "—", "same_bool": status == "same"},
                                           {"asset_str": "KRE", "live_str": "BUY · 10.0%", "backtest_str": "BUY", "same_bool": True}]},
            "record_list": [{"session_str": "2026-12-09", "status_str": "same"}, {"session_str": "2026-12-10", "status_str": status}],
            "curve_dict": {"point_list": [{"date_str": "2026-12-09", "live_float": 0.01, "backtest_float": 0.011},
                                          {"date_str": "2026-12-10", "live_float": 0.02, "backtest_float": 0.023}],
                           "start_str": "2026-12-09", "gap_float": -0.003, "gap_month_float": -0.001},
            "reference_status_str": "ok", "rare_loss_dict": rare}


def _report(pod_list, generated=NOW - timedelta(hours=1), gate=None):
    return {"schema_version_str": btd.SCHEMA_VERSION_STR, "generated_at_utc_str": generated.isoformat(),
            "pod_list": pod_list, "cost_gate_dict": gate}


def _provider(tmp_path, report=None, identities=(("pod_a", "U1", "r1"),)):
    log_path = tmp_path / "live_events.jsonl"
    log_path.write_text("", encoding="utf-8")
    if report is not None:
        (tmp_path / btd.REPORT_FILE_NAME_STR).write_text(json.dumps(report), encoding="utf-8")
    target_list = [SimpleNamespace(release_obj=SimpleNamespace(pod_id_str=pod, account_route_str=account, release_id_str=release,
                                                               mode_str="live", enabled_bool=True)) for pod, account, release in identities]
    return SimpleNamespace(event_log_path_str=str(log_path), get_target_list=lambda: target_list)


def test_no_report_means_no_view(tmp_path):
    provider = _provider(tmp_path)
    assert btd.load_report_dict(provider, as_of_ts=NOW) is None
    assert not btd.report_file_exists_bool(provider)
    assert btd.pod_view_dict(None, "pod_a") is None
    assert btd.attention_list(None, {"pod_a": "A"}) == []


def test_report_is_scoped_to_current_live_identities(tmp_path):
    report = _report([_pod(), _pod(pod_id="pod_old", account="U9", release="r9"), _pod(pod_id="pod_other", release="r_other")])
    loaded = btd.load_report_dict(_provider(tmp_path, report), as_of_ts=NOW)
    assert [pod["pod_id_str"] for pod in loaded["pod_list"]] == ["pod_a"]
    assert loaded["stale_bool"] is False


@pytest.mark.parametrize('status_str', ['skipped', 'not_run', 'incomplete'])
def test_no_completed_decision_check_is_not_green(tmp_path, status_str):
    pod_dict = _pod(status='not_checked')
    pod_dict['reference_status_str'] = status_str
    report_dict = btd.load_report_dict(_provider(tmp_path, _report([pod_dict])), as_of_ts=NOW)
    row_dict = btd.system_row_dict(report_dict, as_of_ts=NOW)
    assert row_dict['state_str'] == 'warning'
    assert '0 of 1 pods checked' in row_dict['now_str']


def test_replaced_release_and_partially_missing_pods_are_not_green(tmp_path):
    for identity_tuple, expected_str in [((('pod_a', 'U1', 'r2'),), '0 of 1'),
                                          ((('pod_a', 'U1', 'r1'), ('pod_b', 'U2', 'r2')), '1 of 2')]:
        report_dict = btd.load_report_dict(_provider(tmp_path, _report([_pod()]), identities=identity_tuple), as_of_ts=NOW)
        row_dict = btd.system_row_dict(report_dict, as_of_ts=NOW)
        assert row_dict['state_str'] == 'warning'
        assert expected_str in row_dict['now_str']


def test_empty_report_is_not_green(tmp_path):
    report_dict = btd.load_report_dict(_provider(tmp_path, _report([]), identities=()), as_of_ts=NOW)
    assert btd.system_row_dict(report_dict, as_of_ts=NOW)['state_str'] == 'warning'


def test_incomplete_cost_sources_survive_validation_and_show_the_reason(tmp_path):
    gate_dict = {'status_str': 'incomplete', 'fill_count_int': 200, 'reason_str': 'Unreadable MR sources: pod_a',
                 'average_bps_float': None, 'pod_row_list': [{'pod_id_str': 'pod_a'}]}
    report_dict = btd.load_report_dict(_provider(tmp_path, _report([_pod()], gate=gate_dict)), as_of_ts=NOW)
    assert report_dict is not None
    view_dict = btd.gate_view_dict(report_dict)
    assert view_dict['state_str'] == 'late'
    assert view_dict['reason_str'] == gate_dict['reason_str']
    assert view_dict['average_str'] == '—'
    assert btd.system_row_dict(report_dict, as_of_ts=NOW)['state_str'] == 'warning'


@pytest.mark.parametrize('session_str,release_str,decision_id_int,expected_tab_bool', [
    ('2026-09-04', 'DEMO-release-1', 1, True), ('2026-09-08', 'DEMO-release-1', 1, False),
    ('2026-09-04', 'other-release', 1, False), ('2026-09-04', 'DEMO-release-1', 2, False),
    ('2026-09-04', 'DEMO-release-1', None, False),
])
def test_historical_backtest_tab_requires_exact_decision(monkeypatch, session_str, release_str, decision_id_int, expected_tab_bool):
    from flask import template_rendered
    from alpha.live.dashboard_v4 import backtest_check_demo
    from alpha.live.dashboard_v4.demo import create_demo_app
    original_fn = backtest_check_demo.build_demo_report_dict
    def report_fn(*arg_tuple, **kwarg_dict):
        report_dict = original_fn(*arg_tuple, **kwarg_dict)
        report_dict['pod_list'][0]['decision_dict']['session_str'] = session_str
        report_dict['pod_list'][0]['decision_plan_id_int'] = decision_id_int
        report_dict['pod_list'][0]['release_id_str'] = release_str
        return report_dict
    monkeypatch.setattr(backtest_check_demo, 'build_demo_report_dict', report_fn)
    app_obj = create_demo_app(demo_backtest_bool=True)
    context_list = []
    def capture_fn(sender_obj, template, context, **kwarg_dict):
        if 'pod_page_dict' in context:
            context_list.append(context['pod_page_dict'])
    with template_rendered.connected_to(capture_fn, app_obj):
        response_obj = app_obj.test_client().get('/pods/demo_1_0?cycle=vplan:1&tab=backtest')
    assert response_obj.status_code == 200
    context_dict = context_list[-1]
    assert context_dict['historical_bool']
    assert context_dict['selected_cycle_dict']['session_date_str'] == '2026-09-04'
    assert bool(context_dict['backtest_dict']) is expected_tab_bool
    assert any(tab_dict['key_str'] == 'backtest' for tab_dict in context_dict['tab_list']) is expected_tab_bool


def test_incomplete_gate_reason_is_rendered(monkeypatch):
    from alpha.live.dashboard_v4 import backtest_check_demo
    from alpha.live.dashboard_v4.demo import create_demo_app
    original_fn = backtest_check_demo.build_demo_report_dict
    def report_fn(*arg_tuple, **kwarg_dict):
        report_dict = original_fn(*arg_tuple, **kwarg_dict)
        report_dict['cost_gate_dict'].update(status_str='incomplete', average_bps_float=None,
                                             reason_str='Unreadable MR sources: missing_test_pod')
        return report_dict
    monkeypatch.setattr(backtest_check_demo, 'build_demo_report_dict', report_fn)
    response_obj = create_demo_app(demo_backtest_bool=True).test_client().get('/performance?level=backtest')
    assert response_obj.status_code == 200
    assert 'Cost component incomplete' in response_obj.text
    assert 'Unreadable MR sources: missing_test_pod' in response_obj.text


@pytest.mark.parametrize("bad", [{"schema_version_str": "other"}, {"schema_version_str": btd.SCHEMA_VERSION_STR, "pod_list": "x",
                                                                    "generated_at_utc_str": NOW.isoformat()}])
def test_invalid_report_is_ignored(tmp_path, bad):
    assert btd.load_report_dict(_provider(tmp_path, bad), as_of_ts=NOW) is None


def test_future_and_oversized_reports_are_ignored(tmp_path, monkeypatch):
    assert btd.load_report_dict(_provider(tmp_path, _report([_pod()], generated=NOW + timedelta(minutes=5))), as_of_ts=NOW) is None
    btd._CACHE_DICT.clear()
    monkeypatch.setattr(btd, "REPORT_BYTE_LIMIT_INT", 10)
    assert btd.load_report_dict(_provider(tmp_path, _report([_pod()])), as_of_ts=NOW) is None


def test_reading_writes_nothing(tmp_path):
    provider = _provider(tmp_path, _report([_pod()]))
    before = sorted(path.name for path in tmp_path.iterdir())
    btd.load_report_dict(provider, as_of_ts=NOW)
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_stale_report_hides_alerts_and_rare_loss(tmp_path):
    rare = {"window_int": 3, "live_float": -0.06, "backtest_float": 0.0, "line_float": -0.05, "worst_float": -0.1, "cause_str": "ours"}
    report = btd.load_report_dict(_provider(tmp_path, _report([_pod(status="differs", rare=rare)], generated=NOW - timedelta(hours=40))), as_of_ts=NOW)
    assert report["stale_bool"]
    assert btd.attention_list(report, {"pod_a": "A"}) == []
    assert btd.pod_view_dict(report, "pod_a")["rare_dict"] is None
    assert btd.pod_view_dict(report, "pod_a")["status_str"] == "not_checked"


def test_views_and_attention_rows(tmp_path):
    rare = {"window_int": 3, "live_float": -0.053, "backtest_float": -0.052, "line_float": -0.048, "worst_float": -0.11, "cause_str": "strategy"}
    report = btd.load_report_dict(_provider(tmp_path, _report([_pod(status="differs", rare=rare)])), as_of_ts=NOW)
    view = btd.pod_view_dict(report, "pod_a")
    assert view["title_str"] == "1 of 2 targets differ from the backtest"
    assert view["table_dict"]["row_list"][0]["state_str"] == "fail"
    assert view["record_dict"]["same_int"] == 1 and view["record_dict"]["checked_int"] == 2
    assert view["rare_dict"]["state_str"] == "late" and "the strategy did this" in view["rare_dict"]["sub_str"]
    rows = btd.attention_list(report, {"pod_a": "Pod A"})
    assert [row["kind_str"] for row in rows] == ["backtest_check"]  # a loss the strategy explains adds no row
    assert rows[0]["state_str"] == "late"  # a difference is something to check, never a red X
    ours = dict(rare, backtest_float=0.0, cause_str="ours")
    report = btd.load_report_dict(_provider(tmp_path, _report([_pod(rare=ours)], generated=NOW - timedelta(minutes=30))), as_of_ts=NOW)
    assert [row["kind_str"] for row in btd.attention_list(report, {"pod_a": "Pod A"})] == ["rare_loss"]


def test_gate_view_hides_average_before_30_fills():
    gate = {"status_str": "collecting", "fill_count_int": 12, "fill_target_int": 200, "gate_bps_float": 4.0, "average_bps_float": 1.0,
            "cost_bps_list": [1.0] * 12, "pod_row_list": []}
    view = btd.gate_view_dict({"cost_gate_dict": gate})
    assert view["average_str"] == "—" and view["state_str"] == "next"
    above = btd.gate_view_dict({"cost_gate_dict": dict(gate, status_str="above", fill_count_int=64, average_bps_float=4.6)})
    assert above["average_str"] == "4.6"


def test_backtest_series_is_rebased_to_the_chart_start():
    view = {"point_list": [{"date_str": "2026-12-08", "live_float": 0.0, "backtest_float": 0.10},
                           {"date_str": "2026-12-09", "live_float": 0.0, "backtest_float": 0.21},
                           {"date_str": "2026-12-10", "live_float": 0.0, "backtest_float": 0.10}]}
    path = [{"market_date_str": "2026-12-08 SOD", "cumulative_return_float": 0.0},
            {"market_date_str": "2026-12-09", "cumulative_return_float": 0.05},
            {"market_date_str": "2026-12-10", "cumulative_return_float": 0.01}]
    chart = btd.add_backtest_series({"available_bool": True, "drawing_dict": {}}, path, view)
    series = chart["drawing_dict"]["series_list"]
    assert [item["name_str"] for item in series] == ["Live", "Backtest, same days"]
    assert series[1]["dash_bool"] and not series[1]["area_bool"]
    assert [point["value_float"] for point in series[1]["point_list"]] == pytest.approx([0.0, 0.1, 0.0])
    assert btd.add_backtest_series({"available_bool": False}, path, view) == {"available_bool": False}


def test_system_row_reports_failures_and_age(tmp_path):
    pod = _pod()
    pod["reference_status_str"] = "failed"
    report = btd.load_report_dict(_provider(tmp_path, _report([pod])), as_of_ts=NOW)
    row = btd.system_row_dict(report, as_of_ts=NOW)
    assert row["state_str"] == "warning" and "1 failed" in row["now_str"]
    assert btd.system_row_dict(None, as_of_ts=NOW)["state_str"] == "unknown"


def test_demo_routes_with_and_without_the_report():
    from alpha.live.dashboard_v4.demo import create_demo_app

    plain = create_demo_app().test_client()
    assert plain.get("/performance?level=backtest").status_code == 404
    assert "Vs backtest" not in plain.get("/performance").data.decode()
    assert "Backtest check" not in plain.get("/system").data.decode()
    assert ">Backtest<" not in plain.get("/pods/demo_1_0").data.decode()

    client = create_demo_app(demo_backtest_bool=True).test_client()
    overview = client.get("/").data.decode()
    assert "Decision differs from the backtest." in overview
    tab = client.get("/pods/demo_1_0?tab=backtest").data.decode()
    assert "data-backtest-tab" in tab and "data-backtest-issue" in tab
    assert "data-backtest-rare" in client.get("/pods/demo_1_1").data.decode()
    page = client.get("/pods/demo_1_2").data.decode()
    assert "data-backtest-gap" in page and "Backtest, same days" in page
    performance = client.get("/performance?level=backtest")
    assert performance.status_code == 200 and "Cost gate · MR capsule" in performance.data.decode()
    assert client.get("/performance?level=backtest&download=csv&expected=" + "0" * 64).status_code == 400
    assert "Backtest check" in client.get("/system").data.decode()


def test_a_friday_report_stays_current_over_the_weekend():
    friday = datetime(2026, 12, 11, 13, 5, tzinfo=timezone.utc)  # 08:05 ET
    payload = dict(_report([_pod()], generated=friday), as_of_date_str="2026-12-11")
    assert not btd._stale_bool(payload, friday, datetime(2026, 12, 13, 20, 0, tzinfo=timezone.utc))  # Sunday
    assert not btd._stale_bool(payload, friday, datetime(2026, 12, 14, 14, 0, tzinfo=timezone.utc))  # Monday 09:00 ET
    assert btd._stale_bool(payload, friday, datetime(2026, 12, 14, 15, 30, tzinfo=timezone.utc))     # Monday 10:30 ET


@pytest.mark.parametrize("field,value", [("record_list", 7), ("curve_dict", {"point_list": 3}), ("rare_loss_dict", {"window_int": "x", "live_float": -0.1})])
def test_malformed_report_is_dropped_not_rendered(tmp_path, field, value):
    pod = _pod()
    pod[field] = value
    report = btd.load_report_dict(_provider(tmp_path, _report([pod])), as_of_ts=NOW)
    assert report is None or btd.pod_view_dict(report, "pod_a") is not None


def test_gate_pass_and_unknown_cause_add_no_overview_row(tmp_path):
    gate = {"status_str": "passed", "fill_count_int": 200, "fill_target_int": 200, "gate_bps_float": 4.0, "average_bps_float": 2.0,
            "cost_bps_list": [2.0] * 200, "pod_row_list": [{"pod_id_str": "pod_a", "fill_count_int": 200}]}
    unknown = {"window_int": 3, "live_float": -0.06, "backtest_float": None, "line_float": -0.05, "worst_float": -0.1, "cause_str": "unknown"}
    report = btd.load_report_dict(_provider(tmp_path, _report([_pod(rare=unknown)], gate=gate)), as_of_ts=NOW)
    assert btd.attention_list(report, {"pod_a": "A"}) == []
    assert btd.pod_view_dict(report, "pod_a")["rare_dict"]["state_str"] == "late"


@pytest.mark.parametrize("malformed_field_str,malformed_value_obj", [
    ("decision_dict", ["malformed"]),
    ("curve_dict", {"point_list": [{"date_str": "2026-12-09", "live_float": 0.0, "backtest_float": -1.0}]}),
    ("curve_dict", {"point_list": "bad"}),
])
def test_malformed_reports_are_rejected_before_system_or_chart_use(tmp_path, malformed_field_str, malformed_value_obj):
    pod_dict = _pod()
    pod_dict[malformed_field_str] = malformed_value_obj
    report_dict = btd.load_report_dict(_provider(tmp_path, _report([pod_dict])), as_of_ts=NOW)
    assert report_dict is None
    assert btd.system_row_dict(report_dict, as_of_ts=NOW)["state_str"] == "unknown"


def test_chart_rebase_handles_invalid_baseline_even_for_a_direct_caller():
    view_dict = {"point_list": [{"date_str": "2026-12-08", "live_float": 0.0, "backtest_float": -1.0},
                               {"date_str": "2026-12-09", "live_float": 0.0, "backtest_float": 0.0}]}
    chart_dict = {"available_bool": True}
    assert btd.add_backtest_series(chart_dict, [{"market_date_str": "2026-12-09", "cumulative_return_float": 0.0}],
                                  view_dict) == chart_dict


def test_chart_does_not_overlay_an_incomplete_selected_period():
    view_dict = {"point_list": [{"date_str": "2026-12-09", "live_float": 0.0, "backtest_float": 0.0},
                               {"date_str": "2026-12-10", "live_float": 0.0, "backtest_float": 0.0}]}
    chart_dict = {"available_bool": True}
    path_list = [{"market_date_str": "2026-12-08", "cumulative_return_float": -0.2},
                 {"market_date_str": "2026-12-09", "cumulative_return_float": 0.0}]
    assert btd.add_backtest_series(chart_dict, path_list, view_dict) == chart_dict


def test_single_pod_run_preserves_old_pod_and_cost_evidence_dates(tmp_path, monkeypatch):
    from alpha.live import backtest_check as bc, dashboard

    old_ts = NOW - timedelta(days=4)
    rare_dict = {"window_int": 3, "live_float": -0.06, "backtest_float": 0.0, "line_float": -0.05,
                 "worst_float": -0.2, "cause_str": "ours"}
    old_gate_dict = {"status_str": "failed", "fill_count_int": 200, "pod_row_list": [{"pod_id_str": "pod_b"}]}
    previous_dict = _report([_pod(), _pod("pod_b", "U2", status="differs", rare=rare_dict)], generated=old_ts, gate=old_gate_dict)
    previous_dict["as_of_date_str"] = old_ts.date().isoformat()
    provider_obj = _provider(tmp_path, previous_dict, identities=(("pod_a", "U1", "r1"), ("pod_b", "U2", "r1")))
    monkeypatch.setattr(dashboard, "DashboardApp", lambda **kwargs: provider_obj)
    monkeypatch.setattr(bc, "datetime", SimpleNamespace(now=lambda timezone_obj: NOW))
    fresh_dict = dict(_report([_pod()], generated=NOW), as_of_date_str=NOW.date().isoformat())
    monkeypatch.setattr(bc, "build_report_dict", lambda *args, **kwargs: fresh_dict)
    assert bc.main(["--pod-id", "pod_a", "--event-log-path", provider_obj.event_log_path_str,
                    "--performance-db", str(tmp_path / "absent"), "--skip-env-file"]) == 0
    report_dict = btd.load_report_dict(provider_obj, as_of_ts=NOW)
    assert not report_dict["stale_bool"]
    stale_view_dict = btd.pod_view_dict(report_dict, "pod_b")
    assert stale_view_dict["status_str"] == "not_checked"
    assert stale_view_dict["rare_dict"] is None
    assert "12-06" in stale_view_dict["detail_str"]
    assert btd.gate_view_dict(report_dict)["stale_bool"]
    assert btd.attention_list(report_dict, {"pod_a": "A", "pod_b": "B"}) == []
    assert btd.system_row_dict(report_dict, as_of_ts=NOW)["state_str"] == "warning"


def test_stale_report_never_renders_the_red_stop_rule_box(monkeypatch):
    from alpha.live.dashboard_v4.demo import create_demo_app
    from flask import render_template

    old_ts = NOW - timedelta(days=4)
    provider_obj = SimpleNamespace(get_backtest_check_report_dict=lambda **kwargs: _report([_pod(status="differs")], generated=old_ts))
    report_dict = btd.load_report_dict(provider_obj, as_of_ts=NOW)
    view_dict = btd.pod_view_dict(report_dict, "pod_a")
    app_obj = create_demo_app()
    with app_obj.test_request_context():
        rendered_str = render_template("_backtest_alerts.html", pod_page_dict={"backtest_dict": view_dict})
    assert "data-backtest-issue" not in rendered_str


def test_cost_pass_is_explicitly_a_component_and_coverage_is_visible(tmp_path):
    gate_dict = {"status_str": "passed", "fill_count_int": 200, "cost_bps_list": [], "pod_row_list": [{"pod_id_str": "pod_a"}]}
    pod_dict = _pod()
    pod_dict["coverage_reason_str"] = "Incomplete same-days coverage: 1 live / 0 reference dates missing."
    report_dict = btd.load_report_dict(_provider(tmp_path, _report([pod_dict], gate=gate_dict)), as_of_ts=NOW)
    gate_view_dict = btd.gate_view_dict(report_dict)
    assert gate_view_dict["label_str"] == "Cost component passed"
    assert "not full gate approval" in gate_view_dict["halt_str"]
    assert "Incomplete" in btd.performance_view_dict(report_dict, {})["pod_row_list"][0]["now_str"]


@pytest.mark.parametrize("field_str,value_obj", [
    ("pod_id_str", []), ("account_route_str", {}), ("release_id_str", None),
    ("record_list", [{"session_str": "2026-12-09", "status_str": []}]),
    ("record_list", [{"session_str": "bad", "status_str": "same"}]),
    ("reference_status_str", []), ("decision_dict", {"status_str": []}),
])
def test_corrupt_scalar_fields_are_dropped_for_both_reader_and_nightly_cache(tmp_path, field_str, value_obj):
    from alpha.live import backtest_check as bc
    pod_dict = _pod()
    pod_dict[field_str] = value_obj
    provider_obj = _provider(tmp_path, _report([pod_dict]))
    assert btd.load_report_dict(provider_obj, as_of_ts=NOW) is None
    assert bc._read_previous_dict(tmp_path / btd.REPORT_FILE_NAME_STR) == {}


def test_duplicate_pod_identities_are_not_ambiguous_evidence(tmp_path):
    from alpha.live import backtest_check as bc
    provider_obj = _provider(tmp_path, _report([_pod(), _pod(release="other")]))
    assert btd.load_report_dict(provider_obj, as_of_ts=NOW) is None
    assert bc._read_previous_dict(tmp_path / btd.REPORT_FILE_NAME_STR) == {}


def test_cost_evidence_is_not_reused_for_a_changed_account_or_release(tmp_path):
    gate_dict = {"status_str": "passed", "fill_count_int": 200, "pod_row_list": [{"pod_id_str": "pod_a"}],
                 "pod_identity_list": [{"pod_id_str": "pod_a", "account_route_str": "OLD_ACCOUNT", "release_id_str": "r1"}]}
    report_dict = btd.load_report_dict(_provider(tmp_path, _report([_pod()], gate=gate_dict)), as_of_ts=NOW)
    assert report_dict is not None
    assert report_dict["cost_gate_dict"] is None


def test_malformed_cost_rows_are_also_dropped_before_prior_cache_merge(tmp_path):
    from alpha.live import backtest_check as bc
    gate_dict = {"status_str": "failed", "fill_count_int": 200, "pod_row_list": ["bad"]}
    provider_obj = _provider(tmp_path, _report([_pod()], gate=gate_dict))
    assert btd.load_report_dict(provider_obj, as_of_ts=NOW) is None
    assert bc._read_previous_dict(tmp_path / btd.REPORT_FILE_NAME_STR) == {}


def test_overflowing_prior_numbers_do_not_prevent_a_fresh_nightly_report(tmp_path):
    from alpha.live import backtest_check as bc
    pod_dict = _pod()
    pod_dict["curve_dict"]["point_list"][0]["live_float"] = 10 ** 400
    provider_obj = _provider(tmp_path, _report([pod_dict]))
    assert bc._read_previous_dict(tmp_path / btd.REPORT_FILE_NAME_STR) == {}
    assert btd.load_report_dict(provider_obj, as_of_ts=NOW) is None
