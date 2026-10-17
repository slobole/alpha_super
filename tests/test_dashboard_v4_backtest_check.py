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
    report = _report([_pod(), _pod(pod_id="pod_old", account="U9", release="r9"), _pod(release="r_other")])
    loaded = btd.load_report_dict(_provider(tmp_path, report), as_of_ts=NOW)
    assert [pod["pod_id_str"] for pod in loaded["pod_list"]] == ["pod_a"]
    assert loaded["stale_bool"] is False


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
