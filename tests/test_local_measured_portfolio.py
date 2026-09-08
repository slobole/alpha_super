"""Single-VPS measured history, using synthetic expanded IBKR source rows."""

from datetime import UTC, datetime
from dataclasses import replace
from io import BytesIO

import pytest
from pypdf import PdfReader

from alpha.live.ibkr_performance import PerformanceStore, PodPerformanceBinding
from alpha.live.release_manifest import load_release_list
from test_client_reporting import xml_text_str
from test_dashboard_local_workspace import build_fixture_app, file_snapshot_dict
from test_ibkr_nav_profile import expanded_nav_attributes_dict
from test_live_dashboard import _seed_eod_pod_state


def measured_fixture_app(tmp_path, monkeypatch, *, missing_day_str=None, retire_bool=False):
    app_obj = build_fixture_app(tmp_path, monkeypatch, finance_bool=False, new_pod_bool=False, expanded_bool=True)
    binding_list = [
        PodPerformanceBinding(account_route_str="U100", pod_id_str="pod_a",
            return_start_date_str="2026-06-01", return_end_date_str=None, enabled_bool=True),
        PodPerformanceBinding(account_route_str="U200", pod_id_str="pod_b",
            return_start_date_str="2026-06-17", enabled_bool=not retire_bool,
            return_end_date_str="2026-06-17" if retire_bool else None),
    ]
    # Real Pod ledgers establish the baseline; the normal dashboard derives
    # June1 / June17 without a reporting date override or mocked discovery.
    for release_obj in load_release_list(str(tmp_path / "releases")):
        baseline_ts = datetime(2026, 5, 29, 20, 10, tzinfo=UTC) if release_obj.pod_id_str == "pod_a" else datetime(2026, 6, 16, 20, 10, tzinfo=UTC)
        _seed_eod_pod_state(tmp_path / (release_obj.pod_id_str + ".sqlite3"), release_obj,
            total_value_float=1000, updated_timestamp_ts=baseline_ts)
    if retire_bool:
        # The separate retirement case supplies a historical closed window.
        monkeypatch.setattr("alpha.live.dashboard_v3.local_workspace.build_live_binding_obj_list",
            lambda **kwargs: binding_list)
    date_list = [f"2026-06-{day_int:02}" for day_int in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18)]
    attribute_list = [expanded_nav_attributes_dict(account_str, date_str="2026-05-29")
        for account_str in ("U100", "U200")]
    for index_int, date_str in enumerate(date_list):
        attribute_list.append(expanded_nav_attributes_dict("U100", date_str=date_str,
            opening_str=str(1000 + index_int * 10), closing_str=str(1010 + index_int * 10), mtm="10"))
        if date_str >= "2026-06-17":
            opening_int = 5000 if date_str == "2026-06-17" else 5050
            attribute_list.append(expanded_nav_attributes_dict("U200", date_str=date_str,
                opening_str=str(opening_int), closing_str=str(opening_int + 50), mtm="50", twr_str="1"))
        else:
            # Real broker history can precede the measured strategy. Neither
            # this large NAV nor its return / missing bridge fields may leak in.
            attribute_dict = expanded_nav_attributes_dict("U200", date_str=date_str,
                opening_str="100000", closing_str="99000", mtm="-1000", twr_str="-1")
            del attribute_dict["mtmAtPaxos"]
            attribute_list.append(attribute_dict)
    PerformanceStore(app_obj.config["performance_db_path_str"]).replace_range(
        xml_text_str=xml_text_str(attribute_list).replace('queryName="TEST_NAV"', 'queryName="ALPHA_DAILY_TWR"'),
        query_name_str="ALPHA_DAILY_TWR", request_from_date_str="2026-05-29", request_to_date_str="2026-06-18",
        binding_obj_list=binding_list, imported_timestamp_str="2026-09-04T12:00:00+00:00")
    if missing_day_str:
        from alpha.live.dashboard_v3 import client_views
        original_fn = client_views.load_broker_reporting_snapshot

        def missing_snapshot_fn(*args, **kwargs):
            snapshot_obj = original_fn(*args, **kwargs)
            return replace(snapshot_obj, row_tuple=tuple(row_obj for row_obj in snapshot_obj.row_tuple
                if (row_obj.account_route_str, row_obj.market_date_str) != ("U100", missing_day_str)))

        monkeypatch.setattr(client_views, "load_broker_reporting_snapshot", missing_snapshot_fn)
    return app_obj


def test_staggered_local_history_same_scope_on_screen_daily_and_pdf(tmp_path, monkeypatch):
    app_obj = measured_fixture_app(tmp_path, monkeypatch)
    client_obj = app_obj.test_client()
    before_dict = file_snapshot_dict(tmp_path)
    result_list = []
    # Twelve days of the original account, then entrant SOD capital included
    # in day 13's denominator. No reset and no investment gain from the entry.
    expected_return_float = 1.12 * (1 + 60 / 6120) - 1
    for view_str in ("overview", "performance", "report"):
        path_str = f"/clients/local/{view_str}?from=2026-06-01&to=2026-06-17"
        result_dict = client_obj.get(path_str + "&download=json").get_json()
        result_list.append(result_dict)
        assert result_dict["status_str"] == "ready"
        assert result_dict["scope_complete_bool"] and result_dict["coverage_complete_bool"]
        assert result_dict["opening_nav_float"] == 1000
        assert result_dict["closing_nav_float"] == 6180
        assert result_dict["pnl_float"] == 180
        assert result_dict["capital_movement_float"] == 0
        assert result_dict["scope_movement_float"] == 5000
        assert result_dict["twr_float"] == pytest.approx(expected_return_float)
        assert len(result_dict["twr_daily_list"]) == 13
        assert [row_dict["account_count_int"] for row_dict in result_dict["daily_book_list"]] == [1] * 12 + [2]
        assert [row_dict["pnl_float"] for row_dict in result_dict["daily_book_list"]] == [10] * 12 + [60]
        assert [row_dict["nav_float"] for row_dict in result_dict["daily_book_list"]] == list(range(1010, 1130, 10)) + [6180]
        assert [row_dict["from_date_str"] for row_dict in result_dict["strategy_list"]] == ["2026-06-01", "2026-06-17"]
        assert result_dict["strategy_list"][1]["twr_float"] == .01
        assert not result_dict["issue_list"]
        html_str = client_obj.get(path_str).get_data(as_text=True)
        assert "$180.00" in html_str and "13.10%" in html_str
        if view_str == "overview":
            assert 'data-account-unit="pct" aria-pressed="true"' in html_str
            assert "Incomplete IBKR return data" not in html_str
    assert len({result_dict["report_hash_str"] for result_dict in result_list}) == 1
    pdf_response_obj = client_obj.get("/clients/local/report?from=2026-06-01&to=2026-06-17&download=pdf&expected=" + result_list[0]["report_hash_str"])
    assert pdf_response_obj.status_code == 200
    pdf_text_str = " ".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(pdf_response_obj.data)).pages)
    assert "$180.00" in pdf_text_str and "13.10%" in pdf_text_str and "FINAL" in pdf_text_str
    assert file_snapshot_dict(tmp_path) == before_dict


def test_preentry_period_excludes_future_strategy_without_missing_data(tmp_path, monkeypatch):
    app_obj = measured_fixture_app(tmp_path, monkeypatch)
    result_dict = app_obj.test_client().get("/clients/local/overview?from=2026-06-01&to=2026-06-16&download=json").json
    assert result_dict["status_str"] == "ready"
    assert result_dict["pnl_float"] == 120 and result_dict["closing_nav_float"] == 1120
    assert result_dict["twr_float"] == pytest.approx(.12)
    assert len(result_dict["strategy_list"]) == 1


def test_missing_active_day_blocks_full_return_not_scope_or_account_history(tmp_path, monkeypatch):
    app_obj = measured_fixture_app(tmp_path, monkeypatch, missing_day_str="2026-06-08")
    result_dict = app_obj.test_client().get("/clients/local/overview?from=2026-06-01&to=2026-06-17&download=json").json
    assert result_dict["scope_complete_bool"] and not result_dict["coverage_complete_bool"]
    assert result_dict["pnl_float"] is None and result_dict["twr_float"] is None
    assert not result_dict["twr_daily_list"] and not result_dict["return_path_list"]
    assert result_dict["closing_nav_float"] == 6180
    assert next(row_dict for row_dict in result_dict["daily_book_list"] if row_dict["market_date_str"] == "2026-06-08")["nav_float"] is None


def test_retirement_removes_only_measured_scope_not_a_cash_withdrawal(tmp_path, monkeypatch):
    app_obj = measured_fixture_app(tmp_path, monkeypatch, retire_bool=True)
    result_dict = app_obj.test_client().get("/clients/local/overview?from=2026-06-01&to=2026-06-18&download=json").json
    assert result_dict["status_str"] == "ready"
    assert result_dict["closing_nav_float"] == 1140 and result_dict["pnl_float"] == 190
    assert result_dict["scope_movement_float"] == -50  # +5000 entry -5050 exit.
    assert result_dict["capital_movement_float"] == 0
    assert result_dict["twr_float"] == pytest.approx(1.12 * (1 + 60/6120) * (1 + 10/1130) - 1)


def test_all_preserves_first_strategy_date_and_dplus1_nav(tmp_path, monkeypatch):
    app_obj = measured_fixture_app(tmp_path, monkeypatch)
    from alpha.live.dashboard_v3 import client_views

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 6, 18, 23, tzinfo=UTC)

    monkeypatch.setattr(client_views, "datetime", FixedDateTime)
    client_obj = app_obj.test_client()
    result_dict = client_obj.get("/clients/local/overview?window=all&download=json").json
    assert result_dict["requested_from_date_str"] == "2026-06-01"
    assert result_dict["requested_to_date_str"] == "2026-06-17"
    assert result_dict["status_str"] == "ready"
    today_dict = client_obj.get("/clients/local/overview?from=2026-06-01&to=2026-06-18&download=json").json
    assert today_dict["closing_nav_float"] is None
    assert today_dict["twr_float"] is None and today_dict["status_str"] == "draft"
    assert all(row_dict["closing_nav_float"] is None for row_dict in today_dict["strategy_list"])


@pytest.mark.parametrize("empty_store_bool", [True, False])
def test_future_first_measurement_without_raw_rows_keeps_overview_accessible(tmp_path, monkeypatch, empty_store_bool):
    app_obj = build_fixture_app(tmp_path, monkeypatch, finance_bool=False, new_pod_bool=False, expanded_bool=True)
    if empty_store_bool:
        PerformanceStore(app_obj.config["performance_db_path_str"]).initialize()
    binding_list = [PodPerformanceBinding(account_route_str=account_str, pod_id_str=pod_str,
        return_start_date_str="2026-09-09", return_end_date_str=None, enabled_bool=True)
        for account_str, pod_str in (("U100", "pod_a"), ("U200", "pod_b"))]
    monkeypatch.setattr("alpha.live.dashboard_v3.local_workspace.build_live_binding_obj_list", lambda **kwargs: binding_list)
    from alpha.live.dashboard_v3 import client_views

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 8, 23, tzinfo=UTC)

    monkeypatch.setattr(client_views, "datetime", FixedDateTime)
    before_dict = file_snapshot_dict(tmp_path)
    response_obj = app_obj.test_client().get("/clients/local/overview")
    assert response_obj.status_code == 200
    assert 'aria-label="Current client operations"' in response_obj.get_data(as_text=True)
    assert file_snapshot_dict(tmp_path) == before_dict
