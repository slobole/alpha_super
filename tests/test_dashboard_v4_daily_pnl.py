"""Daily P&L views must preserve selected-period facts and missing evidence."""

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal

import pytest

from alpha.live.dashboard_v4.daily_pnl import build_daily_pnl_dict
from alpha.live.dashboard_v4.performance import _heat_class_str, build_performance_page_dict
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple


def _panel_dict(value_dict, *, session_set=None, from_str="2026-09-07", to_str="2026-09-11", return_dict=None):
    return build_daily_pnl_dict([{"market_date_str": key_str, "pnl_float": value_float} for key_str, value_float in value_dict.items()],
        [{"market_date_str": key_str, "return_float": value_float} for key_str, value_float in (return_dict or {}).items()],
        session_set if session_set is not None else {"2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"},
        from_str=from_str, to_str=to_str, heat_fn=_heat_class_str)


def test_full_period_not_thirty_and_does_not_mutate_financial_facts():
    book_list = [{"market_date_str": (date(2026, 1, 1) + timedelta(days=index_int)).isoformat(), "pnl_float": index_int}
        for index_int in range(370)]
    saved_list = deepcopy(book_list)
    session_set = {row_dict["market_date_str"] for row_dict in book_list}
    result_dict = build_daily_pnl_dict(book_list, [], session_set, from_str="2026-01-01", to_str="2027-01-05", heat_fn=_heat_class_str)
    assert len(result_dict["day_list"]) == 370
    assert result_dict["total_str"] == "+$68,265.00"
    assert len(result_dict["chart_dict"]["drawing_dict"]["series_list"][0]["point_list"]) == 370
    assert result_dict["last_dict"]["date_str"] == "2027-01-05"
    assert book_list == saved_list


def test_holiday_is_closed_missing_session_is_unknown_and_withholds_totals():
    result_dict = _panel_dict({"2026-09-07": 0, "2026-09-08": 20, "2026-09-10": -10, "2026-09-11": 0})
    cell_list = result_dict["week_list"][0]["cell_list"]
    assert cell_list[0]["day_dict"] is None and cell_list[0]["empty_str"] == "closed"
    assert cell_list[2]["day_dict"]["pnl_float"] is None
    assert cell_list[2]["day_dict"]["pnl_str"] == "—"
    assert result_dict["week_list"][0]["total_str"] == result_dict["total_str"] == "—"
    assert result_dict["missing_count_int"] == 1
    assert result_dict["best_dict"] is result_dict["worst_dict"] is None


def test_weekly_and_panel_totals_keep_cashflows_out_of_daily_return():
    result_dict = _panel_dict({"2026-09-08": 1.01, "2026-09-09": -.51, "2026-09-10": 0, "2026-09-11": 2},
        return_dict={"2026-09-08": .023, "2026-09-11": -.001})
    assert result_dict["total_str"] == "+$2.50"
    assert result_dict["week_list"][0]["total_detail_str"] == "+$2.50"
    assert result_dict["last_dict"]["return_str"] == "-0.10%"
    assert result_dict["day_list"][0]["heat_str"] == "heat-pos-3"
    assert result_dict["day_list"][1]["heat_str"] == "heat-neg-1"
    assert result_dict["last_dict"]["heat_str"] == "heat-pos-1"  # dollar sign, not return sign
    assert (result_dict["up_int"], result_dict["down_int"], result_dict["flat_int"]) == (2, 1, 1)
    assert result_dict["best_dict"]["date_str"] == "2026-09-11"
    assert result_dict["worst_dict"]["date_str"] == "2026-09-09"


def test_non_session_money_is_not_silently_discarded():
    result_dict = _panel_dict({"2026-09-07": 4, "2026-09-08": 1, "2026-09-09": 2, "2026-09-10": 3,
        "2026-09-11": 4, "2026-09-12": 5}, to_str="2026-09-13")
    assert result_dict["extra_count_int"] == 2
    assert result_dict["total_str"] == "+$19.00"
    assert result_dict["week_list"][0]["extra_list"][0]["date_str"] == "2026-09-12"
    assert result_dict["last_dict"]["date_str"] == "2026-09-11"
    assert result_dict["week_list"][0]["cell_list"][0]["day_dict"]["pnl_float"] == 4


@pytest.mark.parametrize("value_float", [0., -10., 10., None])
def test_single_session_flat_and_unavailable(value_float):
    result_dict = _panel_dict({"2026-09-08": value_float}, session_set={"2026-09-08"}, from_str="2026-09-08", to_str="2026-09-08")
    assert len(result_dict["day_list"]) == 1
    assert result_dict["complete_bool"] == (value_float is not None)
    if value_float is not None:
        assert sum(day_dict["extreme_bool"] for day_dict in result_dict["day_list"]) == 1
        assert result_dict["day_list"][0]["hit_left_float"] == 0
        assert result_dict["day_list"][0]["hit_width_float"] == 100
    else:
        assert result_dict["chart_dict"] is None


def test_empty_interval():
    result_dict = _panel_dict({}, session_set=set())
    assert result_dict["day_list"] == result_dict["week_list"] == []
    assert result_dict["last_dict"] is None and result_dict["total_str"] == "—"


def test_missing_dollars_are_never_green_even_with_a_known_return():
    result_dict = _panel_dict({"2026-09-08": None}, session_set={"2026-09-08"}, from_str="2026-09-08", to_str="2026-09-08",
        return_dict={"2026-09-08": .03})
    assert result_dict["last_dict"]["heat_str"] == ""
    assert result_dict["last_dict"]["return_str"] == "+3.00%"


def test_weekend_only_numbers_has_keyboard_entry():
    from pathlib import Path
    from jinja2 import Environment, FileSystemLoader
    result_dict = _panel_dict({"2026-09-12": 12.34}, session_set=set(), from_str="2026-09-12", to_str="2026-09-13")
    environment_obj = Environment(loader=FileSystemLoader(Path(__file__).parents[1] / "alpha/live/dashboard_v4/templates"))
    html_str = environment_obj.get_template("_daily_pnl.html").render(performance_dict={"daily_panel_dict": result_dict})
    numbers_str = html_str.split('id="daily-numbers"')[1]
    assert 'tabindex="0"' in numbers_str
    assert result_dict["total_str"] == "+$12.34"


@pytest.mark.parametrize("pnl_float,tone_str", [(12.34, "pos"), (-12.34, "neg"), (0., ""), (None, "")])
def test_rendered_readout_and_selectable_days_share_server_values(pnl_float, tone_str):
    import re
    from pathlib import Path
    from jinja2 import Environment, FileSystemLoader
    result_dict = _panel_dict({"2026-09-08": pnl_float}, session_set={"2026-09-08"}, from_str="2026-09-08", to_str="2026-09-08")
    environment_obj = Environment(loader=FileSystemLoader(Path(__file__).parents[1] / "alpha/live/dashboard_v4/templates"), autoescape=True)
    html_str = environment_obj.get_template("_daily_pnl.html").render(performance_dict={"daily_panel_dict": result_dict})
    day_dict = result_dict["last_dict"]
    source_list = re.findall(r'<(?:section|button)\b[^>]*data-daily-(?:panel|day)[^>]*>', html_str)
    assert len(source_list) == (2 if pnl_float is None else 3)  # panel, Numbers, optional bar
    for source_str in source_list:
        assert f'data-date-label="{day_dict["date_label_str"]}"' in source_str
        assert f'data-pnl="{day_dict["pnl_str"]}"' in source_str
        assert f'data-return="{day_dict["return_str"]}"' in source_str
        assert f'data-tone="{tone_str}"' in source_str
    assert f'<b data-daily-date>{day_dict["date_label_str"]}</b>' in html_str
    assert f'<b data-daily-value class="{tone_str}">{day_dict["pnl_str"]}</b>' in html_str
    assert f'<span data-daily-return>{day_dict["return_str"]}</span>' in html_str


def test_selection_bounds_and_partial_weeks():
    result_dict = _panel_dict({"2026-09-08": 10, "2026-09-09": 20, "2026-09-10": 30}, from_str="2026-09-09", to_str="2026-09-10")
    assert [day_dict["date_str"] for day_dict in result_dict["day_list"]] == ["2026-09-09", "2026-09-10"]
    assert result_dict["total_str"] == "+$50.00"
    assert result_dict["week_list"][0]["cell_list"][1]["empty_str"] == "·"


def test_zero_session_is_keyboard_reachable_and_hit_columns_cover_plot():
    result_dict = _panel_dict({"2026-09-08": 20, "2026-09-09": 0, "2026-09-10": -5, "2026-09-11": 2})
    day_list = result_dict["day_list"]
    assert all(day_dict["hit_width_float"] > 0 for day_dict in day_list)
    assert sum(day_dict["hit_width_float"] for day_dict in day_list) == pytest.approx(100)
    assert day_list[1]["pnl_float"] == 0
    assert len(result_dict["chart_dict"]["drawing_dict"]["bar_list"]) == 3
    assert sum(day_dict["extreme_bool"] for day_dict in day_list) == 2


def test_demo_all_matches_canonical_pnl_and_spans_more_than_thirty_sessions():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    try:
        result_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=DEMO_NOW_TS)
        daily_dict = result_dict["daily_panel_dict"]
        assert daily_dict["session_count_int"] > 30
        assert sum(Decimal(str(day_dict["pnl_float"])) for day_dict in daily_dict["day_list"]) == Decimal(str(result_dict["report_dict"]["pnl_float"]))
        assert daily_dict["day_list"][0]["date_str"] == result_dict["from_date_str"]
        assert daily_dict["last_dict"]["date_str"] == result_dict["to_date_str"]
    finally:
        provider_obj.close()
