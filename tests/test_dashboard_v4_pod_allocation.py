from copy import deepcopy
import math

import pytest

from alpha.live.dashboard_v4.pod_allocation import build_pod_allocation_dict


def _source_dict(position_list=None, *, cash_float=10, nav_float=100):
    return {"available_bool": True, "close_date_str": "2026-09-18", "nav_float": nav_float,
        "cash_float": cash_float, "position_list": position_list if position_list is not None else [
            {"symbol_str": "ZZZ", "shares_float": 2.5, "value_float": 30},
            {"symbol_str": "AAA", "shares_float": 7, "value_float": 45},
            {"symbol_str": "BBB", "shares_float": .125, "value_float": 15}]}


def _allocation_dict(source_dict):
    return build_pod_allocation_dict(source_dict, color_str="#2a78d6")


def test_sorted_clockwise_geometry_preserves_close_values_and_source():
    source_dict = _source_dict()
    source_dict["holdings_changed_bool"] = True
    before_dict = deepcopy(source_dict)
    result_dict = _allocation_dict(source_dict)
    assert result_dict["available_bool"] and result_dict["donut_available_bool"]
    assert result_dict["close_date_str"] == "2026-09-18"
    assert result_dict["holdings_changed_bool"] is True
    assert result_dict["cash_percent_str"] == "10.0%"
    assert [row_dict["symbol_str"] for row_dict in result_dict["row_list"]] == ["AAA", "BBB", "ZZZ"]
    assert [slice_dict["label_str"] for slice_dict in result_dict["slice_list"]] == ["AAA", "ZZZ", "BBB", "Cash"]
    assert [slice_dict["color_str"] for slice_dict in result_dict["slice_list"]] == ["#2a78d6"] * 3 + ["#dfe3e9"]
    assert result_dict["slice_list"][0]["path_str"].startswith("M 116.000,43.000 A 82,82 0 0 1")
    assert result_dict["label_list"][0]["x_float"] > 116
    assert [label_dict["label_str"] for label_dict in result_dict["label_list"]] == ["AAA", "ZZZ", "BBB"]
    assert all(row_dict["target_percent_float"] is None and row_dict["new_bool"] is False for row_dict in result_dict["row_list"])
    assert result_dict["row_list"][1]["shares_str"] == "0.125"
    assert result_dict["row_list"][0]["value_str"] == "45.00"
    assert result_dict["row_list"][0]["bar_width_float"] == 100
    assert result_dict["row_list"][1]["bar_width_float"] == pytest.approx(100 / 3)
    assert result_dict["color_str"] == "#2a78d6"
    assert result_dict["cash_row_dict"]["key_str"] == "cash"
    assert source_dict == before_dict


@pytest.mark.parametrize("cash_only_bool", [True, False])
def test_cash_only_and_single_holding_make_a_complete_circle(cash_only_bool):
    source_dict = _source_dict([] if cash_only_bool else [{"symbol_str": "ONLY", "shares_float": 3, "value_float": 100}],
        cash_float=100 if cash_only_bool else 0)
    result_dict = _allocation_dict(source_dict)
    assert result_dict["available_bool"] and result_dict["donut_available_bool"]
    assert len(result_dict["slice_list"]) == 1
    assert result_dict["slice_list"][0]["path_str"].count(" A ") == 4
    assert result_dict["slice_list"][0]["label_str"] == ("Cash" if cash_only_bool else "ONLY")
    assert result_dict["cash_percent_str"] == ("100.0%" if cash_only_bool else "0.0%")
    assert result_dict["cash_row_dict"]["value_float"] == (100 if cash_only_bool else 0)


def test_more_than_twelve_holdings_are_not_grouped_or_dropped_and_ties_are_stable():
    position_list = [{"symbol_str": f"S{index_int:02}", "shares_float": index_int + 1, "value_float": 10}
        for index_int in reversed(range(15))]
    result_dict = _allocation_dict(_source_dict(position_list, cash_float=50, nav_float=200))
    assert len(result_dict["row_list"]) == 15 and len(result_dict["slice_list"]) == 16
    assert [slice_dict["label_str"] for slice_dict in result_dict["slice_list"]] == [f"S{index_int:02}" for index_int in range(15)] + ["Cash"]
    assert [label_dict["label_str"] for label_dict in result_dict["label_list"]] == ["Cash"]
    assert all(row_dict["bar_width_float"] == 100 for row_dict in result_dict["row_list"])
    assert all(row_dict["weight_float"] == .05 for row_dict in result_dict["row_list"])
    assert all(math.isfinite(label_dict[axis_str]) for label_dict in result_dict["label_list"] for axis_str in ("x_float", "y_float"))


@pytest.mark.parametrize("position_list,cash_float", [
    ([{"symbol_str": "SHORT", "shares_float": -2, "value_float": -20}], 120),
    ([{"symbol_str": "LONG", "shares_float": 12, "value_float": 120}], -20),
    ([{"symbol_str": "SHORT", "shares_float": -2, "value_float": 0}], 100),
])
def test_shorts_and_negative_cash_keep_signed_table_without_donut(position_list, cash_float):
    result_dict = _allocation_dict(_source_dict(position_list, cash_float=cash_float))
    assert result_dict["available_bool"] and not result_dict["donut_available_bool"]
    assert not result_dict["slice_list"] and not result_dict["label_list"]
    assert result_dict["row_list"][0]["weight_float"] == position_list[0]["value_float"] / 100
    assert result_dict["cash_row_dict"]["weight_float"] == cash_float / 100
    weight_float = result_dict["row_list"][0]["weight_float"]
    assert result_dict["row_list"][0]["bar_width_float"] == (weight_float * 100 if 0 <= weight_float <= 1 else None)
    assert "table only" in result_dict["reason_str"]


@pytest.mark.parametrize("nav_float,available_bool", [(100.01, True), (99.99, True), (100.011, False), (99.989, False), (200, False)])
def test_absolute_cent_gate_never_uses_relative_tolerance_or_residual_cash(nav_float, available_bool):
    result_dict = _allocation_dict(_source_dict(nav_float=nav_float))
    assert result_dict["available_bool"] is available_bool
    if available_bool:
        assert result_dict["row_list"][0]["weight_float"] == pytest.approx(45 / nav_float)
        assert result_dict["cash_row_dict"]["value_float"] == 10
    else:
        assert result_dict["row_list"] == [] and result_dict["slice_list"] == []
        assert result_dict["reason_str"] == "Closing values do not match NAV"


def test_one_cent_is_not_scaled_with_account_size():
    result_dict = _allocation_dict(_source_dict([{"symbol_str": "BIG", "shares_float": 500, "value_float": 999999}], cash_float=.98, nav_float=1000000))
    assert not result_dict["available_bool"]


@pytest.mark.parametrize("cash_only_bool", [True, False])
@pytest.mark.parametrize("nav_float,value_float", [(.01, .02), (100, 100.01)])
def test_accepted_penny_residual_cannot_draw_a_slice_above_one_hundred_percent(cash_only_bool, nav_float, value_float):
    position_list = [] if cash_only_bool else [{"symbol_str": "ONLY", "shares_float": 1, "value_float": value_float}]
    result_dict = _allocation_dict(_source_dict(position_list, cash_float=value_float if cash_only_bool else 0, nav_float=nav_float))
    assert result_dict["available_bool"] and not result_dict["donut_available_bool"]
    assert result_dict["slice_list"] == [] and result_dict["label_list"] == []
    row_dict = result_dict["cash_row_dict"] if cash_only_bool else result_dict["row_list"][0]
    assert row_dict["weight_float"] == pytest.approx(value_float / nav_float)
    assert row_dict["bar_width_float"] is None
    assert result_dict["reason_str"] == "Weight exceeds 100%: table only"


@pytest.mark.parametrize("field_str,value_obj", [
    ("available_bool", False), ("available_bool", 1), ("close_date_str", None),
    ("close_date_str", "2026-09-31"), ("close_date_str", "20260918"),
    ("nav_float", 0), ("nav_float", -100), ("nav_float", True), ("nav_float", float("nan")), ("nav_float", 10 ** 400),
    ("cash_float", None), ("cash_float", float("inf")), ("position_list", None), ("position_list", {}),
])
def test_unverified_or_malformed_snapshot_has_no_values_or_geometry(field_str, value_obj):
    source_dict = _source_dict()
    source_dict[field_str] = value_obj
    result_dict = _allocation_dict(source_dict)
    assert not result_dict["available_bool"] and not result_dict["donut_available_bool"]
    assert result_dict["row_list"] == [] and result_dict["slice_list"] == [] and result_dict["cash_row_dict"] == {}


@pytest.mark.parametrize("field_str,value_obj", [
    ("symbol_str", ""), ("symbol_str", " AAA "), ("symbol_str", "AAA"),
    ("shares_float", True), ("shares_float", None), ("shares_float", 0), ("shares_float", -1),
    ("value_float", float("nan")), ("value_float", float("inf")), ("value_float", -30),
    ("close_date_str", "2026-09-17"),
])
def test_one_invalid_duplicate_or_differently_dated_holding_withholds_whole_value_table(field_str, value_obj):
    source_dict = _source_dict()
    source_dict["position_list"][0][field_str] = value_obj
    result_dict = _allocation_dict(source_dict)
    assert not result_dict["available_bool"] and not result_dict["donut_available_bool"]
    assert result_dict["row_list"] == []


def test_missing_source_and_reason_are_safe_and_zero_value_holding_is_not_a_slice():
    assert _allocation_dict(None)["available_bool"] is False
    assert _allocation_dict({"available_bool": False, "reason_str": "Closing values missing"})["reason_str"] == "Closing values missing"
    result_dict = _allocation_dict(_source_dict([{"symbol_str": "ZERO", "shares_float": 2, "value_float": 0}], cash_float=100))
    assert result_dict["available_bool"] and result_dict["donut_available_bool"]
    assert len(result_dict["row_list"]) == 1 and result_dict["row_list"][0]["weight_str"] == "0.0%"
    assert [slice_dict["label_str"] for slice_dict in result_dict["slice_list"]] == ["Cash"]
