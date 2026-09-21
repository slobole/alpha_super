"""Independent numerical and scope checks for the Positions enrichment layer."""

from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4 import positions_enrichment
from alpha.live.dashboard_v4.positions import _empty_page_dict


NOW_TS = datetime(2026, 9, 21, 21, tzinfo=UTC)
OBSERVED_STR = "2026-09-21T20:10:00+00:00"


def _mark_dict(symbol_str, shares_float, price_float, cost_float, conid_int=1):
    return {"symbol_str": symbol_str, "shares_float": shares_float,
        "market_price_float": price_float, "value_float": shares_float * price_float,
        "average_cost_float": cost_float, "unrealized_pnl_float": shares_float * (price_float - cost_float),
        "currency_str": "USD", "conid_int": conid_int}


def _activity_dict(*, before_float, after_float, delta_float, changed_bool=True, status_str="Filled"):
    return {"before_float": before_float, "after_float": after_float, "filled_delta_float": delta_float,
        "changed_bool": changed_bool, "status_str": status_str,
        "new_bool": before_float == 0 and after_float != 0,
        "closed_bool": before_float != 0 and after_float == 0}


@pytest.fixture
def valuation_case_obj(monkeypatch):
    broker_dict = {
        "a": {"available_bool": True, "position_list": [_mark_dict("AAA", 10., 110., 100.)],
            "cash_float": 100., "broker_nav_float": 1200., "observed_timestamp_str": OBSERVED_STR},
        "b": {"available_bool": True, "position_list": [_mark_dict("AAA", 20., 110., 120.),
            _mark_dict("BBB", 5., 100., 80., 2)], "cash_float": 400.,
            "broker_nav_float": 3100., "observed_timestamp_str": OBSERVED_STR},
    }
    activity_dict = {pod_id_str: {"available_bool": True, "symbol_dict": {}} for pod_id_str in broker_dict}
    source_dict = {}
    result_dict = _empty_page_dict()
    result_dict["source_fresh_bool"] = True
    for pod_id_str, holding_dict in broker_dict.items():
        identity_dict = {"pod_id_str": pod_id_str, "name_str": "Pod " + pod_id_str.upper(), "color_str": "#2a78d6"}
        source_dict[pod_id_str] = {
            "target_obj": SimpleNamespace(pod_id_str=pod_id_str), "identity_dict": identity_dict,
            "position_map_dict": {row_dict["symbol_str"]: row_dict["shares_float"] for row_dict in holding_dict["position_list"]},
            "position_asof_str": "2026-09-21 16:11:00 ET", "position_timestamp_str": "2026-09-21T20:11:00+00:00",
            "source_str": "broker_snapshot", "timestamp_basis_str": "observed"}
        result_dict["pod_row_list"].append({**identity_dict, "invested_str": "—", "cash_str": "—",
            "weight_str": "—", "pnl_str": "—", "pnl_tone_str": ""})
    call_list = []

    def marks_reader(target_obj, *, as_of_ts):
        assert as_of_ts == NOW_TS
        call_list.append(("marks", target_obj.pod_id_str))
        return deepcopy(broker_dict[target_obj.pod_id_str])

    def activity_reader(target_obj, *, as_of_ts):
        assert as_of_ts == NOW_TS
        call_list.append(("activity", target_obj.pod_id_str))
        return deepcopy(activity_dict[target_obj.pod_id_str])

    monkeypatch.setattr(positions_enrichment, "load_broker_holdings_dict", marks_reader)
    monkeypatch.setattr(positions_enrichment, "load_position_activity_dict", activity_reader)
    return SimpleNamespace(broker_dict=broker_dict, activity_dict=activity_dict,
        source_dict=source_dict, result_dict=result_dict, call_list=call_list)


def _enriched_tuple(valuation_case_obj, pod_str="all"):
    result_dict = deepcopy(valuation_case_obj.result_dict)
    row_list = positions_enrichment.enrich_positions_dict(result_dict,
        deepcopy(valuation_case_obj.source_dict), pod_str=pod_str, as_of_ts=NOW_TS)
    return result_dict, row_list


def test_full_book_uses_cash_in_weights_and_merges_shared_symbol_by_value(valuation_case_obj):
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    assert result_dict["money_complete_bool"] and result_dict["values_available_bool"]
    assert [row_dict["symbol_str"] for row_dict in row_list] == ["AAA", "BBB"]
    aaa_dict = row_list[0]
    assert aaa_dict["share_str"] == "30" and len(aaa_dict["pod_list"]) == 2
    assert aaa_dict["value_float"] == 3300 and aaa_dict["value_str"] == "$3,300.00"
    assert aaa_dict["weight_percent_float"] == pytest.approx(3300 / 4300 * 100)
    assert result_dict["tile_list"][0]["value_str"] == "$3,800.00"
    assert result_dict["total_dict"]["cash_str"] == "$500.00"
    assert result_dict["total_dict"]["weight_str"] == "100.0%"
    assert result_dict["pod_row_list"][0]["weight_str"] == "27.9%"
    assert result_dict["pod_row_list"][1]["weight_str"] == "72.1%"
    assert "2026-09-21 16:10:00" in result_dict["values_asof_str"]


def test_pnl_percent_is_cost_weighted_and_best_worst_rank_merged_dollars(valuation_case_obj):
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    aaa_dict, bbb_dict = row_list
    assert aaa_dict["pnl_float"] == -100 and aaa_dict["cost_float"] == 3400
    assert aaa_dict["pl_percent_str"] == "-2.9%"  # -100 / (10*100 + 20*120), not average return.
    assert bbb_dict["pnl_float"] == 100 and bbb_dict["pl_percent_str"] == "+25.0%"
    assert result_dict["tile_list"][1]["value_str"] == "$0.00"
    assert result_dict["tile_list"][2]["value_str"] == "BBB +$100.00"
    assert result_dict["tile_list"][3]["value_str"] == "AAA −$100.00"


def test_pod_filter_scopes_rows_without_changing_portfolio_tiles_or_denominator(valuation_case_obj):
    full_dict, _ = _enriched_tuple(valuation_case_obj)
    selected_dict, selected_list = _enriched_tuple(valuation_case_obj, "a")
    assert [row_dict["symbol_str"] for row_dict in selected_list] == ["AAA"]
    assert selected_list[0]["share_str"] == "10" and selected_list[0]["value_float"] == 1100
    assert selected_list[0]["weight_percent_float"] == pytest.approx(1100 / 4300 * 100)
    assert selected_list[0]["pl_percent_str"] == "+10.0%"
    for field_str in ("tile_list", "pod_row_list", "total_dict"):
        assert selected_dict[field_str] == full_dict[field_str]


def test_missing_one_pod_values_never_become_partial_portfolio_totals(valuation_case_obj):
    valuation_case_obj.broker_dict["a"] = {"available_bool": False, "reason_str": "Missing sample"}
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    row_by_symbol_dict = {row_dict["symbol_str"]: row_dict for row_dict in row_list}
    assert not result_dict["money_complete_bool"]
    assert result_dict["tile_list"][0]["value_str"] == "—"
    assert result_dict["total_dict"]["cash_str"] == result_dict["total_dict"]["weight_str"] == "—"
    assert row_by_symbol_dict["AAA"]["value_str"] == "—"  # Must not expose only Pod B's part.
    assert row_by_symbol_dict["BBB"]["value_str"] == "$500.00"
    assert all(row_dict["weight_percent_float"] is None for row_dict in row_list)
    assert result_dict["pod_row_list"][0]["invested_str"] == "—"
    assert result_dict["pod_row_list"][1]["invested_str"] == "$2,700.00"


def test_different_observation_days_keep_individual_facts_but_withhold_combined_values(valuation_case_obj):
    valuation_case_obj.broker_dict["a"]["observed_timestamp_str"] = "2026-09-18T20:10:00+00:00"
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    aaa_dict = next(row_dict for row_dict in row_list if row_dict["symbol_str"] == "AAA")
    assert not result_dict["money_complete_bool"] and aaa_dict["value_str"] == "—"
    assert result_dict["tile_list"][0]["value_str"] == "—"
    assert "different dates" in result_dict["values_note_str"]
    assert "2026-09-18" in result_dict["values_asof_str"] and "2026-09-21" in result_dict["values_asof_str"]


def test_complete_older_saved_day_stays_explicit_without_claiming_live_prices(valuation_case_obj):
    for holding_dict in valuation_case_obj.broker_dict.values():
        holding_dict["observed_timestamp_str"] = "2026-09-18T20:10:00+00:00"
    result_dict, _ = _enriched_tuple(valuation_case_obj)
    assert result_dict["money_complete_bool"]
    assert result_dict["values_asof_str"] == "IBKR values · 2026-09-18 16:10:00 ET"
    assert result_dict["financial_basis_str"] == "Saved IBKR holdings + cash"


def test_changed_quantity_never_receives_an_old_value_or_portfolio_weight(valuation_case_obj):
    valuation_case_obj.source_dict["a"]["position_map_dict"]["AAA"] = 99.
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    aaa_dict = next(row_dict for row_dict in row_list if row_dict["symbol_str"] == "AAA")
    assert aaa_dict["share_str"] == "119"
    assert aaa_dict["value_str"] == aaa_dict["pl_str"] == "—"
    assert not result_dict["money_complete_bool"]
    assert result_dict["values_note_str"] == "Holdings changed; values pending."
    assert all(row_dict["weight_percent_float"] is None for row_dict in row_list)


def test_absent_owned_position_source_prevents_complete_totals(valuation_case_obj):
    valuation_case_obj.source_dict.pop("a")
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    assert not result_dict["money_complete_bool"]
    assert result_dict["total_dict"]["cash_str"] == "—"
    assert row_list and all(row_dict["weight_percent_float"] is None for row_dict in row_list)
    assert not result_dict["changed_available_bool"]


def test_missing_cost_or_pnl_does_not_hide_values_or_invent_pnl(valuation_case_obj):
    valuation_case_obj.broker_dict["a"]["position_list"][0].pop("unrealized_pnl_float")
    valuation_case_obj.broker_dict["a"]["position_list"][0].pop("average_cost_float")
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    assert result_dict["money_complete_bool"]
    assert row_list[0]["value_float"] == 3300 and row_list[0]["pl_str"] == "—"
    assert result_dict["tile_list"][1]["value_str"] == "—"
    assert result_dict["tile_list"][1]["detail_str"] == "P&L incomplete"
    assert result_dict["pod_row_list"][0]["pnl_str"] == "—"
    assert result_dict["pod_row_list"][1]["pnl_str"] == "−$100.00"


def test_offsetting_long_short_legs_keep_gross_cost_for_pnl_percentage(valuation_case_obj):
    valuation_case_obj.broker_dict["b"].update(position_list=[_mark_dict("AAA", -10., 110., 120.)], cash_float=2000.)
    valuation_case_obj.source_dict["b"]["position_map_dict"] = {"AAA": -10.}
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    assert len(row_list) == 1
    row_dict = row_list[0]
    assert row_dict["share_str"] == "0" and row_dict["offset_bool"]
    assert row_dict["value_float"] == 0 and len(row_dict["pod_list"]) == 2
    assert row_dict["pnl_float"] == 200 and row_dict["cost_float"] == 2200
    assert row_dict["pl_percent_str"] == "+9.1%"
    assert result_dict["tile_list"][1]["value_str"] == "+$200.00"


def test_cash_only_has_valid_total_and_no_invented_best_or_worst(valuation_case_obj):
    for pod_id_str in valuation_case_obj.broker_dict:
        valuation_case_obj.broker_dict[pod_id_str]["position_list"] = []
        valuation_case_obj.source_dict[pod_id_str]["position_map_dict"] = {}
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    assert result_dict["money_complete_bool"] and row_list == []
    assert result_dict["total_dict"]["invested_str"] == "$0.00"
    assert result_dict["total_dict"]["cash_str"] == "$500.00"
    assert result_dict["tile_list"][1]["value_str"] == "$0.00"
    assert result_dict["tile_list"][2]["value_str"] == result_dict["tile_list"][3]["value_str"] == "—"


def test_closed_today_row_survives_zero_current_shares_and_has_no_open_pnl(valuation_case_obj):
    valuation_case_obj.activity_dict["a"]["symbol_dict"]["CLOSED"] = _activity_dict(before_float=8., after_float=0., delta_float=-8.)
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    closed_dict = next(row_dict for row_dict in row_list if row_dict["symbol_str"] == "CLOSED")
    assert closed_dict["share_str"] == "0" and closed_dict["value_float"] == 0
    assert closed_dict["changed_bool"] and closed_dict["today_str"] == "Closed · -8"
    assert closed_dict["pl_str"] == "$0.00"
    assert result_dict["changed_count_int"] == 1
    assert result_dict["tile_list"][1]["value_str"] == "$0.00"


@pytest.mark.parametrize("baseline_float", [0., 10.])
def test_net_zero_round_trip_is_traded_not_new_or_closed(valuation_case_obj, baseline_float):
    symbol_str = "ROUNDTRIP"
    valuation_case_obj.activity_dict["a"]["symbol_dict"][symbol_str] = _activity_dict(
        before_float=baseline_float, after_float=baseline_float, delta_float=0.)
    if baseline_float:
        valuation_case_obj.source_dict["a"]["position_map_dict"][symbol_str] = baseline_float
        valuation_case_obj.broker_dict["a"]["position_list"].append(_mark_dict(symbol_str, baseline_float, 5., 5., 3))
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    row_dict = next(row_dict for row_dict in row_list if row_dict["symbol_str"] == symbol_str)
    assert row_dict["changed_bool"] and row_dict["today_str"] == "Traded"
    assert result_dict["changed_count_int"] == 1


def test_pending_intent_is_visible_but_not_counted_as_filled_change(valuation_case_obj):
    valuation_case_obj.activity_dict["a"]["symbol_dict"]["PENDING"] = _activity_dict(
        before_float=None, after_float=None, delta_float=0., changed_bool=False, status_str="Buy pending")
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    pending_dict = next(row_dict for row_dict in row_list if row_dict["symbol_str"] == "PENDING")
    assert pending_dict["share_str"] == "0" and not pending_dict["changed_bool"]
    assert pending_dict["today_pending_bool"] and pending_dict["today_detail_str"] == "Buy pending"
    assert result_dict["changed_count_int"] == 0


def test_another_existing_holder_prevents_a_misleading_new_symbol_tag(valuation_case_obj):
    valuation_case_obj.activity_dict["a"]["symbol_dict"]["AAA"] = _activity_dict(before_float=0., after_float=10., delta_float=10.)
    _, row_list = _enriched_tuple(valuation_case_obj)
    row_dict = next(row_dict for row_dict in row_list if row_dict["symbol_str"] == "AAA")
    assert row_dict["today_str"] == "+10"


def test_unknown_activity_does_not_turn_into_no_trades_and_filter_scope_is_independent(valuation_case_obj):
    valuation_case_obj.activity_dict["b"] = {"available_bool": False, "symbol_dict": {}}
    full_dict, full_list = _enriched_tuple(valuation_case_obj)
    assert not full_dict["changed_available_bool"]
    assert all(row_dict["today_str"] == "Unknown" for row_dict in full_list)
    selected_dict, _ = _enriched_tuple(valuation_case_obj, "a")
    assert selected_dict["changed_available_bool"] and selected_dict["changed_count_int"] == 0


def test_stale_refresh_keeps_dated_money_but_does_not_claim_current_activity(valuation_case_obj):
    valuation_case_obj.result_dict["source_fresh_bool"] = False
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    assert result_dict["money_complete_bool"] and not result_dict["changed_available_bool"]
    assert all(row_dict["today_str"] == "Unknown" for row_dict in row_list)
    assert not any(call_tuple[0] == "activity" for call_tuple in valuation_case_obj.call_list)


def test_date_alignment_uses_et_instead_of_utc_calendar_date(valuation_case_obj):
    valuation_case_obj.broker_dict["a"]["observed_timestamp_str"] = "2026-09-20T22:00:00+00:00"
    valuation_case_obj.broker_dict["b"]["observed_timestamp_str"] = "2026-09-21T01:00:00+00:00"
    result_dict, _ = _enriched_tuple(valuation_case_obj)
    assert result_dict["money_complete_bool"]
    assert result_dict["values_asof_str"] == "IBKR values · 2026-09-20 18:00:00 → 2026-09-20 21:00:00 ET"


def test_negative_cash_is_signed_and_not_replaced_with_residual_or_clamped(valuation_case_obj):
    valuation_case_obj.broker_dict["a"]["cash_float"] = -200.
    valuation_case_obj.broker_dict["b"]["cash_float"] = 0.
    result_dict, _ = _enriched_tuple(valuation_case_obj)
    assert result_dict["money_complete_bool"]
    assert result_dict["total_dict"]["cash_str"] == "−$200.00"
    assert result_dict["tile_list"][0]["bar_percent_float"] == pytest.approx(3800 / 3600 * 100)
    assert result_dict["tile_list"][0]["detail_str"] == "105.6% · cash -5.6%"


def test_no_positive_book_denominator_never_produces_infinite_weights(valuation_case_obj):
    for pod_id_str in valuation_case_obj.broker_dict:
        valuation_case_obj.broker_dict[pod_id_str].update(position_list=[], cash_float=0.)
        valuation_case_obj.source_dict[pod_id_str]["position_map_dict"] = {}
    result_dict, row_list = _enriched_tuple(valuation_case_obj)
    assert row_list == []
    assert result_dict["total_dict"]["weight_str"] == "—"
    assert result_dict["tile_list"][0]["bar_percent_float"] is None


def test_activity_endpoint_mismatch_does_not_claim_new_or_closed_position(valuation_case_obj):
    valuation_case_obj.activity_dict["a"]["symbol_dict"]["AAA"] = _activity_dict(
        before_float=0., after_float=8., delta_float=8.)
    _, row_list = _enriched_tuple(valuation_case_obj, "a")
    assert row_list[0]["share_str"] == "10"
    assert row_list[0]["today_str"] == "+8"
