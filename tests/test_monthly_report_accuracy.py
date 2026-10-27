"""Independent investor-report checks over synthetic broker evidence only."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import hashlib
import xml.etree.ElementTree as ElementTree

from flask import Flask, render_template
import pytest

from alpha.live.client_reporting import ClientReportingError, parse_broker_nav_import
from alpha.live.dashboard_v4 import monthly_report
from alpha.live.dashboard_v4.monthly_report import build_monthly_report_dict, _money_str, _percent_str
from test_client_reporting import xml_text_str
from test_dashboard_v4_monthly_report import _sample_tuple, SAMPLE_AS_OF_TS, _visible_text_str


def _expected_tuple(snapshot_obj, route_set, *, first_str="2026-08-03", last_str="2026-08-31",
                    first_by_route_dict=None, last_by_route_dict=None):
    """Independent EOD NAV identity; no report or Performance calculation helper."""
    selected_list = [row_obj for row_obj in snapshot_obj.row_tuple
        if row_obj.account_route_str in route_set and max(first_str, (first_by_route_dict or {}).get(row_obj.account_route_str, first_str))
        <= row_obj.market_date_str <= min(last_str, (last_by_route_dict or {}).get(row_obj.account_route_str, last_str))]
    growth_decimal, profit_decimal = Decimal(1), Decimal(0)
    for date_str in sorted({row_obj.market_date_str for row_obj in selected_list}):
        daily_list = [row_obj for row_obj in selected_list if row_obj.market_date_str == date_str]
        opening_decimal = sum((row_obj.opening_nav_decimal for row_obj in daily_list), Decimal(0))
        daily_profit_decimal = sum((row_obj.closing_nav_decimal - row_obj.opening_nav_decimal
            - sum((Decimal(row_obj.attribute_dict[field_str]) for field_str in (
                "depositsWithdrawals", "internalCashTransfers", "assetTransfers", "debitCardActivity", "billPay")), Decimal(0))
            - Decimal(row_obj.attribute_dict["linkingAdjustments"]) for row_obj in daily_list), Decimal(0))
        growth_decimal *= 1 + daily_profit_decimal / opening_decimal
        profit_decimal += daily_profit_decimal
    closing_decimal = sum((row_obj.closing_nav_decimal for row_obj in selected_list
        if row_obj.market_date_str == last_str), Decimal(0))
    return float(growth_decimal - 1), float(profit_decimal), float(closing_decimal)


def _report_dict(workspace_dict, snapshot_obj, *, month_str="2026-08", program_dict=None):
    return build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS,
        month_str=month_str, program_by_pod_dict=program_dict)


def _printed_body_str(report_dict):
    template_path_obj = Path(__file__).resolve().parents[1] / "alpha/live/dashboard_v4/templates"
    app_obj = Flask(__name__, template_folder=str(template_path_obj))
    app_obj.add_url_rule("/assets/<path:filename>", endpoint="assets", view_func=lambda filename: "")
    with app_obj.test_request_context():
        html_str = render_template("monthly_report.html", report_dict=report_dict)
    return _visible_text_str(html_str.split('<main class="mr-pages"', 1)[1].split(">", 1)[1].split("</main>", 1)[0])


def _amount_decimal(value_str):
    return Decimal(value_str.replace("$", "").replace(",", "").replace("−", "-") or "0")


def test_retired_program_keeps_its_own_return_and_zero_closing_allocation():
    workspace_dict, snapshot_obj = _sample_tuple(effective_to_dict={"U200": "2026-08-13"})
    report_dict = _report_dict(workspace_dict, snapshot_obj, program_dict={"pod_taa": "Growth", "pod_ndx": "Steady"})
    program_dict = {card_dict["name_str"]: card_dict for card_dict in report_dict["program_list"]}
    growth_return_float, growth_pnl_float, growth_close_float = _expected_tuple(snapshot_obj, {"U100"})
    steady_return_float, steady_pnl_float, _ = _expected_tuple(snapshot_obj, {"U200"}, last_str="2026-08-13")
    assert set(program_dict) == {"Growth", "Steady"}
    for name_str, return_float, pnl_float in (("Growth", growth_return_float, growth_pnl_float),
            ("Steady", steady_return_float, steady_pnl_float)):
        assert program_dict[name_str]["stat_list"][-1]["value_str"] == _percent_str(return_float)
        assert program_dict[name_str]["change_str"] == _money_str(pnl_float, signed_bool=True) + " this month"
    assert program_dict["Growth"]["value_str"] == _money_str(growth_close_float)
    assert program_dict["Steady"]["value_str"] == "$0" and program_dict["Steady"]["closed_bool"]
    assert "closed 2026-08-13" in program_dict["Steady"]["since_str"]
    assert "Steady" not in [group_dict["name_str"] for group_dict in report_dict["split_dict"]["group_list"]]
    assert [row_dict["name_str"] for row_dict in report_dict["month_table_dict"]["row_list"]] == [
        "Growth", "Pod 1 · U···100", "Steady", "Pod 2 · U···200"]
    assert not report_dict["draft_bool"]


def test_cross_program_transfer_resolves_without_changing_broker_rows_or_account_twr():
    workspace_dict, snapshot_obj = _sample_tuple()
    replacement_list = []
    for row_obj in snapshot_obj.row_tuple:
        attribute_dict = dict(row_obj.attribute_dict)
        if row_obj.market_date_str == "2026-08-24" and row_obj.account_route_str in {"U100", "U200"}:
            # Replace an existing capital field without changing net capital or NAV.
            attribute_dict["internalCashTransfers"] = attribute_dict["depositsWithdrawals"]
            attribute_dict["depositsWithdrawals"] = "0"
        if row_obj.account_route_str == "U100":
            attribute_dict["twr"] = ".5"
            row_obj = replace(row_obj, twr_decimal=Decimal(".005"))
        replacement_list.append(replace(row_obj, attribute_dict=attribute_dict))
    snapshot_obj = replace(snapshot_obj, row_tuple=tuple(replacement_list))
    original_obj, original_workspace_dict = deepcopy(snapshot_obj), deepcopy(workspace_dict)
    report_dict = _report_dict(workspace_dict, snapshot_obj, program_dict={"pod_taa": "Growth", "pod_ndx": "Steady"})
    assert not report_dict["draft_bool"]
    for card_dict in report_dict["program_list"]:
        route_str = "U100" if card_dict["name_str"] == "Growth" else "U200"
        expected_return_float, _, _ = _expected_tuple(snapshot_obj, {route_str})
        assert card_dict["stat_list"][-1]["value_str"] == _percent_str(expected_return_float)
        assert card_dict["chart_dict"]["available_bool"]
    account_days_int = sum(row_obj.account_route_str == "U100" and row_obj.market_date_str <= "2026-08-31"
        for row_obj in snapshot_obj.row_tuple)
    pod_row_dict = next(row_dict for row_dict in report_dict["month_table_dict"]["row_list"]
        if row_dict["name_str"] == "Pod 1 · U···100")
    assert pod_row_dict["since_str"] == _percent_str(float(Decimal("1.005") ** account_days_int - 1), suffix_str="")
    assert snapshot_obj == original_obj and workspace_dict == original_workspace_dict


def test_usd_vault_fx_gain_changes_investor_headline_but_not_program_performance():
    base_workspace_dict, base_snapshot_obj = _sample_tuple(extra_event_dict={("U900", "2026-08-31"): {"mtm": "0"}})
    fx_workspace_dict, fx_snapshot_obj = _sample_tuple(extra_event_dict={("U900", "2026-08-31"): {"mtm": "300"}})
    base_dict = _report_dict(base_workspace_dict, base_snapshot_obj)
    fx_dict = _report_dict(fx_workspace_dict, fx_snapshot_obj)
    expected_return_float, expected_pnl_float, expected_close_float = _expected_tuple(fx_snapshot_obj, {"U100", "U200", "U900"})
    assert fx_dict["summary_scope_str"] == "All your accounts · USD"
    assert fx_dict["money_row_list"][0]["value_str"] == _money_str(expected_close_float)
    assert fx_dict["money_row_list"][1]["value_str"] == _money_str(expected_pnl_float, signed_bool=True)
    assert fx_dict["money_row_list"][1]["pct_str"] == _percent_str(expected_return_float)
    assert _amount_decimal(fx_dict["money_row_list"][0]["value_str"]) - _amount_decimal(base_dict["money_row_list"][0]["value_str"]) == 300
    assert fx_dict["program_list"] == base_dict["program_list"]
    assert fx_dict["other_account_list"][0]["pnl_str"] == "+$300"
    assert "Non-USD holdings" in fx_dict["scope_note_str"] and "FX" in fx_dict["scope_note_str"]
    assert sum(float(group_dict["pct_str"].rstrip("%")) for group_dict in fx_dict["split_dict"]["group_list"]) == pytest.approx(100, abs=.15)


@pytest.mark.parametrize("currency_target_str", ["account", "nav"])
def test_actual_non_usd_base_source_is_rejected_instead_of_silently_converted(currency_target_str):
    _, snapshot_obj = _sample_tuple()
    root_obj = ElementTree.fromstring(xml_text_str([dict(row_obj.attribute_dict) for row_obj in snapshot_obj.row_tuple]))
    element_obj = root_obj.find(".//AccountInformation" if currency_target_str == "account" else ".//ChangeInNAV")
    element_obj.set("currency", "ILS")
    raw_xml_str = ElementTree.tostring(root_obj, encoding="unicode")
    with pytest.raises(ClientReportingError, match="currency|USD"):
        parse_broker_nav_import(raw_xml_str,
            allowed_account_set={"U100", "U200", "U900"}, query_name_str="TEST_NAV",
            source_import_id_int=1, source_checksum_str=hashlib.sha256(raw_xml_str.encode()).hexdigest())


def _holding_fixture_tuple(*, position_float, cash_float=None, cash_date_str=None):
    workspace_dict, snapshot_obj = _sample_tuple()
    nav_row_obj = next(row_obj for row_obj in snapshot_obj.row_tuple
        if row_obj.account_route_str == "U100" and row_obj.market_date_str == "2026-08-31")
    snapshot_obj = replace(snapshot_obj, row_tuple=(replace(nav_row_obj, closing_nav_decimal=Decimal("1000")),))
    close_dict = {"available_bool": True, "position_list": [{"symbol_str": "SPY", "shares_float": 1., "value_float": position_float}],
        "cash_float": cash_float, "cash_date_str": cash_date_str}
    provider_obj = SimpleNamespace(get_close_holdings_dict=lambda *args, **kwargs: close_dict)
    pod_dict = {"pod_id_str": "pod_taa", "account_route_str": "U100", "close_str": "2026-08-31", "active_bool": True}
    holding_dict = monthly_report._pod_holdings_dict(pod_dict,
        client_dict={**workspace_dict["client_dict"], "is_demo": True}, snapshot_obj=snapshot_obj,
        provider_obj=provider_obj, as_of_ts=SAMPLE_AS_OF_TS, database_path_str=None)
    return pod_dict, holding_dict


@pytest.mark.parametrize("cash_float,cash_date_str", [(None, None), (100., "2026-08-28"),
    (float("nan"), "2026-08-31"), (float("inf"), "2026-08-31"), (99., "2026-08-31")])
def test_missing_stale_or_invalid_cash_never_becomes_nav_residual(cash_float, cash_date_str):
    _, holding_dict = _holding_fixture_tuple(position_float=900., cash_float=cash_float, cash_date_str=cash_date_str)
    assert not holding_dict["available_bool"]
    assert holding_dict["cash_float"] is None and holding_dict["position_list"] == []


def test_verified_penny_reconciliation_keeps_nav_denominator_and_negative_cash_has_no_donut():
    pod_dict, holding_dict = _holding_fixture_tuple(position_float=999.99, cash_float=0., cash_date_str="2026-08-31")
    assert holding_dict["available_bool"] and holding_dict["total_float"] == 1000.
    assert monthly_report._program_holding_dict("Growth", [pod_dict], {"pod_taa": holding_dict})["available_bool"]
    _, invalid_dict = _holding_fixture_tuple(position_float=999.98, cash_float=0., cash_date_str="2026-08-31")
    assert not invalid_dict["available_bool"]
    pod_dict, negative_dict = _holding_fixture_tuple(position_float=1000.001, cash_float=-.001, cash_date_str="2026-08-31")
    assert negative_dict["available_bool"]
    assert not monthly_report._program_holding_dict("Growth", [pod_dict], {"pod_taa": negative_dict})["available_bool"]


def test_many_month_movements_fold_without_losing_any_dollars():
    _, snapshot_obj = _sample_tuple()
    august_day_list = sorted({row_obj.market_date_str for row_obj in snapshot_obj.row_tuple if row_obj.market_date_str.startswith("2026-08")})
    event_dict = {("U100", day_str): {"depositsWithdrawals": "1"} for day_str in august_day_list}
    event_dict[("U200", "2026-08-24")] = {"depositsWithdrawals": "0"}
    workspace_dict, snapshot_obj = _sample_tuple(extra_event_dict=event_dict)
    flow_dict = _report_dict(workspace_dict, snapshot_obj)["flow_dict"]
    assert len(august_day_list) > 12 and len(flow_dict["row_list"]) == 12
    assert flow_dict["row_list"][0]["kind_str"] == "Earlier, net"
    assert sum(_amount_decimal(row_dict["total_str"]) for row_dict in flow_dict["row_list"]) == 30000 + len(august_day_list)
    for row_dict in flow_dict["row_list"]:
        assert sum(_amount_decimal(cell_dict["value_str"]) for cell_dict in row_dict["cell_list"]) == _amount_decimal(row_dict["total_str"])


def test_current_month_movements_exclude_all_prior_month_flows():
    workspace_dict, snapshot_obj = _sample_tuple(extra_event_dict={
        ("U100", "2026-09-01"): {"depositsWithdrawals": "37"},
        ("U100", "2026-09-02"): {"depositsWithdrawals": "-12"}})
    report_dict = _report_dict(workspace_dict, snapshot_obj, month_str="2026-09")
    assert [row_dict["date_str"] for row_dict in report_dict["flow_dict"]["row_list"]] == ["Sep 1, 2026", "Sep 2, 2026"]
    assert sum(_amount_decimal(row_dict["total_str"]) for row_dict in report_dict["flow_dict"]["row_list"]) == 25
    assert "Opening capital" not in [row_dict["kind_str"] for row_dict in report_dict["flow_dict"]["row_list"]]


def test_printed_flow_cells_still_add_up_when_source_movements_have_cents():
    workspace_dict, snapshot_obj = _sample_tuple(extra_event_dict={
        ("U100", "2026-09-01"): {"depositsWithdrawals": ".50"},
        ("U200", "2026-09-01"): {"depositsWithdrawals": ".50"}})
    report_dict = _report_dict(workspace_dict, snapshot_obj, month_str="2026-09")
    row_dict = report_dict["flow_dict"]["row_list"][0]
    assert row_dict["total_str"] == "+$1.00"
    assert [cell_dict["value_str"] for cell_dict in row_dict["cell_list"]] == ["+$0.50", "+$0.50"]
    assert sum(_amount_decimal(cell_dict["value_str"]) for cell_dict in row_dict["cell_list"]) == _amount_decimal(row_dict["total_str"])
    _, expected_pnl_float, expected_close_float = _expected_tuple(snapshot_obj, {"U100", "U200", "U900"},
        first_str="2026-09-01", last_str="2026-09-03")
    expected_open_float = float(sum((row_obj.opening_nav_decimal for row_obj in snapshot_obj.row_tuple
        if row_obj.market_date_str == "2026-09-01"), Decimal(0)))
    bridge_dict = {item_dict["label_str"]: item_dict["value_str"] for item_dict in report_dict["bridge_row_list"]}
    assert bridge_dict == {"Opening value": _money_str(expected_open_float, cents_bool=True),
        "Net capital movements": "+$1.00", "Investment gain/loss": _money_str(expected_pnl_float, signed_bool=True, cents_bool=True),
        "Closing value": _money_str(expected_close_float, cents_bool=True)}
    assert _amount_decimal(bridge_dict["Opening value"]) + _amount_decimal(bridge_dict["Net capital movements"]) \
        + _amount_decimal(bridge_dict["Investment gain/loss"]) == _amount_decimal(bridge_dict["Closing value"])


def test_printed_body_omits_method_section_but_keeps_missing_evidence_and_partial_dates():
    workspace_dict, snapshot_obj = _sample_tuple()
    report_dict = _report_dict(workspace_dict, snapshot_obj)
    body_str = _printed_body_str(report_dict)
    for text_str in ("Holdings on Aug 31 were not saved.", "from 2026-08-03", "All your accounts", "USD", "VAULT"):
        assert text_str in body_str
    for text_str in ("About these numbers", "end-of-day capital flows", "Prepared ", "Source:"):
        assert text_str not in body_str
    partial_dict = _report_dict(workspace_dict, snapshot_obj, month_str="2026-09")
    partial_body_str = _printed_body_str(partial_dict)
    assert partial_dict["draft_bool"] and partial_body_str.count("Draft · Month in progress, through Sep 3") == 2
    assert "September 2026 · to Sep 3" in partial_body_str


def test_printed_demo_cannot_be_mistaken_for_real_broker_report():
    workspace_dict, snapshot_obj = _sample_tuple()
    workspace_dict["client_dict"]["is_demo"] = True
    body_str = _printed_body_str(_report_dict(workspace_dict, snapshot_obj))
    assert "DEMONSTRATION · synthetic data" in body_str
    assert body_str.count("DEMONSTRATION · synthetic data") == 2
    assert "Source: saved IBKR statements" not in body_str


def test_vault_horizon_continues_after_every_program_retires_without_inventing_returns():
    closing_date_dict = {"U100": "2026-08-13", "U200": "2026-08-13"}
    workspace_dict, snapshot_obj = _sample_tuple(effective_to_dict=closing_date_dict, last_row_dict=closing_date_dict)
    program_map_dict = {"pod_taa": "Growth", "pod_ndx": "Steady"}
    august_dict = _report_dict(workspace_dict, snapshot_obj, program_dict=program_map_dict)
    expected_return_float, expected_pnl_float, expected_close_float = _expected_tuple(snapshot_obj,
        {"U100", "U200", "U900"}, last_by_route_dict=closing_date_dict)
    assert august_dict["available_bool"] and not august_dict["draft_bool"]
    assert august_dict["period_label_str"] == "August 2026"
    assert august_dict["money_row_list"][0]["label_str"] == "Value on August 31"
    assert august_dict["money_row_list"][0]["value_str"] == _money_str(expected_close_float) == "$50,100"
    assert august_dict["money_row_list"][1]["value_str"] == _money_str(expected_pnl_float, signed_bool=True)
    assert august_dict["money_row_list"][1]["pct_str"] == _percent_str(expected_return_float)
    september_dict = _report_dict(workspace_dict, snapshot_obj, month_str="2026-09", program_dict=program_map_dict)
    assert september_dict is not None and september_dict["available_bool"] and september_dict["draft_bool"]
    assert september_dict["period_label_str"] == "September 2026 · to Sep 3"
    assert september_dict["money_row_list"][0]["value_str"] == "$50,100"
    assert september_dict["money_row_list"][1]["value_str"] == "$0"
    assert september_dict["money_row_list"][1]["pct_str"] == "0.0%"
    august_program_dict = {card_dict["name_str"]: card_dict for card_dict in august_dict["program_list"]}
    for card_dict in september_dict["program_list"]:
        assert card_dict["value_str"] == "$0" and card_dict["closed_bool"]
        assert card_dict["since_str"].endswith("closed 2026-08-13")
        assert card_dict["stat_list"][-1] == august_program_dict[card_dict["name_str"]]["stat_list"][-1]
        assert card_dict["chart_dict"]["end_label_str"] == august_program_dict[card_dict["name_str"]]["chart_dict"]["end_label_str"]
        assert card_dict["stat_list"][0]["value_str"] == "—"
    assert [group_dict["name_str"] for group_dict in september_dict["split_dict"]["group_list"]] == ["Vault"]
    assert september_dict["flow_dict"]["row_list"] == []


def test_newer_owned_source_does_not_hide_missing_required_month_end_account():
    workspace_dict, snapshot_obj = _sample_tuple(last_row_dict={"U200": "2026-08-28"})
    report_dict = _report_dict(workspace_dict, snapshot_obj)
    assert report_dict["available_bool"] and report_dict["draft_bool"]
    assert report_dict["period_label_str"] == "August 2026"
    assert report_dict["money_row_list"][0]["label_str"] == "Value on August 31"
    assert report_dict["money_row_list"][0]["value_str"] == "—"
    assert report_dict["money_row_list"][1]["value_str"] == "—"
    assert "Month data incomplete" in report_dict["draft_reason_str"]
    assert any(option_dict["label_str"] == "September 2026 · to date" for option_dict in report_dict["month_option_list"])


def test_reserve_only_client_has_monthly_summary_without_synthetic_program():
    workspace_dict, snapshot_obj = _sample_tuple()
    workspace_dict["client_dict"]["accounts"] = [account_dict for account_dict in workspace_dict["client_dict"]["accounts"]
        if account_dict.get("role") == "reserve"]
    workspace_dict["valuation_account_list"] = [account_dict for account_dict in workspace_dict["valuation_account_list"]
        if account_dict.get("role") == "reserve"]
    workspace_dict["operations_account_list"] = []
    report_dict = _report_dict(workspace_dict, snapshot_obj)
    assert report_dict["available_bool"] and not report_dict["draft_bool"]
    assert report_dict["money_row_list"][0]["value_str"] == "$50,100"
    assert report_dict["money_row_list"][1]["value_str"] == "+$100"
    assert report_dict["money_row_list"][1]["pct_str"] == _percent_str(float(Decimal(100) / Decimal(50000)))
    assert report_dict["program_list"] == [] and report_dict["flow_dict"]["reason_str"] == ""
    assert report_dict["other_account_list"][0]["value_str"] == "$50,100"
    assert report_dict["split_dict"]["group_list"][0]["pct_str"] == "100.0%"


def test_vault_before_first_pod_keeps_full_client_start_and_actual_program_start():
    workspace_dict, snapshot_obj = _sample_tuple()
    for account_dict in workspace_dict["client_dict"]["accounts"]:
        if account_dict.get("role", "strategy") == "strategy":
            account_dict["effective_from"] = "2026-08-17"
    report_dict = _report_dict(workspace_dict, snapshot_obj)
    expected_return_float, expected_pnl_float, _ = _expected_tuple(snapshot_obj, {"U100", "U200", "U900"},
        first_by_route_dict={"U100": "2026-08-17", "U200": "2026-08-17"})
    assert report_dict["available_bool"] and not report_dict["draft_bool"]
    assert report_dict["money_row_list"][1]["value_str"] == _money_str(expected_pnl_float, signed_bool=True)
    assert report_dict["money_row_list"][1]["pct_str"] == _percent_str(expected_return_float)
    assert report_dict["program_list"][0]["since_str"] == "from 2026-08-17"
    expected_program_return_float, _, _ = _expected_tuple(snapshot_obj, {"U100", "U200"}, first_str="2026-08-17")
    assert report_dict["program_list"][0]["stat_list"][-1]["value_str"] == _percent_str(expected_program_return_float)
