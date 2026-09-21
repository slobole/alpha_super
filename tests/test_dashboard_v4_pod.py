"""Selected cycle joins and route boundaries must preserve saved evidence."""

from copy import deepcopy
from datetime import timedelta
from html import unescape
import re

import pytest

from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple, create_demo_app
from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.pod import build_evidence_tables_dict, build_pod_page_dict
from alpha.live.dashboard_v4.overview import build_overview_dict


@pytest.fixture
def pod_fixture_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    pod_id_str = workspace_dict["operations_account_list"][0]["pod_id"]
    source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)
    return workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str


def test_request_join_preserves_same_symbol_entry_and_exit(pod_fixture_tuple):
    source_dict = pod_fixture_tuple[3]
    for list_key_str in ("plan_row_list", "order_list", "ack_list", "fill_list"):
        for row_dict in source_dict[list_key_str]:
            row_dict["asset_str"] = "SPY"
    source_dict["vplan_dict"]["current_broker_position_map_dict"] = {"SPY": 100}
    source_dict["reconciliation_dict"].update(model_position_map_dict={"SPY": 104}, broker_position_map_dict={"SPY": 104})
    tables_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)
    table_dict = tables_dict["plan"]
    assert len(table_dict["row_list"]) == 3
    assert [row_dict["order_str"] for row_dict in table_dict["row_list"]] == ["+31", "+17", "-44"]
    assert [row_dict["cell_list"][6] for row_dict in tables_dict["orders"]["row_list"]] == ["158.42", "291.05", "112.80"]
    assert {row_dict["before_str"] for row_dict in table_dict["row_list"]} == {"100"}
    assert {row_dict["after_str"] for row_dict in table_dict["row_list"]} == {"104"}


def test_partial_execution_prices_are_weighted_within_order_only(pod_fixture_tuple):
    source_dict = pod_fixture_tuple[3]
    first_dict = source_dict["fill_list"][0]
    first_dict.update(fill_amount_float=10, fill_price_float=100)
    source_dict["fill_list"].append({**first_dict, "fill_amount_float": 21, "fill_price_float": 200})
    table_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)["orders"]
    assert table_dict["row_list"][0]["cell_list"][6] == "167.74"
    assert "Slip bps" not in table_dict["column_list"]  # No verified comparison source.


@pytest.mark.parametrize("missing_str", ["reconcile", "quantities", "before", "duplicate_order"])
def test_missing_or_ambiguous_evidence_never_manufactures_values(pod_fixture_tuple, missing_str):
    source_dict = pod_fixture_tuple[3]
    if missing_str == "reconcile":
        source_dict["reconciliation_dict"] = {}
    elif missing_str == "quantities":
        source_dict["cycle_evidence_dict"] = {}
    elif missing_str == "before":
        source_dict["vplan_dict"] = {}
    else:
        source_dict["order_list"].append(deepcopy(source_dict["order_list"][0]))
    tables_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)
    table_dict = tables_dict["plan"]
    if missing_str == "reconcile":
        assert {row_dict["match_str"] for row_dict in table_dict["row_list"]} == {"unk"}
        assert {row_dict["after_str"] for row_dict in table_dict["row_list"]} == {"—"}
    elif missing_str == "before":
        assert table_dict["row_list"][0]["before_str"] == "—"
    else:
        assert tables_dict["orders"]["row_list"][0]["cell_list"][5:7] == ["—", "—"]


def test_compact_position_uses_saved_broker_result_not_order_target_or_fill_total():
    source_dict = {
        "plan_row_list": [{"asset_str": "ABC", "order_delta_share_float": 19}],
        "vplan_dict": {"current_broker_position_map_dict": {"ABC": 7}, "target_share_map_dict": {"ABC": 26}},
        "reconciliation_dict": {"broker_position_map_dict": {"ABC": 18}, "model_position_map_dict": {"ABC": 26}},
        "fill_list": [{"asset_str": "ABC", "fill_amount_float": 2, "fill_price_float": 100}],
    }
    before_dict = deepcopy(source_dict)
    table_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)["plan"]
    assert table_dict["column_list"] == ["Symbol", "Order", "Position", "Broker = model"]
    assert table_dict["row_list"] == [{"symbol_str": "ABC", "order_str": "+19",
        "before_str": "7", "after_str": "18", "match_str": "fail"}]
    assert source_dict == before_dict


@pytest.mark.parametrize("before_map,broker_map,model_map,before_str,after_str,match_str", [
    (None, {"ABC": 18}, {"ABC": 18}, "—", "18", "done"),
    ({"ABC": 7}, None, {"ABC": 26}, "7", "—", "unk"),
    ({"ABC": 7}, {"ABC": 18}, None, "7", "18", "unk"),
    ({}, {}, {}, "0", "0", "done"),
])
def test_compact_positions_preserve_independent_snapshot_availability(before_map, broker_map, model_map,
        before_str, after_str, match_str):
    source_dict = {"plan_row_list": [{"asset_str": "ABC", "order_delta_share_float": 19}],
        "vplan_dict": {"current_broker_position_map_dict": before_map, "target_share_map_dict": {"ABC": 26}},
        "reconciliation_dict": {"broker_position_map_dict": broker_map, "model_position_map_dict": model_map}}
    row_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)["plan"]["row_list"][0]
    assert (row_dict["before_str"], row_dict["after_str"], row_dict["match_str"]) == (before_str, after_str, match_str)


@pytest.mark.parametrize("amount_obj,order_str", [(0, "0"), (-5, "-5"),
    (0.125, "+0.125"), (None, "—"), (float("nan"), "—")])
def test_compact_requested_shares_do_not_invent_an_order(amount_obj, order_str):
    tables_dict = build_evidence_tables_dict({"plan_row_list": [{"asset_str": "ABC", "order_delta_share_float": amount_obj}]},
        as_of_ts=DEMO_NOW_TS, fresh_bool=True)
    assert tables_dict["plan"]["row_list"][0]["order_str"] == order_str
    if amount_obj == 0:
        assert tables_dict["orders"]["row_list"] == []


def test_compact_table_escapes_symbols_and_keeps_mobile_and_selection_fields(pod_fixture_tuple, monkeypatch):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    source_dict["plan_row_list"][0]["asset_str"] = '<img src=x onerror="alert(1)">'
    monkeypatch.setattr(provider_obj, "get_pod_cycles_dict", lambda *args, **kwargs: deepcopy(source_dict))
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    html_str = app_obj.test_client().get(f"/pods/{pod_id_str}?tab=plan").get_data(as_text=True)
    table_html_str = html_str.split('data-selection-key="evidence:plan"')[1].split('</table>')[0]
    assert '<img src=x' not in table_html_str and '&lt;img src=x' in table_html_str
    assert table_html_str.count('scope="col"') == 4
    assert all(f'data-column="{label_str}"' in table_html_str for label_str in ("Symbol", "Order", "Position", "Broker = model"))
    assert '<span class="sr-only">Before </span>' in table_html_str
    assert '<span class="sr-only"> After </span>' in table_html_str
    assert 'data-observed-state' in table_html_str
    assert 'Fill px' not in table_html_str and 'Filled' not in table_html_str


def test_order_has_its_own_cell_and_cannot_rederive_saved_position(pod_fixture_tuple, monkeypatch):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    source_dict["plan_row_list"] = [{"asset_str": "ABC", "order_delta_share_float": 19}]
    source_dict["vplan_dict"].update(current_broker_position_map_dict={"ABC": 7}, target_share_map_dict={"ABC": 999})
    source_dict["reconciliation_dict"].update(broker_position_map_dict={"ABC": 18}, model_position_map_dict={"ABC": 999})
    source_dict["fill_list"] = [{"asset_str": "ABC", "fill_amount_float": 2, "fill_price_float": 100}]
    original_dict = deepcopy(source_dict)
    monkeypatch.setattr(provider_obj, "get_pod_cycles_dict", lambda *args, **kwargs: deepcopy(source_dict))
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    html_str = app_obj.test_client().get(f"/pods/{pod_id_str}?tab=plan").get_data(as_text=True)
    table_str = html_str.split('data-selection-key="evidence:plan"')[1].split('</table>')[0]
    order_str = re.search(r'<td\b[^>]*data-column="Order"[^>]*>(.*?)</td>', table_str, re.S)[1]
    position_str = re.search(r'<td\b[^>]*data-column="Position"[^>]*>(.*?)</td>', table_str, re.S)[1]
    assert unescape(re.sub(r"<[^>]+>", "", order_str)).strip() == "+19"
    visible_position_str = " ".join(unescape(re.sub(r"<[^>]+>", " ", position_str)).split())
    assert visible_position_str == "Before 7 → After 18"
    assert "+19" not in position_str and "Order" not in position_str
    assert source_dict == original_dict


@pytest.mark.parametrize("book_str,note_str", [
    ("incremental_entry_exit_book", "Entry and exit targets"), ("unknown_book", "Saved decision targets"),
])
def test_decision_notes_preserve_incremental_meaning_and_target_precision(book_str, note_str):
    source_dict = {"decision_dict": {"decision_book_type_str": book_str,
        "target_weight_map_dict": {"HELD": .75}, "display_target_weight_map_dict": {"NEW": .12345678},
        "exit_asset_list": ["OLD"]}}
    original_dict = deepcopy(source_dict)
    table_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)["decision"]
    assert table_dict["column_list"] == ["Symbol", "Target"]
    assert [row_dict["cell_list"] for row_dict in table_dict["row_list"]] == [["NEW", "12.345678%"], ["OLD", "Exit"]]
    assert table_dict["note_str"] == note_str
    assert source_dict == original_dict


@pytest.mark.parametrize("book_str", ["full_target_weight_book", "incremental_entry_exit_book"])
def test_decision_hides_only_full_portfolio_note_and_cycle_legend(pod_fixture_tuple, monkeypatch, book_str):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    source_dict["decision_dict"].update(decision_book_type_str=book_str,
        display_target_weight_map_dict={"NEW": .12345678}, exit_asset_list=["OLD"])
    monkeypatch.setattr(provider_obj, "get_pod_cycles_dict", lambda *args, **kwargs: deepcopy(source_dict))
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    html_str = app_obj.test_client().get(f"/pods/{pod_id_str}?tab=decision").get_data(as_text=True)
    assert "Black = actual" not in html_str and "gray = plan" not in html_str
    assert "Full portfolio targets" not in html_str
    assert ("Entry and exit targets" in html_str) is (book_str == "incremental_entry_exit_book")
    table_tag_str = re.search(r'<table\b[^>]*data-selection-key="evidence:decision"[^>]*>', html_str)[0]
    assert "decision-table" in table_tag_str
    table_str = html_str.split('data-selection-key="evidence:decision"')[1].split('</table>')[0]
    assert table_str.count('scope="col"') == 2
    assert "12.345678%" in table_str and "Exit" in table_str


@pytest.mark.parametrize("position_time_available_bool", [True, False])
def test_pod_stamp_markup_keeps_positions_time_separate_from_cash_close(pod_fixture_tuple, position_time_available_bool):
    workspace_dict, snapshot_obj, provider_obj, _, pod_id_str = pod_fixture_tuple
    if not position_time_available_bool:
        workspace_dict["summary_dict"]["pod_row_dict_list"][0]["latest_pod_state_timestamp_str"] = None
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    html_str = app_obj.test_client().get(f"/pods/{pod_id_str}?tab=plan").get_data(as_text=True)
    position_header_str = re.search(r'<div\b[^>]*class="panel-h pod-positions-head"[^>]*>(.*?)</div>', html_str, re.S)[1]
    cash_header_str = re.search(r'<div\b[^>]*class="pod-cash"[^>]*>(.*?)</div>', html_str, re.S)[1]
    position_visible_str = re.sub(r'<span class="sr-only">.*?</span>', "", position_header_str, flags=re.S)
    cash_visible_str = re.sub(r'<span class="sr-only">.*?</span>', "", cash_header_str, flags=re.S)
    position_text_str = " ".join(unescape(re.sub(r"<[^>]+>", " ", position_visible_str)).split())
    cash_text_str = " ".join(unescape(re.sub(r"<[^>]+>", " ", cash_visible_str)).split())
    assert 'class="pod-stamp"' in position_header_str and 'class="pod-stamp"' in cash_header_str
    assert '<svg class="ic"' in position_header_str and '<svg class="ic"' in cash_header_str
    assert "2026-09-04 close" in cash_text_str and "09:36:12" not in cash_text_str
    assert 'title="Demo · Cash · broker end-of-day · 2026-09-04"' in cash_header_str
    assert '<span class="sr-only">Demo · Cash · broker end-of-day · 2026-09-04</span>' in cash_header_str
    assert '<span aria-hidden="true">2026-09-04 close</span>' in cash_header_str
    if position_time_available_bool:
        assert "2026-09-08 09:36:12 ET" in position_text_str
        assert 'title="Saved positions · 2026-09-08 09:36:12 ET"' in position_header_str
        assert '<span class="sr-only">Saved positions · 2026-09-08 09:36:12 ET</span>' in position_header_str
        assert '<span aria-hidden="true">2026-09-08 09:36:12 ET</span>' in position_header_str
    else:
        assert "—" in position_text_str and "2026-09-04" not in position_text_str
        assert 'title="Saved positions time unavailable"' in position_header_str
        assert '<span class="sr-only">Saved positions time unavailable</span>' in position_header_str
        assert '<span aria-hidden="true">—</span>' in position_header_str
    assert "Saved positions" not in position_text_str
    assert "broker end-of-day" not in cash_text_str


@pytest.mark.parametrize("fresh_bool,reason_str", [(True, "Order ownership could not be checked."), (True, ""), (False, "")])
def test_unverified_fill_records_remain_visible_without_verified_order_totals(pod_fixture_tuple, fresh_bool, reason_str):
    source_dict = pod_fixture_tuple[3]
    source_dict["cycle_evidence_dict"] = {"state_str": "unknown", "reason_str": reason_str, "order_list": []}
    tables_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=fresh_bool)
    assert len(tables_dict["fills"]["row_list"]) == len(source_dict["fill_list"])
    assert all(row_dict["cell_list"][5:7] == ["—", "—"] for row_dict in tables_dict["orders"]["row_list"])
    assert tables_dict["fills"]["note_str"] == (
        "Not verified. " + (reason_str or "Fill details could not be checked.") if fresh_bool else
        "Saved fill records; current verification unavailable.")


def test_unverified_fill_reason_is_escaped_with_saved_fill_rows(pod_fixture_tuple, monkeypatch):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    source_dict["cycle_evidence_dict"] = {"state_str": "unknown", "reason_str": '<script>alert("reason")</script>', "order_list": []}
    monkeypatch.setattr(provider_obj, "get_pod_cycles_dict", lambda *args, **kwargs: deepcopy(source_dict))
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    html_str = app_obj.test_client().get(f"/pods/{pod_id_str}?tab=fills").get_data(as_text=True)
    assert 'Not verified. &lt;script&gt;' in html_str and '<script>alert("reason")</script>' not in html_str
    assert 'data-selection-key="evidence:fills"' in html_str
    assert '158.42' in html_str


def test_stale_cycle_never_keeps_green_states(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS + timedelta(seconds=121))
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS + timedelta(seconds=121))
    assert view_dict["pill_str"] == "Unknown"
    assert {step_dict["state_str"] for step_dict in view_dict["step_list"]} == {"Unknown"}
    assert {row_dict["match_str"] for row_dict in view_dict["tables_dict"]["plan"]["row_list"]} == {"unk"}
    assert {row_dict["cell_list"][2] for row_dict in view_dict["tables_dict"]["orders"]["row_list"]} == {"Unknown"}


@pytest.mark.parametrize("tab_str", ["plan", "decision", "orders", "fills", "reconcile", "events", "files"])
def test_pod_tabs_and_refresh_are_real_read_only_routes(pod_fixture_tuple, tab_str):
    pod_id_str = pod_fixture_tuple[4]
    app_obj = create_demo_app()
    response_obj = app_obj.test_client().get(f"/pods/{pod_id_str}?tab={tab_str}")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert 'aria-label="Seven cycle steps"' in html_str and html_str.count('class="step ') == 7
    assert html_str.count('hx-get="') == 1
    assert 'href="/pods/' in html_str
    assert pod_fixture_tuple[3]["pod_row_dict"]["account_route_str"] not in html_str
    assert '09:30:02' in html_str
    assert app_obj.test_client().get(f"/pods/{pod_id_str}/refresh?tab={tab_str}").status_code == 200
    assert app_obj.test_client().post(f"/pods/{pod_id_str}?tab={tab_str}").status_code == 403


def test_history_navigation_changes_evidence_and_survives_poll(pod_fixture_tuple):
    pod_id_str = pod_fixture_tuple[4]
    app_obj = create_demo_app()
    html_str = app_obj.test_client().get(f"/pods/{pod_id_str}?cycle=vplan:1&tab=fills").get_data(as_text=True)
    assert "2026-09-04 · Open" in html_str
    assert "09-04 09:30:02" in html_str
    assert "cycle=vplan:1" in html_str and "tab=fills" in html_str
    assert 'aria-label="Next cycle"' in html_str
    assert app_obj.test_client().get(f"/pods/{pod_id_str}?cycle=vplan:999").status_code == 404


def test_issue_defaults_to_orders_and_preserves_ack_warning(pod_fixture_tuple):
    pod_id_str = pod_fixture_tuple[0]["operations_account_list"][1]["pod_id"]
    html_str = create_demo_app().test_client().get(f"/pods/{pod_id_str}").get_data(as_text=True)
    assert 'aria-label="Pod needs attention"' in html_str
    assert "No ack" in html_str and "Do not resubmit blindly." in html_str
    assert "Next: Review broker ACK" in html_str and "Next: Submit" not in html_str
    assert "Tools for this step" not in html_str
    assert 'class="step is-fail is-sel"' in html_str


@pytest.mark.parametrize("query_str", ["mode=paper", "tab=raw", "cycle=1", "cycle=vplan:0", "cycle=vplan:-1", "cycle=vplan:2&cycle=vplan:1", "tab=plan&tab=fills", "period=3M&period=All", "path=C:/secret"])
def test_invalid_pod_queries_rejected_before_acquisition(pod_fixture_tuple, query_str):
    response_obj = create_demo_app().test_client().get(f"/pods/{pod_fixture_tuple[4]}?{query_str}")
    assert response_obj.status_code == 400


def test_unknown_pod_is_not_a_path_or_non_live_fallback():
    app_obj = create_demo_app()
    assert app_obj.test_client().get("/pods/foreign_account").status_code == 404
    assert app_obj.test_client().get("/pods/..%2fconfig.env").status_code == 404


def test_overview_pod_links_work(pod_fixture_tuple):
    html_str = create_demo_app().test_client().get("/").get_data(as_text=True)
    assert f'/pods/{pod_fixture_tuple[4]}' in html_str
    assert 'Pod page is not available yet' not in html_str


@pytest.mark.parametrize("field_str,value_obj,expected_str", [("ack_status_str", "missing_critical", "No ack"), ("asset_str", "OTHER", "Unknown"), ("broker_order_id_str", "other-order", "Unknown")])
def test_conflicting_ack_is_not_acked(pod_fixture_tuple, field_str, value_obj, expected_str):
    source_dict = pod_fixture_tuple[3]
    source_dict["ack_list"][0][field_str] = value_obj
    table_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)["orders"]
    assert table_dict["row_list"][0]["cell_list"][2] == expected_str


def test_old_selected_assessment_clears_table_status_too(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    source_dict["pod_row_dict"]["as_of_timestamp_str"] = (DEMO_NOW_TS - timedelta(seconds=121)).isoformat()
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert not view_dict["fresh_bool"]
    assert view_dict["tables_dict"]["orders"]["row_list"][0]["cell_list"][2] == "Unknown"
    assert view_dict["tables_dict"]["plan"]["row_list"][0]["match_str"] == "unk"


@pytest.mark.parametrize("query_str", ["", "?cycle=vplan:2&tab=orders", "?cycle=vplan:2&period=All"])
def test_current_cycle_failure_survives_tab_and_period_navigation(pod_fixture_tuple, query_str):
    workspace_dict, snapshot_obj, provider_obj, _, pod_id_str = pod_fixture_tuple
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["reconcile_read_failure_dict"] = {"timestamp_str": DEMO_NOW_TS.isoformat(), "error_str": "Read failed"}
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    html_str = app_obj.test_client().get(f"/pods/{pod_id_str}{query_str}").get_data(as_text=True)
    assert "Broker read failed" in html_str


def test_old_cycle_does_not_borrow_current_failure(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, _, pod_id_str = pod_fixture_tuple
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["reconcile_read_failure_dict"] = {"timestamp_str": DEMO_NOW_TS.isoformat(), "error_str": "Read failed"}
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    html_str = app_obj.test_client().get(f"/pods/{pod_id_str}?cycle=vplan:1").get_data(as_text=True)
    assert "Broker read failed" in html_str  # The current Pod warning stays visible.
    cycle_html_str = html_str.split('aria-label="Selected cycle"')[1].split('</section>')[0]
    assert "Broker read failed" not in cycle_html_str and "Saved cycle" in cycle_html_str


def test_slow_detail_read_cannot_extend_fresh_source_lifetime(pod_fixture_tuple, monkeypatch):
    workspace_dict, snapshot_obj, provider_obj, _, pod_id_str = pod_fixture_tuple
    clock_list = [DEMO_NOW_TS]
    original_fn = provider_obj.get_pod_cycles_dict
    def delayed_source(pod_id_str, **options_dict):
        source_dict = original_fn(pod_id_str, **options_dict)
        clock_list[0] += timedelta(seconds=121)
        return source_dict
    monkeypatch.setattr(provider_obj, "get_pod_cycles_dict", delayed_source)
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: clock_list[0])
    response_obj = app_obj.test_client().get(f"/pods/{pod_id_str}")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert 'data-source-valid-ms="0"' in html_str and "Status out of date" in html_str
    assert "Acked" not in html_str


def test_malformed_unrelated_summary_row_does_not_break_pod(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, _, pod_id_str = pod_fixture_tuple
    workspace_dict["summary_dict"]["pod_row_dict_list"].append(None)
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    response_obj = app_obj.test_client().get(f"/pods/{pod_id_str}")
    assert response_obj.status_code == 200
    assert "Financial data unavailable" in response_obj.get_data(as_text=True)


@pytest.mark.parametrize("conflict_str", ["asset", "order_id", "request", "response", "timestamp", "duplicate_ack", "duplicate_order", "missing_ack", "missing_order", "extra_ack", "order_asset", "shared_order_id", "order_type"])
def test_inconsistent_detailed_ack_cannot_leave_cycle_green(pod_fixture_tuple, conflict_str):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    ack_dict, order_dict = source_dict["ack_list"][0], source_dict["order_list"][0]
    if conflict_str == "asset":
        ack_dict["asset_str"] = "OTHER"
    elif conflict_str == "order_id":
        ack_dict["broker_order_id_str"] = "OTHER"
    elif conflict_str == "request":
        ack_dict["order_request_key_str"] = "OTHER"
    elif conflict_str == "response":
        ack_dict["broker_response_ack_bool"] = False
    elif conflict_str == "timestamp":
        ack_dict["response_timestamp_str"] = (DEMO_NOW_TS + timedelta(seconds=1)).isoformat()
    elif conflict_str == "duplicate_ack":
        source_dict["ack_list"].append(deepcopy(ack_dict))
    elif conflict_str == "duplicate_order":
        source_dict["order_list"].append(deepcopy(order_dict))
    elif conflict_str == "missing_ack":
        source_dict["ack_list"] = source_dict["ack_list"][1:]
    elif conflict_str == "missing_order":
        source_dict["order_list"] = source_dict["order_list"][1:]
    elif conflict_str == "extra_ack":
        source_dict["ack_list"].append({**ack_dict, "order_request_key_str": "OTHER"})
    elif conflict_str == "order_asset":
        order_dict["asset_str"] = "OTHER"
    elif conflict_str == "shared_order_id":
        source_dict["order_list"][1]["broker_order_id_str"] = order_dict["broker_order_id_str"]
    else:
        source_dict["plan_row_list"][0]["broker_order_type_str"] = "MOO"
        order_dict["broker_order_type_str"] = ack_dict["broker_order_type_str"] = "MOC"
    before_dict = deepcopy(source_dict)
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["step_list"][3]["state_str"] == "Unknown"
    assert view_dict["pill_str"] == "Unknown"
    assert source_dict == before_dict


def test_detailed_critical_ack_overrides_complete_summary_and_no_order_claim(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    source_dict["ack_list"][0]["ack_status_str"] = "missing_critical"
    source_dict["cycle_evidence_dict"]["state_str"] = "no_orders"
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["step_list"][3]["state_str"] == "Failed"
    assert view_dict["pill_str"] == "Action needed"
    assert view_dict["issue_title_str"] == "Review broker ACK"
    assert view_dict["tables_dict"]["orders"]["row_list"][0]["cell_list"][2] == "No ack"


def test_upcoming_unsubmitted_plan_keeps_waiting_without_ack_alarm(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    row_dict = source_dict["pod_row_dict"]
    row_dict.update(latest_vplan_status_str="ready", latest_submit_ack_status_str="not_checked",
        broker_order_count_int=0, broker_ack_count_int=0, missing_ack_count_int=0,
        latest_vplan_submission_timestamp_str=(DEMO_NOW_TS + timedelta(minutes=5)).isoformat(),
        latest_vplan_target_execution_timestamp_str=(DEMO_NOW_TS + timedelta(minutes=10)).isoformat())
    source_dict["order_list"], source_dict["ack_list"], source_dict["cycle_evidence_dict"] = [], [], {}
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["step_list"][3]["state_str"] == "Planned"
    assert view_dict["step_list"][3]["fact_str"] == "Waiting to submit"
    assert not view_dict["issue_bool"]


@pytest.mark.parametrize("explicit_bool", [True, False])
def test_historical_failed_cycle_wording_preserves_default_unresolved_warning(pod_fixture_tuple, explicit_bool):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    source_dict["selected_cycle_dict"].update(current_bool=False, unresolved_bool=True, session_date_str="2026-09-04")
    source_dict["selected_explicit_bool"] = explicit_bool
    source_dict["ack_list"][0]["ack_status_str"] = "missing_critical"
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["issue_bool"]
    if explicit_bool:
        assert view_dict["verdict_str"].startswith("Saved cycle:")
        assert "Next:" not in view_dict["verdict_detail_str"]
        assert view_dict["issue_title_str"] == "Saved cycle · Missing broker ACK"
        assert view_dict["issue_detail_str"] == "Review the saved records for 2026-09-04."
    else:
        assert not view_dict["verdict_str"].startswith("Saved cycle:")
        assert view_dict["issue_title_str"] == "Review broker ACK"
        assert "Do not resubmit blindly." in view_dict["issue_detail_str"]
