"""Backtest check job (alpha/live/backtest_check.py): pure maths and the read-only pod database read."""

from datetime import UTC, date, datetime
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from alpha.live import backtest_check as bc


def _fill(asset_str, amount_float, price_float, open_float, session_str="2026-11-02"):
    return {"asset_str": asset_str, "fill_amount_float": amount_float, "fill_price_float": price_float,
            "official_open_price_float": open_float, "session_date_str": session_str}


def test_fill_cost_sign_matches_runner_slippage():
    assert bc.fill_cost_bps(_fill("AAA", 10, 100.04, 100.0)) == pytest.approx(4.0)   # bought above the open: cost
    assert bc.fill_cost_bps(_fill("AAA", -10, 99.98, 100.0)) == pytest.approx(2.0)   # sold below the open: cost
    assert bc.fill_cost_bps(_fill("AAA", -10, 100.02, 100.0)) == pytest.approx(-2.0)  # sold above: gain
    assert bc.fill_cost_bps(_fill("AAA", 10, 100.0, None)) is None
    assert bc.fill_cost_bps(_fill("AAA", 0, 100.0, 100.0)) is None


def test_cost_gate_counts_stocks_only_and_reports_missing_opens():
    fills = {"pod_a": [_fill("AAA", 10, 100.02, 100.0), _fill("BIL", 5, 91.5, 91.5), _fill("BBB", -3, 50.0, None)],
             "pod_b": [_fill("CCC", -4, 49.99, 50.0)]}
    gate = bc.cost_gate_dict(fills, as_of_date_str="2026-11-10")
    assert gate["fill_count_int"] == 2
    assert gate["parking_fill_count_int"] == 1
    assert gate["missing_open_count_int"] == 1
    assert gate["average_bps_float"] == pytest.approx(2.0)
    assert gate["buy_bps_float"] == pytest.approx(2.0) and gate["sell_bps_float"] == pytest.approx(2.0)
    assert gate["status_str"] == "collecting"
    assert gate["time_cap_date_str"] == "2027-05-04"


@pytest.mark.parametrize("cost_float,count_int,status_str", [
    (2.0, 40, "on_track"), (5.0, 40, "above"), (3.9, 200, "passed"), (4.1, 200, "failed")])
def test_cost_gate_status(cost_float, count_int, status_str):
    fills = {"pod_a": [_fill("AAA", 10, 100.0 * (1 + cost_float / 10000), 100.0) for _ in range(count_int)]}
    assert bc.cost_gate_dict(fills, as_of_date_str="2026-12-01")["status_str"] == status_str


def test_cost_gate_time_cap():
    fills = {"pod_a": [_fill("AAA", 10, 100.0, 100.0, "2026-01-02")] * 40}
    assert bc.cost_gate_dict(fills, as_of_date_str="2026-08-01")["status_str"] == "time_cap"


def test_full_book_compares_weights_with_tolerance():
    decision = {"decision_book_type_str": "full_target_weight_book",
                "full_target_weight_map_dict": {"QQQ": 0.60, "TLT": 0.40}}
    same = bc.compare_decision_dict(decision, reference_weight_map_dict={"QQQ": 0.565, "TLT": 0.405, "Cash": 0.028}, reference_trade_map_dict={})
    assert same["status_str"] == "same" and same["differ_count_int"] == 0
    differs = bc.compare_decision_dict(decision, reference_weight_map_dict={"QQQ": 0.60, "GLD": 0.40}, reference_trade_map_dict={})
    assert differs["status_str"] == "differs"
    assert [row["asset_str"] for row in differs["row_list"] if not row["same_bool"]] == ["GLD", "TLT"]


def test_entry_book_compares_entries_and_exits_and_skips_parking():
    decision = {"decision_book_type_str": "incremental_entry_exit_book",
                "entry_target_weight_map_dict": {"XLE": 0.10, "KRE": 0.10}, "exit_asset_list": ["XLK"]}
    result = bc.compare_decision_dict(decision, reference_weight_map_dict={},
                                      reference_trade_map_dict={"XLU": 12.0, "KRE": 9.0, "XLK": -6.0, "BIL": 40.0},
                                      reference_entry_weight_map_dict={"XLU": 0.10, "KRE": 0.10},
                                      skip_set=frozenset({"BIL"}))
    assert result["differ_count_int"] == 2
    by_asset = {row["asset_str"]: row for row in result["row_list"]}
    assert not by_asset["XLE"]["same_bool"] and not by_asset["XLU"]["same_bool"]
    assert by_asset["KRE"]["same_bool"] and by_asset["XLK"]["same_bool"]
    assert "BIL" not in by_asset


def test_curve_gap_is_relative_and_month_to_date():
    live = {"2026-11-27": 0.01, "2026-11-30": 0.0, "2026-12-01": -0.02}
    test = {"2026-11-27": 0.01, "2026-11-30": 0.0, "2026-12-01": -0.01}
    curve = bc.curve_dict(live, test)
    assert curve["start_str"] == "2026-11-27"
    assert curve["gap_float"] == pytest.approx(0.98 / 0.99 - 1)
    assert curve["gap_month_float"] == pytest.approx(0.98 / 0.99 - 1)
    assert bc.curve_dict({}, test)["gap_float"] is None


def test_rare_loss_flags_only_below_the_one_in_100_line_and_names_the_cause():
    history = [0.001] * 600 + [-0.03] * 12
    ruler = bc.build_ruler_dict(history)
    assert ruler["1"]["worst_float"] == pytest.approx(-0.03)
    assert bc.rare_loss_dict({"2026-12-10": -0.5}, {}, {"1": {"one_in_100_float": 0.001, "worst_float": -0.1}}) is None
    dates = [f"2026-12-{day:02d}" for day in (7, 8, 9, 10)]
    calm = dict(zip(dates, [0.001, 0.0, -0.002, 0.001]))
    assert bc.rare_loss_dict(calm, calm, ruler) is None
    bad = dict(zip(dates, [0.0, -0.035, -0.035, -0.035]))
    strategy = bc.rare_loss_dict(bad, dict(zip(dates, [0.0, -0.034, -0.035, -0.035])), ruler)
    assert strategy is not None and strategy["cause_str"] == "strategy"
    ours = bc.rare_loss_dict(bad, dict(zip(dates, [0.0, 0.0, 0.0, -0.001])), ruler)
    assert ours["cause_str"] == "ours"


def test_record_merge_keeps_one_mark_per_session():
    record = bc.merge_record_list([{"session_str": "2026-12-09", "status_str": "same"}], "2026-12-10", "differs")
    assert record == [{"session_str": "2026-12-09", "status_str": "same"}, {"session_str": "2026-12-10", "status_str": "differs"}]
    record = bc.merge_record_list(record, "2026-12-10", "same")
    assert record[-1]["status_str"] == "same" and len(record) == 2
    assert bc.merge_record_list(record, None, "not_checked") == record


def test_month_end_window():
    assert bc.month_end_window_bool(date(2026, 10, 30)) and bc.month_end_window_bool(date(2026, 11, 2))
    assert not bc.month_end_window_bool(date(2026, 11, 10))
    assert bc.month_end_window_bool(date(2026, 11, 28))  # Saturday: not a session, stay safe


def _pod_db(tmp_path):
    db_path = tmp_path / "pod.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.executescript("""
            CREATE TABLE decision_plan (decision_plan_id_int INTEGER PRIMARY KEY, decision_book_type_str TEXT,
                entry_target_weight_json_str TEXT, full_target_weight_json_str TEXT, exit_asset_json_str TEXT,
                snapshot_metadata_json_str TEXT);
            CREATE TABLE vplan (vplan_id_int INTEGER PRIMARY KEY, pod_id_str TEXT, decision_plan_id_int INTEGER,
                target_execution_timestamp_str TEXT, pod_budget_float REAL, status_str TEXT);
            CREATE TABLE vplan_fill (fill_record_id_int INTEGER PRIMARY KEY, vplan_id_int INTEGER, asset_str TEXT,
                fill_amount_float REAL, fill_price_float REAL, official_open_price_float REAL, fill_timestamp_str TEXT);
            INSERT INTO decision_plan VALUES (1, 'full_target_weight_book', '{}', '{"QQQ": 1.0}', '[]', '{"norgate_snapshot_date_str": "2026-11-01"}');
            INSERT INTO decision_plan VALUES (2, 'full_target_weight_book', '{}', '{"TLT": 1.0}', '[]', '{}');
            INSERT INTO decision_plan VALUES (3, 'full_target_weight_book', '{}', '{"GLD": 1.0}', '[]', '{}');
            INSERT INTO vplan VALUES (10, 'pod_a', 1, '2026-11-02T14:30:00+00:00', 25000, 'completed');
            INSERT INTO vplan VALUES (11, 'pod_a', 2, '2026-12-01T14:30:00+00:00', 25500, 'completed');
            INSERT INTO vplan VALUES (12, 'pod_a', 3, '2027-01-04T14:30:00+00:00', 26000, 'ready');
            INSERT INTO vplan VALUES (13, 'pod_b', 3, '2026-11-02T14:30:00+00:00', 9000, 'completed');
            INSERT INTO vplan_fill VALUES (1, 10, 'QQQ', 40, 500.1, 500.0, '2026-11-02T14:30:05+00:00');
            INSERT INTO vplan_fill VALUES (2, 13, 'SPY', 4, 600.0, 600.0, '2026-11-02T14:30:05+00:00');
        """)
    return db_path


def test_pod_read_is_read_only_and_scoped(tmp_path):
    db_path = _pod_db(tmp_path)
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    source = bc.read_pod_source_dict(str(db_path), "pod_a", "2026-12-10")
    assert source["first_vplan_dict"]["session_date_str"] == "2026-11-02"
    assert source["latest_vplan_dict"]["vplan_id_int"] == 11  # the future 'ready' plan is not checked
    assert source["decision_dict"]["full_target_weight_map_dict"] == {"TLT": 1.0}
    assert [fill["asset_str"] for fill in source["fill_list"]] == ["QQQ"]
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    with pytest.raises(sqlite3.OperationalError):
        with bc._connect_ro(str(db_path)) as connection:
            connection.execute("DELETE FROM vplan")


def test_missing_database_is_not_created(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        bc.read_pod_source_dict(str(tmp_path / "absent.sqlite3"), "pod_a", "2026-12-10")
    assert not (tmp_path / "absent.sqlite3").exists()


def _target(db_path, pod_id_str, strategy_import_str, account_str):
    release = SimpleNamespace(pod_id_str=pod_id_str, account_route_str=account_str, release_id_str="r-" + pod_id_str,
                              strategy_import_str=strategy_import_str, mode_str="live", enabled_bool=True,
                              data_profile_str="")
    return SimpleNamespace(release_obj=release, db_path_str=str(db_path))


def test_report_without_backtests_keeps_record_and_builds_gate(tmp_path):
    db_path = _pod_db(tmp_path)
    targets = [_target(db_path, "pod_a", "strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil:X", "U1"),
               _target(db_path, "pod_b", "strategies.taa_df.strategy_taa_df_btal:Y", "U2")]
    previous = {"pod_list": [{"pod_id_str": "pod_a", "record_list": [{"session_str": "2026-11-02", "status_str": "same"}],
                              "curve_dict": {"point_list": [{"date_str": "2026-11-02", "live_float": 0.0, "backtest_float": 0.0}]}}]}
    report = bc.build_report_dict(targets, as_of_ts=datetime(2026, 12, 10, 23, 30, tzinfo=UTC), performance_db_path_str=None,
                                  previous_dict=previous, cache_dir_path=tmp_path / "cache", reference_bool=False)
    assert report["schema_version_str"] == bc.SCHEMA_VERSION_STR
    by_pod = {pod["pod_id_str"]: pod for pod in report["pod_list"]}
    assert by_pod["pod_a"]["decision_dict"]["status_str"] == "not_checked"
    assert by_pod["pod_a"]["record_list"] == previous["pod_list"][0]["record_list"]
    assert by_pod["pod_a"]["curve_dict"]["point_list"]  # last curve kept on a skipped night
    assert "fill_list" not in by_pod["pod_a"]
    assert report["cost_gate_dict"]["fill_count_int"] == 1  # only the MR pod's fills
    json.dumps(report)  # serialisable


def test_last_finished_session_is_before_the_run_date():
    assert bc.last_finished_session_str("2026-12-14") == "2026-12-11"  # Monday morning run: Friday
    assert bc.last_finished_session_str("2026-11-27") == "2026-11-25"  # after Thanksgiving


@pytest.mark.parametrize('failed_pod_list', [['pod_b'], ['pod_a', 'pod_b']])
def test_unreadable_mr_source_never_passes_the_pooled_gate(tmp_path, monkeypatch, failed_pod_list):
    target_list = [_target(tmp_path / 'missing', pod_id_str, 'strategies.mr_capsule.test:X', pod_id_str)
                   for pod_id_str in ('pod_a', 'pod_b')]
    def check_fn(target_obj, **kwarg_dict):
        pod_id_str = target_obj.release_obj.pod_id_str
        if pod_id_str in failed_pod_list:
            raise OSError('source unreadable')
        return {'pod_id_str': pod_id_str, 'reference_status_str': 'ok', 'decision_dict': {'status_str': 'same'},
                'fill_list': [_fill('AAA', 10, 100, 100, '2026-10-01')] * 200}
    monkeypatch.setattr(bc, 'check_pod_dict', check_fn)
    monkeypatch.setattr(bc, 'live_return_dict_by_account', lambda *arg_tuple: {})
    report_dict = bc.build_report_dict(target_list, as_of_ts=datetime(2026, 10, 16, 12, tzinfo=UTC),
                                     performance_db_path_str=None, previous_dict={}, cache_dir_path=tmp_path)
    gate_dict = report_dict['cost_gate_dict']
    assert gate_dict['status_str'] == 'incomplete'
    assert gate_dict['unavailable_pod_list'] == failed_pod_list
    assert gate_dict['average_bps_float'] is None
    assert gate_dict['time_cap_date_str'] is None


def test_ruler_failure_keeps_the_decision_and_curve(tmp_path, monkeypatch):
    import pandas as pd
    from alpha.live import reference_compare

    db_path = _pod_db(tmp_path)
    monkeypatch.setattr(reference_compare, "inspect_auto_reference_support_dict", lambda release: {"supported_bool": True, "reason_str": "ok"})
    calls = []

    def fake_run(release, *, start_str, end_str, capital_float, output_dir_path):
        calls.append((start_str, end_str))
        if start_str is None:
            raise MemoryError("ruler run")
        session_list = bc.session_date_list(start_str, end_str)
        return {"transaction_map_dict": {}, "realized_weight_history_df": pd.DataFrame({"TLT": [1.0]}, index=["2026-12-01"]),
                "equity_by_date_dict": {session_str: capital_float for session_str in session_list}}

    monkeypatch.setattr(bc, "run_reference_maps_dict", fake_run)
    target = _target(db_path, "pod_a", "strategies.ndx.strategy_ndx:Z", "U1")
    pod = bc.check_pod_dict(target, as_of_date_str="2026-12-04",
                            live_return_dict={session_str: 0.0 for session_str in bc.session_date_list("2026-11-02", "2026-12-03")},
                            previous_pod_dict=None, cache_dir_path=tmp_path / "cache", reference_bool=True,
                            refresh_ruler_bool=True, month_end_bool=False)
    assert pod["reference_status_str"] == "ok"
    assert pod["decision_dict"]["status_str"] == "same"
    assert len(pod["curve_dict"]["point_list"]) == len(bc.session_date_list("2026-11-02", "2026-12-03"))
    assert pod["ruler_reason_str"].startswith("Ruler unavailable")
    assert calls[0] == ("2026-11-02", "2026-12-03")  # the backtest runs to the last finished session, not the plan date


def test_different_low_exposure_symbol_books_never_match():
    decision_dict = {"decision_book_type_str": "full_target_weight_book",
                     "full_target_weight_map_dict": {f"LIVE{symbol_int}": 0.025 for symbol_int in range(10)}}
    comparison_dict = bc.compare_decision_dict(decision_dict,
        reference_weight_map_dict={f"REF{symbol_int}": 0.025 for symbol_int in range(10)}, reference_trade_map_dict={})
    assert comparison_dict["status_str"] == "differs"
    assert comparison_dict["differ_count_int"] == 20


@pytest.mark.parametrize("exposure_sign_int", [1, -1])
def test_per_symbol_tolerance_cannot_hide_doubled_total_exposure(exposure_sign_int):
    decision_dict = {"decision_book_type_str": "full_target_weight_book",
                     "full_target_weight_map_dict": {f"AAA{symbol_int}": exposure_sign_int * 0.05 for symbol_int in range(10)}}
    comparison_dict = bc.compare_decision_dict(decision_dict,
        reference_weight_map_dict={f"AAA{symbol_int}": exposure_sign_int * 0.025 for symbol_int in range(10)},
        reference_trade_map_dict={})
    assert comparison_dict["status_str"] == "differs"
    assert comparison_dict["differ_count_int"] == 1
    assert comparison_dict["row_list"][0] == {"asset_str": "Total exposure", "live_str": "50.0%",
                                             "backtest_str": "25.0%", "same_bool": False}


@pytest.mark.parametrize("live_weight_float,reference_weight_float,expected_str", [(0.1, 0.1, "same"), (0.9, 0.1, "differs")])
def test_incremental_entry_checks_the_intended_weight(live_weight_float, reference_weight_float, expected_str):
    comparison_dict = bc.compare_decision_dict(
        {"decision_book_type_str": "incremental_entry_exit_book", "entry_target_weight_map_dict": {"AAA": live_weight_float}},
        reference_weight_map_dict={"AAA": 0.075}, reference_trade_map_dict={"AAA": 20},
        reference_entry_weight_map_dict={"AAA": reference_weight_float})
    assert comparison_dict["status_str"] == expected_str


def test_unknown_entry_weight_does_not_approve_a_decision():
    comparison_dict = bc.compare_decision_dict(
        {"decision_book_type_str": "incremental_entry_exit_book", "entry_target_weight_map_dict": {"AAA": 0.1}},
        reference_weight_map_dict={"AAA": 0.1}, reference_trade_map_dict={"AAA": 20})
    assert comparison_dict["status_str"] == "not_checked"
    assert comparison_dict["row_list"][0]["same_bool"] is None


def test_cost_component_deadline_uses_when_the_sample_completed():
    first_fill_dict = _fill("AAA", 10, 100, 100, "2026-01-02")
    late_gate_dict = bc.cost_gate_dict({"pod_a": [first_fill_dict] * 199 + [_fill("AAA", 10, 100, 100, "2026-08-01")]},
                                     as_of_date_str="2026-08-01")
    assert late_gate_dict["status_str"] == "time_cap"
    early_gate_dict = bc.cost_gate_dict({"pod_a": [first_fill_dict] * 199 + [_fill("AAA", 10, 100, 100, "2026-06-01")]},
                                      as_of_date_str="2026-08-01")
    assert early_gate_dict["status_str"] == "passed"
    assert early_gate_dict["scope_str"] == "cost_component_only"


def test_missing_official_open_does_not_move_the_deadline():
    gate_dict = bc.cost_gate_dict({"pod_a": [_fill("AAA", 10, 100, None, "2026-01-02"),
                                            _fill("AAA", 10, 100, 100, "2026-06-01")]}, as_of_date_str="2026-08-01")
    assert gate_dict["time_cap_date_str"] == "2026-07-04"
    assert gate_dict["status_str"] == "time_cap"


def test_first_session_return_includes_the_initial_cash_baseline():
    return_dict = bc.daily_returns_from_equity({"2026-10-12": 900.0, "2026-10-13": 990.0}, initial_capital_float=1000)
    assert return_dict == pytest.approx({"2026-10-12": -0.1, "2026-10-13": 0.1})


@pytest.mark.parametrize("missing_side_str", ["live", "reference"])
def test_incomplete_coverage_never_displays_a_partial_gap(missing_side_str):
    live_dict = {"2026-10-12": 0.0, "2026-10-13": -0.2, "2026-10-14": 0.0}
    reference_dict = {session_str: 0.0 for session_str in live_dict}
    (live_dict if missing_side_str == "live" else reference_dict).pop("2026-10-13")
    curve_dict = bc.curve_dict(live_dict, reference_dict)
    assert curve_dict["coverage_status_str"] == "incomplete"
    assert curve_dict["gap_float"] is None
    assert curve_dict["point_list"] == []


def test_rare_windows_are_consecutive_and_end_on_the_required_session():
    ruler_dict = {"3": {"one_in_100_float": -0.05, "worst_float": -0.2}}
    sparse_return_dict = {"2026-10-12": 0.0, "2026-10-14": -0.03, "2026-10-16": -0.03}
    assert bc.rare_loss_dict(sparse_return_dict, sparse_return_dict, ruler_dict) is None
    daily_ruler_dict = {"1": {"one_in_100_float": -0.02, "worst_float": -0.2}}
    assert bc.rare_loss_dict({"2026-10-14": -0.03}, {"2026-10-14": -0.03}, daily_ruler_dict,
                             expected_end_str="2026-10-16") is None


def test_historical_run_resolves_the_declared_start_and_entry_slot_weight(tmp_path, monkeypatch):
    import pandas as pd
    from alpha.live import reference_compare
    from strategies.mr_capsule.dv2_vix_gated import DV2VixGatedStrategy

    strategy_obj = DV2VixGatedStrategy(name="reference", benchmarks=[], capital_base=1000)
    strategy_obj.max_positions = 5
    captured_dict = {}
    def run_reference_fn(**kwargs):
        captured_dict.update(kwargs)
        return strategy_obj
    monkeypatch.setattr(reference_compare, "run_auto_reference_strategy", run_reference_fn)
    monkeypatch.setattr(reference_compare, "build_reference_maps_dict", lambda strategy_obj: {
        "reference_strategy_obj": strategy_obj, "transaction_map_dict": {("2026-10-12", "AAA"): {"quantity_float": 5}},
        "equity_by_date_dict": {"2026-10-12": 1000}, "realized_weight_history_df": pd.DataFrame()})
    release_obj = _target(tmp_path / "absent", "pod_a", "strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil:X", "U1").release_obj
    maps_dict = bc.run_reference_maps_dict(release_obj, start_str=None, end_str="2026-10-12", capital_float=1000,
                                           output_dir_path=tmp_path / "reference")
    assert captured_dict["deployment_start_date_str"] == "2004-01-01"
    assert maps_dict["entry_weight_by_date_dict"] == {("2026-10-12", "AAA"): 0.2}


def test_malformed_ruler_never_discards_the_valid_nightly_check(tmp_path, monkeypatch):
    import pandas as pd
    from alpha.live import reference_compare

    db_path = _pod_db(tmp_path)
    session_list = bc.session_date_list("2026-11-02", "2026-12-03")
    monkeypatch.setattr(reference_compare, "inspect_auto_reference_support_dict", lambda release_obj: {"supported_bool": True})
    monkeypatch.setattr(bc, "run_reference_maps_dict", lambda *args, **kwargs: {
        "transaction_map_dict": {}, "realized_weight_history_df": pd.DataFrame({"TLT": [1.0]}, index=["2026-12-01"]),
        "equity_by_date_dict": {session_str: 25000 for session_str in session_list}})
    monkeypatch.setattr(bc, "load_or_build_ruler_dict", lambda *args, **kwargs: {
        "window_dict": {"1": {"one_in_100_float": "bad", "worst_float": -0.2}}})
    report_dict = bc.build_report_dict([_target(db_path, "pod_a", "strategies.ndx.strategy_ndx:X", "U1")],
        as_of_ts=datetime(2026, 12, 4, 13, tzinfo=UTC), performance_db_path_str=None, previous_dict={}, cache_dir_path=tmp_path)
    pod_dict = report_dict["pod_list"][0]
    assert pod_dict["reference_status_str"] == "ok"
    assert pod_dict["decision_dict"]["status_str"] == "same"
    assert pod_dict["ruler_reason_str"].startswith("Ruler unavailable")


def test_missing_exact_decision_session_does_not_use_previous_holdings(tmp_path, monkeypatch):
    import pandas as pd
    from alpha.live import reference_compare

    monkeypatch.setattr(reference_compare, "inspect_auto_reference_support_dict", lambda release_obj: {"supported_bool": True})
    monkeypatch.setattr(bc, "run_reference_maps_dict", lambda *args, **kwargs: {
        "transaction_map_dict": {}, "realized_weight_history_df": pd.DataFrame({"TLT": [1.0]}, index=["2026-11-30"]),
        "equity_by_date_dict": {"2026-11-30": 25000}})
    pod_dict = bc.check_pod_dict(_target(_pod_db(tmp_path), "pod_a", "strategies.ndx.strategy_ndx:X", "U1"),
        as_of_date_str="2026-12-04", live_return_dict={}, previous_pod_dict=None, cache_dir_path=tmp_path,
        reference_bool=True, refresh_ruler_bool=False, month_end_bool=False)
    assert pod_dict["decision_dict"]["status_str"] == "not_checked"
    assert pod_dict["reference_status_str"] == "incomplete"


@pytest.mark.parametrize("saved_json_str", ["broken", "[]", '{"QQQ":NaN}', '{"QQQ":true}', '{"QQQ":"0.1"}'])
def test_malformed_saved_decisions_are_not_coerced_to_empty_books(tmp_path, saved_json_str):
    db_path = _pod_db(tmp_path)
    with sqlite3.connect(db_path) as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET full_target_weight_json_str=? WHERE decision_plan_id_int=2", (saved_json_str,))
    with pytest.raises(ValueError):
        bc.read_pod_source_dict(str(db_path), "pod_a", "2026-12-10")


def test_unknown_saved_book_type_is_not_treated_as_incremental(tmp_path):
    db_path = _pod_db(tmp_path)
    with sqlite3.connect(db_path) as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET decision_book_type_str='unknown' WHERE decision_plan_id_int=2")
    with pytest.raises(ValueError):
        bc.read_pod_source_dict(str(db_path), "pod_a", "2026-12-10")


def test_first_budget_and_start_ignore_an_earlier_unexecuted_plan(tmp_path):
    db_path = _pod_db(tmp_path)
    with sqlite3.connect(db_path) as connection_obj:
        connection_obj.execute("INSERT INTO vplan VALUES (1, 'pod_a', 1, '2026-10-01T13:30:00+00:00', 999, 'blocked')")
    source_dict = bc.read_pod_source_dict(str(db_path), "pod_a", "2026-12-10")
    assert source_dict["first_vplan_dict"]["vplan_id_int"] == 10
    assert source_dict["first_vplan_dict"]["pod_budget_float"] == 25000


def test_opposite_exposure_directions_never_match_inside_weight_tolerance():
    comparison_dict = bc.compare_decision_dict(
        {"decision_book_type_str": "full_target_weight_book", "full_target_weight_map_dict": {"AAA": 0.02}},
        reference_weight_map_dict={"AAA": -0.02}, reference_trade_map_dict={})
    assert comparison_dict["status_str"] == "differs"


def test_malformed_previous_report_is_dropped_before_partial_merge(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"schema_version_str": bc.SCHEMA_VERSION_STR, "pod_list": [1]}), encoding="utf-8")
    assert bc._read_previous_dict(report_path) == {}


@pytest.mark.parametrize("fail_run_bool", [False, True])
def test_reference_wrapper_isolates_fred_writes_even_when_the_run_fails(tmp_path, monkeypatch, fail_run_bool):
    import pandas as pd
    from alpha.data import fred_loader
    from alpha.live import reference_compare
    from strategies.dv2.strategy_mr_dv2 import DVO2Strategy

    source_path = tmp_path / "live" / "DTB3.csv"
    source_path.parent.mkdir()
    original_bytes = b"DATE,DTB3\n2026-10-12,4.0\n"
    source_path.write_bytes(original_bytes)
    download_bytes = b"DATE,DTB3\n2026-10-12,5.0\n"
    from contextlib import contextmanager
    @contextmanager
    def response_fn(*args, **kwargs):
        yield SimpleNamespace(read=lambda: download_bytes)
    monkeypatch.setattr(fred_loader, "urlopen", response_fn)
    def run_reference_fn(**kwargs):
        fred_loader.load_daily_fred_series_snapshot("DTB3", str(source_path), datetime(2026, 10, 12, tzinfo=UTC), "backtest")
        if fail_run_bool:
            raise RuntimeError("reference failed")
        return DVO2Strategy(name="reference", benchmarks=[])
    monkeypatch.setattr(reference_compare, "run_auto_reference_strategy", run_reference_fn)
    monkeypatch.setattr(reference_compare, "build_reference_maps_dict", lambda strategy_obj: {
        "transaction_map_dict": {}, "realized_weight_history_df": pd.DataFrame(), "equity_by_date_dict": {}})
    release_obj = _target(tmp_path / "absent", "pod_a", "strategies.taa_df.strategy_taa_df:X", "U1").release_obj
    run_kwarg_dict = dict(start_str="2026-10-12", end_str="2026-10-12", capital_float=1000, output_dir_path=tmp_path / "reference")
    if fail_run_bool:
        with pytest.raises(RuntimeError):
            bc.run_reference_maps_dict(release_obj, **run_kwarg_dict)
    else:
        bc.run_reference_maps_dict(release_obj, **run_kwarg_dict)
    assert source_path.read_bytes() == original_bytes
    assert (tmp_path / "reference" / "macro_cache" / "DTB3.csv").read_text(encoding="utf-8") == download_bytes.decode("utf-8")
    fred_loader.load_daily_fred_series_snapshot("DTB3", str(source_path), datetime(2026, 10, 12, tzinfo=UTC), "live")
    assert source_path.read_text(encoding="utf-8") == download_bytes.decode("utf-8")  # ordinary live write restored


def test_taa_historical_start_retains_its_config_default(tmp_path, monkeypatch):
    from alpha.live import reference_compare
    from strategies.taa_df import strategy_taa_df
    from strategies.dv2.strategy_mr_dv2 import DVO2Strategy
    captured_dict = {}
    def run_variant_fn(**kwargs):
        captured_dict.update(kwargs)
        return DVO2Strategy(name="reference", benchmarks=[], capital_base=kwargs["capital_base_float"])
    monkeypatch.setattr(strategy_taa_df, "run_defense_first_variant", run_variant_fn)
    release_obj = _target(tmp_path / "absent", "pod_a", "strategies.taa_df.strategy_taa_df:X", "U1").release_obj
    bc.run_reference_maps_dict(release_obj, start_str=None, end_str="2026-10-12", capital_float=1000, output_dir_path=tmp_path / "reference")
    assert captured_dict["backtest_start_date_str"] is None
    assert captured_dict["config"].start_date_str == strategy_taa_df.DEFAULT_CONFIG.start_date_str


@pytest.mark.parametrize("window_int", [1, 3, 5])
def test_missing_prior_nav_never_attributes_a_multi_session_loss_as_daily(window_int):
    session_list = bc.session_date_list("2026-10-12", "2026-10-30")
    window_date_list = session_list[2:2 + window_int]
    equity_dict = {session_list[0]: 1000.0, **{session_str: 900.0 for session_str in window_date_list}}
    reference_return_dict = bc.daily_returns_from_equity(equity_dict, initial_capital_float=1000.0)
    assert window_date_list[0] not in reference_return_dict  # prior Oct 13 NAV is missing
    live_return_dict = {session_str: -0.1 if session_str == window_date_list[0] else 0.0 for session_str in window_date_list}
    rare_dict = bc.rare_loss_dict(live_return_dict, reference_return_dict,
        {str(window_int): {"one_in_100_float": -0.05, "worst_float": -0.2}}, expected_end_str=window_date_list[-1])
    assert rare_dict is not None
    assert rare_dict["cause_str"] == "unknown"
    assert rare_dict["backtest_float"] is None


def test_nontrading_days_do_not_break_consecutive_session_nav_returns():
    return_dict = bc.daily_returns_from_equity({"2026-10-09": 1000.0, "2026-10-12": 900.0})
    assert return_dict == pytest.approx({"2026-10-12": -0.1})


def test_missing_initial_session_is_not_rebased_into_a_later_daily_return():
    return_dict = bc.daily_returns_from_equity({"2026-10-14": 900.0, "2026-10-15": 900.0},
                                               initial_capital_float=1000, initial_session_str="2026-10-12")
    assert "2026-10-14" not in return_dict
    assert return_dict["2026-10-15"] == 0.0
