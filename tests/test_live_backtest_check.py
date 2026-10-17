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
        return {"transaction_map_dict": {}, "realized_weight_history_df": pd.DataFrame({"TLT": [1.0]}, index=["2026-12-01"]),
                "equity_by_date_dict": {"2026-12-01": 100.0, "2026-12-02": 101.0, "2026-12-03": 100.5}}

    monkeypatch.setattr(bc, "run_reference_maps_dict", fake_run)
    target = _target(db_path, "pod_a", "strategies.ndx.strategy_ndx:Z", "U1")
    pod = bc.check_pod_dict(target, as_of_date_str="2026-12-04", live_return_dict={"2026-12-02": 0.01, "2026-12-03": -0.005},
                            previous_pod_dict=None, cache_dir_path=tmp_path / "cache", reference_bool=True,
                            refresh_ruler_bool=True, month_end_bool=False)
    assert pod["reference_status_str"] == "ok"
    assert pod["decision_dict"]["status_str"] == "same"
    assert len(pod["curve_dict"]["point_list"]) == 2
    assert pod["ruler_reason_str"].startswith("Ruler unavailable")
    assert calls[0] == ("2026-11-02", "2026-12-03")  # the backtest runs to the last finished session, not the plan date
