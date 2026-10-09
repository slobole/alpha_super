"""Optional broker history/open references never gate daily holdings settlement."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from alpha.live import daily_reporting, runner
from alpha.live.daily_reporting import claim_post_close_report_attempt
from alpha.live.state_store_v2 import LiveStateStore
from test_live_daily_reconcile import CLOSE_TS, INTRADAY_TS, MARKET_ZONE_OBJ, daily_case


def _reporting_counts_dict(broker_obj, failure_str=None):
    count_dict = {"history": 0, "open_price": 0}

    def history_fn(**_argument_dict):
        count_dict["history"] += 1
        if failure_str == "history":
            raise RuntimeError("synthetic reporting history failure")
        return [], [], []

    def open_price_fn(**_argument_dict):
        count_dict["open_price"] += 1
        if failure_str == "open_price":
            raise RuntimeError("synthetic open reference failure")
        return []

    broker_obj.get_recent_order_state_snapshot = history_fn
    broker_obj.get_session_open_price_list = open_price_fn
    return count_dict


def _poll(store_obj, broker_obj, tmp_path):
    return runner._reconcile_daily_cycles(store_obj,
        SimpleNamespace(get_adapter=lambda _release_obj: broker_obj), broker_obj.as_of_ts,
        "paper", None, str(tmp_path / "report.log"), False, str(tmp_path / "traces"))


def test_intraday_polls_and_completion_records_never_query_history_or_open(daily_case, tmp_path):
    store_obj, _, decision_obj, _, broker_obj = daily_case
    count_dict = _reporting_counts_dict(broker_obj)
    for minute_int in range(3):
        broker_obj.as_of_ts = INTRADAY_TS + timedelta(minutes=minute_int)
        previous_count_int = broker_obj.refresh_count_int
        assert _poll(store_obj, broker_obj, tmp_path) == 0
        assert broker_obj.refresh_count_int > previous_count_int
    # The first poll sends the permitted MSFT/BIL remainder sales and persists
    # their returned reporting payloads, still without another broker query.
    assert {request_obj.asset_str for request_obj in broker_obj.sent_request_list} == {"MSFT", "BIL"}
    assert count_dict == {"history": 0, "open_price": 0}
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "submitted"


def test_first_post_close_report_is_durable_across_failed_finalize_and_restart(daily_case, tmp_path, monkeypatch):
    store_obj, _, decision_obj, plan_obj, broker_obj = daily_case
    broker_obj.position_dict.update(plan_obj.target_share_map)
    broker_obj.as_of_ts = CLOSE_TS
    count_dict = _reporting_counts_dict(broker_obj)

    def fail_finalize_fn(*_argument_list, **_argument_dict):
        raise RuntimeError("synthetic failure after reporting")

    monkeypatch.setattr(store_obj, "complete_daily_cycle", fail_finalize_fn)
    assert _poll(store_obj, broker_obj, tmp_path) == 0
    assert count_dict == {"history": 1, "open_price": 1}
    previous_count_int = broker_obj.refresh_count_int
    broker_obj.as_of_ts += timedelta(seconds=30)
    assert _poll(store_obj, broker_obj, tmp_path) == 0
    assert count_dict == {"history": 1, "open_price": 1}
    assert broker_obj.refresh_count_int > previous_count_int
    restarted_store_obj = LiveStateStore(store_obj.db_path_str)
    previous_count_int = broker_obj.refresh_count_int
    broker_obj.as_of_ts += timedelta(seconds=30)
    assert _poll(restarted_store_obj, broker_obj, tmp_path) == 1
    assert count_dict == {"history": 1, "open_price": 1}
    assert broker_obj.refresh_count_int > previous_count_int
    assert restarted_store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "completed"
    with restarted_store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT COUNT(*) FROM daily_post_close_report").fetchone()[0] == 1
        assert connection_obj.execute("SELECT status_str,error_str FROM daily_post_close_report").fetchone()[:] == ("reported", None)


def test_post_close_settlement_refreshes_again_after_optional_reporting(daily_case, tmp_path):
    store_obj, _, decision_obj, plan_obj, broker_obj = daily_case
    broker_obj.position_dict.update(plan_obj.target_share_map)
    broker_obj.as_of_ts = CLOSE_TS
    def history_fn(**_argument_dict):
        broker_obj.position_dict["FOREIGN"] = 3.0
        return [], [], []
    broker_obj.get_recent_order_state_snapshot = history_fn
    broker_obj.get_session_open_price_list = lambda **_argument_dict: []
    assert _poll(store_obj, broker_obj, tmp_path) == 1
    assert broker_obj.refresh_count_int == 2  # pre-report and fresh settlement
    saved_decision_obj = store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int)
    assert saved_decision_obj.status_str == "completed_with_exceptions"
    assert any(row_dict["asset_str"] == "FOREIGN" for row_dict in
        saved_decision_obj.snapshot_metadata_dict["daily_execution_result_dict"]["exception_list"])


@pytest.mark.parametrize("failure_str", ["history", "open_price"])
def test_reporting_failure_does_not_block_actual_holdings_completion(daily_case, tmp_path, failure_str):
    store_obj, _, decision_obj, plan_obj, broker_obj = daily_case
    broker_obj.position_dict.update(plan_obj.target_share_map)
    broker_obj.as_of_ts = CLOSE_TS
    count_dict = _reporting_counts_dict(broker_obj, failure_str)
    assert _poll(store_obj, broker_obj, tmp_path) == 1
    assert broker_obj.refresh_count_int > 0
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "completed"
    assert count_dict == {"history": 1, "open_price": int(failure_str != "history")}
    log_str = (tmp_path / "report.log").read_text(encoding="utf-8")
    assert ("daily_fill_reporting_unavailable" if failure_str == "history"
        else "daily_open_price_reporting_unavailable") in log_str
    with store_obj._connect() as connection_obj:
        audit_row_obj = connection_obj.execute("SELECT status_str,error_str FROM daily_post_close_report").fetchone()
    assert audit_row_obj["status_str"] == "failed"
    assert "synthetic" in audit_row_obj["error_str"]
    retry_count_dict = _reporting_counts_dict(broker_obj)
    broker_obj.as_of_ts += timedelta(seconds=30)
    assert _poll(store_obj, broker_obj, tmp_path) == 0
    assert retry_count_dict == {"history": 1, "open_price": 1}
    with store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT status_str,error_str FROM daily_post_close_report").fetchone()[:] == (
            "reported", None)
    broker_obj.as_of_ts += timedelta(seconds=30)
    assert _poll(store_obj, broker_obj, tmp_path) == 0
    assert retry_count_dict == {"history": 1, "open_price": 1}


@pytest.mark.parametrize("failure_str", ["claim_post_close_report_attempt", "finish_post_close_report_attempt"])
def test_reporting_audit_failure_cannot_skip_mandatory_snapshot(daily_case, tmp_path, monkeypatch, failure_str):
    store_obj, _, decision_obj, plan_obj, broker_obj = daily_case
    broker_obj.position_dict.update(plan_obj.target_share_map)
    broker_obj.as_of_ts = CLOSE_TS
    count_dict = _reporting_counts_dict(broker_obj)

    def fail_audit_fn(*_argument_list, **_argument_dict):
        raise RuntimeError("synthetic optional reporting audit failure")

    monkeypatch.setattr(daily_reporting, failure_str, fail_audit_fn)
    assert _poll(store_obj, broker_obj, tmp_path) == 1
    assert broker_obj.refresh_count_int > 0
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "completed"
    expected_count_int = int(failure_str == "finish_post_close_report_attempt")
    assert count_dict == {"history": expected_count_int, "open_price": expected_count_int}
    assert "synthetic optional reporting audit failure" in (tmp_path / "report.log").read_text(encoding="utf-8")


def test_failed_first_refresh_leaves_post_close_report_unclaimed_for_retry(daily_case, tmp_path):
    store_obj, _, _, plan_obj, broker_obj = daily_case
    broker_obj.position_dict.update(plan_obj.target_share_map)
    broker_obj.as_of_ts = CLOSE_TS
    broker_obj.error_obj = TimeoutError("synthetic mandatory snapshot failure")
    count_dict = _reporting_counts_dict(broker_obj)
    assert _poll(store_obj, broker_obj, tmp_path) == 0
    assert count_dict == {"history": 0, "open_price": 0}
    with store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT 1 FROM sqlite_master WHERE name='daily_post_close_report'").fetchone() is None
    broker_obj.error_obj = None
    broker_obj.as_of_ts += timedelta(seconds=30)
    restarted_store_obj = LiveStateStore(store_obj.db_path_str)
    assert _poll(restarted_store_obj, broker_obj, tmp_path) == 1
    assert count_dict == {"history": 1, "open_price": 1}
    assert broker_obj.refresh_count_int == 3  # failed pass, pre-report retry, fresh settlement
    with restarted_store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT status_str FROM daily_post_close_report").fetchone()[0] == "reported"
    broker_obj.as_of_ts += timedelta(seconds=30)
    assert _poll(restarted_store_obj, broker_obj, tmp_path) == 0
    assert count_dict == {"history": 1, "open_price": 1}


def test_early_close_gates_history_at_exchange_close_not_1600(daily_case, tmp_path):
    store_obj, _, decision_obj, plan_obj, broker_obj = daily_case
    target_ts = datetime(2026, 11, 27, 9, 30, tzinfo=MARKET_ZONE_OBJ)
    close_ts = target_ts.replace(hour=13, minute=0)
    with store_obj._connect() as connection_obj:
        for table_str in ("decision_plan", "vplan"):
            connection_obj.execute(f"UPDATE {table_str} SET target_execution_timestamp_str=? WHERE decision_plan_id_int=?",
                (target_ts.isoformat(), decision_obj.decision_plan_id_int))
    broker_obj.position_dict.update(plan_obj.target_share_map)
    count_dict = _reporting_counts_dict(broker_obj)
    broker_obj.as_of_ts = close_ts - timedelta(seconds=1)
    assert _poll(store_obj, broker_obj, tmp_path) == 0
    assert count_dict == {"history": 0, "open_price": 0}
    previous_count_int = broker_obj.refresh_count_int
    broker_obj.as_of_ts = close_ts
    assert _poll(store_obj, broker_obj, tmp_path) == 1
    assert count_dict == {"history": 1, "open_price": 1}
    assert broker_obj.refresh_count_int > previous_count_int


def test_post_close_claim_is_atomic_for_competing_workers(daily_case):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    assert not claim_post_close_report_attempt(store_obj, release_obj, decision_obj, CLOSE_TS - timedelta(seconds=1))
    with ThreadPoolExecutor(max_workers=4) as executor_obj:
        result_list = list(executor_obj.map(lambda _index_int:
            claim_post_close_report_attempt(store_obj, release_obj, decision_obj, CLOSE_TS), range(4)))
    assert result_list.count(True) == 1
    assert not claim_post_close_report_attempt(LiveStateStore(store_obj.db_path_str),
        release_obj, decision_obj, CLOSE_TS + timedelta(seconds=30))


def test_failed_report_retry_is_scoped_to_enabled_release_and_mode(daily_case):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    assert claim_post_close_report_attempt(store_obj, release_obj, decision_obj, CLOSE_TS)
    daily_reporting.finish_post_close_report_attempt(store_obj, decision_obj, "broker outage")
    retry_ids = daily_reporting.get_retry_post_close_report_decision_id_list
    assert retry_ids(store_obj, CLOSE_TS, env_mode_str="paper") == [decision_obj.decision_plan_id_int]
    assert retry_ids(store_obj, CLOSE_TS, env_mode_str="live") == []
    store_obj.upsert_release(replace(release_obj, enabled_bool=False))
    assert retry_ids(store_obj, CLOSE_TS, env_mode_str="paper") == []


def test_reporting_claim_rejects_monthly_release_without_creating_daily_table(daily_case):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    monthly_release_obj = replace(release_obj, strategy_import_str="strategies.ndx.strategy_ndx")
    with pytest.raises(ValueError, match="limited to CORE5 and MR capsule"):
        claim_post_close_report_attempt(store_obj, monthly_release_obj, decision_obj, CLOSE_TS)
    with store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT 1 FROM sqlite_master WHERE name='daily_post_close_report'").fetchone() is None
