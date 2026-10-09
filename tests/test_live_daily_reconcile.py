"""Daily CORE5/capsule lifecycle uses fresh holdings, never fill evidence."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo

import pytest

from alpha.live.core5_adapter import CORE5_CONTRACT_STR, CORE5_STRATEGY_IMPORT_STR
from alpha.live.daily_broker import DailyExecutionSnapshot
from alpha.live.daily_reconcile import (
    DAILY_TERMINAL_STATUS_SET, claim_daily_completion_request, ensure_daily_reconcile_schema,
    is_daily_reconcile_release_bool, reconcile_daily_cycle,
)
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerSnapshot, DecisionPlan, LiveRelease, SubmitBatchResult, VPlan, VPlanRow
from alpha.live.state_store_v2 import LiveStateStore
from alpha.live.scheduler_service import _build_stuck_operator_message_spec_list


MARKET_ZONE_OBJ = ZoneInfo("America/New_York")
OPEN_TS = datetime(2026, 10, 5, 9, 30, tzinfo=MARKET_ZONE_OBJ)
INTRADAY_TS = OPEN_TS + timedelta(minutes=10)
CLOSE_TS = OPEN_TS.replace(hour=16, minute=0)


class DailyBroker:
    def __init__(self, position_dict, as_of_ts=INTRADAY_TS):
        self.position_dict = dict(position_dict)
        self.as_of_ts = as_of_ts
        self.open_row_list = []
        self.sent_request_list = []
        self.cancel_ref_list = []
        self.refresh_count_int = 0
        self.snapshot_transform_fn = lambda snapshot_obj: snapshot_obj
        self.error_obj = None
        self.send_error_obj = None
        self.cancel_confirmed_bool = True

    def get_daily_execution_snapshot(self, account_route_str):
        self.refresh_count_int += 1
        if self.error_obj:
            raise self.error_obj
        snapshot_obj = BrokerSnapshot(account_route_str, self.as_of_ts, 1000.0, 100000.0,
            position_amount_map=dict(self.position_dict), net_liq_float=100000.0)
        return self.snapshot_transform_fn(DailyExecutionSnapshot(snapshot_obj,
            [dict(row_dict) for row_dict in self.open_row_list], self.as_of_ts, self.as_of_ts))

    def cancel_daily_owned_orders(self, account_route_str, owned_order_ref_set, *, session_close_timestamp_ts):
        assert self.as_of_ts >= session_close_timestamp_ts
        self.cancel_ref_list.append(set(owned_order_ref_set))
        if self.cancel_confirmed_bool:
            self.open_row_list = [row_dict for row_dict in self.open_row_list if row_dict["order_ref_str"] not in owned_order_ref_set]
        return self.get_daily_execution_snapshot(account_route_str)

    def submit_order_request_list(self, *, account_route_str, broker_order_request_list, submitted_timestamp_ts):
        self.sent_request_list.extend(broker_order_request_list)
        if self.send_error_obj:
            raise self.send_error_obj
        for request_obj in broker_order_request_list:
            self.position_dict[request_obj.asset_str] = self.position_dict.get(request_obj.asset_str, 0.0) + request_obj.amount_float
        return SubmitBatchResult()  # Deliberately no fills or ACKs.


@pytest.fixture(params=["capsule", "core5"])
def daily_case(request, tmp_path):
    strategy_str = ("strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil"
        if request.param == "capsule" else CORE5_STRATEGY_IMPORT_STR)
    release_obj = LiveRelease("daily.v1", "test_user", "daily_pod", "DU_TEST", strategy_str,
        "paper", "XNYS", "eod_snapshot_ready", "next_open_moo", "synthetic", {}, "standard", True,
        str(tmp_path / "release.yaml"), pod_budget_fraction_float=1.0)
    store_obj = LiveStateStore(str(tmp_path / "daily.sqlite3"))
    with store_obj._connect() as connection_obj:
        ensure_daily_reconcile_schema(connection_obj)
    store_obj.upsert_release(release_obj)
    base_dict = {"MSFT": 5.0, "BIL": 100.0, "UNTOUCHED": 2.0}
    target_dict = {"AAPL": 10.0, "MSFT": 0.0, "BIL": 80.0}
    if request.param == "core5":
        target_dict["DBC"] = -5.0
    metadata_dict = {"sizing_contract_str": CORE5_CONTRACT_STR if request.param == "core5" else "mr_capsule_close_targets_v1"}
    if request.param == "core5":
        metadata_dict["fixed_target_share_map_dict"] = dict(target_dict)
    decision_obj = store_obj.insert_decision_plan(DecisionPlan(
        release_obj.release_id_str, release_obj.user_id_str, release_obj.pod_id_str, release_obj.account_route_str,
        OPEN_TS - timedelta(days=3), OPEN_TS - timedelta(minutes=8), OPEN_TS, "next_open_moo", base_dict,
        metadata_dict, {"committed_after_close_bool": True}, entry_target_weight_map_dict={"AAPL": 0.1},
        exit_asset_set={"MSFT"}, target_share_map_dict={"BIL": 80.0}))
    delta_dict = {asset_str: target_float - base_dict.get(asset_str, 0.0) for asset_str, target_float in target_dict.items()}
    plan_obj = store_obj.insert_vplan(VPlan(
        release_obj.release_id_str, release_obj.user_id_str, release_obj.pod_id_str, release_obj.account_route_str,
        decision_obj.decision_plan_id_int, decision_obj.signal_timestamp_ts, decision_obj.submission_timestamp_ts,
        OPEN_TS, "next_open_moo", decision_obj.submission_timestamp_ts, decision_obj.submission_timestamp_ts,
        "synthetic", 100000.0, None, None, 1.0, 100000.0, base_dict,
        {asset_str: 100.0 for asset_str in target_dict}, target_dict, delta_dict,
        [VPlanRow(asset_str, base_dict.get(asset_str, 0.0), target_dict[asset_str], delta_dict[asset_str],
            100.0, target_dict[asset_str] * 100.0, "MOO") for asset_str in sorted(target_dict)],
        submission_key_str="daily:1"))
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitted")
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "submitted")
    return store_obj, release_obj, decision_obj, plan_obj, DailyBroker(base_dict)


def reconcile_case(case_tuple, **override_dict):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = case_tuple
    return reconcile_daily_cycle(store_obj, broker_obj, release_obj, decision_obj, broker_obj.as_of_ts,
        vplan_obj=plan_obj, **override_dict)


def open_row(asset_str, ref_str="manual-order", client_id_int=99):
    return {"account_route_str": "DU_TEST", "asset_str": asset_str, "order_ref_str": ref_str,
        "client_id_int": client_id_int, "order_id_int": 999, "perm_id_int": 9999,
        "side_str": "SELL", "amount_float": 100.0, "remaining_amount_float": 100.0, "status_str": "Submitted"}


def test_session_polls_even_when_all_holdings_match_and_no_fills_exist(daily_case):
    store_obj, _, _, plan_obj, broker_obj = daily_case
    broker_obj.position_dict.update(plan_obj.target_share_map)
    result_obj = reconcile_case(daily_case)
    assert result_obj.status_str == "pending"
    assert result_obj.exception_list == [] and broker_obj.sent_request_list == []
    assert store_obj.get_fill_row_dict_list_for_vplan(plan_obj.vplan_id_int) == []


def test_per_asset_exit_and_bil_completion_ignores_unrelated_open_symbols_and_ids(daily_case):
    _, _, _, _, broker_obj = daily_case
    broker_obj.open_row_list = [open_row("AAPL")]
    result_obj = reconcile_case(daily_case)
    assert result_obj.status_str == "pending"
    assert {request_obj.asset_str: request_obj.amount_float for request_obj in broker_obj.sent_request_list} == {"MSFT": -5.0, "BIL": -20.0}
    assert all(request_obj.broker_order_type_str == "MKT" and request_obj.execution_deadline_timestamp_str == CLOSE_TS.isoformat()
        for request_obj in broker_obj.sent_request_list)
    assert broker_obj.position_dict["UNTOUCHED"] == 2.0
    assert broker_obj.refresh_count_int == 4  # first poll, two pre-send checks, final observation


def test_completion_candidates_are_frozen_from_first_poll(daily_case):
    _, _, _, _, broker_obj = daily_case
    broker_obj.position_dict["MSFT"] = 0.0
    def change_positions_fn(daily_snapshot_obj):
        if broker_obj.refresh_count_int == 2:
            broker_obj.position_dict["MSFT"] = 5.0
            return replace(daily_snapshot_obj, broker_snapshot_obj=replace(daily_snapshot_obj.broker_snapshot_obj,
                position_amount_map=dict(broker_obj.position_dict)))
        return daily_snapshot_obj
    broker_obj.snapshot_transform_fn = change_positions_fn
    reconcile_case(daily_case)
    assert [request_obj.asset_str for request_obj in broker_obj.sent_request_list] == ["BIL"]


def test_any_client_open_order_for_symbol_blocks_only_that_symbol(daily_case):
    _, _, _, _, broker_obj = daily_case
    broker_obj.open_row_list = [open_row("MSFT", "unknown-no-local-order-id")]
    reconcile_case(daily_case)
    assert [request_obj.asset_str for request_obj in broker_obj.sent_request_list] == ["BIL"]


def test_funding_dropped_buys_preserve_bil_but_keep_stock_exit(daily_case):
    _, _, _, _, broker_obj = daily_case
    reconcile_case(daily_case, funding_buys_dropped_bool=True)
    assert [(request_obj.asset_str, request_obj.amount_float) for request_obj in broker_obj.sent_request_list] == [("MSFT", -5.0)]
    assert broker_obj.position_dict["BIL"] == 100.0


def test_no_completion_candidate_uses_only_first_poll_snapshot(daily_case):
    _, _, _, plan_obj, broker_obj = daily_case
    broker_obj.position_dict.update(plan_obj.target_share_map)
    reconcile_case(daily_case)
    assert broker_obj.refresh_count_int == 1
    assert broker_obj.sent_request_list == []


def test_daily_submitting_vplan_has_no_intraday_stuck_alert(daily_case):
    store_obj, _, _, plan_obj, _ = daily_case
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitting")
    message_list = _build_stuck_operator_message_spec_list(state_store_obj=store_obj,
        as_of_ts=INTRADAY_TS, reconcile_grace_seconds_int=300)
    assert all(row_dict["phase_action_str"] != "submit_vplan.stuck" for row_dict in message_list)


@pytest.mark.parametrize("target_float,expected_side_str,expected_reason_str", [
    (120.0, "BUY", "bil_buy_withheld_after_funding_buys_dropped"),
    (80.0, "SELL", "bil_sale_withheld_after_funding_buys_dropped"),
])
def test_bil_funding_drop_label_uses_residual_side(daily_case, target_float, expected_side_str, expected_reason_str):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    broker_obj.as_of_ts = CLOSE_TS
    plan_obj = replace(plan_obj, target_share_map={**plan_obj.target_share_map, "BIL": target_float})
    result_obj = reconcile_daily_cycle(store_obj, broker_obj, release_obj, decision_obj, CLOSE_TS,
        vplan_obj=plan_obj, funding_buys_dropped_bool=True)
    bil_row_dict = next(row_dict for row_dict in result_obj.exception_list if row_dict["asset_str"] == "BIL")
    assert (bil_row_dict["side_str"], bil_row_dict["reason_str"]) == (expected_side_str, expected_reason_str)


def test_never_buy_or_complete_nonzero_stock_reduction_or_short_target(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    plan_obj = replace(plan_obj, target_share_map={"AAPL": 10.0, "MSFT": 3.0, "DBC": -10.0, "BIL": 120.0})
    reconcile_daily_cycle(store_obj, broker_obj, release_obj, decision_obj, broker_obj.as_of_ts, vplan_obj=plan_obj)
    assert broker_obj.sent_request_list == []


@pytest.mark.parametrize("no_plan_bool", [False, True])
def test_no_dispatched_vplan_never_triggers_intraday_completion(daily_case, no_plan_bool):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    result_obj = reconcile_daily_cycle(store_obj, broker_obj, release_obj, decision_obj, broker_obj.as_of_ts,
        vplan_obj=None if no_plan_bool else plan_obj, vplan_sent_bool=False)
    assert result_obj.status_str == "pending" and broker_obj.sent_request_list == []


def test_uncertain_completion_attempt_is_durable_and_not_repeated_after_restart(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    broker_obj.send_error_obj = TimeoutError("synthetic timeout after possible send")
    with pytest.raises(TimeoutError):
        reconcile_case(daily_case)
    assert [request_obj.asset_str for request_obj in broker_obj.sent_request_list] == ["BIL"]
    with store_obj._connect() as connection_obj:
        row_obj, = connection_obj.execute("SELECT * FROM daily_completion_request").fetchall()
    assert "synthetic timeout" in row_obj["send_error_str"]
    broker_obj.send_error_obj = None
    restarted_store_obj = LiveStateStore(store_obj.db_path_str)
    reconcile_daily_cycle(restarted_store_obj, broker_obj, release_obj, decision_obj, broker_obj.as_of_ts, vplan_obj=plan_obj)
    assert [request_obj.asset_str for request_obj in broker_obj.sent_request_list] == ["BIL", "MSFT"]


def test_refresh_before_each_sale_uses_actual_remainder(daily_case):
    _, _, _, _, broker_obj = daily_case
    def change_positions_fn(daily_snapshot_obj):
        if broker_obj.refresh_count_int == 3:
            broker_obj.position_dict["MSFT"] = 2.0
            return replace(daily_snapshot_obj, broker_snapshot_obj=replace(daily_snapshot_obj.broker_snapshot_obj,
                position_amount_map=dict(broker_obj.position_dict)))
        return daily_snapshot_obj
    broker_obj.snapshot_transform_fn = change_positions_fn
    reconcile_case(daily_case)
    assert next(request_obj.amount_float for request_obj in broker_obj.sent_request_list if request_obj.asset_str == "MSFT") == -2.0


@pytest.mark.parametrize("holdings_match_bool", [False, True])
def test_close_uses_actual_holdings_without_any_orders_acks_or_fills(daily_case, holdings_match_bool):
    store_obj, _, _, plan_obj, broker_obj = daily_case
    broker_obj.as_of_ts = CLOSE_TS
    if holdings_match_bool:
        broker_obj.position_dict.update(plan_obj.target_share_map)
    result_obj = reconcile_case(daily_case)
    assert result_obj.status_str == ("completed" if holdings_match_bool else "completed_with_exceptions")
    assert result_obj.broker_snapshot_obj.position_amount_map == broker_obj.position_dict
    assert broker_obj.sent_request_list == []
    assert store_obj.get_fill_row_dict_list_for_vplan(plan_obj.vplan_id_int) == []
    if not holdings_match_bool:
        exception_dict = {row_dict["asset_str"]: row_dict for row_dict in result_obj.exception_list}
        assert (exception_dict["AAPL"]["side_str"], exception_dict["AAPL"]["quantity_float"]) == ("BUY", 10.0)
        assert (exception_dict["MSFT"]["side_str"], exception_dict["MSFT"]["quantity_float"]) == ("SELL", 5.0)
        assert "UNTOUCHED" not in exception_dict


def test_close_cancels_only_our_exact_refs_and_waits_for_confirmation(daily_case):
    _, _, _, plan_obj, broker_obj = daily_case
    broker_obj.as_of_ts = CLOSE_TS
    owned_ref_str = build_broker_order_request_list_from_vplan(plan_obj)[0].order_request_key_str
    broker_obj.open_row_list = [open_row("AAPL", owned_ref_str, 31), open_row("MSFT", "manual-order", 12)]
    broker_obj.cancel_confirmed_bool = False
    with pytest.raises(RuntimeError, match="waiting for confirmation"):
        reconcile_case(daily_case)
    broker_obj.cancel_confirmed_bool = True
    assert reconcile_case(daily_case).status_str in DAILY_TERMINAL_STATUS_SET
    assert [row_dict["order_ref_str"] for row_dict in broker_obj.open_row_list] == ["manual-order"]
    assert "manual-order" not in broker_obj.cancel_ref_list[-1]


def test_close_cancels_prior_cycle_owned_orders_too(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    next_decision_obj = store_obj.insert_decision_plan(replace(decision_obj, decision_plan_id_int=None,
        signal_timestamp_ts=decision_obj.signal_timestamp_ts + timedelta(days=1)))
    broker_obj.as_of_ts = CLOSE_TS
    prior_ref_str = build_broker_order_request_list_from_vplan(plan_obj)[0].order_request_key_str
    broker_obj.open_row_list = [open_row("AAPL", prior_ref_str)]
    result_obj = reconcile_daily_cycle(store_obj, broker_obj, release_obj, next_decision_obj, CLOSE_TS,
        vplan_obj=None, vplan_sent_bool=False)
    assert result_obj.status_str == "completed_with_exceptions"
    assert broker_obj.open_row_list == [] and prior_ref_str in broker_obj.cancel_ref_list[0]


def test_closing_old_pending_cycle_never_cancels_newer_intraday_orders(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    next_open_ts = OPEN_TS + timedelta(days=1)
    next_decision_obj = store_obj.insert_decision_plan(replace(decision_obj, decision_plan_id_int=None,
        signal_timestamp_ts=CLOSE_TS, submission_timestamp_ts=next_open_ts - timedelta(minutes=8),
        target_execution_timestamp_ts=next_open_ts))
    next_plan_obj = store_obj.insert_vplan(replace(plan_obj, vplan_id_int=None,
        decision_plan_id_int=next_decision_obj.decision_plan_id_int, signal_timestamp_ts=CLOSE_TS,
        submission_timestamp_ts=next_decision_obj.submission_timestamp_ts, target_execution_timestamp_ts=next_open_ts,
        submission_key_str="daily:2"))
    store_obj.mark_vplan_status(next_plan_obj.vplan_id_int, "submitted")
    store_obj.mark_decision_plan_status(next_decision_obj.decision_plan_id_int, "submitted")
    broker_obj.as_of_ts = next_open_ts + timedelta(minutes=10)
    prior_ref_str = build_broker_order_request_list_from_vplan(plan_obj)[0].order_request_key_str
    newer_request_list = build_broker_order_request_list_from_vplan(next_plan_obj)
    newer_ref_str = newer_request_list[0].order_request_key_str
    completion_obj = replace(next(request_obj for request_obj in newer_request_list if request_obj.asset_str == "MSFT"),
        broker_order_type_str="MKT", order_request_key_str="daily:2:daily-completion:MSFT")
    assert claim_daily_completion_request(store_obj, release_obj, next_decision_obj, next_plan_obj,
        completion_obj, broker_obj.as_of_ts)
    broker_obj.open_row_list = [open_row("AAPL", prior_ref_str), open_row("AAPL", newer_ref_str),
        open_row("MSFT", completion_obj.order_request_key_str)]

    result_obj = reconcile_daily_cycle(store_obj, broker_obj, release_obj, decision_obj, broker_obj.as_of_ts,
        vplan_obj=plan_obj)
    assert result_obj.status_str in DAILY_TERMINAL_STATUS_SET
    assert {row_dict["order_ref_str"] for row_dict in broker_obj.open_row_list} == {
        newer_ref_str, completion_obj.order_request_key_str}
    assert prior_ref_str in broker_obj.cancel_ref_list[0]
    assert newer_ref_str not in broker_obj.cancel_ref_list[0]
    assert completion_obj.order_request_key_str not in broker_obj.cancel_ref_list[0]


def test_missed_vplan_lists_unsized_decision_intentions_without_quote_or_buy(daily_case):
    store_obj, release_obj, decision_obj, _, broker_obj = daily_case
    broker_obj.as_of_ts = CLOSE_TS
    result_obj = reconcile_daily_cycle(store_obj, broker_obj, release_obj, decision_obj, CLOSE_TS, vplan_sent_bool=False)
    assert result_obj.status_str == "completed_with_exceptions"
    exception_dict = {row_dict["asset_str"]: row_dict for row_dict in result_obj.exception_list}
    assert exception_dict["MSFT"]["quantity_float"] == 5.0
    assert exception_dict["BIL"]["quantity_float"] == 20.0
    if release_obj.strategy_import_str == CORE5_STRATEGY_IMPORT_STR:
        assert exception_dict["AAPL"]["quantity_float"] == 10.0
        assert exception_dict["DBC"]["side_str"] == "SELL" and exception_dict["DBC"]["quantity_float"] == 5.0
    else:
        assert exception_dict["AAPL"]["quantity_float"] is None
        assert exception_dict["AAPL"]["target_weight_float"] == 0.1
    assert broker_obj.sent_request_list == []


def test_no_sent_plan_with_known_targets_already_met_closes_without_zero_share_exceptions(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    decision_obj = replace(decision_obj, entry_target_weight_map_dict={}, target_weight_map={},
        snapshot_metadata_dict={"no_order_bool": True})
    broker_obj.position_dict.update({"MSFT": 0.0, "BIL": 80.0})
    broker_obj.as_of_ts = CLOSE_TS
    result_obj = reconcile_daily_cycle(store_obj, broker_obj, release_obj, decision_obj, CLOSE_TS, vplan_sent_bool=False)
    assert result_obj.status_str == "completed"
    assert result_obj.exception_list == [] and broker_obj.sent_request_list == []


def test_no_sent_plan_still_reports_unexpected_actual_holdings(daily_case):
    store_obj, release_obj, decision_obj, _, broker_obj = daily_case
    broker_obj.as_of_ts = CLOSE_TS
    broker_obj.position_dict["FOREIGN"] = 7.0
    result_obj = reconcile_daily_cycle(store_obj, broker_obj, release_obj, decision_obj, CLOSE_TS, vplan_sent_bool=False)
    row_dict = next(row_dict for row_dict in result_obj.exception_list if row_dict["asset_str"] == "FOREIGN")
    assert (row_dict["quantity_float"], row_dict["side_str"]) == (7.0, "SELL")
    assert result_obj.status_str == "completed_with_exceptions"


def test_stale_worker_cannot_claim_after_decision_is_superseded(daily_case):
    store_obj, _, decision_obj, _, broker_obj = daily_case
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "superseded")
    reconcile_case(daily_case)
    assert broker_obj.sent_request_list == []


@pytest.mark.parametrize("fault_str", ["incomplete", "stale", "wrong_account", "nonfinite_holdings", "nonfinite_cash", "missing_symbol", "wrong_order_account"])
def test_invalid_broker_snapshot_never_closes_or_sends(daily_case, fault_str):
    _, _, _, _, broker_obj = daily_case
    broker_obj.as_of_ts = CLOSE_TS
    def damage_fn(daily_snapshot_obj):
        if fault_str == "incomplete":
            return replace(daily_snapshot_obj, complete_bool=False)
        if fault_str == "stale":
            return replace(daily_snapshot_obj, refresh_started_timestamp_ts=CLOSE_TS - timedelta(seconds=1))
        if fault_str in {"missing_symbol", "wrong_order_account"}:
            row_dict = open_row("MSFT")
            row_dict["asset_str" if fault_str == "missing_symbol" else "account_route_str"] = ""
            return replace(daily_snapshot_obj, open_order_row_list=[row_dict])
        override_dict = ({"account_route_str": "OTHER"} if fault_str == "wrong_account" else
            {"position_amount_map": {"MSFT": float("nan")}} if fault_str == "nonfinite_holdings" else {"cash_float": float("nan")})
        return replace(daily_snapshot_obj, broker_snapshot_obj=replace(daily_snapshot_obj.broker_snapshot_obj, **override_dict))
    broker_obj.snapshot_transform_fn = damage_fn
    with pytest.raises(ValueError):
        reconcile_case(daily_case)
    assert broker_obj.sent_request_list == [] and broker_obj.cancel_ref_list == []


def test_unreachable_broker_can_retry_later_but_cannot_close_on_cache(daily_case):
    _, _, _, _, broker_obj = daily_case
    broker_obj.as_of_ts = CLOSE_TS
    broker_obj.error_obj = ConnectionError("synthetic disconnected broker")
    with pytest.raises(ConnectionError):
        reconcile_case(daily_case)
    broker_obj.error_obj = None
    assert reconcile_case(daily_case).status_str == "completed_with_exceptions"


def test_upgrade_migrates_legacy_completion_claim_and_does_not_repeat(daily_case):
    store_obj, _, decision_obj, plan_obj, broker_obj = daily_case
    request_obj = replace(build_broker_order_request_list_from_vplan(plan_obj)[0], asset_str="BIL", amount_float=-20.0,
        broker_order_type_str="MKT", order_request_key_str="daily:1:late:BIL")
    with store_obj._connect() as connection_obj:
        connection_obj.execute("""CREATE TABLE IF NOT EXISTS mr_capsule_execution_request (
            vplan_id_int INTEGER,order_request_key_str TEXT,asset_str TEXT,request_kind_str TEXT,
            request_json_str TEXT,claimed_timestamp_str TEXT,PRIMARY KEY(vplan_id_int,order_request_key_str))""")
        connection_obj.execute("INSERT INTO mr_capsule_execution_request VALUES (?,?,?,?,?,?)",
            (plan_obj.vplan_id_int, request_obj.order_request_key_str, "BIL", "recovery", json.dumps(asdict(request_obj)), INTRADAY_TS.isoformat()))
        ensure_daily_reconcile_schema(connection_obj)
        ensure_daily_reconcile_schema(connection_obj)
        assert connection_obj.execute("SELECT COUNT(*) FROM daily_completion_request WHERE decision_plan_id_int=?", (decision_obj.decision_plan_id_int,)).fetchone()[0] == 1
    reconcile_case(daily_case)
    assert [request_obj.asset_str for request_obj in broker_obj.sent_request_list] == ["MSFT"]
    broker_obj.as_of_ts = CLOSE_TS
    broker_obj.open_row_list = [open_row("BIL", request_obj.order_request_key_str)]
    reconcile_case(daily_case)
    assert request_obj.order_request_key_str in broker_obj.cancel_ref_list[-1]


def test_concurrent_workers_get_only_one_durable_completion_claim(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    request_obj = replace(build_broker_order_request_list_from_vplan(plan_obj)[0], asset_str="MSFT", amount_float=-5.0,
        broker_order_type_str="MKT", order_request_key_str="daily:1:daily-completion:MSFT")
    with ThreadPoolExecutor(max_workers=2) as executor_obj:
        result_list = list(executor_obj.map(lambda _: claim_daily_completion_request(store_obj,
            release_obj, decision_obj, plan_obj, request_obj, INTRADAY_TS), range(2)))
    assert sorted(result_list) == [False, True]


@pytest.mark.parametrize("minute_int,expect_sell_bool", [(59, True), (60, False), (61, False)])
def test_completion_uses_exchange_early_close(daily_case, minute_int, expect_sell_bool):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    early_open_ts = datetime(2026, 11, 27, 9, 30, tzinfo=MARKET_ZONE_OBJ)
    decision_obj = replace(decision_obj, target_execution_timestamp_ts=early_open_ts)
    plan_obj = replace(plan_obj, target_execution_timestamp_ts=early_open_ts)
    broker_obj.as_of_ts = early_open_ts.replace(hour=12, minute=0) + timedelta(minutes=minute_int)
    result_obj = reconcile_daily_cycle(store_obj, broker_obj, release_obj, decision_obj, broker_obj.as_of_ts, vplan_obj=plan_obj)
    assert bool(broker_obj.sent_request_list) == expect_sell_bool
    assert (result_obj.status_str == "pending") == expect_sell_bool


def test_scope_never_includes_ndx_taa_or_other_daily_strategies(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    for strategy_str in ("strategies.dv2.strategy_mr_dv2:DVO2Strategy", "strategies.taa.strategy_taa", "strategies.hpi.strategy_hpi"):
        other_release_obj = replace(release_obj, strategy_import_str=strategy_str)
        assert not is_daily_reconcile_release_bool(other_release_obj)
        with pytest.raises(ValueError, match="limited"):
            reconcile_daily_cycle(store_obj, broker_obj, other_release_obj, decision_obj, CLOSE_TS, vplan_obj=plan_obj)
    assert broker_obj.refresh_count_int == 0
