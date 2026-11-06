"""Month-End Flow Vote live route: the next-session closing auction, frozen Close_T shares, and the guards around them."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha.live import daily_watchdog, runner, strategy_host
from alpha.live import flow_vote_broker as broker_module
from alpha.live import scheduler_utils
from alpha.live.daily_broker import DailyExecutionSnapshot
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.daily_reconcile import MOC_SETTLEMENT_GRACE_SECONDS_INT, daily_settlement_timestamp_ts, is_daily_reconcile_release_bool
from alpha.live.daily_reporting import claim_post_close_report_attempt
from alpha.live.dispatch_state import is_guarded_release_bool
from alpha.live.execution_engine import _broker_order_type_from_execution_policy_str
from alpha.live.flow_vote_adapter import (
    FLOW_VOTE_ASSET_TUPLE,
    build_flow_vote_decision_from_prices,
    validate_flow_vote_release,
    validated_flow_vote_target_share_dict,
)
from alpha.live.guarded_dispatch import MOC_DISPATCH_CUTOFF_MINUTES_INT, dispatch_deadline_ts, moo_dispatch_deadline_ts
from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.models import BrokerSnapshot, PodState
from alpha.live.release_manifest import parse_release_manifest, validate_release_list, validate_release_manifest
from alpha.live.state_store_v2 import LiveStateStore
from data import norgate_loader, norgate_snapshot_store as snapshot_module
from data.flow_vote_snapshot_contract import FLOW_VOTE_PROFILE_STR, flow_vote_price_coverage_dict
from scripts import export_norgate_snapshot as export_module
from strategies.taa_beyond_6040 import strategy_taa_month_end_flow_vote as vote_module
from test_dashboard_v4_daily_cycle import CLOSE_TS as V4_CLOSE_TS, _daily_dict as _v4_daily_dict, _view as _v4_view
from test_live_daily_reconcile import CLOSE_TS, OPEN_TS, daily_case  # noqa: F401  (fixture)

TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "docs" / "live" / "release_templates" / "pod_taa_month_end_flow_vote_daily_moc.yaml.example"
ROUTE_STR = "DU_FLOW_TEST"
MARKET_TZ = scheduler_utils.to_market_timestamp_ts(datetime(2026, 1, 5, 15, tzinfo=UTC), "XNYS").tzinfo


def _release(**change_dict):
    return replace(parse_release_manifest(str(TEMPLATE_PATH)), mode_str="paper", enabled_bool=True, account_route_str=ROUTE_STR, **change_dict)


def _et(year_int, month_int, day_int, hour_int, minute_int=0):
    return scheduler_utils.to_market_timestamp_ts(datetime(year_int, month_int, day_int, hour_int, minute_int, tzinfo=MARKET_TZ), "XNYS")


# ---------------------------------------------------------------- timing
@pytest.mark.parametrize("signal_str,next_close_tuple", [("2026-11-19", (2026, 11, 20, 16)), ("2026-11-25", (2026, 11, 27, 13))])
def test_next_close_moc_submits_before_the_next_sessions_close_and_stops_fifteen_minutes_before_it(signal_str, next_close_tuple):
    release_obj = _release()
    signal_ts = pd.Timestamp(signal_str)
    close_ts = _et(*next_close_tuple)
    submission_ts = scheduler_utils.build_submission_timestamp_ts(signal_ts, release_obj)
    target_ts = scheduler_utils.build_target_execution_timestamp_ts(signal_ts, release_obj)
    assert target_ts == close_ts  # the NEXT session's close, a 13:00 half day included
    assert submission_ts == close_ts - timedelta(minutes=20)
    plan_obj = SimpleNamespace(execution_policy_str="next_close_moc", target_execution_timestamp_ts=target_ts)
    assert dispatch_deadline_ts(plan_obj) == close_ts - timedelta(minutes=MOC_DISPATCH_CUTOFF_MINUTES_INT)
    assert submission_ts < dispatch_deadline_ts(plan_obj) < target_ts
    # The signal session's own close is never the target.
    assert target_ts > scheduler_utils.get_session_close_timestamp_ts(signal_ts, "XNYS")


def test_open_auction_pods_keep_their_deadline_and_order_type():
    target_ts = _et(2026, 11, 20, 9, 30)
    for policy_str in ("next_open_moo", "next_month_first_open"):
        plan_obj = SimpleNamespace(execution_policy_str=policy_str, target_execution_timestamp_ts=target_ts)
        assert dispatch_deadline_ts(plan_obj) == moo_dispatch_deadline_ts(plan_obj) == target_ts - timedelta(minutes=2)
    assert _broker_order_type_from_execution_policy_str("next_open_moo") == "MOO"
    assert _broker_order_type_from_execution_policy_str("next_close_moc") == "MOC"
    assert _broker_order_type_from_execution_policy_str("same_day_moc") == "MOC"


def test_a_closing_cycle_settles_after_the_auction_prints_and_an_opening_one_at_the_close():
    target_ts = _et(2026, 11, 20, 16)
    moc_plan = SimpleNamespace(execution_policy_str="next_close_moc", target_execution_timestamp_ts=target_ts)
    moo_plan = SimpleNamespace(execution_policy_str="next_open_moo", target_execution_timestamp_ts=_et(2026, 11, 20, 9, 30))
    assert daily_settlement_timestamp_ts(moc_plan, "XNYS") == target_ts + timedelta(seconds=MOC_SETTLEMENT_GRACE_SECONDS_INT)
    assert daily_settlement_timestamp_ts(moo_plan, "XNYS") == target_ts


def test_the_post_close_report_waits_for_the_closing_prints(tmp_path):
    store_obj = LiveStateStore(str(tmp_path / "state.db"))
    release_obj = _release()
    close_ts = _et(2026, 11, 20, 16)
    decision_obj = SimpleNamespace(decision_plan_id_int=7, execution_policy_str="next_close_moc", target_execution_timestamp_ts=close_ts)
    grace_ts = close_ts + timedelta(seconds=MOC_SETTLEMENT_GRACE_SECONDS_INT)
    assert not claim_post_close_report_attempt(store_obj, release_obj, decision_obj, close_ts)
    assert not claim_post_close_report_attempt(store_obj, release_obj, decision_obj, grace_ts - timedelta(seconds=1))
    assert claim_post_close_report_attempt(store_obj, release_obj, decision_obj, grace_ts)


def test_socket_client_accepts_auction_and_recovery_send_deadlines_but_not_limits(monkeypatch):
    socket_obj = IBKRSocketClient()
    reached_list = []

    @contextmanager
    def stop_at_connect():
        reached_list.append(True)
        raise RuntimeError("stop before any broker call")
        yield

    monkeypatch.setattr(socket_obj, "connect", stop_at_connect)
    monkeypatch.setattr(socket_obj, "_guarded_dispatch_connection", stop_at_connect, raising=False)
    deadline_str = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    from alpha.live.models import BrokerOrderRequest

    def request(order_type_str):
        return BrokerOrderRequest(release_id_str="r", pod_id_str="p", account_route_str=ROUTE_STR, submission_key_str="k",
            order_request_key_str=f"k:{order_type_str}", asset_str="TLT", broker_order_type_str=order_type_str, order_class_str="MarketOrder",
            unit_str="shares", amount_float=-10.0, target_bool=False, trade_id_int=None, sizing_reference_price_float=90.0,
            portfolio_value_float=1e5, decision_plan_id_int=1, vplan_id_int=1, submission_deadline_timestamp_str=deadline_str)

    # v4 recovery adds a bounded MKT send deadline; auction order behavior remains unchanged.
    for order_type_str in ("MOC", "MKT"):
        reached_list.clear()
        with pytest.raises(RuntimeError, match="stop before any broker call"):
            socket_obj.submit_order_request_list(account_route_str=ROUTE_STR, broker_order_request_list=[request(order_type_str)], submitted_timestamp_ts=datetime.now(UTC))
        assert reached_list
    # The guarded path wraps the refusal so the send claim is kept for reconciliation; nothing reached the broker.
    reached_list.clear()
    with pytest.raises(Exception, match="an MOO, MOC or MKT order"):
        socket_obj.submit_order_request_list(account_route_str=ROUTE_STR, broker_order_request_list=[request("LMT")], submitted_timestamp_ts=datetime.now(UTC))
    assert not reached_list


# ---------------------------------------------------------------- release rules
def test_template_is_a_disabled_dedicated_daily_moc_pod():
    template_obj = parse_release_manifest(str(TEMPLATE_PATH))
    assert (template_obj.enabled_bool, template_obj.execution_policy_str, template_obj.data_profile_str) == (False, "next_close_moc", FLOW_VOTE_PROFILE_STR)
    validate_release_manifest(template_obj)
    release_obj = _release()
    assert is_daily_reconcile_release_bool(release_obj) and is_guarded_release_bool(release_obj)


@pytest.mark.parametrize("change_dict", [
    {"execution_policy_str": "next_open_moo"}, {"signal_clock_str": "month_end_snapshot_ready"}, {"data_profile_str": "norgate_eod_core5"},
    {"pod_budget_fraction_float": 0.5}, {"params_dict": {"early_rule_str": "flat"}},
])
def test_the_frozen_route_refuses_any_other_shape(change_dict):
    with pytest.raises(ValueError):
        validate_flow_vote_release(_release(**change_dict))


def test_the_policy_and_the_profile_are_reserved_for_the_flow_vote():
    core5_obj = parse_release_manifest(str(TEMPLATE_PATH.with_name("pod_taa_adaptive_macro_core5_daily_moo.yaml.example")))
    with pytest.raises(ValueError, match="reserved"):
        validate_release_manifest(replace(core5_obj, strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy", data_profile_str="norgate_eod_sp500_pit",
                                          execution_policy_str="next_close_moc"))
    with pytest.raises(ValueError, match="reserved"):
        validate_release_manifest(replace(core5_obj, strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy", data_profile_str=FLOW_VOTE_PROFILE_STR))


@pytest.mark.parametrize("flow_vote_first_bool", [True, False])
def test_two_enabled_pods_cannot_share_the_flow_vote_account(flow_vote_first_bool):
    flow_obj = _release()
    other_obj = replace(parse_release_manifest(str(TEMPLATE_PATH.with_name("pod_dv2_daily_moo.yaml.example"))),
        mode_str="paper", enabled_bool=True, account_route_str=ROUTE_STR, broker_client_id_int=57)
    with pytest.raises(ValueError, match="Flow Vote requires one dedicated account"):
        validate_release_list([flow_obj, other_obj] if flow_vote_first_bool else [other_obj, flow_obj])
    with pytest.raises(ValueError, match="Flow Vote requires one dedicated account"):
        validate_release_list([flow_obj, replace(flow_obj, release_id_str="other.v1", pod_id_str="pod_other", broker_client_id_int=57)])


def test_the_flow_vote_refuses_incubation():
    # The virtual broker would need the next session's Norgate close, which only a later snapshot holds.
    with pytest.raises(ValueError, match="paper or live"):
        validate_flow_vote_release(replace(_release(), mode_str="incubation"))


# ---------------------------------------------------------------- the decision
def _synthetic_frame(end_str="2012-12-31", seed_int=5):
    import exchange_calendars as xcals

    session_idx = pd.DatetimeIndex(xcals.get_calendar("XNYS", start="2002-01-01").sessions_in_range("2002-07-26", end_str))
    rng_obj = np.random.default_rng(seed_int)
    close_df = pd.DataFrame({s: 100.0 * np.cumprod(1.0 + rng_obj.normal(0.0002, 0.01, len(session_idx))) for s in ("SPY", "TLT", "IEF")}, index=session_idx)
    field_dict = {}
    for asset_str in ("SPY", "TLT"):
        for field_str in ("Open", "High", "Low", "Close"):
            field_dict[(asset_str, field_str)] = close_df[asset_str]
        field_dict[(asset_str, "Dividend")] = pd.Series(0.0, index=session_idx)
    for asset_str in ("SPY", "IEF", "TLT"):
        field_dict[(f"FLOW_TR_{asset_str}", "Close")] = close_df[asset_str]
    for field_str in ("Open", "High", "Low", "Close"):
        field_dict[("$SPX", field_str)] = close_df["SPY"]
    pricing_data_df = pd.DataFrame(field_dict, index=session_idx)
    pricing_data_df.attrs.update({"norgate_adjustment_by_symbol_dict": {"SPY": "CAPITALSPECIAL", "TLT": "CAPITALSPECIAL", "$SPX": "TOTALRETURN",
        "FLOW_TR_SPY": "TOTALRETURN", "FLOW_TR_IEF": "TOTALRETURN", "FLOW_TR_TLT": "TOTALRETURN"},
        "benchmark_data_symbol_dict": {"$SPX": "$SPXTR"}, "price_padding_policy_str": "NONE"})
    return pricing_data_df


@pytest.fixture(scope="module")
def decision_case():
    pricing_data_df = _synthetic_frame()
    strategy_obj = vote_module.MonthEndVoteFlowStrategy(vote_module.DEFAULT_CONFIG)
    strategy_obj.compute_signals(pricing_data_df)
    schedule_df = strategy_obj.order_schedule_df
    session_list = list(pricing_data_df.index)
    # A fill eve whose book is non-flat, after the rank has its 24 prior months and inside the live calendar's 20-year reach.
    fill_ts = next(t for t in schedule_df.index if t >= pd.Timestamp("2011-01-01") and (schedule_df.loc[t, "SPY"] != 0 or schedule_df.loc[t, "TLT"] != 0))
    eve_ts = session_list[session_list.index(fill_ts) - 1]
    hold_ts = session_list[session_list.index(fill_ts) + 1]
    return pricing_data_df, schedule_df, eve_ts, fill_ts, hold_ts


def _pod_state(release_obj, signal_ts, position_dict, cash_float, receipt_dict, **change_dict):
    close_ts = scheduler_utils.get_session_close_timestamp_ts(signal_ts, "XNYS")
    return replace(PodState(pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str,
        position_amount_map=position_dict, cash_float=cash_float, total_value_float=cash_float,
        strategy_state_dict={"flow_vote_state_version_int": 1, "last_signal_date_str": "", "flow_vote_execution_receipt_dict": receipt_dict},
        updated_timestamp_ts=close_ts + timedelta(minutes=12), snapshot_stage_str="eod", snapshot_source_str="broker"), **change_dict)


def _decide(release_obj, signal_ts, pod_state_obj, pricing_data_df, **metadata_change_dict):
    as_of_ts = scheduler_utils.get_session_close_timestamp_ts(signal_ts, "XNYS") + timedelta(minutes=30)
    metadata_dict = {"norgate_snapshot_date_str": signal_ts.date().isoformat(), "norgate_data_profile_str": FLOW_VOTE_PROFILE_STR,
                     "norgate_manifest_hash_str": "test", **metadata_change_dict}
    return build_flow_vote_decision_from_prices(release_obj, as_of_ts, pod_state_obj, pricing_data_df, metadata_dict)


def _receipt(schedule_df, before_ts, share_dict):
    event_ts = schedule_df.index[schedule_df.index <= before_ts][-1]
    return {"last_applied_event_fill_date_str": event_ts.date().isoformat(),
            "last_applied_target_weight_map_dict": {}, "last_applied_target_share_map_dict": {a: float(share_dict.get(a, 0.0)) for a in FLOW_VOTE_ASSET_TUPLE}}


def test_a_fill_eve_freezes_trunc_nav_times_weight_over_close_t(decision_case):
    pricing_data_df, schedule_df, eve_ts, fill_ts, _ = decision_case
    release_obj = _release()
    plan_obj = _decide(release_obj, eve_ts, _pod_state(release_obj, eve_ts, {}, 100_000.0, _receipt(schedule_df, eve_ts, {})), pricing_data_df)
    metadata_dict = plan_obj.snapshot_metadata_dict
    weight_dict = {a: float(schedule_df.loc[fill_ts, a]) for a in FLOW_VOTE_ASSET_TUPLE}
    close_dict = {a: float(pricing_data_df.at[eve_ts, (a, "Close")]) for a in FLOW_VOTE_ASSET_TUPLE}
    assert metadata_dict["rebalance_bool"] and metadata_dict["flow_vote_fill_next_session_bool"]
    assert metadata_dict["fixed_target_share_map_dict"] == {a: float(int(100_000.0 * weight_dict[a] / close_dict[a])) for a in FLOW_VOTE_ASSET_TUPLE}
    assert plan_obj.execution_policy_str == "next_close_moc"
    assert plan_obj.target_execution_timestamp_ts == scheduler_utils.get_session_close_timestamp_ts(fill_ts, "XNYS")
    # The engine path re-derives the same shares and refuses any other quantity.
    assert validated_flow_vote_target_share_dict(plan_obj, release_obj, {}) == metadata_dict["fixed_target_share_map_dict"]
    tampered_obj = replace(plan_obj, snapshot_metadata_dict={**metadata_dict, "fixed_target_share_map_dict": {"SPY": 1.0, "TLT": 1.0}})
    with pytest.raises(ValueError):
        validated_flow_vote_target_share_dict(tampered_obj, release_obj, {})


def test_a_session_after_an_applied_fill_holds(decision_case):
    pricing_data_df, schedule_df, _, fill_ts, hold_ts = decision_case
    release_obj = _release()
    held_dict = {"SPY": 100.0} if float(schedule_df.loc[fill_ts, "SPY"]) > 0 else {"TLT": 100.0}
    plan_obj = _decide(release_obj, hold_ts, _pod_state(release_obj, hold_ts, held_dict, 1_000.0, _receipt(schedule_df, hold_ts, held_dict)), pricing_data_df)
    assert not plan_obj.snapshot_metadata_dict["rebalance_bool"] and plan_obj.snapshot_metadata_dict["no_order_bool"]
    assert plan_obj.snapshot_metadata_dict["fixed_target_share_map_dict"] == {a: float(held_dict.get(a, 0.0)) for a in FLOW_VOTE_ASSET_TUPLE}


def test_a_missed_fill_is_caught_up_at_the_next_close(decision_case):
    pricing_data_df, schedule_df, eve_ts, fill_ts, hold_ts = decision_case
    release_obj = _release()
    stale_receipt_dict = _receipt(schedule_df, eve_ts, {})  # the event before the missed one, still flat
    plan_obj = _decide(release_obj, hold_ts, _pod_state(release_obj, hold_ts, {}, 100_000.0, stale_receipt_dict), pricing_data_df)
    metadata_dict = plan_obj.snapshot_metadata_dict
    assert metadata_dict["rebalance_bool"] and metadata_dict["flow_vote_catch_up_bool"]
    assert metadata_dict["flow_vote_standing_event_dict"]["event_fill_date_str"] == fill_ts.date().isoformat()


def test_a_new_flat_pod_waits_for_the_next_scheduled_fill_and_enters_on_its_eve(decision_case):
    pricing_data_df, _, eve_ts, _, hold_ts = decision_case
    release_obj = _release()
    wait_obj = _decide(release_obj, hold_ts, _pod_state(release_obj, hold_ts, {}, 100_000.0, {}), pricing_data_df)
    assert wait_obj.snapshot_metadata_dict["initialization_bool"] and wait_obj.snapshot_metadata_dict["no_order_bool"]
    assert wait_obj.snapshot_metadata_dict["flow_vote_candidate_execution_receipt_dict"]["last_applied_target_share_map_dict"] == {"SPY": 0.0, "TLT": 0.0}
    enter_obj = _decide(release_obj, eve_ts, _pod_state(release_obj, eve_ts, {}, 100_000.0, {}), pricing_data_df)
    assert enter_obj.snapshot_metadata_dict["rebalance_bool"]


@pytest.mark.parametrize("case_str", ["holding_without_receipt", "unsupported_asset", "spy_short", "fractional", "stale_eod", "default_state",
                                      "wrong_snapshot_date", "wrong_profile", "disabled_live"])
def test_the_decision_fails_closed(decision_case, case_str):
    pricing_data_df, schedule_df, eve_ts, _, _ = decision_case
    release_obj = _release()
    pod_state_obj = _pod_state(release_obj, eve_ts, {}, 100_000.0, _receipt(schedule_df, eve_ts, {}))
    metadata_change_dict = {}
    if case_str == "holding_without_receipt":
        pod_state_obj = _pod_state(release_obj, eve_ts, {"SPY": 10.0}, 100_000.0, {})
    elif case_str == "unsupported_asset":
        pod_state_obj = replace(pod_state_obj, position_amount_map={"QQQ": 10.0})
    elif case_str == "spy_short":
        pod_state_obj = replace(pod_state_obj, position_amount_map={"SPY": -10.0})
    elif case_str == "fractional":
        pod_state_obj = replace(pod_state_obj, position_amount_map={"TLT": 1.5})
    elif case_str == "stale_eod":
        pod_state_obj = replace(pod_state_obj, updated_timestamp_ts=pod_state_obj.updated_timestamp_ts - timedelta(days=1))
    elif case_str == "default_state":
        pod_state_obj = replace(pod_state_obj, snapshot_stage_str="unknown", snapshot_source_str="pod_state")
    elif case_str == "wrong_snapshot_date":
        metadata_change_dict = {"norgate_snapshot_date_str": "2001-01-02"}
    elif case_str == "wrong_profile":
        metadata_change_dict = {"norgate_data_profile_str": "norgate_eod_core5"}
    elif case_str == "disabled_live":
        release_obj = replace(release_obj, mode_str="live", enabled_bool=False)
    with pytest.raises(ValueError):
        _decide(release_obj, eve_ts, pod_state_obj, pricing_data_df, **metadata_change_dict)


def test_prices_after_close_t_cannot_change_the_decision(decision_case):
    pricing_data_df, schedule_df, eve_ts, _, _ = decision_case
    release_obj = _release()
    pod_state_obj = _pod_state(release_obj, eve_ts, {}, 100_000.0, _receipt(schedule_df, eve_ts, {}))
    shocked_df = pricing_data_df.copy()
    shocked_df.loc[shocked_df.index > eve_ts] *= 3.0
    shocked_df.attrs = dict(pricing_data_df.attrs)
    base_obj = _decide(release_obj, eve_ts, pod_state_obj, pricing_data_df)
    shocked_obj = _decide(release_obj, eve_ts, pod_state_obj, shocked_df)
    cut_obj = _decide(release_obj, eve_ts, pod_state_obj, pricing_data_df.loc[:eve_ts].copy())
    assert (base_obj.snapshot_metadata_dict["fixed_target_share_map_dict"] == shocked_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
            == cut_obj.snapshot_metadata_dict["fixed_target_share_map_dict"])


# ---------------------------------------------------------------- receipts and the strict account read
@pytest.mark.parametrize("achieved_bool", [False, True])
def test_the_daily_finalizer_commits_the_flow_vote_receipt_only_when_the_book_is_held(daily_case, achieved_bool):  # noqa: F811
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    prior_dict, candidate_dict = {"last_applied_event_fill_date_str": "prior"}, {"last_applied_event_fill_date_str": "new"}
    target_dict = dict(plan_obj.target_share_map)
    metadata_dict = {**decision_obj.snapshot_metadata_dict, "fixed_target_share_map_dict": target_dict,
        "base_strategy_state_dict": {"flow_vote_execution_receipt_dict": prior_dict}, "flow_vote_candidate_execution_receipt_dict": candidate_dict}
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str=? WHERE decision_plan_id_int=?", (json.dumps(metadata_dict), decision_obj.decision_plan_id_int))
    actual_dict = target_dict if achieved_bool else {"MSFT": 3.0}
    exception_list = [] if achieved_bool else [{"asset_str": "MSFT", "quantity_float": 3.0, "side_str": "SELL", "reason_str": "unfilled"}]
    store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        BrokerSnapshot(release_obj.account_route_str, CLOSE_TS, cash_float=500.0, total_value_float=1e5, net_liq_float=1e5, position_amount_map=actual_dict),
        "completed" if achieved_bool else "completed_with_exceptions", exception_list, CLOSE_TS)
    committed_dict = store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict
    assert committed_dict["flow_vote_execution_receipt_dict"] == (candidate_dict if achieved_bool else prior_dict)
    assert "core5_execution_receipt_dict" not in committed_dict


def _account_value_obj(tag_str, value_obj, currency_str="USD", account_str=ROUTE_STR):
    return SimpleNamespace(tag=tag_str, value=value_obj, currency=currency_str, account=account_str, modelCode="")


def _position_obj(asset_str, shares_float):
    return SimpleNamespace(account=ROUTE_STR, position=shares_float,
        contract=SimpleNamespace(symbol=asset_str, secType="STK", currency="USD", multiplier=""))


class _FakeBroker:
    def __init__(self):
        self.wrapper = SimpleNamespace(_futures={})
        self.summary_list = [_account_value_obj("TotalCashValue", "52000"), _account_value_obj("NetLiquidation", "100000")]
        self.position_list = [_position_obj("SPY", 50), _position_obj("TLT", -300)]

    def managedAccounts(self):
        return [ROUTE_STR]

    def accountSummary(self, account):
        return self.summary_list

    def positions(self, account):
        return self.position_list

    def reqAllOpenOrders(self):
        return [SimpleNamespace(order=SimpleNamespace(orderId=7, account=ROUTE_STR)), SimpleNamespace(order=SimpleNamespace(orderId=8, account="OTHER"))]

    def placeOrder(self, *argument_tuple, **keyword_dict):
        raise AssertionError("an account read must never send an order")


@pytest.fixture
def broker_case(monkeypatch):
    broker_obj, socket_obj = _FakeBroker(), IBKRSocketClient()

    @contextmanager
    def connect_fake():
        yield broker_obj

    monkeypatch.setattr(socket_obj, "connect", connect_fake)
    monkeypatch.setattr(socket_obj, "daily_connection", connect_fake)
    return socket_obj, broker_obj


def test_the_strict_read_keeps_signed_shares_cash_and_every_clients_open_orders(broker_case):
    socket_obj, _ = broker_case
    snapshot_obj = broker_module.get_flow_vote_account_snapshot(socket_obj, ROUTE_STR)
    assert snapshot_obj.position_amount_map == {"SPY": 50.0, "TLT": -300.0}
    assert (snapshot_obj.cash_float, snapshot_obj.net_liq_float, snapshot_obj.open_order_id_list) == (52000.0, 100000.0, ["7"])


POSITION_REFUSAL_STR = "unique SPY/TLT USD stock positions in whole shares; only TLT may be short"


@pytest.mark.parametrize("case_str,match_str", [
    ("spy_short", POSITION_REFUSAL_STR), ("other_asset", POSITION_REFUSAL_STR), ("fractional", POSITION_REFUSAL_STR),
    ("other_currency", POSITION_REFUSAL_STR), ("not_a_stock", POSITION_REFUSAL_STR), ("duplicate", POSITION_REFUSAL_STR),
    ("positions_pending", "positions download is incomplete"), ("not_visible", "not visible"),
    ("no_nav", "unambiguous USD NetLiquidation"), ("nav_not_positive", "positive USD NetLiquidation"),
    ("two_cash_rows", "unambiguous USD TotalCashValue"),
])
def test_the_strict_read_fails_closed(broker_case, case_str, match_str):
    socket_obj, broker_obj = broker_case
    if case_str == "spy_short":
        broker_obj.position_list = [_position_obj("SPY", -1)]
    elif case_str == "other_asset":
        broker_obj.position_list = [_position_obj("QQQ", 1)]
    elif case_str == "fractional":
        broker_obj.position_list = [_position_obj("TLT", 1.5)]
    elif case_str in ("other_currency", "not_a_stock"):
        position_obj = _position_obj("TLT", 10)
        position_obj.contract.currency, position_obj.contract.secType = ("EUR", "STK") if case_str == "other_currency" else ("USD", "OPT")
        broker_obj.position_list = [position_obj]
    elif case_str == "duplicate":
        broker_obj.position_list = [_position_obj("SPY", 1), _position_obj("SPY", 2)]
    elif case_str == "positions_pending":
        broker_obj.wrapper._futures["positions"] = object()
    elif case_str == "not_visible":
        broker_obj.managedAccounts = lambda: ["OTHER"]
    elif case_str == "no_nav":
        broker_obj.summary_list = [_account_value_obj("TotalCashValue", "1")]
    elif case_str == "nav_not_positive":
        broker_obj.summary_list = [_account_value_obj("TotalCashValue", "1"), _account_value_obj("NetLiquidation", "0")]
    elif case_str == "two_cash_rows":
        broker_obj.summary_list.append(_account_value_obj("TotalCashValue", "2"))
    with pytest.raises(ValueError, match=match_str):
        broker_module.get_flow_vote_account_snapshot(socket_obj, ROUTE_STR)


# ---------------------------------------------------------------- the first fill
def test_a_failed_first_fill_is_caught_up_at_the_next_close(decision_case):
    pricing_data_df, _, eve_ts, fill_ts, _ = decision_case
    release_obj = _release()
    first_obj = _decide(release_obj, eve_ts, _pod_state(release_obj, eve_ts, {}, 100_000.0, {}), pricing_data_df)
    assert first_obj.snapshot_metadata_dict["rebalance_bool"]
    # Neither leg filled: the finalizer commits the decision's state but keeps the empty receipt.
    failed_state_dict = {**first_obj.strategy_state_dict, "flow_vote_execution_receipt_dict": {}}
    retry_obj = _decide(release_obj, fill_ts, replace(_pod_state(release_obj, fill_ts, {}, 100_000.0, {}),
        strategy_state_dict=failed_state_dict), pricing_data_df)
    metadata_dict = retry_obj.snapshot_metadata_dict
    assert metadata_dict["rebalance_bool"] and metadata_dict["flow_vote_catch_up_bool"] and not metadata_dict["initialization_bool"]
    assert metadata_dict["fixed_target_share_map_dict"] != {"SPY": 0.0, "TLT": 0.0}


def test_a_partial_first_fill_is_caught_up_instead_of_blocking(decision_case):
    pricing_data_df, _, eve_ts, fill_ts, _ = decision_case
    release_obj = _release()
    first_obj = _decide(release_obj, eve_ts, _pod_state(release_obj, eve_ts, {}, 100_000.0, {}), pricing_data_df)
    target_dict = first_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    filled_asset_str = next(asset_str for asset_str in FLOW_VOTE_ASSET_TUPLE if target_dict[asset_str] > 0)
    partial_dict = {filled_asset_str: target_dict[filled_asset_str]}
    cash_float = 100_000.0 - partial_dict[filled_asset_str] * float(pricing_data_df.at[fill_ts, (filled_asset_str, "Close")])
    failed_state_dict = {**first_obj.strategy_state_dict, "flow_vote_execution_receipt_dict": {}}
    retry_obj = _decide(release_obj, fill_ts, replace(_pod_state(release_obj, fill_ts, partial_dict, cash_float, {}),
        strategy_state_dict=failed_state_dict), pricing_data_df)
    assert retry_obj.snapshot_metadata_dict["rebalance_bool"] and retry_obj.snapshot_metadata_dict["flow_vote_catch_up_bool"]
    # A pod that never decided and already holds shares still needs a reviewed start.
    with pytest.raises(ValueError, match="without an execution receipt"):
        _decide(release_obj, fill_ts, _pod_state(release_obj, fill_ts, partial_dict, cash_float, {}), pricing_data_df)


# ---------------------------------------------------------------- the pair leg and the runner cycle (fake account only)
@pytest.fixture(scope="module")
def pair_case(decision_case):
    """The first next-month pair from 2011: TLT goes from the final week's long to a short at the next close."""
    pricing_data_df, schedule_df, _, _, _ = decision_case
    session_list = list(pricing_data_df.index)
    fill_ts = next(t for t in schedule_df.index if t >= pd.Timestamp("2011-01-01")
                   and (float(schedule_df.loc[t, "SPY"]), float(schedule_df.loc[t, "TLT"])) == (0.5, -0.5))
    eve_ts = session_list[session_list.index(fill_ts) - 1]
    assert float(schedule_df.loc[schedule_df.index <= eve_ts, "TLT"].iloc[-1]) > 0.0
    return pricing_data_df, schedule_df, eve_ts, fill_ts


@pytest.mark.parametrize("held_dict,cash_float", [({"TLT": 800.0}, 1_000.0), ({"SPY": 50.0, "TLT": -300.0}, 150_000.0)])
def test_the_pair_is_sized_from_signed_nav_and_the_short_truncates_toward_zero(pair_case, held_dict, cash_float):
    pricing_data_df, schedule_df, eve_ts, _ = pair_case
    release_obj = _release()
    plan_obj = _decide(release_obj, eve_ts, _pod_state(release_obj, eve_ts, held_dict, cash_float, _receipt(schedule_df, eve_ts, held_dict)), pricing_data_df)
    metadata_dict = plan_obj.snapshot_metadata_dict
    close_dict = {a: float(pricing_data_df.at[eve_ts, (a, "Close")]) for a in FLOW_VOTE_ASSET_TUPLE}
    nav_float = cash_float + sum(shares_float * close_dict[a] for a, shares_float in held_dict.items())
    assert metadata_dict["sizing_close_nav_float"] == pytest.approx(nav_float, rel=1e-12)
    short_float = -0.5 * nav_float / close_dict["TLT"]
    assert math.floor(short_float) != math.ceil(short_float)
    assert metadata_dict["fixed_target_share_map_dict"] == {"SPY": float(math.floor(0.5 * nav_float / close_dict["SPY"])),
                                                           "TLT": float(math.ceil(short_float))}


def _prepared_pair_cycle(tmp_path, monkeypatch, pair_case):
    pricing_data_df, schedule_df, eve_ts, fill_ts = pair_case
    held_dict, cash_float = {"TLT": 800.0}, 1_000.0
    release_obj = _release()
    state_obj = _pod_state(release_obj, eve_ts, held_dict, cash_float, _receipt(schedule_df, eve_ts, held_dict))
    store_obj = LiveStateStore(str(tmp_path / "flow_vote.sqlite3"))
    store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(state_obj)
    decision_obj = store_obj.insert_decision_plan(_decide(release_obj, eve_ts, state_obj, pricing_data_df))
    broker_obj = StubBrokerAdapter()
    broker_obj.seed_account_snapshot(ROUTE_STR, cash_float, 100_000.0, held_dict, decision_obj.submission_timestamp_ts, session_mode_str="paper")

    def fresh_daily_fn(account_route_str):
        snapshot_obj = broker_obj.get_account_snapshot(account_route_str)
        return DailyExecutionSnapshot(snapshot_obj, [], snapshot_obj.snapshot_timestamp_ts, snapshot_obj.snapshot_timestamp_ts)

    monkeypatch.setattr(broker_obj, "get_daily_execution_snapshot", fresh_daily_fn)
    option_dict = {"log_path_str": str(tmp_path / "ops.log"), "trace_enabled_bool": False}
    assert runner.build_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts, "paper", **option_dict)["created_vplan_count_int"] == 1
    return SimpleNamespace(release_obj=release_obj, store_obj=store_obj, broker_obj=broker_obj, decision_obj=decision_obj,
        vplan_obj=store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str), option_dict=option_dict,
        pricing_data_df=pricing_data_df, fill_ts=fill_ts)


def test_a_pair_cycle_sends_signed_moc_legs_settles_after_the_prints_and_then_holds(pair_case, tmp_path, monkeypatch):
    case_obj = _prepared_pair_cycle(tmp_path, monkeypatch, pair_case)
    release_obj, store_obj, broker_obj, decision_obj, vplan_obj = (case_obj.release_obj, case_obj.store_obj, case_obj.broker_obj,
        case_obj.decision_obj, case_obj.vplan_obj)
    target_dict = decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    assert target_dict["TLT"] < 0.0 < target_dict["SPY"]
    phase_list = []
    original_read_fn, original_claim_fn = broker_obj.get_flow_vote_account_snapshot, store_obj.claim_vplan_for_submission

    def read_fn(account_route_str, *argument_tuple, **argument_dict):
        phase_list.append("account_read")
        assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "ready"
        return original_read_fn(account_route_str, *argument_tuple, **argument_dict)

    def claim_fn(vplan_id_int):
        phase_list.append("claim")
        return original_claim_fn(vplan_id_int)

    monkeypatch.setattr(broker_obj, "get_flow_vote_account_snapshot", read_fn)
    monkeypatch.setattr(store_obj, "claim_vplan_for_submission", claim_fn)
    prior_state_dict = dict(store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict)
    assert runner.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **case_obj.option_dict)["submitted_vplan_count_int"] == 1
    assert phase_list == ["account_read", "claim"]
    close_ts = decision_obj.target_execution_timestamp_ts
    request_list = broker_obj.submitted_order_request_list
    assert all(request_obj.broker_order_type_str == "MOC" and request_obj.unit_str == "shares" and request_obj.target_bool is False
        and request_obj.submission_deadline_timestamp_str == (close_ts - timedelta(minutes=15)).isoformat() for request_obj in request_list)
    # The flip keeps the research's close-old and open-new legs: sell the 800 long, then sell the new short.
    assert [request_obj.amount_float for request_obj in request_list if request_obj.asset_str == "TLT"] == [-800.0, target_dict["TLT"]]
    assert [request_obj.amount_float for request_obj in request_list if request_obj.asset_str == "SPY"] == [target_dict["SPY"]]
    for minute_int, completed_int in ((1, 0), (5, 1)):
        as_of_ts = close_ts + timedelta(minutes=minute_int)
        broker_obj._snapshot_map[ROUTE_STR] = replace(broker_obj.get_account_snapshot(ROUTE_STR), snapshot_timestamp_ts=as_of_ts)
        assert runner.post_execution_reconcile(store_obj, broker_obj, as_of_ts, "paper", **case_obj.option_dict)["completed_vplan_count_int"] == completed_int
        if completed_int == 0:
            # The MOC still works at the bell: nothing settles before the prints.
            assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == prior_state_dict
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str == "completed"
    committed_dict = store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict
    assert committed_dict["flow_vote_execution_receipt_dict"] == decision_obj.snapshot_metadata_dict["flow_vote_candidate_execution_receipt_dict"]
    # The next session reads its own committed receipt back and holds the book: no re-sizing.
    account_obj = broker_obj.get_account_snapshot(ROUTE_STR)
    next_state_obj = replace(_pod_state(release_obj, case_obj.fill_ts, {a: s for a, s in account_obj.position_amount_map.items() if s},
        account_obj.cash_float, {}), strategy_state_dict=committed_dict)
    next_obj = _decide(release_obj, case_obj.fill_ts, next_state_obj, case_obj.pricing_data_df)
    assert not next_obj.snapshot_metadata_dict["rebalance_bool"] and next_obj.snapshot_metadata_dict["no_order_bool"]
    assert next_obj.snapshot_metadata_dict["flow_vote_warning_code_list"] == []


@pytest.mark.parametrize("drift_str", ["holdings", "orders", "account", "disabled", "cutoff"])
def test_the_final_account_read_blocks_the_send(pair_case, tmp_path, monkeypatch, drift_str):
    case_obj = _prepared_pair_cycle(tmp_path, monkeypatch, pair_case)
    as_of_ts = case_obj.decision_obj.submission_timestamp_ts
    if drift_str == "disabled":
        case_obj.store_obj.upsert_release(replace(case_obj.release_obj, enabled_bool=False))
    elif drift_str == "cutoff":
        as_of_ts = case_obj.decision_obj.target_execution_timestamp_ts - timedelta(minutes=MOC_DISPATCH_CUTOFF_MINUTES_INT)
    else:
        original_fn = case_obj.broker_obj.get_flow_vote_account_snapshot

        def changed_fn(account_route_str, *argument_tuple, **argument_dict):
            snapshot_obj = original_fn(account_route_str, *argument_tuple, **argument_dict)
            if drift_str == "holdings":
                return replace(snapshot_obj, position_amount_map={**snapshot_obj.position_amount_map, "TLT": 801.0})
            if drift_str == "orders":
                return replace(snapshot_obj, open_order_id_list=["foreign-order"])
            return replace(snapshot_obj, account_route_str="DU_OTHER")

        monkeypatch.setattr(case_obj.broker_obj, "get_flow_vote_account_snapshot", changed_fn)
    assert runner.submit_ready_vplans(case_obj.store_obj, case_obj.broker_obj, as_of_ts, "paper", False,
        vplan_id_int=case_obj.vplan_obj.vplan_id_int, **case_obj.option_dict)["submitted_vplan_count_int"] == 0
    assert not case_obj.broker_obj.submitted_order_request_list
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str != "submitting"
    if drift_str in {"holdings", "orders", "account"}:
        assert "flow_vote_pre_submit_account_changed" in Path(case_obj.option_dict["log_path_str"]).read_text(encoding="utf-8")


def test_the_state_store_refuses_a_plan_or_a_claim_for_an_inactive_decision(pair_case, tmp_path, monkeypatch):
    case_obj = _prepared_pair_cycle(tmp_path, monkeypatch, pair_case)
    case_obj.store_obj.mark_decision_plan_status(case_obj.decision_obj.decision_plan_id_int, "expired")
    assert case_obj.store_obj.claim_vplan_for_submission(case_obj.vplan_obj.vplan_id_int) is False
    with pytest.raises(ValueError, match="active planned decision"):
        case_obj.store_obj.insert_vplan(replace(case_obj.vplan_obj, vplan_id_int=None))


def test_a_flow_vote_decision_error_is_blocked_not_raised(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    releases_root_path_obj.mkdir()
    (releases_root_path_obj / "flow_vote.yaml").write_text(TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("mode: live", "mode: paper").replace("enabled_bool: false", "enabled_bool: true")
        .replace("YOUR_FLOW_VOTE_ACCOUNT", ROUTE_STR), encoding="utf-8")
    # Direct mode: the adapter refuses to decide without the qualified snapshot.
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    monkeypatch.setattr(scheduler_utils, "evaluate_build_gate_dict", lambda *argument_tuple, **argument_dict: {"due_bool": True})
    result_dict = runner.build_decision_plans(LiveStateStore(str(tmp_path / "flow_vote.sqlite3")), _et(2026, 11, 20, 16, 30),
        str(releases_root_path_obj), "paper", auto_sync_norgate_snapshots_bool=False,
        log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert result_dict["created_decision_plan_count_int"] == 0
    assert result_dict["reason_count_map_dict"].get("flow_vote_decision_blocked") == 1


@pytest.mark.parametrize("open_order_bool", [False, True])
def test_the_eod_capture_keeps_signed_holdings_and_refuses_an_account_with_open_orders(tmp_path, open_order_bool):
    release_obj = _release()
    store_obj = LiveStateStore(str(tmp_path / "flow_vote.sqlite3"))
    store_obj.upsert_release(release_obj)
    broker_obj = StubBrokerAdapter()
    as_of_ts = _et(2026, 10, 30, 16, 20)  # a past close: the capture refuses a broker stamp later than the real clock
    broker_obj.seed_account_snapshot(ROUTE_STR, 152_000.0, 100_000.0, {"SPY": 50.0, "TLT": -300.0}, as_of_ts, session_mode_str="paper")
    if open_order_bool:
        broker_obj._snapshot_map[ROUTE_STR] = replace(broker_obj._snapshot_map[ROUTE_STR], open_order_id_list=["9"])
    log_path_str = str(tmp_path / "eod.log")
    result_dict = runner.eod_snapshot(store_obj, broker_obj, as_of_ts, "paper", log_path_str=log_path_str, trace_enabled_bool=False)
    state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    if open_order_bool:
        assert result_dict["eod_snapshot_count_int"] == 0
        assert state_obj is None or state_obj.snapshot_stage_str != "eod"
        assert "flow_vote_eod_source_untrusted" in Path(log_path_str).read_text(encoding="utf-8")
    else:
        assert result_dict["eod_snapshot_count_int"] == 1
        assert (state_obj.snapshot_stage_str, state_obj.snapshot_source_str) == ("eod", "broker")
        assert state_obj.position_amount_map == {"SPY": 50.0, "TLT": -300.0} and state_obj.cash_float == 152_000.0


# ---------------------------------------------------------------- the watchdog and the V4 cycle at the close
def test_the_watchdog_owes_a_closing_send_only_after_its_cutoff_and_grace(daily_case, monkeypatch, tmp_path):  # noqa: F811
    store_obj, saved_release_obj, decision_obj, plan_obj, _ = daily_case
    release_obj = replace(saved_release_obj, execution_policy_str="next_close_moc")
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "planned")
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "ready")
    monkeypatch.setattr(daily_watchdog, "load_release_list", lambda _root_str: [release_obj])
    monkeypatch.setattr(daily_watchdog.dashboard, "resolve_db_path_for_release_str", lambda _release_obj, _config_obj: store_obj.db_path_str)

    def alert_list(as_of_ts):
        return daily_watchdog.daily_heartbeat_alert_list(str(tmp_path), str(tmp_path / "none.yaml"),
            str(tmp_path / "events.jsonl"), as_of_ts, release_obj.mode_str)

    cutoff_ts = CLOSE_TS - timedelta(minutes=MOC_DISPATCH_CUTOFF_MINUTES_INT)
    assert alert_list(OPEN_TS + timedelta(minutes=5)) == []  # the open owes nothing to a closing batch
    assert alert_list(cutoff_ts + timedelta(minutes=4, seconds=59)) == []
    alert_dict, = alert_list(cutoff_ts + timedelta(minutes=5))
    assert alert_dict["kind_str"] == "opening_not_dispatched"


def test_the_v4_cycle_reads_a_closing_plan_against_its_cutoff():
    cutoff_ts = V4_CLOSE_TS - timedelta(minutes=MOC_DISPATCH_CUTOFF_MINUTES_INT)
    unsent_dict = _v4_daily_dict(decision_status_str="vplan_ready", vplan_status_str="ready", execution_policy_str="next_close_moc")
    sent_dict = _v4_daily_dict(execution_policy_str="next_close_moc")
    for daily_dict in (unsent_dict, sent_dict):
        daily_dict["decision_dict"]["target_execution_timestamp_str"] = V4_CLOSE_TS.isoformat()
        daily_dict["vplan_dict"]["target_execution_timestamp_str"] = V4_CLOSE_TS.isoformat()
    ready_dict = _v4_view(deepcopy(unsent_dict), cutoff_ts - timedelta(minutes=10))
    # The plan shows its send time (close minus 20), never the cutoff at which sending is already refused.
    assert (ready_dict["daily_status_dict"]["key_str"], ready_dict["daily_status_dict"]["label_str"]) == ("ready", "Ready · sends 15:40")
    assert ready_dict["step_dict_list"][3]["fact_str"] == "Sends 15:40"
    assert _v4_view(deepcopy(unsent_dict), cutoff_ts + timedelta(minutes=4, seconds=59))["daily_status_dict"]["key_str"] == "ready"
    not_sent_dict = _v4_view(deepcopy(unsent_dict), cutoff_ts + timedelta(minutes=5))["daily_status_dict"]
    assert not_sent_dict["key_str"] == "not_sent" and "closing batch" in not_sent_dict["detail_str"]
    trading_dict = _v4_view(deepcopy(sent_dict), cutoff_ts)
    assert trading_dict["daily_status_dict"]["key_str"] == "trading"
    assert trading_dict["step_dict_list"][4]["fact_str"] == "Fills at the close"


def test_the_v4_cycle_shows_the_half_day_send_time():
    half_close_ts = datetime(2026, 11, 27, 18, 0, tzinfo=UTC)  # 13:00 ET, the day after Thanksgiving
    daily_dict = _v4_daily_dict(decision_status_str="vplan_ready", vplan_status_str="ready", execution_policy_str="next_close_moc",
        session_close_timestamp_str=half_close_ts.isoformat())
    daily_dict["decision_dict"]["target_execution_timestamp_str"] = half_close_ts.isoformat()
    daily_dict["vplan_dict"]["target_execution_timestamp_str"] = half_close_ts.isoformat()
    view_dict = _v4_view(daily_dict, half_close_ts - timedelta(minutes=30))
    assert view_dict["daily_status_dict"]["label_str"] == "Ready · sends 12:40"


# ---------------------------------------------------------------- the data path: exporter -> snapshot -> reader -> decision
FLOW_END_STR = "2026-10-29"  # the eve of October's last session: a fill eve


class FlowProvider:
    """Norgate's call shape for the six Flow Vote series: observed rows only, native float32 prices."""
    StockPriceAdjustmentType = SimpleNamespace(CAPITALSPECIAL="CAPITALSPECIAL", TOTALRETURN="TOTALRETURN")
    PaddingType = SimpleNamespace(ALLMARKETDAYS="ALLMARKETDAYS", NONE="NONE")

    def __init__(self, end_str):
        import exchange_calendars as xcals

        # A plain index, as Norgate returns it (no calendar frequency attached).
        self.date_idx = pd.DatetimeIndex(list(xcals.get_calendar("XNYS", start="2002-01-01").sessions_in_range("2002-07-26", end_str)), name="Date")
        rng_obj = np.random.default_rng(11)
        self.frame_dict, self.stale_key_tuple = {}, None
        for symbol_str in ("SPY", "TLT", "IEF", "$SPXTR"):
            capital_vec = 100.0 * np.cumprod(1.0 + rng_obj.normal(0.0002, 0.01, len(self.date_idx)))
            for adjustment_str, growth_obj in (("CAPITALSPECIAL", 1.0), ("TOTALRETURN", np.linspace(1.0, 1.8, len(self.date_idx)))):
                close_vec = (capital_vec * growth_obj).astype(np.float32)
                frame_df = pd.DataFrame({"Open": close_vec, "High": (close_vec * 1.01).astype(np.float32),
                    "Low": (close_vec * 0.99).astype(np.float32), "Close": close_vec,
                    "Volume": np.full(len(close_vec), 1_000_000, dtype=np.int64), "Turnover": close_vec.astype(float) * 1e6},
                    index=self.date_idx)
                if symbol_str != "$SPXTR":
                    frame_df["Unadjusted Close"] = close_vec
                    frame_df["Dividend"] = np.zeros(len(close_vec), dtype=np.float32)
                self.frame_dict[(symbol_str, adjustment_str)] = frame_df

    def price_timeseries(self, symbol_str, **option_dict):
        key_tuple = (symbol_str, option_dict["stock_price_adjustment_setting"])
        frame_df = self.frame_dict[key_tuple].iloc[:-1] if key_tuple == self.stale_key_tuple else self.frame_dict[key_tuple]
        return frame_df.loc[option_dict["start_date"]:option_dict["end_date"]].copy()


@pytest.fixture
def flow_snapshot_case(tmp_path, monkeypatch):
    provider_obj = FlowProvider(FLOW_END_STR)
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: provider_obj)
    monkeypatch.setattr(norgate_loader, "_load_direct_norgate_module", lambda: provider_obj)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    config_obj = replace(vote_module.DEFAULT_CONFIG, end_date_str=FLOW_END_STR)
    direct_df = vote_module.get_month_end_flow_data(config_obj)
    snapshot_path_obj = export_module.export_profile_snapshot(snapshot_root_str=str(tmp_path), profile_str=FLOW_VOTE_PROFILE_STR,
        snapshot_date_str=FLOW_END_STR)
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setattr(norgate_loader, "_load_direct_norgate_module", lambda: pytest.fail("the snapshot consumer called direct Norgate"))
    signal_ts = pd.Timestamp(FLOW_END_STR)
    return SimpleNamespace(provider_obj=provider_obj, direct_df=direct_df, snapshot_path_obj=snapshot_path_obj, config_obj=config_obj,
        signal_ts=signal_ts, as_of_ts=scheduler_utils.get_session_close_timestamp_ts(signal_ts, "XNYS") + timedelta(minutes=30))


def test_the_snapshot_reader_returns_the_research_frame_and_the_same_decision(flow_snapshot_case):
    case_obj = flow_snapshot_case
    snapshot_df = vote_module.get_month_end_flow_data(case_obj.config_obj)
    pd.testing.assert_frame_equal(snapshot_df, case_obj.direct_df, check_exact=True)
    assert snapshot_df.attrs == case_obj.direct_df.attrs
    release_obj = _release()
    state_obj = _pod_state(release_obj, case_obj.signal_ts, {}, 100_000.0, {})
    host_obj = strategy_host.build_decision_plan_for_release(release_obj, case_obj.as_of_ts, state_obj)
    direct_obj = build_flow_vote_decision_from_prices(release_obj, case_obj.as_of_ts, state_obj, case_obj.direct_df,
        {key_str: host_obj.snapshot_metadata_dict[key_str] for key_str in ("norgate_snapshot_date_str", "norgate_data_profile_str", "norgate_manifest_hash_str")})
    assert host_obj.snapshot_metadata_dict["flow_vote_fill_next_session_bool"]
    for key_str in ("fixed_target_share_map_dict", "rebalance_bool", "flow_vote_standing_event_dict", "sizing_close_price_map_dict"):
        assert host_obj.snapshot_metadata_dict[key_str] == direct_obj.snapshot_metadata_dict[key_str]


def _rewrite_snapshot(case_obj, price_fn=None, contract_fn=None, recount_bool=False):
    price_path_obj = case_obj.snapshot_path_obj / "prices.parquet"
    price_df = pd.read_parquet(price_path_obj)
    if price_fn is not None:
        price_df = price_fn(price_df)
        price_df.to_parquet(price_path_obj, index=False)
    manifest_path_obj = case_obj.snapshot_path_obj / "manifest.json"
    manifest_dict = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
    if recount_bool:
        manifest_dict["data_contract"]["series_coverage_dict"] = flow_vote_price_coverage_dict(price_df)
    if contract_fn is not None:
        contract_fn(manifest_dict["data_contract"])
    manifest_dict["files"]["prices.parquet"]["sha256"] = hashlib.sha256(price_path_obj.read_bytes()).hexdigest()
    manifest_path_obj.write_text(json.dumps(manifest_dict), encoding="utf-8")


def _pair_mask_ser(price_df, symbol_str, adjustment_str):
    return price_df["symbol_str"].eq(symbol_str) & price_df["adjustment_str"].eq(adjustment_str)


@pytest.mark.parametrize("mutation_str,match_str", [
    ("endpoint", "stale endpoint"), ("gap", "missing session"), ("coverage", "declared source coverage"),
    ("padding", "contract mismatch"), ("pair", "exact symbol/adjustment pairs"), ("dtype", "invalid native dtype"),
])
def test_a_corrupt_snapshot_fails_closed_even_with_an_updated_hash(flow_snapshot_case, mutation_str, match_str):
    case_obj = flow_snapshot_case

    def price_fn(price_df):
        mask_ser = _pair_mask_ser(price_df, "TLT", "TOTALRETURN")
        if mutation_str == "endpoint":
            return price_df.drop(index=price_df.index[mask_ser][-1])
        if mutation_str in ("gap", "coverage"):
            return price_df.drop(index=price_df.index[mask_ser][100])
        if mutation_str == "pair":
            return price_df.loc[~_pair_mask_ser(price_df, "IEF", "TOTALRETURN")]
        price_df["Close"] = price_df["Close"].astype(float)
        price_df.loc[price_df.index[mask_ser][50], "Close"] += 1e-7
        return price_df

    def contract_fn(contract_dict):
        contract_dict["price_padding_setting_str"] = "ALLMARKETDAYS"

    _rewrite_snapshot(case_obj, price_fn=None if mutation_str == "padding" else price_fn,
        contract_fn=contract_fn if mutation_str == "padding" else None, recount_bool=mutation_str in ("endpoint", "gap"))
    with pytest.raises((ValueError, snapshot_module.NorgateSnapshotValidationError), match=match_str):
        snapshot_module.load_valid_snapshot_manifest(FLOW_VOTE_PROFILE_STR)


def test_the_exporter_refuses_a_series_without_an_observed_bar_at_t(tmp_path, monkeypatch):
    provider_obj = FlowProvider(FLOW_END_STR)
    provider_obj.stale_key_tuple = ("TLT", "TOTALRETURN")
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: provider_obj)
    with pytest.raises(ValueError, match="observed bar"):
        export_module.export_profile_snapshot(snapshot_root_str=str(tmp_path), profile_str=FLOW_VOTE_PROFILE_STR, snapshot_date_str=FLOW_END_STR)


def test_a_snapshot_published_between_the_six_reads_is_refused(flow_snapshot_case, monkeypatch):
    original_fn, call_list = snapshot_module.load_valid_snapshot_manifest, []

    def changing_fn(*argument_tuple, **argument_dict):
        manifest_obj = original_fn(*argument_tuple, **argument_dict)
        call_list.append(manifest_obj)
        return manifest_obj if len(call_list) == 1 else replace(manifest_obj, manifest_hash_str="another vintage")

    monkeypatch.setattr(snapshot_module, "load_valid_snapshot_manifest", changing_fn)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="changed during the data load"):
        vote_module.get_month_end_flow_data(flow_snapshot_case.config_obj)


def test_the_decision_refuses_metadata_from_another_snapshot(flow_snapshot_case, monkeypatch):
    case_obj = flow_snapshot_case
    monkeypatch.setattr(snapshot_module, "build_data_source_metadata_dict", lambda _profile_str: {"norgate_manifest_hash_str": "another"})
    release_obj = _release()
    with pytest.raises(ValueError, match="changed while preparing"):
        strategy_host.build_decision_plan_for_release(release_obj, case_obj.as_of_ts, _pod_state(release_obj, case_obj.signal_ts, {}, 100_000.0, {}))
