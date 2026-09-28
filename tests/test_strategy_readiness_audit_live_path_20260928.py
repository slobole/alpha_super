"""
Strategy readiness audit 2026-09-28 - live execution path evidence (read-only).

These tests document CURRENT behaviour of the live path after/around the DecisionPlan
(scheduler gate, calendar, VPlan sizing, broker snapshot parsing). They use synthetic
inputs and fakes only; no broker, no network, no Norgate. A passing test here means
"the audited behaviour is as described in LIVE_PATH_FINDINGS.md", not "the behaviour is
desirable". Tests whose name contains `documents_risk` pin a finding.
"""
from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from alpha.live import scheduler_utils
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan, build_vplan
from alpha.live.models import BrokerSnapshot, DecisionPlan, LivePriceSnapshot, LiveRelease
from alpha.live.strategy_host import _build_full_target_weight_decision_plan
from strategies.momentum.strategy_mo_atr_normalized_ndx import get_monthly_decision_close_df

ET = ZoneInfo("America/New_York")


def _release(
    *,
    strategy_import_str: str = "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
    data_profile_str: str = "norgate_eod_etf_plus_vix_helper",
    pod_budget_fraction_float: float = 1.0,
) -> LiveRelease:
    return LiveRelease(
        release_id_str="audit.release",
        user_id_str="audit_user",
        pod_id_str="pod_audit",
        account_route_str="U000AUDIT",
        strategy_import_str=strategy_import_str,
        mode_str="live",
        session_calendar_id_str="XNYS",
        signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open",
        data_profile_str=data_profile_str,
        params_dict={},
        risk_profile_str="audit",
        enabled_bool=True,
        source_path_str="audit.yaml",
        pod_budget_fraction_float=pod_budget_fraction_float,
        auto_submit_enabled_bool=False,
    )


def _et(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=ET)


# ---------------------------------------------------------------------------
# B2 - calendar and invocation timing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "signal_date_str, expected_open_et",
    [
        ("2026-09-30", _et(2026, 10, 1, 9, 30)),   # next live cycle
        ("2026-10-30", _et(2026, 11, 2, 9, 30)),   # Oct 31 is a Saturday
        ("2026-11-30", _et(2026, 12, 1, 9, 30)),
        ("2026-12-31", _et(2027, 1, 4, 9, 30)),    # Jan 1 2027 holiday (Fri) + weekend
        ("2027-03-31", _et(2027, 4, 1, 9, 30)),    # Good Friday 2027 is Mar 26
        ("2027-04-30", _et(2027, 5, 3, 9, 30)),
        ("2027-07-30", _et(2027, 8, 2, 9, 30)),    # Jul 31 2027 is a Saturday
    ],
)
def test_next_month_first_open_uses_real_xnys_sessions(signal_date_str, expected_open_et):
    release_obj = _release()
    signal_date_ts = pd.Timestamp(signal_date_str).to_pydatetime()
    target_ts = scheduler_utils.build_target_execution_timestamp_ts(signal_date_ts, release_obj)
    submit_ts = scheduler_utils.build_submission_timestamp_ts(signal_date_ts, release_obj)
    assert target_ts == expected_open_et
    # 6.5 minutes before the open = 09:23:30 ET (before the 09:28 MOO cutoff).
    assert (target_ts - submit_ts).total_seconds() == 390.0
    assert scheduler_utils.is_last_session_of_month_bool(pd.Timestamp(signal_date_str), "XNYS")


def test_xnys_calendar_knows_2026_2027_holidays_and_early_closes():
    cal = scheduler_utils.get_exchange_calendar_obj("XNYS")
    for closed_str in ("2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-03-26", "2027-07-05"):
        assert not cal.is_session(pd.Timestamp(closed_str)), closed_str
    for early_str in ("2026-11-27", "2026-12-24"):
        close_et = scheduler_utils.get_session_close_timestamp_ts(pd.Timestamp(early_str), "XNYS")
        assert (close_et.hour, close_et.minute) == (13, 0), early_str
    # Month-end resolution of weekend month ends.
    assert not scheduler_utils.is_last_session_of_month_bool(pd.Timestamp("2026-10-29"), "XNYS")
    assert scheduler_utils.is_last_session_of_month_bool(pd.Timestamp("2026-10-30"), "XNYS")


@pytest.mark.parametrize(
    "heartbeat_str, as_of_et, expected_due_bool, expected_reason_str",
    [
        # Normal month-end evening after the snapshot lands.
        ("2026-09-30", _et(2026, 9, 30, 18, 0), True, "snapshot_ready"),
        # Catch-up on the first session of the month, before the 09:23:30 submit.
        ("2026-09-30", _et(2026, 10, 1, 9, 0), True, "carry_forward_snapshot_ready"),
        # After the submit time on the first session: the whole month is skipped.
        ("2026-09-30", _et(2026, 10, 1, 9, 24), False, "snapshot_window_expired"),
        # VPS down on Sep 30 AND Oct 1: no catch-up later in the month.
        ("2026-09-30", _et(2026, 10, 2, 9, 0), False, "snapshot_not_ready_for_session"),
        # Mid-month heartbeat: never due.
        ("2026-09-15", _et(2026, 9, 15, 18, 0), False, "not_month_end_session"),
        # Snapshot did not refresh on the month-end (heartbeat = Sep 29): skipped.
        ("2026-09-29", _et(2026, 10, 1, 9, 0), False, "not_month_end_session"),
        # Weekend month-end: Saturday is not due, Monday pre-open is.
        ("2026-10-30", _et(2026, 10, 31, 10, 0), False, "snapshot_not_ready_for_session"),
        ("2026-10-30", _et(2026, 11, 2, 9, 0), True, "carry_forward_snapshot_ready"),
        # Year end across the Jan 1 holiday.
        ("2026-12-31", _et(2027, 1, 1, 12, 0), False, "snapshot_not_ready_for_session"),
        ("2026-12-31", _et(2027, 1, 4, 9, 0), True, "carry_forward_snapshot_ready"),
    ],
)
def test_month_end_build_gate_matrix(monkeypatch, heartbeat_str, as_of_et, expected_due_bool, expected_reason_str):
    monkeypatch.setattr(
        scheduler_utils,
        "load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp(heartbeat_str),
    )
    gate_dict = scheduler_utils.evaluate_build_gate_dict(_release(), as_of_et.astimezone(UTC))
    assert gate_dict["due_bool"] is expected_due_bool
    assert gate_dict["reason_code_str"] == expected_reason_str


def _synthetic_close_df(end_date_str: str) -> pd.DataFrame:
    cal = scheduler_utils.get_exchange_calendar_obj("XNYS")
    session_index = pd.DatetimeIndex(cal.sessions_in_range("2025-06-02", end_date_str)).tz_localize(None)
    return pd.DataFrame({"AAA": np.linspace(100.0, 120.0, len(session_index))}, index=session_index)


@pytest.mark.parametrize(
    "data_end_str, expected_last_decision_str",
    [
        ("2026-09-15", "2026-08-31"),  # mid-month invocation -> previous month-end
        ("2026-09-29", "2026-08-31"),  # month-end bar missing -> previous month-end
        ("2026-09-30", "2026-09-30"),  # complete month
        ("2026-10-30", "2026-10-30"),  # weekend month-end
        ("2026-10-01", "2026-09-30"),  # first session of next month present
    ],
)
def test_ndx_monthly_decision_drops_partial_trailing_month(data_end_str, expected_last_decision_str):
    monthly_df = get_monthly_decision_close_df(_synthetic_close_df(data_end_str))
    assert monthly_df.index[-1] == pd.Timestamp(expected_last_decision_str)


def test_ndx_partial_month_plan_would_target_a_past_open_and_only_expire(monkeypatch):
    """NDX host has no stale-target guard (TAA host raises at strategy_host.py:1092).

    A mid-month call resolves to the previous month-end (test above), so the plan it
    builds targets an open that already happened. The runner inserts it
    (runner.py:2695) and the next expire pass marks it expired (runner.py:2518-2524).
    """
    monkeypatch.setattr(
        "alpha.live.strategy_host.build_data_source_metadata_dict",
        lambda data_profile_str=None: {"norgate_data_profile_str": data_profile_str},
    )
    release_obj = _release(
        strategy_import_str="strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy",
        data_profile_str="norgate_eod_ndx_pit_plus_vxn_helper",
    )
    plan_obj = _build_full_target_weight_decision_plan(
        release_obj=release_obj,
        signal_date_ts=pd.Timestamp("2026-08-31").to_pydatetime(),
        decision_base_position_map_dict={},
        full_target_weight_map_dict={"AAA": 0.1},
        cash_reserve_weight_float=0.9,
        strategy_state_dict={},
    )
    assert plan_obj.target_execution_timestamp_ts == _et(2026, 9, 1, 9, 30)
    assert scheduler_utils.is_execution_window_expired_bool(
        plan_obj.execution_policy_str, plan_obj.target_execution_timestamp_ts, _et(2026, 9, 15, 18, 0)
    )


def test_documents_risk_moo_submit_allowed_after_opg_cutoff():
    """Submit is accepted until the open itself (09:30:00), not until the 09:28 MOO cutoff."""
    target_ts = _et(2026, 10, 1, 9, 30)
    assert not scheduler_utils.is_execution_window_expired_bool("next_month_first_open", target_ts, _et(2026, 10, 1, 9, 29, 45))
    assert scheduler_utils.is_execution_window_expired_bool("next_month_first_open", target_ts, _et(2026, 10, 1, 9, 30, 0))


# ---------------------------------------------------------------------------
# B5 - VPlan sizing and order conversion
# ---------------------------------------------------------------------------


def _full_target_plan(weight_map_dict: dict[str, float]) -> DecisionPlan:
    return DecisionPlan(
        release_id_str="audit.release",
        user_id_str="audit_user",
        pod_id_str="pod_audit",
        account_route_str="U000AUDIT",
        signal_timestamp_ts=_et(2026, 9, 30, 16, 0),
        submission_timestamp_ts=_et(2026, 10, 1, 9, 23, 30),
        target_execution_timestamp_ts=_et(2026, 10, 1, 9, 30),
        execution_policy_str="next_month_first_open",
        decision_base_position_map={},
        snapshot_metadata_dict={"norgate_data_profile_str": "norgate_eod_etf_plus_vix_helper"},
        strategy_state_dict={},
        decision_book_type_str="full_target_weight_book",
        full_target_weight_map_dict=weight_map_dict,
        cash_reserve_weight_float=max(0.0, 1.0 - sum(weight_map_dict.values())),
        preserve_untouched_positions_bool=False,
        rebalance_omitted_assets_to_zero_bool=True,
        decision_plan_id_int=7,
    )


def _broker(net_liq_float: float, position_map_dict: dict[str, float]) -> BrokerSnapshot:
    return BrokerSnapshot(
        account_route_str="U000AUDIT",
        snapshot_timestamp_ts=_et(2026, 10, 1, 9, 23, 30),
        cash_float=0.0,
        total_value_float=net_liq_float,
        position_amount_map=position_map_dict,
        net_liq_float=net_liq_float,
    )


def _prices(price_map_dict: dict[str, float]) -> LivePriceSnapshot:
    return LivePriceSnapshot(
        account_route_str="U000AUDIT",
        snapshot_timestamp_ts=_et(2026, 10, 1, 9, 23, 31),
        price_source_str="auction_225",
        asset_reference_price_map=price_map_dict,
    )


def test_documents_risk_full_target_book_liquidates_unrelated_holding():
    vplan_obj = build_vplan(
        _release(),
        _full_target_plan({"TQQQ": 1.0}),
        _broker(100_000.0, {"TQQQ": 500.0, "AAPL": 10.0}),
        _prices({"TQQQ": 100.0, "AAPL": 250.0}),
    )
    assert vplan_obj.target_share_map == {"AAPL": 0.0, "TQQQ": 1000.0}
    assert vplan_obj.order_delta_map == {"AAPL": -10.0, "TQQQ": 500.0}


def test_full_target_sizing_is_floor_of_full_netliq_with_no_cash_buffer():
    vplan_obj = build_vplan(
        _release(),
        _full_target_plan({"TQQQ": 0.6, "BTAL": 0.4}),
        _broker(100_000.0, {}),
        _prices({"TQQQ": 87.37, "BTAL": 19.91}),
    )
    assert vplan_obj.pod_budget_float == 100_000.0
    assert vplan_obj.target_share_map == {"BTAL": 2009.0, "TQQQ": 686.0}
    invested_float = sum(row.estimated_target_notional_float for row in vplan_obj.vplan_row_list)
    assert 99_900.0 < invested_float <= 100_000.0
    # A +1% gap between the 09:23:30 reference and the open fill overdraws cash by ~USD 1K.
    assert invested_float * 1.01 > 100_000.0


def test_documents_risk_sells_and_buys_are_one_alphabetical_basket():
    vplan_obj = build_vplan(
        _release(),
        _full_target_plan({"AAA": 1.0}),
        _broker(100_000.0, {"ZZZ": 1000.0}),
        _prices({"AAA": 50.0, "ZZZ": 100.0}),
    )
    request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    assert [(r.asset_str, r.amount_float, r.broker_order_type_str) for r in request_list] == [
        ("AAA", 2000.0, "MOO"),   # buy first
        ("ZZZ", -1000.0, "MOO"),  # funded only by a same-auction sale
    ]


def test_invalid_live_price_fails_closed_in_vplan_builder():
    for bad_price_float in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            build_vplan(
                _release(),
                _full_target_plan({"TQQQ": 1.0}),
                _broker(100_000.0, {}),
                _prices({"TQQQ": bad_price_float}),
            )


def test_incremental_exit_of_unheld_name_sends_no_order():
    plan_obj = DecisionPlan(
        release_id_str="audit.release",
        user_id_str="audit_user",
        pod_id_str="pod_audit",
        account_route_str="U000AUDIT",
        signal_timestamp_ts=_et(2026, 9, 30, 16, 0),
        submission_timestamp_ts=_et(2026, 10, 1, 9, 23, 30),
        target_execution_timestamp_ts=_et(2026, 10, 1, 9, 30),
        execution_policy_str="next_open_moo",
        decision_base_position_map={"OLD": 10.0},
        snapshot_metadata_dict={},
        strategy_state_dict={},
        decision_book_type_str="incremental_entry_exit_book",
        entry_target_weight_map_dict={"NEW": 0.2, "HELD": 0.2},
        exit_asset_set={"OLD"},
        entry_priority_list=["NEW", "HELD"],
        decision_plan_id_int=8,
    )
    vplan_obj = build_vplan(
        _release(strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy"),
        plan_obj,
        _broker(50_000.0, {"HELD": 30.0, "UNRELATED": 5.0}),
        _prices({"NEW": 100.0, "HELD": 100.0, "OLD": 10.0}),
    )
    # OLD is not held -> delta 0 -> no request; UNRELATED is preserved; HELD is re-targeted.
    assert vplan_obj.order_delta_map == {"OLD": 0.0, "NEW": 100.0, "HELD": 70.0}
    request_asset_list = [r.asset_str for r in build_broker_order_request_list_from_vplan(vplan_obj)]
    assert request_asset_list == ["NEW", "HELD"]


# ---------------------------------------------------------------------------
# B4/B5 - broker snapshot parsing used for sizing (fake ib_async objects)
# ---------------------------------------------------------------------------


class _FakeIB:
    def __init__(self, account_value_list, position_list):
        self._account_value_list = account_value_list
        self._position_list = position_list

    def accountSummary(self, account):
        return self._account_value_list

    def positions(self, account):
        return self._position_list

    def reqOpenOrders(self):
        return []


def _fake_client(account_value_list, position_list):
    from alpha.live.ibkr_socket_client import IBKRSocketClient

    client_obj = IBKRSocketClient("127.0.0.1", 7496, 99, 1.0)
    client_obj.connect = lambda: contextlib.nullcontext(_FakeIB(account_value_list, position_list))
    return client_obj


def _account_value(tag_str, value_str, currency_str):
    return SimpleNamespace(account="U000AUDIT", tag=tag_str, value=value_str, currency=currency_str, modelCode="")


def _position(symbol_str, sec_type_str, amount_float, currency_str="USD"):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol_str, secType=sec_type_str, currency=currency_str),
        position=amount_float,
    )


def test_documents_risk_netliq_currency_is_not_checked_for_sizing():
    """A non-USD base currency (e.g. ILS) NetLiquidation is used as a USD budget."""
    client_obj = _fake_client(
        [_account_value("NetLiquidation", "370000", "ILS"), _account_value("TotalCashValue", "370000", "ILS")],
        [],
    )
    snapshot_obj = client_obj.get_account_snapshot("U000AUDIT")
    assert snapshot_obj.net_liq_float == 370_000.0
    vplan_obj = build_vplan(_release(), _full_target_plan({"TQQQ": 1.0}), snapshot_obj, _prices({"TQQQ": 100.0}))
    # ~USD 100K of equity would be sized as USD 370K of TQQQ (3.7x).
    assert vplan_obj.target_share_map == {"TQQQ": 3700.0}


def test_documents_risk_position_map_is_keyed_by_symbol_only():
    """A non-stock position on the same symbol overwrites the stock share count."""
    client_obj = _fake_client(
        [_account_value("NetLiquidation", "100000", "USD")],
        [_position("TQQQ", "STK", 1000.0), _position("TQQQ", "OPT", -5.0)],
    )
    snapshot_obj = client_obj.get_account_snapshot("U000AUDIT")
    assert snapshot_obj.position_amount_map == {"TQQQ": -5.0}
    vplan_obj = build_vplan(_release(), _full_target_plan({"TQQQ": 1.0}), snapshot_obj, _prices({"TQQQ": 100.0}))
    # The account already holds 1000 shares; the VPlan buys 1005 more (2x target).
    assert vplan_obj.order_delta_map == {"TQQQ": 1005.0}


def test_documents_risk_missing_netliq_tag_parses_as_zero_and_blocks():
    client_obj = _fake_client([], [_position("TQQQ", "STK", 1000.0)])
    snapshot_obj = client_obj.get_account_snapshot("U000AUDIT")
    # runner.build_vplans blocks non_positive_net_liq (runner.py:2956), so this fails closed.
    assert snapshot_obj.net_liq_float == 0.0


def test_documents_risk_fractional_broker_position_produces_fractional_moo_order():
    vplan_obj = build_vplan(
        _release(),
        _full_target_plan({"TQQQ": 1.0}),
        _broker(100_000.0, {"TQQQ": 400.37}),
        _prices({"TQQQ": 100.0}),
    )
    request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    assert request_list[0].amount_float == pytest.approx(599.63)


def test_documents_risk_small_account_high_price_name_silently_gets_zero_shares():
    # NDX-style book: 10 names x 0.1 x VXN scale 0.25 = 2.5% per name.
    weight_map_dict = {f"N{i}": 0.025 for i in range(9)} | {"BKNG": 0.025}
    price_map_dict = {f"N{i}": 150.0 for i in range(9)} | {"BKNG": 5200.0}
    vplan_obj = build_vplan(_release(), _full_target_plan(weight_map_dict), _broker(30_000.0, {}), _prices(price_map_dict))
    # USD 750 budget < one BKNG share: target 0 shares, no order, no warning field.
    assert vplan_obj.target_share_map["BKNG"] == 0.0
    assert "BKNG" not in [r.asset_str for r in build_broker_order_request_list_from_vplan(vplan_obj)]


@pytest.mark.parametrize(
    "label_str, data_end_str, as_of_et, expected_str",
    [
        ("2026-10-31", "2026-10-30", _et(2026, 10, 30, 18, 0), "2026-10-30"),
        ("2026-09-30", "2026-09-30", _et(2026, 9, 30, 18, 0), "2026-09-30"),
        ("2026-12-31", "2026-12-31", _et(2027, 1, 4, 9, 0), "2026-12-31"),
    ],
)
def test_taa_month_end_label_resolves_to_last_xnys_session(label_str, data_end_str, as_of_et, expected_str):
    available_index = _synthetic_close_df(data_end_str).index
    resolved_ts = scheduler_utils.resolve_calendar_month_end_label_to_last_tradable_session(
        pd.Timestamp(label_str), available_index, "XNYS", as_of_ts=as_of_et
    )
    assert resolved_ts == pd.Timestamp(expected_str)


def test_taa_partial_month_label_is_rejected_as_not_complete():
    with pytest.raises(scheduler_utils.CalendarMonthNotCompleteError):
        scheduler_utils.resolve_calendar_month_end_label_to_last_tradable_session(
            pd.Timestamp("2026-09-30"), _synthetic_close_df("2026-09-25").index, "XNYS", as_of_ts=_et(2026, 9, 25, 18, 0)
        )


def test_documents_risk_netliq_marked_at_prior_close_relevers_on_overnight_gap():
    """Unchanged target (100% TQQQ) after a -8% overnight gap still trades.

    NetLiq before the open is marked at the prior close (assumed: 1000 x 100), while the
    target is priced at the auction indicative (92). A backtest sized on the open NAV
    would hold 1000 shares and trade nothing.
    """
    vplan_obj = build_vplan(
        _release(),
        _full_target_plan({"TQQQ": 1.0}),
        _broker(100_000.0, {"TQQQ": 1000.0}),
        _prices({"TQQQ": 92.0}),
    )
    assert vplan_obj.order_delta_map == {"TQQQ": 86.0}
    open_nav_float = 1000.0 * 92.0
    assert vplan_obj.target_share_map["TQQQ"] * 92.0 / open_nav_float == pytest.approx(1.086, abs=1e-3)
