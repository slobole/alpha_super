from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from ib_async import IB, Stock
from ib_async.objects import AccountValue
from ib_async.util import run

from alpha.live.ibkr_socket_client import IBKRSocketClient


ACCOUNT_STR = "U_CAPTURE"
OTHER_ACCOUNT_STR = "U_OTHER"


def _contract_obj(symbol_str="AAA", conid_int=1, **attribute_dict):
    contract_obj = Stock(symbol_str, "SMART", "USD", conId=conid_int)
    for attribute_str, attribute_obj in attribute_dict.items():
        setattr(contract_obj, attribute_str, attribute_obj)
    return contract_obj


def _portfolio_tuple(
    symbol_str="AAA", conid_int=1, shares_float=2.0,
    mark_float=25.0, value_float=50.0, account_str=ACCOUNT_STR,
    average_cost_obj=20.0, unrealized_pnl_obj=10.0, **attribute_dict,
):
    return (
        _contract_obj(symbol_str, conid_int, **attribute_dict), shares_float,
        mark_float, value_float, average_cost_obj, unrealized_pnl_obj, 0.0, account_str,
    )


def _account_value_list(account_str=ACCOUNT_STR, cash_str="100", nav_str="150"):
    return [
        AccountValue(account_str, "TotalCashValue", cash_str, "USD", ""),
        AccountValue(account_str, "NetLiquidation", nav_str, "USD", ""),
        AccountValue(account_str, "accountReady", "true", "", ""),
    ]


@pytest.fixture
def capture_case(monkeypatch):
    """Exercise the installed IB request/Wrapper/Event machinery, without sockets."""
    ib_obj = IB()
    ib_obj.wrapper.accounts = [ACCOUNT_STR, OTHER_ACCOUNT_STR]
    contract_obj = _contract_obj()
    ib_obj.wrapper.position(ACCOUNT_STR, contract_obj, 2.0, 20.0)
    summary_list = _account_value_list(cash_str="90", nav_str="140") + [
        AccountValue(ACCOUNT_STR, "AvailableFunds", "80", "USD", ""),
        AccountValue(ACCOUNT_STR, "ExcessLiquidity", "70", "USD", ""),
        AccountValue(ACCOUNT_STR, "Cushion", ".5", "", ""),
    ]
    monkeypatch.setattr(ib_obj, "accountSummary", lambda account: summary_list)
    monkeypatch.setattr(ib_obj, "reqOpenOrders", lambda: [
        SimpleNamespace(order=SimpleNamespace(orderId=11, account=ACCOUNT_STR)),
        SimpleNamespace(order=SimpleNamespace(orderId=99, account=OTHER_ACCOUNT_STR)),
    ])
    case_obj = SimpleNamespace(
        ib_obj=ib_obj, client_obj=IBKRSocketClient(timeout_seconds_float=0.02),
        portfolio_list=[_portfolio_tuple()], account_value_list=_account_value_list(),
        end_account_str=ACCOUNT_STR, request_list=[], connection_count_int=0,
        no_reply_bool=False, before_reply_fn=None,
    )

    def send_reply():
        if case_obj.before_reply_fn is not None:
            case_obj.before_reply_fn()
        for account_value_obj in case_obj.account_value_list:
            ib_obj.wrapper.updateAccountValue(
                account_value_obj.tag, account_value_obj.value,
                account_value_obj.currency, account_value_obj.account,
            )
        for portfolio_tuple in case_obj.portfolio_list:
            ib_obj.wrapper.updatePortfolio(*portfolio_tuple)
        if case_obj.end_account_str is not None:
            ib_obj.wrapper.accountDownloadEnd(case_obj.end_account_str)

    def request_account_updates(subscribe_bool, account_str):
        case_obj.request_list.append((subscribe_bool, account_str))
        if subscribe_bool and not case_obj.no_reply_bool:
            asyncio.get_event_loop().call_soon(send_reply)

    monkeypatch.setattr(ib_obj.client, "reqAccountUpdates", request_account_updates)

    @contextmanager
    def fake_connect():
        case_obj.connection_count_int += 1
        yield ib_obj

    monkeypatch.setattr(case_obj.client_obj, "connect", fake_connect)
    yield case_obj
    ib_obj.wrapper.reset()


def _capture(case_obj):
    return case_obj.client_obj.get_account_snapshot(
        ACCOUNT_STR, include_portfolio_valuation_bool=True
    )


def test_default_snapshot_makes_no_valuation_request(capture_case):
    snapshot_obj = capture_case.client_obj.get_account_snapshot(ACCOUNT_STR)
    assert snapshot_obj.portfolio_valuation_dict is None
    assert snapshot_obj.position_amount_map == {"AAA": 2.0}
    assert snapshot_obj.cash_float == 90.0
    assert snapshot_obj.open_order_id_list == ["11"]
    assert capture_case.request_list == []
    assert capture_case.connection_count_int == 1


@pytest.mark.parametrize("include_bool", [False, True])
def test_snapshot_timestamp_preserves_default_disconnect_boundary(capture_case, monkeypatch, include_bool):
    start_ts = datetime(2026, 9, 21, 20, 10, tzinfo=UTC)
    clock_dict = {"now_ts": start_ts}
    capture_call_list = []

    class SnapshotClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock_dict["now_ts"].astimezone(tz) if tz is not None else clock_dict["now_ts"]

    @contextmanager
    def disconnect_advances_clock():
        yield capture_case.ib_obj
        clock_dict["now_ts"] += timedelta(seconds=10)

    def capture_advances_clock(*argument_tuple):
        capture_call_list.append(argument_tuple)
        clock_dict["now_ts"] += timedelta(seconds=3)
        return {"available_bool": False, "observed_timestamp_str": clock_dict["now_ts"].isoformat()}

    monkeypatch.setattr("alpha.live.ibkr_socket_client.datetime", SnapshotClock)
    monkeypatch.setattr(capture_case.client_obj, "connect", disconnect_advances_clock)
    monkeypatch.setattr(capture_case.client_obj, "_capture_portfolio_valuation_dict", capture_advances_clock)
    snapshot_obj = capture_case.client_obj.get_account_snapshot(
        ACCOUNT_STR, include_portfolio_valuation_bool=include_bool
    )
    assert snapshot_obj.snapshot_timestamp_ts == (
        start_ts if include_bool else start_ts + timedelta(seconds=10)
    )
    assert len(capture_call_list) == int(include_bool)


def test_complete_target_download_adds_values_without_changing_base(capture_case):
    before_snapshot_obj = capture_case.client_obj.get_account_snapshot(ACCOUNT_STR)
    original_callback_fn = capture_case.ib_obj.wrapper.accountDownloadEnd
    snapshot_obj = _capture(capture_case)
    valuation_dict = snapshot_obj.portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    assert valuation_dict["source_str"] == "IBKR portfolio"
    assert valuation_dict["account_route_str"] == ACCOUNT_STR
    assert valuation_dict["cash_float"] == 100.0
    assert valuation_dict["broker_nav_float"] == 150.0
    assert valuation_dict["position_list"] == [{
        "symbol_str": "AAA", "conid_int": 1, "currency_str": "USD",
        "shares_float": 2.0, "market_price_float": 25.0, "value_float": 50.0,
        "average_cost_float": 20.0, "unrealized_pnl_float": 10.0,
    }]
    before_dict, after_dict = asdict(before_snapshot_obj), asdict(snapshot_obj)
    for field_str in ("snapshot_timestamp_ts", "portfolio_valuation_dict"):
        before_dict.pop(field_str)
        after_dict.pop(field_str)
    assert before_dict == after_dict
    assert capture_case.request_list == [(True, ACCOUNT_STR)]
    assert capture_case.ib_obj.wrapper.accountDownloadEnd == original_callback_fn
    assert capture_case.connection_count_int == 2
    observed_ts = datetime.fromisoformat(valuation_dict["observed_timestamp_str"])
    assert snapshot_obj.snapshot_timestamp_ts <= observed_ts <= datetime.now(UTC)


def test_timeout_never_uses_preexisting_cached_values(capture_case):
    for portfolio_tuple in capture_case.portfolio_list:
        capture_case.ib_obj.wrapper.updatePortfolio(*portfolio_tuple)
    for account_value_obj in capture_case.account_value_list:
        capture_case.ib_obj.wrapper.updateAccountValue(
            account_value_obj.tag, account_value_obj.value,
            account_value_obj.currency, account_value_obj.account,
        )
    capture_case.no_reply_bool = True
    snapshot_obj = _capture(capture_case)
    assert snapshot_obj.portfolio_valuation_dict["available_bool"] is False
    assert "position_list" not in snapshot_obj.portfolio_valuation_dict
    assert "timed out" in snapshot_obj.portfolio_valuation_dict["reason_str"]
    assert snapshot_obj.cash_float == 90.0
    assert len(capture_case.ib_obj.updatePortfolioEvent) == 0
    assert len(capture_case.ib_obj.accountValueEvent) == 0


def test_installed_wrapper_timeout_retains_incomplete_request_marker():
    ib_obj = IB()
    request_future = ib_obj.wrapper.startReq("accountValues")
    with pytest.raises(TimeoutError):
        run(request_future, timeout=0.001)
    assert request_future.cancelled()
    assert "accountValues" in ib_obj.wrapper._futures
    ib_obj.wrapper.accountDownloadEnd(ACCOUNT_STR)
    assert "accountValues" not in ib_obj.wrapper._futures


@pytest.mark.parametrize("field_str, invalid_obj", [
    ("market_price_float", float("nan")), ("market_price_float", float("inf")),
    ("market_price_float", 0), ("market_price_float", -1),
    ("market_price_float", 1e308), ("value_float", float("nan")),
    ("value_float", 0), ("value_float", -50), ("value_float", 55),
    ("shares_float", 3), ("shares_float", float("nan")),
])
def test_bad_or_changed_position_fails_without_partial_values(capture_case, field_str, invalid_obj):
    row_list = list(capture_case.portfolio_list[0])
    row_list[{"shares_float": 1, "market_price_float": 2, "value_float": 3}[field_str]] = invalid_obj
    capture_case.portfolio_list = [tuple(row_list)]
    snapshot_obj = _capture(capture_case)
    assert snapshot_obj.portfolio_valuation_dict["available_bool"] is False
    assert "position_list" not in snapshot_obj.portfolio_valuation_dict
    assert snapshot_obj.position_amount_map == {"AAA": 2.0}


@pytest.mark.parametrize("attribute_dict", [
    {"currency": "EUR"}, {"secType": "OPT"}, {"multiplier": "100"},
    {"symbol": ""}, {"conId": 0},
])
def test_unsupported_or_ambiguous_contract_fails(capture_case, attribute_dict):
    for attribute_str, value_obj in attribute_dict.items():
        setattr(capture_case.portfolio_list[0][0], attribute_str, value_obj)
    assert _capture(capture_case).portfolio_valuation_dict["available_bool"] is False


def test_unexpected_foreign_completion_without_target_rows_is_not_success(capture_case):
    capture_case.account_value_list = _account_value_list(account_str=OTHER_ACCOUNT_STR)
    capture_case.portfolio_list = [_portfolio_tuple(account_str=OTHER_ACCOUNT_STR)]
    capture_case.end_account_str = OTHER_ACCOUNT_STR
    assert _capture(capture_case).portfolio_valuation_dict["available_bool"] is False


def test_foreign_rows_are_ignored(capture_case):
    capture_case.account_value_list += _account_value_list(account_str=OTHER_ACCOUNT_STR)
    capture_case.portfolio_list += [_portfolio_tuple("OTHER", 2, account_str=OTHER_ACCOUNT_STR)]
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    assert len(valuation_dict["position_list"]) == 1


@pytest.mark.parametrize("tag_str", ["TotalCashValue", "NetLiquidation"])
def test_missing_fresh_account_total_fails(capture_case, tag_str):
    capture_case.account_value_list = [
        row_obj for row_obj in capture_case.account_value_list if row_obj.tag != tag_str
    ]
    assert _capture(capture_case).portfolio_valuation_dict["available_bool"] is False


def test_not_ready_account_fails(capture_case):
    capture_case.account_value_list += [AccountValue(ACCOUNT_STR, "accountReady", "false", "", "")]
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is False
    assert "not ready" in valuation_dict["reason_str"]


def test_missing_holding_fails(capture_case):
    capture_case.portfolio_list = []
    assert _capture(capture_case).portfolio_valuation_dict["available_bool"] is False


def test_duplicate_symbol_with_different_contracts_fails(capture_case):
    capture_case.portfolio_list += [_portfolio_tuple(conid_int=2)]
    assert _capture(capture_case).portfolio_valuation_dict["available_bool"] is False


def test_repeat_same_contract_uses_last_update(capture_case):
    capture_case.portfolio_list += [_portfolio_tuple(mark_float=26.0, value_float=52.0)]
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    assert len(valuation_dict["position_list"]) == 1
    assert valuation_dict["position_list"][0]["value_float"] == 52.0


def test_cash_only_requires_complete_fresh_account_totals(capture_case):
    capture_case.ib_obj.wrapper.positions[ACCOUNT_STR].clear()
    capture_case.portfolio_list = []
    capture_case.account_value_list = _account_value_list(nav_str="100")
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    assert valuation_dict["position_list"] == []


def test_short_retains_signed_broker_value(capture_case):
    capture_case.ib_obj.wrapper.position(ACCOUNT_STR, _contract_obj(), -2.0, 20.0)
    capture_case.portfolio_list = [_portfolio_tuple(shares_float=-2.0, value_float=-50.0)]
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    assert valuation_dict["position_list"][0]["value_float"] == -50.0


def test_non_usd_base_total_is_not_inferred_from_stock_currency(capture_case):
    capture_case.account_value_list = [
        row_obj._replace(currency="BASE") if row_obj.currency == "USD" else row_obj
        for row_obj in capture_case.account_value_list
    ]
    assert _capture(capture_case).portfolio_valuation_dict["available_bool"] is False


def test_unknown_account_makes_no_valuation_request(capture_case):
    capture_case.ib_obj.wrapper.accounts = [OTHER_ACCOUNT_STR]
    assert _capture(capture_case).portfolio_valuation_dict["available_bool"] is False
    assert capture_case.request_list == []


def _seed_single_account_startup(case_obj, complete_bool=True):
    case_obj.ib_obj.wrapper.accounts = [ACCOUNT_STR]
    request_future = case_obj.ib_obj.wrapper.startReq("accountValues")
    for account_value_obj in case_obj.account_value_list:
        case_obj.ib_obj.wrapper.updateAccountValue(
            account_value_obj.tag, account_value_obj.value,
            account_value_obj.currency, account_value_obj.account,
        )
    for portfolio_tuple in case_obj.portfolio_list:
        case_obj.ib_obj.wrapper.updatePortfolio(*portfolio_tuple)
    if complete_bool:
        case_obj.ib_obj.wrapper.accountDownloadEnd(ACCOUNT_STR)
        assert request_future.done()
    return request_future


def test_single_account_uses_completed_download_on_same_fresh_connection(capture_case):
    _seed_single_account_startup(capture_case)
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    assert valuation_dict["cash_float"] == 100.0
    assert valuation_dict["position_list"][0]["value_float"] == 50.0
    assert capture_case.request_list == []
    assert capture_case.connection_count_int == 1


@pytest.mark.parametrize("cancel_bool", [False, True])
def test_single_account_partial_or_timed_out_startup_stays_unavailable(capture_case, cancel_bool):
    request_future = _seed_single_account_startup(capture_case, complete_bool=False)
    if cancel_bool:
        request_future.cancel()
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is False
    assert "position_list" not in valuation_dict
    assert capture_case.request_list == []


def test_partial_request_does_not_interrupt_existing_account_download(capture_case):
    request_future = capture_case.ib_obj.wrapper.startReq("accountValues")
    assert _capture(capture_case).portfolio_valuation_dict["available_bool"] is False
    assert capture_case.ib_obj.wrapper._futures["accountValues"] is request_future
    assert not request_future.cancelled()
    assert capture_case.request_list == []


def test_early_other_completion_with_partial_target_rows_is_unavailable(capture_case):
    capture_case.ib_obj.wrapper.position(ACCOUNT_STR, _contract_obj("BBB", 2), 3.0, 30.0)
    capture_case.end_account_str = OTHER_ACCOUNT_STR
    assert _capture(capture_case).portfolio_valuation_dict["available_bool"] is False


@pytest.mark.parametrize("tag_str, invalid_str", [
    ("TotalCashValue", "nan"), ("TotalCashValue", "inf"),
    ("TotalCashValue", "1e308"), ("NetLiquidation", "0"),
    ("NetLiquidation", "-10"), ("NetLiquidation", "nan"),
])
def test_invalid_account_totals_fail(capture_case, tag_str, invalid_str):
    capture_case.account_value_list = [
        row_obj._replace(value=invalid_str) if row_obj.tag == tag_str else row_obj
        for row_obj in capture_case.account_value_list
    ]
    assert _capture(capture_case).portfolio_valuation_dict["available_bool"] is False


def test_request_exception_preserves_snapshot_and_removes_own_listeners(capture_case, monkeypatch):
    def fail_request(account_str):
        raise ConnectionError("private broker detail must not reach UI")

    monkeypatch.setattr(capture_case.ib_obj, "reqAccountUpdatesAsync", fail_request)
    snapshot_obj = _capture(capture_case)
    assert snapshot_obj.portfolio_valuation_dict["available_bool"] is False
    assert snapshot_obj.portfolio_valuation_dict["reason_str"] == "IBKR portfolio unavailable"
    assert snapshot_obj.position_amount_map == {"AAA": 2.0}
    assert len(capture_case.ib_obj.updatePortfolioEvent) == 0
    assert len(capture_case.ib_obj.accountValueEvent) == 0


def test_foreign_single_account_rows_are_never_read(capture_case):
    capture_case.account_value_list = _account_value_list(account_str=OTHER_ACCOUNT_STR)
    capture_case.portfolio_list = [_portfolio_tuple(account_str=OTHER_ACCOUNT_STR)]
    _seed_single_account_startup(capture_case)
    assert _capture(capture_case).portfolio_valuation_dict["available_bool"] is False


def test_more_than_twelve_holdings_are_preserved(capture_case):
    capture_case.ib_obj.wrapper.positions[ACCOUNT_STR].clear()
    capture_case.portfolio_list = []
    for holding_int in range(20):
        symbol_str = f"STK{holding_int:02}"
        conid_int = holding_int + 1
        capture_case.ib_obj.wrapper.position(ACCOUNT_STR, _contract_obj(symbol_str, conid_int), 2.0, 20.0)
        capture_case.portfolio_list.append(_portfolio_tuple(symbol_str, conid_int))
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    assert len(valuation_dict["position_list"]) == 20


def test_ib_rounding_tolerance_does_not_recompute_saved_market_value(capture_case):
    capture_case.portfolio_list = [_portfolio_tuple(mark_float=25.0001, value_float=50.01)]
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    assert valuation_dict["position_list"][0]["value_float"] == 50.01


@pytest.mark.parametrize("average_cost_obj,unrealized_pnl_obj", [
    (None, 10.0), (20.0, None), (-1.0, 52.0), (float("nan"), 10.0),
    (20.0, float("nan")), (float("inf"), 10.0), (20.0, float("inf")),
    (True, 48.0), (20.0, True), ("20", 10.0), (20.0, "10"),
    (1e308, -1e308), (10 ** 500, 10.0), (20.0, 10 ** 500),
    (20.0, 0.0), (20.0, 9.0), (0.0, 0.0),
])
def test_missing_invalid_or_inconsistent_pnl_does_not_hide_valid_marks(
    capture_case, average_cost_obj, unrealized_pnl_obj,
):
    capture_case.portfolio_list = [_portfolio_tuple(
        average_cost_obj=average_cost_obj, unrealized_pnl_obj=unrealized_pnl_obj)]
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    row_dict = valuation_dict["position_list"][0]
    assert row_dict["shares_float"] == 2.0 and row_dict["value_float"] == 50.0
    assert "average_cost_float" not in row_dict and "unrealized_pnl_float" not in row_dict


@pytest.mark.parametrize("shares_float,value_float,cost_float,pnl_float", [
    (2.0, 50.0, 30.0, -10.0), (-2.0, -50.0, 30.0, 10.0),
    (-2.0, -50.0, 20.0, -10.0), (2.0, 50.0, 0.0, 50.0),
    (2.0, 50.0, 25.0, 0.0),
])
def test_pnl_keeps_broker_sign_and_zero_cost_without_inventing_percentage(
    capture_case, shares_float, value_float, cost_float, pnl_float,
):
    capture_case.ib_obj.wrapper.position(ACCOUNT_STR, _contract_obj(), shares_float, cost_float)
    capture_case.portfolio_list = [_portfolio_tuple(shares_float=shares_float,
        value_float=value_float, average_cost_obj=cost_float, unrealized_pnl_obj=pnl_float)]
    row_dict = _capture(capture_case).portfolio_valuation_dict["position_list"][0]
    assert row_dict["average_cost_float"] == cost_float
    assert row_dict["unrealized_pnl_float"] == pnl_float
    assert not any("percent" in key_str for key_str in row_dict)


@pytest.mark.parametrize("pnl_float,accepted_bool", [(10.05, True), (10.051, False)])
def test_pnl_five_cent_rounding_boundary(capture_case, pnl_float, accepted_bool):
    capture_case.portfolio_list = [_portfolio_tuple(unrealized_pnl_obj=pnl_float)]
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    row_dict = valuation_dict["position_list"][0]
    assert ("unrealized_pnl_float" in row_dict) is accepted_bool
    if accepted_bool:
        assert row_dict["unrealized_pnl_float"] == pnl_float


@pytest.mark.parametrize("pnl_float,accepted_bool", [(200001.0, True), (200001.01, False)])
def test_pnl_rounding_relative_branch(capture_case, pnl_float, accepted_bool):
    capture_case.ib_obj.wrapper.position(ACCOUNT_STR, _contract_obj(), 20.0, 40000.0)
    capture_case.portfolio_list = [_portfolio_tuple(shares_float=20.0, mark_float=50000.0,
        value_float=1000000.0, average_cost_obj=40000.0, unrealized_pnl_obj=pnl_float)]
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    assert ("unrealized_pnl_float" in valuation_dict["position_list"][0]) is accepted_bool


def test_missing_pnl_on_one_holding_does_not_drop_another_holdings_pnl(capture_case):
    capture_case.ib_obj.wrapper.position(ACCOUNT_STR, _contract_obj("BBB", 2), 2.0, 20.0)
    capture_case.portfolio_list.append(_portfolio_tuple("BBB", 2, unrealized_pnl_obj=None))
    valuation_dict = _capture(capture_case).portfolio_valuation_dict
    assert valuation_dict["available_bool"] is True
    assert valuation_dict["position_list"][0]["unrealized_pnl_float"] == 10.0
    assert "unrealized_pnl_float" not in valuation_dict["position_list"][1]
