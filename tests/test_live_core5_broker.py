from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from alpha.live import core5_broker as broker_module
from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.models import BrokerOrderRequest


ROUTE_STR = "U_CORE5_TEST"


def _account_value_obj(tag_str, value_obj, currency_str="USD", account_str=ROUTE_STR, model_str=""):
    return SimpleNamespace(tag=tag_str, value=value_obj, currency=currency_str, account=account_str, modelCode=model_str)


def _position_obj(asset_str="DBC", shares_float=-5.0, account_str=ROUTE_STR, **contract_dict):
    return SimpleNamespace(account=account_str, position=shares_float,
        contract=SimpleNamespace(**{"symbol": asset_str, "secType": "STK", "currency": "USD", "multiplier": "",
                                   **contract_dict}))


def _request_obj(asset_str="SPY", amount_float=10.0, key_str="batch:SPY:1"):
    return BrokerOrderRequest(
        release_id_str="core5", pod_id_str="core5", account_route_str=ROUTE_STR,
        submission_key_str="batch", order_request_key_str=key_str, asset_str=asset_str,
        broker_order_type_str="MOO", order_class_str="MarketOrder", unit_str="shares",
        amount_float=amount_float, target_bool=False, trade_id_int=None,
        sizing_reference_price_float=100.0, portfolio_value_float=100_000.0,
    )


def _preview_obj(init_float=100.0, maint_float=80.0):
    return SimpleNamespace(status="PreSubmitted", initMarginBefore="1000", initMarginAfter=str(1000 + init_float),
        initMarginChange=str(init_float), maintMarginBefore="800", maintMarginAfter=str(800 + maint_float),
        maintMarginChange=str(maint_float), equityWithLoanBefore="100000", equityWithLoanAfter="99999", warningText="")


class _FakeBroker:
    def __init__(self):
        self.account_list = [ROUTE_STR]
        self.wrapper = SimpleNamespace(_futures={})
        self.summary_list = [_account_value_obj("TotalCashValue", "12000"), _account_value_obj("NetLiquidation", "100000")]
        self.account_value_list = [
            _account_value_obj("TradingType-S", "STKNOPT", ""),
            _account_value_obj("AccountType", "INDIVIDUAL", ""),
            _account_value_obj("accountReady", "true", ""),
            _account_value_obj("AvailableFunds", "1000"),
            _account_value_obj("ExcessLiquidity", "800"),
        ]
        self.position_list = []
        self.open_trade_list = []
        self.preview_result_list = []
        self.preview_list = []
        self.subscription_list = []
        self.cancelled_subscription_list = []
        self.shortable_shares_float = 1000.0
        self.download_list = []
        self.account_read_count_int = 0
        self.after_first_read_fn = None
        self.contract_mutation_dict = {}
        self.open_read_count_int = 0

    def managedAccounts(self):
        return self.account_list

    def accountSummary(self, account):
        assert account == ROUTE_STR
        return self.summary_list

    def positions(self, account):
        assert account == ROUTE_STR
        return self.position_list

    def reqAllOpenOrders(self):
        self.open_read_count_int += 1
        return self.open_trade_list

    def reqOpenOrders(self):
        raise AssertionError("CORE5 must inspect every client, not only current-client open orders")

    def reqAccountUpdates(self, account):
        self.download_list.append(account)

    def accountValues(self, account):
        assert account == ROUTE_STR
        self.account_read_count_int += 1
        if self.account_read_count_int > 1 and self.after_first_read_fn:
            self.after_first_read_fn(self)
        return self.account_value_list

    def qualifyContracts(self, *contract_tuple):
        for contract_obj in contract_tuple:
            for field_str, value_obj in self.contract_mutation_dict.items():
                setattr(contract_obj, field_str, value_obj)
        return list(contract_tuple)

    def whatIfOrder(self, contract_obj, order_obj):
        assert order_obj.whatIf is True
        self.preview_list.append((contract_obj.symbol, order_obj))
        return self.preview_result_list.pop(0) if self.preview_result_list else _preview_obj()

    def reqMktData(self, contract_obj, genericTickList, snapshot):
        assert genericTickList == "236" and snapshot is False and contract_obj.symbol == "DBC"
        self.subscription_list.append(contract_obj)
        return SimpleNamespace(shortableShares=self.shortable_shares_float)

    def sleep(self, seconds_float):
        assert 0 < seconds_float <= 2

    def cancelMktData(self, contract_obj):
        self.cancelled_subscription_list.append(contract_obj)
        return True

    def placeOrder(self, *argument_tuple, **keyword_dict):
        raise AssertionError("CORE5 evidence must never submit orders")

    def cancelOrder(self, *argument_tuple, **keyword_dict):
        raise AssertionError("CORE5 evidence must never cancel orders")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(IBKRSocketClient, "connect", lambda *argument_tuple: pytest.fail("Real broker forbidden"))


@pytest.fixture
def broker_case_tuple(monkeypatch):
    broker_obj = _FakeBroker()
    socket_obj = IBKRSocketClient()
    socket_obj.test_connect_count_int = 0

    @contextmanager
    def connect_fake():
        socket_obj.test_connect_count_int += 1
        yield broker_obj

    monkeypatch.setattr(socket_obj, "connect", connect_fake)
    return socket_obj, broker_obj


def test_snapshot_requires_one_connection_and_keeps_signed_cash_and_other_client_orders(broker_case_tuple):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.summary_list[0].value = "-12000"
    broker_obj.summary_list.extend([
        _account_value_obj("TotalCashValue", "999999", "EUR"),
        _account_value_obj("TotalCashValue", "999999", account_str="OTHER"),
        _account_value_obj("TotalCashValue", "999999", model_str="MODEL"),
    ])
    broker_obj.position_list = [_position_obj(), _position_obj("SPY", 10),
                               _position_obj("UNSUPPORTED", 20, account_str="OTHER")]
    broker_obj.open_trade_list = [SimpleNamespace(order=SimpleNamespace(account=ROUTE_STR, orderId=91, clientId=99)),
                                  SimpleNamespace(order=SimpleNamespace(account="OTHER", orderId=92))]
    before_ts = datetime.now(tz=UTC)
    snapshot_obj = broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR)
    assert socket_obj.test_connect_count_int == 1
    assert broker_obj.open_read_count_int == 1 and broker_obj.RequestTimeout == 4
    assert snapshot_obj.cash_float == -12000 and snapshot_obj.net_liq_float == snapshot_obj.total_value_float == 100000
    assert snapshot_obj.position_amount_map == {"DBC": -5, "SPY": 10}
    assert snapshot_obj.open_order_id_list == ["91"]
    assert snapshot_obj.available_funds_float is None
    assert before_ts <= snapshot_obj.snapshot_timestamp_ts <= datetime.now(tz=UTC)


def test_snapshot_optional_valuation_uses_same_connection_and_original_observation(broker_case_tuple, monkeypatch):
    socket_obj, broker_obj = broker_case_tuple
    valuation_dict = {"available_bool": True}
    monkeypatch.setattr(socket_obj, "_capture_portfolio_valuation_dict",
        lambda current_broker_obj, route_str, position_dict: valuation_dict
        if current_broker_obj is broker_obj and route_str == ROUTE_STR and position_dict == {} else pytest.fail("Wrong capture"))
    snapshot_obj = broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR, True)
    assert snapshot_obj.portfolio_valuation_dict is valuation_dict
    assert socket_obj.test_connect_count_int == 1


@pytest.mark.parametrize("tag_str,value_obj", [
    ("TotalCashValue", None), ("TotalCashValue", ""), ("TotalCashValue", "nan"), ("TotalCashValue", "inf"),
    ("TotalCashValue", "1.7976931348623157e308"), ("TotalCashValue", True),
    ("NetLiquidation", "0"), ("NetLiquidation", "-1"), ("NetLiquidation", "nan"),
])
def test_snapshot_rejects_missing_invalid_or_sentinel_totals(broker_case_tuple, tag_str, value_obj):
    socket_obj, broker_obj = broker_case_tuple
    next(value_row_obj for value_row_obj in broker_obj.summary_list if value_row_obj.tag == tag_str).value = value_obj
    with pytest.raises(ValueError):
        broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR)


@pytest.mark.parametrize("mutation_str", ["missing_cash", "ambiguous_cash", "wrong_currency", "incomplete_positions", "unsupported_sdk", "invisible"])
def test_snapshot_fails_closed_on_incomplete_account_evidence(broker_case_tuple, mutation_str):
    socket_obj, broker_obj = broker_case_tuple
    if mutation_str == "missing_cash":
        broker_obj.summary_list.pop(0)
    elif mutation_str == "ambiguous_cash":
        broker_obj.summary_list.append(_account_value_obj("TotalCashValue", "12000"))
    elif mutation_str == "wrong_currency":
        broker_obj.summary_list[0].currency = "BASE"
    elif mutation_str == "incomplete_positions":
        broker_obj.wrapper._futures["positions"] = object()
    elif mutation_str == "unsupported_sdk":
        del broker_obj.wrapper
    else:
        broker_obj.account_list = ["OTHER"]
    with pytest.raises(ValueError):
        broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR)


@pytest.mark.parametrize("position_obj", [
    _position_obj(shares_float=-.5), _position_obj(shares_float=float("nan")),
    _position_obj("SPY", -1), _position_obj("QQQ", 1),
    _position_obj(currency="EUR"), _position_obj(secType="OPT"), _position_obj(multiplier="100"),
])
def test_snapshot_rejects_unsupported_or_fractional_positions(broker_case_tuple, position_obj):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.position_list = [position_obj]
    with pytest.raises(ValueError):
        broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR)


def test_snapshot_rejects_duplicate_symbol_contracts(broker_case_tuple):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.position_list = [_position_obj(), _position_obj()]
    with pytest.raises(ValueError, match="unique"):
        broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR)


@pytest.mark.parametrize("trading_type_str", ["STKNOPT", "STKMRGN"])
def test_funding_sums_positive_margin_and_does_not_credit_sales(broker_case_tuple, trading_type_str):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.account_value_list[0].value = trading_type_str
    broker_obj.position_list = [_position_obj("BIL", 100)]
    broker_obj.preview_result_list = [_preview_obj(-300, -200), _preview_obj(500, 300), _preview_obj(400, 400)]
    evidence_dict = broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR,
        [_request_obj("BIL", -100, "BIL"), _request_obj(), _request_obj("GLD", 10, "GLD")], {"BIL": 100})
    assert evidence_dict["total_initial_margin_increase_float"] == 900
    assert evidence_dict["total_maintenance_margin_increase_float"] == 700
    assert evidence_dict["total_preview_equity_debit_float"] == 3
    assert evidence_dict["required_initial_margin_float"] == 903
    assert evidence_dict["required_maintenance_margin_float"] == 703
    assert all(preview_dict["preview_equity_debit_float"] == 1 for preview_dict in evidence_dict["preview_list"])
    assert evidence_dict["pending_sell_credit_float"] == 0
    assert evidence_dict["basket_margin_guaranteed_bool"] is False
    assert evidence_dict["borrow_reserved_bool"] is False and evidence_dict["borrow_rate_float"] is None
    assert socket_obj.test_connect_count_int == 1
    assert broker_obj.download_list == []
    assert [(asset_str, order_obj.action, order_obj.totalQuantity, order_obj.tif, order_obj.orderType)
            for asset_str, order_obj in broker_obj.preview_list] == [
                ("BIL", "SELL", 100, "OPG", "MKT"),
                ("SPY", "BUY", 10, "OPG", "MKT"), ("GLD", "BUY", 10, "OPG", "MKT")]


@pytest.mark.parametrize("initial_float,amount_list,expected_list,short_float", [
    (10, [-10, -5], [-10, -15], 5),
    (-10, [10, 5], [10, 15], 0),
    (-10, [-5], [-5], 5),
    (0, [-5], [-5], 5),
    (-10, [5], [5], 0),
])
def test_dbc_flip_previews_use_current_account_cumulative_shares_and_incremental_inventory(
    broker_case_tuple, initial_float, amount_list, expected_list, short_float,
):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.position_list = [_position_obj(shares_float=initial_float)] if initial_float else []
    request_list = [_request_obj("DBC", amount_float, f"DBC:{index_int}")
                    for index_int, amount_float in enumerate(amount_list)]
    evidence_dict = broker_module.get_core5_funding_evidence(
        socket_obj, ROUTE_STR, request_list, {"DBC": initial_float})
    assert [preview_dict["preview_amount_float"] for preview_dict in evidence_dict["preview_list"]] == expected_list
    assert [order_obj.orderRef for _, order_obj in broker_obj.preview_list] == [
        preview_dict["order_request_key_str"] for preview_dict in evidence_dict["preview_list"]]
    assert evidence_dict["required_additional_dbc_short_shares_float"] == short_float
    assert len(broker_obj.subscription_list) == int(short_float > 0)
    assert broker_obj.cancelled_subscription_list == broker_obj.subscription_list


@pytest.mark.parametrize("shortable_obj", [None, "", float("nan"), float("inf"), 1.7976931348623157e308, -1, 4])
def test_increased_dbc_short_fails_closed_and_always_releases_subscription(broker_case_tuple, shortable_obj):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.shortable_shares_float = shortable_obj
    with pytest.raises(ValueError, match="shortable"):
        broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR, [_request_obj("DBC", -5, "DBC")], {})
    assert len(broker_obj.subscription_list) == 1
    assert broker_obj.cancelled_subscription_list == broker_obj.subscription_list


@pytest.mark.parametrize("tag_str,currency_str,value_obj", [
    ("AvailableFunds", "BASE", "1000"), ("AvailableFunds", "USD", "nan"),
    ("AvailableFunds", "USD", "1.7976931348623157e308"), ("ExcessLiquidity", "USD", ""),
    ("ExcessLiquidity", "EUR", "1000"), ("ExcessLiquidity", "USD", "inf"),
])
def test_funding_rejects_invalid_margin_capacity(broker_case_tuple, tag_str, currency_str, value_obj):
    socket_obj, broker_obj = broker_case_tuple
    account_value_obj = next(value_obj for value_obj in broker_obj.account_value_list if value_obj.tag == tag_str)
    account_value_obj.currency = currency_str
    account_value_obj.value = value_obj
    with pytest.raises(ValueError):
        broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR, [_request_obj()], {})


@pytest.mark.parametrize("init_float,maint_float", [(501, 0), (0, 401)])
def test_funding_checks_combined_not_merely_individual_margin(broker_case_tuple, init_float, maint_float):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.preview_result_list = [_preview_obj(init_float, maint_float), _preview_obj(init_float, maint_float)]
    with pytest.raises(ValueError, match="combined"):
        broker_module.get_core5_funding_evidence(
            socket_obj, ROUTE_STR, [_request_obj(), _request_obj("GLD", 10, "GLD")], {})


def test_combined_equity_debits_are_included_at_exact_margin_boundary(broker_case_tuple):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.preview_result_list = [_preview_obj(499, 399), _preview_obj(499, 399)]
    evidence_dict = broker_module.get_core5_funding_evidence(
        socket_obj, ROUTE_STR, [_request_obj(), _request_obj("GLD", 10, "GLD")], {})
    assert evidence_dict["total_preview_equity_debit_float"] == 2
    assert evidence_dict["required_initial_margin_float"] == evidence_dict["available_funds_float"] == 1000
    assert evidence_dict["required_maintenance_margin_float"] == evidence_dict["excess_liquidity_float"] == 800


@pytest.mark.parametrize("init_float,maint_float", [(499, 0), (0, 399)])
def test_combined_equity_debits_block_otherwise_affordable_preview_basket(broker_case_tuple, init_float, maint_float):
    socket_obj, broker_obj = broker_case_tuple
    first_preview_obj = _preview_obj(init_float, maint_float)
    second_preview_obj = _preview_obj(init_float, maint_float)
    second_preview_obj.equityWithLoanAfter = "99998"
    broker_obj.preview_result_list = [first_preview_obj, second_preview_obj]
    with pytest.raises(ValueError, match="combined positive margin increases and equity debits"):
        broker_module.get_core5_funding_evidence(
            socket_obj, ROUTE_STR, [_request_obj(), _request_obj("GLD", 10, "GLD")], {})


def test_reducing_preview_never_offsets_another_margin_increase(broker_case_tuple):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.preview_result_list = [_preview_obj(-500, -400), _preview_obj(1001, 0)]
    with pytest.raises(ValueError, match="combined"):
        broker_module.get_core5_funding_evidence(
            socket_obj, ROUTE_STR, [_request_obj(), _request_obj("GLD", 10, "GLD")], {})


def test_selling_an_existing_position_cannot_hide_margin_increase(broker_case_tuple):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.account_value_list[0].value = "STKMRGN"
    broker_obj.position_list = [_position_obj("BIL", 100)]
    broker_obj.preview_result_list = [_preview_obj(950, 0), _preview_obj(100, 0)]
    with pytest.raises(ValueError, match="combined"):
        broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR,
            [_request_obj("BIL", -100, "BIL"), _request_obj()], {"BIL": 100})
    assert [order_obj.action for _, order_obj in broker_obj.preview_list] == ["SELL", "BUY"]


@pytest.mark.parametrize("field_str,value_obj", [
    ("status", "Inactive"), ("initMarginChange", ""), ("maintMarginChange", "nan"),
    ("initMarginBefore", "1.7976931348623157e308"), ("initMarginAfter", "1200"),
    ("equityWithLoanAfter", "1"), ("maintMarginAfter", "-1"), ("equityWithLoanBefore", None),
])
def test_funding_requires_complete_consistent_preview(broker_case_tuple, field_str, value_obj):
    socket_obj, broker_obj = broker_case_tuple
    preview_obj = _preview_obj()
    setattr(preview_obj, field_str, value_obj)
    broker_obj.preview_result_list = [preview_obj]
    with pytest.raises(ValueError, match="preview"):
        broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR, [_request_obj()], {})


@pytest.mark.parametrize("trading_type_str", ["", "STKCASH", "IRAMRGN", "INDIVIDUAL", "unknown", "PMRGN", "GPMRGN"])
def test_funding_uses_trading_type_not_broad_account_type(broker_case_tuple, trading_type_str):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.account_value_list[0].value = trading_type_str
    with pytest.raises(ValueError, match="verified standard margin"):
        broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR, [_request_obj()], {})
    assert not broker_obj.preview_list


@pytest.mark.parametrize("mutation_str", ["duplicate_margin", "unready", "pending_download", "changed_position", "other_client_order"])
def test_funding_rejects_ambiguous_or_changed_account(broker_case_tuple, mutation_str):
    socket_obj, broker_obj = broker_case_tuple
    if mutation_str == "duplicate_margin":
        broker_obj.account_value_list.append(_account_value_obj("AvailableFunds", "1000"))
    elif mutation_str == "unready":
        broker_obj.account_value_list[2].value = "false"
    elif mutation_str == "pending_download":
        broker_obj.wrapper._futures["accountValues"] = object()
    elif mutation_str == "changed_position":
        broker_obj.position_list = [_position_obj()]
    else:
        broker_obj.open_trade_list = [SimpleNamespace(order=SimpleNamespace(account=ROUTE_STR, orderId=3, clientId=99))]
    with pytest.raises(ValueError):
        broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR, [_request_obj()], {})
    assert not broker_obj.preview_list


def test_multi_account_funding_downloads_only_routed_account(broker_case_tuple):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.account_list.append("OTHER")
    broker_obj.account_value_list.append(_account_value_obj("AvailableFunds", "1", account_str="OTHER"))
    evidence_dict = broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR, [_request_obj()], {})
    assert broker_obj.download_list == [ROUTE_STR]
    assert evidence_dict["available_funds_float"] == 1000


def test_margin_deteriorating_during_preview_blocks_batch(broker_case_tuple):
    socket_obj, broker_obj = broker_case_tuple

    def reduce_margin(current_broker_obj):
        next(value_obj for value_obj in current_broker_obj.account_value_list if value_obj.tag == "AvailableFunds").value = "99"

    broker_obj.after_first_read_fn = reduce_margin
    with pytest.raises(ValueError, match="combined"):
        broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR, [_request_obj()], {})


@pytest.mark.parametrize("mutation_str", ["unready", "margin_type", "position", "open_order"])
def test_final_read_rejects_account_change_during_previews(broker_case_tuple, mutation_str):
    socket_obj, broker_obj = broker_case_tuple

    def change_account(current_broker_obj):
        if mutation_str == "unready":
            current_broker_obj.account_value_list[2].value = "false"
        elif mutation_str == "margin_type":
            current_broker_obj.account_value_list[0].value = "PMRGN"
        elif mutation_str == "position":
            current_broker_obj.position_list = [_position_obj()]
        else:
            current_broker_obj.open_trade_list = [SimpleNamespace(order=SimpleNamespace(account=ROUTE_STR, orderId=9))]

    broker_obj.after_first_read_fn = change_account
    with pytest.raises(ValueError):
        broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR, [_request_obj()], {})
    assert len(broker_obj.preview_list) == 1


@pytest.mark.parametrize("mutation_dict", [
    {"account_route_str": "OTHER"}, {"unit_str": "percent"}, {"target_bool": True},
    {"asset_str": "QQQ"}, {"amount_float": .5}, {"amount_float": float("nan")},
    {"broker_order_type_str": "MKT"}, {"order_request_key_str": ""},
])
def test_invalid_order_contract_rejected_before_connect(broker_case_tuple, mutation_dict):
    socket_obj, _ = broker_case_tuple
    with pytest.raises(ValueError):
        broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR, [replace(_request_obj(), **mutation_dict)], {})
    assert socket_obj.test_connect_count_int == 0


def test_duplicate_request_identity_rejected_before_connect(broker_case_tuple):
    socket_obj, _ = broker_case_tuple
    with pytest.raises(ValueError):
        broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR, [_request_obj(), _request_obj()], {})
    assert socket_obj.test_connect_count_int == 0


def test_opposing_legs_rejected_before_connect(broker_case_tuple):
    socket_obj, _ = broker_case_tuple
    with pytest.raises(ValueError, match="opposing"):
        broker_module.get_core5_funding_evidence(socket_obj, ROUTE_STR,
            [_request_obj("DBC", -5, "DBC:1"), _request_obj("DBC", 5, "DBC:2")], {})
    assert socket_obj.test_connect_count_int == 0
