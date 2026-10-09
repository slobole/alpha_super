from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from alpha.live import ibkr_socket_client
import test_live_ibkr_socket_client as fixture_module


@pytest.fixture(scope="module")
def baseline_client_class():
    repository_path_obj = Path(__file__).resolve().parents[1]
    source_str = subprocess.check_output(
        ["git", "-c", f"safe.directory={repository_path_obj.as_posix()}",
         "-c", "core.fsmonitor=false", "show", "54b417f:alpha/live/ibkr_socket_client.py"],
        cwd=repository_path_obj, text=True, encoding="utf-8",
    )
    namespace_dict = {"__name__": "baseline_54b417f_ibkr_socket_client"}
    exec(compile(source_str, "54b417f:alpha/live/ibkr_socket_client.py", "exec"), namespace_dict)
    return namespace_dict["IBKRSocketClient"]


def _freeze_parity_clock(monkeypatch, baseline_client_class, observation_ts):
    class ClockMeta(type):
        def __instancecheck__(cls, timestamp_obj):
            return isinstance(timestamp_obj, datetime)

    class ClockDateTime(datetime, metaclass=ClockMeta):
        @classmethod
        def now(cls, tz=None):
            return observation_ts.astimezone(tz or UTC)

    monkeypatch.setattr(ibkr_socket_client, "datetime", ClockDateTime)
    monkeypatch.setitem(baseline_client_class.get_account_snapshot.__globals__, "datetime", ClockDateTime)


def _fill_obj(order_id_int, timestamp_ts, *, side_str="BOT", execution_id_str="execution-1"):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol="AAPL"), time=timestamp_ts,
        execution=SimpleNamespace(acctNumber="SIM_pod", orderId=order_id_int, permId=1000 + order_id_int,
            side=side_str, shares=4, price=101.25, execId=execution_id_str),
    )


class _SnapshotIB:
    def __init__(self, source_str):
        self.call_list = []
        self.open_trade_list = [fixture_module._snapshot_trade_obj(1, status_str="Submitted")]
        self.completed_trade_list = [
            fixture_module._snapshot_trade_obj(1),
            fixture_module._snapshot_trade_obj(2, account_str="OTHER"),
            fixture_module._snapshot_trade_obj(3, order_ref_str="unrelated:AAPL:1"),
        ]
        if source_str == "open":
            self.completed_trade_list = []
        elif source_str == "completed":
            self.open_trade_list = []
        elif source_str == "without_logs":
            for trade_obj in self.open_trade_list + self.completed_trade_list:
                trade_obj.log = []
        self.fill_list = [
            _fill_obj(1, datetime(2024, 1, 3, 15, 0, tzinfo=UTC)),
            _fill_obj(3, datetime(2024, 1, 3, 15, 0, tzinfo=UTC), side_str="SLD", execution_id_str="execution-3"),
            _fill_obj(1, datetime(2024, 1, 2, 15, 0, tzinfo=UTC), execution_id_str="older-execution"),
        ]

    def reqOpenOrders(self):
        self.call_list.append("reqOpenOrders")
        return self.open_trade_list

    def reqAllOpenOrders(self):
        raise AssertionError("Default monthly calls must keep the legacy order query")

    def reqCompletedOrders(self, apiOnly):
        assert apiOnly is False
        self.call_list.append("reqCompletedOrders")
        return self.completed_trade_list

    def reqExecutions(self, execution_filter_obj):
        assert execution_filter_obj.acctCode == "SIM_pod"
        self.call_list.append("reqExecutions")
        return self.fill_list

    def accountSummary(self, account):
        self.call_list.append("accountSummary")
        return [SimpleNamespace(account=account, tag=tag_str, value=value_str) for tag_str, value_str in (
            ("TotalCashValue", "1000"), ("NetLiquidation", "10000"), ("AvailableFunds", "9000"),
            ("ExcessLiquidity", "8000"), ("Cushion", "0.8"),
        )] + [SimpleNamespace(account="OTHER", tag="TotalCashValue", value="999999")]

    def positions(self, account):
        assert account == "SIM_pod"
        self.call_list.append("positions")
        return [SimpleNamespace(contract=SimpleNamespace(symbol=asset_str), position=amount_int)
                for asset_str, amount_int in (("AAPL", 5), ("IEF", 40))]


@pytest.mark.parametrize("source_str", ["open", "completed", "both", "without_logs"])
@pytest.mark.parametrize("filter_dict", [
    {}, {"submission_key_str": "batch"}, {"allowed_broker_order_id_set": {"1"}},
    {"allowed_broker_order_id_set": {"1001"}},
])
def test_default_public_order_history_matches_54b417f_full_payloads(
    monkeypatch, baseline_client_class, source_str, filter_dict,
):
    result_list = []
    call_list = []
    for client_class in (baseline_client_class, ibkr_socket_client.IBKRSocketClient):
        broker_obj = _SnapshotIB(source_str)
        client_obj = client_class()
        monkeypatch.setattr(client_obj, "connect", lambda: nullcontext(broker_obj))
        result_list.append(client_obj.get_recent_order_state_snapshot(
            "SIM_pod", datetime(2024, 1, 3, 14, 0, tzinfo=UTC), **filter_dict))
        call_list.append(broker_obj.call_list)
    assert result_list[1] == result_list[0]
    assert call_list[1] == call_list[0] == ["reqOpenOrders", "reqCompletedOrders", "reqExecutions"]
    for record_obj in result_list[1][0] + result_list[1][1]:
        assert "snapshot_source_str" not in record_obj.raw_payload_dict
        assert "open_order_observed_bool" not in record_obj.raw_payload_dict


def test_default_account_snapshot_and_fills_match_54b417f(monkeypatch, baseline_client_class):
    observation_ts = datetime(2024, 1, 3, 15, 30, tzinfo=UTC)
    _freeze_parity_clock(monkeypatch, baseline_client_class, observation_ts)
    result_list = []
    call_list = []
    for client_class in (baseline_client_class, ibkr_socket_client.IBKRSocketClient):
        broker_obj = _SnapshotIB("both")
        client_obj = client_class()
        monkeypatch.setattr(client_obj, "connect", lambda: nullcontext(broker_obj))
        result_list.append((client_obj.get_account_snapshot("SIM_pod"),
            client_obj.get_recent_fill_list("SIM_pod", observation_ts - timedelta(hours=1))))
        call_list.append(broker_obj.call_list)
    assert result_list[1] == result_list[0]
    assert len(result_list[1][1]) == 2
    assert call_list[1] == call_list[0] == ["accountSummary", "positions", "reqOpenOrders", "reqExecutions"]


class _SubmitIB(fixture_module._ExpirySubmitIB):
    def __init__(self, clock_list, fill_bool):
        super().__init__(clock_list)
        self.fill_bool = fill_bool

    def placeOrder(self, contract_obj, order_obj):
        trade_obj = super().placeOrder(contract_obj, order_obj)
        if self.fill_bool:
            trade_obj.fills = [_fill_obj(order_obj.orderId, self.clock_list[0])]
            trade_obj.orderStatus.filled = 4
            trade_obj.orderStatus.remaining = order_obj.totalQuantity - 4
            trade_obj.orderStatus.avgFillPrice = 101.25
        return trade_obj

    def reqExecutions(self, execution_filter_obj):
        assert execution_filter_obj.acctCode == "SIM_pod"
        return [fill_obj for trade_obj in self.trade_list for fill_obj in trade_obj.fills]


@pytest.mark.parametrize("order_type_str", ["MOO", "MOC", "MKT", "LMT"])
@pytest.mark.parametrize("fill_bool", [False, True])
def test_default_submission_matches_54b417f_full_orders_records_events_fills_and_acks(
    monkeypatch, baseline_client_class, order_type_str, fill_bool,
):
    observation_ts = datetime(2024, 1, 3, 14, 25, tzinfo=UTC)
    _freeze_parity_clock(monkeypatch, baseline_client_class, observation_ts)
    request_obj = replace(fixture_module._funding_request_obj(), broker_order_type_str=order_type_str,
        limit_price_float=123.45 if order_type_str == "LMT" else None)
    result_list = []
    order_list = []
    for client_class in (baseline_client_class, ibkr_socket_client.IBKRSocketClient):
        broker_obj = _SubmitIB([observation_ts], fill_bool)
        client_obj = client_class()
        monkeypatch.setattr(client_obj, "connect", lambda: nullcontext(broker_obj))
        result_list.append(client_obj.submit_order_request_list("SIM_pod", [request_obj], observation_ts))
        order_list.append([asdict(order_obj) for order_obj in broker_obj.placed_order_list])
    assert asdict(result_list[1]) == asdict(result_list[0])
    assert order_list[1] == order_list[0]
    assert bool(result_list[1].broker_order_fill_list) is fill_bool
    assert result_list[1].submit_ack_status_str == "complete"
    assert "submission_deadline_timestamp_str" not in result_list[1].broker_order_record_list[0].raw_payload_dict


def test_explicit_submission_deadline_remains_in_daily_local_record_payload(monkeypatch):
    observation_ts = datetime(2024, 1, 3, 14, 25, tzinfo=UTC)
    broker_obj = _SubmitIB([observation_ts], False)
    client_obj = fixture_module._expiry_client_obj(monkeypatch, broker_obj)
    request_obj = replace(fixture_module._funding_request_obj(),
        submission_deadline_timestamp_str="2024-01-03T14:28:00+00:00")
    result_obj = client_obj.submit_order_request_list("SIM_pod", [request_obj], observation_ts)
    assert result_obj.submit_ack_status_str == "complete"
    assert result_obj.broker_order_record_list[0].raw_payload_dict["submission_deadline_timestamp_str"] == (
        request_obj.submission_deadline_timestamp_str)
