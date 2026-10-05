"""Refreshed absence evidence must honestly bound its history and query completeness."""
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.order_evidence import refreshed_order_evidence_dict


NOW_TS = datetime(2026, 10, 5, 21, 0, tzinfo=UTC)
SINCE_TS = datetime(2026, 10, 5, 13, 22, tzinfo=UTC)


class ErrorEvent:
    def __init__(self):
        self.handler_list = []
    def __iadd__(self, handler_fn):
        self.handler_list.append(handler_fn)
        return self
    def __isub__(self, handler_fn):
        self.handler_list.remove(handler_fn)
        return self
    def emit(self, code_int):
        for handler_fn in self.handler_list:
            handler_fn(-1, code_int, "synthetic error")


class EvidenceIB:
    def __init__(self):
        self.TimezoneTWS = "America/New_York"
        self.errorEvent = ErrorEvent()
        self.call_list = []
        self.fill_list = []
        self.completed_list = []
        self.open_list = []
        self.error_code_int = None
        self.connected_bool = True
    def isConnected(self):
        return self.connected_bool
    def managedAccounts(self):
        return ["DU_TEST"]
    def reqAllOpenOrders(self):
        self.call_list.append("all_open")
        return self.open_list
    def reqCompletedOrders(self, apiOnly):
        assert apiOnly is False
        self.call_list.append("completed")
        if self.error_code_int is not None:
            self.errorEvent.emit(self.error_code_int)
        return self.completed_list
    def reqExecutions(self, filter_obj):
        assert filter_obj.acctCode == "DU_TEST"
        self.call_list.append("executions")
        return self.fill_list


@pytest.fixture
def evidence_case(monkeypatch):
    monkeypatch.delenv("ALPHA_IBKR_TWS_TIMEZONE", raising=False)
    clock_list = [NOW_TS, NOW_TS + timedelta(seconds=1)]
    class EvidenceClock:
        @classmethod
        def now(cls, timezone_obj):
            return clock_list.pop(0).astimezone(timezone_obj)
    monkeypatch.setattr("alpha.live.order_evidence.datetime", EvidenceClock)
    broker_obj = EvidenceIB()
    client_obj = IBKRSocketClient(timeout_seconds_float=2.0)
    monkeypatch.setattr(client_obj, "connect", lambda: nullcontext(broker_obj))
    return client_obj, broker_obj, clock_list


def test_complete_refresh_queries_each_source_and_records_verified_timezone(evidence_case):
    client_obj, broker_obj, _ = evidence_case
    result_dict = refreshed_order_evidence_dict(client_obj, "DU_TEST", SINCE_TS)
    assert broker_obj.call_list == ["all_open", "completed", "executions"]
    assert broker_obj.RequestTimeout == 2.0 and broker_obj.RaiseRequestErrors is True
    assert broker_obj.errorEvent.handler_list == []
    assert all(result_dict[key_str] for key_str in ("open_orders_complete_bool", "completed_orders_complete_bool", "executions_complete_bool"))
    assert result_dict["tws_timezone_str"] == "America/New_York"
    assert result_dict["tws_timezone_source_str"] == "ib_async.TimezoneTWS"
    assert datetime.fromisoformat(result_dict["coverage_since_timestamp_str"]) <= SINCE_TS


def test_unknown_tws_timezone_does_not_claim_current_day_history(evidence_case):
    client_obj, broker_obj, _ = evidence_case
    broker_obj.TimezoneTWS = ""
    result_dict = refreshed_order_evidence_dict(client_obj, "DU_TEST", SINCE_TS)
    assert not result_dict["executions_complete_bool"]
    assert not result_dict["completed_orders_complete_bool"]
    assert result_dict["tws_timezone_source_str"] == "unknown"
    assert result_dict["coverage_since_timestamp_str"] == result_dict["refreshed_timestamp_str"]


def test_explicit_verified_timezone_only_changes_evidence_coverage(evidence_case, monkeypatch):
    client_obj, broker_obj, _ = evidence_case
    broker_obj.TimezoneTWS = ""
    monkeypatch.setenv("ALPHA_IBKR_TWS_TIMEZONE", "Asia/Tokyo")
    result_dict = refreshed_order_evidence_dict(client_obj, "DU_TEST", SINCE_TS)
    assert result_dict["tws_timezone_source_str"] == "ALPHA_IBKR_TWS_TIMEZONE"
    assert result_dict["tws_timezone_str"] == "Asia/Tokyo"
    assert broker_obj.TimezoneTWS == ""
    # Tokyo has rolled to Oct 6, so today's TWS history cannot cover this morning's order.
    assert datetime.fromisoformat(result_dict["coverage_since_timestamp_str"]) == datetime(2026, 10, 5, 15, 0, tzinfo=UTC)
    assert datetime.fromisoformat(result_dict["coverage_since_timestamp_str"]) > SINCE_TS


@pytest.mark.parametrize("timezone_str", ["Invalid/Zone", "Eastern Standard Time"])
def test_invalid_explicit_timezone_fails_before_queries(evidence_case, monkeypatch, timezone_str):
    client_obj, broker_obj, _ = evidence_case
    monkeypatch.setenv("ALPHA_IBKR_TWS_TIMEZONE", timezone_str)
    with pytest.raises(ValueError, match="verified IANA"):
        refreshed_order_evidence_dict(client_obj, "DU_TEST", SINCE_TS)
    assert broker_obj.call_list == []


def test_refresh_crossing_tws_midnight_does_not_claim_complete_history(evidence_case):
    client_obj, broker_obj, clock_list = evidence_case
    broker_obj.TimezoneTWS = "Asia/Tokyo"
    clock_list[:] = [datetime(2026, 10, 5, 14, 59, 59, tzinfo=UTC), datetime(2026, 10, 5, 15, 0, 1, tzinfo=UTC)]
    result_dict = refreshed_order_evidence_dict(client_obj, "DU_TEST", SINCE_TS)
    assert not result_dict["completed_orders_complete_bool"]
    assert not result_dict["executions_complete_bool"]


@pytest.mark.parametrize("error_code_int", [1100, 502, 504, 162])
def test_broker_error_during_refresh_cannot_become_absence_proof(evidence_case, error_code_int):
    client_obj, broker_obj, _ = evidence_case
    broker_obj.error_code_int = error_code_int
    with pytest.raises(RuntimeError, match="incomplete"):
        refreshed_order_evidence_dict(client_obj, "DU_TEST", SINCE_TS)
    assert broker_obj.errorEvent.handler_list == []


def test_informational_farm_notification_does_not_invalidate_completed_query(evidence_case):
    client_obj, broker_obj, _ = evidence_case
    broker_obj.error_code_int = 2104
    assert refreshed_order_evidence_dict(client_obj, "DU_TEST", SINCE_TS)["executions_complete_bool"]


def test_request_timeout_propagates_and_removes_event_handler(evidence_case, monkeypatch):
    client_obj, broker_obj, _ = evidence_case
    def timeout_fn(apiOnly):
        raise TimeoutError("incomplete completed-order query")
    monkeypatch.setattr(broker_obj, "reqCompletedOrders", timeout_fn)
    with pytest.raises(TimeoutError):
        refreshed_order_evidence_dict(client_obj, "DU_TEST", SINCE_TS)
    assert broker_obj.errorEvent.handler_list == []


def test_orders_and_executions_are_account_scoped_with_request_identity(evidence_case):
    client_obj, broker_obj, _ = evidence_case
    def trade_obj(account_str):
        return SimpleNamespace(order=SimpleNamespace(account=account_str, orderRef="request", orderId=3, permId=103),
            orderStatus=SimpleNamespace(orderId=3, permId=103, status="Cancelled"), contract=SimpleNamespace(symbol="BRK B"))
    broker_obj.completed_list = [trade_obj("DU_TEST"), trade_obj("DU_OTHER")]
    broker_obj.fill_list = [SimpleNamespace(execution=SimpleNamespace(acctNumber="DU_TEST", execId="execution-1",
        orderRef="request", permId=103, orderId=3), contract=SimpleNamespace(symbol="BRK B"), time=SINCE_TS)]
    result_dict = refreshed_order_evidence_dict(client_obj, "DU_TEST", SINCE_TS)
    assert len(result_dict["order_row_list"]) == len(result_dict["execution_row_list"]) == 1
    assert result_dict["order_row_list"][0]["asset_str"] == "BRK.B"
    assert result_dict["execution_row_list"][0]["broker_execution_id_str"] == "execution-1"
    assert result_dict["execution_row_list"][0]["order_request_key_str"] == "request"


def test_disconnection_during_query_fails_closed(evidence_case, monkeypatch):
    client_obj, broker_obj, _ = evidence_case
    def disconnect_fn(filter_obj):
        broker_obj.connected_bool = False
        return []
    monkeypatch.setattr(broker_obj, "reqExecutions", disconnect_fn)
    with pytest.raises(RuntimeError, match="incomplete"):
        refreshed_order_evidence_dict(client_obj, "DU_TEST", SINCE_TS)
