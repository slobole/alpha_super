"""Fresh DAILY account state and exact owned-order cancellation; no fill history."""
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from time import monotonic

from alpha.live.models import BrokerSnapshot


@dataclass(frozen=True)
class DailyExecutionSnapshot:
    broker_snapshot_obj: BrokerSnapshot
    open_order_row_list: list[dict]
    refresh_started_timestamp_ts: datetime
    refreshed_timestamp_ts: datetime
    complete_bool: bool = True


class DailySnapshotChangedError(RuntimeError):
    """Retry a changing observation; it is not a complete account snapshot."""


def _number_float(value_obj, label_str):
    value_float = float(value_obj)
    if isinstance(value_obj, bool) or not isfinite(value_float) or abs(value_float) >= 1e15:
        raise ValueError(f"DAILY invalid {label_str}.")
    return value_float


def _open_rows_list(trade_list, account_route_str):
    from alpha.live.ibkr_socket_client import _canonical_stock_symbol_str
    row_list = []
    identity_set = set()
    for trade_obj in trade_list:
        order_obj = trade_obj.order
        route_str = str(order_obj.account or "").strip()
        if not route_str:
            raise ValueError("DAILY open-order response lacks account identity.")
        client_id_int = int(order_obj.clientId)
        order_id_int = int(order_obj.orderId)
        perm_id_int = int(order_obj.permId or trade_obj.orderStatus.permId or 0)
        # cancelOrder transmits only orderId in the connected client's namespace.
        # Conflicting accounts/permIds for that same namespace are unsafe.
        identity_tuple = (("unbound_manual", perm_id_int) if client_id_int == 0 and order_id_int <= 0
            else (client_id_int, order_id_int))
        if identity_tuple in identity_set:
            raise ValueError("DAILY ambiguous client/order ID in all-client response.")
        identity_set.add(identity_tuple)
        if route_str != account_route_str:
            continue
        asset_str = _canonical_stock_symbol_str(trade_obj.contract.symbol)
        side_str = str(order_obj.action).upper()
        amount_float = _number_float(order_obj.totalQuantity, "order quantity")
        remaining_float = _number_float(trade_obj.orderStatus.remaining, "remaining quantity")
        if (not asset_str or client_id_int < 0 or not (order_id_int or perm_id_int)
                or side_str not in {"BUY", "SELL"} or amount_float <= 0 or remaining_float < 0):
            raise ValueError("DAILY ambiguous or invalid open-order response.")
        row_list.append({"account_route_str": route_str, "asset_str": asset_str,
            "order_ref_str": str(order_obj.orderRef or ""), "order_id_int": order_id_int,
            "perm_id_int": perm_id_int, "client_id_int": client_id_int, "side_str": side_str,
            "amount_float": amount_float, "remaining_amount_float": remaining_float,
            "status_str": str(trade_obj.orderStatus.status)})
    return sorted(row_list, key=lambda row_dict: (row_dict["client_id_int"], row_dict["order_id_int"], row_dict["perm_id_int"]))


def _refresh_snapshot(ib_obj, socket_client_obj, account_route_str):
    from alpha.live.ibkr_socket_client import _canonical_stock_symbol_str
    started_ts = datetime.now(UTC)
    if not ib_obj.isConnected() or not account_route_str or account_route_str not in ib_obj.managedAccounts():
        raise ConnectionError("DAILY account is not connected and visible.")
    ib_obj.RequestTimeout = _number_float(socket_client_obj.timeout_seconds_float, "request timeout")
    if ib_obj.RequestTimeout <= 0:
        raise ValueError("DAILY request timeout must be positive.")
    ib_obj.RaiseRequestErrors = True
    error_code_list = []
    account_value_list = []
    def error_fn(request_id_int, error_code_int, *detail_tuple):
        if error_code_int not in {202, 2104, 2106, 2107, 2108, 2158}:
            error_code_list.append(error_code_int)
    def account_fn(value_obj):
        account_value_list.append(value_obj)
    ib_obj.errorEvent += error_fn
    ib_obj.accountSummaryEvent += account_fn
    try:
        first_order_list = _open_rows_list(ib_obj.reqAllOpenOrders(), account_route_str)
        position_list = ib_obj.reqPositions()
        # Use this completed request's callbacks, never an accountSummary cache.
        # IB allows only two active summary subscriptions. Own the request ID so
        # every confirmation refresh cancels its subscription even on timeout.
        summary_request_id_int = ib_obj.client.getReqId()
        summary_future_obj = ib_obj.wrapper.startReq(summary_request_id_int)
        try:
            ib_obj.client.reqAccountSummary(summary_request_id_int, "All",
                "TotalCashValue,NetLiquidation,AvailableFunds,ExcessLiquidity")
            ib_obj._run(summary_future_obj)
        finally:
            ib_obj.client.cancelAccountSummary(summary_request_id_int)
        trade_list = ib_obj.reqAllOpenOrders()
        order_row_list = _open_rows_list(trade_list, account_route_str)
        if error_code_list or not ib_obj.isConnected():
            raise ConnectionError("DAILY broker response was incomplete or disconnected.")
        if first_order_list != order_row_list:
            raise DailySnapshotChangedError("DAILY open orders changed while refreshing holdings; retry the snapshot.")
    finally:
        ib_obj.errorEvent -= error_fn
        ib_obj.accountSummaryEvent -= account_fn
    position_dict = {}
    for position_obj in position_list:
        if not position_obj.account:
            raise ValueError("DAILY position response lacks account identity.")
        if position_obj.account != account_route_str:
            continue
        amount_float = _number_float(position_obj.position, "position quantity")
        if not amount_float:
            continue
        contract_obj = position_obj.contract
        asset_str = _canonical_stock_symbol_str(contract_obj.symbol)
        if (not asset_str or contract_obj.secType != "STK" or contract_obj.currency != "USD"
                or asset_str in position_dict):
            raise ValueError("DAILY requires unique USD stock positions.")
        position_dict[asset_str] = amount_float
    def usd_float(tag_str, required_bool=True):
        value_set = {_number_float(value_obj.value, tag_str) for value_obj in account_value_list
            if value_obj.account == account_route_str and value_obj.tag == tag_str
            and value_obj.currency == "USD" and not getattr(value_obj, "modelCode", "")}
        if not value_set and not required_bool:
            return None
        if len(value_set) != 1:
            raise ValueError(f"DAILY missing or ambiguous fresh USD {tag_str}.")
        return value_set.pop()
    cash_float = usd_float("TotalCashValue")
    nav_float = usd_float("NetLiquidation")
    refreshed_ts = datetime.now(UTC)
    snapshot_obj = BrokerSnapshot(account_route_str=account_route_str, snapshot_timestamp_ts=refreshed_ts,
        cash_float=cash_float, total_value_float=nav_float, net_liq_float=nav_float,
        available_funds_float=usd_float("AvailableFunds", False),
        excess_liquidity_float=usd_float("ExcessLiquidity", False), position_amount_map=position_dict,
        open_order_id_list=[str(row_dict["perm_id_int"] or row_dict["order_id_int"]) for row_dict in order_row_list])
    return DailyExecutionSnapshot(snapshot_obj, order_row_list, started_ts, refreshed_ts), trade_list


def get_daily_execution_snapshot(socket_client_obj, account_route_str):
    with socket_client_obj.daily_connection() as ib_obj:
        return _refresh_snapshot(ib_obj, socket_client_obj, account_route_str)[0]


def cancel_daily_owned_orders(socket_client_obj, account_route_str, owned_order_ref_set,
        *, session_close_timestamp_ts):
    if session_close_timestamp_ts.tzinfo is None or datetime.now(UTC) < session_close_timestamp_ts:
        raise ValueError("DAILY cancellation is allowed only after the target session closes.")
    if any(not isinstance(ref_str, str) or not ref_str.strip() for ref_str in owned_order_ref_set):
        raise ValueError("DAILY cancellation requires exact nonempty owned order references.")
    initial_obj = get_daily_execution_snapshot(socket_client_obj, account_route_str)
    owned_row_list = [row_dict for row_dict in initial_obj.open_order_row_list
        if row_dict["order_ref_str"] in owned_order_ref_set]
    if any(row_dict["client_id_int"] <= 0 for row_dict in owned_row_list):
        raise ValueError("DAILY cannot cancel client-zero orders without forbidden manual-order binding.")
    for client_id_int in sorted({row_dict["client_id_int"] for row_dict in owned_row_list}):
        # IBKR cancelOrder acts in the connected client's order-ID namespace.
        # Reconnect to that original positive client; never bind or global-cancel.
        with socket_client_obj.daily_connection(client_id_int=client_id_int) as ib_obj:
            refreshed_obj, trade_list = _refresh_snapshot(ib_obj, socket_client_obj, account_route_str)
            original_identity_set = {(row_dict["order_ref_str"], row_dict["order_id_int"], row_dict["perm_id_int"])
                for row_dict in owned_row_list if row_dict["client_id_int"] == client_id_int}
            current_owned_list = [row_dict for row_dict in refreshed_obj.open_order_row_list
                if row_dict["order_ref_str"] in owned_order_ref_set and row_dict["client_id_int"] == client_id_int]
            if any((row_dict["order_ref_str"], row_dict["order_id_int"], row_dict["perm_id_int"])
                    not in original_identity_set for row_dict in current_owned_list):
                raise ValueError("DAILY owned order identity changed before cancellation.")
            for trade_obj in trade_list:
                order_obj = trade_obj.order
                identity_tuple = (str(order_obj.orderRef or ""), int(order_obj.orderId),
                    int(order_obj.permId or trade_obj.orderStatus.permId or 0))
                if (order_obj.account == account_route_str and int(order_obj.clientId) == client_id_int
                        and identity_tuple in original_identity_set):
                    ib_obj.cancelOrder(order_obj)
            deadline_float = monotonic() + socket_client_obj.timeout_seconds_float
            while True:
                try:
                    observed_obj, _ = _refresh_snapshot(ib_obj, socket_client_obj, account_route_str)
                except DailySnapshotChangedError:
                    observed_obj = None
                if observed_obj is not None and not any(row_dict["order_ref_str"] in owned_order_ref_set
                        and row_dict["client_id_int"] == client_id_int for row_dict in observed_obj.open_order_row_list):
                    break
                if monotonic() >= deadline_float:
                    raise TimeoutError("DAILY owned-order cancellation is not yet confirmed.")
                ib_obj.sleep(min(0.1, max(0, deadline_float - monotonic())))
    final_obj = get_daily_execution_snapshot(socket_client_obj, account_route_str)
    if any(row_dict["order_ref_str"] in owned_order_ref_set for row_dict in final_obj.open_order_row_list):
        raise RuntimeError("DAILY owned orders remain open after cancellation refresh.")
    return final_obj
