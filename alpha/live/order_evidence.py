"""Refreshed, account-scoped broker evidence; empty caches are not absence proof."""
from datetime import UTC, datetime
import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from alpha.live import scheduler_utils


def refreshed_order_evidence_dict(socket_client_obj, account_route_str, since_timestamp_ts):
    from alpha.live.ibkr_socket_client import ExecutionFilter, _canonical_stock_symbol_str

    if since_timestamp_ts.tzinfo is None:
        raise ValueError("Order evidence requires an aware coverage boundary.")
    started_ts = datetime.now(UTC)
    error_code_list = []
    def record_error_fn(request_id_int, error_code_int, *detail_tuple):
        del request_id_int, detail_tuple
        if error_code_int not in {2104, 2106, 2107, 2108, 2158}:
            error_code_list.append(error_code_int)

    with socket_client_obj.connect() as ib_obj:
        # This explicit setting describes the actual TWS login timezone. It
        # only constrains evidence coverage; it does not change order timing.
        # ib_async's empty default means "assume host timezone", not broker proof.
        configured_timezone_str = os.environ.get("ALPHA_IBKR_TWS_TIMEZONE", "").strip()
        tws_timezone_str = configured_timezone_str or str(getattr(ib_obj, "TimezoneTWS", "") or "").strip()
        timezone_source_str = ("ALPHA_IBKR_TWS_TIMEZONE" if configured_timezone_str else
            "ib_async.TimezoneTWS" if tws_timezone_str else "unknown")
        try:
            tws_timezone_obj = ZoneInfo(tws_timezone_str) if tws_timezone_str else None
        except (ValueError, ZoneInfoNotFoundError):
            raise ValueError("ALPHA_IBKR_TWS_TIMEZONE / TimezoneTWS must identify the verified IANA TWS login timezone.")
        ib_obj.RequestTimeout = socket_client_obj.timeout_seconds_float
        ib_obj.RaiseRequestErrors = True
        if not ib_obj.isConnected() or account_route_str not in ib_obj.managedAccounts():
            raise ValueError("Order evidence account is not connected and visible.")
        ib_obj.errorEvent += record_error_fn
        try:
            open_list = ib_obj.reqAllOpenOrders()
            completed_list = ib_obj.reqCompletedOrders(apiOnly=False)
            fill_list = ib_obj.reqExecutions(ExecutionFilter(acctCode=account_route_str))
            if error_code_list or not ib_obj.isConnected():
                raise RuntimeError("Broker order evidence refresh was incomplete.")
        finally:
            ib_obj.errorEvent -= record_error_fn
    refreshed_ts = datetime.now(UTC)
    # Current-day TWS queries do not establish absence on an earlier day.
    # Restrict coverage to the UTC, exchange-local and verified TWS current
    # day, rejecting a refresh that crosses any boundary. Persist valid proofs
    # while their target day is still covered; do not manufacture old history.
    market_started_ts = scheduler_utils.to_market_timestamp_ts(started_ts, "XNYS")
    market_refreshed_ts = scheduler_utils.to_market_timestamp_ts(refreshed_ts, "XNYS")
    same_day_bool = (started_ts.date() == refreshed_ts.date()
        and market_started_ts.date() == market_refreshed_ts.date())
    coverage_ts = max(started_ts.replace(hour=0, minute=0, second=0, microsecond=0),
        market_started_ts.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC))
    if tws_timezone_obj is None:
        same_day_bool = False
        coverage_ts = refreshed_ts
    else:
        tws_started_ts = started_ts.astimezone(tws_timezone_obj)
        tws_refreshed_ts = refreshed_ts.astimezone(tws_timezone_obj)
        same_day_bool = same_day_bool and tws_started_ts.date() == tws_refreshed_ts.date()
        coverage_ts = max(coverage_ts,
            tws_started_ts.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC))
    order_row_list, execution_row_list = [], []
    for trade_obj in [*open_list, *completed_list]:
        route_str = str(getattr(trade_obj.order, "account", ""))
        if not route_str:
            raise ValueError("Broker order evidence has no account identity.")
        if route_str != account_route_str:
            continue
        order_id_str = socket_client_obj._build_broker_order_id_str(trade_obj.order, trade_obj.orderStatus)
        if not order_id_str or order_id_str == "0":
            raise ValueError("Broker order evidence has no order identity.")
        order_row_list.append({"order_request_key_str": str(getattr(trade_obj.order, "orderRef", "") or ""),
            "broker_order_id_str": order_id_str,
            "asset_str": _canonical_stock_symbol_str(trade_obj.contract.symbol),
            "status_str": str(trade_obj.orderStatus.status)})
    for fill_obj in fill_list:
        execution_obj = fill_obj.execution
        route_str = str(getattr(execution_obj, "acctNumber", ""))
        if not route_str:
            raise ValueError("Broker execution evidence has no account identity.")
        if route_str != account_route_str:
            continue
        execution_id_str = str(getattr(execution_obj, "execId", "") or "")
        if not execution_id_str:
            raise ValueError("Broker execution evidence has no execution identity.")
        execution_row_list.append({"order_request_key_str": str(getattr(execution_obj, "orderRef", "") or ""),
            "broker_order_id_str": str(execution_obj.permId or execution_obj.orderId),
            "broker_execution_id_str": execution_id_str,
            "asset_str": _canonical_stock_symbol_str(fill_obj.contract.symbol),
            "fill_timestamp_str": socket_client_obj._as_utc_timestamp_ts(fill_obj.time).isoformat()})
    return {"account_route_str": account_route_str, "source_str": "ibkr.refreshed_current_day",
        "refresh_started_timestamp_str": started_ts.isoformat(), "refreshed_timestamp_str": refreshed_ts.isoformat(),
        "coverage_since_timestamp_str": coverage_ts.isoformat(),
        "tws_timezone_str": tws_timezone_str, "tws_timezone_source_str": timezone_source_str,
        "open_orders_complete_bool": same_day_bool,
        "completed_orders_complete_bool": same_day_bool, "executions_complete_bool": same_day_bool,
        "order_row_list": order_row_list, "execution_row_list": execution_row_list}
