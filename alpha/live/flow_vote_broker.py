"""Fail-closed, read-only Month-End Flow Vote account snapshot. No orders are submitted.

The same strict read as core5_broker.py, for this pod's own tradables: SPY long, TLT long or short, whole USD shares.
"""
from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite, trunc

from alpha.live.flow_vote_adapter import FLOW_VOTE_ASSET_TUPLE, FLOW_VOTE_SHORT_ASSET_SET
from alpha.live.models import BrokerSnapshot


def _finite_number_float(value_obj, label_str: str) -> float:
    try:
        number_float = float(value_obj)
    except (TypeError, ValueError, OverflowError) as exception_obj:
        raise ValueError(f"Flow Vote invalid/missing {label_str}.") from exception_obj
    if isinstance(value_obj, bool) or not isfinite(number_float) or abs(number_float) >= 1e15:
        raise ValueError(f"Flow Vote invalid/missing {label_str}.")
    return number_float


def _usd_value_float(account_value_list: list, tag_str: str, required_bool: bool = True) -> float | None:
    value_list = [value_obj.value for value_obj in account_value_list
                  if value_obj.tag == tag_str and value_obj.currency == "USD"]
    if not value_list and not required_bool:
        return None
    if len(value_list) != 1:
        raise ValueError(f"Flow Vote requires unambiguous USD {tag_str}.")
    return _finite_number_float(value_list[0], tag_str)


def _check_connection(ib_obj, socket_client_obj, account_route_str: str) -> None:
    timeout_float = _finite_number_float(socket_client_obj.timeout_seconds_float, "request timeout")
    if timeout_float <= 0:
        raise ValueError("Flow Vote requires a positive request timeout.")
    ib_obj.RequestTimeout = timeout_float
    if not account_route_str or account_route_str not in ib_obj.managedAccounts():
        raise ValueError("Flow Vote account is not visible to IBKR.")
    future_dict = getattr(getattr(ib_obj, "wrapper", None), "_futures", None)
    if not isinstance(future_dict, dict):
        raise ValueError("Flow Vote cannot verify IBKR startup download completeness.")
    # ib_async can log a startup timeout without raising; a pending positions Future means the cache is not proof.
    if "positions" in future_dict:
        raise ValueError("Flow Vote positions download is incomplete.")


def _position_map_dict(ib_obj, account_route_str: str) -> dict[str, float]:
    position_dict: dict[str, float] = {}
    for position_obj in ib_obj.positions(account=account_route_str):
        if position_obj.account != account_route_str:
            continue
        amount_float = _finite_number_float(position_obj.position, "position shares")
        if amount_float == 0:
            continue
        contract_obj = position_obj.contract
        asset_str = str(contract_obj.symbol)
        if (contract_obj.secType != "STK" or contract_obj.currency != "USD"
                or str(getattr(contract_obj, "multiplier", "") or "1") != "1"
                or asset_str not in FLOW_VOTE_ASSET_TUPLE or asset_str in position_dict
                or amount_float != trunc(amount_float) or (amount_float < 0 and asset_str not in FLOW_VOTE_SHORT_ASSET_SET)):
            raise ValueError("Flow Vote requires unique SPY/TLT USD stock positions in whole shares; only TLT may be short.")
        position_dict[asset_str] = amount_float
    return position_dict


def get_flow_vote_account_snapshot(socket_client_obj, account_route_str: str, include_portfolio_valuation_bool: bool = False) -> BrokerSnapshot:
    """Capture strict USD totals, signed holdings and all-client open orders."""
    connection_fn = socket_client_obj.daily_connection if include_portfolio_valuation_bool else socket_client_obj.connect
    with connection_fn() as ib_obj:
        _check_connection(ib_obj, socket_client_obj, account_route_str)
        account_value_list = [value_obj for value_obj in ib_obj.accountSummary(account=account_route_str)
                              if value_obj.account == account_route_str and not getattr(value_obj, "modelCode", "")]
        cash_float = _usd_value_float(account_value_list, "TotalCashValue")
        nav_float = _usd_value_float(account_value_list, "NetLiquidation")
        if nav_float <= 0:
            raise ValueError("Flow Vote requires positive USD NetLiquidation.")
        available_funds_float = _usd_value_float(account_value_list, "AvailableFunds", False)
        excess_liquidity_float = _usd_value_float(account_value_list, "ExcessLiquidity", False)
        position_dict = _position_map_dict(ib_obj, account_route_str)
        order_id_list = [str(trade_obj.order.orderId) for trade_obj in ib_obj.reqAllOpenOrders()
                         if str(trade_obj.order.account) == account_route_str]
        snapshot_timestamp_ts = datetime.now(tz=UTC)
        valuation_dict = (socket_client_obj._capture_portfolio_valuation_dict(ib_obj, account_route_str, position_dict)
                          if include_portfolio_valuation_bool else None)
    return BrokerSnapshot(
        account_route_str=account_route_str, snapshot_timestamp_ts=snapshot_timestamp_ts,
        cash_float=cash_float, total_value_float=nav_float, net_liq_float=nav_float,
        available_funds_float=available_funds_float, excess_liquidity_float=excess_liquidity_float,
        position_amount_map=position_dict, open_order_id_list=order_id_list,
        portfolio_valuation_dict=valuation_dict,
    )
