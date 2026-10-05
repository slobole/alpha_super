"""Fail-closed, read-only CORE5 broker account and pre-submit evidence.

What-if previews do not reserve margin or borrow, or guarantee auction fills.
Only the fresh subscription created here is cancelled; no orders are submitted.
"""
from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite, trunc

from alpha.live.core5_adapter import CORE5_ASSET_TUPLE
from alpha.live.models import BrokerOrderRequest, BrokerSnapshot


def _finite_number_float(value_obj, label_str: str) -> float:
    try:
        number_float = float(value_obj)
    except (TypeError, ValueError, OverflowError) as exception_obj:
        raise ValueError(f"CORE5 invalid/missing {label_str}.") from exception_obj
    if isinstance(value_obj, bool) or not isfinite(number_float) or abs(number_float) >= 1e15:
        raise ValueError(f"CORE5 invalid/missing {label_str}.")
    return number_float


def _usd_value_float(account_value_list: list, tag_str: str, required_bool: bool = True) -> float | None:
    value_list = [value_obj.value for value_obj in account_value_list
                  if value_obj.tag == tag_str and value_obj.currency == "USD"]
    if not value_list and not required_bool:
        return None
    if len(value_list) != 1:
        raise ValueError(f"CORE5 requires unambiguous USD {tag_str}.")
    return _finite_number_float(value_list[0], tag_str)


def _account_rows_list(account_value_list: list, account_route_str: str) -> list:
    return [value_obj for value_obj in account_value_list
            if value_obj.account == account_route_str and not getattr(value_obj, "modelCode", "")]


def _ready_margin_type_str(account_value_list: list) -> str:
    trading_type_set = {str(value_obj.value).strip().upper() for value_obj in account_value_list
                        if value_obj.tag == "TradingType-S"}
    # Independent order previews cannot bound joint portfolio-margin offsets.
    # PMRGN/GPMRGN need a separately qualified joint-basket protocol.
    if len(trading_type_set) != 1 or not trading_type_set.issubset({"STKNOPT", "STKMRGN"}):
        raise ValueError("CORE5 requires a verified standard margin account (IBKR TradingType-S).")
    if any(str(value_obj.value).strip().lower() != "true" for value_obj in account_value_list
           if value_obj.tag.lower() == "accountready"):
        raise ValueError("CORE5 funding account is not ready.")
    return next(iter(trading_type_set))


def _check_connection(ib_obj, socket_client_obj, account_route_str: str) -> list[str]:
    timeout_float = _finite_number_float(socket_client_obj.timeout_seconds_float, "request timeout")
    if timeout_float <= 0:
        raise ValueError("CORE5 requires a positive request timeout.")
    ib_obj.RequestTimeout = timeout_float
    account_list = ib_obj.managedAccounts()
    if not account_route_str or account_route_str not in account_list:
        raise ValueError("CORE5 account is not visible to IBKR.")
    future_dict = getattr(getattr(ib_obj, "wrapper", None), "_futures", None)
    if not isinstance(future_dict, dict):
        raise ValueError("CORE5 cannot verify IBKR startup download completeness.")
    # ib_async can log a startup timeout without raising; its pending/cancelled
    # positions Future remains until positionEnd, so an empty cache is not proof.
    if "positions" in future_dict:
        raise ValueError("CORE5 positions download is incomplete.")
    return account_list


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
                or asset_str not in CORE5_ASSET_TUPLE or asset_str in position_dict
                or amount_float != trunc(amount_float) or (amount_float < 0 and asset_str != "DBC")):
            raise ValueError("CORE5 requires unique supported USD stock positions and whole shares; only DBC may be short.")
        position_dict[asset_str] = amount_float
    return position_dict


def _open_order_id_list(ib_obj, account_route_str: str) -> list[str]:
    return [str(trade_obj.order.orderId) for trade_obj in ib_obj.reqAllOpenOrders()
            if str(trade_obj.order.account) == account_route_str]


def get_core5_account_snapshot(
    socket_client_obj,
    account_route_str: str,
    include_portfolio_valuation_bool: bool = False,
) -> BrokerSnapshot:
    """Capture strict USD totals, signed holdings and all-client open orders."""
    with socket_client_obj.connect() as ib_obj:
        _check_connection(ib_obj, socket_client_obj, account_route_str)
        account_value_list = _account_rows_list(ib_obj.accountSummary(account=account_route_str), account_route_str)
        cash_float = _usd_value_float(account_value_list, "TotalCashValue")
        nav_float = _usd_value_float(account_value_list, "NetLiquidation")
        if nav_float <= 0:
            raise ValueError("CORE5 requires positive USD NetLiquidation.")
        available_funds_float = _usd_value_float(account_value_list, "AvailableFunds", False)
        excess_liquidity_float = _usd_value_float(account_value_list, "ExcessLiquidity", False)
        position_dict = _position_map_dict(ib_obj, account_route_str)
        order_id_list = _open_order_id_list(ib_obj, account_route_str)
        snapshot_timestamp_ts = datetime.now(tz=UTC)
        valuation_dict = (socket_client_obj._capture_portfolio_valuation_dict(
            ib_obj, account_route_str, position_dict) if include_portfolio_valuation_bool else None)
    return BrokerSnapshot(
        account_route_str=account_route_str, snapshot_timestamp_ts=snapshot_timestamp_ts,
        cash_float=cash_float, total_value_float=nav_float, net_liq_float=nav_float,
        available_funds_float=available_funds_float, excess_liquidity_float=excess_liquidity_float,
        position_amount_map=position_dict, open_order_id_list=order_id_list,
        portfolio_valuation_dict=valuation_dict,
    )


def _preview_margin_dict(preview_obj, asset_str: str) -> dict[str, float]:
    if str(getattr(preview_obj, "status", "")) not in {"PreSubmitted", "Submitted", "PendingSubmit"}:
        raise ValueError(f"CORE5 margin preview rejected/unavailable for {asset_str}.")
    margin_dict = {field_str: _finite_number_float(getattr(preview_obj, field_str, None),
                   f"preview {field_str} for {asset_str}") for field_str in (
        "initMarginBefore", "initMarginAfter", "initMarginChange",
        "maintMarginBefore", "maintMarginAfter", "maintMarginChange",
        "equityWithLoanBefore", "equityWithLoanAfter",
    )}
    for prefix_str in ("initMargin", "maintMargin"):
        before_float = margin_dict[prefix_str + "Before"]
        after_float = margin_dict[prefix_str + "After"]
        change_float = margin_dict[prefix_str + "Change"]
        if (min(before_float, after_float) < 0
                or abs(after_float - before_float - change_float) > max(.05, abs(after_float) * 1e-6)):
            raise ValueError(f"CORE5 inconsistent margin preview for {asset_str}.")
    if margin_dict["equityWithLoanAfter"] < max(margin_dict["initMarginAfter"], margin_dict["maintMarginAfter"]):
        raise ValueError(f"CORE5 margin preview has insufficient equity for {asset_str}.")
    return margin_dict


def get_core5_funding_evidence(
    socket_client_obj,
    account_route_str: str,
    request_list: list[BrokerOrderRequest],
    current_position_dict: dict[str, float],
) -> dict[str, object]:
    """Check current margin and incremental DBC inventory without placing orders.

    Positive per-preview margin increases and equity debits are summed;
    reducing sells earn no margin credit. Repeated same-symbol legs use
    cumulative shares so a flip's opening leg uses the actual account baseline.
    """
    from ib_async.order import Order

    position_dict = {}
    for asset_str, value_obj in current_position_dict.items():
        amount_float = _finite_number_float(value_obj, "current position")
        if amount_float == 0:
            continue
        if (asset_str not in CORE5_ASSET_TUPLE or amount_float != trunc(amount_float)
                or (amount_float < 0 and asset_str != "DBC")):
            raise ValueError("CORE5 funding requires supported whole-share current positions.")
        position_dict[asset_str] = amount_float
    projected_position_dict = dict(position_dict)
    cumulative_delta_dict: dict[str, float] = {}
    request_key_set: set[str] = set()
    preview_request_list: list[tuple[BrokerOrderRequest, float]] = []
    for request_obj in request_list:
        asset_str = request_obj.asset_str
        amount_float = _finite_number_float(request_obj.amount_float, "order shares")
        if (request_obj.account_route_str != account_route_str or asset_str not in CORE5_ASSET_TUPLE
                or request_obj.unit_str != "shares" or request_obj.target_bool
                or request_obj.broker_order_type_str != "MOO"
                or request_obj.execution_deadline_timestamp_str is not None
                or amount_float == 0 or amount_float != trunc(amount_float)
                or not request_obj.order_request_key_str or request_obj.order_request_key_str in request_key_set):
            raise ValueError("CORE5 funding requires unique routed MOO share deltas.")
        request_key_set.add(request_obj.order_request_key_str)
        prior_delta_float = cumulative_delta_dict.get(asset_str, 0.0)
        if prior_delta_float * amount_float < 0:
            raise ValueError("CORE5 cannot preview opposing same-symbol order legs.")
        prior_position_float = projected_position_dict.get(asset_str, 0.0)
        next_position_float = prior_position_float + amount_float
        if next_position_float < 0 and asset_str != "DBC":
            raise ValueError("CORE5 only permits DBC shorts.")
        cumulative_delta_dict[asset_str] = prior_delta_float + amount_float
        projected_position_dict[asset_str] = next_position_float
        # A reducing sell can remove a portfolio-margin hedge. Preview every
        # leg; its negative margin change still never funds another order.
        preview_request_list.append((request_obj, cumulative_delta_dict[asset_str]))
    required_short_float = max(0.0, -projected_position_dict.get("DBC", 0.0)) - max(0.0, -position_dict.get("DBC", 0.0))
    required_short_float = max(0.0, required_short_float)

    with socket_client_obj.connect() as ib_obj:
        account_list = _check_connection(ib_obj, socket_client_obj, account_route_str)
        if "accountValues" in ib_obj.wrapper._futures:
            raise ValueError("CORE5 account download is incomplete.")
        if account_list != [account_route_str]:
            ib_obj.reqAccountUpdates(account=account_route_str)
        account_value_list = _account_rows_list(ib_obj.accountValues(account=account_route_str), account_route_str)
        trading_type_str = _ready_margin_type_str(account_value_list)
        available_funds_float = _usd_value_float(account_value_list, "AvailableFunds")
        excess_liquidity_float = _usd_value_float(account_value_list, "ExcessLiquidity")
        if _position_map_dict(ib_obj, account_route_str) != position_dict:
            raise ValueError("CORE5 positions changed before margin preview.")
        if _open_order_id_list(ib_obj, account_route_str):
            raise ValueError("CORE5 account has outstanding orders before margin preview.")
        contract_dict = socket_client_obj._build_stock_contract_map(
            ib_obj, list(dict.fromkeys(request_obj.asset_str for request_obj, _ in preview_request_list)))
        for asset_str, contract_obj in contract_dict.items():
            if (contract_obj.secType != "STK" or contract_obj.currency != "USD"
                    or str(contract_obj.symbol) != asset_str
                    or str(getattr(contract_obj, "multiplier", "") or "1") != "1"):
                raise ValueError("CORE5 preview requires the qualified USD stock contract.")
        preview_list = []
        init_increase_float = 0.0
        maint_increase_float = 0.0
        total_preview_equity_debit_float = 0.0
        for request_obj, preview_amount_float in preview_request_list:
            preview_order_obj = Order(
                action="BUY" if preview_amount_float > 0 else "SELL",
                totalQuantity=abs(preview_amount_float), orderType="MKT", tif="OPG",
                account=account_route_str, orderRef=request_obj.order_request_key_str, whatIf=True,
            )
            preview_obj = ib_obj.whatIfOrder(contract_dict[request_obj.asset_str], preview_order_obj)
            margin_dict = _preview_margin_dict(preview_obj, request_obj.asset_str)
            # *** CRITICAL *** These are current-account independent previews:
            # required budget = sum(max(0, margin change)) +
            # sum(max(0, equity before - equity after)); never credit exits.
            init_increase_float += max(0.0, margin_dict["initMarginChange"])
            maint_increase_float += max(0.0, margin_dict["maintMarginChange"])
            preview_equity_debit_float = max(0.0, margin_dict["equityWithLoanBefore"] - margin_dict["equityWithLoanAfter"])
            total_preview_equity_debit_float += preview_equity_debit_float
            preview_list.append({
                "asset_str": request_obj.asset_str, "order_request_key_str": request_obj.order_request_key_str,
                "amount_float": float(request_obj.amount_float), "preview_amount_float": preview_amount_float,
                "status_str": str(preview_obj.status), "warning_str": str(getattr(preview_obj, "warningText", "") or ""),
                "margin_dict": margin_dict, "preview_equity_debit_float": preview_equity_debit_float,
            })
        shortable_shares_float = None
        if required_short_float > 0:
            contract_obj = contract_dict["DBC"]
            ticker_obj = ib_obj.reqMktData(contract_obj, genericTickList="236", snapshot=False)
            try:
                ib_obj.sleep(min(2.0, socket_client_obj.timeout_seconds_float))
                shortable_shares_float = _finite_number_float(getattr(ticker_obj, "shortableShares", None), "DBC shortableShares")
                if shortable_shares_float < required_short_float:
                    raise ValueError("CORE5 DBC shortable shares are insufficient.")
            finally:
                ib_obj.cancelMktData(contract_obj)
        # Account updates stream on this dedicated connection. Do not let
        # available margin improving during previews inflate the initial bound.
        refreshed_value_list = _account_rows_list(ib_obj.accountValues(account=account_route_str), account_route_str)
        if _ready_margin_type_str(refreshed_value_list) != trading_type_str:
            raise ValueError("CORE5 margin account type changed during preview.")
        available_funds_float = min(available_funds_float, _usd_value_float(refreshed_value_list, "AvailableFunds"))
        excess_liquidity_float = min(excess_liquidity_float, _usd_value_float(refreshed_value_list, "ExcessLiquidity"))
        required_initial_margin_float = init_increase_float + total_preview_equity_debit_float
        required_maintenance_margin_float = maint_increase_float + total_preview_equity_debit_float
        if required_initial_margin_float > available_funds_float or required_maintenance_margin_float > excess_liquidity_float:
            raise ValueError("CORE5 combined positive margin increases and equity debits exceed available account margin.")
        if _position_map_dict(ib_obj, account_route_str) != position_dict or _open_order_id_list(ib_obj, account_route_str):
            raise ValueError("CORE5 account changed during margin preview.")
        checked_timestamp_str = datetime.now(tz=UTC).isoformat()
    return {
        "required_bool": bool(preview_request_list), "account_route_str": account_route_str,
        "currency_str": "USD", "trading_type_str": trading_type_str,
        "available_funds_float": available_funds_float, "excess_liquidity_float": excess_liquidity_float,
        "total_initial_margin_increase_float": init_increase_float,
        "total_maintenance_margin_increase_float": maint_increase_float,
        "total_preview_equity_debit_float": total_preview_equity_debit_float,
        "required_initial_margin_float": required_initial_margin_float,
        "required_maintenance_margin_float": required_maintenance_margin_float,
        "pending_sell_credit_float": 0.0, "preview_list": preview_list,
        "required_additional_dbc_short_shares_float": required_short_float,
        "dbc_shortable_shares_float": shortable_shares_float,
        "borrow_rate_float": None, "borrow_reserved_bool": False, "basket_margin_guaranteed_bool": False,
        "checked_at_str": checked_timestamp_str,
    }
