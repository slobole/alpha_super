from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from math import isfinite

from ib_async import IB, Stock
from ib_async.objects import ExecutionFilter
from ib_async.order import MarketOrder, Order

from alpha.live.models import BrokerOrderFill, BrokerOrderRecord, BrokerSnapshot, LivePriceSnapshot


class IBKRSocketClient:
    def __init__(
        self,
        host_str: str = "127.0.0.1",
        port_int: int = 7497,
        client_id_int: int = 31,
        timeout_seconds_float: float = 4.0,
    ):
        self.host_str = str(host_str)
        self.port_int = int(port_int)
        self.client_id_int = int(client_id_int)
        self.timeout_seconds_float = float(timeout_seconds_float)

    @contextmanager
    def connect(self):
        ib_obj = IB()
        ib_obj.connect(
            self.host_str,
            self.port_int,
            clientId=self.client_id_int,
            timeout=self.timeout_seconds_float,
        )
        try:
            yield ib_obj
        finally:
            if ib_obj.isConnected():
                ib_obj.disconnect()

    @staticmethod
    def _as_utc_timestamp_ts(raw_timestamp_obj) -> datetime:
        if isinstance(raw_timestamp_obj, datetime):
            if raw_timestamp_obj.tzinfo is None:
                return raw_timestamp_obj.replace(tzinfo=UTC)
            return raw_timestamp_obj.astimezone(UTC)
        return datetime.now(tz=UTC)

    @staticmethod
    def _build_stock_contract_map(ib_obj: IB, asset_str_list: list[str]) -> dict[str, object]:
        contract_list = [Stock(asset_str, "SMART", "USD") for asset_str in asset_str_list]
        qualified_contract_list = ib_obj.qualifyContracts(*contract_list)
        return {
            contract_obj.symbol: contract_obj
            for contract_obj in qualified_contract_list
        }

    def get_visible_account_route_set(self) -> set[str]:
        with self.connect() as ib_obj:
            return set(ib_obj.managedAccounts())

    def get_account_snapshot(self, account_route_str: str) -> BrokerSnapshot:
        with self.connect() as ib_obj:
            account_value_list = ib_obj.accountSummary(account=account_route_str)
            account_value_map = {
                account_value_obj.tag: account_value_obj.value
                for account_value_obj in account_value_list
                if account_value_obj.account == account_route_str
            }
            position_amount_map = {
                position_obj.contract.symbol: float(position_obj.position)
                for position_obj in ib_obj.positions(account=account_route_str)
            }
            open_order_id_list = [
                str(trade_obj.order.orderId)
                for trade_obj in ib_obj.reqOpenOrders()
                if str(trade_obj.order.account) == account_route_str
            ]

        return BrokerSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=datetime.now(tz=UTC),
            cash_float=float(account_value_map.get("TotalCashValue", 0.0)),
            total_value_float=float(account_value_map.get("NetLiquidation", 0.0)),
            net_liq_float=float(account_value_map.get("NetLiquidation", 0.0)),
            available_funds_float=float(account_value_map["AvailableFunds"])
            if "AvailableFunds" in account_value_map
            else None,
            excess_liquidity_float=float(account_value_map["ExcessLiquidity"])
            if "ExcessLiquidity" in account_value_map
            else None,
            cushion_float=float(account_value_map["Cushion"]) if "Cushion" in account_value_map else None,
            position_amount_map=position_amount_map,
            open_order_id_list=open_order_id_list,
        )

    def get_live_price_snapshot(
        self,
        account_route_str: str,
        asset_str_list: list[str],
    ) -> LivePriceSnapshot:
        del account_route_str
        with self.connect() as ib_obj:
            contract_map = self._build_stock_contract_map(ib_obj, asset_str_list)
            ticker_list = ib_obj.reqTickers(*contract_map.values())
            asset_reference_price_map: dict[str, float] = {}
            for ticker_obj in ticker_list:
                market_price_float = float(ticker_obj.marketPrice())
                if isfinite(market_price_float) and market_price_float > 0.0:
                    asset_reference_price_map[ticker_obj.contract.symbol] = market_price_float

        return LivePriceSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=datetime.now(tz=UTC),
            price_source_str="ib_async.reqTickers.marketPrice",
            asset_reference_price_map=asset_reference_price_map,
        )

    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list,
        submitted_timestamp_ts: datetime,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderFill]]:
        with self.connect() as ib_obj:
            contract_map = self._build_stock_contract_map(
                ib_obj,
                [broker_order_request_obj.asset_str for broker_order_request_obj in broker_order_request_list],
            )
            broker_order_record_list: list[BrokerOrderRecord] = []
            broker_order_fill_list: list[BrokerOrderFill] = []

            for broker_order_request_obj in broker_order_request_list:
                order_action_str = "BUY" if float(broker_order_request_obj.amount_float) > 0.0 else "SELL"
                total_quantity_float = abs(float(broker_order_request_obj.amount_float))
                if broker_order_request_obj.broker_order_type_str == "MOO":
                    broker_order_obj = Order(
                        action=order_action_str,
                        totalQuantity=total_quantity_float,
                        orderType="MKT",
                        tif="OPG",
                        account=account_route_str,
                        orderRef=broker_order_request_obj.submission_key_str,
                    )
                elif broker_order_request_obj.broker_order_type_str == "MOC":
                    broker_order_obj = Order(
                        action=order_action_str,
                        totalQuantity=total_quantity_float,
                        orderType="MOC",
                        tif="DAY",
                        account=account_route_str,
                        orderRef=broker_order_request_obj.submission_key_str,
                    )
                else:
                    broker_order_obj = MarketOrder(
                        action=order_action_str,
                        totalQuantity=total_quantity_float,
                        account=account_route_str,
                        orderRef=broker_order_request_obj.submission_key_str,
                    )

                trade_obj = ib_obj.placeOrder(
                    contract_map[broker_order_request_obj.asset_str],
                    broker_order_obj,
                )
                broker_order_id_str = str(trade_obj.order.permId or trade_obj.order.orderId)
                broker_order_record_list.append(
                    BrokerOrderRecord(
                        broker_order_id_str=broker_order_id_str,
                        order_plan_id_int=broker_order_request_obj.order_plan_id_int,
                        order_intent_id_int=broker_order_request_obj.order_intent_id_int,
                        decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                        vplan_id_int=broker_order_request_obj.vplan_id_int,
                        account_route_str=account_route_str,
                        asset_str=broker_order_request_obj.asset_str,
                        broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                        unit_str=broker_order_request_obj.unit_str,
                        amount_float=float(broker_order_request_obj.amount_float),
                        filled_amount_float=float(trade_obj.orderStatus.filled),
                        status_str=str(trade_obj.orderStatus.status),
                        submitted_timestamp_ts=submitted_timestamp_ts,
                        raw_payload_dict={
                            "submission_key_str": broker_order_request_obj.submission_key_str,
                            "order_id_int": int(trade_obj.order.orderId),
                        },
                    )
                )
                for fill_obj in trade_obj.fills:
                    signed_fill_amount_float = float(fill_obj.execution.shares)
                    if str(fill_obj.execution.side).upper() == "SLD":
                        signed_fill_amount_float *= -1.0
                    broker_order_fill_list.append(
                        BrokerOrderFill(
                            broker_order_id_str=broker_order_id_str,
                            order_plan_id_int=broker_order_request_obj.order_plan_id_int,
                            decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                            vplan_id_int=broker_order_request_obj.vplan_id_int,
                            account_route_str=account_route_str,
                            asset_str=broker_order_request_obj.asset_str,
                            fill_amount_float=signed_fill_amount_float,
                            fill_price_float=float(fill_obj.execution.price),
                            fill_timestamp_ts=self._as_utc_timestamp_ts(fill_obj.time),
                            raw_payload_dict={"exec_id_str": fill_obj.execution.execId},
                        )
                    )

        return broker_order_record_list, broker_order_fill_list

    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
    ) -> list[BrokerOrderFill]:
        with self.connect() as ib_obj:
            ib_fill_list = ib_obj.reqExecutions(ExecutionFilter(acctCode=account_route_str))

        broker_order_fill_list: list[BrokerOrderFill] = []
        for fill_obj in ib_fill_list:
            fill_timestamp_ts = self._as_utc_timestamp_ts(fill_obj.time)
            if fill_timestamp_ts < since_timestamp_ts:
                continue
            signed_fill_amount_float = float(fill_obj.execution.shares)
            if str(fill_obj.execution.side).upper() == "SLD":
                signed_fill_amount_float *= -1.0
            broker_order_fill_list.append(
                BrokerOrderFill(
                    broker_order_id_str=str(fill_obj.execution.permId or fill_obj.execution.orderId),
                    order_plan_id_int=None,
                    decision_plan_id_int=None,
                    vplan_id_int=None,
                    account_route_str=account_route_str,
                    asset_str=fill_obj.contract.symbol,
                    fill_amount_float=signed_fill_amount_float,
                    fill_price_float=float(fill_obj.execution.price),
                    fill_timestamp_ts=fill_timestamp_ts,
                    raw_payload_dict={"exec_id_str": fill_obj.execution.execId},
                )
            )
        return broker_order_fill_list
