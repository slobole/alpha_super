from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from math import isfinite

from ib_async import IB, Stock
from ib_async.objects import ExecutionFilter
from ib_async.order import MarketOrder, Order

from alpha.live import scheduler_utils
from alpha.live.models import (
    BrokerOrderEvent,
    BrokerOrderFill,
    BrokerOrderRecord,
    BrokerSnapshot,
    LivePriceSnapshot,
    SessionOpenPrice,
)


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

    @staticmethod
    def _build_broker_order_id_str(order_obj, order_status_obj) -> str:
        perm_id_int = int(getattr(order_obj, "permId", 0) or getattr(order_status_obj, "permId", 0) or 0)
        if perm_id_int > 0:
            return str(perm_id_int)
        return str(int(getattr(order_obj, "orderId", 0) or getattr(order_status_obj, "orderId", 0) or 0))

    @staticmethod
    def _signed_fill_amount_float_from_execution(fill_obj) -> float:
        signed_fill_amount_float = float(fill_obj.execution.shares)
        if str(fill_obj.execution.side).upper() == "SLD":
            signed_fill_amount_float *= -1.0
        return signed_fill_amount_float

    def _build_broker_order_record_obj(
        self,
        trade_obj,
        account_route_str: str,
        submitted_timestamp_ts: datetime,
        decision_plan_id_int: int | None,
        vplan_id_int: int | None,
        fallback_asset_str: str,
    ) -> BrokerOrderRecord:
        last_status_timestamp_ts = submitted_timestamp_ts
        if len(trade_obj.log) > 0:
            last_status_timestamp_ts = self._as_utc_timestamp_ts(trade_obj.log[-1].time)
        order_status_obj = trade_obj.orderStatus
        broker_order_type_str = str(trade_obj.order.orderType)
        if broker_order_type_str == "MKT" and str(getattr(trade_obj.order, "tif", "")).upper() == "OPG":
            broker_order_type_str = "MOO"
        return BrokerOrderRecord(
            broker_order_id_str=self._build_broker_order_id_str(trade_obj.order, order_status_obj),
            decision_plan_id_int=decision_plan_id_int,
            vplan_id_int=vplan_id_int,
            account_route_str=account_route_str,
            asset_str=str(getattr(trade_obj.contract, "symbol", fallback_asset_str)),
            broker_order_type_str=broker_order_type_str,
            unit_str="shares",
            amount_float=float(trade_obj.order.totalQuantity) * (
                1.0 if str(trade_obj.order.action).upper() == "BUY" else -1.0
            ),
            filled_amount_float=float(order_status_obj.filled),
            remaining_amount_float=float(order_status_obj.remaining),
            avg_fill_price_float=(
                None
                if float(order_status_obj.avgFillPrice or 0.0) <= 0.0
                else float(order_status_obj.avgFillPrice)
            ),
            status_str=str(order_status_obj.status),
            last_status_timestamp_ts=last_status_timestamp_ts,
            submitted_timestamp_ts=submitted_timestamp_ts,
            raw_payload_dict={
                "order_id_int": int(trade_obj.order.orderId),
                "perm_id_int": int(order_status_obj.permId or trade_obj.order.permId or 0),
            },
        )

    def _build_broker_order_event_list(
        self,
        trade_obj,
        account_route_str: str,
        decision_plan_id_int: int | None,
        vplan_id_int: int | None,
        fallback_asset_str: str,
    ) -> list[BrokerOrderEvent]:
        broker_order_id_str = self._build_broker_order_id_str(trade_obj.order, trade_obj.orderStatus)
        asset_str = str(getattr(trade_obj.contract, "symbol", fallback_asset_str))
        broker_order_event_list: list[BrokerOrderEvent] = []
        for trade_log_entry_obj in trade_obj.log:
            event_timestamp_ts = self._as_utc_timestamp_ts(trade_log_entry_obj.time)
            broker_order_event_list.append(
                BrokerOrderEvent(
                    broker_order_id_str=broker_order_id_str,
                    decision_plan_id_int=decision_plan_id_int,
                    vplan_id_int=vplan_id_int,
                    account_route_str=account_route_str,
                    asset_str=asset_str,
                    status_str=str(trade_log_entry_obj.status),
                    filled_amount_float=float(trade_obj.orderStatus.filled),
                    remaining_amount_float=float(trade_obj.orderStatus.remaining),
                    avg_fill_price_float=(
                        None
                        if float(trade_obj.orderStatus.avgFillPrice or 0.0) <= 0.0
                        else float(trade_obj.orderStatus.avgFillPrice)
                    ),
                    event_timestamp_ts=event_timestamp_ts,
                    event_source_str="ibkr.trade_log",
                    message_str=str(getattr(trade_log_entry_obj, "message", "") or ""),
                    raw_payload_dict={
                        "error_code_int": int(getattr(trade_log_entry_obj, "errorCode", 0) or 0),
                    },
                )
            )
        return broker_order_event_list

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
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        with self.connect() as ib_obj:
            contract_map = self._build_stock_contract_map(
                ib_obj,
                [broker_order_request_obj.asset_str for broker_order_request_obj in broker_order_request_list],
            )
            broker_order_record_list: list[BrokerOrderRecord] = []
            broker_order_event_list: list[BrokerOrderEvent] = []
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
                broker_order_record_obj = self._build_broker_order_record_obj(
                    trade_obj=trade_obj,
                    account_route_str=account_route_str,
                    submitted_timestamp_ts=submitted_timestamp_ts,
                    decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                    vplan_id_int=broker_order_request_obj.vplan_id_int,
                    fallback_asset_str=broker_order_request_obj.asset_str,
                )
                broker_order_record_list.append(
                    BrokerOrderRecord(
                        **{
                            **broker_order_record_obj.__dict__,
                            "broker_order_type_str": broker_order_request_obj.broker_order_type_str,
                            "unit_str": broker_order_request_obj.unit_str,
                            "amount_float": float(broker_order_request_obj.amount_float),
                            "raw_payload_dict": {
                                **broker_order_record_obj.raw_payload_dict,
                                "submission_key_str": broker_order_request_obj.submission_key_str,
                            },
                        }
                    )
                )
                broker_order_id_str = broker_order_record_list[-1].broker_order_id_str
                broker_order_event_list.extend(
                    self._build_broker_order_event_list(
                        trade_obj=trade_obj,
                        account_route_str=account_route_str,
                        decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                        vplan_id_int=broker_order_request_obj.vplan_id_int,
                        fallback_asset_str=broker_order_request_obj.asset_str,
                    )
                )
                for fill_obj in trade_obj.fills:
                    broker_order_fill_list.append(
                        BrokerOrderFill(
                            broker_order_id_str=broker_order_id_str,
                            decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                            vplan_id_int=broker_order_request_obj.vplan_id_int,
                            account_route_str=account_route_str,
                            asset_str=broker_order_request_obj.asset_str,
                            fill_amount_float=self._signed_fill_amount_float_from_execution(fill_obj),
                            fill_price_float=float(fill_obj.execution.price),
                            fill_timestamp_ts=self._as_utc_timestamp_ts(fill_obj.time),
                            raw_payload_dict={"exec_id_str": fill_obj.execution.execId},
                        )
                    )

        return broker_order_record_list, broker_order_event_list, broker_order_fill_list

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
            broker_order_fill_list.append(
                BrokerOrderFill(
                    broker_order_id_str=str(fill_obj.execution.permId or fill_obj.execution.orderId),
                    decision_plan_id_int=None,
                    vplan_id_int=None,
                    account_route_str=account_route_str,
                    asset_str=fill_obj.contract.symbol,
                    fill_amount_float=self._signed_fill_amount_float_from_execution(fill_obj),
                    fill_price_float=float(fill_obj.execution.price),
                    fill_timestamp_ts=fill_timestamp_ts,
                    raw_payload_dict={"exec_id_str": fill_obj.execution.execId},
                )
            )
        return broker_order_fill_list

    def get_recent_order_state_snapshot(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        with self.connect() as ib_obj:
            open_trade_list = [
                trade_obj
                for trade_obj in ib_obj.reqOpenOrders()
                if str(trade_obj.order.account) == account_route_str
            ]
            completed_trade_list = [
                trade_obj
                for trade_obj in ib_obj.reqCompletedOrders(apiOnly=False)
                if str(trade_obj.order.account) == account_route_str
            ]

        broker_order_fill_list = self.get_recent_fill_list(
            account_route_str=account_route_str,
            since_timestamp_ts=since_timestamp_ts,
        )
        relevant_broker_order_id_set = {
            str(broker_order_fill_obj.broker_order_id_str)
            for broker_order_fill_obj in broker_order_fill_list
        }
        broker_order_record_map: dict[str, BrokerOrderRecord] = {}
        broker_order_event_list: list[BrokerOrderEvent] = []
        for trade_obj in [*open_trade_list, *completed_trade_list]:
            candidate_broker_order_id_str = self._build_broker_order_id_str(
                trade_obj.order,
                trade_obj.orderStatus,
            )
            candidate_broker_order_event_list = [
                broker_order_event_obj
                for broker_order_event_obj in self._build_broker_order_event_list(
                    trade_obj=trade_obj,
                    account_route_str=account_route_str,
                    decision_plan_id_int=None,
                    vplan_id_int=None,
                    fallback_asset_str=str(getattr(trade_obj.contract, "symbol", "")),
                )
                if (
                    broker_order_event_obj.event_timestamp_ts is not None
                    and broker_order_event_obj.event_timestamp_ts >= since_timestamp_ts
                )
            ]
            if (
                len(candidate_broker_order_event_list) == 0
                and candidate_broker_order_id_str not in relevant_broker_order_id_set
            ):
                continue
            broker_order_record_obj = self._build_broker_order_record_obj(
                trade_obj=trade_obj,
                account_route_str=account_route_str,
                submitted_timestamp_ts=since_timestamp_ts,
                decision_plan_id_int=None,
                vplan_id_int=None,
                fallback_asset_str=str(getattr(trade_obj.contract, "symbol", "")),
            )
            broker_order_record_map[broker_order_record_obj.broker_order_id_str] = broker_order_record_obj
            broker_order_event_list.extend(candidate_broker_order_event_list)
        return list(broker_order_record_map.values()), broker_order_event_list, broker_order_fill_list

    def get_session_open_price_list(
        self,
        account_route_str: str,
        asset_str_list: list[str],
        session_open_timestamp_ts: datetime,
        session_calendar_id_str: str,
    ) -> list[SessionOpenPrice]:
        if len(asset_str_list) == 0:
            return []

        market_open_timestamp_ts = scheduler_utils.to_market_timestamp_ts(
            session_open_timestamp_ts,
            session_calendar_id_str,
        )
        session_date_str = market_open_timestamp_ts.date().isoformat()
        with self.connect() as ib_obj:
            contract_map = self._build_stock_contract_map(ib_obj, asset_str_list)
            ticker_list = ib_obj.reqTickers(*contract_map.values())
            ticker_open_map: dict[str, float] = {}
            for ticker_obj in ticker_list:
                official_open_price_float = float(getattr(ticker_obj, "open", float("nan")))
                if isfinite(official_open_price_float) and official_open_price_float > 0.0:
                    ticker_open_map[str(ticker_obj.contract.symbol)] = official_open_price_float

            session_open_price_list: list[SessionOpenPrice] = []
            for asset_str in asset_str_list:
                if asset_str in ticker_open_map:
                    session_open_price_list.append(
                        SessionOpenPrice(
                            session_date_str=session_date_str,
                            account_route_str=account_route_str,
                            asset_str=asset_str,
                            official_open_price_float=ticker_open_map[asset_str],
                            open_price_source_str="ibkr.tick_open",
                            snapshot_timestamp_ts=datetime.now(tz=UTC),
                            raw_payload_dict={"ticker_open_float": ticker_open_map[asset_str]},
                        )
                    )
                    continue

                historical_bar_list = ib_obj.reqHistoricalData(
                    contract_map[asset_str],
                    endDateTime=market_open_timestamp_ts + timedelta(minutes=5),
                    durationStr="1800 S",
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=2,
                )
                official_open_price_float = None
                raw_payload_dict = {}
                for historical_bar_obj in historical_bar_list:
                    bar_timestamp_ts = self._as_utc_timestamp_ts(historical_bar_obj.date)
                    market_bar_timestamp_ts = scheduler_utils.to_market_timestamp_ts(
                        bar_timestamp_ts,
                        session_calendar_id_str,
                    )
                    if market_bar_timestamp_ts < market_open_timestamp_ts:
                        continue
                    official_open_price_float = float(historical_bar_obj.open)
                    raw_payload_dict = {
                        "bar_timestamp_str": bar_timestamp_ts.isoformat(),
                    }
                    break
                session_open_price_list.append(
                    SessionOpenPrice(
                        session_date_str=session_date_str,
                        account_route_str=account_route_str,
                        asset_str=asset_str,
                        official_open_price_float=official_open_price_float,
                        open_price_source_str=(
                            "ibkr.historical_1m_open"
                            if official_open_price_float is not None
                            else None
                        ),
                        snapshot_timestamp_ts=datetime.now(tz=UTC),
                        raw_payload_dict=raw_payload_dict,
                    )
                )

        return session_open_price_list
