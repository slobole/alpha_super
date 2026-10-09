from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from math import isfinite
from time import monotonic

try:
    from ib_async import IB, Stock
    from ib_async.objects import ExecutionFilter
    from ib_async.order import Order
except ModuleNotFoundError:  # pragma: no cover - exercised only in lightweight test envs.
    class IB:  # type: ignore[no-redef]
        def connect(self, *args, **kwargs):
            raise ModuleNotFoundError("ib_async is required for live IBKR connectivity.")

        def isConnected(self) -> bool:
            return False

        def disconnect(self) -> None:
            return None

    def Stock(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError("ib_async is required for live IBKR connectivity.")

    class ExecutionFilter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("ib_async is required for live IBKR connectivity.")

    class Order:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("ib_async is required for live IBKR connectivity.")

from alpha.live import scheduler_utils
from alpha.live.models import (
    BrokerOrderAck,
    BrokerOrderEvent,
    BrokerOrderFill,
    BrokerOrderRecord,
    BrokerOrderRequest,
    BrokerSnapshot,
    LivePriceSnapshot,
    SessionOpenPrice,
    SubmitBatchResult,
)


DEFAULT_AUCTION_REFERENCE_WAIT_SECONDS_FLOAT = 2.0
DEFAULT_SUBMIT_ACK_TIMEOUT_SECONDS_FLOAT = 10.0
DEFAULT_SUBMIT_ACK_POLL_SECONDS_FLOAT = 0.25
AUCTION_REFERENCE_SOURCE_STR = "ib_async.reqMktData.225.auctionPrice"
REQ_MKTDATA_MARKET_PRICE_FALLBACK_SOURCE_STR = "ib_async.reqMktData.marketPrice.fallback"
REQ_TICKERS_MARKET_PRICE_FALLBACK_SOURCE_STR = "ib_async.reqTickers.marketPrice.fallback"
LEGACY_REQ_TICKERS_MARKET_PRICE_SOURCE_STR = "ib_async.reqTickers.marketPrice"
IBKR_TICK_OPEN_SOURCE_STR = "ibkr.tick_open"
LOOPBACK_HOST_ALIAS_TUPLE: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
CONNECTION_REFUSED_WINERROR_INT_SET: set[int] = {1225, 10061}
# Explicit share-class aliases only; punctuation is meaningful for other assets.
IBKR_STOCK_SYMBOL_MAP_DICT = {"BRK.B": "BRK B", "BF.B": "BF B"}
CANONICAL_STOCK_SYMBOL_MAP_DICT = {
    broker_symbol_str: asset_str for asset_str, broker_symbol_str in IBKR_STOCK_SYMBOL_MAP_DICT.items()
}


def _canonical_stock_symbol_str(symbol_obj) -> str:
    symbol_str = str(symbol_obj).strip()
    return CANONICAL_STOCK_SYMBOL_MAP_DICT.get(symbol_str, symbol_str)


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
        last_exception_obj: Exception | None = None
        for host_str in self._build_connect_host_str_list():
            ib_obj = IB()
            try:
                ib_obj.connect(
                    host_str,
                    self.port_int,
                    clientId=self.client_id_int,
                    timeout=self.timeout_seconds_float,
                )
                try:
                    yield ib_obj
                finally:
                    if ib_obj.isConnected():
                        ib_obj.disconnect()
                return
            except Exception as exception_obj:
                last_exception_obj = exception_obj
                if ib_obj.isConnected():
                    ib_obj.disconnect()
                if not self._can_retry_connect_with_alternate_loopback_host(
                    attempted_host_str=host_str,
                    exception_obj=exception_obj,
                ):
                    raise
        if last_exception_obj is not None:
            raise last_exception_obj
        raise RuntimeError("IBKRSocketClient.connect() exhausted all hosts without an exception.")

    @contextmanager
    def daily_connection(self, client_id_int: int | None = None):
        """Fresh DAILY queries without execution history or client-zero binding."""
        from ib_async import StartupFetchNONE

        client_id_int = self.client_id_int if client_id_int is None else client_id_int
        if isinstance(client_id_int, bool) or not isinstance(client_id_int, int) or client_id_int <= 0:
            raise ValueError("DAILY account operations require a positive API client ID; client zero auto-binds manual orders.")
        for host_str in self._build_connect_host_str_list():
            ib_obj = IB()
            try:
                ib_obj.connect(host_str, self.port_int, clientId=client_id_int,
                    timeout=self.timeout_seconds_float, readonly=True,
                    raiseSyncErrors=True, fetchFields=StartupFetchNONE)
            except Exception as exception_obj:
                if ib_obj.isConnected():
                    ib_obj.disconnect()
                if self._can_retry_connect_with_alternate_loopback_host(
                        attempted_host_str=host_str, exception_obj=exception_obj):
                    continue
                raise
            # Body failures cannot trigger the host fallback or repeat cancellation.
            try:
                yield ib_obj
            finally:
                if ib_obj.isConnected():
                    ib_obj.disconnect()
            return
        raise RuntimeError("DAILY broker connection exhausted configured hosts.")

    def _build_connect_host_str_list(self) -> list[str]:
        normalized_host_str = str(self.host_str).strip()
        if normalized_host_str not in LOOPBACK_HOST_ALIAS_TUPLE:
            return [normalized_host_str]
        return [
            normalized_host_str,
            *[
                candidate_host_str
                for candidate_host_str in LOOPBACK_HOST_ALIAS_TUPLE
                if candidate_host_str != normalized_host_str
            ],
        ]

    def _can_retry_connect_with_alternate_loopback_host(
        self,
        *,
        attempted_host_str: str,
        exception_obj: Exception,
    ) -> bool:
        normalized_host_str = str(self.host_str).strip()
        if normalized_host_str not in LOOPBACK_HOST_ALIAS_TUPLE:
            return False
        if attempted_host_str == LOOPBACK_HOST_ALIAS_TUPLE[-1]:
            return False
        if isinstance(exception_obj, TimeoutError):
            return True
        if isinstance(exception_obj, ConnectionRefusedError):
            return True
        return int(getattr(exception_obj, "winerror", 0) or 0) in CONNECTION_REFUSED_WINERROR_INT_SET

    @staticmethod
    def _as_utc_timestamp_ts(raw_timestamp_obj) -> datetime:
        if isinstance(raw_timestamp_obj, datetime):
            if raw_timestamp_obj.tzinfo is None:
                return raw_timestamp_obj.replace(tzinfo=UTC)
            return raw_timestamp_obj.astimezone(UTC)
        return datetime.now(tz=UTC)

    @staticmethod
    def _build_stock_contract_map(ib_obj: IB, asset_str_list: list[str]) -> dict[str, object]:
        requested_asset_str_list = [str(asset_str) for asset_str in asset_str_list]
        contract_list = [Stock(IBKR_STOCK_SYMBOL_MAP_DICT.get(asset_str, asset_str), "SMART", "USD")
                         for asset_str in requested_asset_str_list]
        qualified_contract_list = ib_obj.qualifyContracts(*contract_list)
        contract_map_dict: dict[str, object] = {}
        for contract_obj in qualified_contract_list:
            contract_symbol_str = _canonical_stock_symbol_str(getattr(contract_obj, "symbol", "") or "")
            if contract_symbol_str == "":
                continue
            contract_map_dict[contract_symbol_str] = contract_obj
        missing_asset_str_list = sorted(
            {asset_str for asset_str in requested_asset_str_list if asset_str not in contract_map_dict}
        )
        if len(missing_asset_str_list) > 0:
            missing_asset_csv_str = ", ".join(missing_asset_str_list)
            raise ValueError(f"IBKR contract qualification failed for assets: {missing_asset_csv_str}")
        return contract_map_dict

    @staticmethod
    def _build_broker_order_id_str(order_obj, order_status_obj) -> str:
        perm_id_int = int(getattr(order_obj, "permId", 0) or getattr(order_status_obj, "permId", 0) or 0)
        if perm_id_int > 0:
            return str(perm_id_int)
        return str(int(getattr(order_obj, "orderId", 0) or getattr(order_status_obj, "orderId", 0) or 0))

    @staticmethod
    def _build_broker_order_id_alias_set(
        order_obj=None,
        order_status_obj=None,
        execution_obj=None,
    ) -> set[str]:
        broker_order_id_set: set[str] = set()
        raw_broker_order_id_obj_list = [
            getattr(order_obj, "orderId", 0) if order_obj is not None else 0,
            getattr(order_obj, "permId", 0) if order_obj is not None else 0,
            getattr(order_status_obj, "orderId", 0) if order_status_obj is not None else 0,
            getattr(order_status_obj, "permId", 0) if order_status_obj is not None else 0,
            getattr(execution_obj, "orderId", 0) if execution_obj is not None else 0,
            getattr(execution_obj, "permId", 0) if execution_obj is not None else 0,
        ]
        for raw_broker_order_id_obj in raw_broker_order_id_obj_list:
            try:
                broker_order_id_int = int(raw_broker_order_id_obj or 0)
            except (TypeError, ValueError):
                continue
            if broker_order_id_int > 0:
                broker_order_id_set.add(str(broker_order_id_int))
        return broker_order_id_set

    @staticmethod
    def _build_order_request_key_str(trade_obj) -> str | None:
        order_request_key_str = str(getattr(trade_obj.order, "orderRef", "") or "").strip()
        if order_request_key_str == "":
            return None
        return order_request_key_str

    @staticmethod
    def _derive_submission_key_str_from_order_request_key(
        order_request_key_str: str | None,
    ) -> str | None:
        normalized_order_request_key_str = str(order_request_key_str or "").strip()
        if normalized_order_request_key_str == "":
            return None
        submit_batch_key_piece_list = normalized_order_request_key_str.rsplit(":", 2)
        if len(submit_batch_key_piece_list) == 3:
            return str(submit_batch_key_piece_list[0])
        return normalized_order_request_key_str

    @classmethod
    def _build_submission_key_str(cls, trade_obj) -> str | None:
        return cls._derive_submission_key_str_from_order_request_key(
            cls._build_order_request_key_str(trade_obj)
        )

    @staticmethod
    def _build_perm_id_int(
        order_obj=None,
        order_status_obj=None,
        execution_obj=None,
    ) -> int | None:
        raw_perm_id_obj_list = [
            getattr(order_obj, "permId", 0) if order_obj is not None else 0,
            getattr(order_status_obj, "permId", 0) if order_status_obj is not None else 0,
            getattr(execution_obj, "permId", 0) if execution_obj is not None else 0,
        ]
        for raw_perm_id_obj in raw_perm_id_obj_list:
            try:
                perm_id_int = int(raw_perm_id_obj or 0)
            except (TypeError, ValueError):
                continue
            if perm_id_int > 0:
                return perm_id_int
        return None

    @staticmethod
    def _safe_positive_price_float(raw_price_obj) -> float | None:
        try:
            price_float = float(raw_price_obj)
        except (TypeError, ValueError):
            return None
        if not isfinite(price_float) or price_float <= 0.0:
            return None
        return price_float

    @staticmethod
    def _build_live_price_source_summary_str(
        asset_reference_source_map_dict: dict[str, str],
        default_price_source_str: str,
    ) -> str:
        if len(asset_reference_source_map_dict) == 0:
            return str(default_price_source_str)
        source_set = {
            str(source_str)
            for source_str in asset_reference_source_map_dict.values()
            if str(source_str) != ""
        }
        if len(source_set) == 0:
            return str(default_price_source_str)
        if source_set == {AUCTION_REFERENCE_SOURCE_STR}:
            return "auction_225"
        market_price_fallback_source_set = {
            REQ_MKTDATA_MARKET_PRICE_FALLBACK_SOURCE_STR,
            REQ_TICKERS_MARKET_PRICE_FALLBACK_SOURCE_STR,
        }
        if source_set.issubset(market_price_fallback_source_set):
            return "marketPrice_fallback"
        if AUCTION_REFERENCE_SOURCE_STR in source_set and len(source_set - {AUCTION_REFERENCE_SOURCE_STR}) > 0:
            return "mixed"
        if len(source_set) == 1:
            return next(iter(source_set))
        return str(default_price_source_str)

    def _build_req_tickers_price_map_dict(
        self,
        ib_obj: IB,
        contract_map_dict: dict[str, object],
        price_source_str: str,
    ) -> tuple[dict[str, float], dict[str, str]]:
        ticker_list = ib_obj.reqTickers(*contract_map_dict.values())
        asset_reference_price_map_dict: dict[str, float] = {}
        asset_reference_source_map_dict: dict[str, str] = {}
        for ticker_obj in ticker_list:
            market_price_float = self._safe_positive_price_float(ticker_obj.marketPrice())
            if market_price_float is None:
                continue
            asset_str = _canonical_stock_symbol_str(ticker_obj.contract.symbol)
            asset_reference_price_map_dict[asset_str] = market_price_float
            asset_reference_source_map_dict[asset_str] = str(price_source_str)
        return asset_reference_price_map_dict, asset_reference_source_map_dict

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
        *,
        use_completed_quantity_bool: bool = False,
    ) -> BrokerOrderRecord:
        last_status_timestamp_ts = submitted_timestamp_ts
        if len(trade_obj.log) > 0:
            last_status_timestamp_ts = self._as_utc_timestamp_ts(trade_obj.log[-1].time)
        order_status_obj = trade_obj.orderStatus
        broker_order_type_str = str(trade_obj.order.orderType)
        if broker_order_type_str == "MKT" and str(getattr(trade_obj.order, "tif", "")).upper() == "OPG":
            broker_order_type_str = "MOO"
        submission_key_str = self._build_submission_key_str(trade_obj)
        order_request_key_str = self._build_order_request_key_str(trade_obj)
        filled_amount_float = float(order_status_obj.filled)
        remaining_amount_float = float(order_status_obj.remaining)
        quantity_evidence_dict: dict[str, object] = {}
        if use_completed_quantity_bool:
            # ib_async.completedOrder creates an empty OrderStatus (filled=0).
            # The completed feed's actual quantity lives on Order.filledQuantity.
            try:
                completed_filled_float = float(getattr(trade_obj.order, "filledQuantity", float("nan")))
            except (TypeError, ValueError):
                completed_filled_float = float("nan")
            completed_quantity_verified_bool = (
                isfinite(completed_filled_float)
                and 0 <= completed_filled_float <= float(trade_obj.order.totalQuantity)
            )
            quantity_evidence_dict["completed_quantity_verified_bool"] = completed_quantity_verified_bool
            if completed_quantity_verified_bool:
                filled_amount_float = completed_filled_float
                remaining_amount_float = float(trade_obj.order.totalQuantity) - completed_filled_float
                quantity_evidence_dict["filled_quantity_source_str"] = "completed_order.filledQuantity"
        return BrokerOrderRecord(
            broker_order_id_str=self._build_broker_order_id_str(trade_obj.order, order_status_obj),
            decision_plan_id_int=decision_plan_id_int,
            vplan_id_int=vplan_id_int,
            account_route_str=account_route_str,
            asset_str=_canonical_stock_symbol_str(getattr(trade_obj.contract, "symbol", fallback_asset_str)),
            order_request_key_str=order_request_key_str,
            broker_order_type_str=broker_order_type_str,
            unit_str="shares",
            amount_float=float(trade_obj.order.totalQuantity) * (
                1.0 if str(trade_obj.order.action).upper() == "BUY" else -1.0
            ),
            filled_amount_float=filled_amount_float,
            remaining_amount_float=remaining_amount_float,
            avg_fill_price_float=(
                None
                if float(order_status_obj.avgFillPrice or 0.0) <= 0.0
                else float(order_status_obj.avgFillPrice)
            ),
            status_str=str(order_status_obj.status),
            last_status_timestamp_ts=last_status_timestamp_ts,
            submitted_timestamp_ts=submitted_timestamp_ts,
            submission_key_str=submission_key_str,
            raw_payload_dict={
                "order_id_int": int(trade_obj.order.orderId),
                "perm_id_int": int(self._build_perm_id_int(trade_obj.order, order_status_obj) or 0),
                "order_ref_str": order_request_key_str,
                **quantity_evidence_dict,
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
        asset_str = _canonical_stock_symbol_str(getattr(trade_obj.contract, "symbol", fallback_asset_str))
        submission_key_str = self._build_submission_key_str(trade_obj)
        order_request_key_str = self._build_order_request_key_str(trade_obj)
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
                    order_request_key_str=order_request_key_str,
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
                    submission_key_str=submission_key_str,
                    raw_payload_dict={
                        "error_code_int": int(getattr(trade_log_entry_obj, "errorCode", 0) or 0),
                        "order_ref_str": order_request_key_str,
                    },
                )
            )
        return broker_order_event_list

    def get_visible_account_route_set(self) -> set[str]:
        with self.connect() as ib_obj:
            return set(ib_obj.managedAccounts())

    def get_account_snapshot(
        self,
        account_route_str: str,
        *,
        include_portfolio_valuation_bool: bool = False,
        include_all_client_orders_bool: bool = False,
    ) -> BrokerSnapshot:
        with self.connect() as ib_obj:
            account_value_list = ib_obj.accountSummary(account=account_route_str)
            account_value_map = {
                account_value_obj.tag: account_value_obj.value
                for account_value_obj in account_value_list
                if account_value_obj.account == account_route_str
            }
            position_amount_map = {
                _canonical_stock_symbol_str(position_obj.contract.symbol): float(position_obj.position)
                for position_obj in ib_obj.positions(account=account_route_str)
            }
            open_order_id_list = [
                str(trade_obj.order.orderId)
                for trade_obj in (ib_obj.reqAllOpenOrders() if include_all_client_orders_bool else ib_obj.reqOpenOrders())
                if str(trade_obj.order.account) == account_route_str
            ]
            snapshot_timestamp_ts = datetime.now(tz=UTC) if include_portfolio_valuation_bool else None
            portfolio_valuation_dict = (
                self._capture_portfolio_valuation_dict(ib_obj, account_route_str, position_amount_map)
                if include_portfolio_valuation_bool else None
            )

        return BrokerSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=snapshot_timestamp_ts or datetime.now(tz=UTC),
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
            portfolio_valuation_dict=portfolio_valuation_dict,
        )

    def get_capsule_account_snapshot(self, account_route_str: str) -> BrokerSnapshot:
        """Observe all clients' account orders without binding or changing them."""
        return self.get_account_snapshot(account_route_str, include_all_client_orders_bool=True)

    def _capture_portfolio_valuation_dict(
        self,
        ib_obj: IB,
        account_route_str: str,
        position_amount_map: dict[str, float],
    ) -> dict[str, object]:
        """Optional EOD display values; never replace the trading snapshot.

        connect() owns a new IB instance per call. No shared subscriptions or
        callbacks are replaced, and the enclosing context still disconnects it.
        """
        valuation_dict: dict[str, object] = {
            "available_bool": False,
            "reason_str": "IBKR portfolio unavailable",
            "account_route_str": account_route_str,
            "source_str": "IBKR portfolio",
            "currency_str": "USD",
        }
        try:
            account_list = ib_obj.managedAccounts()
            if not account_route_str or account_route_str not in account_list:
                return valuation_dict
            # ib_async 2.1's single-account startup awaits accountValues. A
            # timeout cancels its Future but leaves this key until the full
            # accountDownloadEnd arrives. Reading the key never changes it.
            if "accountValues" in ib_obj.wrapper._futures:
                return valuation_dict
            if account_list == [account_route_str]:
                portfolio_item_list = ib_obj.portfolio(account=account_route_str)
                account_value_list = ib_obj.accountValues(account=account_route_str)
            else:
                # Multi-account startup only uses accountUpdatesMulti (no
                # portfolio marks). This is the sole classic account download
                # on this dedicated connection. Collect fresh target rows;
                # the SDK's unscoped completion alone is not completeness proof.
                portfolio_item_list: list[object] = []
                account_value_list: list[object] = []

                def collect_portfolio_item(portfolio_item_obj):
                    if portfolio_item_obj.account == account_route_str:
                        portfolio_item_list.append(portfolio_item_obj)

                def collect_account_value(account_value_obj):
                    if account_value_obj.account == account_route_str:
                        account_value_list.append(account_value_obj)

                ib_obj.updatePortfolioEvent += collect_portfolio_item
                ib_obj.accountValueEvent += collect_account_value
                try:
                    ib_obj.run(
                        ib_obj.reqAccountUpdatesAsync(account_route_str),
                        timeout=max(0.1, min(self.timeout_seconds_float, 10.0)),
                    )
                finally:
                    ib_obj.updatePortfolioEvent -= collect_portfolio_item
                    ib_obj.accountValueEvent -= collect_account_value
            valuation_dict.update(self._build_portfolio_valuation_values_dict(
                portfolio_item_list, account_value_list, position_amount_map
            ))
        except TimeoutError:
            valuation_dict["reason_str"] = "IBKR portfolio request timed out"
        except Exception:
            # A missing display value must not fail the existing EOD capture.
            pass
        finally:
            valuation_dict["observed_timestamp_str"] = datetime.now(tz=UTC).isoformat()
        return valuation_dict

    @staticmethod
    def _build_portfolio_valuation_values_dict(
        portfolio_item_list: list[object],
        account_value_list: list[object],
        position_amount_map: dict[str, float],
    ) -> dict[str, object]:
        unavailable_dict = {"available_bool": False, "reason_str": "IBKR portfolio incomplete"}
        account_value_map = {
            (account_value_obj.tag, account_value_obj.currency): account_value_obj.value
            for account_value_obj in account_value_list if not account_value_obj.modelCode
        }
        if any(
            account_value_obj.tag == "accountReady"
            and str(account_value_obj.value).lower() != "true"
            for account_value_obj in account_value_list
        ):
            return {"available_bool": False, "reason_str": "IBKR account not ready"}
        # Never infer that an unlabelled BASE total is USD from stock currency.
        cash_float = float(account_value_map[("TotalCashValue", "USD")])
        broker_nav_float = float(account_value_map[("NetLiquidation", "USD")])
        if not all(isfinite(value_float) and abs(value_float) < 1e15
                   for value_float in (cash_float, broker_nav_float)) or broker_nav_float <= 0:
            return unavailable_dict
        portfolio_item_map = {
            int(portfolio_item_obj.contract.conId): portfolio_item_obj
            for portfolio_item_obj in portfolio_item_list
        }
        position_list: list[dict[str, object]] = []
        valuation_shares_map: dict[str, float] = {}
        for conid_int, portfolio_item_obj in portfolio_item_map.items():
            contract_obj = portfolio_item_obj.contract
            symbol_str = _canonical_stock_symbol_str(contract_obj.symbol)
            shares_float = float(portfolio_item_obj.position)
            if shares_float == 0:
                continue
            market_price_float = float(portfolio_item_obj.marketPrice)
            value_float = float(portfolio_item_obj.marketValue)
            if (
                not symbol_str or symbol_str in valuation_shares_map or conid_int <= 0
                or contract_obj.secType != "STK" or contract_obj.currency != "USD"
                or str(contract_obj.multiplier or "1") != "1"
                or not all(isfinite(number_float) and abs(number_float) < 1e15
                           for number_float in (shares_float, market_price_float, value_float))
                or market_price_float <= 0 or shares_float * value_float <= 0
                or abs(shares_float * market_price_float - value_float) > max(.02, abs(value_float) * 1e-6)
            ):
                return unavailable_dict
            valuation_shares_map[symbol_str] = shares_float
            position_dict = {
                "symbol_str": symbol_str, "conid_int": conid_int, "currency_str": "USD",
                "shares_float": shares_float, "market_price_float": market_price_float,
                "value_float": value_float,
            }
            average_cost_obj = getattr(portfolio_item_obj, "averageCost", None)
            unrealized_pnl_obj = getattr(portfolio_item_obj, "unrealizedPNL", None)
            if (all(type(number_obj) in {int, float} and abs(number_obj) < 1e15 and isfinite(number_obj)
                    for number_obj in (average_cost_obj, unrealized_pnl_obj)) and average_cost_obj >= 0):
                cost_decimal = Decimal(str(shares_float)) * Decimal(str(average_cost_obj))
                value_decimal = Decimal(str(value_float))
                # Optional display evidence: P&L = value - signed shares * average cost.
                # Accept five cents or one ppm of value/cost for IB rounding;
                # a missing/inconsistent cost must not suppress valid marks.
                tolerance_decimal = max(Decimal(".05"), max(abs(value_decimal), abs(cost_decimal)) * Decimal(".000001"))
                if abs(Decimal(str(unrealized_pnl_obj)) - (value_decimal - cost_decimal)) <= tolerance_decimal:
                    position_dict.update(average_cost_float=float(average_cost_obj),
                        unrealized_pnl_float=float(unrealized_pnl_obj))
            position_list.append(position_dict)
        expected_shares_map = {
            symbol_str: shares_float for symbol_str, shares_float in position_amount_map.items()
            if shares_float != 0
        }
        if valuation_shares_map != expected_shares_map:
            return {"available_bool": False, "reason_str": "IBKR holdings changed or incomplete"}
        return {
            "available_bool": True, "reason_str": "", "cash_float": cash_float,
            "broker_nav_float": broker_nav_float,
            "position_list": sorted(position_list, key=lambda position_dict: position_dict["symbol_str"]),
        }

    def get_live_price_snapshot(
        self,
        account_route_str: str,
        asset_str_list: list[str],
        execution_policy_str: str | None = None,
    ) -> LivePriceSnapshot:
        with self.connect() as ib_obj:
            contract_map = self._build_stock_contract_map(ib_obj, asset_str_list)
            open_auction_reference_policy_set = {"next_open_moo", "next_month_first_open"}
            # *** CRITICAL*** MOO-style execution policies must size from the
            # same auction/open reference path they will execute against.
            if str(execution_policy_str or "") not in open_auction_reference_policy_set:
                asset_reference_price_map, asset_reference_source_map_dict = (
                    self._build_req_tickers_price_map_dict(
                        ib_obj=ib_obj,
                        contract_map_dict=contract_map,
                        price_source_str=LEGACY_REQ_TICKERS_MARKET_PRICE_SOURCE_STR,
                    )
                )
                return LivePriceSnapshot(
                    account_route_str=account_route_str,
                    snapshot_timestamp_ts=datetime.now(tz=UTC),
                    price_source_str=LEGACY_REQ_TICKERS_MARKET_PRICE_SOURCE_STR,
                    asset_reference_price_map=asset_reference_price_map,
                    asset_reference_source_map_dict=asset_reference_source_map_dict,
                )

            asset_reference_price_map: dict[str, float] = {}
            asset_reference_source_map_dict: dict[str, str] = {}
            ticker_map_dict: dict[str, object] = {}
            for asset_str, contract_obj in contract_map.items():
                ticker_map_dict[str(asset_str)] = ib_obj.reqMktData(
                    contract_obj,
                    genericTickList="225",
                    snapshot=False,
                )
            try:
                ib_obj.sleep(DEFAULT_AUCTION_REFERENCE_WAIT_SECONDS_FLOAT)
                unresolved_asset_list: list[str] = []
                for asset_str, ticker_obj in ticker_map_dict.items():
                    auction_price_float = self._safe_positive_price_float(
                        getattr(ticker_obj, "auctionPrice", None)
                    )
                    if auction_price_float is not None:
                        asset_reference_price_map[asset_str] = auction_price_float
                        asset_reference_source_map_dict[asset_str] = AUCTION_REFERENCE_SOURCE_STR
                        continue
                    market_price_float = self._safe_positive_price_float(ticker_obj.marketPrice())
                    if market_price_float is not None:
                        asset_reference_price_map[asset_str] = market_price_float
                        asset_reference_source_map_dict[asset_str] = (
                            REQ_MKTDATA_MARKET_PRICE_FALLBACK_SOURCE_STR
                        )
                        continue
                    unresolved_asset_list.append(asset_str)
            finally:
                for contract_obj in contract_map.values():
                    try:
                        ib_obj.cancelMktData(contract_obj)
                    except Exception:
                        continue

            if len(unresolved_asset_list) > 0:
                unresolved_contract_map_dict = {
                    asset_str: contract_map[asset_str]
                    for asset_str in unresolved_asset_list
                    if asset_str in contract_map
                }
                fallback_price_map_dict, fallback_source_map_dict = (
                    self._build_req_tickers_price_map_dict(
                        ib_obj=ib_obj,
                        contract_map_dict=unresolved_contract_map_dict,
                        price_source_str=REQ_TICKERS_MARKET_PRICE_FALLBACK_SOURCE_STR,
                    )
                )
                asset_reference_price_map.update(fallback_price_map_dict)
                asset_reference_source_map_dict.update(fallback_source_map_dict)

        price_source_str = self._build_live_price_source_summary_str(
            asset_reference_source_map_dict=asset_reference_source_map_dict,
            default_price_source_str="mixed",
        )
        return LivePriceSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=datetime.now(tz=UTC),
            price_source_str=price_source_str,
            asset_reference_price_map=asset_reference_price_map,
            asset_reference_source_map_dict=asset_reference_source_map_dict,
        )

    def get_capsule_funding_evidence(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
    ) -> dict[str, object]:
        """Check current buying power and each buy without crediting pending sells.

        BuyingPower is a broker notional capacity, not AvailableFunds or cash.
        A WhatIf is an individual order preview against the current account;
        these independent previews do not prove cumulative basket margin or
        guarantee acceptance at the opening price. No order is transmitted.
        This opt-in check is called only by the capsule submission path.
        """
        buy_request_list: list[BrokerOrderRequest] = []
        total_buy_notional_float = 0.0
        for request_obj in broker_order_request_list:
            amount_float = float(request_obj.amount_float)
            if (request_obj.account_route_str != account_route_str
                    or request_obj.unit_str != "shares" or not isfinite(amount_float)):
                raise ValueError("Capsule funding requires finite share orders for the routed account.")
            if amount_float <= 0:
                continue
            reference_price_float = float(request_obj.sizing_reference_price_float)
            if not isfinite(reference_price_float) or reference_price_float <= 0:
                raise ValueError(f"Capsule funding has no valid reference price for {request_obj.asset_str}.")
            if request_obj.broker_order_type_str not in {"MOO", "MKT"}:
                raise ValueError("Capsule funding previews support MOO/MKT share buys only.")
            buy_request_list.append(request_obj)
            total_buy_notional_float += amount_float * reference_price_float
        if not buy_request_list:
            return {"required_bool": False, "total_buy_notional_float": 0.0}
        if not isfinite(total_buy_notional_float):
            raise ValueError("Capsule total buy notional is not finite.")

        with self.connect() as ib_obj:
            # A dedicated connection and completed download avoid reading another
            # account's startup cache. RequestTimeout bounds blocked API replies.
            ib_obj.RequestTimeout = self.timeout_seconds_float
            managed_account_list = ib_obj.managedAccounts()
            if account_route_str not in managed_account_list:
                raise ValueError("Capsule funding account is not visible to IBKR.")
            if managed_account_list == [account_route_str]:
                # The new single-account connection already downloads these
                # values at startup; do not start a duplicate subscription.
                if "accountValues" in ib_obj.wrapper._futures:
                    raise ValueError("Capsule funding account download is incomplete.")
            else:
                ib_obj.reqAccountUpdates(account=account_route_str)
            account_value_list = [
                account_value_obj for account_value_obj in ib_obj.accountValues(account=account_route_str)
                if account_value_obj.account == account_route_str
                and not getattr(account_value_obj, "modelCode", "")
            ]
            trading_type_set = {
                str(account_value_obj.value).strip().upper()
                for account_value_obj in account_value_list if account_value_obj.tag == "TradingType-S"
            }
            # IBKR reference margin types: STKNOPT/STKMRGN are standard margin;
            # PMRGN/GPMRGN are portfolio margin. IRA limited margin is excluded.
            if (len(trading_type_set) != 1
                    or not trading_type_set.issubset({"STKNOPT", "STKMRGN", "PMRGN", "GPMRGN"})):
                raise ValueError("Capsule requires a verified margin account (IBKR TradingType-S).")
            if any(str(account_value_obj.value).lower() != "true"
                   for account_value_obj in account_value_list
                   if account_value_obj.tag.lower() == "accountready"):
                raise ValueError("Capsule funding account is not ready.")
            buying_power_value_list = [
                account_value_obj.value for account_value_obj in account_value_list
                if account_value_obj.tag == "BuyingPower" and account_value_obj.currency == "USD"
            ]
            if len(buying_power_value_list) != 1:
                raise ValueError("Capsule funding requires unambiguous USD BuyingPower from IBKR.")
            try:
                buying_power_float = float(buying_power_value_list[0])
            except (TypeError, ValueError) as exception_obj:
                raise ValueError("Capsule BuyingPower is invalid.") from exception_obj
            if not isfinite(buying_power_float) or not 0 <= buying_power_float < 1e15:
                raise ValueError("Capsule BuyingPower is invalid.")
            if total_buy_notional_float > buying_power_float:
                raise ValueError(
                    f"Capsule buys require {total_buy_notional_float:.2f} USD before sells; "
                    f"IBKR BuyingPower is {buying_power_float:.2f} USD."
                )
            contract_map_dict = self._build_stock_contract_map(
                ib_obj, [request_obj.asset_str for request_obj in buy_request_list]
            )
            preview_list: list[dict[str, object]] = []
            for request_obj in buy_request_list:
                preview_order_obj = Order(
                    action="BUY", totalQuantity=float(request_obj.amount_float), orderType="MKT",
                    tif="OPG" if request_obj.broker_order_type_str == "MOO" else "DAY",
                    account=account_route_str, orderRef=request_obj.order_request_key_str, whatIf=True,
                )
                preview_obj = ib_obj.whatIfOrder(contract_map_dict[request_obj.asset_str], preview_order_obj)
                preview_status_str = str(getattr(preview_obj, "status", ""))
                if preview_status_str not in {"PreSubmitted", "Submitted", "PendingSubmit"}:
                    raise ValueError(f"Capsule buy preview rejected/unavailable for {request_obj.asset_str}: {preview_status_str}")
                margin_field_dict: dict[str, float] = {}
                for field_str in ("initMarginBefore", "initMarginAfter", "maintMarginBefore",
                                  "maintMarginAfter", "equityWithLoanBefore", "equityWithLoanAfter"):
                    try:
                        value_float = float(getattr(preview_obj, field_str, ""))
                    except (TypeError, ValueError) as exception_obj:
                        raise ValueError(f"Capsule buy preview missing {field_str} for {request_obj.asset_str}.") from exception_obj
                    if not isfinite(value_float) or abs(value_float) >= 1e15:
                        raise ValueError(f"Capsule buy preview invalid {field_str} for {request_obj.asset_str}.")
                    margin_field_dict[field_str] = value_float
                if (min(margin_field_dict["initMarginAfter"], margin_field_dict["maintMarginAfter"]) < 0
                        or margin_field_dict["equityWithLoanAfter"] < max(
                            margin_field_dict["initMarginAfter"], margin_field_dict["maintMarginAfter"])):
                    raise ValueError(f"Capsule buy preview has insufficient margin for {request_obj.asset_str}.")
                preview_list.append({
                    "asset_str": request_obj.asset_str,
                    "order_request_key_str": request_obj.order_request_key_str,
                    "amount_float": float(request_obj.amount_float),
                    "status_str": preview_status_str,
                    "warning_str": str(getattr(preview_obj, "warningText", "") or ""),
                    "margin_dict": margin_field_dict,
                })
        return {
            "required_bool": True, "account_route_str": account_route_str,
            "trading_type_str": next(iter(trading_type_set)), "currency_str": "USD",
            "buying_power_float": buying_power_float, "total_buy_notional_float": total_buy_notional_float,
            "pending_sell_credit_float": 0.0, "preview_list": preview_list,
            "basket_margin_guaranteed_bool": False,
            "checked_timestamp_str": datetime.now(tz=UTC).isoformat(),
        }

    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> SubmitBatchResult:
        if not any(request_obj.submission_deadline_timestamp_str is not None or request_obj.execution_deadline_timestamp_str is not None
                for request_obj in broker_order_request_list):
            return self._submit_order_request_list(account_route_str, broker_order_request_list, submitted_timestamp_ts)
        from alpha.live.guarded_dispatch import DispatchFailure, partial_submit_result_obj
        progress_dict = {"attempted_key_list": []}
        try:
            return self._submit_order_request_list(account_route_str, broker_order_request_list,
                submitted_timestamp_ts, progress_dict=progress_dict)
        except Exception as exception_obj:
            raise DispatchFailure(exception_obj, broker_order_request_list,
                progress_dict["attempted_key_list"], partial_submit_result_obj(progress_dict)) from exception_obj

    @contextmanager
    def _guarded_dispatch_connection(self):
        # Keep body errors out of connect's host-fallback loop. A placeOrder
        # exception is an uncertain send, never permission to reconnect/replay it.
        dispatch_exception_obj = None
        with self.connect() as ib_obj:
            try:
                yield ib_obj
            except Exception as exception_obj:
                dispatch_exception_obj = exception_obj
        if dispatch_exception_obj is not None:
            raise dispatch_exception_obj

    def _submit_order_request_list(
        self, account_route_str: str, broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime, *, progress_dict=None,
    ) -> SubmitBatchResult:
        # Validate the whole batch before connecting. Absolute broker expiry keeps
        # a delayed recovery ticket from becoming a new-session DAY market order.
        expiry_fields_by_request_key_dict = {}
        submission_deadline_by_request_key_dict = {}
        for request_obj in broker_order_request_list:
            submission_deadline_str = request_obj.submission_deadline_timestamp_str
            if submission_deadline_str is not None:
                try:
                    submission_deadline_ts = datetime.fromisoformat(submission_deadline_str)
                except (TypeError, ValueError) as exception_obj:
                    raise ValueError("Submission deadline requires an aware ISO timestamp.") from exception_obj
                if submission_deadline_ts.tzinfo is None or request_obj.broker_order_type_str != "MOO":
                    raise ValueError("Submission deadline requires an aware timestamp and MOO order.")
                submission_deadline_by_request_key_dict[request_obj.order_request_key_str] = submission_deadline_ts
            deadline_str = request_obj.execution_deadline_timestamp_str
            if deadline_str is None:
                continue
            if request_obj.broker_order_type_str != "MKT":
                raise ValueError("execution_deadline_timestamp_str is supported only for MKT orders.")
            try:
                deadline_ts = datetime.fromisoformat(deadline_str)
            except (TypeError, ValueError) as exception_obj:
                raise ValueError("execution_deadline_timestamp_str requires an aware ISO timestamp.") from exception_obj
            if deadline_ts.tzinfo is None or deadline_ts.utcoffset() is None:
                raise ValueError("execution_deadline_timestamp_str requires an aware ISO timestamp.")
            expiry_fields_by_request_key_dict[request_obj.order_request_key_str] = {
                # IBKR's hyphenated format is explicitly UTC, independent of TWS timezone.
                "goodTillDate": deadline_ts.astimezone(UTC).strftime("%Y%m%d-%H:%M:%S"),
                "outsideRth": False,
            }
        with (self.connect() if progress_dict is None else self._guarded_dispatch_connection()) as ib_obj:
            contract_map = self._build_stock_contract_map(
                ib_obj,
                [broker_order_request_obj.asset_str for broker_order_request_obj in broker_order_request_list],
            )
            local_broker_order_record_list: list[BrokerOrderRecord] = []
            local_broker_order_event_list: list[BrokerOrderEvent] = []
            local_broker_order_fill_list: list[BrokerOrderFill] = []
            if progress_dict is not None:
                progress_dict.update(record_list=local_broker_order_record_list,
                    event_list=local_broker_order_event_list, fill_list=local_broker_order_fill_list)
            local_broker_order_id_alias_set_by_request_key_map_dict: dict[str, set[str]] = {}

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
                        orderRef=broker_order_request_obj.order_request_key_str,
                    )
                elif broker_order_request_obj.broker_order_type_str == "MOC":
                    broker_order_obj = Order(
                        action=order_action_str,
                        totalQuantity=total_quantity_float,
                        orderType="MOC",
                        tif="DAY",
                        account=account_route_str,
                        orderRef=broker_order_request_obj.order_request_key_str,
                    )
                elif broker_order_request_obj.broker_order_type_str == "MKT":
                    expiry_fields_dict = expiry_fields_by_request_key_dict.get(
                        broker_order_request_obj.order_request_key_str, {})
                    broker_order_obj = Order(
                        action=order_action_str,
                        totalQuantity=total_quantity_float,
                        orderType="MKT",
                        tif="GTD" if expiry_fields_dict else "DAY",
                        account=account_route_str,
                        orderRef=broker_order_request_obj.order_request_key_str,
                        **expiry_fields_dict,
                    )
                elif broker_order_request_obj.broker_order_type_str == "LMT":
                    if (
                        broker_order_request_obj.limit_price_float is None
                        or float(broker_order_request_obj.limit_price_float) <= 0.0
                    ):
                        raise ValueError("LMT orders require a positive limit_price_float.")
                    broker_order_obj = Order(
                        action=order_action_str,
                        totalQuantity=total_quantity_float,
                        orderType="LMT",
                        tif="DAY",
                        lmtPrice=float(broker_order_request_obj.limit_price_float),
                        account=account_route_str,
                        orderRef=broker_order_request_obj.order_request_key_str,
                    )
                else:
                    raise ValueError(
                        "Unsupported broker_order_type_str "
                        f"'{broker_order_request_obj.broker_order_type_str}'."
                    )

                submission_deadline_ts = submission_deadline_by_request_key_dict.get(broker_order_request_obj.order_request_key_str)
                # *** CRITICAL *** Check after connect/contract qualification,
                # immediately before EACH dispatch. Preserve MKT/OPG semantics.
                if submission_deadline_ts is not None and datetime.now(UTC) >= submission_deadline_ts:
                    raise ValueError("MOO submission deadline passed before broker dispatch; reconcile any earlier legs.")
                if progress_dict is not None:
                    progress_dict["attempted_key_list"].append(broker_order_request_obj.order_request_key_str)
                trade_obj = ib_obj.placeOrder(
                    contract_map[broker_order_request_obj.asset_str],
                    broker_order_obj,
                )
                local_broker_order_id_alias_set_by_request_key_map_dict[
                    broker_order_request_obj.order_request_key_str
                ] = self._build_broker_order_id_alias_set(
                    order_obj=trade_obj.order,
                    order_status_obj=trade_obj.orderStatus,
                )
                broker_order_record_obj = self._build_broker_order_record_obj(
                    trade_obj=trade_obj,
                    account_route_str=account_route_str,
                    submitted_timestamp_ts=submitted_timestamp_ts,
                    decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                    vplan_id_int=broker_order_request_obj.vplan_id_int,
                    fallback_asset_str=broker_order_request_obj.asset_str,
                )
                local_broker_order_record_list.append(
                    BrokerOrderRecord(
                        **{
                            **broker_order_record_obj.__dict__,
                            "order_request_key_str": broker_order_request_obj.order_request_key_str,
                            "broker_order_type_str": broker_order_request_obj.broker_order_type_str,
                            "unit_str": broker_order_request_obj.unit_str,
                            "amount_float": float(broker_order_request_obj.amount_float),
                            "submission_key_str": broker_order_request_obj.submission_key_str,
                            "raw_payload_dict": {
                                **broker_order_record_obj.raw_payload_dict,
                                "submission_key_str": broker_order_request_obj.submission_key_str,
                                "order_request_key_str": broker_order_request_obj.order_request_key_str,
                                "limit_price_float": broker_order_request_obj.limit_price_float,
                                **({"submission_deadline_timestamp_str": broker_order_request_obj.submission_deadline_timestamp_str}
                                    if broker_order_request_obj.submission_deadline_timestamp_str is not None else {}),
                            },
                        }
                    )
                )
                broker_order_id_str = local_broker_order_record_list[-1].broker_order_id_str
                local_broker_order_event_list.extend(
                    [
                        BrokerOrderEvent(
                            **{
                                **broker_order_event_obj.__dict__,
                                "order_request_key_str": broker_order_request_obj.order_request_key_str,
                                "submission_key_str": broker_order_request_obj.submission_key_str,
                                "raw_payload_dict": {
                                    **broker_order_event_obj.raw_payload_dict,
                                    "submission_key_str": broker_order_request_obj.submission_key_str,
                                    "order_request_key_str": broker_order_request_obj.order_request_key_str,
                                    "limit_price_float": broker_order_request_obj.limit_price_float,
                                },
                            }
                        )
                        for broker_order_event_obj in self._build_broker_order_event_list(
                            trade_obj=trade_obj,
                            account_route_str=account_route_str,
                            decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                            vplan_id_int=broker_order_request_obj.vplan_id_int,
                            fallback_asset_str=broker_order_request_obj.asset_str,
                        )
                    ]
                )
                for fill_obj in trade_obj.fills:
                    local_broker_order_fill_list.append(
                        BrokerOrderFill(
                            broker_order_id_str=broker_order_id_str,
                            decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                            vplan_id_int=broker_order_request_obj.vplan_id_int,
                            account_route_str=account_route_str,
                            asset_str=broker_order_request_obj.asset_str,
                            fill_amount_float=self._signed_fill_amount_float_from_execution(fill_obj),
                            fill_price_float=float(fill_obj.execution.price),
                            fill_timestamp_ts=self._as_utc_timestamp_ts(fill_obj.time),
                            raw_payload_dict={
                                "exec_id_str": fill_obj.execution.execId,
                                "order_request_key_str": broker_order_request_obj.order_request_key_str,
                            },
                        )
                    )

            broker_snapshot_order_record_list: list[BrokerOrderRecord] = []
            broker_snapshot_order_event_list: list[BrokerOrderEvent] = []
            broker_snapshot_fill_list: list[BrokerOrderFill] = []
            broker_order_ack_list: list[BrokerOrderAck] = []
            request_count_int = len(broker_order_request_list)
            request_submission_key_str = (
                None
                if request_count_int == 0
                else broker_order_request_list[0].submission_key_str
            )
            normalized_allowed_broker_order_id_set = {
                broker_order_id_str
                for broker_order_id_alias_set in local_broker_order_id_alias_set_by_request_key_map_dict.values()
                for broker_order_id_str in broker_order_id_alias_set
            }
            submit_ack_deadline_float = monotonic() + DEFAULT_SUBMIT_ACK_TIMEOUT_SECONDS_FLOAT
            while True:
                (
                    broker_snapshot_order_record_list,
                    broker_snapshot_order_event_list,
                    broker_snapshot_fill_list,
                ) = self._get_recent_order_state_snapshot_from_connection(
                    ib_obj=ib_obj,
                    account_route_str=account_route_str,
                    since_timestamp_ts=submitted_timestamp_ts,
                    submission_key_str=request_submission_key_str,
                    allowed_broker_order_id_set=normalized_allowed_broker_order_id_set,
                )
                broker_order_ack_list = self._build_broker_order_ack_list(
                    broker_order_request_list=broker_order_request_list,
                    broker_order_record_list=broker_snapshot_order_record_list,
                    broker_order_event_list=broker_snapshot_order_event_list,
                    broker_order_fill_list=broker_snapshot_fill_list,
                    local_broker_order_id_alias_set_by_request_key_map_dict=(
                        local_broker_order_id_alias_set_by_request_key_map_dict
                    ),
                )
                if all(
                    broker_order_ack_obj.broker_response_ack_bool
                    for broker_order_ack_obj in broker_order_ack_list
                ):
                    break
                if monotonic() >= submit_ack_deadline_float:
                    break
                ib_obj.sleep(DEFAULT_SUBMIT_ACK_POLL_SECONDS_FLOAT)

        broker_ack_count_int = sum(
            1 for broker_order_ack_obj in broker_order_ack_list if broker_order_ack_obj.broker_response_ack_bool
        )
        ack_coverage_ratio_float = (
            1.0
            if request_count_int == 0
            else float(broker_ack_count_int) / float(request_count_int)
        )
        missing_ack_asset_list = sorted(
            {
                str(broker_order_ack_obj.asset_str)
                for broker_order_ack_obj in broker_order_ack_list
                if not broker_order_ack_obj.broker_response_ack_bool
            }
        )
        submit_ack_status_str = (
            "complete"
            if len(missing_ack_asset_list) == 0
            else "missing_critical"
        )
        return SubmitBatchResult(
            broker_order_record_list=[
                *local_broker_order_record_list,
                *broker_snapshot_order_record_list,
            ],
            broker_order_event_list=[
                *local_broker_order_event_list,
                *broker_snapshot_order_event_list,
            ],
            broker_order_fill_list=[
                *local_broker_order_fill_list,
                *broker_snapshot_fill_list,
            ],
            broker_order_ack_list=broker_order_ack_list,
            ack_coverage_ratio_float=ack_coverage_ratio_float,
            missing_ack_asset_list=missing_ack_asset_list,
            submit_ack_status_str=submit_ack_status_str,
        )

    def _get_recent_fill_list_from_connection(
        self,
        ib_obj: IB,
        account_route_str: str,
        since_timestamp_ts: datetime,
        allowed_broker_order_id_set: set[str] | None = None,
        *,
        require_account_match_bool: bool = False,
    ) -> list[BrokerOrderFill]:
        ib_fill_list = ib_obj.reqExecutions(ExecutionFilter(acctCode=account_route_str))
        normalized_allowed_broker_order_id_set = (
            None
            if allowed_broker_order_id_set is None
            else {
                str(broker_order_id_str)
                for broker_order_id_str in allowed_broker_order_id_set
            }
        )
        broker_order_fill_list: list[BrokerOrderFill] = []
        for fill_obj in ib_fill_list:
            if require_account_match_bool and str(getattr(fill_obj.execution, "acctNumber", "")) != account_route_str:
                continue
            fill_timestamp_ts = self._as_utc_timestamp_ts(fill_obj.time)
            if fill_timestamp_ts < since_timestamp_ts:
                continue
            if normalized_allowed_broker_order_id_set is not None:
                candidate_broker_order_id_set = self._build_broker_order_id_alias_set(
                    execution_obj=fill_obj.execution,
                )
                if len(candidate_broker_order_id_set & normalized_allowed_broker_order_id_set) == 0:
                    continue
            broker_order_fill_list.append(
                BrokerOrderFill(
                    broker_order_id_str=str(fill_obj.execution.permId or fill_obj.execution.orderId),
                    decision_plan_id_int=None,
                    vplan_id_int=None,
                    account_route_str=account_route_str,
                    asset_str=_canonical_stock_symbol_str(fill_obj.contract.symbol),
                    fill_amount_float=self._signed_fill_amount_float_from_execution(fill_obj),
                    fill_price_float=float(fill_obj.execution.price),
                    fill_timestamp_ts=fill_timestamp_ts,
                    raw_payload_dict={
                        "exec_id_str": fill_obj.execution.execId,
                        "perm_id_int": int(self._build_perm_id_int(execution_obj=fill_obj.execution) or 0),
                    },
                )
            )
        return broker_order_fill_list

    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> list[BrokerOrderFill]:
        with self.connect() as ib_obj:
            return self._get_recent_fill_list_from_connection(
                ib_obj=ib_obj,
                account_route_str=account_route_str,
                since_timestamp_ts=since_timestamp_ts,
                allowed_broker_order_id_set=allowed_broker_order_id_set,
            )

    def _get_recent_order_state_snapshot_from_connection(
        self,
        ib_obj: IB,
        account_route_str: str,
        since_timestamp_ts: datetime,
        submission_key_str: str | None = None,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
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

        normalized_allowed_broker_order_id_set = (
            None
            if allowed_broker_order_id_set is None
            else {
                str(broker_order_id_str)
                for broker_order_id_str in allowed_broker_order_id_set
            }
        )
        broker_order_record_map: dict[str, BrokerOrderRecord] = {}
        broker_order_event_list: list[BrokerOrderEvent] = []
        relevant_broker_order_id_set: set[str] = set()
        for snapshot_source_str, trade_list in (
            ("open_order", open_trade_list),
            ("completed_order", completed_trade_list),
        ):
            for trade_obj in trade_list:
                candidate_broker_order_id_set = self._build_broker_order_id_alias_set(
                    order_obj=trade_obj.order,
                    order_status_obj=trade_obj.orderStatus,
                )
                submission_key_matches_bool = (
                    submission_key_str is not None
                    and self._build_submission_key_str(trade_obj) == str(submission_key_str)
                )
                broker_order_id_matches_bool = (
                    normalized_allowed_broker_order_id_set is not None
                    and len(candidate_broker_order_id_set & normalized_allowed_broker_order_id_set) > 0
                )
                candidate_broker_order_event_list = [
                    broker_order_event_obj.__class__(
                        **{
                            **broker_order_event_obj.__dict__,
                            "raw_payload_dict": {
                                **broker_order_event_obj.raw_payload_dict,
                                "snapshot_source_str": snapshot_source_str,
                            },
                        }
                    )
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
                include_trade_bool = submission_key_matches_bool or broker_order_id_matches_bool
                if submission_key_str is None and normalized_allowed_broker_order_id_set is None:
                    include_trade_bool = len(candidate_broker_order_event_list) > 0
                if not include_trade_bool:
                    continue
                broker_order_record_obj = self._build_broker_order_record_obj(
                    trade_obj=trade_obj,
                    account_route_str=account_route_str,
                    submitted_timestamp_ts=since_timestamp_ts,
                    decision_plan_id_int=None,
                    vplan_id_int=None,
                    fallback_asset_str=str(getattr(trade_obj.contract, "symbol", "")),
                )
                broker_order_record_map[broker_order_record_obj.broker_order_id_str] = (
                    broker_order_record_obj.__class__(
                        **{
                            **broker_order_record_obj.__dict__,
                            "raw_payload_dict": {
                                **broker_order_record_obj.raw_payload_dict,
                                "snapshot_source_str": snapshot_source_str,
                            },
                        }
                    )
                )
                broker_order_event_list.extend(candidate_broker_order_event_list)
                relevant_broker_order_id_set.update(candidate_broker_order_id_set)
        if normalized_allowed_broker_order_id_set is not None:
            relevant_broker_order_id_set.update(normalized_allowed_broker_order_id_set)
        broker_order_fill_list = self._get_recent_fill_list_from_connection(
            ib_obj=ib_obj,
            account_route_str=account_route_str,
            since_timestamp_ts=since_timestamp_ts,
            allowed_broker_order_id_set=(
                relevant_broker_order_id_set
                if len(relevant_broker_order_id_set) > 0
                else normalized_allowed_broker_order_id_set
            ),
        )
        if submission_key_str is None and normalized_allowed_broker_order_id_set is None:
            relevant_broker_order_id_set.update(
                str(broker_order_fill_obj.broker_order_id_str)
                for broker_order_fill_obj in broker_order_fill_list
            )
            if len(relevant_broker_order_id_set) > 0:
                broker_order_record_map = {
                    broker_order_id_str: broker_order_record_obj
                    for broker_order_id_str, broker_order_record_obj in broker_order_record_map.items()
                    if broker_order_id_str in relevant_broker_order_id_set
                }
        return list(broker_order_record_map.values()), broker_order_event_list, broker_order_fill_list

    def _build_broker_order_ack_list(
        self,
        broker_order_request_list: list[BrokerOrderRequest],
        broker_order_record_list: list[BrokerOrderRecord],
        broker_order_event_list: list[BrokerOrderEvent],
        broker_order_fill_list: list[BrokerOrderFill],
        local_broker_order_id_alias_set_by_request_key_map_dict: dict[str, set[str]],
    ) -> list[BrokerOrderAck]:
        latest_broker_order_record_by_request_key_map_dict: dict[str, BrokerOrderRecord] = {}
        broker_order_event_list_by_request_key_map_dict: dict[str, list[BrokerOrderEvent]] = {}
        broker_order_id_alias_set_by_request_key_map_dict: dict[str, set[str]] = {
            str(order_request_key_str): set(broker_order_id_alias_set)
            for order_request_key_str, broker_order_id_alias_set in local_broker_order_id_alias_set_by_request_key_map_dict.items()
        }

        for broker_order_record_obj in broker_order_record_list:
            order_request_key_str = str(broker_order_record_obj.order_request_key_str or "").strip()
            if order_request_key_str != "":
                existing_broker_order_record_obj = latest_broker_order_record_by_request_key_map_dict.get(
                    order_request_key_str
                )
                candidate_timestamp_ts = (
                    broker_order_record_obj.last_status_timestamp_ts
                    or broker_order_record_obj.submitted_timestamp_ts
                )
                existing_timestamp_ts = None
                if existing_broker_order_record_obj is not None:
                    existing_timestamp_ts = (
                        existing_broker_order_record_obj.last_status_timestamp_ts
                        or existing_broker_order_record_obj.submitted_timestamp_ts
                    )
                if existing_broker_order_record_obj is None or (
                    existing_timestamp_ts is not None
                    and candidate_timestamp_ts >= existing_timestamp_ts
                ) or existing_timestamp_ts is None:
                    latest_broker_order_record_by_request_key_map_dict[order_request_key_str] = broker_order_record_obj
                broker_order_id_alias_set_by_request_key_map_dict.setdefault(order_request_key_str, set()).add(
                    str(broker_order_record_obj.broker_order_id_str)
                )
                perm_id_int = int(broker_order_record_obj.raw_payload_dict.get("perm_id_int", 0) or 0)
                if perm_id_int > 0:
                    broker_order_id_alias_set_by_request_key_map_dict[order_request_key_str].add(str(perm_id_int))

        for broker_order_event_obj in broker_order_event_list:
            order_request_key_str = str(broker_order_event_obj.order_request_key_str or "").strip()
            if order_request_key_str == "":
                continue
            broker_order_event_list_by_request_key_map_dict.setdefault(order_request_key_str, []).append(
                broker_order_event_obj
            )
            broker_order_id_alias_set_by_request_key_map_dict.setdefault(order_request_key_str, set()).add(
                str(broker_order_event_obj.broker_order_id_str)
            )

        broker_order_ack_list: list[BrokerOrderAck] = []
        for broker_order_request_obj in broker_order_request_list:
            order_request_key_str = broker_order_request_obj.order_request_key_str
            candidate_broker_order_id_alias_set = set(
                broker_order_id_alias_set_by_request_key_map_dict.get(order_request_key_str, set())
            )
            broker_order_record_obj = latest_broker_order_record_by_request_key_map_dict.get(order_request_key_str)
            broker_order_event_candidate_list = broker_order_event_list_by_request_key_map_dict.get(
                order_request_key_str,
                [],
            )
            broker_response_ack_bool = False
            ack_source_str = "missing"
            broker_order_id_str = None
            perm_id_int = None
            response_timestamp_ts = None
            raw_payload_dict = {
                "submission_key_str": broker_order_request_obj.submission_key_str,
                "order_request_key_str": order_request_key_str,
            }

            if broker_order_record_obj is None and len(candidate_broker_order_id_alias_set) > 0:
                broker_order_record_obj = next(
                    (
                        candidate_broker_order_record_obj
                        for candidate_broker_order_record_obj in broker_order_record_list
                        if str(candidate_broker_order_record_obj.broker_order_id_str)
                        in candidate_broker_order_id_alias_set
                    ),
                    None,
                )

            if broker_order_record_obj is not None:
                broker_response_ack_bool = True
                ack_source_str = str(
                    broker_order_record_obj.raw_payload_dict.get("snapshot_source_str", "snapshot_record")
                )
                broker_order_id_str = str(broker_order_record_obj.broker_order_id_str)
                perm_id_raw_obj = broker_order_record_obj.raw_payload_dict.get("perm_id_int")
                perm_id_int = None if int(perm_id_raw_obj or 0) <= 0 else int(perm_id_raw_obj)
                response_timestamp_ts = (
                    broker_order_record_obj.last_status_timestamp_ts
                    or broker_order_record_obj.submitted_timestamp_ts
                )
                raw_payload_dict.update(
                    {
                        "broker_status_str": broker_order_record_obj.status_str,
                        "snapshot_source_str": broker_order_record_obj.raw_payload_dict.get(
                            "snapshot_source_str"
                        ),
                    }
                )
            elif len(broker_order_event_candidate_list) > 0:
                latest_broker_order_event_obj = max(
                    broker_order_event_candidate_list,
                    key=lambda broker_order_event_obj: (
                        broker_order_event_obj.event_timestamp_ts or datetime.min.replace(tzinfo=UTC)
                    ),
                )
                broker_response_ack_bool = True
                ack_source_str = str(latest_broker_order_event_obj.event_source_str or "event")
                broker_order_id_str = str(latest_broker_order_event_obj.broker_order_id_str)
                response_timestamp_ts = latest_broker_order_event_obj.event_timestamp_ts
                raw_payload_dict.update(
                    {
                        "broker_status_str": latest_broker_order_event_obj.status_str,
                        "event_source_str": latest_broker_order_event_obj.event_source_str,
                    }
                )
            elif len(candidate_broker_order_id_alias_set) > 0:
                matching_broker_order_fill_obj = next(
                    (
                        broker_order_fill_obj
                        for broker_order_fill_obj in broker_order_fill_list
                        if str(broker_order_fill_obj.broker_order_id_str)
                        in candidate_broker_order_id_alias_set
                    ),
                    None,
                )
                if matching_broker_order_fill_obj is not None:
                    broker_response_ack_bool = True
                    ack_source_str = "fill"
                    broker_order_id_str = str(matching_broker_order_fill_obj.broker_order_id_str)
                    perm_id_raw_obj = matching_broker_order_fill_obj.raw_payload_dict.get("perm_id_int")
                    perm_id_int = None if int(perm_id_raw_obj or 0) <= 0 else int(perm_id_raw_obj)
                    response_timestamp_ts = matching_broker_order_fill_obj.fill_timestamp_ts
                    raw_payload_dict.update(
                        {
                            "fill_price_float": float(matching_broker_order_fill_obj.fill_price_float),
                            "fill_amount_float": float(matching_broker_order_fill_obj.fill_amount_float),
                        }
                    )

            broker_order_ack_list.append(
                BrokerOrderAck(
                    decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                    vplan_id_int=broker_order_request_obj.vplan_id_int,
                    account_route_str=broker_order_request_obj.account_route_str,
                    order_request_key_str=order_request_key_str,
                    asset_str=broker_order_request_obj.asset_str,
                    broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                    local_submit_ack_bool=True,
                    broker_response_ack_bool=broker_response_ack_bool,
                    ack_status_str=(
                        "broker_acked"
                        if broker_response_ack_bool
                        else "missing_critical"
                    ),
                    ack_source_str=ack_source_str,
                    broker_order_id_str=broker_order_id_str,
                    perm_id_int=perm_id_int,
                    response_timestamp_ts=response_timestamp_ts,
                    raw_payload_dict=raw_payload_dict,
                )
            )
        return broker_order_ack_list

    def get_recent_order_state_snapshot(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        submission_key_str: str | None = None,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        with self.connect() as ib_obj:
            return self._get_recent_order_state_snapshot_from_connection(
                ib_obj=ib_obj,
                account_route_str=account_route_str,
                since_timestamp_ts=since_timestamp_ts,
                submission_key_str=submission_key_str,
                allowed_broker_order_id_set=allowed_broker_order_id_set,
            )

    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> list[BrokerOrderFill]:
        with self.connect() as ib_obj:
            ib_fill_list = ib_obj.reqExecutions(ExecutionFilter(acctCode=account_route_str))

        normalized_allowed_broker_order_id_set = (
            None
            if allowed_broker_order_id_set is None
            else {
                str(broker_order_id_str)
                for broker_order_id_str in allowed_broker_order_id_set
            }
        )
        broker_order_fill_list: list[BrokerOrderFill] = []
        for fill_obj in ib_fill_list:
            fill_timestamp_ts = self._as_utc_timestamp_ts(fill_obj.time)
            if fill_timestamp_ts < since_timestamp_ts:
                continue
            if normalized_allowed_broker_order_id_set is not None:
                candidate_broker_order_id_set = self._build_broker_order_id_alias_set(
                    execution_obj=fill_obj.execution,
                )
                if len(candidate_broker_order_id_set & normalized_allowed_broker_order_id_set) == 0:
                    continue
            broker_order_fill_list.append(
                BrokerOrderFill(
                    broker_order_id_str=str(fill_obj.execution.permId or fill_obj.execution.orderId),
                    decision_plan_id_int=None,
                    vplan_id_int=None,
                    account_route_str=account_route_str,
                    asset_str=_canonical_stock_symbol_str(fill_obj.contract.symbol),
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
        submission_key_str: str | None = None,
        allowed_broker_order_id_set: set[str] | None = None,
        *,
        capsule_evidence_bool: bool = False,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        with self.connect() as ib_obj:
            open_trade_list = [
                trade_obj
                for trade_obj in (ib_obj.reqAllOpenOrders() if capsule_evidence_bool else ib_obj.reqOpenOrders())
                if str(trade_obj.order.account) == account_route_str
            ]
            completed_trade_list = [
                trade_obj
                for trade_obj in ib_obj.reqCompletedOrders(apiOnly=False)
                if str(trade_obj.order.account) == account_route_str
            ]
            capsule_fill_list = (
                self._get_recent_fill_list_from_connection(ib_obj, account_route_str, since_timestamp_ts,
                    require_account_match_bool=True) if capsule_evidence_bool else None
            )
        recent_fill_order_id_set = {fill_obj.broker_order_id_str for fill_obj in capsule_fill_list or []}

        normalized_allowed_broker_order_id_set = (
            None
            if allowed_broker_order_id_set is None
            else {
                str(broker_order_id_str)
                for broker_order_id_str in allowed_broker_order_id_set
            }
        )
        broker_order_record_map: dict[str, BrokerOrderRecord] = {}
        broker_order_event_list: list[BrokerOrderEvent] = []
        relevant_broker_order_id_set: set[str] = set()
        open_broker_order_id_set = {
            broker_order_id_str
            for trade_obj in open_trade_list
            for broker_order_id_str in self._build_broker_order_id_alias_set(
                order_obj=trade_obj.order,
                order_status_obj=trade_obj.orderStatus,
            )
        }
        snapshot_trade_pair_list = [
            ("open_order", trade_obj) for trade_obj in open_trade_list
        ] + [
            ("completed_order", trade_obj) for trade_obj in completed_trade_list
        ]
        for snapshot_source_str, trade_obj in snapshot_trade_pair_list:
            candidate_broker_order_id_set = self._build_broker_order_id_alias_set(
                order_obj=trade_obj.order,
                order_status_obj=trade_obj.orderStatus,
            )
            submission_key_matches_bool = (
                submission_key_str is not None
                and self._build_submission_key_str(trade_obj) == str(submission_key_str)
            )
            broker_order_id_matches_bool = (
                normalized_allowed_broker_order_id_set is not None
                and len(candidate_broker_order_id_set & normalized_allowed_broker_order_id_set) > 0
            )
            candidate_broker_order_event_list = [
                broker_order_event_obj.__class__(
                    **{
                        **broker_order_event_obj.__dict__,
                        "raw_payload_dict": {
                            **broker_order_event_obj.raw_payload_dict,
                            "snapshot_source_str": snapshot_source_str,
                        },
                    }
                ) if capsule_evidence_bool else broker_order_event_obj
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
            include_trade_bool = submission_key_matches_bool or broker_order_id_matches_bool
            if submission_key_str is None and normalized_allowed_broker_order_id_set is None:
                include_trade_bool = len(candidate_broker_order_event_list) > 0 or (
                    capsule_evidence_bool
                    and bool(candidate_broker_order_id_set & recent_fill_order_id_set)
                )
            if not include_trade_bool:
                continue
            broker_order_record_obj = self._build_broker_order_record_obj(
                trade_obj=trade_obj,
                account_route_str=account_route_str,
                submitted_timestamp_ts=since_timestamp_ts,
                decision_plan_id_int=None,
                vplan_id_int=None,
                fallback_asset_str=str(getattr(trade_obj.contract, "symbol", "")),
                use_completed_quantity_bool=capsule_evidence_bool and snapshot_source_str == "completed_order",
            )
            broker_order_record_map[broker_order_record_obj.broker_order_id_str] = (
                broker_order_record_obj.__class__(
                    **{
                        **broker_order_record_obj.__dict__,
                        "raw_payload_dict": {
                            **broker_order_record_obj.raw_payload_dict,
                            "snapshot_source_str": snapshot_source_str,
                            # Preserve precedence while retaining an open observation.
                            "open_order_observed_bool": bool(candidate_broker_order_id_set & open_broker_order_id_set),
                        },
                    }
                ) if capsule_evidence_bool else broker_order_record_obj
            )
            broker_order_event_list.extend(candidate_broker_order_event_list)
            relevant_broker_order_id_set.update(candidate_broker_order_id_set)
        if normalized_allowed_broker_order_id_set is not None:
            relevant_broker_order_id_set.update(normalized_allowed_broker_order_id_set)
        if capsule_fill_list is not None:
            broker_order_fill_list = [fill_obj for fill_obj in capsule_fill_list
                if fill_obj.broker_order_id_str in relevant_broker_order_id_set]
        else:
            broker_order_fill_list = self.get_recent_fill_list(
                account_route_str=account_route_str,
                since_timestamp_ts=since_timestamp_ts,
                allowed_broker_order_id_set=(
                    relevant_broker_order_id_set
                    if len(relevant_broker_order_id_set) > 0
                    else normalized_allowed_broker_order_id_set
                ),
            )
        if submission_key_str is None and normalized_allowed_broker_order_id_set is None:
            relevant_broker_order_id_set.update(
                str(broker_order_fill_obj.broker_order_id_str)
                for broker_order_fill_obj in broker_order_fill_list
            )
            if len(relevant_broker_order_id_set) > 0:
                filtered_broker_order_record_map = {
                    broker_order_id_str: broker_order_record_obj
                    for broker_order_id_str, broker_order_record_obj in broker_order_record_map.items()
                    if broker_order_id_str in relevant_broker_order_id_set
                }
                broker_order_record_map = filtered_broker_order_record_map
        return list(broker_order_record_map.values()), broker_order_event_list, broker_order_fill_list

    def get_capsule_order_state_snapshot(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        submission_key_str: str | None = None,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        """Completed quantities and recent manual fills, opt-in for capsule only."""
        return self.get_recent_order_state_snapshot(
            account_route_str, since_timestamp_ts, submission_key_str,
            allowed_broker_order_id_set, capsule_evidence_bool=True)

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
                    ticker_open_map[_canonical_stock_symbol_str(ticker_obj.contract.symbol)] = official_open_price_float

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

    def get_tick_open_price_list(
        self,
        account_route_str: str,
        asset_str_list: list[str],
        session_open_timestamp_ts: datetime,
        session_calendar_id_str: str,
    ) -> list[SessionOpenPrice]:
        if len(asset_str_list) == 0:
            return []

        # *** CRITICAL*** ticker.open is a current field, not a historical query.
        # Only read it during the requested exchange session, after its real open.
        market_open_timestamp_ts = scheduler_utils.to_market_timestamp_ts(
            session_open_timestamp_ts,
            session_calendar_id_str,
        )
        session_label_ts = scheduler_utils.session_label_from_timestamp_ts(
            market_open_timestamp_ts, session_calendar_id_str,
        )
        if session_label_ts is None:
            raise ValueError("IBKR tick-open request must name a trading session.")
        canonical_open_ts = scheduler_utils.get_session_open_timestamp_ts(session_label_ts, session_calendar_id_str)
        if market_open_timestamp_ts != canonical_open_ts:
            raise ValueError("IBKR tick-open request must use the exchange session open.")
        session_date_str = market_open_timestamp_ts.date().isoformat()

        def checked_observation_timestamp_ts() -> datetime:
            observed_ts = datetime.now(tz=UTC)
            market_observed_ts = scheduler_utils.to_market_timestamp_ts(observed_ts, session_calendar_id_str)
            if market_observed_ts.date() != market_open_timestamp_ts.date() or observed_ts < canonical_open_ts:
                raise RuntimeError(
                    f"Uncached IBKR tick-open is available only during target session {session_date_str} "
                    "after its open; a different day requires a saved target-session price."
                )
            return observed_ts

        checked_observation_timestamp_ts()
        normalized_asset_str_list = sorted({str(asset_str) for asset_str in asset_str_list})
        with self.connect() as ib_obj:
            contract_map_dict = self._build_stock_contract_map(ib_obj, normalized_asset_str_list)
            # *** CRITICAL*** Option A intentionally reads only ticker.open.
            # No historical one-minute fallback is allowed for incubation open fills.
            ticker_list = ib_obj.reqTickers(*contract_map_dict.values())
            ticker_open_map_dict: dict[str, float] = {}
            for ticker_obj in ticker_list:
                official_open_price_float = self._safe_positive_price_float(
                    getattr(ticker_obj, "open", None)
                )
                if official_open_price_float is None:
                    continue
                ticker_open_map_dict[_canonical_stock_symbol_str(ticker_obj.contract.symbol)] = official_open_price_float

        # Check again after I/O, so a request crossing the session date cannot
        # relabel a new day's value. This is capture time, not exchange print time.
        observed_ts = checked_observation_timestamp_ts()

        session_open_price_list: list[SessionOpenPrice] = []
        for asset_str in normalized_asset_str_list:
            official_open_price_float = ticker_open_map_dict.get(asset_str)
            session_open_price_list.append(
                SessionOpenPrice(
                    session_date_str=session_date_str,
                    account_route_str=account_route_str,
                    asset_str=asset_str,
                    official_open_price_float=official_open_price_float,
                    open_price_source_str=(
                        IBKR_TICK_OPEN_SOURCE_STR
                        if official_open_price_float is not None
                        else None
                    ),
                    snapshot_timestamp_ts=observed_ts,
                    raw_payload_dict=(
                        {"ticker_open_float": official_open_price_float}
                        if official_open_price_float is not None
                        else {}
                    ),
                )
            )
        return session_open_price_list
