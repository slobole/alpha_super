from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import UTC, datetime
from typing import Iterable
import uuid

from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live import scheduler_utils
from alpha.live.models import (
    BrokerOrderFill,
    BrokerOrderEvent,
    BrokerOrderRecord,
    BrokerOrderRequest,
    BrokerSnapshot,
    LivePriceSnapshot,
    SessionOpenPrice,
)


PAPER_ACCOUNT_PREFIX_TUPLE: tuple[str, ...] = ("DU",)


def account_route_looks_like_paper_bool(account_route_str: str) -> bool:
    normalized_account_route_str = str(account_route_str).strip().upper()
    return normalized_account_route_str.startswith(PAPER_ACCOUNT_PREFIX_TUPLE)


def infer_ibkr_account_mode_str(account_route_str: str) -> str:
    if account_route_looks_like_paper_bool(account_route_str):
        return "paper"
    return "live"


def validate_account_route_matches_mode(
    mode_str: str,
    account_route_str: str,
) -> None:
    inferred_mode_str = infer_ibkr_account_mode_str(account_route_str)
    if mode_str == "paper" and inferred_mode_str != "paper":
        raise ValueError(
            f"Manifest mode_str '{mode_str}' expects a paper-style IBKR account_route_str, "
            f"but received '{account_route_str}'. Paper accounts are expected to use a {PAPER_ACCOUNT_PREFIX_TUPLE} prefix heuristic."
        )
    if mode_str == "live" and inferred_mode_str != "live":
        raise ValueError(
            f"Manifest mode_str '{mode_str}' expects a live-style IBKR account_route_str, "
            f"but received '{account_route_str}', which matches the paper-account prefix heuristic {PAPER_ACCOUNT_PREFIX_TUPLE}."
        )


def _amount_in_shares_float(
    unit_str: str,
    amount_float: float,
    target_bool: bool,
    sizing_reference_price_float: float,
    portfolio_value_float: float,
    current_position_float: float,
) -> float:
    if unit_str == "shares":
        intended_amount_float = float(amount_float)
    elif unit_str == "value":
        intended_amount_float = float(int(amount_float / sizing_reference_price_float))
    elif unit_str == "percent":
        intended_amount_float = float(
            int(portfolio_value_float * amount_float / sizing_reference_price_float)
        )
    else:
        raise ValueError(f"Unsupported unit_str '{unit_str}'.")

    if target_bool:
        return intended_amount_float - float(current_position_float)
    return intended_amount_float


class BrokerAdapter(ABC):
    @abstractmethod
    def get_visible_account_route_set(self) -> set[str] | None:
        raise NotImplementedError

    @abstractmethod
    def get_session_mode_str(self, account_route_str: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def is_session_ready(self, account_route_str: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_account_snapshot(self, account_route_str: str) -> BrokerSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_live_price_snapshot(
        self,
        account_route_str: str,
        asset_str_list: list[str],
    ) -> LivePriceSnapshot:
        raise NotImplementedError

    @abstractmethod
    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        raise NotImplementedError

    @abstractmethod
    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
    ) -> list[BrokerOrderFill]:
        raise NotImplementedError

    @abstractmethod
    def get_recent_order_state_snapshot(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        raise NotImplementedError

    @abstractmethod
    def get_session_open_price_list(
        self,
        account_route_str: str,
        asset_str_list: list[str],
        session_open_timestamp_ts: datetime,
        session_calendar_id_str: str,
    ) -> list[SessionOpenPrice]:
        raise NotImplementedError


class IBKRGatewayBrokerAdapter(BrokerAdapter):
    def __init__(
        self,
        host_str: str = "127.0.0.1",
        port_int: int = 7497,
        client_id_int: int = 31,
        timeout_seconds_float: float = 4.0,
    ):
        self.socket_client_obj = IBKRSocketClient(
            host_str=host_str,
            port_int=port_int,
            client_id_int=client_id_int,
            timeout_seconds_float=timeout_seconds_float,
        )

    def get_visible_account_route_set(self) -> set[str] | None:
        try:
            return self.socket_client_obj.get_visible_account_route_set()
        except Exception:
            return None

    def get_session_mode_str(self, account_route_str: str) -> str | None:
        visible_account_route_set = self.get_visible_account_route_set()
        if visible_account_route_set is not None and account_route_str in visible_account_route_set:
            return infer_ibkr_account_mode_str(account_route_str)
        return None

    def is_session_ready(self, account_route_str: str) -> bool:
        visible_account_route_set = self.get_visible_account_route_set()
        if visible_account_route_set is None:
            return False
        return account_route_str in visible_account_route_set

    def get_account_snapshot(self, account_route_str: str) -> BrokerSnapshot:
        return self.socket_client_obj.get_account_snapshot(account_route_str)

    def get_live_price_snapshot(
        self,
        account_route_str: str,
        asset_str_list: list[str],
    ) -> LivePriceSnapshot:
        return self.socket_client_obj.get_live_price_snapshot(account_route_str, asset_str_list)

    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        return self.socket_client_obj.submit_order_request_list(
            account_route_str=account_route_str,
            broker_order_request_list=broker_order_request_list,
            submitted_timestamp_ts=submitted_timestamp_ts,
        )

    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
    ) -> list[BrokerOrderFill]:
        return self.socket_client_obj.get_recent_fill_list(
            account_route_str=account_route_str,
            since_timestamp_ts=since_timestamp_ts,
        )

    def get_recent_order_state_snapshot(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        return self.socket_client_obj.get_recent_order_state_snapshot(
            account_route_str=account_route_str,
            since_timestamp_ts=since_timestamp_ts,
        )

    def get_session_open_price_list(
        self,
        account_route_str: str,
        asset_str_list: list[str],
        session_open_timestamp_ts: datetime,
        session_calendar_id_str: str,
    ) -> list[SessionOpenPrice]:
        return self.socket_client_obj.get_session_open_price_list(
            account_route_str=account_route_str,
            asset_str_list=asset_str_list,
            session_open_timestamp_ts=session_open_timestamp_ts,
            session_calendar_id_str=session_calendar_id_str,
        )


class StubBrokerAdapter(BrokerAdapter):
    def __init__(self):
        self._snapshot_map: dict[str, BrokerSnapshot] = {}
        self._fill_map: dict[str, list[BrokerOrderFill]] = defaultdict(list)
        self._broker_order_record_map: dict[str, dict[str, BrokerOrderRecord]] = defaultdict(dict)
        self._broker_order_event_map: dict[str, list[BrokerOrderEvent]] = defaultdict(list)
        self._session_open_price_map: dict[tuple[str, str, str], SessionOpenPrice] = {}
        self._session_mode_map: dict[str, str] = {}
        self._fill_price_multiplier_map: dict[str, float] = {}
        self._live_price_snapshot_map: dict[str, LivePriceSnapshot] = {}
        self.submitted_order_request_list: list[BrokerOrderRequest] = []

    def seed_account_snapshot(
        self,
        account_route_str: str,
        cash_float: float,
        total_value_float: float,
        position_amount_map: dict[str, float] | None = None,
        snapshot_timestamp_ts: datetime | None = None,
        session_mode_str: str | None = None,
        net_liq_float: float | None = None,
        available_funds_float: float | None = None,
        excess_liquidity_float: float | None = None,
        cushion_float: float | None = None,
    ) -> None:
        if snapshot_timestamp_ts is None:
            snapshot_timestamp_ts = datetime.now(UTC)
        if session_mode_str is None:
            session_mode_str = infer_ibkr_account_mode_str(account_route_str)
        self._snapshot_map[account_route_str] = BrokerSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=snapshot_timestamp_ts,
            cash_float=float(cash_float),
            total_value_float=float(total_value_float),
            net_liq_float=float(total_value_float if net_liq_float is None else net_liq_float),
            available_funds_float=(
                float(total_value_float if available_funds_float is None else available_funds_float)
            ),
            excess_liquidity_float=(
                float(total_value_float if excess_liquidity_float is None else excess_liquidity_float)
            ),
            cushion_float=float(1.0 if cushion_float is None else cushion_float),
            position_amount_map=dict(position_amount_map or {}),
            open_order_id_list=[],
        )
        self._session_mode_map[account_route_str] = str(session_mode_str)

    def seed_live_price_snapshot(
        self,
        account_route_str: str,
        asset_reference_price_map: dict[str, float],
        snapshot_timestamp_ts: datetime | None = None,
        price_source_str: str = "stub",
    ) -> None:
        if snapshot_timestamp_ts is None:
            snapshot_timestamp_ts = datetime.now(UTC)
        self._live_price_snapshot_map[account_route_str] = LivePriceSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=snapshot_timestamp_ts,
            price_source_str=price_source_str,
            asset_reference_price_map={
                asset_str: float(price_float)
                for asset_str, price_float in asset_reference_price_map.items()
            },
        )

    def set_fill_price_multiplier(
        self,
        asset_str: str,
        fill_price_multiplier_float: float,
    ) -> None:
        self._fill_price_multiplier_map[str(asset_str)] = float(fill_price_multiplier_float)

    def seed_session_open_price(
        self,
        account_route_str: str,
        session_date_str: str,
        asset_str: str,
        official_open_price_float: float | None,
        open_price_source_str: str = "stub.seeded_open",
        snapshot_timestamp_ts: datetime | None = None,
    ) -> None:
        if snapshot_timestamp_ts is None:
            snapshot_timestamp_ts = datetime.now(UTC)
        self._session_open_price_map[(account_route_str, session_date_str, asset_str)] = SessionOpenPrice(
            session_date_str=str(session_date_str),
            account_route_str=str(account_route_str),
            asset_str=str(asset_str),
            official_open_price_float=(
                None if official_open_price_float is None else float(official_open_price_float)
            ),
            open_price_source_str=str(open_price_source_str),
            snapshot_timestamp_ts=snapshot_timestamp_ts,
        )

    def get_session_mode_str(self, account_route_str: str) -> str | None:
        return self._session_mode_map.get(account_route_str)

    def get_visible_account_route_set(self) -> set[str] | None:
        return set(self._snapshot_map.keys())

    def is_session_ready(self, account_route_str: str) -> bool:
        return account_route_str in self._snapshot_map

    def get_account_snapshot(self, account_route_str: str) -> BrokerSnapshot:
        if account_route_str not in self._snapshot_map:
            raise RuntimeError(f"No stub account snapshot seeded for {account_route_str}.")
        return self._snapshot_map[account_route_str]

    def get_live_price_snapshot(
        self,
        account_route_str: str,
        asset_str_list: list[str],
    ) -> LivePriceSnapshot:
        if account_route_str not in self._live_price_snapshot_map:
            raise RuntimeError(f"No stub live price snapshot seeded for {account_route_str}.")
        live_price_snapshot_obj = self._live_price_snapshot_map[account_route_str]
        asset_reference_price_map = {
            asset_str: float(live_price_snapshot_obj.asset_reference_price_map[asset_str])
            for asset_str in asset_str_list
            if asset_str in live_price_snapshot_obj.asset_reference_price_map
        }
        return LivePriceSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=live_price_snapshot_obj.snapshot_timestamp_ts,
            price_source_str=live_price_snapshot_obj.price_source_str,
            asset_reference_price_map=asset_reference_price_map,
        )

    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        snapshot_obj = self.get_account_snapshot(account_route_str)
        updated_position_map = dict(snapshot_obj.position_amount_map)
        cash_float = float(snapshot_obj.cash_float)
        broker_order_record_list: list[BrokerOrderRecord] = []
        broker_order_event_list: list[BrokerOrderEvent] = []
        broker_order_fill_list: list[BrokerOrderFill] = []

        for broker_order_request_obj in broker_order_request_list:
            self.submitted_order_request_list.append(broker_order_request_obj)
            current_position_float = float(updated_position_map.get(broker_order_request_obj.asset_str, 0.0))
            filled_amount_float = _amount_in_shares_float(
                unit_str=broker_order_request_obj.unit_str,
                amount_float=broker_order_request_obj.amount_float,
                target_bool=broker_order_request_obj.target_bool,
                sizing_reference_price_float=broker_order_request_obj.sizing_reference_price_float,
                portfolio_value_float=broker_order_request_obj.portfolio_value_float,
                current_position_float=current_position_float,
            )
            updated_position_map[broker_order_request_obj.asset_str] = (
                current_position_float + filled_amount_float
            )
            fill_price_multiplier_float = self._fill_price_multiplier_map.get(
                broker_order_request_obj.asset_str,
                1.0,
            )
            fill_price_float = (
                broker_order_request_obj.sizing_reference_price_float * fill_price_multiplier_float
            )
            cash_float -= filled_amount_float * fill_price_float

            broker_order_id_str = uuid.uuid4().hex
            requested_quantity_float = abs(float(broker_order_request_obj.amount_float))
            broker_order_record_list.append(
                BrokerOrderRecord(
                    broker_order_id_str=broker_order_id_str,
                    decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                    vplan_id_int=broker_order_request_obj.vplan_id_int,
                    account_route_str=account_route_str,
                    asset_str=broker_order_request_obj.asset_str,
                    broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                    unit_str=broker_order_request_obj.unit_str,
                    amount_float=broker_order_request_obj.amount_float,
                    filled_amount_float=0.0,
                    remaining_amount_float=requested_quantity_float,
                    avg_fill_price_float=None,
                    status_str="PendingSubmit",
                    last_status_timestamp_ts=submitted_timestamp_ts,
                    submitted_timestamp_ts=submitted_timestamp_ts,
                    raw_payload_dict={
                        "trade_id_int": broker_order_request_obj.trade_id_int,
                        "target_bool": broker_order_request_obj.target_bool,
                        "submission_key_str": broker_order_request_obj.submission_key_str,
                    },
                )
            )
            pending_event_obj = BrokerOrderEvent(
                broker_order_id_str=broker_order_id_str,
                decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                vplan_id_int=broker_order_request_obj.vplan_id_int,
                account_route_str=account_route_str,
                asset_str=broker_order_request_obj.asset_str,
                status_str="PendingSubmit",
                filled_amount_float=0.0,
                remaining_amount_float=requested_quantity_float,
                avg_fill_price_float=None,
                event_timestamp_ts=submitted_timestamp_ts,
                event_source_str="stub.submit",
                message_str="submitted",
                raw_payload_dict={
                    "submission_key_str": broker_order_request_obj.submission_key_str,
                },
            )
            filled_event_obj = BrokerOrderEvent(
                broker_order_id_str=broker_order_id_str,
                decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                vplan_id_int=broker_order_request_obj.vplan_id_int,
                account_route_str=account_route_str,
                asset_str=broker_order_request_obj.asset_str,
                status_str="Filled",
                filled_amount_float=filled_amount_float,
                remaining_amount_float=0.0,
                avg_fill_price_float=fill_price_float,
                event_timestamp_ts=submitted_timestamp_ts,
                event_source_str="stub.fill",
                message_str="fill complete",
                raw_payload_dict={
                    "trade_id_int": broker_order_request_obj.trade_id_int,
                },
            )
            broker_order_event_list.append(pending_event_obj)
            broker_order_fill_obj = BrokerOrderFill(
                broker_order_id_str=broker_order_id_str,
                decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                vplan_id_int=broker_order_request_obj.vplan_id_int,
                account_route_str=account_route_str,
                asset_str=broker_order_request_obj.asset_str,
                fill_amount_float=filled_amount_float,
                fill_price_float=fill_price_float,
                fill_timestamp_ts=submitted_timestamp_ts,
                raw_payload_dict={
                    "unit_str": broker_order_request_obj.unit_str,
                },
            )
            broker_order_fill_list.append(broker_order_fill_obj)
            self._fill_map[account_route_str].append(broker_order_fill_obj)
            self._broker_order_event_map[account_route_str].extend(
                [pending_event_obj, filled_event_obj]
            )
            self._broker_order_record_map[account_route_str][broker_order_id_str] = BrokerOrderRecord(
                broker_order_id_str=broker_order_id_str,
                decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                vplan_id_int=broker_order_request_obj.vplan_id_int,
                account_route_str=account_route_str,
                asset_str=broker_order_request_obj.asset_str,
                broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                unit_str=broker_order_request_obj.unit_str,
                amount_float=broker_order_request_obj.amount_float,
                filled_amount_float=filled_amount_float,
                remaining_amount_float=0.0,
                avg_fill_price_float=fill_price_float,
                status_str="Filled",
                last_status_timestamp_ts=submitted_timestamp_ts,
                submitted_timestamp_ts=submitted_timestamp_ts,
                raw_payload_dict={
                    "trade_id_int": broker_order_request_obj.trade_id_int,
                    "target_bool": broker_order_request_obj.target_bool,
                    "submission_key_str": broker_order_request_obj.submission_key_str,
                },
            )

        asset_reference_price_map: dict[str, float] = {}
        if account_route_str in self._live_price_snapshot_map:
            asset_reference_price_map.update(
                self._live_price_snapshot_map[account_route_str].asset_reference_price_map
            )
        for broker_order_request_obj in broker_order_request_list:
            asset_reference_price_map[broker_order_request_obj.asset_str] = (
                broker_order_request_obj.sizing_reference_price_float
                * self._fill_price_multiplier_map.get(broker_order_request_obj.asset_str, 1.0)
            )
        position_value_float = 0.0
        for asset_str, position_amount_float in updated_position_map.items():
            position_value_float += float(position_amount_float) * float(asset_reference_price_map.get(asset_str, 0.0))
        total_value_float = cash_float + position_value_float
        self._snapshot_map[account_route_str] = BrokerSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=submitted_timestamp_ts,
            cash_float=cash_float,
            total_value_float=total_value_float,
            net_liq_float=total_value_float,
            available_funds_float=total_value_float,
            excess_liquidity_float=total_value_float,
            cushion_float=1.0,
            position_amount_map=updated_position_map,
            open_order_id_list=[],
        )

        return broker_order_record_list, broker_order_event_list, []

    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
    ) -> list[BrokerOrderFill]:
        fill_list = self._fill_map.get(account_route_str, [])
        return [
            fill_obj
            for fill_obj in fill_list
            if fill_obj.fill_timestamp_ts >= since_timestamp_ts
        ]

    def get_recent_order_state_snapshot(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        broker_order_record_list = list(self._broker_order_record_map.get(account_route_str, {}).values())
        broker_order_event_list = [
            broker_order_event_obj
            for broker_order_event_obj in self._broker_order_event_map.get(account_route_str, [])
            if (
                broker_order_event_obj.event_timestamp_ts is not None
                and broker_order_event_obj.event_timestamp_ts >= since_timestamp_ts
            )
        ]
        broker_order_fill_list = self.get_recent_fill_list(
            account_route_str=account_route_str,
            since_timestamp_ts=since_timestamp_ts,
        )
        return broker_order_record_list, broker_order_event_list, broker_order_fill_list

    def get_session_open_price_list(
        self,
        account_route_str: str,
        asset_str_list: list[str],
        session_open_timestamp_ts: datetime,
        session_calendar_id_str: str,
    ) -> list[SessionOpenPrice]:
        market_open_timestamp_ts = scheduler_utils.to_market_timestamp_ts(
            session_open_timestamp_ts,
            session_calendar_id_str,
        )
        session_date_str = market_open_timestamp_ts.date().isoformat()
        session_open_price_list: list[SessionOpenPrice] = []
        live_price_snapshot_obj = self._live_price_snapshot_map.get(account_route_str)
        for asset_str in asset_str_list:
            seeded_session_open_price_obj = self._session_open_price_map.get(
                (account_route_str, session_date_str, asset_str)
            )
            if seeded_session_open_price_obj is not None:
                session_open_price_list.append(seeded_session_open_price_obj)
                continue
            official_open_price_float = None
            open_price_source_str = None
            if (
                live_price_snapshot_obj is not None
                and asset_str in live_price_snapshot_obj.asset_reference_price_map
            ):
                official_open_price_float = float(live_price_snapshot_obj.asset_reference_price_map[asset_str])
                open_price_source_str = "stub.live_price"
            session_open_price_list.append(
                SessionOpenPrice(
                    session_date_str=session_date_str,
                    account_route_str=account_route_str,
                    asset_str=asset_str,
                    official_open_price_float=official_open_price_float,
                    open_price_source_str=open_price_source_str,
                    snapshot_timestamp_ts=market_open_timestamp_ts,
                    raw_payload_dict={},
                )
            )
        return session_open_price_list
