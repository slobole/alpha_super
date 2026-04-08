from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import UTC, datetime
from typing import Iterable
import uuid

from alpha.live.models import (
    BrokerOrderFill,
    BrokerOrderRecord,
    BrokerOrderRequest,
    BrokerPositionSnapshot,
    FrozenOrderIntent,
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
    intent_obj: FrozenOrderIntent,
    current_position_float: float,
) -> float:
    if intent_obj.unit_str == "shares":
        intended_amount_float = float(intent_obj.amount_float)
    elif intent_obj.unit_str == "value":
        intended_amount_float = float(int(intent_obj.amount_float / intent_obj.sizing_reference_price_float))
    elif intent_obj.unit_str == "percent":
        intended_amount_float = float(
            int(intent_obj.portfolio_value_float * intent_obj.amount_float / intent_obj.sizing_reference_price_float)
        )
    else:
        raise ValueError(f"Unsupported unit_str '{intent_obj.unit_str}'.")

    if intent_obj.target_bool:
        return intended_amount_float - float(current_position_float)
    return intended_amount_float


def validate_order_intent_list(
    order_intent_list: Iterable[FrozenOrderIntent],
    allowed_symbol_set: set[str] | None = None,
) -> None:
    for order_intent_obj in order_intent_list:
        if len(order_intent_obj.asset_str.strip()) == 0:
            raise ValueError("Order intent asset_str must not be empty.")
        if allowed_symbol_set is not None and order_intent_obj.asset_str not in allowed_symbol_set:
            raise ValueError(
                f"Order intent asset_str '{order_intent_obj.asset_str}' is outside the allowed symbol set."
            )
        if order_intent_obj.sizing_reference_price_float <= 0.0:
            raise ValueError(
                f"Order intent sizing_reference_price_float must be positive for {order_intent_obj.asset_str}."
            )


def build_broker_order_request_list(
    order_plan_id_int: int,
    release_id_str: str,
    pod_id_str: str,
    account_route_str: str,
    submission_key_str: str,
    order_intent_list: list[FrozenOrderIntent],
    order_intent_id_list: list[int],
) -> list[BrokerOrderRequest]:
    broker_order_request_list: list[BrokerOrderRequest] = []

    for order_intent_id_int, order_intent_obj in zip(order_intent_id_list, order_intent_list, strict=True):
        broker_order_request_list.append(
            BrokerOrderRequest(
                order_plan_id_int=order_plan_id_int,
                order_intent_id_int=int(order_intent_id_int),
                release_id_str=release_id_str,
                pod_id_str=pod_id_str,
                account_route_str=account_route_str,
                submission_key_str=submission_key_str,
                asset_str=order_intent_obj.asset_str,
                broker_order_type_str=order_intent_obj.broker_order_type_str,
                order_class_str=order_intent_obj.order_class_str,
                unit_str=order_intent_obj.unit_str,
                amount_float=float(order_intent_obj.amount_float),
                target_bool=bool(order_intent_obj.target_bool),
                trade_id_int=order_intent_obj.trade_id_int,
                sizing_reference_price_float=float(order_intent_obj.sizing_reference_price_float),
                portfolio_value_float=float(order_intent_obj.portfolio_value_float),
            )
        )

    return broker_order_request_list


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
    def get_account_snapshot(self, account_route_str: str) -> BrokerPositionSnapshot:
        raise NotImplementedError

    @abstractmethod
    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderFill]]:
        raise NotImplementedError

    @abstractmethod
    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
    ) -> list[BrokerOrderFill]:
        raise NotImplementedError


class IBKRGatewayBrokerAdapter(BrokerAdapter):
    def get_visible_account_route_set(self) -> set[str] | None:
        return None

    def get_session_mode_str(self, account_route_str: str) -> str | None:
        return None

    def is_session_ready(self, account_route_str: str) -> bool:
        return False

    def get_account_snapshot(self, account_route_str: str) -> BrokerPositionSnapshot:
        raise RuntimeError(
            "IBKR broker adapter is a thin boundary only in v1. "
            "Ensure IBC + IB Gateway/TWS are running and wire a concrete adapter before live submission."
        )

    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderFill]]:
        raise RuntimeError(
            "IBKR broker adapter submission is not implemented in v1. "
            "Use the stub adapter in tests or provide a concrete broker integration."
        )

    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
    ) -> list[BrokerOrderFill]:
        raise RuntimeError(
            "IBKR broker adapter recent-fill lookup is not implemented in v1."
        )


class StubBrokerAdapter(BrokerAdapter):
    def __init__(self):
        self._snapshot_map: dict[str, BrokerPositionSnapshot] = {}
        self._fill_map: dict[str, list[BrokerOrderFill]] = defaultdict(list)
        self._session_mode_map: dict[str, str] = {}
        self._fill_price_multiplier_map: dict[str, float] = {}
        self.submitted_order_request_list: list[BrokerOrderRequest] = []

    def seed_account_snapshot(
        self,
        account_route_str: str,
        cash_float: float,
        total_value_float: float,
        position_amount_map: dict[str, float] | None = None,
        snapshot_timestamp_ts: datetime | None = None,
        session_mode_str: str | None = None,
    ) -> None:
        if snapshot_timestamp_ts is None:
            snapshot_timestamp_ts = datetime.now(UTC)
        if session_mode_str is None:
            session_mode_str = infer_ibkr_account_mode_str(account_route_str)
        self._snapshot_map[account_route_str] = BrokerPositionSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=snapshot_timestamp_ts,
            cash_float=float(cash_float),
            total_value_float=float(total_value_float),
            position_amount_map=dict(position_amount_map or {}),
            open_order_id_list=[],
        )
        self._session_mode_map[account_route_str] = str(session_mode_str)

    def set_fill_price_multiplier(
        self,
        asset_str: str,
        fill_price_multiplier_float: float,
    ) -> None:
        self._fill_price_multiplier_map[str(asset_str)] = float(fill_price_multiplier_float)

    def get_session_mode_str(self, account_route_str: str) -> str | None:
        return self._session_mode_map.get(account_route_str)

    def get_visible_account_route_set(self) -> set[str] | None:
        return set(self._snapshot_map.keys())

    def is_session_ready(self, account_route_str: str) -> bool:
        return account_route_str in self._snapshot_map

    def get_account_snapshot(self, account_route_str: str) -> BrokerPositionSnapshot:
        if account_route_str not in self._snapshot_map:
            raise RuntimeError(f"No stub account snapshot seeded for {account_route_str}.")
        return self._snapshot_map[account_route_str]

    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderFill]]:
        snapshot_obj = self.get_account_snapshot(account_route_str)
        updated_position_map = dict(snapshot_obj.position_amount_map)
        cash_float = float(snapshot_obj.cash_float)
        broker_order_record_list: list[BrokerOrderRecord] = []
        broker_order_fill_list: list[BrokerOrderFill] = []

        for broker_order_request_obj in broker_order_request_list:
            self.submitted_order_request_list.append(broker_order_request_obj)
            current_position_float = float(updated_position_map.get(broker_order_request_obj.asset_str, 0.0))
            intent_obj = FrozenOrderIntent(
                asset_str=broker_order_request_obj.asset_str,
                order_class_str=broker_order_request_obj.order_class_str,
                unit_str=broker_order_request_obj.unit_str,
                amount_float=broker_order_request_obj.amount_float,
                target_bool=broker_order_request_obj.target_bool,
                trade_id_int=broker_order_request_obj.trade_id_int,
                broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                sizing_reference_price_float=broker_order_request_obj.sizing_reference_price_float,
                portfolio_value_float=broker_order_request_obj.portfolio_value_float,
            )
            filled_amount_float = _amount_in_shares_float(
                intent_obj=intent_obj,
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
            broker_order_record_list.append(
                BrokerOrderRecord(
                    broker_order_id_str=broker_order_id_str,
                    order_plan_id_int=broker_order_request_obj.order_plan_id_int,
                    order_intent_id_int=broker_order_request_obj.order_intent_id_int,
                    account_route_str=account_route_str,
                    asset_str=broker_order_request_obj.asset_str,
                    broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                    unit_str=broker_order_request_obj.unit_str,
                    amount_float=broker_order_request_obj.amount_float,
                    filled_amount_float=filled_amount_float,
                    status_str="filled",
                    submitted_timestamp_ts=submitted_timestamp_ts,
                    raw_payload_dict={
                        "trade_id_int": broker_order_request_obj.trade_id_int,
                        "target_bool": broker_order_request_obj.target_bool,
                        "submission_key_str": broker_order_request_obj.submission_key_str,
                    },
                )
            )
            broker_order_fill_obj = BrokerOrderFill(
                broker_order_id_str=broker_order_id_str,
                order_plan_id_int=broker_order_request_obj.order_plan_id_int,
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

        position_value_float = 0.0
        for broker_order_request_obj in broker_order_request_list:
            position_value_float += (
                updated_position_map.get(broker_order_request_obj.asset_str, 0.0)
                * (
                    broker_order_request_obj.sizing_reference_price_float
                    * self._fill_price_multiplier_map.get(broker_order_request_obj.asset_str, 1.0)
                )
            )
        total_value_float = cash_float + position_value_float
        self._snapshot_map[account_route_str] = BrokerPositionSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=submitted_timestamp_ts,
            cash_float=cash_float,
            total_value_float=total_value_float,
            position_amount_map=updated_position_map,
            open_order_id_list=[],
        )

        return broker_order_record_list, broker_order_fill_list

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
