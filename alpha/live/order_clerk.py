from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import UTC, datetime
from typing import Iterable
import uuid

from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live import scheduler_utils
from alpha.live.models import (
    BrokerOrderAck,
    BrokerOrderFill,
    BrokerOrderEvent,
    BrokerOrderRecord,
    BrokerOrderRequest,
    BrokerSnapshot,
    LivePriceSnapshot,
    SessionOpenPrice,
    SubmitBatchResult,
)


PAPER_ACCOUNT_PREFIX_TUPLE: tuple[str, ...] = ("DU",)
INCUBATION_ACCOUNT_PREFIX_TUPLE: tuple[str, ...] = ("SIM_",)


def account_route_looks_like_incubation_bool(account_route_str: str) -> bool:
    normalized_account_route_str = str(account_route_str).strip().upper()
    return normalized_account_route_str.startswith(INCUBATION_ACCOUNT_PREFIX_TUPLE)


def account_route_looks_like_paper_bool(account_route_str: str) -> bool:
    normalized_account_route_str = str(account_route_str).strip().upper()
    return normalized_account_route_str.startswith(PAPER_ACCOUNT_PREFIX_TUPLE)


def infer_ibkr_account_mode_str(account_route_str: str) -> str:
    if account_route_looks_like_incubation_bool(account_route_str):
        return "incubation"
    if account_route_looks_like_paper_bool(account_route_str):
        return "paper"
    return "live"


def validate_account_route_matches_mode(
    mode_str: str,
    account_route_str: str,
) -> None:
    inferred_mode_str = infer_ibkr_account_mode_str(account_route_str)
    if mode_str == "incubation" and inferred_mode_str != "incubation":
        raise ValueError(
            f"Manifest mode_str '{mode_str}' expects a virtual incubation account_route_str, "
            f"but received '{account_route_str}'. Incubation accounts must use a "
            f"{INCUBATION_ACCOUNT_PREFIX_TUPLE} prefix."
        )
    if mode_str in ("paper", "live") and inferred_mode_str == "incubation":
        raise ValueError(
            f"Manifest mode_str '{mode_str}' expects an IBKR account_route_str, "
            f"but received virtual incubation route '{account_route_str}'."
        )
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


def _compute_submit_ack_summary(
    broker_order_ack_list: list[BrokerOrderAck],
) -> tuple[float, list[str], str]:
    request_count_int = len(broker_order_ack_list)
    if request_count_int == 0:
        return 1.0, [], "complete"

    broker_ack_count_int = sum(
        1
        for broker_order_ack_obj in broker_order_ack_list
        if broker_order_ack_obj.broker_response_ack_bool
    )
    ack_coverage_ratio_float = float(broker_ack_count_int) / float(request_count_int)
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
    return ack_coverage_ratio_float, missing_ack_asset_list, submit_ack_status_str


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
        execution_policy_str: str | None = None,
    ) -> LivePriceSnapshot:
        raise NotImplementedError

    @abstractmethod
    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> SubmitBatchResult:
        raise NotImplementedError

    @abstractmethod
    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> list[BrokerOrderFill]:
        raise NotImplementedError

    @abstractmethod
    def get_recent_order_state_snapshot(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        submission_key_str: str | None = None,
        allowed_broker_order_id_set: set[str] | None = None,
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
        execution_policy_str: str | None = None,
    ) -> LivePriceSnapshot:
        return self.socket_client_obj.get_live_price_snapshot(
            account_route_str,
            asset_str_list,
            execution_policy_str=execution_policy_str,
        )

    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> SubmitBatchResult:
        return self.socket_client_obj.submit_order_request_list(
            account_route_str=account_route_str,
            broker_order_request_list=broker_order_request_list,
            submitted_timestamp_ts=submitted_timestamp_ts,
        )

    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> list[BrokerOrderFill]:
        return self.socket_client_obj.get_recent_fill_list(
            account_route_str=account_route_str,
            since_timestamp_ts=since_timestamp_ts,
            allowed_broker_order_id_set=allowed_broker_order_id_set,
        )

    def get_recent_order_state_snapshot(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        submission_key_str: str | None = None,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        return self.socket_client_obj.get_recent_order_state_snapshot(
            account_route_str=account_route_str,
            since_timestamp_ts=since_timestamp_ts,
            submission_key_str=submission_key_str,
            allowed_broker_order_id_set=allowed_broker_order_id_set,
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
        asset_reference_source_map_dict: dict[str, str] | None = None,
    ) -> None:
        if snapshot_timestamp_ts is None:
            snapshot_timestamp_ts = datetime.now(UTC)
        if asset_reference_source_map_dict is None:
            asset_reference_source_map_dict = {
                str(asset_str): str(price_source_str)
                for asset_str in asset_reference_price_map
            }
        self._live_price_snapshot_map[account_route_str] = LivePriceSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=snapshot_timestamp_ts,
            price_source_str=price_source_str,
            asset_reference_price_map={
                asset_str: float(price_float)
                for asset_str, price_float in asset_reference_price_map.items()
            },
            asset_reference_source_map_dict={
                str(asset_str): str(source_str)
                for asset_str, source_str in asset_reference_source_map_dict.items()
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

    def seed_broker_order_state(
        self,
        broker_order_record_obj: BrokerOrderRecord,
        broker_order_event_list: list[BrokerOrderEvent] | None = None,
        broker_order_fill_list: list[BrokerOrderFill] | None = None,
    ) -> None:
        account_route_str = str(broker_order_record_obj.account_route_str)
        self._broker_order_record_map[account_route_str][
            str(broker_order_record_obj.broker_order_id_str)
        ] = broker_order_record_obj
        if broker_order_event_list is not None:
            self._broker_order_event_map[account_route_str].extend(broker_order_event_list)
        if broker_order_fill_list is not None:
            self._fill_map[account_route_str].extend(broker_order_fill_list)

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
        execution_policy_str: str | None = None,
    ) -> LivePriceSnapshot:
        del execution_policy_str
        if account_route_str not in self._live_price_snapshot_map:
            raise RuntimeError(f"No stub live price snapshot seeded for {account_route_str}.")
        live_price_snapshot_obj = self._live_price_snapshot_map[account_route_str]
        asset_reference_price_map = {
            asset_str: float(live_price_snapshot_obj.asset_reference_price_map[asset_str])
            for asset_str in asset_str_list
            if asset_str in live_price_snapshot_obj.asset_reference_price_map
        }
        asset_reference_source_map_dict = {
            asset_str: str(
                live_price_snapshot_obj.asset_reference_source_map_dict.get(
                    asset_str,
                    live_price_snapshot_obj.price_source_str,
                )
            )
            for asset_str in asset_reference_price_map
        }
        return LivePriceSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=live_price_snapshot_obj.snapshot_timestamp_ts,
            price_source_str=live_price_snapshot_obj.price_source_str,
            asset_reference_price_map=asset_reference_price_map,
            asset_reference_source_map_dict=asset_reference_source_map_dict,
        )

    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> SubmitBatchResult:
        snapshot_obj = self.get_account_snapshot(account_route_str)
        updated_position_map = dict(snapshot_obj.position_amount_map)
        cash_float = float(snapshot_obj.cash_float)
        broker_order_record_list: list[BrokerOrderRecord] = []
        broker_order_event_list: list[BrokerOrderEvent] = []
        broker_order_fill_list: list[BrokerOrderFill] = []
        broker_order_ack_list: list[BrokerOrderAck] = []

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
                    order_request_key_str=broker_order_request_obj.order_request_key_str,
                    broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                    unit_str=broker_order_request_obj.unit_str,
                    amount_float=broker_order_request_obj.amount_float,
                    filled_amount_float=0.0,
                    remaining_amount_float=requested_quantity_float,
                    avg_fill_price_float=None,
                    status_str="PendingSubmit",
                    last_status_timestamp_ts=submitted_timestamp_ts,
                    submitted_timestamp_ts=submitted_timestamp_ts,
                    submission_key_str=broker_order_request_obj.submission_key_str,
                    raw_payload_dict={
                        "trade_id_int": broker_order_request_obj.trade_id_int,
                        "target_bool": broker_order_request_obj.target_bool,
                        "submission_key_str": broker_order_request_obj.submission_key_str,
                        "order_request_key_str": broker_order_request_obj.order_request_key_str,
                    },
                )
            )
            pending_event_obj = BrokerOrderEvent(
                broker_order_id_str=broker_order_id_str,
                decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                vplan_id_int=broker_order_request_obj.vplan_id_int,
                account_route_str=account_route_str,
                asset_str=broker_order_request_obj.asset_str,
                order_request_key_str=broker_order_request_obj.order_request_key_str,
                status_str="PendingSubmit",
                filled_amount_float=0.0,
                remaining_amount_float=requested_quantity_float,
                avg_fill_price_float=None,
                event_timestamp_ts=submitted_timestamp_ts,
                event_source_str="stub.submit",
                message_str="submitted",
                submission_key_str=broker_order_request_obj.submission_key_str,
                raw_payload_dict={
                    "submission_key_str": broker_order_request_obj.submission_key_str,
                    "order_request_key_str": broker_order_request_obj.order_request_key_str,
                },
            )
            filled_event_obj = BrokerOrderEvent(
                broker_order_id_str=broker_order_id_str,
                decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                vplan_id_int=broker_order_request_obj.vplan_id_int,
                account_route_str=account_route_str,
                asset_str=broker_order_request_obj.asset_str,
                order_request_key_str=broker_order_request_obj.order_request_key_str,
                status_str="Filled",
                filled_amount_float=filled_amount_float,
                remaining_amount_float=0.0,
                avg_fill_price_float=fill_price_float,
                event_timestamp_ts=submitted_timestamp_ts,
                event_source_str="stub.fill",
                message_str="fill complete",
                submission_key_str=broker_order_request_obj.submission_key_str,
                raw_payload_dict={
                    "trade_id_int": broker_order_request_obj.trade_id_int,
                    "order_request_key_str": broker_order_request_obj.order_request_key_str,
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
                order_request_key_str=broker_order_request_obj.order_request_key_str,
                broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                unit_str=broker_order_request_obj.unit_str,
                amount_float=broker_order_request_obj.amount_float,
                filled_amount_float=filled_amount_float,
                remaining_amount_float=0.0,
                avg_fill_price_float=fill_price_float,
                status_str="Filled",
                last_status_timestamp_ts=submitted_timestamp_ts,
                submitted_timestamp_ts=submitted_timestamp_ts,
                submission_key_str=broker_order_request_obj.submission_key_str,
                raw_payload_dict={
                    "trade_id_int": broker_order_request_obj.trade_id_int,
                    "target_bool": broker_order_request_obj.target_bool,
                    "submission_key_str": broker_order_request_obj.submission_key_str,
                    "order_request_key_str": broker_order_request_obj.order_request_key_str,
                },
            )
            broker_order_ack_list.append(
                BrokerOrderAck(
                    decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                    vplan_id_int=broker_order_request_obj.vplan_id_int,
                    account_route_str=account_route_str,
                    order_request_key_str=broker_order_request_obj.order_request_key_str,
                    asset_str=broker_order_request_obj.asset_str,
                    broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                    local_submit_ack_bool=True,
                    broker_response_ack_bool=True,
                    ack_status_str="broker_acked",
                    ack_source_str="stub.state_snapshot",
                    broker_order_id_str=broker_order_id_str,
                    response_timestamp_ts=submitted_timestamp_ts,
                    raw_payload_dict={
                        "submission_key_str": broker_order_request_obj.submission_key_str,
                        "order_request_key_str": broker_order_request_obj.order_request_key_str,
                    },
                )
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

        ack_coverage_ratio_float, missing_ack_asset_list, submit_ack_status_str = (
            _compute_submit_ack_summary(broker_order_ack_list)
        )
        return SubmitBatchResult(
            broker_order_record_list=broker_order_record_list,
            broker_order_event_list=broker_order_event_list,
            broker_order_fill_list=[],
            broker_order_ack_list=broker_order_ack_list,
            ack_coverage_ratio_float=ack_coverage_ratio_float,
            missing_ack_asset_list=missing_ack_asset_list,
            submit_ack_status_str=submit_ack_status_str,
        )

    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> list[BrokerOrderFill]:
        fill_list = self._fill_map.get(account_route_str, [])
        filtered_fill_list = [
            fill_obj
            for fill_obj in fill_list
            if fill_obj.fill_timestamp_ts >= since_timestamp_ts
        ]
        if allowed_broker_order_id_set is None:
            return filtered_fill_list
        normalized_allowed_broker_order_id_set = {
            str(broker_order_id_str)
            for broker_order_id_str in allowed_broker_order_id_set
        }
        return [
            fill_obj
            for fill_obj in filtered_fill_list
            if str(fill_obj.broker_order_id_str) in normalized_allowed_broker_order_id_set
        ]

    def get_recent_order_state_snapshot(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        submission_key_str: str | None = None,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        broker_order_record_list = list(self._broker_order_record_map.get(account_route_str, {}).values())
        normalized_allowed_broker_order_id_set = (
            None
            if allowed_broker_order_id_set is None
            else {
                str(broker_order_id_str)
                for broker_order_id_str in allowed_broker_order_id_set
            }
        )
        if submission_key_str is not None or normalized_allowed_broker_order_id_set is not None:
            filtered_broker_order_record_list: list[BrokerOrderRecord] = []
            for broker_order_record_obj in broker_order_record_list:
                submission_key_matches_bool = (
                    submission_key_str is not None
                    and str(broker_order_record_obj.submission_key_str or "") == str(submission_key_str)
                )
                broker_order_id_matches_bool = (
                    normalized_allowed_broker_order_id_set is not None
                    and str(broker_order_record_obj.broker_order_id_str) in normalized_allowed_broker_order_id_set
                )
                if submission_key_matches_bool or broker_order_id_matches_bool:
                    filtered_broker_order_record_list.append(broker_order_record_obj)
            broker_order_record_list = filtered_broker_order_record_list
        broker_order_event_list = [
            broker_order_event_obj
            for broker_order_event_obj in self._broker_order_event_map.get(account_route_str, [])
            if (
                broker_order_event_obj.event_timestamp_ts is not None
                and broker_order_event_obj.event_timestamp_ts >= since_timestamp_ts
            )
        ]
        if submission_key_str is not None or normalized_allowed_broker_order_id_set is not None:
            filtered_broker_order_event_list: list[BrokerOrderEvent] = []
            for broker_order_event_obj in broker_order_event_list:
                submission_key_matches_bool = (
                    submission_key_str is not None
                    and str(broker_order_event_obj.submission_key_str or "") == str(submission_key_str)
                )
                broker_order_id_matches_bool = (
                    normalized_allowed_broker_order_id_set is not None
                    and str(broker_order_event_obj.broker_order_id_str) in normalized_allowed_broker_order_id_set
                )
                if submission_key_matches_bool or broker_order_id_matches_bool:
                    filtered_broker_order_event_list.append(broker_order_event_obj)
            broker_order_event_list = filtered_broker_order_event_list
        effective_broker_order_id_set = {
            str(broker_order_record_obj.broker_order_id_str)
            for broker_order_record_obj in broker_order_record_list
        }
        effective_broker_order_id_set.update(
            str(broker_order_event_obj.broker_order_id_str)
            for broker_order_event_obj in broker_order_event_list
        )
        broker_order_fill_list = self.get_recent_fill_list(
            account_route_str=account_route_str,
            since_timestamp_ts=since_timestamp_ts,
            allowed_broker_order_id_set=(
                effective_broker_order_id_set
                if len(effective_broker_order_id_set) > 0
                else normalized_allowed_broker_order_id_set
            ),
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
