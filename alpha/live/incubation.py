from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import math
from typing import Any

import pandas as pd

from alpha.live import scheduler_utils
from alpha.live.ibkr_socket_client import IBKRSocketClient, IBKR_TICK_OPEN_SOURCE_STR
from alpha.live.models import (
    BrokerOrderAck,
    BrokerOrderEvent,
    BrokerOrderFill,
    BrokerOrderRecord,
    BrokerOrderRequest,
    BrokerSnapshot,
    CashLedgerEntry,
    LivePriceSnapshot,
    LiveRelease,
    PodState,
    SessionOpenPrice,
    SubmitBatchResult,
    VPlan,
)
from alpha.live.order_clerk import BrokerAdapter
from alpha.live.state_store_v2 import LiveStateStore


OfficialPriceLookup_Func = Callable[[list[str], str, str], dict[str, float]]
PreClosePriceLookup_Func = Callable[[list[str], str], dict[str, float]]
IBKRTickOpenLookup_Func = Callable[[str, list[str], datetime, str], list[SessionOpenPrice]]


OPEN_EXECUTION_POLICY_SET: set[str] = {
    "next_open_moo",
    "next_open_market",
    "next_month_first_open",
}


def load_official_price_map_from_norgate(
    asset_str_list: list[str],
    session_date_str: str,
    price_field_str: str,
) -> dict[str, float]:
    try:
        import norgatedata
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError(
            "Incubation official-price settlement requires the norgatedata package."
        ) from exc

    official_price_map_dict: dict[str, float] = {}
    for asset_str in sorted({str(asset_str) for asset_str in asset_str_list}):
        price_df = norgatedata.price_timeseries(
            asset_str,
            stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
            padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
            start_date=session_date_str,
            end_date=session_date_str,
            timeseriesformat="pandas-dataframe",
        )
        if price_df is None or len(price_df.index) == 0 or price_field_str not in price_df.columns:
            raise RuntimeError(
                "Missing Norgate official price for incubation settlement: "
                f"asset={asset_str} date={session_date_str} field={price_field_str}."
            )
        price_float = float(price_df.iloc[-1][price_field_str])
        if not math.isfinite(price_float) or price_float <= 0.0:
            raise RuntimeError(
                "Invalid Norgate official price for incubation settlement: "
                f"asset={asset_str} date={session_date_str} field={price_field_str} "
                f"price={price_float}."
            )
        official_price_map_dict[asset_str] = price_float
    return official_price_map_dict


def incubation_broker_order_id_str(order_request_key_str: str) -> str:
    return f"incubation:{order_request_key_str}"


class IncubationBrokerAdapter(BrokerAdapter):
    def __init__(
        self,
        *,
        state_store_obj: LiveStateStore,
        as_of_ts: datetime,
        official_price_lookup_func: OfficialPriceLookup_Func | None = None,
        preclose_price_lookup_func: PreClosePriceLookup_Func | None = None,
        ibkr_tick_open_lookup_func: IBKRTickOpenLookup_Func | None = None,
        ibkr_host_str: str = "127.0.0.1",
        ibkr_port_int: int = 7497,
        ibkr_client_id_int: int = 91,
        ibkr_timeout_seconds_float: float = 4.0,
    ) -> None:
        self.state_store_obj = state_store_obj
        self.as_of_ts = as_of_ts
        self.official_price_lookup_func = (
            official_price_lookup_func
            if official_price_lookup_func is not None
            else load_official_price_map_from_norgate
        )
        self.preclose_price_lookup_func = preclose_price_lookup_func
        self.ibkr_tick_open_lookup_func = ibkr_tick_open_lookup_func
        self.ibkr_socket_client_obj = IBKRSocketClient(
            host_str=ibkr_host_str,
            port_int=ibkr_port_int,
            client_id_int=ibkr_client_id_int,
            timeout_seconds_float=ibkr_timeout_seconds_float,
        )
        self._settled_state_by_vplan_id_dict: dict[
            int,
            tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]],
        ] = {}

    def get_visible_account_route_set(self) -> set[str] | None:
        return {
            str(release_obj.account_route_str)
            for release_obj in self.state_store_obj.get_enabled_release_list()
            if release_obj.mode_str == "incubation"
        }

    def get_session_mode_str(self, account_route_str: str) -> str | None:
        if str(account_route_str).upper().startswith("SIM_"):
            return "incubation"
        return None

    def is_session_ready(self, account_route_str: str) -> bool:
        return self._get_release_for_account_route(account_route_str) is not None

    def get_account_snapshot(self, account_route_str: str) -> BrokerSnapshot:
        self._settle_due_vplan_list_for_account(account_route_str)
        release_obj = self._require_release_for_account_route(account_route_str)
        pod_state_obj = self._get_pod_state_or_default(release_obj)
        mark_price_map_dict = self._load_mark_price_map_dict(
            release_obj=release_obj,
            asset_str_list=sorted(pod_state_obj.position_amount_map),
        )
        position_value_float = sum(
            float(share_float) * float(mark_price_map_dict.get(asset_str, 0.0))
            for asset_str, share_float in pod_state_obj.position_amount_map.items()
        )
        total_value_float = float(pod_state_obj.cash_float) + float(position_value_float)
        return BrokerSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=self.as_of_ts,
            cash_float=float(pod_state_obj.cash_float),
            total_value_float=total_value_float,
            net_liq_float=total_value_float,
            available_funds_float=total_value_float,
            excess_liquidity_float=total_value_float,
            cushion_float=1.0,
            position_amount_map={
                str(asset_str): float(share_float)
                for asset_str, share_float in pod_state_obj.position_amount_map.items()
                if abs(float(share_float)) > 1e-9
            },
            open_order_id_list=[],
        )

    def get_live_price_snapshot(
        self,
        account_route_str: str,
        asset_str_list: list[str],
        execution_policy_str: str | None = None,
    ) -> LivePriceSnapshot:
        release_obj = self._require_release_for_account_route(account_route_str)
        execution_policy_str = str(execution_policy_str or release_obj.execution_policy_str)
        normalized_asset_str_list = sorted({str(asset_str) for asset_str in asset_str_list})

        if execution_policy_str in OPEN_EXECUTION_POLICY_SET:
            # *** CRITICAL*** Open-policy incubation sizing uses the previous
            # completed session close. Using target-session open would leak the
            # execution price into pre-open share sizing.
            reference_date_str = self._previous_session_date_str(release_obj)
            price_map_dict = self.official_price_lookup_func(
                normalized_asset_str_list,
                reference_date_str,
                "Close",
            )
            price_source_str = f"incubation.norgate.official_close.{reference_date_str}"
        elif execution_policy_str == "same_day_moc":
            if self.preclose_price_lookup_func is None:
                raise RuntimeError(
                    "Incubation same_day_moc requires a causal pre-close snapshot source. "
                    "Official close is not allowed as the sizing reference."
                )
            # *** CRITICAL*** MOC sizing must come from a pre-close snapshot
            # observed before the auction cutoff, never from official close.
            reference_date_str = self._current_session_date_str(release_obj)
            price_map_dict = self.preclose_price_lookup_func(
                normalized_asset_str_list,
                reference_date_str,
            )
            price_source_str = f"incubation.preclose_snapshot.{reference_date_str}"
        else:
            raise ValueError(f"Unsupported execution_policy_str '{execution_policy_str}'.")

        missing_asset_list = sorted(
            asset_str for asset_str in normalized_asset_str_list if asset_str not in price_map_dict
        )
        if len(missing_asset_list) > 0:
            raise RuntimeError(
                "Missing incubation sizing reference prices for assets: "
                f"{missing_asset_list}."
            )
        return LivePriceSnapshot(
            account_route_str=account_route_str,
            snapshot_timestamp_ts=self.as_of_ts,
            price_source_str=price_source_str,
            asset_reference_price_map={
                asset_str: float(price_map_dict[asset_str])
                for asset_str in normalized_asset_str_list
            },
            asset_reference_source_map_dict={
                asset_str: price_source_str for asset_str in normalized_asset_str_list
            },
        )

    def submit_order_request_list(
        self,
        account_route_str: str,
        broker_order_request_list: list[BrokerOrderRequest],
        submitted_timestamp_ts: datetime,
    ) -> SubmitBatchResult:
        broker_order_record_list: list[BrokerOrderRecord] = []
        broker_order_event_list: list[BrokerOrderEvent] = []
        broker_order_ack_list: list[BrokerOrderAck] = []

        for broker_order_request_obj in broker_order_request_list:
            broker_order_id_str = incubation_broker_order_id_str(
                broker_order_request_obj.order_request_key_str
            )
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
                    amount_float=float(broker_order_request_obj.amount_float),
                    filled_amount_float=0.0,
                    remaining_amount_float=requested_quantity_float,
                    avg_fill_price_float=None,
                    status_str="PendingSubmit",
                    last_status_timestamp_ts=submitted_timestamp_ts,
                    submitted_timestamp_ts=submitted_timestamp_ts,
                    submission_key_str=broker_order_request_obj.submission_key_str,
                    raw_payload_dict={
                        "broker_str": "incubation",
                        "submission_key_str": broker_order_request_obj.submission_key_str,
                        "order_request_key_str": broker_order_request_obj.order_request_key_str,
                    },
                )
            )
            broker_order_event_list.append(
                BrokerOrderEvent(
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
                    event_source_str="incubation.submit",
                    message_str="accepted by Clearinghouse-Lite",
                    submission_key_str=broker_order_request_obj.submission_key_str,
                    raw_payload_dict={},
                )
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
                    ack_source_str="incubation.accepted",
                    broker_order_id_str=broker_order_id_str,
                    response_timestamp_ts=submitted_timestamp_ts,
                    raw_payload_dict={
                        "submission_key_str": broker_order_request_obj.submission_key_str,
                    },
                )
            )

        return SubmitBatchResult(
            broker_order_record_list=broker_order_record_list,
            broker_order_event_list=broker_order_event_list,
            broker_order_fill_list=[],
            broker_order_ack_list=broker_order_ack_list,
            ack_coverage_ratio_float=1.0,
            missing_ack_asset_list=[],
            submit_ack_status_str="complete",
        )

    def get_recent_fill_list(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> list[BrokerOrderFill]:
        _, _, broker_order_fill_list = self.get_recent_order_state_snapshot(
            account_route_str=account_route_str,
            since_timestamp_ts=since_timestamp_ts,
            allowed_broker_order_id_set=allowed_broker_order_id_set,
        )
        return broker_order_fill_list

    def get_recent_order_state_snapshot(
        self,
        account_route_str: str,
        since_timestamp_ts: datetime,
        submission_key_str: str | None = None,
        allowed_broker_order_id_set: set[str] | None = None,
    ) -> tuple[list[BrokerOrderRecord], list[BrokerOrderEvent], list[BrokerOrderFill]]:
        del since_timestamp_ts, allowed_broker_order_id_set
        self._settle_due_vplan_list_for_account(account_route_str)
        if submission_key_str is None:
            return [], [], []

        for vplan_obj in self.state_store_obj.get_submitted_vplan_list():
            expected_submission_key_str = str(
                vplan_obj.submission_key_str or f"vplan:{vplan_obj.decision_plan_id_int}"
            )
            if vplan_obj.account_route_str != account_route_str:
                continue
            if expected_submission_key_str != str(submission_key_str):
                continue
            return self._settled_state_by_vplan_id_dict.get(
                int(vplan_obj.vplan_id_int or 0),
                ([], [], []),
            )
        return [], [], []

    def get_session_open_price_list(
        self,
        account_route_str: str,
        asset_str_list: list[str],
        session_open_timestamp_ts: datetime,
        session_calendar_id_str: str,
    ) -> list[SessionOpenPrice]:
        session_date_str = scheduler_utils.to_market_timestamp_ts(
            session_open_timestamp_ts,
            session_calendar_id_str,
        ).date().isoformat()
        price_field_str = self._reference_price_field_for_account_session_str(
            account_route_str=account_route_str,
            session_date_str=session_date_str,
        )
        if price_field_str == "Open":
            return self._load_cached_or_ibkr_tick_open_price_list(
                account_route_str=account_route_str,
                asset_str_list=asset_str_list,
                session_open_timestamp_ts=session_open_timestamp_ts,
                session_calendar_id_str=session_calendar_id_str,
            )

        official_price_map_dict = self.official_price_lookup_func(
            sorted({str(asset_str) for asset_str in asset_str_list}),
            session_date_str,
            price_field_str,
        )
        source_str = (
            "incubation.norgate.official_close"
            if price_field_str == "Close"
            else "incubation.norgate.official_open"
        )
        return [
            SessionOpenPrice(
                session_date_str=session_date_str,
                account_route_str=account_route_str,
                asset_str=str(asset_str),
                official_open_price_float=official_price_map_dict.get(str(asset_str)),
                open_price_source_str=source_str,
                snapshot_timestamp_ts=self.as_of_ts,
                raw_payload_dict={"price_field_str": price_field_str},
            )
            for asset_str in sorted({str(asset_str) for asset_str in asset_str_list})
        ]

    def _load_cached_or_ibkr_tick_open_price_list(
        self,
        *,
        account_route_str: str,
        asset_str_list: list[str],
        session_open_timestamp_ts: datetime,
        session_calendar_id_str: str,
    ) -> list[SessionOpenPrice]:
        normalized_asset_str_list = sorted({str(asset_str) for asset_str in asset_str_list})
        if len(normalized_asset_str_list) == 0:
            return []
        session_date_str = scheduler_utils.to_market_timestamp_ts(
            session_open_timestamp_ts,
            session_calendar_id_str,
        ).date().isoformat()
        existing_session_open_price_map_dict = self.state_store_obj.get_session_open_price_map_dict(
            account_route_str=account_route_str,
            session_date_str=session_date_str,
        )
        session_open_price_list: list[SessionOpenPrice] = []
        missing_asset_str_list: list[str] = []
        for asset_str in normalized_asset_str_list:
            existing_session_open_price_obj = existing_session_open_price_map_dict.get(asset_str)
            if (
                existing_session_open_price_obj is not None
                and existing_session_open_price_obj.official_open_price_float is not None
                and str(existing_session_open_price_obj.open_price_source_str) == IBKR_TICK_OPEN_SOURCE_STR
            ):
                session_open_price_list.append(existing_session_open_price_obj)
                continue
            missing_asset_str_list.append(asset_str)

        if len(missing_asset_str_list) == 0:
            return session_open_price_list

        if self.ibkr_tick_open_lookup_func is not None:
            fetched_session_open_price_list = self.ibkr_tick_open_lookup_func(
                account_route_str,
                missing_asset_str_list,
                session_open_timestamp_ts,
                session_calendar_id_str,
            )
        else:
            fetched_session_open_price_list = self.ibkr_socket_client_obj.get_tick_open_price_list(
                account_route_str=account_route_str,
                asset_str_list=missing_asset_str_list,
                session_open_timestamp_ts=session_open_timestamp_ts,
                session_calendar_id_str=session_calendar_id_str,
            )
        return session_open_price_list + fetched_session_open_price_list

    def _require_ibkr_tick_open_price_map_dict(
        self,
        *,
        account_route_str: str,
        asset_str_list: list[str],
        session_open_timestamp_ts: datetime,
        session_calendar_id_str: str,
        reason_str: str,
    ) -> tuple[dict[str, float], dict[str, str], list[SessionOpenPrice]]:
        normalized_asset_str_list = sorted({str(asset_str) for asset_str in asset_str_list})
        session_date_str = scheduler_utils.to_market_timestamp_ts(
            session_open_timestamp_ts,
            session_calendar_id_str,
        ).date().isoformat()
        session_open_price_list = self._load_cached_or_ibkr_tick_open_price_list(
            account_route_str=account_route_str,
            asset_str_list=normalized_asset_str_list,
            session_open_timestamp_ts=session_open_timestamp_ts,
            session_calendar_id_str=session_calendar_id_str,
        )
        session_open_price_by_asset_map_dict = {
            str(session_open_price_obj.asset_str): session_open_price_obj
            for session_open_price_obj in session_open_price_list
        }
        price_map_dict: dict[str, float] = {}
        source_map_dict: dict[str, str] = {}
        missing_asset_str_list: list[str] = []
        for asset_str in normalized_asset_str_list:
            session_open_price_obj = session_open_price_by_asset_map_dict.get(asset_str)
            price_float = (
                None
                if session_open_price_obj is None
                else session_open_price_obj.official_open_price_float
            )
            if price_float is None or not math.isfinite(float(price_float)) or float(price_float) <= 0.0:
                missing_asset_str_list.append(asset_str)
                continue
            price_map_dict[asset_str] = float(price_float)
            source_map_dict[asset_str] = str(
                session_open_price_obj.open_price_source_str or IBKR_TICK_OPEN_SOURCE_STR
            )

        if len(missing_asset_str_list) > 0:
            raise RuntimeError(
                "Missing IBKR tick-open price for incubation "
                f"{reason_str}: assets={missing_asset_str_list} "
                f"date={session_date_str} source={IBKR_TICK_OPEN_SOURCE_STR}."
            )
        return price_map_dict, source_map_dict, session_open_price_list

    def _get_release_for_account_route(self, account_route_str: str) -> LiveRelease | None:
        for release_obj in self.state_store_obj.get_enabled_release_list():
            if release_obj.mode_str != "incubation":
                continue
            if str(release_obj.account_route_str) == str(account_route_str):
                return release_obj
        return None

    def _require_release_for_account_route(self, account_route_str: str) -> LiveRelease:
        release_obj = self._get_release_for_account_route(account_route_str)
        if release_obj is None:
            raise RuntimeError(f"No enabled incubation release for account_route_str '{account_route_str}'.")
        return release_obj

    def _get_pod_state_or_default(self, release_obj: LiveRelease) -> PodState:
        pod_state_obj = self.state_store_obj.get_pod_state(release_obj.pod_id_str)
        if pod_state_obj is not None:
            return pod_state_obj
        capital_base_float = float(release_obj.params_dict.get("capital_base_float", 100_000.0))
        return PodState(
            pod_id_str=release_obj.pod_id_str,
            user_id_str=release_obj.user_id_str,
            account_route_str=release_obj.account_route_str,
            position_amount_map={},
            cash_float=capital_base_float,
            total_value_float=capital_base_float,
            strategy_state_dict={},
            updated_timestamp_ts=self.as_of_ts,
        )

    def _current_session_label_ts(self, release_obj: LiveRelease) -> pd.Timestamp:
        session_label_ts = scheduler_utils.session_label_from_timestamp_ts(
            self.as_of_ts,
            release_obj.session_calendar_id_str,
        )
        if session_label_ts is None:
            raise RuntimeError(
                "Incubation command timestamp is not inside a supported market session: "
                f"{self.as_of_ts.isoformat()}."
            )
        return session_label_ts

    def _current_session_date_str(self, release_obj: LiveRelease) -> str:
        return self._current_session_label_ts(release_obj).date().isoformat()

    def _previous_session_date_str(self, release_obj: LiveRelease) -> str:
        current_session_label_ts = self._current_session_label_ts(release_obj)
        calendar_obj = scheduler_utils.get_exchange_calendar_obj(release_obj.session_calendar_id_str)
        previous_session_label_ts = calendar_obj.previous_session(current_session_label_ts)
        return previous_session_label_ts.date().isoformat()

    def _load_mark_price_map_dict(
        self,
        release_obj: LiveRelease,
        asset_str_list: list[str],
    ) -> dict[str, float]:
        normalized_asset_str_list = sorted({str(asset_str) for asset_str in asset_str_list})
        if len(normalized_asset_str_list) == 0:
            return {}
        session_label_ts = self._current_session_label_ts(release_obj)
        market_timestamp_ts = scheduler_utils.to_market_timestamp_ts(
            self.as_of_ts,
            release_obj.session_calendar_id_str,
        )
        session_open_timestamp_ts = scheduler_utils.get_session_open_timestamp_ts(
            session_label_ts,
            release_obj.session_calendar_id_str,
        )
        session_close_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(
            session_label_ts,
            release_obj.session_calendar_id_str,
        )
        if market_timestamp_ts >= session_close_timestamp_ts:
            # *** CRITICAL*** After the session close, incubation may mark with
            # same-session official close. Before that point, close is not
            # known and must not be used.
            session_date_str = session_label_ts.date().isoformat()
            price_field_str = "Close"
        elif market_timestamp_ts >= session_open_timestamp_ts:
            # *** CRITICAL*** During the execution session after the open,
            # IBKR tick-open is the only same-day open mark allowed in
            # incubation. Same-day close is not known yet.
            price_map_dict, _, _ = self._require_ibkr_tick_open_price_map_dict(
                account_route_str=release_obj.account_route_str,
                asset_str_list=normalized_asset_str_list,
                session_open_timestamp_ts=session_open_timestamp_ts,
                session_calendar_id_str=release_obj.session_calendar_id_str,
                reason_str="post-open mark",
            )
            return price_map_dict
        else:
            # *** CRITICAL*** Pre-open marks use previous completed close so
            # account equity cannot depend on not-yet-observed open prices.
            session_date_str = self._previous_session_date_str(release_obj)
            price_field_str = "Close"
        return self.official_price_lookup_func(
            normalized_asset_str_list,
            session_date_str,
            price_field_str,
        )

    def _settle_due_vplan_list_for_account(self, account_route_str: str) -> None:
        for vplan_obj in self.state_store_obj.get_submitted_vplan_list():
            if vplan_obj.account_route_str != account_route_str:
                continue
            if self.as_of_ts < vplan_obj.target_execution_timestamp_ts:
                continue
            vplan_id_int = int(vplan_obj.vplan_id_int or 0)
            if len(self.state_store_obj.get_fill_row_dict_list_for_vplan(vplan_id_int)) > 0:
                continue
            self._settle_vplan(vplan_obj)

    def _settle_vplan(self, vplan_obj: VPlan) -> None:
        release_obj = self.state_store_obj.get_release_by_id(vplan_obj.release_id_str)
        pod_state_obj = self._get_pod_state_or_default(release_obj)
        fill_price_field_str = "Close" if vplan_obj.execution_policy_str == "same_day_moc" else "Open"
        # *** CRITICAL*** Settlement date is the target execution session from
        # the VPlan, not the wall-clock date of this reconcile command.
        target_session_label_ts = scheduler_utils.session_label_from_timestamp_ts(
            vplan_obj.target_execution_timestamp_ts,
            release_obj.session_calendar_id_str,
        )
        if target_session_label_ts is None:
            raise RuntimeError(
                "Cannot settle incubation VPlan because target_execution_timestamp_ts "
                "does not map to a trading session."
            )
        target_session_date_str = target_session_label_ts.date().isoformat()
        target_session_open_timestamp_ts = scheduler_utils.get_session_open_timestamp_ts(
            target_session_label_ts,
            release_obj.session_calendar_id_str,
        )
        settlement_asset_str_list = [
            str(vplan_row_obj.asset_str)
            for vplan_row_obj in vplan_obj.vplan_row_list
            if abs(float(vplan_row_obj.order_delta_share_float)) > 1e-9
        ]
        session_open_price_by_asset_map_dict: dict[str, SessionOpenPrice] = {}
        if fill_price_field_str == "Open":
            official_fill_price_map_dict, fill_price_source_map_dict, session_open_price_list = (
                self._require_ibkr_tick_open_price_map_dict(
                    account_route_str=vplan_obj.account_route_str,
                    asset_str_list=settlement_asset_str_list,
                    session_open_timestamp_ts=target_session_open_timestamp_ts,
                    session_calendar_id_str=release_obj.session_calendar_id_str,
                    reason_str="settlement",
                )
            )
            session_open_price_by_asset_map_dict.update(
                {
                    str(session_open_price_obj.asset_str): session_open_price_obj
                    for session_open_price_obj in session_open_price_list
                }
            )
        else:
            official_fill_price_map_dict = self.official_price_lookup_func(
                settlement_asset_str_list,
                target_session_date_str,
                fill_price_field_str,
            )
            fill_price_source_map_dict = {
                asset_str: "incubation.norgate.official_close"
                for asset_str in settlement_asset_str_list
            }
        updated_position_map_dict = {
            str(asset_str): float(share_float)
            for asset_str, share_float in pod_state_obj.position_amount_map.items()
        }
        cash_float = float(pod_state_obj.cash_float)
        broker_order_record_list: list[BrokerOrderRecord] = []
        broker_order_event_list: list[BrokerOrderEvent] = []
        broker_order_fill_list: list[BrokerOrderFill] = []
        cash_ledger_entry_list: list[CashLedgerEntry] = []
        submission_key_str = str(vplan_obj.submission_key_str or f"vplan:{vplan_obj.decision_plan_id_int}")

        for request_idx_int, vplan_row_obj in enumerate(vplan_obj.vplan_row_list, start=1):
            fill_quantity_float = float(vplan_row_obj.order_delta_share_float)
            if abs(fill_quantity_float) <= 1e-9:
                continue
            asset_str = str(vplan_row_obj.asset_str)
            if asset_str not in official_fill_price_map_dict:
                raise RuntimeError(
                    "Missing official incubation fill price: "
                    f"asset={asset_str} date={target_session_date_str} field={fill_price_field_str}."
                )
            fill_price_float = float(official_fill_price_map_dict[asset_str])
            fill_price_source_str = str(fill_price_source_map_dict.get(asset_str, ""))
            order_request_key_str = f"{submission_key_str}:{asset_str}:{request_idx_int}"
            broker_order_id_str = incubation_broker_order_id_str(order_request_key_str)
            commission_float = self._compute_commission_float(
                release_obj=release_obj,
                quantity_float=fill_quantity_float,
            )
            trade_cash_delta_float = -fill_quantity_float * fill_price_float
            commission_cash_delta_float = -commission_float
            cash_float += trade_cash_delta_float + commission_cash_delta_float
            updated_position_float = float(updated_position_map_dict.get(asset_str, 0.0)) + fill_quantity_float
            if abs(updated_position_float) <= 1e-9:
                updated_position_map_dict.pop(asset_str, None)
            else:
                updated_position_map_dict[asset_str] = updated_position_float

            broker_order_record_list.append(
                BrokerOrderRecord(
                    broker_order_id_str=broker_order_id_str,
                    decision_plan_id_int=int(vplan_obj.decision_plan_id_int),
                    vplan_id_int=int(vplan_obj.vplan_id_int or 0),
                    account_route_str=vplan_obj.account_route_str,
                    asset_str=asset_str,
                    order_request_key_str=order_request_key_str,
                    broker_order_type_str=vplan_row_obj.broker_order_type_str,
                    unit_str="shares",
                    amount_float=fill_quantity_float,
                    filled_amount_float=fill_quantity_float,
                    remaining_amount_float=0.0,
                    avg_fill_price_float=fill_price_float,
                    status_str="Filled",
                    submitted_timestamp_ts=vplan_obj.submission_timestamp_ts,
                    last_status_timestamp_ts=vplan_obj.target_execution_timestamp_ts,
                    submission_key_str=submission_key_str,
                    raw_payload_dict={"broker_str": "incubation"},
                )
            )
            broker_order_event_list.append(
                BrokerOrderEvent(
                    broker_order_id_str=broker_order_id_str,
                    decision_plan_id_int=int(vplan_obj.decision_plan_id_int),
                    vplan_id_int=int(vplan_obj.vplan_id_int or 0),
                    account_route_str=vplan_obj.account_route_str,
                    asset_str=asset_str,
                    order_request_key_str=order_request_key_str,
                    status_str="Filled",
                    filled_amount_float=fill_quantity_float,
                    remaining_amount_float=0.0,
                    avg_fill_price_float=fill_price_float,
                    event_timestamp_ts=vplan_obj.target_execution_timestamp_ts,
                    event_source_str="incubation.settlement",
                    message_str="configured-price fill",
                    submission_key_str=submission_key_str,
                    raw_payload_dict={
                        "price_field_str": fill_price_field_str,
                        "price_source_str": fill_price_source_str,
                    },
                )
            )
            broker_order_fill_list.append(
                BrokerOrderFill(
                    broker_order_id_str=broker_order_id_str,
                    decision_plan_id_int=int(vplan_obj.decision_plan_id_int),
                    vplan_id_int=int(vplan_obj.vplan_id_int or 0),
                    account_route_str=vplan_obj.account_route_str,
                    asset_str=asset_str,
                    fill_amount_float=fill_quantity_float,
                    fill_price_float=fill_price_float,
                    fill_timestamp_ts=vplan_obj.target_execution_timestamp_ts,
                    official_open_price_float=fill_price_float,
                    open_price_source_str=fill_price_source_str,
                    raw_payload_dict={
                        "price_field_str": fill_price_field_str,
                        "price_source_str": fill_price_source_str,
                    },
                )
            )
            cash_ledger_entry_list.extend(
                [
                    CashLedgerEntry(
                        pod_id_str=vplan_obj.pod_id_str,
                        account_route_str=vplan_obj.account_route_str,
                        vplan_id_int=int(vplan_obj.vplan_id_int or 0),
                        broker_order_id_str=broker_order_id_str,
                        asset_str=asset_str,
                        entry_type_str="trade_notional",
                        cash_delta_float=trade_cash_delta_float,
                        entry_timestamp_ts=vplan_obj.target_execution_timestamp_ts,
                        raw_payload_dict={
                            "quantity_float": fill_quantity_float,
                            "fill_price_float": fill_price_float,
                        },
                    ),
                    CashLedgerEntry(
                        pod_id_str=vplan_obj.pod_id_str,
                        account_route_str=vplan_obj.account_route_str,
                        vplan_id_int=int(vplan_obj.vplan_id_int or 0),
                        broker_order_id_str=broker_order_id_str,
                        asset_str=asset_str,
                        entry_type_str="commission",
                        cash_delta_float=commission_cash_delta_float,
                        entry_timestamp_ts=vplan_obj.target_execution_timestamp_ts,
                        raw_payload_dict={
                            "commission_per_share_float": float(
                                release_obj.params_dict.get("commission_per_share", 0.005)
                            ),
                            "commission_minimum_float": float(
                                release_obj.params_dict.get("commission_minimum", 1.0)
                            ),
                        },
                    ),
                ]
            )

        settlement_mark_price_map_dict: dict[str, float] = {}
        if len(updated_position_map_dict) > 0:
            if fill_price_field_str == "Open":
                settlement_mark_price_map_dict.update(
                    {
                        asset_str: float(official_fill_price_map_dict[asset_str])
                        for asset_str in updated_position_map_dict
                        if asset_str in official_fill_price_map_dict
                    }
                )
                missing_mark_asset_str_list = [
                    asset_str
                    for asset_str in sorted(updated_position_map_dict)
                    if asset_str not in settlement_mark_price_map_dict
                ]
                if len(missing_mark_asset_str_list) > 0:
                    mark_price_map_dict, _, mark_session_open_price_list = (
                        self._require_ibkr_tick_open_price_map_dict(
                            account_route_str=vplan_obj.account_route_str,
                            asset_str_list=missing_mark_asset_str_list,
                            session_open_timestamp_ts=target_session_open_timestamp_ts,
                            session_calendar_id_str=release_obj.session_calendar_id_str,
                            reason_str="post-settlement mark",
                        )
                    )
                    settlement_mark_price_map_dict.update(mark_price_map_dict)
                    session_open_price_by_asset_map_dict.update(
                        {
                            str(session_open_price_obj.asset_str): session_open_price_obj
                            for session_open_price_obj in mark_session_open_price_list
                        }
                    )
            else:
                settlement_mark_price_map_dict = self.official_price_lookup_func(
                    sorted(updated_position_map_dict),
                    target_session_date_str,
                    fill_price_field_str,
                )
        position_value_float = sum(
            float(share_float) * float(settlement_mark_price_map_dict[asset_str])
            for asset_str, share_float in updated_position_map_dict.items()
        )
        if len(session_open_price_by_asset_map_dict) > 0:
            self.state_store_obj.upsert_session_open_price_list(
                session_open_price_by_asset_map_dict.values()
            )
        self.state_store_obj.upsert_vplan_broker_order_record_list(broker_order_record_list)
        self.state_store_obj.insert_vplan_broker_order_event_list(broker_order_event_list)
        self.state_store_obj.upsert_vplan_fill_list(broker_order_fill_list)
        self.state_store_obj.insert_cash_ledger_entry_list(cash_ledger_entry_list)
        self.state_store_obj.upsert_pod_state(
            PodState(
                pod_id_str=vplan_obj.pod_id_str,
                user_id_str=vplan_obj.user_id_str,
                account_route_str=vplan_obj.account_route_str,
                position_amount_map=updated_position_map_dict,
                cash_float=cash_float,
                total_value_float=cash_float + position_value_float,
                strategy_state_dict=dict(pod_state_obj.strategy_state_dict),
                updated_timestamp_ts=self.as_of_ts,
            ),
            snapshot_stage_str="post_execution",
            snapshot_source_str="virtual_broker",
        )
        self._settled_state_by_vplan_id_dict[int(vplan_obj.vplan_id_int or 0)] = (
            broker_order_record_list,
            broker_order_event_list,
            broker_order_fill_list,
        )

    def _compute_commission_float(
        self,
        release_obj: LiveRelease,
        quantity_float: float,
    ) -> float:
        commission_per_share_float = float(release_obj.params_dict.get("commission_per_share", 0.005))
        commission_minimum_float = float(release_obj.params_dict.get("commission_minimum", 1.0))
        if commission_per_share_float == 0.0:
            return 0.0
        return max(commission_minimum_float, commission_per_share_float * abs(float(quantity_float)))

    def _reference_price_field_for_account_session_str(
        self,
        account_route_str: str,
        session_date_str: str,
    ) -> str:
        for vplan_obj in self.state_store_obj.get_submitted_vplan_list():
            if vplan_obj.account_route_str != account_route_str:
                continue
            target_session_label_ts = scheduler_utils.session_label_from_timestamp_ts(
                vplan_obj.target_execution_timestamp_ts,
                self.state_store_obj.get_release_by_id(vplan_obj.release_id_str).session_calendar_id_str,
            )
            if target_session_label_ts is None:
                continue
            if target_session_label_ts.date().isoformat() != session_date_str:
                continue
            if vplan_obj.execution_policy_str == "same_day_moc":
                return "Close"
        return "Open"
