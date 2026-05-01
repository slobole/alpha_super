from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable
import uuid

from alpha.data import FredSeriesLoadError
from alpha.live import reference_compare, scheduler_utils, strategy_host
from alpha.live.execution_engine import (
    build_broker_order_request_list_from_vplan,
    build_vplan,
    get_touched_asset_list_for_decision_plan,
)
from alpha.live.incubation import IncubationBrokerAdapter
from alpha.live.logging_utils import DEFAULT_LOG_PATH_STR, log_event
from alpha.live.models import BrokerOrderEvent, DecisionPlan, LiveRelease, PodState, VPlan
from alpha.live.order_clerk import BrokerAdapter, IBKRGatewayBrokerAdapter
from alpha.live.reconcile import reconcile_account_state
from alpha.live.release_manifest import (
    load_release_list,
    select_enabled_release_list_for_mode,
    validate_enabled_deployment_for_mode,
)
from alpha.live.state_store import V1_EXECUTION_TABLE_NAME_TUPLE
from alpha.live.state_store_v2 import LiveStateStore


DEFAULT_RELEASES_ROOT_PATH_STR = str(Path(__file__).resolve().parent / "releases")
DEFAULT_DB_PATH_STR = str(Path(__file__).resolve().parent / "live_state.sqlite3")
DEFAULT_INCUBATION_DB_PATH_STR = str(Path(__file__).resolve().parent / "incubation_state.sqlite3")
DEFAULT_POD_STATE_ROOT_PATH_STR = str(Path(__file__).resolve().parent / "state")
DEFAULT_BROKER_HOST_STR = "127.0.0.1"
DEFAULT_BROKER_PORT_INT = 7497
DEFAULT_BROKER_CLIENT_ID_INT = 31
DEFAULT_BROKER_TIMEOUT_SECONDS_FLOAT = 4.0
DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT = 10
TERMINAL_ORDER_STATUS_SET: set[str] = {
    "Filled",
    "Cancelled",
    "ApiCancelled",
    "Rejected",
    "Expired",
    "Inactive",
}
PARKED_TERMINAL_ORDER_STATUS_SET: set[str] = {
    "Cancelled",
    "ApiCancelled",
    "Rejected",
    "Expired",
    "Inactive",
}

BrokerAdapterKey_Tuple = tuple[str, int, int, float]


class BrokerAdapterResolver:
    def __init__(
        self,
        *,
        broker_adapter_obj: BrokerAdapter | None = None,
        broker_host_str: str | None = None,
        broker_port_int: int | None = None,
        broker_client_id_int: int | None = None,
        broker_timeout_seconds_float: float | None = None,
        adapter_factory_func: Callable[[str, int, int, float], BrokerAdapter] | None = None,
        state_store_obj: LiveStateStore | None = None,
        as_of_ts: datetime | None = None,
    ) -> None:
        self._broker_adapter_obj = broker_adapter_obj
        self._broker_host_str = broker_host_str
        self._broker_port_int = broker_port_int
        self._broker_client_id_int = broker_client_id_int
        self._broker_timeout_seconds_float = broker_timeout_seconds_float
        self._adapter_factory_func = (
            adapter_factory_func if adapter_factory_func is not None else _build_ibkr_gateway_broker_adapter
        )
        self._state_store_obj = state_store_obj
        self._as_of_ts = as_of_ts
        self._adapter_by_key_tup_dict: dict[BrokerAdapterKey_Tuple, BrokerAdapter] = {}
        self._incubation_adapter_by_account_route_dict: dict[
            tuple[str, BrokerAdapterKey_Tuple],
            BrokerAdapter,
        ] = {}

    def set_incubation_context(
        self,
        *,
        state_store_obj: LiveStateStore | None,
        as_of_ts: datetime | None,
    ) -> None:
        context_changed_bool = (
            state_store_obj is not None
            and state_store_obj is not self._state_store_obj
        ) or (as_of_ts is not None and as_of_ts != self._as_of_ts)
        if state_store_obj is not None:
            self._state_store_obj = state_store_obj
        if as_of_ts is not None:
            self._as_of_ts = as_of_ts
        if context_changed_bool:
            self._incubation_adapter_by_account_route_dict.clear()

    def _build_effective_broker_adapter_key_tup(
        self,
        release_obj: LiveRelease,
    ) -> BrokerAdapterKey_Tuple:
        return (
            str(
                release_obj.broker_host_str
                if self._broker_host_str is None
                else self._broker_host_str
            ),
            int(
                release_obj.broker_port_int
                if self._broker_port_int is None
                else self._broker_port_int
            ),
            int(
                release_obj.broker_client_id_int
                if self._broker_client_id_int is None
                else self._broker_client_id_int
            ),
            float(
                release_obj.broker_timeout_seconds_float
                if self._broker_timeout_seconds_float is None
                else self._broker_timeout_seconds_float
            ),
        )

    def get_connection_field_map_dict(
        self,
        release_obj: LiveRelease,
    ) -> dict[str, object]:
        broker_adapter_key_tup = self._build_effective_broker_adapter_key_tup(release_obj)
        return {
            "broker_host_str": broker_adapter_key_tup[0],
            "broker_port_int": int(broker_adapter_key_tup[1]),
            "broker_client_id_int": int(broker_adapter_key_tup[2]),
            "broker_timeout_seconds_float": float(broker_adapter_key_tup[3]),
        }

    def get_adapter(
        self,
        release_obj: LiveRelease,
    ) -> BrokerAdapter:
        if self._broker_adapter_obj is not None:
            return self._broker_adapter_obj
        if release_obj.mode_str == "incubation":
            if self._state_store_obj is None or self._as_of_ts is None:
                raise RuntimeError(
                    "Incubation mode requires state_store_obj and as_of_ts when building the broker adapter."
                )
            account_route_str = str(release_obj.account_route_str)
            broker_adapter_key_tup = self._build_effective_broker_adapter_key_tup(release_obj)
            incubation_cache_key_tup = (account_route_str, broker_adapter_key_tup)
            if incubation_cache_key_tup not in self._incubation_adapter_by_account_route_dict:
                self._incubation_adapter_by_account_route_dict[incubation_cache_key_tup] = IncubationBrokerAdapter(
                    state_store_obj=self._state_store_obj,
                    as_of_ts=self._as_of_ts,
                    ibkr_host_str=str(broker_adapter_key_tup[0]),
                    ibkr_port_int=int(broker_adapter_key_tup[1]),
                    ibkr_client_id_int=int(broker_adapter_key_tup[2]),
                    ibkr_timeout_seconds_float=float(broker_adapter_key_tup[3]),
                )
            return self._incubation_adapter_by_account_route_dict[incubation_cache_key_tup]

        broker_adapter_key_tup = self._build_effective_broker_adapter_key_tup(release_obj)
        if broker_adapter_key_tup not in self._adapter_by_key_tup_dict:
            self._adapter_by_key_tup_dict[broker_adapter_key_tup] = self._adapter_factory_func(
                str(broker_adapter_key_tup[0]),
                int(broker_adapter_key_tup[1]),
                int(broker_adapter_key_tup[2]),
                float(broker_adapter_key_tup[3]),
            )
        return self._adapter_by_key_tup_dict[broker_adapter_key_tup]


def _build_ibkr_gateway_broker_adapter(
    host_str: str,
    port_int: int,
    client_id_int: int,
    timeout_seconds_float: float,
) -> BrokerAdapter:
    return IBKRGatewayBrokerAdapter(
        host_str=host_str,
        port_int=port_int,
        client_id_int=client_id_int,
        timeout_seconds_float=timeout_seconds_float,
    )


def _pluralize_label_str(count_int: int, singular_label_str: str, plural_label_str: str | None = None) -> str:
    if count_int == 1:
        return singular_label_str
    if plural_label_str is not None:
        return plural_label_str
    return f"{singular_label_str}s"


def _parse_as_of_timestamp_ts(as_of_timestamp_str: str | None) -> datetime:
    if as_of_timestamp_str is None:
        return datetime.now(tz=UTC)
    return datetime.fromisoformat(as_of_timestamp_str)


def _same_resolved_path_bool(left_path_str: str, right_path_str: str) -> bool:
    return Path(left_path_str).resolve() == Path(right_path_str).resolve()


def _validate_pod_id_for_path(pod_id_str: str) -> None:
    if len(str(pod_id_str).strip()) == 0:
        raise ValueError("pod_id_str must not be empty.")
    if any(separator_str in str(pod_id_str) for separator_str in ("/", "\\", ":")):
        raise ValueError("pod_id_str must not contain path separators.")


def _build_default_pod_db_path_str(env_mode_str: str, pod_id_str: str) -> str:
    _validate_pod_id_for_path(pod_id_str)
    return str(Path(DEFAULT_POD_STATE_ROOT_PATH_STR) / env_mode_str / f"{pod_id_str}.sqlite3")


def _resolve_db_path_for_mode_str(
    db_path_str: str | None,
    env_mode_str: str,
    pod_id_str: str | None = None,
) -> str:
    if db_path_str is None:
        if pod_id_str is not None:
            return _build_default_pod_db_path_str(env_mode_str, pod_id_str)
        if env_mode_str == "incubation":
            return DEFAULT_INCUBATION_DB_PATH_STR
        return DEFAULT_DB_PATH_STR

    if env_mode_str == "incubation" and _same_resolved_path_bool(db_path_str, DEFAULT_DB_PATH_STR):
        raise ValueError(
            "Incubation must not use the paper/live default DB. "
            f"Use '{DEFAULT_INCUBATION_DB_PATH_STR}' or pass a different --db-path."
        )
    if env_mode_str in ("paper", "live") and _same_resolved_path_bool(
        db_path_str,
        DEFAULT_INCUBATION_DB_PATH_STR,
    ):
        raise ValueError(
            "Paper/live must not use the incubation default DB. "
            f"Use '{DEFAULT_DB_PATH_STR}' or pass a different --db-path."
        )
    return db_path_str


def _build_v1_cutover_archive_dir_path_obj(
    db_path_str: str,
    as_of_ts: datetime,
    archive_root_path_str: str | None,
) -> Path:
    archive_root_path_obj = (
        Path(archive_root_path_str).resolve()
        if archive_root_path_str is not None
        else Path(db_path_str).resolve().parent / "schema_archives"
    )
    timestamp_label_str = as_of_ts.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    db_stem_str = Path(db_path_str).resolve().stem
    return archive_root_path_obj / f"{db_stem_str}_v1_cutover_{timestamp_label_str}"


def _load_release_list_and_sync(
    releases_root_path_str: str,
    state_store_obj: LiveStateStore,
    pod_id_str: str | None = None,
) -> list[LiveRelease]:
    release_list = load_release_list(releases_root_path_str)
    selected_release_list = _filter_release_list_for_pod(release_list, pod_id_str)
    state_store_obj.upsert_release_list(selected_release_list)
    return selected_release_list


def _load_release_list_validate_and_sync(
    releases_root_path_str: str,
    state_store_obj: LiveStateStore,
    env_mode_str: str,
    pod_id_str: str | None = None,
) -> list[LiveRelease]:
    release_list = load_release_list(releases_root_path_str)
    validate_enabled_deployment_for_mode(release_list, env_mode_str)
    selected_release_list = _filter_release_list_for_pod(release_list, pod_id_str)
    _validate_selected_pod_enabled_for_mode(selected_release_list, env_mode_str, pod_id_str)
    state_store_obj.upsert_release_list(selected_release_list)
    return selected_release_list


def _filter_release_list_for_pod(
    release_list: list[LiveRelease],
    pod_id_str: str | None,
) -> list[LiveRelease]:
    if pod_id_str is None:
        return release_list
    if len(release_list) == 0:
        return []
    selected_release_list = [
        release_obj
        for release_obj in release_list
        if release_obj.pod_id_str == pod_id_str
    ]
    if len(selected_release_list) == 0:
        raise ValueError(f"No release found for pod_id_str '{pod_id_str}'.")
    return selected_release_list


def _validate_selected_pod_enabled_for_mode(
    release_list: list[LiveRelease],
    env_mode_str: str,
    pod_id_str: str | None,
) -> None:
    if pod_id_str is None:
        return
    enabled_release_list = select_enabled_release_list_for_mode(
        release_list=release_list,
        env_mode_str=env_mode_str,
    )
    if len(enabled_release_list) == 0:
        raise ValueError(
            f"No enabled release found for pod_id_str '{pod_id_str}' in mode '{env_mode_str}'."
        )


def _validate_release_root_for_mutation(
    releases_root_path_str: str,
    env_mode_str: str,
    pod_id_str: str | None = None,
) -> None:
    release_list = load_release_list(releases_root_path_str)
    validate_enabled_deployment_for_mode(release_list, env_mode_str)
    selected_release_list = _filter_release_list_for_pod(release_list, pod_id_str)
    _validate_selected_pod_enabled_for_mode(selected_release_list, env_mode_str, pod_id_str)


def _validate_state_store_releases_for_mutation(
    state_store_obj: LiveStateStore,
    env_mode_str: str,
    pod_id_str: str | None = None,
) -> None:
    release_list = state_store_obj.get_enabled_release_list()
    validate_enabled_deployment_for_mode(release_list, env_mode_str)
    selected_release_list = _filter_release_list_for_pod(release_list, pod_id_str)
    _validate_selected_pod_enabled_for_mode(selected_release_list, env_mode_str, pod_id_str)


def _coerce_broker_adapter_resolver_obj(
    *,
    broker_adapter_obj: BrokerAdapter | None,
    broker_adapter_resolver_obj: BrokerAdapterResolver | None,
    broker_host_str: str | None,
    broker_port_int: int | None,
    broker_client_id_int: int | None,
    broker_timeout_seconds_float: float | None,
    state_store_obj: LiveStateStore | None = None,
    as_of_ts: datetime | None = None,
    adapter_factory_func: Callable[[str, int, int, float], BrokerAdapter] | None = None,
) -> BrokerAdapterResolver:
    if broker_adapter_resolver_obj is not None:
        broker_adapter_resolver_obj.set_incubation_context(
            state_store_obj=state_store_obj,
            as_of_ts=as_of_ts,
        )
        return broker_adapter_resolver_obj
    return BrokerAdapterResolver(
        broker_adapter_obj=broker_adapter_obj,
        broker_host_str=broker_host_str,
        broker_port_int=broker_port_int,
        broker_client_id_int=broker_client_id_int,
        broker_timeout_seconds_float=broker_timeout_seconds_float,
        adapter_factory_func=adapter_factory_func,
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
    )


def _get_model_state_or_default(
    release_obj: LiveRelease,
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
) -> PodState:
    pod_state_obj = state_store_obj.get_pod_state(release_obj.pod_id_str)
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
        updated_timestamp_ts=as_of_ts,
    )


def _build_release_log_payload_dict(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    extra_payload_dict: dict | None = None,
) -> dict:
    payload_dict = {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "pod_id_str": release_obj.pod_id_str,
        "release_id_str": release_obj.release_id_str,
        "account_route_str": release_obj.account_route_str,
        "session_calendar_id_str": release_obj.session_calendar_id_str,
        "signal_clock_str": release_obj.signal_clock_str,
        "execution_policy_str": release_obj.execution_policy_str,
        "pod_budget_fraction_float": float(release_obj.pod_budget_fraction_float),
        "auto_submit_enabled_bool": bool(release_obj.auto_submit_enabled_bool),
    }
    if extra_payload_dict is not None:
        payload_dict.update(extra_payload_dict)
    return payload_dict


def _build_fred_series_error_log_payload_dict(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    exception_obj: FredSeriesLoadError,
) -> dict:
    extra_payload_dict: dict[str, object] = {
        "reason_code_str": str(exception_obj.reason_code_str),
        "error_str": str(exception_obj),
        "series_id_str": str(exception_obj.series_id_str),
    }
    if exception_obj.series_snapshot_obj is not None:
        extra_payload_dict.update(
            {
                "source_name_str": str(exception_obj.series_snapshot_obj.source_name_str),
                "latest_observation_date_str": (
                    exception_obj.series_snapshot_obj.latest_observation_date_ts.date().isoformat()
                ),
                "download_attempt_timestamp_str": (
                    exception_obj.series_snapshot_obj.download_attempt_timestamp_ts.isoformat()
                ),
                "download_status_str": str(exception_obj.series_snapshot_obj.download_status_str),
                "used_cache_bool": bool(exception_obj.series_snapshot_obj.used_cache_bool),
                "freshness_business_days_int": int(
                    exception_obj.series_snapshot_obj.freshness_business_days_int
                ),
            }
        )
    return _build_release_log_payload_dict(
        release_obj=release_obj,
        as_of_ts=as_of_ts,
        extra_payload_dict=extra_payload_dict,
    )


def _build_decision_plan_log_payload_dict(
    release_obj: LiveRelease,
    decision_plan_obj: DecisionPlan,
    as_of_ts: datetime,
    extra_payload_dict: dict | None = None,
) -> dict:
    payload_dict = _build_release_log_payload_dict(
        release_obj,
        as_of_ts,
        {
            "decision_plan_id_int": int(decision_plan_obj.decision_plan_id_int or 0),
            "decision_plan_status_str": decision_plan_obj.status_str,
            "signal_timestamp_str": decision_plan_obj.signal_timestamp_ts.isoformat(),
            "submission_timestamp_str": decision_plan_obj.submission_timestamp_ts.isoformat(),
            "target_execution_timestamp_str": decision_plan_obj.target_execution_timestamp_ts.isoformat(),
        },
    )
    if extra_payload_dict is not None:
        payload_dict.update(extra_payload_dict)
    return payload_dict


def _build_vplan_log_payload_dict(
    release_obj: LiveRelease,
    vplan_obj: VPlan,
    as_of_ts: datetime,
    extra_payload_dict: dict | None = None,
) -> dict:
    payload_dict = _build_release_log_payload_dict(
        release_obj,
        as_of_ts,
        {
            "vplan_id_int": int(vplan_obj.vplan_id_int or 0),
            "decision_plan_id_int": int(vplan_obj.decision_plan_id_int),
            "vplan_status_str": vplan_obj.status_str,
            "signal_timestamp_str": vplan_obj.signal_timestamp_ts.isoformat(),
            "submission_timestamp_str": vplan_obj.submission_timestamp_ts.isoformat(),
            "target_execution_timestamp_str": vplan_obj.target_execution_timestamp_ts.isoformat(),
        },
    )
    if extra_payload_dict is not None:
        payload_dict.update(extra_payload_dict)
    return payload_dict


def _is_terminal_order_status_bool(status_str: str) -> bool:
    return str(status_str) in TERMINAL_ORDER_STATUS_SET


def _session_open_context_dict(
    release_obj: LiveRelease,
    reference_timestamp_ts: datetime,
) -> dict[str, object]:
    session_label_ts = scheduler_utils.session_label_from_timestamp_ts(
        reference_timestamp_ts,
        release_obj.session_calendar_id_str,
    )
    if session_label_ts is None:
        raise ValueError(
            f"No market session found for timestamp '{reference_timestamp_ts.isoformat()}'."
        )
    session_open_timestamp_ts = scheduler_utils.get_session_open_timestamp_ts(
        session_label_ts,
        release_obj.session_calendar_id_str,
    )
    return {
        "session_date_str": session_label_ts.date().isoformat(),
        "session_open_timestamp_ts": session_open_timestamp_ts,
    }


def _annotate_fill_list_with_session_open_price(
    fill_list,
    session_open_price_map_dict,
) -> list:
    annotated_fill_list = []
    for fill_obj in fill_list:
        session_open_price_obj = session_open_price_map_dict.get(fill_obj.asset_str)
        annotated_fill_list.append(
            fill_obj.__class__(
                **{
                    **fill_obj.__dict__,
                    "official_open_price_float": (
                        None
                        if session_open_price_obj is None
                        else session_open_price_obj.official_open_price_float
                    ),
                    "open_price_source_str": (
                        None
                        if session_open_price_obj is None
                        else session_open_price_obj.open_price_source_str
                    ),
                }
            )
        )
    return annotated_fill_list


def _synthesize_terminal_filled_event_list(
    broker_order_record_list,
    broker_order_event_list,
    broker_order_fill_list,
):
    broker_order_event_list = list(broker_order_event_list)
    broker_order_event_by_order_map = {}
    for broker_order_event_obj in broker_order_event_list:
        broker_order_event_by_order_map.setdefault(
            broker_order_event_obj.broker_order_id_str,
            [],
        ).append(broker_order_event_obj)

    broker_order_fill_by_order_map = {}
    for broker_order_fill_obj in broker_order_fill_list:
        broker_order_fill_by_order_map.setdefault(
            broker_order_fill_obj.broker_order_id_str,
            [],
        ).append(broker_order_fill_obj)

    synthesized_event_list = []
    normalized_record_list = []
    for broker_order_record_obj in broker_order_record_list:
        broker_order_fill_row_list = broker_order_fill_by_order_map.get(
            broker_order_record_obj.broker_order_id_str,
            [],
        )
        cumulative_filled_amount_float = sum(
            abs(float(broker_order_fill_obj.fill_amount_float))
            for broker_order_fill_obj in broker_order_fill_row_list
        )
        requested_amount_float = abs(float(broker_order_record_obj.amount_float))
        existing_terminal_filled_bool = any(
            str(broker_order_event_obj.status_str) == "Filled"
            for broker_order_event_obj in broker_order_event_by_order_map.get(
                broker_order_record_obj.broker_order_id_str,
                [],
            )
        )
        if (
            requested_amount_float > 0.0
            and abs(cumulative_filled_amount_float - requested_amount_float) <= 1e-9
            and not existing_terminal_filled_bool
            and len(broker_order_fill_row_list) > 0
        ):
            last_fill_timestamp_ts = max(
                broker_order_fill_obj.fill_timestamp_ts
                for broker_order_fill_obj in broker_order_fill_row_list
            )
            weighted_notional_float = sum(
                abs(float(broker_order_fill_obj.fill_amount_float))
                * float(broker_order_fill_obj.fill_price_float)
                for broker_order_fill_obj in broker_order_fill_row_list
            )
            avg_fill_price_float = (
                weighted_notional_float / cumulative_filled_amount_float
                if cumulative_filled_amount_float > 0.0
                else None
            )
            synthesized_event_list.append(
                BrokerOrderEvent(
                    broker_order_id_str=broker_order_record_obj.broker_order_id_str,
                    decision_plan_id_int=broker_order_record_obj.decision_plan_id_int,
                    vplan_id_int=broker_order_record_obj.vplan_id_int,
                    account_route_str=broker_order_record_obj.account_route_str,
                    asset_str=broker_order_record_obj.asset_str,
                    order_request_key_str=broker_order_record_obj.order_request_key_str,
                    status_str="Filled",
                    filled_amount_float=cumulative_filled_amount_float,
                    remaining_amount_float=0.0,
                    avg_fill_price_float=avg_fill_price_float,
                    event_timestamp_ts=last_fill_timestamp_ts,
                    event_source_str="synthetic.fill_closeout",
                    message_str="fills imply filled closeout",
                    submission_key_str=broker_order_record_obj.submission_key_str,
                    raw_payload_dict={},
                )
            )
            broker_order_record_obj = broker_order_record_obj.__class__(
                **{
                    **broker_order_record_obj.__dict__,
                    "filled_amount_float": cumulative_filled_amount_float,
                    "remaining_amount_float": 0.0,
                    "avg_fill_price_float": avg_fill_price_float,
                    "status_str": "Filled",
                    "last_status_timestamp_ts": last_fill_timestamp_ts,
                }
            )
        normalized_record_list.append(broker_order_record_obj)

    return normalized_record_list, broker_order_event_list + synthesized_event_list


def _normalize_position_map_dict(
    position_map_dict: dict[str, float],
    tolerance_float: float = 1e-9,
) -> dict[str, float]:
    normalized_position_map_dict: dict[str, float] = {}
    for asset_str, share_float in position_map_dict.items():
        normalized_share_float = float(share_float)
        if abs(normalized_share_float) <= tolerance_float:
            continue
        normalized_position_map_dict[str(asset_str)] = normalized_share_float
    return normalized_position_map_dict


def _build_expected_broker_position_map_dict(
    vplan_obj: VPlan,
    tolerance_float: float = 1e-9,
) -> dict[str, float]:
    expected_broker_position_map_dict = _normalize_position_map_dict(
        vplan_obj.current_broker_position_map,
        tolerance_float=tolerance_float,
    )
    for asset_str, target_share_float in vplan_obj.target_share_map.items():
        normalized_target_share_float = float(target_share_float)
        if abs(normalized_target_share_float) <= tolerance_float:
            expected_broker_position_map_dict.pop(str(asset_str), None)
            continue
        expected_broker_position_map_dict[str(asset_str)] = normalized_target_share_float
    return expected_broker_position_map_dict


def _build_latest_broker_order_row_by_asset_map_dict(
    broker_order_row_dict_list: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    latest_broker_order_row_by_asset_map_dict: dict[str, dict[str, object]] = {}
    for broker_order_row_dict in broker_order_row_dict_list:
        asset_str = str(broker_order_row_dict["asset_str"])
        candidate_timestamp_str = str(
            broker_order_row_dict.get("last_status_timestamp_str")
            or broker_order_row_dict.get("submitted_timestamp_str")
            or ""
        )
        existing_broker_order_row_dict = latest_broker_order_row_by_asset_map_dict.get(asset_str)
        existing_timestamp_str = ""
        if existing_broker_order_row_dict is not None:
            existing_timestamp_str = str(
                existing_broker_order_row_dict.get("last_status_timestamp_str")
                or existing_broker_order_row_dict.get("submitted_timestamp_str")
                or ""
            )
        if existing_broker_order_row_dict is None or candidate_timestamp_str >= existing_timestamp_str:
            latest_broker_order_row_by_asset_map_dict[asset_str] = broker_order_row_dict
    return latest_broker_order_row_by_asset_map_dict


def _build_missing_ack_row_dict_list(
    broker_ack_row_dict_list: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        broker_ack_row_dict
        for broker_ack_row_dict in broker_ack_row_dict_list
        if not bool(broker_ack_row_dict.get("broker_response_ack_bool", False))
    ]


def _signed_execution_direction_float_from_amount(
    amount_float: float | None,
) -> float | None:
    if amount_float is None:
        return None
    normalized_amount_float = float(amount_float)
    if normalized_amount_float > 0.0:
        return 1.0
    if normalized_amount_float < 0.0:
        return -1.0
    return None


def _compute_execution_cost_bps_float(
    execution_price_float: float,
    official_open_price_float: float,
    signed_execution_direction_float: float | None,
) -> float | None:
    if signed_execution_direction_float is None:
        return None
    if official_open_price_float <= 0.0:
        return None
    execution_cost_bps_float = 10000.0 * float(signed_execution_direction_float) * (
        (float(execution_price_float) / float(official_open_price_float)) - 1.0
    )
    return round(execution_cost_bps_float, 10)


def _enrich_fill_row_dict_list(
    fill_row_dict_list: list[dict[str, object]],
) -> list[dict[str, object]]:
    enriched_fill_row_dict_list: list[dict[str, object]] = []
    for fill_row_dict in fill_row_dict_list:
        fill_amount_float = float(fill_row_dict["fill_amount_float"])
        fill_price_float = float(fill_row_dict["fill_price_float"])
        official_open_price_float = fill_row_dict.get("official_open_price_float")
        signed_fill_direction_float = 1.0 if fill_amount_float >= 0.0 else -1.0
        slippage_share_float = None
        slippage_notional_float = None
        fill_slippage_bps_float = None
        if official_open_price_float is not None:
            slippage_share_float = signed_fill_direction_float * (
                fill_price_float - float(official_open_price_float)
            )
            slippage_notional_float = abs(fill_amount_float) * float(slippage_share_float)
            fill_slippage_bps_float = _compute_execution_cost_bps_float(
                execution_price_float=fill_price_float,
                official_open_price_float=float(official_open_price_float),
                signed_execution_direction_float=signed_fill_direction_float,
            )
        enriched_fill_row_dict_list.append(
            {
                **fill_row_dict,
                "signed_fill_direction_float": signed_fill_direction_float,
                "slippage_share_float": slippage_share_float,
                "slippage_notional_float": slippage_notional_float,
                "fill_slippage_bps_float": fill_slippage_bps_float,
            }
        )
    return enriched_fill_row_dict_list


def _build_fill_stat_by_asset_map_dict(
    fill_row_dict_list: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    fill_stat_by_asset_map_dict: dict[str, dict[str, object]] = {}
    for fill_row_dict in fill_row_dict_list:
        asset_str = str(fill_row_dict["asset_str"])
        fill_stat_row_dict = fill_stat_by_asset_map_dict.setdefault(
            asset_str,
            {
                "filled_share_float": 0.0,
                "absolute_fill_share_float": 0.0,
                "weighted_fill_notional_float": 0.0,
                "weighted_avg_fill_price_float": None,
                "official_open_price_float": None,
                "latest_fill_timestamp_str": None,
                "latest_fill_price_float": None,
                "latest_fill_slippage_bps_float": None,
            },
        )
        fill_amount_float = float(fill_row_dict["fill_amount_float"])
        fill_price_float = float(fill_row_dict["fill_price_float"])
        absolute_fill_share_float = abs(fill_amount_float)
        fill_stat_row_dict["filled_share_float"] = float(fill_stat_row_dict["filled_share_float"]) + fill_amount_float
        fill_stat_row_dict["absolute_fill_share_float"] = float(
            fill_stat_row_dict["absolute_fill_share_float"]
        ) + absolute_fill_share_float
        fill_stat_row_dict["weighted_fill_notional_float"] = float(
            fill_stat_row_dict["weighted_fill_notional_float"]
        ) + (absolute_fill_share_float * fill_price_float)
        if float(fill_stat_row_dict["absolute_fill_share_float"]) > 0.0:
            fill_stat_row_dict["weighted_avg_fill_price_float"] = float(
                fill_stat_row_dict["weighted_fill_notional_float"]
            ) / float(fill_stat_row_dict["absolute_fill_share_float"])
        if (
            fill_stat_row_dict["official_open_price_float"] is None
            and fill_row_dict.get("official_open_price_float") is not None
        ):
            fill_stat_row_dict["official_open_price_float"] = float(
                fill_row_dict["official_open_price_float"]
            )
        latest_fill_timestamp_str = fill_stat_row_dict["latest_fill_timestamp_str"]
        candidate_fill_timestamp_str = fill_row_dict.get("fill_timestamp_str")
        if latest_fill_timestamp_str is None or str(candidate_fill_timestamp_str) >= str(latest_fill_timestamp_str):
            fill_stat_row_dict["latest_fill_timestamp_str"] = candidate_fill_timestamp_str
            fill_stat_row_dict["latest_fill_price_float"] = fill_price_float
            fill_stat_row_dict["latest_fill_slippage_bps_float"] = fill_row_dict.get("fill_slippage_bps_float")
    return fill_stat_by_asset_map_dict


def _resolve_avg_execution_quality_dict(
    broker_order_row_dict: dict[str, object] | None,
    fill_stat_row_dict: dict[str, object] | None,
    fallback_direction_amount_float: float | None = None,
) -> dict[str, object]:
    avg_fill_price_float = None
    official_open_price_float = None
    signed_execution_direction_float = None
    if broker_order_row_dict is not None:
        raw_avg_fill_price_float = broker_order_row_dict.get("avg_fill_price_float")
        if raw_avg_fill_price_float is not None:
            avg_fill_price_float = float(raw_avg_fill_price_float)
        signed_execution_direction_float = _signed_execution_direction_float_from_amount(
            broker_order_row_dict.get("amount_float")
        )
    if fill_stat_row_dict is not None:
        if avg_fill_price_float is None and fill_stat_row_dict.get("weighted_avg_fill_price_float") is not None:
            avg_fill_price_float = float(fill_stat_row_dict["weighted_avg_fill_price_float"])
        if fill_stat_row_dict.get("official_open_price_float") is not None:
            official_open_price_float = float(fill_stat_row_dict["official_open_price_float"])
        if signed_execution_direction_float is None:
            signed_execution_direction_float = _signed_execution_direction_float_from_amount(
                fill_stat_row_dict.get("filled_share_float")
            )
    if signed_execution_direction_float is None:
        signed_execution_direction_float = _signed_execution_direction_float_from_amount(
            fallback_direction_amount_float
        )
    avg_slippage_bps_float = None
    if avg_fill_price_float is not None and official_open_price_float is not None:
        avg_slippage_bps_float = _compute_execution_cost_bps_float(
            execution_price_float=avg_fill_price_float,
            official_open_price_float=official_open_price_float,
            signed_execution_direction_float=signed_execution_direction_float,
        )
    return {
        "avg_fill_price_float": avg_fill_price_float,
        "official_open_price_float": official_open_price_float,
        "avg_slippage_bps_float": avg_slippage_bps_float,
    }


def _enrich_broker_order_row_dict_list_for_display(
    broker_order_row_dict_list: list[dict[str, object]],
    fill_row_dict_list: list[dict[str, object]],
) -> list[dict[str, object]]:
    fill_stat_by_asset_map_dict = _build_fill_stat_by_asset_map_dict(fill_row_dict_list)
    enriched_broker_order_row_dict_list: list[dict[str, object]] = []
    for broker_order_row_dict in broker_order_row_dict_list:
        execution_quality_dict = _resolve_avg_execution_quality_dict(
            broker_order_row_dict=broker_order_row_dict,
            fill_stat_row_dict=fill_stat_by_asset_map_dict.get(str(broker_order_row_dict["asset_str"])),
        )
        enriched_broker_order_row_dict_list.append(
            {
                **broker_order_row_dict,
                **execution_quality_dict,
            }
        )
    return enriched_broker_order_row_dict_list


def _build_execution_row_dict_list(
    vplan_obj: VPlan,
    broker_position_map_dict: dict[str, float],
    broker_order_row_dict_list: list[dict[str, object]],
    fill_row_dict_list: list[dict[str, object]],
    tolerance_float: float = 1e-9,
) -> list[dict[str, object]]:
    latest_broker_order_row_by_asset_map_dict = _build_latest_broker_order_row_by_asset_map_dict(
        broker_order_row_dict_list
    )
    fill_stat_by_asset_map_dict = _build_fill_stat_by_asset_map_dict(fill_row_dict_list)
    asset_str_set = {
        str(vplan_row_obj.asset_str)
        for vplan_row_obj in vplan_obj.vplan_row_list
    }
    asset_str_set.update(str(asset_str) for asset_str in vplan_obj.target_share_map)
    asset_str_set.update(str(asset_str) for asset_str in latest_broker_order_row_by_asset_map_dict)
    asset_str_set.update(str(asset_str) for asset_str in fill_stat_by_asset_map_dict)

    execution_row_dict_list: list[dict[str, object]] = []
    for asset_str in sorted(asset_str_set):
        current_share_float = float(vplan_obj.current_broker_position_map.get(asset_str, 0.0))
        target_share_float = float(vplan_obj.target_share_map.get(asset_str, current_share_float))
        broker_share_float = float(broker_position_map_dict.get(asset_str, 0.0))
        residual_share_float = target_share_float - broker_share_float
        live_reference_price_float = float(vplan_obj.live_reference_price_map.get(asset_str, 0.0))
        residual_notional_float = abs(residual_share_float) * live_reference_price_float
        latest_broker_order_row_dict = latest_broker_order_row_by_asset_map_dict.get(asset_str)
        fill_stat_row_dict = fill_stat_by_asset_map_dict.get(
            asset_str,
            {
                "filled_share_float": 0.0,
                "absolute_fill_share_float": 0.0,
                "weighted_fill_notional_float": 0.0,
                "weighted_avg_fill_price_float": None,
                "official_open_price_float": None,
                "latest_fill_timestamp_str": None,
                "latest_fill_price_float": None,
                "latest_fill_slippage_bps_float": None,
            },
        )
        unresolved_bool = abs(residual_share_float) > tolerance_float
        exit_breach_bool = abs(target_share_float) <= tolerance_float and unresolved_bool
        direction_reference_amount_float = float(vplan_obj.order_delta_map.get(asset_str, 0.0))
        if abs(direction_reference_amount_float) <= tolerance_float:
            direction_reference_amount_float = float(fill_stat_row_dict.get("filled_share_float", 0.0))
        execution_quality_dict = _resolve_avg_execution_quality_dict(
            broker_order_row_dict=latest_broker_order_row_dict,
            fill_stat_row_dict=fill_stat_row_dict,
            fallback_direction_amount_float=direction_reference_amount_float,
        )
        execution_row_dict_list.append(
            {
                "asset_str": asset_str,
                "current_share_float": current_share_float,
                "target_share_float": target_share_float,
                "broker_share_float": broker_share_float,
                "residual_share_float": residual_share_float,
                "residual_notional_float": residual_notional_float,
                "open_share_float": abs(residual_share_float),
                "planned_order_delta_share_float": float(vplan_obj.order_delta_map.get(asset_str, 0.0)),
                "filled_share_float": float(fill_stat_row_dict["filled_share_float"]),
                "latest_fill_timestamp_str": fill_stat_row_dict["latest_fill_timestamp_str"],
                "latest_fill_price_float": fill_stat_row_dict["latest_fill_price_float"],
                "latest_fill_slippage_bps_float": fill_stat_row_dict["latest_fill_slippage_bps_float"],
                "latest_broker_order_status_str": (
                    None
                    if latest_broker_order_row_dict is None
                    else latest_broker_order_row_dict["status_str"]
                ),
                "latest_broker_order_timestamp_str": (
                    None
                    if latest_broker_order_row_dict is None
                    else (
                        latest_broker_order_row_dict.get("last_status_timestamp_str")
                        or latest_broker_order_row_dict.get("submitted_timestamp_str")
                    )
                ),
                "live_reference_price_float": live_reference_price_float,
                "unresolved_bool": unresolved_bool,
                "exit_breach_bool": exit_breach_bool,
                "problem_bool": unresolved_bool,
                **execution_quality_dict,
            }
        )
    execution_row_dict_list.sort(
        key=lambda execution_row_dict: (
            0 if bool(execution_row_dict["exit_breach_bool"]) else 1,
            0 if bool(execution_row_dict["problem_bool"]) else 1,
            str(execution_row_dict["asset_str"]),
        )
    )
    return execution_row_dict_list


def _build_exception_row_dict_list(
    execution_row_dict_list: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        execution_row_dict
        for execution_row_dict in execution_row_dict_list
        if bool(execution_row_dict["problem_bool"])
    ]


def _build_terminal_unresolved_row_dict_list(
    execution_row_dict_list: list[dict[str, object]],
) -> list[dict[str, object]]:
    terminal_unresolved_row_dict_list: list[dict[str, object]] = []
    for execution_row_dict in execution_row_dict_list:
        if not bool(execution_row_dict["unresolved_bool"]):
            continue
        latest_broker_order_status_str = execution_row_dict["latest_broker_order_status_str"]
        if latest_broker_order_status_str not in PARKED_TERMINAL_ORDER_STATUS_SET:
            continue
        terminal_unresolved_row_dict_list.append(execution_row_dict)
    return terminal_unresolved_row_dict_list


def is_vplan_execution_exception_parked(
    state_store_obj: LiveStateStore,
    vplan_obj: VPlan,
    tolerance_float: float = 1e-9,
) -> bool:
    if vplan_obj.vplan_id_int is None:
        return False
    if not state_store_obj.has_post_execution_reconciliation_snapshot(int(vplan_obj.vplan_id_int)):
        return False
    latest_broker_snapshot_obj = state_store_obj.get_latest_broker_snapshot_for_account(
        vplan_obj.account_route_str
    )
    broker_position_map_dict = (
        vplan_obj.current_broker_position_map
        if latest_broker_snapshot_obj is None
        else latest_broker_snapshot_obj.position_amount_map
    )
    broker_order_row_dict_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(
        int(vplan_obj.vplan_id_int)
    )
    fill_row_dict_list = _enrich_fill_row_dict_list(
        state_store_obj.get_fill_row_dict_list_for_vplan(int(vplan_obj.vplan_id_int))
    )
    execution_row_dict_list = _build_execution_row_dict_list(
        vplan_obj=vplan_obj,
        broker_position_map_dict=broker_position_map_dict,
        broker_order_row_dict_list=broker_order_row_dict_list,
        fill_row_dict_list=fill_row_dict_list,
        tolerance_float=tolerance_float,
    )
    terminal_unresolved_row_dict_list = _build_terminal_unresolved_row_dict_list(execution_row_dict_list)
    return len(terminal_unresolved_row_dict_list) > 0


def _format_optional_float_str(value_obj: object, precision_int: int = 4) -> str:
    if value_obj is None:
        return "unavailable"
    return f"{float(value_obj):.{precision_int}f}"


def _format_bps_str(value_obj: object, precision_int: int = 1) -> str:
    if value_obj is None:
        return "unavailable"
    return f"{float(value_obj):.{precision_int}f} bps"


def _humanize_reason_code_str(reason_code_str: str | None) -> str:
    reason_map_dict = {
        "active_decision_plan_exists": "active decision plan already exists",
        "account_not_visible": "target account is not visible in the broker session",
        "broker_not_ready": "broker session is not ready for the target account",
        "carry_forward_snapshot_ready": "signal is ready and waiting for the next submission window",
        "completed": "cycle completed",
        "dtb3_stale": "macro data is stale and decision creation was skipped",
        "duplicate_submission_guard": "submission skipped because broker orders already exist for this order plan",
        "eod_snapshot_already_exists": "EOD snapshot already exists for this market date",
        "eod_snapshot_completed": "EOD broker snapshot recorded",
        "env_mode_mismatch": "release mode does not match the current runtime mode",
        "execution_exception_parked": "execution exception is parked for manual review",
        "live_price_snapshot_error": "live price snapshot failed and the order plan was blocked",
        "live_reference_fallback": "using fallback market prices instead of auction prices",
        "missing_broker_response_ack": "broker did not acknowledge all submitted orders",
        "missing_live_price": "missing live reference price for at least one asset",
        "no_due_work": "no due work right now",
        "non_positive_net_liq": "broker net liquidation value is non-positive",
        "not_month_end_session": "waiting for the last session of the month",
        "position_reconciliation_warning": "model and broker positions differ before sizing",
        "ready_to_build_decision_plan": "ready to build the next decision plan",
        "ready_to_build_vplan": "ready to build the order plan",
        "ready_to_reconcile": "ready to reconcile post-execution broker state",
        "session_mode_mismatch": "broker session mode does not match the release mode",
        "snapshot_not_ready_for_session": "snapshot is not ready for this session yet",
        "snapshot_ready": "snapshot is ready for this cycle",
        "snapshot_window_expired": "snapshot submit window already passed for this session",
        "submission_claim_failed": "submit claim failed because the order plan is no longer ready",
        "submission_window_expired": "submit window already passed for this session",
        "submitted": "orders submitted to the broker",
        "unresolved_execution": "submitted execution is not reconciled yet",
        "vplan_ready": "execution plan is ready to submit",
        "waiting_for_post_execution_reconcile": "waiting for the reconcile grace window",
        "waiting_for_submission_window": "waiting for the submission window to open",
    }
    normalized_reason_code_str = str(reason_code_str or "").strip()
    if normalized_reason_code_str in reason_map_dict:
        return reason_map_dict[normalized_reason_code_str]
    if normalized_reason_code_str == "":
        return "unknown"
    return normalized_reason_code_str.replace("_", " ")


def _humanize_next_action_str(next_action_str: str | None) -> str:
    next_action_map_dict = {
        "wait": "no action due",
        "build_decision_plan": "build decision plan",
        "build_vplan": "build order plan",
        "submit_vplan": "submit orders automatically",
        "eod_snapshot": "record EOD broker state",
        "review_vplan": "review orders manually",
        "post_execution_reconcile": "reconcile broker execution",
        "manual_review": "manual review required",
    }
    normalized_next_action_str = str(next_action_str or "").strip()
    if normalized_next_action_str in next_action_map_dict:
        return next_action_map_dict[normalized_next_action_str]
    if normalized_next_action_str == "":
        return "unknown"
    return normalized_next_action_str.replace("_", " ")


def _humanize_status_pair_str(
    decision_status_str: str | None,
    vplan_status_str: str | None,
) -> str:
    if vplan_status_str == "ready":
        return "order plan ready"
    if vplan_status_str == "submitting":
        return "orders are being submitted"
    if vplan_status_str == "submitted":
        return "orders submitted"
    if vplan_status_str == "completed":
        return "cycle completed"
    if vplan_status_str == "expired":
        return "order plan expired"
    if decision_status_str == "planned":
        return "decision planned"
    if decision_status_str == "vplan_ready":
        return "order plan ready"
    if decision_status_str == "blocked":
        return "decision blocked"
    if decision_status_str == "expired":
        return "decision expired"
    if decision_status_str == "completed":
        return "cycle completed"
    if decision_status_str is None and vplan_status_str is None:
        return "waiting for next signal"
    return (
        f"decision={decision_status_str or 'none'} | "
        f"order_plan={vplan_status_str or 'none'}"
    )


def _render_tick_detail_str(detail_dict: dict[str, object]) -> str:
    if not bool(detail_dict.get("lease_acquired_bool", False)):
        return "\n".join(
            [
                "Tick Result",
                "- Another tick is already running.",
                "- No sleeve needed action in this run.",
                "",
                "Next",
                "- Wait for the current tick to finish, then run status.",
            ]
        )

    def _count_line_str(verb_str: str, count_int: int, singular_label_str: str) -> str:
        return f"- {verb_str} {count_int} {_pluralize_label_str(count_int, singular_label_str)}."

    def _append_count_map_line_list(line_list: list[str], count_map_dict: dict[object, object]) -> None:
        for code_str, count_obj in sorted(count_map_dict.items()):
            line_list.append(f"- {_humanize_reason_code_str(str(code_str))}: {int(count_obj)}")

    eod_snapshot_count_int = int(detail_dict.get("eod_snapshot_count_int", 0))
    created_decision_plan_count_int = int(detail_dict.get("created_decision_plan_count_int", 0))
    skipped_decision_plan_count_int = int(detail_dict.get("skipped_decision_plan_count_int", 0))
    expired_decision_plan_count_int = int(detail_dict.get("expired_decision_plan_count_int", 0))
    created_vplan_count_int = int(detail_dict.get("created_vplan_count_int", 0))
    submitted_vplan_count_int = int(detail_dict.get("submitted_vplan_count_int", 0))
    blocked_action_count_int = int(detail_dict.get("blocked_action_count_int", 0))
    completed_vplan_count_int = int(detail_dict.get("completed_vplan_count_int", 0))
    warning_count_map_dict = dict(detail_dict.get("warning_count_map_dict", {}))
    warning_action_count_int = sum(int(warning_count_int) for warning_count_int in warning_count_map_dict.values())
    reason_count_map_dict = dict(detail_dict.get("reason_count_map_dict", {}))
    work_done_count_int = (
        eod_snapshot_count_int
        + created_decision_plan_count_int
        + skipped_decision_plan_count_int
        + expired_decision_plan_count_int
        + created_vplan_count_int
        + submitted_vplan_count_int
        + blocked_action_count_int
        + completed_vplan_count_int
        + warning_action_count_int
    )

    line_list = ["Tick Result"]
    if work_done_count_int == 0:
        line_list.append("- No sleeve needed action in this run.")
    line_list.extend(
        [
            _count_line_str("Recorded", eod_snapshot_count_int, "EOD snapshot"),
            _count_line_str("Built", created_decision_plan_count_int, "decision plan"),
            _count_line_str("Built", created_vplan_count_int, "order plan"),
            _count_line_str("Submitted", submitted_vplan_count_int, "order plan"),
            _count_line_str("Completed", completed_vplan_count_int, "reconciliation"),
            f"- Blocked actions: {blocked_action_count_int}",
            f"- Warnings: {warning_action_count_int}",
        ]
    )
    if skipped_decision_plan_count_int > 0:
        line_list.append(
            _count_line_str("Skipped duplicate", skipped_decision_plan_count_int, "decision plan")
        )
    if expired_decision_plan_count_int > 0:
        line_list.append(
            _count_line_str("Expired stale", expired_decision_plan_count_int, "decision plan")
        )

    if len(reason_count_map_dict) > 0:
        line_list.extend(["", "Reasons"])
        _append_count_map_line_list(line_list, reason_count_map_dict)
    if len(warning_count_map_dict) > 0:
        line_list.extend(["", "Warnings"])
        _append_count_map_line_list(line_list, warning_count_map_dict)

    next_line_str = "Run status to see the current sleeve state."
    if created_vplan_count_int > 0 and submitted_vplan_count_int == 0:
        next_line_str = "Run show_vplan or status before submitting."
    if submitted_vplan_count_int > 0:
        next_line_str = "Run status to inspect broker ACK and reconcile state."
    if blocked_action_count_int > 0 or warning_action_count_int > 0:
        next_line_str = "Run status for details before trusting the deployment."

    line_list.extend(["", "Next", f"- {next_line_str}"])
    return "\n".join(line_list)


def _render_eod_snapshot_detail_str(detail_dict: dict[str, object]) -> str:
    reason_count_map_dict = dict(detail_dict.get("reason_count_map_dict", {}))
    line_list = [
        "EOD Snapshot",
        f"- Recorded snapshots: {int(detail_dict.get('eod_snapshot_count_int', 0))}",
        f"- Skipped snapshots: {int(detail_dict.get('skipped_snapshot_count_int', 0))}",
        f"- Blocked actions: {int(detail_dict.get('blocked_action_count_int', 0))}",
    ]
    if len(reason_count_map_dict) > 0:
        line_list.append("")
        line_list.append("Reasons")
        for code_str, count_obj in sorted(reason_count_map_dict.items()):
            line_list.append(f"- {_humanize_reason_code_str(str(code_str))}: {int(count_obj)}")
    return "\n".join(line_list)


def _render_status_detail_str(detail_dict: dict[str, object]) -> str:
    pod_status_dict_list = list(detail_dict.get("pod_status_dict_list", []))
    line_list = [
        "Status",
        f"- Deployment: {detail_dict.get('mode_str', 'paper')}",
        f"- Client: {detail_dict.get('client_user_id_str', 'unknown')}",
        f"- Enabled sleeves: {int(detail_dict.get('enabled_pod_count_int', 0))}",
        f"- As of: {detail_dict.get('as_of_timestamp_str', 'unknown')}",
    ]
    deployment_warning_str = str(detail_dict.get("deployment_warning_str") or "")
    if deployment_warning_str != "":
        line_list.append(f"- Warning: {deployment_warning_str}")
    if len(pod_status_dict_list) == 0:
        line_list.append("- No enabled sleeves were found for this deployment.")
        return "\n".join(line_list)

    for pod_status_dict in pod_status_dict_list:
        exception_row_dict_list = list(pod_status_dict.get("exception_row_dict_list", []))
        missing_ack_row_dict_list = list(pod_status_dict.get("missing_ack_row_dict_list", []))
        latest_decision_plan_status_str = pod_status_dict.get("latest_decision_plan_status_str")
        latest_vplan_status_str = pod_status_dict.get("latest_vplan_status_str")
        submit_ack_status_str = pod_status_dict.get("submit_ack_status_str")
        missing_ack_count_int = int(pod_status_dict.get("missing_ack_count_int", 0))
        next_action_obj = pod_status_dict.get("next_action_str")
        reason_code_obj = pod_status_dict.get("reason_code_str")
        line_list.extend(
            [
                "",
                f"{pod_status_dict['pod_id_str']} sleeve",
                f"- Release: {pod_status_dict['release_id_str']}",
                f"- Account: {pod_status_dict['account_route_str']}",
                f"- Auto-submit: {'on' if bool(pod_status_dict.get('auto_submit_enabled_bool')) else 'off'}",
                f"- State: {_humanize_status_pair_str(latest_decision_plan_status_str, latest_vplan_status_str)}",
                f"- Next: {_humanize_next_action_str(None if next_action_obj is None else str(next_action_obj))}",
                f"- Reason: {_humanize_reason_code_str(None if reason_code_obj is None else str(reason_code_obj))}",
            ]
        )
        if pod_status_dict.get("latest_signal_timestamp_str") is not None:
            line_list.append(f"- Latest signal: {pod_status_dict['latest_signal_timestamp_str']}")
        if pod_status_dict.get("latest_submission_timestamp_str") is not None:
            line_list.append(f"- Planned submit: {pod_status_dict['latest_submission_timestamp_str']}")
        if pod_status_dict.get("latest_broker_snapshot_timestamp_str") is not None:
            line_list.append(
                f"- Latest broker snapshot: {pod_status_dict['latest_broker_snapshot_timestamp_str']}"
            )
        if pod_status_dict.get("latest_fill_timestamp_str") is not None:
            line_list.append(f"- Latest fill: {pod_status_dict['latest_fill_timestamp_str']}")
        if (
            submit_ack_status_str not in (None, "not_checked", "complete")
            or missing_ack_count_int > 0
        ):
            line_list.append(
                "- Broker ACK: "
                f"status={submit_ack_status_str or 'none'} | "
                f"coverage={_format_optional_float_str(pod_status_dict.get('ack_coverage_ratio_float'))} | "
                f"missing={missing_ack_count_int}"
            )
        for missing_ack_row_dict in missing_ack_row_dict_list:
            line_list.append(
                "- CRITICAL ACK: "
                f"{missing_ack_row_dict['asset_str']} | "
                f"request={missing_ack_row_dict['order_request_key_str']} | "
                f"source={missing_ack_row_dict['ack_source_str']}"
            )
        if len(exception_row_dict_list) > 0:
            line_list.append(f"- Exceptions: {len(exception_row_dict_list)}")
        for exception_row_dict in exception_row_dict_list:
            label_str = "CRITICAL EXIT" if bool(exception_row_dict["exit_breach_bool"]) else "Issue"
            line_list.append(
                f"- {label_str}: "
                f"{exception_row_dict['asset_str']} | "
                f"target={_format_optional_float_str(exception_row_dict['target_share_float'])} | "
                f"broker={_format_optional_float_str(exception_row_dict['broker_share_float'])} | "
                f"residual={_format_optional_float_str(exception_row_dict['residual_share_float'])} | "
                f"filled={_format_optional_float_str(exception_row_dict['filled_share_float'])} | "
                f"latest_order={exception_row_dict['latest_broker_order_status_str'] or 'none'} | "
                f"open={_format_optional_float_str(exception_row_dict['open_share_float'])}"
            )
    return "\n".join(line_list)


def _render_vplan_detail_str(detail_dict: dict[str, object]) -> str:
    vplan_dict_list = list(detail_dict.get("vplan_dict_list", []))
    line_list = ["VPlan"]
    if len(vplan_dict_list) == 0:
        line_list.append("- No VPlan was found.")
        return "\n".join(line_list)

    for vplan_dict in vplan_dict_list:
        line_list.extend(
            [
                "",
                f"Pod: {vplan_dict['pod_id_str']}",
                f"- VPlan id: {vplan_dict['vplan_id_int']}",
                f"- DecisionPlan id: {vplan_dict['decision_plan_id_int']}",
                f"- Status: {vplan_dict['status_str']}",
                "- Submit ACK: "
                f"status={vplan_dict.get('submit_ack_status_str') or 'none'} | "
                f"coverage={_format_optional_float_str(vplan_dict.get('ack_coverage_ratio_float'))} | "
                f"missing={int(vplan_dict.get('missing_ack_count_int', 0))}",
                f"- NetLiq: {vplan_dict['net_liq_float']}",
                f"- Pod budget: {vplan_dict['pod_budget_float']}",
                f"- Price source: {vplan_dict['live_price_source_str']}",
            ]
        )
        if str(vplan_dict.get("display_mode_str", "planned")) == "planned":
            warning_row_dict_list = list(vplan_dict.get("warning_row_dict_list", []))
            if len(warning_row_dict_list) > 0:
                line_list.append(f"- Position warnings: {len(warning_row_dict_list)}")
            for vplan_row_dict in vplan_dict["vplan_row_dict_list"]:
                line_list.append(
                    "- Plan: "
                    f"{vplan_row_dict['asset_str']} | "
                    f"current={_format_optional_float_str(vplan_row_dict['current_share_float'])} | "
                    f"target={_format_optional_float_str(vplan_row_dict['target_share_float'])} | "
                    f"delta={_format_optional_float_str(vplan_row_dict['order_delta_share_float'])} | "
                    f"ref_price={_format_optional_float_str(vplan_row_dict['live_reference_price_float'])} | "
                    f"ref_source={vplan_row_dict['live_reference_source_str']} | "
                    f"est_notional={_format_optional_float_str(vplan_row_dict['estimated_target_notional_float'])}"
                )
            for warning_row_dict in warning_row_dict_list:
                line_list.append(
                    "- Warning: "
                    f"{warning_row_dict['asset_str']} | "
                    f"decision_base={_format_optional_float_str(warning_row_dict['decision_base_share_float'])} | "
                    f"current={_format_optional_float_str(warning_row_dict['current_share_float'])} | "
                    f"drift={_format_optional_float_str(warning_row_dict['drift_share_float'])}"
                )
            continue

        exception_row_dict_list = list(vplan_dict.get("exception_row_dict_list", []))
        missing_ack_row_dict_list = list(vplan_dict.get("missing_ack_row_dict_list", []))
        execution_row_dict_list = list(vplan_dict.get("execution_row_dict_list", []))
        fill_row_dict_list = list(vplan_dict.get("fill_row_dict_list", []))
        for missing_ack_row_dict in missing_ack_row_dict_list:
            line_list.append(
                "- CRITICAL ACK: "
                f"{missing_ack_row_dict['asset_str']} | "
                f"request={missing_ack_row_dict['order_request_key_str']} | "
                f"source={missing_ack_row_dict['ack_source_str']}"
            )
        line_list.append(f"- Exceptions: {len(exception_row_dict_list)}")
        if len(exception_row_dict_list) == 0:
            line_list.append("- OK: no unresolved broker-vs-target residuals.")
        for exception_row_dict in exception_row_dict_list:
            label_str = "CRITICAL EXIT" if bool(exception_row_dict["exit_breach_bool"]) else "Issue"
            line_list.append(
                f"- {label_str}: "
                f"{exception_row_dict['asset_str']} | "
                f"target={_format_optional_float_str(exception_row_dict['target_share_float'])} | "
                f"broker={_format_optional_float_str(exception_row_dict['broker_share_float'])} | "
                f"residual={_format_optional_float_str(exception_row_dict['residual_share_float'])} | "
                f"filled={_format_optional_float_str(exception_row_dict['filled_share_float'])} | "
                f"latest_order={exception_row_dict['latest_broker_order_status_str'] or 'none'} | "
                f"open={_format_optional_float_str(exception_row_dict['open_share_float'])}"
            )
        for execution_row_dict in execution_row_dict_list:
            line_list.append(
                "- Row: "
                f"{execution_row_dict['asset_str']} | "
                f"target={_format_optional_float_str(execution_row_dict['target_share_float'])} | "
                f"broker={_format_optional_float_str(execution_row_dict['broker_share_float'])} | "
                f"residual={_format_optional_float_str(execution_row_dict['residual_share_float'])} | "
                f"filled={_format_optional_float_str(execution_row_dict['filled_share_float'])} | "
                f"avg_fill={_format_optional_float_str(execution_row_dict['avg_fill_price_float'])} | "
                f"open={_format_optional_float_str(execution_row_dict['official_open_price_float'])} | "
                f"avg_bps={_format_bps_str(execution_row_dict['avg_slippage_bps_float'])} | "
                f"latest_order={execution_row_dict['latest_broker_order_status_str'] or 'none'}"
            )
        for broker_order_row_dict in vplan_dict.get("broker_order_row_dict_list", []):
            line_list.append(
                "- Order: "
                f"{broker_order_row_dict['asset_str']} | "
                f"requested={_format_optional_float_str(broker_order_row_dict['amount_float'])} | "
                f"filled={_format_optional_float_str(broker_order_row_dict['filled_amount_float'])} | "
                f"remaining={_format_optional_float_str(broker_order_row_dict['remaining_amount_float'])} | "
                f"avg_fill={_format_optional_float_str(broker_order_row_dict['avg_fill_price_float'])} | "
                f"open={_format_optional_float_str(broker_order_row_dict['official_open_price_float'])} | "
                f"avg_bps={_format_bps_str(broker_order_row_dict['avg_slippage_bps_float'])} | "
                f"status={broker_order_row_dict['status_str']} | "
                f"time={broker_order_row_dict['last_status_timestamp_str'] or broker_order_row_dict['submitted_timestamp_str']}"
            )
        for fill_row_dict in fill_row_dict_list:
            line_list.append(
                "- Fill: "
                f"{fill_row_dict['asset_str']} | "
                f"shares={_format_optional_float_str(fill_row_dict['fill_amount_float'])} | "
                f"fill={_format_optional_float_str(fill_row_dict['fill_price_float'])} | "
                f"official_open={_format_optional_float_str(fill_row_dict['official_open_price_float'])} | "
                f"slippage_bps={_format_bps_str(fill_row_dict['fill_slippage_bps_float'])} | "
                f"slippage/share={_format_optional_float_str(fill_row_dict['slippage_share_float'])} | "
                f"slippage_notional={_format_optional_float_str(fill_row_dict['slippage_notional_float'])} | "
                f"time={fill_row_dict['fill_timestamp_str']}"
            )
    return "\n".join(line_list)


def _render_execution_report_detail_str(detail_dict: dict[str, object]) -> str:
    execution_report_dict_list = list(detail_dict.get("execution_report_dict_list", []))
    line_list = ["Execution Report"]
    if len(execution_report_dict_list) == 0:
        line_list.append("- No fills were found yet.")
        return "\n".join(line_list)

    for execution_report_dict in execution_report_dict_list:
        line_list.extend(
            [
                "",
                f"Pod: {execution_report_dict['pod_id_str']}",
                f"- VPlan id: {execution_report_dict['latest_vplan_id_int']}",
                f"- Fill count: {execution_report_dict['fill_count_int']}",
                f"- Fills with official open: {execution_report_dict['fill_with_open_count_int']}",
                f"- Aggregate slippage notional: {_format_optional_float_str(execution_report_dict['aggregate_slippage_notional_float'])}",
            ]
        )
        for fill_row_dict in execution_report_dict["fill_row_dict_list"]:
            line_list.append(
                "- Fill: "
                f"{fill_row_dict['asset_str']} | "
                f"shares={_format_optional_float_str(fill_row_dict['fill_amount_float'])} | "
                f"fill={_format_optional_float_str(fill_row_dict['fill_price_float'])} | "
                f"official_open={_format_optional_float_str(fill_row_dict['official_open_price_float'])} | "
                f"slippage_bps={_format_bps_str(fill_row_dict['fill_slippage_bps_float'])} | "
                f"slippage/share={_format_optional_float_str(fill_row_dict['slippage_share_float'])} | "
                f"slippage_notional={_format_optional_float_str(fill_row_dict['slippage_notional_float'])} | "
                f"open_source={fill_row_dict['open_price_source_str'] or 'unavailable'} | "
                f"time={fill_row_dict['fill_timestamp_str']}"
            )
    return "\n".join(line_list)


def _render_compare_reference_detail_str(detail_dict: dict[str, object]) -> str:
    compare_report_dict_list = list(detail_dict.get("compare_report_dict_list", []))
    line_list = ["Reference Compare"]
    if len(compare_report_dict_list) == 0:
        line_list.append("- No comparable VPlans were found.")
        return "\n".join(line_list)

    for compare_report_dict in compare_report_dict_list:
        line_list.extend(
            [
                "",
                f"Pod: {compare_report_dict['pod_id_str']}",
                f"- VPlan id: {compare_report_dict['vplan_id_int']}",
                f"- Status: {compare_report_dict.get('status_str', 'unknown')}",
                f"- Deployment start: {compare_report_dict.get('deployment_start_date_str')}",
                f"- Target session: {compare_report_dict['target_session_date_str']}",
                f"- Actual equity: {_format_optional_float_str(compare_report_dict.get('actual_equity_float'))}",
                f"- Actual equity source: {compare_report_dict.get('actual_equity_source_str') or 'unknown'}",
                f"- Backtest equity: {_format_optional_float_str(compare_report_dict.get('backtest_equity_float'))}",
                f"- Equity tracking error: {_format_optional_float_str(compare_report_dict.get('equity_tracking_error_float'))}",
                f"- Actual cash: {_format_optional_float_str(compare_report_dict.get('actual_cash_float'))}",
                f"- Backtest cash: {_format_optional_float_str(compare_report_dict.get('backtest_cash_float'))}",
                f"- Cash diff: {_format_optional_float_str(compare_report_dict.get('cash_diff_float'))}",
            ]
        )
        artifact_path_dict = dict(compare_report_dict.get("artifact_path_dict", {}))
        if artifact_path_dict.get("html_path_str") is not None:
            line_list.append(f"- HTML report: {artifact_path_dict['html_path_str']}")
        if compare_report_dict.get("reference_strategy_pickle_path_str") is not None:
            line_list.append(f"- Reference pickle: {compare_report_dict['reference_strategy_pickle_path_str']}")
        for compare_row_dict in compare_report_dict["compare_row_dict_list"]:
            line_list.append(
                "- Row: "
                f"{compare_row_dict['asset_str']} | "
                f"live_w={_format_optional_float_str(compare_row_dict.get('live_target_weight_float'))} | "
                f"ref_w={_format_optional_float_str(compare_row_dict.get('reference_realized_weight_float'))} | "
                f"w_diff={_format_optional_float_str(compare_row_dict.get('weight_diff_float'))} | "
                f"planned={_format_optional_float_str(compare_row_dict['planned_order_delta_share_float'])} | "
                f"filled={_format_optional_float_str(compare_row_dict['filled_share_float'])} | "
                f"quantity_diff={_format_optional_float_str(compare_row_dict['quantity_diff_float'])} | "
                f"target_pos={_format_optional_float_str(compare_row_dict['target_position_float'])} | "
                f"actual_pos={_format_optional_float_str(compare_row_dict['actual_position_float'])} | "
                f"ref_pos={_format_optional_float_str(compare_row_dict.get('reference_position_float'))} | "
                f"position_diff={_format_optional_float_str(compare_row_dict['position_diff_float'])} | "
                f"ref_position_diff={_format_optional_float_str(compare_row_dict.get('reference_position_diff_float'))} | "
                f"avg_fill={_format_optional_float_str(compare_row_dict['avg_fill_price_float'])} | "
                f"reference={_format_optional_float_str(compare_row_dict['reference_price_float'])} | "
                f"slippage_bps={_format_bps_str(compare_row_dict['fill_slippage_bps_float'])} | "
                f"bt_qty_diff={_format_optional_float_str(compare_row_dict.get('backtest_quantity_diff_float'))} | "
                f"bt_px_diff={_format_optional_float_str(compare_row_dict.get('backtest_fill_price_diff_float'))}"
            )
    return "\n".join(line_list)


def _render_preflight_contract_detail_str(detail_dict: dict[str, object]) -> str:
    contract_report_dict_list = list(detail_dict.get("contract_report_dict_list", []))
    line_list = [
        "Preflight Contract",
        f"- Enabled releases: {int(detail_dict.get('enabled_release_count_int', 0))}",
        f"- Passed: {int(detail_dict.get('passed_release_count_int', 0))}",
        f"- Failed: {int(detail_dict.get('failed_release_count_int', 0))}",
    ]
    if len(contract_report_dict_list) == 0:
        line_list.append("- No enabled releases were found.")
        return "\n".join(line_list)

    for contract_report_dict in contract_report_dict_list:
        line_list.extend(
            [
                "",
                f"Release: {contract_report_dict['release_id_str']}",
                f"- Pod: {contract_report_dict['pod_id_str']}",
                f"- Strategy: {contract_report_dict['strategy_import_str']}",
                f"- Decision book: {contract_report_dict['decision_book_type_str']}",
                f"- Contract status: {contract_report_dict['contract_status_str']}",
                f"- Accepted shapes: {int(contract_report_dict['accepted_shape_count_int'])}",
                f"- Unsupported shapes: {int(contract_report_dict['unsupported_shape_count_int'])}",
            ]
        )
        unsupported_shape_example_dict_list = list(
            contract_report_dict.get("unsupported_shape_example_dict_list", [])
        )
        if len(unsupported_shape_example_dict_list) > 0:
            line_list.append("- Unsupported shape examples:")
            for unsupported_shape_dict in unsupported_shape_example_dict_list:
                line_list.append(
                    "  - "
                    f"asset={unsupported_shape_dict['asset_str']} | "
                    f"class={unsupported_shape_dict['order_class_str']} | "
                    f"unit={unsupported_shape_dict['unit_str']} | "
                    f"target={unsupported_shape_dict['target_bool']} | "
                    f"amount={unsupported_shape_dict['amount_float']} | "
                    f"trade_id={unsupported_shape_dict['trade_id_int']}"
                )
        if contract_report_dict.get("error_str") is not None:
            line_list.append(f"- Error: {contract_report_dict['error_str']}")

    return "\n".join(line_list)


def _render_v1_cutover_detail_str(detail_dict: dict[str, object]) -> str:
    table_export_dict_list = list(detail_dict.get("table_export_dict_list", []))
    line_list = [
        "V1 Schema Cutover",
        f"- DB path: {detail_dict['db_path_str']}",
        f"- Archive path: {detail_dict['archive_dir_path_str']}",
        f"- Exported tables: {int(detail_dict.get('exported_table_count_int', 0))}",
        f"- Dropped tables: {int(detail_dict.get('dropped_table_count_int', 0))}",
    ]
    remaining_v1_table_name_list = list(detail_dict.get("remaining_v1_table_name_list", []))
    if len(remaining_v1_table_name_list) > 0:
        line_list.append(f"- Remaining v1 tables: {', '.join(remaining_v1_table_name_list)}")
    else:
        line_list.append("- Remaining v1 tables: none")
    if len(table_export_dict_list) == 0:
        line_list.append("- No v1 tables were present.")
        return "\n".join(line_list)

    for table_export_dict in table_export_dict_list:
        line_list.extend(
            [
                "",
                f"Table: {table_export_dict['table_name_str']}",
                f"- Source rows: {int(table_export_dict['source_row_count_int'])}",
                f"- Exported rows: {int(table_export_dict['exported_row_count_int'])}",
                f"- Archive file: {table_export_dict['archive_file_path_str']}",
            ]
        )

    return "\n".join(line_list)


def _render_command_output_str(command_name_str: str, detail_dict: dict[str, object]) -> str:
    if command_name_str == "tick":
        return _render_tick_detail_str(detail_dict)
    if command_name_str == "status":
        return _render_status_detail_str(detail_dict)
    if command_name_str == "show_vplan":
        return _render_vplan_detail_str(detail_dict)
    if command_name_str == "execution_report":
        return _render_execution_report_detail_str(detail_dict)
    if command_name_str == "compare_reference":
        return _render_compare_reference_detail_str(detail_dict)
    if command_name_str == "eod_snapshot":
        return _render_eod_snapshot_detail_str(detail_dict)
    if command_name_str == "preflight_contract":
        return _render_preflight_contract_detail_str(detail_dict)
    if command_name_str == "cutover_v1_schema":
        return _render_v1_cutover_detail_str(detail_dict)
    return json.dumps(detail_dict, indent=2, sort_keys=True)


def expire_stale_decision_plans(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str | None = None,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
    pod_id_str: str | None = None,
) -> dict[str, object]:
    del releases_root_path_str
    expired_decision_plan_count_int = 0
    reason_counter_obj: Counter[str] = Counter()
    for decision_plan_obj in state_store_obj.get_expirable_decision_plan_list(as_of_ts):
        if pod_id_str is not None and decision_plan_obj.pod_id_str != pod_id_str:
            continue
        release_obj = state_store_obj.get_release_by_id(decision_plan_obj.release_id_str)
        if env_mode_str is not None and release_obj.mode_str != env_mode_str:
            continue
        if not scheduler_utils.is_execution_window_expired_bool(
            decision_plan_obj.execution_policy_str,
            decision_plan_obj.target_execution_timestamp_ts,
            as_of_ts,
        ):
            continue
        state_store_obj.mark_decision_plan_status(int(decision_plan_obj.decision_plan_id_int or 0), "expired")
        latest_vplan_obj = state_store_obj.get_latest_vplan_for_decision(int(decision_plan_obj.decision_plan_id_int or 0))
        if latest_vplan_obj is not None and latest_vplan_obj.status_str == "ready":
            state_store_obj.mark_vplan_status(int(latest_vplan_obj.vplan_id_int or 0), "expired")
        expired_decision_plan_count_int += 1
        reason_counter_obj["submission_window_expired"] += 1
        log_event(
            "decision_plan_expired",
            _build_decision_plan_log_payload_dict(
                release_obj,
                decision_plan_obj,
                as_of_ts,
                {"reason_code_str": "submission_window_expired"},
            ),
            log_path_str=log_path_str,
        )
    return {
        "expired_decision_plan_count_int": expired_decision_plan_count_int,
        "reason_count_map_dict": dict(reason_counter_obj),
    }


def build_decision_plans(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str = "paper",
    log_path_str: str = DEFAULT_LOG_PATH_STR,
    pod_id_str: str | None = None,
) -> dict[str, object]:
    release_list = _load_release_list_validate_and_sync(
        releases_root_path_str,
        state_store_obj,
        env_mode_str,
        pod_id_str=pod_id_str,
    )
    selected_release_list = select_enabled_release_list_for_mode(release_list, env_mode_str)
    due_release_list = scheduler_utils.select_due_release_list(selected_release_list, as_of_ts)
    created_decision_plan_count_int = 0
    skipped_decision_plan_count_int = 0
    reason_counter_obj: Counter[str] = Counter()

    for release_obj in due_release_list:
        pod_state_obj = _get_model_state_or_default(release_obj, state_store_obj, as_of_ts)
        try:
            decision_plan_obj = strategy_host.build_decision_plan_for_release(
                release_obj=release_obj,
                as_of_ts=as_of_ts,
                pod_state_obj=pod_state_obj,
            )
        except FredSeriesLoadError as exception_obj:
            skipped_decision_plan_count_int += 1
            reason_counter_obj[str(exception_obj.reason_code_str)] += 1
            log_event(
                "build_decision_plan_data_dependency_error",
                _build_fred_series_error_log_payload_dict(
                    release_obj=release_obj,
                    as_of_ts=as_of_ts,
                    exception_obj=exception_obj,
                ),
                log_path_str=log_path_str,
            )
            continue

        if state_store_obj.has_active_decision_plan(
            pod_id_str=release_obj.pod_id_str,
            signal_timestamp_ts=decision_plan_obj.signal_timestamp_ts,
            execution_policy_str=decision_plan_obj.execution_policy_str,
        ):
            skipped_decision_plan_count_int += 1
            reason_counter_obj["active_decision_plan_exists"] += 1
            continue

        inserted_decision_plan_obj = state_store_obj.insert_decision_plan(decision_plan_obj)
        created_decision_plan_count_int += 1
        log_event(
            "build_decision_plan_created",
            _build_decision_plan_log_payload_dict(
                release_obj,
                inserted_decision_plan_obj,
                as_of_ts,
                {"reason_code_str": "snapshot_ready"},
            ),
            log_path_str=log_path_str,
        )

    return {
        "created_decision_plan_count_int": created_decision_plan_count_int,
        "skipped_decision_plan_count_int": skipped_decision_plan_count_int,
        "reason_count_map_dict": dict(reason_counter_obj),
    }


def build_vplans(
    state_store_obj: LiveStateStore,
    broker_adapter_obj: BrokerAdapter | None,
    as_of_ts: datetime,
    env_mode_str: str,
    releases_root_path_str: str | None = None,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
    broker_adapter_resolver_obj: BrokerAdapterResolver | None = None,
    broker_host_str: str | None = None,
    broker_port_int: int | None = None,
    broker_client_id_int: int | None = None,
    broker_timeout_seconds_float: float | None = None,
    adapter_factory_func: Callable[[str, int, int, float], BrokerAdapter] | None = None,
    pod_id_str: str | None = None,
) -> dict[str, object]:
    created_vplan_count_int = 0
    blocked_action_count_int = 0
    warning_counter_obj: Counter[str] = Counter()
    reason_counter_obj: Counter[str] = Counter()
    if releases_root_path_str is None:
        _validate_state_store_releases_for_mutation(state_store_obj, env_mode_str, pod_id_str=pod_id_str)
    else:
        _load_release_list_validate_and_sync(
            releases_root_path_str,
            state_store_obj,
            env_mode_str,
            pod_id_str=pod_id_str,
        )
    broker_adapter_resolver_obj = _coerce_broker_adapter_resolver_obj(
        broker_adapter_obj=broker_adapter_obj,
        broker_adapter_resolver_obj=broker_adapter_resolver_obj,
        broker_host_str=broker_host_str,
        broker_port_int=broker_port_int,
        broker_client_id_int=broker_client_id_int,
        broker_timeout_seconds_float=broker_timeout_seconds_float,
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
        adapter_factory_func=adapter_factory_func,
    )

    for decision_plan_obj in state_store_obj.get_due_decision_plan_list(as_of_ts):
        if pod_id_str is not None and decision_plan_obj.pod_id_str != pod_id_str:
            continue
        release_obj = state_store_obj.get_release_by_id(decision_plan_obj.release_id_str)
        latest_vplan_obj = state_store_obj.get_latest_vplan_for_decision(int(decision_plan_obj.decision_plan_id_int or 0))
        if latest_vplan_obj is not None and latest_vplan_obj.status_str in ("ready", "submitted", "completed"):
            continue
        if release_obj.mode_str != env_mode_str:
            reason_counter_obj["env_mode_mismatch"] += 1
            continue
        broker_adapter_obj = broker_adapter_resolver_obj.get_adapter(release_obj)
        visible_account_route_set = broker_adapter_obj.get_visible_account_route_set()
        if scheduler_utils.is_execution_window_expired_bool(
            decision_plan_obj.execution_policy_str,
            decision_plan_obj.target_execution_timestamp_ts,
            as_of_ts,
        ):
            blocked_action_count_int += 1
            reason_counter_obj["submission_window_expired"] += 1
            state_store_obj.mark_decision_plan_status(int(decision_plan_obj.decision_plan_id_int or 0), "expired")
            continue
        if not broker_adapter_obj.is_session_ready(decision_plan_obj.account_route_str):
            blocked_action_count_int += 1
            reason_counter_obj["broker_not_ready"] += 1
            continue
        if (
            visible_account_route_set is not None
            and decision_plan_obj.account_route_str not in visible_account_route_set
        ):
            blocked_action_count_int += 1
            reason_counter_obj["account_not_visible"] += 1
            state_store_obj.mark_decision_plan_status(int(decision_plan_obj.decision_plan_id_int or 0), "blocked")
            continue
        session_mode_str = broker_adapter_obj.get_session_mode_str(decision_plan_obj.account_route_str)
        if session_mode_str is not None and session_mode_str != release_obj.mode_str:
            blocked_action_count_int += 1
            reason_counter_obj["session_mode_mismatch"] += 1
            state_store_obj.mark_decision_plan_status(int(decision_plan_obj.decision_plan_id_int or 0), "blocked")
            continue

        broker_snapshot_obj = broker_adapter_obj.get_account_snapshot(decision_plan_obj.account_route_str)
        state_store_obj.upsert_broker_snapshot_cache(broker_snapshot_obj)
        reconciliation_result_obj = reconcile_account_state(
            model_position_map=decision_plan_obj.decision_base_position_map,
            model_cash_float=0.0,
            broker_snapshot_obj=broker_snapshot_obj,
        )
        state_store_obj.insert_vplan_reconciliation_snapshot(
            pod_id_str=decision_plan_obj.pod_id_str,
            decision_plan_id_int=decision_plan_obj.decision_plan_id_int,
            vplan_id_int=None,
            stage_str="pre_vplan",
            reconciliation_result_obj=reconciliation_result_obj,
        )
        if not reconciliation_result_obj.passed_bool:
            warning_counter_obj["position_reconciliation_warning"] += 1
            log_event(
                "build_vplan_position_warning",
                _build_decision_plan_log_payload_dict(
                    release_obj,
                    decision_plan_obj,
                    as_of_ts,
                    {
                        "warning_code_str": "position_reconciliation_warning",
                        "mismatch_dict": reconciliation_result_obj.mismatch_dict,
                    },
                ),
                log_path_str=log_path_str,
            )
        if broker_snapshot_obj.net_liq_float <= 0.0:
            blocked_action_count_int += 1
            reason_counter_obj["non_positive_net_liq"] += 1
            state_store_obj.mark_decision_plan_status(int(decision_plan_obj.decision_plan_id_int or 0), "blocked")
            continue

        touched_asset_list = get_touched_asset_list_for_decision_plan(
            decision_plan_obj,
            broker_position_map_dict=broker_snapshot_obj.position_amount_map,
        )
        try:
            live_price_snapshot_obj = broker_adapter_obj.get_live_price_snapshot(
                decision_plan_obj.account_route_str,
                touched_asset_list,
                execution_policy_str=decision_plan_obj.execution_policy_str,
            )
        except (RuntimeError, ValueError) as exc:
            blocked_action_count_int += 1
            reason_counter_obj["live_price_snapshot_error"] += 1
            state_store_obj.mark_decision_plan_status(int(decision_plan_obj.decision_plan_id_int or 0), "blocked")
            log_event(
                "build_vplan_blocked",
                _build_decision_plan_log_payload_dict(
                    release_obj,
                    decision_plan_obj,
                    as_of_ts,
                    {
                        "reason_code_str": "live_price_snapshot_error",
                        "error_str": str(exc),
                    },
                ),
                log_path_str=log_path_str,
            )
            continue
        live_reference_source_map_dict = {
            str(asset_str): str(source_str)
            for asset_str, source_str in live_price_snapshot_obj.asset_reference_source_map_dict.items()
        }
        for asset_str, live_reference_source_str in sorted(live_reference_source_map_dict.items()):
            if live_reference_source_str not in (
                "ib_async.reqMktData.marketPrice.fallback",
                "ib_async.reqTickers.marketPrice.fallback",
            ):
                continue
            log_event(
                "build_vplan_live_reference_fallback_warning",
                _build_decision_plan_log_payload_dict(
                    release_obj,
                    decision_plan_obj,
                    as_of_ts,
                    {
                        "severity_str": "warning",
                        "reason_code_str": "live_reference_fallback",
                        "asset_str": asset_str,
                        "primary_source_str": "ib_async.reqMktData.225.auctionPrice",
                        "fallback_source_str": live_reference_source_str,
                        "live_reference_snapshot_timestamp_str": (
                            live_price_snapshot_obj.snapshot_timestamp_ts.isoformat()
                        ),
                    },
                ),
                log_path_str=log_path_str,
            )
        missing_asset_list = sorted(
            asset_str
            for asset_str in touched_asset_list
            if asset_str not in live_price_snapshot_obj.asset_reference_price_map
        )
        if len(missing_asset_list) > 0:
            blocked_action_count_int += 1
            reason_counter_obj["missing_live_price"] += 1
            state_store_obj.mark_decision_plan_status(int(decision_plan_obj.decision_plan_id_int or 0), "blocked")
            log_event(
                "build_vplan_blocked",
                _build_decision_plan_log_payload_dict(
                    release_obj,
                    decision_plan_obj,
                    as_of_ts,
                    {
                        "reason_code_str": "missing_live_price",
                        "missing_asset_list": missing_asset_list,
                    },
                ),
                log_path_str=log_path_str,
            )
            continue

        vplan_obj = build_vplan(
            release_obj=release_obj,
            decision_plan_obj=decision_plan_obj,
            broker_snapshot_obj=broker_snapshot_obj,
            live_price_snapshot_obj=live_price_snapshot_obj,
        )
        inserted_vplan_obj = state_store_obj.insert_vplan(vplan_obj)
        state_store_obj.mark_decision_plan_status(int(decision_plan_obj.decision_plan_id_int or 0), "vplan_ready")
        created_vplan_count_int += 1
        log_event(
            "build_vplan_created",
            _build_vplan_log_payload_dict(
                release_obj,
                inserted_vplan_obj,
                as_of_ts,
                {"reason_code_str": "vplan_ready"},
            ),
            log_path_str=log_path_str,
        )

    return {
        "created_vplan_count_int": created_vplan_count_int,
        "blocked_action_count_int": blocked_action_count_int,
        "warning_count_map_dict": dict(warning_counter_obj),
        "reason_count_map_dict": dict(reason_counter_obj),
    }


def submit_ready_vplans(
    state_store_obj: LiveStateStore,
    broker_adapter_obj: BrokerAdapter | None,
    as_of_ts: datetime,
    env_mode_str: str,
    manual_only_bool: bool,
    vplan_id_int: int | None = None,
    releases_root_path_str: str | None = None,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
    broker_adapter_resolver_obj: BrokerAdapterResolver | None = None,
    broker_host_str: str | None = None,
    broker_port_int: int | None = None,
    broker_client_id_int: int | None = None,
    broker_timeout_seconds_float: float | None = None,
    adapter_factory_func: Callable[[str, int, int, float], BrokerAdapter] | None = None,
    pod_id_str: str | None = None,
) -> dict[str, object]:
    submitted_vplan_count_int = 0
    blocked_action_count_int = 0
    warning_counter_obj: Counter[str] = Counter()
    reason_counter_obj: Counter[str] = Counter()
    if releases_root_path_str is None:
        _validate_state_store_releases_for_mutation(state_store_obj, env_mode_str, pod_id_str=pod_id_str)
    else:
        _load_release_list_validate_and_sync(
            releases_root_path_str,
            state_store_obj,
            env_mode_str,
            pod_id_str=pod_id_str,
        )
    broker_adapter_resolver_obj = _coerce_broker_adapter_resolver_obj(
        broker_adapter_obj=broker_adapter_obj,
        broker_adapter_resolver_obj=broker_adapter_resolver_obj,
        broker_host_str=broker_host_str,
        broker_port_int=broker_port_int,
        broker_client_id_int=broker_client_id_int,
        broker_timeout_seconds_float=broker_timeout_seconds_float,
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
        adapter_factory_func=adapter_factory_func,
    )
    if vplan_id_int is None:
        candidate_vplan_list = []
        for release_obj in state_store_obj.get_enabled_release_list():
            if pod_id_str is not None and release_obj.pod_id_str != pod_id_str:
                continue
            if release_obj.mode_str != env_mode_str:
                continue
            latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
            if latest_vplan_obj is None or latest_vplan_obj.status_str != "ready":
                continue
            if manual_only_bool:
                continue
            if not release_obj.auto_submit_enabled_bool:
                continue
            candidate_vplan_list.append(latest_vplan_obj)
    else:
        candidate_vplan_list = [state_store_obj.get_vplan_by_id(vplan_id_int)]

    for vplan_obj in candidate_vplan_list:
        if pod_id_str is not None and vplan_obj.pod_id_str != pod_id_str:
            reason_counter_obj["pod_id_mismatch"] += 1
            continue
        release_obj = state_store_obj.get_release_by_id(vplan_obj.release_id_str)
        if release_obj.mode_str != env_mode_str:
            blocked_action_count_int += 1
            reason_counter_obj["env_mode_mismatch"] += 1
            continue
        broker_adapter_obj = broker_adapter_resolver_obj.get_adapter(release_obj)
        if scheduler_utils.is_execution_window_expired_bool(
            vplan_obj.execution_policy_str,
            vplan_obj.target_execution_timestamp_ts,
            as_of_ts,
        ):
            blocked_action_count_int += 1
            reason_counter_obj["submission_window_expired"] += 1
            state_store_obj.mark_vplan_status(int(vplan_obj.vplan_id_int or 0), "expired")
            state_store_obj.mark_decision_plan_status(int(vplan_obj.decision_plan_id_int), "expired")
            continue
        if state_store_obj.count_broker_orders_for_vplan(int(vplan_obj.vplan_id_int or 0)) > 0:
            blocked_action_count_int += 1
            reason_counter_obj["duplicate_submission_guard"] += 1
            continue
        if not state_store_obj.claim_vplan_for_submission(int(vplan_obj.vplan_id_int or 0)):
            blocked_action_count_int += 1
            reason_counter_obj["submission_claim_failed"] += 1
            continue

        broker_order_request_list = build_broker_order_request_list_from_vplan(vplan_obj)
        submit_batch_result_obj = broker_adapter_obj.submit_order_request_list(
            account_route_str=vplan_obj.account_route_str,
            broker_order_request_list=broker_order_request_list,
            submitted_timestamp_ts=as_of_ts,
        )
        normalized_broker_order_record_list = [
            broker_order_record_obj.__class__(
                **{
                    **broker_order_record_obj.__dict__,
                    "decision_plan_id_int": int(vplan_obj.decision_plan_id_int),
                    "vplan_id_int": int(vplan_obj.vplan_id_int or 0),
                    "submission_key_str": (
                        broker_order_record_obj.submission_key_str
                        if broker_order_record_obj.submission_key_str is not None
                        else vplan_obj.submission_key_str
                    ),
                }
            )
            for broker_order_record_obj in submit_batch_result_obj.broker_order_record_list
        ]
        normalized_broker_order_event_list = [
            broker_order_event_obj.__class__(
                **{
                    **broker_order_event_obj.__dict__,
                    "decision_plan_id_int": int(vplan_obj.decision_plan_id_int),
                    "vplan_id_int": int(vplan_obj.vplan_id_int or 0),
                    "submission_key_str": (
                        broker_order_event_obj.submission_key_str
                        if broker_order_event_obj.submission_key_str is not None
                        else vplan_obj.submission_key_str
                    ),
                }
            )
            for broker_order_event_obj in submit_batch_result_obj.broker_order_event_list
        ]
        normalized_broker_order_fill_list = [
            broker_order_fill_obj.__class__(
                **{
                    **broker_order_fill_obj.__dict__,
                    "decision_plan_id_int": int(vplan_obj.decision_plan_id_int),
                    "vplan_id_int": int(vplan_obj.vplan_id_int or 0),
                }
            )
            for broker_order_fill_obj in submit_batch_result_obj.broker_order_fill_list
        ]
        normalized_broker_order_ack_list = [
            broker_order_ack_obj.__class__(
                **{
                    **broker_order_ack_obj.__dict__,
                    "decision_plan_id_int": int(vplan_obj.decision_plan_id_int),
                    "vplan_id_int": int(vplan_obj.vplan_id_int or 0),
                }
            )
            for broker_order_ack_obj in submit_batch_result_obj.broker_order_ack_list
        ]
        state_store_obj.upsert_vplan_broker_order_record_list(
            normalized_broker_order_record_list
        )
        state_store_obj.insert_vplan_broker_order_event_list(
            normalized_broker_order_event_list
        )
        state_store_obj.upsert_vplan_fill_list(
            normalized_broker_order_fill_list
        )
        state_store_obj.upsert_vplan_broker_ack_list(
            normalized_broker_order_ack_list
        )
        state_store_obj.update_vplan_submit_ack_summary(
            vplan_id_int=int(vplan_obj.vplan_id_int or 0),
            submit_ack_status_str=submit_batch_result_obj.submit_ack_status_str,
            ack_coverage_ratio_float=submit_batch_result_obj.ack_coverage_ratio_float,
            missing_ack_count_int=len(submit_batch_result_obj.missing_ack_asset_list),
            submit_ack_checked_timestamp_ts=as_of_ts,
        )
        state_store_obj.mark_vplan_status(int(vplan_obj.vplan_id_int or 0), "submitted")
        state_store_obj.mark_decision_plan_status(int(vplan_obj.decision_plan_id_int), "submitted")
        submitted_vplan_count_int += 1
        if len(submit_batch_result_obj.missing_ack_asset_list) > 0:
            warning_counter_obj["missing_broker_response_ack"] += len(
                submit_batch_result_obj.missing_ack_asset_list
            )
            log_event(
                "submit_vplan_missing_broker_ack",
                _build_vplan_log_payload_dict(
                    release_obj,
                    vplan_obj,
                    as_of_ts,
                    {
                        "severity_str": "critical",
                        "reason_code_str": "missing_broker_response_ack",
                        "request_count_int": len(broker_order_request_list),
                        "ack_coverage_ratio_float": float(
                            submit_batch_result_obj.ack_coverage_ratio_float
                        ),
                        "missing_ack_count_int": len(
                            submit_batch_result_obj.missing_ack_asset_list
                        ),
                        "missing_ack_asset_list": list(
                            submit_batch_result_obj.missing_ack_asset_list
                        ),
                    },
                ),
                log_path_str=log_path_str,
            )
        log_event(
            "submit_vplan_completed",
            _build_vplan_log_payload_dict(
                release_obj,
                vplan_obj,
                as_of_ts,
                    {
                        "reason_code_str": "submitted",
                        "broker_order_count_int": len(normalized_broker_order_record_list),
                        "broker_order_event_count_int": len(normalized_broker_order_event_list),
                        "fill_count_int": len(normalized_broker_order_fill_list),
                        "broker_order_ack_count_int": len(normalized_broker_order_ack_list),
                        "submit_ack_status_str": submit_batch_result_obj.submit_ack_status_str,
                        "ack_coverage_ratio_float": float(
                            submit_batch_result_obj.ack_coverage_ratio_float
                    ),
                    "missing_ack_count_int": len(submit_batch_result_obj.missing_ack_asset_list),
                },
            ),
            log_path_str=log_path_str,
        )

    return {
        "submitted_vplan_count_int": submitted_vplan_count_int,
        "blocked_action_count_int": blocked_action_count_int,
        "warning_count_map_dict": dict(warning_counter_obj),
        "reason_count_map_dict": dict(reason_counter_obj),
    }


def eod_snapshot(
    state_store_obj: LiveStateStore,
    broker_adapter_obj: BrokerAdapter | None,
    as_of_ts: datetime,
    env_mode_str: str = "paper",
    releases_root_path_str: str | None = None,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
    broker_adapter_resolver_obj: BrokerAdapterResolver | None = None,
    broker_host_str: str | None = None,
    broker_port_int: int | None = None,
    broker_client_id_int: int | None = None,
    broker_timeout_seconds_float: float | None = None,
    adapter_factory_func: Callable[[str, int, int, float], BrokerAdapter] | None = None,
    require_due_bool: bool = False,
    eod_snapshot_buffer_minutes_int: int = DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT,
    pod_id_str: str | None = None,
) -> dict[str, object]:
    if releases_root_path_str is None:
        _validate_state_store_releases_for_mutation(state_store_obj, env_mode_str, pod_id_str=pod_id_str)
    else:
        _load_release_list_validate_and_sync(
            releases_root_path_str,
            state_store_obj,
            env_mode_str,
            pod_id_str=pod_id_str,
        )

    broker_adapter_resolver_obj = _coerce_broker_adapter_resolver_obj(
        broker_adapter_obj=broker_adapter_obj,
        broker_adapter_resolver_obj=broker_adapter_resolver_obj,
        broker_host_str=broker_host_str,
        broker_port_int=broker_port_int,
        broker_client_id_int=broker_client_id_int,
        broker_timeout_seconds_float=broker_timeout_seconds_float,
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
        adapter_factory_func=adapter_factory_func,
    )

    snapshot_source_str = _snapshot_source_str_for_mode(env_mode_str)
    eod_snapshot_count_int = 0
    skipped_snapshot_count_int = 0
    blocked_action_count_int = 0
    reason_counter_obj: Counter[str] = Counter()

    for release_obj in state_store_obj.get_enabled_release_list():
        if pod_id_str is not None and release_obj.pod_id_str != pod_id_str:
            continue
        if release_obj.mode_str != env_mode_str:
            continue

        if require_due_bool:
            eod_due_timestamp_ts = _eod_snapshot_due_timestamp_ts(
                release_obj=release_obj,
                as_of_ts=as_of_ts,
                eod_snapshot_buffer_minutes_int=eod_snapshot_buffer_minutes_int,
            )
            if eod_due_timestamp_ts is None or as_of_ts < eod_due_timestamp_ts:
                continue

        eod_market_date_str = _eod_snapshot_market_date_str(
            release_obj=release_obj,
            as_of_ts=as_of_ts,
        )
        # *** CRITICAL*** Do not let an EOD state sample overwrite the latest
        # pod state while a submitted execution has not been reconciled.
        # Reconcile owns fill/position truth; EOD is only the clean close-state
        # sample after execution has settled.
        if _pod_has_unresolved_execution_bool(state_store_obj, release_obj.pod_id_str):
            skipped_snapshot_count_int += 1
            reason_counter_obj["unresolved_execution"] += 1
            continue

        if _pod_has_stage_snapshot_for_market_date_bool(
            state_store_obj=state_store_obj,
            release_obj=release_obj,
            snapshot_stage_str="eod",
            market_date_str=eod_market_date_str,
        ):
            skipped_snapshot_count_int += 1
            reason_counter_obj["eod_snapshot_already_exists"] += 1
            continue

        broker_adapter_for_release_obj = broker_adapter_resolver_obj.get_adapter(release_obj)
        visible_account_route_set = broker_adapter_for_release_obj.get_visible_account_route_set()
        if (
            visible_account_route_set is not None
            and release_obj.account_route_str not in visible_account_route_set
        ):
            blocked_action_count_int += 1
            reason_counter_obj["account_not_visible"] += 1
            continue
        if not broker_adapter_for_release_obj.is_session_ready(release_obj.account_route_str):
            blocked_action_count_int += 1
            reason_counter_obj["broker_not_ready"] += 1
            continue
        session_mode_str = broker_adapter_for_release_obj.get_session_mode_str(release_obj.account_route_str)
        if session_mode_str is not None and session_mode_str != release_obj.mode_str:
            blocked_action_count_int += 1
            reason_counter_obj["session_mode_mismatch"] += 1
            continue

        broker_snapshot_obj = broker_adapter_for_release_obj.get_account_snapshot(release_obj.account_route_str)
        state_store_obj.upsert_broker_snapshot_cache(broker_snapshot_obj)
        previous_pod_state_obj = state_store_obj.get_pod_state(release_obj.pod_id_str)
        strategy_state_dict = (
            {} if previous_pod_state_obj is None else dict(previous_pod_state_obj.strategy_state_dict)
        )
        state_store_obj.upsert_pod_state(
            PodState(
                pod_id_str=release_obj.pod_id_str,
                user_id_str=release_obj.user_id_str,
                account_route_str=release_obj.account_route_str,
                position_amount_map=broker_snapshot_obj.position_amount_map,
                cash_float=broker_snapshot_obj.cash_float,
                total_value_float=broker_snapshot_obj.net_liq_float,
                strategy_state_dict=strategy_state_dict,
                updated_timestamp_ts=as_of_ts,
            ),
            snapshot_stage_str="eod",
            snapshot_source_str=snapshot_source_str,
        )
        eod_snapshot_count_int += 1
        reason_counter_obj["eod_snapshot_completed"] += 1
        log_event(
            "eod_snapshot_completed",
            _build_release_log_payload_dict(
                release_obj,
                as_of_ts,
                {
                    "snapshot_stage_str": "eod",
                    "snapshot_source_str": snapshot_source_str,
                    "eod_market_date_str": eod_market_date_str,
                    "broker_snapshot_timestamp_str": broker_snapshot_obj.snapshot_timestamp_ts.isoformat(),
                    "cash_float": float(broker_snapshot_obj.cash_float),
                    "net_liq_float": float(broker_snapshot_obj.net_liq_float),
                },
            ),
            log_path_str=log_path_str,
        )

    return {
        "lease_acquired_bool": True,
        "eod_snapshot_count_int": int(eod_snapshot_count_int),
        "skipped_snapshot_count_int": int(skipped_snapshot_count_int),
        "blocked_action_count_int": int(blocked_action_count_int),
        "reason_count_map_dict": dict(reason_counter_obj),
    }


def post_execution_reconcile(
    state_store_obj: LiveStateStore,
    broker_adapter_obj: BrokerAdapter | None,
    as_of_ts: datetime,
    env_mode_str: str = "paper",
    releases_root_path_str: str | None = None,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
    broker_adapter_resolver_obj: BrokerAdapterResolver | None = None,
    broker_host_str: str | None = None,
    broker_port_int: int | None = None,
    broker_client_id_int: int | None = None,
    broker_timeout_seconds_float: float | None = None,
    adapter_factory_func: Callable[[str, int, int, float], BrokerAdapter] | None = None,
    pod_id_str: str | None = None,
) -> dict[str, object]:
    if releases_root_path_str is None:
        _validate_state_store_releases_for_mutation(state_store_obj, env_mode_str, pod_id_str=pod_id_str)
    else:
        _load_release_list_validate_and_sync(
            releases_root_path_str,
            state_store_obj,
            env_mode_str,
            pod_id_str=pod_id_str,
        )
    log_path_obj = Path(log_path_str)
    log_path_obj.parent.mkdir(parents=True, exist_ok=True)
    log_path_obj.touch(exist_ok=True)
    completed_vplan_count_int = 0
    tolerance_float = 1e-9
    broker_adapter_resolver_obj = _coerce_broker_adapter_resolver_obj(
        broker_adapter_obj=broker_adapter_obj,
        broker_adapter_resolver_obj=broker_adapter_resolver_obj,
        broker_host_str=broker_host_str,
        broker_port_int=broker_port_int,
        broker_client_id_int=broker_client_id_int,
        broker_timeout_seconds_float=broker_timeout_seconds_float,
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
        adapter_factory_func=adapter_factory_func,
    )
    for vplan_obj in state_store_obj.get_submitted_vplan_list():
        if pod_id_str is not None and vplan_obj.pod_id_str != pod_id_str:
            continue
        if as_of_ts < vplan_obj.target_execution_timestamp_ts:
            continue
        post_execution_snapshot_exists_bool = state_store_obj.has_post_execution_reconciliation_snapshot(
            int(vplan_obj.vplan_id_int or 0)
        )
        release_obj = state_store_obj.get_release_by_id(vplan_obj.release_id_str)
        if release_obj.mode_str != env_mode_str:
            continue
        broker_adapter_obj = broker_adapter_resolver_obj.get_adapter(release_obj)
        decision_plan_obj = state_store_obj.get_decision_plan_by_id(int(vplan_obj.decision_plan_id_int))
        broker_snapshot_obj = broker_adapter_obj.get_account_snapshot(vplan_obj.account_route_str)
        state_store_obj.upsert_broker_snapshot_cache(broker_snapshot_obj)
        existing_broker_order_row_dict_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(
            int(vplan_obj.vplan_id_int or 0)
        )
        known_broker_order_id_set = {
            str(broker_order_row_dict["broker_order_id_str"])
            for broker_order_row_dict in existing_broker_order_row_dict_list
        }
        broker_order_record_list, broker_order_event_list, broker_fill_list = (
            broker_adapter_obj.get_recent_order_state_snapshot(
                account_route_str=vplan_obj.account_route_str,
                since_timestamp_ts=vplan_obj.submission_timestamp_ts,
                submission_key_str=(
                    str(vplan_obj.submission_key_str)
                    if vplan_obj.submission_key_str is not None
                    else f"vplan:{vplan_obj.decision_plan_id_int}"
                ),
                allowed_broker_order_id_set=known_broker_order_id_set,
            )
        )
        session_open_context_dict = _session_open_context_dict(
            release_obj=release_obj,
            reference_timestamp_ts=vplan_obj.target_execution_timestamp_ts,
        )
        open_universe_asset_set = set(broker_snapshot_obj.position_amount_map) | set(
            get_touched_asset_list_for_decision_plan(
                decision_plan_obj,
                broker_position_map_dict=broker_snapshot_obj.position_amount_map,
            )
        )
        # *** CRITICAL*** Session-open lookup must be anchored to the target execution session,
        # never a later timestamp, or fill-vs-open slippage would leak future session context.
        session_open_price_list = broker_adapter_obj.get_session_open_price_list(
            account_route_str=vplan_obj.account_route_str,
            asset_str_list=sorted(open_universe_asset_set),
            session_open_timestamp_ts=session_open_context_dict["session_open_timestamp_ts"],
            session_calendar_id_str=release_obj.session_calendar_id_str,
        )
        state_store_obj.upsert_session_open_price_list(session_open_price_list)
        session_open_price_map_dict = state_store_obj.get_session_open_price_map_dict(
            account_route_str=vplan_obj.account_route_str,
            session_date_str=str(session_open_context_dict["session_date_str"]),
        )
        normalized_broker_order_record_list = [
            broker_order_record_obj.__class__(
                **{
                    **broker_order_record_obj.__dict__,
                    "decision_plan_id_int": int(vplan_obj.decision_plan_id_int),
                    "vplan_id_int": int(vplan_obj.vplan_id_int or 0),
                    "submission_key_str": (
                        broker_order_record_obj.submission_key_str
                        if broker_order_record_obj.submission_key_str is not None
                        else vplan_obj.submission_key_str
                    ),
                }
            )
            for broker_order_record_obj in broker_order_record_list
        ]
        normalized_broker_order_event_list = [
            broker_order_event_obj.__class__(
                **{
                    **broker_order_event_obj.__dict__,
                    "decision_plan_id_int": int(vplan_obj.decision_plan_id_int),
                    "vplan_id_int": int(vplan_obj.vplan_id_int or 0),
                    "submission_key_str": (
                        broker_order_event_obj.submission_key_str
                        if broker_order_event_obj.submission_key_str is not None
                        else vplan_obj.submission_key_str
                    ),
                }
            )
            for broker_order_event_obj in broker_order_event_list
        ]
        normalized_fill_list = [
            fill_obj.__class__(
                **{
                    **fill_obj.__dict__,
                    "decision_plan_id_int": int(vplan_obj.decision_plan_id_int),
                    "vplan_id_int": int(vplan_obj.vplan_id_int or 0),
                }
            )
            for fill_obj in broker_fill_list
        ]
        normalized_fill_list = _annotate_fill_list_with_session_open_price(
            fill_list=normalized_fill_list,
            session_open_price_map_dict=session_open_price_map_dict,
        )
        normalized_broker_order_record_list, normalized_broker_order_event_list = (
            _synthesize_terminal_filled_event_list(
                broker_order_record_list=normalized_broker_order_record_list,
                broker_order_event_list=normalized_broker_order_event_list,
                broker_order_fill_list=normalized_fill_list,
            )
        )
        state_store_obj.upsert_vplan_broker_order_record_list(normalized_broker_order_record_list)
        state_store_obj.insert_vplan_broker_order_event_list(normalized_broker_order_event_list)
        state_store_obj.upsert_vplan_fill_list(normalized_fill_list)
        expected_broker_position_map_dict = _build_expected_broker_position_map_dict(
            vplan_obj,
            tolerance_float=tolerance_float,
        )
        reconciliation_result_obj = reconcile_account_state(
            model_position_map=expected_broker_position_map_dict,
            model_cash_float=broker_snapshot_obj.cash_float,
            broker_snapshot_obj=broker_snapshot_obj,
            tolerance_float=tolerance_float,
        )
        state_store_obj.insert_vplan_reconciliation_snapshot(
            pod_id_str=vplan_obj.pod_id_str,
            decision_plan_id_int=int(vplan_obj.decision_plan_id_int),
            vplan_id_int=int(vplan_obj.vplan_id_int or 0),
            stage_str="post_execution",
            reconciliation_result_obj=reconciliation_result_obj,
        )
        strategy_state_dict = {} if decision_plan_obj is None else dict(decision_plan_obj.strategy_state_dict)
        state_store_obj.upsert_pod_state(
            PodState(
                pod_id_str=vplan_obj.pod_id_str,
                user_id_str=release_obj.user_id_str,
                account_route_str=vplan_obj.account_route_str,
                position_amount_map=broker_snapshot_obj.position_amount_map,
                cash_float=broker_snapshot_obj.cash_float,
                total_value_float=broker_snapshot_obj.net_liq_float,
                strategy_state_dict=strategy_state_dict,
                updated_timestamp_ts=as_of_ts,
            ),
            snapshot_stage_str="post_execution",
            snapshot_source_str=("virtual_broker" if env_mode_str == "incubation" else "broker"),
        )
        persisted_broker_order_row_dict_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(
            int(vplan_obj.vplan_id_int or 0)
        )
        persisted_fill_row_dict_list = _enrich_fill_row_dict_list(
            state_store_obj.get_fill_row_dict_list_for_vplan(int(vplan_obj.vplan_id_int or 0))
        )
        execution_row_dict_list = _build_execution_row_dict_list(
            vplan_obj=vplan_obj,
            broker_position_map_dict=broker_snapshot_obj.position_amount_map,
            broker_order_row_dict_list=persisted_broker_order_row_dict_list,
            fill_row_dict_list=persisted_fill_row_dict_list,
            tolerance_float=tolerance_float,
        )
        exit_breach_row_dict_list = [
            execution_row_dict
            for execution_row_dict in execution_row_dict_list
            if bool(execution_row_dict["exit_breach_bool"])
        ]
        terminal_unresolved_row_dict_list = _build_terminal_unresolved_row_dict_list(execution_row_dict_list)
        for exit_breach_row_dict in exit_breach_row_dict_list:
            log_event(
                "exit_residual_detected",
                _build_vplan_log_payload_dict(
                    release_obj,
                    vplan_obj,
                    as_of_ts,
                    {
                        "severity_str": "critical",
                        "reason_code_str": "exit_residual_detected",
                        "asset_str": exit_breach_row_dict["asset_str"],
                        "target_share_float": float(exit_breach_row_dict["target_share_float"]),
                        "broker_share_float": float(exit_breach_row_dict["broker_share_float"]),
                        "residual_share_float": float(exit_breach_row_dict["residual_share_float"]),
                        "latest_broker_order_status_str": exit_breach_row_dict["latest_broker_order_status_str"],
                    },
                ),
                log_path_str=log_path_str,
            )
        if len(terminal_unresolved_row_dict_list) > 0 and not post_execution_snapshot_exists_bool:
            log_event(
                "execution_exception_parked",
                _build_vplan_log_payload_dict(
                    release_obj,
                    vplan_obj,
                    as_of_ts,
                    {
                        "severity_str": "critical",
                        "reason_code_str": "execution_exception_parked",
                        "terminal_unresolved_count_int": len(terminal_unresolved_row_dict_list),
                        "terminal_unresolved_asset_list": [
                            str(terminal_unresolved_row_dict["asset_str"])
                            for terminal_unresolved_row_dict in terminal_unresolved_row_dict_list
                        ],
                        "terminal_unresolved_status_map_dict": {
                            str(terminal_unresolved_row_dict["asset_str"]): terminal_unresolved_row_dict[
                                "latest_broker_order_status_str"
                            ]
                            for terminal_unresolved_row_dict in terminal_unresolved_row_dict_list
                        },
                        "terminal_unresolved_residual_share_map_dict": {
                            str(terminal_unresolved_row_dict["asset_str"]): float(
                                terminal_unresolved_row_dict["residual_share_float"]
                            )
                            for terminal_unresolved_row_dict in terminal_unresolved_row_dict_list
                        },
                    },
                ),
                log_path_str=log_path_str,
            )
        completed_bool = bool(reconciliation_result_obj.passed_bool)
        if completed_bool:
            state_store_obj.mark_vplan_status(int(vplan_obj.vplan_id_int or 0), "completed")
            state_store_obj.mark_decision_plan_status(int(vplan_obj.decision_plan_id_int), "completed")
            completed_vplan_count_int += 1
            log_event(
                "post_execution_reconcile_completed",
                _build_vplan_log_payload_dict(
                    release_obj,
                    vplan_obj,
                    as_of_ts,
                    {
                        "reason_code_str": "completed",
                        "broker_order_count_int": len(normalized_broker_order_record_list),
                        "broker_order_event_count_int": len(normalized_broker_order_event_list),
                        "fill_count_int": len(normalized_fill_list),
                        "session_open_price_count_int": len(session_open_price_list),
                    },
                ),
                log_path_str=log_path_str,
            )
    return {"completed_vplan_count_int": completed_vplan_count_int}


def get_status_summary(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str = "paper",
    pod_id_str: str | None = None,
) -> dict[str, object]:
    release_list = _load_release_list_and_sync(
        releases_root_path_str,
        state_store_obj,
        pod_id_str=pod_id_str,
    )
    enabled_release_list = select_enabled_release_list_for_mode(
        release_list=release_list,
        env_mode_str=env_mode_str,
    )
    selected_user_id_list = sorted(
        {str(release_obj.user_id_str).strip() for release_obj in enabled_release_list}
    )
    if len(selected_user_id_list) == 0:
        client_user_id_str = "none"
        deployment_warning_str = None
    elif len(selected_user_id_list) == 1:
        client_user_id_str = selected_user_id_list[0]
        deployment_warning_str = None
    else:
        client_user_id_str = "mixed"
        deployment_warning_str = (
            "selected enabled sleeves contain multiple client identities: "
            f"{', '.join(selected_user_id_list)}"
        )
    pod_status_dict_list: list[dict[str, object]] = []

    for release_obj in enabled_release_list:
        latest_decision_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
        latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
        latest_broker_snapshot_obj = state_store_obj.get_latest_broker_snapshot_for_account(release_obj.account_route_str)
        latest_fill_timestamp_str = None
        latest_broker_order_row_dict_list: list[dict[str, object]] = []
        broker_ack_row_dict_list: list[dict[str, object]] = []
        missing_ack_row_dict_list: list[dict[str, object]] = []
        exception_row_dict_list: list[dict[str, object]] = []
        parked_execution_exception_bool = False
        if latest_vplan_obj is not None and latest_vplan_obj.vplan_id_int is not None:
            fill_row_dict_list = _enrich_fill_row_dict_list(
                state_store_obj.get_fill_row_dict_list_for_vplan(int(latest_vplan_obj.vplan_id_int))
            )
            if len(fill_row_dict_list) > 0:
                latest_fill_timestamp_str = str(fill_row_dict_list[-1]["fill_timestamp_str"])
            latest_broker_order_row_dict_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(
                int(latest_vplan_obj.vplan_id_int)
            )
            broker_ack_row_dict_list = state_store_obj.get_broker_ack_row_dict_list_for_vplan(
                int(latest_vplan_obj.vplan_id_int)
            )
            missing_ack_row_dict_list = _build_missing_ack_row_dict_list(broker_ack_row_dict_list)
            if latest_vplan_obj.status_str in ("submitted", "completed") or len(
                latest_broker_order_row_dict_list
            ) > 0 or len(fill_row_dict_list) > 0:
                broker_position_map_dict = (
                    latest_vplan_obj.current_broker_position_map
                    if latest_broker_snapshot_obj is None
                    else latest_broker_snapshot_obj.position_amount_map
                )
                execution_row_dict_list = _build_execution_row_dict_list(
                    vplan_obj=latest_vplan_obj,
                    broker_position_map_dict=broker_position_map_dict,
                    broker_order_row_dict_list=latest_broker_order_row_dict_list,
                    fill_row_dict_list=fill_row_dict_list,
                )
                exception_row_dict_list = _build_exception_row_dict_list(execution_row_dict_list)
                terminal_unresolved_row_dict_list = _build_terminal_unresolved_row_dict_list(
                    execution_row_dict_list
                )
                parked_execution_exception_bool = bool(
                    latest_vplan_obj.status_str == "submitted"
                    and state_store_obj.has_post_execution_reconciliation_snapshot(
                        int(latest_vplan_obj.vplan_id_int)
                    )
                    and len(terminal_unresolved_row_dict_list) > 0
                )
        build_gate_dict = scheduler_utils.evaluate_build_gate_dict(release_obj, as_of_ts)

        next_action_str = "wait"
        reason_code_str = str(build_gate_dict["reason_code_str"])
        if latest_decision_plan_obj is None:
            if bool(build_gate_dict["due_bool"]):
                next_action_str = "build_decision_plan"
                reason_code_str = "ready_to_build_decision_plan"
        elif latest_decision_plan_obj.status_str == "planned":
            if scheduler_utils.is_execution_window_expired_bool(
                latest_decision_plan_obj.execution_policy_str,
                latest_decision_plan_obj.target_execution_timestamp_ts,
                as_of_ts,
            ):
                reason_code_str = "submission_window_expired"
            elif latest_decision_plan_obj.submission_timestamp_ts <= as_of_ts:
                next_action_str = "build_vplan"
                reason_code_str = "ready_to_build_vplan"
            else:
                reason_code_str = "waiting_for_submission_window"
        elif latest_vplan_obj is not None and latest_vplan_obj.status_str == "ready":
            next_action_str = "submit_vplan" if release_obj.auto_submit_enabled_bool else "review_vplan"
            reason_code_str = "vplan_ready"
        elif latest_vplan_obj is not None and latest_vplan_obj.status_str == "submitted":
            if parked_execution_exception_bool:
                next_action_str = "manual_review"
                reason_code_str = "execution_exception_parked"
            else:
                next_action_str = "post_execution_reconcile"
                reason_code_str = "waiting_for_post_execution_reconcile"
        elif latest_decision_plan_obj.status_str in ("completed", "expired", "blocked"):
            if bool(build_gate_dict["due_bool"]):
                next_action_str = "build_decision_plan"
                reason_code_str = "ready_to_build_decision_plan"

        pod_status_dict_list.append(
            {
                "release_id_str": release_obj.release_id_str,
                "user_id_str": release_obj.user_id_str,
                "pod_id_str": release_obj.pod_id_str,
                "mode_str": release_obj.mode_str,
                "account_route_str": release_obj.account_route_str,
                "auto_submit_enabled_bool": bool(release_obj.auto_submit_enabled_bool),
                "latest_decision_plan_status_str": (
                    None if latest_decision_plan_obj is None else latest_decision_plan_obj.status_str
                ),
                "latest_vplan_status_str": None if latest_vplan_obj is None else latest_vplan_obj.status_str,
                "latest_signal_timestamp_str": (
                    None if latest_decision_plan_obj is None else latest_decision_plan_obj.signal_timestamp_ts.isoformat()
                ),
                "latest_submission_timestamp_str": (
                    None
                    if latest_decision_plan_obj is None
                    else latest_decision_plan_obj.submission_timestamp_ts.isoformat()
                ),
                "latest_broker_snapshot_timestamp_str": (
                    None
                    if latest_broker_snapshot_obj is None
                    else latest_broker_snapshot_obj.snapshot_timestamp_ts.isoformat()
                ),
                "next_action_str": next_action_str,
                "reason_code_str": reason_code_str,
                "latest_fill_timestamp_str": latest_fill_timestamp_str,
                "submit_ack_status_str": (
                    None if latest_vplan_obj is None else latest_vplan_obj.submit_ack_status_str
                ),
                "ack_coverage_ratio_float": (
                    None if latest_vplan_obj is None else latest_vplan_obj.ack_coverage_ratio_float
                ),
                "missing_ack_count_int": (
                    0 if latest_vplan_obj is None else int(latest_vplan_obj.missing_ack_count_int)
                ),
                "missing_ack_row_dict_list": missing_ack_row_dict_list,
                "latest_broker_order_row_dict_list": latest_broker_order_row_dict_list,
                "broker_ack_row_dict_list": broker_ack_row_dict_list,
                "exception_row_dict_list": exception_row_dict_list,
                "exception_count_int": len(exception_row_dict_list),
            }
        )

    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "mode_str": env_mode_str,
        "client_user_id_str": client_user_id_str,
        "deployment_warning_str": deployment_warning_str,
        "enabled_pod_count_int": len(enabled_release_list),
        "pod_status_dict_list": pod_status_dict_list,
    }


def show_vplan_summary(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    vplan_id_int: int | None = None,
    pod_id_str: str | None = None,
) -> dict[str, object]:
    _load_release_list_and_sync(
        releases_root_path_str,
        state_store_obj,
        pod_id_str=pod_id_str,
    )
    if vplan_id_int is not None:
        vplan_obj_list = [state_store_obj.get_vplan_by_id(vplan_id_int)]
    elif pod_id_str is not None:
        latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod(pod_id_str)
        vplan_obj_list = [] if latest_vplan_obj is None else [latest_vplan_obj]
    else:
        vplan_obj_list = []
        for release_obj in state_store_obj.get_enabled_release_list():
            latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
            if latest_vplan_obj is not None:
                vplan_obj_list.append(latest_vplan_obj)

    def _build_position_warning_row_dict_list(
        decision_base_position_map: dict[str, float],
        current_broker_position_map: dict[str, float],
        tolerance_float: float = 1e-9,
    ) -> list[dict[str, object]]:
        compared_asset_set = set(decision_base_position_map) | set(current_broker_position_map)
        warning_row_dict_list: list[dict[str, object]] = []
        for asset_str in sorted(compared_asset_set):
            decision_base_share_float = float(decision_base_position_map.get(asset_str, 0.0))
            current_share_float = float(current_broker_position_map.get(asset_str, 0.0))
            drift_share_float = current_share_float - decision_base_share_float
            if abs(drift_share_float) <= tolerance_float:
                continue
            warning_row_dict_list.append(
                {
                    "asset_str": asset_str,
                    "decision_base_share_float": decision_base_share_float,
                    "current_share_float": current_share_float,
                    "drift_share_float": drift_share_float,
                }
            )
        return warning_row_dict_list

    def _build_vplan_dict(vplan_obj: VPlan) -> dict[str, object]:
        decision_plan_obj = state_store_obj.get_decision_plan_by_id(int(vplan_obj.decision_plan_id_int))
        decision_base_position_map = decision_plan_obj.decision_base_position_map
        warning_row_dict_list = _build_position_warning_row_dict_list(
            decision_base_position_map,
            vplan_obj.current_broker_position_map,
        )
        broker_order_row_dict_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(
            int(vplan_obj.vplan_id_int or 0)
        )
        broker_ack_row_dict_list = state_store_obj.get_broker_ack_row_dict_list_for_vplan(
            int(vplan_obj.vplan_id_int or 0)
        )
        missing_ack_row_dict_list = _build_missing_ack_row_dict_list(broker_ack_row_dict_list)
        fill_row_dict_list = _enrich_fill_row_dict_list(
            state_store_obj.get_fill_row_dict_list_for_vplan(int(vplan_obj.vplan_id_int or 0))
        )
        broker_order_row_dict_list = _enrich_broker_order_row_dict_list_for_display(
            broker_order_row_dict_list=broker_order_row_dict_list,
            fill_row_dict_list=fill_row_dict_list,
        )
        latest_broker_snapshot_obj = state_store_obj.get_latest_broker_snapshot_for_account(vplan_obj.account_route_str)
        display_mode_str = (
            "planned"
            if len(broker_order_row_dict_list) == 0 and len(fill_row_dict_list) == 0 and vplan_obj.status_str == "ready"
            else "execution"
        )
        execution_row_dict_list: list[dict[str, object]] = []
        exception_row_dict_list: list[dict[str, object]] = []
        if display_mode_str == "execution":
            broker_position_map_dict = (
                vplan_obj.current_broker_position_map
                if latest_broker_snapshot_obj is None
                else latest_broker_snapshot_obj.position_amount_map
            )
            execution_row_dict_list = _build_execution_row_dict_list(
                vplan_obj=vplan_obj,
                broker_position_map_dict=broker_position_map_dict,
                broker_order_row_dict_list=broker_order_row_dict_list,
                fill_row_dict_list=fill_row_dict_list,
            )
            exception_row_dict_list = _build_exception_row_dict_list(execution_row_dict_list)
        return {
            "vplan_id_int": vplan_obj.vplan_id_int,
            "decision_plan_id_int": vplan_obj.decision_plan_id_int,
            "pod_id_str": vplan_obj.pod_id_str,
            "status_str": vplan_obj.status_str,
            "submit_ack_status_str": vplan_obj.submit_ack_status_str,
            "ack_coverage_ratio_float": vplan_obj.ack_coverage_ratio_float,
            "missing_ack_count_int": int(vplan_obj.missing_ack_count_int),
            "net_liq_float": vplan_obj.net_liq_float,
            "pod_budget_float": vplan_obj.pod_budget_float,
            "live_price_source_str": vplan_obj.live_price_source_str,
            "live_reference_source_map_dict": {
                asset_str: source_str
                for asset_str, source_str in vplan_obj.live_reference_source_map_dict.items()
            },
            "display_mode_str": display_mode_str,
            "latest_broker_snapshot_timestamp_str": (
                None
                if latest_broker_snapshot_obj is None
                else latest_broker_snapshot_obj.snapshot_timestamp_ts.isoformat()
            ),
            "broker_order_row_dict_list": broker_order_row_dict_list,
            "broker_ack_row_dict_list": broker_ack_row_dict_list,
            "missing_ack_row_dict_list": missing_ack_row_dict_list,
            "fill_row_dict_list": fill_row_dict_list,
            "execution_row_dict_list": execution_row_dict_list,
            "exception_row_dict_list": exception_row_dict_list,
            "warning_row_dict_list": warning_row_dict_list,
            "vplan_row_dict_list": [
                {
                    "asset_str": vplan_row_obj.asset_str,
                    "decision_base_share_float": float(
                        decision_base_position_map.get(vplan_row_obj.asset_str, 0.0)
                    ),
                    "current_share_float": vplan_row_obj.current_share_float,
                    "drift_share_float": float(
                        vplan_row_obj.current_share_float
                        - decision_base_position_map.get(vplan_row_obj.asset_str, 0.0)
                    ),
                    "target_share_float": vplan_row_obj.target_share_float,
                    "order_delta_share_float": vplan_row_obj.order_delta_share_float,
                    "live_reference_price_float": vplan_row_obj.live_reference_price_float,
                    "live_reference_source_str": vplan_row_obj.live_reference_source_str,
                    "estimated_target_notional_float": vplan_row_obj.estimated_target_notional_float,
                    "warning_bool": bool(
                        abs(
                            vplan_row_obj.current_share_float
                            - decision_base_position_map.get(vplan_row_obj.asset_str, 0.0)
                        )
                        > 1e-9
                    ),
                }
                for vplan_row_obj in vplan_obj.vplan_row_list
            ],
        }

    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "vplan_dict_list": [_build_vplan_dict(vplan_obj) for vplan_obj in vplan_obj_list],
    }


def get_execution_report_summary(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str | None = None,
    pod_id_str: str | None = None,
) -> dict[str, object]:
    _load_release_list_and_sync(releases_root_path_str, state_store_obj, pod_id_str=pod_id_str)
    execution_report_dict_list: list[dict[str, object]] = []
    for release_obj in state_store_obj.get_enabled_release_list():
        if pod_id_str is not None and release_obj.pod_id_str != pod_id_str:
            continue
        if env_mode_str is not None and release_obj.mode_str != env_mode_str:
            continue
        latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
        if latest_vplan_obj is None or latest_vplan_obj.vplan_id_int is None:
            continue
        fill_row_dict_list = _enrich_fill_row_dict_list(
            state_store_obj.get_fill_row_dict_list_for_vplan(int(latest_vplan_obj.vplan_id_int))
        )
        if len(fill_row_dict_list) == 0:
            continue
        slippage_notional_float_list = [
            float(fill_row_dict["slippage_notional_float"])
            for fill_row_dict in fill_row_dict_list
            if fill_row_dict.get("slippage_notional_float") is not None
        ]
        execution_report_dict_list.append(
            {
                "pod_id_str": release_obj.pod_id_str,
                "latest_vplan_id_int": latest_vplan_obj.vplan_id_int,
                "fill_count_int": len(fill_row_dict_list),
                "fill_with_open_count_int": sum(
                    1 for fill_row_dict in fill_row_dict_list if fill_row_dict["official_open_price_float"] is not None
                ),
                "aggregate_slippage_notional_float": (
                    None
                    if len(slippage_notional_float_list) == 0
                    else sum(slippage_notional_float_list)
                ),
                "fill_row_dict_list": fill_row_dict_list,
            }
        )
    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "execution_report_dict_list": execution_report_dict_list,
    }


def _load_backtest_reference_maps_dict(
    reference_strategy_pickle_path_str: str | None,
) -> dict[str, object]:
    if reference_strategy_pickle_path_str is None:
        return {
            "transaction_map_dict": {},
            "equity_by_date_dict": {},
            "cash_by_date_dict": {},
        }
    return reference_compare.load_reference_maps_from_pickle(reference_strategy_pickle_path_str)


def _target_session_date_str_from_vplan(
    vplan_obj: VPlan,
    release_obj: LiveRelease,
) -> str:
    target_session_label_ts = scheduler_utils.session_label_from_timestamp_ts(
        vplan_obj.target_execution_timestamp_ts,
        release_obj.session_calendar_id_str,
    )
    if target_session_label_ts is None:
        raise RuntimeError(
            "Cannot compare VPlan because target_execution_timestamp_ts does not map to a trading session."
        )
    return target_session_label_ts.date().isoformat()


def _market_date_str_from_timestamp_str(
    *,
    timestamp_str: str,
    release_obj: LiveRelease,
) -> str:
    timestamp_ts = datetime.fromisoformat(timestamp_str)
    if timestamp_ts.tzinfo is None:
        timestamp_ts = timestamp_ts.replace(tzinfo=UTC)
    market_timestamp_ts = scheduler_utils.to_market_timestamp_ts(
        timestamp_ts,
        release_obj.session_calendar_id_str,
    )
    return market_timestamp_ts.date().isoformat()


def _pod_has_unresolved_execution_bool(
    state_store_obj: LiveStateStore,
    pod_id_str: str,
) -> bool:
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod(pod_id_str)
    return latest_vplan_obj is not None and latest_vplan_obj.status_str in ("submitted", "submitting")


def _pod_has_stage_snapshot_for_market_date_bool(
    *,
    state_store_obj: LiveStateStore,
    release_obj: LiveRelease,
    snapshot_stage_str: str,
    market_date_str: str,
) -> bool:
    for history_row_dict in state_store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str):
        if str(history_row_dict.get("snapshot_stage_str", "unknown")) != snapshot_stage_str:
            continue
        # *** CRITICAL*** Snapshot date matching uses the release market
        # calendar date, not a UTC string prefix, so EOD equity is aligned to
        # the same session label as the reference backtest close.
        history_market_date_str = _market_date_str_from_timestamp_str(
            timestamp_str=str(history_row_dict["updated_timestamp_str"]),
            release_obj=release_obj,
        )
        if history_market_date_str == market_date_str:
            return True
    return False


def _latest_stage_snapshot_for_market_date_dict(
    *,
    state_store_obj: LiveStateStore,
    release_obj: LiveRelease,
    snapshot_stage_str: str,
    market_date_str: str,
) -> dict[str, object] | None:
    matched_row_dict_list: list[dict[str, object]] = []
    for history_row_dict in state_store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str):
        if str(history_row_dict.get("snapshot_stage_str", "unknown")) != snapshot_stage_str:
            continue
        # *** CRITICAL*** Equity comparison must choose the EOD snapshot by
        # market session date. Comparing against report wall-clock date can
        # mix close-marked reference equity with the wrong broker sample.
        history_market_date_str = _market_date_str_from_timestamp_str(
            timestamp_str=str(history_row_dict["updated_timestamp_str"]),
            release_obj=release_obj,
        )
        if history_market_date_str == market_date_str:
            matched_row_dict_list.append(dict(history_row_dict))
    if len(matched_row_dict_list) == 0:
        return None
    return sorted(
        matched_row_dict_list,
        key=lambda row_dict: (
            str(row_dict["updated_timestamp_str"]),
            int(row_dict["pod_state_history_id_int"]),
        ),
    )[-1]


def _snapshot_source_str_for_mode(env_mode_str: str) -> str:
    if env_mode_str == "incubation":
        return "virtual_broker"
    return "broker"


def _eod_snapshot_market_date_str(
    *,
    release_obj: LiveRelease,
    as_of_ts: datetime,
) -> str:
    market_timestamp_ts = scheduler_utils.to_market_timestamp_ts(
        as_of_ts,
        release_obj.session_calendar_id_str,
    )
    return market_timestamp_ts.date().isoformat()


def _eod_snapshot_due_timestamp_ts(
    *,
    release_obj: LiveRelease,
    as_of_ts: datetime,
    eod_snapshot_buffer_minutes_int: int = DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT,
) -> datetime | None:
    session_label_ts = scheduler_utils.session_label_from_timestamp_ts(
        as_of_ts,
        release_obj.session_calendar_id_str,
    )
    if session_label_ts is None:
        return None
    # *** CRITICAL*** Automatic EOD sampling is anchored to the current
    # market session close plus a buffer. It must not sample before the close
    # and then label the account state as close-of-day equity.
    return scheduler_utils.get_session_close_timestamp_ts(
        session_label_ts,
        release_obj.session_calendar_id_str,
    ) + timedelta(minutes=int(eod_snapshot_buffer_minutes_int))


def _resolve_actual_state_for_compare_dict(
    *,
    state_store_obj: LiveStateStore,
    release_obj: LiveRelease,
    target_session_date_str: str,
) -> dict[str, object | None]:
    eod_history_row_dict = _latest_stage_snapshot_for_market_date_dict(
        state_store_obj=state_store_obj,
        release_obj=release_obj,
        snapshot_stage_str="eod",
        market_date_str=target_session_date_str,
    )
    if eod_history_row_dict is not None:
        # *** CRITICAL*** Prefer same-session EOD equity for reference
        # comparison because the reference backtest equity is close-marked.
        # Post-open NetLiq and close-marked backtest equity are different
        # measurement bases.
        return {
            "actual_equity_float": float(eod_history_row_dict["total_value_float"]),
            "actual_cash_float": float(eod_history_row_dict["cash_float"]),
            "actual_equity_source_str": "pod_state_history.eod",
            "actual_equity_basis_str": "eod_broker_netliq",
            "actual_equity_timestamp_str": str(eod_history_row_dict["updated_timestamp_str"]),
        }

    pod_state_obj = state_store_obj.get_pod_state(release_obj.pod_id_str)
    return {
        "actual_equity_float": None if pod_state_obj is None else float(pod_state_obj.total_value_float),
        "actual_cash_float": None if pod_state_obj is None else float(pod_state_obj.cash_float),
        "actual_equity_source_str": "pod_state.latest" if pod_state_obj is not None else None,
        "actual_equity_basis_str": "latest_broker_netliq" if pod_state_obj is not None else None,
        "actual_equity_timestamp_str": None if pod_state_obj is None else pod_state_obj.updated_timestamp_ts.isoformat(),
    }


def get_compare_reference_summary(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str = "paper",
    pod_id_str: str | None = None,
    reference_strategy_pickle_path_str: str | None = None,
    html_output_bool: bool = False,
    output_dir_str: str = "results",
) -> dict[str, object]:
    _load_release_list_and_sync(
        releases_root_path_str,
        state_store_obj,
        pod_id_str=pod_id_str,
    )
    compare_report_dict_list: list[dict[str, object]] = []
    for release_obj in state_store_obj.get_enabled_release_list():
        if release_obj.mode_str != env_mode_str:
            continue
        if pod_id_str is not None and release_obj.pod_id_str != pod_id_str:
            continue

        latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
        if latest_vplan_obj is None or latest_vplan_obj.vplan_id_int is None:
            continue
        first_vplan_obj = state_store_obj.get_first_vplan_for_pod(release_obj.pod_id_str)
        if first_vplan_obj is None:
            continue

        # *** CRITICAL*** Deployment reference starts on the first target
        # execution session, not the signal date. The live contract is:
        # decision_t = f(I_{t-1}); fill_t = Open_t.
        deployment_start_date_str = _target_session_date_str_from_vplan(first_vplan_obj, release_obj)
        deployment_initial_cash_float = float(first_vplan_obj.pod_budget_float)
        target_session_date_str = _target_session_date_str_from_vplan(latest_vplan_obj, release_obj)

        reference_output_dir_path_obj = reference_compare.build_reference_output_dir_path(
            output_dir_str=output_dir_str,
            env_mode_str=env_mode_str,
            pod_id_str=release_obj.pod_id_str,
            as_of_ts=as_of_ts,
        )
        reference_strategy_pickle_path_for_report_str = reference_strategy_pickle_path_str
        if reference_strategy_pickle_path_str is None:
            reference_output_dir_path_obj.mkdir(parents=True, exist_ok=True)
            reference_strategy_obj = reference_compare.run_auto_reference_strategy(
                release_obj=release_obj,
                deployment_start_date_str=deployment_start_date_str,
                reference_end_date_str=target_session_date_str,
                deployment_initial_cash_float=deployment_initial_cash_float,
                output_dir_path_obj=reference_output_dir_path_obj,
            )
            reference_strategy_pickle_path_for_report_str = str(
                reference_output_dir_path_obj / "reference_strategy.pkl"
            )
            reference_strategy_obj.to_pickle(reference_strategy_pickle_path_for_report_str)
            reference_maps_dict = reference_compare.build_reference_maps_dict(reference_strategy_obj)
        else:
            reference_maps_dict = _load_backtest_reference_maps_dict(reference_strategy_pickle_path_str)

        transaction_map_dict = dict(reference_maps_dict["transaction_map_dict"])
        equity_by_date_dict = dict(reference_maps_dict["equity_by_date_dict"])
        cash_by_date_dict = dict(reference_maps_dict.get("cash_by_date_dict", {}))
        latest_broker_snapshot_obj = state_store_obj.get_latest_broker_snapshot_for_account(
            latest_vplan_obj.account_route_str
        )
        post_execution_history_row_dict = _latest_stage_snapshot_for_market_date_dict(
            state_store_obj=state_store_obj,
            release_obj=release_obj,
            snapshot_stage_str="post_execution",
            market_date_str=target_session_date_str,
        )
        # *** CRITICAL*** Execution-row position comparison prefers the
        # post-execution snapshot for the target session. EOD can update
        # equity/cash for close-marked comparison, but fills and order-plan
        # reconciliation stay anchored to post-execution evidence.
        if post_execution_history_row_dict is not None:
            broker_position_map_dict = {
                str(asset_str): float(share_float)
                for asset_str, share_float in dict(
                    post_execution_history_row_dict["position_amount_map"]
                ).items()
            }
        elif latest_broker_snapshot_obj is None:
            broker_position_map_dict = latest_vplan_obj.current_broker_position_map
        else:
            broker_position_map_dict = latest_broker_snapshot_obj.position_amount_map
        fill_row_dict_list = _enrich_fill_row_dict_list(
            state_store_obj.get_fill_row_dict_list_for_vplan(int(latest_vplan_obj.vplan_id_int))
        )
        broker_order_row_dict_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(
            int(latest_vplan_obj.vplan_id_int)
        )
        execution_row_dict_list = _build_execution_row_dict_list(
            vplan_obj=latest_vplan_obj,
            broker_position_map_dict=broker_position_map_dict,
            broker_order_row_dict_list=broker_order_row_dict_list,
            fill_row_dict_list=fill_row_dict_list,
        )

        # *** CRITICAL*** Backtest/reference matching is anchored to the target
        # execution session, not the report timestamp, to avoid comparing fills
        # against a later date.
        actual_state_dict = _resolve_actual_state_for_compare_dict(
            state_store_obj=state_store_obj,
            release_obj=release_obj,
            target_session_date_str=target_session_date_str,
        )
        actual_equity_float = actual_state_dict["actual_equity_float"]
        actual_cash_float = actual_state_dict["actual_cash_float"]
        backtest_equity_float = reference_compare.value_on_or_before_date_float(
            equity_by_date_dict,
            target_session_date_str,
        )
        backtest_cash_float = reference_compare.value_on_or_before_date_float(
            cash_by_date_dict,
            target_session_date_str,
        )
        equity_tracking_error_float = None
        if actual_equity_float is not None and backtest_equity_float is not None and backtest_equity_float != 0.0:
            equity_tracking_error_float = (actual_equity_float / backtest_equity_float) - 1.0
        cash_diff_float = None
        if actual_cash_float is not None and backtest_cash_float is not None:
            cash_diff_float = actual_cash_float - backtest_cash_float

        reference_position_map_dict = reference_compare.get_reference_position_map_dict(
            reference_maps_dict,
            target_session_date_str,
        )
        reference_realized_weight_map_dict = reference_compare.get_reference_realized_weight_map_dict(
            reference_maps_dict,
            target_session_date_str,
        )
        decision_plan_obj = state_store_obj.get_decision_plan_by_id(int(latest_vplan_obj.decision_plan_id_int))
        live_target_weight_map_dict = (
            decision_plan_obj.full_target_weight_map_dict
            if decision_plan_obj.decision_book_type_str == "full_target_weight_book"
            else decision_plan_obj.entry_target_weight_map_dict
        )

        compare_row_dict_list: list[dict[str, object]] = []
        for execution_row_dict in execution_row_dict_list:
            asset_str = str(execution_row_dict["asset_str"])
            planned_order_delta_share_float = float(
                execution_row_dict["planned_order_delta_share_float"]
            )
            filled_share_float = float(execution_row_dict["filled_share_float"])
            target_position_float = float(execution_row_dict["target_share_float"])
            actual_position_float = float(execution_row_dict["broker_share_float"])
            avg_fill_price_float = execution_row_dict.get("avg_fill_price_float")
            reference_price_float = execution_row_dict.get("official_open_price_float")
            if reference_price_float is None:
                reference_price_float = execution_row_dict.get("live_reference_price_float")
            side_float = _signed_execution_direction_float_from_amount(
                planned_order_delta_share_float
            )
            fill_slippage_bps_float = None
            if avg_fill_price_float is not None and reference_price_float is not None:
                fill_slippage_bps_float = _compute_execution_cost_bps_float(
                    execution_price_float=float(avg_fill_price_float),
                    official_open_price_float=float(reference_price_float),
                    signed_execution_direction_float=side_float,
                )

            reference_transaction_dict = transaction_map_dict.get(
                (target_session_date_str, asset_str),
                {},
            )
            backtest_quantity_diff_float = None
            backtest_fill_price_diff_float = None
            if "quantity_float" in reference_transaction_dict:
                backtest_quantity_diff_float = filled_share_float - float(
                    reference_transaction_dict["quantity_float"]
                )
            if (
                avg_fill_price_float is not None
                and reference_transaction_dict.get("avg_price_float") is not None
            ):
                backtest_fill_price_diff_float = float(avg_fill_price_float) - float(
                    reference_transaction_dict["avg_price_float"]
                )

            reference_position_float = float(reference_position_map_dict.get(asset_str, 0.0))
            reference_position_diff_float = actual_position_float - reference_position_float
            live_target_weight_float = float(live_target_weight_map_dict.get(asset_str, 0.0))
            reference_realized_weight_float = float(reference_realized_weight_map_dict.get(asset_str, 0.0))
            weight_diff_float = live_target_weight_float - reference_realized_weight_float

            compare_row_dict_list.append(
                {
                    "asset_str": asset_str,
                    "live_target_weight_float": live_target_weight_float,
                    "reference_realized_weight_float": reference_realized_weight_float,
                    "weight_diff_float": weight_diff_float,
                    "planned_order_delta_share_float": planned_order_delta_share_float,
                    "filled_share_float": filled_share_float,
                    "quantity_diff_float": filled_share_float - planned_order_delta_share_float,
                    "target_position_float": target_position_float,
                    "actual_position_float": actual_position_float,
                    "reference_position_float": reference_position_float,
                    "position_diff_float": actual_position_float - target_position_float,
                    "reference_position_diff_float": reference_position_diff_float,
                    "avg_fill_price_float": avg_fill_price_float,
                    "reference_price_float": reference_price_float,
                    "fill_slippage_bps_float": fill_slippage_bps_float,
                    "backtest_quantity_diff_float": backtest_quantity_diff_float,
                    "backtest_fill_price_diff_float": backtest_fill_price_diff_float,
                }
            )

        compare_report_dict = {
            "pod_id_str": release_obj.pod_id_str,
            "release_id_str": release_obj.release_id_str,
            "vplan_id_int": int(latest_vplan_obj.vplan_id_int),
            "deployment_start_date_str": deployment_start_date_str,
            "deployment_initial_cash_float": deployment_initial_cash_float,
            "target_session_date_str": target_session_date_str,
            "actual_equity_float": actual_equity_float,
            "backtest_equity_float": backtest_equity_float,
            "equity_tracking_error_float": equity_tracking_error_float,
            "actual_cash_float": actual_cash_float,
            "actual_equity_source_str": actual_state_dict["actual_equity_source_str"],
            "actual_equity_basis_str": actual_state_dict["actual_equity_basis_str"],
            "actual_equity_timestamp_str": actual_state_dict["actual_equity_timestamp_str"],
            "backtest_cash_float": backtest_cash_float,
            "cash_diff_float": cash_diff_float,
            "reference_strategy_pickle_path_str": reference_strategy_pickle_path_for_report_str,
            "reference_auto_generated_bool": reference_strategy_pickle_path_str is None,
            "compare_row_dict_list": compare_row_dict_list,
            "cash_ledger_row_dict_list": state_store_obj.get_cash_ledger_row_dict_list_for_vplan(
                int(latest_vplan_obj.vplan_id_int)
            ),
        }
        compare_report_dict.update(reference_compare.classify_compare_status_dict(compare_report_dict))

        if html_output_bool:
            live_history_row_dict_list = state_store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str)
            compare_report_dict["artifact_path_dict"] = reference_compare.write_reference_compare_artifacts(
                output_dir_path_obj=reference_output_dir_path_obj,
                release_obj=release_obj,
                compare_report_dict=compare_report_dict,
                reference_maps_dict=reference_maps_dict,
                live_history_row_dict_list=live_history_row_dict_list,
                reference_strategy_pickle_path_str=reference_strategy_pickle_path_for_report_str,
            )

        compare_report_dict_list.append(
            compare_report_dict
        )

    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "mode_str": env_mode_str,
        "reference_strategy_pickle_path_str": reference_strategy_pickle_path_str,
        "reference_auto_generated_bool": reference_strategy_pickle_path_str is None,
        "html_output_bool": bool(html_output_bool),
        "output_dir_str": output_dir_str,
        "compare_report_dict_list": compare_report_dict_list,
    }


def preflight_contract_summary(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str | None = None,
    pod_id_str: str | None = None,
) -> dict[str, object]:
    release_list = _load_release_list_and_sync(
        releases_root_path_str,
        state_store_obj,
        pod_id_str=pod_id_str,
    )
    enabled_release_list = [
        release_obj
        for release_obj in release_list
        if release_obj.enabled_bool
        and (env_mode_str is None or release_obj.mode_str == env_mode_str)
    ]
    contract_report_dict_list: list[dict[str, object]] = []
    passed_release_count_int = 0
    failed_release_count_int = 0

    for release_obj in enabled_release_list:
        pod_state_obj = _get_model_state_or_default(release_obj, state_store_obj, as_of_ts)
        contract_report_dict = strategy_host.preflight_decision_contract_for_release(
            release_obj=release_obj,
            as_of_ts=as_of_ts,
            pod_state_obj=pod_state_obj,
        )
        contract_report_dict_list.append(contract_report_dict)
        if contract_report_dict["contract_status_str"] == "pass":
            passed_release_count_int += 1
        else:
            failed_release_count_int += 1

    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "mode_str": env_mode_str,
        "enabled_release_count_int": len(enabled_release_list),
        "passed_release_count_int": passed_release_count_int,
        "failed_release_count_int": failed_release_count_int,
        "contract_report_dict_list": contract_report_dict_list,
    }


def cutover_v1_schema(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    db_path_str: str,
    archive_root_path_str: str | None = None,
) -> dict[str, object]:
    archive_dir_path_obj = _build_v1_cutover_archive_dir_path_obj(
        db_path_str=db_path_str,
        as_of_ts=as_of_ts,
        archive_root_path_str=archive_root_path_str,
    )
    archive_dir_path_obj.mkdir(parents=True, exist_ok=True)
    timestamp_label_str = as_of_ts.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    existing_table_name_set = set(state_store_obj.get_existing_table_name_list())

    table_export_dict_list: list[dict[str, object]] = []
    existing_v1_table_name_list = [
        table_name_str
        for table_name_str in V1_EXECUTION_TABLE_NAME_TUPLE
        if table_name_str in existing_table_name_set
    ]
    for table_name_str in existing_v1_table_name_list:
        source_row_count_int = state_store_obj.get_table_row_count_int(table_name_str)
        row_dict_list = state_store_obj.get_table_row_dict_list(table_name_str)
        exported_row_count_int = len(row_dict_list)
        if source_row_count_int != exported_row_count_int:
            raise RuntimeError(
                "V1 archive row-count mismatch: "
                f"table_name_str={table_name_str} "
                f"source_row_count_int={source_row_count_int} "
                f"exported_row_count_int={exported_row_count_int}"
            )
        archive_file_path_obj = archive_dir_path_obj / f"{timestamp_label_str}_{table_name_str}.json"
        archive_file_path_obj.write_text(
            json.dumps(row_dict_list, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        table_export_dict_list.append(
            {
                "table_name_str": table_name_str,
                "source_row_count_int": source_row_count_int,
                "exported_row_count_int": exported_row_count_int,
                "archive_file_path_str": str(archive_file_path_obj),
            }
        )

    manifest_dict = {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "db_path_str": str(Path(db_path_str).resolve()),
        "existing_v1_table_name_list": existing_v1_table_name_list,
        "table_export_dict_list": table_export_dict_list,
    }
    manifest_path_obj = archive_dir_path_obj / f"{timestamp_label_str}_manifest.json"
    manifest_path_obj.write_text(
        json.dumps(manifest_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    dropped_table_name_list: list[str] = []
    for table_name_str in V1_EXECUTION_TABLE_NAME_TUPLE:
        if table_name_str not in existing_table_name_set:
            continue
        state_store_obj.drop_table_if_exists(table_name_str)
        dropped_table_name_list.append(table_name_str)

    remaining_table_name_list = state_store_obj.get_existing_table_name_list()
    remaining_table_name_set = set(remaining_table_name_list)
    remaining_v1_table_name_list = [
        table_name_str
        for table_name_str in V1_EXECUTION_TABLE_NAME_TUPLE
        if table_name_str in remaining_table_name_set
    ]
    if len(remaining_v1_table_name_list) > 0:
        raise RuntimeError(
            "V1 cutover left tables behind: "
            + ", ".join(remaining_v1_table_name_list)
        )

    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "db_path_str": str(Path(db_path_str).resolve()),
        "archive_dir_path_str": str(archive_dir_path_obj),
        "manifest_path_str": str(manifest_path_obj),
        "existing_v1_table_name_list": existing_v1_table_name_list,
        "exported_table_count_int": len(table_export_dict_list),
        "dropped_table_count_int": len(dropped_table_name_list),
        "table_export_dict_list": table_export_dict_list,
        "dropped_table_name_list": dropped_table_name_list,
        "remaining_table_name_list": remaining_table_name_list,
        "remaining_v1_table_name_list": remaining_v1_table_name_list,
    }


def tick(
    state_store_obj: LiveStateStore,
    broker_adapter_obj: BrokerAdapter | None,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
    broker_adapter_resolver_obj: BrokerAdapterResolver | None = None,
    broker_host_str: str | None = None,
    broker_port_int: int | None = None,
    broker_client_id_int: int | None = None,
    broker_timeout_seconds_float: float | None = None,
    adapter_factory_func: Callable[[str, int, int, float], BrokerAdapter] | None = None,
    pod_id_str: str | None = None,
) -> dict[str, object]:
    _validate_release_root_for_mutation(releases_root_path_str, env_mode_str, pod_id_str=pod_id_str)
    lease_owner_token_str = uuid.uuid4().hex
    lease_acquired_bool = state_store_obj.acquire_scheduler_lease(
        lease_name_str="tick",
        owner_token_str=lease_owner_token_str,
        expires_timestamp_ts=scheduler_utils.utc_now_ts() + timedelta(minutes=5),
    )
    if not lease_acquired_bool:
        return {
            "lease_acquired_bool": False,
            "eod_snapshot_count_int": 0,
            "skipped_snapshot_count_int": 0,
            "created_decision_plan_count_int": 0,
            "skipped_decision_plan_count_int": 0,
            "expired_decision_plan_count_int": 0,
            "created_vplan_count_int": 0,
            "submitted_vplan_count_int": 0,
            "blocked_action_count_int": 0,
            "completed_vplan_count_int": 0,
            "warning_count_map_dict": {},
        }

    try:
        broker_adapter_resolver_obj = _coerce_broker_adapter_resolver_obj(
            broker_adapter_obj=broker_adapter_obj,
            broker_adapter_resolver_obj=broker_adapter_resolver_obj,
            broker_host_str=broker_host_str,
            broker_port_int=broker_port_int,
            broker_client_id_int=broker_client_id_int,
            broker_timeout_seconds_float=broker_timeout_seconds_float,
            state_store_obj=state_store_obj,
            as_of_ts=as_of_ts,
            adapter_factory_func=adapter_factory_func,
        )
        build_detail_dict = build_decision_plans(
            state_store_obj=state_store_obj,
            as_of_ts=as_of_ts,
            releases_root_path_str=releases_root_path_str,
            env_mode_str=env_mode_str,
            log_path_str=log_path_str,
            pod_id_str=pod_id_str,
        )
        expire_detail_dict = expire_stale_decision_plans(
            state_store_obj=state_store_obj,
            as_of_ts=as_of_ts,
            releases_root_path_str=releases_root_path_str,
            env_mode_str=env_mode_str,
            log_path_str=log_path_str,
            pod_id_str=pod_id_str,
        )
        vplan_detail_dict = build_vplans(
            state_store_obj=state_store_obj,
            broker_adapter_obj=broker_adapter_obj,
            as_of_ts=as_of_ts,
            env_mode_str=env_mode_str,
            releases_root_path_str=releases_root_path_str,
            log_path_str=log_path_str,
            broker_adapter_resolver_obj=broker_adapter_resolver_obj,
            pod_id_str=pod_id_str,
        )
        submit_detail_dict = submit_ready_vplans(
            state_store_obj=state_store_obj,
            broker_adapter_obj=broker_adapter_obj,
            as_of_ts=as_of_ts,
            env_mode_str=env_mode_str,
            manual_only_bool=False,
            releases_root_path_str=releases_root_path_str,
            log_path_str=log_path_str,
            broker_adapter_resolver_obj=broker_adapter_resolver_obj,
            pod_id_str=pod_id_str,
        )
        reconcile_detail_dict = post_execution_reconcile(
            state_store_obj=state_store_obj,
            broker_adapter_obj=broker_adapter_obj,
            as_of_ts=as_of_ts,
            env_mode_str=env_mode_str,
            releases_root_path_str=releases_root_path_str,
            log_path_str=log_path_str,
            broker_adapter_resolver_obj=broker_adapter_resolver_obj,
            pod_id_str=pod_id_str,
        )
        warning_counter_obj: Counter[str] = Counter()
        reason_counter_obj: Counter[str] = Counter()
        for detail_piece_dict in (
            build_detail_dict,
            expire_detail_dict,
            vplan_detail_dict,
            submit_detail_dict,
        ):
            warning_counter_obj.update(detail_piece_dict.get("warning_count_map_dict", {}))
            reason_counter_obj.update(detail_piece_dict.get("reason_count_map_dict", {}))
        return {
            "lease_acquired_bool": True,
            "eod_snapshot_count_int": 0,
            "skipped_snapshot_count_int": 0,
            "created_decision_plan_count_int": int(build_detail_dict.get("created_decision_plan_count_int", 0)),
            "skipped_decision_plan_count_int": int(build_detail_dict.get("skipped_decision_plan_count_int", 0)),
            "expired_decision_plan_count_int": int(expire_detail_dict.get("expired_decision_plan_count_int", 0)),
            "created_vplan_count_int": int(vplan_detail_dict.get("created_vplan_count_int", 0)),
            "submitted_vplan_count_int": int(submit_detail_dict.get("submitted_vplan_count_int", 0)),
            "blocked_action_count_int": int(vplan_detail_dict.get("blocked_action_count_int", 0))
            + int(submit_detail_dict.get("blocked_action_count_int", 0)),
            "completed_vplan_count_int": int(reconcile_detail_dict.get("completed_vplan_count_int", 0)),
            "warning_count_map_dict": dict(warning_counter_obj),
            "reason_count_map_dict": dict(reason_counter_obj),
        }
    finally:
        state_store_obj.release_scheduler_lease(
            lease_name_str="tick",
            owner_token_str=lease_owner_token_str,
        )


def main(argv_list: list[str] | None = None) -> int:
    parser_obj = argparse.ArgumentParser(description="alpha.live runner")
    parser_obj.add_argument(
        "command_name_str",
        choices=(
            "build_decision_plans",
            "build_vplan",
            "show_vplan",
            "submit_vplan",
            "post_execution_reconcile",
            "eod_snapshot",
            "tick",
            "status",
            "execution_report",
            "compare_reference",
            "preflight_contract",
            "cutover_v1_schema",
        ),
    )
    parser_obj.add_argument("--db-path", dest="db_path_str", default=None)
    parser_obj.add_argument("--releases-root", dest="releases_root_path_str", default=DEFAULT_RELEASES_ROOT_PATH_STR)
    parser_obj.add_argument("--as-of-ts", dest="as_of_timestamp_str", default=None)
    parser_obj.add_argument("--mode", dest="env_mode_str", default="paper")
    parser_obj.add_argument("--log-path", dest="log_path_str", default=DEFAULT_LOG_PATH_STR)
    parser_obj.add_argument("--archive-root", dest="archive_root_path_str", default=None)
    parser_obj.add_argument("--json", dest="json_output_bool", action="store_true")
    parser_obj.add_argument("--vplan-id", dest="vplan_id_int", type=int, default=None)
    parser_obj.add_argument("--pod-id", dest="pod_id_str", default=None)
    parser_obj.add_argument("--broker-host", dest="broker_host_str", default=None)
    parser_obj.add_argument("--broker-port", dest="broker_port_int", type=int, default=None)
    parser_obj.add_argument("--broker-client-id", dest="broker_client_id_int", type=int, default=None)
    parser_obj.add_argument("--broker-timeout-seconds", dest="broker_timeout_seconds_float", type=float, default=None)
    parser_obj.add_argument("--reference-strategy-pickle", dest="reference_strategy_pickle_path_str", default=None)
    parser_obj.add_argument("--html", dest="html_output_bool", action="store_true")
    parser_obj.add_argument("--output-dir", dest="output_dir_str", default="results")
    parsed_args_obj = parser_obj.parse_args(argv_list)

    db_path_str = _resolve_db_path_for_mode_str(
        db_path_str=parsed_args_obj.db_path_str,
        env_mode_str=parsed_args_obj.env_mode_str,
        pod_id_str=parsed_args_obj.pod_id_str,
    )
    state_store_obj = LiveStateStore(db_path_str)
    job_run_id_int = state_store_obj.record_job_start(parsed_args_obj.command_name_str)
    as_of_ts = _parse_as_of_timestamp_ts(parsed_args_obj.as_of_timestamp_str)
    broker_adapter_obj = None

    try:
        if parsed_args_obj.command_name_str == "build_decision_plans":
            detail_dict = build_decision_plans(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                env_mode_str=parsed_args_obj.env_mode_str,
                log_path_str=parsed_args_obj.log_path_str,
                pod_id_str=parsed_args_obj.pod_id_str,
            )
        elif parsed_args_obj.command_name_str == "build_vplan":
            detail_dict = build_vplans(
                state_store_obj=state_store_obj,
                broker_adapter_obj=broker_adapter_obj,
                as_of_ts=as_of_ts,
                env_mode_str=parsed_args_obj.env_mode_str,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                log_path_str=parsed_args_obj.log_path_str,
                broker_host_str=parsed_args_obj.broker_host_str,
                broker_port_int=parsed_args_obj.broker_port_int,
                broker_client_id_int=parsed_args_obj.broker_client_id_int,
                broker_timeout_seconds_float=parsed_args_obj.broker_timeout_seconds_float,
                pod_id_str=parsed_args_obj.pod_id_str,
            )
        elif parsed_args_obj.command_name_str == "submit_vplan":
            detail_dict = submit_ready_vplans(
                state_store_obj=state_store_obj,
                broker_adapter_obj=broker_adapter_obj,
                as_of_ts=as_of_ts,
                env_mode_str=parsed_args_obj.env_mode_str,
                manual_only_bool=False,
                vplan_id_int=parsed_args_obj.vplan_id_int,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                log_path_str=parsed_args_obj.log_path_str,
                broker_host_str=parsed_args_obj.broker_host_str,
                broker_port_int=parsed_args_obj.broker_port_int,
                broker_client_id_int=parsed_args_obj.broker_client_id_int,
                broker_timeout_seconds_float=parsed_args_obj.broker_timeout_seconds_float,
                pod_id_str=parsed_args_obj.pod_id_str,
            )
        elif parsed_args_obj.command_name_str == "post_execution_reconcile":
            detail_dict = post_execution_reconcile(
                state_store_obj=state_store_obj,
                broker_adapter_obj=broker_adapter_obj,
                as_of_ts=as_of_ts,
                env_mode_str=parsed_args_obj.env_mode_str,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                log_path_str=parsed_args_obj.log_path_str,
                broker_host_str=parsed_args_obj.broker_host_str,
                broker_port_int=parsed_args_obj.broker_port_int,
                broker_client_id_int=parsed_args_obj.broker_client_id_int,
                broker_timeout_seconds_float=parsed_args_obj.broker_timeout_seconds_float,
                pod_id_str=parsed_args_obj.pod_id_str,
            )
        elif parsed_args_obj.command_name_str == "eod_snapshot":
            detail_dict = eod_snapshot(
                state_store_obj=state_store_obj,
                broker_adapter_obj=broker_adapter_obj,
                as_of_ts=as_of_ts,
                env_mode_str=parsed_args_obj.env_mode_str,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                log_path_str=parsed_args_obj.log_path_str,
                broker_host_str=parsed_args_obj.broker_host_str,
                broker_port_int=parsed_args_obj.broker_port_int,
                broker_client_id_int=parsed_args_obj.broker_client_id_int,
                broker_timeout_seconds_float=parsed_args_obj.broker_timeout_seconds_float,
                pod_id_str=parsed_args_obj.pod_id_str,
            )
        elif parsed_args_obj.command_name_str == "tick":
            detail_dict = tick(
                state_store_obj=state_store_obj,
                broker_adapter_obj=broker_adapter_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                env_mode_str=parsed_args_obj.env_mode_str,
                log_path_str=parsed_args_obj.log_path_str,
                broker_host_str=parsed_args_obj.broker_host_str,
                broker_port_int=parsed_args_obj.broker_port_int,
                broker_client_id_int=parsed_args_obj.broker_client_id_int,
                broker_timeout_seconds_float=parsed_args_obj.broker_timeout_seconds_float,
                pod_id_str=parsed_args_obj.pod_id_str,
            )
        elif parsed_args_obj.command_name_str == "cutover_v1_schema":
            detail_dict = cutover_v1_schema(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                db_path_str=db_path_str,
                archive_root_path_str=parsed_args_obj.archive_root_path_str,
            )
        elif parsed_args_obj.command_name_str == "show_vplan":
            detail_dict = show_vplan_summary(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                vplan_id_int=parsed_args_obj.vplan_id_int,
                pod_id_str=parsed_args_obj.pod_id_str,
            )
        elif parsed_args_obj.command_name_str == "execution_report":
            detail_dict = get_execution_report_summary(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                env_mode_str=parsed_args_obj.env_mode_str,
                pod_id_str=parsed_args_obj.pod_id_str,
            )
        elif parsed_args_obj.command_name_str == "compare_reference":
            detail_dict = get_compare_reference_summary(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                env_mode_str=parsed_args_obj.env_mode_str,
                pod_id_str=parsed_args_obj.pod_id_str,
                reference_strategy_pickle_path_str=parsed_args_obj.reference_strategy_pickle_path_str,
                html_output_bool=parsed_args_obj.html_output_bool,
                output_dir_str=parsed_args_obj.output_dir_str,
            )
        elif parsed_args_obj.command_name_str == "preflight_contract":
            detail_dict = preflight_contract_summary(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                env_mode_str=parsed_args_obj.env_mode_str,
                pod_id_str=parsed_args_obj.pod_id_str,
            )
        else:
            detail_dict = get_status_summary(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                env_mode_str=parsed_args_obj.env_mode_str,
                pod_id_str=parsed_args_obj.pod_id_str,
            )

        state_store_obj.record_job_finish(job_run_id_int, "completed", detail_dict)
        if parsed_args_obj.json_output_bool:
            print(json.dumps(detail_dict, indent=2, sort_keys=True))
        else:
            print(_render_command_output_str(parsed_args_obj.command_name_str, detail_dict))
    except Exception as exc:  # pragma: no cover - CLI safety
        error_detail_dict = {"error_str": str(exc)}
        state_store_obj.record_job_finish(job_run_id_int, "failed", error_detail_dict)
        log_event(
            "runner_failed",
            {
                "command_name_str": parsed_args_obj.command_name_str,
                "as_of_timestamp_str": as_of_ts.isoformat(),
                "error_str": str(exc),
            },
            log_path_str=parsed_args_obj.log_path_str,
        )
        raise

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
