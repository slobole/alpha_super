"""
LiveRelease = what to run
FrozenOrderIntent = one planned trade
FrozenOrderPlan = the full saved decision
BrokerPositionSnapshot = what the broker currently has
BrokerOrderRequest = what we want to send
BrokerOrderRecord = what we sent
BrokerOrderFill = what got filled
ReconciliationResult = did model and broker match?
PodState = saved live memory of the pod
ExecutionQualitySnapshot = derived execution analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class LiveRelease:
    release_id_str: str  # Exact live deployment id.
    user_id_str: str  # Logical owner of the release.
    pod_id_str: str  # Stable live sleeve id.
    account_route_str: str  # Broker account route.
    strategy_import_str: str  # Research strategy import path.
    mode_str: str  # paper or live.
    session_calendar_id_str: str  # Exchange calendar id, e.g. XNYS.
    signal_clock_str: str  # When the strategy may decide.
    execution_policy_str: str  # How and when orders should be sent.
    data_profile_str: str  # Data contract expected by the pod.
    params_dict: dict[str, Any]  # Strategy-specific parameters.
    risk_profile_str: str  # Risk label / policy hook.
    enabled_bool: bool  # Operational kill switch.
    source_path_str: str  # YAML manifest path on disk.


@dataclass(frozen=True)
class FrozenOrderIntent:
    asset_str: str  # Symbol to trade.
    order_class_str: str  # MarketOrder / LimitOrder / StopOrder.
    unit_str: str  # shares / value / percent.
    amount_float: float  # Signed order amount in the given unit.
    target_bool: bool  # True if amount is a target, not a delta.
    trade_id_int: int | None  # Strategy trade id, if used.
    broker_order_type_str: str  # Broker-facing order type, e.g. MOO.
    sizing_reference_price_float: float  # Price used for sizing math.
    portfolio_value_float: float  # Portfolio value used when the intent was frozen.


@dataclass(frozen=True)
class FrozenOrderPlan:
    release_id_str: str  # Release that produced this plan.
    user_id_str: str  # Owning user.
    pod_id_str: str  # Pod that owns the plan.
    account_route_str: str  # Broker account that should receive the orders.
    signal_timestamp_ts: datetime  # Decision timestamp.
    submission_timestamp_ts: datetime  # Earliest allowed submit time.
    target_execution_timestamp_ts: datetime  # Expected market execution time.
    execution_policy_str: str  # next_open_moo / same_day_moc / next_month_first_open.
    snapshot_metadata_dict: dict[str, Any]  # Audit metadata about the signal snapshot.
    strategy_state_dict: dict[str, Any]  # Strategy memory carried to the next run.
    order_intent_list: list[FrozenOrderIntent]  # Planned trades inside this plan.
    submission_key_str: str | None = None  # Stable id used to guard duplicate submits.
    status_str: str = "frozen"  # frozen / submitting / submitted / completed / blocked / failed.
    order_plan_id_int: int | None = None  # Database primary key.


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    account_route_str: str  # Broker account id.
    snapshot_timestamp_ts: datetime  # When the broker snapshot was taken.
    cash_float: float  # Broker cash balance.
    total_value_float: float  # Broker account total value.
    position_amount_map: dict[str, float]  # Current broker positions by symbol.
    open_order_id_list: list[str] = field(default_factory=list)  # Broker open order ids.


@dataclass(frozen=True)
class BrokerOrderRequest:
    order_plan_id_int: int  # Parent frozen plan id.
    order_intent_id_int: int  # Parent intent id.
    release_id_str: str  # Release that produced the request.
    pod_id_str: str  # Pod that owns the request.
    account_route_str: str  # Broker account to route to.
    submission_key_str: str  # Duplicate-submit guard key.
    asset_str: str  # Symbol to trade.
    broker_order_type_str: str  # Broker order type.
    order_class_str: str  # Original engine order class.
    unit_str: str  # shares / value / percent.
    amount_float: float  # Signed amount in the given unit.
    target_bool: bool  # True if amount is target-based.
    trade_id_int: int | None  # Strategy trade id, if used.
    sizing_reference_price_float: float  # Reference price used in sizing.
    portfolio_value_float: float  # Portfolio value used in sizing.


@dataclass(frozen=True)
class BrokerOrderRecord:
    broker_order_id_str: str  # Broker order id.
    order_plan_id_int: int  # Parent frozen plan id.
    order_intent_id_int: int  # Parent intent id.
    account_route_str: str  # Routed account.
    asset_str: str  # Symbol traded.
    broker_order_type_str: str  # Submitted broker order type.
    unit_str: str  # shares / value / percent.
    amount_float: float  # Requested order amount.
    filled_amount_float: float  # Filled quantity reported by broker.
    status_str: str  # Broker order status.
    submitted_timestamp_ts: datetime  # Submit timestamp.
    raw_payload_dict: dict[str, Any]  # Extra broker payload for audit.


@dataclass(frozen=True)
class BrokerOrderFill:
    broker_order_id_str: str  # Broker order id that produced the fill.
    order_plan_id_int: int  # Parent frozen plan id.
    account_route_str: str  # Routed account.
    asset_str: str  # Symbol filled.
    fill_amount_float: float  # Signed filled quantity.
    fill_price_float: float  # Actual broker fill price.
    fill_timestamp_ts: datetime  # Fill timestamp.
    raw_payload_dict: dict[str, Any]  # Extra broker fill payload.


@dataclass(frozen=True)
class ReconciliationResult:
    passed_bool: bool  # True if model and broker match well enough.
    status_str: str  # pass / fail style status label.
    mismatch_dict: dict[str, Any]  # Details of any mismatches found.
    model_position_map: dict[str, float]  # Model positions used in the check.
    broker_position_map: dict[str, float]  # Broker positions used in the check.
    model_cash_float: float  # Model cash used in the check.
    broker_cash_float: float  # Broker cash used in the check.


@dataclass(frozen=True)
class PodState:
    pod_id_str: str  # Pod id.
    user_id_str: str  # Owning user.
    account_route_str: str  # Broker account route.
    position_amount_map: dict[str, float]  # Saved positions by symbol.
    cash_float: float  # Saved cash balance.
    total_value_float: float  # Saved total portfolio value.
    strategy_state_dict: dict[str, Any]  # Saved strategy memory.
    updated_timestamp_ts: datetime  # Last pod-state update timestamp.


@dataclass(frozen=True)
class ExecutionQualitySnapshot:
    order_plan_id_int: int  # Parent frozen plan id.
    pod_id_str: str  # Pod id.
    reference_notional_float: float  # Model reference notional.
    actual_notional_float: float  # Realized fill notional.
    slippage_cash_float: float  # Cash slippage versus the current reference convention.
    slippage_bps_float: float  # Slippage in basis points.
    fill_count_int: int  # Number of fills used in the calculation.
    computed_timestamp_ts: datetime  # When the snapshot was computed.
