"""
LiveRelease = what to run
DecisionPlan = frozen strategy decision without final shares
VPlan = pre-submit execution plan with final shares
BrokerSnapshot = what the broker currently has
LivePriceSnapshot = pre-submit quote snapshot used for sizing
BrokerOrderRequest = what we want to send
BrokerOrderRecord = what we sent
BrokerOrderFill = what got filled
ReconciliationResult = did model and broker match?
PodState = saved live memory of the pod
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SUPPORTED_DECISION_BOOK_TYPE_TUPLE: tuple[str, ...] = (
    "incremental_entry_exit_book",
    "full_target_weight_book",
)


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
    broker_host_str: str = "127.0.0.1"  # Broker API host for this release.
    broker_port_int: int = 7497  # Broker API port for this release.
    broker_client_id_int: int = 31  # Broker API client id for this release.
    broker_timeout_seconds_float: float = 4.0  # Broker API timeout for this release.
    pod_budget_fraction_float: float = 0.03  # Fraction of broker NetLiq reserved for this pod.
    auto_submit_enabled_bool: bool = True  # True if tick may auto-submit the VPlan.


@dataclass(frozen=True)
class DecisionPlan:
    release_id_str: str  # Release that produced this decision.
    user_id_str: str  # Owning user.
    pod_id_str: str  # Pod that owns the decision.
    account_route_str: str  # Broker account route for execution.
    signal_timestamp_ts: datetime  # Timestamp of the approved signal snapshot.
    submission_timestamp_ts: datetime  # Earliest allowed submit time for the derived VPlan.
    target_execution_timestamp_ts: datetime  # Expected execution timestamp.
    execution_policy_str: str  # next_open_moo / same_day_moc / next_month_first_open.
    decision_base_position_map: dict[str, float]  # Position snapshot used by the strategy when the decision was built.
    snapshot_metadata_dict: dict[str, Any]  # Audit metadata about the signal snapshot.
    strategy_state_dict: dict[str, Any]  # Strategy memory carried to the next cycle.
    decision_book_type_str: str = "incremental_entry_exit_book"  # Explicit decision-book semantics consumed by execution.
    entry_target_weight_map_dict: dict[str, float] = field(default_factory=dict)  # Incremental entry slot weights only.
    full_target_weight_map_dict: dict[str, float] = field(default_factory=dict)  # Full target book weights for rebalance-style strategies.
    target_weight_map: dict[str, float] = field(default_factory=dict)  # Legacy alias kept during the v2 transition.
    exit_asset_set: set[str] = field(default_factory=set)  # Assets that must be driven to zero.
    entry_priority_list: list[str] = field(default_factory=list)  # Ranked entry list used when sizing the VPlan.
    cash_reserve_weight_float: float = 0.0  # Reserved cash weight left unallocated by design.
    preserve_untouched_positions_bool: bool = True  # True if omitted assets should keep current broker shares.
    rebalance_omitted_assets_to_zero_bool: bool = False  # True if omitted held assets should be liquidated in a full rebalance book.
    status_str: str = "planned"  # planned / vplan_ready / submitted / completed / expired / blocked.
    decision_plan_id_int: int | None = None  # Database primary key.

    def __post_init__(self) -> None:
        if self.decision_book_type_str not in SUPPORTED_DECISION_BOOK_TYPE_TUPLE:
            raise ValueError(
                "Unsupported decision_book_type_str "
                f"'{self.decision_book_type_str}'. Expected one of {SUPPORTED_DECISION_BOOK_TYPE_TUPLE}."
            )

        entry_target_weight_map_dict = {
            str(asset_str): float(target_weight_float)
            for asset_str, target_weight_float in dict(self.entry_target_weight_map_dict).items()
            if abs(float(target_weight_float)) > 1e-12
        }
        full_target_weight_map_dict = {
            str(asset_str): float(target_weight_float)
            for asset_str, target_weight_float in dict(self.full_target_weight_map_dict).items()
            if abs(float(target_weight_float)) > 1e-12
        }
        legacy_target_weight_map_dict = {
            str(asset_str): float(target_weight_float)
            for asset_str, target_weight_float in dict(self.target_weight_map).items()
            if abs(float(target_weight_float)) > 1e-12
        }

        if self.decision_book_type_str == "incremental_entry_exit_book":
            if len(entry_target_weight_map_dict) == 0 and len(legacy_target_weight_map_dict) > 0:
                entry_target_weight_map_dict = dict(legacy_target_weight_map_dict)
            if len(legacy_target_weight_map_dict) == 0 and len(entry_target_weight_map_dict) > 0:
                legacy_target_weight_map_dict = dict(entry_target_weight_map_dict)
            if len(full_target_weight_map_dict) > 0:
                raise ValueError(
                    "incremental_entry_exit_book must not carry full_target_weight_map_dict."
                )
        else:
            if len(full_target_weight_map_dict) == 0 and len(legacy_target_weight_map_dict) > 0:
                full_target_weight_map_dict = dict(legacy_target_weight_map_dict)
            if len(legacy_target_weight_map_dict) == 0 and len(full_target_weight_map_dict) > 0:
                legacy_target_weight_map_dict = dict(full_target_weight_map_dict)

        object.__setattr__(self, "entry_target_weight_map_dict", entry_target_weight_map_dict)
        object.__setattr__(self, "full_target_weight_map_dict", full_target_weight_map_dict)
        object.__setattr__(self, "target_weight_map", legacy_target_weight_map_dict)

    def get_execution_touched_asset_list(
        self,
        broker_position_map_dict: dict[str, float] | None = None,
        position_tolerance_float: float = 1e-9,
    ) -> list[str]:
        if self.decision_book_type_str == "incremental_entry_exit_book":
            touched_asset_set = set(self.entry_target_weight_map_dict) | set(self.exit_asset_set)
            entry_rank_map_dict = {
                asset_str: rank_idx_int
                for rank_idx_int, asset_str in enumerate(self.entry_priority_list)
            }
            return sorted(
                touched_asset_set,
                key=lambda asset_str: (
                    0 if asset_str in self.exit_asset_set else 1,
                    entry_rank_map_dict.get(asset_str, len(entry_rank_map_dict)),
                    asset_str,
                ),
            )

        touched_asset_set = set(self.full_target_weight_map_dict)
        if self.rebalance_omitted_assets_to_zero_bool and broker_position_map_dict is not None:
            touched_asset_set |= {
                str(asset_str)
                for asset_str, amount_float in broker_position_map_dict.items()
                if abs(float(amount_float)) > position_tolerance_float
            }
        return sorted(touched_asset_set)


@dataclass(frozen=True)
class VPlanRow:
    asset_str: str  # Symbol covered by this execution row.
    current_share_float: float  # Current broker share count.
    target_share_float: float  # Final target share count after sizing.
    order_delta_share_float: float  # Shares to buy (>0) or sell (<0).
    live_reference_price_float: float  # Live quote snapshot price used for sizing.
    estimated_target_notional_float: float  # Target share count multiplied by the live reference price.
    broker_order_type_str: str  # Broker order type, e.g. MOO / MOC.
    live_reference_source_str: str = ""  # Per-asset provenance for the live reference price.


@dataclass(frozen=True)
class VPlan:
    release_id_str: str  # Release that produced this execution plan.
    user_id_str: str  # Owning user.
    pod_id_str: str  # Pod that owns the execution plan.
    account_route_str: str  # Broker account route used for execution.
    decision_plan_id_int: int  # Parent decision plan id.
    signal_timestamp_ts: datetime  # Original signal timestamp from the decision plan.
    submission_timestamp_ts: datetime  # Earliest allowed submit time.
    target_execution_timestamp_ts: datetime  # Expected execution timestamp.
    execution_policy_str: str  # Execution policy used by the VPlan.
    broker_snapshot_timestamp_ts: datetime  # Timestamp of the broker account snapshot used for sizing.
    live_reference_snapshot_timestamp_ts: datetime  # Timestamp of the quote snapshot used for sizing.
    live_price_source_str: str  # Broker / quote source label used for the live prices.
    net_liq_float: float  # Broker Net Liquidation value sampled pre-submit.
    available_funds_float: float | None  # Broker AvailableFunds sampled pre-submit.
    excess_liquidity_float: float | None  # Broker ExcessLiquidity sampled pre-submit.
    pod_budget_fraction_float: float  # Fraction of NetLiq allocated to this pod.
    pod_budget_float: float  # Dollar budget allocated to this pod.
    current_broker_position_map: dict[str, float]  # Broker shares at sizing time.
    live_reference_price_map: dict[str, float]  # Price snapshot used for sizing.
    target_share_map: dict[str, float]  # Final target shares after sizing.
    order_delta_map: dict[str, float]  # Shares to send after subtracting broker positions.
    vplan_row_list: list[VPlanRow]  # Human-readable execution rows.
    live_reference_source_map_dict: dict[str, str] = field(default_factory=dict)  # Per-asset provenance for live reference prices.
    submission_key_str: str | None = None  # Stable id used to guard duplicate submits.
    status_str: str = "ready"  # ready / submitted / completed / expired / blocked.
    submit_ack_status_str: str = "not_checked"  # not_checked / complete / missing_critical.
    ack_coverage_ratio_float: float | None = None  # broker_ack_count / request_count.
    missing_ack_count_int: int = 0  # Number of outbound requests with no broker-correlated response.
    submit_ack_checked_timestamp_ts: datetime | None = None  # Timestamp of the bounded post-submit ACK poll.
    vplan_id_int: int | None = None  # Database primary key.


@dataclass(frozen=True)
class BrokerSnapshot:
    account_route_str: str  # Broker account id.
    snapshot_timestamp_ts: datetime  # When the broker snapshot was taken.
    cash_float: float  # Broker cash balance.
    total_value_float: float  # Broker NetLiq carried through the legacy total-value field.
    position_amount_map: dict[str, float] = field(default_factory=dict)  # Current broker positions by symbol.
    net_liq_float: float = 0.0  # Broker Net Liquidation value.
    available_funds_float: float | None = None  # Broker AvailableFunds value.
    excess_liquidity_float: float | None = None  # Broker ExcessLiquidity value.
    cushion_float: float | None = None  # Broker cushion ratio, if available.
    open_order_id_list: list[str] = field(default_factory=list)  # Broker open order ids.


BrokerPositionSnapshot = BrokerSnapshot


@dataclass(frozen=True)
class LivePriceSnapshot:
    account_route_str: str  # Broker account route used for the quote snapshot.
    snapshot_timestamp_ts: datetime  # When the live quote snapshot was sampled.
    price_source_str: str  # Human-readable source label for the prices.
    asset_reference_price_map: dict[str, float]  # Symbol-to-price map used for VPlan sizing.
    asset_reference_source_map_dict: dict[str, str] = field(default_factory=dict)  # Per-asset provenance for each reference price.


@dataclass(frozen=True)
class BrokerOrderRequest:
    release_id_str: str  # Release that produced the request.
    pod_id_str: str  # Pod that owns the request.
    account_route_str: str  # Broker account to route to.
    submission_key_str: str  # Duplicate-submit guard key.
    order_request_key_str: str  # Per-order correlation key sent as broker orderRef.
    asset_str: str  # Symbol to trade.
    broker_order_type_str: str  # Broker order type.
    order_class_str: str  # Original engine order class.
    unit_str: str  # shares / value / percent.
    amount_float: float  # Signed amount in the given unit.
    target_bool: bool  # True if amount is target-based.
    trade_id_int: int | None  # Strategy trade id, if used.
    sizing_reference_price_float: float  # Reference price used in sizing.
    portfolio_value_float: float  # Portfolio value used in sizing.
    decision_plan_id_int: int | None = None  # Parent decision plan id in v2 flows.
    vplan_id_int: int | None = None  # Parent VPlan id in v2 flows.


@dataclass(frozen=True)
class BrokerOrderRecord:
    broker_order_id_str: str  # Broker order id.
    decision_plan_id_int: int | None  # Parent decision plan id in v2 flows.
    vplan_id_int: int | None  # Parent VPlan id in v2 flows.
    account_route_str: str  # Routed account.
    asset_str: str  # Symbol traded.
    order_request_key_str: str | None  # Per-order correlation key / broker orderRef when available.
    broker_order_type_str: str  # Submitted broker order type.
    unit_str: str  # shares / value / percent.
    amount_float: float  # Requested order amount.
    filled_amount_float: float  # Filled quantity reported by broker.
    status_str: str  # Broker order status.
    submitted_timestamp_ts: datetime  # Submit timestamp.
    raw_payload_dict: dict[str, Any] = field(default_factory=dict)  # Extra broker payload for audit.
    remaining_amount_float: float | None = None  # Remaining quantity reported by broker.
    avg_fill_price_float: float | None = None  # Average fill price reported by broker.
    last_status_timestamp_ts: datetime | None = None  # Timestamp of the latest known broker status.
    submission_key_str: str | None = None  # Stable vplan submission key / broker orderRef when available.


@dataclass(frozen=True)
class BrokerOrderEvent:
    broker_order_id_str: str  # Broker order id.
    decision_plan_id_int: int | None  # Parent decision plan id in v2 flows.
    vplan_id_int: int | None  # Parent VPlan id in v2 flows.
    account_route_str: str  # Routed account.
    asset_str: str  # Symbol traded.
    order_request_key_str: str | None  # Per-order correlation key / broker orderRef when available.
    status_str: str  # Broker order status observed at this event.
    filled_amount_float: float  # Filled quantity reported at this event.
    remaining_amount_float: float | None = None  # Remaining quantity reported at this event.
    avg_fill_price_float: float | None = None  # Average fill price reported at this event.
    event_timestamp_ts: datetime | None = None  # Timestamp of the event.
    event_source_str: str = ""  # Source of the event, e.g. ibkr.trade_log.
    message_str: str = ""  # Human-readable broker message.
    raw_payload_dict: dict[str, Any] = field(default_factory=dict)  # Extra broker payload for audit.
    submission_key_str: str | None = None  # Stable vplan submission key / broker orderRef when available.


@dataclass(frozen=True)
class BrokerOrderAck:
    decision_plan_id_int: int | None  # Parent decision plan id in v2 flows.
    vplan_id_int: int | None  # Parent VPlan id in v2 flows.
    account_route_str: str  # Routed account.
    order_request_key_str: str  # Per-order request correlation key.
    asset_str: str  # Symbol traded.
    broker_order_type_str: str  # Submitted broker order type.
    local_submit_ack_bool: bool  # True if the client-side placeOrder call returned.
    broker_response_ack_bool: bool  # True if a broker-correlated response was observed in the ACK window.
    ack_status_str: str  # broker_acked / missing_critical.
    ack_source_str: str  # open_order / completed_order / event / fill / missing.
    broker_order_id_str: str | None = None  # Broker order id if known.
    perm_id_int: int | None = None  # Broker permanent id if known.
    response_timestamp_ts: datetime | None = None  # Timestamp of the broker response used as the ACK witness.
    raw_payload_dict: dict[str, Any] = field(default_factory=dict)  # Extra ACK audit payload.


@dataclass(frozen=True)
class SubmitBatchResult:
    broker_order_record_list: list[BrokerOrderRecord] = field(default_factory=list)  # Submitted order rows persisted immediately after submit.
    broker_order_event_list: list[BrokerOrderEvent] = field(default_factory=list)  # Order events observed during submit + ACK poll.
    broker_order_fill_list: list[BrokerOrderFill] = field(default_factory=list)  # Fills observed during submit + ACK poll.
    broker_order_ack_list: list[BrokerOrderAck] = field(default_factory=list)  # Explicit local-vs-broker ACK rows.
    ack_coverage_ratio_float: float = 0.0  # broker_ack_count / request_count.
    missing_ack_asset_list: list[str] = field(default_factory=list)  # Assets with no broker-correlated ACK in the bounded window.
    submit_ack_status_str: str = "not_checked"  # complete / missing_critical / not_checked.


@dataclass(frozen=True)
class SessionOpenPrice:
    session_date_str: str  # Market-session date in YYYY-MM-DD form.
    account_route_str: str  # Routed account.
    asset_str: str  # Symbol for this session-open reference.
    official_open_price_float: float | None  # Official session-open price if available.
    open_price_source_str: str | None  # Source label for the open price.
    snapshot_timestamp_ts: datetime  # When the open reference was sampled.
    raw_payload_dict: dict[str, Any] = field(default_factory=dict)  # Extra broker payload for audit.


@dataclass(frozen=True)
class BrokerOrderFill:
    broker_order_id_str: str  # Broker order id that produced the fill.
    decision_plan_id_int: int | None  # Parent decision plan id in v2 flows.
    vplan_id_int: int | None  # Parent VPlan id in v2 flows.
    account_route_str: str  # Routed account.
    asset_str: str  # Symbol filled.
    fill_amount_float: float  # Signed filled quantity.
    fill_price_float: float  # Actual broker fill price.
    fill_timestamp_ts: datetime  # Fill timestamp.
    raw_payload_dict: dict[str, Any] = field(default_factory=dict)  # Extra broker fill payload.
    official_open_price_float: float | None = None  # Official session-open price for slippage checks.
    open_price_source_str: str | None = None  # Source label for the official session-open price.


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
