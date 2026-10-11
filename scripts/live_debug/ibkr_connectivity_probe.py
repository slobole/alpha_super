"""
Probe IBKR API connectivity and fetch a minimal account snapshot.

Usage
-----

    uv run python scripts/live_debug/ibkr_connectivity_probe.py ^
        --release-manifest-path alpha/live/releases/caspersky_account/pod_dv2_caspersky_account_live_01.yaml

    uv run python scripts/live_debug/ibkr_connectivity_probe.py ^
        --host 127.0.0.1 --port 7496 --client-id 31 --account-route U21192795

Daily pods (CORE5, MR capsule) before go-live: add --daily-pod-checks with an
unused client ID. After the connectivity report it runs the serve's own account
reads, daily close read and pre-submit margin/funding/borrow checks, using
one-share what-if previews only; no order is transmitted. Owner decision
2026-10-11: if a check fails only because IBKR does not return a field it reads,
remove that check rather than working around it.

    uv run python scripts/live_debug/ibkr_connectivity_probe.py ^
        --release-manifest-path <pod YAML> --client-id 90 --daily-pod-checks
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.models import BrokerOrderRequest
from alpha.live.order_clerk import infer_ibkr_account_mode_str
from alpha.live.release_manifest import parse_release_manifest


SELECTED_ACCOUNT_TAG_TUPLE: tuple[str, ...] = (
    "TotalCashValue",
    "NetLiquidation",
    "AvailableFunds",
    "ExcessLiquidity",
    "Cushion",
    "BuyingPower",
    "InitMarginReq",
    "MaintMarginReq",
)
# Capsule funding compares buy notional with BuyingPower; one share at this
# placeholder price only proves the field is read, not that a real batch fits.
DAILY_PROBE_PLACEHOLDER_PRICE_FLOAT = 100.0
CAPSULE_SAMPLE_ASSET_TUPLE: tuple[str, ...] = ("AAPL", "BIL")


def _safe_float_or_none(raw_value_obj: Any) -> float | None:
    try:
        value_float = float(raw_value_obj)
    except (TypeError, ValueError):
        return None
    return value_float


def _resolve_probe_config_dict(parsed_args_obj: argparse.Namespace) -> dict[str, Any]:
    manifest_release_obj = None
    if parsed_args_obj.release_manifest_path_str is not None:
        manifest_release_obj = parse_release_manifest(parsed_args_obj.release_manifest_path_str)

    host_str = (
        str(parsed_args_obj.host_str)
        if parsed_args_obj.host_str is not None
        else (
            str(manifest_release_obj.broker_host_str)
            if manifest_release_obj is not None
            else "127.0.0.1"
        )
    )
    port_int = (
        int(parsed_args_obj.port_int)
        if parsed_args_obj.port_int is not None
        else (
            int(manifest_release_obj.broker_port_int)
            if manifest_release_obj is not None
            else 7497
        )
    )
    client_id_int = (
        int(parsed_args_obj.client_id_int)
        if parsed_args_obj.client_id_int is not None
        else (
            int(manifest_release_obj.broker_client_id_int)
            if manifest_release_obj is not None
            else 31
        )
    )
    timeout_seconds_float = (
        float(parsed_args_obj.timeout_seconds_float)
        if parsed_args_obj.timeout_seconds_float is not None
        else (
            float(manifest_release_obj.broker_timeout_seconds_float)
            if manifest_release_obj is not None
            else 4.0
        )
    )
    account_route_str = (
        str(parsed_args_obj.account_route_str)
        if parsed_args_obj.account_route_str is not None
        else (
            str(manifest_release_obj.account_route_str)
            if manifest_release_obj is not None
            else None
        )
    )

    if account_route_str is None:
        raise ValueError(
            "Probe requires --account-route or --release-manifest-path so it knows "
            "which IBKR account to inspect."
        )

    return {
        "release_manifest_path_str": (
            str(parsed_args_obj.release_manifest_path_str)
            if parsed_args_obj.release_manifest_path_str is not None
            else None
        ),
        "host_str": host_str,
        "port_int": port_int,
        "client_id_int": client_id_int,
        "timeout_seconds_float": timeout_seconds_float,
        "account_route_str": account_route_str,
    }


def _build_account_value_map_dict(
    account_value_obj_list: list[Any],
    account_route_str: str,
) -> dict[str, str]:
    return {
        str(account_value_obj.tag): str(account_value_obj.value)
        for account_value_obj in account_value_obj_list
        if str(account_value_obj.account) == str(account_route_str)
    }


def _build_position_amount_map_dict(position_obj_list: list[Any]) -> dict[str, float]:
    position_amount_map_dict: dict[str, float] = {}
    for position_obj in position_obj_list:
        position_float = float(position_obj.position)
        if abs(position_float) <= 1e-12:
            continue
        asset_str = str(position_obj.contract.symbol)
        position_amount_map_dict[asset_str] = position_float
    return dict(sorted(position_amount_map_dict.items()))


def _build_open_order_row_dict_list(
    trade_obj_list: list[Any],
    account_route_str: str,
) -> list[dict[str, Any]]:
    open_order_row_dict_list: list[dict[str, Any]] = []
    for trade_obj in trade_obj_list:
        if str(getattr(trade_obj.order, "account", "")) != str(account_route_str):
            continue
        open_order_row_dict_list.append(
            {
                "broker_order_id_str": str(getattr(trade_obj.order, "orderId", "")),
                "perm_id_int": int(getattr(trade_obj.order, "permId", 0) or 0),
                "asset_str": str(getattr(getattr(trade_obj, "contract", None), "symbol", "")),
                "action_str": str(getattr(trade_obj.order, "action", "")),
                "total_quantity_float": float(getattr(trade_obj.order, "totalQuantity", 0.0) or 0.0),
                "order_type_str": str(getattr(trade_obj.order, "orderType", "")),
                "status_str": str(getattr(trade_obj.orderStatus, "status", "")),
            }
        )
    return sorted(
        open_order_row_dict_list,
        key=lambda row_dict: (row_dict["asset_str"], row_dict["broker_order_id_str"]),
    )


def _collect_probe_report_dict(probe_config_dict: dict[str, Any]) -> dict[str, Any]:
    socket_client_obj = IBKRSocketClient(
        host_str=str(probe_config_dict["host_str"]),
        port_int=int(probe_config_dict["port_int"]),
        client_id_int=int(probe_config_dict["client_id_int"]),
        timeout_seconds_float=float(probe_config_dict["timeout_seconds_float"]),
    )
    requested_account_route_str = str(probe_config_dict["account_route_str"])
    probe_timestamp_ts = datetime.now(tz=UTC)

    with socket_client_obj.connect() as ib_obj:
        visible_account_route_list = sorted(
            str(account_route_str) for account_route_str in ib_obj.managedAccounts()
        )
        account_visible_bool = requested_account_route_str in visible_account_route_list
        account_value_obj_list = ib_obj.accountSummary(account=requested_account_route_str)
        account_value_map_dict = _build_account_value_map_dict(
            account_value_obj_list=account_value_obj_list,
            account_route_str=requested_account_route_str,
        )
        position_obj_list = ib_obj.positions(account=requested_account_route_str)
        position_amount_map_dict = _build_position_amount_map_dict(position_obj_list)
        open_trade_obj_list = list(ib_obj.reqOpenOrders())
        open_order_row_dict_list = _build_open_order_row_dict_list(
            trade_obj_list=open_trade_obj_list,
            account_route_str=requested_account_route_str,
        )

    if not account_visible_bool:
        raise RuntimeError(
            "IBKR API connected, but the requested account_route_str was not visible "
            "in managedAccounts(). "
            f"requested_account_route_str={requested_account_route_str} "
            f"visible_account_route_list={visible_account_route_list}"
        )

    numeric_account_value_map_dict = {
        tag_str: _safe_float_or_none(account_value_map_dict.get(tag_str))
        for tag_str in SELECTED_ACCOUNT_TAG_TUPLE
    }
    gross_position_share_float = sum(
        abs(position_float) for position_float in position_amount_map_dict.values()
    )

    return {
        "probe_timestamp_str": probe_timestamp_ts.isoformat(),
        "connected_bool": True,
        "host_str": str(probe_config_dict["host_str"]),
        "port_int": int(probe_config_dict["port_int"]),
        "client_id_int": int(probe_config_dict["client_id_int"]),
        "timeout_seconds_float": float(probe_config_dict["timeout_seconds_float"]),
        "requested_account_route_str": requested_account_route_str,
        "requested_account_mode_str": infer_ibkr_account_mode_str(requested_account_route_str),
        "visible_account_route_list": visible_account_route_list,
        "open_order_count_int": len(open_order_row_dict_list),
        "position_count_int": len(position_amount_map_dict),
        "gross_position_share_float": gross_position_share_float,
        "formula_str": "gross_position_share_float = sum_i abs(position_share_i)",
        "account_value_map_dict": numeric_account_value_map_dict,
        "position_amount_map_dict": position_amount_map_dict,
        "open_order_row_dict_list": open_order_row_dict_list,
    }


def _render_probe_report_str(probe_report_dict: dict[str, Any]) -> str:
    account_value_map_dict = dict(probe_report_dict["account_value_map_dict"])
    position_amount_map_dict = dict(probe_report_dict["position_amount_map_dict"])
    open_order_row_dict_list = list(probe_report_dict["open_order_row_dict_list"])

    output_line_list = [
        "IBKR connectivity probe",
        f"  connected_bool = {probe_report_dict['connected_bool']}",
        f"  host_str = {probe_report_dict['host_str']}",
        f"  port_int = {probe_report_dict['port_int']}",
        f"  client_id_int = {probe_report_dict['client_id_int']}",
        f"  requested_account_route_str = {probe_report_dict['requested_account_route_str']}",
        f"  visible_account_route_list = {probe_report_dict['visible_account_route_list']}",
        "Account values",
    ]

    for tag_str in SELECTED_ACCOUNT_TAG_TUPLE:
        output_line_list.append(f"  {tag_str} = {account_value_map_dict.get(tag_str)}")

    output_line_list.extend(
        [
            "Positions",
            f"  position_count_int = {probe_report_dict['position_count_int']}",
            f"  gross_position_share_float = {probe_report_dict['gross_position_share_float']}",
            f"  formula_str = {probe_report_dict['formula_str']}",
        ]
    )
    if len(position_amount_map_dict) == 0:
        output_line_list.append("  position_amount_map_dict = {}")
    else:
        for asset_str, position_float in position_amount_map_dict.items():
            output_line_list.append(f"  {asset_str} = {position_float}")

    output_line_list.extend(
        [
            "Open orders",
            f"  open_order_count_int = {probe_report_dict['open_order_count_int']}",
        ]
    )
    if len(open_order_row_dict_list) == 0:
        output_line_list.append("  open_order_row_dict_list = []")
    else:
        for open_order_row_dict in open_order_row_dict_list:
            output_line_list.append("  " + json.dumps(open_order_row_dict, sort_keys=True))

    if "daily_pod_check_list" in probe_report_dict:
        output_line_list.append("Daily pod checks (read-only: what-if previews only; no order is transmitted)")
        for step_dict in probe_report_dict["daily_pod_check_list"]:
            output_line_list.append(f"  {step_dict['status_str']}  {step_dict['label_str']}: {step_dict['detail_str']}")
        output_line_list.append(
            "  RESULT: all daily pod checks passed." if probe_report_dict["daily_pod_checks_passed_bool"]
            else "  RESULT: at least one daily pod check failed; send this output for review."
        )

    return "\n".join(output_line_list)


class ReadOnlyBroker:
    """Forwards only the read calls the daily pod checks need; anything else raises."""

    ALLOWED_METHOD_TUPLE: tuple[str, ...] = (
        "get_visible_account_route_set", "get_core5_account_snapshot", "get_eod_account_snapshot",
        "get_capsule_account_snapshot", "get_daily_execution_snapshot",
        "get_core5_funding_evidence", "get_capsule_funding_evidence",
    )

    def __init__(self, adapter_obj: Any):
        object.__setattr__(self, "_adapter_obj", adapter_obj)

    def __getattr__(self, name_str: str) -> Any:
        if name_str not in self.ALLOWED_METHOD_TUPLE:
            raise AttributeError(f"The probe is read-only; {name_str} is not allowed.")
        return getattr(object.__getattribute__(self, "_adapter_obj"), name_str)


def _daily_sample_request(release_obj: Any, asset_str: str, amount_float: float, index_int: int) -> BrokerOrderRequest:
    return BrokerOrderRequest(
        release_id_str=release_obj.release_id_str, pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str, submission_key_str="probe",
        order_request_key_str=f"probe:{asset_str}:{index_int}", asset_str=asset_str,
        broker_order_type_str="MOO", order_class_str="probe", unit_str="shares",
        amount_float=float(amount_float), target_bool=False, trade_id_int=None,
        sizing_reference_price_float=DAILY_PROBE_PLACEHOLDER_PRICE_FLOAT, portfolio_value_float=0.0,
    )


def core5_sample_request_list(release_obj: Any, position_dict: dict[str, float]) -> list[BrokerOrderRequest]:
    """One-share buys of every CORE5 asset except DBC, plus a DBC sell that grows
    the short by one share so the shortable-shares check also runs."""
    from alpha.live.core5_adapter import CORE5_ASSET_TUPLE

    request_list = []
    for index_int, asset_str in enumerate(CORE5_ASSET_TUPLE):
        amount_float = -(max(float(position_dict.get("DBC", 0.0)), 0.0) + 1.0) if asset_str == "DBC" else 1.0
        request_list.append(_daily_sample_request(release_obj, asset_str, amount_float, index_int))
    return request_list


def capsule_sample_request_list(release_obj: Any) -> list[BrokerOrderRequest]:
    return [_daily_sample_request(release_obj, asset_str, 1.0, index_int)
            for index_int, asset_str in enumerate(CAPSULE_SAMPLE_ASSET_TUPLE)]


def _plain(value_obj: Any) -> Any:
    if is_dataclass(value_obj):
        return _plain(asdict(value_obj))
    if isinstance(value_obj, dict):
        return {str(key_obj): _plain(item_obj) for key_obj, item_obj in value_obj.items()}
    if isinstance(value_obj, (list, tuple, set)):
        return [_plain(item_obj) for item_obj in value_obj]
    if isinstance(value_obj, datetime):
        return value_obj.isoformat()
    return value_obj


def _snapshot_summary_str(snapshot_obj: Any) -> str:
    return (f"cash {snapshot_obj.cash_float:,.2f}, NAV {snapshot_obj.total_value_float:,.2f}, "
            f"positions {dict(sorted(snapshot_obj.position_amount_map.items()))}, "
            f"open orders {len(snapshot_obj.open_order_id_list)}")


def _funding_summary_str(evidence_dict: dict[str, Any]) -> str:
    preview_str = ", ".join(f"{item_dict['asset_str']} {item_dict['status_str']}"
                            for item_dict in evidence_dict.get("preview_list") or [])
    fact_list = [f"type {evidence_dict.get('trading_type_str')}"]
    for key_str, label_str in (("available_funds_float", "AvailableFunds"), ("excess_liquidity_float", "ExcessLiquidity"),
                               ("buying_power_float", "BuyingPower"), ("dbc_shortable_shares_float", "DBC shortable")):
        if evidence_dict.get(key_str) is not None:
            fact_list.append(f"{label_str} {evidence_dict[key_str]:,.0f}")
    return "; ".join(fact_list) + (f"; previews: {preview_str}" if preview_str else "")


def run_daily_pod_checks(release_obj: Any, broker_obj: Any) -> list[dict[str, Any]]:
    """The serve's own reads and pre-submit checks; one failure never stops the others."""
    from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR

    account_str = release_obj.account_route_str
    core5_bool = release_obj.strategy_import_str == CORE5_STRATEGY_IMPORT_STR
    step_list: list[dict[str, Any]] = []
    context_dict: dict[str, Any] = {}

    def run_step(label_str, step_fn):
        try:
            detail_str, raw_obj = step_fn()
            step_list.append({"label_str": label_str, "status_str": "PASS", "detail_str": detail_str, "raw": _plain(raw_obj)})
        except Exception as exception_obj:  # Every IBKR or check failure is the finding itself.
            step_list.append({"label_str": label_str, "status_str": "FAIL",
                              "detail_str": f"{type(exception_obj).__name__}: {exception_obj}", "raw": None})

    def visible_step():
        account_set = broker_obj.get_visible_account_route_set()
        if account_set is None or account_str not in account_set:
            raise ValueError(f"account {account_str} is not visible on this IBKR session")
        return f"account {account_str} visible", sorted(account_set)

    run_step("IBKR session sees the account", visible_step)
    if core5_bool:
        def account_step():
            snapshot_obj = broker_obj.get_core5_account_snapshot(account_str, include_portfolio_valuation_bool=True)
            context_dict["position_dict"] = dict(snapshot_obj.position_amount_map)
            return _snapshot_summary_str(snapshot_obj), snapshot_obj
        run_step("Account read (16:10 and before sending)", account_step)
    else:
        def eod_step():
            snapshot_obj = broker_obj.get_eod_account_snapshot(account_str)
            return _snapshot_summary_str(snapshot_obj), snapshot_obj
        run_step("Account read at 16:10", eod_step)

        def presubmit_step():
            snapshot_obj = broker_obj.get_capsule_account_snapshot(account_str)
            return _snapshot_summary_str(snapshot_obj), snapshot_obj
        run_step("Account read before sending", presubmit_step)

    def daily_step():
        daily_obj = broker_obj.get_daily_execution_snapshot(account_str)
        if not daily_obj.complete_bool:
            raise ValueError("the daily account refresh reported incomplete data")
        return (_snapshot_summary_str(daily_obj.broker_snapshot_obj)
                + f"; open order rows {len(daily_obj.open_order_row_list)}"), daily_obj
    run_step("Daily close read (holdings and all-client open orders)", daily_step)

    if core5_bool:
        def core5_funding_step():
            if "position_dict" not in context_dict:
                raise ValueError("skipped: needs the account read above to pass first")
            request_list = core5_sample_request_list(release_obj, context_dict["position_dict"])
            evidence_dict = broker_obj.get_core5_funding_evidence(account_str, request_list, context_dict["position_dict"])
            return _funding_summary_str(evidence_dict), evidence_dict
        run_step("Pre-submit margin, funding and DBC borrow check (what-if only)", core5_funding_step)
    else:
        def capsule_funding_step():
            evidence_dict = broker_obj.get_capsule_funding_evidence(account_str, capsule_sample_request_list(release_obj))
            return _funding_summary_str(evidence_dict), evidence_dict
        run_step("Pre-submit buying power and margin check (what-if only)", capsule_funding_step)
    return step_list


def _daily_pod_release_obj(parsed_args_obj: argparse.Namespace) -> Any:
    """Preconditions for --daily-pod-checks; raises ValueError with an operator message."""
    from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
    from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE

    if parsed_args_obj.release_manifest_path_str is None:
        raise ValueError("--daily-pod-checks requires --release-manifest-path for a CORE5 or MR capsule pod.")
    release_obj = parse_release_manifest(parsed_args_obj.release_manifest_path_str)
    if (release_obj.strategy_import_str != CORE5_STRATEGY_IMPORT_STR
            and release_obj.strategy_import_str not in MR_CAPSULE_STRATEGY_IMPORT_TUPLE):
        raise ValueError("--daily-pod-checks is for CORE5 and MR capsule releases only.")
    client_id_obj = parsed_args_obj.client_id_int
    if client_id_obj is None or client_id_obj <= 0 or client_id_obj == release_obj.broker_client_id_int:
        raise ValueError("--daily-pod-checks requires an explicit --client-id that no other process uses "
                         "(never the pod's own client ID).")
    return release_obj


def _build_error_report_dict(
    probe_config_dict: dict[str, Any] | None,
    exception_obj: Exception,
    traceback_bool: bool,
) -> dict[str, Any]:
    return {
        "connected_bool": False,
        "host_str": None if probe_config_dict is None else probe_config_dict.get("host_str"),
        "port_int": None if probe_config_dict is None else probe_config_dict.get("port_int"),
        "client_id_int": None if probe_config_dict is None else probe_config_dict.get("client_id_int"),
        "requested_account_route_str": (
            None if probe_config_dict is None else probe_config_dict.get("account_route_str")
        ),
        "error_type_str": exception_obj.__class__.__name__,
        "error_str": str(exception_obj),
        "traceback_str": traceback.format_exc() if traceback_bool else None,
    }


def parse_args(argv_list: list[str] | None = None) -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Probe IBKR API connectivity and account state."
    )
    parser_obj.add_argument("--release-manifest-path", dest="release_manifest_path_str", default=None)
    parser_obj.add_argument("--host", dest="host_str", default=None)
    parser_obj.add_argument("--port", dest="port_int", type=int, default=None)
    parser_obj.add_argument("--client-id", dest="client_id_int", type=int, default=None)
    parser_obj.add_argument("--timeout-seconds", dest="timeout_seconds_float", type=float, default=None)
    parser_obj.add_argument("--account-route", dest="account_route_str", default=None)
    parser_obj.add_argument("--json", dest="json_output_bool", action="store_true")
    parser_obj.add_argument("--traceback", dest="traceback_bool", action="store_true")
    parser_obj.add_argument(
        "--daily-pod-checks", dest="daily_pod_checks_bool", action="store_true",
        help="CORE5/MR capsule only: also run the serve's account reads and pre-submit checks "
             "(what-if only, no order sent). Requires --release-manifest-path and an unused --client-id.",
    )
    return parser_obj.parse_args(argv_list)


def main(argv_list: list[str] | None = None, *, adapter_factory_fn: Any = None) -> int:
    parsed_args_obj = parse_args(argv_list)
    daily_release_obj = None
    if parsed_args_obj.daily_pod_checks_bool:
        # Validate before connecting: never borrow the pod's own client ID.
        try:
            daily_release_obj = _daily_pod_release_obj(parsed_args_obj)
        except ValueError as exception_obj:
            print(f"daily pod checks: {exception_obj}", file=sys.stderr)
            return 2
    probe_config_dict: dict[str, Any] | None = None
    try:
        probe_config_dict = _resolve_probe_config_dict(parsed_args_obj)
        probe_report_dict = _collect_probe_report_dict(probe_config_dict)
        if daily_release_obj is not None:
            if adapter_factory_fn is None:
                from alpha.live.order_clerk import IBKRGatewayBrokerAdapter as adapter_factory_fn
            broker_obj = ReadOnlyBroker(adapter_factory_fn(
                host_str=str(probe_config_dict["host_str"]), port_int=int(probe_config_dict["port_int"]),
                client_id_int=int(probe_config_dict["client_id_int"]),
                timeout_seconds_float=float(probe_config_dict["timeout_seconds_float"]),
            ))
            daily_pod_check_list = run_daily_pod_checks(daily_release_obj, broker_obj)
            probe_report_dict["daily_pod_check_list"] = daily_pod_check_list
            probe_report_dict["daily_pod_checks_passed_bool"] = all(
                step_dict["status_str"] == "PASS" for step_dict in daily_pod_check_list)
        if parsed_args_obj.json_output_bool:
            print(json.dumps(probe_report_dict, indent=2, sort_keys=True))
        else:
            print(_render_probe_report_str(probe_report_dict))
        return 1 if probe_report_dict.get("daily_pod_checks_passed_bool") is False else 0
    except Exception as exception_obj:
        error_report_dict = _build_error_report_dict(
            probe_config_dict=probe_config_dict,
            exception_obj=exception_obj,
            traceback_bool=bool(parsed_args_obj.traceback_bool),
        )
        if parsed_args_obj.json_output_bool:
            print(json.dumps(error_report_dict, indent=2, sort_keys=True))
        else:
            print("IBKR connectivity probe")
            print("  connected_bool = False")
            print(f"  error_type_str = {error_report_dict['error_type_str']}")
            print(f"  error_str = {error_report_dict['error_str']}")
            if error_report_dict["traceback_str"] is not None:
                print(error_report_dict["traceback_str"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
