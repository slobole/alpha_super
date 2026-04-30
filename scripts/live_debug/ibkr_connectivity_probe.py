"""
Probe IBKR API connectivity and fetch a minimal account snapshot.

Usage
-----

    uv run python scripts/live_debug/ibkr_connectivity_probe.py ^
        --release-manifest-path alpha/live/releases/caspersky_account/pod_dv2_caspersky_account_live_01.yaml

    uv run python scripts/live_debug/ibkr_connectivity_probe.py ^
        --host 127.0.0.1 --port 7496 --client-id 31 --account-route U21192795
"""

from __future__ import annotations

import argparse
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

    return "\n".join(output_line_list)


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
    return parser_obj.parse_args(argv_list)


def main(argv_list: list[str] | None = None) -> int:
    parsed_args_obj = parse_args(argv_list)
    probe_config_dict: dict[str, Any] | None = None
    try:
        probe_config_dict = _resolve_probe_config_dict(parsed_args_obj)
        probe_report_dict = _collect_probe_report_dict(probe_config_dict)
        if parsed_args_obj.json_output_bool:
            print(json.dumps(probe_report_dict, indent=2, sort_keys=True))
        else:
            print(_render_probe_report_str(probe_report_dict))
        return 0
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
