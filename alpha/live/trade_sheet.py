"""Trade-sheet export: render the persisted DecisionPlan + VPlan as an xlsx.

Operator escape hatch (RealTest-style): one file with the trades to take so a
missed submission window or a broker outage can be recovered by manual
execution, and so the operator can review the order plan before the open.

This module is deliberately read-only with respect to trading. It renders what
the live system already decided and persisted in SQLite — it does not size,
price, submit, or mutate anything. Manual fills need no special state handling:
VPlan sizing reads fresh broker truth and EOD snapshots record the actual
account, so manual trades are absorbed by the next cycle automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha.live.models import DecisionPlan, VPlan

# Same artifact timestamp-label convention as alpha/live/reference_compare.py.
TIMESTAMP_LABEL_FORMAT_STR = "%Y%m%dT%H%M%SZ"

ORDERS_SHEET_NAME_STR = "Orders"
DECISION_SHEET_NAME_STR = "Decision"
CONTEXT_SHEET_NAME_STR = "Context"

_ORDERS_COLUMN_LIST = [
    "asset_str",
    "side_str",
    "order_delta_share_float",
    "broker_order_type_str",
    "live_reference_price_float",
    "estimated_delta_notional_float",
    "current_share_float",
    "target_share_float",
    "estimated_target_notional_float",
    "live_reference_source_str",
]

_DECISION_COLUMN_LIST = [
    "asset_str",
    "role_str",
    "target_weight_float",
    "entry_priority_int",
]

_SHARE_TOLERANCE_FLOAT = 1e-9


def _side_str_for_delta(order_delta_share_float: float) -> str:
    if order_delta_share_float > _SHARE_TOLERANCE_FLOAT:
        return "BUY"
    if order_delta_share_float < -_SHARE_TOLERANCE_FLOAT:
        return "SELL"
    return "HOLD"


def _isoformat_str(value_ts: datetime | None) -> str:
    if value_ts is None:
        return ""
    return value_ts.isoformat()


_SIDE_RANK_MAP_DICT = {"SELL": 0, "BUY": 1, "HOLD": 2}


def _build_orders_df(vplan_obj: VPlan | None) -> pd.DataFrame:
    if vplan_obj is None:
        return pd.DataFrame(columns=_ORDERS_COLUMN_LIST)
    # The state store does not preserve VPlan row insertion order, so impose an
    # explicit manual-execution order instead: SELL first (frees cash for the
    # buys), then BUY, then HOLD; alphabetical within each group.
    row_dict_list = [
        {
            "asset_str": row_obj.asset_str,
            "side_str": _side_str_for_delta(float(row_obj.order_delta_share_float)),
            "order_delta_share_float": float(row_obj.order_delta_share_float),
            "broker_order_type_str": row_obj.broker_order_type_str,
            "live_reference_price_float": float(row_obj.live_reference_price_float),
            "estimated_delta_notional_float": float(row_obj.order_delta_share_float)
            * float(row_obj.live_reference_price_float),
            "current_share_float": float(row_obj.current_share_float),
            "target_share_float": float(row_obj.target_share_float),
            "estimated_target_notional_float": float(row_obj.estimated_target_notional_float),
            "live_reference_source_str": row_obj.live_reference_source_str,
        }
        for row_obj in vplan_obj.vplan_row_list
    ]
    row_dict_list.sort(
        key=lambda row_dict: (
            _SIDE_RANK_MAP_DICT[row_dict["side_str"]],
            row_dict["asset_str"],
        )
    )
    return pd.DataFrame(row_dict_list, columns=_ORDERS_COLUMN_LIST)


def _build_decision_df(decision_plan_obj: DecisionPlan | None) -> pd.DataFrame:
    if decision_plan_obj is None:
        return pd.DataFrame(columns=_DECISION_COLUMN_LIST)

    entry_rank_map_dict = {
        asset_str: rank_idx_int
        for rank_idx_int, asset_str in enumerate(decision_plan_obj.entry_priority_list)
    }

    if decision_plan_obj.decision_book_type_str == "incremental_entry_exit_book":
        weight_map_dict = decision_plan_obj.entry_target_weight_map_dict
        weight_role_str = "ENTRY"
    else:
        weight_map_dict = decision_plan_obj.full_target_weight_map_dict
        weight_role_str = "TARGET"

    row_dict_list = [
        {
            "asset_str": asset_str,
            "role_str": weight_role_str,
            "target_weight_float": float(target_weight_float),
            "entry_priority_int": entry_rank_map_dict.get(asset_str),
        }
        for asset_str, target_weight_float in sorted(weight_map_dict.items())
    ]
    row_dict_list.extend(
        {
            "asset_str": asset_str,
            "role_str": "EXIT",
            "target_weight_float": 0.0,
            "entry_priority_int": None,
        }
        for asset_str in sorted(decision_plan_obj.exit_asset_set)
    )
    return pd.DataFrame(row_dict_list, columns=_DECISION_COLUMN_LIST)


def _build_context_df(
    pod_id_str: str,
    env_mode_str: str,
    decision_plan_obj: DecisionPlan | None,
    vplan_obj: VPlan | None,
    generated_at_ts: datetime,
) -> pd.DataFrame:
    # All values are rendered as strings: openpyxl rejects tz-aware datetimes,
    # and a key/value audit tab does not need native Excel types.
    field_value_pair_list: list[tuple[str, str]] = [
        ("generated_at", _isoformat_str(generated_at_ts)),
        ("pod_id", pod_id_str),
        ("mode", env_mode_str),
    ]
    if decision_plan_obj is not None:
        field_value_pair_list.extend(
            [
                ("account_route", decision_plan_obj.account_route_str),
                ("decision_plan_id", str(decision_plan_obj.decision_plan_id_int)),
                ("decision_status", decision_plan_obj.status_str),
                ("decision_book_type", decision_plan_obj.decision_book_type_str),
                ("execution_policy", decision_plan_obj.execution_policy_str),
                ("signal_timestamp", _isoformat_str(decision_plan_obj.signal_timestamp_ts)),
                ("submission_timestamp", _isoformat_str(decision_plan_obj.submission_timestamp_ts)),
                (
                    "target_execution_timestamp",
                    _isoformat_str(decision_plan_obj.target_execution_timestamp_ts),
                ),
            ]
        )
    if vplan_obj is not None:
        field_value_pair_list.extend(
            [
                ("vplan_id", str(vplan_obj.vplan_id_int)),
                ("vplan_status", vplan_obj.status_str),
                (
                    "broker_snapshot_timestamp",
                    _isoformat_str(vplan_obj.broker_snapshot_timestamp_ts),
                ),
                (
                    "live_reference_snapshot_timestamp",
                    _isoformat_str(vplan_obj.live_reference_snapshot_timestamp_ts),
                ),
                ("live_price_source", vplan_obj.live_price_source_str),
                ("net_liq", str(float(vplan_obj.net_liq_float))),
                ("pod_budget_fraction", str(float(vplan_obj.pod_budget_fraction_float))),
                ("pod_budget", str(float(vplan_obj.pod_budget_float))),
                (
                    "available_funds",
                    "" if vplan_obj.available_funds_float is None else str(float(vplan_obj.available_funds_float)),
                ),
            ]
        )
    else:
        field_value_pair_list.append(
            ("vplan_status", "NO VPLAN YET - decision intent only, no broker-sized orders")
        )
    return pd.DataFrame(field_value_pair_list, columns=["field_str", "value_str"])


def build_trade_sheet_data(
    state_store_obj,
    pod_id_str: str,
    generated_at_ts: datetime,
    vplan_id_int: int | None = None,
) -> dict[str, object]:
    """Load the persisted plans for one pod and shape them into sheet frames.

    Read-only: only state-store getters are called. Raises a plain-language
    ValueError when the pod has no decision plan and no VPlan at all.
    """
    if vplan_id_int is not None:
        vplan_obj: VPlan | None = state_store_obj.get_vplan_by_id(int(vplan_id_int))
        if vplan_obj is not None and vplan_obj.pod_id_str != pod_id_str:
            raise ValueError(
                f"VPlan {vplan_id_int} belongs to pod '{vplan_obj.pod_id_str}', "
                f"not requested pod '{pod_id_str}'. Refusing to export a mismatched sheet."
            )
    else:
        vplan_obj = state_store_obj.get_latest_vplan_for_pod(pod_id_str)

    if vplan_obj is not None:
        decision_plan_obj: DecisionPlan | None = state_store_obj.get_decision_plan_by_id(
            int(vplan_obj.decision_plan_id_int)
        )
    else:
        decision_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(pod_id_str)

    if vplan_obj is None and decision_plan_obj is None:
        raise ValueError(
            f"No DecisionPlan and no VPlan found for pod '{pod_id_str}' in this DB. "
            "There is nothing to export: run a tick first, or check --mode/--db-path."
        )

    return {
        "orders_df": _build_orders_df(vplan_obj),
        "decision_df": _build_decision_df(decision_plan_obj),
        "vplan_obj": vplan_obj,
        "decision_plan_obj": decision_plan_obj,
    }


def write_trade_sheet_xlsx(
    trade_sheet_dict: dict[str, object],
    output_path_obj: Path,
    pod_id_str: str,
    env_mode_str: str,
    generated_at_ts: datetime,
) -> Path:
    context_df = _build_context_df(
        pod_id_str=pod_id_str,
        env_mode_str=env_mode_str,
        decision_plan_obj=trade_sheet_dict["decision_plan_obj"],
        vplan_obj=trade_sheet_dict["vplan_obj"],
        generated_at_ts=generated_at_ts,
    )
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path_obj, engine="openpyxl") as writer_obj:
        trade_sheet_dict["orders_df"].to_excel(
            writer_obj, sheet_name=ORDERS_SHEET_NAME_STR, index=False
        )
        trade_sheet_dict["decision_df"].to_excel(
            writer_obj, sheet_name=DECISION_SHEET_NAME_STR, index=False
        )
        context_df.to_excel(writer_obj, sheet_name=CONTEXT_SHEET_NAME_STR, index=False)
    return output_path_obj


def build_default_trade_sheet_path_obj(
    output_dir_str: str,
    env_mode_str: str,
    pod_id_str: str,
    generated_at_ts: datetime,
) -> Path:
    timestamp_label_str = generated_at_ts.astimezone(timezone.utc).strftime(
        TIMESTAMP_LABEL_FORMAT_STR
    )
    return (
        Path(output_dir_str)
        / "trade_sheets"
        / env_mode_str
        / pod_id_str
        / f"trade_sheet_{timestamp_label_str}.xlsx"
    )


def export_trade_sheet_detail_dict(
    state_store_obj,
    pod_id_str: str,
    env_mode_str: str,
    generated_at_ts: datetime,
    vplan_id_int: int | None = None,
    output_path_str: str | None = None,
    output_dir_str: str = "results",
) -> dict[str, object]:
    """Build + write the trade sheet; return the runner-style detail dict."""
    trade_sheet_dict = build_trade_sheet_data(
        state_store_obj=state_store_obj,
        pod_id_str=pod_id_str,
        generated_at_ts=generated_at_ts,
        vplan_id_int=vplan_id_int,
    )
    if output_path_str is not None:
        output_path_obj = Path(output_path_str)
    else:
        output_path_obj = build_default_trade_sheet_path_obj(
            output_dir_str=output_dir_str,
            env_mode_str=env_mode_str,
            pod_id_str=pod_id_str,
            generated_at_ts=generated_at_ts,
        )
    written_path_obj = write_trade_sheet_xlsx(
        trade_sheet_dict=trade_sheet_dict,
        output_path_obj=output_path_obj,
        pod_id_str=pod_id_str,
        env_mode_str=env_mode_str,
        generated_at_ts=generated_at_ts,
    )

    vplan_obj = trade_sheet_dict["vplan_obj"]
    decision_plan_obj = trade_sheet_dict["decision_plan_obj"]
    orders_df = trade_sheet_dict["orders_df"]
    return {
        "command_name_str": "export_trade_sheet",
        "pod_id_str": pod_id_str,
        "mode_str": env_mode_str,
        "output_path_str": str(written_path_obj),
        "vplan_id_int": None if vplan_obj is None else vplan_obj.vplan_id_int,
        "vplan_status_str": "" if vplan_obj is None else vplan_obj.status_str,
        "decision_plan_id_int": (
            None if decision_plan_obj is None else decision_plan_obj.decision_plan_id_int
        ),
        "order_count_int": int(len(orders_df)),
        "decision_only_bool": vplan_obj is None,
        "generated_at_timestamp_str": _isoformat_str(generated_at_ts),
    }
