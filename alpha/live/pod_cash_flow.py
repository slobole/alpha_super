"""Operator-declared external cash flows per pod — read-only.

Broker EOD NetLiq snapshots cannot distinguish trading P&L from a deposit:
a $10k top-up into a $10k pod read as +100% "return". Time-weighted math
needs the flow dates, and the live schema records none, so the operator
declares them here — one YAML file, hand-edited, versioned with the repo.
The dashboard only ever reads it; recording a deposit is a text edit, not
a dashboard action, which keeps the console's zero-write property intact.

Schema (``pod_cash_flows.yaml`` beside this module, or the path in
``ALPHA_POD_CASH_FLOWS_PATH_STR``)::

    flows:
      - pod_id: pod_taa_df_btal_fallback_live_01
        date: 2026-07-01          # market date the money landed (ET)
        amount: 10000.0           # positive = deposit, negative = withdrawal
        note: July top-up         # optional, for the audit trail

Convention: a flow dated D is treated as arriving at the END of market
date D — day D's return is (E_D - F_D) / E_{D-1} - 1. At daily
resolution this is the standard fund-unitization treatment; the
approximation is recorded in ASSUMPTIONS_AND_GAPS.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import date
import math
from pathlib import Path
from typing import Any

import yaml

DEFAULT_POD_CASH_FLOWS_PATH = Path(__file__).resolve().parent / "pod_cash_flows.yaml"
POD_CASH_FLOWS_PATH_ENV_STR = "ALPHA_POD_CASH_FLOWS_PATH_STR"


def resolve_pod_cash_flows_path() -> Path:
    override_str = os.environ.get(POD_CASH_FLOWS_PATH_ENV_STR)
    return Path(override_str) if override_str else DEFAULT_POD_CASH_FLOWS_PATH


def load_flow_by_date_dict(
    pod_id_str: str,
    flows_path: Path | None = None,
) -> dict[str, float]:
    """Net external flow per market date for one pod: ``{'YYYY-MM-DD': amount}``.

    A missing default file or an empty list means no declared flows. A missing
    explicit override and every malformed entry fail loud: silently dropping a
    deposit would re-inflate the very returns this module exists to fix.
    """
    flow_entry_list = _load_normalized_flow_entry_list(flows_path)
    flow_by_date_dict: dict[str, float] = defaultdict(float)
    for entry_dict in flow_entry_list:
        if entry_dict["pod_id_str"] != pod_id_str:
            continue
        date_str = entry_dict["date_str"]
        amount_float = entry_dict["amount_float"]
        flow_by_date_dict[date_str] += amount_float
    return dict(flow_by_date_dict)


def validate_pod_cash_flow_pod_ids(
    known_pod_id_set: set[str],
    flows_path: Path | None = None,
) -> None:
    """Fail when a declared flow names no release known to this dashboard."""
    declared_pod_id_set = {
        str(entry_dict["pod_id_str"])
        for entry_dict in _load_normalized_flow_entry_list(flows_path)
    }
    unknown_pod_id_list = sorted(declared_pod_id_set - known_pod_id_set)
    if unknown_pod_id_list:
        raise ValueError(
            "pod cash flows contain unknown pod_id value(s): "
            f"{unknown_pod_id_list}"
        )


def _load_normalized_flow_entry_list(
    flows_path: Path | None,
) -> list[dict[str, Any]]:
    explicit_path_bool = flows_path is not None or bool(
        os.environ.get(POD_CASH_FLOWS_PATH_ENV_STR)
    )
    path_obj = flows_path or resolve_pod_cash_flows_path()
    if not path_obj.is_file():
        if explicit_path_bool:
            raise FileNotFoundError(f"pod cash flows file does not exist: {path_obj}")
        return []

    payload_obj = yaml.safe_load(path_obj.read_text(encoding="utf-8"))
    if payload_obj is None:
        return []
    if not isinstance(payload_obj, dict):
        raise ValueError(f"{path_obj}: top level must be a mapping")
    if "flows" not in payload_obj:
        raise ValueError(f"{path_obj}: missing required 'flows' list")
    flow_entry_list = payload_obj["flows"]
    if not isinstance(flow_entry_list, list):
        raise ValueError(f"{path_obj}: 'flows' must be a list")

    normalized_flow_entry_list: list[dict[str, Any]] = []
    for entry_dict in flow_entry_list:
        if not isinstance(entry_dict, dict):
            raise ValueError(f"{path_obj}: flow entries must be mappings, got {entry_dict!r}")
        missing_key_list = [
            key_str
            for key_str in ("pod_id", "date", "amount")
            if key_str not in entry_dict
        ]
        if missing_key_list:
            raise ValueError(
                f"{path_obj}: flow entry missing {missing_key_list}: {entry_dict!r}"
            )

        pod_id_obj = entry_dict["pod_id"]
        if not isinstance(pod_id_obj, str) or not pod_id_obj.strip():
            raise ValueError(
                f"{path_obj}: flow pod_id must be a non-empty string: {entry_dict!r}"
            )
        pod_id_str = pod_id_obj.strip()

        date_obj = entry_dict["date"]
        date_str = date_obj.isoformat() if isinstance(date_obj, date) else str(date_obj)
        if isinstance(entry_dict["amount"], bool):
            raise ValueError(
                f"{path_obj}: flow amount must be numeric, not boolean"
            )
        try:
            parsed_date_obj = date.fromisoformat(date_str)
        except ValueError as exception_obj:
            raise ValueError(
                f"{path_obj}: flow date must be an ISO market date YYYY-MM-DD: {date_str!r}"
            ) from exception_obj
        if parsed_date_obj.isoformat() != date_str:
            raise ValueError(
                f"{path_obj}: flow date must be an ISO market date YYYY-MM-DD: {date_str!r}"
            )

        try:
            amount_float = float(entry_dict["amount"])
        except (TypeError, ValueError) as exception_obj:
            raise ValueError(
                f"{path_obj}: flow amount must be numeric: {entry_dict['amount']!r}"
            ) from exception_obj
        if not math.isfinite(amount_float):
            raise ValueError(
                f"{path_obj}: flow amount must be finite: {entry_dict['amount']!r}"
            )

        normalized_flow_entry_list.append(
            {
                "pod_id_str": pod_id_str,
                "date_str": date_str,
                "amount_float": amount_float,
            }
        )
    return normalized_flow_entry_list


def net_contribution_through_date_float(
    flow_by_date_dict: dict[str, float],
    market_date_str: str,
) -> float:
    """Sum of declared flows dated on or before ``market_date_str``."""
    return float(
        sum(
            amount_float
            for date_str, amount_float in flow_by_date_dict.items()
            if date_str <= market_date_str
        )
    )
