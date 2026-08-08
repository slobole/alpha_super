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

    Missing file or empty list means no declared flows — identical to today's
    behaviour. A malformed entry fails loud: silently dropping a deposit
    would quietly re-inflate the very returns this module exists to fix.
    """
    path_obj = flows_path or resolve_pod_cash_flows_path()
    if not path_obj.is_file():
        return {}
    payload_obj = yaml.safe_load(path_obj.read_text(encoding="utf-8")) or {}
    flow_entry_list = payload_obj.get("flows") or []
    if not isinstance(flow_entry_list, list):
        raise ValueError(f"{path_obj}: 'flows' must be a list")
    flow_by_date_dict: dict[str, float] = defaultdict(float)
    for entry_dict in flow_entry_list:
        if not isinstance(entry_dict, dict):
            raise ValueError(f"{path_obj}: flow entries must be mappings, got {entry_dict!r}")
        missing_key_list = [k for k in ("pod_id", "date", "amount") if k not in entry_dict]
        if missing_key_list:
            raise ValueError(f"{path_obj}: flow entry missing {missing_key_list}: {entry_dict!r}")
        if str(entry_dict["pod_id"]) != pod_id_str:
            continue
        date_str = str(entry_dict["date"])
        amount_float = float(entry_dict["amount"])
        flow_by_date_dict[date_str] += amount_float
    return dict(flow_by_date_dict)


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
