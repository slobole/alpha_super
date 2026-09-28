"""Probe: what does the NDX live host return when invoked mid-month / after the month-end?

B2 check. Calls the real host entry point with several as_of timestamps and records the
signal and target-execution dates it produces. No state, no broker.
"""
from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ndx_live_parity_replay import _build_release, _cached_build_universe  # noqa: E402
import strategies.momentum.strategy_mo_atr_normalized_ndx as atr_module  # noqa: E402
from alpha.live import strategy_host  # noqa: E402
NY = ZoneInfo("America/New_York")
atr_module.build_index_constituent_matrix = _cached_build_universe
release_obj = _build_release("vxn")
probe_list = [
    ("mid_month_2026_09_25_evening", datetime(2026, 9, 25, 20, 0, tzinfo=NY)),
    ("mid_month_2026_09_28_morning", datetime(2026, 9, 28, 9, 0, tzinfo=NY)),
    ("month_end_2026_08_31_evening", datetime(2026, 8, 31, 20, 0, tzinfo=NY)),
    ("first_session_2026_09_01_premarket", datetime(2026, 9, 1, 8, 0, tzinfo=NY)),
    ("first_session_2026_09_01_after_open", datetime(2026, 9, 1, 11, 0, tzinfo=NY)),
    ("third_session_2026_09_03_evening", datetime(2026, 9, 3, 20, 0, tzinfo=NY)),
]
out = []
for label, as_of in probe_list:
    row = {"probe": label, "as_of": as_of.isoformat()}
    try:
        plan = strategy_host.build_decision_plan_for_release(release_obj, as_of, None)
        row.update({
            "status": "plan_built",
            "signal_date": pd.Timestamp(plan.signal_timestamp_ts).tz_convert(NY).isoformat(),
            "target_execution": pd.Timestamp(plan.target_execution_timestamp_ts).tz_convert(NY).isoformat(),
            "execution_in_past": pd.Timestamp(plan.target_execution_timestamp_ts) <= pd.Timestamp(as_of),
            "n_targets": len(plan.full_target_weight_map_dict),
            "weights": plan.full_target_weight_map_dict,
        })
    except Exception as e:  # noqa: BLE001
        row.update({"status": f"error: {type(e).__name__}: {e}"[:400]})
    print(json.dumps(row, default=str), flush=True)
    out.append(row)
out_path = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/ndx/ndx_invocation_timing_probe.json"
out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
