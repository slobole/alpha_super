"""
Audit probe (read-only, local Norgate direct mode, no broker): what does the live NDX
VXN host return when invoked inside a partial month?

The local Norgate vintage ends 2026-09-25 (mid-month for September 2026), so calling
strategy_host._build_atr_normalized_ndx_decision_plan with as_of = 2026-09-25 18:00 ET
reproduces "invoked mid-month / month-end bar missing". Expected (code read): signal =
2026-08-31, target execution = 2026-09-01 09:30 ET (already past), no exception.

Output: results/research/strategy_readiness_audit_20260928/live_path/ndx_host_partial_month_probe.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from alpha.live import scheduler_utils
from alpha.live.models import LiveRelease
from alpha.live.strategy_host import build_decision_plan_for_release

OUTPUT_PATH_OBJ = Path(
    "results/research/strategy_readiness_audit_20260928/live_path/ndx_host_partial_month_probe.json"
)


def main() -> None:
    release_obj = LiveRelease(
        release_id_str="audit.ndx_vxn.partial_month",
        user_id_str="audit_user",
        pod_id_str="pod_audit_ndx_vxn",
        account_route_str="U000AUDIT",
        strategy_import_str=(
            "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy"
        ),
        mode_str="live",
        session_calendar_id_str="XNYS",
        signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open",
        data_profile_str="norgate_eod_ndx_pit_plus_vxn_helper",
        params_dict={
            "max_positions_int": 10,
            "lookback_month_int": 12,
            "index_trend_window_int": 200,
            "stock_trend_window_int": 100,
            "regime_symbol_str": "SPY",
            "vxn_symbol_str": "$VXN",
            "target_vxn_pct_float": 22.0,
            "min_exposure_scale_float": 0.25,
            "max_exposure_scale_float": 1.0,
        },
        risk_profile_str="audit",
        enabled_bool=True,
        source_path_str="audit.yaml",
        pod_budget_fraction_float=1.0,
        auto_submit_enabled_bool=False,
    )
    as_of_ts = datetime(2026, 9, 25, 18, 0, tzinfo=ZoneInfo("America/New_York"))
    plan_obj = build_decision_plan_for_release(release_obj=release_obj, as_of_ts=as_of_ts, pod_state_obj=None)
    result_dict = {
        "as_of_str": as_of_ts.isoformat(),
        "signal_timestamp_str": plan_obj.signal_timestamp_ts.isoformat(),
        "submission_timestamp_str": plan_obj.submission_timestamp_ts.isoformat(),
        "target_execution_timestamp_str": plan_obj.target_execution_timestamp_ts.isoformat(),
        "target_already_past_bool": bool(
            scheduler_utils.is_execution_window_expired_bool(
                plan_obj.execution_policy_str, plan_obj.target_execution_timestamp_ts, as_of_ts
            )
        ),
        "full_target_weight_map_dict": plan_obj.full_target_weight_map_dict,
        "snapshot_metadata_dict": plan_obj.snapshot_metadata_dict,
    }
    OUTPUT_PATH_OBJ.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH_OBJ.write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result_dict, indent=2, default=str))


if __name__ == "__main__":
    main()
