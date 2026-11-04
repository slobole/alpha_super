"""Old-vs-new gate: the live NDX and TAA pods' decisions from one checkout, to compare a base checkout with the branch.

Usage: gate_ndx_taa.py <repo root> <output json>. Run once per checkout, one after the other (Norgate loads), then compare the two files with `gate_ndx_taa.py compare <a> <b>`. Never connects to a broker. Direct Norgate, the same data for both runs.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

NEW_YORK_TZ = ZoneInfo("America/New_York")
VARIANT_DICT = {
    "taa_rank_tqqq": ("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash", "norgate_eod_etf_plus_vix_helper"),
    "taa_equal_tqqq": ("strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", "norgate_eod_etf_plus_vix_helper"),
    "ndx_atr": ("strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy", "norgate_eod_ndx_pit"),
    "ndx_atr_vxn": ("strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy", "norgate_eod_ndx_pit_plus_vxn_helper"),
}
DATE_TUPLE = ("2026-09-30", "2026-08-31", "2026-06-30", "2026-03-31", "2025-12-31", "2025-04-30", "2022-06-30", "2020-03-31")


def plan_dict(plan_obj) -> dict:
    return {
        "full_target_weight_map_dict": {str(k): float(v) for k, v in sorted(plan_obj.full_target_weight_map_dict.items())},
        "entry_target_weight_map_dict": {str(k): float(v) for k, v in sorted(plan_obj.entry_target_weight_map_dict.items())},
        "exit_asset_list": sorted(str(a) for a in plan_obj.exit_asset_set),
        "cash_reserve_weight_float": float(plan_obj.cash_reserve_weight_float),
        "signal_timestamp_str": str(plan_obj.signal_timestamp_ts),
        "target_execution_timestamp_str": str(plan_obj.target_execution_timestamp_ts),
        "decision_book_type_str": str(plan_obj.decision_book_type_str),
        "execution_policy_str": str(plan_obj.execution_policy_str),
        "strategy_state_dict": json.loads(json.dumps(plan_obj.strategy_state_dict, sort_keys=True, default=str)),
    }


def run(repo_str: str, output_str: str) -> None:
    sys.path.insert(0, repo_str)
    os.environ["ALPHA_USE_NORGATE_SNAPSHOT_BOOL"] = "false"
    import alpha

    from alpha.live import strategy_host
    from alpha.live.models import LiveRelease

    out: dict = {"alpha_file_str": alpha.__file__, "rows": {}}
    print(out["alpha_file_str"], flush=True)
    for variant_str, (import_str, profile_str) in VARIANT_DICT.items():
        release_obj = LiveRelease(
            release_id_str=f"gate.{variant_str}", user_id_str="gate", pod_id_str=variant_str, account_route_str="DU_GATE_DISABLED", strategy_import_str=import_str, mode_str="paper",
            session_calendar_id_str="XNYS", signal_clock_str="month_end_snapshot_ready", execution_policy_str="next_month_first_open", data_profile_str=profile_str,
            params_dict={"capital_base_float": 1_000_000.0}, risk_profile_str="audit_only", enabled_bool=False, source_path_str="isolated_gate", pod_budget_fraction_float=1.0,
        )
        for date_str in DATE_TUPLE:
            year_int, month_int, day_int = (int(part_str) for part_str in date_str.split("-"))
            as_of_ts = datetime(year_int, month_int, day_int, 20, 0, tzinfo=NEW_YORK_TZ)
            try:
                row_dict = plan_dict(strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, None))
            except Exception as exception_obj:  # the same refusal on both sides is a match, and it is reported
                row_dict = {"error_str": f"{type(exception_obj).__name__}: {exception_obj}"[:300]}
            out["rows"][f"{variant_str}|{date_str}"] = row_dict
            print(variant_str, date_str, row_dict.get("error_str", "ok"), flush=True)
    with open(output_str, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)


def compare(a_str: str, b_str: str) -> None:
    with open(a_str, encoding="utf-8") as handle:
        a_dict = json.load(handle)
    with open(b_str, encoding="utf-8") as handle:
        b_dict = json.load(handle)
    print(a_dict["alpha_file_str"], "\n", b_dict["alpha_file_str"])
    key_list = sorted(set(a_dict["rows"]) | set(b_dict["rows"]))
    differ_list = [key_str for key_str in key_list if a_dict["rows"].get(key_str) != b_dict["rows"].get(key_str)]
    error_int = sum("error_str" in a_dict["rows"].get(key_str, {}) for key_str in key_list)
    print(f"identical {len(key_list) - len(differ_list)}/{len(key_list)}; errors on the base {error_int}; differ {differ_list}")
    for key_str in key_list:
        if "error_str" in a_dict["rows"].get(key_str, {}):
            print(key_str, a_dict["rows"][key_str]["error_str"][:200])

    if not key_list or differ_list:
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2])
