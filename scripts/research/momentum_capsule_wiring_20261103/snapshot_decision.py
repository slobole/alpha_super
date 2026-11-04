"""One real snapshot decision for each leg through the live host, against the backtest path read directly from Norgate on the same session. Flat account, no broker, no order.

Usage: snapshot_decision.py <snapshot_root> <YYYY-MM-DD> [leg numbers]. Each leg runs in its own process, so its peak memory is its own. Writes results/research/momentum_capsule_wiring_20261103/snapshot_decision_<date>.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_PATH))
OUT_PATH = REPO_PATH / "results" / "research" / "momentum_capsule_wiring_20261103"
MARKET_ZONE_OBJ = ZoneInfo("America/New_York")
TEMPLATE_DICT = {
    "1": "pod_mo_capsule_leg1_natr_ndx_month_swap.yaml.example",
    "2": "pod_mo_capsule_leg2_residual_ndx_month_swap.yaml.example",
    "3": "pod_mo_capsule_leg3_residual_r1000_month_swap.yaml.example",
}


def path_dict_for(leg_str: str, date_str: str) -> dict:
    """Every decision of the leg's path through T, in the active data mode."""
    from dataclasses import replace

    from strategies.momentum.momentum_capsule_legs import LEG1_CONFIG, LEG2_CONFIG, LEG3_CONFIG, build_leg_panel, build_target_path, load_leg_inputs

    config_obj = replace({"1": LEG1_CONFIG, "2": LEG2_CONFIG, "3": LEG3_CONFIG}[leg_str], end_date_str=date_str)
    input_dict = load_leg_inputs(config_obj)
    pricing_data_df = input_dict.pop("pricing_data_df")
    path_obj = build_target_path(config_obj, build_leg_panel(config_obj, pricing_data_df, **input_dict))
    return {str(decision_ts.date()): {str(k): float(v) for k, v in target_ser.items()} for decision_ts, target_ser in path_obj.target_by_decision_dict.items()}


def decide_live(leg_str: str, date_str: str) -> dict:
    import psutil

    from alpha.live import strategy_host
    from alpha.live.release_manifest import parse_release_manifest
    from data.norgate_snapshot_store import use_norgate_data_profile

    t0_float = time.time()
    release_obj = parse_release_manifest(str(REPO_PATH / f"docs/live/release_templates/{TEMPLATE_DICT[leg_str]}"))
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, datetime.fromisoformat(date_str).replace(hour=18, tzinfo=MARKET_ZONE_OBJ), None)
    metadata_dict = decision_obj.snapshot_metadata_dict
    seconds_float, peak_float = round(time.time() - t0_float, 1), round(psutil.Process().memory_info().peak_wset / 2**30, 2)
    with use_norgate_data_profile(release_obj.data_profile_str):
        path_dict = path_dict_for(leg_str, date_str)
    return {
        "path_dict": path_dict,
        "snapshot_date_str": metadata_dict.get("norgate_snapshot_date_str"),
        "manifest_hash_str": str(metadata_dict.get("norgate_manifest_hash_str"))[:12],
        "book_offset_int": metadata_dict.get("book_offset_int"),
        "gate_open_bool": metadata_dict.get("gate_open_bool"),
        "member_count_int": metadata_dict.get("member_count_int"),
        "finite_score_member_count_int": metadata_dict.get("finite_score_member_count_int"),
        "target_without_unpadded_close_list": metadata_dict.get("target_without_unpadded_close_list"),
        "weight_dict": decision_obj.full_target_weight_map_dict,
        "cash_float": decision_obj.cash_reserve_weight_float,
        "target_execution_str": decision_obj.target_execution_timestamp_ts.isoformat(),
        "seconds_float": seconds_float,
        "peak_working_set_gb_float": peak_float,
    }


def decide_direct(leg_str: str, date_str: str) -> dict:
    import psutil

    t0_float = time.time()
    path_dict = path_dict_for(leg_str, date_str)
    return {"path_dict": path_dict, "weight_dict": path_dict[date_str], "seconds_float": round(time.time() - t0_float, 1), "peak_working_set_gb_float": round(psutil.Process().memory_info().peak_wset / 2**30, 2)}


def run_one(mode_str: str, leg_str: str, date_str: str, snapshot_root_str: str) -> dict:
    environment_dict = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    if mode_str == "live":
        environment_dict.update({"ALPHA_USE_NORGATE_SNAPSHOT_BOOL": "true", "NORGATE_SNAPSHOT_ROOT": snapshot_root_str})
    else:
        environment_dict.pop("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", None)
    done_obj = subprocess.run([sys.executable, __file__, "--one", mode_str, leg_str, date_str], env=environment_dict, cwd=REPO_PATH, capture_output=True, text=True, encoding="utf-8")
    line_list = [line_str for line_str in done_obj.stdout.splitlines() if line_str.startswith("RESULT ")]
    return json.loads(line_list[-1][7:]) if line_list else {"error_str": done_obj.stderr[-1800:]}


def main(snapshot_root_str: str, date_str: str, leg_list: list[str]) -> None:
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    out_file_path = OUT_PATH / f"snapshot_decision_{date_str}.json"
    result_dict = json.loads(out_file_path.read_text(encoding="utf-8")) if out_file_path.exists() else {}
    for leg_str in leg_list:
        live_dict, direct_dict = run_one("live", leg_str, date_str, snapshot_root_str), run_one("direct", leg_str, date_str, snapshot_root_str)
        same_names_bool, largest_gap_float, path_summary_dict = None, None, {}
        if "weight_dict" in live_dict and "weight_dict" in direct_dict:
            same_names_bool = set(live_dict["weight_dict"]) == set(direct_dict["weight_dict"])
            largest_gap_float = max((abs(live_dict["weight_dict"][k] - direct_dict["weight_dict"][k]) for k in set(live_dict["weight_dict"]) & set(direct_dict["weight_dict"])), default=0.0)
            live_path_dict, direct_path_dict = live_dict.pop("path_dict"), direct_dict.pop("path_dict")
            differ_list = [date_str_ for date_str_ in sorted(set(live_path_dict) | set(direct_path_dict)) if live_path_dict.get(date_str_) != direct_path_dict.get(date_str_)]
            path_summary_dict = {"decisions_int": len(direct_path_dict), "differ_int": len(differ_list), "first_differ_list": differ_list[:5]}
        result_dict[leg_str] = {"live": live_dict, "direct": direct_dict, "same_names_bool": same_names_bool, "largest_weight_gap_float": largest_gap_float, "whole_path_dict": path_summary_dict}
        print("leg", leg_str, json.dumps(result_dict[leg_str], default=str)[:1800], flush=True)
    out_file_path.write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")
    print("SNAPSHOT DECISIONS DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        result_dict = (decide_live if sys.argv[2] == "live" else decide_direct)(sys.argv[3], sys.argv[4])
        print("RESULT " + json.dumps(result_dict, default=str), flush=True)
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3:] or list(TEMPLATE_DICT))
