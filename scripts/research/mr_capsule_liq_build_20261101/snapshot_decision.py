"""One real snapshot-mode decision for each sized identity, beside the 1/N identity on the same data and account: the route a VPS uses, which the Norgate replays do not.

Each decision runs in its own process, so the peak memory is comparable. Flat account, $1M, on the latest session of the given snapshot root. No Norgate, no broker.
Usage: snapshot_decision.py <snapshot_root> <YYYY-MM-DD>. Writes results/research/mr_capsule_liq_build_20261101/snapshot_decision.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "research" / "mr_capsule_liq_build_20261101"
MARKET_ZONE_OBJ = ZoneInfo("America/New_York")
NAV_FLOAT = 1_000_000.0


def decide(template_str: str, date_str: str) -> dict:
    import psutil

    from alpha.live import strategy_host
    from alpha.live.models import PodState
    from alpha.live.release_manifest import parse_release_manifest

    date_ts = datetime.fromisoformat(date_str)
    release_obj = parse_release_manifest(str(REPO / f"docs/live/release_templates/{template_str}"))
    state_obj = PodState(
        pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str, position_amount_map={},
        cash_float=NAV_FLOAT, total_value_float=NAV_FLOAT, strategy_state_dict={"mr_capsule_strategy_import_str": release_obj.strategy_import_str},
        updated_timestamp_ts=date_ts.replace(hour=16, minute=30, tzinfo=MARKET_ZONE_OBJ), snapshot_stage_str="eod", snapshot_source_str="broker",
    )
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, date_ts.replace(hour=18, tzinfo=MARKET_ZONE_OBJ), state_obj)
    metadata_dict = decision_obj.snapshot_metadata_dict
    return {
        "strategy_import_str": release_obj.strategy_import_str, "snapshot_date_str": metadata_dict.get("norgate_snapshot_date_str"), "profile_str": metadata_dict.get("norgate_data_profile_str"),
        "manifest_hash_str": str(metadata_dict.get("norgate_manifest_hash_str"))[:12], "gate_open_bool": metadata_dict.get("gate_open_bool"),
        "entry_weight_dict": {asset_str: round(float(weight_float), 5) for asset_str, weight_float in decision_obj.entry_target_weight_map_dict.items()},
        "entry_priority_list": list(decision_obj.entry_priority_list), "target_share_dict": dict(decision_obj.target_share_map_dict), "exit_list": sorted(decision_obj.exit_asset_set),
        "liquidity_reference_float": metadata_dict.get("liquidity_reference_float"), "wanted_weight_dict": metadata_dict.get("liquidity_entry_wanted_weight_dict"), "skipped_list": metadata_dict.get("liquidity_skipped_entry_list"),
        "peak_working_set_gb_float": round(psutil.Process().memory_info().peak_wset / 2**30, 2),
    }


def main(snapshot_root_str: str, date_str: str) -> None:
    result_dict = {}
    environment_dict = {**os.environ, "ALPHA_USE_NORGATE_SNAPSHOT_BOOL": "true", "NORGATE_SNAPSHOT_ROOT": snapshot_root_str, "PYTHONDONTWRITEBYTECODE": "1"}
    for pod_str in ("dv2_vix_gated", "hpi_vote_vix_gated"):
        for mode_str in ("bil", "liq_bil"):
            key_str = f"{pod_str}_{mode_str}"
            completed_obj = subprocess.run([sys.executable, __file__, "--one", f"pod_mr_{pod_str}_{mode_str}_daily_moo.yaml.example", date_str], env=environment_dict, capture_output=True, text=True)
            line_list = [line_str for line_str in completed_obj.stdout.splitlines() if line_str.startswith("RESULT ")]
            result_dict[key_str] = json.loads(line_list[-1][7:]) if line_list else {"error_str": completed_obj.stderr[-1500:]}
            print(key_str, json.dumps(result_dict[key_str], default=str)[:1100], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snapshot_decision.json").write_text(json.dumps(result_dict, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        try:
            print("RESULT " + json.dumps(decide(sys.argv[2], sys.argv[3]), default=str), flush=True)
        except Exception as exception_obj:  # report, never hide: a block on real data is a finding
            print("RESULT " + json.dumps({"error_str": f"{type(exception_obj).__name__}: {exception_obj}"}), flush=True)
    else:
        main(sys.argv[1], sys.argv[2])
