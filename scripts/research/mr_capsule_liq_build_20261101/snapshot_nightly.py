"""A pod's first day on the route a VPS uses: the live decision from the snapshot of session T, then the nightly reference from the snapshot of the next session, and the real compare.

Flat account, $1M, both sides. Each step runs in its own process, each with one snapshot root, because the loader takes the root's latest date. No Norgate, no broker.
Usage: snapshot_nightly.py <root_with_T> <root_with_S> <T> <S>. Writes results/research/mr_capsule_liq_build_20261101/snapshot_nightly.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "research" / "mr_capsule_liq_build_20261101"
MARKET_ZONE_OBJ = ZoneInfo("America/New_York")
NAV_FLOAT = 1_000_000.0
POD_LIST = ["dv2_vix_gated", "hpi_vote_vix_gated"]


def _release(pod_str: str):
    from alpha.live.release_manifest import parse_release_manifest

    return parse_release_manifest(str(REPO / f"docs/live/release_templates/pod_mr_{pod_str}_liq_bil_daily_moo.yaml.example"))


def decide(pod_str: str, date_str: str) -> dict:
    from alpha.live import strategy_host
    from alpha.live.models import PodState

    release_obj = _release(pod_str)
    date_ts = datetime.fromisoformat(date_str)
    state_obj = PodState(
        pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str, position_amount_map={},
        cash_float=NAV_FLOAT, total_value_float=NAV_FLOAT, strategy_state_dict={"mr_capsule_strategy_import_str": release_obj.strategy_import_str},
        updated_timestamp_ts=date_ts.replace(hour=16, minute=30, tzinfo=MARKET_ZONE_OBJ), snapshot_stage_str="eod", snapshot_source_str="broker",
    )
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, date_ts.replace(hour=18, tzinfo=MARKET_ZONE_OBJ), state_obj)
    metadata_dict = decision_obj.snapshot_metadata_dict
    return {
        "snapshot_date_str": metadata_dict["norgate_snapshot_date_str"], "manifest_hash_str": metadata_dict["norgate_manifest_hash_str"][:12],
        "plan_weight_dict": dict(decision_obj.entry_target_weight_map_dict), "wanted_weight_dict": metadata_dict["liquidity_entry_wanted_weight_dict"],
        "skipped_list": metadata_dict["liquidity_skipped_entry_list"], "exit_list": sorted(decision_obj.exit_asset_set), "target_share_dict": dict(decision_obj.target_share_map_dict),
        "reference_float": metadata_dict["liquidity_reference_float"],
    }


def reference(pod_str: str, session_str: str) -> dict:
    from alpha.live import backtest_check

    release_obj = _release(pod_str)
    with tempfile.TemporaryDirectory() as folder_str:
        maps_dict = backtest_check.run_reference_maps_dict(release_obj, start_str=session_str, end_str=session_str, capital_float=NAV_FLOAT, output_dir_path=Path(folder_str) / "reference")
    return {
        "trade_dict": {asset_str: float(trade_dict["quantity_float"]) for (date_str, asset_str), trade_dict in maps_dict["transaction_map_dict"].items() if date_str == session_str},
        "weight_dict": {asset_str: float(weight_float) for (date_str, asset_str), weight_float in maps_dict["entry_weight_by_date_dict"].items() if date_str == session_str},
    }


def _child(step_str: str, pod_str: str, date_str: str, root_str: str) -> dict:
    environment_dict = {**os.environ, "ALPHA_USE_NORGATE_SNAPSHOT_BOOL": "true", "NORGATE_SNAPSHOT_ROOT": root_str, "PYTHONDONTWRITEBYTECODE": "1"}
    completed_obj = subprocess.run([sys.executable, __file__, "--step", step_str, pod_str, date_str], env=environment_dict, capture_output=True, text=True)
    line_list = [line_str for line_str in completed_obj.stdout.splitlines() if line_str.startswith("RESULT ")]
    return json.loads(line_list[-1][7:]) if line_list else {"error_str": completed_obj.stderr[-1500:]}


def main(root_t_str: str, root_s_str: str, date_str: str, session_str: str) -> None:
    from alpha.live import backtest_check

    result_dict = {}
    for pod_str in POD_LIST:
        decision_dict, reference_dict = _child("decide", pod_str, date_str, root_t_str), _child("reference", pod_str, session_str, root_s_str)
        row_dict = {"decision_dict": decision_dict, "reference_dict": reference_dict}
        if "error_str" not in decision_dict and "error_str" not in reference_dict:
            comparison_dict = backtest_check.compare_decision_dict(
                {"decision_book_type_str": "incremental_entry_exit_book", "exit_asset_list": decision_dict["exit_list"], "entry_target_weight_map_dict": decision_dict["wanted_weight_dict"]},
                reference_weight_map_dict={}, reference_trade_map_dict=reference_dict["trade_dict"], reference_entry_weight_map_dict=reference_dict["weight_dict"], skip_set=backtest_check.PARKING_ASSET_SET,
                entry_weight_tolerance_float=backtest_check.SIZED_ENTRY_WEIGHT_TOLERANCE_FLOAT,
            )
            row_dict.update(status_str=comparison_dict["status_str"], differ_count_int=comparison_dict["differ_count_int"], row_list=comparison_dict["row_list"])
        result_dict[pod_str] = row_dict
        print(pod_str, json.dumps(row_dict, default=str)[:2200], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snapshot_nightly.json").write_text(json.dumps(result_dict, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "--step":
        step_fn = {"decide": decide, "reference": reference}[sys.argv[2]]
        try:
            print("RESULT " + json.dumps(step_fn(sys.argv[3], sys.argv[4]), default=str), flush=True)
        except Exception as exception_obj:  # report, never hide: a block on real data is a finding
            print("RESULT " + json.dumps({"error_str": f"{type(exception_obj).__name__}: {exception_obj}"}), flush=True)
    else:
        main(*sys.argv[1:5])
