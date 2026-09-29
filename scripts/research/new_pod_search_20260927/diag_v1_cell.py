"""Diagnostic for the V1 cell flagged with position differences (research only; changes no verdict).

Re-runs the flagged cell on the base and rescaled universes with the engine's phantom re-size fills cancelled in both
(trend study note N6) and compares decisions and daily positions; also lists the mismatching sessions of the
engine-rule runs with the raw price of the names involved, to show whether the flips are 0-vs-1 raw-share targets.

    python scripts/research/new_pod_search_20260927/diag_v1_cell.py "S|SE_2_5|N50|HEDGED|k+0"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from new_pod_search_20260927 import cells as cells_module  # noqa: E402
from new_pod_search_20260927 import common  # noqa: E402
from new_pod_search_20260927 import data as data_module  # noqa: E402
from new_pod_search_20260927 import invariance  # noqa: E402
from new_pod_search_20260927.simulate import simulate  # noqa: E402


def main() -> None:
    key_str = sys.argv[1] if len(sys.argv) > 1 else "S|SE_2_5|N50|HEDGED|k+0"
    cell = next(c for c in cells_module.family_s_cells() if c.key_str == key_str)
    universe_dict = data_module.get_universe("SP500", sh_bool=True)
    base_ctx = invariance._context("S", universe_dict)
    scaled_ctx = invariance._context("S", invariance.scaled_universe(universe_dict))
    report_dict = {"cell": key_str}
    for mode_str, cancel_bool in (("engine_rule", False), ("phantom_free", True)):
        base_sim = simulate(base_ctx["features"], invariance._policy(base_ctx, cell), record_positions_bool=True, phantom_cancel_bool=cancel_bool)
        scaled_sim = simulate(scaled_ctx["features"], invariance._policy(scaled_ctx, cell), record_positions_bool=True, phantom_cancel_bool=cancel_bool)
        mismatch_list = []
        for (p0, h0, _), (p1, h1, _) in zip(base_sim["position_log"], scaled_sim["position_log"]):
            if set(h0) != set(h1):
                diff_idx = sorted(set(h0) ^ set(h1))
                mismatch_list.append({"date": str(base_ctx["features"].date_index[p0].date()), "symbols": [base_ctx["features"].symbol_list[i] for i in diff_idx],
                                      "raw_close": [float(universe_dict["unadjusted_close_arr"][p0, i]) for i in diff_idx], "nav_base": float(base_sim["total_ser"].iloc[p0 - base_sim["start_pos_int"]]),
                                      "nav_scaled": float(scaled_sim["total_ser"].iloc[p0 - scaled_sim["start_pos_int"]])})
        b_df, s_df = base_sim["intent_df"], scaled_sim["intent_df"]
        col_list = ["decision_pos", "symbol_idx", "kind", "reason"]
        report_dict[mode_str] = {
            "position_mismatch_sessions_int": len(mismatch_list),
            "intents_identical_bool": len(b_df) == len(s_df) and bool((b_df[col_list].to_numpy() == s_df[col_list].to_numpy()).all()),
            "max_rel_nav_diff_float": float(np.max(np.abs(scaled_sim["total_ser"].to_numpy() / base_sim["total_ser"].to_numpy() - 1.0))),
            "phantom_fills_base_int": int(base_sim["phantom_fill_int"]), "phantom_fills_scaled_int": int(scaled_sim["phantom_fill_int"]),
            "first_mismatches": mismatch_list[:6],
            "mismatch_names_min_raw_close": float(min(x for m in mismatch_list for x in m["raw_close"])) if mismatch_list else None,
        }
        print(mode_str, json.dumps({k: v for k, v in report_dict[mode_str].items() if k != "first_mismatches"}), flush=True)
    common.write_json("diag_v1_flagged_cell.json", report_dict)


if __name__ == "__main__":
    main()
