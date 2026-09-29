"""V1 diagnostic (research only): which candidate events differ between the base panel and the per-stock rescaled
panel, and how far their jump return sits from the threshold J. Confirmations, intents and positions were identical
in invariance_v1.json; this script documents the event-flag flips. Writes _cache/diag_v1_events.json.

    python scripts/research/merger_arb_v2_20260927/diag_v1_events.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merger_arb_v2_20260927 import common  # noqa: E402
from merger_arb_v2_20260927 import data as data_module  # noqa: E402
from merger_arb_v2_20260927 import invariance  # noqa: E402
from merger_arb_v2_20260927.features import V2FeatureBook, jump_ret_arr  # noqa: E402


def main() -> None:
    panel_dict = data_module.load_panel()
    base_obj = V2FeatureBook(panel_dict)
    scaled_dict = invariance.scaled_panel(panel_dict)
    scaled_obj = V2FeatureBook(scaled_dict)
    jump_base_arr = jump_ret_arr(base_obj.close_arr)
    jump_scaled_arr = jump_ret_arr(scaled_obj.close_arr)
    out: dict = {}
    for jump_float in (0.10, 0.15):
        base_event_arr, scaled_event_arr = base_obj.event(jump_float), scaled_obj.event(jump_float)
        diff_pairs = np.argwhere(base_event_arr != scaled_event_arr)
        row_list = []
        for d_int, s_int in diff_pairs:
            row_list.append({"date": str(base_obj.date_index[d_int].date()), "symbol": base_obj.symbol_list[s_int],
                             "close_prev": float(panel_dict["close_arr"][d_int - 1, s_int]), "close_d": float(panel_dict["close_arr"][d_int, s_int]),
                             "jump_base_minus_J": float(jump_base_arr[d_int, s_int] - jump_float), "jump_scaled_minus_J": float(jump_scaled_arr[d_int, s_int] - jump_float),
                             "base_event": bool(base_event_arr[d_int, s_int]), "scaled_event": bool(scaled_event_arr[d_int, s_int])})
        # do any of the flipped events confirm at the loosest theta (0.8%) for W 3/5/10 in either book?
        conf_flip_int = 0
        for w_int in (3, 5, 10):
            conf_base_arr = base_obj.confirmation(jump_float, 0.008, w_int)["confirmed_at_event"]
            conf_scaled_arr = scaled_obj.confirmation(jump_float, 0.008, w_int)["confirmed_at_event"]
            for d_int, s_int in diff_pairs:
                conf_flip_int += int(conf_base_arr[d_int, s_int] != conf_scaled_arr[d_int, s_int])
        out[f"J{int(round(jump_float * 100))}"] = {
            "events_base_int": int(base_event_arr.sum()), "events_scaled_int": int(scaled_event_arr.sum()), "differing_events_int": int(len(diff_pairs)),
            "max_abs_distance_to_J_base": float(max([abs(r["jump_base_minus_J"]) for r in row_list] + [0.0])),
            "max_abs_distance_to_J_scaled": float(max([abs(r["jump_scaled_minus_J"]) for r in row_list] + [0.0])),
            "confirmation_flips_theta0.8_W3_5_10_int": conf_flip_int, "rows": row_list[:60]}
        print(jump_float, {k: v for k, v in out[f"J{int(round(jump_float * 100))}"].items() if k != "rows"}, flush=True)
    path_obj = Path(common.CACHE_DIR_PATH) / "diag_v1_events.json"
    path_obj.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written", path_obj)


if __name__ == "__main__":
    main()
