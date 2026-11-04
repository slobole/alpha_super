"""On real data: each sampled decision rebuilt from rows through its own T only must equal the same decision in the full run. Direct Norgate, one load, no account.

Usage: replay_cut.py <leg 1|2|3> [sample count]. Writes results/research/momentum_capsule_wiring_20261103/replay_cut_leg<leg>.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_PATH))
from strategies.momentum.momentum_capsule_legs import LEG1_CONFIG, LEG2_CONFIG, LEG3_CONFIG, build_leg_panel, build_target_path, load_leg_inputs  # noqa: E402

CONFIG_DICT = {"1": LEG1_CONFIG, "2": LEG2_CONFIG, "3": LEG3_CONFIG}
OUT_PATH = REPO_PATH / "results" / "research" / "momentum_capsule_wiring_20261103"


def main(leg_str: str, sample_int: int) -> None:
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    t0_float = time.time()
    config_obj = CONFIG_DICT[leg_str]
    input_dict = load_leg_inputs(config_obj)
    pricing_data_df = input_dict.pop("pricing_data_df")
    full_path_obj = build_target_path(config_obj, build_leg_panel(config_obj, pricing_data_df, **input_dict))
    decision_list = sorted(full_path_obj.target_by_decision_dict)
    print("full path", f"{time.time() - t0_float:.0f}s", len(decision_list), flush=True)
    rng = np.random.default_rng(int(leg_str))
    recent_list = decision_list[-(sample_int // 2):]
    older_list = list(rng.choice(np.array(decision_list[:-(sample_int // 2)], dtype="datetime64[ns]"), size=sample_int - len(recent_list), replace=False))
    row_list = []
    for decision_ts in sorted(pd.Timestamp(date_obj) for date_obj in [*recent_list, *older_list]):
        cut_price_df = pricing_data_df.loc[:decision_ts]
        cut_universe_df = input_dict["universe_df"].loc[:decision_ts]
        cut_helper_dict = {symbol_str: close_ser.loc[:decision_ts] for symbol_str, close_ser in input_dict["helper_close_dict"].items()}
        cut_path_obj = build_target_path(config_obj, build_leg_panel(config_obj, cut_price_df, cut_universe_df, cut_helper_dict, input_dict["sector_label_dict"], input_dict["group_label_dict"]))
        full_ser, cut_ser = full_path_obj.target_by_decision_dict[decision_ts], cut_path_obj.target_by_decision_dict[decision_ts]
        same_names_bool = list(full_ser.index) == list(cut_ser.index)
        gap_float = float((full_ser - cut_ser).abs().max()) if same_names_bool and len(full_ser) else (0.0 if same_names_bool else float("nan"))
        row_list.append({"decision_str": str(decision_ts.date()), "names_int": int(len(full_ser)), "same_names_bool": same_names_bool, "largest_gap_float": gap_float})
        print(row_list[-1], f"{time.time() - t0_float:.0f}s", flush=True)
    summary_dict = {
        "leg": leg_str,
        "decisions_checked": len(row_list),
        "same_names": sum(row_dict["same_names_bool"] for row_dict in row_list),
        "largest_gap": max(row_dict["largest_gap_float"] for row_dict in row_list),
        "rows": row_list,
    }
    (OUT_PATH / f"replay_cut_leg{leg_str}.json").write_text(json.dumps(summary_dict, indent=1), encoding="utf-8")
    print("REPLAY DONE", {key_str: value_obj for key_str, value_obj in summary_dict.items() if key_str != "rows"}, f"{time.time() - t0_float:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
