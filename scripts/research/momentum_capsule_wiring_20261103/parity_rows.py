"""The wired legs' decisions against the closed design's, book by book. Reads the saved research rows; runs on this checkout's modules.

Usage: python scripts/research/momentum_capsule_wiring_20261103/parity_rows.py <leg 1|2|3> <saved rows parquet>
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

CONFIG_DICT = {"1": (LEG1_CONFIG, "A"), "2": (LEG2_CONFIG, "B"), "3": (LEG3_CONFIG, "C")}
OUT_PATH = REPO_PATH / "results" / "research" / "momentum_capsule_wiring_20261103"


def main() -> None:
    leg_str, rows_path_str = sys.argv[1], sys.argv[2]
    config_obj, research_leg_str = CONFIG_DICT[leg_str]
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    t0_float = time.time()
    inputs_dict = load_leg_inputs(config_obj)
    print("loaded", f"{time.time() - t0_float:.0f}s", inputs_dict["pricing_data_df"].shape, flush=True)
    panel = build_leg_panel(config_obj, inputs_dict["pricing_data_df"], inputs_dict["universe_df"], inputs_dict["helper_close_dict"], inputs_dict["sector_label_dict"], inputs_dict["group_label_dict"])
    path_obj = build_target_path(config_obj, panel)
    print("path", f"{time.time() - t0_float:.0f}s", len(path_obj.book_by_decision_dict), flush=True)
    saved_df = pd.read_parquet(rows_path_str)
    saved_df = saved_df[saved_df["leg"] == research_leg_str]
    rows_list = []
    for (k, decision_ts), part_df in saved_df.groupby(["k", "decision_date"]):
        decision_ts = pd.Timestamp(decision_ts)
        saved_ser = part_df[part_df["symbol"] != "__CASH__"].set_index("symbol")["weight"].astype(float)
        book_ser = path_obj.book_by_decision_dict.get(decision_ts)
        if book_ser is None or path_obj.book_offset_by_decision_dict[decision_ts] != int(k):
            rows_list.append({"k": int(k), "decision_date": decision_ts, "state": "missing", "only_saved": ",".join(saved_ser.index), "only_new": "", "weight_gap": np.nan})
            continue
        only_saved_list, only_new_list = sorted(set(saved_ser.index) - set(book_ser.index)), sorted(set(book_ser.index) - set(saved_ser.index))
        common_list = sorted(set(saved_ser.index) & set(book_ser.index))
        gap_float = float((saved_ser.reindex(common_list) - book_ser.reindex(common_list)).abs().max()) if common_list else 0.0
        rows_list.append({"k": int(k), "decision_date": decision_ts, "state": "same" if not only_saved_list and not only_new_list else "names", "only_saved": ",".join(only_saved_list), "only_new": ",".join(only_new_list), "weight_gap": gap_float, "names": len(saved_ser)})
    compare_df = pd.DataFrame(rows_list).sort_values("decision_date")
    compare_df.to_csv(OUT_PATH / f"parity_rows_leg{leg_str}.csv", index=False)
    extra_list = sorted(set(path_obj.book_by_decision_dict) - set(pd.to_datetime(saved_df["decision_date"].unique())))
    summary_dict = {
        "leg": leg_str,
        "decisions": int(len(compare_df)),
        "same": int((compare_df["state"] == "same").sum()),
        "names_differ": int((compare_df["state"] == "names").sum()),
        "missing": int((compare_df["state"] == "missing").sum()),
        "largest_weight_gap_when_same": float(compare_df.loc[compare_df["state"] == "same", "weight_gap"].max()),
        "extra_decisions_in_new": [str(d.date()) for d in extra_list][:10],
        "first_difference": str(compare_df.loc[compare_df["state"] != "same", "decision_date"].min()),
    }
    (OUT_PATH / f"parity_rows_leg{leg_str}.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    print(json.dumps(summary_dict, indent=2), flush=True)
    print(compare_df[compare_df["state"] != "same"].tail(25).to_string(index=False), flush=True)
    print("PARITY DONE", f"{time.time() - t0_float:.0f}s", flush=True)


if __name__ == "__main__":
    main()
