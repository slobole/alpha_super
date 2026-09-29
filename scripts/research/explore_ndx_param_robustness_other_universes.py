"""EXPLORATORY (not pre-registered): universe-fitted settings for S&P 500 and Russell 1000.

Question from the owner after the NDX study: what if the same momentum pod ran on S&P 500 or Russell 1000 with
settings suited to those universes? Two ways of fitting, both reported:
- "in-sample": each stage's plateau centre chosen on standalone 2000-26 Sharpe (optimistic, uses the test data);
- "walk-forward": each stage's plateau centre chosen on 2000-11 only, judged on 2012-21, 2022-26 and inside G3.
The fitted stage choices (score, N/weights, filters, buffer, VXN) are then combined into one configuration per
universe and simulated once. Nothing here is a verdict; any adoption needs its own frozen study.

    uv run python scripts/research/explore_ndx_param_robustness_other_universes.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_ndx_param_robustness_study as an  # noqa: E402
import ndx_param_robustness_core as core  # noqa: E402
from run_ndx_param_robustness_study import capacity_stats  # noqa: E402

STAGE_FIELD_DICT = {
    "S1_score": ("numerator_str", "denominator_str"),
    "S2_n_weight": ("n_int", "weight_str"),
    "S3_filters": ("stock_filter_int", "regime_str"),
    "S4_buffer_offset": ("buffer_int",),
    "S5_vxn": ("vxn_target_float", "vxn_floor_float"),
}


def cell_from_key(key_str: str) -> core.Cell:
    for cell in core.all_cell_list():
        if cell.key_str == key_str:
            return cell
    raise KeyError(key_str)


def main() -> None:
    stage_map_dict = json.loads((an.OUT_PATH / "stage_map.json").read_text())
    taa_long_ser = an.load_taa_long_ser()
    out_dict: dict = {"note": "EXPLORATORY, not pre-registered"}
    for universe_str in ("NDX", "SP500", "R1000"):
        ret_df = pd.read_parquet(an.OUT_PATH / f"returns_{universe_str}_engine.parquet")
        standalone, g3 = an.all_metrics(ret_df, taa_long_ser)
        universe_dict = core.load_universe(universe_str)
        feature_obj = core.FeatureBook(universe_dict)
        adv_arr = feature_obj.adv20_dollar()
        uni_res: dict = {"fits": {}}
        for fit_str, block_str in (("in_sample", "FULL"), ("walk_forward", "P1")):
            fitted_cell = core.ANCHOR_CELL
            centre_dict = {}
            for stage_str, field_tuple in STAGE_FIELD_DICT.items():
                row_list, col_list, key_arr = an.stage_matrix(stage_map_dict[stage_str])
                a0_pos = tuple(int(x) for x in np.argwhere(key_arr == an.A0_KEY)[0])
                cand = an.stage_candidate(stage_str, key_arr, lambda k: standalone[k][block_str]["sharpe"], a0_pos)
                centre_cell = cell_from_key(key_arr[cand["centre"]])
                centre_dict[stage_str] = key_arr[cand["centre"]]
                fitted_cell = dataclasses.replace(fitted_cell, **{f: getattr(centre_cell, f) for f in field_tuple})
            # simulate the combined fitted configuration once
            target_list = core.build_target_list(feature_obj, fitted_cell)
            sim = core.simulate(universe_dict, target_list, adv_arr=adv_arr)
            fitted_ret_ser = sim["return_ser"]
            frame_df = pd.DataFrame({"fitted": fitted_ret_ser, "A0": ret_df[an.A0_KEY], "L": ret_df[an.L_KEY]})
            s_dict, g_dict = an.all_metrics(frame_df, taa_long_ser)
            uni_res["fits"][fit_str] = {
                "stage_centres": centre_dict,
                "combined_cell": fitted_cell.key_str,
                "standalone": s_dict["fitted"],
                "g3": g_dict["fitted"],
                "capacity_2021_2026": capacity_stats(sim["order_frac_arr"], universe_dict["date_index"], pd.Timestamp("2021-01-01")),
                "turnover_x_per_year": float(sim["traded_notional_ser"].loc["2000-01-04":an.END_TS].sum()
                                             / sim["total_ser"].loc["2000-01-04":an.END_TS].mean()
                                             / (len(sim["total_ser"].loc["2000-01-04":an.END_TS]) / 252.0)),
            }
        uni_res["A0"] = {"standalone": standalone[an.A0_KEY], "g3": g3[an.A0_KEY]}
        uni_res["L"] = {"standalone": standalone[an.L_KEY], "g3": g3[an.L_KEY]}
        taa_ser = taa_long_ser.loc["2012-10-02":an.END_TS]
        uni_res["corr_with_taa_2012_2026"] = {
            "A0": float(ret_df[an.A0_KEY].reindex(taa_ser.index).corr(taa_ser)),
            "L": float(ret_df[an.L_KEY].reindex(taa_ser.index).corr(taa_ser)),
        }
        out_dict[universe_str] = uni_res
    ndx_l = pd.read_parquet(an.OUT_PATH / "returns_NDX_engine.parquet")[an.L_KEY]
    for universe_str in ("SP500", "R1000"):
        r = pd.read_parquet(an.OUT_PATH / f"returns_{universe_str}_engine.parquet")[an.A0_KEY]
        out_dict[universe_str]["corr_A0_with_NDX_L_2000_2026"] = float(r.loc["2000-01-04":an.END_TS].corr(ndx_l.loc["2000-01-04":an.END_TS]))
    (an.OUT_PATH / "exploratory_other_universes.json").write_text(json.dumps(out_dict, indent=1, default=str))
    for universe_str in ("NDX", "SP500", "R1000"):
        u = out_dict[universe_str]
        print(f"\n===== {universe_str}  corr w TAA {u['corr_with_taa_2012_2026']}")
        for lab in ("L", "A0"):
            print(f"  {lab:12s} SA", {b: (round(v['cagr'], 3), round(v['sharpe'], 2), round(v['max_dd'], 3)) for b, v in u[lab]["standalone"].items()})
            print(f"  {'':12s} G3", {b: (round(v['cagr'], 3), round(v['sharpe'], 2), round(v['max_dd'], 3)) for b, v in u[lab]["g3"].items()})
        for fit_str, f in u["fits"].items():
            print(f"  {fit_str:12s} {f['combined_cell']}  turnover {f['turnover_x_per_year']:.1f}")
            print(f"  {'':12s} SA", {b: (round(v['cagr'], 3), round(v['sharpe'], 2), round(v['max_dd'], 3)) for b, v in f["standalone"].items()})
            print(f"  {'':12s} G3", {b: (round(v['cagr'], 3), round(v['sharpe'], 2), round(v['max_dd'], 3)) for b, v in f["g3"].items()})
            print(f"  {'':12s} cap 2021+", {k: f"{v / 1e6:.1f}M" for k, v in f["capacity_2021_2026"].items() if "aum" in k})
    for universe_str in ("SP500", "R1000"):
        print(universe_str, "corr A0 vs NDX L", round(out_dict[universe_str]["corr_A0_with_NDX_L_2000_2026"], 2))


if __name__ == "__main__":
    main()
