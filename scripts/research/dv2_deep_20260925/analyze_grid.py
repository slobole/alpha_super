"""Apply the frozen promotion rules to the Phase 3 grid (research-only)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch  # noqa: E402
import phase3_grid as g3  # noqa: E402

OUT = batch.OUT
PRIOR_TRIALS = 50  # ~15 DV2 variant files, deeper-dip 24 cells, liquidity test 3, the 5% change, source/course variants


def main():
    grid = pd.read_csv(OUT / "phase3_grid_summary.csv", index_col=0)
    stress = pd.read_csv(OUT / "phase3_grid_stress_summary.csv", index_col=0)
    nav = batch.load_nav(list(dict.fromkeys(grid["run_id"].tolist())))
    ret = nav.pct_change()
    base_id = grid.loc["floor", "run_id"]
    wired_id = grid.loc["wired", "run_id"]
    luck = grid[grid.group == "luck"]
    cand = grid[grid.group != "luck"]
    n_trials = len(cand) + PRIOR_TRIALS
    trial_sr = cand["sharpe"].to_numpy()
    rows = []
    for name, row in cand.iterrows():
        rid = row["run_id"]
        bs = batch.paired_bootstrap(ret[rid], ret[base_id])
        dsr = batch.deflated_sharpe(ret[rid].dropna().to_numpy(), trial_sr, n_trials)
        srow = stress.loc[f"{name}__stress"] if f"{name}__stress" in stress.index else None
        rows.append({"name": name, "group": row["group"], "cagr": row["cagr"], "sharpe": row["sharpe"], "maxdd": row["maxdd"],
                     "calmar": row["calmar"], "turnover_x": row["turnover_x"], "entries_py": row["entries_per_year"],
                     "P1": row["P1_sharpe"], "P2": row["P2_sharpe"], "P3": row["P3_sharpe"],
                     "d_sharpe": bs["dsharpe"], "p_le_0": bs["p_le_0"], "d_q05": bs["q05"], "d_q95": bs["q95"], "corr_base": bs["corr"],
                     "dP1": row["P1_sharpe"] - grid.loc["floor", "P1_sharpe"], "dP2": row["P2_sharpe"] - grid.loc["floor", "P2_sharpe"],
                     "dP3": row["P3_sharpe"] - grid.loc["floor", "P3_sharpe"], "dsr": dsr["dsr"],
                     "stress_sharpe": srow["sharpe"] if srow is not None else np.nan,
                     "stress_cagr": srow["cagr"] if srow is not None else np.nan,
                     "stress_calmar": srow["calmar"] if srow is not None else np.nan})
    df = pd.DataFrame(rows).set_index("name")
    base = df.loc["floor"]
    df["d_stress_sharpe"] = df["stress_sharpe"] - base["stress_sharpe"]
    df["d_maxdd"] = df["maxdd"] - base["maxdd"]
    df["d_turnover_pct"] = df["turnover_x"] / base["turnover_x"] - 1
    # "better" rule (i)-(vi); plateau (iii) evaluated below for axis members
    df["better_i"] = df["p_le_0"] < 0.05
    df["better_ii"] = (df[["dP1", "dP2", "dP3"]] > 0).all(axis=1)
    df["better_iv"] = df["dsr"] >= 0.95
    df["better_v"] = df["d_turnover_pct"] <= 0.25
    df["better_vi"] = df["d_stress_sharpe"] > 0
    df["better_candidate"] = df[["better_i", "better_ii", "better_iv", "better_v", "better_vi"]].all(axis=1)
    # non-inferiority for robustness replacements
    df["noninferior"] = ((df["d_sharpe"] >= -0.05) & (df["d_q05"] >= -0.15) & (df["d_maxdd"] >= -0.03)
                         & (df[["dP1", "dP2", "dP3"]] >= -0.10).all(axis=1))
    df.to_csv(OUT / "phase3_grid_evaluated.csv", float_format="%.4f")
    # luck band summary
    lb = {"n": int(len(luck)), "sharpe_q": {q: float(luck.sharpe.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
          "cagr_q": {q: float(luck.cagr.quantile(q)) for q in (0.05, 0.5, 0.95)},
          "maxdd_q": {q: float(luck.maxdd.quantile(q)) for q in (0.05, 0.5, 0.95)},
          "floor_natr_sharpe": float(base["sharpe"]), "floor_natr_pct_in_band": float((luck.sharpe < base["sharpe"]).mean()),
          "rank_adv_pct_in_band": float((luck.sharpe < df.loc["rank_adv", "sharpe"]).mean()),
          "rank_dv_pct_in_band": float((luck.sharpe < df.loc["rank_dv", "sharpe"]).mean()),
          "n_trials_for_dsr": n_trials}
    (OUT / "phase3_luck_band.json").write_text(json.dumps(lb, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cols = ["cagr", "sharpe", "maxdd", "turnover_x", "P1", "P2", "P3", "d_sharpe", "p_le_0", "d_q05", "dsr", "stress_sharpe", "better_candidate", "noninferior"]
    print(json.dumps(lb, indent=1))
    print(df[cols].round(3).to_string())


if __name__ == "__main__":
    main()
