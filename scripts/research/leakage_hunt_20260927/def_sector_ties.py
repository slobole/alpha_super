"""Diagnose sector signal-invariance cell differences: are they exact-threshold ties flipped by 1-ulp arithmetic?

For every (symbol, k) and every differing entry/exit cell, record the feature values of reference and candidate and
their distance to the threshold.  Usage: python def_sector_ties.py vox_iyr|kie_ihi
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import def_sector as ds  # noqa: E402  (reads sys.argv[1])
from def_common import FACTORS, OUT, cached, dump_json, harness  # noqa: E402

import numpy as np
import pandas as pd


def features(px):
    sig = ds.signals(px)
    f = {}
    for s in ds.SYMS:
        d = {"ibs": sig[(s, "ibs_value_ser")], "entry": sig[(s, "entry_signal_bool")], "exit": sig[(s, "exit_signal_bool")]}
        if ds.POD == "vox_iyr":
            d["range_ratio"] = sig[(s, "range_ratio_ser")]
            d["downshock"] = sig[(s, "downshock_atr_ser")]
        else:
            d["range_ratio"] = sig[(s, "relative_range_ser")]
            d["sma_gap"] = sig[(s, "Close")] / sig[(s, ds.m.ASSET_SMA_FIELD_STR)] - 1.0
        f[s] = pd.DataFrame(d)
    return f


def main():
    import os
    f64 = os.environ.get("DEF_F64") == "1"
    px = cached(f"sector_{ds.POD}_pricing", ds.load)
    if f64:
        px = px.astype("float64")
    tag = "_float64" if f64 else ""
    cfg = ds.CFG
    thr = {"entry_ibs": cfg.entry_ibs_max_float, "exit_ibs": cfg.exit_ibs_min_float,
           "range": 1.0 if ds.POD == "vox_iyr" else cfg.min_relative_range_float}
    base = features(px)
    rows = []
    for sym in ds.SYMS:
        for k in FACTORS:
            cand = features(harness.rescale_symbol_history(px, sym, k))
            for s in ds.SYMS:
                b, c = base[s], cand[s]
                for kind in ("entry", "exit"):
                    diff = b[kind].astype(bool) != c[kind].astype(bool)
                    for ts in b.index[diff.to_numpy()]:
                        rec = {"rescaled": sym, "k": k, "symbol": s, "date": ts.date().isoformat(), "kind": kind,
                               "ref": bool(b.at[ts, kind]), "cand": bool(c.at[ts, kind]),
                               "ibs_ref": b.at[ts, "ibs"], "ibs_cand": c.at[ts, "ibs"],
                               "rr_ref": b.at[ts, "range_ratio"], "rr_cand": c.at[ts, "range_ratio"]}
                        if "downshock" in b:
                            rec["ds_ref"], rec["ds_cand"] = b.at[ts, "downshock"], c.at[ts, "downshock"]
                        if "sma_gap" in b:
                            rec["sma_gap_ref"], rec["sma_gap_cand"] = b.at[ts, "sma_gap"], c.at[ts, "sma_gap"]
                        dists = [abs(rec["rr_ref"] - thr["range"]),
                                 abs(rec["ibs_ref"] - (thr["entry_ibs"] if kind == "entry" else thr["exit_ibs"]))]
                        if "ds_ref" in rec:
                            dists.append(abs(rec["ds_ref"] - cfg.downshock_atr_max_float))
                        if "sma_gap_ref" in rec:
                            dists.append(abs(rec["sma_gap_ref"]))
                        rec["min_dist_to_threshold"] = float(np.nanmin(dists))
                        rec["rr_ref_minus_1_exact"] = float(rec["rr_ref"] - thr["range"])
                        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"sector_{ds.POD}_invariance_cell_diffs{tag}.csv", index=False)
    summ = {"n_cell_diffs": int(len(df))}
    if len(df):
        summ["max_min_dist_to_threshold"] = float(df["min_dist_to_threshold"].max())
        summ["n_with_dist_gt_1e-9"] = int((df["min_dist_to_threshold"] > 1e-9).sum())
        summ["n_range_ratio_exactly_1_in_ref"] = int((df["rr_ref"] == 1.0).sum())
        summ["by_kind"] = df["kind"].value_counts().to_dict()
    # base-rate of exact ties in the reference run
    ties = {}
    for s in ds.SYMS:
        b = base[s]
        ties[s] = {"range_ratio_exactly_threshold": int((b["range_ratio"] == thr["range"]).sum()),
                   "ibs_exactly_entry_thr": int((b["ibs"] == thr["entry_ibs"]).sum()),
                   "ibs_exactly_exit_thr": int((b["ibs"] == thr["exit_ibs"]).sum())}
    summ["exact_ties_in_reference"] = ties
    dump_json(summ, f"sector_{ds.POD}_invariance_cell_diff_summary{tag}.json")
    print(summ)


if __name__ == "__main__":
    main()
