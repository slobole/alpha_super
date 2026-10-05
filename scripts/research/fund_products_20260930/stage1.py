"""SPEC 3: family blocks. Viability, near-duplicates (corr >= 0.97), dominance (>= 90% of paths by Sharpe), equal split.

Usage: python stage1.py [frame]   (default main). Writes <study>/<frame>/stage1.json and block_returns.csv.gz.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import fp_lib as fp
from fp_lib import TBILL, Book, ga, lib


def main(frame_name: str = "main") -> int:
    data = lib.load_inputs()
    frame, start = ga.frames(data)[frame_name]
    meta = data["meta"]
    index_all = data["index"]
    tb = frame[TBILL]
    out = {"frame": frame_name, "families": {}}
    blocks = {}
    for fam, variants in fp.FAMILIES.items():
        R = frame.loc[start:ga.END, variants]
        info = {}
        for v in variants:
            r = R[v]
            ok_hist = r.notna().all()
            xs = {b: (lib.excess_cagr(lib.window(r, *ga.BLOCK_DICT[b]), tb, index_all) if ok_hist and b != "A" or
                      (ok_hist and ga.BLOCK_DICT[b][0] >= r.index[0]) else np.nan) for b in ("B", "C", "RECENT")}
            st = ga.window_stats(r.to_frame(), index_all) if ok_hist else None
            info[v] = {"tier": fp.tier(v, meta), "daily": v in fp.DAILY, "full_history": bool(ok_hist),
                       "xs": {k: float(x) for k, x in xs.items()},
                       "viable": bool(ok_hist and all(x > 0 for x in xs.values())),
                       "cagr": float(st["gross_cagr"][0]) if st else None, "sharpe": float(st["gross_sharpe"][0]) if st else None,
                       "dd": float(st["gross_dd"][0]) if st else None}
        viable = [v for v in variants if info[v]["viable"]]
        corr = R[viable].corr() if len(viable) > 1 else pd.DataFrame()
        spx = data["bench"]["SPXTR"].reindex(R.index)
        worst = spx <= spx.quantile(0.05)
        corr_falls = R.loc[worst, viable].corr() if len(viable) > 1 else pd.DataFrame()
        # Near-duplicates: keep the more mature tier, then the higher LONG Sharpe.
        rank_tier = {"wired": 0, "pm-ready": 1, "shadow": 2}
        keep = sorted(viable, key=lambda v: (rank_tier[info[v]["tier"]], -info[v]["sharpe"]))
        merged = []
        for v in keep:
            if any(corr.at[v, u] >= 0.97 for u in merged):
                info[v]["merged_into"] = next(u for u in merged if corr.at[v, u] >= 0.97)
                continue
            merged.append(v)
        # Dominance by bootstrap Sharpe.
        if len(merged) > 1:
            A = R[merged].to_numpy()
            idx = ga.boot_index(len(A))
            sh = np.empty((idx.shape[0], len(merged)))
            for k in range(idx.shape[0]):
                s = A[idx[k]]
                sh[k] = s.mean(axis=0) / s.std(axis=0, ddof=1) * np.sqrt(252)
            beats = {f"{a}>{b}": float(np.mean(sh[:, i] > sh[:, j])) for i, a in enumerate(merged) for j, b in enumerate(merged) if i != j}
            dominated = {b for i, a in enumerate(merged) for j, b in enumerate(merged) if i != j and np.mean(sh[:, i] > sh[:, j]) >= 0.90}
        else:
            beats, dominated = {}, set()
        final_all = [v for v in merged if v not in dominated]
        final_dep = [v for v in final_all if info[v]["tier"] != "shadow" and v not in fp.NOT_LIVE_TRADABLE]
        final_lt = [v for v in final_dep if v not in fp.DAILY]
        out_f = {"variants": info, "viable": viable, "after_merge": merged, "dominated": sorted(dominated),
                 "block_target": final_all, "block_deployable": final_dep, "block_monthly": final_lt, "sharpe_beats": beats,
                 "corr": corr.round(3).to_dict() if len(corr) else {}, "corr_falls": corr_falls.round(3).to_dict() if len(corr_falls) else {}}
        for kind, members in (("target", final_all), ("deployable", final_dep), ("monthly", final_lt)):
            if not members:
                continue
            w = {m: 1.0 / len(members) for m in members}
            r_blk = lib.book_returns(frame, Book(f"{fam}|{kind}", tuple(members), "EQ", w), start)
            blocks[f"{fam}|{kind}"] = r_blk
            st = ga.window_stats(r_blk.to_frame(), index_all)
            out_f[f"split_{kind}"] = {"members": members, "cagr": float(st["gross_cagr"][0]), "sharpe": float(st["gross_sharpe"][0]),
                                      "dd": float(st["gross_dd"][0])}
        out["families"][fam] = out_f
    blocks["CASH"] = tb.loc[blocks["TAA|target"].index]
    out_dir = fp.STUDY / frame_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(blocks).to_csv(out_dir / "block_returns.csv.gz", float_format="%.10g", compression="gzip")
    (out_dir / "stage1.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    fp.ledger("stage1_finished", frame=frame_name)
    for fam, f in out["families"].items():
        print(fam, "viable", f["viable"], "| merged", f["after_merge"], "| dominated", f["dominated"],
              "| target", f["block_target"], "| deployable", f["block_deployable"], "| monthly", f["block_monthly"])
        for v, i in f["variants"].items():
            print("   ", v, i["tier"], "viable" if i["viable"] else "NOT viable", {k: round(x, 3) for k, x in i["xs"].items()},
                  i["cagr"] and round(i["cagr"], 3), i["sharpe"] and round(i["sharpe"], 2), i["dd"] and round(i["dd"], 3), i.get("merged_into", ""))
        for k in ("target", "deployable", "monthly"):
            if f.get(f"split_{k}"):
                s = f[f"split_{k}"]
                print("    split", k, s["members"], round(s["cagr"], 3), round(s["sharpe"], 2), round(s["dd"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
