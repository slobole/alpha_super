"""Compliance reviewer: GR1 against the incumbent S9 by block (the plan prints block gaps for every challenger),
and the planning-column (k = 0.75, +5 bps) rung breach of each product at every rung. Read-only."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
WT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(WT / "scripts" / "research" / "fund_products_20261005"))
import g_lib as g
from g_lib import BLOCK_DICT, TBILL, lib
import battery as bt
OUTD = WT / "results/research/portfolio/fund_products_20261005/audit/review/compliance"
lab = g.Lab()
lab._disk_path = OUTD / "never_written_tail_cache.json"
rf = lab.rf
out = {"blocks": {}, "planning_breach": {}}
r1, r9 = lab.ret(g.PRODUCTS["GR1"]), lab.ret(g.INCUMBENT)
r1_5, r9_5 = lab.ret(g.PRODUCTS["GR1"], "s3_plus_5bps"), lab.ret(g.INCUMBENT, "s3_plus_5bps")
for k, (lo, hi) in BLOCK_DICT.items():
    a, b = lib.window(r1, lo, hi), lib.window(r9, lo, hi)
    pr = lab.paired(a, b)
    a5, b5 = lib.window(r1_5, lo, hi), lib.window(r9_5, lo, hi)
    out["blocks"][k] = {"window": [str(lo), str(hi)], "gr1": g.stats(a, rf), "s9": g.stats(b, rf), "gr1_plus5": g.stats(a5, rf), "s9_plus5": g.stats(b5, rf),
                        "share_xs_gr1_over_s9": pr["share_xs"], "share_cagr_gr1_over_s9": pr["share_cagr"], "gap_xs_p5_50_95": pr["gap_xs_p5_50_95"], "paths": pr["paths"]}
    print(k, lo, hi, "GR1", round(out["blocks"][k]["gr1"]["cagr"], 4), round(out["blocks"][k]["gr1"]["xs"], 3), "S9", round(out["blocks"][k]["s9"]["cagr"], 4), round(out["blocks"][k]["s9"]["xs"], 3),
          "share xs", round(pr["share_xs"], 3), "share cagr", round(pr["share_cagr"], 3), flush=True)
# planning convention (k = 0.75, +5 bps): breach at -20 / -25 / -30, mean and worst seed
f5 = lab.frames["s3_plus_5bps"][0]
fr = bt.decay_frame(f5, dict.fromkeys(bt.ALL_CAPS, 0.75))
names = list(g.PRODUCTS) + ["S9"]
R = np.column_stack([lab.ret({**g.PRODUCTS, "S9": g.INCUMBENT}[n], frame_key="s3_plus_5bps", frame=fr).to_numpy() for n in names])
tm = lab.tail_matrix(R, limits=(-0.20, -0.25, -0.30))
for j, n in enumerate(names):
    out["planning_breach"][n] = {str(L): {"mean": float(tm[L][:, j].mean()), "worst": float(tm[L][:, j].max())} for L in tm}
    print("PLAN", n, out["planning_breach"][n], flush=True)
(OUTD / "cp_blocks.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
