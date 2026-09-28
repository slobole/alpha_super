"""Linearity-family split-harness positive control with an ADDITIVE (non-homogeneous) planted defect: log(P + 5)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq_taa_positive_controls as pc

real_lin = pc.linearity_module.compute_daily_linearity_score_df
def additive_lin(signal_close_df, lookback_day_vec):
    return real_lin(signal_close_df=signal_close_df + 5.0, lookback_day_vec=lookback_day_vec)
rows = []
for sym in ("GLD", "TLT", "BTAL", "UUP"):
    clean = pc._split_case("btal_qqq", sym, 40.0)
    pc.linearity_module.compute_daily_linearity_score_df = additive_lin
    try:
        planted = pc._split_case("btal_qqq", sym, 40.0)
    finally:
        pc.linearity_module.compute_daily_linearity_score_df = real_lin
    rows.append({"symbol": sym, "k": 40.0, "clean": clean, "planted_additive": planted, "caught": planted > 1e-12})
    print(rows[-1], flush=True)
(pc.OUT_DIR_PATH / "rq_taa_pc_linearity_split.json").write_text(json.dumps(rows, indent=2))
