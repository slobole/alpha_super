"""Distribution of |momentum score - DTB3 hurdle| per decision vs. one-session DTB3 moves (TAA 3x standard family)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq_taa_fee_and_hurdle_margin  # noqa: F401  (sets offline FRED)
import rq_taa_checks as rq

cap = {}
real_fn = rq.base_module.compute_month_end_weight_df
def spy(sc, cr, cfg):
    s, w = real_fn(sc, cr, cfg); cap["score"] = s; cap["cash_daily"] = cr; cap["cash"] = cr.resample("ME").last(); return s, w
rq.base_module.compute_month_end_weight_df = spy
rq._month_end_weights("taa3x", rq._config("taa3x"))
rq.base_module.compute_month_end_weight_df = real_fn
score, cash = cap["score"], cap["cash"]
idx = score.dropna().index.intersection(cash.dropna().index)
margin = score.loc[idx].sub(cash.loc[idx], axis=0)
absm = margin.abs()
d1 = cap["cash_daily"].diff().abs().dropna()
out = {
 "decisions": int(len(idx)),
 "one_day_hurdle_change_p50": float(d1.median()), "p95": float(d1.quantile(.95)), "p99": float(d1.quantile(.99)),
 "asset_months_with_margin_below_p95_daily_move": int((absm < d1.quantile(.95)).sum().sum()),
 "asset_months_with_margin_below_p99_daily_move": int((absm < d1.quantile(.99)).sum().sum()),
 "smallest_10": [(str(i.date()), c, float(margin.loc[i, c])) for (i, c) in absm.stack().nsmallest(10).index],
}
print(json.dumps(out, indent=1))
(rq.OUT_DIR_PATH / "rq_taa_hurdle_margin_dist.json").write_text(json.dumps(out, indent=2))
