"""Review (BC quant lens): does the NATR20 study's data path equal the NATR20 module's own data path?

The 5b study built NATR20 runs from the ATR-VXN module's loader and schedule. This compares, at HEAD, the two
loaders' pricing frame, universe, schedule and VXN frame (trimmed production builder, end 2026-09-25), and runs the
NATR20 module's own month-end helper on prefixes of the protocol's A3 cut-off types. Read-only; no repo file changed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_bc_quant"
OUT.mkdir(parents=True, exist_ok=True)

import strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module  # noqa: E402
import strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled as natr_module  # noqa: E402

END = "2026-09-25"
a = vxn_module.get_vxn_scaled_atr_normalized_ndx_data(replace(vxn_module.DEFAULT_CONFIG, end_date_str=END),
                                                      include_total_return_benchmark_bool=True)
n = natr_module.get_natr20_vxn_scaled_ndx_data(replace(natr_module.DEFAULT_CONFIG, end_date_str=END),
                                               include_total_return_benchmark_bool=True)
out = {}
for name, x, y in zip(("pricing", "universe", "schedule", "vxn"), a, n):
    same_shape = x.shape == y.shape
    cols_equal = set(map(str, x.columns)) == set(map(str, y.columns))
    idx_equal = x.index.equals(y.index)
    val_equal = None
    if same_shape and cols_equal and idx_equal:
        try:
            val_equal = bool(x.loc[:, y.columns].equals(y))
        except Exception as exc:  # noqa: BLE001
            val_equal = repr(exc)[:120]
    out[name] = {"shape_atr_loader": list(x.shape), "shape_natr_loader": list(y.shape), "columns_equal": cols_equal,
                 "index_equal": idx_equal, "values_equal": val_equal,
                 "cols_only_atr": sorted(map(str, set(map(str, x.columns)) - set(map(str, y.columns))))[:10],
                 "cols_only_natr": sorted(map(str, set(map(str, y.columns)) - set(map(str, x.columns))))[:10]}
sched_a, sched_n = a[2], n[2]
out["schedule_tail_atr"] = [str(d.date()) for d in sched_a["decision_date_ts"].iloc[-3:]]
out["schedule_tail_natr"] = [str(d.date()) for d in sched_n["decision_date_ts"].iloc[-3:]]

# NATR20 own month-end helper on A3 cut-off types (prefix of the SPY close series).
spy = a[0][("SPY", "Close")].astype(float).to_frame("SPY")
cut = {"mid_month": "2026-09-15", "weekend_month_end_fri": "2022-04-29", "first_session": "2026-09-01",
       "current_partial": "2026-09-25", "pre_good_friday": "2024-03-28", "last_completed": "2026-08-31"}
me = {}
for label, t in cut.items():
    pref = spy.loc[:t]
    idx = natr_module.get_monthly_decision_close_df(pref).index
    me[label] = {"T": t, "last_decision": str(idx[-1].date())}
out["natr_month_end_prefix"] = me
(OUT / "rbq_natr20_loader_parity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
