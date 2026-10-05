"""Margin route (debt pod) and GR1 / defensive-launch correlation, recomputed from own frames."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir_core as c

main = pd.read_pickle(c.OUT / "main_frame.pkl")
plus5 = pd.read_pickle(c.OUT / "plus5_frame.pkl")
bil = main["BIL"]
study = json.loads((c.REPORT / "study.json").read_text(encoding="utf-8"))
B = dict(c.BOOKS)

y = c.dtb3_rate(main.index)
days = pd.Series(main.index, index=main.index).diff().dt.days
days.iloc[0] = 1.0  # 2008-03-03 -> 2008-03-04
debt = (y + 0.015) * days / 360.0


def levered(frame, w, L, spread=0.015):
    f = frame.copy()
    f["debt"] = (y + spread) * days / 360.0
    ww = {k: v * L for k, v in w.items()}
    ww["debt"] = -(L - 1.0)
    # book_returns normalises nothing: weights sum to 1 (L*1 - (L-1))
    return c.book_returns(f, ww)


for name, L in (("GR1 x1.19", 1.19), ("GR1 x1.17", 1.17)):
    r = levered(main, B["GR1"], L)
    m = c.metrics(r, bil)
    q = study["books"][name]["q"]
    print(f"{name}: mine cagr {m['cagr']:.6f} vol {m['vol']:.6f} xs {m['xs']:.6f} dd {m['dd']:.6f} | study {q['cagr']} {q['vol']} {q['xs']} {q['dd']}")
vm = study["margin"]["GR1 -> GR2"]["vol_matched"]
r5 = levered(plus5, B["GR1"], 1.19)
print(f"   +5bps mine {c.metrics(r5, bil)['cagr']:.6f} | study {vm['plus5']['cagr']}")
for sp, key in ((0.005, "spread_050"), (0.025, "spread_250")):
    print(f"   spread {sp}: mine {c.metrics(levered(main, B['GR1'], 1.19, sp), bil)['cagr']:.6f} | study {vm[key]['cagr']}")
g1 = c.metrics(c.book_returns(main, B["GR1"]), bil)
g2 = c.metrics(c.book_returns(main, B["GR2"]), bil)
print(f"   exact vol-matched L = {g2['vol']/g1['vol']:.4f}; GR2 vol {g2['vol']:.6f}; levered vol at L=1.19 vs GR2: see above")
print(f"   first-day debt accrual days: mine uses 1 day for 2008-03-04")

print("")
dl = {"core5": 0.54, "btal_qqq": 0.36, "BIL": 0.10}
print("study defensive launch weights:", study["books"]["defensive launch"]["weights"])
rd = c.book_returns(main, dl)
rg = c.book_returns(main, B["GR1"])
print(f"corr GR1 vs defensive launch: mine {np.corrcoef(rg, rd)[0,1]:.6f} | study {study['corr_gr1_defensive']['all']}")
spx = c.load_tr("$SPXTR", main.index)
thr = spx.quantile(0.05)
mask = spx <= thr
print(f"   on S&P worst 5% days ({int(mask.sum())}): mine {np.corrcoef(rg[mask], rd[mask])[0,1]:.6f} | study {study['corr_gr1_defensive']['spx_worst5']}")
md = c.metrics(rd, bil)
qd = study["books"]["defensive launch"]["q"]
print(f"   defensive launch mine cagr {md['cagr']:.6f} xs {md['xs']:.6f} dd {md['dd']:.6f} | study {qd['cagr']} {qd['xs']} {qd['dd']}")
bl = {"taa3x": 1 / 6, "ndx_atr_cap": 1 / 12, "ndx_natr_cap": 1 / 12, "dv2_g": 1 / 12, "hpi_g": 1 / 12, "core5": 0.27, "btal_qqq": 0.18, "BIL": 0.05}
print("study blend 50/50 weights:", study["books"]["GR1 50 / defensive 50"]["weights"])
mb = c.metrics(c.book_returns(main, bl), bil)
qb = study["books"]["GR1 50 / defensive 50"]["q"]
print(f"   blend 50/50 mine cagr {mb['cagr']:.6f} xs {mb['xs']:.6f} dd {mb['dd']:.6f} | study {qb['cagr']} {qb['xs']} {qb['dd']}")
