"""Feature parity: replica QPI / RSI2 vs alpha.indicators and talib (research-only)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qpi_lib as q
from alpha.indicators import qp_indicator
p = q.rp.Panel("sp500")
F = q.Features(p)
C = np.asarray(p.C)
for s in ["AAPL", "MSFT", "XOM", "JPM", "NVDA"]:
    if s not in p.symbols:
        continue
    i = p.symbols.index(s)
    ref = qp_indicator(pd.Series(C[:, i], index=p.dates)).to_numpy()
    m = np.isfinite(F.qpi[:, i])
    print(s, int(m.sum()), "max |QPI diff|", float(np.nanmax(np.abs(F.qpi[m, i] - ref[m]))), "ref finite where r3<0 & member but ours NaN:",
          int((np.isfinite(ref) & ~m & np.asarray(p.member)[:, i] & (F.r3[:, i] < 0)).sum()))
r = q.rp.rsi2(p)
m = np.isfinite(r) & np.isfinite(F.rsi2)
print("RSI2 max diff", float(np.max(np.abs(r[m] - F.rsi2[m]))), "talib finite but ours NaN", int((np.isfinite(r) & ~np.isfinite(F.rsi2)).sum()))
print("entry candidates per day 2016+", float(F.entry[p.dates >= "2016"].sum(1).mean()))
