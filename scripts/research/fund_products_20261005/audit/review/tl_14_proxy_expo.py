"""Timing lens 14: TQQQ weight in the proxy era (2008-03..2012-09), measured on the proxy runs' own TQQQ fill dates (shares x fill price / NAV),
against the same measure in the real-data era. Tells whether leaving the proxy era out of the look-through understates exposure."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import g_lib as g
lib = g.lib
EX, END, LONG = g.EXACT_START, g.END, g.LONG_START
for a in ("taa3x", "taa3x_1n"):
    ptx = lib.read_tx(lib.PROXY / "splice_scaled", a); ppath = lib.read_path(lib.PROXY / "splice_scaled", a)
    t = ptx[ptx.asset_str == "TQQQ"].copy()
    sh = t.groupby("date").amount_float.sum().cumsum()                    # shares after each TQQQ fill date
    px = t.groupby("date").fill_price_float.last()
    w = (sh * px / ppath.total_value_float.reindex(sh.index)).dropna()
    inv = (ppath.portfolio_value_float / ppath.total_value_float)
    pro, real = w.loc[LONG:EX - pd.Timedelta(days=1)], w.loc[EX:END]
    # month-start weight for every month (carry the last weight when no TQQQ fill): approximate with fill dates only
    print(f"{a}: proxy run {ppath.index[0].date()}..{ppath.index[-1].date()}; TQQQ fill dates proxy era {len(pro)}, real era {len(real)}")
    print(f"   TQQQ weight on TQQQ fill dates: proxy era mean {pro.mean():.3f} p90 {pro.quantile(.9):.3f} max {pro.max():.3f} | real era (same measure) mean {real.mean():.3f} p90 {real.quantile(.9):.3f} max {real.max():.3f}")
    print(f"   share of proxy-era fill dates with weight 0: {(pro < 0.01).mean():.2f}; real era {(real < 0.01).mean():.2f}; invested weight proxy era mean {inv.loc[LONG:EX].mean():.3f}")
    yr = pro.groupby(pro.index.year).agg(["mean", "max", "count"]).round(3)
    print(yr.to_string())
