"""A4b: robustness of the trend-tilt specification (is the SMA10/k=0.5 result a plateau or a peak?)."""
from common import *
import itertools
panel = dev(load())
B0 = run(panel, Params())
rows = []
# SMA family: engine uses 210 sessions (~10 months). Emulate other lengths via a custom column trick is not available,
# so extend: monkeypatch engine._sma10_trend length through a module variable.
import engine as E
def sma_trend_factory(months):
    def f(R):
        px = (1 + R.fillna(0)).cumprod(); sma = px.rolling(21 * months).mean()
        return np.sign(px - sma).where(sma.notna())
    return f
for months, k in itertools.product([6, 8, 10, 12], [0.25, 0.5, 0.75, 1.0]):
    E._sma10_trend = sma_trend_factory(months)
    p = Params(tilt="sma10", tilt_strength=k, tilt_freq="M")
    res = E.run(panel, p)
    row = {"tilt": f"sma{months}m", "k": k, "freq": "M"} | summarize(res)
    row |= {f"adopt_{kk}": v for kk, v in adoption_check(res, B0, panel).items() if kk in ("adopt", "boot_diff_median", "boot_diff_p05")}
    rows.append(row); print(row["tilt"], k, round(row["sharpe"], 3), round(row["max_dd"], 3), flush=True)
E._sma10_trend = sma_trend_factory(10)
for k in [0.25, 0.5, 0.75, 1.0]:
    for freq in ["M", "W"]:
        p = Params(tilt="ensemble", tilt_strength=k, tilt_freq=freq, rebalance=freq if freq == "W" else "M")
        res = E.run(panel, p)
        row = {"tilt": "ensemble", "k": k, "freq": freq} | summarize(res)
        row |= {f"adopt_{kk}": v for kk, v in adoption_check(res, B0, panel).items() if kk in ("adopt", "boot_diff_median", "boot_diff_p05")}
        rows.append(row); print("ens", k, freq, round(row["sharpe"], 3), round(row["max_dd"], 3), flush=True)
# lag-2 check for the two leading candidates
for nm, p in [("sma10 k0.5 M lag2", Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M", lag=2)),
              ("ensemble k0.5 M lag2", Params(tilt="ensemble", tilt_strength=0.5, tilt_freq="M", lag=2)),
              ("sma10 k0.5 M cost10", Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M", cost_bps=10)),
              ("sma10 k0.5 M cash0", Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M", cash="zero"))]:
    res = E.run(panel, p); rows.append({"tilt": nm, "k": 0.5, "freq": "M"} | summarize(res))
df = pd.DataFrame(rows); save(df, "tilt_grid", "a4")
cols = ["tilt", "k", "freq", "cagr", "vol", "sharpe", "max_dd", "worst_12m", "turnover_per_year", "avg_exposure", "sh_1970s", "sh_1980s", "sh_1990s", "sh_2000s", "sh_2010s", "sh_2020-22", "sh_ETF era 2005-22", "adopt_boot_diff_median", "adopt_boot_diff_p05", "adopt_adopt"]
print(df[cols].round(3).to_string())
