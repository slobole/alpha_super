"""A11 regimes and A13 walk-forward / bootstrap for B0 and the tilt candidate. Development sample."""
from common import *
import itertools
panel = dev(load())
B0 = run(panel, Params())
T1 = run(panel, Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M"))
bm = benchmarks(panel)
series = {"B0": B0, "B0+trend(sma10,k=.5,M)": T1, "60/40": bm["60/40"], "EW3": bm["EW3"]}
cash = B0["cash_daily"]
# --- regimes
sb = panel["r_stock"].rolling(120).corr(panel["r_bond"]).shift(1)
cpi = panel["cpi_yoy"]
infl_trend = cpi - cpi.rolling(252 * 10, min_periods=252 * 3).mean()
infl = pd.Series(np.where(infl_trend.shift(1) > 0, "inflation above 10y avg", "inflation below 10y avg"), index=panel.index)
corr = pd.Series(np.where(sb > 0.3, "stock-bond corr > 0.3", np.where(sb < -0.3, "stock-bond corr < -0.3", "corr between")), index=panel.index)
rows = {}
for nm, r in series.items():
    ex = (r["ret"] - cash).loc["1970":]
    for grp_name, grp in [("infl", infl), ("corr", corr)]:
        for g, v in ex.groupby(grp.loc["1970":]):
            rows.setdefault(nm, {})[f"{g} | ann excess"] = v.mean() * 252
            rows[nm][f"{g} | sharpe"] = v.mean() / v.std() * np.sqrt(252)
            rows[nm][f"{g} | years"] = len(v) / 252
df = pd.DataFrame(rows); save(df, "regimes", "a11"); print(df.round(3).to_string())
# worst 10 months of B0 and what the tilt did
m = pd.DataFrame({k: (1 + v["ret"]).resample("ME").prod() - 1 for k, v in series.items()}).loc["1970":]
w = m.nsmallest(12, "B0"); save(w, "worst_months_b0", "a11"); print(); print(w.round(3).to_string())
w2 = m.nsmallest(12, "60/40"); save(w2, "worst_months_6040", "a11"); print(); print(w2.round(3).to_string())
# --- walk-forward (A13 / G4): small grid, parameters re-chosen every 5 years on the expanding past
grid = list(itertools.product([42, 63, 126], [0.07, 0.08, 0.09], ["none", "sma10"]))
cells = {}
for vw, tv, tilt in grid:
    p = Params(vol_window=vw, port_vol_window=vw, target_vol=tv, tilt=tilt, tilt_strength=0.5, tilt_freq="M")
    cells[(vw, tv, tilt)] = run(panel, p)["ret"]
ex = pd.DataFrame({k: v - cash for k, v in cells.items()}).loc["1970":]
boundaries = pd.date_range("1980-01-01", "2022-12-31", freq="5YS")
wf = []
chosen = []
for i, b in enumerate(boundaries):
    e = boundaries[i + 1] if i + 1 < len(boundaries) else pd.Timestamp("2022-12-31")
    past = ex.loc[:b - pd.Timedelta(days=1)]
    sh = past.mean() / past.std() * np.sqrt(252)
    best = sh.idxmax()
    chosen.append({"from": str(b.date()), "to": str(e.date()), "chosen": str(best), "past_sharpe": sh.max()})
    wf.append(ex.loc[b:e - pd.Timedelta(days=1), best])
wf = pd.concat(wf)
def sh(x): return x.mean() / x.std() * np.sqrt(252)
out = {"walk_forward_sharpe_1980_2022": sh(wf), "B0_fixed_sharpe_1980_2022": sh(ex.loc["1980":, (63, 0.08, "none")]),
       "T1_fixed_sharpe_1980_2022": sh(ex.loc["1980":, (63, 0.08, "sma10")]),
       "best_in_hindsight_sharpe_1980_2022": float((ex.loc["1980":].mean() / ex.loc["1980":].std() * np.sqrt(252)).max()),
       "grid_sharpe_1980_2022": {str(k): float(v) for k, v in (ex.loc["1980":].mean() / ex.loc["1980":].std() * np.sqrt(252)).items()},
       "chosen": chosen}
save_json(out, "walk_forward", "a13"); print(); print(json.dumps({k: v for k, v in out.items() if k != "grid_sharpe_1980_2022"}, indent=1, default=str))
print(pd.Series(out["grid_sharpe_1980_2022"]).round(3).to_string())
# --- bootstrap bands for B0 and T1 (for the vault gate G6)
for nm, r in [("B0", B0), ("T1", T1)]:
    bs = block_bootstrap_diff(r["ret"].loc["1970":], bm["60/40"]["ret"].loc["1970":], cash.loc["1970":])
    save_json(bs, f"bootstrap_{nm}_vs_6040", "a13"); print(nm, {k: round(v, 3) for k, v in bs.items()})
pd.DataFrame({k: v["ret"] for k, v in series.items()}).to_parquet(OUT / "a13" / "series_daily.parquet")
