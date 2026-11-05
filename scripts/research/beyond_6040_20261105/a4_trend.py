"""A4: per-asset trend tilt as insurance. Development sample 1970-2022."""
from common import *
panel = dev(load())
B0 = run(panel, Params())
arms = {
    "B0": Params(),
    "tilt ensemble k=0.5 daily": Params(tilt="ensemble", tilt_strength=0.5, tilt_freq="D"),
    "tilt ensemble k=0.5 monthly": Params(tilt="ensemble", tilt_strength=0.5, tilt_freq="M"),
    "tilt ensemble k=0.25 daily": Params(tilt="ensemble", tilt_strength=0.25, tilt_freq="D"),
    "tilt ensemble k=0.5 + band5": Params(tilt="ensemble", tilt_strength=0.5, tilt_freq="D", band=0.05),
    "tilt sma10 k=0.5 monthly": Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M"),
    "tilt sma10 k=0.25 monthly": Params(tilt="sma10", tilt_strength=0.25, tilt_freq="M"),
    # scaled: the author lifts the tilted book back to 8% with a slow (10y) scalar; approximate with higher target
    "tilt ensemble k=0.5 daily, target 10%": Params(tilt="ensemble", tilt_strength=0.5, tilt_freq="D", target_vol=0.10, trigger_buffer=0.005),
}
results = {k: run(panel, p) for k, p in arms.items()}
rows = {k: summarize(r) for k, r in results.items()}
for nm, r in benchmarks(panel).items():
    rows[nm] = summarize(r)
df = pd.DataFrame(rows).T
save(df, "trend_table", "a4")
print(df[[c for c in df.columns if not c.startswith(("sh_", "dd_"))]].round(3).to_string())
print(); print(df[[c for c in df.columns if c.startswith("sh_")]].round(2).to_string())
print(); print(df[[c for c in df.columns if c.startswith("dd_")]].round(3).to_string())
# adoption rule
ad = {k: adoption_check(r, B0, panel) for k, r in results.items() if k != "B0"}
adf = pd.DataFrame(ad).T
save(adf, "trend_adoption", "a4")
print(); print(adf.round(3).to_string())
# regime view: stock-bond 120d correlation sign x trailing 12m stock return sign
sb = panel["r_stock"].rolling(120).corr(panel["r_bond"]).shift(1)
st = panel["r_stock"].rolling(252).sum().shift(1)
reg = pd.Series(np.where(sb > 0, np.where(st < 0, "corr>0 & stocks down", "corr>0 & stocks up"),
                         np.where(st < 0, "corr<0 & stocks down", "corr<0 & stocks up")), index=panel.index)
out = {}
for k in ["B0", "tilt ensemble k=0.5 daily", "tilt ensemble k=0.25 daily", "tilt sma10 k=0.5 monthly"]:
    r = results[k]["ret"].loc["1970":]; c = results[k]["cash_daily"].loc["1970":]
    ex = r - c
    g = ex.groupby(reg.loc["1970":])
    out[k] = {f"{n} ann excess ret": v.mean() * 252 for n, v in g} | {f"{n} days": len(v) for n, v in g}
rdf = pd.DataFrame(out)
save(rdf, "trend_by_regime", "a4")
print(); print(rdf.round(4).to_string())
# save daily series for the report
pd.DataFrame({k: r["ret"] for k, r in results.items()}).to_parquet(OUT / "a4" / "trend_daily.parquet")
