from pathlib import Path
src = open(Path(__file__).with_name("research_checks.py"), encoding="utf8").read()
exec(src.split('W0 = weights(idx); r0, t0 = sim(W0)')[0])
eng = pd.read_parquet(OUT / "engine_ret.parquet")["engine"]; feat = pd.read_parquet(OUT / "engine_features.parquet")
W0 = weights(idx)
common = W0.index.intersection(feat.index); a_ = W0.loc[common]; b_ = feat.loc[common, ["target_SVXY", "target_VIXM"]].fillna(0.0); b_.columns = ["SVXY", "VIXM"]
d = (a_ - b_).abs().max(axis=1); print("target max diff all", round(d.max(), 6), "after 2011-12-01", round(d.loc["2011-12-01":].max(), 12), "days >1e-9 after Dec", int((d.loc["2011-12-01":] > 1e-9).sum()))
zero = pd.Series(0.0, index=idx)
r_rep, _ = sim(W0, rf_=zero)   # replica with 0% cash, 5 bp, free daily reset
r_rep_d, _ = sim(W0, rf_=zero) 
x = pd.concat([eng, r_rep], axis=1, keys=["eng", "rep"]).dropna().loc["2012-01-01":"2026-10-30"]
sh = lambda s: s.mean() / s.std() * 252 ** .5
print("corr", round(x.corr().iloc[0, 1], 4), "raw Sharpe engine", round(sh(x.eng), 3), "replica", round(sh(x.rep), 3), "CAGR", round((1 + x.eng).prod() ** (252 / len(x)) - 1, 4), round((1 + x.rep).prod() ** (252 / len(x)) - 1, 4))
w_cash = 1 - feat[["target_SVXY", "target_VIXM"]].fillna(0).sum(axis=1)
# engine plus fair cash on the decided cash share (approximation for the headline only)
e2 = (eng + rf.reindex(eng.index).fillna(0) * w_cash.shift(1).reindex(eng.index).fillna(1)).loc[S:EL]
print("engine + cash on target cash share, 2011-10-10..2026-08-19:", {k: round(v, 3) for k, v in met(e2).items()}, "| replica locked", {k: round(v, 3) for k, v in met(sim(W0)[0].loc[S:EL]).items()})
print("engine 2018 Feb window", round((1 + eng.loc["2018-01-26":"2018-02-12"]).prod() - 1, 4), "COVID", round((1 + eng.loc["2020-02-19":"2020-03-23"]).prod() - 1, 4), "Aug 2024", round((1 + eng.loc["2024-07-31":"2024-08-08"]).prod() - 1, 4))
