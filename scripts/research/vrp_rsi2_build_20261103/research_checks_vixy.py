from pathlib import Path
src = open(Path(__file__).with_name("research_checks.py"), encoding="utf8").read()
exec(src.split('W0 = weights(idx); r0, t0 = sim(W0)')[0])
W = weights(idx, long_col="VIXY"); W["VIXY"] = W["VIXY"] * .5
r, t = sim(W); r5, _ = sim(W, {"SVXY": 10, "VIXY": 10}); m = met(r.loc[S:EL]); print("half VIXY", {k: round(x, 3) for k, x in m.items()}, "COVID", round(tot(r, "2020-02-19", "2020-03-23"), 3), "turn", round(t.loc[S:EL].mean() * 252, 2))
ix_lab = L.index
def t11(q):
    z = q.reindex(ix_lab).fillna(rfL); z.loc[:S] = rfL.loc[:S]; return z
pd.DataFrame({"vixy_half": t11(r), "vixy_half_p5": t11(r5)}).to_parquet(OUT / "locked_series2.parquet")
