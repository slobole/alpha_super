from pathlib import Path
src = open(Path(__file__).with_name("research_checks.py"), encoding="utf8").read()
exec(src.split('W0 = weights(idx); r0, t0 = sim(W0)')[0])
W0 = weights(idx); ix_lab = L.index
def t11(q):
    z = q.reindex(ix_lab).fillna(rfL); z.loc[:S] = rfL.loc[:S]; return z
out = {}
for c in (5, 10, 15, 20, 25, 30, 35, 40, 45):
    r, _ = sim(W0, {"SVXY": c, "VIXM": c}); out[f"c{c}"] = t11(r); print(c, round(met(r.loc[S:EL])["xs"], 3), round(met(r.loc[S:EL])["cagr"], 3))
# VX splice with 10 bp fill kept, ETF part at 5 and 10
pd.DataFrame(out).to_parquet(OUT / "locked_costs.parquet")
