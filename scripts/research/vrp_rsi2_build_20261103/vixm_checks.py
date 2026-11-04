"""The VIXM leg's own risks, after the house audit of the VIXM backwardation pod (TIERC_HEDGE_FINDINGS.md C-VIXM-02):
extra cost on VIXM legs only, VIXM filled at the close instead of the thin opening print, and how much of the leg's
P&L sits on the first day of an episode.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe scripts/research/vrp_rsi2_build_20261103/vixm_checks.py
"""
from pathlib import Path
src = open(Path(__file__).with_name("research_checks.py"), encoding="utf8").read()
exec(src.split('W0 = weights(idx); r0, t0 = sim(W0)')[0])
W0 = weights(idx); r0, _ = sim(W0)
rows = {"locked, 5 bp": met(r0.loc[S:EL])}
for bp in (20, 45):
    r, _ = sim(W0, {"SVXY": 5, "VIXM": bp}); rows[f"VIXM legs {bp} bp, SVXY 5 bp"] = met(r.loc[S:EL])
O_close = O.copy(); O_close["VIXM"] = C["VIXM"]
r, _ = sim(W0, O_=O_close); rows["VIXM filled at the T+1 close"] = met(r.loc[S:EL]); r_close = r
print(pd.DataFrame(rows).T.round(3).to_string())
print("COVID window: open fill", round(tot(r0, "2020-02-19", "2020-03-23"), 3), "close fill", round(tot(r_close, "2020-02-19", "2020-03-23"), 3))
held = W0["VIXM"].shift(1).fillna(0) > 0
start = held & ~held.shift(1, fill_value=False)
leg = W0["VIXM"].shift(1).fillna(0) * (C["VIXM"].pct_change().fillna(0))
first_oc = (W0["VIXM"].shift(1).fillna(0) * (C["VIXM"] / O["VIXM"] - 1)).where(start, 0.0)
print("VIXM episodes", int(start.sum()), "days held", int(held.sum()), "| leg P&L sum", round(leg.sum(), 3), "first-day open-to-close share", round(first_oc.sum() / leg.sum(), 3))
oc_eq = ((O["VIXM"] == d["VIXM"]["High"].reindex(idx)) | (O["VIXM"] == d["VIXM"]["Low"].reindex(idx))).loc[S:EL].mean()
print("share of sessions where VIXM open equals the day's high or low", round(oc_eq, 3))
