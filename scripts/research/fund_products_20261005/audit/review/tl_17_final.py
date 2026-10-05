"""Timing lens 17: final quantification - sessions above the published look-through peak; S9 combined BTAL orders vs the per-leg screen."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import g_lib as g
from g_lib import END, TBILL, ga, lib
lab = g.Lab(); data = lab.data; frame = lab.frame
EX, LONG = g.EXACT_START, g.LONG_START
tq = pickle.load(open(g.STUDY / "audit/review/timing_lens/tqqq.pkl", "rb"))["cs_close"]
exp = json.loads((g.OUT / "exposure.json").read_text()); cap = json.loads((g.OUT / "capacity.json").read_text())["books"]
idx = data["nav"]["taa3x"].loc[EX:END].index
def tqw(a):
    tx = data["tx"][a]; nav = data["nav"][a]
    sh = tx[tx.asset_str == "TQQQ"].groupby("date")["amount_float"].sum().reindex(nav.index).fillna(0).cumsum()
    return (sh * tq.reindex(nav.index) / nav).reindex(idx)
inv = lambda a: (data["path"][a]["portfolio_value_float"] / data["path"][a]["total_value_float"]).reindex(idx)
for n in ("GR1", "GR2", "GR3"):
    w = g.PRODUCTS[n]; taa = "taa3x" if "taa3x" in w else "taa3x_1n"
    wp = []; g.book_returns(frame, w, LONG, weight_path=wp); ix, cols, pw = wp[0]
    prior = pd.DataFrame(pw, index=ix, columns=cols); grown = prior * (1 + frame.loc[LONG:END, cols]); cw = grown.div(grown.sum(axis=1), axis=0).reindex(idx)
    look = cw[taa] * 3 * tqw(taa) + cw["ndx_atr_cap"] * inv("ndx_atr_cap") + cw["ndx_natr_cap"] * inv("ndx_natr_cap")
    pub = exp["books"][n]["nasdaq_lookthrough"]["max"]
    above = look[look > pub + 1e-9]
    print(f"{n}: published peak {pub:.3f}; sessions with the carried look-through above it: {len(above)} of {len(look)} ({above.index.min().date() if len(above) else None} .. {above.index.max().date() if len(above) else None}); max {look.max():.3f}; max TAA capital share {cw[taa].max():.3f}; p99 {look.quantile(.99):.3f}")
# S9: combined BTAL orders of taa3x_1n and btal_qqq
sys.path.insert(1, str(ga.MAIN_REPO / "scripts" / "research" / "growth_shelf_v2_20260926"))
import shelf_books as sb
start = sb.CAPACITY_START
w = g.INCUMBENT
pw = lib.common.book_return_ser(data["sleeve"].loc["2023-01-01":END, list(w)], w, "annual")[1]
rows = []
for a in ("taa3x_1n", "btal_qqq"):
    t = data["tx"][a]; t = t[(t.date >= start) & (t.date <= END) & (t.asset_str == "BTAL")]
    f = t.signed_notional_float / data["nav"][a].shift(1).reindex(t.date).to_numpy() * pw[a].reindex(t.date).to_numpy()
    rows.append(pd.DataFrame({"date": t.date.to_numpy(), "alias": a, "signed": f.to_numpy()}))
o = pd.concat(rows)
px = sb.load_price_timeseries("BTAL", start_date_str="2023-03-01", end_date_str=END.strftime("%Y-%m-%d")); px.index = pd.to_datetime(px.index).normalize()
adv = (px.Close * px.Volume).replace(0, np.nan).rolling(60, min_periods=20).median().shift(1)
piv = o.pivot_table(index="date", columns="alias", values="signed", aggfunc="sum").fillna(0)
same_dir = (np.sign(piv["taa3x_1n"]) == np.sign(piv["btal_qqq"])) & (piv["taa3x_1n"] != 0) & (piv["btal_qqq"] != 0)
comb = piv.abs().sum(axis=1) / adv.reindex(piv.index)
single = {a: (piv[a].abs() / adv.reindex(piv.index)) for a in piv}
print(f"S9 BTAL order dates {len(piv)}; both pods trade BTAL on the same date {int(((piv != 0).all(axis=1)).sum())}, same direction on {int(same_dir.sum())}")
print(f"   AUM at which the largest BTAL order = 5% of a median day: combined ${0.05 / comb.max() / 1e6:.2f}M (capacity.json book max ${cap[g.S9]['participation']['aum_max_at_5pct'] / 1e6:.2f}M); per leg taa3x_1n ${0.05 / single['taa3x_1n'].max() / 1e6:.2f}M, btal_qqq ${0.05 / single['btal_qqq'].max() / 1e6:.2f}M; published per-leg P99 ${cap[g.S9]['participation_by_leg']['aum_p99_at_5pct']['aum'] / 1e6:.2f}M, per-leg max ${cap[g.S9]['participation_by_leg']['aum_max_at_5pct']['aum'] / 1e6:.2f}M")
print(f"   combined BTAL orders: P90 = 5% at ${0.05 / comb.quantile(.9) / 1e6:.2f}M; GR1 published per-leg P99 ${cap['GR1']['participation_by_leg']['aum_p99_at_5pct']['aum'] / 1e6:.2f}M, max ${cap['GR1']['participation_by_leg']['aum_max_at_5pct']['aum'] / 1e6:.2f}M")
