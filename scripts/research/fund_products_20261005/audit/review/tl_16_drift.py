"""Timing lens 16: Nasdaq look-through with the pod weights the book actually carries (annual reset, drift inside the year).
Checks the author's new nasdaq_lookthrough_drift (prior-close weights x same-day close exposure) against close-of-day weights."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import g_lib as g
lab = g.Lab(); data = lab.data; frame = lab.frame
EX, END, LONG = g.EXACT_START, g.END, g.LONG_START
tq = pickle.load(open(g.STUDY / "audit/review/timing_lens/tqqq.pkl", "rb"))["cs_close"]
exp = json.loads((g.OUT / "exposure.json").read_text())
idx = data["nav"]["taa3x"].loc[EX:END].index
def tqw(a):
    tx = data["tx"][a]; nav = data["nav"][a]
    sh = tx[tx.asset_str == "TQQQ"].groupby("date")["amount_float"].sum().reindex(nav.index).fillna(0).cumsum()
    return (sh * tq.reindex(nav.index) / nav).reindex(idx)
inv = lambda a: (data["path"][a]["portfolio_value_float"] / data["path"][a]["total_value_float"]).reindex(idx)
for n in ("GR1", "GR2", "GR3"):
    w = g.PRODUCTS[n]; taa = "taa3x" if "taa3x" in w else "taa3x_1n"
    wp = []; g.book_returns(frame, w, LONG, weight_path=wp); ix, cols, pw = wp[0]
    prior = pd.DataFrame(pw, index=ix, columns=cols)
    close_w = prior.shift(-1)                                   # pod weights at the close of t = prior-close weights of t+1 ...
    yr_end = pd.Series(ix.year, index=ix).shift(-1) != pd.Series(ix.year, index=ix)
    r = frame.loc[LONG:END, cols]; grown = prior * (1 + r); grown = grown.div(grown.sum(axis=1), axis=0)
    close_w = grown                                              # ... computed directly, so year-end closes carry the drifted weights (the reset applies from the next session)
    def look(wdf):
        return (wdf[taa] * 3 * tqw(taa) + wdf["ndx_atr_cap"] * inv("ndx_atr_cap") + wdf["ndx_natr_cap"] * inv("ndx_natr_cap")).dropna()
    a_, b_ = look(prior.reindex(idx)), look(close_w.reindex(idx))
    tgt = w[taa] * 3 * tqw(taa) + w["ndx_atr_cap"] * inv("ndx_atr_cap") + w["ndx_natr_cap"] * inv("ndx_natr_cap")
    mr = (close_w.reindex(idx)["dv2_g"] * inv("dv2_g_cash") + close_w.reindex(idx)["hpi_g"] * inv("hpi_g_cash"))
    eq = (b_ + mr).dropna()
    ref = exp["books"][n]
    print(f"{n}: Nasdaq look-through at target weights mean {tgt.mean():.3f} p90 {tgt.quantile(.9):.3f} max {tgt.max():.3f} ({tgt.idxmax().date()}) [published {ref['nasdaq_lookthrough']['max']:.3f}]")
    print(f"     prior-close drifted weights (author's new field): max {a_.max():.3f} on {a_.idxmax().date()} [exposure.json {ref['nasdaq_lookthrough_drift']['max']:.3f} {ref['nasdaq_lookthrough_drift']['peak_date']}]")
    print(f"     close-of-day drifted weights: mean {b_.mean():.3f} p90 {b_.quantile(.9):.3f} max {b_.max():.3f} on {b_.idxmax().date()}; TAA share then {close_w.loc[b_.idxmax(), taa]:.3f}; since 2015 max {b_.loc['2015':].max():.3f}")
    print(f"     equity exposure incl. MR with close-of-day drifted weights: mean {eq.mean():.3f} p90 {eq.quantile(.9):.3f} max {eq.max():.3f} on {eq.idxmax().date()} [published at target weights {ref['equity_exposure']['max']:.3f}] -> 10% gap at peak {-0.1 * eq.max():.3f} vs published {ref['gap_table']['10%']['peak']:.3f}")
