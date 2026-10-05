"""Amendment A1: the same families, gates and tie-break with LONG Sharpe as the objective."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shelf_rebuild_20260929"))
import lib, defensive_v2 as dv

TIE = ["pods", "not_live_share", "p_breach10", "trade_days_per_year", "sharpe"]

def boot_sharpe(R):
    idx = lib.boot_index(R.shape[0])
    out = np.empty((idx.shape[0], R.shape[1]))
    for k in range(idx.shape[0]):
        s = R[idx[k]]
        out[k] = s.mean(axis=0) / s.std(axis=0, ddof=1) * np.sqrt(252)
    return out

def select(t, names, boot, pool):
    top = t.loc[pool, "sharpe"].idxmax(); jt = names.index(top)
    share = pd.Series({b: float(np.mean(boot[:, jt] > boot[:, names.index(b)])) for b in pool}); share[top] = 0
    band = t.loc[share.index[share < 0.9]].copy(); band["beaten_by_top"] = share
    band = band.sort_values(TIE, ascending=[True, True, True, True, False])
    return top, band

out = {}
for line in ("low", "main"):
    t = pd.read_csv(dv.OUT / f"{line}_books.csv", index_col=0)
    R = pd.read_csv(dv.OUT / f"{line}_long_returns.csv.gz", index_col=0, parse_dates=True)
    names = list(R.columns)
    t["sharpe"] = (R.mean() / R.std() * np.sqrt(252)).reindex(t.index)
    boot = boot_sharpe(R.to_numpy())
    t["sharpe_beats_champion"] = pd.Series(np.mean(boot > boot[:, [names.index(dv.CHAMPION)]], axis=0), index=names)
    res = {"gate_passers": int(t.gates_pass.sum())}
    for label, pool in (("all", list(t.index[t.gates_pass])), ("no_eom", list(t.index[t.gates_pass & ~t.pods_list.str.contains("eom_flow")]))):
        if not pool:
            continue
        top, band = select(t, names, boot, pool)
        pick = band.index[0]
        holds = not (t.at[pick, "sharpe_beats_champion"] >= 0.80 and t.at[pick, "p_breach10"] <= t.at[dv.CHAMPION, "p_breach10"] + 0.02)
        res[label] = {"top": top, "pick": pick, "band": list(band.index), "beats_champion": t.at[pick, "sharpe_beats_champion"],
                      "recommendation": dv.CHAMPION if holds else pick}
    t.to_csv(dv.OUT / f"{line}_books_sharpe.csv", float_format="%.6g")
    out[line] = res
(dv.OUT / "selection_sharpe.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
print(json.dumps(out, indent=2, default=float))
