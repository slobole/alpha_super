"""C1 order size vs ADV (native Norgate Turnover, 20-session median before the fill) at USD 30K / 1M / 10M, full
history and last 3 years; house auction guardrail share (0.05% / 0.10% of ADV, alpha/engine/capacity_analysis.py:91-92);
C2 whole shares at USD 30K (and 15K) from recorded rebalance targets and nominal decision closes; current ADV table.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

import tc_common as c

sys.path.insert(0, str(c.HERE.parent / "common"))
import tradability as t  # noqa: E402

LOADERS = {"vixm": c.load_vixm, "ctc": c.load_ctc_workaround, "trin": c.load_trin}
ASSETS = {"vixm": c.vixm.TRADEABLE_ASSET_TUPLE, "ctc": c.ctc.TRADEABLE_ASSET_TUPLE, "trin": c.trin.TRADEABLE_ASSET_TUPLE}


class _S:  # minimal adapter for tradability.participation_table_df
    def __init__(self, tx, tv):
        self._tx = tx
        self.total_value_series = tv

    def get_transactions(self):
        return self._tx


def run(name: str) -> dict:
    base = pickle.load(open(c.CACHE / f"{name}_baseline.pkl", "rb"))
    df = LOADERS[name]()
    tv = base["results"]["total_value"].astype(float)
    tv.index = pd.to_datetime(tv.index)
    turnover = {a: df[(a, "Turnover")].astype(float) for a in ASSETS[name]}
    part = t.participation_table_df(_S(base["tx"], tv), turnover)
    summ = t.summarize_participation_df(part)
    part.to_csv(c.OUT / name / "c1_participation_orders.csv.gz", index=False)
    summ.to_csv(c.OUT / name / "c1_participation_summary.csv", index=False)
    out = {"participation": summ.to_dict(orient="records")}
    # auction guardrail share (last 3y)
    recent = part[part["bar"] >= pd.Timestamp("2023-09-25")]
    guard = {}
    for cap in (30_000, 1_000_000, 10_000_000):
        col = f"part_{cap}"
        guard[str(cap)] = {"share_orders_over_0.05pct_adv": float((recent[col] > 0.0005).mean()),
                           "share_orders_over_0.10pct_adv": float((recent[col] > 0.0010).mean()),
                           "worst_asset_p99_last3y": str(recent.groupby("asset")[col].quantile(0.99).idxmax())}
    out["auction_guardrail_last3y"] = guard
    # per-asset last-3y p99 at 1M/10M
    per = recent.groupby("asset").agg(n=("bar", "size"),
                                      p99_1m=("part_1000000", lambda s: float(s.quantile(0.99) * 100)),
                                      p99_10m=("part_10000000", lambda s: float(s.quantile(0.99) * 100)),
                                      med_10m=("part_10000000", lambda s: float(s.median() * 100)))
    out["per_asset_last3y_pct_adv"] = per.round(4).to_dict(orient="index")
    # ADV table today
    adv = {}
    for a in ASSETS[name]:
        s = turnover[a]
        adv[a] = {"adv_median_last1y_usd_m": float(s[s.index >= "2025-09-25"].median() / 1e6),
                  "adv_median_last3y_usd_m": float(s[s.index >= "2023-09-25"].median() / 1e6),
                  "adv_median_full_usd_m": float(s.median() / 1e6),
                  "unadj_close_last": float(df[(a, "Unadjusted Close")].iloc[-1])}
    out["adv_today"] = adv
    # C2 whole shares from rebalance targets
    if name == "trin":
        tw = base["daily_target"]
        tx = base["tx"].copy()
        tx["bar"] = pd.to_datetime(tx["bar"])
        idx = df.index
        dec_days = sorted({idx[idx.get_loc(d) - 1] for d in set(tx["bar"])})
        tw = tw.loc[tw.index.intersection(dec_days)]
        cols = list(c.trin.TRADEABLE_ASSET_TUPLE)
    elif name == "ctc":
        tw = base["rebal"]
        cols = list(c.ctc.TRADEABLE_ASSET_TUPLE)
    else:
        tw = base["rebal"]
        cols = list(c.vixm.TRADEABLE_ASSET_TUPLE)
    tw = tw[tw.index >= pd.Timestamp("2016-01-01")]
    rows = []
    for d, w in tw.iterrows():
        for a in cols:
            wt = float(w[a])
            if abs(wt) < 1e-12:
                continue
            p = float(df.loc[d, (a, "Unadjusted Close")])
            for cap in (15_000.0, 30_000.0, 100_000.0):
                sh = np.floor(abs(wt) * cap / p)
                err = abs(wt) - sh * p / cap
                rows.append({"d": d, "asset": a, "cap": cap, "w": wt, "p": p, "shares": sh, "err": err,
                             "zero": sh == 0})
    ws = pd.DataFrame(rows)
    ws.to_csv(c.OUT / name / "c2_whole_share_rows.csv.gz", index=False)
    wsum = {}
    for cap, g in ws.groupby("cap"):
        per_dec = g.groupby("d")["err"].sum()
        wsum[str(int(cap))] = {"max_per_name_err_pct_nav": float(g["err"].max() * 100),
                               "p99_per_name_err_pct_nav": float(g["err"].quantile(0.99) * 100),
                               "mean_rounding_cash_pct_nav": float(per_dec.mean() * 100),
                               "max_rounding_cash_pct_nav": float(per_dec.max() * 100),
                               "zero_share_legs": int(g["zero"].sum()), "legs": int(len(g)),
                               "zero_share_assets": sorted(set(g.loc[g["zero"], "asset"])),
                               "last3y_max_per_name_err_pct_nav": float(g.loc[g["d"] >= "2023-09-25", "err"].max() * 100)
                               if (g["d"] >= "2023-09-25").any() else None}
    out["whole_shares_since_2016"] = wsum
    return out


if __name__ == "__main__":
    import json
    for n in sys.argv[1:] or ["vixm", "ctc", "trin"]:
        o = run(n)
        c.dump(o, f"{n}/c_tradability.json")
        print(n, json.dumps({k: v for k, v in o.items() if k != "per_asset_last3y_pct_adv"}, indent=1, default=str)[:5000], flush=True)
