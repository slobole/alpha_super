"""C1/C2/C4 tradability and A7 padded-bar counts for DV2 and QPI (research-only).

Inputs: full_runs/<fam>_base (production run, $100k) and full_runs/<fam>_hsu_30k (raw whole shares at $30k).

C1  Every engine fill (synthetic liquidations excluded): order weight w = |shares x fill price| / NAV_T (NAV at the
    decision close).  Notional at capital C = w x C.  ADV_T = median native Turnover (nominal $) over the 20 sessions
    ending at the decision date T (known at decision).  Ratio = notional / ADV_T at C in {30k, 1M, 10M}: median, p99,
    max, share > 1% / > 5%, full sample and last 3 years (fills >= 2023-09-25).
C2  $30k raw whole shares: for every entry of the hsu_30k run, slot value V = NAV_T / 10, raw price P = Unadjusted
    Close_T, shares = floor(V / P); weight error = (V - shares x P) / NAV_T.  Zero-share entries = P > V.
C4  Buying power: at each decision the backtest submits entries for the same open as its exits.  Required buying power
    beyond cash at submission = max(0, sum(entry values) - cash_T) / NAV_T.  End-of-day negative cash from daily.csv.
A7  Fills on padded bars (Volume == 0 on the fill date; ALLMARKETDAYS pads halts with the prior close) and entries
    decided on a padded bar (Volume_T == 0).

Usage: uv run python aud_tradability.py <dv2|qpi>
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import pandas as pd

import aud_common as ac
import aud_data

CAPITALS = {"30k": 30_000.0, "1m": 1_000_000.0, "10m": 10_000_000.0}
LAST3_START = pd.Timestamp("2023-09-25")


def _stats(x: pd.Series) -> dict:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median": float(x.median()), "p99": float(x.quantile(0.99)), "max": float(x.max()),
            "share_gt_1pct": float((x > 0.01).mean()), "share_gt_5pct": float((x > 0.05).mean())}


def main(fam: str) -> None:
    data = aud_data.load()
    pricing = data["pricing_df"]
    del data
    idx = pricing.index
    base = ac.OUT / "full_runs" / f"{fam}_base"
    tx = pd.read_csv(base / "transactions.csv.gz", parse_dates=["bar"])
    daily = pd.read_csv(base / "daily.csv.gz", index_col=0, parse_dates=True)
    nav = daily["total_value"].astype(float)
    tx = tx[tx["order_id"].astype(int) != -1].copy()
    pos = idx.get_indexer(tx["bar"])
    tx["decision_date"] = idx[pos - 1]
    tx["notional"] = (tx["amount"].astype(float) * tx["price"].astype(float)).abs()
    tx["nav_T"] = nav.reindex(tx["decision_date"]).to_numpy()
    tx["w"] = tx["notional"] / tx["nav_T"]
    turnover = pricing.xs("Turnover", axis=1, level=1)
    adv20 = turnover.rolling(20, min_periods=20).median()
    volume = pricing.xs("Volume", axis=1, level=1)
    tx["adv20"] = [adv20.at[d, a] if a in adv20.columns else np.nan for d, a in zip(tx["decision_date"], tx["asset"])]
    tx["vol_fill"] = [volume.at[d, a] if a in volume.columns else np.nan for d, a in zip(tx["bar"], tx["asset"])]
    tx["vol_T"] = [volume.at[d, a] if a in volume.columns else np.nan for d, a in zip(tx["decision_date"], tx["asset"])]
    out = {"fam": fam, "n_fills": int(len(tx)), "adv_definition": "median native Turnover, 20 sessions ending at T",
           "n_adv_missing_or_zero": int(((tx["adv20"] <= 0) | tx["adv20"].isna()).sum())}
    c1 = {}
    for label, cap in CAPITALS.items():
        ratio = tx["w"] * cap / tx["adv20"]
        c1[label] = {"all": _stats(ratio), "last_3y": _stats(ratio[tx["bar"] >= LAST3_START]),
                     "median_order_usd": float((tx["w"] * cap).median()),
                     "p99_order_usd": float((tx["w"] * cap).quantile(0.99))}
    out["C1_order_over_adv20"] = c1
    worst = tx.assign(ratio_10m=tx["w"] * 1e7 / tx["adv20"]).sort_values("ratio_10m", ascending=False).head(10)
    out["C1_worst_10m"] = worst[["bar", "asset", "w", "adv20", "ratio_10m"]].astype(str).to_dict("records")
    out["C1_adv20_usd_percentiles_last3y"] = {q: float(tx.loc[tx["bar"] >= LAST3_START, "adv20"].quantile(q))
                                              for q in (0.01, 0.05, 0.5)}

    # A7 padded bars
    out["A7_fills_on_zero_volume_bar"] = int((tx["vol_fill"] == 0).sum())
    out["A7_entries_decided_on_zero_volume_bar"] = int(((tx["vol_T"] == 0) & (tx["amount"] > 0)).sum())
    out["A7_examples"] = tx.loc[(tx["vol_fill"] == 0) | (tx["vol_T"] == 0), ["bar", "asset", "amount"]].astype(str).head(10).to_dict("records")

    # C4 buying power, from the recorded production decisions
    with (base / "decision_log.pkl").open("rb") as handle:
        log = pickle.load(handle)
    rows = []
    for r in log:
        entries = sum(o["amount"] for o in r["orders"] if not o["target"] and o["amount"] > 0)
        n_exit = sum(1 for o in r["orders"] if o["target"])
        n_entry = sum(1 for o in r["orders"] if not o["target"])
        rows.append({"date": r["decision_date"], "nav": r["prev_total_value"], "cash": r["cash"], "entries": entries,
                     "n_exit": n_exit, "n_entry": n_entry, "n_held": len(r["positions"])})
    bp = pd.DataFrame(rows)
    bp["need_over_cash"] = (bp["entries"] - bp["cash"]).clip(lower=0) / bp["nav"]
    active = bp[bp["n_entry"] > 0]
    bp["reuse"] = np.minimum(bp["n_exit"], np.maximum(0, bp["n_entry"] - (10 - bp["n_held"])))
    out["C4_buying_power"] = {
        "n_decisions_with_entries": int(len(active)),
        "share_needing_buying_power_beyond_cash": float((active["need_over_cash"] > 1e-9).mean()),
        "median_need_over_nav_when_needed": float(active.loc[active["need_over_cash"] > 1e-9, "need_over_cash"].median()),
        "p99_need_over_nav": float(active["need_over_cash"].quantile(0.99)),
        "max_need_over_nav": float(active["need_over_cash"].max()),
        "same_open_slot_reuse_decisions": int((bp["reuse"] > 0).sum()),
        "entries_total": int(bp["n_entry"].sum()),
        "entries_using_slot_freed_same_open": int(bp["reuse"].sum()),
    }
    out["C4_eod_cash"] = {"min_cash_over_nav": float((daily["cash"] / nav).min()),
                          "n_days_negative_cash": int((daily["cash"] < -1e-6).sum()),
                          "p1_cash_over_nav": float((daily["cash"] / nav).quantile(0.01))}

    # C2 whole shares at $30k (raw shares)
    hsu = ac.OUT / "full_runs" / f"{fam}_hsu_30k"
    if (hsu / "decision_log.pkl").exists():
        with (hsu / "decision_log.pkl").open("rb") as handle:
            log30 = pickle.load(handle)
        uc = pricing.xs("Unadjusted Close", axis=1, level=1)
        er = []
        for r in log30:
            for o in r["orders"]:
                if o["target"] or o["amount"] <= 0:
                    continue
                p = float(uc.at[r["decision_date"], o["asset"]])
                v = float(o["amount"])
                sh = np.floor(v / p)
                er.append({"date": r["decision_date"], "asset": o["asset"], "raw_price": p, "slot_value": v,
                           "shares": sh, "weight_error": (v - sh * p) / r["prev_total_value"]})
        er = pd.DataFrame(er)
        er.to_csv(ac.OUT / f"{fam}_whole_share_30k_entries.csv.gz", index=False)
        last3 = er[er["date"] >= LAST3_START]
        out["C2_whole_shares_30k"] = {
            "n_entries": int(len(er)), "median_weight_error": float(er["weight_error"].median()),
            "p99_weight_error": float(er["weight_error"].quantile(0.99)),
            "share_error_gt_2pct": float((er["weight_error"] > 0.02).mean()),
            "share_error_gt_5pct": float((er["weight_error"] > 0.05).mean()),
            "n_zero_share_entries": int((er["shares"] == 0).sum()),
            "last3y_n_entries": int(len(last3)), "last3y_share_error_gt_2pct": float((last3["weight_error"] > 0.02).mean()),
            "last3y_share_error_gt_5pct": float((last3["weight_error"] > 0.05).mean()),
            "last3y_n_zero_share": int((last3["shares"] == 0).sum()),
            "last3y_zero_share_names": sorted(last3.loc[last3["shares"] == 0, "asset"].unique().tolist()),
            "last3y_names_error_gt_5pct": sorted(last3.loc[last3["weight_error"] > 0.05, "asset"].unique().tolist()),
            "mean_cash_drag_weight": float(er["weight_error"].mean()),
        }
    # C2 at a FIXED $30k NAV (owner size today): every production entry, slot = $3,000, raw price = Unadjusted Close_T
    uc_all = pricing.xs("Unadjusted Close", axis=1, level=1)
    fx = []
    for r in log:
        for o in r["orders"]:
            if o["target"] or o["amount"] <= 0:
                continue
            p = float(uc_all.at[r["decision_date"], o["asset"]])
            sh = np.floor(3_000.0 / p)
            fx.append({"date": r["decision_date"], "asset": o["asset"], "raw_price": p, "shares": sh,
                       "weight_error": (3_000.0 - sh * p) / 30_000.0})
    fx = pd.DataFrame(fx)
    fx.to_csv(ac.OUT / f"{fam}_whole_share_fixed30k_entries.csv.gz", index=False)
    for label, sub in (("all", fx), ("last_3y", fx[fx["date"] >= LAST3_START])):
        out[f"C2_fixed_nav_30k_{label}"] = {
            "n_entries": int(len(sub)), "median_weight_error": float(sub["weight_error"].median()),
            "p99_weight_error": float(sub["weight_error"].quantile(0.99)),
            "max_weight_error": float(sub["weight_error"].max()),
            "share_error_gt_2pct": float((sub["weight_error"] > 0.02).mean()),
            "share_error_gt_5pct": float((sub["weight_error"] > 0.05).mean()),
            "n_zero_share": int((sub["shares"] == 0).sum()),
            "zero_share_names": sorted(sub.loc[sub["shares"] == 0, "asset"].unique().tolist())[:30],
            "names_error_gt_2pct": sorted(sub.loc[sub["weight_error"] > 0.02, "asset"].unique().tolist())[:30],
            "mean_weight_error": float(sub["weight_error"].mean()),
            "share_raw_price_gt_1000": float((sub["raw_price"] > 1000).mean())}
    path = ac.OUT / f"{fam}_tradability.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
