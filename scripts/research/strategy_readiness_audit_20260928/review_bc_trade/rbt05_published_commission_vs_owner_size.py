"""Review BC-trade RBT05: what the PUBLISHED backtest charged in commission (last 3y, bp of NAV per year) versus
what IBKR charges the same order pattern at owner pod sizes (from RBT02).

published_bp = sum(commission_i / NAV_(t-1)) / years over the last-3y window, from the auditors' saved ledgers
(USD 100K start, compounding; adjusted-unit share counts as run). Zero-commission models (Compass, Tactical FI, CTC,
VIXM) are 0 by construction. gap = IBKR(C) - published: the part of owner-size cost the published record omits.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rbt02_uniform_capacity_and_friction as R  # noqa: E402

RES = R.RES


def pub(tx: pd.DataFrame, nav: pd.Series) -> float:
    tx = tx.copy()
    tx["bar"] = pd.to_datetime(tx["bar"])
    nav = nav.copy()
    nav.index = pd.to_datetime(nav.index)
    navp = nav.shift(1).bfill()
    w = tx[(tx["bar"] >= R.L3Y) & (tx["bar"] <= R.END)]
    return round(1e4 * float((w["commission"] / w["bar"].map(navp)).sum()) / R.YEARS, 1)


def main() -> None:
    led = {}
    for k, key in (("vox", "vox_iyr"), ("eom", "eom"), ("xlc", "disp_kie_ihi_xlc"), ("xlc200", "disp_xlc_sma200"),
                   ("kie200", "disp_kie_ihi_sma200")):
        led[key] = (pd.read_parquet(RES / f"tierb_etf_mr/fills_{k}.parquet"),
                    pd.read_parquet(RES / f"tierb_etf_mr/equity_{k}.parquet")["total_value"])
    led["etf_dv2"] = (pd.read_csv(RES / "tierbc_dv2etf_taa2x/etf/baseline_transactions.csv"),
                      pd.read_csv(RES / "tierbc_dv2etf_taa2x/etf/baseline_nav.csv", index_col=0)["total_value"])
    for k in ("qld_1n", "sso_1n", "btal_qld_1n", "lin_qqq"):
        led[k] = (pd.read_csv(RES / f"tierbc_dv2etf_taa2x/taa2x/transactions_{k}_100k.csv"),
                  pd.read_csv(RES / f"tierbc_dv2etf_taa2x/taa2x/nav_{k}_100k.csv", index_col=0)["total_value"])
    tb = pickle.load(open(RES / "tierc_hedge/_cache/trin_baseline.pkl", "rb"))
    led["trinity"] = (tb["tx"], tb["results"]["portfolio_value"])
    r2 = json.loads((R.OUT / "rbt02_uniform_capacity_and_friction.json").read_text(encoding="utf-8"))["pods"]
    out = {}
    for key in r2:
        if "fees" not in r2[key]:
            continue
        p = pub(*led[key]) if key in led else (0.0 if R.PODS[key][4] == (0.0, 0.0) else None)
        row = {"published_bp_yr": p}
        for C, f in r2[key]["fees"].items():
            row[C] = {"fixed_bp": f["ibkr_fixed_bp_yr"], "tiered_bp": f["ibkr_tiered_bp_yr"],
                      "gap_fixed_pp": None if p is None else round((f["ibkr_fixed_bp_yr"] - p) / 100, 2),
                      "gap_tiered_pp": None if p is None else round((f["ibkr_tiered_bp_yr"] - p) / 100, 2),
                      "orders_per_yr": f["orders_per_yr"], "zero_share_orders_3y": f["orders_zero_share"],
                      "max_share_price_pct_nav": f["max_share_price_pct_nav"]}
        out[key] = row
        print(key, p, {C: (row[C]["gap_fixed_pp"], row[C]["gap_tiered_pp"]) for C in r2[key]["fees"]}, flush=True)
    (R.OUT / "rbt05_published_commission_vs_owner_size.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
