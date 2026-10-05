"""inputs_map audit, step 4: are the cash-only capsule runs the same stock strategy as the BIL runs / the PM pods?

Stock fill events (asset, date, side) of: research cash run vs research BIL run ($100K each), research BIL run vs the
PM-book pod ($500K). Also the BIL-parking cost split from the BIL fills. No Norgate, no strategy imports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
WT = Path(__file__).resolve().parents[4]
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/inputs_map"
CAP = MAIN / "results/research/mr_capsule_build_20261004"
PM = MAIN / "results/research/portfolio/mr_capsule_bil/vanilla_backtest/2026-10-04_115751/pods"
PARK = ("BIL", "SPMO")
LO, HI = pd.Timestamp("2008-03-04"), pd.Timestamp("2026-08-19")


def events(tx: pd.DataFrame) -> set:
    s = tx[~tx["asset"].isin(PARK)]
    s = s[(s["bar"] >= LO) & (s["bar"] <= HI)]
    return set(zip(s["asset"].astype(str), s["bar"].dt.normalize(), np.sign(s["amount"]).astype(int)))


def jac(a: set, b: set) -> dict:
    return {"a": len(a), "b": len(b), "common": len(a & b), "a_only": len(a - b), "b_only": len(b - a), "jaccard": len(a & b) / max(len(a | b), 1)}


def main() -> None:
    rep = {}
    for pod, folder in (("dv2", "pod_mr_dv2_gated_bil"), ("hpi", "pod_mr_hpi_vote_gated_bil")):
        cash = pd.read_csv(CAP / f"{pod}_cash_transactions.csv", parse_dates=["bar"])
        bil = pd.read_csv(CAP / f"{pod}_bil_transactions.csv", parse_dates=["bar"])
        pm = pd.read_csv(PM / folder / "transactions.csv", parse_dates=["bar"])
        nav = pd.read_csv(CAP / f"{pod}_bil_nav.csv", index_col=0, parse_dates=True)["total_value"].astype(float)
        b = bil[(bil["asset"] == "BIL") & (bil["bar"] >= LO) & (bil["bar"] <= HI)]
        years = len(nav.loc[LO:HI]) / 252
        prior = nav.shift(1).reindex(b["bar"]).to_numpy()
        notional = (b["amount"].abs() * b["price"]).to_numpy()
        rep[pod] = {
            "cash_vs_bil_100k": jac(events(cash), events(bil)),
            "bil_100k_vs_pm_500k": jac(events(bil), events(pm)),
            "bil_orders_per_year": float(len(b) / years),
            "bil_one_way_turnover_x_nav_per_year": float((notional / prior).sum() / years),
            "bil_commission_pp_per_year": float((b["commission"].to_numpy() / prior).sum() / years * 100),
            "bil_slippage_2p5bps_pp_per_year": float((notional / prior).sum() / years * 0.00025 * 100),
        }
    (OUT / "04_parity.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
