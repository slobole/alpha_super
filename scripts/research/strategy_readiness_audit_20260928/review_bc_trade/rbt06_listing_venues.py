"""Review BC-trade RBT06: listing venue of every symbol each Tier B/C pod trades (Norgate metadata), and the share of
last-3y order notional that goes to a Cboe BZX-listed or Nasdaq-listed ETF (opening/closing-auction route to verify).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rbt02_uniform_capacity_and_friction as R  # noqa: E402
from data.norgate_loader import norgatedata  # noqa: E402

def main():
    out = {"venue": {}, "pods": {}}
    for key, (path, fmt, *_rest) in R.PODS.items():
        if key == "ndx_vxn":
            continue
        df = R.load(path, fmt)
        w = df[df["bar"] >= R.L3Y]
        syms = sorted(df["asset"].unique())
        for s in syms:
            if s not in out["venue"]:
                try:
                    out["venue"][s] = {"exchange": str(norgatedata.exchange_name(s)), "name": str(norgatedata.security_name(s))}
                except Exception as e:  # noqa: BLE001
                    out["venue"][s] = {"exchange": f"n/a {type(e).__name__}", "name": ""}
        ex = w["asset"].map(lambda s: out["venue"][s]["exchange"])
        tot = float(w["order_frac_of_nav"].sum()) or 1.0
        out["pods"][key] = {
            "symbols_by_exchange": {e: sorted(set(df.loc[df["asset"].map(lambda s: out["venue"][s]["exchange"]) == e, "asset"])) for e in sorted({out["venue"][s]["exchange"] for s in syms})},
            "last3y_notional_share_by_exchange": {e: round(float(w.loc[ex == e, "order_frac_of_nav"].sum()) / tot, 3) for e in sorted(set(ex))},
        }
        print(key, out["pods"][key], flush=True)
    (R.OUT / "rbt06_listing_venues.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
