"""Stocks whose CAPITALSPECIAL price exceeds a pod slot (zero-share cancellations at small capital). Read-only."""
from pathlib import Path
import pandas as pd
WT = Path(__file__).resolve().parents[4]
MAIN = Path(r"C:/Users/User/Documents/workspace/alpha_super")
PM = MAIN / "results/research/portfolio/mr_capsule_bil/vanilla_backtest/2026-10-04_115751/pods"
for pod, pm_dir in (("dv2", "pod_mr_dv2_gated_bil"), ("hpi", "pod_mr_hpi_vote_gated_bil")):
    a = pd.read_csv(WT / f"results/research/mr_capsule_build_20261004/{pod}_bil_transactions.csv", parse_dates=["bar"])
    b = pd.read_csv(PM / pm_dir / "transactions.csv", parse_dates=["bar"])
    nav_a = pd.read_csv(WT / f"results/research/mr_capsule_build_20261004/{pod}_bil_nav.csv", index_col=0, parse_dates=True)["total_value"]
    for name, t, scale in (("100K", a, 1.0), ("500K", b, 5.0)):
        buys = t[(t["amount"] > 0) & (~t["asset"].isin(["BIL", "SPMO"]))].copy()
        slot = (nav_a.shift(1).reindex(buys["bar"]).to_numpy() * scale / 10.0)
        buys["fill_vs_slot"] = buys["amount"] * buys["price"] / slot
        hi = buys[buys["price"] > 2000]
        under = buys[buys["fill_vs_slot"] < 0.8]
        print(pod, name, "stock buys", len(buys), "price>2000:", len(hi), sorted(hi["asset"].unique())[:10], "| fills <80% of slot:", len(under), sorted(under["asset"].unique())[:12],
              "| median fill/slot", round(float(buys["fill_vs_slot"].median()), 4), "min", round(float(buys["fill_vs_slot"].min()), 4))
    ea = set(zip(a["asset"], a["bar"], (a["amount"] > 0)))
    eb = set(zip(b["asset"], b["bar"], (b["amount"] > 0)))
    only_b = sorted({x[0] for x in eb - ea if x[0] not in ("BIL",)})
    only_a = sorted({x[0] for x in ea - eb if x[0] not in ("BIL",)})
    print(pod, "assets with events only in 500K:", only_b)
    print(pod, "assets with events only in 100K:", only_a)
    yrs = sorted({x[1].year for x in (ea ^ eb) if x[0] != "BIL"})
    print(pod, "years with differing stock events:", yrs)
