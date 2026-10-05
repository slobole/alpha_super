"""First diverging trades: run_engine pod ($100K) vs the PM pod ($500K). Read-only."""
from pathlib import Path
import pandas as pd
WT = Path(__file__).resolve().parents[4]
MAIN = Path(r"C:/Users/User/Documents/workspace/alpha_super")
PM = MAIN / "results/research/portfolio/mr_capsule_bil/vanilla_backtest/2026-10-04_115751/pods"
for pod, pm_dir in (("dv2", "pod_mr_dv2_gated_bil"), ("hpi", "pod_mr_hpi_vote_gated_bil")):
    a = pd.read_csv(WT / f"results/research/mr_capsule_build_20261004/{pod}_bil_transactions.csv", parse_dates=["bar"])
    b = pd.read_csv(PM / pm_dir / "transactions.csv", parse_dates=["bar"])
    print("=====", pod, len(a), len(b))
    lo, hi = "2006-06-08", "2006-06-22"
    for name, t in (("100K", a), ("500K", b)):
        s = t[(t["bar"] >= lo) & (t["bar"] <= hi)]
        print(name)
        print(s[["bar", "asset", "amount", "price", "total_value", "commission"]].to_string(index=False))
    # scale check before divergence: amounts ratio
    m = a.merge(b, on=["bar", "asset"], suffixes=("_a", "_b"))
    m = m[m["bar"] < "2006-06-01"]
    print("pre-2006-06 common trades", len(m), "amount ratio 500K/100K min/median/max", (m["amount_b"] / m["amount_a"]).describe()[["min", "50%", "max"]].to_dict())
