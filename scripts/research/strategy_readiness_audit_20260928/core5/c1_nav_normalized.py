"""C1 with constant NAV: order notional / NAV at the fill date x target NAV (30K, 1M, 10M), vs 20-session median
dollar volume through the decision session. Uses the 100K engine run's orders (share rounding at 100K)."""
import pickle, json
import numpy as np, pandas as pd
import core5_common as c
df = c.load_pricing()
with (c.CACHE / "baseline_strategy.pkl").open("rb") as fh:
    base = pickle.load(fh)
tx = base["tx"].copy(); tx["bar"] = pd.to_datetime(tx["bar"])
tv = base["results"]["total_value"].astype(float)
prev_nav = tv.shift(1).reindex(tx["bar"]).to_numpy()
tx["frac_nav"] = (tx["amount"].astype(float) * tx["price"].astype(float)).abs().to_numpy() / prev_nav
dv = {s: df[(s, "Turnover")].astype(float).rolling(20, min_periods=15).median() for s in c.TRADEABLES}
pos = df.index.get_indexer(tx["bar"])
tx["decision"] = df.index[pos - 1]
tx["adv"] = [float(dv[a].loc[d]) for a, d in zip(tx["asset"], tx["decision"])]
rows = []
for nav in (30_000.0, 1_000_000.0, 10_000_000.0):
    tx["pct_adv"] = 100 * tx["frac_nav"] * nav / tx["adv"]
    for window, sub in (("full", tx), ("last3y", tx[tx["bar"] >= "2023-09-25"])):
        for asset in list(c.TRADEABLES) + ["ALL"]:
            x = sub if asset == "ALL" else sub[sub["asset"] == asset]
            rows.append({"nav": int(nav), "window": window, "asset": asset, "n": int(len(x)),
                         "median_pct_adv": float(x["pct_adv"].median()), "p99_pct_adv": float(x["pct_adv"].quantile(.99)),
                         "max_pct_adv": float(x["pct_adv"].max())})
r = pd.DataFrame(rows); r.to_csv(c.OUT / "c1_order_vs_adv_constant_nav.csv", index=False)
print(r[r.asset.isin(["ALL", "DBC", "UUP", "BIL", "IEF"])].round(4).to_string())
# largest single-order fraction of NAV (sleeve moves are 20%)
print("max order frac NAV", float(tx["frac_nav"].max()), "p99", float(tx["frac_nav"].quantile(.99)))
# NAV at which DBC/UUP p99 order hits 1%/5% of ADV, last 3y
l3 = tx[tx["bar"] >= "2023-09-25"]
for a in ("DBC", "UUP"):
    x = l3[l3.asset == a]; q = float((x["frac_nav"] / x["adv"]).quantile(.99))
    print(a, "NAV where p99 order = 5% ADV:", round(0.05 / q), " = 1% ADV:", round(0.01 / q))
