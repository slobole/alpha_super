"""Standalone capacity of each DV2 variant (orders 2023-08-21 -> 2026-08-19; house auction limits; growth-shelf model).

Recommended AUM = largest grid size where every route gate passes and modelled impact cost <= 25% of the
variant's 2012-2026 excess return over 2% (same rule as dv2_liquidity_eval.py).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase5_book as pb  # noqa: E402
from phase5_book import SRC, CAP_START, AUM_GRID, gc, gd, load_price_timeseries  # noqa: E402


def main():
    rows = []
    liq_cache: dict = {}
    for a in pb.VARIANTS:
        path = pd.read_csv(SRC / f"{a}__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"]
        t = pd.read_csv(SRC / f"{a}__transactions.csv.gz", parse_dates=["date"])
        sub = path.loc["2012-10-02":"2026-08-19"]
        cagr = (sub.iloc[-1] / sub.iloc[0]) ** (365.25 / (sub.index[-1] - sub.index[0]).days) - 1
        w = t[(t["date"] >= CAP_START) & (t["date"] <= gd.END_TS)]
        prior = path.shift(1)
        o = pd.DataFrame({"date": w["date"], "asset_str": w["asset_str"],
                          "book_fraction_float": w["signed_notional_float"].abs().values / prior.reindex(w["date"]).values})
        o = o.groupby(["date", "asset_str"], as_index=False)["book_fraction_float"].sum()
        o["is_urgent"] = True
        o["is_etf"] = a.startswith("etf")
        o["is_nasdaq"] = False
        for tk in o["asset_str"].unique():
            if tk in liq_cache:
                continue
            pr = load_price_timeseries(tk, start_date_str="2023-03-01", end_date_str=gd.END_TS.strftime("%Y-%m-%d"))
            pr.index = pd.to_datetime(pr.index).normalize()
            dol = (pr["Close"] * pr["Volume"]).replace(0.0, np.nan)
            # *** CRITICAL*** shift(1): liquidity known before the trade
            liq_cache[tk] = (dol.rolling(20, min_periods=10).median().shift(1), dol.rolling(60, min_periods=20).median().shift(1),
                             pr["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1))
        for j, col in enumerate(("adv20", "adv60", "sigma")):
            o[col] = [liq_cache[tk][j].get(d, np.nan) for d, tk in zip(o["date"], o["asset_str"])]
        o = o.dropna()
        yrs = (gd.END_TS - CAP_START).days / 365.25
        row = {"variant": a, "cagr_2012_26": cagr}
        for route in ("MOO", "MOC"):
            rec, fail = None, ""
            for aum in AUM_GRID:
                cost, gates = gc.route_cost_and_gates(o, route, aum)
                cpy = cost / aum / yrs
                if aum in (5e6, 2.5e7):
                    row[f"{route}_cost_{int(aum / 1e6)}m"] = cpy
                ok = all(v for k, v in gates.items() if k.endswith("_ok")) and cpy <= 0.25 * (cagr - 0.02)
                if ok and not fail:
                    rec = aum
                elif not fail:
                    bad = [k.replace("_ok", "") + f" ({gates[k.replace('_ok', '_worst')]})" for k, v in gates.items() if k.endswith("_ok") and not v]
                    fail = f"${aum / 1e6:g}M: " + ("; ".join(bad) if bad else f"cost {cpy:.2%}")
            row[f"{route}_recommended"] = rec
            row[f"{route}_first_fail"] = fail
        rows.append(row)
    df = pd.DataFrame(rows).set_index("variant")
    df.to_csv(pb.batch.OUT / ("phase5_capacity_alone.csv" if "wired" in pb.VARIANTS else "phase5_capacity_followup.csv"), float_format="%.5g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(df.to_string())


if __name__ == "__main__":
    main()
