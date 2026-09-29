"""Stage 2, family B: frozen pod grid per LIVE index group and pooled (amendment A5 tradability filter),
plus the labelled exploratory index-flow long/short pod.

Usage: python stage2_b.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_PATH))
import features as ft  # noqa: E402
import pod_stats as mt  # noqa: E402
import sim_events as se  # noqa: E402

OUT_PATH = ft.STUDY_OUT_PATH / "stage2_b"
LIVE_GROUPS = {"ndx": ("ndx",), "sp500": ("sp500",), "sp400": ("sp400",), "sp600": ("sp600",), "r1000": ("r1000",),
               "pooled": ("ndx", "sp500", "sp400", "sp600", "r1000")}


def main():
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    etf = ft.Panel("etfx")
    ev = pd.read_csv(se.EVENTS_PATH, parse_dates=["first_day"])
    all_rule = se.EventRule(groups=LIVE_GROUPS["pooled"], short_add_groups=("sp500", "ndx"))
    prices = se.load_prices(sorted(se.tradable_events(all_rule, ev).symbol.unique()), etf.dates)
    rows, navs = [], {}
    for gname, groups in LIVE_GROUPS.items():
        for slots in (10, 20):
            for hold in (10, 20):
                for hedge in ("none", "spy"):
                    rule = se.EventRule(groups=groups, slots=slots, hold=hold, hedge=hedge)
                    nav, gross, trades = se.run(rule, ev=ev, etf=etf, prices=prices)
                    st = mt.full(nav, gross, trades)
                    stress = mt.basic(se.run(se.EventRule(groups=groups, slots=slots, hold=hold, hedge=hedge,
                                                          extra_slippage=0.0005), ev=ev, etf=etf, prices=prices)[0])
                    st.update({"group": gname, "slots": slots, "hold": hold, "hedge": hedge, "stress_sharpe": stress["sharpe"],
                               "kind": "frozen"})
                    st.update(mt.difference_stats(nav.pct_change()))
                    rows.append(st)
                    navs[f"{gname}|S{slots}|h{hold}|{hedge}"] = nav
    # exploratory (A5): long tradable deletions + short tradable S&P 500 / Nasdaq-100 additions, SPY-hedged
    for slots in (10, 20):
        for hold in (10, 20):
            rule = se.EventRule(groups=LIVE_GROUPS["pooled"], short_add_groups=("sp500", "ndx"), slots=slots, hold=hold, hedge="spy")
            nav, gross, trades = se.run(rule, ev=ev, etf=etf, prices=prices)
            st = mt.full(nav, gross, trades)
            st.update({"group": "flow_LS", "slots": slots, "hold": hold, "hedge": "spy", "kind": "exploratory"})
            st.update(mt.difference_stats(nav.pct_change()))
            rows.append(st)
            navs[f"flow_LS|S{slots}|h{hold}|spy"] = nav
            rule_s = se.EventRule(groups=(), short_add_groups=("sp500", "ndx"), slots=slots, hold=hold, hedge="spy")
            nav, gross, trades = se.run(rule_s, ev=ev, etf=etf, prices=prices)
            st = mt.full(nav, gross, trades)
            st.update({"group": "adds_short_only", "slots": slots, "hold": hold, "hedge": "spy", "kind": "exploratory"})
            st.update(mt.difference_stats(nav.pct_change()))
            rows.append(st)
            navs[f"adds_short_only|S{slots}|h{hold}|spy"] = nav
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH / "grid.csv", index=False)
    pd.DataFrame(navs).to_parquet(OUT_PATH / "navs.parquet")
    cols = ["kind", "group", "slots", "hold", "hedge", "cagr", "sharpe", "maxdd", "stress_sharpe", "calm_sharpe", "calm_all_pos",
            "C1_2010_14_cagr", "C2_2015_19_cagr", "C3_2023_26_cagr", "avg_gross", "days_invested", "entries_per_year",
            "corr_dv2_adv", "corr_hpi_vote", "beta_spx"]
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print(df[cols].round(3).to_string())


if __name__ == "__main__":
    main()
