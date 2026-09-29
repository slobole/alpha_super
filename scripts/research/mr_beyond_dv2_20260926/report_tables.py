"""Every number quoted in docs/research/MR_BEYOND_DV2_20260926.md that is not printed by a stage script.

Reads saved stage outputs (run the stage scripts first) and writes results/research/mr_beyond_dv2_20260926/report_tables/.
Sections: family-A tally; S&P 500 signal table by block and VIX tercile (full sample and 2010-2026 only); Nasdaq-100
deletion and turn-of-year diagnostics; S&P 500 addition fade; EOM flow gates and the no-signal TLT month-end control.

    calendar control: position = +1 on the last 5 sessions of each month, -1 on the first 5, else 0;
    r_t = position_t * TR_t(TLT) - 2.5 bps * |position_t - position_{t-1}|
    EOM flow = a + b * control + e   (OLS on daily returns; a annualised x 252)

Usage: python report_tables.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_PATH))
import features as ft  # noqa: E402
import pod_stats as ps  # noqa: E402
from stage1_a import nw_t  # noqa: E402
from stage1_b import cluster_t  # noqa: E402

OUT_PATH = ft.STUDY_OUT_PATH / "report_tables"
UNIVERSE_LIST = ["sp500", "ndx", "sp400", "sp600", "r1000x"]


def family_a_tally() -> pd.DataFrame:
    rows = []
    for u in UNIVERSE_LIST:
        df = pd.read_csv(ft.STUDY_OUT_PATH / "stage1_a" / f"map_{u}.csv")
        for bucket in ("dec", "b10"):
            d = df[df.bucket == bucket]
            calm = d[d.block == "CALM"].set_index(["tier", "signal", "h", "hedge"])
            piv = d.pivot_table(index=["tier", "signal", "h", "hedge"], columns="block", values="mean_excess")
            live = (calm.t_net >= 2) & (piv[["C1_2010_14", "C2_2015_19", "C3_2023_26"]] > 0).all(axis=1)
            rows.append({"universe": u, "bucket": bucket, "cells": len(calm), "live": int(live.sum()),
                         "best_calm_excess_bps": calm.mean_excess.max() * 1e4, "best_calm_t_net": calm.t_net.max()})
    return pd.DataFrame(rows)


def sp500_signal_table() -> pd.DataFrame:
    df = pd.read_csv(ft.STUDY_OUT_PATH / "stage1_a" / "map_sp500.csv")
    daily = pd.read_parquet(ft.STUDY_OUT_PATH / "stage1_a" / "daily_excess_sp500.parquet")
    etf = ft.Panel("etfx")
    vix = pd.Series(np.asarray(etf.extra["vix_close"]), index=etf.dates)
    q = np.nanquantile(vix.loc["1990-01-01":], [1 / 3, 2 / 3])
    vix_bin = pd.cut(vix, [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"]).reindex(daily.index)
    rows = []
    for sig, h, hg in [("DV2", 5, "none"), ("R5raw", 5, "spy"), ("E5_SPY", 5, "spy"), ("E5_SEC", 5, "sec"), ("INTRA5", 5, "spy")]:
        x = df[(df.tier == "L") & (df.bucket == "dec") & (df.signal == sig) & (df.h == h) & (df.hedge == hg)].set_index("block")
        row = {"signal": sig, "h": h, "hedge": hg}
        for blk in ["H_1991_99", "E_2000_09", "C1_2010_14", "C2_2015_19", "S_2020_22", "C3_2023_26", "VIX_low", "VIX_mid", "VIX_high"]:
            row[blk] = x.loc[blk, "mean_excess"] * 1e4
        ser = daily[f"sp500|L|{sig}|{h}|{hg}|dec"].loc["2010-01-01":"2026-08-19"]
        for b in ("low", "mid", "high"):
            row[f"2010_26_VIX_{b}"] = nw_t(ser[vix_bin.loc[ser.index] == b].to_numpy(), h - 1)[0] * 1e4
        row["calm_t_net"] = x.loc["CALM", "t_net"]
        rows.append(row)
    return pd.DataFrame(rows), pd.Series(q, index=["tercile_1", "tercile_2"])


def event_diagnostics() -> pd.DataFrame:
    ev = pd.read_csv(ft.STUDY_OUT_PATH / "stage1_b" / "events.csv", parse_dates=["first_day"])
    tradable = (ev.raw_close > 5) & (ev.adv63 > 5e6)
    main = (ev.first_day >= "2000-01-01") & (ev.first_day <= "2026-08-19")
    rows = []
    ndx = ev[(ev["index"] == "ndx") & (ev.kind == "exit") & (ev.other == "none") & tradable & main]
    for label, sub in {"ndx_exits_all": ndx, "ndx_exits_december": ndx[ndx.first_day.dt.month == 12],
                       "ndx_exits_other_months": ndx[ndx.first_day.dt.month != 12],
                       "ndx_exits_2000_12": ndx[ndx.first_day <= "2012-12-31"], "ndx_exits_2013_26": ndx[ndx.first_day >= "2013-01-01"]}.items():
        m, t, n = cluster_t(sub.car_d2_20.to_numpy(), sub.first_day.to_numpy())
        rows.append({"set": label, "window": "CAR 20d", "n": n, "mean_bps": m * 1e4, "median_bps": sub.car_d2_20.median() * 1e4, "t": t,
                     "share_positive": (sub.car_d2_20 > 0).mean()})
    add = ev[(ev["index"] == "sp500") & (ev.kind == "add") & (ev.other == "none")]
    for label, sub in {"sp500_adds_1991_99": add[(add.first_day >= "1991-01-01") & (add.first_day <= "1999-12-31")],
                       "sp500_adds_2000_26": add[main.loc[add.index]], "sp500_adds_2000_26_tradable": add[main.loc[add.index] & tradable.loc[add.index]]}.items():
        m, t, n = cluster_t(sub.car_d2_5.to_numpy(), sub.first_day.to_numpy())
        rows.append({"set": label, "window": "CAR 5d", "n": n, "mean_bps": m * 1e4, "median_bps": sub.car_d2_5.median() * 1e4, "t": t,
                     "share_positive": (sub.car_d2_5 > 0).mean()})
    for u in ("sp500", "ndx"):
        f = ft.STUDY_OUT_PATH / "stage1_e" / f"map_e_{u}.csv"
        if f.exists():
            e = pd.read_csv(f)
            e = e[(e.signal == "YTD") & (e.hedge == "spy") & (e.tier == "L") & (e.h == 5)]
            for blk in ("MAIN_2000_26", "HALF1_2000_12", "HALF2_2013_26"):
                r = e[e.block == blk].iloc[0]
                rows.append({"set": f"{u}_december_ytd_losers_{blk}", "window": "5 January sessions", "n": r.n, "mean_bps": r.mean_excess * 1e4,
                             "median_bps": np.nan, "t": r.t, "share_positive": np.nan})
    return pd.DataFrame(rows)


def eom_gates_and_calendar_control() -> tuple[pd.DataFrame, dict]:
    sl = ps.sleeve_returns()
    rows = []
    for c in ["eom_flow", "dv2", "hpi_vote", "qpi", "sector_vox_iyr", "disp_kie_ihi_sma"]:
        r = sl[c].dropna()
        st = ps.full((1 + r).cumprod())
        st.update(ps.difference_stats(r))
        st["sleeve"], st["start"] = c, str(r.index.min().date())
        rows.append(st)
    gates = pd.DataFrame(rows).set_index("sleeve")
    etf = ft.Panel("etfx")
    c = etf.col("TLT")
    C = pd.Series(np.asarray(etf.C[:, c]), index=etf.dates)
    D = pd.Series(np.nan_to_num(np.asarray(etf.DIV[:, c])), index=etf.dates)
    # *** CRITICAL*** Dividend_p is paid to holders at Close_p, so it belongs to the return from p to p+1.
    tr = ((C + D.shift(1)) / C.shift(1) - 1.0).loc["2003-01-01":"2026-08-19"]
    grp = pd.Series(1, index=tr.index).groupby([tr.index.year, tr.index.month])
    pos_start = grp.cumsum()
    pos_end = grp.transform("sum") - pos_start + 1
    position = pd.Series(0.0, index=tr.index)
    position[pos_end <= 5] = 1.0
    position[pos_start <= 5] = -1.0
    control = position * tr - position.diff().abs().fillna(position.abs()) * 0.00025
    j = pd.concat([control.rename("control"), sl["eom_flow"]], axis=1).dropna()
    X = np.column_stack([np.ones(len(j)), j["control"].to_numpy()])
    y = j["eom_flow"].to_numpy()
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    se = np.sqrt(res.var(ddof=2) * np.linalg.inv(X.T @ X).diagonal())
    ctl = {"control_sharpe": control.mean() / control.std() * np.sqrt(252),
           "control_sharpe_2003_14": control.loc[:"2014-12-31"].mean() / control.loc[:"2014-12-31"].std() * np.sqrt(252),
           "control_sharpe_2015_26": control.loc["2015-01-01":].mean() / control.loc["2015-01-01":].std() * np.sqrt(252),
           "control_cagr": (1 + control).prod() ** (252 / len(control)) - 1, "corr_control_eom": j.corr().iloc[0, 1],
           "eom_alpha_over_control_per_year": b[0] * 252, "eom_alpha_t": b[0] / se[0], "eom_beta_to_control": b[1],
           "eom_variance_share_explained": 1 - res.var() / y.var()}
    return gates, ctl


def main():
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    tally = family_a_tally()
    tally.to_csv(OUT_PATH / "family_a_tally.csv", index=False)
    print(tally.round(2).to_string(index=False))
    table, q = sp500_signal_table()
    table.to_csv(OUT_PATH / "sp500_signal_table.csv", index=False)
    print("VIX terciles", q.round(2).to_dict())
    print(table.round(1).to_string(index=False))
    diag = event_diagnostics()
    diag.to_csv(OUT_PATH / "event_diagnostics.csv", index=False)
    print(diag.round(2).to_string(index=False))
    gates, ctl = eom_gates_and_calendar_control()
    gates.to_csv(OUT_PATH / "eom_and_mr_gates.csv")
    pd.Series(ctl).to_csv(OUT_PATH / "eom_calendar_control.csv")
    cols = ["start", "cagr", "sharpe", "maxdd", "calm_sharpe", "calm_all_pos", "E_2000_09_sharpe", "C1_2010_14_sharpe", "C2_2015_19_sharpe",
            "S_2020_22_sharpe", "C3_2023_26_sharpe", "corr_dv2_adv", "corr_hpi_vote", "beta_spx"]
    print(gates[cols].round(3).to_string())
    print({k: round(v, 4) for k, v in ctl.items()})


if __name__ == "__main__":
    main()
