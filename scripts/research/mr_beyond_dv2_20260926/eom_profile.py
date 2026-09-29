"""EOM month-end rebalancing flow: statistics and chart data for the owner (research-only).

Source: the fund-menu standalone engine run of strategies/taa_beyond_6040/strategy_taa_month_end_rebalancing_flow.py
($1M, scheduled close-auction fills, 1% TLT borrow, engine costs), 2003-01-24 -> 2026-08-19.

    r_t          = NAV_t / NAV_{t-1} - 1
    stress r_t   = r_t - 5 bps * sum(|signed notional traded on t|) / NAV_{t-1}          (+5 bps per side)
    Sortino      = mean(r) * 252 / (std(r | r < 0) * sqrt(252))
    control      = no-signal TLT month-end cycle (report_tables.eom_gates_and_calendar_control)

Usage: python eom_profile.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_PATH))
import features as ft  # noqa: E402
import pod_stats as ps  # noqa: E402

SOURCE_PATH = HERE_PATH.parents[2] / "results" / "research" / "portfolio" / "fund_product_menu_20260923" / "sources"
OUT_PATH = ft.STUDY_OUT_PATH / "eom_profile"
START_TS, END_TS = pd.Timestamp("2003-01-24"), pd.Timestamp("2026-08-19")
CRISES = {"GFC": ("2008-09-01", "2009-03-09"), "2018 Q4": ("2018-10-01", "2018-12-24"), "COVID": ("2020-02-19", "2020-03-23"),
          "2022 bear": ("2022-01-03", "2022-10-12"), "April 2025": ("2025-02-19", "2025-04-08")}


def control_returns() -> pd.Series:
    etf = ft.Panel("etfx")
    c = etf.col("TLT")
    C = pd.Series(np.asarray(etf.C[:, c]), index=etf.dates)
    D = pd.Series(np.nan_to_num(np.asarray(etf.DIV[:, c])), index=etf.dates)
    # *** CRITICAL*** Dividend_p is paid to holders at Close_p: it belongs to the return from p to p+1.
    tr = ((C + D.shift(1)) / C.shift(1) - 1.0).loc["2003-01-01":END_TS]
    grp = pd.Series(1, index=tr.index).groupby([tr.index.year, tr.index.month])
    pos_start = grp.cumsum()
    pos_end = grp.transform("sum") - pos_start + 1
    position = pd.Series(0.0, index=tr.index)
    position[pos_end <= 5] = 1.0
    position[pos_start <= 5] = -1.0
    return position * tr - position.diff().abs().fillna(position.abs()) * 0.00025


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    nav = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    dd = nav / nav.cummax() - 1
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    rec = nav.loc[trough:][nav.loc[trough:] >= nav.loc[peak]]
    under = (dd < 0).astype(int)
    runs = under.groupby((under != under.shift()).cumsum()).transform("size") * under
    mon = (1 + r).resample("ME").prod() - 1
    yr = (1 + r).resample("YE").prod() - 1
    return {"cagr": cagr, "vol": r.std() * np.sqrt(252), "sharpe": r.mean() / r.std() * np.sqrt(252),
            "sortino": r.mean() * 252 / (r[r < 0].std() * np.sqrt(252)), "maxdd": dd.min(),
            "maxdd_peak": str(peak.date()), "maxdd_trough": str(trough.date()),
            "maxdd_recovery": str(rec.index[0].date()) if len(rec) else None,
            "longest_underwater_sessions": int(runs.max()), "calmar": cagr / abs(dd.min()),
            "worst_month": mon.min(), "worst_month_date": str(mon.idxmin().date()), "best_month": mon.max(),
            "positive_months": float((mon > 0).mean()), "worst_year": yr.min(), "worst_year_label": int(yr.idxmin().year),
            "best_year": yr.max(), "best_year_label": int(yr.idxmax().year), "positive_years": float((yr > 0).mean())}


def main():
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    path = pd.read_csv(SOURCE_PATH / "eom_flow__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"]
    trades = pd.read_csv(SOURCE_PATH / "eom_flow__transactions.csv.gz", parse_dates=["date"])
    r = path.pct_change().loc[START_TS:END_TS]
    sl = ps.sleeve_returns()
    assert np.allclose(r.values, sl["eom_flow"].loc[START_TS:END_TS].reindex(r.index).values, atol=1e-12, equal_nan=True)
    traded = trades.groupby("date")["signed_notional_float"].apply(lambda x: x.abs().sum()).reindex(r.index).fillna(0.0)
    r_stress = r - 0.0005 * traded / path.shift(1).reindex(r.index)
    ctl = control_returns().loc[START_TS:END_TS]
    bm = ps.benchmark_returns()["SPXTR"].loc[START_TS:END_TS]
    dv2 = sl["dv2"].loc[START_TS:END_TS]
    out = {"period": [str(r.index[0].date()), str(r.index[-1].date())],
           "eom": stats(r), "eom_stress_5bps": stats(r_stress), "control": stats(ctl), "spx_tr": stats(bm)}
    out["eom"].update(ps.difference_stats(r))
    j = pd.concat([r, bm], axis=1).dropna()
    up, down = j.iloc[:, 1] > 0, j.iloc[:, 1] < 0
    mon = (1 + j).resample("ME").prod() - 1
    out["eom"]["corr_spx_monthly"] = mon.corr().iloc[0, 1]
    out["eom"]["down_month_capture"] = mon.iloc[:, 0][mon.iloc[:, 1] < 0].mean() / mon.iloc[:, 1][mon.iloc[:, 1] < 0].mean()
    out["eom"]["share_days_in_market"] = float((r != 0).mean())
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    out["eom"]["trade_days_per_year"] = trades["date"].nunique() / yrs
    out["eom"]["one_way_turnover_per_year"] = float((trades.signed_notional_float.abs().sum() / 2) / path.loc[START_TS:END_TS].mean() / yrs)
    blocks = {}
    for blk, (a, b) in {"2003-09": ("2003-01-01", "2009-12-31"), **{k: v for k, v in ps.BLOCKS.items() if k != "E_2000_09"}}.items():
        x = r.loc[a:b]
        blocks[blk] = {"sharpe": x.mean() / x.std() * np.sqrt(252), "cagr": (1 + x).prod() ** (252 / len(x)) - 1}
    out["blocks"] = blocks
    out["crises"] = {k: {"eom": float((1 + r.loc[a:b]).prod() - 1), "spx_tr": float((1 + bm.loc[a:b]).prod() - 1),
                         "dv2": float((1 + dv2.loc[a:b]).prod() - 1)} for k, (a, b) in CRISES.items()}
    yr = pd.DataFrame({"eom": (1 + r).resample("YE").prod() - 1, "control": (1 + ctl).resample("YE").prod() - 1,
                       "spx_tr": (1 + bm).resample("YE").prod() - 1})
    yr.index = yr.index.year
    out["annual"] = {int(k): {c: float(v) for c, v in row.items()} for k, row in yr.iterrows()}
    # weekly chart series (last session of each week), growth of 1 from the close before the first trade
    base_ts = path.index[path.index.get_loc(START_TS) - 1]
    nav = pd.DataFrame({"eom": (1 + r).cumprod(), "control": (1 + ctl.reindex(r.index).fillna(0)).cumprod(),
                        "spx_tr": (1 + bm.reindex(r.index).fillna(0)).cumprod()})
    nav.loc[base_ts] = 1.0
    nav = nav.sort_index()
    wk = nav.resample("W-FRI").last().dropna()
    dd = (nav["eom"] / nav["eom"].cummax() - 1).resample("W-FRI").min().reindex(wk.index)
    out["weekly"] = {"dates": [str(d.date()) for d in wk.index], "eom": wk["eom"].round(4).tolist(),
                     "control": wk["control"].round(4).tolist(), "spx_tr": wk["spx_tr"].round(4).tolist(),
                     "eom_dd": dd.round(4).tolist()}
    (OUT_PATH / "eom_profile.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    view = {k: v for k, v in out.items() if k not in ("weekly", "annual")}
    print(json.dumps(view, indent=1, default=lambda x: round(float(x), 4)))
    print(pd.DataFrame(out["annual"]).T.round(3).to_string())


if __name__ == "__main__":
    main()
