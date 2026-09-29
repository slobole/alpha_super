"""G8 book test: the growth book G3+MR "DV2-ADV 9 + ETF 9" with candidate pods added at 10% (others x 0.9), and
the labelled alternative of swapping HPI for the month-end flow pod. Pod model with annual reset
(fund_menu common.book_return_ser). Exact window 2012-10-02 -> 2026-08-19; long window from 2008-03-04 uses the
growth-shelf TAA stand-in (taa_1n_qld) before 2012-10-02 and the industry-ETF research run before 2012-01-03,
exactly as the DV2 deep study's phase 6.

Usage: python stage3_book.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
REPO_ROOT_PATH = HERE_PATH.parents[2]
sys.path.insert(0, str(HERE_PATH))
sys.path.insert(0, str(REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923"))
import common  # noqa: E402
import features as ft  # noqa: E402
import pod_stats as mt  # noqa: E402

DV2_OUT = REPO_ROOT_PATH / "results" / "research" / "dv2_deep_20260925"
CUT_TS, LONG_TS, END_TS = pd.Timestamp("2012-10-02"), pd.Timestamp("2008-03-04"), pd.Timestamp("2026-08-19")
BASE = {"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2_adv": 0.09, "etf_ind": 0.09, "hpi_vote": 0.18}


def sleeves(extra: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sl = mt.sleeve_returns().loc[:END_TS].copy()
    sl["dv2_adv"] = mt.dv2_adv_returns().reindex(sl.index)
    etf = pd.read_csv(DV2_OUT / "wired_check" / "etf__path.csv", index_col=0, parse_dates=True)["total_value"]
    sl["etf_ind"] = etf.pct_change().reindex(sl.index)
    for k, v in extra.items():
        sl[k] = v.reindex(sl.index)
    long_df = sl.copy()
    pre = long_df.index < CUT_TS
    long_df.loc[pre, "taa_btal_tqqq"] = sl.loc[pre, "taa_1n_qld"]
    etf_long = pd.read_csv(DV2_OUT / "sources" / "etf_ind_adv50__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"].pct_change()
    pre_etf = long_df.index < pd.Timestamp("2012-01-03")
    long_df.loc[pre_etf, "etf_ind"] = etf_long.reindex(long_df.index)[pre_etf]
    return sl, long_df


def stats(r: pd.Series) -> dict:
    nav = (1 + r).cumprod()
    b = mt.basic(nav)
    b["calmar"] = b["cagr"] / abs(b["maxdd"])
    half = len(r) // 2
    b["sharpe_h1"] = r.iloc[:half].mean() / r.iloc[:half].std() * np.sqrt(252)
    b["sharpe_h2"] = r.iloc[half:].mean() / r.iloc[half:].std() * np.sqrt(252)
    b["worst_year"] = ((1 + r).resample("YE").prod() - 1).min()
    return b


def main():
    nav_b = pd.read_parquet(ft.STUDY_OUT_PATH / "stage2_b" / "navs.parquet")
    # ndx_del = the frozen family-B pick (Nasdaq-100 exits, 20 slots, 20-session hold, unhedged)
    extra = {"ndx_del": nav_b["ndx|S20|h20|none"].pct_change(), "pooled_del": nav_b["pooled|S10|h20|none"].pct_change(),
             "flow_ls": nav_b["flow_LS|S20|h20|spy"].pct_change()}
    sl, long_df = sleeves(extra)
    books = {"G3 (TAA 50 / NDX 50)": {"taa_btal_tqqq": 0.5, "ndx_vxn": 0.5}, "BASE G3+MR DV2-ADV 9 + ETF 9": dict(BASE)}
    for cand in ["eom_flow", "ndx_del", "pooled_del", "flow_ls"]:
        w = {k: v * 0.9 for k, v in BASE.items()}
        w[cand] = 0.10
        books[f"BASE + {cand} 10"] = w
    swap = dict(BASE)
    swap["eom_flow"] = swap.pop("hpi_vote")
    books["BASE, HPI 18 -> EOM 18 (labelled alternative)"] = swap
    rows = []
    for name, w in books.items():
        ex = common.book_return_ser(sl.loc[CUT_TS:, list(w)].fillna(0.0), w, "annual")[0]
        lg = common.book_return_ser(long_df.loc[LONG_TS:, list(w)].fillna(0.0), w, "annual")[0]
        s = stats(ex)
        s["maxdd_incl_2008"] = min(s["maxdd"], mt.basic((1 + lg).cumprod())["maxdd"])
        s["calmar_incl_2008"] = s["cagr"] / abs(s["maxdd_incl_2008"])
        s["book"] = name
        rows.append(s)
    df = pd.DataFrame(rows).set_index("book")
    out = ft.STUDY_OUT_PATH / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "book_g8.csv")
    corr = sl.loc[CUT_TS:, ["taa_btal_tqqq", "ndx_vxn", "dv2_adv", "etf_ind", "hpi_vote", "eom_flow", "ndx_del", "pooled_del", "flow_ls"]].corr()
    corr.to_csv(out / "book_sleeve_corr.csv")
    pd.set_option("display.width", 250)
    print(df[["cagr", "vol", "sharpe", "maxdd", "calmar", "maxdd_incl_2008", "calmar_incl_2008", "sharpe_h1", "sharpe_h2", "worst_year"]].round(3).to_string())
    print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
