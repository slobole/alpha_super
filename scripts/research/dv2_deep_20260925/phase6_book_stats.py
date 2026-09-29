"""Full statistics for the growth book with the two new modules (owner question C, 2026-09-26).

Sleeves: TAA 3x (taa_btal_tqqq), NDX VXN, HPI vote from the growth-shelf inventory; DV2 = the ADV-ranked floor
module (replica with the module's median rule, verified against the engine); ETF = the industry-ETF module run
from 2012-01-03 through the real engine. Books reset annually (as all growth-shelf books). Window 2012-10-02 ->
2026-08-19 (TAA 3x inception); drawdown incl. the 2008 proxy uses the growth-shelf stand-in for TAA.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replica as rp  # noqa: E402
import phase5_book as pb  # noqa: E402
from phase5_book import common, gd  # noqa: E402

OUT = pb.batch.OUT


def main():
    sleeve = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    bench = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    p = rp.Panel("sp500")
    f1 = rp.run(p, rp.Rule(floor=True, rank="adv", floor_median_all=True))
    sleeve["dv2_adv"] = pd.Series(f1.nav, index=f1.dates).pct_change().reindex(sleeve.index)
    etf = pd.read_csv(OUT / "wired_check" / "etf__path.csv", index_col=0, parse_dates=True)["total_value"]
    sleeve["etf_ind"] = etf.pct_change().reindex(sleeve.index)
    long_df = sleeve.copy()
    long_df.loc[long_df.index < gd.CUT_TS, "taa_btal_tqqq"] = sleeve.loc[sleeve.index < gd.CUT_TS, "taa_1n_qld"]
    # 2008 proxy for the ETF sleeve: the same rules run from 2000 (research source), before the module's 2012 start
    etf_long = pd.read_csv(OUT / "sources" / "etf_ind_adv50__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"].pct_change()
    pre = long_df.index < pd.Timestamp("2012-01-03")
    long_df.loc[pre, "etf_ind"] = etf_long.reindex(long_df.index)[pre]
    books = {
        "G3 (TAA 50 / NDX 50)": {"taa_btal_tqqq": 0.5, "ndx_vxn": 0.5},
        "G3+MR, WIRED DV2 18": {"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2": 0.18, "hpi_vote": 0.18},
        "G3+MR, DV2-ADV 18": {"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2_adv": 0.18, "hpi_vote": 0.18},
        "G3+MR, DV2-ADV 9 + ETF 9": {"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2_adv": 0.09, "etf_ind": 0.09, "hpi_vote": 0.18},
    }
    spx = bench["SPXTR"]
    rows, yearly, rets = [], {}, {}
    for name, w in books.items():
        ex = common.book_return_ser(sleeve.loc[gd.CUT_TS:, list(w)].fillna(0.0), w, "annual")[0]
        lg = common.book_return_ser(long_df.loc[gd.LONG_TS:, list(w)].fillna(0.0), w, "annual")[0]
        rets[name] = ex
        nav = (1 + ex).cumprod()
        yrs = (ex.index[-1] - ex.index[0]).days / 365.25
        cagr = nav.iloc[-1] ** (1 / yrs) - 1
        vol = ex.std() * np.sqrt(252)
        down = ex[ex < 0].std() * np.sqrt(252)
        dd = (nav / nav.cummax() - 1).min()
        mon = (1 + ex).resample("ME").prod() - 1
        m = spx.reindex(ex.index)
        beta = np.cov(ex, m)[0, 1] / m.var()
        half = len(ex) // 2
        rows.append({"book": name, "CAGR": cagr, "vol": vol, "Sharpe": ex.mean() / ex.std() * np.sqrt(252),
                     "Sortino": ex.mean() * 252 / down, "maxDD_2012_26": dd, "Calmar": cagr / abs(dd),
                     "maxDD_incl_2008_proxy": (min(dd, ((1 + lg).cumprod() / (1 + lg).cumprod().cummax() - 1).min()) if lg is not None else np.nan),
                     "Sharpe_H1": ex.iloc[:half].mean() / ex.iloc[:half].std() * np.sqrt(252), "Sharpe_H2": ex.iloc[half:].mean() / ex.iloc[half:].std() * np.sqrt(252),
                     "worst_month": mon.min(), "best_month": mon.max(), "pct_pos_months": (mon > 0).mean(), "beta_spx": beta,
                     "corr_spx": ex.corr(m), "worst_year": ((1 + ex).resample("YE").prod() - 1).min()})
        yearly[name] = (1 + ex).resample("YE").prod() - 1
    df = pd.DataFrame(rows).set_index("book")
    yr = pd.DataFrame(yearly)
    yr.index = yr.index.year
    yr["S&P 500 TR"] = ((1 + spx.loc[gd.CUT_TS:]).resample("YE").prod() - 1).values[: len(yr)]
    corr = sleeve.loc[gd.CUT_TS:, ["taa_btal_tqqq", "ndx_vxn", "hpi_vote", "dv2_adv", "etf_ind"]].corr()
    # crisis windows
    crises = {"2015-08 flash": ("2015-08-10", "2015-08-31"), "2018-Q4": ("2018-10-01", "2018-12-24"), "2020 COVID": ("2020-02-19", "2020-03-23"),
              "2022 bear": ("2022-01-03", "2022-10-12"), "2025 April": ("2025-02-19", "2025-04-08")}
    cr = pd.DataFrame({name: {k: (1 + r.loc[a:b]).prod() - 1 for k, (a, b) in crises.items()} for name, r in rets.items()})
    cr["S&P 500 TR"] = [(1 + spx.loc[a:b]).prod() - 1 for a, b in crises.values()]
    out = {"stats": df.to_dict(orient="index"), "yearly": yr.to_dict(), "sleeve_corr": corr.to_dict(), "crises": cr.to_dict()}
    (OUT / "phase6_book_stats.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(df.round(3).T.to_string())
    print(yr.round(3).to_string())
    print(corr.round(2).to_string())
    print(cr.round(3).to_string())


if __name__ == "__main__":
    main()
