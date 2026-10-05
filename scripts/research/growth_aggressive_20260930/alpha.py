"""EA2 (reported, not selecting): factor alpha of the verdict books, G3 and the benchmarks, gross and net of 2/20.

Method as fund_menu_20260923/allocator_first_look.py: weekly (W-FRI) excess returns over T-bills (BIL TR here, the
study's hurdle) regressed on the excess returns of QQQ, IEF, GLD, DBC, UUP (M1) plus the QQQ 200-day trend rule (M2; M0 = QQQ alone,
QQQ above its 200-day average -> QQQ next session, else IEF); Newey-West 4 lags; the partial first and last weeks
dropped. Windows: LONG, EXACT and the two halves of EXACT. Usage: python alpha.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ga_lib as ga
from ga_lib import END, EXACT_START, LONG_START, lib
import select_ext as se
import allocator_first_look as afl  # noqa: E402  (fund menu folder is on sys.path via lib)

BOOKS = {"G3": ga.G3_NAME,
         "GROWTH verdict": "TAA3x-1N + NDX-VXN @60:40 | def2@36",
         "AGGRESSIVE verdict": "TAA3x-1N + NDX-VXN @70:30 | def2@18",
         "AGGRESSIVE rule (NDX-ATR)": "TAA3x-1N + NDX-ATR @70:30 | def2@18",
         "GROWTH first report (50:50)": "TAA3x-1N + NDX-VXN | def2@36"}


def main() -> int:
    data = lib.load_inputs()
    data["meta"]["ndx_rm"] = {"tier_str": "shadow"}
    frames = se.ext_frames(data)
    books = {b.name: b for b in se.family()}
    closes = pd.concat([lib.common.load_total_return_close_ser(s, "2005-01-01", END.strftime("%Y-%m-%d")) for s in afl.ETF_LIST], axis=1)
    closes.columns = afl.ETF_LIST
    closes = closes.reindex(data["index"]).loc["2005-01-01":]
    tbill = data["sleeve"][ga.TBILL]
    naive = afl.naive_rule_return_df(closes, tbill)
    etf_r = closes.pct_change(fill_method=None)
    tb_w = afl.weekly_ser(tbill)
    fac_w = pd.DataFrame({s: afl.weekly_ser(etf_r[s]) for s in afl.FACTOR_M1_LIST}).sub(tb_w, axis=0)
    trend_w = afl.weekly_ser(naive["NAIVE_QQQ_TREND200"]) - tb_w

    series = {}
    for label, name in BOOKS.items():
        r_long = lib.book_returns(frames["main"][0], books[name], LONG_START)
        r_exact = lib.book_returns(frames["s6_exact"][0], books[name], EXACT_START)
        series[(label, "gross")] = (r_long, r_exact)
        series[(label, "net")] = (ga.net_return_series(r_long), ga.net_return_series(r_exact))
    for label, col in (("S&P 500 TR", "SPXTR"), ("60/40", "SIXTY_FORTY")):
        b = data["bench"][col]
        series[(label, "gross")] = (b.loc[LONG_START:END], b.loc[EXACT_START:END])
    series[("QQQ 200-day rule", "gross")] = (naive["NAIVE_QQQ_TREND200"].loc[LONG_START:END], naive["NAIVE_QQQ_TREND200"].loc[EXACT_START:END])

    rows = []
    for (label, basis), (r_long, r_exact) in series.items():
        halves = np.array_split(r_exact.index, 2)
        for window, ser in (("long", r_long), ("exact", r_exact), ("exact_h1", r_exact.loc[halves[0]]), ("exact_h2", r_exact.loc[halves[1]])):
            y = (afl.weekly_ser(ser.dropna()) - tb_w).dropna().iloc[1:-1]
            x1 = fac_w.reindex(y.index)
            x2 = x1.assign(TREND200=trend_w.reindex(y.index))
            x0 = x1[["QQQ"]]
            for model, x in (("M0", x0), ("M1", x1), ("M2", x2)):
                if label == "QQQ 200-day rule" and model == "M2":
                    continue
                res = afl.newey_west_ols(y, x, afl.NW_LAG_INT)
                rows.append({"series": label, "basis": basis, "window": window, "model": model,
                             "start": str(ser.index[0].date()), **res})
    out = pd.DataFrame(rows)
    out.to_csv(ga.STUDY / "report" / "alpha.csv", index=False, float_format="%.6g")
    (ga.STUDY / "report" / "alpha.json").write_text(out.to_json(orient="records"), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = out[out["model"] == "M2"].pivot_table(index=["series", "basis"], columns="window", values=["alpha_ann", "alpha_t"], sort=False)
    print(show.round(3).to_string())
    print(out[(out["model"] == "M2") & (out["window"] == "long")][["series", "basis", "r2", "b_QQQ", "b_IEF", "b_GLD", "b_TREND200"]].round(2).to_string())
    ga.ledger("ea2_alpha_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
