"""Descriptive overview of the defensive candidates (owner request 2026-09-30): full statistics, robustness,
year-by-year, crises and chart series. Nothing is selected here.

Usage: python candidates_overview.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "shelf_rebuild_20260929"))

import lib  # noqa: E402
import defensive_v2 as dv  # noqa: E402
from lib import END, EXACT_START, LONG_START, TBILL  # noqa: E402

OUT = dv.OUT / "overview"
CANDIDATES = [
    ("core5", "CORE5 לבד (בנצ׳מרק)", dv.DefBook("c", ("core5",), "EQ", {}, 0.0)),
    ("champ", "‏60/40 ‏CORE5/BTAL_QQQ", dv.DefBook("c", ("core5", "btal_qqq"), "FIXED", {"core5": 0.6, "btal_qqq": 0.4}, 0.0)),
    ("three", "‏CORE5 + BTAL_QQQ + DV2 תעשייתי", dv.DefBook("c", ("core5", "btal_qqq", "etf_dv2"), "EQ", {}, 0.0)),
    ("three_iv", "אותו הדבר, הפוך-לתנודתיות", dv.DefBook("c", ("core5", "btal_qqq", "etf_dv2"), "IV", {}, 0.0)),
    ("c5dv2", "‏CORE5 + DV2 תעשייתי", dv.DefBook("c", ("core5", "etf_dv2"), "EQ", {}, 0.0)),
    ("bqdv2", "‏BTAL_QQQ + DV2 תעשייתי", dv.DefBook("c", ("btal_qqq", "etf_dv2"), "EQ", {}, 0.0)),
    ("eom", "‏CORE5 + EOM + DISP", dv.DefBook("c", ("core5", "eom_flow", "disp"), "IV", {}, 0.0)),
]
CRISES = [("gfc", "משבר 2008", "2008-05-19", "2009-03-09"), ("eu2011", "אירופה 2011", "2011-04-29", "2011-10-03"),
          ("china2015", "סין 2015–16", "2015-07-20", "2016-02-11"), ("feb2018", "פברואר 2018", "2018-01-26", "2018-02-08"),
          ("q4_2018", "רבעון 4 2018", "2018-09-20", "2018-12-24"), ("covid", "קורונה 2020", "2020-02-19", "2020-03-23"),
          ("bear2022", "שוק דובי 2022", "2022-01-03", "2022-10-12"), ("bonds2023", "מפולת אג״ח 2023", "2023-07-31", "2023-10-27"),
          ("tariffs2025", "מכסים 2025", "2025-02-19", "2025-04-08")]


def stats(r: pd.Series, tb: pd.Series, spx: pd.Series, idx) -> dict:
    base = lib.base_date(idx, r)
    ex = r - tb.reindex(r.index)
    monthly = (1 + r).resample("ME").prod() - 1
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    roll21 = (1 + r).rolling(21).apply(np.prod, raw=True) - 1
    nav = (1 + r).cumprod()
    dd = nav / nav.cummax() - 1
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    rec = nav.loc[trough:][nav.loc[trough:] >= nav.loc[peak]]
    s = spx.reindex(r.index)
    return {"cagr": lib.cagr(r, base), "vol": float(r.std() * np.sqrt(252)), "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
            "xs_sharpe": float(ex.mean() / ex.std() * np.sqrt(252)), "sortino": float(r.mean() / r[r < 0].std() * np.sqrt(252)),
            "maxdd": float(dd.min()), "dd_peak": peak.date().isoformat(), "dd_trough": trough.date().isoformat(),
            "dd_recovery": rec.index[0].date().isoformat() if len(rec) else None,
            "calmar": lib.cagr(r, base) / abs(float(dd.min())),
            "cvar5_21d": float(roll21[roll21 <= roll21.quantile(0.05)].mean()),
            "worst_month": float(monthly.min()), "worst_year": float(yearly.min()), "pos_months": float((monthly > 0).mean()),
            "beta": float(r.cov(s) / s.var()), "corr": float(r.corr(s)), "crisis_corr": lib.crisis_corr(r, spx)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = lib.load_inputs()
    idx = data["index"]
    fair, house = data["cash_long"], data["long"]
    spx = data["bench"]["SPXTR"]
    swap_tfi = data["long"].copy()
    swap_tfi["tactical_fi"] = data["tfi_frozen"]
    sens_frames = {"house_zero_cash": house, "proxy_unscaled": dv.fair(data["long_unscaled"], data),
                   "plus_5bps": dv.fair(data["stressed_long"], data), "etf_idle_2008_09": dv.fair(data["long_etf_cash"], data)}
    series, rows, sens, years, crises, subs = {}, [], [], {}, {}, []
    for key, label, b in CANDIDATES:
        r = dv.book_series(fair, b)
        series[key] = r
        log: list = []
        if b.rule == "IV":
            lib.book_returns(fair, b.defensive(), LONG_START, weight_log=log)
            w = lib.average_weights(log)
        else:
            w = b.defensive().targets()
        row = {"key": key, "label": label, "weights": {k: round(v, 3) for k, v in w.items()}, "pods": len(b.pods)}
        for win, start in (("long", LONG_START), ("exact", EXACT_START), ("recent", lib.BLOCK_DICT["RECENT"][0])):
            rw = r.loc[start:END]
            row.update({f"{win}_{k}": v for k, v in stats(rw, fair[TBILL], spx, idx).items()})
        row["trade_days"] = len(set().union(*(lib.trade_dates(data, p) for p in b.pods))) / ((END - EXACT_START).days / 365.25)
        rows.append(row)
        for sl, frame in sens_frames.items():
            rs = dv.book_series(frame, b)
            sens.append({"key": key, "sensitivity": sl, "cagr": lib.cagr(rs, lib.base_date(idx, rs)),
                         "sharpe": float(rs.mean() / rs.std() * np.sqrt(252)), "maxdd": lib.maxdd(rs)})
        for block, (lo, hi) in lib.BLOCK_DICT.items():
            rb = lib.window(r, lo, hi)
            subs.append({"key": key, "block": block, "sharpe": float(rb.mean() / rb.std() * np.sqrt(252)),
                         "cagr": lib.cagr(rb, lib.base_date(idx, rb)), "maxdd": lib.maxdd(rb)})
        years[key] = {int(y): float(v) for y, v in ((1 + r).groupby(r.index.year).prod() - 1).items()}
        crises[key] = {c: lib.common.window_return_float(r, lo, hi) for c, _, lo, hi in CRISES}
    for key, col in (("spx", "SPXTR"), ("sixty", "SIXTY_FORTY"), ("bil", "BIL")):
        r = data["bench"][col].loc[LONG_START:END]
        series[key] = r
        years[key] = {int(y): float(v) for y, v in ((1 + r).groupby(r.index.year).prod() - 1).items()}
        crises[key] = {c: lib.common.window_return_float(r, lo, hi) for c, _, lo, hi in CRISES}
    R = pd.DataFrame({k: series[k] for k, _, _ in CANDIDATES})
    boot_idx = lib.boot_index(len(R))
    A, tb = R.to_numpy(), fair[TBILL].reindex(R.index).to_numpy()
    sh = np.empty((boot_idx.shape[0], A.shape[1]))
    ddb = np.empty_like(sh)
    for k in range(boot_idx.shape[0]):
        s = A[boot_idx[k]]
        sh[k] = s.mean(axis=0) / s.std(axis=0, ddof=1) * np.sqrt(252)
        ddb[k] = lib.path_stats(s)[1]
    champ = [k for k, _, _ in CANDIDATES].index("champ")
    for j, row in enumerate(rows):
        row["p_beats_champ_sharpe"] = float(np.mean(sh[:, j] > sh[:, champ]))
        row["p_breach10"] = float(np.mean(ddb[:, j] < -0.10))
        row["p_breach8"] = float(np.mean(ddb[:, j] < -0.08))
        row["boot_dd_p50"] = float(np.median(ddb[:, j]))
        row["boot_dd_p05"] = float(np.percentile(ddb[:, j], 5))
        row["boot_sharpe_p05"] = float(np.percentile(sh[:, j], 5))
        row["boot_sharpe_p50"] = float(np.median(sh[:, j]))
    curves = {k: [[d.strftime("%Y-%m"), round(float(v), 5)] for d, v in (1 + s).cumprod().resample("ME").last().items()]
              for k, s in series.items()}
    dds = {}
    for k, s in series.items():
        nav = (1 + s).cumprod()
        dd = (nav / nav.cummax() - 1).resample("W-FRI").min()
        dds[k] = [[d.strftime("%Y-%m-%d"), round(float(v), 5)] for d, v in dd.items()]
    corr = R.loc[EXACT_START:END].corr().round(3)
    payload = {"rows": rows, "sens": sens, "subs": subs, "years": years, "crises": crises,
               "crisis_list": [[c, l, lo, hi] for c, l, lo, hi in CRISES], "curves": curves, "drawdowns": dds,
               "labels": {k: l for k, l, _ in CANDIDATES}}
    (OUT / "overview.json").write_text(json.dumps(payload, ensure_ascii=False, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = pd.DataFrame(rows).set_index("key")[["long_cagr", "long_sharpe", "long_maxdd", "exact_sharpe", "recent_sharpe",
                                                 "p_beats_champ_sharpe", "p_breach10", "boot_sharpe_p05", "trade_days"]]
    print(show.round(3).to_string())
    print(pd.DataFrame(sens).pivot(index="key", columns="sensitivity", values="sharpe").round(3).to_string())
    print(corr.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
