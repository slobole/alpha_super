"""More statistics for the frozen MR final-slot test (owner request 2026-10-04); the decision is mr_final_slot.py.

Same legs and conventions as mr_final_slot.py at engine costs (replica + HPI engine run, T-bill sweep, annual reset).
Book = TAA 0.5 + NDX 0.25 + slot 0.25, 2008-03-04 .. 2026-08-19; slot standalone 2004-01-02 .. 2026-09-24.
Writes results/research/mr_capsule_build_20261004/mr_final_slot_stats.json and prints two tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mr_final_slot as mfs  # noqa: E402

ev, st, npc, tbc = mfs.ev, mfs.st, mfs.npc, mfs.tbc
WINDOWS = {"GFC 08-03..09-03": ("2008-03-04", "2009-03-09"), "Aug 2011": ("2011-07-22", "2011-10-03"),
           "Q4 2018": ("2018-10-01", "2018-12-24"), "COVID 2020": ("2020-02-19", "2020-03-23"), "2022": ("2022-01-03", "2022-12-30")}


def stats(ret: pd.Series, rate: pd.Series) -> dict:
    ret = ret.dropna()
    wealth = (1.0 + ret).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    years = len(ret) / 252.0
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
    vol = float(ret.std() * np.sqrt(252.0))
    downside = ret[ret < 0]
    excess = ret - rate.reindex(ret.index).fillna(0.0)
    yearly = (1.0 + ret).groupby(ret.index.year).prod() - 1.0
    monthly = (1.0 + ret).groupby(ret.index.to_period("M")).prod() - 1.0
    underwater = (drawdown < 0).astype(int)
    longest = int(underwater.groupby((underwater == 0).cumsum()).sum().max())
    return {
        "cagr": cagr, "vol": vol, "sharpe": float(ret.mean() / ret.std() * np.sqrt(252.0)),
        "sharpe_excess_tbill": float(excess.mean() / excess.std() * np.sqrt(252.0)),
        "sortino": float(ret.mean() * 252.0 / (np.sqrt((downside ** 2).sum() / len(ret)) * np.sqrt(252.0))),
        "max_dd": float(drawdown.min()), "calmar": cagr / abs(float(drawdown.min())),
        "worst_year": float(yearly.min()), "worst_year_label": int(yearly.idxmin()), "best_year": float(yearly.max()),
        "worst_month": float(monthly.min()), "pct_positive_months": float((monthly > 0).mean()),
        "longest_underwater_sessions": longest, "wealth_100k": float(100_000 * wealth.iloc[-1]),
    }


def main() -> None:
    taa, ndx = tbc.load_taa_ser(), npc.load_l_ret_ser("engine")
    leg_dict = mfs.legs("engine")
    rate = st.cash_rate(pd.DatetimeIndex(pd.concat(list(leg_dict.values()), axis=1).index))
    spx = st.npc.load_total_return_ret_ser("$SPXTR", "$SPXTR") if hasattr(st.npc, "load_total_return_ret_ser") else None
    report = {}
    for cand, weights in mfs.CANDIDATES.items():
        slot = ev.capsule({k: leg_dict[k] for k in weights}, weights, start=mfs.SLOT_START, end=mfs.SLOT_END)
        book = tbc.book_window_return_ser({"taa": taa, "L": ndx, "X": slot}, npc.CANDIDATE_WEIGHT_DICT, *mfs.BOOK)
        entry = {"book": stats(book, rate), "slot": stats(slot.loc[mfs.SLOT_START:mfs.SLOT_END], rate)}
        entry["book_windows"] = {k: float((1.0 + book.loc[a:b]).prod() - 1.0) for k, (a, b) in WINDOWS.items()}
        entry["slot_windows"] = {k: float((1.0 + slot.loc[a:b]).prod() - 1.0) for k, (a, b) in WINDOWS.items()}
        window = slot.loc[mfs.BOOK[0]:mfs.BOOK[1]]
        entry["slot_corr"] = {"taa": float(pd.concat([window, taa], axis=1).dropna().corr().iloc[0, 1]),
                              "ndx": float(pd.concat([window, ndx], axis=1).dropna().corr().iloc[0, 1])}
        if spx is not None:
            both = pd.concat([window, spx.reindex(window.index)], axis=1).dropna()
            beta = float(np.polyfit(both.iloc[:, 1], both.iloc[:, 0], 1)[0])
            entry["slot_beta_spx"] = beta
        report[cand] = entry
    (mfs.OUT / "mr_final_slot_stats.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print("BOOK 2008-03..2026-08")
    print(f"{'':<4}{'CAGR':>7}{'Vol':>7}{'Sharpe':>8}{'xsTB':>6}{'Sortino':>8}{'MaxDD':>8}{'Calmar':>7}{'WorstY':>8}{'WorstM':>8}{'%M+':>6}{'UW d':>6}{'$100K':>9}"
          + "".join(f"{k:>18}" for k in WINDOWS))
    for cand, e in report.items():
        b = e["book"]
        print(f"{cand:<4}{b['cagr'] * 100:>6.1f}%{b['vol'] * 100:>6.1f}%{b['sharpe']:>8.3f}{b['sharpe_excess_tbill']:>6.2f}{b['sortino']:>8.2f}{b['max_dd'] * 100:>7.1f}%"
              f"{b['calmar']:>7.2f}{b['worst_year'] * 100:>6.1f}%{b['worst_year_label'] % 100:>2}{b['worst_month'] * 100:>7.1f}%{b['pct_positive_months'] * 100:>5.0f}%"
              f"{b['longest_underwater_sessions']:>6}{b['wealth_100k'] / 1e3:>8.0f}K" + "".join(f"{v * 100:>17.1f}%" for v in e["book_windows"].values()))
    print("SLOT standalone 2004-01..2026-09")
    print(f"{'':<4}{'CAGR':>7}{'Vol':>7}{'Sharpe':>8}{'xsTB':>6}{'MaxDD':>8}{'Calmar':>7}{'WorstY':>8}{'corrTAA':>8}{'corrNDX':>8}{'betaSPX':>8}"
          + "".join(f"{k:>18}" for k in WINDOWS))
    for cand, e in report.items():
        s = e["slot"]
        print(f"{cand:<4}{s['cagr'] * 100:>6.1f}%{s['vol'] * 100:>6.1f}%{s['sharpe']:>8.3f}{s['sharpe_excess_tbill']:>6.2f}{s['max_dd'] * 100:>7.1f}%{s['calmar']:>7.2f}"
              f"{s['worst_year'] * 100:>6.1f}%{s['worst_year_label'] % 100:>2}{e['slot_corr']['taa']:>8.2f}{e['slot_corr']['ndx']:>8.2f}{e.get('slot_beta_spx', float('nan')):>8.2f}"
              + "".join(f"{v * 100:>17.1f}%" for v in e["slot_windows"].values()))


if __name__ == "__main__":
    main()
