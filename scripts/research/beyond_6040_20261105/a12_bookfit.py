"""A12 Book fit: how does Beyond 60/40 (B0) sit next to the house pods? Research-only, reads caches, runs no backtest.

Pod daily returns (all with the fund study's fair-cash add, DTB3 - 0.5% on idle cash):
  frozen fund frame  results/scout/growth_hunt_20261030/shared/frame_main.parquet  -> core5, taa3x, taa3x_1n, btal_qqq,
                     ndx_atr_cap, ndx_natr_cap, dv2_g, hpi_g, tbill (the Engine Map's history source, build.py)
  saved research runs results/research/strategy/<stem>/vanilla_backtest/<newest>/<stem>.pkl -> dv2_g_liq, hpi_g_liq, eom_vote
                     (read with alpha.bench.engine_map.refresh.saved_run, same arithmetic as build.saved_history)
B0 itself: common.run(dev(load()), Params()) -> Fed-funds idle cash, 5 bp cost, one-session lag.
Window: 2008-03-04 (first TAA session, proxy run before 2012-10-02) -> 2022-12-30 (development sample end).
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from common import *  # noqa: E402,F401
from alpha.bench.engine_map.refresh import saved_run, path_of, house_returns, cash_add, dtb3_rate  # noqa: E402

SUB = "a12"
START, END = "2008-03-04", DEV_END
EXACT_START = "2012-10-02"   # real TQQQ/BTAL history
FROZEN = REPO / "results/scout/growth_hunt_20261030/shared/frame_main.parquet"
SAVED = {"dv2_g_liq": "strategy_mr_dv2_vix_gated_liq_bil", "hpi_g_liq": "strategy_mr_hpi_vote_vix_gated_liq_bil",
         "eom_vote": "strategy_taa_month_end_flow_vote"}
NAMES = {"b0": "B0 Beyond 60/40", "core5": "CORE5", "taa3x": "TAA 3x", "taa3x_1n": "TAA 1N 3x", "btal_qqq": "TAA BTAL_QQQ",
         "dv2_g_liq": "MR DV2 (liq)", "hpi_g_liq": "MR HPI (liq)", "mr_capsule": "MR capsule 50/50", "eom_vote": "Flow Vote",
         "ndx_atr_cap": "NDX ATR cap", "ndx_natr_cap": "NDX NATR cap", "dv2_g": "MR DV2 (bil)", "hpi_g": "MR HPI (bil)", "tbill": "T-bill",
         "sixty_forty": "60/40"}
# weights read from portfolios/*.yaml (working tree, 2026-11-05)
BOOKS = {"fund_growth": {"taa3x": .40, "ndx_atr_cap": .15, "ndx_natr_cap": .15, "dv2_g": .15, "hpi_g": .15},
         "fund_growth_plus": {"taa3x_1n": .40, "ndx_atr_cap": .15, "ndx_natr_cap": .15, "dv2_g": .15, "hpi_g": .15},
         "fund_defensive": {"core5": .54, "btal_qqq": .36, "tbill": .10}}
sources = []


def load_pods():
    frame = pd.read_parquet(FROZEN)
    sources.append(f"{FROZEN.relative_to(REPO)}  columns core5, taa3x, taa3x_1n, btal_qqq, ndx_atr_cap, ndx_natr_cap, dv2_g, hpi_g, tbill  {frame.index[0].date()} -> {frame.index[-1].date()}")
    pods = {k: frame[k] for k in ["core5", "taa3x", "taa3x_1n", "btal_qqq", "ndx_atr_cap", "ndx_natr_cap", "dv2_g", "hpi_g", "tbill"]}
    for key, stem in SAVED.items():
        name, state = saved_run(stem, covers=START)
        path = path_of(state["results"])
        pods[key] = house_returns(path) + cash_add(path, dtb3_rate(path.index)).reindex(path.index).fillna(0.0)
        sources.append(f"results/research/strategy/{stem}/vanilla_backtest/{name}/{stem}.pkl  {path.index[0].date()} -> {path.index[-1].date()}  (+ fair-cash add)")
    return pods


def blend(parts: dict, weights: dict, reset: str) -> pd.Series:
    """Fixed mix, weights restored at the first session of each month or year (the Engine Map's convention; books rebalance annually)."""
    frame = pd.DataFrame({k: parts[k] for k in weights}).dropna()
    w = np.array([weights[k] for k in weights], float)
    held, out, last = w.copy(), [], None
    for date, row in zip(frame.index, frame.to_numpy()):
        period = date.month if reset == "month" else date.year
        if last is not None and period != last:
            held = w * held.sum()
        held = held * (1 + row)
        out.append(held.sum())
        last = period
    eq = pd.Series(out, index=frame.index)
    r = eq.pct_change(); r.iloc[0] = eq.iloc[0] - 1
    return r


def stats(r: pd.Series, ff: pd.Series) -> dict:
    r = r.dropna(); ex = r - ff.reindex(r.index).fillna(0.0) / TRADING_DAYS
    eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    roll12 = eq / eq.shift(TRADING_DAYS) - 1
    m = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return {"cagr_%": (eq.iloc[-1] ** (TRADING_DAYS / len(r)) - 1) * 100, "vol_%": r.std() * np.sqrt(TRADING_DAYS) * 100,
            "sharpe_excess": ex.mean() / ex.std() * np.sqrt(TRADING_DAYS), "max_dd_%": dd.min() * 100,
            "worst_12m_%": roll12.min() * 100, "worst_month_%": m.min() * 100, "n_days": len(r)}


def main():
    panel = dev(load())
    b0 = run(panel, Params())["ret"]
    ff = panel["dff"]
    pods = load_pods()
    pods["b0"] = b0
    pods["sixty_forty"] = fixed_mix(panel, {"r_stock": .6, "r_bond": .4}, cash="ff", cost_bps=5.0)["ret"]
    pods["mr_capsule"] = blend(pods, {"dv2_g_liq": .5, "hpi_g_liq": .5}, "month")
    idx = pd.DataFrame({k: pods[k] for k in ["b0", "core5", "taa3x", "taa3x_1n", "btal_qqq", "mr_capsule", "eom_vote"]}).loc[START:END].dropna().index
    print("common window", idx[0].date(), "->", idx[-1].date(), len(idx), "sessions")
    R = pd.DataFrame({k: pods[k].reindex(idx) for k in pods}).fillna(0.0)

    # 1. correlations
    cols = ["b0", "core5", "taa3x", "taa3x_1n", "btal_qqq", "dv2_g_liq", "hpi_g_liq", "mr_capsule", "eom_vote", "sixty_forty"]
    daily = R[cols].corr().rename(index=NAMES, columns=NAMES)
    monthly = R[cols].resample("ME").apply(lambda x: (1 + x).prod() - 1).corr().rename(index=NAMES, columns=NAMES)
    save(daily.round(3), "corr_daily", SUB); save(monthly.round(3), "corr_monthly", SUB)
    print("\nmonthly correlation\n", monthly.round(2).to_string())

    # standalone pods
    solo = pd.DataFrame({NAMES[k]: stats(R[k], ff) for k in cols}).T
    save(solo.round(3), "pods_standalone", SUB)
    print("\nstandalone, same window\n", solo.round(2).to_string())

    # 2. books with B0 added
    rows = {}
    for book, w in BOOKS.items():
        for win_name, s in (("2008", START), ("2012", EXACT_START)):
            base = blend(R.loc[s:], w, "year")
            rows[(book, win_name, "as is")] = stats(base, ff)
            for share in (0.20, 0.33):
                ww = {k: v * (1 - share) for k, v in w.items()}; ww["b0"] = share
                rows[(book, win_name, f"+B0 {int(share*100)}%")] = stats(blend(R.loc[s:], ww, "year"), ff)
        if "core5" in w:  # the replace question, on the defensive book
            for win_name, s in (("2008", START), ("2012", EXACT_START)):
                ww = dict(w); ww["b0"] = ww.pop("core5")
                rows[(book, win_name, "B0 replaces CORE5")] = stats(blend(R.loc[s:], ww, "year"), ff)
                ww = dict(w); half = ww.pop("core5") / 2; ww["core5"] = half; ww["b0"] = half
                rows[(book, win_name, "CORE5 half -> B0")] = stats(blend(R.loc[s:], ww, "year"), ff)
    books = pd.DataFrame(rows).T
    books.index.names = ["book", "from", "variant"]
    save(books.round(3), "books_marginal", SUB)
    print("\nbooks with B0 added\n", books.round(2).to_string())

    # 3. B0 vs CORE5 head to head
    h2h = {"B0": stats(R["b0"], ff), "CORE5": stats(R["core5"], ff),
           "50/50 B0+CORE5 (monthly)": stats(blend(R, {"b0": .5, "core5": .5}, "month"), ff),
           "60/40": stats(R["sixty_forty"], ff)}
    h2h = pd.DataFrame(h2h).T
    h2h["corr_daily_vs_B0"] = [1.0, daily.loc["B0 Beyond 60/40", "CORE5"], np.nan, daily.loc["B0 Beyond 60/40", "60/40"]]
    h2h["corr_monthly_vs_B0"] = [1.0, monthly.loc["B0 Beyond 60/40", "CORE5"], np.nan, monthly.loc["B0 Beyond 60/40", "60/40"]]
    save(h2h.round(3), "b0_vs_core5", SUB)
    print("\nB0 vs CORE5\n", h2h.round(2).to_string())
    # calendar years side by side
    yrs = pd.DataFrame({k: R[k].resample("YE").apply(lambda x: ((1 + x).prod() - 1) * 100) for k in ["b0", "core5", "sixty_forty"]}).rename(columns=NAMES)
    yrs.index = yrs.index.year
    save(yrs.round(1), "b0_core5_by_year", SUB)
    print("\nby year\n", yrs.round(1).to_string())

    # 4. stock-bond correlation stress: months whose month-end 120d SPY-TLT correlation > 0.3
    corr120 = panel["r_stock"].rolling(120).corr(panel["r_bond"]).reindex(idx)
    me = corr120.resample("ME").last()
    mret = R[["b0", "core5", "sixty_forty", "taa3x_1n", "mr_capsule", "eom_vote"]].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mff = (ff.reindex(idx) / TRADING_DAYS).resample("ME").sum()
    # 2008-2022 never shows a month-end 120d correlation above 0.3 (max 0.24, Dec 2022): the > 0.3 cut is reported as empty,
    # the usable cuts are corr > 0, the top quintile of months, and the direct stress: stocks and the long bond both down in the month.
    m_sb = panel[["r_stock", "r_bond"]].reindex(idx).resample("ME").apply(lambda x: (1 + x).prod() - 1)
    both_down = (m_sb["r_stock"] < 0) & (m_sb["r_bond"] < 0)
    q80 = me.quantile(0.8)
    out = {}
    for label, mask in ((f"corr>0.3 ({int((me > 0.3).sum())} months)", me > 0.3), ("corr>0", me > 0), (f"corr top quintile (>{q80:.2f})", me > q80),
                        ("corr bottom 80%", me <= q80), ("stocks and bond both down", both_down), ("all", me.notna())):
        sub = mret[mask.reindex(mret.index).fillna(False)]
        if len(sub) == 0:
            out[label] = pd.DataFrame({"months": 0}, index=mret.columns).rename(index=NAMES); continue
        ex = sub.sub(mff.reindex(sub.index), axis=0)
        out[label] = pd.DataFrame({"months": len(sub), "mean_month_%": sub.mean() * 100, "ann_return_%": ((1 + sub).prod() ** (12 / len(sub)) - 1) * 100,
                                   "ann_excess_sharpe": ex.mean() / ex.std() * np.sqrt(12), "hit_rate_%": (sub > 0).mean() * 100,
                                   "worst_month_%": sub.min() * 100, "total_%": ((1 + sub).prod() - 1) * 100}).rename(index=NAMES)
    stress = pd.concat(out, names=["regime", "strategy"])
    save(stress.round(3), "stress_stock_bond_corr", SUB)
    print("\nstock-bond correlation regimes (month-end 120d corr)\n", stress.round(2).to_string())
    print("month-end 120d corr: max", round(me.max(), 2), "at", me.idxmax().date(), "; months > 0:", sorted(str(d.date()) for d in me[me > 0].index))
    print("both-down months:", sorted(str(d.date()) for d in both_down[both_down].index))
    (OUT / SUB / "sources.txt").write_text("\n".join(sources) + f"\ncommon window {idx[0].date()} -> {idx[-1].date()} ({len(idx)} sessions)\n", encoding="utf-8")
    print("\nsources:\n" + "\n".join(sources))
    t1_pass(panel, ff, R, idx, both_down)


def t1_pass(panel, ff, R, idx, both_down):
    """Follow-up: the trend-tilted candidate T1 (A13 series) through the same head-to-head, stress and book tables, files t1_*.csv."""
    a13 = OUT / "a13" / "series_daily.parquet"
    t1 = pd.read_parquet(a13)["B0+trend(sma10,k=.5,M)"]
    sources.append(f"{a13.relative_to(REPO)}  column B0+trend(sma10,k=.5,M)  {t1.index[0].date()} -> {t1.index[-1].date()}")
    R = R.copy(); R["t1"] = t1.reindex(idx).fillna(0.0)
    cols = ["t1", "b0", "core5", "taa3x", "taa3x_1n", "btal_qqq", "mr_capsule", "eom_vote", "sixty_forty"]
    names = dict(NAMES, t1="T1 trend-tilted")
    cd = R[cols].corr(); cm = R[cols].resample("ME").apply(lambda x: (1 + x).prod() - 1).corr()
    corr = pd.DataFrame({"corr_daily_vs_T1": cd["t1"], "corr_monthly_vs_T1": cm["t1"]}).rename(index=names)
    save(corr.round(3), "t1_corr", SUB); print("\nT1 correlations\n", corr.round(2).to_string())
    h2h = {"T1": stats(R["t1"], ff), "B0": stats(R["b0"], ff), "CORE5": stats(R["core5"], ff),
           "50/50 T1+CORE5 (monthly)": stats(blend(R, {"t1": .5, "core5": .5}, "month"), ff),
           "T1 2012->": stats(R["t1"].loc[EXACT_START:], ff), "CORE5 2012->": stats(R["core5"].loc[EXACT_START:], ff),
           "50/50 T1+CORE5 2012->": stats(blend(R.loc[EXACT_START:], {"t1": .5, "core5": .5}, "month"), ff)}
    h2h = pd.DataFrame(h2h).T
    save(h2h.round(3), "t1_vs_core5", SUB); print("\nT1 vs CORE5\n", h2h.round(2).to_string())
    yrs = pd.DataFrame({k: R[k].resample("YE").apply(lambda x: ((1 + x).prod() - 1) * 100) for k in ["t1", "b0", "core5", "sixty_forty"]}).rename(columns=names)
    yrs.index = yrs.index.year
    save(yrs.round(1), "t1_by_year", SUB); print("\nby year\n", yrs.round(1).to_string())
    mret = R[["t1", "b0", "core5", "sixty_forty"]].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mff = (ff.reindex(idx) / TRADING_DAYS).resample("ME").sum()
    out = {}
    for label, mask in (("stocks and bond both down", both_down), ("other months", ~both_down), ("all", both_down.notna())):
        sub = mret[mask.reindex(mret.index).fillna(False)]; ex = sub.sub(mff.reindex(sub.index), axis=0)
        out[label] = pd.DataFrame({"months": len(sub), "mean_month_%": sub.mean() * 100, "ann_excess_sharpe": ex.mean() / ex.std() * np.sqrt(12),
                                   "hit_rate_%": (sub > 0).mean() * 100, "worst_month_%": sub.min() * 100, "total_%": ((1 + sub).prod() - 1) * 100}).rename(index=names)
    stress = pd.concat(out, names=["regime", "strategy"])
    save(stress.round(3), "t1_stress_both_down", SUB); print("\nT1 stress\n", stress.round(2).to_string())
    rows = {}
    for book, w in BOOKS.items():
        for win_name, s in (("2008", START), ("2012", EXACT_START)):
            rows[(book, win_name, "as is")] = stats(blend(R.loc[s:], w, "year"), ff)
            for share in (0.20, 0.33):
                ww = {k: v * (1 - share) for k, v in w.items()}; ww["t1"] = share
                rows[(book, win_name, f"+T1 {int(share*100)}%")] = stats(blend(R.loc[s:], ww, "year"), ff)
            if "core5" in w:
                ww = dict(w); ww["t1"] = ww.pop("core5")
                rows[(book, win_name, "T1 replaces CORE5")] = stats(blend(R.loc[s:], ww, "year"), ff)
                ww = dict(w); half = ww.pop("core5") / 2; ww["core5"] = half; ww["t1"] = half
                rows[(book, win_name, "CORE5 half -> T1")] = stats(blend(R.loc[s:], ww, "year"), ff)
    books = pd.DataFrame(rows).T; books.index.names = ["book", "from", "variant"]
    save(books.round(3), "t1_books_marginal", SUB); print("\nbooks with T1\n", books.round(2).to_string())
    (OUT / SUB / "sources.txt").write_text("\n".join(sources) + f"\ncommon window {idx[0].date()} -> {idx[-1].date()} ({len(idx)} sessions)\n", encoding="utf-8")


if __name__ == "__main__":
    main()
