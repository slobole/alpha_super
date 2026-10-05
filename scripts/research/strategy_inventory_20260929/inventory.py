"""Strategy inventory after the 22-29 Sep 2026 fixes (descriptive; nothing is selected here).

Reads the fresh sleeve runs of shelf_rebuild_20260929 (27 runs at HEAD f9ad358, $1M, to 2026-08-19) and reports,
per strategy and window, return, risk, tail and exposure statistics, plus factor loadings and correlations.

Fair cash (owner request 2026-09-29): the headline numbers credit each sleeve's idle cash like a real account would,
positive cash at max(DTB3 - 0.5%, 0) and negative cash at DTB3 + 1.5% (shelf_rebuild lib.cash_realism_add, prior
close cash and prior DTB3 observation). The house convention (idle cash 0%) is shown beside it. T-bills = BIL TR.

Factor model (weekly, Friday closes, EXACT window 2012-10-02 -> 2026-08-19):
    r_s - r_f = alpha + b_eq (SPXTR - r_f) + b_bd (IEF - r_f) + b_au (GLD - r_f) + b_usd (UUP - r_f) + e
Total-return series from Norgate; r_f = BIL. Alpha annualised (x 52); t-statistics with Newey-West (4 lags).

Usage: python inventory.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SHELF = REPO / "scripts" / "research" / "shelf_rebuild_20260929"
sys.path.insert(0, str(SHELF))

import lib  # noqa: E402  (shelf rebuild: inputs, cash realism, metrics)
from lib import END, EXACT_START, LONG_START, TBILL  # noqa: E402

OUT = REPO / "results" / "research" / "portfolio" / "strategy_inventory_20260929"
RECENT_START = lib.BLOCK_DICT["RECENT"][0]

INFO = {
    # alias: (display name, family, what it does, instruments, cadence)
    "core5": ("CORE5", "Defensive TAA", "Five fixed 20% sleeves (SPY, IEF, GLD, DBC, UUP); each holds its ETF when its trend is up, else BIL; small DBC short overlay", "ETFs", "monthly"),
    "btal_qqq": ("BTAL_QQQ", "Defensive TAA", "Defense First 1/N with BTAL: equal-weight momentum among defensive assets, QQQ as the risk-on fallback, cash on VIX stress", "ETFs incl. BTAL", "monthly"),
    "taa_lin_qqq": ("TAA 1x QQQ (no BTAL)", "Defensive TAA", "BTAL_QQQ's twin without BTAL", "ETFs", "monthly"),
    "tactical_fi": ("Tactical FI", "Defensive TAA", "IEF when the yield curve is steep, LQD when credit spreads are wide (vs expanding medians), else BIL", "IEF/LQD/BIL", "monthly"),
    "trinity": ("Trinity 8% vol", "Defensive TAA", "Inverse-vol VTI/GLD/TLT scaled to 8% volatility, rest in BIL", "VTI/GLD/TLT/BIL", "daily band + monthly"),
    "eom_flow": ("EOM flow", "Calendar", "Trades the month-end 60/40 rebalancing flow in SPY/TLT around the last sessions of the month (MOC, TLT short)", "SPY/TLT", "month-end MOC"),
    "downshock": ("DOWNSHOCK", "ETF mean reversion", "Buys sector ETFs after a sharp IBS down-shock, exits on recovery", "11 sector ETFs", "daily"),
    "disp": ("DISP", "ETF mean reversion", "Buys the most oversold of five industry ETFs (IBS dispersion) when above its SMA200", "SOXX/IGV/IBB/KIE/IHI", "daily"),
    "disp_xlc": ("DISP (XLC)", "ETF mean reversion", "Dispersion variant with XLC (from 2018)", "sector ETFs", "daily"),
    "disp_xlc_sma": ("DISP (XLC, SMA200)", "ETF mean reversion", "Dispersion variant with XLC and SMA200 gate (from 2019)", "sector ETFs", "daily"),
    "etf_dv2": ("DV2 industry ETFs", "ETF mean reversion", "DV2 dip-buying on 19 liquid industry ETFs (shadow)", "industry ETFs", "daily"),
    "taa3x": ("TAA 3x (live)", "Growth TAA", "Defense First rank-weighted with BTAL, TQQQ as the risk-on fallback", "ETFs incl. TQQQ", "monthly"),
    "taa3x_1n": ("TAA 3x 1/N", "Growth TAA", "Same signals, equal weights, TQQQ fallback", "ETFs incl. TQQQ", "monthly"),
    "taa2x_1n": ("TAA 2x 1/N", "Growth TAA", "Same signals, equal weights, QLD fallback", "ETFs incl. QLD", "monthly"),
    "taa_1n_qld": ("TAA 2x 1/N (no BTAL)", "Growth TAA", "Defense First 1/N without BTAL, QLD fallback", "ETFs incl. QLD", "monthly"),
    "taa_1n_sso": ("TAA 2x SPX (no BTAL)", "Growth TAA", "Defense First 1/N without BTAL, SSO fallback", "ETFs incl. SSO", "monthly"),
    "ndx_vxn": ("NDX-VXN (live)", "Equity momentum", "Top 10 Nasdaq-100 stocks by ROC12 / dollar ATR20, stock and SPY trend filters, exposure scaled by VXN", "~10 stocks", "monthly"),
    "ndx_atr": ("NDX-ATR", "Equity momentum", "Same ranking and filters without the VXN scaling", "~10 stocks", "monthly"),
    "ndx_natr20": ("NDX-NATR20", "Equity momentum", "Scale-free ranking ROC12 / (ATR20 / price), VXN scaling (shadow)", "~10 stocks", "monthly"),
    "compass": ("Inflation Compass", "Macro rotation", "Rotates sector ETFs by growth (SPY vs SMA200) and inflation (T5YIE) regime", "sector ETFs", "monthly"),
    "compass_qqq": ("Compass QQQ", "Macro rotation", "Compass with QQQ instead of XLK in the goldilocks regime", "sector ETFs + QQQ", "monthly"),
    "dv2": ("DV2", "Stock mean reversion", "Buys S&P 500 dips (DV2 < 10, above SMA200, 6-month momentum), exits on strength", "~10 stocks", "daily"),
    "dv2_adv": ("DV2 floor + ADV", "Stock mean reversion", "DV2 on the more liquid half of the S&P 500, ranked by dollar volume (shadow)", "~10 stocks", "daily"),
    "dv2_floor": ("DV2 floor", "Stock mean reversion", "DV2 on the more liquid half of the S&P 500 (shadow)", "~10 stocks", "daily"),
    "hpi_vote": ("HPI vote", "Stock mean reversion", "S&P 500 high-probability IBS/RSI dip entries, 2/3/5-day vote", "~10 stocks", "daily"),
    "hpi_ibs_rsi": ("HPI IBS/RSI", "Stock mean reversion", "S&P 500 high-probability dip entries, IBS/RSI exit", "~10 stocks", "daily"),
    "tactical_fi_frozen": ("Tactical FI (frozen FRED)", "Defensive TAA", "Governed current-vintage FRED mode (sensitivity only)", "IEF/LQD/BIL", "monthly"),
}


def window_stats(r: pd.Series, r_house: pd.Series, data: dict, invested: pd.Series) -> dict:
    index_all = data["index"]
    tb = data["sleeve"][TBILL]
    base = lib.base_date(index_all, r)
    nav = (1 + r).cumprod()
    daily_cut = r.quantile(0.05)
    roll21 = (1 + r).rolling(21).apply(np.prod, raw=True) - 1
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    cagr = lib.cagr(r, base)
    maxdd = lib.maxdd(r)
    return {"cagr": cagr, "cagr_house_cash": lib.cagr(r_house, base), "vol": float(r.std() * np.sqrt(252)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)), "maxdd": maxdd, "calmar": cagr / abs(maxdd),
            "cvar5_daily": float(r[r <= daily_cut].mean()), "cvar5_21d": float(roll21[roll21 <= roll21.quantile(0.05)].mean()),
            "worst_year": float(yearly.min()), "excess_cagr": cagr - lib.cagr(tb.reindex(r.index), base),
            "invested_share": float(invested.reindex(r.index).mean()),
            "crisis_corr": lib.crisis_corr(r, data["bench"]["SPXTR"]), "dd_trough": (nav / nav.cummax() - 1).idxmin().date().isoformat()}


def newey_west_t(X: np.ndarray, y: np.ndarray, beta: np.ndarray, lags: int = 4) -> np.ndarray:
    resid = y - X @ beta
    n = len(y)
    xtx_inv = np.linalg.inv(X.T @ X)
    s = (X * resid[:, None]).T @ (X * resid[:, None])
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = (X[lag:] * resid[lag:, None]).T @ (X[:-lag] * resid[:-lag, None])
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv * n / (n - X.shape[1])
    return beta / np.sqrt(np.diag(cov))


def factor_table(frame: pd.DataFrame, aliases: list[str]) -> pd.DataFrame:
    tr = {s: lib.common.load_total_return_close_ser(s, "2012-01-01", END.strftime("%Y-%m-%d")) for s in ("IEF", "GLD", "UUP")}
    idx = frame.loc[EXACT_START:END].index
    daily = pd.DataFrame({"EQ": frame["_SPXTR"], "BD": tr["IEF"].reindex(frame.index).pct_change(fill_method=None),
                          "AU": tr["GLD"].reindex(frame.index).pct_change(fill_method=None),
                          "USD": tr["UUP"].reindex(frame.index).pct_change(fill_method=None), "RF": frame[TBILL]}).loc[idx]
    # *** CRITICAL*** weekly compounding uses only returns inside each Mon-Fri week; no forward alignment.
    weekly_f = (1 + daily).resample("W-FRI").prod() - 1
    rows = {}
    for alias in aliases:
        s = frame[alias].loc[idx]
        if s.isna().any():
            continue
        ws = (1 + s).resample("W-FRI").prod() - 1
        y = (ws - weekly_f["RF"]).to_numpy()
        X = np.column_stack([np.ones(len(ws))] + [(weekly_f[c] - weekly_f["RF"]).to_numpy() for c in ("EQ", "BD", "AU", "USD")])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        t = newey_west_t(X, y, beta)
        fitted = X @ beta
        r2 = 1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
        rows[alias] = {"alpha_ann": float(beta[0] * 52), "alpha_t": float(t[0]), "b_equity": float(beta[1]),
                       "b_bonds": float(beta[2]), "b_gold": float(beta[3]), "b_dollar": float(beta[4]),
                       "t_equity": float(t[1]), "r2": float(r2)}
    return pd.DataFrame(rows).T


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = lib.load_inputs()
    meta = data["meta"]
    aliases = [a for a in INFO if a in meta and a != "tactical_fi_frozen"]
    fair_long, fair_exact = data["cash_long"], data["cash_exact"]
    rows = []
    for alias in aliases:
        path = lib.read_path(lib.SOURCE, alias)
        invested = (path["portfolio_value_float"].abs() / path["total_value_float"]).clip(upper=1.5)
        row = {"alias": alias, "name": INFO[alias][0], "family": INFO[alias][1], "what": INFO[alias][2],
               "instruments": INFO[alias][3], "cadence": INFO[alias][4], "tier": meta[alias]["tier_str"],
               "first_invested": meta[alias]["first_invested_date_str"],
               "trade_days_per_year": len(lib.trade_dates(data, alias)) / ((END - EXACT_START).days / 365.25),
               "proxy_before_2012": alias in lib.PROXY_ALIAS_LIST, "filled_before_2010": alias == "etf_dv2"}
        for label, frame, house, start in (("long", fair_long, data["long"], LONG_START),
                                           ("exact", fair_exact, data["sleeve"], EXACT_START),
                                           ("recent", fair_exact, data["sleeve"], RECENT_START)):
            r = frame[alias].loc[start:END]
            if r.isna().any() or len(r) == 0:
                continue
            stats = window_stats(r, house[alias].loc[start:END], data, invested)
            row.update({f"{label}_{k}": v for k, v in stats.items()})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("alias")

    factor_frame = fair_exact.copy()
    factor_frame["_SPXTR"] = data["bench"]["SPXTR"]
    factors = factor_table(factor_frame, aliases)
    table = table.join(factors)

    bench_rows = []
    for name, column in (("S&P 500 TR", "SPXTR"), ("60/40 SPY/AGG", "SIXTY_FORTY"), ("T-bills (BIL)", "BIL")):
        row = {"alias": name, "name": name, "family": "Benchmark"}
        for label, start in (("long", LONG_START), ("exact", EXACT_START), ("recent", RECENT_START)):
            r = data["bench"][column].loc[start:END]
            stats = window_stats(r, r, data, pd.Series(1.0, index=r.index))
            row.update({f"{label}_{k}": v for k, v in stats.items()})
        bench_rows.append(row)
    bench = pd.DataFrame(bench_rows).set_index("alias")

    corr_daily = fair_exact.loc[EXACT_START:END, [a for a in aliases if fair_exact[a].loc[EXACT_START:END].notna().all()]].corr()
    monthly = (1 + fair_exact.loc[EXACT_START:END]).resample("ME").prod() - 1
    corr_monthly = monthly[corr_daily.columns].corr()
    spx = data["bench"]["SPXTR"].loc[EXACT_START:END]
    stress_days = spx[spx <= spx.quantile(0.05)].index
    corr_stress = fair_exact.loc[stress_days, corr_daily.columns].corr()

    table.to_csv(OUT / "inventory.csv", float_format="%.6g")
    bench.to_csv(OUT / "benchmarks.csv", float_format="%.6g")
    corr_daily.to_csv(OUT / "corr_daily_exact.csv", float_format="%.4f")
    corr_monthly.to_csv(OUT / "corr_monthly_exact.csv", float_format="%.4f")
    corr_stress.to_csv(OUT / "corr_stress_exact.csv", float_format="%.4f")

    def clean(v):
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else round(float(v), 6)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    payload = {"strategies": [{k: clean(v) for k, v in r.items()} for r in table.reset_index().to_dict(orient="records")],
               "benchmarks": [{k: clean(v) for k, v in r.items()} for r in bench.reset_index().to_dict(orient="records")],
               "corr_monthly": {"labels": list(corr_monthly.columns),
                                "values": [[clean(x) for x in row] for row in corr_monthly.to_numpy()]},
               "corr_stress": {"labels": list(corr_stress.columns),
                               "values": [[clean(x) for x in row] for row in corr_stress.to_numpy()]},
               "meta": {"end": END.date().isoformat(), "long_start": LONG_START.date().isoformat(),
                        "exact_start": EXACT_START.date().isoformat(), "recent_start": RECENT_START.date().isoformat()}}
    (OUT / "inventory.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    show = ["exact_cagr", "exact_cagr_house_cash", "exact_vol", "exact_sharpe", "exact_maxdd", "exact_cvar5_21d",
            "long_cagr", "long_sharpe", "long_maxdd", "recent_excess_cagr", "exact_invested_share", "b_equity",
            "b_bonds", "b_gold", "alpha_ann", "alpha_t", "r2"]
    print(table[show].round(3).to_string())
    print(bench[["long_cagr", "long_sharpe", "long_maxdd", "exact_cagr", "exact_sharpe", "exact_maxdd"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
