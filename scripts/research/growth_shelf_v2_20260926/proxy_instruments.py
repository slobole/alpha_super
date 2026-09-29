"""Part A1 + A2 of SPEC_FROZEN.md: synthetic leveraged ETFs and a synthetic BTAL, with their validation.

A1  r_syn(t) = L * r_und(t) - (L - 1) * f(t) - d / 252        (TQQQ / QLD from QQQ, SPXL from SPY)
A2  anti-beta long/short on Russell 1000 members: long the lowest-beta fifth, short the highest-beta fifth
    (within GICS sectors for V1 / V3), equal dollars per name, monthly, drifting between rebalances,
    r(t) = [sum_long w r - sum_short w r] + f(t) - d / 252

f(t) is the lagged DTB3 accrual (prior observation, calendar days / 360). d is calibrated only where the real fund
exists (A1: 2010-02-11 -> 2026-08-19; A2: 2011-09-13 -> 2026-08-19). Outputs go to
results/research/portfolio/growth_shelf_v2_20260926/proxy/. Research only; nothing live or in strategies/ changes.

Usage: python proxy_instruments.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (REPO, REPO / "scripts" / "research" / "fund_menu_20260923"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common  # noqa: E402  (fund menu study: lagged T-bill accrual)
from data.norgate_loader import (  # noqa: E402
    CAPITALSPECIAL_ADJUSTMENT_STR as CS,
    TOTALRETURN_ADJUSTMENT_STR as TR,
    load_price_timeseries,
)

OUT_DIR = REPO / "results" / "research" / "portfolio" / "growth_shelf_v2_20260926" / "proxy"
R1000_DIR = REPO / "results" / "research" / "mr_beyond_dv2_20260926" / "cache" / "r1000"
CAL_START, CAL_END = pd.Timestamp("2010-02-11"), pd.Timestamp("2026-08-19")
BTAL_START = pd.Timestamp("2011-09-13")
GFC = (pd.Timestamp("2008-05-19"), pd.Timestamp("2009-03-09"))
LEVERED_DICT = {"TQQQ": ("QQQ", 3.0), "QLD": ("QQQ", 2.0), "SPXL": ("SPY", 3.0)}
VARIANT_DICT = {"V1": {"sector": True, "window": 252, "min_obs": 200},
                "V2": {"sector": False, "window": 252, "min_obs": 200},
                "V3": {"sector": True, "window": 126, "min_obs": 100}}  # min_obs for V3: amendment A1
QUINTILE = 0.20
MIN_PRICE = 5.0


def bars(symbol_str: str, adjustment_str: str, start_str: str = "1990-01-01") -> pd.DataFrame:
    frame = load_price_timeseries(symbol_str, adjustment_str=adjustment_str, start_date_str=start_str)
    frame.index = pd.to_datetime(frame.index).normalize()
    return frame


def tbill_accrual(index: pd.DatetimeIndex) -> pd.Series:
    return common.load_tbill_return_ser(pd.DatetimeIndex(index))


def cum_return(r: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Compounded return from the close of `start` to the close of `end`."""
    window = r.loc[(r.index > start) & (r.index <= end)]
    return float((1.0 + window.fillna(0.0)).prod() - 1.0)


def calibrate_drag(gross: pd.Series, real_tr_close: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Constant annual drag d so that prod(1 + gross - d/252) equals the real fund's growth over (start, end]."""
    real = real_tr_close.loc[start:end].dropna()
    window = gross.reindex(real.index).iloc[1:].fillna(0.0).to_numpy()
    target = float(np.log(real.iloc[-1] / real.iloc[0]))
    return float(brentq(lambda d: np.log1p(window - d / 252.0).sum() - target, -0.5, 0.5))


def yearly(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).groupby(r.index.year).prod() - 1.0


# ─── A1: leveraged ETFs ──────────────────────────────────────────────────────


def levered_gross(und_close: pd.Series, lev: float, f: pd.Series) -> pd.Series:
    """L * r_und - (L - 1) * f, before the calibrated drag."""
    return lev * und_close.pct_change(fill_method=None) - (lev - 1.0) * f.reindex(und_close.index).fillna(0.0)


def levered_bars(und: pd.DataFrame, lev: float, r_syn: pd.Series) -> pd.DataFrame:
    """OHLC bars for the synthetic fund: intraday prices map through L times the underlying's move from its
    prior close; the close follows r_syn (financing and drag accrue by the close)."""
    r = r_syn.reindex(und.index).fillna(0.0)
    close = 100.0 * (1.0 + r).cumprod()
    prev_close = close.shift(1)
    und_prev = und["Close"].shift(1)
    out = pd.DataFrame(index=und.index)
    for field in ("Open", "High", "Low"):
        out[field] = prev_close * (1.0 + lev * (und[field] / und_prev - 1.0))
    out["Close"] = close
    out.iloc[0, :3] = close.iloc[0]
    out["High"] = out[["Open", "High", "Close"]].max(axis=1)
    out["Low"] = out[["Open", "Low", "Close"]].min(axis=1)
    out["Volume"] = und["Volume"]
    out["Turnover"] = out["Close"] * out["Volume"]
    out["Unadjusted Close"] = out["Close"]
    out["Dividend"] = 0.0
    return out


def build_levered() -> dict:
    report = {}
    und_cs = {s: bars(s, CS) for s in ("QQQ", "SPY")}
    for fund, (und_symbol, lev) in LEVERED_DICT.items():
        und = und_cs[und_symbol]
        f = tbill_accrual(und.index)
        gross = levered_gross(und["Close"], lev, f)
        real_tr = bars(fund, TR)["Close"]
        drag = calibrate_drag(gross, real_tr, max(CAL_START, real_tr.index[0]), CAL_END)
        r_syn = gross - drag / 252.0
        r_real = real_tr.pct_change(fill_method=None)
        fit = pd.concat([r_syn, r_real], axis=1, keys=["syn", "real"]).loc[CAL_START:CAL_END].dropna().iloc[1:]
        year_diff = (yearly(fit["syn"]) - yearly(fit["real"])).loc[2011:2025]
        entry = {"underlying": und_symbol, "leverage": lev, "drag_per_year": drag,
                 "fit_daily_corr": float(fit.corr().iloc[0, 1]),
                 "fit_median_abs_year_diff": float(year_diff.abs().median()),
                 "fit_year_diff": {int(k): float(v) for k, v in year_diff.items()}}
        first_real = real_tr.index[0]
        if first_real < CAL_START:  # out-of-sample crisis test (QLD, SPXL)
            oos = pd.concat([r_syn, r_real], axis=1, keys=["syn", "real"]).loc[first_real:CAL_START - pd.Timedelta(days=1)].dropna().iloc[1:]
            entry["oos_start"] = str(first_real.date())
            entry["oos_daily_corr"] = float(oos.corr().iloc[0, 1])
            entry["oos_cum_syn"] = float((1 + oos["syn"]).prod() - 1)
            entry["oos_cum_real"] = float((1 + oos["real"]).prod() - 1)
            if first_real < GFC[0]:
                entry["gfc_syn"] = cum_return(r_syn, *GFC)
                entry["gfc_real"] = cum_return(r_real, *GFC)
        report[fund] = entry
        if fund == "TQQQ":
            levered_bars(und, lev, r_syn).to_csv(OUT_DIR / "synthetic_TQQQ_bars.csv.gz", float_format="%.8g")
            r_syn.rename("r").to_csv(OUT_DIR / "synthetic_TQQQ_returns.csv.gz", float_format="%.10g")
    return report


# ─── A2: anti-beta long/short (synthetic BTAL) ───────────────────────────────


def load_r1000() -> dict:
    arr = {k: np.load(R1000_DIR / f"{k}.npy", mmap_mode="r") for k in ("Close", "Open", "Dividend", "Unadjusted Close", "member")}
    dates = pd.DatetimeIndex(np.load(R1000_DIR / "dates.npy"))
    symbols = np.load(R1000_DIR / "symbols.npy").astype(str)
    start = int(np.searchsorted(dates, np.datetime64("1994-01-01")))
    close = np.asarray(arr["Close"][start - 1:], dtype=np.float64)
    div = np.nan_to_num(np.asarray(arr["Dividend"][start - 1:], dtype=np.float64))
    opn = np.asarray(arr["Open"][start - 1:], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        # *** CRITICAL*** Norgate stamps the dividend on the last cum session t-1; the price drop is on t.
        tot = (close[1:] + div[:-1]) / close[:-1] - 1.0
        overnight = (opn[1:] + div[:-1]) / close[:-1] - 1.0
    return {"dates": dates[start:], "symbols": symbols, "ret": tot, "overnight": overnight,
            "unadj": np.asarray(arr["Unadjusted Close"][start:], dtype=np.float64),
            "member": np.asarray(arr["member"][start:]).astype(bool)}


def sector_labels(symbols: np.ndarray) -> np.ndarray:
    cache_path = OUT_DIR / "gics_sector_by_symbol.json"
    if cache_path.exists():
        label_dict = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        import norgatedata
        label_dict = {}
        for symbol in symbols:
            try:
                value = norgatedata.classification_at_level(str(symbol), "GICS", "ClassificationId", 1)
            except Exception:  # noqa: BLE001 - unclassified names form their own bucket
                value = None
            label_dict[str(symbol)] = str(value) if value else "UNKNOWN"
        cache_path.write_text(json.dumps(label_dict, indent=0), encoding="utf-8")
    return np.array([label_dict.get(str(s), "UNKNOWN") for s in symbols])


def betas_at(ret: np.ndarray, mkt: np.ndarray, row: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Beta of every column on the market over rows (row - window, row], pairwise-valid days only.
    *** CRITICAL*** the window ends at the decision close `row`; nothing after it is read."""
    x = ret[row - window + 1: row + 1]
    y = mkt[row - window + 1: row + 1]
    valid = np.isfinite(x) & np.isfinite(y)[:, None]
    n = valid.sum(axis=0)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = xv.sum(0) / n, yv.sum(0) / n
        cov = (xv * yv).sum(0) / n - mx * my
        var = (yv * yv).sum(0) / n - my * my
        beta = cov / var
    return beta, n


def select_legs(beta: np.ndarray, eligible: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    long_list, short_list = [], []
    for group in np.unique(groups[eligible]):
        idx = np.nonzero(eligible & (groups == group))[0]
        if len(idx) < 5:
            continue
        k = max(1, int(round(QUINTILE * len(idx))))
        order = idx[np.argsort(beta[idx], kind="mergesort")]
        long_list.append(order[:k])
        short_list.append(order[-k:])
    return np.concatenate(long_list), np.concatenate(short_list)


def anti_beta_returns(data: dict, mkt_ret: np.ndarray, sectors: np.ndarray, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily gross long/short return (before financing and drag) and the overnight part, plus the holdings log."""
    dates, ret, overnight = data["dates"], data["ret"], data["overnight"]
    groups = sectors if spec["sector"] else np.full(len(sectors), "ALL")
    month = dates.to_period("M")
    decision_rows = np.nonzero(np.r_[month[1:] != month[:-1], True])[0]
    decision_rows = decision_rows[decision_rows >= spec["window"] + 5]
    gross = np.full(len(dates), np.nan)
    gross_on = np.full(len(dates), np.nan)
    log_rows = []
    for k, m in enumerate(decision_rows[:-1]):
        beta, n_obs = betas_at(ret, mkt_ret, m, spec["window"])
        eligible = data["member"][m] & (data["unadj"][m] > MIN_PRICE) & (n_obs >= spec["min_obs"]) & np.isfinite(beta)
        if eligible.sum() < 50:
            continue
        long_idx, short_idx = select_legs(beta, eligible, groups)
        nxt = decision_rows[k + 1]
        seg = slice(m + 1, nxt + 1)
        # *** CRITICAL*** buy-and-hold from close m: a name without a return that day keeps its value.
        r_long = np.nan_to_num(ret[seg][:, long_idx])
        r_short = np.nan_to_num(ret[seg][:, short_idx])
        g_long = np.vstack([np.ones(len(long_idx)), np.cumprod(1.0 + r_long, axis=0)])
        g_short = np.vstack([np.ones(len(short_idx)), np.cumprod(1.0 + r_short, axis=0)])
        v_long, v_short = g_long.mean(axis=1), g_short.mean(axis=1)
        nav = 1.0 + v_long - v_short
        gross[seg] = nav[1:] / nav[:-1] - 1.0
        on_long = (g_long[:-1] * np.nan_to_num(overnight[seg][:, long_idx])).mean(axis=1)
        on_short = (g_short[:-1] * np.nan_to_num(overnight[seg][:, short_idx])).mean(axis=1)
        gross_on[seg] = (on_long - on_short) / nav[:-1]
        log_rows.append({"decision": dates[m], "eligible": int(eligible.sum()), "long": len(long_idx), "short": len(short_idx),
                         "beta_long": float(np.mean(beta[long_idx])), "beta_short": float(np.mean(beta[short_idx]))})
    frame = pd.DataFrame({"gross": gross, "gross_overnight": gross_on}, index=dates)
    return frame, pd.DataFrame(log_rows)


def btal_bars(r_syn: pd.Series, r_on: pd.Series) -> pd.DataFrame:
    r = r_syn.fillna(0.0)
    close = 100.0 * (1.0 + r).cumprod()
    out = pd.DataFrame(index=r.index)
    out["Open"] = close.shift(1) * (1.0 + r_on.reindex(r.index).fillna(0.0))
    out.iloc[0, 0] = close.iloc[0]
    out["Close"] = close
    out["High"] = out[["Open", "Close"]].max(axis=1)
    out["Low"] = out[["Open", "Close"]].min(axis=1)
    out["Volume"] = 1.0e6
    out["Turnover"] = out["Close"] * out["Volume"]
    out["Unadjusted Close"] = out["Close"]
    out["Dividend"] = 0.0
    return out[["Open", "High", "Low", "Close", "Volume", "Turnover", "Unadjusted Close", "Dividend"]]


def build_btal() -> dict:
    data = load_r1000()
    spy_tr = bars("SPY", TR)["Close"].reindex(data["dates"])
    mkt = spy_tr.pct_change(fill_method=None).to_numpy()
    sectors = sector_labels(data["symbols"])
    f = tbill_accrual(data["dates"])
    btal_tr = bars("BTAL", TR)["Close"]
    r_btal = btal_tr.pct_change(fill_method=None)
    spy_r = spy_tr.pct_change(fill_method=None)
    report, gross_by_variant = {"variants": {}}, {}
    for name, spec in VARIANT_DICT.items():
        frame, log_df = anti_beta_returns(data, mkt, sectors, spec)
        log_df.to_csv(OUT_DIR / f"btal_{name}_holdings_log.csv", index=False)
        gross_by_variant[name] = frame
        r0 = frame["gross"] + f
        both = pd.concat([r0, r_btal], axis=1, keys=["syn", "real"]).loc[BTAL_START:CAL_END].dropna().iloc[1:]
        monthly = (1 + both).resample("ME").prod() - 1
        monthly = monthly.loc["2011-10":]
        report["variants"][name] = {"daily_corr": float(both.corr().iloc[0, 1]), "monthly_corr": float(monthly.corr().iloc[0, 1]),
                                    "median_names_per_leg": float(log_df["long"].median())}
    best = max(report["variants"], key=lambda v: report["variants"][v]["monthly_corr"])
    if report["variants"][best]["monthly_corr"] - report["variants"]["V1"]["monthly_corr"] < 0.01:
        best = "V1"  # pre-declared tie rule
    frame = gross_by_variant[best]
    report["chosen"] = best
    # Amendment A3: exposure scale k from the overlap only (OLS slope of BTAL's excess return on the replica).
    fit = pd.concat([r_btal - f, frame["gross"]], axis=1, keys=["real_excess", "gross"]).loc[BTAL_START:CAL_END].dropna().iloc[1:]
    scale = float(np.cov(fit["real_excess"], fit["gross"])[0, 1] / fit["gross"].var())
    report["scale_k"] = scale
    series = {}
    for label, k in (("scaled", scale), ("unscaled", 1.0)):
        gross = k * frame["gross"] + f
        drag = calibrate_drag(gross, btal_tr, BTAL_START, CAL_END)
        r_syn = (gross - drag / 252.0).rename("r")
        r_on = (k * frame["gross_overnight"]).rename("r_overnight")
        series[label] = (r_syn, r_on)
        both = pd.concat([r_syn, r_btal, spy_r], axis=1, keys=["syn", "real", "spy"]).loc[BTAL_START:CAL_END].dropna().iloc[1:]
        report[label] = {"k": k, "drag_per_year": drag,
                         "beta_syn": float(np.cov(both["syn"], both["spy"])[0, 1] / both["spy"].var()),
                         "beta_real": float(np.cov(both["real"], both["spy"])[0, 1] / both["spy"].var()),
                         "vol_syn": float(both["syn"].std() * np.sqrt(252)), "vol_real": float(both["real"].std() * np.sqrt(252)),
                         "year_syn": {int(y): float(v) for y, v in yearly(both["syn"]).items()},
                         "year_real": {int(y): float(v) for y, v in yearly(both["real"]).items()},
                         "covid_syn": cum_return(r_syn, pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")),
                         "covid_real": cum_return(r_btal, pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")),
                         "bear2022_syn": cum_return(r_syn, pd.Timestamp("2022-01-03"), pd.Timestamp("2022-10-12")),
                         "bear2022_real": cum_return(r_btal, pd.Timestamp("2022-01-03"), pd.Timestamp("2022-10-12")),
                         "gfc_syn": cum_return(r_syn, *GFC),
                         "year_2008_syn": float(yearly(r_syn.loc["2008"]).iloc[0]),
                         "cagr_syn_overlap": float((1 + both["syn"]).prod() ** (252 / len(both)) - 1),
                         "cagr_real_overlap": float((1 + both["real"]).prod() ** (252 / len(both)) - 1)}
        btal_bars(r_syn.loc["1995-01-01":], r_on).to_csv(OUT_DIR / f"synthetic_BTAL_bars_{label}.csv.gz", float_format="%.8g")
    pd.concat([series["scaled"][0].rename("r_scaled"), series["unscaled"][0].rename("r_unscaled"),
               series["scaled"][1].rename("r_overnight_scaled")], axis=1).to_csv(OUT_DIR / "synthetic_BTAL_returns.csv.gz", float_format="%.10g")
    return report


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"levered": build_levered(), "btal": build_btal()}
    (OUT_DIR / "proxy_validation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
