"""Pre-2003 holdout DATA module for the Inflation Compass deep research (2026-09-28; research-only).

DATA ONLY. This module builds and sanity-checks proxy inputs for the locked holdout (SPEC_FROZEN.md, "Pre-2003
holdout"). It computes no SMA, no regime, no signal and no portfolio return, and it must stay that way.

Proxies (1983-01..2002-12; the frame is built from 1982 so a 200-session warm-up exists):
  XLE <- FF12 Enrgy   XLK <- FF12 BusEq   XLU <- FF12 Utils   XLP <- FF12 NoDur
  XLI <- FF12 Manuf   XLF <- FF12 Money   XLB <- FF12 Chems   XLV <- FF12 Hlth
  (Kenneth French "12 Industry Portfolios", DAILY, VALUE-weighted, CRSP total returns incl. dividends, gross of
  any fee.)
  SPY <- Norgate $SPXTR TOTALRETURN close from 1988-01-04 (its first row), spliced backwards with the Norgate $SPX
        PRICE index before that (see proxy_total_return_index docstring for the dividend gap).
  IEF <- synthetic 8.5-year par Treasury total return from FRED DGS10 (see synthetic_bond_returns).
Inflation: FRED EXPINF5YR (Cleveland Fed 5-year expected inflation, monthly, percent) with a conservative
availability lag: the value labelled month m is visible only from the last session of month m+1.

Caches: every downloaded raw file and Norgate pickle made here lives in
results/research/inflation_compass_deep_20260928/_cache/holdout/. The sector-ETF TOTALRETURN pickles used by the
sanity checks are read through common._load_norgate (already cached by earlier phases).

Run the sanity checks from the worktree root:
    uv run python scripts/research/inflation_compass_deep_20260928/holdout_data.py
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402

HOLD_CACHE = common.CACHE / "holdout"
HOLD_CACHE.mkdir(parents=True, exist_ok=True)
CHECKS_PATH = common.OUT / "holdout_data_checks.json"

FF12_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_daily_CSV.zip"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FF12_COLS = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money",
             "Other"]
# ETF column <- FF12 industry
FF_PROXY_MAP = {"XLE": "Enrgy", "XLK": "BusEq", "XLU": "Utils", "XLP": "NoDur", "XLI": "Manuf", "XLF": "Money",
                "XLB": "Chems", "XLV": "Hlth"}
BOND_MATURITY_YEARS = 8.5
BOND_FFILL_LIMIT = 3
SPXTR_FIRST = pd.Timestamp("1988-01-04")


# ---------------------------------------------------------------------------------------------------------------
# Raw downloads (cached)
# ---------------------------------------------------------------------------------------------------------------
def _download(url: str, path: Path) -> Path:
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research data download)"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    return path


def _norgate_hold(symbol_str: str, adjustment_str: str, start_str: str = "1970-01-01") -> pd.DataFrame:
    """Norgate series with full history (common._load_norgate starts at 1990), cached in the holdout folder."""
    path = HOLD_CACHE / f"norgate_{symbol_str.replace('$', 'IDX_')}_{adjustment_str}_from{start_str[:4]}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    from data.norgate_loader import load_price_timeseries

    df = load_price_timeseries(symbol_str, adjustment_str=adjustment_str, start_date_str=start_str,
                               end_date_str=None)
    df.to_pickle(path)
    return df


def _fred_download(series_id_str: str) -> pd.Series:
    path = _download(FRED_CSV_URL.format(sid=series_id_str), HOLD_CACHE / f"{series_id_str}.csv")
    df = pd.read_csv(path)
    ser = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    ser.index = pd.to_datetime(df.iloc[:, 0])
    ser = ser.dropna().sort_index()
    ser.name = series_id_str
    return ser


# ---------------------------------------------------------------------------------------------------------------
# 1. Kenneth French 12 industries, daily value-weighted
# ---------------------------------------------------------------------------------------------------------------
_FF12: pd.DataFrame | None = None


def ff12_daily_returns() -> pd.DataFrame:
    """FF12 DAILY VALUE-weighted returns in DECIMAL, 1926-07-01 onward, columns FF12_COLS.

    Parses only the "Average Value Weighted Returns -- Daily" block of the Tuck CSV (the equal-weighted block that
    follows is ignored). Missing codes -99.99 / -999 become NaN.
    """
    global _FF12
    if _FF12 is not None:
        return _FF12
    path = _download(FF12_URL, HOLD_CACHE / "12_Industry_Portfolios_daily_CSV.zip")
    with zipfile.ZipFile(path) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        text = zf.read(name).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if "Average Value Weighted Returns -- Daily" in ln)
    header = lines[start + 1]
    cols = [c.strip() for c in header.split(",")][1:]
    if cols != FF12_COLS:
        raise ValueError(f"unexpected FF12 header: {cols}")
    rows = []
    for ln in lines[start + 2:]:
        if not ln.strip() or not ln.strip()[0].isdigit():
            break  # blank line ends the value-weighted block
        rows.append(ln)
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=["date"] + cols)
    df.index = pd.to_datetime(df.pop("date").astype(str).str.strip(), format="%Y%m%d")
    df = df.astype(float)
    df = df.mask(df.isin([-99.99, -999.0]) | (df <= -99.0))  # missing codes
    if df.index.duplicated().any() or not df.index.is_monotonic_increasing:
        raise ValueError("FF12 dates not unique/increasing")
    _FF12 = df / 100.0
    return _FF12


# ---------------------------------------------------------------------------------------------------------------
# Synthetic 7-10y Treasury (IEF stand-in)
# ---------------------------------------------------------------------------------------------------------------
def par_bond_duration_convexity(y: np.ndarray, maturity_years: float = BOND_MATURITY_YEARS
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Modified duration D (years) and convexity C (years^2) of a semiannual par bond (coupon = yield y, decimal).

    Cash flows at k = 1..n half-years (n = 2*maturity, 17 for 8.5y): c/2 each (c = y), plus 1 at k = n.
    With v = 1/(1+y/2) and price P = 1 (par):
        D = sum_k CF_k * (k/2) * v^(k+1)           (modified duration)
        C = sum_k CF_k * k*(k+1)/4 * v^(k+2)       (convexity)
    """
    y = np.asarray(y, dtype=float)
    n = int(round(2 * maturity_years))
    k = np.arange(1, n + 1, dtype=float)
    cf = np.broadcast_to(y[:, None] / 2.0, (len(y), n)).copy()
    cf[:, -1] += 1.0
    v = 1.0 / (1.0 + y[:, None] / 2.0)
    price = (cf * v ** k).sum(axis=1)  # = 1 at par; kept to normalise exactly
    dur = (cf * (k / 2.0) * v ** (k + 1)).sum(axis=1) / price
    conv = (cf * (k * (k + 1) / 4.0) * v ** (k + 2)).sum(axis=1) / price
    return dur, conv


def synthetic_bond_returns(yield_pct: pd.Series) -> pd.Series:
    """Daily total return of a constant-maturity 8.5y par Treasury from a yield series in PERCENT.

        y_t decimal;  dy = y_t - y_(t-1)
        r_t = y_(t-1)/252  +  (-D_(t-1) * dy)  +  0.5 * C_(t-1) * dy^2
    D, C = modified duration and convexity of an 8.5-year semiannual par bond at yield y_(t-1)
    (par_bond_duration_convexity). The bond is re-struck at par every day (constant maturity), so the carry term is
    the full yield and roll-down is ignored. Carry accrues 1/252 per ROW of the input (per trading day), not per
    calendar day. 8.5y ~ midpoint of IEF's 7-10y bucket; DGS10 is the 10y yield, so the level is slightly above a
    true 8.5y yield (upward-sloping curve) - a small carry bias, documented in the checks.
    """
    y = yield_pct.astype(float) / 100.0
    y_prev = y.shift(1)
    dur, conv = par_bond_duration_convexity(y_prev.fillna(y).to_numpy())
    dy = (y - y_prev).to_numpy()
    r = y_prev.to_numpy() / 252.0 - dur * dy + 0.5 * conv * dy ** 2
    return pd.Series(r, index=yield_pct.index, name="IEF")


def dgs10() -> pd.Series:
    """FRED DGS10 (percent, daily). Frozen copy of the shared ../1_data cache (common.MACRO_SRC; starts 1962, ends
    2026-03-31 at build time) kept in the holdout folder; downloaded from FRED only if it does not reach 1982."""
    frozen = HOLD_CACHE / "DGS10_frozen.csv"
    if not frozen.exists():
        shutil.copy2(common.MACRO_SRC / "DGS10.csv", frozen)
    df = pd.read_csv(frozen)
    ser = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    ser.index = pd.to_datetime(df.iloc[:, 0])
    ser = ser.dropna().sort_index()
    ser.name = "DGS10"
    if ser.index[0] > pd.Timestamp("1982-01-01"):
        ser = _fred_download("DGS10")
    return ser


# ---------------------------------------------------------------------------------------------------------------
# SPY stand-in
# ---------------------------------------------------------------------------------------------------------------
def spx_spliced_close() -> tuple[pd.Series, dict]:
    """$SPXTR TOTALRETURN close from 1988-01-04, chained backwards with $SPX price returns before that.

    Norgate $SPXTR begins 1988-01-04. Before it the level moves with the $SPX PRICE index only, so the SPY proxy
    misses dividends from its start to 1988-01-04 (S&P 500 dividend yield ~3-5%/yr in 1982-87, i.e. roughly
    1.2-1.5 bp/day; measured $SPXTR-minus-$SPX CAGR 1988-1992 = 3.84%/yr). Returns after 1988-01-04 are pure
    $SPXTR. The level is sampled on FF12 dates only (every FF12 date 1982-2002 exists in Norgate $SPX).
    """
    tr = _norgate_hold("$SPXTR", "TOTALRETURN")["Close"].dropna()
    px = _norgate_hold("$SPX", "CAPITALSPECIAL")["Close"].dropna()
    first = tr.index[0]
    r_px = px.pct_change(fill_method=None)
    r_tr = tr.pct_change(fill_method=None)
    r = pd.concat([r_px[r_px.index <= first], r_tr[r_tr.index > first]]).sort_index()
    lvl = (1.0 + r.fillna(0.0)).cumprod()
    lvl.name = "SPY"
    info = {"spxtr_first_date": str(first.date()), "spx_first_date": str(px.index[0].date()),
            "price_only_before": str(first.date())}
    return lvl, info


# ---------------------------------------------------------------------------------------------------------------
# 2. Proxy total-return index
# ---------------------------------------------------------------------------------------------------------------
def _proxy_returns(start=None, end=None) -> tuple[pd.DataFrame, dict]:
    ff = ff12_daily_returns()
    idx = ff.index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    out = pd.DataFrame(index=idx)
    for etf, ind in FF_PROXY_MAP.items():
        out[etf] = ff.loc[idx, ind]
    # SPY: returns of the spliced level on FF days (no fill: an FF day absent from Norgate stays NaN)
    spx, spx_info = spx_spliced_close()
    spx_on = spx.reindex(ff.index)  # full FF calendar so the first in-window return uses the prior FF day
    out["SPY"] = spx_on.pct_change(fill_method=None).reindex(idx)
    # IEF: DGS10 on the FF calendar, forward-filled at most 3 FF days
    y = dgs10()
    y_ff = y.reindex(ff.index)
    n_missing_raw = int(y_ff.loc[idx].isna().sum())
    y_ff = y_ff.ffill(limit=BOND_FFILL_LIMIT)
    out["IEF"] = synthetic_bond_returns(y_ff).reindex(idx)
    info = {"spx": spx_info, "ff_days_without_norgate_spx": [str(d.date()) for d in idx.difference(spx.index)],
            "ff_days_missing_dgs10_before_ffill": n_missing_raw,
            "ff_days_missing_dgs10_after_ffill": int(y_ff.loc[idx].isna().sum())}
    return out, info


def proxy_total_return_index(start="1982-01-01", end="2002-12-31") -> pd.DataFrame:
    """Daily total-return INDEX levels (1.0 on the first FF business day in [start, end]).

    Columns: XLE XLK XLU XLP XLI XLF XLB XLV (FF12 value-weighted industries, see FF_PROXY_MAP), SPY (Norgate
    $SPXTR, spliced with $SPX PRICE before 1988-01-04 - dividends missing before that date), IEF (synthetic 8.5y
    par Treasury from DGS10: r_t = y_(t-1)/252 - D*dy + 0.5*C*dy^2, D/C of the par bond at y_(t-1)).
    Index = FF12 dates. Nothing is forward-filled except DGS10 yields (max 3 FF days). A NaN return leaves the
    level NaN on that day and the chain continues from the last valid level (count reported by the checks).
    """
    r, _ = _proxy_returns(start, end)
    r = r.copy()
    r.iloc[0] = 0.0  # index starts at 1.0 on the first day
    lvl = (1.0 + r.fillna(0.0)).cumprod()
    return lvl.where(r.notna())


# ---------------------------------------------------------------------------------------------------------------
# 3. Expected inflation with availability lag
# ---------------------------------------------------------------------------------------------------------------
def _session_calendar() -> pd.DatetimeIndex:
    return ff12_daily_returns().index


def _last_session_of_month(month: pd.Period, cal: pd.DatetimeIndex) -> pd.Timestamp:
    """Last calendar session of `month`; beyond the calendar's last full month, the last weekday of the month."""
    if month.end_time.normalize() <= cal[-1]:
        in_m = cal[(cal >= month.start_time) & (cal <= month.end_time)]
        if len(in_m):
            return in_m[-1]
    return pd.offsets.BMonthEnd().rollback(month.end_time.normalize())


def expected_inflation_monthly() -> pd.Series:
    """Cleveland Fed 5-year expected inflation (FRED EXPINF5YR, percent), indexed by AVAILABILITY date.

    The value FRED dates YYYY-MM-01 (month m) becomes usable only on the last session of month m+1 (the Cleveland
    Fed releases month m during month m or m+1; the one-month lag is conservative). Sessions = FF12 dates; after the
    FF12 file ends, the last weekday of the month. Attribute `attrs["label_month"]` keeps the source month.
    """
    raw = _fred_download("EXPINF5YR")
    cal = _session_calendar()
    avail = [_last_session_of_month(pd.Period(d, "M") + 1, cal) for d in raw.index]
    ser = pd.Series(raw.to_numpy(), index=pd.DatetimeIndex(avail), name="EXPINF5YR")
    ser.index.name = "available"
    ser.attrs["label_month"] = [str(pd.Period(d, "M")) for d in raw.index]
    ser.attrs["earliest_label"] = str(raw.index[0].date())
    ser.attrs["earliest_available"] = str(ser.index[0].date())
    return ser


# ---------------------------------------------------------------------------------------------------------------
# 4. Holdout inputs
# ---------------------------------------------------------------------------------------------------------------
def holdout_daily_inputs(start="1982-01-01", end="2002-12-31") -> dict:
    """{"tr_index": proxy_total_return_index(start, end),
        "infl_available": EXPINF5YR on the same daily index, stepping on each availability date (a value is visible
                          on and after its availability date; NaN before the first one)}."""
    tr = proxy_total_return_index(start, end)
    ei = expected_inflation_monthly()
    infl = ei.reindex(tr.index.union(ei.index)).ffill().reindex(tr.index)
    infl.name = "EXPINF5YR_available"
    return {"tr_index": tr, "infl_available": infl}


# ---------------------------------------------------------------------------------------------------------------
# Sanity checks (data only)
# ---------------------------------------------------------------------------------------------------------------
def _pair_stats(rp: pd.Series, re: pd.Series) -> dict:
    df = pd.concat([rp.rename("p"), re.rename("e")], axis=1).dropna()
    lp = (1 + df["p"]).cumprod()
    le = (1 + df["e"]).cumprod()
    mp = lp.groupby(lp.index.to_period("M")).last().pct_change().dropna()
    me = le.groupby(le.index.to_period("M")).last().pct_change().dropna()
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    beta = float(np.cov(df["e"], df["p"])[0, 1] / df["p"].var())
    return {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n_days": int(len(df)),
            "corr_daily": round(float(df["p"].corr(df["e"])), 4),
            "corr_monthly": round(float(mp.corr(me)), 4),
            "cagr_proxy_pct": round(float((lp.iloc[-1] ** (1 / yrs) - 1) * 100), 2),
            "cagr_etf_pct": round(float((le.iloc[-1] ** (1 / yrs) - 1) * 100), 2),
            "vol_proxy_pct": round(float(df["p"].std() * np.sqrt(252) * 100), 2),
            "vol_etf_pct": round(float(df["e"].std() * np.sqrt(252) * 100), 2),
            "te_daily_ann_pct": round(float((df["p"] - df["e"]).std() * np.sqrt(252) * 100), 2),
            "te_monthly_ann_pct": round(float((mp - me).std() * np.sqrt(12) * 100), 2),
            "beta_etf_on_proxy": round(beta, 3)}


def run_checks() -> dict:
    checks: dict = {}
    ff = ff12_daily_returns()
    checks["ff12"] = {"first": str(ff.index[0].date()), "last": str(ff.index[-1].date()), "rows": int(len(ff)),
                      "nan_counts_all": {c: int(v) for c, v in ff.isna().sum().items()},
                      "nan_counts_1982_2002": {c: int(v) for c, v in
                                               ff.loc["1982":"2002"].isna().sum().items()}}

    # Holdout frame coverage
    r_hold, info = _proxy_returns("1982-01-01", "2002-12-31")
    tr = proxy_total_return_index("1982-01-01", "2002-12-31")
    checks["holdout_frame"] = {"first": str(tr.index[0].date()), "last": str(tr.index[-1].date()),
                               "rows": int(len(tr)),
                               "nan_return_counts": {c: int(v) for c, v in r_hold.iloc[1:].isna().sum().items()},
                               **info}
    # dividend gap before $SPXTR: TR minus price, 1988-1992 (first 5 years with both), annualised
    tr_n = _norgate_hold("$SPXTR", "TOTALRETURN")["Close"]
    px_n = _norgate_hold("$SPX", "CAPITALSPECIAL")["Close"]
    both = pd.concat([tr_n, px_n], axis=1, keys=["tr", "px"]).dropna().loc["1988-01-04":"1992-12-31"]
    yrs = (both.index[-1] - both.index[0]).days / 365.25
    g = lambda s: (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1  # noqa: E731
    checks["spy_splice_dividend_gap"] = {
        "tr_minus_price_cagr_1988_1992_pct": round(float((g(both["tr"]) - g(both["px"])) * 100), 2),
        "note": "SPY proxy is price-only 1982-01..1988-01-04; expect ~3.5-5%/yr missing dividends there"}

    # Overlap: FF proxies vs real ETFs (Norgate TOTALRETURN)
    ov_start, ov_end = "1999-01-01", common.MAIN_END_STR
    r_ov, _ = _proxy_returns("1998-12-01", ov_end)
    r_ov = r_ov.loc[ov_start:]
    ov = {}
    for etf in list(FF_PROXY_MAP) + ["SPY", "IEF"]:
        c = common._load_norgate(etf, "TOTALRETURN")["Close"]
        c = c.loc[: ov_end]
        re = c.pct_change(fill_method=None).loc[ov_start:]
        if etf == "IEF":  # DGS10 cache ends before the overlap end: stop at its last yield (no ffilled tail)
            re = re.loc[: dgs10().index[-1]]
        st = _pair_stats(r_ov[etf], re)
        st["proxy"] = FF_PROXY_MAP.get(etf, "$SPXTR" if etf == "SPY" else "DGS10 8.5y par bond")
        ov[etf] = st
    checks["overlap_vs_etf"] = ov
    checks["dgs10_last_date"] = str(dgs10().index[-1].date())

    # Expected inflation vs T5YIE
    ei = expected_inflation_monthly()
    raw = _fred_download("EXPINF5YR")
    t5 = common.fred_series("T5YIE")
    t5_me = t5.groupby(t5.index.to_period("M")).last()
    lab = pd.Series(raw.to_numpy(), index=raw.index.to_period("M"))
    both_lab = pd.concat([lab, t5_me], axis=1, keys=["exp", "t5"]).dropna()
    ei_av = pd.Series(ei.to_numpy(), index=ei.index.to_period("M"))  # value visible at end of that month
    both_av = pd.concat([ei_av, t5_me], axis=1, keys=["exp", "t5"]).dropna()
    h = lab.loc["1983-01":"2002-12"]
    h_av = ei_av.loc["1983-01":"2002-12"]
    checks["expinf5yr"] = {
        "earliest_label_month": ei.attrs["earliest_label"], "earliest_available": ei.attrs["earliest_available"],
        "last_label_month": str(raw.index[-1].date()), "n_obs": int(len(raw)), "nan": int(raw.isna().sum()),
        "vs_t5yie_same_label_month": {"months": int(len(both_lab)), "first": str(both_lab.index[0]),
                                      "corr_levels": round(float(both_lab["exp"].corr(both_lab["t5"])), 3),
                                      "corr_monthly_changes": round(float(both_lab.diff().dropna().corr().iloc[0, 1]), 3),
                                      "mean_expinf": round(float(both_lab["exp"].mean()), 3),
                                      "mean_t5yie": round(float(both_lab["t5"].mean()), 3),
                                      "share_expinf_gt_2": round(float((both_lab["exp"] > 2).mean()), 3),
                                      "share_t5yie_gt_2": round(float((both_lab["t5"] > 2).mean()), 3),
                                      "agree_gt_2": round(float(((both_lab["exp"] > 2) == (both_lab["t5"] > 2)).mean()), 3)},
        "vs_t5yie_as_available_at_month_end": {"months": int(len(both_av)),
                                               "corr_levels": round(float(both_av["exp"].corr(both_av["t5"])), 3),
                                               "agree_gt_2": round(float(((both_av["exp"] > 2) == (both_av["t5"] > 2)).mean()), 3)},
        "holdout_1983_2002_by_label_month": {"months": int(len(h)), "gt_2": int((h > 2).sum()),
                                             "min": round(float(h.min()), 3), "max": round(float(h.max()), 3),
                                             "min_month": str(h.idxmin())},
        "holdout_1983_2002_by_availability_month": {"months": int(len(h_av)), "gt_2": int((h_av > 2).sum())},
    }
    CHECKS_PATH.write_text(json.dumps(checks, indent=2))
    return checks


def _print_checks(c: dict) -> None:
    print(json.dumps({k: v for k, v in c.items() if k != "overlap_vs_etf"}, indent=2))
    rows = pd.DataFrame(c["overlap_vs_etf"]).T[["proxy", "start", "end", "corr_daily", "corr_monthly",
                                                 "cagr_proxy_pct", "cagr_etf_pct", "vol_proxy_pct", "vol_etf_pct",
                                                 "te_daily_ann_pct", "te_monthly_ann_pct", "beta_etf_on_proxy"]]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(rows)
    print(f"written: {CHECKS_PATH}")


if __name__ == "__main__":
    _print_checks(run_checks())
