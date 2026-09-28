"""Shared helpers for the Inflation Compass deep research (2026-09-28; research-only).

- Data: Norgate TOTALRETURN signal closes and CAPITALSPECIAL execution bars (with Dividend) for every symbol the
  study touches, cached once as pickles; T5YIE and DTB3 from the shared ../1_data FRED cache (frozen copy).
- Signal: a vectorized re-implementation of the fixed module's month-end rule with every parameter exposed.
  It is validated against strategies.taa_df.strategy_taa_inflation_compass (phase 0) before use.
- Replica: a fast share-accounting replica of the Vanilla engine for monthly target weights:
    shares_i = V_(d-1) * w_i / Close_(d-1)     (sized from the prior close, like DefenseFirstStrategy)
    fill at Open_d * (1 +/- slippage)          (buys pay up, sells receive less)
    dividends: shares held at Close_(d-1) receive Dividend_(d-1) * (1 - 25% withholding) in cash before Open_d
    cash earns 0 unless cash_rate_ser is given (then lagged DTB3 accrues ACT/360)
  Whole-share rounding is ignored (fractional shares).

Run from the worktree root: uv run python scripts/research/inflation_compass_deep_20260928/<phase>.py
"""

from __future__ import annotations

import os
import pickle
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "research" / "inflation_compass_deep_20260928"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)

MAIN_START_STR = "2003-05-01"
MAIN_END_STR = "2026-08-19"
PERIODS = {
    "P1": ("2003-05-01", "2012-12-31"),
    "P2": ("2013-01-01", "2019-12-31"),
    "P3": ("2020-01-01", "2026-08-19"),
}
SLIPPAGE = 0.0005
WITHHOLDING = 0.25  # engine DEFAULT_DIVIDEND_WITHHOLDING_RATE_FLOAT

SIGNAL_SYMBOLS = ("SPY", "XLE", "XLI", "XLF", "XLB", "XLU", "XLV", "XLP", "XLK", "XLY", "QQQ", "IEF", "TLT",
                  "SHY", "DBC", "GLD")
EXEC_SYMBOLS = SIGNAL_SYMBOLS
MACRO_SRC = REPO.parent / "1_data"


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------
def _load_norgate(symbol_str: str, adjustment_str: str) -> pd.DataFrame:
    path = CACHE / f"norgate_{symbol_str.replace('$', 'IDX_')}_{adjustment_str}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    from data.norgate_loader import load_price_timeseries

    df = load_price_timeseries(symbol_str, adjustment_str=adjustment_str, start_date_str="1990-01-01",
                               end_date_str=None)
    df.to_pickle(path)
    return df


def signal_close_df(symbols=SIGNAL_SYMBOLS) -> pd.DataFrame:
    return pd.DataFrame({s: _load_norgate(s, "TOTALRETURN")["Close"] for s in symbols}).sort_index()


def exec_frames(symbols=EXEC_SYMBOLS) -> dict[str, pd.DataFrame]:
    return {s: _load_norgate(s, "CAPITALSPECIAL") for s in symbols}


def spxtr_close() -> pd.Series:
    return _load_norgate("$SPXTR", "TOTALRETURN")["Close"]


def fred_series(series_id_str: str) -> pd.Series:
    frozen = CACHE / f"{series_id_str}_frozen.csv"
    if not frozen.exists():
        shutil.copy2(MACRO_SRC / f"{series_id_str}.csv", frozen)
    df = pd.read_csv(frozen)
    ser = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    ser.index = pd.to_datetime(df.iloc[:, 0])
    ser = ser.dropna().sort_index()
    ser.name = series_id_str
    return ser


def sessions() -> pd.DatetimeIndex:
    """XNYS sessions = SPY CAPITALSPECIAL rows with volume (the module's execution calendar)."""
    spy = _load_norgate("SPY", "CAPITALSPECIAL")
    return pd.DatetimeIndex(spy.index[spy["Volume"].fillna(0) > 0])


# ---------------------------------------------------------------------------------------------------------------
# Signal (vectorized, parameterized; must equal the module on its defaults)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    threshold: float = 2.0
    be_lookback: int = 60
    slope_lookback: int = 60
    sma: int = 200
    strict_ties: bool = True          # '>' as in the module; False -> '>='
    anchor_lag_extra: int = 0         # 0 = anchor dated T-L (module); 1 = anchor lagged one more session
    hysteresis: tuple | None = None   # (on_level, off_level) for the level gate (C5)


def _align(t5: pd.Series, idx: pd.DatetimeIndex, same_date: bool, tol_days: int = 7) -> pd.Series:
    left = pd.DataFrame({"d": pd.DatetimeIndex(idx)})
    right = t5.dropna().rename("v").reset_index()
    right.columns = ["o", "v"]
    m = pd.merge_asof(left, right, left_on="d", right_on="o", direction="backward",
                      tolerance=pd.Timedelta(days=tol_days), allow_exact_matches=same_date)
    return pd.Series(m["v"].to_numpy(dtype=float), index=idx)


def rolling_slope(y: pd.Series, n: int) -> pd.Series:
    """OLS slope over the trailing n values (window ends at t), closed form; NaN if any value is missing."""
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    # slope = sum((x - xm) * y_window) / sxx ; weights applied by a rolling dot product
    w = (x - xm) / sxx
    v = y.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        win = np.lib.stride_tricks.sliding_window_view(v, n)
        out[n - 1:] = win @ w
    return pd.Series(out, index=y.index)


@dataclass
class Features:
    growth_on: pd.Series
    level: pd.Series           # published T5YIE at T
    prior: pd.Series           # anchor
    asset_slope: pd.Series
    idx: pd.DatetimeIndex


_FEATURE_CACHE: dict = {}


def daily_features(sig: pd.DataFrame, t5: pd.Series, p: Params, t5_same_date: bool = False) -> Features:
    """Daily features on the signal index. t5_same_date=True reproduces the OLD (leaky) module for Phase 1."""
    key = (id(sig), id(t5), p.be_lookback, p.slope_lookback, p.sma, p.anchor_lag_extra, t5_same_date)
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]
    idx = pd.DatetimeIndex(sig.index)
    s = sig[["SPY", "XLE", "XLI", "XLF", "XLB", "XLU", "XLV", "XLP"]].astype(float)
    r = s.pct_change(fill_method=None)
    pos = (0.5 * r["XLE"] + (r["XLI"] + r["XLF"] + r["XLB"]) / 6.0)
    neg = (r["XLU"] + r["XLV"] + r["XLP"]) / 3.0
    pos[r[["XLE", "XLI", "XLF", "XLB"]].isna().any(axis=1)] = np.nan
    neg[r[["XLU", "XLV", "XLP"]].isna().any(axis=1)] = np.nan
    ratio = (1 + pos).cumprod() / (1 + neg).cumprod()
    slope = rolling_slope(ratio, p.slope_lookback)
    sma = s["SPY"].rolling(p.sma, min_periods=p.sma).mean()
    growth = s["SPY"] > sma
    if t5_same_date:
        level = _align(t5, idx, same_date=True)
        prior = level.shift(p.be_lookback)
    else:
        level = _align(t5, idx, same_date=False)
        dated = _align(t5, idx, same_date=True)
        # *** CRITICAL*** anchor = observation dated on/before session T-L (published by T-L+1 <= T)
        prior = dated.shift(p.be_lookback + p.anchor_lag_extra) if p.anchor_lag_extra == 0 else \
            level.shift(p.be_lookback)
    f = Features(growth_on=growth.where(sma.notna()), level=level, prior=prior, asset_slope=slope, idx=idx)
    _FEATURE_CACHE[key] = f
    return f


REGIME_WEIGHTS = {
    (True, True): {"XLE": 1.0},
    (True, False): {"XLK": 1.0},
    (False, True): {"XLU": 1.0},
    (False, False): {"XLP": 0.5, "IEF": 0.5},
}


def month_end_index(idx: pd.DatetimeIndex, offset: int = 0) -> pd.DatetimeIndex:
    """Last session of each month, moved `offset` sessions (negative = earlier) within the session index."""
    ser = pd.Series(np.arange(len(idx)), index=idx)
    last_pos = ser.groupby(idx.to_period("M")).max().to_numpy()
    pos = last_pos + offset
    pos = pos[(pos >= 0) & (pos < len(idx))]
    return idx[pos]


def regime_frame(f: Features, p: Params, decision_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """growth / inflation booleans at the decision dates; drops dates with missing features."""
    cmp = (lambda a, b: a > b) if p.strict_ties else (lambda a, b: a >= b)
    lvl = f.level.reindex(decision_idx)
    pri = f.prior.reindex(decision_idx)
    slo = f.asset_slope.reindex(decision_idx)
    gro = f.growth_on.reindex(decision_idx)
    ok = lvl.notna() & pri.notna() & slo.notna() & gro.notna()
    if p.hysteresis is None:
        level_on = cmp(lvl, p.threshold)
    else:
        on_lvl, off_lvl = p.hysteresis
        state, vals = False, []
        for v in lvl.to_numpy():
            if np.isnan(v):
                vals.append(False)
                continue
            if state and v < off_lvl:
                state = False
            elif (not state) and v > on_lvl:
                state = True
            vals.append(state)
        level_on = pd.Series(vals, index=decision_idx)
    infl = level_on & (cmp(lvl, pri) | (slo > 0))
    out = pd.DataFrame({"growth": gro.astype(bool), "infl": infl.astype(bool), "level": lvl, "prior": pri,
                        "slope": slo}, index=decision_idx)
    return out[ok]


def weights_from_regimes(reg: pd.DataFrame, cell_map: dict | None = None) -> pd.DataFrame:
    cell_map = cell_map or REGIME_WEIGHTS
    rows = [cell_map[(bool(g), bool(i))] for g, i in zip(reg["growth"], reg["infl"])]
    return pd.DataFrame(rows, index=reg.index).fillna(0.0)


def compass_weights(sig, t5, p: Params = Params(), offset: int = 0, cell_map=None, t5_same_date=False):
    f = daily_features(sig, t5, p, t5_same_date=t5_same_date)
    reg = regime_frame(f, p, month_end_index(f.idx, offset))
    return weights_from_regimes(reg, cell_map), reg


# ---------------------------------------------------------------------------------------------------------------
# Replica backtest
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Bars:
    idx: pd.DatetimeIndex
    open: pd.DataFrame
    close: pd.DataFrame
    div: pd.DataFrame


_BARS: dict = {}


def bars(symbols) -> Bars:
    key = tuple(sorted(symbols))
    if key in _BARS:
        return _BARS[key]
    fr = exec_frames(key)
    idx = sessions()
    o = pd.DataFrame({s: fr[s]["Open"] for s in key}).reindex(idx)
    c = pd.DataFrame({s: fr[s]["Close"] for s in key}).reindex(idx)
    d = pd.DataFrame({s: fr[s]["Dividend"] if "Dividend" in fr[s] else 0.0 for s in key}).reindex(idx).fillna(0.0)
    b = Bars(idx=idx, open=o, close=c, div=d)
    _BARS[key] = b
    return b


def map_to_execution(decision_weights: pd.DataFrame, idx: pd.DatetimeIndex, lag: int = 1) -> pd.DataFrame:
    """Decision at date T -> execute at the lag-th session after T (lag=1: first session after T, i.e. the next
    open; lag=0: the session on T itself, for same-close fills; T must then be a session)."""
    dec = pd.DatetimeIndex(decision_weights.index)
    if lag == 0:
        exe = idx.get_indexer(dec)
        if (exe < 0).any():
            raise ValueError("same-close execution needs decision dates that are sessions")
    else:
        # *** CRITICAL*** first session strictly after the decision date, then lag-1 more
        exe = idx.searchsorted(dec, side="right") + (lag - 1)
    keep = exe < len(idx)
    out = decision_weights[keep].copy()
    out.index = idx[exe[keep]]
    return out[~out.index.duplicated(keep="last")]


def run_replica(exec_weights: pd.DataFrame, start: str = MAIN_START_STR, end: str = MAIN_END_STR,
                fill: str = "open", slippage: float = SLIPPAGE, capital: float = 100_000.0,
                cash_rate_ser: pd.Series | None = None, scale_ser: pd.Series | None = None,
                withholding: float = WITHHOLDING) -> pd.Series:
    """Daily NAV. exec_weights index = execution sessions. fill='open' (Open_d, sized from Close_(d-1)) or
    'close' (Close_d, sized from Close_d). scale_ser (optional, by execution date) multiplies target weights
    (volatility targeting); the remainder stays in cash."""
    syms = list(exec_weights.columns)
    b = bars(syms)
    idx = b.idx[(b.idx >= pd.Timestamp(start)) & (b.idx <= pd.Timestamp(end))]
    ew = exec_weights.reindex(columns=syms).fillna(0.0)
    ew = ew[(ew.index >= idx[0]) & (ew.index <= idx[-1])]
    # first execution on/after start: the calendar starts at the first rebalance (module behaviour)
    idx = idx[idx >= ew.index[0]]
    O = b.open.reindex(idx)[syms].to_numpy(float)
    C = b.close.reindex(idx)[syms].to_numpy(float)
    D = b.div.reindex(idx)[syms].to_numpy(float)
    full_idx = b.idx
    prev_close_all = b.close[syms].shift(1).reindex(idx).to_numpy(float)
    prev_div_all = b.div[syms].shift(1).reindex(idx).fillna(0.0).to_numpy(float)
    wmap = {d: ew.loc[d].to_numpy(float) for d in ew.index}
    if scale_ser is not None:
        scale_map = scale_ser.reindex(ew.index).fillna(1.0).to_dict()
    rate = None
    if cash_rate_ser is not None:
        rate = cash_rate_ser.reindex(full_idx).ffill().shift(1).reindex(idx).fillna(0.0).to_numpy(float) / 100.0
        gap = np.r_[1.0, np.diff(idx.values).astype("timedelta64[D]").astype(float)]
    n = len(idx)
    k = len(syms)
    shares = np.zeros(k)
    cash = capital
    nav = np.empty(n)
    prev_nav = capital
    for t in range(n):
        d = idx[t]
        if t > 0:
            # engine default: long dividends credited net of 25% withholding
            cash += float(np.nansum(shares * prev_div_all[t])) * (1.0 - withholding)
            if rate is not None and cash > 0:
                cash *= 1.0 + rate[t] * gap[t] / 360.0
        if d in wmap:
            w = wmap[d] * (scale_map[d] if scale_ser is not None else 1.0)
            if fill == "open":
                ref = prev_close_all[t] if t > 0 else C[t]
                px = O[t]
            else:
                ref = C[t]
                px = C[t]
            budget = prev_nav if fill == "open" else float(cash + np.nansum(shares * C[t]))
            with np.errstate(invalid="ignore", divide="ignore"):
                target = np.where(w > 0, budget * w / ref, 0.0)
            delta = target - shares
            fillpx = np.where(delta > 0, px * (1 + slippage), px * (1 - slippage))
            trade = np.where(delta != 0, delta * fillpx, 0.0)
            if np.isnan(trade).any():
                raise RuntimeError(f"missing price for a traded asset on {d.date()}")
            cash -= float(trade.sum())
            shares = target
        nav[t] = cash + float(np.nansum(shares * C[t]))
        prev_nav = nav[t]
    return pd.Series(nav, index=idx, name="nav")


# ---------------------------------------------------------------------------------------------------------------
# Metrics and tests
# ---------------------------------------------------------------------------------------------------------------
def metrics(nav: pd.Series, start=None, end=None) -> dict:
    s = nav
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    r = s.pct_change().dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    dd = s / s.cummax() - 1
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    vol = r.std(ddof=1) * np.sqrt(252)
    return {"start": str(s.index[0].date()), "end": str(s.index[-1].date()), "cagr": cagr * 100,
            "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(252), "vol": vol * 100, "maxdd": dd.min() * 100,
            "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan}


def period_table(nav: pd.Series) -> dict:
    out = {"ALL": metrics(nav)}
    for k, (a, b) in PERIODS.items():
        out[k] = metrics(nav, a, b)
    return out


def stationary_bootstrap_idx(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    p = 1.0 / mean_block
    i = rng.integers(n)
    for t in range(n):
        if t > 0 and rng.random() < p:
            i = rng.integers(n)
        else:
            i = (i + 1) % n if t > 0 else i
        idx[t] = i
    return idx


def paired_sharpe_bootstrap(ra: pd.Series, rb: pd.Series, draws: int = 5000, mean_block: int = 21,
                            seed: int = 7) -> dict:
    """dSharpe = Sharpe(a) - Sharpe(b) on aligned daily returns; P(dSharpe <= 0)."""
    df = pd.concat([ra, rb], axis=1).dropna()
    a, b = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(a)

    def sh(x):
        return x.mean() / x.std(ddof=1) * np.sqrt(252)

    base = sh(a) - sh(b)
    ds = np.empty(draws)
    for j in range(draws):
        ii = stationary_bootstrap_idx(n, mean_block, rng)
        ds[j] = sh(a[ii]) - sh(b[ii])
    return {"dsharpe": base, "p_le_0": float((ds <= 0).mean()), "p05": float(np.percentile(ds, 5)),
            "p95": float(np.percentile(ds, 95))}


def deflated_sharpe(sr_annual: float, n_obs: int, n_trials: int, skew: float, kurt: float,
                    sr_trials_std_annual: float) -> float:
    """Bailey & Lopez de Prado DSR on daily data. Returns P(true SR > expected max of n_trials null SRs)."""
    from scipy.stats import norm

    sr = sr_annual / np.sqrt(252)
    sd = sr_trials_std_annual / np.sqrt(252)
    emc = 0.5772156649
    e_max = sd * ((1 - emc) * norm.ppf(1 - 1.0 / n_trials) + emc * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2)
    return float(norm.cdf((sr - e_max) * np.sqrt(n_obs - 1) / denom))
