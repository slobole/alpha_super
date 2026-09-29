"""Panels and causal features for the MR-beyond-DV2 study (research-only).

Every feature on row t uses data through Close_t only; decisions made after Close_t fill at Open_{t+1}.

    r_{i,t}      = C_{i,t} / C_{i,t-1} - 1                                  (CAPITALSPECIAL)
    beta_{i,t}   = Cov_252(r_i, r_H) / Var_252(r_H) over t-251..t, >= 200 pairs
    b_{i,t}      = clip(0.67 * beta_{i,t} + 0.33, 0.3, 2.0)
    e_{i,t}      = r_{i,t} - b_{i,t-1} * r_{H,t}
    sigma_{i,t}  = std(e_{i,t-62..t}), >= 50 values
    E_k          = sum_{j<k} e_{i,t-j} / (sigma_{i,t} * sqrt(k))
    F_h          = O_{t+1+h} / O_{t+1} - 1   (missing exit open -> last finite close on or before t+1+h)

Sector H for stock i on date t = argmax over SPDRs of the 252-day return correlation measured at the last
month-end strictly before t (fixed for the month; causal).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
REPO_ROOT_PATH = HERE_PATH.parents[2]
DV2_DEEP_DIR_PATH = REPO_ROOT_PATH / "scripts" / "research" / "dv2_deep_20260925"
for path in (REPO_ROOT_PATH, DV2_DEEP_DIR_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import replica as rp  # noqa: E402  (DV2 deep study: indicators with engine parity)

STUDY_OUT_PATH = REPO_ROOT_PATH / "results" / "research" / "mr_beyond_dv2_20260926"
CACHE_DIR_PATH = STUDY_OUT_PATH / "cache"
FEATURE_DIR_PATH = STUDY_OUT_PATH / "features"
DV2_CACHE_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "dv2_deep_20260925" / "cache"
SECTOR_SPDR_LIST = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
BETA_WINDOW_INT, BETA_MIN_INT = 252, 200
RESID_WINDOW_INT, RESID_MIN_INT = 63, 50
HORIZON_LIST = [1, 5, 10, 21]


# ----------------------------------------------------------------------------- loading
def _array_dir_path(label_str: str) -> Path:
    own_path = CACHE_DIR_PATH / label_str
    if (own_path / "_done").exists():
        return own_path
    dv2_path = DV2_CACHE_DIR_PATH / label_str
    if not (dv2_path / "_done").exists():
        rp.load_arrays(label_str)  # unpacks the DV2 deep .npz into memory-mappable .npy files
    return dv2_path


class Panel:
    """Dates x symbols arrays for one universe; attribute names match replica.Panel so its indicators work."""

    def __init__(self, label_str: str):
        d = _array_dir_path(label_str)
        arr = {f.stem: np.load(f, mmap_mode="r") for f in d.glob("*.npy")}
        self.label_str = label_str
        self.dates = pd.DatetimeIndex(arr["dates"])
        self.symbols = [str(s) for s in arr["symbols"]]
        self.O, self.H, self.L, self.C = arr["Open"], arr["High"], arr["Low"], arr["Close"]
        self.V, self.RAW, self.DIV = arr["Volume"], arr["Unadjusted Close"], arr["Dividend"]
        self.valid = arr["all_fields_valid"]
        self.member = arr["member"]
        self.spx = arr.get("spx_close")
        self.eng = {k[4:]: v for k, v in arr.items() if k.startswith("eng_")}
        self.extra = {k: v for k, v in arr.items() if k.endswith("_close") or k.endswith("_open")}
        self._cache: dict = {}

    def feat(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def col(self, symbol_str: str) -> int:
        return self.symbols.index(symbol_str)


class Hedges:
    """ETF panel (etfx) aligned to a stock panel's dates: SPY (with a $SPX price-index proxy before SPY exists)
    and the sector SPDRs."""

    def __init__(self, stock_panel: Panel, etf_panel: Panel | None = None):
        self.etf = etf_panel or Panel("etfx")
        date_idx = stock_panel.dates
        pos_arr = self.etf.dates.get_indexer(date_idx)
        if (pos_arr < 0).any():
            missing_idx = date_idx[pos_arr < 0]
            raise RuntimeError(f"ETF panel lacks {len(missing_idx)} stock dates, first {missing_idx[0]}")
        self.pos = pos_arr

        def take(arr2d, col_int):
            return np.asarray(arr2d[:, col_int])[pos_arr]

        self.close, self.open = {}, {}
        for sym in ["SPY"] + SECTOR_SPDR_LIST:
            c = self.etf.col(sym)
            self.close[sym], self.open[sym] = take(self.etf.C, c), take(self.etf.O, c)
        # *** CRITICAL*** SPY proxy before its first trade (1993-01-29): the $SPX price index, spliced by returns so
        # the series is continuous; only used as a hedge in the 1991-1992 holdout rows.
        spx_c = np.asarray(self.etf.extra["spx_close"])[pos_arr]
        spx_o = np.asarray(self.etf.extra["spx_open"])[pos_arr]
        spy_c, spy_o = self.close["SPY"], self.open["SPY"]
        first = int(np.argmax(np.isfinite(spy_c)))
        scale = spy_c[first] / spx_c[first]
        pre = np.arange(len(spy_c)) < first
        self.close["SPY"] = np.where(pre, spx_c * scale, spy_c)
        self.open["SPY"] = np.where(pre, spx_o * scale, spy_o)
        self.vix = np.asarray(self.etf.extra["vix_close"])[pos_arr]


# ----------------------------------------------------------------------------- rolling helpers
def _lag(a: np.ndarray, n: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    if a.ndim == 1:
        out[n:] = a[:-n]
    else:
        out[n:] = a[:-n]
    return out


def _roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Sum over t-w+1..t along axis 0 (x must have no NaN; mask separately)."""
    c = np.cumsum(x, axis=0, dtype=np.float64)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def roll_beta_corr(y: np.ndarray, x: np.ndarray, w: int = BETA_WINDOW_INT, min_n: int = BETA_MIN_INT):
    """Rolling OLS beta of y (T x N) on x (T,) and their correlation, over pairs where both are finite."""
    ok = np.isfinite(y) & np.isfinite(x)[:, None]
    y0 = np.where(ok, y, 0.0)
    x0 = np.where(ok, x[:, None], 0.0)
    n = _roll_sum(ok.astype(np.float64), w)
    sx, sy = _roll_sum(x0, w), _roll_sum(y0, w)
    sxx, syy, sxy = _roll_sum(x0 * x0, w), _roll_sum(y0 * y0, w), _roll_sum(x0 * y0, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (sxy - sx * sy / n) / (n - 1)
        vx = (sxx - sx * sx / n) / (n - 1)
        vy = (syy - sy * sy / n) / (n - 1)
        beta = cov / vx
        corr = cov / np.sqrt(vx * vy)
    bad = (n < min_n) | ~np.isfinite(beta)
    beta[bad] = np.nan
    corr[bad] = np.nan
    return beta, corr


def roll_std(x: np.ndarray, w: int, min_n: int) -> np.ndarray:
    ok = np.isfinite(x)
    x0 = np.where(ok, x, 0.0)
    n = _roll_sum(ok.astype(np.float64), w)
    s, ss = _roll_sum(x0, w), _roll_sum(x0 * x0, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (ss - s * s / n) / (n - 1)
    out = np.sqrt(np.maximum(var, 0.0))
    out[n < min_n] = np.nan
    return out


def roll_sum_strict(x: np.ndarray, k: int) -> np.ndarray:
    """Sum of x[t-k+1..t]; NaN unless all k values are finite."""
    ok = np.isfinite(x)
    s = _roll_sum(np.where(ok, x, 0.0), k)
    n = _roll_sum(ok.astype(np.float64), k)
    s[n < k] = np.nan
    s[: k - 1] = np.nan
    return s


def roll_mean(x: np.ndarray, w: int, min_n: int) -> np.ndarray:
    ok = np.isfinite(x)
    s = _roll_sum(np.where(ok, x, 0.0), w)
    n = _roll_sum(ok.astype(np.float64), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / n
    out[n < min_n] = np.nan
    return out


# ----------------------------------------------------------------------------- features
def returns(p: Panel) -> np.ndarray:
    def f():
        C = np.asarray(p.C, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return C / _lag(C) - 1.0
    return p.feat("r", f)


def adv63(p: Panel) -> np.ndarray:
    return p.feat("adv63", lambda: roll_mean(np.asarray(p.RAW) * np.asarray(p.V), 63, 63))


def history_ok(p: Panel, n: int = 252) -> np.ndarray:
    """At least n prior finite closes (the stock has a year of history for betas and ranks)."""
    def f():
        ok = np.isfinite(np.asarray(p.C))
        return np.cumsum(ok, axis=0) >= n
    return p.feat(("hist", n), f)


def month_end_rows(dates: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(np.arange(len(dates)), index=dates)
    return s.groupby([dates.year, dates.month]).max().to_numpy()


def hedge_returns(h: Hedges, sym: str) -> np.ndarray:
    c = h.close[sym]
    with np.errstate(invalid="ignore", divide="ignore"):
        return c / _lag(c) - 1.0


def sector_assignment(p: Panel, h: Hedges) -> tuple[np.ndarray, dict]:
    """assign[t, i] = index into SECTOR_SPDR_LIST (-1 = none) from the last month-end strictly before t.
    Also returns the per-sector lagged shrunk beta arrays b_s[t, i] (beta measured through t)."""
    def f():
        r = returns(p)
        T, N = r.shape
        me_rows = month_end_rows(p.dates)
        best_corr = np.full((len(me_rows), N), -np.inf)
        best_idx = np.full((len(me_rows), N), -1, dtype=np.int16)
        betas = {}
        for s_i, sym in enumerate(SECTOR_SPDR_LIST):
            beta, corr = roll_beta_corr(r, hedge_returns(h, sym))
            betas[sym] = np.clip(0.67 * beta + 0.33, 0.3, 2.0).astype(np.float32)
            c_me = corr[me_rows]
            better = np.isfinite(c_me) & (c_me > best_corr)
            best_corr[better] = c_me[better]
            best_idx[better] = s_i
            del beta, corr
        # *** CRITICAL*** a decision on row t uses the assignment measured at the last month-end row < t.
        me_pos_for_t = np.searchsorted(me_rows, np.arange(T), side="left") - 1
        assign = np.full((T, N), -1, dtype=np.int16)
        ok = me_pos_for_t >= 0
        assign[ok] = best_idx[me_pos_for_t[ok]]
        return assign, betas
    return p.feat("sector_assign", f)


def spy_beta(p: Panel, h: Hedges) -> np.ndarray:
    def f():
        beta, _ = roll_beta_corr(returns(p), hedge_returns(h, "SPY"))
        return np.clip(0.67 * beta + 0.33, 0.3, 2.0).astype(np.float32)
    return p.feat("spy_beta", f)


def residual_z(p: Panel, h: Hedges, hedge_str: str, k_list=(1, 5, 21)) -> dict:
    """E_k for hedge 'spy' or 'sec' (per-stock assigned sector, the assignment valid on the signal date)."""
    def f():
        r = returns(p)
        out = {k: np.full(r.shape, np.nan, dtype=np.float32) for k in k_list}
        if hedge_str == "spy":
            jobs = [("SPY", spy_beta(p, h), None)]
        else:
            assign, betas = sector_assignment(p, h)
            jobs = [(sym, betas[sym], assign == s_i) for s_i, sym in enumerate(SECTOR_SPDR_LIST)]
        for sym, b, sel in jobs:
            if sel is not None and not sel.any():
                continue
            # *** CRITICAL*** the beta applied to day t's hedge return is measured through t-1.
            e = r - _lag(np.asarray(b, dtype=np.float64)) * hedge_returns(h, sym)[:, None]
            sig = roll_std(e, RESID_WINDOW_INT, RESID_MIN_INT)
            for k in k_list:
                with np.errstate(invalid="ignore", divide="ignore"):
                    z = roll_sum_strict(e, k) / (sig * np.sqrt(k))
                if sel is None:
                    out[k][:] = z
                else:
                    out[k][sel] = z[sel]
            del e, sig
        return out
    return p.feat(("resid_z", hedge_str, tuple(k_list)), f)


def raw_z(p: Panel, k_list=(1, 5, 21)) -> dict:
    def f():
        r = returns(p)
        sd = roll_std(r, 63, 50)
        out = {}
        for k in k_list:
            with np.errstate(invalid="ignore", divide="ignore"):
                out[k] = (roll_sum_strict(r, k) / (sd * np.sqrt(k))).astype(np.float32)
        return out
    return p.feat(("raw_z", tuple(k_list)), f)


def raw_ret(p: Panel, k: int) -> np.ndarray:
    def f():
        C = np.asarray(p.C, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (C / _lag(C, k) - 1.0).astype(np.float32)
    return p.feat(("raw_ret", k), f)


def intra_overnight_z(p: Panel, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    def f():
        C, O = np.asarray(p.C, dtype=np.float64), np.asarray(p.O, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            intra = np.log(C / O)
            ovn = np.log(O / _lag(C))
        out = []
        for x in (intra, ovn):
            sd = roll_std(x, 63, 50)
            with np.errstate(invalid="ignore", divide="ignore"):
                out.append((roll_sum_strict(x, k) / (sd * np.sqrt(k))).astype(np.float32))
        return tuple(out)
    return p.feat(("intra_ovn", k), f)


def ibs(p: Panel) -> np.ndarray:
    def f():
        H, L, C = np.asarray(p.H), np.asarray(p.L), np.asarray(p.C)
        rng = H - L
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(rng > 0, (C - L) / rng, 0.5).astype(np.float32)
    return p.feat("ibs", f)


def dv2(p: Panel) -> np.ndarray:
    return rp.dvpct(p, 2, 126)


def abnormal_turnover(p: Panel) -> np.ndarray:
    def f():
        V = np.asarray(p.V, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (roll_mean(V, 5, 5) / roll_mean(V, 63, 50)).astype(np.float32)
    return p.feat("abn_turn", f)


def forward_open_returns(p: Panel, h: int) -> np.ndarray:
    """F_h[t] = exit / O_{t+1} - 1 with exit = O_{t+1+h}, or the last finite close on or before t+1+h."""
    def f():
        O, C = np.asarray(p.O, dtype=np.float64), np.asarray(p.C, dtype=np.float64)
        T = O.shape[0]
        last_close = pd.DataFrame(C).ffill().to_numpy()
        entry = np.full_like(O, np.nan)
        entry[:-1] = O[1:]
        exit_ = np.full_like(O, np.nan)
        ex_open = O[1 + h:]
        ex_close = last_close[1 + h:]
        exit_[: T - 1 - h] = np.where(np.isfinite(ex_open), ex_open, ex_close)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (exit_ / entry - 1.0).astype(np.float32)
    return p.feat(("fwd", h), f)


def forward_hedge_returns(h: Hedges, sym: str, hz: int) -> np.ndarray:
    O = h.open[sym]
    T = len(O)
    out = np.full(T, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[: T - 1 - hz] = O[1 + hz:] / O[1: T - hz] - 1.0
    return out


def hedged_forward(p: Panel, h: Hedges, hz: int, hedge_str: str) -> np.ndarray:
    """Stock forward return minus b x hedge forward return (b measured through t, hedge fixed at t)."""
    def f():
        F = forward_open_returns(p, hz).astype(np.float64)
        if hedge_str == "none":
            return F.astype(np.float32)
        if hedge_str == "spy":
            return (F - spy_beta(p, h) * forward_hedge_returns(h, "SPY", hz)[:, None]).astype(np.float32)
        assign, betas = sector_assignment(p, h)
        out = np.full(F.shape, np.nan, dtype=np.float32)
        for s_i, sym in enumerate(SECTOR_SPDR_LIST):
            sel = assign == s_i
            if not sel.any():
                continue
            v = F - betas[sym] * forward_hedge_returns(h, sym, hz)[:, None]
            out[sel] = v[sel]
        return out
    return p.feat(("hfwd", hz, hedge_str), f)


def eligible(p: Panel, tier_str: str) -> np.ndarray:
    """PIT member, all raw fields finite, raw close > $5, >= 252 sessions of history, tradable next open;
    tier L adds ADV63 > same-day median of the other eligible names (the DV2 floor)."""
    def f():
        O = np.asarray(p.O)
        next_open_ok = np.zeros(O.shape, dtype=bool)
        next_open_ok[:-1] = np.isfinite(O[1:])
        base = np.asarray(p.member) & np.asarray(p.valid) & (np.asarray(p.RAW) > 5.0) & history_ok(p) & next_open_ok
        if tier_str == "ALL":
            return base
        adv = adv63(p)
        ok = base & np.isfinite(adv)
        med = np.full(adv.shape[0], np.nan)
        for t in range(adv.shape[0]):
            v = adv[t][ok[t]]
            if v.size:
                med[t] = np.median(v)
        return ok & (adv > med[:, None])
    return p.feat(("elig", tier_str), f)
