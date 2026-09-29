"""Fast replica of the Vanilla engine for the DV2 rule family (research-only).

It reproduces the engine contract used by strategies/dv2/strategy_mr_dv2.py:

  decision after Close_p (p = previous session), fill at Open_t (t = p + 1):
      exit   if Close_p > High_{p-1}                       (X0; other exits selectable)
      slots  = S - held + exits
      entry  = top-ranked eligible symbols not held, shares = int((NAV_p / S) / Close_p)
      fill   = Open_t * (1 +/- slippage), commission = max(1, 0.005 * |shares|)
  before the open of t: dividend cash = shares_p * Dividend_p * (1 - 0.25) for longs
  held symbol missing Open_t or Close_t: sold at the last close <= p (no slippage), commission charged
  NAV_t = cash_t + sum shares * Close_t

Timing "moc": decision at ~15:45 of day t from a decision-time price/high/low of day t (DecisionState),
fill at Close_t with the same slippage; dividends and missing-data rules unchanged.

Eligibility mirrors the engine's `close.unstack().dropna()`: every raw field and every feature the rule uses must
be finite on the decision row; the liquidity-floor median is taken over members with complete rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from numba import njit

CACHE_DIR_PATH = Path(__file__).resolve().parents[3] / "results" / "research" / "dv2_deep_20260925" / "cache"
WITHHOLDING_RATE_FLOAT = 0.25


# ----------------------------------------------------------------------------- data
def load_arrays(label_str: str) -> dict:
    """Arrays as read-only memory maps (one .npy per array) so parallel workers share the OS page cache."""
    d = CACHE_DIR_PATH / label_str
    if not (d / "_done").exists():
        npz = np.load(CACHE_DIR_PATH / f"{label_str}.npz", allow_pickle=False)
        d.mkdir(exist_ok=True)
        for k in npz.files:
            np.save(d / f"{k}.npy", npz[k])
        (d / "_done").write_text("ok")
    out = {f.stem: np.load(f, mmap_mode="r") for f in d.glob("*.npy")}
    out["files"] = [k for k in out]
    return out


class _Arrays(dict):
    @property
    def files(self):
        return [k for k in self if k != "files"]


class Panel:
    """Arrays for one universe, dates x symbols."""

    def __init__(self, label_str: str):
        npz = load_arrays(label_str)
        self.label_str = label_str
        self.dates = pd.DatetimeIndex(npz["dates"])
        self.symbols = [str(s) for s in npz["symbols"]]
        self.O, self.H, self.L, self.C = npz["Open"], npz["High"], npz["Low"], npz["Close"]
        self.V, self.RAW, self.DIV = npz["Volume"], npz["Unadjusted Close"], npz["Dividend"]
        self.valid = npz["all_fields_valid"]
        self.member = npz["member"]
        self.spx = npz["spx_close"]
        self.eng = {k[4:]: npz[k] for k in npz["files"] if k.startswith("eng_")}
        self._cache: dict = {}

    def feat(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]


# ----------------------------------------------------------------------------- indicators
@njit(cache=True)
def _dv1(C, H, L):
    T, N = C.shape
    out = np.full((T, N), np.nan)
    for i in range(N):
        for t in range(T):
            c, h, l = C[t, i], H[t, i], L[t, i]
            if np.isnan(c) or np.isnan(h) or np.isnan(l):
                continue
            mid = (h + l) / 2.0
            if np.isnan(mid) or mid == 0.0:
                continue
            out[t, i] = (c / mid) - 1.0
    return out


@njit(cache=True)
def _roll_mean_all(x, k):
    """Mean of x[t-k+1..t] summed oldest-first; NaN unless all k values finite (k=2 == engine DV)."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for i in range(N):
        for t in range(k - 1, T):
            s = 0.0
            ok = True
            for j in range(t - k + 1, t + 1):
                v = x[j, i]
                if np.isnan(v):
                    ok = False
                    break
                s += v
            if ok:
                out[t, i] = s / k if k != 2 else (x[t - 1, i] + x[t, i]) / 2.0
    return out


@njit(cache=True)
def _pct_rank(x, w):
    """100 * average rank of x[t] inside x[t-w+1..t] / w; NaN if any value in the window is NaN (engine)."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for i in range(N):
        for t in range(w - 1, T):
            last = x[t, i]
            if np.isnan(last):
                continue
            less = 0
            eq = 0
            bad = False
            for j in range(t - w + 1, t + 1):
                v = x[j, i]
                if np.isnan(v):
                    bad = True
                    break
                if v < last:
                    less += 1
                elif v == last:
                    eq += 1
            if bad:
                continue
            out[t, i] = (less + (eq + 1.0) / 2.0) / w * 100.0
    return out


@njit(cache=True)
def _pct_rank_replace_last(x_hist, x_new, w):
    """DV percentile at t where today's value is x_new[t] and the w-1 prior values are x_hist[t-w+1..t-1]."""
    T, N = x_hist.shape
    out = np.full((T, N), np.nan)
    for i in range(N):
        for t in range(w - 1, T):
            last = x_new[t, i]
            if np.isnan(last):
                continue
            less = 0
            eq = 1  # itself
            bad = False
            for j in range(t - w + 1, t):
                v = x_hist[j, i]
                if np.isnan(v):
                    bad = True
                    break
                if v < last:
                    less += 1
                elif v == last:
                    eq += 1
            if bad:
                continue
            out[t, i] = (less + (eq - 1 + 1.0) / 2.0) / w * 100.0
    return out


def per_column(arr_list, fn):
    T, N = arr_list[0].shape
    out = np.full((T, N), np.nan)
    for i in range(N):
        cols = [a[:, i] for a in arr_list]
        if np.isfinite(cols[0]).sum() == 0:
            continue
        try:
            out[:, i] = fn(*cols)
        except Exception:  # talib raises on all-NaN input
            pass
    return out


def natr(p: Panel, n: int):
    if n == 14 and "natr" in p.eng:
        return p.eng["natr"]
    return p.feat(("natr", n), lambda: per_column([p.H, p.L, p.C], lambda h, l, c: talib.NATR(h, l, c, n)))


def sma(p: Panel, n: int):
    if n == 200 and "sma_200" in p.eng:
        return p.eng["sma_200"]
    return p.feat(("sma", n), lambda: pd.DataFrame(p.C).rolling(n).mean().to_numpy())


def mom(p: Panel, n: int):
    if n == 126 and "p126d_return" in p.eng:
        return p.eng["p126d_return"]
    return p.feat(("mom", n), lambda: (pd.DataFrame(p.C) / pd.DataFrame(p.C).shift(n) - 1.0).to_numpy())


def dv1(p: Panel):
    return p.feat("dv1", lambda: _dv1(p.C, p.H, p.L))


def dvk(p: Panel, k: int):
    return p.feat(("dvk", k), lambda: _roll_mean_all(dv1(p), k))


def dvpct(p: Panel, k: int, w: int):
    if k == 2 and w == 126 and "dv2" in p.eng:
        return p.eng["dv2"]
    return p.feat(("dvpct", k, w), lambda: _pct_rank(dvk(p, k), w))


def adv63(p: Panel):
    return p.feat("adv63", lambda: pd.DataFrame(p.RAW * p.V).rolling(63, min_periods=63).mean().to_numpy())


def rsi2(p: Panel):
    return p.feat("rsi2", lambda: per_column([p.C], lambda c: talib.RSI(c, 2)))


def ibs(p: Panel):
    rng = p.H - p.L
    with np.errstate(invalid="ignore", divide="ignore"):
        return p.feat("ibs", lambda: np.where(rng > 0, (p.C - p.L) / rng, 0.5))


# ----------------------------------------------------------------------------- rules
@dataclass(frozen=True)
class Rule:
    k: int = 2
    dv_window: int = 126
    dv_thr: float = 10.0
    ensemble: str | None = None          # None | "avg" | "vote"
    mom_lb: int | None = 126             # None = no momentum filter
    mom_thr: float = 0.05
    mom_vote: bool = False               # at least 2 of R63, R126, R252 > 0
    sma_n: int | None = 200
    rank: str = "natr14"                 # natr14 natr5 natr30 adv dv random
    slots: int = 10
    floor: bool = False
    floor_q: float = 0.5                 # member ADV63 quantile the floor requires (0.5 = HPI median rule)
    adv_min: float = 0.0                 # absolute ADV63 minimum in dollars (ETF liquidity screen)
    floor_median_all: bool = False
    side: str = "long"                   # "short": DV2 > 100 - dv_thr, exit Close < Low_{t-1}, negative shares
    short_trend: str = "mirror"          # short trend gate: mirror (C<SMA200, R126<-mom_thr) | none | up
    borrow_bps_yr: float = 0.0           # borrow fee on short market value (bps per year)
    downshock: float | None = None       # entry also needs (C_t/C_{t-1}-1)/(ATR14_{t-1}/C_{t-1}) < downshock       # True = median over every member with ADV (module rewrite of 2026-09-25 18:46)
    exit: str = "X0"
    time_limit: int = 10
    seed: int = 0
    timing: str = "next_open"            # next_open | moc
    slippage: float = 0.00025
    commission_per_share: float = 0.005
    commission_min: float = 1.0

    def with_(self, **kw):
        return replace(self, **kw)


ENSEMBLE_KS = (2, 3, 5)
ENSEMBLE_WS = (63, 126, 252)


def oversold_mask_and_feats(p: Panel, r: Rule, dec=None):
    """Boolean entry-signal matrix at the decision row plus the list of features that must be finite."""
    need = []
    if dec is not None:
        return dec.oversold(r)
    if r.ensemble is None:
        x = dvpct(p, r.k, r.dv_window)
        need.append(x)
        sig = x < r.dv_thr
    else:
        pcts = [dvpct(p, k, w) for k in ENSEMBLE_KS for w in ENSEMBLE_WS]
        stack = np.stack(pcts)
        finite = np.isfinite(stack).all(axis=0)
        if r.ensemble == "avg":
            sig = finite & (stack.mean(axis=0) < r.dv_thr)
        else:
            sig = finite & ((stack < r.dv_thr).sum(axis=0) >= 5)
        need.append(np.where(finite, 0.0, np.nan))
    return sig, need


def trend_mask(p: Panel, r: Rule, dec=None):
    C = p.C if dec is None else dec.P
    mask = np.ones(p.C.shape, dtype=bool)
    need = []
    if r.sma_n is not None:
        s = sma(p, r.sma_n) if dec is None else dec.sma(r.sma_n)
        need.append(s)
        mask &= C > s
    if r.mom_vote:
        ms = [mom(p, n) if dec is None else dec.mom(n) for n in (63, 126, 252)]
        need += ms
        mask &= (np.stack([m > 0 for m in ms]).sum(axis=0) >= 2)
    elif r.mom_lb is not None:
        m = mom(p, r.mom_lb) if dec is None else dec.mom(r.mom_lb)
        need.append(m)
        mask &= m > r.mom_thr
    return mask, need


def rank_score(p: Panel, r: Rule):
    if r.rank.startswith("natr"):
        return natr(p, int(r.rank[4:]))
    if r.rank == "adv":
        return adv63(p)
    if r.rank == "dv":
        return -dvpct(p, r.k, r.dv_window)
    if r.rank == "random":
        rng = np.random.default_rng(r.seed)
        return rng.random(p.C.shape)
    raise ValueError(r.rank)


def exit_mask(p: Panel, r: Rule, dec=None):
    """exit_now[p_row, i] on the decision row (X4 time limit handled in the loop)."""
    C = p.C if dec is None else dec.P
    prevH = np.vstack([np.full((1, p.H.shape[1]), np.nan), p.H[:-1]])
    prevC = np.vstack([np.full((1, p.C.shape[1]), np.nan), p.C[:-1]])
    x0 = C > prevH
    with np.errstate(invalid="ignore"):
        if r.exit in ("X0", "X4"):
            return x0
        if r.exit == "X1":
            d = dvpct(p, 2, 126) if dec is None else dec.dvpct(2, 126)
            return d > 50
        if r.exit == "X2":
            s5 = sma(p, 5) if dec is None else dec.sma(5)
            return C > s5
        if r.exit == "X3":
            return C > prevC
        if r.exit == "X5":
            s200 = sma(p, 200) if dec is None else dec.sma(200)
            return x0 | (C < s200)
        if r.exit == "X6":
            return (ibs(p) > 0.9) | (rsi2(p) > 90) if dec is None else dec.ibs_rsi_exit()
    raise ValueError(r.exit)


# ----------------------------------------------------------------------------- simulator
@dataclass
class Result:
    rule: Rule
    dates: pd.DatetimeIndex
    nav: np.ndarray
    invested: np.ndarray
    trades: pd.DataFrame
    fills: list = field(default_factory=list)
    diag: dict = field(default_factory=dict)


def run(p: Panel, r: Rule, start="2000-01-03", end="2026-08-19", capital=1_000_000.0, dec=None) -> Result:
    """dec: optional DecisionState for timing='moc' (decision-time price/high/low of the fill day)."""
    dates = p.dates
    t0 = int(dates.searchsorted(pd.Timestamp(start)))
    t1 = int(dates.searchsorted(pd.Timestamp(end), side="right"))
    T, N = p.C.shape
    moc = r.timing == "moc"
    if moc and dec is None:
        dec = PerfectDecision(p)  # same-close with the final close known (optimistic bound)

    if r.side == "short":
        assert not moc, "shorts: next-open only"
        x = dvpct(p, r.k, r.dv_window)
        sig, need_sig = x > (100.0 - r.dv_thr), [x]
        s200, m126 = sma(p, 200), mom(p, 126)
        with np.errstate(invalid="ignore"):
            if r.short_trend == "mirror":
                trend, need_trend = (p.C < s200) & (m126 < -r.mom_thr), [s200, m126]
            elif r.short_trend == "up":
                trend, need_trend = (p.C > s200) & (m126 > r.mom_thr), [s200, m126]
            else:
                trend, need_trend = np.ones(p.C.shape, dtype=bool), []
    else:
        sig, need_sig = oversold_mask_and_feats(p, r, dec)
        trend, need_trend = trend_mask(p, r, dec)
    score = rank_score(p, r)
    if moc and (r.rank.startswith("natr") or r.rank == "adv"):
        # *** CRITICAL*** at 15:45 today's full bar is unknown: NATR / ADV ranks use the previous session
        score = np.vstack([np.full((1, N), np.nan), score[:-1]])
    need = need_sig + need_trend + [natr(p, 14)]
    complete = p.valid.copy()
    for a in need:
        complete &= np.isfinite(a)
    base_ok = complete & p.member
    if r.floor:
        adv = adv63(p) if not moc else np.vstack([np.full((1, N), np.nan), adv63(p)[:-1]])
        raw_dec = p.RAW if not moc else dec.raw_price()
        comp_f = base_ok & np.isfinite(adv)
        med_pool = (p.member & np.isfinite(adv)) if r.floor_median_all else comp_f
        med = np.full(T, np.nan)
        for t in range(t0 - 1, t1):
            v = adv[t][med_pool[t]]
            if v.size:
                med[t] = np.median(v) if r.floor_q == 0.5 else np.quantile(v, r.floor_q)
        base_ok = comp_f & (raw_dec > 5.0) & (adv > med[:, None])
    if r.adv_min > 0:
        adv_m = adv63(p) if not moc else np.vstack([np.full((1, N), np.nan), adv63(p)[:-1]])
        base_ok = base_ok & (adv_m > r.adv_min)
    if r.downshock is not None:
        atr = p.feat("atr14", lambda: per_column([p.H, p.L, p.C], lambda h, l, c: talib.ATR(h, l, c, 14)))
        prevC = np.vstack([np.full((1, N), np.nan), p.C[:-1]])
        prevA = np.vstack([np.full((1, N), np.nan), atr[:-1]])
        Cd = p.C if dec is None else dec.P
        with np.errstate(invalid="ignore", divide="ignore"):
            # *** CRITICAL*** today's move scaled by the PREVIOUS day's ATR (known before today's close)
            shock = (Cd / prevC - 1.0) / (prevA / prevC)
        base_ok = base_ok & (shock < r.downshock)
    entry_ok = base_ok & sig & trend & np.isfinite(score)
    if r.side == "short":
        prevL = np.vstack([np.full((1, N), np.nan), p.L[:-1]])
        with np.errstate(invalid="ignore"):
            ex_now = p.C < prevL  # mirror of Close > High_{t-1}
    else:
        ex_now = exit_mask(p, r, dec)
    sgn = -1.0 if r.side == "short" else 1.0

    cash = capital
    shares = np.zeros(N)
    entry_row = np.full(N, -1)
    nav = np.full(T, np.nan)
    invested = np.zeros(T, dtype=bool)
    trade_rows = []
    fills = []
    prev_total = capital
    slip = r.slippage
    comm = lambda q: max(r.commission_min, r.commission_per_share * abs(q)) if r.commission_per_share else 0.0
    open_trade = {}

    for t in range(t0, t1):
        pr = t - 1 if not moc else t  # decision row
        # dividends credited before the open of t for shares held at close t-1
        held = np.nonzero(shares)[0]
        if held.size:
            d = p.DIV[t - 1, held]
            d = np.where(np.isfinite(d), d, 0.0)
            g = shares[held] * d
            cash += float(np.sum(g - np.maximum(g, 0.0) * WITHHOLDING_RATE_FLOAT))
            if r.borrow_bps_yr:
                short_val = -np.sum(np.minimum(shares[held], 0.0) * np.nan_to_num(p.C[t - 1, held]))
                cash -= short_val * r.borrow_bps_yr / 1e4 / 252.0
        # --- decisions
        exits = []
        for i in held:
            if moc and not np.isfinite(p.C[t, i]):
                continue
            e = bool(ex_now[pr, i]) if np.isfinite(ex_now[pr, i]) else False
            if r.exit == "X4" and entry_row[i] >= 0 and (pr - entry_row[i]) >= r.time_limit:
                e = True
            if e:
                exits.append(i)
        slots = r.slots - held.size + len(exits)
        entries = []
        if slots > 0:
            cand = np.nonzero(entry_ok[pr])[0]
            if cand.size:
                order = cand[np.argsort(-score[pr, cand], kind="stable")]
                for i in order:
                    if slots <= 0:
                        break
                    if shares[i] != 0:
                        continue
                    entries.append(i)
                    slots -= 1
        cap = prev_total / r.slots
        # --- execution at t
        fill_px_arr = p.O[t] if not moc else p.C[t]
        # missing-price liquidation (held, missing Open_t or Close_t)
        liquidated = set()
        for i in np.nonzero(shares)[0]:
            if not (np.isfinite(p.O[t, i]) and np.isfinite(p.C[t, i])):
                hist = p.C[: t, i]
                last = hist[np.isfinite(hist)][-1]
                q = -shares[i]
                c = comm(q)
                cash -= q * last + c
                trade_rows.append((open_trade.get(i), dates[t], p.symbols[i], q, last, c, "liquidate"))
                shares[i] = 0.0
                entry_row[i] = -1
                liquidated.add(i)
        for i in exits:
            if i in liquidated:
                continue
            q = -shares[i]
            px0 = fill_px_arr[i]
            if not np.isfinite(px0):
                continue
            px = px0 * (1 + np.sign(q) * slip)
            c = comm(q)
            cash -= q * px + c
            trade_rows.append((open_trade.get(i), dates[t], p.symbols[i], q, px, c, "exit"))
            shares[i] = 0.0
            entry_row[i] = -1
        for i in entries:
            if i in liquidated:
                continue
            px0 = fill_px_arr[i]
            if not np.isfinite(px0):
                continue
            size_px = p.C[pr, i] if not moc else dec.P[t, i]
            if not (np.isfinite(size_px) and size_px > 0):
                size_px = px0
            q = sgn * float(int(cap / size_px))
            if q == 0:
                continue
            px = px0 * (1 + np.sign(q) * slip)
            c = comm(q)
            cash -= q * px + c
            shares[i] = q
            entry_row[i] = t if moc else t  # fill row
            open_trade[i] = len(trade_rows)
            trade_rows.append((open_trade[i], dates[t], p.symbols[i], q, px, c, "entry"))
        held = np.nonzero(shares)[0]
        pv = float(np.sum(shares[held] * p.C[t, held])) if held.size else 0.0
        total = cash + pv
        nav[t] = total
        invested[t] = held.size > 0
        prev_total = total
    trades = pd.DataFrame(trade_rows, columns=["trade_id", "date", "asset", "amount", "price", "commission", "kind"])
    sl = slice(t0, t1)
    return Result(r, dates[sl], nav[sl], invested[sl], trades)


# ----------------------------------------------------------------------------- decision states (MOC)
class PerfectDecision:
    """Decision at the close with the final close known: P = Close_t, H/L = final."""

    def __init__(self, p: Panel, P=None, Hd=None, Ld=None):
        self.p = p
        self.P = p.C if P is None else P
        self.Hd = p.H if Hd is None else Hd
        self.Ld = p.L if Ld is None else Ld
        self._c = {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def raw_price(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.p.RAW * (self.P / self.p.C)

    def dvpct(self, k, w):
        def f():
            p = self.p
            dv1_true = dv1(p)
            dv1_new = _dv1(self.P, self.Hd, self.Ld)
            if k == 1:
                new = dv1_new
            else:
                prev_sum = _roll_mean_all(dv1_true, k - 1) * (k - 1)
                prev_sum = np.vstack([np.full((1, prev_sum.shape[1]), np.nan), prev_sum[:-1]])
                new = (prev_sum + dv1_new) / k if k != 2 else (np.vstack([np.full((1, dv1_true.shape[1]), np.nan), dv1_true[:-1]]) + dv1_new) / 2.0
            return _pct_rank_replace_last(dvk(p, k), new, w)
        if self.P is self.p.C and self.Hd is self.p.H and self.Ld is self.p.L:
            return dvpct(self.p, k, w)
        return self._get(("dvpct", k, w), f)

    def oversold(self, r: Rule):
        if r.ensemble is None:
            x = self.dvpct(r.k, r.dv_window)
            return x < r.dv_thr, [x]
        pcts = np.stack([self.dvpct(k, w) for k in ENSEMBLE_KS for w in ENSEMBLE_WS])
        finite = np.isfinite(pcts).all(axis=0)
        sig = finite & ((pcts.mean(axis=0) < r.dv_thr) if r.ensemble == "avg" else ((pcts < r.dv_thr).sum(axis=0) >= 5))
        return sig, [np.where(finite, 0.0, np.nan)]

    def sma(self, n):
        if self.P is self.p.C:
            return sma(self.p, n)
        def f():
            s = sma(self.p, n)
            prev = np.vstack([np.full((1, s.shape[1]), np.nan), s[:-1]])
            oldest = np.vstack([np.full((n, s.shape[1]), np.nan), self.p.C[:-n]])
            # SMA_t with today's close replaced: SMA_{t-1} + (P_t - C_{t-n}) / n
            return prev + (self.P - oldest) / n
        return self._get(("sma", n), f)

    def mom(self, n):
        if self.P is self.p.C:
            return mom(self.p, n)
        return self._get(("mom", n), lambda: self.P / np.vstack([np.full((n, self.P.shape[1]), np.nan), self.p.C[:-n]]) - 1.0)

    def ibs_rsi_exit(self):
        rng = self.Hd - self.Ld
        with np.errstate(invalid="ignore", divide="ignore"):
            ib = np.where(rng > 0, (self.P - self.Ld) / rng, 0.5)
        return (ib > 0.9) | (rsi2(self.p) > 90)


# ----------------------------------------------------------------------------- metrics
def stats(nav: np.ndarray, dates: pd.DatetimeIndex, invested=None, trades=None) -> dict:
    s = pd.Series(nav, index=dates).dropna()
    if invested is not None:
        inv = pd.Series(invested, index=dates)
        first = inv[inv].index[0] if inv.any() else s.index[0]
        base = s.index[max(s.index.get_loc(first) - 1, 0)]
        s = s.loc[base:]
    r = s.pct_change().dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    dd = (s / s.cummax() - 1).min()
    out = {"cagr": cagr, "sharpe": sharpe, "maxdd": dd, "vol": r.std() * np.sqrt(252), "calmar": cagr / abs(dd) if dd < 0 else np.nan}
    if trades is not None and len(trades):
        notional = (trades["amount"].abs() * trades["price"]).groupby(trades["date"]).sum()
        out["turnover_x"] = float(notional.sum() / s.mean() / yrs)
        out["entries_per_year"] = float((trades["kind"] == "entry").sum() / yrs)
    if invested is not None:
        out["exposure_days"] = float(np.mean(invested))
    return out


def daily_returns(res: Result) -> pd.Series:
    return pd.Series(res.nav, index=res.dates).pct_change()


PERIODS = {"P1": ("2000-01-01", "2014-12-31"), "P2": ("2015-01-01", "2020-12-31"), "P3": ("2021-01-01", "2026-12-31")}


def period_stats(res: Result) -> dict:
    out = {}
    for k, (a, b) in PERIODS.items():
        m = (res.dates >= a) & (res.dates <= b)
        if m.sum() > 50:
            st = stats(res.nav[m], res.dates[m])
            out[f"{k}_sharpe"] = st["sharpe"]
            out[f"{k}_cagr"] = st["cagr"]
    return out


def summarize(res: Result) -> dict:
    d = stats(res.nav, res.dates, res.invested, res.trades)
    d.update(period_stats(res))
    return d
