"""Limit-on-close (LOC) and limit-below-open execution for the QPI and DV2 replicas (SPEC_LOC.md, research-only).

LOC buy limit (highest close that still passes the close-monotone entry conditions), computed at 15:45:
    QPI:  min( C_{t-3} * (1 + r*),  L45 + 0.10 * (H45 - L45) )
          r* = largest r < 0 with 100 * (less(r) + 1) / (down_hist + 1) < 30   (QPI rank formula, unique values)
    DV2:  mid45 * (1 + 2 * x* - dv1_{t-1}),  x* = 13th smallest of DV(2)_{t-125..t-1}  (pct rank < 10)
LOC sell limit (QPI exit), lowest close that triggers IBS > 0.9 or RSI2 > 90:
    min( L45 + 0.9 * (H45 - L45),  C_{t-1} + d* ),  d* = 9 al' - ag' if >= 0 else (9 al' - ag') / 9
    (ag', al' = Wilder averages at t-1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qpi_lib as q  # noqa: E402

rp = q.rp
lag = q.lag


# ----------------------------------------------------------------------------- thresholds
@njit(cache=True)
def _qpi_rstar(r_hist, mask, L, thr):
    T, N = r_hist.shape
    out = np.full((T, N), np.nan)
    buf = np.empty(L - 1)
    for i in range(N):
        for t in range(L - 1, T):
            if not mask[t, i]:
                continue
            bad = False
            dh = 0
            for k in range(L - 1):
                v = r_hist[t - L + 1 + k, i]
                if np.isnan(v):
                    bad = True
                    break
                buf[k] = v
                if v <= 0.0:
                    dh += 1
            if bad:
                continue
            b = thr / 100.0 * (dh + 1) - 1.0     # need less < b
            kmax = int(math.ceil(b)) - 1
            if kmax < 0:
                continue
            s = np.sort(buf)
            if kmax >= L - 1:
                continue
            out[t, i] = min(s[kmax], 0.0)
    return out


@njit(cache=True)
def _dv_xstar(x_hist, mask, w, kth):
    T, N = x_hist.shape
    out = np.full((T, N), np.nan)
    buf = np.empty(w - 1)
    for i in range(N):
        for t in range(w - 1, T):
            if not mask[t, i]:
                continue
            bad = False
            for k in range(w - 1):
                v = x_hist[t - w + 1 + k, i]
                if np.isnan(v):
                    bad = True
                    break
                buf[k] = v
            if bad:
                continue
            out[t, i] = np.sort(buf)[kth]
    return out


EPS = 1e-9


def qpi_loc(p, S):
    """S: dict with P, H, L (15:45 state). Returns buy limit, lower bound, sell limit."""
    C = np.asarray(p.C)
    r_hist = p.feat("qpi_r3", lambda: C / lag(C, 3) - 1.0)
    P, H, L = S["P"], S["H"], S["L"]
    sma_prev = lag(rp.sma(p, 200))
    with np.errstate(invalid="ignore", divide="ignore"):
        lb = (sma_prev - lag(C, 200) / 200.0) / (199.0 / 200.0)     # C > SMA200_t(C)
        m = np.asarray(p.member) & (P / lag(C, 3) - 1.0 < 0.03) & (P > 0.97 * lb) & np.isfinite(P)
        rstar = _qpi_rstar(r_hist, m, q.QPI_L, 30.0)
        lim_q = lag(C, 3) * (1.0 + rstar) * (1.0 - EPS)
        lim_i = np.where(H > L, (L + 0.10 * (H - L)) * (1.0 - EPS), np.nan)
        buy = np.fmin(lim_q, lim_i)
        buy = np.where(np.isfinite(lim_q) & np.isfinite(lim_i) & (buy > lb), buy, np.nan)
        ag, al = p.feat("wilder2", lambda: q._wilder_avgs(C, 2))
        ag1, al1 = lag(ag), lag(al)
        raw = 9.0 * al1 - ag1
        dstar = np.where(raw >= 0, raw, raw / 9.0)
        sell_r = lag(C) + dstar * (1.0 + EPS)
        sell_i = np.where(H > L, (L + 0.90 * (H - L)) * (1.0 + EPS), np.nan)
        sell = np.fmin(sell_r, sell_i)
    return {"buy": buy, "lb": lb, "sell": sell}


def dv2_loc(p, S, rule):
    C = np.asarray(p.C)
    P, H, L = S["P"], S["H"], S["L"]
    dv1 = rp.dv1(p)
    xh = rp.dvk(p, 2)
    sma_prev = lag(rp.sma(p, 200))
    with np.errstate(invalid="ignore", divide="ignore"):
        lb_sma = (sma_prev - lag(C, 200) / 200.0) / (199.0 / 200.0)
        lb_mom = (1.0 + rule.mom_thr) * lag(C, rule.mom_lb)
        lb = np.fmax(lb_sma, lb_mom)
        m = np.asarray(p.member) & (P > 0.97 * lb) & np.isfinite(P)
        # pct = (less + 0.5) / w * 100 < dv_thr  ->  less <= kth, x < sorted[kth]
        kth = int(math.ceil(rule.dv_thr / 100.0 * rule.dv_window - 0.5)) - 1
        xstar = _dv_xstar(xh, m, rule.dv_window, kth)
        # dv1(C) < y with mid(C) = (H45 + min(L45, C)) / 2 (a close below the 15:45 low becomes the day's low):
        #   C >= L45: C < mid45 * (1 + y);   C < L45: (C - H45) / (C + H45) < y  <=>  C < H45 * (1 + y) / (1 - y)
        y = 2.0 * xstar - lag(dv1)
        b1 = (H + L) / 2.0 * (1.0 + y)
        b2 = np.fmin(L, H * (1.0 + y) / (1.0 - y))
        buy = np.where(b1 >= L, b1, b2) * (1.0 - EPS)
        buy = np.where(np.isfinite(buy) & (buy > lb) & (H > L) & (y < 1.0), buy, np.nan)
    return {"buy": buy, "lb": lb, "kth": kth}


# ----------------------------------------------------------------------------- generic runner
@dataclass
class Spec:
    name: str
    open_entry: np.ndarray | None          # final-state entry mask (decision row t -> fill t+1)
    open_score: np.ndarray | None          # higher = better
    exit_open: np.ndarray                  # final-state exit mask (decision row t -> exit at open t+1)
    loc_buy: np.ndarray | None = None      # LOC buy limits on row t
    loc_P: np.ndarray | None = None        # 15:45 price (gap ranking)
    loc_policy: str = "A"
    loc_sell: np.ndarray | None = None
    buffer_bps: float = 0.0
    open_limit_k: float | None = None      # S2 arm
    open_anchor: str = "open"              # "open" | "prevclose"
    size_mult: np.ndarray | None = None    # per decision row: scales the slot budget of next-open entries
    slip_extra_bps: float = 0.0            # stress cost per side
    extra: dict = field(default_factory=dict)


def run(p, spec: Spec, start, end, capital=1_000_000.0, slots=10):
    dates = p.dates
    t0 = int(dates.searchsorted(pd.Timestamp(start)))
    t1 = int(dates.searchsorted(pd.Timestamp(end), side="right"))
    O, H, Lo, C, DIV = (np.asarray(a) for a in (p.O, p.H, p.L, p.C, p.DIV))
    N = C.shape[1]
    slip = 0.00025 + spec.slip_extra_bps / 1e4
    buf = spec.buffer_bps / 1e4
    comm = lambda x: max(1.0, 0.005 * abs(x))
    cash, sh = capital, np.zeros(N)
    epx = np.full(N, np.nan)
    edt = np.empty(N, dtype=object)
    nav = np.full(len(dates), np.nan)
    gross = np.zeros(len(dates))
    npos = np.zeros(len(dates), dtype=int)
    prev_total = capital
    pend_entry, pend_exit = [], []
    closed = []
    orders = {"loc_buy_sent": 0, "loc_buy_filled": 0, "loc_below_lb": 0, "loc_sell_filled": 0, "open_lim_sent": 0, "open_lim_filled": 0}

    def close_pos(i, px, t, kind):
        nonlocal cash
        x = -sh[i]
        cash -= x * px + comm(x)
        closed.append((p.symbols[i], edt[i], dates[t], px / epx[i] - 1.0, kind))
        sh[i] = 0.0

    def open_pos(i, px, size_px, cap, t):
        nonlocal cash
        if not (np.isfinite(size_px) and size_px > 0 and np.isfinite(px)):
            return False
        x = float(int(cap / size_px))
        if x <= 0:
            return False
        cash -= x * px + comm(x)
        sh[i], epx[i], edt[i] = x, px, dates[t]
        return True

    for t in range(t0, t1):
        held = np.nonzero(sh)[0]
        if held.size:
            d = np.where(np.isfinite(DIV[t - 1, held]), DIV[t - 1, held], 0.0)
            g = sh[held] * d
            cash += float(np.sum(g - np.maximum(g, 0.0) * rp.WITHHOLDING_RATE_FLOAT))
        for i in held:
            if not (np.isfinite(O[t, i]) and np.isfinite(C[t, i])):
                h = C[:t, i]
                close_pos(i, h[np.isfinite(h)][-1], t, "liquidate")
        cap = prev_total / slots
        cap_open = cap * (spec.size_mult[t - 1] if spec.size_mult is not None else 1.0)  # decided at close t-1
        # --- open
        for i in pend_exit:
            if sh[i] != 0 and np.isfinite(O[t, i]):
                close_pos(i, O[t, i] * (1 - slip), t, "exit_open")
        for i in pend_entry:
            if sh[i] != 0 or not np.isfinite(O[t, i]):
                continue
            if spec.open_limit_k is None:
                open_pos(i, O[t, i] * (1 + slip), C[t - 1, i], cap_open, t)
                continue
            orders["open_lim_sent"] += 1
            k = spec.open_limit_k
            if spec.open_anchor == "prevclose":
                lim = C[t - 1, i] * (1 - k)
                if O[t, i] <= lim:
                    if open_pos(i, O[t, i], C[t - 1, i], cap, t):
                        orders["open_lim_filled"] += 1
                    continue
            else:
                lim = O[t, i] * (1 - k)
            if np.isfinite(Lo[t, i]) and Lo[t, i] <= lim * (1 - 0.001):
                if open_pos(i, lim, C[t - 1, i], cap, t):
                    orders["open_lim_filled"] += 1
        # --- close: LOC sells, then LOC buys
        if spec.loc_sell is not None:
            for i in np.nonzero(sh)[0]:
                if edt[i] == dates[t]:
                    continue
                lim = spec.loc_sell[t, i]
                if np.isfinite(lim) and np.isfinite(C[t, i]) and C[t, i] >= lim * (1 + buf):
                    close_pos(i, C[t, i] * (1 - slip), t, "exit_loc")
                    orders["loc_sell_filled"] += 1
        if spec.loc_buy is not None:
            free = slots - np.count_nonzero(sh)
            if free > 0:
                lim_row = spec.loc_buy[t]
                cand = np.nonzero(np.isfinite(lim_row) & (sh == 0))[0]
                if cand.size:
                    gap = lim_row[cand] / spec.loc_P[t, cand] - 1.0
                    order = cand[np.lexsort((cand, -gap))]
                    n_send = free if spec.loc_policy == "A" else 2 * free
                    for i in order[:n_send]:
                        orders["loc_buy_sent"] += 1
                        lim = lim_row[i]
                        if np.isfinite(C[t, i]) and C[t, i] <= lim * (1 - buf):
                            if open_pos(i, C[t, i] * (1 + slip), lim, cap, t):
                                orders["loc_buy_filled"] += 1
                                if "lb" in spec.extra and not C[t, i] > spec.extra["lb"][t, i]:
                                    orders["loc_below_lb"] += 1
        # --- after the close
        pend_exit = [i for i in np.nonzero(sh)[0] if spec.exit_open[t, i]]
        pend_entry = []
        if spec.open_entry is not None:
            free = slots - np.count_nonzero(sh) + len(pend_exit)
            cand = np.nonzero(spec.open_entry[t] & np.isfinite(spec.open_score[t]))[0]
            order = cand[np.lexsort((cand, -spec.open_score[t, cand]))]
            pend_entry = [i for i in order if sh[i] == 0][:max(free, 0)]
        held = np.nonzero(sh)[0]
        pv = float(np.sum(sh[held] * C[t, held])) if held.size else 0.0
        total = cash + pv
        nav[t], gross[t], npos[t] = total, pv / total, held.size
        prev_total = total
    sl = slice(t0, t1)
    tr = pd.DataFrame(closed, columns=["asset", "entry_date", "exit_date", "ret", "kind"])
    res = rp.Result(None, dates[sl], nav[sl], npos[sl] > 0, tr)
    res.diag = {"orders": orders, "gross_mean": float(gross[sl].mean()), "gross_max": float(gross[sl].max()),
                "max_positions": int(npos[sl].max()), "gross_ser": gross[sl]}
    return res


def summarize(res):
    s = q.summarize(res)
    s.update({k: v for k, v in res.diag.items() if k not in ("orders", "gross_ser")})
    s.update(res.diag.get("orders", {}))
    return s


# ----------------------------------------------------------------------------- DV2 final-state masks (wired rule)
def dv2_masks(p, rule):
    sig, need_sig = rp.oversold_mask_and_feats(p, rule)
    trend, need_trend = rp.trend_mask(p, rule)
    score = rp.rank_score(p, rule)
    complete = np.asarray(p.valid).copy()
    for a in need_sig + need_trend + [rp.natr(p, 14)]:
        complete &= np.isfinite(a)
    entry = complete & np.asarray(p.member) & sig & trend & np.isfinite(score)
    with np.errstate(invalid="ignore"):
        ex = np.asarray(rp.exit_mask(p, rule)) == True  # noqa: E712
    return entry, score, ex
