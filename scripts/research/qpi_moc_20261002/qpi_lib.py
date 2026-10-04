"""QPI + IBS + RSI2 replica with selectable entry / exit timing and decision state (research-only).

Engine contract (strategies/qpi/strategy_mr_qpi_ibs_rsi_exit.py), as in the DV2 replica:
    decision after Close_t; next-open fills at Open_{t+1} * (1 +/- slippage)
    commission = max(1, 0.005 * |shares|); dividends net of 25% withholding; slots = 10, cap = NAV_{t-1} / 10
Close fills (MOC): decision from a decision state (P, Hd, Ld) of day t, fill at Close_t * (1 +/- slippage).

Decision-state features with today's value replaced (history uses final closes only):
    r3_t   = P_t / C_{t-3} - 1
    QPI_t  = QPI of r3_t inside [r3_{t-L+1..t-1}, r3_t], L = 5 * 252
    SMA_t  = SMA_{t-1} + (P_t - C_{t-200}) / 200
    IBS_t  = (P_t - Ld_t) / (Hd_t - Ld_t)            (NaN if Hd == Ld)
    RSI2_t = Wilder RSI with today's change P_t - C_{t-1}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from numba import njit

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/research/dv2_deep_20260925"))
import replica as rp  # noqa: E402

OUT = REPO / "results/research/qpi_moc_20261002"
QPI_L = 5 * 252


def lag(x, k=1):
    return np.vstack([np.full((k, x.shape[1]), np.nan), x[:-k]])


@njit(cache=True)
def _qpi_replace_last(r_hist, r_new, mask, L):
    T, N = r_hist.shape
    out = np.full((T, N), np.nan)
    for i in range(N):
        for t in range(L - 1, T):
            if not mask[t, i]:
                continue
            x = r_new[t, i]
            if np.isnan(x):
                continue
            less = 0
            eq = 1
            down = 1 if x <= 0.0 else 0
            bad = False
            for j in range(t - L + 1, t):
                v = r_hist[j, i]
                if np.isnan(v):
                    bad = True
                    break
                if v < x:
                    less += 1
                elif v == x:
                    eq += 1
                if v <= 0.0:
                    down += 1
            if bad:
                continue
            rank = (less + (eq + 1.0) / 2.0) / L
            pd_ = down / L
            if x <= 0.0:
                out[t, i] = 100.0 * rank / pd_
            else:
                out[t, i] = 100.0 * (1.0 - rank) / (1.0 - pd_)
    return out


@njit(cache=True)
def _wilder_avgs(C, n):
    """talib-style RSI averages: seed = mean of the first n changes after the first valid close, then Wilder."""
    T, N = C.shape
    ag = np.full((T, N), np.nan)
    al = np.full((T, N), np.nan)
    for i in range(N):
        start = -1
        for t in range(T):
            if not np.isnan(C[t, i]):
                start = t
                break
        if start < 0:
            continue
        g = 0.0
        l = 0.0
        cnt = 0
        seeded = False
        for t in range(start + 1, T):
            c0, c1 = C[t - 1, i], C[t, i]
            if np.isnan(c0) or np.isnan(c1):
                break  # talib output is unreliable after a gap; stop (engine features then NaN too)
            d = c1 - c0
            gg = d if d > 0 else 0.0
            ll = -d if d < 0 else 0.0
            if not seeded:
                g += gg
                l += ll
                cnt += 1
                if cnt == n:
                    g /= n
                    l /= n
                    seeded = True
                    ag[t, i] = g
                    al[t, i] = l
            else:
                g = (g * (n - 1) + gg) / n
                l = (l * (n - 1) + ll) / n
                ag[t, i] = g
                al[t, i] = l
    return ag, al


def rsi_from(ag, al):
    s = ag + al
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 1e-12, 100.0 * ag / s, 0.0) * np.where(np.isfinite(s), 1.0, np.nan)


class Features:
    """Entry / exit features of one decision state."""

    def __init__(self, p: rp.Panel, P=None, Hd=None, Ld=None, cand_mask=None):
        C = np.asarray(p.C)
        self.final = P is None
        P = C if P is None else P
        Hd = np.asarray(p.H) if Hd is None else Hd
        Ld = np.asarray(p.L) if Ld is None else Ld
        r3_hist = p.feat("qpi_r3", lambda: C / lag(C, 3) - 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.r3 = P / lag(C, 3) - 1.0
            rng = Hd - Ld
            self.ibs = np.where(rng > 0, (P - Ld) / rng, np.nan)
        sma200 = rp.sma(p, 200)
        self.sma = sma200 if self.final else lag(sma200) + (P - lag(C, 200)) / 200.0
        m = np.asarray(p.member) & (self.r3 < 0) & np.isfinite(self.r3)
        if cand_mask is not None:
            m &= cand_mask
        self.qpi = _qpi_replace_last(r3_hist, self.r3, m, QPI_L)
        ag, al = p.feat("wilder2", lambda: _wilder_avgs(C, 2))
        if self.final:
            self.rsi2 = rsi_from(ag, al)
        else:
            d = P - lag(C)
            ag1 = (lag(ag) * (2 - 1) + np.maximum(d, 0)) / 2.0  # Wilder, n = 2
            al1 = (lag(al) * (2 - 1) + np.maximum(-d, 0)) / 2.0
            self.rsi2 = rsi_from(ag1, al1)
        self.P = P
        with np.errstate(invalid="ignore"):
            self.entry = (np.asarray(p.valid) & np.asarray(p.member) & (self.qpi < 30) & (P > self.sma)
                          & (self.r3 < 0) & (self.ibs < 0.10) & np.isfinite(self.sma) & np.isfinite(self.rsi2))
            self.exit = (self.ibs > 0.90) | (self.rsi2 > 90)


@dataclass
class Mode:
    name: str
    entry: str            # "open" | "close"
    exit: str             # "open" | "close"
    entry_state: str = "final"   # "final" | "1545"
    exit_state: str = "final"
    rank: str = "today"   # "today" | "prev"
    extra_bps: float = 0.0


def run(p: rp.Panel, mode: Mode, F: dict, start: str, end: str, capital=1_000_000.0, slots=10):
    """F: {"final": Features, "1545": Features}. Returns rp.Result (trades carry entry/exit pairing)."""
    dates = p.dates
    t0 = int(dates.searchsorted(pd.Timestamp(start)))
    t1 = int(dates.searchsorted(pd.Timestamp(end), side="right"))
    O, C, DIV = np.asarray(p.O), np.asarray(p.C), np.asarray(p.DIV)
    N = C.shape[1]
    turn = p.feat("turn", lambda: np.asarray(p.RAW) * np.asarray(p.V))
    score = turn if mode.rank == "today" else lag(turn)
    fin = F["final"]
    Fe = F[mode.entry_state] if mode.entry == "close" else fin
    Fx = F[mode.exit_state] if mode.exit == "close" else fin
    slip = 0.00025 + mode.extra_bps / 1e4
    comm = lambda q: max(1.0, 0.005 * abs(q))

    cash, shares = capital, np.zeros(N)
    entry_px = np.full(N, np.nan)
    entry_dt = np.empty(N, dtype=object)
    nav = np.full(len(dates), np.nan)
    invested = np.zeros(len(dates), dtype=bool)
    prev_total = capital
    pend_entry, pend_exit = [], []
    closed = []  # (asset, entry_date, exit_date, gross_ret)

    def close_pos(i, px, t, kind):
        nonlocal cash
        q = -shares[i]
        cash -= q * px + comm(q)
        closed.append((p.symbols[i], entry_dt[i], dates[t], px / entry_px[i] - 1.0, kind))
        shares[i] = 0.0

    def open_pos(i, px, size_px, cap, t):
        nonlocal cash
        if not (np.isfinite(size_px) and size_px > 0 and np.isfinite(px)):
            return
        q = float(int(cap / size_px))
        if q <= 0:
            return
        cash -= q * px + comm(q)
        shares[i] = q
        entry_px[i] = px
        entry_dt[i] = dates[t]

    for t in range(t0, t1):
        held = np.nonzero(shares)[0]
        if held.size:
            d = np.where(np.isfinite(DIV[t - 1, held]), DIV[t - 1, held], 0.0)
            g = shares[held] * d
            cash += float(np.sum(g - np.maximum(g, 0.0) * rp.WITHHOLDING_RATE_FLOAT))
        for i in held:  # missing price: sold at the last close (engine rule)
            if not (np.isfinite(O[t, i]) and np.isfinite(C[t, i])):
                h = C[:t, i]
                close_pos(i, h[np.isfinite(h)][-1], t, "liquidate")
        cap = prev_total / slots
        # --- open fills (decided after the previous close, final state)
        for i in pend_exit:
            if shares[i] != 0 and np.isfinite(O[t, i]):
                close_pos(i, O[t, i] * (1 - slip), t, "exit_open")
        for i in pend_entry:
            if shares[i] == 0 and np.isfinite(O[t, i]):
                open_pos(i, O[t, i] * (1 + slip), C[t - 1, i], cap, t)
        # --- close fills (decided at 15:45 or with the final close)
        if mode.exit == "close":
            for i in np.nonzero(shares)[0]:
                if entry_dt[i] == dates[t] and mode.entry == "close":
                    continue
                if Fx.exit[t, i] and np.isfinite(C[t, i]):
                    close_pos(i, C[t, i] * (1 - slip), t, "exit_close")
        if mode.entry == "close":
            # with next-open exits, tonight's exit decisions do not free a slot at today's close
            free = slots - np.count_nonzero(shares)
            cand = np.nonzero(Fe.entry[t] & np.isfinite(score[t]))[0]
            order = cand[np.lexsort((cand, -score[t, cand]))]
            for i in order:
                if free <= 0:
                    break
                if shares[i] != 0 or not np.isfinite(C[t, i]):
                    continue
                open_pos(i, C[t, i] * (1 + slip), Fe.P[t, i], cap, t)
                if shares[i] != 0:
                    free -= 1
        # --- after the close: decisions for the next open (final state)
        pend_exit = []
        if mode.exit == "open":
            pend_exit = [i for i in np.nonzero(shares)[0] if fin.exit[t, i]]
        pend_entry = []
        if mode.entry == "open":
            free = slots - np.count_nonzero(shares) + len(pend_exit)
            cand = np.nonzero(fin.entry[t] & np.isfinite(score[t]))[0]
            order = cand[np.lexsort((cand, -score[t, cand]))]
            pend_entry = [i for i in order if shares[i] == 0][:max(free, 0)]
        held = np.nonzero(shares)[0]
        total = cash + (float(np.sum(shares[held] * C[t, held])) if held.size else 0.0)
        nav[t] = total
        invested[t] = held.size > 0
        prev_total = total
    sl = slice(t0, t1)
    tr = pd.DataFrame(closed, columns=["asset", "entry_date", "exit_date", "ret", "kind"])
    res = rp.Result(None, dates[sl], nav[sl], invested[sl], tr)
    return res


def summarize(res, trades_per_year=True):
    s = rp.stats(res.nav, res.dates)
    out = {k: s[k] for k in ("cagr", "sharpe", "maxdd", "vol")}
    yrs = (res.dates[-1] - res.dates[0]).days / 365.25
    out["trades_per_year"] = len(res.trades) / yrs
    out["avg_trade_ret"] = float(res.trades["ret"].mean()) if len(res.trades) else np.nan
    for a, b in (("2016-01-01", "2020-12-31"), ("2021-01-01", "2026-12-31"), ("2004-01-01", "2015-12-31")):
        m = (res.dates >= a) & (res.dates <= b)
        if m.sum() > 250:
            out[f"sharpe_{a[:4]}_{b[2:4]}"] = rp.stats(res.nav[m], res.dates[m])["sharpe"]
    return out
