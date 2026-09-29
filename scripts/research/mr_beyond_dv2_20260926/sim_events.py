"""Family-B event pods: index-exit rebound (frozen Stage 2) and the labelled exploratory index-flow long/short pod.

Engine contract (replica style): a decision after Close_{t-1} fills at Open_t.
- Event with first day out d+1 is known after Close_{d+1}; entry Open_{d+2}; exit at the open h sessions later.
- Slot size = NAV_{t-1} / S; shares = int(size / Close_{t-1}); fills at Open_t * (1 +/- 2.5 bps) plus
  max($1, $0.005 * shares).
- Dividends before the open of t for positions held at Close_{t-1}: longs +75% of gross, shorts -100%.
- Borrow 0.5%/yr ACT/360 on short market value at Close_{t-1} (SPY hedge and, in the exploratory pod, stocks).
- A held name without Open_t or Close_t is closed at its last finite close (engine rule).
- Hedge: at entry, SPY shares = int(b * size / SPY_Close_{t-1}) in the opposite direction; unwound at the trade's
  exit. b = the event's 252-day shrunk beta to SPY.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_PATH))
import features as ft  # noqa: E402
from data.norgate_loader import load_price_timeseries  # noqa: E402

EVENTS_PATH = ft.STUDY_OUT_PATH / "stage1_b" / "events.csv"
PRICE_CACHE_PATH = ft.STUDY_OUT_PATH / "stage2_b" / "event_prices.npz"


@dataclass(frozen=True)
class EventRule:
    groups: tuple = ("ndx",)                 # index keys whose exit-to-none events are traded long
    short_add_groups: tuple = ()             # exploratory: index keys whose additions are shorted
    slots: int = 10
    hold: int = 20
    hedge: str = "none"                      # none | spy
    min_raw_close: float = 5.0
    min_adv: float = 5e6
    slippage: float = 0.00025
    extra_slippage: float = 0.0              # +5 bps/side stress
    borrow_rate: float = 0.005
    stock_borrow_rate: float = 0.005


def tradable_events(rule: EventRule, ev: pd.DataFrame) -> pd.DataFrame:
    longs = ev[(ev.kind == "exit") & (ev.other == "none") & ev["index"].isin(rule.groups)].assign(side=1)
    shorts = ev[(ev.kind == "add") & ev["index"].isin(rule.short_add_groups)].assign(side=-1)
    sel = pd.concat([longs, shorts])
    sel = sel[(sel.raw_close > rule.min_raw_close) & (sel.adv63 > rule.min_adv)]
    # the same stock can leave two indices on one day (e.g. Nasdaq-100 and S&P 500): trade it once per side/date
    return sel.drop_duplicates(["symbol", "first_day", "side"]).sort_values(["first_day", "adv63"], ascending=[True, False])


def load_prices(symbols: list[str], cal: pd.DatetimeIndex) -> dict:
    cache = {}
    if PRICE_CACHE_PATH.exists():
        z = np.load(PRICE_CACHE_PATH, allow_pickle=False)
        have = [str(s) for s in z["symbols"]]
        for j, s in enumerate(have):
            cache[s] = (z["O"][:, j], z["C"][:, j], z["D"][:, j])
    missing = [s for s in symbols if s not in cache]
    for s in missing:
        px = load_price_timeseries(s, start_date_str="1989-01-01").reindex(cal)
        div = px["Dividend"] if "Dividend" in px else pd.Series(0.0, index=cal)
        cache[s] = (px["Open"].to_numpy(float), px["Close"].to_numpy(float), div.fillna(0.0).to_numpy(float))
    if missing:
        PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(cache)
        np.savez(PRICE_CACHE_PATH, symbols=np.array(keys), O=np.column_stack([cache[k][0] for k in keys]),
                 C=np.column_stack([cache[k][1] for k in keys]), D=np.column_stack([cache[k][2] for k in keys]))
    return cache


def run(rule: EventRule, start="2000-01-03", end="2026-08-19", capital=1_000_000.0, ev: pd.DataFrame | None = None,
        etf: ft.Panel | None = None, prices: dict | None = None):
    etf = etf or ft.Panel("etfx")
    cal = etf.dates
    ev = pd.read_csv(EVENTS_PATH, parse_dates=["first_day"]) if ev is None else ev
    sel = tradable_events(rule, ev)
    prices = prices or load_prices(sorted(sel.symbol.unique()), cal)
    spy_o = np.asarray(etf.O[:, etf.col("SPY")])
    spy_c = np.asarray(etf.C[:, etf.col("SPY")])
    spy_d = np.nan_to_num(np.asarray(etf.DIV[:, etf.col("SPY")]))
    t0, t1 = int(cal.searchsorted(pd.Timestamp(start))), int(cal.searchsorted(pd.Timestamp(end), side="right"))
    # *** CRITICAL*** first_day = d+1 (first session out/in); known after Close_{d+1}; entry row = d+2.
    sel = sel.assign(entry_row=cal.get_indexer(sel.first_day) + 1)
    by_entry = {r: g for r, g in sel.groupby("entry_row")}
    slip = rule.slippage + rule.extra_slippage
    comm = lambda q: max(1.0, 0.005 * abs(q))
    cash, spy_sh = capital, 0.0
    pos = {}   # key -> dict(symbol, shares, exit_row, hedge_shares)
    nav = np.full(len(cal), np.nan)
    gross = np.zeros(len(cal))
    prev_total = capital
    trades = []
    for t in range(t0, t1):
        days = (cal[t] - cal[t - 1]).days
        # dividends and borrow for positions held at Close_{t-1}
        for k, p_ in pos.items():
            O, C, D = prices[p_["symbol"]]
            g = p_["shares"] * D[t - 1]
            cash += g * 0.75 if g > 0 else g
            if p_["shares"] < 0 and np.isfinite(C[t - 1]):
                cash -= rule.stock_borrow_rate * abs(p_["shares"]) * C[t - 1] * days / 360.0
        g = spy_sh * spy_d[t - 1]
        cash += g * 0.75 if g > 0 else g
        if spy_sh < 0:
            cash -= rule.borrow_rate * abs(spy_sh) * spy_c[t - 1] * days / 360.0
        spy_order = 0.0
        # missing-data liquidation and scheduled exits (decided after Close_{t-1}, filled at Open_t)
        for k in list(pos):
            p_ = pos[k]
            O, C, D = prices[p_["symbol"]]
            if not (np.isfinite(O[t]) and np.isfinite(C[t])):
                hist = C[:t][np.isfinite(C[:t])]
                px = hist[-1]
                cash -= -p_["shares"] * px + comm(p_["shares"])
                spy_order -= p_["hedge"]
                trades.append((cal[t], p_["symbol"], -p_["shares"], px, "liquidate"))
                del pos[k]
            elif t >= p_["exit_row"]:
                q = -p_["shares"]
                px = O[t] * (1 - slip) if q < 0 else O[t] * (1 + slip)
                cash -= q * px + comm(q)
                spy_order -= p_["hedge"]
                trades.append((cal[t], p_["symbol"], q, px, "exit"))
                del pos[k]
        # entries
        grp = by_entry.get(t)
        if grp is not None:
            size = prev_total / rule.slots
            for _, e in grp.iterrows():
                if len(pos) >= rule.slots:
                    break
                key = (e.symbol, e.side)
                if key in pos:
                    continue
                O, C, D = prices[e.symbol]
                if not (np.isfinite(O[t]) and np.isfinite(C[t - 1]) and C[t - 1] > 0):
                    continue
                q = float(int(size / C[t - 1])) * e.side
                if q == 0:
                    continue
                px = O[t] * (1 + slip) if q > 0 else O[t] * (1 - slip)
                cash -= q * px + comm(q)
                h_sh = 0.0
                if rule.hedge == "spy":
                    h_sh = -np.sign(q) * float(int(e.beta * size / spy_c[t - 1]))
                    spy_order += h_sh
                pos[key] = {"symbol": e.symbol, "shares": q, "exit_row": t + rule.hold, "hedge": h_sh}
                trades.append((cal[t], e.symbol, q, px, "entry"))
        if spy_order != 0.0:
            px = spy_o[t] * (1 + slip) if spy_order > 0 else spy_o[t] * (1 - slip)
            cash -= spy_order * px + comm(spy_order)
            spy_sh += spy_order
        mv = 0.0
        g_exp = 0.0
        for p_ in pos.values():
            v = p_["shares"] * prices[p_["symbol"]][1][t]
            mv += v
            g_exp += abs(v)
        mv += spy_sh * spy_c[t]
        total = cash + mv
        nav[t] = total
        gross[t] = g_exp / total if total > 0 else np.nan
        prev_total = total
    sl = slice(t0, t1)
    return pd.Series(nav[sl], index=cal[sl]), pd.Series(gross[sl], index=cal[sl]), pd.DataFrame(
        trades, columns=["date", "symbol", "shares", "price", "kind"])
