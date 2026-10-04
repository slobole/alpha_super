# Limit-on-close and limit-below-open study (frozen 2026-10-02, before any result)

Follows SPEC_FROZEN.md (QPI at 15:45 fails). Idea: the entry conditions are monotone in today's close, so a
limit-on-close (LOC) order whose limit is the highest close that still satisfies them fills exactly when the final
close "passes" the signal, without knowing the close at decision time. NYSE LOC cutoff is 15:50; we decide at 15:45.

## Strategies
- QPI: `strategy_mr_qpi_ibs_rsi_exit` replica (qpi_lib). Baseline M0 = next open / next open.
- DV2: wired rule (DV2(126) < 10, Close > SMA200, 126-day return > 5%, NATR14 rank, exit Close > High_{t-1}).
  Baseline = replica.run next open.

## LOC entry limit (computed at 15:45 from the 15:45 state P, H45, L45 and final history)
- QPI: C <= C_{t-3} * (1 + r*), r* = largest 3-day return with QPI < 30 (exact rank formula, r* < 0);
  and C <= L45 + 0.10 * (H45 - L45) (IBS < 0.10 with the 15:45 range; a new low gives IBS 0).
- DV2: dv1 = C / mid45 - 1, mid45 = (H45 + L45) / 2; C <= mid45 * (1 + 2 * x* - dv1_{t-1}),
  x* = 13th smallest of the previous 125 DV(2) values (DV2 < 10).
- Limit = min of the upper bounds. The lower bounds (Close > SMA200; DV2 also C > 1.05 * C_{t-126}) cannot be
  enforced by a LOC buy: an order is sent only if limit > lower bound; a fill below the lower bound is kept.
- Fill: buy at the official close if Close_t <= limit (base) or Close_t <= limit * (1 - 5 bps) (conservative,
  "trade through"). Slippage 2.5 bps, $0.005/share, $1 minimum, as the engine.
- Order policy: gap = limit / P - 1 (closest to filling first).
  A: orders to the top F candidates, F = free slots at 15:45 (no overfill).
  B: orders to the top 2F candidates; every fill is kept at NAV/10 (positions may exceed 10; gross reported).

## QPI LOC exit (secondary QPI arms)
Sell LOC at the lowest close that triggers IBS > 0.90 (15:45 range) or RSI2 > 90 (Wilder, n = 2); fill if
Close_t >= limit (base) or >= limit * (1 + 5 bps) (conservative). Positions not sold this way still exit at the next
open on the final-state rule (fallback).

## Arms (2016-01-04 -> 2026-09-24)
QPI: LOC-A / LOC-B entries x {exit next open, LOC exit + fallback}; DV2: LOC-A / LOC-B entries, exit next open.

## Gate (per arm)
Pass if Sharpe >= baseline + 0.10, >= baseline in 2016-20 and 2021-26, and the conservative fill version is
>= baseline. Six arms are tested; a single marginal pass is reported as such (no best-of claim).

## Secondary arms (same gate)
- S1 QPI next open ranked by the previous session's turnover (found post hoc on 2016-26 / 2004-26; reported, not
  promotable from this study alone).
- S2 limit below the open, QPI and DV2, k in {0.25%, 0.5%, 1.0%}, day order, sized NAV/10 / Close_{t-1}:
  anchor "open": limit = Open_t * (1 - k), placed at the open, fill at the limit if Low_t <= limit * (1 - 10 bps);
  anchor "prior close": limit = Close_{t-1} * (1 - k) placed pre-open; Open_t <= limit fills at Open_t, otherwise
  as above. Unfilled orders lapse. Windows 2016-26 (gate) and 2004-26 (context).
Coverage of the 15:45 data is reported; names without it use the final state (optimistic for LOC).

## Amendment L1 (2026-10-02, validation, before any result)
The DV2 limit with a fixed mid45 satisfied DV2 < 10 at the limit in only 37% of cases: a close below the 15:45 low
becomes the day's low and moves the mid. The limit now solves dv1(C) < y with mid(C) = (H45 + min(L45, C)) / 2:
C < mid45 * (1 + y) when that is >= L45, else C < min(L45, H45 * (1 + y) / (1 - y)). At the limit DV2 < 10 holds in
100% of cases; 5 bps above it fails in 99.7%. QPI's limit already handles this (a new low gives IBS 0): 100% / 99.995%.

## Amendment L2 (2026-10-02, after the results, before any 1995-2003 run) — holdout of the two near-misses
Owner asked whether "limit k% below the prior close" (QPI) and the previous-day turnover rank are real.
Untouched holdout: 1995-01-03 -> 2003-12-31 (QPI needs 5 years of history; every earlier run started in 2004).
Arms: QPI base, QPI limit 0.5% and 1% below the prior close, QPI previous-day rank. Fixed, no other k.
Pass (per arm): holdout Sharpe >= base + 0.05 AND 2004-2026 paired block bootstrap (20-day blocks, 2,000 draws)
P(Sharpe difference > 0) >= 0.90. Also reported: return with idle cash at the T-bill rate (limit arms hold more cash).
