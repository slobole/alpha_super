# DV2 deep research — frozen specification

Written 2026-09-25 before any result of this study was computed. Owner approval: "run everything, final verdict
at the end" (/goal, 2026-09-25). Research-only: no live, release, scheduler or broker path changes.
Later changes to this file are appended as dated amendments; nothing above the amendment log is edited after results.

## Question

Can the DV2 mean-reversion pod be improved in a way that survives honest testing, and can it be traded at the
same-day close (MOC) at fund scale? Final output: one verdict (keep / adopt a named change / reject DV2 for the fund).

## Data and fixed settings

- Norgate US equities incl. delisted, S&P 500 Current & Past point-in-time membership (starts 1990-01-02).
- Signals CAPITALSPECIAL; fills/marks CAPITALSPECIAL; liquidity from Unadjusted Close x Volume.
- Main window 2000-01-03 -> 2026-08-19, $1M, same as the growth-shelf DV2 sources (reproduction anchor:
  `results/research/portfolio/growth_shelf_20260924/dv2_sources/dv2_check__*`).
- Costs: engine default (2.5 bps slippage per side, $0.005/share, $1 minimum). Stress: +5 bps per side.
- Periods: P1 2000-2014, P2 2015-2020, P3 2021-2026-08 (same split as the pakal DV2 studies).
- LOCKED HOLDOUT: S&P 500 1991-01-02 -> 1999-12-31 (never used for DV2). Used once, only for the baseline and
  the finalists, after the finalists are chosen.

## Baselines

- `wired`: strategies/dv2/strategy_mr_dv2.py (DV2(126)<10, Close>SMA200, R126>5%, NATR14 rank, 10 slots,
  exit Close>High[-1], next open).
- `floor`: wired + liquidity floor (raw price > $5, ADV63 > same-day S&P 500 member median). Promotion baseline,
  because a fund version needs the floor.
- `source`: wired with R126 > 0 (the Quantitativo source; the 5% was chosen post-hoc by the owner).

## Instrument (Phase 1)

A fast numpy replica of the Vanilla engine for this rule family. Acceptance: on `wired` it must match the engine
run `dv2_check` trade for trade (same fills; NAV path max abs daily return diff < 1e-8) or the residual must be
explained and bounded (< 0.05pp CAGR). Luck band: 200 runs of `floor` with a random rank instead of NATR.

## Pre-declared search (Phase 3), all on the replica, around `floor` unless stated

Signal axes (one at a time, plus the joint grids marked *):
- DV smoothing k in {1,2,3,4,5,10}; DV rank window in {63,126,252}; DV threshold in {5,10,15,20};
  * k x window; * k x threshold.
- Trend: momentum lookback {63,126,252} x threshold {none, 0, 5%, 10%} *; SMA filter {none,100,150,200,250}.
- Rank: NATR14 (base), NATR5, NATR30, ADV63 desc, DV2 asc, random (luck band).
- Slots {5,10,15,20}.
Robustness candidates declared now:
- `E_avg`: DV-percentile averaged over k in {2,3,5} x window {63,126,252}; entry if average < 10.
- `E_vote`: entry if at least 5 of those 9 DV-percentiles are < 10.
- `T_vote`: replace R126>5% with "at least 2 of R63, R126, R252 > 0" (SMA200 kept).
- `E_avg+T_vote`.
Exit menu (same entries): X0 Close>High[-1] (base); X1 DV2(126)>50; X2 Close>SMA5; X3 first up-close
(Close>Close[-1]); X4 X0 or 10-day time limit; X5 X0 or Close<SMA200; X6 IBS>0.9 or RSI2>90 (already failed on DV2
in March; control).
Trial count for deflation = this study's configurations + prior DV2 history (~15 variant files, 3 pakal studies,
deeper-dip 24 cells, liquidity test 3, the 5% change); reported explicitly.

## Promotion rules (decided before results)

"Better" change: (i) paired block-bootstrap P(dSharpe <= 0) < 0.05 on 2000-2026; (ii) dSharpe > 0 in P1, P2, P3;
(iii) plateau: nearest neighbours on each changed axis also dSharpe > 0; (iv) deflated Sharpe >= 0.95 with the
full trial count; (v) turnover not > +25% and capacity not lower; (vi) still better under +5 bps/side.
"Robustness replacement" (E_avg, E_vote, T_vote, source 0%): non-inferior if dSharpe >= -0.05, bootstrap 5th pct
>= -0.15, max DD not worse by > 3pp, each period's Sharpe within -0.10 of baseline. Non-inferior + fewer fitted
choices => preferred.
Holdout: a finalist must have positive Sharpe in 1991-1999 and not trail the baseline by > 0.15 Sharpe there.
If nothing passes, the verdict is "keep", and that is a valid result.

## MOC (same-day close) test

- Decision at 15:45 New York (MOC entry cutoff 15:50), fill at the official close, same costs.
- Layer 1: closed-form flip distance of every entry/exit/trend decision vs the measured 15:45->close basis.
- Layer 2 (exact, 2021-01-04 -> latest): Alpaca SIP 15-min bars, all S&P 500 members; decision from price,
  high and low up to 15:45; fill at the official close.
- Layer 3 (2000-2020 model): replace each stock-day's close/high/low with a 15:45 state resampled from the 2021+
  sample, matched on volatility band and close-location band; many seeds. Validation: applied to 2021+ it must
  match layer 2 (CAGR within 1pp, daily return correlation >= 0.9).
- Gate: MOC viable if its Sharpe >= next-open Sharpe - 0.05 and CAGR >= next-open CAGR - 1pp, in layer 2 and in
  layer 3 (2000-2020).

## ETF block (owner request, 2026-09-25)

Frozen DV2 rules (wired parameters, 10 slots, no floor, next open) on a fixed liquid equity-ETF list declared
here, eligible from their own history (warm-up 252 days):
sectors XLB XLE XLF XLI XLK XLP XLU XLV XLY XLRE XLC; industries XBI IBB SMH SOXX KRE KBE XHB ITB XRT XOP OIH XME
GDX IYT IGV ITA IHI XSD XPH; countries EWJ EWG EWU EWC EWA EWZ EWH EWT EWY EWW EWS EWP EWQ EWI EWL EWN FXI INDA
EZA EEM EFA; broad/style SPY QQQ IWM DIA MDY IWD IWF IWN IWO. Symbols Norgate lacks are dropped (data, not results).
Survivorship caveat: the list is today's ETFs. Viable diversifier if net Sharpe >= 0.5 (engine costs and
10 bps round trip), positive in P1-P3, daily correlation with stock DV2 < 0.8. Also sectors-only and
industries-only subsets.

## Universes

Exact rules on S&P MidCap 400 and S&P SmallCap 600 (mechanism check of the July transfer failure with the real
exit; not independent evidence).

## Book level (Phase 5)

At most 5 finalists through the real engine, then the growth-shelf book G3+MR (TAA 32 / NDX 32 / DV2 18 / HPI 18,
annual reset) against the owner's growth gates, stressed Calmar and capacity (house MOC limits).

## Amendment log

- A1 (2026-09-25, before any study result): Phase 1 replica validated trade-for-trade vs the engine for wired
  (25,086 fills), floor (23,184) and ADV rank (25,636); only the final day differs (one APO dividend absent from the
  engine's end-truncated load). Data quirk: Norgate returns VLO dividends as zero when the request starts in 1989;
  the cache takes every field from a 1998-start load from 1998 on (only VLO changed).
- A2 (2026-09-25, before any MOC result): Alpaca free SIP history reaches back to 2016 (91% member coverage on
  2016-03-01). The exact MOC layer is extended to 2016-01-04 -> 2026-09-24. Members without Alpaca bars on a day
  keep the final-close decision (flagged); coverage is reported by year and a full-coverage-years sensitivity run.
  Layer 3 model is then calibrated on 2016+ and validated on it; 2000-2015 is the modeled period.
- A3 (2026-09-25, download throughput, before any MOC result): Alpaca returns ~750 bars/s, so each session
  requests only names DV2 could act on at 15:45: members in the last 30 sessions whose close-based DV2(126) was
  < 35 on any of those sessions. Names outside the set keep the final-close state (they cannot enter; a holding
  outside the set is flagged). 30-min bars to cutoff-15 plus one 15-min bar ending at the cutoff (same state as
  15-min bars; verified on 2016-03-01).
- A4 (2026-09-25, after the grid): the deflated Sharpe as specified tests "edge > 0 after N trials", not "better
  than baseline"; every configuration scores ~1.0. It is reported as evidence on the edge itself; the paired
  bootstrap and per-period rules decide promotions.
- A5 (2026-09-25, before the holdout was run): no configuration passes the "better" rule. Finalists:
  F0 floor; F1 floor + ADV63 rank (non-inferior, dSharpe +0.11, p=0.04, fails only P3 by -0.015; fund capacity
  objective); F2 = F1 with 15 slots (post-hoc capacity combination, labelled as such); F3 E_vote (owner's
  ensemble idea; misses non-inferiority by 0.003 on the bootstrap 5th pct); F4 DV window 252 (non-inferior, all
  periods positive, p=0.09; Varadi's original length). ETF-industries DV2 passes the ETF gate and goes to the
  book as a separate pod candidate, not as a DV2 replacement.
- A6 (2026-09-25, after results, labelled exploratory): MOC gate failed in layer 2 for every version; layer 3 failed
  its validation (CAGR +4pp vs exact) and was not used. Post-result explorations, not promotable: entry-only /
  exit-only MOC hybrids; stricter liquidity floors (75th/90th ADV percentile); ETF ADV > $50M screen; the
  rewritten floor module's median definition (floor_median_all) to explain the F4 engine difference.
