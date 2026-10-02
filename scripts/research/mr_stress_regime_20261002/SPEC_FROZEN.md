# Stress-gated mean reversion (frozen 2026-10-02, before any gated result)

## Question
Owner (2026-10-02): MR edge per trade is near zero in calm markets since 2010 (Nagel 2012; MR_BEYOND_DV2: −4 to +5 bps
per trade in low/mid VIX vs 21–33 bps in high VIX; DV2 study: 32%/yr below the S&P 500 SMA200 vs 19% above, 8%/yr in
the calmest volatility tercile). Does taking new MR entries only (or mostly) in stress improve DV2 enough to matter
in the book, against T-bills in the same slot?

## Known before freezing (disclosed)
- The regime facts above (descriptive, from earlier studies on overlapping data). No gated DV2/QPI run exists.
- DV2 vs T-bills in the quarter slot: roughly a tie in 2022–26 (new-pod search, 2026-09-27).

## Strategies (replicas; generic runner `qpi_moc_20261002/loc_lib.py`, parity checked: DV2 1.0687 vs 1.0693)
- Primary: DV2 wired (DV2(126) < 10, Close > SMA200, 126-day return > 5%, NATR14 rank, 10 slots, exit Close > High_{t-1}).
- Cross-check (R4): QPI (`strategy_mr_qpi_ibs_rsi_exit` rules).
- Engine costs (2.5 bps + $0.005/share, $1 min); stress +5 bps per side.

## Gates (state known at close t; they apply only to NEW entries decided at close t; exits are never gated)
- V20: VIX_t > 20.
- VREL: VIX_t > median(VIX_{t-251..t}).
- MKT: S&P 500 price index < its SMA200.
- ANY: V20 or MKT.
Modes: OFF (no new entries when the gate is closed), HALF (entries when closed get half the slot budget).
8 variants per strategy, all reported. No other thresholds.

## Cash
Idle cash earns T-bills in every variant AND the ungated baseline: BIL total return (book windows, as the new-pod
search); DTB3 (lagged one day) for standalone windows before BIL exists (2007-05-31).

## Book (official pod model, annual reset, each window its own run; new-pod-search code)
Candidate {TAA 0.5, L 0.25, X 0.25}; controls C_BIL (X = BIL) and C_DV2 (X = ungated DV2 with the cash sweep).
Blocks G-P1 2008-03-04..2011, G-P2 2012-10-02..2021, G-P3 2022..2026-08-19; G-FULL, G-LONG.

## Decision rule (per DV2 variant)
- R1: book Sharpe > max(C_BIL, C_DV2) in each of G-P1, G-P2, G-P3.
- R2: book max DD not worse than C_BIL's by more than 2.0 pp on G-FULL and G-LONG.
- R3: R1 and R2 hold with +5 bps per side on X (and the stored stress L).
- R4: the same gate and mode on QPI raise its standalone Sharpe (T-bill cash) over ungated QPI, 2004–2026.
A variant passes only with R1–R4. With 8 variants, a single pass is reported with that count.

## Reported, not gating
- Standalone Sharpe / CAGR / DD, 2000–26 and blocks 2000–09, 2010–19, 2020–26.
- Locked 1991–1999 run (DV2) and 1995–2003 (QPI), once, after the main results. The 1990s edge was large in calm
  markets, so a gate may fairly lose there; this tests whether the regime story is a post-2010 feature.
- Per-trade market-adjusted return of ungated DV2/QPI by gate state at entry, per block (descriptive).
- Share of time each gate is open; trades per year; exposure.
