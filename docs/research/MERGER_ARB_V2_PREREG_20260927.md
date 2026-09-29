# PREREG (frozen) - merger arbitrage from price behaviour, version 2: Russell 3000 coverage and a robust pin test

Frozen: 2026-09-27, before any version-2 cell was computed. Freeze evidence: SHA-256 and timestamp in
`results/research/merger_arb_v2_20260927/prereg_freeze.json` (no git commit). Later changes are dated, labelled
amendments. Research only; nothing live changes. Code: `scripts/research/merger_arb_v2_20260927/` (may import the
`new_pod_search_20260927` and `trend_breakout_20260927` packages), tests `tests/test_research_merger_arb_v2_20260927.py`,
results `results/research/merger_arb_v2_20260927/`. Designed by the lead.

## 0. What is already known (disclosed; this is a follow-up hypothesis)

- **Why version 2 exists.** It follows `NEW_POD_SEARCH_PREREG_20260927.md`, whose verdict was that no new pod passed.
  The version-1 detector (family M: Russell 1000; jump >= 10-20% on a 5x Turnover shock, then the mean of TR/Close over
  5-10 sessions <= 0.6-1.5%) proved accurate but too narrow:
  - 78 events a year passed the jump and volume tests (J 15%), but only 2.4-7.4 a year passed the pin test.
  - Anchor M0 made 3.8 entries a year and held 1.2 positions on average (exposure 6%).
  - 87-90% of entries ended in a completed deal within 252 sessions.
  - Completed deals averaged +2.6%; breaks averaged -7.3% (worst episode -25%).
  - Beta 0.00, correlation 0.04 with TAA and 0.09 with L.
  - Its book beat the T-bill control by only +0.002..+0.007 Sharpe and failed the cross-half test by 0.0003.
  Version 2 was designed after seeing those coverage and precision diagnostics. It therefore tests one idea - broaden
  coverage without losing precision - and its evidence is weaker than a first test.
- **Book controls** (official pod model, annual reset, each window its own run; addition slot = TAA 0.5, L 0.25,
  X 0.25):

  | Slot | 2008-11 | 2012-21 | 2022-26 | 2012-26 Sharpe / MaxDD | 2008-26 Sharpe / MaxDD |
  |---|---|---|---|---|---|
  | none (G3) | 0.689 | 1.303 | 1.257 | 1.288 / -14.1% | 1.167 / -15.3% |
  | BIL (T-bills, total return) | 0.728 | 1.334 | 1.437 | 1.365 / -11.7% | 1.226 / -14.3% |
- **Merger-arbitrage ETFs.** MNA (from 2009-11-17, about $0.8M a day) is the only investable merger-arbitrage ETF with a
  usable history. It is a reference row here, not a candidate. Nothing about MNA's performance has been computed yet.
- **Literature.** Mitchell and Pulvino (2001): about 4%/yr excess return, low beta in calm markets and short-put-like in
  crashes. Deal counts are much larger among small caps.

## 1. Question

Can a price-only detector of pending cash takeovers, applied across the Russell 3000, hold enough deals to fill the
book's quarter slot and beat T-bills in it?

## 2. Hard constraints

As in `NEW_POD_SEARCH_PREREG_20260927.md` section 2:
- causal features; next-open fills; PIT membership; no survivorship;
- scale-free features and native Turnover; long-only;
- engine semantics with historical share units; 2.5 bps slippage per side;
- the idle-cash sweep into BIL total return is the primary evaluation; no-sweep results are reported alongside.

## 3. Data

- Universe: PIT Russell 3000 = Russell 1000 or Russell 2000 member at day d (Norgate constituent histories), 1999-2026.
- For memory reasons, detection streams symbol by symbol. Full daily panels are built only for symbols with at least
  one confirmed event, plus SPY, BIL and MNA.
- Windows and blocks as in the new-pod PREREG section 3 (END = 2026-08-19); TAA and L legs as there.

## 4. Family V - broad price-detected merger arbitrage (14 cells)

**Event at session d** (data through d; PIT Russell 3000 member at d):
- jump: Close_d / Close_{d-1} - 1 >= J;
- volume shock: Turnover_d >= 5 x median(Turnover_{d-60..d-1}), with 60 valid positive values;
- liquidity (REL25): ADV20_{d-1}, the median Turnover over d-20..d-1, is at least the 25th percentile of that measure
  among the Russell 3000 members at d.

**Pin window** s = d+1..d+W. All of these must hold:
- every bar has Turnover > 0;
- the median of |Close_s / Close_{s-1} - 1| <= theta;
- the maximum of |Close_s / Close_{s-1} - 1| <= 3%;
- the minimum Close >= 0.97 x Close_d.
Define pin_ref = the median Close over the window, fixed for the position's life.

**Entry**
- Decision at the close of d+W; fill at the next open if a slot is free and the name is not held.
- Budget b = min(V_t / K, (C_t + P_t) / n_t); raw shares = trunc(b / U_t).
- If slots are full, the name waits in a persistent FIFO queue, ordered by confirmation date, then median |r|, then
  symbol.
- A queued name is dropped when any of these occurs:
  - its series ends;
  - any close since confirmation is <= 0.95 x pin_ref;
  - 60 sessions have passed since confirmation.

**Exits**
- terminal liquidation at the last close when the series ends (the deal closes);
- break stop: close <= 0.95 x pin_ref, sold at the next open;
- time stop: sold at the next open after 252 sessions held.
- One position per symbol; re-entry only on a new event after the exit.
- No regime gate, no scaler, no re-sizing.

**Grids**
- Stage P (event x pin): J in {10%, 15%} x theta in {0.3%, 0.5%, 0.8%}, with W 5 and K 10 (6 cells).
- Stage S (speed x slots): W in {3, 5, 10} x K in {5, 10, 20}, with J 15% and theta 0.5% (9 cells).
- Anchor V0 = J 15%, theta 0.5%, W 5, K 10 (in both stages); 14 distinct cells.

**Sensitivities** (reported, not candidates): V0 with terminal proceeds x 0.995; x 0.99; with no break stop; with a 10%
break stop.

**Halves (R4).** The same cells run on events of Russell 1000 members at d, and on events of Russell 2000-only members
at d. These are disjoint.

**References** (labels only): MNA in the slot (from 2009-11-17; blocks G-P2, G-P3 and G-FULL); version-1 anchor M0.

## 5. Measurements

- **Standalone, with and without the sweep:** CAGR, Sharpe, max DD, beta to SPY, and correlation with TAA and with L.
  Also events, confirmations and entries per year; mean and maximum positions held; exposure; holding periods.
- **Precision and P&L by exit:** share of entries ending in a terminal liquidation within 252 sessions (diagnostic only).
  P&L by exit class, mean per completed deal and per break, and fills versus break levels.
- **Books:** by window with engine costs; +5 bps per side on X and L (TAA and BIL unchanged).
- **Capacity proxy:** the largest and p95 order against ADV20 from Turnover, 2021-26 and full history.
- **Small-account economics:** commission share of P&L at a $25k pod (K 10).
- **Charts:** heatmaps per stage; equity and drawdown of the candidates against C_BIL, G3 and MNA.

## 6. Parity and tests (before any grid cell is read)

- The real engine replays the replica's intents for V0 on the Russell 3000 (a pricing frame of the traded names). It
  must match to daily-return correlation >= 0.9999, CAGR within 0.05 pp, and identical positions.
- Unit tests cover:
  - the median-based pin and the max-move rule;
  - the queue-drop rules;
  - streaming detection equal to panel detection on a synthetic set;
  - the Russell 3000 membership union;
  - the sweep;
  - the control books reproducing the table above.
- If anything fails, it is fixed before any grid cell is run or read.

## 7. Invariance

- V1: random per-stock OHLC constants leave events, confirmations, positions and pre-fee NAV identical.
- V2: event and pin features recomputed from NONE bars restated to day t, for all confirmations plus 200 random events,
  must be identical up to float rounding.

## 8. Candidates and decision rule (frozen; same rule as the new-pod PREREG section 8)

- **Neighbourhoods.** Stage P: the cells within one theta step, in both J rows. Stage S: the 3 x 3 box, clipped.
  Plateau value = median over the neighbourhood.
- **Candidates.** Per stage, the highest plateau value of standalone FULL Sharpe with the sweep; ties go to the cell
  nearest V0.
- **R1.** Candidate-book Sharpe strictly above C_BIL in each of G-P1, G-P2, G-P3.
- **R2.** Candidate-book max DD no worse than C_BIL's by more than 2.0 pp, on G-FULL and G-LONG.
- **R3.** R1 and R2 also hold with +5 bps per side.
- **R4.** On each half (Russell 1000 events; Russell 2000-only events), candidate-book G-FULL Sharpe > C_BIL's.
- **R5.** In 2021-26 the largest order stays below 5% of ADV20 up to a pod AUM of at least $5M.
- **Labels.** Whether it also beats G3, C_SPY and the MNA slot (2012-26); whether it meets the owner's gates; the
  no-sweep result against the cash-0% slot; the terminal-proceeds sensitivities.
- **If a candidate passes:** recommend a forward paper line first; say that the evidence is a follow-up to a seen
  version.
- **If none passes:** say so; no further merger-arbitrage variant is tried in this session.
- **Confidence.**
  - Reality Check over the 14 cells versus C_BIL (stationary bootstrap, block 21, 2,000 draws, seed 20260927).
  - Paired bootstrap of each candidate versus C_BIL.
  - DSR with N = 377, a lower bound: 327 + 36 + 14.
  - Walk-forward: candidates re-chosen on P1 only.

## 9. Assumptions

- A1: Terminal liquidation at the last close equals the consideration of a completed cash deal. The x 0.995 and x 0.99
  sensitivities bound small mismatches.
- A2: The sweep is frictionless and daily.
- A3: Small-cap deals have wider spreads, more breaks and thinner trading; the Russell 2000 half measures this
  separately.
- A4: No deal database is used; detection errors are part of the result.
- A5: G-P1 uses the synthetic TAA proxy. One history.
