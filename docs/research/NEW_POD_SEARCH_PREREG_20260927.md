# PREREG (frozen) - a new pod for the growth book: price-detected merger arbitrage and return seasonality

Frozen: 2026-09-27, before any grid cell below was computed. Freeze evidence: SHA-256 and timestamp in
`results/research/new_pod_search_20260927/prereg_freeze.json` (no git commit; files untracked in worktree
`nice-banzai-a4e788`). Later changes are dated, labelled amendments that cannot change a computed verdict. Research only:
no live pod, release, scheduler, broker code, released YAML or WIRED strategy file is modified. Code:
`scripts/research/new_pod_search_20260927/` (it may import `scripts/research/trend_breakout_20260927/` and
`ndx_param_robustness_core.py`), tests `tests/test_research_new_pod_search_20260927.py`, results
`results/research/new_pod_search_20260927/`. Designed by the lead (Opus 5.5).

## 0. What is already known (disclosed; this weakens the evidence)

- **The owner's book.** G3 = 0.5 TAA (`taa_btal_tqqq`) + 0.5 NDX pod L, official pod model (annual reset).
  2012-10-02..2026-08-19: Sharpe 1.288, max DD -14.1%; blocks 2008-11 / 2012-21 / 2022-26: 0.689 / 1.303 / 1.257. Owner
  gates: Sharpe >= 1.35 and DD >= -20% on 2012-26.
- **Trivial slot controls, computed by the lead before this freeze** (addition role = TAA 0.5, L 0.25, slot 0.25; official
  pod model; each window its own run):

  | Slot | 2008-11 | 2012-21 | 2022-26 | 2012-26 Sharpe / MaxDD | 2008-26 Sharpe / MaxDD |
  |---|---|---|---|---|---|
  | none (G3) | 0.689 | 1.303 | 1.257 | 1.288 / -14.1% | 1.167 / -15.3% |
  | cash at 0% | 0.722 | 1.325 | 1.357 | 1.334 / -11.7% | 1.201 / -14.5% |
  | BIL (T-bills, total return) | 0.728 | 1.334 | 1.437 | 1.365 / -11.7% | 1.226 / -14.3% |
  | SPY (total return) | 0.628 | 1.368 | 1.325 | 1.354 / -13.3% | 1.198 / -20.6% |

  So shrinking L already beats G3. The yesterday-passing breakout leg (1.340 in this slot) does not beat T-bills. Any
  new pod must beat **T-bills in the same slot**. That is this study's primary control.
- **Tried and not promoted** (knowledge base, 112 studies + the 26-27 Sep studies):
  - stock and ETF momentum / trend / breakout, including per-position stops;
  - short-horizon mean reversion (DV2/HPI; a liquidity premium paid in crises);
  - calendar flows (end of month, turn of month, holidays);
  - overnight stock selection; volatility hedges; low-volatility; factor momentum; country momentum;
  - index-deletion rebounds (faded after 2013); S&P 500 post-inclusion fade (needs shorts).
  Nothing in this repo has tested merger arbitrage or same-calendar-month return seasonality.
- **The merger-arbitrage idea came from data already seen.** Yesterday's breakout leg (quiet-breakout rank) bought 43
  pending takeover targets, held them to deal completion, and made typically +1% to +10% each. The idea was therefore
  prompted by a post-hoc observation. The premium itself is documented independently: Mitchell and Pulvino (2001), about
  4%/yr excess, low beta in calm markets and a short-put profile in crashes.
- **Seasonality priors.** Heston and Sadka (2008) and Keloharju, Linnainmaa and Nyberg (2016) find same-calendar-month
  return persistence; it persisted out of sample after 2002 in broad US data. Some practitioners report it faded in
  large caps.
- **The earnings-announcement premium was considered and dropped** before any test: the literature reports it
  disappeared outside the largest caps after 2004-2011, and it needs about 100%/month turnover.

## 1. Question and mechanisms

Is there a new pod that, placed in the book's quarter slot, beats T-bills in that slot robustly?
- **Family M (merger arbitrage from price behaviour).** After a cash takeover is announced, the target jumps and then
  trades "pinned" just below the offer until the deal closes; holders earn the spread as pay for bearing deal-break
  risk. Detected purely from daily bars: a jump on a volume shock followed by a collapse of daily ranges.
  - Expected: low equity beta, positive carry, occasional -15..-40% single-name breaks, losses concentrated in crashes.
  - Value to the book: low correlation with TAA and L.
- **Family S (return seasonality).** Stocks that did well in a calendar month in past years tend to do well in the same
  month again (seasonal risk premia, recurring flows and information cycles). Two forms keep equity beta out of the
  slot test:
  - GATED: invested only when SPY > SMA200, else cash;
  - HEDGED: half in stocks, half in SH, the -1x S&P 500 ETF.

## 2. Hard constraints

- Causal features only (data through the decision close t; fills at open t+1); PIT membership as-of t; no survivorship.
- Every feature is scale-free (returns, ratios, same-day comparisons) or uses native `Turnover` for dollar liquidity; no
  dollar-price feature.
- Long-only positions (SH is held long). Engine semantics with historical share units, as in
  `TREND_BREAKOUT_PREREG_20260927.md` section 3: 2.5 bps slippage per side; $0.005/share commission on raw-equivalent
  shares, $1 minimum; dividend ledger; terminal liquidation at the last close; no cash interest inside the engine.
- **Idle-cash sweep (primary evaluation).** Pod return_t = engine return_t + max(cash_{t-1}, 0) / V_{t-1} x BIL total
  return_t. This models parking idle cash in T-bills and makes the pod comparable with the T-bill control. It is a
  frictionless post-processing adjustment; the no-sweep returns are reported next to it.
- One stage varies one question at its anchor; no chaining.

## 3. Fixed settings and data

Norgate through the repo loaders:
- universe caches from `trend_breakout_20260927` (1999-01-04..2026-09-25, with native Turnover), rebuilt if needed;
- month-end closes from 1990-01-01 for seasonality look-backs (feeding only the scores);
- SH (from 2006-06-21) as a tradeable symbol (CAPITALSPECIAL, dividend ledger);
- BIL and SPY TOTALRETURN for the sweep and the controls;
- the SPY close for the gate.

Standalone windows:
- P1 2000-01-04..2011-12-31, P2 2012-01-01..2021-12-31, P3 2022-01-01..2026-08-19, FULL 2000-01-04..2026-08-19;
- HEDGED cells start 2006-07-03, when SH exists; their FULL is 2006-07-03..2026-08-19.

Book windows, official pod model (`fund_menu_20260923/common.py::book_return_ser`, "annual"), each window its own run:
- G-P1 2008-03-04..2011-12-31 (TAA proxy), G-P2 2012-10-02..2021-12-31, G-P3 2022-01-01..2026-08-19;
- G-FULL 2012-10-02..2026-08-19, G-LONG 2008-03-04..2026-08-19.

Books and legs:
- TAA = `taa_btal_tqqq` (sleeve_series_incl_2008).
- L = the replica's live pod (equal to `ndx_atrfix`, trend study G1).
- Candidate book = {TAA 0.5, L 0.25, X 0.25}.
- Control C_BIL = {TAA 0.5, L 0.25, BIL 0.25}; references C_SPY (SPY total return in the slot) and G3.

## 4. Families and grids (declared now)

### Family M - merger arbitrage detected from price behaviour (primary Russell 1000; 12 cells)

Event at session d (all data through d; PIT Russell 1000 member at d):
- jump: Close_d / Close_{d-1} - 1 >= J;
- volume shock: Turnover_d >= 5 x median(Turnover_{d-60..d-1}) (60 valid);
- liquidity: ADV20 before the event (median Turnover d-20..d-1) >= the 25th percentile among that day's members (REL25).

Pin window s = d+1..d+W; all W bars must have Turnover > 0 (no halted or padded bars):
- pin_vol = mean over the window of TR_s / Close_s <= theta;
- hold: min Close over the window >= 0.97 x Close_d.

Entry and position management:
- Entry at the open of d+W+1 (decision at the close of d+W) if a slot is free and the name is not held.
- Budget b = min(V_t / K, (C_t + P_t) / n_t) per entry; raw shares trunc(b / U_t).
- Queue order: earliest confirmation first, then lowest pin_vol, then symbol.
- One position per symbol; re-entry only on a new event after the exit.
- No re-sizing, no regime gate, no VXN scaler.

Exits:
- terminal liquidation at the last close when the series ends (the deal closes);
- break stop: at close t, if Close_t <= 0.95 x pin_ref (pin_ref = median close over the pin window), sell at the next
  open;
- time stop: sell at the next open after the 252nd session held.

Grids:
- Stage M1 (event x pin): J in {10%, 15%, 20%} x theta in {0.6%, 1.0%, 1.5%}, with W 5 and K 20 (9 cells, ordered axes).
- Stage M2 (window x slots): W in {5, 10} x K in {10, 20}, with J 15% and theta 1.0% (4 cells).
- Anchor M0 = J 15%, theta 1.0%, W 5, K 20 (in both stages); 12 distinct cells.
- Sensitivities (reported, not candidates): M0 with no break stop; M0 with a 10% break stop; M0 with terminal proceeds
  x 0.99.
- Cross-checks: the same cells on the two disjoint halves of the Russell 1000 - members of the S&P 500 at d, and members
  not in the S&P 500 at d (mid caps).

### Family S - return seasonality (primary S&P 500; 24 cells)

At each repo month-end decision t (for next month m+1):
- Score = mean of the stock's monthly price returns in calendar month m+1 over past years, from month-end
  CAPITALSPECIAL closes.
- Horizons (ordered):
  - SE_1 = year -1 only;
  - SE_2_5 = years -2..-5 (at least 3 valid);
  - SE_1_10 = years -1..-10 (at least 5 valid);
  - SE_1_20 = years -1..-20 (at least 10 valid).
- Eligible: PIT S&P 500 member at t, REL25, finite score. Top N by score, ties by symbol. Monthly hold, next-open
  fills, historical share units.
- Forms (rows):
  - GATED: if SPY <= SMA200 at t, hold nothing (cash, swept); else top N equal weight, 1/N of NAV each.
  - HEDGED: 50% of NAV in the top N equal weight plus 50% in SH, rebalanced monthly, no gate.
- Grid: 4 horizons x N in {10, 20, 50} x 2 forms = 24 cells.
- Anchor per form: SE_1_10, N 20.
- Cross-checks: Russell 1000 members not in the S&P 500 (disjoint mid caps), and NDX.

Trial count: 12 (M) + 24 (S) = 36 configurations, one role. Prior trials, lower bound: 235 (trend PREREG) + 56 (trend
study) = 291. N for DSR = 327, a lower bound.

## 5. Measurements

Standalone (with and without the sweep):
- CAGR, Sharpe (mean / sd x sqrt 252, no risk-free rate), max DD, Calmar;
- beta to SPY; correlation with TAA and with L;
- turnover, mean exposure, positions, holding periods.
Books: the candidate and control books by window; engine costs and +5 bps per side on X and L (TAA and BIL unchanged).

Family M diagnostics:
- events and entries per year; exits by reason with P&L by reason; share of entries ending in terminal liquidation
  within 252 sessions (ex-post precision, diagnostic only); mean return per completed deal and per break;
- fills versus stop levels.

Family S diagnostics:
- score coverage; turnover; realized beta; timing luck of the two anchors at 21 rebalance offsets (diagnostic).

Capacity proxy: order $ / ADV20 (Turnover); pod AUM at which the 95th-percentile and the largest order reach 1% and 5% of
ADV20, full history and 2021-2026 (market-on-open capacity is lower).

Charts: heatmaps per stage (standalone FULL Sharpe; book G-FULL Sharpe minus C_BIL); equity and drawdown of candidates
against C_BIL, G3 and C_SPY.

## 6. Parity and tests (before any grid cell is read)

The replica executes policy intents, and the real engine replays the same intents through a research subclass
(historical share units). The replay must match to daily-return correlation >= 0.9999, CAGR within 0.05 pp and
identical daily positions for:
- M0 on the Russell 1000;
- the GATED anchor on the S&P 500;
- the HEDGED anchor on the S&P 500 (with SH).
Unit tests on synthetic panels cover:
- event and pin detection, including halted and padded bars rejected;
- queue order, budgets, break and time stops;
- terminal liquidation;
- seasonality scores against a hand computation;
- the gate, and the SH hedge weights;
- the sweep arithmetic;
- the book controls reproducing the section 0 table.
If any check fails, it is fixed before any grid cell is run or read.

## 7. Download-date invariance

- V1: random per-stock OHLC constants (Turnover and Unadjusted Close unchanged). All decisions, positions and pre-fee
  NAV must be identical.
- V2: decision-day restatement from Norgate NONE bars for all month-ends plus 200 random sessions. Event, pin and score
  features must match (float rounding excepted).

## 8. Candidates and decision rule (frozen)

Neighbourhoods:
- M1: 3 x 3 box, clipped at the edges;
- M2: the 2 x 2 box (all four cells);
- S: 3 x 3 box (horizon x N), within the same form.
Plateau value = median over the neighbourhood.

Candidates (chosen on standalone FULL Sharpe with the sweep, so the book rule is not the selection metric):
- M: per stage, the cell with the highest plateau value; ties go to the cell nearest M0.
- S: per form, the same; ties go to the anchor.

A candidate is recommended only if all hold (neighbourhood medians throughout):
- **R1:** candidate-book Sharpe is strictly above C_BIL's in each of G-P1, G-P2 and G-P3.
- **R2:** candidate-book max DD is not worse than C_BIL's by more than 2.0 pp, on G-FULL and on G-LONG.
- **R3:** R1 and R2 also hold with +5 bps per side on X and L.
- **R4 (cross-checks):** on each cross-check universe, the candidate-book G-FULL Sharpe exceeds C_BIL's
  - M: both Russell 1000 halves;
  - S: Russell 1000 ex S&P 500, and NDX.
- **R5:** in 2021-2026 the largest order stays below 5% of ADV20 up to a pod AUM of at least $5M.

Reported labels, which do not change the rule:
- whether the candidate book also beats G3 and C_SPY;
- whether it meets the owner's gates (Sharpe >= 1.35 and DD >= -20% on G-FULL);
- the no-sweep result against the cash-0% slot.

If several pass: recommend the one with the largest minimum block margin over C_BIL. The report may present the two
families' passing candidates together as a post-hoc combined idea, but that idea is not tested here.

If none passes: no new pod is recommended. The forward shadow candidate is the one with the largest minimum block margin
over C_BIL, among those satisfying R2 and R5.

Confidence labels:
- White's Reality Check (stationary bootstrap, mean block 21, 2,000 draws, seed 20260927) of the best G-FULL book
  Sharpe difference versus C_BIL over the 36 configurations;
- paired bootstrap of each candidate versus C_BIL;
- DSR with N = 327, on standalone returns;
- walk-forward: candidates re-chosen on P1 only.
If a recommended candidate is not significant after these, the report says so.

## 9. Assumptions

- A1: Terminal liquidation at the last close equals the deal consideration for completed cash deals. Stock deals and
  mixed deals are mostly excluded by the pin filter, and when included the last close reflects their value.
- A2: The sweep is frictionless and daily. Real T-bill ETF trades cost a little, and interest on idle cash at IBKR is
  similar.
- A3: SH carries its expense ratio and daily-reset path. HEDGED cells start in 2006-07. The 50/50 hedge assumes beta 1.
- A4: G-P1 uses the synthetic TAA proxy.
- A5: No deal database is used. Detection errors (non-deal jumps, stock deals) are part of the strategy's measured
  result.
- A6: One history. A 14-year book Sharpe has a standard error of about 0.27.
