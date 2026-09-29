# PREREG (frozen) - why ROC12, why 10 stocks, why these filters? NDX momentum pod parameter robustness

Frozen: 2026-09-26, before any grid cell below was computed. The freeze is the git commit that adds this file; every
later change to this plan is an amendment, dated and labelled, and cannot change a verdict already computed.
Research only. No live pod, release, scheduler, broker code or WIRED strategy file is modified. Code lives in
`scripts/research/`, results in `results/research/ndx_param_robustness_20260926/`. Strategies are subclassed or
re-implemented in memory only.

## 0. What is already known (disclosed)

- Stage 2 (`0_papers/index/review/stage2/ndx_split_bias/REPORT.md`): the backtest score B = ROC12 / ATR20 on
  CAPITALSPECIAL dollars leaks future split factors. The live pod trades **L** = ROC12 / ATR20 in decision-day
  dollars. Standalone and G3 results of B, L, S (= S20 = ROC12/NATR20) and R (= ROC12) are known.
- Stage 3 score redesign (`.../stage3/ndx_score_redesign/FINDINGS.md`, and `..._base/`): 8 scale-free scores
  (S20, S63, V252, M121V, BLENDV, TQ, ENS, R) were compared with L inside G3 under a frozen rule; none passed.
  Known G3 Sharpe of L by block: 2008-11 (proxy) 0.70, 2012-21 1.30, 2022-26 1.29, 2012-26 1.30, max DD -14.1%.
  S20: 0.87 / 1.33 / 1.06 / 1.24. L's recent edge rests on 2025.
- Shelf re-baseline (`.../stage3/shelf_rebaseline/FINDINGS.md`): true G3 is about 20%/yr, Sharpe 1.28-1.30.
- A July 2026 NDX correlation-penalty sweep (on the biased score B) found that raising N from 10 to 20 cut CAGR
  with no drawdown benefit. The repo contains older NDX sweep scripts (`strategies/momentum/run_*_sweep.py`,
  `run_ndx_vxn_roc_variant_suite.py`) whose results I have not read; they all ran on the biased score B.
- Nothing in the grids below has been computed on the scale-free score before this freeze. The anchor cell A0
  (defined in section 2) equals S20, whose results are known.

## 1. Question

The live NDX pod: point-in-time Nasdaq-100 members; stock close > SMA100; regime SPY > SMA200, else cash; rank by
ROC12 / ATR20; top 10 equal weight; total exposure x clip(22 / VXN, 0.25, 1); month-end decision; next-open fills.
For each design choice, is the live value on a broad plateau, or is there a clearly better and robust region?
Would any change beat the live pod (L) inside the G3 book?

## 2. Hard constraints

- **Scale-free only.** Every variant uses only features that are unchanged when all of a stock's prices are
  multiplied by one constant: returns, ratios of prices, ATR divided by price, return volatility, price-vs-SMA
  comparisons. No dollar-unit feature (no dollar ATR, no price level, no dollar-volume filter) enters selection or
  weighting. Consequence: every variant is download-date invariant (verified, section 7).
- **The incumbent is L**, the live score, re-computed exactly as stage 2 did (ATR20 in decision-day dollars,
  R(t) = Unadjusted Close / Close restatement). L is not a grid cell; it is the benchmark every candidate must beat.
- **Anchor A0** = the live design with L's score replaced by its scale-free twin:
  score ROC12 / NATR20, N = 10 equal weight, stock filter SMA100, regime SPY > SMA200, no rank buffer, month-end
  decision (offset 0), VXN target 22, floor 0.25, cap 1.0. A0 = S20 from stage 3.
- **One stage varies one design question; every other setting stays at A0.** Stages are NOT chained: no stage
  inherits another stage's winner. This keeps each grid small and avoids sequential optimisation.

## 3. Fixed for every cell (= the live configuration)

Data: Norgate, CAPITALSPECIAL for all prices, loaded through the repo's own
`get_vxn_scaled_atr_normalized_ndx_data` (history from 1999-01-01, trading from 2000-01-01, data to the latest
session, 2026-09-25). PIT Nasdaq-100 membership as-of the decision date. Decision at the last actual close of each
month (the repo's own rebalance schedule; first decision 2000-01-31), market orders filled at the next open.
Engine semantics (replicated, section 6): target shares = int(V_{t} x w_i / Close_{i,t}) sized on the decision
close with V_t = total value at the decision close; held names re-sized to target each rebalance; names leaving
the list sold in full; 2.5 bps slippage per side; commission max($1, $0.005 x shares) on engine share counts;
dividend cash ledger (entitlement day T credited before open T+1, 25% withholding, no reinvestment); no cash
interest; positions whose price disappears are liquidated at the last available close. Capital $100,000.
Ranking ties broken by symbol, as in the repo. Names with a missing score are ineligible.

Feature definitions (all at decision date t, using data up to and including the close of t):
- ME(k) = close at the k-th previous decision date (ME(0) = t). ROCk = ME(0)/ME(k) - 1.
  ROC12-1 = ME(1)/ME(12) - 1.
- NATRn = mean true range over the last n sessions (all n valid, repo TR formula) / Close(t).
- sigma63 = standard deviation of daily close-to-close returns over the last 63 sessions (at least 60 valid).
- SMAn filter = Close(t) > mean of the last n closes (n valid); "none" = no stock filter.
- Regime X > SMA200 = close of X at t above the mean of its last 200 closes; "none" = always invested.
- VXN scale = clip(target / VXN(t), floor, 1.0), VXN as-of t (latest close on or before t).

## 4. Staged grids (declared now; nothing else will be added without an amendment)

Stage 1 - **what to rank on** (21 cells). Score = numerator / denominator.
- Numerator axis, ordered by mean look-back horizon: ROC3, ROC6, B3612 = mean(ROC3, ROC6, ROC12), ROC9,
  B612 = mean(ROC6, ROC12), ROC12, ROC12-1.
- Denominator axis, ordered from no adjustment to the fastest risk estimate: none (score = numerator), NATR63, NATR20.
- A0 = (ROC12, NATR20).

Stage 2 - **how many and how to weight** (12 cells).
- N axis: 5, 8, 10, 15, 20, 30.
- Weighting rows: EW (equal weight 1/N of the VXN-scaled exposure) and IV (weights proportional to 1/sigma63 over
  the selected names, summing to the VXN-scaled exposure; a name with missing sigma63 gets the median sigma63 of
  the selected names; no single-name cap).
- A0 = (N 10, EW).

Stage 3 - **which filters** (12 cells).
- Stock filter axis: none, SMA50, SMA100, SMA200.
- Regime rows: none, SPY > SMA200, QQQ > SMA200 (QQQ stands for the Nasdaq-100; the price index gives the same
  trend signal up to dividends, so NDX is not run separately).
- A0 = (SMA100, SPY).

Stage 4 - **rank buffer and timing luck** (84 cells).
- Buffer axis b: 0, 2, 5, 10. A held name (in the previous decision's list) stays while it is eligible and ranked
  within N + b; free slots are filled by the best-ranked names not kept. b = 0 is the live rule.
- Offset axis k: -10 ... +10 sessions (21 schedules). Decision date = the month's last session shifted by k
  sessions; ROCk anchors use the same shifted schedule; fills at the next open. k = 0 is the live schedule.
- The offset is NOT a candidate parameter: nobody may pick the best rebalance day. Offsets measure timing luck.
  The buffer is judged by its median over the 21 offsets.
- Diagnostic (not a candidate): the incumbent L at all 21 offsets, to see how much of L's lead over A0 is
  timing luck.
- A0 = (b 0, k 0).

Stage 5 - **VXN scaler** (25 cells + 1 reference).
- Target axis: 18, 20, 22, 24, 26. Floor axis: 0, 0.125, 0.25, 0.375, 0.5. Cap 1.0 throughout.
- Reference (not in any neighbourhood): no VXN scaling (exposure 1.0).
- A0 = (22, 0.25).

Total: 21 + 12 + 12 + 84 + 25 + 1 = 155 grid entries on NDX, 151 distinct configurations (A0 repeats).
For multiplicity the trial count is 151 + the 8 scale-free scores of stage 3 = **159**.

## 5. Measurements

Standalone blocks (daily returns of the NAV): P1 2000-01-04..2011-12-31, P2 2012-01-01..2021-12-31,
P3 2022-01-01..2026-07-24, FULL 2000-01-04..2026-07-24 (end = last TAA return).
**G3** = 0.5 TAA + 0.5 NDX leg, rebalanced daily. TAA = `taa_rank_tqqq` from
`0_papers/index/review/stage3/rescreen_true_g3/book_daily_returns_true.csv` (from 2012-10-02); before that the
owner's synthetic proxy `results/research/portfolio/growth_shelf_v2_20260926/proxy_runs/splice_scaled/
taa_btal_tqqq__path.csv.gz` (returns from 2008-03-04). G3 blocks: G-P1 2008-03-04..2011-12-31 (proxy),
G-P2 2012-10-02..2021-12-31, G-P3 2022-01-01..2026-07-24, G-FULL 2012-10-02..2026-07-24,
G-LONG 2008-03-04..2026-07-24.
Metrics: CAGR (252 days/yr), Sharpe = mean/sd x sqrt(252) with no risk-free rate, max drawdown on the daily NAV,
Calmar. Costs: engine costs, and a stress with +5 bps per side (7.5 bps slippage). Turnover = both-sided traded
notional / mean NAV / year; share of the list replaced per rebalance.
Secondary (candidates and L only): commission re-priced on real share counts (engine shares / R(t)), as in stage 2.

Capacity proxy (L, A0, every stage candidate): for each rebalance order, participation = order notional /
ADV20, ADV20 = median of Close x Volume over the 20 sessions up to the decision date (the product is split
invariant). Participation scales with pod AUM. Report the AUM at which the 95th percentile and the maximum order
participation reach 1% and 5% of ADV20, full history and 2021-2026. This is a proxy, not the CapacityAnalysis v2
"Recommended Max".

Charts: heatmaps of every grid for standalone FULL Sharpe, G3 FULL Sharpe, and the minimum over G3 blocks of the
Sharpe margin versus L, with L's value and A0 marked; timing-luck distributions for A0, L and each buffer.

## 6. Engine replica and parity gate

The grids run on a fast replica of the engine written for this study (same data, same order semantics as section
3). Before any grid cell is computed it must pass a parity gate against real engine runs:
- the stage-2 engine run of L (`stage2/ndx_split_bias/outputs/variant_L_daily.csv.gz`) and the stage-3 engine runs
  of S20 and R: daily-return correlation >= 0.9999 and FULL CAGR within 0.05 pp, identical top-N lists;
- and, after the grids, new engine runs (repo engine, subclass that executes the replica's target weights) for at
  least 4 non-default cells covering N != 10, IV weights, a non-zero offset with buffer, and non-default VXN
  settings: same thresholds. The replica's selection code is the same function in both paths.
If parity fails, the failure is reported and fixed before any result is read.

## 7. Download-date invariance (verified, not assumed)

- V1: every stock's OHLC is multiplied by a random constant (log-uniform 0.01..100, volumes divided by it); every
  NDX cell's list must be identical on every decision date.
- V2: on every decision date t, each feature is recomputed from Norgate unadjusted bars restated into day-t units
  (NONE x R(t)/R(d)), i.e. what a download on day t shows; maximum relative difference versus the CAPITALSPECIAL
  computation is reported per feature, and the stage-1 grid's lists must be identical.
- L is invariant by construction (stage 2); B is shown for contrast.

## 8. Plateaus, candidates and the decision rule (frozen)

**Neighbourhood** of a cell = the cell and its immediate neighbours on each ordered axis (a box of up to 3 x 3,
clipped at the edges). Stage 2 and stage 3: neighbours only along the N / stock-filter axis, within the same
weighting / regime row. Stage 4: neighbours along the buffer axis; each buffer's value is its median over the 21
offsets. The no-VXN reference has no neighbourhood.

**Plateau value** of a cell = the median over its neighbourhood.

**Stage candidate C_s** = the cell with the highest plateau value of standalone FULL Sharpe (engine costs); ties go
to the cell nearest A0. For stage 4, C_s is the buffer level at k = 0 and its neighbourhood is the buffer
neighbours at k = 0 (what would actually be traded).

A stage candidate **replaces L** only if all four hold:
- R1 (G3, all blocks). For each of G-P1, G-P2 and G-P3, the neighbourhood median of G3 Sharpe is strictly
  greater than L's G3 Sharpe in that block.
- R2 (drawdown). The neighbourhood median of G3 max drawdown is not worse than L's by more than 2.0 pp, on G-FULL
  and on G-LONG.
- R3 (costs). R1 and R2 still hold when the candidate cells and L are all re-run with +5 bps per side.
- R4 (other universes). On PIT S&P 500 and on PIT Russell 1000 (identical rules, only the universe changes;
  standalone FULL, engine costs), the neighbourhood median Sharpe of the same cells is >= A0's Sharpe in that
  universe. If C_s = A0 this holds trivially and the stage verdict is "the live value sits on the plateau".

If no stage passes: **keep L** and recommend a forward shadow line: the stage candidate with the largest minimum
G3-block margin versus L (neighbourhood medians); A0 if every candidate is A0.
If one or more stages pass: the passing changes are combined into one configuration and evaluated once (labelled
post-selection). Recommend the combination if it also passes R1-R4; otherwise the single passing stage with the
largest minimum G3-block margin.

**Confidence labels** (do not change the rule):
- Deflated Sharpe ratio (Bailey and Lopez de Prado) of each stage candidate's standalone FULL daily returns, with
  N = 159 trials and the cross-trial variance of standalone FULL Sharpe over all NDX cells.
- White's Reality Check (stationary bootstrap, mean block 21 days, 2,000 draws, seed 20260926): p-value that the
  best NDX cell beats L in G-FULL Sharpe, over all 151 configurations; the same for standalone FULL.
- Paired stationary bootstrap of each candidate centre's G-FULL Sharpe minus L's, one-sided p.
If a replacing candidate is not significant after these corrections, the report says the switch rests on
implementability and robustness, not on a proven higher return.

**Walk-forward diagnostic** (reported, does not change the rule): the stage candidate re-chosen using only P1
(2000-2011) standalone Sharpe plateau values, then its P2, P3, G-P2 and G-P3 results versus A0 and L.

**Other-universe surfaces**: every stage grid is also run on S&P 500 and Russell 1000 (standalone), shown next to
NDX, with the rank correlation of cell Sharpe between universes.

## 9. Assumptions

- A1: CAPITALSPECIAL applies one factor to O/H/L/C of a bar (stage 2 verified it for NDX; V2 re-checks it).
- A2: Ordinary dividends are not in any score (price returns), as live.
- A3: G-P1 rests on the owner's synthetic TAA proxy and is weaker evidence than G-P2 and G-P3; the rule treats it
  like the other blocks.
- A4: The other universes keep the NDX pod's rules unchanged (SPY regime, VXN scaler); they test whether a
  parameter choice generalises, not what is best for those universes.
- A5: Live fills are not inspected (they are on the VPS). L is what a day-t snapshot computes.
- A6: One history. A 14-year G3 Sharpe has a standard error of about 0.27; block-level differences of a few
  hundredths are noise.
