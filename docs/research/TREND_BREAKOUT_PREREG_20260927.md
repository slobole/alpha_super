# PREREG (frozen) - trend / momentum / breakout with movement-based exits: stop overlays on the live NDX pod, a daily breakout leg, and residual momentum

Frozen: 2026-09-27, before any grid cell below was computed and before any engine-parity run. Freeze evidence: SHA-256
and timestamp written by the lead to `results/research/trend_breakout_20260927/prereg_freeze.json` (no git commit; the
study files are untracked in worktree `nice-banzai-a4e788` at fb81e86). Every later change is a dated, labelled
amendment and cannot change a verdict already computed. Research only: no live pod, release, scheduler, broker code,
released YAML or WIRED strategy file is modified. Code: `scripts/research/trend_breakout_20260927/`, tests
`tests/test_research_trend_breakout_20260927.py`, results `results/research/trend_breakout_20260927/`. Strategies are
subclassed or re-implemented in memory only. Designed by the research agent (Fable 5.1) and reviewed by the lead
(section 10 records the lead's decisions).

## 0. What is already known (disclosed; this weakens the evidence)

Owner's book and incumbent (corrected after the fb81e86 leak fix, engine costs, Sharpe with no risk-free rate, 252 d/yr):
- G3 = 0.5 TAA (`taa_btal_tqqq`) + 0.5 NDX live pod L (`ndx_atrfix`), official pod model (pods compound independently,
  reset to target weights at each year-end; `scripts/research/fund_menu_20260923/common.py::book_return_ser`, "annual").
  2012-10-02..2026-08-19: CAGR 19.9%, Sharpe 1.29, max DD -14.1%; Sharpe 1.30 in 2012-21 and 1.26 in 2022-26. From
  2008-03-04 (synthetic TAA proxy before 2012-10-02): Sharpe 1.17, DD -15.3%; 2008-11 block 0.69. Owner gates: Sharpe
  >= 1.35 and DD >= -20% on 2012-26. G3 fails the Sharpe gate. Books that pass need daily mean-reversion pods
  (G3 + capsule 1.41, ladder_4 1.43); no monthly-only book passes.
- L alone 2012-26: 15.5% / 0.93 / -20.5%; from 2000: 12.1% / 0.77 / -29.3%. TAA alone 2012-26: 23.8% / 1.32 / -17.7%.
  MOSAIC (Russell 1000 momentum) is dead after the leak fix (Sharpe ~0.62).
- NDX parameter robustness (2026-09-26, frozen rule): keep L; 12-month lookback and N = 10 sit on NDX plateaus that do
  not transfer to S&P 500 / Russell 1000 (score-grid rank correlation -0.16 / -0.04); the scale-free twin A0
  (ROC12/NATR20) loses to L inside G3 only because of 2025; month-end is a lucky rebalance day (L standalone Sharpe
  0.66-0.79 across 21 days); buffer 2 changes nothing; VXN target is a risk dial. Post-hoc, not pre-registered: no stock
  filter and no regime gate inside G3 gives Sharpe 1.40 with DD 2.8 pp worse; averaging 21 rebalance days ("tranching")
  gives 1.32 / -16.1%. Reality Check p = 0.61 over 151 configurations. Shadow line: A0 + SMA200.
- Russell 1000 liquid momentum leg W-REL25 (2026-09-26): ties G3 (1.28 vs 1.30) with less drawdown (-11.8% vs -14.1%)
  and less return (16.0% vs 20.0%); fails R1 by -0.04 in 2012-21; three-leg book 0.5 TAA / 0.25 L / 0.25 W 1.31,
  -12.9% (descriptive). Corr(W, L) 0.75. (Those two studies used a daily-rebalanced 50/50 book.)
- Repo momentum sweeps (biased score B or legacy share units, all pre-fix): Clenow top-10 / vol63 monthly with index
  regime (2000-01..2026-06, engine costs): Sharpe 0.64 S&P 500 / 0.70 NDX / 0.66 R1000, DD -26% / -37% / -35%,
  turnover 10-13x; JT 12-1 top-20 sweeps (36 configurations); PTA winner continuation (4); NDX ROC-window suite
  (>= 9 variants: 1m, prior 1m, 3m, 12-1, 6-1, 3-1, blends); universe comparison (6 universes); NDX correlation-penalty
  sweep (N 20 cut CAGR with no DD benefit); EV/LRB-252 NDX ranking; smooth-trend suites; `strategy_mo_alpha23_breakout`
  (S&P 500, 20-day high-of-highs breakout with a 5-day time exit - a short-horizon family, no saved results found).
  None of these used per-position stops.
- Knowledge base (pakal), all research-only, none promoted:
  - Han-Zhou-Zhu trend factor: powers-of-two / EWMA / EWMAC (Russell 3000; confirmation Sharpe -0.05..0.29), fixed-N
    long-only (best 0.49, -59% DD), cross-universe (NDX 0.51, -63% DD; gate failed), risk overlays on R1000/NDX
    (inverse vol, 15% vol target, VIX/VXN scale: all rejected), forward shadow armed (NOT_READY).
  - TradeQuantiX USA trend/momentum translations: simple trend (ROC200 rank, stock and index SMA200, vol sizing;
    fails the 2020-25 benchmark gate; 2020-25 NDX 0.59, -37.5%), dual-rank trend (inverse NATR and ROC/dollar-ATR
    sleeves with a 25% trailing close exit; 60.7% stale NAV, terminal economics unresolved), moving-average votes
    (five log-SMA votes, daily 25% close stop; unresolved delisting rights), corrected momentum (monthly entry, daily
    negative-momentum exit; 29% stale NAV), momentum pullback (ROC150 + RSI3/ADX10 with next-open stops; promising but
    unverified terminal economics, demand 7.5% of ADV63). The recurring blocker there was "terminal economics": that
    engine left delisted names as stale NAV. The repo engine instead liquidates at the last available close (gap
    G-014); this study measures how much P&L that convention carries.
  - CMM-lite momentum on Russell 3000 (diagnostic; fixed-z fails); volatility-indicator probability overlays on NDX
    momentum (rejected; keep VXN); NDX variance-ratio overlay (rejected); multi-asset ETF trend allocation (TradeQuantiX
    dynamic allocation, diagnostic; DBMF liquidity); BTAL multi-asset trend (diagnostic); crisis-trend pod (forward
    hypothesis, tail-hedge role, not a growth leg); Quick5 ETF rotation and its offset/hysteresis/improvement studies
    (path-sensitive, nothing passed); Vardi adaptive momentum on ETFs and portfolios (diagnostic / shadow); SPY adaptive
    momentum regime (research candidate, single ETF); three-lens SPY regime filter (diagnostic).
- Stop-loss priors from the literature (stated before results): stops add value when returns show persistence /
  regime shifts and subtract value under mean reversion (Kaminski and Lo, 2014, "When do stop-loss rules stop
  losses?"); single-stock daily returns show short-term reversal, so tight stops are expected to hurt and loose stops to
  be roughly neutral in Sharpe while cutting the left tail. 52-week-high breakouts carry anchoring / underreaction
  evidence (George and Hwang, 2004). Residual momentum (Blitz, Huij and Martens, 2011) keeps momentum returns with
  roughly half the crash exposure by scoring only the part of returns the factors do not explain.
- Genuinely new here: (a) per-position movement-based exits on the live pod with the exact live semantics (historical
  share units, VXN scaler, month-end schedule, next-open fills) - never tested in this repo; (b) a daily breakout-entry /
  ATR-exit stock leg built on the house PIT universes with native-Turnover liquidity, next-open fills, engine terminal
  accounting and a frozen book-level rule; (c) residual momentum, a parked roadmap item never run. Nothing in the grids
  below has been computed before this freeze.

## 1. Question and mechanisms

Does a movement-based stop (ATR chandelier or percent trailing, decided on the close, exited at the next open) improve
the live NDX pod inside the owner's growth book? Does a trend / breakout stock leg with such exits lift the book, either
replacing the NDX leg or added beside it? Is residual momentum a better score for the same pod shell?

Mechanisms (frozen expectations):
- Family A (stop overlays on L): the pod holds ten names for a month regardless of path; a stop truncates single-name
  losses between rebalances (idiosyncratic blow-ups, post-earnings gaps) at the cost of whipsaw (a stopped name may be
  re-bought at the next month-end) and gap risk (the fill is the next open, not the stop level). Expected: tight stops
  lower Sharpe, loose stops change little, drawdown falls modestly; the book test decides.
- Family B (breakout entries, ATR exits): buying N-day closing-high breakouts in a market uptrend and holding until a
  volatility-scaled trailing exit captures continuation after new highs and gives trend-following's positive skew;
  costs are whipsaws and gaps. Expected standalone Sharpe below L; the value, if any, is lower correlation with TAA
  (Nasdaq beta) and with L, hence a book effect.
- Family C (residual momentum): scoring on 36-month market-model residuals removes the market bet that produces part
  of momentum crashes; expected similar return with lower drawdown than total-return momentum in the same shell.

## 2. Hard constraints

- Causal only: every feature at decision close t uses data up to and including t; fills at open t+1. PIT membership
  as-of t. No survivorship.
- Scale-free or decision-date units: every selection, stop and sizing feature is invariant to a constant rescaling of
  a stock's price history (ratios of CAPITALSPECIAL prices, ATR divided by price or compared with prices of the same
  day, closing-high comparisons, returns) or is expressed in decision-date dollars (L's ATR20 x U_t / Close_t, whole-share
  rounding on Unadjusted Close). Dollar liquidity only from native `Turnover`. Verified in section 7.
- Incumbent: L, the live pod, re-computed exactly (score ROC12 / ATR20 in decision-day dollars, SMA100 stock filter,
  SPY > SMA200 regime else cash, top 10 equal weight, exposure x clip(22 / VXN, 0.25, 1), month-end decision, next-open
  fills, historical share units). L is the benchmark; it is not a grid cell.
- One stage varies one design question at its anchor; stages are not chained.
- Stops are decided on the close only; a gap through the stop is filled at the next open. Intraday stop orders, portfolio
  level stops, cooldowns, shorting, leverage and weekly schedules are out of scope.

## 3. Fixed for every cell

Data: Norgate through the repo loaders (`get_vxn_scaled_atr_normalized_ndx_data` with the index name changed; universes
rebuilt into this study's cache with `Turnover`). CAPITALSPECIAL Open/High/Low/Close for fills, marks, ATR, moving
averages, closing highs and returns; `Unadjusted Close` U_t for whole-share rounding, commission units and L's dollar ATR;
native `Turnover` for ADV20; `Dividend` for the cash ledger; $VXN close as-of t; SPY close for the regime. History from
1999-01-01, trading calendar from 2000-01-03, data to the latest session (2026-09-25). PIT membership = latest
constituent row on or before t (the audited forward-filled matrix). Universes: NDX (Nasdaq 100 Current & Past), SP500,
R1000.

Engine semantics replicated (section 6): decision at close t (= engine `previous_bar`), market orders filled at open
t+1 at Open x (1 +/- 0.00025); commission max($1, $0.005 x raw-equivalent shares) where raw shares = ledger shares /
k_E, k_E = U_E / Close_E on the execution bar (liquidation anchor bar for terminal liquidations); historical share units:
a target of D dollars becomes trunc(D / U_t) raw shares mapped to trunc(D / U_t) x k_t ledger units, k_t = U_t / Close_t
at the decision close; an order that would fill zero raw shares is cancelled without commission; dividends: entitlement
day t credited before open t+1 at 75% (25% withholding), no reinvestment; no cash interest; negative cash is allowed by
the engine (implicit 0% borrow) but the sizing rules below keep it near zero; a held name whose Open or Close is
missing on bar t is liquidated at its last available close (no slippage, commission charged) and its pending orders are
cancelled (gap G-014, "terminal liquidation"). Capital $100,000. Ties broken by symbol ascending. Names with a missing
feature are ineligible.

Definitions at session t (all trailing, ending at t inclusive):
- TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|); ATR20_t = mean of the last 20 TR (all valid);
  NATR20_t = ATR20_t / Close_t; ATR20$_t = ATR20_t x U_t / Close_t (L's decision-day dollar ATR).
- SMA_n pass: Close_t > mean of the last n closes (n valid). Regime: SPY_t > SMA200(SPY)_t.
- ME(j) = the j-th previous month-end decision close on the repo schedule (ME(0) = the current month-end);
  ROC12 = ME(0)/ME(12) - 1 (monthly, as live). ROC252_t = Close_t / Close_{t-252} - 1 (daily).
- HH_N(t) = max(Close_{t-N}, ..., Close_{t-1}) (N valid closes; excludes t).
- ADV20_t = median of Turnover over the 20 sessions ending at t (20 valid, positive). Liquidity REL25: exclude PIT
  members whose ADV20 is below the 25th percentile of ADV20 among that day's PIT members with a valid ADV20.
- VXN scale s_t = clip(22 / VXN_t, 0.25, 1), VXN as-of t.
- V_t = total value at close t (cash + ledger shares x Close_t); C_t = cash at close t; P_t = close-t value of the
  positions that will be sold at open t+1.
- Entry session e_i = the session on which a position in i goes from 0 to > 0 shares (its fill day).
  HWM_{i,t} = max(Close_{i,s}, s = e_i..t) - the highest close since entry, including the entry day.
- Stop CH-k fires at close t if Close_{i,t} <= HWM_{i,t} - k x ATR20_{i,t}. Stop PT-x fires if
  Close_{i,t} <= (1 - x) x HWM_{i,t}. Both compare quantities in the same day-t adjusted units, so they are scale-free.
  A stop that fires at close t sells the whole position at open t+1 (market). Stops are checked on every session
  from e_i on (on e_i they cannot fire). A name that fires and a name that was already stopped are not re-entered by
  the stop logic; re-entry happens only through the family's own entry rule.

## 4. Families and grids (declared now; nothing else without an amendment)

### Family A - stop overlays on the live pod L (NDX; 16 cells + L)

Base = L exactly (section 2). Overlay on every held name: stop type x level x freed-slot policy.
- Type / level axis (ordered, tight to loose): CH-k, k in {2, 3, 4, 5}; PT-x, x in {10%, 15%, 20%, 25%}.
- Freed-slot policy rows:
  - CASH: after a stop the slot stays in cash until the next month-end decision, which selects the top 10 as usual
    (a stopped name is eligible again there if it ranks). If a stop fires at a month-end close, that name is excluded
    from that decision only (sold at open t+1, not re-bought that day).
  - REFILL: on the same open as the stop exit, each freed slot is refilled with the best eligible name not held and not
    stopped that day, ranked by the daily score score_d(i,t) = (Close_{i,t} / Close_{i,ME12(t)} - 1) / ATR20$_{i,t},
    where ME12(t) is the month-end decision close 12 months before the month of t (the anchor the coming month-end
    would use); eligibility at t = PIT member, SMA100 pass, regime pass, finite score. Refill budget per name
    b_t = min(V_t x s_m / 10, (C_t + P_t) / n_t), s_m = the VXN scale of the current month's decision, n_t = number of
    refills that day; raw shares trunc(b_t / U_{i,t}); a zero-share refill is skipped. No refill when the regime fails
    at t. Refill positions are re-sized at the next month-end like any other.
- Cells: 2 types x 4 levels x 2 policies = 16, plus L (no stop) as the reference row entry of each row.
- Monthly rebalance unchanged: at the month-end close targets = trunc(V_t x s_t / 10 / U_{i,t}) raw shares for the
  top 10 (a selected name whose slot buys zero shares is dropped, as live); held names re-sized; names leaving sold.
- Book role: replacement only (G3 with L-plus-stop versus G3 with L).

### Family B - daily breakout entries with ATR exits (primary S&P 500; cross-checks NDX, R1000; 17 cells)

Daily state machine, K slots, no re-sizing after entry, no monthly rebalance.
- Entry signal at close t (all with data through t): PIT member and REL25 liquidity pass; regime SPY_t > SMA200;
  stock trend Close_t > SMA200_t; breakout Close_t > HH_N(t); not held; finite rank score.
- Rank when signals exceed free slots: R1 = ROC252_t / NATR20_t descending; R2 = NATR20_t ascending (quiet breakouts).
  Ties by symbol.
- Admission: free slots at close t = K - (positions held at t that are not exiting at open t+1); n_t =
  min(free slots, signals); the top n_t are bought at open t+1 with budget b_t = min(V_t / K, (C_t + P_t) / n_t)
  each, raw shares trunc(b_t / U_{i,t}); zero-share entries are skipped (the slot stays free). This keeps cash >= 0 up
  to slippage and commission.
- Exit: CH-k on HWM since entry, checked every close, sold at the next open. No time stop, no regime exit, no
  re-sizing; a name may re-enter on a later breakout after its exit (no cooldown). Delisting = terminal liquidation.
- Anchor B0: N = 100, k = 5, K = 20, rank R1, no VXN scaler, stock SMA200 filter, SPY entry gate, REL25.
- Stage B1 (entry x exit, 9 cells): N in {50, 100, 250} x k in {3, 5, 8}; both axes ordered.
- Stage B2 (slots x rank, 6 cells): K in {10, 20, 30} x rank in {R1, R2}.
- Sensitivity cells (reported next to B0, not candidates): S-a = B0 with the slot budget scaled by s_t (VXN 22/0.25);
  S-b = B0 with a regime exit (all positions sold at the next open when SPY_t <= SMA200 at close t; entries resume
  when the regime passes).
- The same 17 cells run on NDX and on R1000 (identical rules, REL25 on all three).
- Book roles: replacement (0.5 TAA + 0.5 B) and addition (0.5 TAA + 0.25 L + 0.25 B).

### Family C - residual momentum in the L shell (NDX; 4 cells)

Shell = L's structure with the score replaced: SMA100 filter, SPY gate, N 10 equal weight, VXN 22/0.25, month-end
decision, next-open fills, historical share units.
- Monthly stock returns r_{i,m} from the schedule's month-end CAPITALSPECIAL closes; market factor = SPY monthly price
  return on the same schedule (no external download). Regression per stock at each month-end t over the W months ending
  at t (at least 0.8 x W valid), with intercept, of r_{i,m} on the market return; residuals e_{i,m}.
- To give the first decision (2000-01-31) a full window, the regression inputs (month-end closes of the same symbols and
  of SPY) are loaded from 1996-01-01; they feed only the regression. Membership, eligibility, trading and every other
  feature use the standard 1999 cache. If this extended history cannot be built, every standalone comparison involving C
  starts at C's first invested session for all legs compared, and the report says so.
- Scores: RES12-1 = mean(e_{i,t-11..t-1}) / std(e_{i,t-11..t-1}) (BHM literal, skips the last month);
  RES12-0 = the same over t-11..t; control TOT12-1 = mean(r over t-11..t-1) / std(same) (no regression).
- Cells: C0 = RES12-1 with W = 36 (anchor); RES12-0 / W 36; RES12-1 / W 24; TOT12-1 (control) = 4 cells.
  Cross-checks on SP500 and R1000 (with REL25). Book roles: replacement and addition.

Total new configurations: A 16 + B 17 + C 4 = 37 (primary universes). Book-level trials: A 16 (one role) + B 34 +
C 8 = 58. Prior trials, lower bound: 159 (NDX robustness) + 8 (R1000 liquid) + 36 (JT 12-1 sweeps) + 3 (Clenow) +
4 (PTA) + 9 (NDX ROC suite) + 6 (universe comparison) + 10 (corr-penalty sweep, count not recorded) = 235.
N_trials for DSR = 58 + 235 = 293 (a lower bound; the pakal studies and whatever chose L's parameters are not in it).

## 5. Measurements

Windows (daily NAV returns). Standalone: P1 2000-01-04..2011-12-31, P2 2012-01-01..2021-12-31,
P3 2022-01-01..2026-08-19, FULL 2000-01-04..2026-08-19. Book blocks: G-P1 2008-03-04..2011-12-31 (TAA proxy),
G-P2 2012-10-02..2021-12-31, G-P3 2022-01-01..END, G-FULL 2012-10-02..END, G-LONG 2008-03-04..END, END = 2026-08-19
(the last TAA return).

Books (primary): the official pod model - pods compound independently from target weights at the first session of the
window and are reset to target weights at the last session of each calendar year
(`scripts/research/fund_menu_20260923/common.py::book_return_ser(..., "annual")`); each window is its own run. G3 =
{TAA 0.5, L 0.5}; replacement = {TAA 0.5, T 0.5}; addition = {TAA 0.5, L 0.25, T 0.25}. TAA = `taa_btal_tqqq` from
`results/research/portfolio/portfolio_refresh_20260927/sleeve_series_incl_2008.csv.gz` (proxy before 2012-10-02).
L inside every book (baseline and candidate books alike) is the replica's L in historical-share mode, once gate G1
passes, so that candidate and baseline come from the same simulator; the G3 built from the stored `ndx_atrfix` is shown
as a reference row. Sensitivity: the same books rebalanced daily (weighted sums of same-day returns), as used on 26 Sep.
Stress books: every replica leg (L and T) at 7.5 bps; TAA unchanged (no stressed TAA series exists; the same TAA enters
baseline and candidate).

Metrics: CAGR (252 d/yr), Sharpe = mean / sd x sqrt(252), no risk-free rate; max drawdown on the daily NAV; Calmar;
correlation of the leg with TAA and with L. Costs: engine (2.5 bps + commission on raw-equivalent shares) and stress
+5 bps per side (7.5 bps). Turnover = both-sided traded notional / mean NAV / year; trades per year; median and mean
holding period; mean exposure; share of exits by reason (rebalance / stop / refill-driven / regime / terminal).
Stop realism: for every stop exit, fill slippage beyond the stop = Open_{t+1} / stop level_t - 1 (mean, p5, share
below -2%); share of stop exits that were gaps through the level.
Capacity proxy (L, B0, every candidate): participation = order notional / ADV20 (Turnover) at the decision close;
pod AUM at which the 95th-percentile and the largest order reach 1% and 5% of ADV20, full history and 2021-2026.
This is a proxy (market-on-open fills use a fraction of the day's volume, so open-auction capacity is lower).
Terminal economics: count of terminal liquidations; share of total P&L and of traded notional from terminal
liquidations (P&L of a liquidated lot = proceeds - cost of the shares sold); sensitivity for every candidate and for
B0: terminal liquidations at 0.75 x the last close (a 25% distress haircut). A candidate whose terminal P&L share
exceeds 5% is flagged "terminal-economics dependent".
Charts: per grid, heatmaps of standalone FULL Sharpe, book G-FULL Sharpe (replacement role; addition role for B/C) and
the worst G3-block margin; timing-luck lines (A); stop-slippage histograms; exposure paths; equity and drawdown of the
candidates against G3.

## 6. Engine replica and parity gate (before any grid cell is read)

The grids run on a daily extension of the 26 Sep replica: historical share units and a daily state machine in which
held names keep their shares untouched unless a rebalance re-sizes them, a stop fires, or the price disappears. The
policy code (selection, stops, refills, breakout admission) produces order intents per decision close; the replica
executes them, and the same recorded intents are replayed by the real engine through a research subclass
(`historical_share_units_bool = True`) that only places `order_target_percent` (rebalance targets), `order_value`
(entries / refills) and `order_target(asset, 0)` (exits) on the matching bar. Gate:
- G1 the real engine's own run of the live pod (`run_variant` of `strategy_mo_atr_normalized_ndx_vxn_scaled`,
  historical-share mode is its default since fb81e86) versus the replica's L in historical-share mode: daily-return
  correlation >= 0.9999, FULL CAGR within 0.05 pp, identical top-10 lists on every decision; and both versus
  `ndx_atrfix` (the corrected sleeve): the same thresholds (a data-vintage caveat is allowed and reported).
- G2 at least four daily configurations executed by the real engine via the subclass: A/CH-3/CASH, A/PT-15/REFILL,
  B0 on NDX, and B/N250/k8/K10/R2 on NDX (plus B0 on SP500 if the engine run time allows): correlation >= 0.9999,
  CAGR within 0.05 pp, identical daily position lists.
- Unit tests: stop arithmetic (CH, PT, HWM including the entry day, no reset on re-size, reset on re-entry),
  refill and entry budget caps, breakout eligibility and admission order, historical-share sizing / commission /
  zero-share cancellation / terminal-liquidation arithmetic against hand-computed ledgers, dividend credit timing.
If any part fails, the failure is reported and fixed before any grid cell is run or read.

## 7. Download-date invariance (verified, not assumed)

- V1: every stock's OHLC multiplied by a random constant (log-uniform 0.01..100, seed 20260927); Unadjusted Close and
  Turnover unchanged. Every cell's entry lists, stop decisions, position lists and NAV path must be identical
  (NAV to 1e-9 relative; ledger share counts scale by the constant, raw shares do not).
- V2: for all month-end decision dates and 200 random other sessions (seed 20260927), features are recomputed from
  Norgate NONE bars restated into day-t units, X_t(d) = X_NONE(d) x R(t) / R(d), R(x) = U_x / Close_x: breakout flags,
  SMA200 / SMA100 passes, ROC252, NATR20, ATR20$, the daily refill score, and for every name held on that day the
  HWM-since-entry and the stop decision. Maximum relative difference per feature and the number of list / decision
  mismatches are reported; scale-free cells must show none (float rounding excepted).

## 8. Plateaus, candidates and the decision rule (frozen)

Neighbourhoods. A: rows = (type, policy); the neighbourhood of a cell is the cell and its adjacent levels in the same
row (up to 3). B1: the 3 x 3 box clipped at the edges; B2: adjacent K within the same rank row. C: the neighbourhood of
C0 is {C0, RES12-0 / W 36, RES12-1 / W 24}. Plateau value = median over the neighbourhood.

Candidates. A: per row, the cell with the highest plateau standalone FULL Sharpe (engine costs); ties go to the looser
level (nearest to L = no stop); four row candidates. B: per stage, the cell with the highest plateau standalone FULL
Sharpe on the primary universe; ties go to the cell nearest B0; two stage candidates. C: C0 (pre-declared). Candidates
are chosen on standalone Sharpe so the book rule is not the selection metric.

A candidate is recommended for its role only if all hold (neighbourhood medians throughout; G3 = TAA 0.5 + L 0.5,
official pod model, L from the replica):
- R1 (book, all blocks): book Sharpe strictly above G3's in each of G-P1, G-P2, G-P3.
- R2 (drawdown): book max DD not worse than G3's by more than 2.0 pp on G-FULL and on G-LONG.
- R3 (costs): R1 and R2 also hold with +5 bps per side on the candidate cells and on L (TAA unchanged).
- R4 (other universes, standalone FULL Sharpe, engine costs): A - the same stop cells on the L design on SP500 and
  R1000 have a neighbourhood median >= the no-stop L design there; B - >= B0's Sharpe there (trivial if the centre is
  B0); C - >= A0's Sharpe there (A0 = ROC12/NATR20 shell).
- R5 (capacity): in 2021-2026 the largest order stays below 5% of ADV20 up to a pod AUM of at least $5M (the p95
  order at 1% is reported).
Roles: A replacement only; B and C both roles are tested and both count as trials. If a candidate passes in both roles
the addition role is recommended (the smaller change to the live book). The report also states whether the resulting
book meets the owner's gates (Sharpe >= 1.35 and DD >= -20% on G-FULL); passing R1-R5 without meeting them is reported
as "improves G3, book still below the gate".
If nothing passes: keep G3 and name a forward shadow candidate = the candidate-role pair with the largest minimum
G3-block Sharpe margin among those satisfying R2 and R5 (if none, the largest margin overall, flagged). Anything chosen
after seeing results is labelled post-hoc and gets its own frozen test.

Confidence labels (do not change the rule):
- White's Reality Check (stationary bootstrap, mean block 21 sessions, 2,000 draws, seed 20260927) of the best
  G-FULL book Sharpe difference versus G3 over all 58 book configurations, and separately over family A alone; the
  standalone FULL Reality Check of family A versus L.
- Paired stationary bootstrap of each candidate-role pair's G-FULL Sharpe minus G3's, one-sided p and 90% interval.
- Deflated Sharpe ratio of each candidate's standalone FULL returns with N = 293 trials and the cross-trial variance of
  standalone Sharpe over its family's cells; and of the candidate book with the variance over the 58 book
  configurations.
- Walk-forward diagnostic: every candidate re-chosen on P1 (2000-2011) plateau values only; its P2, P3, G-P2, G-P3
  results are shown against G3.
- Timing luck (A): L and the four A row candidates at 21 rebalance offsets (k = -10..+10 sessions, the 26 Sep
  schedules); min / median / max standalone FULL and G-FULL Sharpe and the month-end percentile. Daily families have no
  rebalance day; their start-date dependence is covered by the blocks.
If a recommended candidate is not significant after these corrections the report says so: the switch would rest on
robustness and implementability, not on a proven higher return.

## 9. Assumptions

- A1 CAPITALSPECIAL applies one factor per bar to O/H/L/C (verified on NDX in stage 2; V2 re-checks); stops on
  adjusted prices ignore ordinary dividends (ex-dividend drops can trigger a stop; NDX yields are small).
- A2 Books follow the official pod model (annual reset); the daily-rebalanced book is a sensitivity only. Pod
  reallocations at year-end are frictionless.
- A3 G-P1 rests on the synthetic TAA proxy and is weaker evidence than G-P2 and G-P3; the rule treats it like the others.
- A4 Cash earns nothing in the engine: cells with low exposure (B in bear markets, A after stops) are penalised in CAGR.
- A5 Terminal liquidation at the last close is conservative bookkeeping, not corporate-action replay; NDX / S&P 500
  delistings are mostly acquisitions, where the last close is near the deal price; bankruptcies end near zero already.
- A6 Market-on-open fills at 2.5 bps: opening-auction capacity is below the ADV proxy; the +5 bps stress covers gaps.
- A7 L is what a day-t snapshot computes; live fills are not inspected. `ndx_atrfix` was produced by the corrected engine
  on 2026-09-26 data; small vintage differences against a fresh run are possible and reported.
- A8 Family C uses a market-only model (SPY price return) because the Fama-French three-factor file is not available
  locally and no external download is made; this is weaker than the literal BHM three-factor residual.
- A9 One history; a 14-year book Sharpe has a standard error of about 0.27; block differences of a few hundredths are
  noise.
- A10 The other-universe cross-checks keep the family's rules unchanged (SPY regime, VXN scaler, REL25); they test
  whether a choice generalises, not what is best there.

## 10. Lead decisions on the design questions (frozen with this file)

1. Family B primary universe: S&P 500 (diversification against the book's Nasdaq beta, enough breakout candidates for
   K = 20, capacity); NDX and R1000 are the R4 cross-checks.
2. Family C is kept, market-only (A8), 4 cells, with the 1996 regression history (section 4).
3. Both roles count as trials for B and C; a pass in both recommends the addition role.
4. END = 2026-08-19 for all book blocks and standalone windows.
5. Books use the official pod model with annual reset; L inside every book is the replica's L after G1 passes; the
   stored-`ndx_atrfix` G3 is a reference row; daily-rebalanced books are a sensitivity.
6. Family B's regime gate blocks entries only; the regime-exit variant is a sensitivity cell, not a candidate.
7. R4 for family A uses the L design (dollar-ATR score in decision-date units) with and without stops on SP500 and
   R1000, recomputed in historical-share mode.
8. The freeze precedes everything, including cache rebuilds and parity runs; no candidate cell is computed before it.
   No research agent may create or edit this file or any report `.md`; tables are written as CSV/JSON/TXT.
9. The new cache (about 1 GB) lives in the main checkout's `results/research/trend_breakout_20260927/_cache/`.
