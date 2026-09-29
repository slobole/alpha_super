# Mean reversion beyond DV2 — frozen specification

Written 2026-09-26 before any result of this study was computed. Owner request (Hebrew, 2026-09-26): research and
arrive at a "real" mean-reversion strategy worth trading, with a reason behind it, given that DV2 behaves like
crisis convexity rather than ordinary mean reversion and HPI is highly correlated with it; other universes than the
S&P 500 are welcome. Research-only: no live, release, scheduler, broker or registry-tier changes.
Later changes are appended as dated amendments; nothing above the amendment log is edited after results.

## Question

Is there a mean-reversion (MR) strategy that is (i) built on a stated economic mechanism, (ii) a different return
stream from the stock-MR family (DV2, HPI, QPI: pairwise daily correlation 0.74-0.87, beta 0.56-0.77, alpha
concentrated in 2020-22 and before 2010), (iii) positive in calm markets, and (iv) tradable in this operation
(IBKR pods, daily bars, next-open fills; same-day close only as a labelled diagnostic)? Output: one named strategy
with a verdict, or "none", with the full search disclosed.

## Context and prior evidence (already seen; not re-litigated)

- DV2 deep study 2026-09-25: DV2 alpha vs S&P 500 ~0 in calm blocks since 2010 (t < 1), large in 2020-22;
  S&P 400/600 transfers fail; industry-ETF DV2 works with low beta.
- Pakal studies: cross-sectional RSI long-short (next open) fails; price-path-convexity 5-day long-short in
  R3000 fails (Sharpe 0.49, DD -36%); end-of-day intraday reversal fails costs; DV2+NATR transfer to R1000-ex and
  R2000 fails; HPI R3000 top-liquidity quintile is a forward hypothesis only.
- Implication taken as the working hypothesis: short-horizon single-stock reversal is a liquidity-provision premium
  paid mainly when liquidity is scarce (Nagel 2012, "Evaporating Liquidity"). Long-only versions add market beta.
  A different MR stream must come from a different source of temporary mispricing, or remove the market component.

## Mechanism families (each with its reason, declared before results)

- A. Residual (peer-relative) stock reversal, hedged. A dip measured relative to the stock's own sector ETF
  (beta-adjusted) is more likely non-informational (Da, Liu & Schaumburg 2014; Hameed & Mian 2015). Hedging that
  ETF removes market beta and the market-wide part of the liquidity premium. Sub-hypotheses, reported as
  conditioning splits only: A2 low abnormal turnover moves revert more (Medhat & Schmeling 2022); A3 residual
  losers in industries with a positive 63-day trend revert more (slow industry-news diffusion, Hou 2007).
- B. Forced-flow rebound. A stock removed from an index while it stays listed is sold by passive funds at a known
  close; the non-informational pressure should revert over the following weeks (Chen, Noronha & Singal 2004).
  Index calendars, not volatility, time these trades.
- C. Cross-asset liquidity provision. Short-term dips in liquid non-equity ETFs (bonds, commodities, currencies,
  real estate) revert; their shocks are asset-class specific, so returns should be less tied to US equity crises.
  Equity ETFs are reported as a separate class for comparison.
- D. Close-substitute relative value. Industry ETFs against their parent sector ETF and country ETFs against
  their region ETF: relative dips caused by narrow-ETF flows should revert. Hedged, so market-neutral.

## Data and fixed settings

- Norgate US equities incl. delisted. Signals, fills and marks CAPITALSPECIAL; cash dividends from the Dividend
  field; liquidity = Unadjusted Close x Volume (ADV63 = 63-day mean). PIT membership for A with the engine's
  semantics (`build_index_constituent_matrix`, latest row <= decision date). B uses raw
  `index_constituent_timeseries` to date the exits (the engine loader trims the last 5 rows of past members).
- Universes. A: S&P 500 (selection); validation: Russell 1000 excluding same-day S&P 500 members (disjoint names),
  S&P MidCap 400, S&P SmallCap 600, Nasdaq-100 (overlaps the S&P 500; secondary). B: exits from S&P 500,
  S&P 400, S&P 600, Nasdaq-100, Russell 1000, Russell 2000. C, D: fixed ETF lists below (today's listings:
  survivorship caveat recorded).
- Hedge instruments: SPY (before 1993-02-01 the $SPX price index as a proxy); the nine original sector SPDRs
  (XLB XLE XLF XLI XLK XLP XLU XLV XLY, from 1998-12-22); XLRE and XLC join once they have 252 sessions.
- $VIX close is used only for descriptive regime splits, never as a trading input.
- Windows: main 2000-01-03 -> 2026-08-19 (end aligned with the pod return inventory). Blocks: E 2000-2009,
  C1 2010-2014, C2 2015-2019, S 2020-2022 (stress), C3 2023-2026-08. "Calm pool" = C1+C2+C3.
  Holdout: 1991-01-02 -> 1999-12-31 (S&P 500; Russell from 1990-07), used once per family after its Stage-2 pick.
  A's sector hedge does not exist before 1999, so A's holdout test uses the SPY-hedged form of the chosen rule.
- Costs: engine default on every fill (2.5 bps slippage per side, $0.005/share, $1 minimum). Short ETF borrow
  0.5%/yr ACT/360 on short market value (stress 2%); short proceeds and idle cash earn 0. Dividends: longs receive
  75% of gross (25% withholding, owner default), shorts pay 100%. Stress: +5 bps per side on every fill.
- Stage-1 cost hurdles per round trip: hedged stock trade 10 bps; unhedged stock 8 bps; ETF 6 bps; hedged ETF
  pair 12 bps.

## Stage 1 — signal map (descriptive; no portfolio mechanics)

### Family A

Eligible on decision date t (after Close_t): PIT member, all raw fields finite, raw close > $5, 252 sessions of
history. Liquidity tiers: L = ADV63 above the same-day eligible median (the DV2 floor); ALL = no ADV condition.

Definitions (r = close-to-close return, CAPITALSPECIAL):
- beta_{i,t} = Cov_252(r_i, r_H) / Var_252(r_H) over t-251..t (>= 200 pairs), shrunk b = 0.67*beta + 0.33,
  clipped to [0.3, 2.0]. Sector H = the SPDR with the highest 252-day return correlation, assigned at each
  month-end and used for the next month (causal). SPY hedge H = SPY.
- residual e_{i,t} = r_{i,t} - b_{i,t-1} * r_{H,t}; sigma_{i,t} = std(e over t-62..t), >= 50 values.
- E_k (k = 1, 5, 21) = sum_{j<k} e_{i,t-j} / (sigma_{i,t} * sqrt(k)), for H = sector (E_k_SEC) and H = SPY
  (E_k_SPY). Raw z: R_k = sum_{j<k} r_{i,t-j} / (std_63(r) * sqrt(k)), k = 1, 5, 21. R5 raw (unscaled) as the
  classic reference. INTRA5 / OVN5 = 5-day sums of log(C/O) and log(O/C_prev), each scaled by its own 63-day
  std * sqrt(5). References: DV2(126) and IBS.
- Forward return: entry Open_{t+1}, exit Open_{t+1+h}, h in {1, 5, 10, 21}; a missing exit open uses the last
  finite close on or before that date (engine-like liquidation). Hedged forward = raw - b_{i,t} * hedge forward.

Per date: bottom decile of the signal among eligible names (>= 20 eligible) and the 10 most extreme names.
Reported: bucket hedged return (what a pod earns) and bucket excess over the eligible mean with the same hedge
(signal quality), per block, VIX tercile (full-sample terciles), and the holdout; Newey-West t with h-1 lags.
Conditioning splits for A2 (abnormal turnover = 5-day mean volume / 63-day mean volume, terciles) and A3 (sign of
the assigned sector ETF's 63-day return) are reported for the E_k_SEC signals only.

Screen: a cell (universe x tier x signal x h x hedge) is LIVE if, in the S&P 500, the bottom-decile excess over the
eligible mean exceeds the round-trip hurdle in the calm pool with NW t >= 2 and is positive in each of C1, C2, C3.
The two LIVE (signal, h) pairs with the highest calm-pool t (at most one per signal) go to Stage 2. If none is
LIVE, family A stops at Stage 1.

### Family B

Event = a symbol with raw membership 1 on day d and 0 on day d+1 that still has finite prices on d+1..d+21.
Known after Close_{d+1} (conservative: the announcement usually precedes it); entry Open_{d+2}. Types: exit to
another index of the same family on the same date (e.g. S&P 500 -> S&P 400, Russell 2000 -> Russell 1000) vs
exit to none. Forward h in {5, 10, 20, 40}; hedged with SPY beta (as in A) and, diagnostically, a size ETF
(MDY for S&P 400 and Russell 1000 exits; IWM for S&P 600 and Russell 2000 exits; SPY for S&P 500 and Nasdaq-100).
Control: index additions (the same statistics; the addition effect should show no rebound or a decline).
Screen: LIVE if pooled 2000-2026 mean SPY-hedged CAR(h) > 30 bps with t >= 2 (events clustered by date) and
positive in 2000-2012 and 2013-2026, for at least one (index group, h).

### Family C

ETF list (Norgate symbols; missing ones dropped as data): bonds TLT IEF LQD HYG TIP EMB AGG; commodities GLD SLV
USO DBC DBA UNG; currencies UUP FXE FXY FXA FXC FXB FXF; real estate VNQ IYR; equity class (comparison) SPY QQQ
IWM MDY EFA EEM EWJ EWG EWU EWZ FXI XLB XLE XLF XLI XLK XLP XLU XLV XLY. Eligible after 252 sessions and
ADV63 > $10M. Signals: DV2(126) < 10; IBS < 0.15; z3 = 3-day return / (std_63 * sqrt(3)) < -1.5. Forward
h in {1, 5}, next open. Excess = conditional mean minus the ETF's unconditional mean in the same block.
Screen per class: LIVE if pooled 2010-2026-08 excess > 6 bps with t >= 2 and positive in 2003-2014 and 2015-2026.

### Family D

Pairs (narrow -> broad): SMH SOXX IGV XSD -> XLK; XBI IBB IHI XPH -> XLV; KRE KBE -> XLF; XOP OIH -> XLE;
XME -> XLB; XHB ITB XRT -> XLY; ITA IYT -> XLI; EWJ EWG EWU EWA EWH EWS EWP EWQ EWI EWL EWN -> EFA;
EWZ EWT EWY EWW FXI INDA EZA -> EEM; IWM MDY -> SPY. Relative residual as in A with H = the broad ETF;
signal E_5 < -2. Forward relative (hedged) return h in {5, 10}. Screen: LIVE if pooled 2005-2026-08 mean > 12 bps
with t >= 2 and positive in 2005-2014 and 2015-2026.

## Stage 2 — pods (daily simulators with the full cost model, next-open fills, $1M, NAV/S per slot)

- A: long S stocks; at entry each is hedged by shorting b x its long value of its assigned sector ETF (variant:
  SPY), unwound at its exit; ETF orders netted per ETF per day. Entry: signal < -z_in, most negative first.
  Exit: signal > -0.5, or T_max = 10 sessions, or loss of eligibility/membership. Grid per LIVE signal:
  z_in in {1.5, 2.0, 2.5} x S in {10, 20} x exit in {revert-or-10d, fixed h of the LIVE cell} x liquidity tier of
  the LIVE cell = 12 configurations; the selected configuration is then also run long-only and SPY-hedged.
- B: long up to S event stocks (oldest event first if full), hold h sessions; S in {10, 20} x h in {10, 20} x hedge
  in {none, SPY}; index group from the LIVE cell.
- C: long-only on the LIVE class(es), S in {5, 10}, the LIVE signal, exit Close > previous High or 10 sessions,
  most oversold first.
- D: one position per pair (long narrow, short b x broad), entry E_5 < -z_in (z_in in {1.5, 2.0, 2.5}), exit
  E_5 > -0.5 or 10 sessions, S = number of pairs of the LIVE group(s).
Pick per family: highest 2000-2026 (C, D: from their data start) net Sharpe among configurations with positive
net CAGR in each of C1, C2, C3.

## Stage 3 — gates for each family's pick ("recommend for forward test" needs all)

- G1 Different: daily correlation <= 0.40 with DV2 (floor + ADV63 rank, F1) and with HPI vote over the common
  window; beta to $SPXTR <= 0.40.
- G2 Calm: net CAGR > 0 in each of C1, C2, C3; calm-pool Sharpe >= 0.6.
- G3 Economics: 2000-2026 net Sharpe >= 0.8; >= 0.6 with +5 bps/side; max drawdown not worse than -20%.
- G4 Out of sample: same frozen rule has Sharpe >= 0.4 and positive calm-pool return on >= 2 validation
  universes/sets (A: R1000-ex-S&P 500, S&P 400, S&P 600, Nasdaq-100; B: index groups not selected;
  C/D: other classes/groups), and a positive holdout (1991-1999) Sharpe where testable.
- G5 Plateau: every one-step neighbour on each gridded axis keeps >= 70% of the pick's Sharpe.
- G6 Luck: the pick beats the 95th percentile of 200 runs that choose randomly among the same eligible candidates
  (B: 200 placebo runs with pseudo-event dates drawn from the same stocks' non-event days).
- G7 Tradable: report the AUM at which the 95th-percentile order exceeds 1% of ADV63 (open auction) and 5%
  (close); pass if >= $2.5M at the open.
- G8 Book: adding the pick at 10% (others scaled by 0.9) to G3+MR "DV2-ADV 9 + ETF 9" (TAA 32 / NDX 32 / DV2-ADV 9
  / industry-ETF 9 / HPI 18, annual reset, 2012-10-02 -> 2026-08-19) raises Sharpe and does not lower Calmar.
Verdicts: "recommend for forward test" (all gates); "diversifier candidate" (G1, G2, G4, G6 pass and G3 misses by
<= 0.2 Sharpe); otherwise "reject". Deflated Sharpe of the final pick with the full trial count of this study.

## Implementation

A winner gets an engine module (research tier, not wired, not in any release), replica = engine trade for trade,
tests. Margin note for hedged pods: long 100% + short ~100% needs portfolio margin or a lower gross (Reg-T).

## Amendment log
- A1 (2026-09-26, before any Stage-1 result; clarifications only): (a) "exceeds the hurdle with NW t >= 2" is
  read strictly: the Newey-West t-statistic of (excess - hurdle) is >= 2 in the calm pool; "positive in each of
  C1, C2, C3" means the gross excess is > 0 in each block. (b) Russell 1000 ex S&P 500 = Russell 1000 members that
  are not S&P 500 members on the same date (S&P 500 PIT mapped by Norgate symbol). (c) Stage-1 eligibility needs a
  finite Open_{t+1} (an order for a name that does not open is not filled); the bucket and the universe mean use
  the same set, so the comparison is symmetric. (d) Feature parity was checked on AAPL 2015-06-15 against an
  independent pandas computation (252-day beta, E5_SEC, F5 all equal to float precision); sector assignment gives
  AAPL XLK, XOM XLE, JPM XLF, JNJ XLV, PG XLP, NEE XLU, CAT XLI, AMZN XLY.
- A2 (2026-09-26, before any family-B result): the frozen event definition required finite prices on d+1..d+21,
  which peeks 20 sessions ahead and drops names that delist soon after leaving (a survivorship filter). Replaced by
  "finite close on d+1" (known at the decision); later delistings stay in the sample and exit at their last finite
  close, as the engine does. Everything else in family B is unchanged.
- A3 (2026-09-26, before any family B, C or D result): the strict reading of A1(a) applies to every family's
  screen: "mean > hurdle with t >= 2" means the t-statistic of (mean - hurdle) is >= 2 (B: clustered by event
  date; C, D: Newey-West on the daily average over names with a signal that day, h-1 lags). Family D is screened
  per group and pooled; LIVE groups (or all pairs if only the pool passes) go to Stage 2.
- A4 (2026-09-26, after families A (S&P 500 and Nasdaq-100 maps), C and D failed their Stage-1 screens; before
  any family-B or family-E result). New family E, disclosed as added after those failures; its trials are counted.
  Reason (declared before its results): mean reversion paid regularly should come from flows that arrive on a
  schedule, not from liquidity shocks that arrive in crises. Institutions sell losers into month- and quarter-ends
  (window dressing, redemptions, rebalancing) and taxable investors sell losers into year-end; turn-of-month
  inflows arrive right after. So losers measured at a period end should rebound at the start of the next period
  more than the same losers on ordinary days.
  - E-M: decision after Close of the last trading day of each month; signals E21_SEC, E21_SPY and R21z (family-A
    definitions); bottom decile among family-A eligible names (tiers L and ALL); entry Open of the first trading day
    of the next month; h in {3, 5}; hedges none / SPY / sector. Subsets reported: quarter-ends, December.
    Control: the same signal on all other days (family-A map values).
  - E-Y: decision after Close of the last trading day of December; signal = year-to-date raw return (Close_t /
    Close at the prior year's last trading day - 1); bottom decile; entry Open of the first January trading day;
    h in {5, 10}; same hedges.
  - Screen: E-M LIVE if in the S&P 500 the per-event bottom-decile excess over the eligible mean exceeds the
    round-trip hurdle with t(excess - hurdle) >= 2 over 2000-2026 (plain t: events are monthly and do not overlap)
    and is positive in each of E, C1, C2, S, C3 (block means). E-Y LIVE if the same holds over 2000-2026 with
    positive means in 2000-2012 and 2013-2026. Stage 2 pod for a LIVE E cell: long the bucket's 10 or 20 most
    extreme names at the next open, equal weight, hold h, hedge per the cell; S in {10, 20}.
- A5 (2026-09-26, after the family-B Stage-1 table; before any family-B pod result). Findings that motivate it:
  the LIVE exit-to-none cells (S&P 500/400/600, Russell 1000, Nasdaq-100) are dominated by distressed low-priced
  names (35-50% had a raw close < $5; e.g. GGP 2008 at $0.35); only the Nasdaq-100 group keeps a positive median.
  Change (conservative, required by the tradability rule of QUANT_PHILOSOPHY): family-B pods trade only events
  with raw close > $5 on the decision day and ADV63 > $5M (63 sessions through the last day in the index).
  The Stage-2 grid is unchanged (S {10, 20} x h {10, 20} x hedge {none, SPY}), run per LIVE index group and pooled.
  Also disclosed: the 1991-1999 rows of the Stage-1 event table were printed with the main table, so the family-B
  holdout is seen; G4 for family B therefore relies on the index groups not selected, not on the holdout.
  Exploratory (post-hoc, labelled, cannot be promoted without forward evidence): an index-flow long/short pod
  (long tradable deletions, short tradable S&P 500 and Nasdaq-100 additions, SPY-hedged), because the Stage-1
  control showed a strong post-inclusion decline (S&P 500 additions -174 bps over 5 sessions, t -4.4, negative in
  both halves), a documented effect with the same forced-flow mechanism.
- A6 (2026-09-26, before downloading any auction data or computing any family-F result). New family F, added
  after families A-E, disclosed and counted. Reason (declared before results): Treasury coupon auctions are
  scheduled supply shocks; dealers and hedgers absorb the new supply, prices concede into the auction and
  recover after it (Lou, Yan & Zhang 2013, "Anticipated and repeated shocks in liquid markets"). A flow on a
  calendar, in a different asset class, so it should be unrelated to equity mean reversion.
  - Data: auction dates from the US Treasury Fiscal Data API (auctions_query; public, no key), security_type Note
    or Bond, excluding TIPS and FRN; buckets LONG = Bond (20- and 30-year incl. reopenings) traded with TLT;
    TEN = Notes with a term of 9 years or more (10-year incl. reopenings) traded with IEF. Several auctions of one
    bucket on the same day count once. The auction date is announced about a week ahead, so orders scheduled on
    it are causal.
  - Windows per event (A = auction day, fills in the bucket's ETF): POST_OPEN_k = buy Open_A, sell Open_{A+k};
    POST_CLOSE_k = buy Close_A (scheduled MOC), sell Close_{A+k}; k in {1, 3, 5}. Control PRE = Close_{A-5} to
    Close_{A-1} (should be negative). Excess = window return minus the ETF's mean daily return (same half) times the
    window length.
  - Screen: LIVE if pooled 2003-2026-08 mean excess > 6 bps with t(excess - 6 bps) >= 2 (plain t over events)
    and positive in 2003-2014 and 2015-2026, for at least one (bucket, window).
  - Holdout (untouched, used once after the screen): FRED DGS30 (LONG) and DGS10 (TEN) daily yields,
    1983-01-01 -> 2002-06-30: mean yield change over the LIVE window (should be negative, t <= -2).
  - Luck: 200 placebo runs with pseudo-auction days drawn from the same months' non-auction days.
  - Stage 2 (if LIVE): long-only pod holding the ETF only in the LIVE window, cash otherwise; engine costs, 75%
    dividends; gates G1-G8 as for the other families (G4 = the other bucket and the yield holdout).
- A7 (2026-09-26, after the family-F screen: no post-auction window is LIVE, so family F fails as pre-declared;
  before touching the holdout). Labelled post-hoc observation: the PRE control is strongly negative for TLT
  (LONG: -32.6 bps over Close_{A-5} -> Close_{A-1}, t -3.2; -33.0 and -32.5 bps in the two halves) and weaker for
  IEF (TEN: -9.7 bps, t -2.0). This is a supply concession without a measurable rebound, not mean reversion; it
  is recorded as a timing rule candidate for existing bond-holding pods. One confirmatory test is declared now,
  on the untouched holdout: FRED DGS30 (LONG) and DGS10 (TEN), auctions 1983-01-01 -> 2002-06-30, yield change from
  Close_{A-5} to Close_{A-1} (FRED daily constant-maturity yields, missing days skipped) should be positive
  (prices down) with t >= 2 for LONG. Pass or fail, it stays a forward hypothesis only.
- Results log (2026-09-26, no rule change): family-F placebo (200 runs, pseudo-auction days from the same months):
  post-auction windows inside the placebo band; the PRE control beyond all 200 runs (LONG -32.6 bps vs placebo
  min -22.7; TEN -9.7 vs -8.1). Holdout (A7): LONG +2.9 bps yield, t 1.7; TEN +3.0, t 1.8 (misses t >= 2).
  Diagnostic after the family-B pick: Nasdaq-100 December YTD losers -> first 5 January sessions +1.6% vs SPY
  (t 2.1; 2000-12 +2.8%, 2013-26 +0.4%); Nasdaq-100 deletions outside December: median +1.8% over 20 sessions (n 81).
- A8 (2026-09-26, after an independent quant-pitfalls review; no rule changes, every item labelled post-review):
  (1) Holdout exactly as A7 declared (raw yield change): LONG +2.48 bps (t 1.46), TEN +2.54 (t 1.54). The first report
  draft quoted an undeclared drift-adjusted version (+2.87 / +3.00, t 1.70 / 1.81); both miss t >= 2.
  (2) Day-of-month control (stage1_f.py dom; the placebo matched only the month): LONG -19.8 bps (t -1.93;
  2003-14 -2.6, 2015-26 -26.1 with t -2.19), TEN -0.4 (t -0.09). The PRE concession is mostly TLT's month-end
  cycle and is dropped as a separate finding.
  (3) EOM flow vs a no-signal TLT month-end control (long the last 5 sessions, short the first 5, 2.5 bps/side):
  control net Sharpe 0.74 (0.80 / 0.68 by half), correlation 0.63 with eom_flow, eom alpha over the control
  6.5%/yr (t 4.0, R^2 0.39). EOM was never run through G3-stress, G4-G7 or a deflated Sharpe here and its MOC timing
  was chosen adaptively in its source study, so it is reported as a forward-test candidate only.
  (4) The book test now uses the frozen family-B pick (20 slots): Sharpe 1.520, Calmar 1.937 (G8 still passes).
  (5) The engine membership trim, recomputed by the reviewer with raw membership, moves family-A cells by at most
  1.2 bps per block with mixed sign (negligible).
  (6) Reproducibility: stage1_b.py stats, stage1_f.py holdout / dom and report_tables.py now produce every number in
  the report; the study files are untracked in git, so the amendment order is documented only in this file.
