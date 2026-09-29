# PREREG (frozen) - "IPOs at all-time highs" (Marsten Parker's rule, as rebuilt by Carlos) and the same rule on recent stock splits

Frozen: 2026-09-28, before any event return, forward return or pod result was computed on our data. The freeze evidence is
the SHA-256 and timestamp in `results/research/ipo_split_ath_20260928/prereg_freeze.json`. There is no git commit, and
the files are untracked in worktree `nice-banzai-a4e788`. Later changes are dated, labelled amendments in a separate file.
They cannot change a verdict that has already been computed.

- Research only: no live pod, release, scheduler, broker code, released YAML or WIRED strategy file is modified.
- Code: `scripts/research/ipo_split_ath_20260928/`. It may import `new_pod_search_20260927` and
  `trend_breakout_20260927` for the book, the sweep and the metrics.
- Tests: `tests/test_research_ipo_split_ath_20260928.py`.
- Results: `results/research/ipo_split_ath_20260928/` in the main checkout.
- Designed by the lead (Opus 5.5).

## 0. What is already known (disclosed; this weakens the evidence)

- **Source.** The owner pasted a talk by "Carlos", who rebuilt a rule that Marsten Parker described on the *Chat With
  Traders* podcast. Parker's own words were: "an IPO is a recently listed company; when it closes at a new all-time
  high I buy, with a profit target and a stop loss". Every detail below is Carlos's interpretation, not Parker's.
  - Data: Sharadar Core US Equities (survivorship-free).
  - Universe: mid, large and mega caps only.
  - "IPO" = fewer than 90 days of trading history. The first day is excluded, so the rule can act on days 2..89.
  - Signal: close = running maximum close.
- **Carlos's reported results.** Sharadar, about 2001-2023. His cost model and fill timing are not stated.
  - Stage A, 20-day hold:
    - IPO all-time-high events: about 19,000, mean +4%, 57% positive, average win +14%, average loss -10%.
    - Non-IPO all-time-high baseline: about 500,000, mean +1.14%, 55% positive, +7.36% / -6.59%.
    - A two-sample test gave p < 5%. The test treats overlapping, same-day events as independent, which inflates
      significance.
  - Final setup:
    - Rules: 20 slots, larger market cap first, profit target +20%, trailing stop 10% below the highest price since
      entry.
    - Pod results: CAGR about 18%, Sharpe about 1.4, max DD below 20%, cash use 67%, about 100 trades a year.
    - Per trade: mean +3.7%, win rate 44%, average win +20%, average loss -9%. Down years were 2008 and 2011 only.
  - The best month, October 2021 (about +50%), came from the Trump SPAC (DWAC). So his universe included SPACs.
  - He tried 5 slot counts and 5 target/stop pairs and kept the best. His numbers are therefore in-sample after a small
    search.
- **Parker's own comment.** The strategy has not worked well since 2022. He thinks this may be permanent, because more
  companies now raise capital privately.
- **Earlier results in this repo.**
  - Stock momentum, trend and breakout rules have been tested repeatedly (knowledge base; the 26-27 Sep studies). None
    beats the T-bill control in the book slot.
  - Per-position trailing stops did not help the NDX pod.
  - No study in this repo has conditioned on IPO age or on stock splits.
- **Book controls** (official pod model, annual reset; slot = TAA 0.5, L 0.25, X 0.25; computed in the 27 Sep study):

  | Slot | 2008-11 | 2012-21 | 2022-26 | 2012-26 Sharpe / MaxDD | 2008-26 Sharpe / MaxDD |
  |---|---|---|---|---|---|
  | none (G3) | 0.689 | 1.303 | 1.257 | 1.288 / -14.1% | 1.167 / -15.3% |
  | BIL (T-bills, total return) = C_BIL | 0.728 | 1.334 | 1.437 | 1.365 / -11.7% | 1.226 / -14.3% |

- **Lead's prior, stated before any result.**
  - Stage A:
    - IPO all-time-high events will beat the non-IPO baseline in raw terms, largely because young stocks are more
      volatile and have higher beta.
    - The market-adjusted gap will be positive in 1993-2000 and 2020-21 and small elsewhere.
    - Clustered by month, the t statistic will be far below the naive one.
  - Pods: IPO-A0 standalone Sharpe 0.5-0.9 with high beta; it fails R1 in 2022-26. Split-A0 behaves like a
    breakout/momentum pod on large caps and fails R1.
  - Probability of passing R1-R5: IPO about 10%, split about 5%.

## 1. Questions and mechanisms

- **Q1 (edge).** After a close at an all-time high, do stocks listed fewer than 90 sessions earlier earn more over the
  next 1, 5, 10 and 20 sessions than stocks at an all-time high that are not recent listings? Is the gap still there
  after the market's move and after honest clustering?
- **Q2 (variant 2, owner's request).** Does the same hold for stocks in the 90 sessions after a forward stock split?
- **Q3 (pod).** Does the rebuilt trading rule beat T-bills in the book's quarter slot under the frozen rule
  (section 8)? Answered for the IPO population and for the split population separately.
- **Mechanisms offered by the source and the literature** (hypotheses, not evidence):
  - IPO:
    - attention and scarce float before lock-up expiry (about 180 days);
    - underwriter price support;
    - analyst initiations when the quiet period ends (about 25 days);
    - momentum while information is revealed.
  - Split: the post-split drift that Ikenberry, Rankine and Stice (1996) and Desai and Jain (1997) reported.
    Management signals confidence; a lower share price draws retail attention. Later work reports the drift weakened
    after the 1990s.
  - Against both: IPO long-run underperformance (Ritter 1991); IPO waves happen in bull markets, so the rule's
    exposure rises with market beta.

## 2. Hard constraints

- Causal features only. Decisions are made at the close of session T. Entries fill at the open of T+1. The
  engine semantics apply: 2.5 bps slippage per side; $0.005 a share with a $1 minimum; dividends credited at 75% net;
  terminal liquidation at the last close when a series ends.
- Long only. Survivorship-free: delisted securities are included.
- Idle cash is swept into BIL total return for the primary evaluation. The sweep is
  `new_pod_search_20260927.common.sweep_return_ser`, and its rate is 0 before BIL's first return on 2007-05-31. Results
  without the sweep are reported alongside.
- No change to live pods, releases, schedulers, broker code or WIRED strategy files.

## 3. Data, universe and windows

- **Securities.** Every symbol in Norgate's `US Equities` and `US Equities Delisted` databases with subtype1 = Equity,
  restricted to operating companies and REITs by subtype2. The exact subtype2 list is recorded in
  `symbol_meta_summary.json` before any event is computed. ETFs, closed-end funds, preferreds, warrants, units and
  rights are excluded.
- **Prices.** CAPITALSPECIAL daily bars with padding NONE, loaded from the earliest date; float64.
  - Fields: Open, High, Low, Close, Volume; native Turnover; Unadjusted Close; Dividend.
  - Bars are mapped onto the `$SPX` session calendar.
- **Listing session L.** A symbol's first quoted session.
  - A symbol is a **new listing** only if all three hold:
    1. first quoted date >= 1991-01-01; Norgate's history starts in 1990, so earlier first dates are data starts;
    2. it was major-exchange listed on that date (not an OTC uplisting);
    3. it was not a blank-check company on that date (SPACs are excluded from the primary IPO set; label S-SPAC keeps
       them).
  - Spin-offs, re-listed reorganized companies and first US listings of foreign companies cannot be told apart from
    IPOs in Norgate. They count as new listings, which matches Parker's "recently listed". This is disclosed.
- **Age a_T** = number of market sessions from L to T (L has age 0).
- **Forward split.** Session E is a forward-split ex-date when:
  - r_E = k_{E-1} / k_E >= 1.24, where k = Unadjusted Close / Close;
  - r_E is within 0.5% of a ratio a/b with integers 1 <= b <= 4 and b < a <= 50.
  - The rule is checked on consecutive bars of the symbol.
  - Other capital events (reverse splits, spin-off adjustments, special dividends) are not split events.
  - The detector's output is reported as counts per year and a spot-check list of 20 random events.
- **Liquidity and size filter (proxy; disclosed deviation).** Norgate has no point-in-time market cap.
  - ADV_T = median native Turnover over the last up to 20 sessions ending at T.
    - The listing session L is excluded.
    - At least 1 value is required.
  - A stock is **eligible at T** when both hold:
    - Unadjusted Close_T >= $5;
    - ADV_T ranks in the top 1,000 among all securities above (any age) with a bar on T. This is roughly a
      Russell-1000-sized liquid universe, standing in for "mid, large and mega cap".
  - Known distortion: young IPOs trade much more of their float than older stocks, so this filter admits smaller IPOs
    than a market-cap filter would.
  - Robustness universe U-500: top 500 instead of top 1,000.
- **All-time high.** ATH_T holds when Close_T >= max(Close_L .. Close_{T-1}), with at least one prior close. Adjusted
  closes are used; the ratio is unaffected by later adjustments.
- **Event populations at the close of T** (all require eligibility at T):
  - **IPO-ATH:** new listing, 1 <= a_T <= 89, ATH_T.
  - **SPLIT-ATH:**
    - a forward-split ex-date E with 0 <= sessions(E..T) <= 89;
    - a_T >= 252, with age measured from the first quoted session, where a pre-1991 data start counts as a listing;
    - ATH_T.
  - **BASE-ATH (baseline, Carlos's "non-event"):** ATH_T, a_T >= 90 or a data-start symbol, and not in a split window.
  - **IPO-ALL (second baseline):** new listing, 1 <= a_T <= 89, any day (at or below its high).
  - **SPLIT-ALL:** in a split window, a_T >= 252, any day.
- **Windows.**
  - Events and trading run 1993-01-04..END, END = 2026-08-19 (the TAA series end). History before 1993 is warm-up.
  - Standalone blocks: P0 1993-01-04..1999-12-31; P1 2000-01-03..2011-12-31; P2 2012-01-01..2021-12-31;
    P3 2022-01-01..END; FULL.
  - Book blocks (unchanged): G-P1 2008-03-04..2011-12-31; G-P2 2012-10-02..2021-12-31; G-P3 2022-01-01..END;
    G-FULL 2012-10-02..END; G-LONG 2008-03-04..END.

## 4. Stage A - event study (the edge question)

- **Forward returns** for h in {1, 5, 10, 20} sessions, entering at the open of T+1:
  - R_h = (Open_{T+1+h} + sum of Dividends from T+1 .. T+h) / Open_{T+1} - 1.
  - The market return M_h is SPY's TOTALRETURN Open_{T+1+h} / Open_{T+1} - 1.
  - The excess return is X_h = R_h - M_h.
  - If the stock has no bar at T+1+h (it was delisted), the last available close replaces Open_{T+1+h}. The count of
    such events is reported.
  - Carlos's close-to-close version, Close_{T+h} / Close_T - 1, is reported as a label only; it cannot be traded
    without a close fill.
- **Per population, per block:**
  - count and events per year;
  - mean, median and share positive of R_20 and X_20;
  - average win and average loss of R_20;
  - the same means for h = 1, 5 and 10.
- **Significance.** The primary test statistic is monthly-clustered:
  - D_m = mean X_20 of IPO-ATH events entered in month m, minus mean X_20 of BASE-ATH events entered in month m.
  - The t statistic is computed over the months that have events in both populations, with Newey-West at 2 lags.
  - The same is done for SPLIT-ATH versus BASE-ATH, and for IPO-ATH versus IPO-ALL (does the high add anything to
    youth?).
  - Carlos's naive Welch t over events is reported beside it for comparison.
- **Stage A answer (frozen).** "Edge present" requires both:
  - clustered t > 2 on FULL;
  - D positive in at least 3 of the 4 blocks P0..P3.

  Stage A does not decide the pod verdict. The pod rule (section 8) does.

## 5. Stage B - the trading rule (rebuilt exactly as described, with the missing details frozen here)

At each close T, per population (IPO-ATH or SPLIT-ATH):

1. **Exit orders for the next session.** They apply only to positions whose entry session is <= T.
   - The profit target is a sell limit at G = F x (1 + P), where F is the entry fill before slippage. G stays fixed.
   - The trailing stop is a sell stop at S = H_T x (1 - L), where H_T = max(F, closes from the entry session through
     T).
   - Both orders are day orders, re-placed each close, as in the engine.
   - On session T+1, as in the engine's LimitOrder and StopOrder:
     - if Low <= S, the stop fills at min(Open, S);
     - otherwise, if High >= G, the target fills at max(Open, G);
     - if both levels are touched, the stop is assumed first (conservative).
   - Slippage applies to every fill.
2. **Entries.**
   - Candidates are the population's events at T that are not held.
   - They are ranked by ADV_T, largest first (the stand-in for "larger market cap first"); ties are broken by symbol.
   - Free slots = N minus the positions held at T.
   - Each candidate is bought at the open of T+1 with a market order until the free slots are used.
   - Budget b = min(V_T / N, max(C_T, 0) / free slots), where V_T and C_T are pod value and cash at the close of T.
   - Shares = floor(b / Unadjusted Close_T), converted to adjusted units with k_T.
   - A name can be bought again after it exits, on a later signal.
3. **No time stop, no re-sizing, no regime filter.**
4. **Accounting.**
   - Daily marks at Close.
   - Dividends: 0.75 x Dividend x adjusted shares.
   - Commission: max($1, $0.005 x nominal shares), where nominal shares = adjusted shares / k on the fill date.
   - A position whose symbol has no bar on a later session is liquidated at its last close.

**Grid per population:** N in {10, 20, 40} x (P, L) in {(7%, 4%), (15%, 10%), (15%, 15%), (20%, 10%), (20%, 15%)}.
- These are Carlos's five target/stop pairs; his "15/16" is read as 15/15 because he calls it one-to-one.
- 15 cells per population, 30 in all.

**Anchor A0 = N 20, P 20%, L 10%**, Carlos's final setup. **Each population's candidate is its A0.** No cell is
selected from our data. The other 14 cells are descriptive, and they are counted in the Reality Check and the DSR.

**Labels (not candidates):**
- **E2, next-open exits.** A close-decided stop or target (Close_T <= S or Close_T >= G) exits at the open of T+1 with
  a market order. This is what the current live stack can execute. E1 (intraday resting orders) would need GTC stop
  and limit orders at IBKR, which is a new execution capability.
- **S-SPAC:** SPACs included.
- **U-500 universe.**
- **Close entry:** Carlos-like fill at Close_T; not tradable without an MOC order placed before the close is known.
- **No sweep.**
- **+20 bps per side.** IPO spreads are wider.
- **Terminal haircut:** liquidation at 0.75 x last close.
- **Owner-size accounts:** $30k and $10k capital, where the $1 minimum commission matters.

## 6. Measurements

- **Standalone, with and without the sweep, per block:**
  - CAGR, Sharpe, max DD;
  - beta to SPY;
  - correlation with TAA, L and the `dv2` sleeve (where available);
  - mean cash utilization;
  - trades a year; mean holding days;
  - win rate, average win and loss, mean return per trade;
  - commission and slippage as % of NAV a year.
- **Market-adjusted return:** regression of daily pod returns on SPY total return per block. Annualized intercept and
  its t.
- **Books:**
  - the slot {TAA 0.5, L 0.25, X 0.25} per book block, at engine costs and with +5 bps per side on X and L;
  - controls C_BIL, C_SPY and G3.
- **Capacity:**
  - largest and p95 entry order against ADV20 in dollars, for 2021-26 and the full history;
  - the pod AUM at which the largest 2021-26 order reaches 5% of ADV20.
- **Charts:**
  - Stage A: X_20 by year for IPO-ATH, SPLIT-ATH and BASE-ATH;
  - the event count through time;
  - the N x (P, L) grid heatmaps;
  - A0 equity and drawdown against C_BIL and SPY.

## 7. Tests, invariance and parity (before any Stage A or pod number is read)

- **Unit tests on synthetic panels:**
  - age and window counting;
  - ATH with ties, and the first-day exclusion;
  - the split detector (2:1, 3:2, reverse 1:10, a spin-off 1.07 and a special dividend);
  - the ADV median excluding the listing day;
  - the top-K rank;
  - exits for gap-down, gap-up, both levels touched, and the trailing update using only closes through T;
  - sizing, commission and dividend units;
  - delisting liquidation.
- **V1, download-date invariance.** For truncation dates 2001-06-29, 2010-06-30 and 2020-06-30:
  - the data after the date are dropped;
  - prices are rescaled as a download on that date would show them (x k_T0; Volume / k_T0; Turnover and Unadjusted
    Close unchanged);
  - every event flag and ADV rank over the last 300 sessions must equal the full-panel values.
- **V2:** a random per-symbol constant rescaling of OHLC must leave every event flag unchanged.
- **Engine parity** for IPO-A0 on 2015-01-02..2019-12-31, with the real engine replaying the replica's intents
  (market entries, day limit and stop exits):
  - identical trade list;
  - daily-return correlation >= 0.9999;
  - CAGR within 0.05 pp.

  If the full engine run is too heavy, the window is the evidence and this is stated.
- If anything fails, it is fixed before any Stage A figure or pod figure is read.

## 8. Decision rule (frozen; the same rule as the earlier new-pod studies)

For each population's candidate (IPO-A0 and SPLIT-A0):

- **R1.** Candidate-book Sharpe is strictly above C_BIL's in each of G-P1, G-P2 and G-P3.
- **R2.** Candidate-book max DD is no worse than C_BIL's by more than 2.0 pp, on G-FULL and G-LONG.
- **R3.** R1 and R2 also hold with +5 bps per side on X and L.
- **R4.** The same cell on U-500: candidate-book G-FULL Sharpe > C_BIL's.
- **R5.** In 2021-26 the largest entry order stays below 5% of ADV20 up to a pod AUM of at least $5M.
- **Labels:**
  - beats G3; beats C_SPY;
  - meets the owner's gates (book Sharpe 1.35, max DD -20% on G-FULL);
  - the E2 form's book;
  - S-SPAC; the cost and haircut labels; the $30k and $10k accounts.
- **Confidence:**
  - Reality Check over the 30 cells versus C_BIL: stationary bootstrap, block 21, 2,000 draws, seed 20260928.
  - Paired bootstrap of each candidate versus C_BIL.
  - DSR on returns in excess of BIL, with N = 535 (a lower bound: 503 earlier + 30 cells + 2 populations).
- **If a candidate passes:** recommend a forward paper line only, never a live change. E1 needs GTC exit orders; the
  report must say whether E2 also passes.
- **If none passes:** say so. No further IPO or split variant is tried in this session. Stage A stays descriptive.

## 9. Assumptions and limitations

- A1: the liquidity rank stands in for market cap; spin-offs and re-listings count as new listings; SPACs are
  excluded from the primary set.
- A2: Carlos's figures come from Sharadar, with a market-cap universe, SPACs, unknown costs and fills, and a search
  over 10 settings. They are context, not a target this test must match.
- A3: daily bars cannot order the stop and the target within a day. Both-touched days assume the stop first.
- A4: IPO first-days liquidity and spreads are worse than the engine's 2.5 bps. The +20 bps label bounds this.
- A5: the sweep is 0 before 2007-05-31, so standalone blocks before then understate a T-bill sweep. No book window is
  affected.
- A6: one history. IPO waves (1995-2000, 2013-14, 2020-21) dominate event counts.
