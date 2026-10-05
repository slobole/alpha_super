# Fund products, final pass: GROWTH rebuilt from three capsules; DEFENSIVE verified (frozen plan)

Written 2026-10-05. **Not a blind test.** The three products are three of the fourteen books of an exploratory grid
seen on 2026-10-04, and this study's own input audit built one product (GR3) and one challenger (S4) in the main-frame
convention before the freeze (section 0). Not seen at the freeze: GR1 and GR2 in the main frame, any bootstrap tail of
a new book, the EXACT window, the challenge tests, the robustness battery. A first draft of this plan was reviewed by
three independent reviewers before the freeze; this text is the amended version. The SHA-256 of this file and the
commit that contains it are recorded in `results/research/portfolio/fund_products_20261005/experiment_ledger.jsonl`
before `build_sources.py` or `study.py` is run. Later amendments go to the log at the end, dated, with the reason and
whether they came before or after a result.

## 0. Owner task and what is already known

Owner (Hebrew, 2026-10-05): close the fund products. DEFENSIVE is believed final ("correct me if I am wrong").
GROWTH changed and is to be rebuilt, most likely from the MR capsule, the momentum capsule and TAA. Robustness and
reliability come first; no optimizer that solves for weights; real quant reasoning (example given: three capsules with
relatively low correlation). Keep the product semantics (launch, more return; "the most aggressive could be based on
the 1N TAA with more weight on it", without overdoing it). Verify that the strategy data is right. Full statistics and
charts as before. Keep the ease and AUM constraints. State every important convention and caveat (example given: idle
cash in TAA earns nothing, which is conservative). The owner delegated the quant design.

Known before this freeze:

- The capsules were designed on the same history. MR capsule: about 110 variants examined on 2004-26 (deflated Sharpe
  0.97 full, 0.78 from 2018). Momentum capsule (E2 + 40% sector cap): about 45 trials plus a 151-configuration grid;
  selection verdict against QQQ at the same exposure UNCLEAR; planning Sharpe 0.75. TAA: Defense First family
  (the 3x variant was one of 48 siblings); synthetic TQQQ/BTAL proxy before 2012-10-02.
- Exploratory weight grid, 2008-03-04..2026-08-19 (`mr_capsule_build_20261004/book_weights.py`; house cash for TAA and
  E2, BIL held inside the MR capsule), CAGR / Sharpe (rf 0) / max DD.
  TAA 3x legs: 1/3 each 17.8% / 1.35 / -12.7%; 50/25/25 18.7% / 1.36 / -15.1%; 50/15/35 18.9% / 1.40 / -15.0%;
  TAA 50 / MR 50 19.3% / 1.43 / -15.9%; 60/20/20 19.2% / 1.34 / -17.4%; 20/40/40 17.0% / 1.31 / -14.2%;
  TAA 50 / momentum 50 17.9% / 1.18 / -15.5%.
  TAA 3x 1N legs: 1/3 each 20.1% / 1.29 / -16.3%; 50/25/25 22.1% / 1.28 / -16.2%; 50/15/35 22.4% / 1.31 / -16.9%;
  TAA 50 / MR 50 22.8% / 1.34 / -18.7%; 60/20/20 23.3% / 1.26 / -18.0%; TAA 50 / momentum 50 21.2% / 1.15 / -18.9%.
- The products are grid books: GR1 = TAA 3x legs 1/3 each; GR2 = TAA 3x 1N legs 1/3 each; GR3 = TAA 3x 1N legs
  50/25/25. Among the seven seen weightings of its TAA variant, GR1 has the shallowest max DD (next -14.2%) and is 4th
  on Sharpe; GR3 has the shallowest max DD and is 4th on Sharpe; GR2 is 2nd on max DD and 3rd on Sharpe.
- The same grid was seen on 2017-11-01..2026-08-19 (close to the second half of the section-4 split), TAA 3x legs,
  CAGR / Sharpe: 1/3 each 19.6% / 1.43; 50/25/25 20.9% / 1.48; 50/15/35 22.0% / 1.58; TAA 50 / MR 50 23.5% / 1.67;
  60/20/20 21.7% / 1.49; 20/40/40 18.6% / 1.34; TAA 50 / momentum 50 18.2% / 1.18.
- `momentum_role.py` (seen 2026-10-05), the 50/25/25 book, 2008-26: momentum slot replaced by T-bills 15.5% / 1.41 /
  -14.2% (P 0.78 that its Sharpe is above the book with E2); no momentum (TAA 50 / MR 50) P 0.77; QQQ in the slot
  P 0.35; the three-capsule book had the higher rolling 3-year Sharpe than the no-momentum book in 36% of windows.
  E2 alpha against QQQ: 6.7%/yr (t 2.2) on the full window, 2.7%/yr (t 0.6) since 2017-11.
- Leg statistics seen (2008-26): volatility TAA 3x 17.5%, TAA 3x 1N 23.4%, E2 16.5%, MR capsule 15.0%. Daily
  correlations: MR-TAA 0.28, MR-E2 0.39, TAA-E2 0.52 (0.41 / 0.63 with the 1N variant).
- Input audit of this study (2026-10-05, `audit/inputs_map_03_capsules.py`, written after the first draft of this
  plan): TAA 50 / MOM 25 / MR 25 in the main-frame convention, with taa3x (= S4) 18.8% / 1.37 / -15.0% and at +5 bps
  18.1% / 1.32 / -15.6%; with taa3x_1n (= GR3) 22.3% / 1.28 / -16.2% and at +5 bps 21.5% / 1.25 / -16.3%. Capsules
  alone: MR with BIL held 16.7% / 1.10 / -21.4% (13.6% with +5 bps on all fills, 14.4% on stock fills only; 17.3% / 1.14
  under the fair-cash convention); momentum 14.2% / 0.89 / -23.0% house cash, 14.5% / 0.91 fair cash. Leg capacity by
  route (`audit/capacity/route_model_by_leg.csv`): house gates hold to about $1M (open) and $5M (close-auction
  limits) for the MR capsule, $1M / $5M / $100M (open / close / worked) for the momentum pair, about $1-2M per TAA pod
  on one-day routes (BTAL). Reviewer diagnostic on the published incumbent S9 only: P(max DD < -20%) is 73% with
  independent days, 32% at block 21, 11% at block 63, 5% at block 126, 2% at block 252; at block 63 it is 31% with
  the excess return cut by a quarter and 70% with it halved.
- Earlier fund-products study (2026-10-01, amendments A6 to A7): growth launch S9 = TAA3x-1N 38.4 / NDX-VXN 25.6 /
  CORE5 18 / BTAL_QQQ 18 (18.1% gross, max DD -13.1%, excess Sharpe 1.14, capacity at least $250M on the worked
  route); growth plus 57.4 / 24.6 / 9 / 9. Stock mean-reversion pods carry the A4 / A6-c gate: no launch slot until
  live slippage on their routes is measured at 3-4 bps per side or less.

Because these weightings were seen, the products are not unselected. What is fixed here is the rule, written before
any main-frame tail, challenge or robustness result. Three choices were made with the grid in view and are therefore
shown beside their alternatives, not defended by the data: equal capital across capsules (rather than across pods,
equal risk, cluster parity, or the 50/25/25 base); the order of the ladder (TAA variant first, then TAA share: seen
+2.3 pp CAGR for the variant step against +0.9 pp for the share step with taa3x); the dial stop at one half.

## 1. Inputs

- Existing sleeves: the shelf-rebuild frames (`MAIN/scripts/research/shelf_rebuild_20260929/lib.py::load_inputs`),
  unchanged: END 2026-08-19; LONG window 2008-03-04 -> END with the validated synthetic TQQQ/BTAL proxy before
  2012-10-02 for taa3x, taa3x_1n, btal_qqq; EXACT window 2012-10-02 -> END (no proxy).
- New sleeves: `ndx_atr_cap` = `strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap`,
  `ndx_natr_cap` = `strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled_sector_cap` (the two pods of E2);
  `dv2_g` = `strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil`, `hpi_g` =
  `strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_bil` (the two pods of the MR capsule, BIL parking);
  parking-off runs `dv2_g_cash` / `hpi_g_cash` = `run_dv2_capsule_pod` / `run_hpi_capsule_pod` with
  `parking_enabled_bool=False`. Each runs once through `build_sources.py` with capital $1,000,000, the module's
  default start, to the latest session (2026-10-02), from this worktree (tracked tree = main 5c0d48d). The runner does
  not import `lib.py` or `ga_lib.py` (they put the main checkout, which has another session's uncommitted edits, first
  on the path). Files are written in the shelf-rebuild source format under
  `results/research/portfolio/fund_products_20261005/sources/`. A wrapper (`g_lib.load_inputs`) calls
  `lib.load_inputs()` and appends the new columns to every frame with `lib.nav_to_returns`, `lib.cash_realism_add`,
  `lib.dtb3_annual_rate` and `evaluation.extra_slippage_cost_ser`, cut at END. Nothing is written under the main
  checkout.
- Frames. MAIN = fair cash: positive idle cash earns DTB3 - 0.5%, negative cash pays DTB3 + 1.5% (ACT/360, prior
  observation). Declared asymmetry: in MAIN the idle cash of TAA and MOM earns DTB3 - 0.5% without trading cost; the
  idle cash of the MR pods is a real BIL position (25% dividend withholding, 2.5 bps per trade), with the fair-cash
  formula applied to their residual cash only; the CASH sleeve is BIL total return. Measured before the freeze: this
  is conservative for the MR capsule by about 0.6 pp of capsule CAGR. Sensitivity frames: house cash (engine as is,
  idle cash 0%); +5 bps per side on every traded dollar, BIL fills included; "+5 bps, stock and risk-ETF fills only"
  (BIL fills of the MR pods excluded); +10 bps (linear extrapolation as in A6); unscaled proxy; EXACT window; "MR fair
  cash" (the parking-off runs plus `cash_realism_add`: the symmetric treatment) and "MR cash 0%" (the parking-off runs
  as they are). Rules and challenge tests use MAIN and the all-fills +5 bps frame. Not used for growth:
  `s4_etf_idle_pre2010` (only DV2-IND) and `s5_hpi_live_gap` (historical since the align-live fix 8b21a2e).
- Cash, stated for the owner. In the engine, Bench and the YAML books idle cash earns 0% (understates returns). The
  MAIN frame used for every headline number is not that convention. Measured size of the difference, pp of CAGR a
  year, LONG / last three years: taa3x +0.02 / +0.05 (it holds about 2% cash, so the point is immaterial for TAA);
  taa3x_1n +0.05 / +0.11; each momentum pod +0.25 / +0.5 (about a quarter to a third idle). Every headline table
  shows the house-cash figure beside the main-frame figure.
- Metrics. Excess Sharpe = mean of (r - BIL) over its standard deviation on daily returns, x sqrt(252): the decision
  metric (as amendment A1 of the predecessor, because BIL is itself a sleeve). The house Sharpe (zero risk-free rate)
  and the excess Sharpe on calendar-month returns (x sqrt(12)) are printed beside it; any pair of books whose order
  differs between the daily and monthly ratio is marked. CAGR is annualised with 252 sessions a year in every table
  and solver of this study (the convention of the A6 tables that stay in force for DEFENSIVE).
- Gross is primary; investor net of 2/20 (daily accrual, yearly payment, high-water mark; `ga_lib.fee_nav`) is
  reported beside it.
- Data verification (finished before any product book is built; result in the ledger). Scope: taa3x, taa3x_1n,
  ndx_vxn, dv2, hpi_vote, core5, btal_qqq and the defensive legs etf_dv2, eom_flow, downshock. Each is re-run as
  `run_sleeves.py` ran it ($1M, requested start 2000-01-03) from the clean worktree and compared with the stored path
  on the stored dates. Pass: max |daily return difference| <= 1 bp and |CAGR difference| <= 0.05 pp. A failing sleeve
  is replaced by its re-run in every frame and the cause is written down; if it is a defensive leg, section 6 becomes
  a re-run of the A6-d rules shown beside the stored slots; the study stops only if a difference cannot be explained.
  The proxy-era runs and the pre-2010 DV2-IND fill cannot be re-run in the engine at HEAD; they are verified by file
  hash against the shelf-rebuild ledger. The four new sleeves have no stored $1M counterpart; their check is the
  capsule-level comparison with the 2026-10-04 builds (CAGR within 0.3 pp, daily correlation >= 0.999). Audit result
  as of the freeze: taa3x, taa3x_1n, core5, btal_qqq, ndx_vxn byte-identical to the stored files (same SHA-256) and
  causal in the end date; the E2 book bit-identical to the stored 2026-10-04 run; all stored sleeves FRESH by code
  lineage (no commit after f9ad358 touches a backtest path); the main checkout's uncommitted edits are
  backtest-neutral and no stored series came from that tree; the dv2 / hpi_vote / defensive-leg re-runs and the MR
  capsule re-run were still running.

## 2. Capsules

A capsule is one return engine. It enters a product as one unit at its internal weights.

| Capsule | Pods (internal weights) | Decisions |
|---|---|---|
| TAA | `taa3x` (default) or `taa3x_1n` (the higher-octane variant; same engine, correlation ~0.93) | monthly |
| MOM | `ndx_atr_cap` 50 / `ndx_natr_cap` 50 (E2 with the 40% sector cap) | monthly |
| MR | `dv2_g` 50 / `hpi_g` 50 (one shared VIX stress gate, BIL parking) | daily |
| CASH | BIL | hold |

The defensive pair CORE5 / BTAL_QQQ is not a capsule of the growth products: BTAL_QQQ is the Defense First engine
again (correlation ~0.86 with the TAA pods), so de-risking inside a product is done with BIL only, and at the client
level by holding less of the growth fund or blending it with the DEFENSIVE fund.

Book model: the house pod model (`lib.book_returns`): pods compound independently and are reset to their target
weights at the first session of each calendar year (a transfer with no trade and no cost). The intended live form of
MOM is one pod holding the average of the two books (50/50 at every monthly rebalance); the product model uses two
pods reset annually. Measured before the freeze on the stand-alone pair: CAGR 14.23% against 14.22%, largest daily
difference 0.11%; treated as equivalent. Ease is reported both ways (five research pods, four live pods).

## 3. Products (fixed by rule, not by search)

Principle. The product has two risk clusters, not three independent bets: a Nasdaq risk-on pair (TAA, whose return
leg is TQQQ, and MOM, which holds Nasdaq-100 stocks; daily correlation about 0.5-0.6) and a stress-regime dip buyer
(MR; about 0.3-0.4 with each). Future Sharpe ratios are unknown and every engine was selected on this history, so the
flagship uses no estimated mean: equal capital over the three engines. Equal capital is the midpoint of two priors
that pull in opposite directions: cluster parity (25 / 25 / 50) would put half the capital in the newest,
not-yet-live, most cost-sensitive and capacity-binding engine; maturity weighting (50 / 25 / 25) would put about 83%
of the risk in the Nasdaq pair. With the section-0 figures equal capital is within 5 points of inverse volatility
(31 / 33 / 36) and of equal risk contribution (31 / 31 / 38), so these three rules are one choice, and it caps the
capital behind any one engine at one third. It is not cluster-neutral (about 73% of GR1's risk is the Nasdaq pair)
and it is not claimed to be optimal or minimax (that would need exchangeable engines). The return dial (GR2, GR3)
deliberately leaves this principle: it is a view that the TAA engine keeps its edge, taken because TAA is the only
engine with built-in, margin-free leverage and trades monthly. The satellites always split the remainder equally.

| Product | Weights (capital) | Target rung |
|---|---|---|
| GR1 Growth (flagship) | `taa3x` 1/3, MOM 1/3, MR 1/3 | GROWTH |
| GR2 Growth Plus | `taa3x_1n` 1/3, MOM 1/3, MR 1/3 | GROWTH PLUS |
| GR3 Aggressive (TAA-led) | `taa3x_1n` 1/2, MOM 1/4, MR 1/4 | AGGRESSIVE |

In earlier studies "G3" means TAA3x 50 / NDX-VXN 50; here it appears only as challenger S10.

Rungs. GROWTH (LONG max DD >= -17%, P(max DD < -20%) <= 15%) and GROWTH PLUS (-22%, -25%) are the owner's limits of
the earlier studies (the second pair was named AGGRESSIVE in the growth study of 2026-09-30). AGGRESSIVE (-27%, -30%)
is new in this plan and extends the same 5-point step (A7 only illustrated -30% / -30%); it is marked "needs the
owner's confirmation" in the report. A rung is a ceiling, not a target: on the seen grid the products sit between
-12.7% and -16.3%, so the historical leg of every target rung is known to pass and only the bootstrap leg is open. A
rung passes only if all hold in the MAIN frame and in the +5 bps frame: LONG historical max DD >= the build limit, and
P(max DD < hard limit) <= 15% for the 10-seed mean and for the worst seed. Bootstrap: stationary, mean block 63, 2,000
paths of the LONG window's length (about 18.5 years, so P is a whole-period figure), seeds 20260929 + 0..9, gross
returns. Declared change: A6 only reported the growth limits and A6-d applied the double test to defensive slots;
here it gates the growth rungs.
Reading, fixed now: the breach figure is a yardstick on one convention (block 63, full backtest edge), not the chance
of that drawdown; the ten seeds measure Monte Carlo noise only. The report never states a breach figure without that
qualifier, and prints beside every rung result, without changing the pass rule: the figure at blocks 1, 21, 126 and
252; in the unscaled-proxy frame and on the EXACT window; over the first 3 and 5 years of each path; with all
capsules at k = 0.75 and k = 0.5 (section 5.3); the book's variance ratio at 21 and 63 sessions; and the edge margin
(the lowest k on a 0.05 grid at which the product still passes its target rung, MAIN frame, 10-seed mean). A product
that passes only at k = 1 is labelled "passes at the full backtest edge only".
If a product fails its target rung, BIL is added: weights x (1 - c) plus BIL c, c = 5%, 10%, ... up to 30%, BIL as its
own pod in the annual reset. The smallest passing c defines the product and its YAML, named "GRx + n% BIL" and
labelled "calibrated to the rung in sample"; the unscaled book is shown beside it. If 30% is not enough the product is
reported as "does not fit its rung" and is not offered at that rung. The label of a product states the strictest rung
it passes. No product is changed after results: no dial is turned up to use spare room in a rung, cash is never
removed, and no product is re-weighted or renamed if two products share their strictest rung.
Menu rule: a higher product is offered only if, after any BIL, its MAIN-frame LONG CAGR is at least 1.5 pp above the
product below it; otherwise the menu has fewer products and the report says so (on the seen grid the steps are +2.3
and +2.0 pp, so three products are expected; the rule exists for the case where BIL scaling erases a step).
Dial guard: no product holds more than 1/2 of capital in the TAA capsule; every product table shows capital share,
risk share per engine and per cluster; if the TAA share of variance exceeds 2/3 in GR3 this is stated beside its
headline with the 40 / 30 / 30 dial point as the alternative.

Stages (fixed now). GR1, GR2 and GR3 are capsule products. They need (a) the momentum capsule and the MR capsule
wired, and (b) the MR gate, carried over from A4 / A6-c and made measurable: paper or live slippage on the MR
capsule's opening orders of at most 4 bps per side over at least 200 stock fills. Until both hold, the growth product
that can run is the incumbent monthly book S9, whose YAML is kept. Declared deviation from the predecessor's champion
rule (the incumbent stays unless beaten on >= 80% of paired paths): here GR1 is the default by construction, on the
owner's direction and because S9's weights came out of a resampled optimiser on this same history; the test GR1
against S9 is reported as primary evidence in both directions, and the report says plainly if GR1 does not beat S9
at 80%. Reported for the gate: GR1's launch form GR1-L (the MR third in BIL) with the same statistics; the paired
share of GR1 over GR1-L on excess Sharpe and on CAGR in the MAIN, +5 and +10 bps frames; and the extra cost per side
at which GR1 and GR1-L have the same excess Sharpe and the same CAGR (linear in the 0 / +5 / +10 bps frames). If the
measured slippage fails the gate, the growth product stays S9 and a new frozen plan is needed.

Reported beside the products, selecting nothing:
- The dial map: TAA share in {1/3, 40%, 50%, 60%, 2/3} x variant in {`taa3x`, `taa3x_1n`}, satellites equal (10
  books; the three products are three of them), with pure `taa3x` and pure `taa3x_1n` as the ceiling.
- Construction table: GR1 beside S6 (equal capital over pods), S11 (cluster parity), S12 (inverse volatility fixed in
  advance), S4 (core and satellites), S7 (walk-forward inverse volatility) and taa3x at one half (the other ladder
  order), with GR1's rank on excess Sharpe, CAGR, max DD and P(max DD < -20%).
- Exposure look-through for every product and dial-map book: TQQQ weight inside the TAA pod (mean, 90th percentile,
  max, at month ends); look-through Nasdaq-100 notional = 3 x TQQQ weight + MOM invested weight, per unit of product
  NAV at target weights (mean, 90th percentile, max, by year); a one-day gap table: product loss for a Nasdaq-100 fall
  of 5 / 10 / 15 / 20% at mean, 90th-percentile and peak exposure (TQQQ = 3 x the index move, MOM beta 1, MR at its
  mean and peak stock weight with beta 1, other assets flat). The bootstrap resamples observed days and cannot
  produce a gap larger than the sample's; the gap table is the complement.
- Margin alternatives (descriptive; the choice between margin and the dial is labelled judgement). Debt is a pod with
  a negative weight: L x w for the product's pods and -(L - 1) for a debt pod that grows at DTB3 + 1.5% (ACT/360),
  reset annually with the book, so leverage drifts inside the year; constant daily leverage is the sensitivity.
  Compared at matched LONG volatility (L = target volatility / own volatility, MAIN frame; the CAGR-matched L is a
  second row): GR1 x L against GR2 and against GR3, and GR2 x L against GR3. L is held fixed in every other frame.
  Reported per route: CAGR (MAIN, +5 bps, EXACT, spread 0.5% and 2.5%), max DD, the comparison product's rung breach
  figure, worst year, crises, largest engine risk share, the engine-dead figure of 5.3, peak Nasdaq look-through and
  the gap table, peak leverage inside a year, worst-case Reg-T initial requirement (3x-ETF pods 75%, other pods 50%,
  times L; above 90% flagged as needing portfolio margin). Expected before the run: the levered GR1 wins on drawdown
  and breach at equal risk because its seen Sharpe is higher (1.35 against 1.29 and 1.28); the informative rows are
  the 2.5% spread, +5 bps, EXACT and Reg-T. No winner is declared from differences inside 0.3 pp of CAGR or 0.5 pp of
  drawdown / breach. Availability by stage is stated: (a) pods in separate Reg-T accounts (today): only the dial is
  practical; (b) a fund with one cross-margined or portfolio-margin account: both.
- Client blends: DEFENSIVE launch + GR1 at 25 / 50 / 75% (annual reset), each beside GR1 diluted with BIL to the same
  LONG volatility, with the Defense First share of capital (TAA + BTAL_QQQ).
- Launch readiness: GR1 with today's wired stand-ins (`ndx_vxn` for MOM; ungated `dv2` 50 / `hpi_vote` 50 for MR;
  each alone and both), each with the GROWTH rung test and the section-4 test against S9. A stand-in book that holds
  ungated dv2 / hpi_vote stays behind the same MR gate.

## 4. Challengers and slot tests

Structures, all with `taa3x` unless stated:
S1 no momentum: TAA 1/2, MR 1/2. S2 no MR: TAA 1/2, MOM 1/2. S3 no TAA: MOM 1/2, MR 1/2.
S4 core and satellites: TAA 50 / MOM 25 / MR 25. S5 MR tilt (seen): TAA 50 / MOM 15 / MR 35.
S6 equal pods (seen): TAA 20 / MOM 40 / MR 40. S7 walk-forward inverse volatility over the three capsules: capsule
return columns are built first (each capsule = its pods at internal weights, annual reset); weights at each annual
reset come from `lib.book_returns(rule="IV")` on those columns in the MAIN frame (trailing 252 sessions strictly
before the reset) and are reused in every frame; a period with fewer than 60 sessions of history (2008 in LONG, 2012
in EXACT) uses equal weights. S8 GR1 75 / defensive pair 25 (TAA, MOM, MR, CORE5 60 / BTAL_QQQ 40 at quarters).
S9 the incumbent growth launch (TAA3x-1N 38.4 / NDX-VXN 25.6 / CORE5 18 / BTAL_QQQ 18). S10 the old G3 (TAA3x 50 /
NDX-VXN 50). S11 cluster parity: TAA 25 / MOM 25 / MR 50. S12 inverse volatility fixed in advance from the section-0
volatilities: TAA 31 / MOM 33 / MR 36.
Slot tests on GR1 (the house slot rule): T1 = MOM replaced by BIL; T2 = MR replaced by BIL (= GR1-L); T3 = TAA
replaced by BIL; T4 = MOM replaced by QQQ total return. Run in the MAIN, +5 bps and +10 bps frames. Reported: excess
Sharpe, CAGR, max DD, P(max DD < -20%), and GR1's paired share over each on excess Sharpe and on CAGR, full window and
by block. Reading, fixed now: a capsule whose BIL replacement has the higher excess Sharpe on at least 80% of paths is
reported as "adds return, not risk-adjusted return, to this book" (seen for momentum in the 50/25/25 book: P 0.78);
it stays in GR1, because the product's aim is growth inside the rung, and the report states its price in Sharpe and
its gain in CAGR.

Challenge test. LONG window, MAIN frame unless stated; paired stationary bootstrap, mean block 63, the same resampled
rows for the challenger, GR1 and BIL. GR1 is the unscaled GR1. A challenger passes only if ALL hold: (0) S1-S8, S11
and S12 pass the GROWTH rung of section 3 with no added cash (S9 and S10 are tested as they are and their rung result
is printed); (1) excess Sharpe above GR1's on at least 80% of the 20,000 paired paths of seeds 0-9 pooled, no
re-draws; a pooled share between 75% and 85% is reported as "borderline"; (2) P(max DD < -20%), 10-seed mean, not
above GR1's; (3) excess Sharpe on the EXACT window above GR1's; (4) excess Sharpe in the +5 bps frame not below GR1's
in the same frame (declared change: A6's growth challenge used +5 bps CAGR); (5, 6) excess Sharpe above GR1's in each
half of the main-frame series (through 2017-06-30, after it; sliced from the full-window book). Printed for every
challenger and not part of the pass rule: the result without check 2 (GR1 had the shallowest drawdown of the seen
grid, so check 2 is partly decided by what was seen); the Sharpe gap with its 90% bootstrap interval; the paired
share on CAGR and the CAGR gap; the gap in each of blocks A (proxy era), B and C; the monthly-return Sharpe gap.
Every test is also run in the reverse direction. GR2 is tested against the old growth plus (57.4 / 24.6 / 9 / 9) by
the same test with the breach at -25%; reported.
Reading, fixed now. One comparison at an 80% paired share is about a one-in-five false-alarm test when the true
Sharpe ratios are equal; with twelve comparisons at least one lucky pass is likely, and a challenger that is truly
better by 0.05-0.07 Sharpe passes less than half the time. The first half is half proxy data and the EXACT check
overlaps both halves, so checks 3, 5 and 6 are about one and a half independent checks. So a pass means "candidate
for forward tracking" and no pass means "no evidence against the default"; neither is confirmation, and the report
does not claim that equal weight is the best book.
Consequences, fixed now. No challenger replaces GR1, GR2 or GR3 in this study, whatever the result; a passing
challenger is listed with its trade-off, receives the full section-5 battery, and is paper-tracked beside GR1; a
product change needs a new frozen plan.

## 5. Robustness battery

Reported for GR1, GR2, GR3 (and their final BIL-scaled versions), S1, S4, S5, S9, S10, S11, the slot-test books
T1-T4, the single capsules, and any other challenger that passes section 4.

1. Sub-periods: blocks A (2008-03..2012-10, proxy era), B (2012-10..2021), C (2022..END), RECENT (2023-08..END),
   halves, calendar years; EXACT window.
2. Frames of section 1.
3. Edge decay (mean shift, risk unchanged). For each pod p let mu_p be the arithmetic mean of its daily return minus
   BIL on the LONG window, in the frame being evaluated. Scenario k: r_p' = r_p - (1 - k) x mu_p on every session; the
   pods of one capsule share k. This models the whole excess return shrinking (alpha and the market premium the pod
   carries), not alpha alone; at k = 0 a pod's CAGR is below BIL by about half its variance (a deliberately harsh dead
   engine). Scenarios: all capsules at k = 0.75 and at k = 0.5; one capsule at k = 0 with the others at 1, and again
   with the others at 0.75; a common shock in which every pod loses beta_p times half the sample QQQ excess return
   (beta_p from a weekly regression of the pod's excess return on QQQ's). Reported per scenario: CAGR, excess Sharpe,
   and the breach figure of the product's own target rung (10-seed mean). The single-path max DD is shown but not
   interpreted (a flat subtraction turns any below-average stretch into an artificial drawdown). A book's worst case
   = its lowest excess Sharpe over the three one-capsule-dead scenarios, with the CAGR beside it, computed with the
   annual reset and with no reset, for the products, GR1's 5.4 neighbours and S1-S12. "Minimax" is used in the report
   only if GR1's worst case is within 0.02 excess Sharpe of the best worst case among its neighbours and the
   challengers. k = 0.75 and 0.5 are conventions, not estimates.
4. Weight plateau: the 19 books at offsets (a, b, c) in {-10, -5, 0, +5, +10} pp with a + b + c = 0 added to the
   product's (TAA, MOM, MR) weights, same TAA variant (the product is one of the 19). Minimum, quartiles and maximum
   of CAGR, excess Sharpe, max DD and the breach figure of the product's target rung (10-seed mean), and the product's
   rank on each. Reading rules, fixed now: a product's rung is called robust only if the median historical max DD and
   the median breach figure of its 18 neighbours also satisfy the rung, otherwise the label reads "passes; its
   neighbours do not"; a product in the best three of its neighbourhood on max DD or on excess Sharpe is flagged
   "local best, expect worse"; in the worst three, "the principle costs x".
5. Reset policy. Fixed: annual, first session of the year; this check cannot change it. It reports the spread over
   the twelve annual start months, and none / quarterly / monthly resets (the study's own period function; resets are
   cost-free transfers, so frequent resets are flattered and the table is descriptive; a second column charges 2.5 bps
   on the capital moved at each reset). If the January figure lies outside the middle half of the twelve start months
   (CAGR or excess Sharpe), the planning figure is the median of the twelve. Also reported: the largest capital share
   each capsule reaches inside a year.
6. Start dates: each calendar year 2008..2021 as the start; rolling windows of 756 and 1,260 sessions (excess Sharpe
   and CAGR: minimum, 10th percentile, median; share of windows above each single capsule and above S9).
7. Leave one calendar year out: range of excess Sharpe and CAGR.
8. Bootstrap: P(max DD < L) for L in -10, -15, -17, -20, -22, -25, -27, -30, -35% (10 seeds, mean and worst), plus the
   rung-reading items of section 3 (other blocks, horizons, frames, k, variance ratio, edge margin); CAGR percentiles
   5 / 25 / 50 / 75, gross and net of 2/20. A separate tail cache keyed by the frame fingerprint and the limit list.
9. Dependence. Correlation matrices of the capsules: full, by block, each half, on the S&P 500's and on QQQ's worst 5%
   of days, in the five named crises, by MR-gate state (gate recomputed with `strategies.mr_capsule.vix_stress_gate`),
   on daily, weekly and 21-session returns; rolling 252-session correlations (min, 10th percentile, median, 90th, max);
   the six co-fall windows; each capsule's risk share (Cov(contribution, book) / Var(book)) and its share of the
   product's mean excess return; the product's T-bill-like share through time (BIL inside MR pods plus idle cash).
   Stability flags, fixed now: for each product the diversification ratio (weighted sum of capsule volatilities over
   book volatility) in each half and in blocks A, B, C; the second-half book volatility predicted from first-half
   correlations and second-half capsule volatilities against the realised one, and the reverse; flag if realised is
   above predicted by more than 15%, if any pairwise correlation moves by more than 0.20 between halves, or if any
   pairwise correlation exceeds 0.70 in a block, a crisis window or on the worst 5% of days. Tail dependence: for each
   pair, P(B in its worst 5% | A in its worst 5%) on daily and 21-session returns against the 5% of independence; the
   number of calendar months in which all three capsules lost and the product's return in them; each capsule's return
   over the product's ten worst 21-session windows and over its deepest drawdown; the effective number of bets from the
   correlation matrix (full sample, each half, worst 5% of days).
10. Factor alpha, gross and net of 2/20: weekly excess returns, Newey-West 4 lags; QQQ alone; QQQ / IEF / GLD / DBC /
    UUP; plus the QQQ 200-day rule; LONG window, halves and EXACT (method of `allocator_first_look.py`).
11. Capacity and ease. Route model = `growth_aggressive_20260930/report_data.py::capacity`, orders 2023-08-21 -> END,
    with these classifications added: `ndx_atr_cap` and `ndx_natr_cap` in the Nasdaq set (monthly, may be worked);
    `dv2_g` and `hpi_g` in the urgent set (stock orders cannot wait); BIL, in any pod, is an ETF order. Cost cap = 25%
    of the product's EXACT-window excess CAGR over BIL in the MAIN frame. Reported per product and per dial-map book,
    at $10M / $25M / $50M and as recommended AUM, for three routes: everything at the open (the only route the
    backtests model), the close auction, and worked + blocks with the MR stock orders in the close auction (not
    modelled by the engine and not supported by the DV2 timing study: shown as an upper bound and labelled so), each
    with the binding leg and symbol. AUM rule, fixed now: a product whose worked-route recommended AUM is below $25M is
    labelled CAPACITY-LIMITED on its headline line with the binding leg named, and S9 (the monthly book) is shown
    beside it as the scalable alternative. Beside the model: a plain participation screen per leg on the dollar volume
    at the order date (pod AUM at which the 90th-percentile and the 99th-percentile order equal 5% of a median day),
    and the BTAL 10%-ownership wall per product. All capacity figures are pre-TCA. Ease per product: pods and accounts
    (one account per pod; a margin account for each MR pod), trading days and orders per year, pods not wired at the
    study's commit, margin needs, minimum clean product size, fee income.
12. Engine confirmation. Each final product as a flat `portfolios/fund_growth*.yaml`: one pod per strategy, weight =
    capsule share x internal weight, rebalance annually / fixed, start 2012-10-02. Run through the PortfolioManager at
    the study's commit. Compared with the research book model in the house-cash frame on 2013-01-02 -> END (from the
    first annual reset, so the cold start of 2012-Q4 drops out). Accepted if |CAGR difference| <= 0.3 pp, |max DD
    difference| <= 1.0 pp and daily return correlation >= 0.995; otherwise the cause is found before the report is
    written.
13. After the window: 2026-08-20 -> latest session, from the verification re-runs (existing sleeves) and the fresh
    runs (new sleeves); house cash; descriptive. Labelled "not out of sample": both new capsules were designed on data
    through 2026-10-02.
14. How much to believe. (a) 90% bootstrap interval of excess Sharpe and CAGR for each product (seeds 0-9 pooled).
    (b) P(excess Sharpe < 1.0), P(< 0.75), P(< 0.5) from that distribution, at k = 1 and at k = 0.75. (c) A deflated
    Sharpe ratio at book level (`alpha/stats/psr_dsr.py`) for N = 100 and N = 1,000 trials, with the note that trials
    are correlated and that passing a test against zero says nothing about the size of the haircut. (d) Headline rule:
    every summary table of the report shows three columns per product: backtest (MAIN frame), planning (k = 0.75 and
    +5 bps) and floor (k = 0.5 and +5 bps); the backtest column is never quoted alone, and the text says that all
    history through 2026-08 was used to choose the engines, so no figure is out of sample.

## 6. DEFENSIVE

No rebuild. Verification (results in the ledger): (a) the stored sleeve files of the defensive legs match their
recorded hashes and their strategy modules and shared engine dependencies are unchanged between f9ad358 and the
study's commit, or their re-run passes the section-1 threshold; (b) `a6.py` and `a6d.py`, unchanged, re-run with an
empty tail cache reproduce `a6.json` / `a6d.json`. If (a) or (b) fails, the defensive menu is re-run by the unchanged
A6-d rules on the corrected frame and shown beside the stored slots.
The defensive slots keep their A6-d books, including "more return" (60/40 + growth slice S9 + cash), whose YAML is
unchanged. Shown beside them as target-stage rows, replacing nothing: (1) the "more return" scan of A6-d with the
growth slice = unscaled GR1 (first a parity run of the same code with S9 must reproduce the stored slot); (2) the
gated upgrade restated with the capsule: 60/40 at 90% + MR capsule 10%, with the smallest cash passing the DEF double
test, challenge-tested against the launch slot, paired shares at 0 / +5 / +10 bps, behind the MR gate. The 22% levered
growth + core mixes of A6 and the higher targets of A7 are not re-run; the margin alternatives of section 3 replace
them for the capsule products, and the stored rows stay in the 2026-10-01 record.

## 7. Deliverables

- YAMLs: `portfolios/fund_growth.yaml` (GR1), `fund_growth_plus.yaml` (GR2), `fund_growth_aggressive.yaml` (GR3). The
  2026-10-01 files are kept under what they are: `fund_growth_monthly.yaml` (S9, the book that can run before the
  capsule conditions hold and the scalable alternative) and `fund_growth_plus_monthly.yaml`; `fund_growth_mr.yaml` is
  deleted as superseded (its HPI-RSI pod was demoted) and this is noted in the record. These files are untracked in
  both checkouts; the main checkout's copies are not touched by this study.
- Records in English: `docs/research/FUND_PRODUCTS_20261005.md` (self-contained); a section "Fund product books" in
  `docs/strategies/book-strategy-caveats.md` with one row per item of the caveat list (direction, size, evidence).
  No Scout registration (book construction is outside Scout's scope).
- Report: the Hebrew page with English statistics, updated in place. Its "important to know" section is a table with
  four columns (convention or caveat; direction: conservative / optimistic / unknown; measured size; source). Minimum
  rows, fixed now: (1) cash convention (engine 0% against MAIN fair cash; sizes per capsule); (2) the products are not
  fully invested (BIL inside the MR pods, idle momentum cash); (3) the MR capsule's BIL is modelled with withholding
  and trade costs (conservative) and three BIL treatments coexist; (4) TAA / BTAL_QQQ proxy before 2012-10-02 (a
  quarter of the sample and the only 2008); (5) TAA commissions on split-adjusted TQQQ shares (conservative); (6)
  negative cash unfinanced in the engine (optimistic, small; charged in MAIN); (7) TQQQ look-through and the gap
  table; no Nasdaq bear of the 2000-02 kind in the sample and two of three engines long Nasdaq in risk-on; (8)
  selection of each capsule and of the book (products seen before the freeze): plan on k = 0.75; (9) breach figures
  are a block-63 convention at the full backtest edge; (10) MR capsule cost sensitivity, turnover, no live fills, the
  unmet slippage gate; (11) momentum capsule: today's GICS labels, selection unproven against QQQ at the same
  exposure, what its slot test shows; (12) annual reset is a cost-free transfer on one date; (13) capacity by route,
  binding legs, pre-TCA; (14) minimum account size and that the capsule products do not fit today's account; (15)
  wiring status of every pod; (16) margin rows: financing convention, no margin-call modelling; (17) gross against
  2/20 net, no fund expenses; (18) Sharpe basis (zero rate against excess over BIL; daily against monthly); (19) the
  weeks after END are not out of sample; (20) the engine BIL pod against BIL total return.
- Forward review triggers, fixed now (a trigger opens a review; it is not an automatic exit): a capsule's live or
  paper drawdown beyond its planning maximum (MOM -30%, MR -21%, TAA -26%); MR slippage above 4 bps per side over 200
  fills; a capsule behind BIL over a rolling three years.
- An independent review before the report is final.

## Amendment log

Everything above this heading is the plan as frozen in commit 70f5c22 (SHA-256 of that file bc3c6521..., first line
of the study ledger) and has not been edited. Every entry below is dated 2026-10-05 and was written **after results**.
None changes a product, a weight, a rung limit or a decision rule; they record where the code or the report departs
from the text above, why, and the numbers before and after where a number moved. Entries R1, I1 and C1 are the ones
the code comments cite.

- **R1 (comparison against the monthly book by cost and by block; `versus.py`).** Added after the independent review:
  the paired comparisons of sections 3 and 4 are also run at +5 and +10 bps and inside blocks A, B, C and RECENT
  (20,000 paths each). Reason: the first report quoted the GR1-over-monthly share only in the main frame (90.2%), and
  the plan (5.1, 4) asked for the block evidence. Result: 90.2% / 71.4% / 42.2% at 0 / +5 / +10 bps; blocks A 52%,
  B 99.5%, C 29%, RECENT 18%. The summary was rewritten: at the planning cost GR1 does not beat the monthly book at
  the 80% bar, its Sharpe lead comes from 2012-2021, and the monthly book has led since 2022.
- **I1 (exposure look-through; `exposure.py`).** (a) The look-through and the gap table cover 2012-10-02 to END only:
  the synthetic TQQQ of the proxy era is not stored as a price series, so 2008 has no look-through. (b) Statistics are
  daily, not month-end (TQQQ weight in TAA 3x: mean 25.6% daily against 25.8% month-end). (c) The gap table uses the
  distribution of the joint same-day equity exposure, not Nasdaq exposure plus MR at its own peak (peak 1.67x against
  1.70x). (d) After review, the look-through with the pod weights the book actually carried between annual resets is
  reported beside the target-weight figure: peak 1.34x / 1.34x / 1.76x at target weights against 1.51x / 1.56x /
  1.97x carried (reached in December 2013; since 2015 the carried maximum is 1.12x / 1.32x / 1.65x).
- **C1 (capacity; `capacity.py`).** Three changes after the review. (a) The MR pods' BIL parking orders are left out
  of the gates and of the cost. This departs from "BIL, in any pod, is an ETF order": those rows were about half of a
  capsule product's ETF order rows and pulled the 95th percentile of the one-day ETF gate down, letting the BTAL
  orders pass at twice the size. Close-auction capacity of GR1-GR3 moved from $10M to $5M; the open ($2.5M) and the
  worked route ($10M) did not move. (b) Pod weights inside the order window are carried from the January 2023 reset
  instead of restarted at target weights at the window start (the TAA pod's share differed by up to 3.8 pp in GR1 and
  8.8 pp in GR3). (c) The participation screen is per leg, as 5.11 wrote it; the first build pooled the orders of all
  legs. GR1 at the 99th-percentile order: $34.5M pooled, $3.0M per leg (binding leg TAA, symbol BTAL). The book-level
  largest same-day order (all pods summed) is printed beside it, because the monthly book's two Defense First pods
  send BTAL orders on the same day.
- **5.5 (start month).** GR1's January reset lies outside the middle half of the twelve start months. The rule's
  figure is the median, 18.01% / 1.266; the headline keeps January (17.92% / 1.260), which is conservative by 0.09 pp.
  The table column is renamed so it is not confused with the 5.14 planning column.
- **5.3 (edge decay).** S7 is left out of the worst-case set (the reviewer's value for it is 0.795; the "not minimax"
  wording does not change). T4's QQQ slot is not decayed. Added after review: the scenario "Defense First dead (TAA
  and BTAL_QQQ)", because for the monthly book "TAA dead" switched off only one of its two pods on that engine (excess
  Sharpe 0.45 with one off, 0.31 with both); the TAA-dead scenario for the dial-map books; edge margins at every rung
  limit. The common-shock column is labelled a regression-beta shock: a regression beta understates the time-average
  exposure of pods that leave the market in high volatility, and the reviewer's figure with time-average exposure
  (GR1 11.9% / 0.85) is quoted in the caption as the reviewer's, not as a study output.
- **Battery scope.** Reset policy (5.5), the rung reading items (5.8), the after-window check (5.13) and intervals and
  DSR (5.14) run for GR1-GR3 and the monthly book only; factor alpha (5.10) for those and the capsules. 5.6: the share
  of windows was computed for 756-session excess Sharpe in the first build and extended after review to 1,260 sessions
  and to CAGR. 5.8: CAGR percentiles use seed 0 only (2,000 paths). Slot-test block shares use the main frame and
  seeds 0-2 (6,000 paths).
- **Rung reading for GR3.** The first build printed the reading items only at -30%, the limit of the new AGGRESSIVE
  rung. GR3 is labelled with the strictest rung it passes (GROWTH PLUS), so the items are now printed at -25% as well:
  5.8% by the rule, edge margin 0.85 (GR2 at the same limit: 2.6%, 0.70). At the planning convention GR3 does not
  meet the GROWTH PLUS cap and GR2 does; the report says so and does not present the fallback label as equivalent.
- **5.14(d) (three-column rule).** The backtest / planning / floor columns are shown on the cards, the menu, the dial
  table and the "how much to believe" table. The margin, slot, challenger, construction and stand-in tables are
  backtest-frame comparisons between books and are labelled as such; the report's self-description was corrected.
- **+10 bps frame.** As section 1 says, it is the linear extrapolation r0 + 2 x (r5 - r0) at book level, not 10 bps
  charged inside each pod; the reviewer's pod-level run moves two printed percentages and the breakeven by one unit in
  the last digit.
- **Margin rows.** L is the ratio of the two unlevered volatilities; leverage decays between annual resets, so the
  realised volatility of the levered book is slightly below the target's (GR1 x1.32: 16.5% against 16.8%). The rows
  are labelled with the realised volatility and the mean leverage. The reading was rewritten after review: levered
  GR1 is ahead of GR2 beyond the tolerance on drawdown, breach and CAGR (a tie at +5 bps and at a 2.5% spread); mixed
  against GR3; GR2 on margin is worse than GR3 on drawdown. No winner is declared; that is a judgement, not a
  tolerance result.
- **5.12 (engine confirmation).** The first YAMLs of GR1 and GR2 were rejected by the PortfolioManager (six-decimal
  weights summed to 1.000005). The weights are now normalised before rounding (GR1: 0.333332 + 4 x 0.166667). The
  first report build ran before the three runs finished and showed "pending"; the published page was built after all
  three were accepted (GR1 20.23% against 20.24%, GR2 22.78% / 22.79%, GR3 25.19% / 25.19%; correlation 1.0000).
- **6(b) (old defensive study).** `a6d.py` reproduces `a6d.json` (1,011 fields, 0 different) and `report_a6.py`
  reproduces `report_a6.json` (14,899 fields, 0 different). `a6.py` does not reproduce `a6.json` exactly: 7 fields,
  all `crises_dd.tariffs_2025` of the old growth rows, differ by up to 0.36 pp. Cause: `a6.py` was edited at 16:28 on
  2026-10-01 (the in-window drawdown excludes the window's start day) after `a6.json` was written at 16:05. No
  defensive field differs beyond 2e-15.
- **"Important to know" table.** The direction column uses a fourth value, "note", for rows that are conventions
  without a direction.
- **Not on the page.** Computed and in the JSON files, listed in the English record instead of printed: correlation
  matrices for blocks B and C, each half and the five crises; rolling-correlation P10 and P90; the ten worst
  21-session windows of GR2 and GR3; breach figures at -10, -15, -17, -22, -27 and -35% with worst-seed values; gap
  tables of the dial-map and margin books.
- **Provenance.** Only `build_sources.py` existed at the freeze. `g_lib.py`, `study.py`, `battery.py`, `exposure.py`,
  `capacity.py`, `defensive.py`, `pm_confirm.py`, `versus.py` and the report builders were written after the freeze
  and before or during the results; `battery.py`, `capacity.py`, `exposure.py`, `pm_confirm.py` and the report
  builders were edited after first results for the items above. The ledger's last line records the SHA-256 of every
  script as published.
- **O1 (owner decision, 2026-10-05, after results and after the first publication): GR1 = TAA 3x 40 / momentum 30 /
  MR 30.** Section 3 fixed GR1 at equal capital (one third each). The owner, having seen the results, chose the
  40 / 30 / 30 point of the pre-registered dial map as the flagship: a mild tilt to the TAA engine, which is the
  oldest and already runs live. This is a choice made after results, not a rule result, and the report says so.
  Equal capital stays in the study as challenger S0 and as a row in the menu. Numbers: equal capital 17.92% / 1.260 /
  -12.75% / P(DD < -20%) 2.8%; 40 / 30 / 30 18.28% / 1.269 / -12.96% / 2.4%. Paired test: 40 / 30 / 30 has the
  higher excess Sharpe on 70% of paths (below the 80% bar: a tie) and the higher CAGR on 93%. Cost of the tilt:
  excess Sharpe with the TAA engine dead 0.69 against 0.77, and a deeper historical drawdown on the planning column
  (-17.0% against -15.5%; floor -20.0% against -18.5%). The breach difference (2.4% against 2.8%) is inside
  simulation noise and is not evidence. At +5 bps 40 / 30 / 30 has the higher Sharpe on 84% of paths because it
  trades less MR. Against the monthly book the paired share is now 94% / 80% / 54% at 0 / +5 / +10 bps (equal
  capital: 90% / 71% / 42%). After this change the three products are no longer consecutive points of one dial:
  GR2 (equal capital with the 1N variant) has less TAA weight than GR1. Consequences in the code (`g_lib.G1`): the slot tests,
  the stand-ins, the plateau neighbours and S8 ("GR1 75 / DEF 25") are defined on the new GR1 weights; S0 is added
  to the challengers (thirteen); GR2 and GR3 are unchanged. The whole pipeline and the GR1 engine confirmation were
  re-run; the outputs of the equal-capital version are kept in `report_equal_capital_snapshot/`.
- **O2 (owner decision, 2026-10-05, after results): GR2 = TAA 3x 1N 40 / momentum 30 / MR 30.** Section 3 fixed GR2
  at one third each with the 1N variant. After O1 that left GR2 with less TAA weight than GR1, so the owner moved it
  to the same 40 / 30 / 30 weights: the three products are again one ladder (a pure 3x to 1N switch, then a higher
  TAA share). Numbers: one third each 20.3% / 1.21 / -16.3%; 40 / 30 / 30 21.1% / 1.21 / -16.3%, P(DD < -25%) 3.5%.
  Consequence by the frozen menu rule (a higher product needs at least 1.5 pp of CAGR over the one below it): GR3
  (22.3%) is 1.2 pp above the new GR2 and is therefore NOT offered. It stays in the study as a reference row and the
  open question of the new AGGRESSIVE rung is moot unless the owner wants a third product anyway.
- **O3 (owner decision, 2026-10-05, after results): two-pod monthly books.** The monthly growth book of 2026-10-01
  (TAA 3x 1N 38.4 / NDX-VXN 25.6 / CORE5 18 / BTAL_QQQ 18) is replaced by Monthly = TAA 3x 1N 50 / CORE5 50 and
  Monthly Plus = TAA 3x 1N 65 / CORE5 35; the old books stay as reference rows (`S13_OLD`, `OLD_PLUS_KEY`; the
  internal keys "S9 incumbent launch" and "old growth plus" now hold the new books). The owner's reasons: NDX-VXN is
  superseded by the momentum capsule, and BTAL_QQQ runs on the same engine as TAA 3x 1N (daily correlation 0.85).
  `monthly.py` (exploratory, written after results, not pre-registered) then asked whether the momentum capsule earns
  a weight in a monthly TAA + CORE5 book: over 2 TAA variants x 4 ratios x momentum 0 / 15 / 30% it never has the
  higher Sharpe on 80% of paired paths, its CAGR is equal or lower in 12 of 16 cells, and it makes 2022 worse in all
  16; without CORE5 the book fails the GROWTH rung. 50 / 50 was chosen for simplicity (17.7% / 1.17 / -13.9%,
  against 18.1% / 1.14 / -13.1% for the old four-pod book: a tie, 68% of paths) and 65 / 35 as the first ladder point
  above 20% CAGR (20.8% / 1.15 / -17.4%, GROWTH PLUS rung). Differences between neighbouring grid rows are inside
  noise. The honest price of two pods: with the TAA engine dead the Monthly book keeps an excess Sharpe of 0.20
  (Monthly Plus 0.11) against 0.69 for GR1, which is why GR1 stays the target. Against the new Monthly, GR1 has the
  higher excess Sharpe on 88% / 70% / 44% of paths at 0 / +5 / +10 bps. Engine confirmation of GR2 and the two
  monthly YAMLs: accepted (23.76% / 23.76%, 20.22% / 20.22%, 23.83% / 23.83%; correlation 1.0000).
- **O4 (owner decision, 2026-10-05): the backtest is the headline; 5.14(d) is narrowed.** The three-column rule
  (backtest / planning / floor beside every headline) is withdrawn from the cards and tables. The owner's argument,
  accepted in part: the engine already charges IBKR commissions and 2.5 bps of slippage per side, so the planning
  column's extra +5 bps double-counted costs for the TAA and momentum pods (it remains a real question for the daily
  MR stock orders, which the paper gate measures), and the headline frame already credits idle cash. What is kept,
  in the "how much to believe" section only: a conservative case (every engine keeps three quarters of its excess
  return, at model costs: GR1 13.5% / 0.95) and a stress case (half the excess return and +5 bps: GR1 8.3% / 0.59).
  Both are conventions, not forecasts. Unchanged facts: nothing is out of sample, and GR1's excess Sharpe was 1.56 in
  2012-2021 and 1.07 since 2022.
- **O5 (presentation, 2026-10-05): the blend dial.** The client blends of section 3 are reported at Growth shares of
  20 / 40 / 60 / 80% with the defensive launch (was 25 / 50 / 75%) as their own section. Against Growth diluted
  with BIL to about the same volatility (the diluted rows carry 1 to 4% more) the blend's historical max DD is
  shallower by about 0.5 to 1.4 pp on the one historical path and its CAGR is within 0.05 pp at 40 / 60 / 80% and
  0.5 pp lower at 20%: a small advantage, not a clear win.
- **O6 (owner decision, 2026-10-05, after results): GR3 = TAA 3x 1N 60 / momentum 20 / MR 20.** After O2 the old GR3
  (50 / 25 / 25) was 1.2 pp above GR2 and not offered by the menu rule. The owner moved it to the 60% point of the
  pre-registered dial: 23.4% / 1.19 / -18.0%, 2.3 pp above GR2, so it is offered again; the ladder is 40% TAA 3x,
  40% TAA 3x 1N, 60% TAA 3x 1N. It passes only the AGGRESSIVE rung of section 3 (max DD >= -27%, cap 15% at -30%:
  2.0% in the main frame, edge margin 0.70, conservative case 9.6%), which the plan marked as needing the owner's
  confirmation; choosing this book in effect uses that rung, and the confirmation is listed as an open decision.
  Price: TAA carries 77% of the risk, excess Sharpe with the TAA engine dead 0.34, Nasdaq look-through peak 2.0x.
- **O7 (owner request, 2026-10-05): fee income (`fees.py`, descriptive).** Income per $1M of AUM for five fee
  schedules (2/20, 1.5/20, 1/15, 1/10, 1/5; management fee accrued daily, performance fee above the high-water mark
  crystallised at calendar year end, no hurdle; no fund expenses), on the backtest and on the conservative case. The
  author's recommendation (1/15 for the growth products, 1/10 for the defensive product and blends) follows a stated
  convention: the fee takes about a quarter of the return above T-bills, about a third in the conservative case. The
  page's header line and the net-of-2/20 rows and columns were removed at the owner's request.
- **O8 (owner decisions, 2026-10-05, after results).** (a) The AGGRESSIVE rung is APPROVED by the owner. (b) The
  monthly menu is one book, Monthly = TAA 3x 1N 60 / CORE5 40 (19.8% / 1.15 / -16.2%); the 50 / 50 and 65 / 35 books
  of O3 are dropped. At 60 / 40 the book no longer passes GROWTH (29.6% of paths beyond -20%) and is read on
  GROWTH PLUS (5.6% beyond -25%, edge margin 0.80, conservative case 18.2% against the 15% cap); with the TAA engine
  dead its excess Sharpe is 0.14. So the book that can run first carries more risk than GR1, not less. (c) Growth
  at a fixed leverage of 1.40 is shown as an alternative route to GR3's level of return, not as a product: 24.1% /
  1.24 / -18.4% against 23.4% / 1.19 / -18.0% for GR3, TAA-dead Sharpe 0.66 against 0.34, slightly behind at +5 bps
  (22.37% against 22.48%); debt at DTB3 + 1.5%, no margin calls or gap days modelled; needs one cross-margined
  account. These later decisions (O6, O7, O8) were re-run, engine-confirmed (GR3 26.58% / 26.59%, Monthly 22.64% /
  22.64%) and checked by the author only; the independent reviews covered the versions before them.
- **Review of v5.2 (2026-10-05).** Two independent reviewers rebuilt Growth Plus, Monthly and Monthly Plus with
  separate code (all figures matched) and found one gap, now on the page: the monthly books hold their rung only
  near the full backtest edge (Monthly down to 0.95 of the edge, Monthly Plus 0.90, against 0.70 for GR1 and 0.75
  for GR2); in the conservative case their breach figures are 30.0% and 27.3% against the 15% cap. `battery.py`
  was extended to compute the edge margin and the after-window rows for the two monthly books.
