# Shelf rebuild after the 22-29 Sep 2026 fixes: defensive core, growth core, portfolio map (frozen plan)

Written 2026-09-29, before any sleeve of this study was re-run and before any book of this study was built.
Later changes go to the amendment log at the end, dated, with the reason and whether they came before or after
a result. Nothing above the log is edited after the first sleeve run starts. SHA-256 of this file is recorded in
`results/research/portfolio/shelf_rebuild_20260929/experiment_ledger.jsonl` at freeze time.

## 0. What is already known (this is not a blind test)

The 2000-2026 history of every sleeve has been studied many times. Known before this freeze, from earlier studies
on pre-fix or partly fixed sleeves: G3 (TAA 3x rank + NDX-VXN) exact-window Sharpe 1.29 after the split-leak fix;
CORE5 + BTAL_QQQ 50/50 Sharpe ~1.44, max drawdown ~-7%; LT_DEF ~1.55; DEF main line ~1.82; MOSAIC dead (demoted);
Tactical FI with BIL cash 1.92% / 0.75 (2012-10..2026-08); EOM flow weak in the last three years (owner remark);
Inflation Compass parameters are a lucky peak. The bootstrap and PBO below measure selection luck inside this
study's own search only; they cannot remove what earlier looks at the same data already did.

## 1. Owner request (Hebrew, 2026-09-29, paraphrased)

Rebuild the portfolios after the week's fixes, rigorously, with the same constraints and ease-of-operation lens as
before, and update the portfolio map. Defensive first (the owner expects it was not really hurt), then growth
(Sharpe is not a strict concern; the drawdown is). Universe: PM_READY + WIRED plus shadow candidates. The 2008
proxy is strongly preferred. Also weigh ease of operation, AUM capacity and similar practical limits. Report in
Hebrew, deep research. Owner answers: pre-register (yes); run the sleeves here, not from another session (yes);
defensive drawdown budget -7% to -10% and growth -20%, both including the 2008 proxy (yes).

## 2. Inputs

### 2.1 Common settings

- Code: git HEAD at run start (recorded; expected f9ad358 or a docs-only successor). One Norgate database vintage
  (recorded at start and end; a change aborts the run).
- End date: 2026-08-19 for every sleeve (Tactical FI's PM_READY contract cannot run past it; one common end keeps
  every book on the same window). Reference capital $1,000,000 per sleeve. Requested start 2000-01-03; a module
  that refuses it falls back to its own default start (logged).
- Engine costs as each module defines them (house: 2.5 bps per side slippage, $0.005/share, $1 minimum, idle cash
  0%, negative cash unfinanced, 25% dividend withholding where the module applies it).

### 2.2 Sleeves

Every registry strategy at PM_READY or WIRED on 2026-09-29 (22) plus these shadow candidates, each run fresh
through its own `run_variant`:

| alias | import | tier | role in this study |
|---|---|---|---|
| core5 | strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5 | PM_READY | defensive anchor (always in Part D) |
| btal_qqq | strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash | WIRED | defensive pool |
| tactical_fi | strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd | PM_READY | defensive pool |
| trinity | strategies.taa_beyond_6040.strategy_taa_trinity_vol_control_8_bil | PM_READY | defensive pool |
| eom_flow | strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow | PM_READY | defensive pool |
| downshock | strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr | PM_READY | defensive pool |
| disp | strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200 | PM_READY | defensive pool (the only dispersion variant with 2008 history) |
| disp_xlc | ...strategy_mr_sector_dispersion_ibs_kie_ihi_xlc | PM_READY | inventory only (starts 2018) |
| disp_xlc_sma | ...strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200 | PM_READY | inventory only (starts 2019) |
| taa_lin_qqq | strategies.taa_df.strategy_taa_df_linearity_1n_fallback_qqq_vix_cash | PM_READY | inventory only (BTAL_QQQ's no-BTAL twin) |
| taa3x | strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash | WIRED (live) | growth TAA leg |
| taa3x_1n | strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash | WIRED | growth TAA leg |
| taa2x_1n | strategies.taa_df.strategy_taa_df_btal_1n_fallback_qld_vix_cash | PM_READY | growth TAA leg |
| taa_1n_qld | strategies.taa_df.strategy_taa_df_1n_fallback_qld_vix_cash | PM_READY | inventory only |
| taa_1n_sso | strategies.taa_df.strategy_taa_df_1n_fallback_sso_vix_cash | PM_READY | inventory only |
| ndx_vxn | strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy | WIRED (live) | growth NDX leg |
| ndx_atr | strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy | WIRED | growth NDX leg |
| ndx_natr20 | strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled | shadow (RESEARCH) | growth NDX leg |
| compass_qqq | strategies.taa_df.strategy_taa_inflation_compass_qqq | PM_READY (shadow per research) | growth third leg |
| compass | strategies.taa_df.strategy_taa_inflation_compass | PM_READY | inventory only |
| dv2 | strategies.dv2.strategy_mr_dv2:DVO2Strategy | WIRED | growth MR option |
| dv2_adv | strategies.dv2.strategy_mr_dv2_liquidity_floor_adv_rank | shadow (RESEARCH) | growth MR option |
| dv2_floor | strategies.dv2.strategy_mr_dv2_liquidity_floor | shadow (RESEARCH) | inventory only |
| hpi_vote | strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote | WIRED | growth MR option |
| hpi_ibs_rsi | strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit | WIRED | inventory only |
| etf_dv2 | strategies.dv2.strategy_mr_dv2_industry_etf | shadow (RESEARCH) | growth MR option |

Excluded by rule: RESEARCH-tier strategies demoted this month (MOSAIC, Crisis Trend Core, VIXM, QPI) and research
ideas without a strategy module (the NDX NATR20+SMA200 shadow line has no module; `ndx_natr20` keeps SMA100 and is
the closest module; the R1000 liquid-momentum leg; merger arbitrage). The run script checks that its alias table
covers the registry's PM_READY+WIRED set exactly, minus nothing, and fails otherwise.

Tactical FI runs in `fred_data_mode_str="alfred_point_in_time"` (vintage = decision date, stale input =
block_and_hold), the point-in-time mode a live run would face; the governed frozen mode is run as a sensitivity.
Its cash sleeve is the default real BIL position.

### 2.3 T-bills

`tbill` = BIL (SPDR 1-3 Month T-Bill ETF) daily TOTALRETURN from Norgate (first bar 2007-05-30), expense ratio
included. It is the T-bill pod, the dilution asset, the slot-test replacement and the hurdle for every "excess"
metric. The lagged DTB3 accrual of the menu study is reported beside it as context.

### 2.4 The 2008 proxy

Only the BTAL TAA sleeves lack pre-2012 history (btal_qqq, taa3x, taa3x_1n, taa2x_1n). Before 2012-10-02 their
returns come from engine runs of their own unchanged code at HEAD with synthetic TQQQ and synthetic BTAL spliced
before each fund's first real bar (the validated instruments of growth_shelf_v2_20260926, reused unchanged, file
hashes recorded): `splice_scaled` is the main series, `splice_unscaled` the sensitivity. From 2012-10-02 on, the
real sleeve runs are used unchanged. The strategy-level check of growth_shelf_v2 (A3) is repeated at HEAD for the
four sleeves (whole-history synthetic run vs real run, 2012-10-02 -> end; monthly corr >= 0.90, |CAGR diff| <= 2 pp,
|max DD diff| <= 5 pp); a failing sleeve keeps its proxy with an "approximate" flag. Every other sleeve uses its
own real history.

Known proxy limits, disclosed not fixed: the synthetic QLD was 0.2-1.8 pp/yr optimistic out of sample; the BTAL
replica's scale k (0.734) is the conservative end of a bracket (unscaled = k 1); the long window starts
2008-03-04, after the S&P 500 had already fallen ~15% from its 2007-10 peak.

### 2.5 Sleeve returns

Daily simple returns from the NAV of each run, starting the day before its first invested day (NaN before, never
filled), as `fund_menu_20260923/common.sleeve_nav_df`.

## 3. Windows

- LONG: 2008-03-04 -> 2026-08-19 (proxy before 2012-10-02 for the four BTAL TAA sleeves). Primary for selection.
- EXACT: 2012-10-02 -> 2026-08-19 (every sleeve real). Reported beside every LONG figure.
- Blocks: A 2008-03-04 -> 2012-10-01 (proxy era), B 2012-10-02 -> 2021-12-31, C 2022-01-03 -> 2026-08-19.
- RECENT: 2023-08-21 -> 2026-08-19 (the last three years).
- Crises: every S&P 500 TR decline of at least 10% inside LONG (objective list from the benchmark,
  `common.equity_drawdown_episode_list`), plus the named windows GFC 2008-05-19 -> 2009-03-09,
  2018 Q4 2018-09-20 -> 2018-12-24, COVID 2020-02-19 -> 2020-03-23, 2022 bear 2022-01-03 -> 2022-10-12,
  2025 tariffs 2025-02-19 -> 2025-04-08.
- Stock-bond co-falls: 21-session windows in LONG where both $SPXTR and AGG (TOTALRETURN) lost money; the six worst
  non-overlapping windows ranked by the 60/40 benchmark's return.

## 4. Book model

Pods compound independently; at the last session of each calendar year the pod values are reset to the target
weights (`common.book_return_ser`, annual reset, the menu product policy). Reallocation is cost-free (disclosed).
Two weighting rules exist in this study:

- EQ: fixed target weights (equal capital unless a structure below says otherwise).
- IV (Part D only): at each reset, target weight_i proportional to 1 / sigma_i, sigma_i = standard deviation of the
  sleeve's daily returns over the last 252 sessions up to and including the reset close (at least 60 valid
  returns). *** CRITICAL*** only returns up to the reset close are used; the new weights apply from the next
  session. The first period of a window uses the 252 sessions before the window's first session; if any pod has
  fewer than 60, the first period uses equal weights.

## 5. Metrics (per book, LONG and EXACT unless noted)

CAGR; volatility; Sharpe with a zero risk-free rate (house rule); excess Sharpe over T-bills; max drawdown with
peak/trough/recovery dates; Calmar = CAGR / |max DD|; excess Calmar = (CAGR - T-bill CAGR) / |max DD|; CVaR 5%
of daily and of 21-session returns; worst month, worst 12 months, worst calendar year; share of positive months;
beta and correlation to $SPXTR; correlation to $SPXTR on the S&P 500's worst 5% of days; returns in every crisis
and co-fall window; excess CAGR over T-bills in blocks A, B, C and RECENT; ease and capacity fields (section 10).

## 6. Part D: the defensive core

Family (83 books): CORE5 alone, and CORE5 plus every subset S of the pool P = {btal_qqq, tactical_fi, trinity,
eom_flow, downshock, disp} with 1 <= |S| <= 3, each under EQ (equal capital over all pods) and IV.

Gates (all must hold):
- D1: LONG max drawdown >= -10%.
- D2: excess CAGR over T-bills > 0 in blocks A, B, C and RECENT.
- D3: T-bill slot test, LONG: for every pod, replacing that pod's capital with T-bills (same weight rule, T-bills
  take the pod's target weight; under IV the T-bill weight is the replaced pod's IV weight) must lower the
  objective. A pod that does not earn its slot fails the book.

Objective: LONG excess Calmar.

Tie band: paired stationary bootstrap of the family's LONG daily returns together with T-bills (2,000 paths,
mean block 63 sessions, seed 20260929). On each path the objective is recomputed. A gate-passing book is tied with
the top gate-passing book when the top book's objective beats it on fewer than 90% of the paths.

Tie-break inside the band, in this order (ease of operation decides among books the data cannot separate):
(1) every pod also passes the slot test on RECENT; (2) fewer pods; (3) lower share in shadow sleeves, then lower
share in PM_READY sleeves (WIRED first); (4) fewer book trading days per year (EXACT); (5) higher objective.
The first book is D*.

If no book passes all gates, D2's RECENT part becomes a reported flag and the selection is repeated; nothing else
is relaxed. Sensitivity family, not selectable: the same pool without CORE5 (every S with 1 <= |S| <= 3, EQ).

## 7. Part G: the growth core

Family (72 books) = TAA leg {taa3x, taa3x_1n, taa2x_1n} x NDX leg {ndx_vxn, ndx_atr, ndx_natr20} x third leg
{none, compass_qqq} x MR option {none; pair = dv2 + hpi_vote; capsule = dv2 + hpi_vote + etf_dv2;
capsule_adv = dv2_adv + hpi_vote + etf_dv2}. Core legs have equal weight (1/2 each, or 1/3 each with the third
leg). With an MR option the MR sleeves take 36% of the book (pair 18/18, capsule 12/12/12; the owner-approved
capsule size of 2026-09-26) and the core legs share 64%. EQ weights, annual reset.

Objective: CAGR at the -20% budget, LONG. For each book, the largest T-bill-free share s in [0, 1] (step 0.01)
such that the mix s * book + (1 - s) * T-bills (pods reset annually) has LONG max drawdown >= -20%; the objective is
that mix's LONG CAGR (s = 1 when the book itself is within budget; no leverage). The same objective at a -16%
design point (20% safety margin) is reported as a robustness check of the ordering.

Gates: G1 excess CAGR over T-bills > 0 in blocks B, C and RECENT (A is reported); G2 T-bill slot test on LONG with
this objective for every pod; G3 a book with compass_qqq must beat its no-Compass twin on at least 90% of the
bootstrap paths (the Compass parameters are a known lucky peak).

Tie band as in Part D, with the objective computed on each path by linear dilution
(s_path = min(1, 0.20 / |maxDD_path|), objective = s_path * CAGR_path + (1 - s_path) * T-bill CAGR_path).
Tie-break: (1) RECENT slot tests pass; (2) fewer pods; (3) lower shadow share, then lower PM_READY share;
(4) no daily mean-reversion pod; (5) fewer trading days per year; (6) higher objective. The first book is G*.

References, evaluated but not selectable: G3 (taa3x 50 / ndx_vxn 50) and every current `portfolios/*.yaml` whose
pods are all in this inventory.

## 8. Part M: the portfolio map

Components: D*, G*, T-bills; and, as the simple alternative, G3 in place of G*. Rungs by LONG drawdown budget:
DEF-7 (-7%), DEF-10 (-10%), BAL (-15%), GRO (-20%). For each rung and each growth component, the weights
(d, g, t) on a 0.05 grid with d + g + t = 1 that maximise LONG CAGR subject to LONG max drawdown >= budget
(ties: more d, then more t). Pod weights = d * D* + g * growth + t * T-bills, annual reset at the pod level.
The neighbouring grid points are shown so a plateau is visible. Choosing (d, g, t) is an in-sample fit of two free
numbers per rung and is labelled so.

For every rung product: all section 5 metrics; bootstrap distribution of LONG max drawdown and the share of paths
worse than the rung's budget; section 9 sensitivities; section 10 ease and capacity; small-account friction.

## 9. Statistics and sensitivities

- PBO (combinatorially symmetric cross-validation) for Parts D and G separately: LONG daily returns cut into 16
  contiguous blocks; for each of the 12,870 half/half splits the in-sample best book by the part's objective is
  taken and its out-of-sample rank recorded; PBO = share of splits where that book ranks at or below the
  out-of-sample median. Reported with the median out-of-sample rank.
- Sensitivities (reported, never used to select): unscaled BTAL proxy; +5 bps per side on every traded dollar;
  Tactical FI in the governed frozen FRED mode; cash realism (positive idle cash earns max(DTB3 - 0.5%, 0), negative
  cash pays DTB3 + 1.5%, from each sleeve's prior-day cash / NAV); EXACT-window ordering; drift instead of annual
  reset for the rung products; commission on real share counts (the engine charges on split-adjusted counts,
  conservative) for the rung products.

## 10. Ease of operation, capacity, small accounts

Per sleeve (documented table): execution route and frequency (monthly MOO, daily MOO, MOC), instruments (ETF or
single stocks, leveraged ETFs, shorts), outside data (FRED/ALFRED, VIX/VXN), live status (WIRED with a live route,
PM_READY without one, shadow), open items from the 2026-09-28 readiness audit. Per book: pods, tier shares, book
trading days per year (EXACT), whether any pod trades daily, whether any needs MOC or a short.

Capacity: the growth-shelf route model (`growth_shelf_v2_20260926/shelf_books.capacity_rows`, house auction limits
for stocks, worked ETF orders, urgent ETF orders worked within one day) on fills of the last three years
(2023-08-21 -> end); recommended AUM for MOO (today), MOC and worked+blocks, the first failing gate, and cost per
year at $25M. Descriptive, not a gate.

Small accounts (rung products only, descriptive): each pod re-run at pod capital = weight x account for accounts of
$30,000 and $100,000 on EXACT; the CAGR loss versus the $1M reference run measures whole-share and minimum-fee
friction.

## 11. What is a result and what is judgement

Gates, objectives, tie bands and tie-breaks above produce D*, G* and the rung weights mechanically. Anything the
report adds beyond that (for example preferring a simpler rung, or a buffer below the budget) is labelled as
judgement. Every book tested is reported (83 + 72 + the map grids + references).

## Amendment log

- A1 (2026-09-29, while the sleeve runs were in progress, before any sleeve or book result of this study was read):
  small-account friction (section 10) is measured on two windows, each against a $1M run with the SAME start, so the
  start state cancels: EXACT (start 2012-10-01; a small account opened in 2012 compounds, so this understates today's
  friction) and RECENT (start 2023-08-21, at today's share prices, the case of a client opening now). Capacity uses
  each product's average pod weights (IV weights averaged over LONG) held fixed with an annual reset over the capacity
  window. Reason: the section-10 wording left the reference run and today's price level undefined.

- A2 (2026-09-29, after the sleeve runs finished, before any book was built and before any sleeve metric was read):
  `etf_dv2` loads prices only from 2009-01-01 (module constant) and needs 252 sessions per ETF, so its engine run
  has no bars before 2009-01-02 and stays idle until its first trade on 2010-01-13. That gap is an artifact of the
  module's history start, not of its rules, and without a fill no capsule book could be measured on LONG. In the LONG
  frames its returns before its first invested day come from the DV2 deep study's research run of the same rules
  (`results/research/dv2_deep_20260925/sources/etf_ind_adv50`, which screened the $50M liquidity on raw Close x
  adjusted Volume, a unit the 2026-09-26 fix later corrected; ETF splits are rare), as growth_shelf_v2 did before
  2012. Its fills set the +5 bps drag on those dates; it gets no cash-realism add there (no cash column). Sensitivity:
  the same dates at 0% (idle cash). Other fallbacks, logged by the run and harmless: `hpi_ibs_rsi` and `eom_flow`
  refused 2000-01-03 and started from their module defaults (2004-01-01 and 2003-01-02); neither is short of LONG.

- A3 (2026-09-29, AFTER the Part D and Part G results; post-result, labelled as such): Part D's mechanical D* is
  CORE5 + EOM [EQ]. Every book in its tie band except one holds `eom_flow`, and every tie-band book fails the RECENT
  slot test, `eom_flow` being the failing pod wherever it is held (the owner's remark that EOM flow is weak in the
  last three years, now confirmed by the study's own test). `eom_flow` also cannot trade live today (month-end MOC and
  a TLT short, gap G-032). Added, without changing D*: D' = the same Part D rules (gates, objective, bootstrap tie
  band, tie-break) applied to the subfamily without `eom_flow`, using the saved bootstrap paths. Part M builds the map
  for D* and D' side by side. Any recommendation that prefers D' over D* is judgement and is labelled so. Part G is
  unchanged (G* stays the mechanical pick; the live G3 is already the declared simple alternative).

- A4 (2026-09-29, AFTER the Part G results; post-result, labelled): G* (TAA3x-1N + NDX-ATR) won the last tie-break
  (objective) over its NDX-VXN twin by 0.30 pp of CAGR, a difference the bootstrap cannot separate (the top book beats
  each on 49% / 47% of paths), while G*'s LONG drawdown (-19.6%) sits at the budget and its rank at the section-7
  design point (-16%) is 22. Added, without changing G*: G' = the gate-passing book that ranks first at the -16%
  design point (TAA3x-1N + NDX-VXN, both WIRED; it keeps the NDX-VXN pod that is live today). Part M builds the map
  for {D*, D'} x {G*, G', G3}. Preferring G' is judgement and is labelled so.

- A4 timing correction (2026-09-29, after the independent review): A4 was written at 11:03 UTC, 12 seconds after a
  full Part M run with A3 had finished, i.e. after the Part M results as well as after Part G.
- A5 (2026-09-29, AFTER all Part D/G/M results; post-result, descriptive + judgement): `breach_frontier.py` shows, for
  every family book, LONG CAGR next to the bootstrap probability of breaking the owner's hard limit (-10% defensive,
  -20% growth). This criterion was not pre-registered. The recommended map in the report (products built one step
  below their hard limit; defensive core D', growth core G3, T-bills) is judgement built on it and on ease, and is
  labelled so next to the mechanical products (D* + G*). Product names differ from design rungs: "Balanced (hard
  -15%)" is the DEF-10 design rung of D' + G3; "Growth (hard -20%)" is the GRO rung of D' + G3, which equals G3.
- A2 correction (2026-09-29, after the review): A2 said "ETF splits are rare"; that is false for this list. 12 of the
  17 industry ETFs trading in the fill window split later (IHI 6x, IGV 5x, IYT 4x; XOP a 1:4 reverse split), so the
  research run's $50M screen on raw Close x adjusted Volume used future split factors in 2008-03 -> 2010-01, and that
  run is about +0.35 pp a year optimistic against the engine after 2010. Only capsule books are affected; none was
  selected. Their LONG figures are flagged in the report.
- A6 (2026-09-29, after the review; post-result checks, nothing re-selected): `selection_checks.py` (1) re-runs the
  complete Part D, D' and Part G rule (gates, tie band on each frame's own bootstrap, tie-break) under every
  section-9 sensitivity, because the parts reported only the top book and D*'s rank; (2) defines G' with the Part G
  tie-break applied at the -16% design point instead of A4's raw argmax, which the review showed is decided by
  0.07 pp against a five-pod shadow book; (3) reports each fixed book's out-of-sample rank in the CSCV splits,
  because the PBO in parts D and G describes an argmax-over-all-books rule; (4) records EOM's RECENT excess under
  cash realism. The review's main finding: sleeves' idle cash earns 0% while every hurdle and slot replacement earns
  the full BIL rate (4.5% a year in RECENT), which biases the slot tests and RECENT gates against mostly-cash pods;
  eom_flow is 89% idle cash in RECENT. A3's premise that EOM is weak in RECENT therefore rests mostly on that
  convention; A3 stands only on EOM's live-tradability gap (month-end MOC and a TLT short, G-032).
