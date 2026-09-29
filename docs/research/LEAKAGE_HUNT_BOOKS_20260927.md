# Leakage hunt over the defensive and growth books — verdict (2026-09-27)

Audit only. No strategy, engine, data, test, portfolio, live or release file was changed; nothing was committed.
Engine flags (for example `historical_share_units_bool`) were set on strategy objects at runtime inside study code
only. Study code: [`scripts/research/leakage_hunt_20260927/`](../../scripts/research/leakage_hunt_20260927/).
Outputs and the four detailed findings files:
[`results/research/leakage_hunt_20260927/`](../../results/research/leakage_hunt_20260927/)
(`taa/TAA_FINDINGS.md`, `ndx/NDX_FINDINGS.md`, `mr/MR_FINDINGS.md`, `def/DEF_FINDINGS.md`).

## Verdict in one paragraph

After the fb81e86 fix there is **no remaining price-scale (future corporate-action) leak** in any strategy of the two
books: every one of them passed a real-data test in which a future split or reverse split is applied to one symbol's
history, and every one passed a truncation test. Three strategies still **need a fix**, and none is a
corporate-action problem:

- **HPI 2/3/5 vote (live)** is overstated by about **2.6 pp CAGR / 0.13 Sharpe** because the backtest refills an
  exited slot at the same open while the live host waits a day. This is a live/backtest divergence rather than
  classic look-ahead, and it is the only finding with real money at stake.
- **Inflation Compass** (fund menu only) uses a breakeven-inflation value one day before it is published, worth
  **0.46 pp CAGR**.
- **Tactical FI** uses Moody's yields that FRED published months late during a 2016–17 outage. That changes 2
  decisions but does **not** lower the returns (the real-time replay does slightly better). It needs a data fix
  before any PAPER/LIVE use, not a restatement.

Everything else is approved, most with caveats that are either conservative (the backtest looks worse than it
should) or about parameter/list selection on the same history rather than leakage. The membership-trim look-ahead
in the shared S&P 500 / Nasdaq-100 universe builder is real but immaterial for DV2 and NDX (≤ 0.12 pp CAGR, mixed
sign). **MOSAIC was not in this audit's scope but is still in several books** (see the last row of the table).

## Owner action list, ranked by money at risk

| # | What | Strategy / books | Money at risk | Decision needed |
|---|---|---|---|---|
| 1 | **HPI backtest refills a slot at the same open; live refills a day later.** | HPI vote — LIVE; growth books with MR (17–18%), DEF main line (9%), fund menu balanced/defensive/growth | Sleeve −2.6 pp CAGR, −0.13 Sharpe, MaxDD −17.7% → −20.7%. Books −0.26 to −0.52 pp CAGR | **Yours:** (a) change the live HPI host to submit replacement entries in the same open-auction batch as the exits (live DV2 already does this; Tier-3 change; needs margin to fund buys before sales settle — the backtest already runs cash to about −2.3% of NAV), or (b) keep live as is and restate the HPI sleeve and every book that holds it. Until you choose, quote the books with the restated HPI (table below). |
| 2 | **Inflation Compass reads T5YIE dated T at Close_T; it is published at T+1.** | Compass — fund menu aggressive 25%, growth 11%, balanced 8%, low-touch growth 18%, low-touch balanced 12% | −0.46 pp CAGR, −0.02 Sharpe (4 of 282 decisions flip). Menu books −0.08 to −0.17 pp | Approve the one-line fix (`allow_exact_matches=False` in the as-of join) and a re-run of its PM artifacts. |
| 3 | Turn on historical share units (raw whole shares and raw per-share fees) in research runs. | TAA 3x, 1/N, DV2, HPI, KIE/IHI, sector VOX/IYR, CORE5 | **Conservative today:** current numbers understate TAA 3x by 0.33 pp, 1/N 0.40 pp, DV2 0.87 pp, HPI 0.34 pp, KIE/IHI 0.24–0.36 pp CAGR. But a future split can move a past result (a hypothetical 40:1 TQQQ split would cut the TAA 3x backtest to 10% CAGR) | Approve making `historical_share_units_bool=True` the research default (engine/Tier-2 change, separate phase). |
| 4 | Remove the 5-day membership trim (`data/norgate_loader.py:107`, `scripts/export_norgate_snapshot.py:134`). | DV2, NDX, and MOSAIC and any other user of `build_index_constituent_matrix` | Immaterial: DV2 ≤ 0.05 pp CAGR, NDX ≤ 0.12 pp, mixed sign | Approve removal in a separate data change (it touches the VPS snapshot exporter). |
| 5 | Tactical FI: Moody's DAAA/DBAA were stale on FRED Oct 2016 – Mar 2017; the frozen contract used values published months later (2 of 254 replayable decisions flip). | Tactical FI — defensive books 17–27% | Not optimistic: the real-time replay does **better** (book CAGR 2.71% → 2.74–2.91%). Pre-2014 unverifiable (no ALFRED vintages) | Approve a stale-input fail-closed rule and an ALFRED-based snapshot for 2014+ before any PAPER/LIVE use. No restatement of book numbers needed. |
| 5b | **MOSAIC was not audited** but is still in G4, ladder_4, ladder_4_1n, LT_DEF (A1) and six fund menus (6–25%). It uses the same trimmed membership builder on the Russell 1000, which has far more removals than the Nasdaq-100, so the trim is unquantified there. | MOSAIC | Unknown; the refresh already showed the corrected sleeve at Sharpe 0.62 | Either drop MOSAIC from the book files and YAMLs (the refresh already recommends it) or commission a MOSAIC audit before quoting any book that holds it. |
| 6 | Small items: DTB3 hurdle dated T (0 flips; live already lags it), BTAL zero-volume padded month-ends (14 of 169 decisions, stale not future), EOM Sandy-2012 closure hindsight (+0.018 pp), VOX 2005 padded fill, float32 threshold ties in the sector pods, CORE5 qualification doc stale after fb81e86, NATR20 VXN has no live-host route | various | each < 0.05 pp CAGR | None needed now; fix opportunistically. |

## Summary table

"Invariance" = future split/reverse-split applied to one symbol's full loaded history (k = 40, 0.1, 1.5; several
symbols, several dates; real Norgate data). "Truncation" = decisions from data ending at T equal full-history
decisions at T. Counts are pass/total. Impact is on the book window 2012-10-02 .. 2026-08-19 unless stated.

| Strategy | Books | Verdict | Evidence (file:line, test) | Fix needed | Estimated impact of the issue |
|---|---|---|---|---|---|
| TAA 3x rank `strategy_taa_df_btal_fallback_tqqq_vix_cash` (LIVE) | growth core (both books), DEF main 8%, LT_DEF 6%, loren 60% | **APPROVED WITH CAVEAT** | Invariance 22/22, truncation 8/8 (`taa/taa_invariance_results.json`); TR adjustment verified multiplicative (`taa_tr_multiplicativity.csv`, max err 1.6e-5); DTB3 hurdle dated T at `strategy_taa_df.py:299` → 0/168 flips with a 1-day lag (re-verified independently) and 0 revisions in ALFRED; adjusted-unit fees/rounding `alpha/engine/strategy.py:726-730` (`taa01_accounting_rerun.csv`); BTAL padded month-ends (`taa03_staleness_scan.json`) | Raw share units (#3); optional DTB3 1-day lag for parity with live | Conservative: raw units give 24.20% → 24.53% CAGR, Sharpe 1.344 → 1.360. Hurdle timing 0. Staleness < 0.05 pp (bound, not measured). Selection: #3 of 48 sibling variants on Sharpe, ~0.1 Sharpe above the grid median |
| TAA 1/N `strategy_taa_df_btal_1n_fallback_tqqq_vix_cash` | growth G3-1N / ladder_4_1n | **APPROVED WITH CAVEAT** | Same tests 22/22, 8/8; same code path | Same as TAA 3x | Conservative: 30.94% → 31.34% CAGR, Sharpe 1.277 → 1.291 |
| BTAL_QQQ `strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash` | defensive (CORE5 + BTAL_QQQ 50%) | **APPROVED WITH CAVEAT** | 22/22, 8/8; log-price regression is scale-free (`strategy_taa_df_btal_linearity.py:102-170`); no FRED input | Raw share units (#3), like its siblings | Raw units +0.0015 pp; shares the BTAL padded month-end endpoints and one padded-bar fill (2015-04-01); #2 of 48 siblings on Sharpe |
| NDX live rule `strategy_mo_atr_normalized_ndx_vxn_scaled` (as fixed in fb81e86) | growth core (G3/G4/ladder), DEF main 8%, LT_DEF 6%, loren 40% | **APPROVED WITH CAVEAT** | Invariance 201/201 incl. AAPL/NVDA/TSLA/AMZN/GOOGL; positive control: the pre-fix formula fails on 7/7 gate-open dates; truncation 320/320 month-ends; live host `_build_atr_normalized_ndx_decision_plan` (`alpha/live/strategy_host.py:1094-1300`) reproduces backtest weights exactly on 7/7 dates; sleeve reproduces the refresh (Sharpe 0.9315); trim `data/norgate_loader.py:107` | Membership trim (#4); document parameter lineage | Trim: +0.00 pp / +0.003 Sharpe (exact window), 8/319 selections change, all acquisition targets. Selection: parameters picked on 2000–26, 151-config study Reality Check p = 0.61 |
| NDX NATR20 `strategy_mo_natr20_ndx_vxn_scaled` | growth core candidate (refresh default) | **APPROVED WITH CAVEAT** | 201/201, 320/320; NATR is scale-free by construction | Live-host route before any deployment (the host calls a function this module lacks); trim (#4) | Trim +0.12 pp / +0.006 Sharpe. Stronger hindsight: introduced after the corrected results were seen |
| DV2 `strategy_mr_dv2:DVO2Strategy` (LIVE) | growth MR pair/capsule 12–18%, ladder_4 16%, menu balanced/growth | **APPROVED WITH CAVEAT** | Invariance 30/30 (engine runs, 3 windows), truncation incl. PIT universe as of the cut; trim re-run (`mr/full_run_metrics.csv`); `close.unstack().dropna()` drops only warm-up rows (`dv2_nan_diag.json`); live builder `strategy_host.py:430-499` matches | Raw share units (#3); trim (#4) | Trim: −0.05 pp / −0.003 Sharpe (2012-10 on), mixed sign. Adjusted units understate DV2 by 0.87 pp / 0.035 Sharpe (conservative), but a later reverse split can silently cancel a past entry (17 vs 13 trades change, ≈ −1.5 bp/yr). One synthetic liquidation (AYE) |
| HPI 2/3/5 vote `strategy_mr_hpi_sp500_2_3_5_vote` (LIVE) | growth MR pair/capsule 12–18%, ladder_4 17%, DEF main 9%, menu balanced/defensive/growth | **NEEDS FIX** | Signal clean: invariance 24/24, truncation pass, own universe untrimmed. **Parity gap:** backtest frees the slot of an exit placed for Open_(T+1) and refills it at that same open (`strategies/hpi/stateful_long.py:564-579`, also reads whether Open_(T+1) exists); live host passes no open (`alpha/live/strategy_host.py:663-680`) so the slot is refilled a day later. 1,655 of 6,184 backtest entries (27%) used a same-open slot. Live-semantics replica `mr_common.py:HPILiveSlotStrategy` | Owner decision #1: align live to the backtest, or restate | Sleeve 16.42% → 13.80% CAGR, Sharpe 1.054 → 0.927, MaxDD −17.7% → −20.7% (2004–2026); book window −2.68 pp / −0.13 Sharpe |
| Industry-ETF DV2 `strategy_mr_dv2_industry_etf` | growth MR capsule 12% | **APPROVED WITH CAVEAT** | Native Turnover verified (24/24 invariance; tests pass); book's engine run equals a fresh run to 1e-11/day | Forward-test (list chosen in 2026) | Pre-2012 splice uses the old buggy ADV but is slightly conservative (0.32% vs 0.41% CAGR 2000–11). Hindsight: 19 ETFs picked in 2026 from a 60-ETF scan; 6 only became liquid 2014–2026 |
| CORE5 `strategy_taa_adaptive_macro_core5` | defensive anchor 33–55% | **APPROVED WITH CAVEAT** | Decision table 33/33, truncation 8/8 (incl. 4 cut-offs the pre-fb81e86 helper would have mis-flagged as month-end); live adapter `alpha/live/core5_adapter.py:84-227` same logic | Doc fix `docs/live/CORE5_ADAPTER_QUALIFICATION.md:87-88` | Accounting < 0.02 pp (raw-unit fees −0.011 pp; BIL adjustment ratio 0.5 pre-2017). Caveat is selection: ~441 trials, DBC-short overlay kept against its gates |
| Tactical FI `strategy_taa_tactical_fixed_income_ief_lqd` | defensive 17–27% | **NEEDS FIX** (data provenance; no return haircut) | ETF invariance 6/6; truncation 16/16; ALFRED replay (`def/tfi_fred_vintage_comparison.csv`, `tfi_vintage_metrics.csv`) | Stale-input rule + ALFRED snapshot (#5) | 2/254 flips from the 2016–17 Moody's outage; real-time replay is not worse (book 2.707% → 2.736–2.909%). Pre-2014 unverifiable. Selection: 38 variants, familywise p = 0.77 |
| EOM flow `strategy_taa_month_end_rebalancing_flow` | DEF main 17% | **APPROVED WITH CAVEAT** | Signal invariance 12/12, engine 3/3, truncation 9/9 | None | Sandy 2012 closure known only after the measure date: +0.43% once ≈ +0.018 pp CAGR (optimistic, immaterial) |
| Sector VOX/IYR `strategy_mr_us_sector_etf_ibs_downshock_vox_iyr` | DEF main 8%, menu | **APPROVED WITH CAVEAT** | Truncation 8/8; invariance differences are exact float ties (IBS exactly 0.90/0.05, re-verified in `def/sector_vox_iyr_invariance_cell_diffs_float64.csv`) | Optional tolerance on thresholds; padded-row mask | Fees conservative (+0.06–0.09 pp with raw units); one 2005 fill on a padded VOX bar, before the book window. Selection: 504-cell grid |
| Sector dispersion KIE/IHI SMA200 `strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200` | fund menu balanced/growth | **APPROVED WITH CAVEAT** | Truncation 8/8; float64 invariance 15/15 | Raw share units (#3) | Fees conservative: +0.24–0.36 pp CAGR, +0.03–0.05 Sharpe with raw units. Strong selection hindsight (149 combinations; SMA200 gate not pre-registered) |
| Inflation Compass `strategy_taa_inflation_compass` | fund menu (8–25%) | **NEEDS FIX** | Price-side invariance 31/31, truncation 8/8. **Macro timing:** `merge_asof(..., allow_exact_matches=True)` at `strategy_taa_inflation_compass.py:185-192` uses T5YIE_T at Close_T (282/283 decisions); published T+1 | `allow_exact_matches=False` (or shift availability one session) + re-run | 4/282 decisions flip; 21.53% → 21.07% CAGR, Sharpe 1.104 → 1.085 (2003–2026). T5YIE is never revised in ALFRED (0 of 660,697) |
| MOSAIC `strategy_mo_mosaic_russell1000` | G4, ladder_4/4_1n 8%, LT_DEF (A1) 6%, six fund menus 6–25% | **NOT AUDITED** (outside the requested scope) | Shares the trimmed membership builder (`data/norgate_loader.py:107`) on the Russell 1000; ADV gate uses native Turnover with an as-of lookup (reviewer code read) | Audit it, or drop it from the book files (refresh already shows corrected Sharpe 0.62) | Unknown |

## What changes in the book numbers

All four book sets (growth shelf v2, defensive books, fund menu, portfolio_refresh_20260927) read the same
2026-09-23 fund-menu sleeve runs to 2026-08-19; the refresh swapped in only the fixed NDX, MOSAIC and ETF-DV2
sleeves. So a sleeve fix moves every set that holds that sleeve.

Measured effect of the two NEEDS FIX items on the refresh's (fixed-momentum) books, same pod model (independent
pods, annual reset), book window 2012-10-02 .. 2026-08-19
(`results/research/leakage_hunt_20260927/book_impact.csv`, `scripts/research/leakage_hunt_20260927/book_impact.py`):

| Book | HPI weight | Compass weight | CAGR before → after | Sharpe before → after |
|---|---|---|---|---|
| G3 (TAA 3x + NDX) | — | — | unchanged 19.94% | unchanged 1.288 |
| G3 + stock pair | 18% | — | 20.10% → 19.58% | 1.400 → 1.373 |
| G3 + capsule | 12% | — | 18.66% → 18.30% | 1.413 → 1.392 |
| ladder_4 (annual) | 17% | — | 19.52% → 19.03% | 1.393 → 1.367 |
| DEF main line | 9% | — | 10.01% → 9.74% | 1.822 → 1.791 |
| DEF main line without EOM | 10.8% | — | 9.94% → 9.62% | 1.638 → 1.603 |
| CORE5 alone / CORE5 + BTAL_QQQ / LT_DEF (A1) / loren | — | — | unchanged | unchanged |
| menu growth | 12% | 11% | 15.17% → 14.75% | 1.391 → 1.363 |
| menu balanced | 8% | 8% | 12.71% → 12.43% | 1.540 → 1.515 |
| menu defensive | 9% | — | 10.01% → 9.74% | 1.822 → 1.791 |
| menu aggressive / low-touch growth / low-touch balanced | — | 25% / 18% / 12% | −0.17 / −0.12 / −0.08 pp | −0.01 |

Notes: menu books are shown at their YAML pod weights without the menu's volatility target, so read them as
direction and size, not as the published menu figures. The HPI correction assumes option (b), keeping live as is;
under option (a) the HPI numbers stand and the live host changes instead.

Other items and which numbers they would move:

- **Raw share units (#3)** would *raise* TAA 3x, 1/N, DV2, HPI and KIE/IHI results slightly. Growth shelf v2's
  "commission-fixed" tables already include a partial version of this (they recover about half of DV2's effect);
  portfolio_refresh_20260927 and the fund menu use the unconverted engine commissions.
- **Membership trim (#4)** moves DV2 and NDX by ≤ 0.12 pp in either direction; no book changes by more than a few
  basis points.
- **Tactical FI (#5)** real-time replay would *raise* the defensive books slightly (no restatement needed for
  leakage reasons).
- The growth shelf's 2008 synthetic TQQQ/BTAL proxy and the ETF-DV2 pre-2012 research splice fill only dates
  before 2012, so they do not touch the book-window numbers above. The proxy is calibrated on 2010/2011–2026 and
  backcast to 2008; the hindsight is in the choice of instruments (and current, not point-in-time, GICS sectors
  for synthetic BTAL), not in fitted parameters. The "incl. 2008" columns depend on them.

## How it was tested

- **Code read** of every feature of every strategy, with formula, file:line, scale-freeness and lag direction
  (feature tables in the four findings files).
- **Future-action invariance on real Norgate data** (shared harness `harness.py`): one symbol's entire loaded history
  is re-based as if a k:1 split happened after the last date. Open/High/Low/Close and Dividend are divided by k,
  Volume is multiplied by k, and `Unadjusted Close` and `Turnover` stay nominal. Decisions compared are selected
  sets, rank order and target weights, or engine trade decisions (date, asset, side) with notionals. Totals: TAA
  family 97 (+32 truncation), NDX 402 (+640 truncation), MR 78, defensive 49 cases, plus positive controls (the pre-fb81e86 NDX formula fails, as it
  must).
- **Truncation tests** at mid-month cut-offs, weekend and Good-Friday month-ends, the book end and the current
  partial month.
- **Macro vintages:** ALFRED real-time vintages for DTB3, T5YIE, DGS10, DGS3MO, DAAA and DBAA, plus causal
  one-day-lag replays for publication timing.
- **Corrected re-runs** through the real engine for every quantified item (DV2/HPI/ETF-DV2 full runs, TAA and
  Compass re-runs, NDX untrimmed runs, Tactical FI real-time replays).
- **Reproduction:** the DV2 $1M run reproduces the book sleeve exactly (25,086 fills), the NDX sleeves match the
  refresh to 1e-9/day, and the ETF-DV2 engine run matches a fresh run to 1e-11/day.
- **Independent checks by the primary auditor:** the DTB3 lag re-derived from scratch (0/168 flips for both TAA
  variants), the HPI live-host behaviour read in both code paths, and the float-tie claim read in the float64 cell
  diffs.
- **Independent quant-pitfalls review** (read-only agent): see "Independent quant-pitfalls review" below.

## Selection and hindsight (not leakage, but it matters for how much to trust the numbers)

Every strategy in both books was designed and tuned on the same 2000–2026 history, with no untouched holdout on
record. Examples:

- **TAA:** 98 modules; BTAL, the TQQQ fallback and the VIX gate were added March–April 2026. TAA 3x ranks #3 of 48
  siblings.
- **NDX:** 151 configurations; Reality Check p = 0.61. NATR20 was added after the corrected results were seen.
- **DV2:** about 148 trials, deflated Sharpe about 1.0.
- **HPI vote:** from a 20-variant sweep.
- **CORE5:** about 441 trials; PBO 49%.
- **Tactical FI:** 38 variants; familywise p = 0.77.
- **KIE/IHI:** 149 combinations.
- **Sector VOX/IYR:** 504 cells.
- **ETF-DV2:** list of 19 surviving funds picked in 2026 (survivorship, bounded by the causal liquidity and
  history gates). The KIE/IHI and VOX/IYR baskets are the same kind of survivorship choice.

A crude estimate of the in-sample selection premium for the chosen TAA variants is 0.05–0.12 Sharpe. None of this
is a code leak, but together it is a larger uncertainty than any single finding above.

## Independent quant-pitfalls review

A read-only reviewer re-checked the method and the numbers:

- **Perturbation fields and recomputation:** correct. Derived features are recomputed from the rescaled raw frames,
  never copied.
- **Methodology:**
  - Truncation plus constant-factor rescaling is logically sufficient to catch decision leaks.
  - NDX coverage is strong (640 cut-offs). MR coverage is thinner (6 cut-offs per strategy).
  - Only NDX has an injected-leak positive control. Elsewhere the wiring is shown indirectly (for example, the
    TQQQ ×40 case moves TAA 3x CAGR from 24.2% to 10.1%).
  - Suggested follow-up: add one standard injected price-threshold control to the harness.
- **Numbers re-verified:** HPI base vs live-slot, Compass base vs causal, Tactical FI vintage metrics, DV2 trim and
  historical-units runs, and the ETF splice all match the findings files.
- **Missed-leak search:**
  - A grep for `resample`, `ffill`, `bfill`, `shift(-`, `center=`, `rank`, `quantile` and `iloc[-1]` across all
    audited modules plus MOSAIC found nothing non-causal.
  - VIX/VXN gates feed next-open trades.
  - EOM's close-auction path never uses the close it trades at.
  - Dividend entitlement is credited before the T+1 fills, so an ex-date-open buyer gets nothing and shorts pay.
    Verified on a real CORE5 DBC short.
  - Short sizing and borrow are correct in dollar terms.
- **Verdict changes adopted here:** BTAL_QQQ to APPROVED WITH CAVEAT (consistency), Tactical FI to NEEDS FIX (the
  literal rule: decision-changing on real data, although not optimistic), MOSAIC flagged NOT AUDITED.
- **Also noted:** if Norgate ever updates index membership a day after prices, the same trim line would drop
  *current* members from the live universe (the condition compares against the $SPX last date). Not verified; one
  more reason to remove the trim.

## Residual risk

- Pre-2014 macro vintages (Moody's, T5YIE) cannot be verified; the causal replays rely on zero observed revisions
  after 2014.
- Tests use one Norgate vintage. The invariance test simulates future corporate actions; it cannot detect a vendor
  restatement of an old unadjusted price.
- Synthetic liquidations at the last close for delisted names (G-014) are bounded, not replayed with deal terms.
- MOSAIC (in G4, ladder_4, LT_DEF and several fund menus) was outside this audit's scope. It shares the membership
  trim; its fb81e86 fix is covered by the existing tests only.
- The audit read the working tree as found. The only pre-existing uncommitted change in an audited module is a
  refreshed `$SPXTR` benchmark hash in the Tactical FI module (reporting only; no signal logic).

## Verification fields

- Tier: 1 (research scripts and docs only; no strategy, engine, data, live or release code changed).
- Agents used: four read-and-test auditors (TAA family, NDX momentum, equity MR, defensive book) and one
  independent read-only quant-pitfalls reviewer.
- Findings fixed: none in code (audit only). Reviewer findings adopted in this report: BTAL_QQQ caveat, Tactical FI
  NEEDS FIX, MOSAIC flagged NOT AUDITED, 2008-proxy wording.
- Tests run: about 1,300 real-data invariance and truncation cases (TAA family 129, NDX 1,042, equity MR 78,
  defensive 49), all passing apart from the explained float ties and the expected positive-control failures; the
  existing `test_dv2_liquidity_corporate_actions` and `test_historical_share_units` suites (77 passed, run by the
  MR group); corrected engine re-runs listed per strategy.
- Residual risk: as listed above.
