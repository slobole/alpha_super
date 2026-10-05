---
title: Book Strategy Caveats
description: Known caveats of PM_READY and research strategies used in portfolio books.
document_type: reference
authority: guide
risk_scope: research
source_paths:
  - alpha/strategy_registry.py
  - docs/research/STRATEGY_READINESS_AUDIT_20260928.md
  - docs/research/MOMENTUM_DECISION_20261004.md
  - docs/research/FUND_PRODUCTS_20261005.md
---

# Book Strategy Caveats

!!! abstract "What this page is"
    Known caveats of the strategies that may join a portfolio book but are not connected to a live account. The
    [wired strategies](index.md) carry their caveats on their own pages. These entries come from the
    [readiness audit of 2026-09-28](../research/STRATEGY_READINESS_AUDIT_20260928.md). Update an entry whenever a
    caveat is measured again or fixed.

*Direction* says whether the published backtest is **conservative** (reality should be better), **optimistic**
(reality should be worse), or neutral. "Friction" means the USD 1 minimum commission plus whole-share rounding at a
small pod size. Capacity is at today's volume with opening-auction orders.

## Adaptive Macro CORE5 — `strategy_taa_adaptive_macro_core5`

| Caveat | Direction | Size | Evidence |
|---|---|---|---|
| Fees on split-adjusted units (BIL was 1:2 before 2017) | Optimistic, negligible | -0.007 pp/yr | `core5/CORE5_FINDINGS.md` A-CORE5-01 |
| 25% withholding on BIL/IEF distributions and 0% on cash | Conservative | Up to +0.16 and +0.04 pp/yr | A-CORE5-02 |
| DBC short needs a margin account and a borrow; borrow is modelled at a fixed 1%/yr | Account requirement | The whole borrow layer costs 0.03% of NAV/yr | A-CORE5-05, A-CORE5-12 |
| Capacity | Limit | DBC/UUP reach 5% of ADV at about USD 3.3-3.6M | A-CORE5-20 |
| One SPY share is 2.6% of NAV at USD 30K | Size-dependent | About -0.18 pp/yr at USD 30K | A-CORE5-21 |
| Selection | Selection | About 441 trials; PBO 49%; the DBC-short overlay was kept against its gates | Leakage hunt 2026-09-27 |
| Live adapter: a missed evening halts the pod permanently; the short has no pre-trade borrow check | Live risk | LIVE mode is refused in code today | Fix #11 |

## Inflation Compass — `strategy_taa_inflation_compass` and `strategy_taa_inflation_compass_qqq`

| Caveat | Direction | Size | Evidence |
|---|---|---|---|
| Parameters sit at a lucky peak | Selection | 826 trials; failed pre-2003 holdout; honest Sharpe about 0.85 | Inflation Compass deep research |
| T5YIE timing: the strict rule uses the observation before T | Neutral | 0 decision flips across 152 ALFRED vintages | `tierb_macro/TIERB_MACRO_FINDINGS.md` B-CMP-01 |
| Unfinanced negative cash | Optimistic, negligible | Min -2.9% of NAV | B-CMP-04 |
| 25% withholding | Conservative | About +0.56 pp/yr | B-CMP-03 |
| Capacity | Limit | Extra auction cost reaches 0.25 pp/yr at about USD 0.6-0.7M (IEF, XLU) | `review_bc_trade/rbt02_*.json` |
| QQQ variant: one share is about 6% of NAV at USD 12K | Size-dependent | — | `review_bc_trade/REVIEW_BC_TRADE.md` |
| A live route must exclude the partial-month decision row | Live risk | Not wired | Fix #26 |

## Tactical Fixed Income — `strategy_taa_tactical_fixed_income_ief_lqd`

| Caveat | Direction | Size | Evidence |
|---|---|---|---|
| Since 2026-09-28 the cash sleeve is BIL with 25% withholding (previously DGS3MO accrual, 0% withholding) | Fixed | Book 2012-10 to 2026-08: 2.71% / 1.06 → 1.92% / 0.75 | Audit section 14 |
| 25% withholding on BIL may be too high if the US interest-related-dividend exemption applies | Conservative | Book result lies between 1.9% and 2.5% | G-029 |
| Before BIL's first bar (2007-05-30) the sleeve holds 0% cash | Conservative | Affects 2002-2007 only; quote the DGS3MO excess metric from 2007-06 | G-029 |
| Selection | Selection | 38 variants; familywise p = 0.77 | Leakage hunt 2026-09-27 |
| Pre-2014 Moody's inputs cannot be checked against vintages | Unknown | 141 decisions | `review_bc_quant/REVIEW_BC_QUANT.md` |
| The strategy has been 100% in the cash sleeve since 2022-05 | Note | Its return is close to T-bills | `tierb_macro/tfi_cash_proxy.json` |

## Month-End Rebalancing Flow — `strategy_taa_month_end_rebalancing_flow`

| Caveat | Direction | Size | Evidence |
|---|---|---|---|
| The Sandy 2012 closure date was known only after the fact | Optimistic, negligible | +0.018 pp/yr | `tierb_etf_mr/TIERB_ETF_MR_FINDINGS.md` B-EOM-01 |
| Fees on split-adjusted units | Neutral today | Under 0.001 pp/yr now; a future 40:1 TLT split would move history by -7 pp | B-EOM-02 |
| Trails T-bills over the last 3 years | Note | 2.47% vs 4.46% | B-EOM-03 |
| TLT short: borrow and margin | Account requirement | Borrow at 3% would cost -0.37 pp/yr | B-EOM-04 |
| A live route needs close-auction orders submitted before the venue cutoff | Live risk | Not wired | Fix #23 |

## Sector ETF IBS Downshock — `strategy_mr_us_sector_etf_ibs_downshock_vox_iyr`

| Caveat | Direction | Size | Evidence |
|---|---|---|---|
| Before 2013 the ETFs were thin, so real opening-auction cost exceeded the modelled 2.5 bp | Optimistic before 2013 | House model +0.51 pp/yr full history; +0.12 pp/yr since 2013. Quote results from 2013 | `review_bc_quant/REVIEW_BC_QUANT.md` |
| Fees on split-adjusted units | Conservative | +0.09 pp/yr | B-VOX-02 |
| float32 ties at IBS thresholds | Neutral | Up to 0.1 pp/yr under hypothetical splits; 0 flips in float64 | B-VOX-03 |
| Capacity | Limit | About USD 0.39M | `review_bc_trade/rbt02_*.json` |
| Selection | Selection | 504-cell grid | Leakage hunt 2026-09-27 |

## Sector Dispersion KIE/IHI — `strategy_mr_sector_dispersion_ibs_kie_ihi_xlc`, `..._xlc_asset_sma200`, `..._asset_sma200`

| Caveat | Direction | Size | Evidence |
|---|---|---|---|
| About 30× NAV/yr of turnover in thin ETFs; the modelled 2.5 bp understates auction cost | Optimistic | House model, full history / last 3y: XLC +0.98/+0.64, XLC SMA200 +0.66/+0.50, SMA200 +3.19/+0.70 pp/yr | B-KIE-07 (model rated low confidence for ETFs; see the backlog research item) |
| Capacity | Limit | About USD 64-93K before extra cost passes 0.25 pp/yr | `review_bc_trade/rbt02_*.json` |
| Fees on split-adjusted units | Conservative | +0.05 to +0.36 pp/yr | B-KIE-01 |
| IGV is listed on Cboe BZX; the IBKR opening-order route there is unverified | Unknown | About 15-18% of order notional | B-KIE-05 |
| Selection | Selection | 149 combinations; the SMA200 gate was not pre-registered | Leakage hunt 2026-09-27 |

## TAA 2x variants — `strategy_taa_df_1n_fallback_qld_vix_cash`, `strategy_taa_df_1n_fallback_sso_vix_cash`, `strategy_taa_df_btal_1n_fallback_qld_vix_cash`, `strategy_taa_df_linearity_1n_fallback_qqq_vix_cash`

| Caveat | Direction | Size | Evidence |
|---|---|---|---|
| Fees on split-adjusted units (QLD splits) | Conservative | +0.44 pp/yr for the QLD variant | `tierbc_dv2etf_taa2x/TIERBC_DV2ETF_TAA2X_FINDINGS.md` |
| Capacity | Limit | About USD 170-220K (no-BTAL variants) and USD 127K (BTAL-QLD) before extra auction cost passes 0.25 pp/yr | `review_bc_trade/rbt02_*.json` |
| A live route is blocked today by the calendar window and the snapshot lacks QLD/SSO | Live risk | Not wired | C-TAA2X-01, C-TAA2X-03; fixes #17 and #25 |
| Stale $VIX flips the gate | Live risk | 10 of 211 months (no-BTAL), 4 of 164 (BTAL-QLD) | C-TAA2X-04 |
| One QQQ share is 6.2% of NAV at USD 12K (linearity) | Size-dependent | — | `review_bc_trade/REVIEW_BC_TRADE.md` |

## Trinity Volatility Control — `strategy_taa_trinity_vol_control_8_bil`

| Caveat | Direction | Size | Evidence |
|---|---|---|---|
| Fees on split-adjusted units | Neutral today | Up to -0.17 pp/yr under a hypothetical future BIL split | `tierc_hedge/TIERC_HEDGE_FINDINGS.md` |
| Friction at small size (daily band trades) | Size-dependent | -0.66 pp/yr at USD 12K, -0.27 at USD 30K (IBKR Fixed) | `review_bc_trade/rbt05_*.json` |
| A live route must derive month-end from the exchange calendar | Live risk | Not wired | Fix #26 |

## NDX momentum, sector-capped pair — `strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap`, `strategy_mo_natr20_ndx_vxn_scaled_sector_cap`

The two books of the NDX design for pods of USD 100K and above and for fund books (book `ndx_e2_sector_cap_5050`,
50/50). Decision record: [Momentum family: map, evidence and decision](../research/MOMENTUM_DECISION_20261004.md).
Figures are for the 50/50 pair, 2000-09 to 2026-10, unless a row says otherwise.

| Caveat | Direction | Size | Evidence |
|---|---|---|---|
| Design chosen on 2000-2026 data after about 45 NDX trials (rankings, 38 trend filters, ensembles, the cap) | Selection, optimistic | Plan on Sharpe 0.75 (luck-band median 0.78), not 0.87 | Decision record, sections 2 and 7 |
| Sector labels are today's GICS, applied to all history | Optimistic, mild | Not measurable without point-in-time labels | House ledger G-025 |
| The 40% sector cap costs return in technology booms | Design | 2023-on CAGR 19.7% vs 24.9% without the cap; 2026 to date +15.0% vs +35.3% | Decision record, section 2 |
| The stock selection is not proven against QQQ held at the same exposure | Unproven | P 0.78, Sharpe +0.08; it beats random picks (99.5th percentile) and equal weight (P 0.98) | Registration `ndx_momentum_decision_controls_20261004` |
| Alpha has faded | Decay | After QQQ and a QQQ 200-day rule: 5.7%/yr (t 2.8) full period, 1.2%/yr (t 0.4) since 2013-09 | Decision record, section 4 |
| Whole shares at small size (about 15 names at 5%) | Size-dependent | Intended exposure left in cash: 22% at USD 12K, 10% at 25K, 6% at 50K, 3% at 100K, 1% at 250K | Decision record, section 6 |
| Small-account friction (whole shares, USD 1 minimum fee), pod started 2023-01-03 | Size-dependent | CAGR 15.9% at USD 12K, 17.6% at 25K, 18.6% at 50K, 19.3% at 100K, 19.7% at 1M: -3.8, -2.1, -1.1 and -0.4 pp/yr | Decision record, section 8 |
| Idle cash earns 0% in the engine (the gates hold cash; mean invested 68%) | Conservative | Sharpe 0.84 at 0% cash vs 0.87 with T-bills; CAGR 12.9% vs 13.6% | Decision record, section 2 |
| Capacity at the opening auction | Size-dependent | House MOO model, recent five years: about USD 0.5M for the dollar-ATR book and USD 1M for the NATR20 book; the next day's close costs 0.01 to 0.02 Sharpe | `capacity_analysis` and `execution_timing_analyzer` runs of 2026-10-04 |
| Recent five years below the S&P 500 | Performance | At 0% cash 10.0% and 9.1% a year against 14.1% (2021-10 to 2026-10) | `capacity_analysis` runs of 2026-10-04 |
| No live route (PM_READY) | Status | A live pod needs one strategy that averages the two books, then the WIRED checks | `alpha/strategy_registry.py` |

## Research strategies used in studies

| Strategy | Caveat | Direction | Size |
|---|---|---|---|
| `strategy_mr_dv2_industry_etf` | The industry group was the best of 4 ETF groups | Selection | Industries 1.19 Sharpe vs 0.87 for all 60 ETFs |
| `strategy_mr_dv2_industry_etf` | Docstring figures are stale | Documentation | 7.51% / 1.19 (2012-2026) |
| `strategy_mr_dv2_industry_etf` | Friction | Size-dependent | -2.1 pp/yr at USD 12K |
| `strategy_mo_natr20_ndx_vxn_scaled` | Not better risk-adjusted than the live ATR rule; no live route | Note | Audit section 5b |
| `strategy_mo_mosaic_russell1000`, `strategy_crisis_trend_core`, `strategy_vixm_backwardation` | Demoted to RESEARCH on 2026-09-28 | Status | Audit section 10 |
| `strategy_mr_dv2_vix_gated_{spmo,bil}`, `strategy_mr_hpi_vote_vix_gated_{spmo,bil}` (MR capsule pods) | Gate and parking chosen after about 100 gate variants and about 10 parking forks on 2000-2026 data | Selection, optimistic | DSR 0.97 (N = 110, engine capsule 2004-26); 0.78 from 2018 |
| MR capsule `_spmo` pods | SPMO parking: the research edge came from 2015-11 to 2018-01, when SPMO did not trade most days; from 2018 it adds about 0.5 pp/yr CAGR but costs about 0.04 Sharpe and 2 pp of drawdown | Optimistic (research record) | [Build record](../research/MR_CAPSULE_20261003.md) |
| MR capsule pods | BIL leg: 25% withholding and the engine's 2.5 bps on BIL trades (spread about 1 bp) | Conservative | About +0.4 pp/yr capsule CAGR together (estimate) |
| MR capsule pods | 0% cash before BIL (2004-07); SPMO only from 2018 (tradability guard B1) | Conservative before 2018 | Quote results from 2007-06 or 2018-02 |
| MR capsule pods | Negative cash from the parents' 10 x 10% sizing, not financed (G-023) | Optimistic, small | DV2-G 136 sessions (133 without parking), minimum -8.8% of NAV |
| MR capsule pods | Friction: about 52-58 parking orders a year at the USD 1 minimum | Size-dependent | -0.35 to -0.4 pp/yr at USD 15K per pod, -0.1 at USD 50K |
| MR capsule pods | PM_READY on 2026-10-04 (books `mr_capsule_{spmo,bil}`); WIRED needs a new live order shape for the parking | Status | Handoff `docs/plans/MR_CAPSULE_REVIEW_HANDOFF.md`, section 9 |

<!-- fund-products-20261005:start -->
## Fund product books (growth capsules and monthly books) — `fund_growth.yaml`, `fund_growth_plus.yaml`, `fund_growth_aggressive.yaml`, `fund_growth_monthly.yaml`

Books of the 2026-10-05 fund-products study ([record](../research/FUND_PRODUCTS_20261005.md)): TAA, the momentum capsule and the MR capsule at fixed capital weights (GR1 TAA 3x 40 / 30 / 30, GR2 the same with TAA 3x 1N, GR3 TAA 3x 1N 60 / 20 / 20), and one monthly book of two pods (TAA 3x 1N 60 / CORE5 40). Each pod's own caveats above still apply; these rows are about the books.

| Caveat | Direction | Size | Evidence |
|---|---|---|---|
| Engine, Bench and YAML books pay 0% on idle cash; the headline frame credits DTB3 - 0.5% and charges DTB3 + 1.5% on negative cash | Engine conservative | CAGR gap, percentage points a year: GR1 0.13%; TAA 3x alone 0.03% (about 2% cash), momentum capsule 0.29%, MR capsule 0.10% | Fund products record 2026-10-05 |
| The products are not fully invested | Note | about 26% of GR1 is BIL or cash on an average day | Fund products record 2026-10-05 |
| Three cash treatments live side by side: the MR capsule's BIL is a real position (25% withholding, trade costs); TAA and momentum idle cash is credited DTB3 - 0.5% without cost; the cash leg of a book is BIL total return | Conservative for MR | about 0.6 pp of capsule CAGR; GR1 0.17% under the symmetric treatment | Fund products record 2026-10-05 |
| TAA before 2012-10-02 is a synthetic TQQQ / BTAL proxy (a quarter of the sample, the only 2008) | Unknown | GR1 on 2012+ only: CAGR 20.3%, excess Sharpe 1.40 | Fund products record 2026-10-05 |
| TAA commissions on split-adjusted TQQQ shares | Conservative | about 0.3 pp a year per TAA pod on average 2012-2026 (0.38 pp in the 1N variant), near zero now | Fund products record 2026-10-05 |
| Negative cash is not financed in the engine | Optimistic, small | MR pods 167 and 168 days, to -9.6% of the pod; momentum pods about 1792 days, to -1.9%; TAA pods about 1,500 days (2012 on), to -1.8%; charged DTB3 + 1.5% in the headline frame | Fund products record 2026-10-05 |
| Two of the three engines are long Nasdaq when risk is on (TAA through TQQQ, momentum through Nasdaq-100 stocks); no 2000-02 type bear in the sample | Optimistic | peak look-through 1.51x (GR1), 2.01x (GR3) at target weights; 1.69x and 2.20x as carried; a 10% one-day Nasdaq fall at peak exposure costs -18.0% and -22.0% | Fund products record 2026-10-05 |
| Every engine was selected on this history; nothing is out of sample (MR: about 110 variants; momentum: about 45 trials plus a 151-configuration grid; the products: 3 of 14 books seen before the freeze) | Optimistic | the backtest is the headline (GR1 CAGR 18.3%, excess Sharpe 1.27); conservative case, 3/4 of the excess return at model costs: 13.5% / 0.95; GR1's excess Sharpe was 1.56 in 2012-2021 and 1.07 since 2022 | Fund products record 2026-10-05 |
| The product weights were chosen by the owner after the results, from the pre-registered dial points (GR1 and GR2 at 40 / 30 / 30; the registered default was equal capital; the 40 / 30 / 30 point was not in the grid seen before the freeze) | Optimistic | GR1 against equal capital: a tie by the 80% rule on Sharpe (70% of paired paths) | Fund products record 2026-10-05 |
| The Monthly book (TAA 3x 1N 60 / CORE5 40) was designed after the results from an exploratory grid that was not pre-registered; the owner chose the ratio | Optimistic | differences between neighbouring grid rows are small; one return engine: with TAA dead the Monthly book keeps an excess Sharpe of 0.14 (GR1 0.69) | Fund products record 2026-10-05 |
| Breach figures are a block-63, full-edge convention | Optimistic | GR1 at -20%: 2.4% by the rule, 10.2% in the conservative case | Fund products record 2026-10-05 |
| MR capsule cost sensitivity; no live fills; the slippage gate (4 bps per side over 200 fills) is unmet; 2020-21 are about a third of its log-wealth; stock turnover about 41 times the pod a year | Optimistic until measured | capsule alone 16.8% -> 13.7% at +5 bps; GR1 18.3% -> 17.0% | Fund products record 2026-10-05 |
| Momentum capsule: today's GICS labels; selection unproven against QQQ at the same exposure; in a drawdown of about 18% from its June 2026 peak (at 2026-10-02) | Unknown | in the slot test BIL in its place has the higher excess Sharpe on 76% of paths and 4.0 pp less CAGR: it adds return, not risk-adjusted return | Fund products record 2026-10-05 |
| Annual reset is a cost-free transfer on one date | Optimistic, small | GR1 CAGR over the twelve start months 18.2% to 18.6% | Fund products record 2026-10-05 |
| Capacity is pre-TCA and route-dependent | Unknown | GR1 $2.5M at the open (modelled), up to $10M if the MR stocks trade at the close (upper bound, not modelled); binding legs: TAA / BTAL on one-day routes, MR stocks (FOX, NWS) on the worked route; Monthly $5M at the open, $235.9M (BTAL wall) worked | Fund products record 2026-10-05 |
| The Monthly book's rung and its margin | Optimistic | The Monthly book passes GROWTH PLUS and not GROWTH (29.6% of paths beyond -20%), so it carries more risk than GR1, not less. Its rung holds down to 0.80 of the edge (GR1 0.70); in the conservative case its breach figure at -25% is 18.2% against the 15% cap. | Fund products record 2026-10-05 |
| Leverage on GR1 (fixed 1.40x) is an alternative, not a product | Optimistic | financing at DTB3 + 1.5%, no margin calls or gap days modelled, leverage also levers the MR capsule's trading cost and divides capacity by 1.4; not available while each pod sits in its own Reg-T account | Fund products record 2026-10-05 |
| Minimum clean size | Note | GR1 about $667K, Monthly about $75K | Fund products record 2026-10-05 |
| Wiring | Note | wired: TAA 3x, TAA 3x 1N, BTAL_QQQ, NDX-VXN (of these only TAA 3x has a pod trading live today; TAA 3x 1N has a live route but is not running); not wired: CORE5 (PM_READY) and the four momentum and MR capsule pods (PM_READY, no live route at 5c0d48d; the MR capsule needs a new order shape and two margin accounts); the Monthly book needs only CORE5 | Fund products record 2026-10-05 |
| Margin rows | Optimistic | DTB3 + 1.5%, no margin calls, no gap; volatility match is approximate; capacity divides by L | Fund products record 2026-10-05 |
| Gross against net of 2/20 | Note | GR1 18.3% gross, 12.8% net; Monthly 19.8% gross, 14.0% net; no fund expenses | Fund products record 2026-10-05 |
| Sharpe basis | Note | GR1 excess Sharpe 1.27 daily, 1.48 monthly, 1.37 at a zero rate | Fund products record 2026-10-05 |
| The weeks after the window are not out of sample | Note | 2026-08-20 to 2026-10-02: GR1 1.1%, Monthly 3.9% | Fund products record 2026-10-05 |
| The engine BIL pod is not BIL total return | Note | 1.08% vs 1.61% a year from 2012 (withholding, no reinvestment) | Fund products record 2026-10-05 |
<!-- fund-products-20261005:end -->
