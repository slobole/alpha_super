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
