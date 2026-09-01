---
title: "Research Knowledge Base"
description: "Searchable summaries of Pakal quantitative research studies."
document_type: research
authority: guide
risk_scope: research
source_paths:
  - "pakal-research/knowledge/research_registry.json"
  - "pakal-research/reports/*/knowledge_record.json"
---

<!-- GENERATED FILE. EDIT THE CANONICAL PAKAL RECORDS, NOT THIS PAGE. -->

# Research Knowledge Base

One place to find what was tested, what survived, what failed, and what remains unknown.

!!! warning "Research-only boundary"
    Inclusion here is not LIVE, allocation, broker, release, or deployment authorization.

<div class="grid cards" markdown>

- :material-flask-outline: **68 published studies**
- :material-clipboard-alert-outline: **12 records need audit**
- :material-tag-multiple-outline: **45 signal families**
- :material-magnify: **Full-text search** is available from the top bar

</div>

## Start here

- [Browse every study](studies.md)
- [Explore signal and portfolio features](features.md)
- [Trace external and internal sources](sources.md)
- [Review records that need repair](needs-audit.md)
- [Understand statuses and evidence boundaries](how-to-read.md)

## Research status

| Status | Count | Meaning |
| --- | ---: | --- |
| research_candidate | 4 | Passed the declared research gate; still research-only. |
| forward_hypothesis | 18 | Frozen idea awaiting genuinely new evidence. |
| diagnostic | 46 | Useful evidence or state variable; not a strategy recommendation. |

## Most recently reviewed

| Study | Status | Family | Reviewed | Verdict |
| --- | --- | --- | --- | --- |
| [שורט קרנות ממונפות: הון, פתיחה הבאה ועלויות מימון](studies/shorting_leveraged_etfs_empirical_study.md) | diagnostic | leveraged_etf_hedged_short | 2026-08-28T19:47:21.753938+00:00 | No robust excess return is established: central bull hedges are close to cash, all four fail conservative costs, and leveraged pairs can breach the EOD margin proxy. Diagnostic only; do not promote. |
| [שורט בקרנות ממונפות: שחיקה, סיכון מסלול ועלויות אמיתיות](studies/shorting_leveraged_etfs_study.md) | diagnostic | leveraged_etf_hedged_short | 2026-08-28T16:10:34.392446+00:00 | Diagnostic only: volatility decay is not a free lunch; source fee arithmetic is consistent, but executable excess returns remain untested. |
| [Volatility-indicator probabilities as risk overlays for SPY and NDX momentum](studies/indicator_volatility_risk_overlay.md) | diagnostic | risk_overlay | 2026-08-28T11:33:59.875904+00:00 | Reject replacing or combining current NDX VXN with this probability multiplier. SPY has useful downside reduction but insufficient stable validation; research-only, no promotion. |
| [HPI Russell 3000 Q5 Exposure-Matched Study](studies/hpi_r3000_q5_exposure_match_study.md) | forward_hypothesis | mean_reversion | 2026-08-22T22:06:09+00:00 | Q5 passed every frozen equal-capital, exposure-efficiency, stability, cost, and capacity gate, but only as a forward hypothesis because the full history was already seen. |
| [HPI Russell 3000 Liquidity Mean-Reversion Study](studies/hpi_r3000_liquidity_mr_study.md) | diagnostic | mean_reversion | 2026-08-22T20:47:10+00:00 | Low ADV did not improve HPI. The least-liquid quintile had weaker return, Sharpe, drawdown, tail behavior, and unusable selected-order capacity; all three frozen gates failed. |
| [Vardi CORE5 SPY versus QQQ Early Sample Study](studies/vardi_core5_spy_qqq_early_sample_study.md) | forward_hypothesis | Frozen adaptive CORE5 timing with SPY or QQQ equity execution vehicle | 2026-08-22T20:18:00Z | QQQ is preferred to SPY on the full 2008-2026 sample, with higher CAGR and Sharpe but modestly deeper downside. |
| [HPI S&P 500 Risk-Filter Transfer](studies/hpi_sp500_risk_filter_transfer.md) | diagnostic | cross-sectional long-only HPI mean reversion | 2026-08-22T19:24:00+00:00 | No tested overlay improved the raw HPI portfolio across frozen economic, stability, multiplicity, exposure, and cost gates; retain raw HPI. |
| [Vardi CORE5 Equity Vehicle Comparison Study](studies/vardi_core5_equity_vehicle_comparison_study.md) | forward_hypothesis | Frozen adaptive CORE5 timing with alternative actual equity execution vehicles | 2026-08-22T18:55:00Z | QQQ is the balanced winner and QLD the growth candidate; UPRO and TQQQ add disproportionate downside. |
| [HPI More Bets, rolling CVaR and fragility filters](studies/hpi_more_bets_cvar_study.md) | diagnostic | cross-sectional long-only HPI mean reversion | 2026-08-22T09:42:05+00:00 | נמצאו שיפורים נקודתיים, אך הם נכשלו באחד או יותר משערי התקופות, החשיפה, המובהקות או הקיבולת; אין מועמד לקידום. |
| [Vardi CORE5 SSO Equity Sleeve Study](studies/vardi_core5_sso_equity_sleeve_study.md) | forward_hypothesis | Adaptive macro-asset timing with an actual SSO execution vehicle and frozen DBC short | 2026-08-21T21:42:51Z | Select E2 full SSO sleeve by frozen hierarchy; keep E1 as the lower-risk alternative; forward evidence required. |

## Largest research families

| Family | Studies |
| --- | ---: |
| Mean Reversion | 10 |
| Cross Sectional Trend | 5 |
| Long Equity Mean Reversion | 4 |
| Cross Asset Momentum | 3 |
| Cross Asset Risk Allocation | 2 |
| Calendar Cross Asset Reversal | 2 |
| Cross Sectional Long Only Hpi Mean Reversion | 2 |
| Market Risk Regime | 2 |
| Leveraged Etf Hedged Short | 2 |
| Dual Momentum | 1 |
| Cross Market Sentiment And Short Horizon Equity Reversal | 1 |
| Cross Market Sentiment | 1 |
| Cross Sectional Momentum | 1 |
| Intraday Vwap Drift Continuation | 1 |
| Cross Asset Dual Momentum | 1 |
| Calendar Mean Reversion | 1 |
| Calendar Conditional Safe Haven Flow | 1 |
| Risk Overlay | 1 |
| Macro Regime Rotation | 1 |
| Cross Sectional Mean Reversion | 1 |
| Defensive Tactical Allocation | 1 |
| Cross Sectional Low Volatility | 1 |
| Market Regime Allocation | 1 |
| Cross Asset Pca Risk Regime | 1 |
| Factor Momentum | 1 |
| Closing Auction Basis | 1 |
| Price Path Convexity Short Horizon Reversal | 1 |
| Cross Sectional Momentum Rotation | 1 |
| Cross Asset Momentum Rotation | 1 |
| Cross Asset Flow Front Running | 1 |
| Adaptive Time Series Momentum Regime | 1 |
| Cross Market Short Volatility Mean Reversion | 1 |
| Weekly Mean Reversion | 1 |
| Historical Yield Spread Rank Tactical Bonds | 1 |
| Factor Etf Rotation | 1 |
| Drawdown Adaptive Time Series Momentum | 1 |
| Drawdown Conditioned Adaptive Moving Average Trend With Independent Fixed Sleeves | 1 |
| Drawdown Conditioned Adaptive Moving Average Trend With Breadth Preserving Core4 Sleeves | 1 |
| Post Selected Dbc Vardi Out State Short With Inverse Volatility Risk Budget | 1 |
| Vardi Drawdown Adaptive Momentum Translated Into Stateful Short Side Etf Sleeves | 1 |
| Frozen Adaptive Core5 Timing With Alternative Actual Equity Execution Vehicles | 1 |
| Frozen Adaptive Core5 Timing With Spy Or Qqq Equity Execution Vehicle | 1 |
| Adaptive Macro Asset Timing With An Actual Sso Execution Vehicle And Frozen Dbc Short | 1 |
| Adaptive Macro Asset Timing With Fixed Sleeves And Optional Dbc Short Risk Budget | 1 |
| Adaptive Macro Asset Timing With Optional Volatility Normalized Uup And Dbc Shorts | 1 |
