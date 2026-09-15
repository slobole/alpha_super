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

- :material-flask-outline: **112 published studies**
- :material-clipboard-alert-outline: **15 records need audit**
- :material-tag-multiple-outline: **80 signal families**
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
| research_candidate | 7 | Passed the declared research gate; still research-only. |
| forward_hypothesis | 22 | Frozen idea awaiting genuinely new evidence. |
| diagnostic | 83 | Useful evidence or state variable; not a strategy recommendation. |

## Most recently reviewed

| Study | Status | Family | Reviewed | Verdict |
| --- | --- | --- | --- | --- |
| [Negative-week depth sizing, VIX and trend regimes across SPY, QQQ, BTC, GLD and SMH](studies/badweek_multiasset_sizing_regime_20260915.md) | diagnostic | mean_reversion | 2026-09-15T18:01:47.241303+00:00 | Diagnostic: normalized depth does not reliably select stronger rebounds across assets. Own-asset MA200 shows a useful historical QQQ/SMH risk-return tradeoff; VIX scaling is not a broad improvement. All seven conditional rules trail own-asset buyhold in full-period CAGR and Sharpe. Preserve all results and treat narrower follow-up ideas as post-result hypotheses requiring a separate prospective freeze. |
| [SPY after a negative week: stateful backtest and VIX forecast portfolios](studies/badweek_spy_backtest_20260915.md) | diagnostic | mean_reversion | 2026-09-15T17:04:32.600723+00:00 | The negative-week strategy earns 6.66% CAGR at 10 bps round trip with 32.74% maximum drawdown versus buyhold 11.00% and 55.19%. It holds equity about 39% of days. Continuous next-open carry greatly reduces forecast-portfolio churn; daily VIX carry earns 9.65% versus AR1 carry 9.37%, an unproven small increment. Every central-cost strategy has lower CAGR than buyhold. diagnostic. These are reproducible research backtests with lower-exposure tradeoffs, not independently confirmed alpha, deployment readiness, or capital-allocation advice. |
| [SPY weekly reversal: VIX level and daily/weekly return regressions](studies/badweek_spy_vix_regression_20260915.md) | diagnostic | mean_reversion | 2026-09-15T16:21:06.111593+00:00 | Weekly reversal direction reproduced. VIX level and weekly change add no stable prediction; daily change helps only in the late period and fails corrected/stability gates. Diagnostic only. |
| [Four dip-buying systems in one shared cash account](studies/tradequantix_four_system_shared_account_study.md) | diagnostic | multi_system_etf_mean_reversion | 2026-09-15T05:34:11.972153+00:00 | Positive conditional historical growth and lower drawdown; validation retains38.5%of funded SPY growth and fails MAR, so later remainsclosed and no promotion. |
| [מניות דיבידנד יציב בארה״ב — יציאה ושווי לא ודאי](studies/tradequantix_dividend_stability_interpretation_study.md) | diagnostic | dividend_stability_trend_exit | 2026-09-14T23:45:21.894588+00:00 | אין קידום: היציאה הקטינה ירידות במודל, אך שימור התשואה נכשל בבדיקה ושווי ניכר נשען על מחירים ישנים. הערך הכלכלי אינו מוכרע. |
| [עונתיות SPY לפני חגים — פרשנות ציבורית](studies/tradequantix_preholiday_interpretation_study.md) | diagnostic | holiday_seasonality | 2026-09-14T22:13:04.537059+00:00 | הפרשנות לחגים אינה מקודמת: תוצאה חיובית ב־2010–2018, אך חלשה ורגישה לעלות ב־1994–2009, ויחס התשואה לסיכון נכשל מול ההשוואה בשתי התקופות. |
| [VIX temporal-trend insurance: protection survives but later return cost fails](studies/tradequantix_vix_trend_interpretation_study.md) | diagnostic | vix_temporal_trend_hedge | 2026-09-14T19:38:00.775852+00:00 | The fixed temporal VIX-trend hedge provides crisis protection, but fails its predeclared later return-cost gate; retain the mechanism as evidence, reject this candidate for promotion. No exact author replication or trading approval. |
| [Coupled momentum-flat ETF short hedge: terminal-policy sensitivity](studies/tradequantix_flat_hedge_interpretation_study.md) | diagnostic | coupled_momentum_flat_hedge | 2026-09-14T18:53:46.145798+00:00 | Unresolved terminal accounting prevents economic approval; the fixed coupled hedge fails its conditional hedge-role gate. Diagnostic/inconclusive, no trading promotion. |
| [Independent inverse-RSI closing-entry tiers](studies/tradequantix_rsi_tiers_interpretation_study.md) | diagnostic | independent_rsi_tier_mean_reversion | 2026-09-14T18:10:26.437899+00:00 | Declared six-fund independent RSI tier interpretation passes the frozen economic gate but remains diagnostic because auction/source/short-later evidence is incomplete. |
| [ATR-stretch ETF intraday buying with inverse-RSI closing exits](studies/tradequantix_sma_atr_rsi_interpretation_study.md) | diagnostic | staged_etf_mean_reversion | 2026-09-14T17:44:55.765131+00:00 | Declared six-fund staged intraday ETF interpretation passes the frozen economic gate but remains diagnostic because auction/source/short-later evidence is incomplete. |

## Largest research families

| Family | Studies |
| --- | ---: |
| Mean Reversion | 13 |
| Cross Sectional Trend | 5 |
| Long Equity Mean Reversion | 4 |
| Staged Etf Mean Reversion | 4 |
| Overnight Return Persistence | 3 |
| Cross Asset Momentum | 3 |
| Cross Asset Risk Allocation | 2 |
| Calendar Cross Asset Rebalance Flow | 2 |
| Calendar Cross Asset Reversal | 2 |
| Cross Sectional Long Only Hpi Mean Reversion | 2 |
| Market Risk Regime | 2 |
| Leveraged Etf Hedged Short | 2 |
| Cross Asset Momentum Robust Covariance Minvar | 1 |
| Dual Momentum | 1 |
| Cross Market Sentiment And Short Horizon Equity Reversal | 1 |
| Valuation Yield Curve Regime Router | 1 |
| Cross Market Sentiment | 1 |
| Cross Sectional Momentum | 1 |
| Asymmetric Cross Asset Time Series Momentum Tail Hedge | 1 |
| Intraday Vwap Drift Continuation | 1 |
| Cross Asset Dual Momentum | 1 |
| Cross Sectional Intraday Reversal | 1 |
| Calendar Mean Reversion | 1 |
| Calendar Rebalancing Flows | 1 |
| Term Structure Inversion Stateful Long Vixy Tail Hedge | 1 |
| Calendar Conditional Safe Haven Flow | 1 |
| Risk Overlay | 1 |
| Macro Regime Rotation | 1 |
| Cross Sectional Mean Reversion | 1 |
| Defensive Tactical Allocation | 1 |
| Cross Sectional Low Volatility | 1 |
| Cross Asset Calendar Reversal | 1 |
| Market Regime Allocation | 1 |
| Cross Asset Momentum Mean Variance Allocation | 1 |
| Index Daily Mean Reversion Variance Ratio | 1 |
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
| Sector Defensive Equity Next To Tail Hedge Pod | 1 |
| Sector Defensive Equity Tail Hedge Claim | 1 |
| Etf Multiasset Trend Anti Beta | 1 |
| Country Etf Momentum | 1 |
| Dividend Stability Trend Exit | 1 |
| Low Volatility And Nominal Atr Momentum | 1 |
| Multiasset Trend Volatility Allocation | 1 |
| Etf Short Horizon Mean Reversion | 1 |
| Coupled Momentum Flat Hedge | 1 |
| Multi System Etf Mean Reversion | 1 |
| Etf Limit Mean Reversion | 1 |
| Mega Cap Limit Mean Reversion | 1 |
| Us Vix Term Structure Long Etn | 1 |
| Momentum Pullback Mean Reversion | 1 |
| Volatility Normalized Equity Momentum | 1 |
| Oex Log Moving Average Vote Trend | 1 |
| Holiday Seasonality | 1 |
| Independent Rsi Tier Mean Reversion | 1 |
| Us R1000 Short Intraday Overextension | 1 |
| Us Etf Short Relief Rally Inverse Rsi | 1 |
| Long Equity Trend Momentum Volatility Sizing | 1 |
| Staged Mean Reversion | 1 |
| Vix Temporal Trend Hedge | 1 |
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
