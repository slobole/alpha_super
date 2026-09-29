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

- :material-flask-outline: **140 published studies**
- :material-clipboard-alert-outline: **3 records need audit**
- :material-tag-multiple-outline: **96 signal families**
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
| research_candidate | 8 | Passed the declared research gate; still research-only. |
| forward_hypothesis | 27 | Frozen idea awaiting genuinely new evidence. |
| diagnostic | 105 | Useful evidence or state variable; not a strategy recommendation. |

## Most recently reviewed

| Study | Status | Family | Reviewed | Verdict |
| --- | --- | --- | --- | --- |
| [Industry/factor residual reversal and momentum (GICS vs sector ETF vs statistical peers vs PCA) on S&P 500, NDX, R1000](studies/gics_residual_reversal_momentum_study.md) | diagnostic | mean_reversion | 2026-09-28T23:45:44+03:00 | Residual (industry/factor-relative) short-term reversal does not beat raw reversal in US large caps 2000-2026 (wins only 2013-19); weekly reversal is mostly industry-level; IBS edge is overnight and decayed; no long-only book beats EW after costs in both holdouts; nothing improves G3. H0 Concretum sector-ETF rule directionally replicated. |
| [Hunt Gather Trade Strategy 6 ATR mean-reversion audit](studies/hgt_strategy6_atr_mean_reversion_audit.md) | diagnostic | short_term_atr_mean_reversion | 2026-09-28T21:05:00+00:00 | Claim not reproduced. The combined book earned nothing after publication (central Sharpe -0.10, CAGR -6%) and 67% of its 2000-2026 growth came from 2000-2010. The long side is simply buying volatile dips with a deep limit; the oversold condition adds nothing (t -0.45). The short side - sell into a second-day spike of an already overbought $5-20 stock and cover at the next open - is a real intraday fade (+1.6 pp per trade vs a same-day control, t 6.8, positive in every slice), but it is small (~4% average exposure), depends on fills near the day's high, loses strength on a broader look-ahead-free universe and carries squeeze trades of -90% to -170%. Do not trade the book; keep the short-side fade as a forward hypothesis. |
| [VIX-gated weekly reversal of sector and industry ETFs](studies/vix_gated_etf_industry_reversal_study.md) | diagnostic | mean_reversion | 2026-09-28T20:52:09+00:00 | Buying the week's weakest sector / industry ETFs when VIX is above its 1-year median beats equal weight only in 2000-12; after 2012 no variant beats EW in either universe, and the VIX gate adds nothing beyond the ungated rule (the gate alone is worse than EW). VIX does raise the reversal IC (3.2% vs 0.5% in sectors). No forward paper test. |
| [Growth and Inflation Sector Timing Model (Varadi 2025) - replication, causal translation, long-history and selection-bias tests](studies/growth_inflation_sector_timing_study.md) | diagnostic | macro_regime_rotation | 2026-09-28T12:02:54+00:00 | Source-like path matches the article volatility and drawdown; the executable path beats SPY over 1999-2026 only because of 1999-2008 and has trailed SPY since Nov 2008. The literal map works on 1927-1989 French industries it never saw (p=0.005), but walk-forward map selection fails and the frozen G1 gate fails; diagnostic only. |
| [Momentum/trend universe search after the NDX split look-ahead: orthogonal stack, era-robust NDX momentum, NDX-RM beside NDX-L](studies/momentum_trend_universe_search.md) | diagnostic | cross_sectional_momentum_with_orthogonal_companions | 2026-09-27T12:00:00+03:00 | GOAL NOT MET ROBUSTLY. The pre-registered design winner (R1000 momentum + pullback + seasonality + lottery avoidance, design Sharpe 1.37) failed the 2012-2026 holdout (Sharpe 0.51, book 1.15). Post-hoc, era-robust NDX residual momentum with size-aware weights (NDX-RM) reaches book 1.34 as a replacement and 1.36 / -17.4% beside NDX-L (TAA 50 / L 25 / RM 25), but only because of 2026 (1.31 through 2025) and not significantly better than G3 (p 0.18). Long-only US momentum sleeves cap the book near 1.30-1.34 because TAA's TQQQ leg already carries the Nasdaq factor. Shadow NDX-RM beside L; do not replace L. |
| [SetupAlpha S&P 500 Short-Term Mean Reversion (Connors/Alvarez pullback) claim audit](studies/setupalpha_sp500_connors_alvarez_audit.md) | forward_hypothesis | mean_reversion | 2026-09-26T12:21:17+00:00 | Vendor profile is replicated (even exceeded) by a free generic RSI2<5/SMA200/4%-limit rule: anchor 18.3%/Sharpe 1.02/-30% at 10 bps with 150% gross. The edge lives in the deep limit fill and the uptrend filter, is concentrated in 2000-2002 and 2024-2026, and is weak 2015-2024 (Sharpe 0.35-0.51). Frozen promotion rule fails; forward hypothesis. |
| [SetupAlpha S&P 500 Mean Reversion 2025 (candlestick-confirmed limit entry) claim audit](studies/setupalpha_sp500_mr2025_audit.md) | diagnostic | mean_reversion | 2026-09-26T12:21:17+00:00 | Vendor profile (Sharpe 1.40, 19.6%) is not reachable by 48 transparent candlestick-confirmed limit-entry rules (median Sharpe 0.14 at 2 bps, best 0.59); confirmation removes the oversold edge and the anchor loses money. Do not buy; nothing to shadow from this product. |
| [SetupAlpha SPX Mean-Reversion (rate-of-decline limit entry) claim audit](studies/setupalpha_sp500_rate_of_decline_audit.md) | diagnostic | mean_reversion | 2026-09-26T12:21:17+00:00 | Vendor profile (Sharpe 1.16, 19.8%) is not reachable (family median 0.43 at 2 bps, best 0.93, ~1e5 trials needed) and the vendor monthly series is uncorrelated with the family; decline signals add nothing at market entry, returns come from the limit discount and 2000-2002. Do not buy. |
| [SetupAlpha Russell 3000 all-time-high pullback mean-reversion audit](studies/setupalpha_r3000_ath_pullback_audit.md) | diagnostic | mean_reversion | 2026-09-26T12:20:42+00:00 | A generic Russell 3000 ATH-pullback family reproduces the vendor's return level (median 15.4% CAGR at 2 bps) but not its risk (Sharpe 0.74 vs 1.09, MaxDD -52% vs -24%). The RSI2 pullback after a recent all-time high adds ~30 bps per 9-day trade over liquid R3000 names and ~20 bps over a generic pullback, but the edge sits in 2000-2014 and 2024-26 and is ~0 in 2015-2024; limit-fill optimism is worth ~1/3 of CAGR; small-cap capacity is soft at $1M and strained at $10M. Do not buy; do not trade live. |
| [SetupAlpha low-drawdown Nasdaq 100 mean-reversion audit (volatility-scaling overlay)](studies/setupalpha_ndx_low_drawdown_mr_audit.md) | diagnostic | mean_reversion | 2026-09-26T12:16:42+00:00 | The low-drawdown profile is not reproduced (family MaxDD -13% to -26% vs vendor -8.6%); the volatility-scaling overlay only lowers exposure and costs Sharpe versus an exposure-matched constant size (0/16 pairs better) because mean-reversion trades pay most in turbulent markets. Timing edge real but thin and decaying; limit-fill optimism worth ~1/3 of Sharpe. Do not buy; do not trade live. |

## Largest research families

| Family | Studies |
| --- | ---: |
| Mean Reversion | 21 |
| Cross Sectional Trend | 5 |
| Long Equity Mean Reversion | 4 |
| Cross Asset Momentum | 4 |
| Staged Etf Mean Reversion | 4 |
| Overnight Return Persistence | 3 |
| Cross Asset Risk Allocation | 2 |
| Calendar Cross Asset Rebalance Flow | 2 |
| Calendar Cross Asset Reversal | 2 |
| Macro Regime Rotation | 2 |
| Cross Sectional Long Only Hpi Mean Reversion | 2 |
| Market Risk Regime | 2 |
| Cross Sectional Path Shape Reversal | 2 |
| Momentum Rotation | 2 |
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
| Short Term Atr Mean Reversion | 1 |
| Risk Overlay | 1 |
| Macro Regime Growth Classifier | 1 |
| Macro Regime Portfolio Construction | 1 |
| Cross Sectional Mean Reversion | 1 |
| Defensive Tactical Allocation | 1 |
| Cross Sectional Low Volatility | 1 |
| Cross Asset Calendar Reversal | 1 |
| Market Regime Allocation | 1 |
| Cross Asset Momentum Mean Variance Allocation | 1 |
| Cross Sectional Momentum With Orthogonal Companions | 1 |
| Index Daily Mean Reversion Variance Ratio | 1 |
| Cross Asset Pca Risk Regime | 1 |
| Tail Risk Overlay | 1 |
| Factor Momentum | 1 |
| Closing Auction Basis | 1 |
| Price Path Convexity Short Horizon Reversal | 1 |
| Short Horizon Cross Sectional Reversal | 1 |
| Cross Sectional Momentum Rotation | 1 |
| Cross Asset Momentum Rotation | 1 |
| Cross Asset Flow Front Running | 1 |
| Short Exhaustion Reversal | 1 |
| Breakout Trend | 1 |
| Mean Reversion Weekly Calendar | 1 |
| Adaptive Time Series Momentum Regime | 1 |
| Cross Market Short Volatility Mean Reversion | 1 |
| Spy Rsi2 Short Volatility Tail Control | 1 |
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
| Trend Following Adaptive Lookback | 1 |
| Factor Etf Rotation | 1 |
| Drawdown Adaptive Time Series Momentum | 1 |
| Drawdown Conditioned Adaptive Moving Average Trend With Independent Fixed Sleeves | 1 |
| Drawdown Conditioned Adaptive Moving Average Trend With Breadth Preserving Core4 Sleeves | 1 |
| Post Selected Dbc Vardi Out State Short With Inverse Volatility Risk Budget | 1 |
| Vardi Drawdown Adaptive Momentum Translated Into Stateful Short Side Etf Sleeves | 1 |
| Frozen Adaptive Core5 Timing With Alternative Actual Equity Execution Vehicles | 1 |
| Asset Local Adaptive Macro Timing With Five Fixed 20% Sleeves And Bil Reserve | 1 |
| Adaptive Momentum Portfolio Construction | 1 |
| Frozen Adaptive Core5 Timing With Spy Or Qqq Equity Execution Vehicle | 1 |
| Adaptive Macro Asset Timing With An Actual Sso Execution Vehicle And Frozen Dbc Short | 1 |
| Adaptive Macro Asset Timing With Fixed Sleeves And Optional Dbc Short Risk Budget | 1 |
| Adaptive Macro Asset Timing With Optional Volatility Normalized Uup And Dbc Shorts | 1 |
| Asset Local Adaptive Trend State And Long Short Asymmetry | 1 |
