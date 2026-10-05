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

- :material-flask-outline: **153 published studies**
- :material-clipboard-alert-outline: **3 records need audit**
- :material-tag-multiple-outline: **106 signal families**
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
| forward_hypothesis | 32 | Frozen idea awaiting genuinely new evidence. |
| diagnostic | 113 | Useful evidence or state variable; not a strategy recommendation. |

## Most recently reviewed

| Study | Status | Family | Reviewed | Verdict |
| --- | --- | --- | --- | --- |
| [Market Meanness Index (financial-hacker.com, jcl 2015): trend-regime filter for 900-style peak/valley trend systems - math audit and post-publication replication](studies/fh_market_meanness_index_study.md) | diagnostic | regime_filter_serial_correlation_median_reversion | 2026-10-04T03:10:00+03:00 | REJECTED. The 75% rule is correct, but MMI(returns) is exactly 50 + 100*arccos(rho1)/(2*pi) of the lag-1 autocorrelation (sampling sd 2.2 pts at N=300 vs 0.8 pts per 0.05 of rho) and MMI(price) is ~50-53% for any random walk with or without drift; on ES/NQ/GC/BTC it is indistinguishable from shuffled returns. Post-publication replication 2017-2026 on 840 intraday trend systems (10 filters x M15/H1/H4 x ES/NQ/GC/BTC, literal Zorro ports, detrended, gross): the MMI(price) gate lowers median profit factor by 1.5-3.4%, improves 32-43% of systems, ranks at the 3rd-29th percentile of 100 time-shifted copies of itself (Holm p 1.0), carries no conditional information (Holm p 1.0), and cuts the share of profitable systems from 67.0% to 51.4% (source claimed +2..+9 pts). No MMI family passes the reality check (p 0.26 gross); the only gross pass is an unfiltered BTC 15m trend system that fails at central cost. Daily ETFs (700 systems) and $SPX 1930+ show the same null with sign flips across MMI windows. Cost-adjusted ensemble gains (+0.04..+0.15 Sharpe) equal those of shifted gates: they come from ~57% fewer trades, not timing. |
| [QuantSeeker momentum & trend archive (15 articles): vol/semivol scaling, IQR momentum timing, 52-week-high neutral, HTP/PTH, informative (earnings) days, smoothness, FCF, linearity TAA, managed-futures ETFs, multi-timeframe BTC, crypto rotation, intraday momentum](studies/quantseeker_momentum_trend_study.md) | forward_hypothesis | momentum_and_trend_overlays | 2026-10-02T19:00:00+03:00 | Most article claims reproduce in direction, but every 'better momentum ranking' (smoothness, 52-week-high neutral, HTP/PTH, FCF, earnings-day or event-weighted returns, R2*slope in TAA, hourly BTC timing, semivolatility) is either not significant, eaten by costs, or reversed after publication. What survives is the WHEN layer: (1) volatility scaling of momentum (French MOM Sharpe 0.51->0.82 since 1963, still +0.20 after 2015; stock L/S max DD -78% -> -29%), and (2) the Liu et al. IQR dispersion switch (cash when last month's cross-sectional return IQR > 80th pct of the prior 60 months): +0.21 Sharpe on 12-1 L/S at central cost (p 0.055), adds +0.10 on top of vol scaling (p 0.09), holds 1995-99 on the French factor, pre-publication and in the 2025+ holdout, leave-one-year-out 0.31-0.42. The vol+IQR 12-1 L/S book is weak standalone (central Sharpe 0.37, CAGR 4.1%, DD -32%) but nearly uncorrelated with BOOK-A (0.07) and raises its monthly Sharpe 1.14 -> 1.21 (2007+) / 1.38 -> 1.53 (2019+) at a 20% overlay. Study-wide BH q for IQR is 0.32: a coherent forward hypothesis, not a confirmed edge. Status: shadow (paper-log the IQR flag and the L/S book), no allocation. |
| [Quant Seeker VIX / volatility-timing archive (9 articles): replication, executable short-vol timing, crisis-only sleeves vs the crisis-trend pod](studies/quantseeker_volatility_study.md) | forward_hypothesis | volatility_term_structure_timing_and_crisis_hedging | 2026-10-02T15:44:41+03:00 | Signal claims mostly reproduce (QS9 curve momentum, QS1 slope switch, QS4 crisis-alpha L/S, QS6 front-end inversion forecasts RV: replicated; QS2 FOMC, QS3 VIX-timed beta, QS8 drawdown probit: directional; QS5 bond-vol HAR improvement: not reproduced). Executable: VIX-curve timing of short vol halves the drawdown (A5 SV-ENS MDD -31% vs -75% buy-and-hold) but does not raise Sharpe vs holding SVXY or the plain level rule; HAR VX-futures timing is a post-2020 artefact (loses 100% on 2008-2020, negative in 2025-26). No VIX/MOVE trigger adds to the pod at equal budget: faster triggers (VIX9D>VIX, negative curve momentum) pay 2-4x more in shocks but bleed 12-33%/yr in calm years; VIX hedges lose in 2022. Crisis-alpha L/S replicates (+4%/yr net, positive in 8/8 stress windows incl. 2022) and improves an 80/20 book only as an extra sleeve (post-hoc). |
| [QuantSeeker short-term mean reversion: stock-bond spread, IBS/MTSI, cross-asset IBS, intraday asymmetry, metals pairs, buy-the-dip, Gatev pairs](studies/quantseeker_mean_reversion_study.md) | forward_hypothesis | mean_reversion | 2026-10-02T15:28:59+03:00 | Most claims replicate as paper-like diagnostics, but almost all of the index-level reversal is earned from Close_T to Open_T+1. At the next open the IBS basket, SPY-TLT spread, intraday asymmetry, silver spreads and Gatev pairs fail; the multi-day IBS/MTSI percentile ensembles that pass the gates are timed equity beta with no alpha versus a beta-matched SPY since 2013. Only a true 15:45 IBS -> MOC trade on SPY/QQQ (signal from ES/NQ bars) keeps the edge (Sharpe 0.85 central 2016-06..2026-08, alpha vs SPY 3.9%/yr t 1.6) and is kept as a shadow forward hypothesis; it misses the frozen validation gate by 0.003 Sharpe. No combined MR sleeve is promoted. |
| [Quant Seeker TAA lineage (Defense First -> +BTAL -> VIX filter -> linearity -> inverse-vol -> fallback filter; stops, sector timing, macro overlays, futures): replication, fragility and book value](studies/quantseeker_taa_study.md) | diagnostic | defensive_tactical_allocation | 2026-10-02T14:26:02+00:00 | Reject V5 (QS final). Its published numbers are reproduced only on a calendar that drops BTAL's 338 no-trade days; on the true calendar CAGR/Sharpe are ~2.5 pp / ~0.25 lower (V5-SPY 9.5%/1.06 vs 11.9%/1.31 in QS's window). Executable V5-SPY 2012-10..2024-12: Sharpe 0.83 vs 60/40 0.80 and book DF 0.84; rebalance-day range 0.60-1.05; only inverse-vol weighting survives as a component; no add-on survives. |
| [eVRP + VIX term-structure + VIX-sizing volatility sleeve (Aziz/Zarattini, Concretum 2026-06-14): full-strategy replication as crisis hedge and diversifying sleeve](studies/evrp_dual_signal_vol_sleeve_study.md) | forward_hypothesis | volatility_risk_premium_short_vol_with_term_structure_switch | 2026-10-02T13:38:07+00:00 | NOT A CRISIS HEDGE; DIVERSIFYING SLEEVE CANDIDATE (forward hypothesis). Full eVRP + VIX/VIX3M + VIX-sizing strategy (short a -0.5x VIX-futures product ~92% of days, long VIXY ~3%, cash ~5%) on 2011-02..2026-09, Open_(T+1), 10 bps: CAGR 7.7%, Sharpe 0.63, MDD -32.8%, corr SPY 0.04. Positive in 2 of 8 post-2011 crisis windows (COVID +36.6%, Euro 2011 +4.0%); lost in Volmageddon (-1.0%), Q4 2018, China 2015, 2022 (-8.3%), yen carry (-3.0%) and tariffs 2025 (-2.3%). It sidestepped Volmageddon and April 2025 through the 'cash when premium positive but curve inverted' state, by a 0.28-point VIX/VIX3M margin in Feb 2018; the long-vol leg fired late in April 2025 and in March 2020 produced the worst days (-17.9%, -14.7%). Gates: hedge H1/H2 fail, H3 passes; sleeve D1-D3 pass. Adding 10% improves Sharpe in BOOK-A (1.23->1.29), BOOK-B (1.68->1.75) and BOOK-C (1.22->1.27) with shallower MDD, and keeps more return than 10% of the crisis pod; ex-COVID the improvement shrinks but stays (BOOK-A 1.30->1.34). Concretum notebook sizing (x2 on SVXY) matches the paper's yearly table best (corr 0.95; 11.7%/yr 2012-2025 vs paper 10.5%) and earns 11.9% CAGR, but with corr SPY 0.26 and -12% in 2022. Post-publication (Jun 2025-Sep 2026) +6.6%. Research only; no PAPER/LIVE/allocation. |
| [Moving Average Distance (MA21/MA200, Avramov-Kaplanski-Subrahmanyam) on US PIT universes](studies/moving_average_distance_study.md) | diagnostic | cross_sectional_momentum_trend | 2026-10-02T00:38:51+03:00 | REJECTED. MRAT reproduces in construction and works in 1991-2000, is flat 2001-2018, and fails out of sample 2019-2026 (primary FM t 0.67). The R1000 top-decile book is ~0.85x a momentum book: Sharpe 0.80 vs momentum 0.83 vs EW 0.66 OOS; alpha vs [EW, MOM] +0.3%/yr (t 0.13). Suddenness is weakly positive but concentrated in bubble names/years; forward hypothesis only. |
| [Concretum 'Build Your Own ETF Trend Portfolio' (Donchian 6/9/12m ensemble, inverse-vol global equity + inflation sleeves, SHV cash): replication and defensive-side audit](studies/concretum_etf_trend_study.md) | diagnostic | time_series_trend_donchian_ensemble_vol_scaled | 2026-10-02T00:00:00+03:00 | REPRODUCED; DEFENSIVE ALLOCATION, NOT A HEDGE; REDUNDANT WITH THE BOOK. Rules reproduce the article's 2026-09-03 orders share-for-share and its headline numbers (with a 100% effective leverage cap: 7.6%/6.6% vol/-9.0% DD vs 7.7/6.8/-8.8; monthly corr 0.99). 2008-2026 out of the article window the defense holds: avg -0.7% in SPY<=-2% months vs -3.1% AOR; 2008 +3%, 2022 +4%. Timing adds defense beyond lower exposure (Holm p 0.014, placebo 100th pct). But it loses like the market in fast shocks after calm markets (Feb 2018 -7..-11%, Aug 2024 -5%) because inverse-vol sizing peaks just before them; hit rate on worst 5% SPY days 33%. Pre-2016 excess Sharpe 0.34-0.40 vs 0.55 for 60/40 SPY/IEF. Correlation 0.66 with Defense First and 0.71 with BOOK-A; adding 20% lowers BOOK-A Sharpe 1.13 -> 1.10, while 20% crisis pod raises it to 1.16. |
| [Trend / breakout / momentum candidates not yet tested in Pakal: industry trend-breakout (Dow Award 2025), trend-smoothness double sort, two-factor rotation with trailing stops, recent-IPO all-time-high breakout](studies/trend_breakout_momentum_candidates_study.md) | forward_hypothesis | long_only_trend_breakout_and_cross_sectional_momentum_candidates | 2026-10-02T00:00:00+03:00 | NO CANDIDATE PASSES THE FROZEN 'REAL HIGH-SHARPE STRATEGY' GATE. (A) The Dow-Award industry trend-breakout replicates on French industries over a century (Sharpe 1.15 vs the paper's 1.39, CAGR 17.8%) but its edge sits in 1927-1989 (decade Sharpe 1.4-2.3 vs market 0.1-1.7); since 2000 it earns Sharpe 0.7 against a market at 0.8-0.9, and the executable sector-ETF version earns Sharpe 0.49 (0.68 in 2012-2026 vs SPY 0.94). (B) The trend-smoothness double sort beats plain 12-1 momentum by 3-4%/yr (t 1.7-2.4, Holm-significant only in the Russell 1000) but inherits the same -69% drawdown and Sharpe 0.6-0.8; it is an ingredient, not a strategy. (D) The two-factor low-vol + momentum rotation with 25% trailing stops reproduces its source: CAGR 11.3%, Sharpe 0.95 (0.89 / 1.01 by half), max DD -25% vs -55% for SPY, alpha 5.7%/yr (t 4.1) at beta 0.48; it misses the Sharpe 1.0 gate by 0.05 and does not beat SPY by 0.25 after 2012, but it is the only candidate that lifts the live book (G3 1.30 -> 1.35 at a 30% mix). (E) The recent-IPO all-time-high breakout earns Sharpe 0.62 with a -39% drawdown and lost 9%/yr after publication. Bottom line: the high-Sharpe, high-CAGR, cross-sectional strategy the owner asked for does not exist among these four; D is a sound low-drawdown equity sleeve worth a 12-month shadow log, nothing more. ROBUSTNESS (SPEC v2, 230 runs): D passes all six frozen robustness gates (parameter grid Sharpe 0.86-0.98, rank-noise median 0.90, lag/MOC 0.96, 40 bps 0.93, four universes 0.83-1.06, bootstrap P(Sharpe>SPY) 0.99 full). Random stock picks with the same stops/filter/sizing already earn 0.79; ranking adds ~0.15 Sharpe (beats 99% of random runs). Versus SPY the edge is a bear-market story (bootstrap 0.67 since 2012; beats SPY in 52% of years). Robust but modest: true Sharpe ~0.9. |
| [Price-path convexity: factor spanning, A/B decomposition and post-paper test (Aligrithm 10.14 / Gulen-Woeppel)](studies/ppc_spanning_oos.md) | diagnostic | price_path_convexity_short_horizon_reversal | 2026-10-01T21:32:07+00:00 | The headline spread reproduces (R3000 equal-weight 0.87%/month vs paper 0.84%), standard factors including 1-month reversal do not explain it (alpha 0.79%, t 4.0), and both halves of the shape matter in the broad universe. But the per-sd slope is -0.26% not -0.45%, it halves after the last-day return, it is mostly a small-cap effect (liquidity-weighted and large-cap versions are weak and negative since 2023), it has been fading since 2013, and after central costs it earns 0.42%/month with Sharpe 0.44 over 1993-2026 and about zero since 2023. Convexity is a recombination of 'distance from the 1-month average' and the 1-month return (MA-distance + reversal hedge -> alpha 0.18%). Not tradable; keep as a diagnostic feature. |

## Largest research families

| Family | Studies |
| --- | ---: |
| Mean Reversion | 22 |
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
| Defensive Tactical Allocation | 2 |
| Price Path Convexity Short Horizon Reversal | 2 |
| Cross Sectional Path Shape Reversal | 2 |
| Momentum Rotation | 2 |
| Leveraged Etf Hedged Short | 2 |
| Cross Asset Momentum Robust Covariance Minvar | 1 |
| Dual Momentum | 1 |
| Cross Market Sentiment And Short Horizon Equity Reversal | 1 |
| Valuation Yield Curve Regime Router | 1 |
| Cross Market Sentiment | 1 |
| Cross Sectional Momentum | 1 |
| Time Series Trend Donchian Ensemble Vol Scaled | 1 |
| Asymmetric Cross Asset Time Series Momentum Tail Hedge | 1 |
| Intraday Vwap Drift Continuation | 1 |
| Cross Asset Dual Momentum | 1 |
| Cross Sectional Intraday Reversal | 1 |
| Calendar Mean Reversion | 1 |
| Calendar Rebalancing Flows | 1 |
| Term Structure Inversion Stateful Long Vixy Tail Hedge | 1 |
| Volatility Risk Premium Short Vol With Term Structure Switch | 1 |
| Regime Filter Serial Correlation Median Reversion | 1 |
| Calendar Conditional Safe Haven Flow | 1 |
| Cross Asset Absolute Momentum Vol Capped | 1 |
| Short Term Atr Mean Reversion | 1 |
| Risk Overlay | 1 |
| Macro Regime Growth Classifier | 1 |
| Macro Regime Portfolio Construction | 1 |
| Cross Sectional Mean Reversion | 1 |
| Cross Sectional Low Volatility | 1 |
| Cross Asset Calendar Reversal | 1 |
| Market Regime Allocation | 1 |
| Cross Asset Momentum Mean Variance Allocation | 1 |
| Cross Sectional Momentum With Orthogonal Companions | 1 |
| Cross Sectional Momentum Trend | 1 |
| Index Daily Mean Reversion Variance Ratio | 1 |
| Cross Asset Pca Risk Regime | 1 |
| Tail Risk Overlay | 1 |
| Factor Momentum | 1 |
| Closing Auction Basis | 1 |
| Short Horizon Cross Sectional Reversal | 1 |
| Momentum And Trend Overlays | 1 |
| Volatility Term Structure Timing And Crisis Hedging | 1 |
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
| Cross Sectional Momentum Vol Targeted | 1 |
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
| Long Only Trend Breakout And Cross Sectional Momentum Candidates | 1 |
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
| Etf Rotation Momentum Mvo And Put Writing | 1 |
