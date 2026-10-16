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

- :material-flask-outline: **165 published studies**
- :material-clipboard-alert-outline: **3 records need audit**
- :material-tag-multiple-outline: **118 signal families**
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
| forward_hypothesis | 36 | Frozen idea awaiting genuinely new evidence. |
| diagnostic | 121 | Useful evidence or state variable; not a strategy recommendation. |

## Most recently reviewed

| Study | Status | Family | Reviewed | Verdict |
| --- | --- | --- | --- | --- |
| [Valuelytica 'The Four-Hour FX Trade' (long foreign currencies vs USD 11:00-15:00 New York) and 'YenStrike' (weekday JPY rule): 15 currencies, 2008-2026 1-minute data, London-fix mechanism, measured spreads](studies/valuelytica_fx_us_hours_study.md) | diagnostic | calendar_intraday_fx_time_of_day | 2026-10-16T17:05:00+03:00 | Both posts reproduce on 2021-2025 (GBP 0.30, MXN 1.11 vs 1.08, JPY -0.67; YenStrike Sharpe 1.30 vs 1.23). The US-hours drift against the dollar is real but tiny: 15-currency basket +0.93 bp/day in 2008-2020 (t 2.3), +0.26 bp in 2021-2025, -1.36 bp in 2026; central and measured costs exceed it in every period after 2020. YenStrike earns zero in 2008-2020 (Sharpe 0.01) and 2026; walk-forward signs earn nothing. Wednesday effect rejected. Mechanism: dollar bought into the London 4pm WM/R fix (11:00 NY) and released after it (pre-fix hour t -2.1). Rejected; diagnostic only. |
| [Bond calendar flow family: Valuelytica ZROZ month-end effect and every comparable bond calendar effect (27 ETFs, 1962-2026 yields, auctions, FOMC, NFP, coupons)](studies/bond_calendar_flow_family_study.md) | forward_hypothesis | calendar_bond_index_flow | 2026-10-16T12:00:00+03:00 | Source replicated (ZROZ EOM 9.9%/0.81 vs claim 9.46%/0.81; EOMS 16.2%/0.91 vs 17.3%/1.01; central cost 0.63/0.66). Month-end bond bid is real and index-driven: absent in 1962-84 yields, present in every era since 1985, larger in refunding months, scales with duration across 27 ETFs (corr 0.95). ZROZ is not a better vehicle (same edge per duration; IEI/IEF/TLT higher Sharpe, far more liquid). Frozen composite gate passed no window. FOMC window, pre-auction long-end short (post-2013 regime) and refunding-month EOM are forward hypotheses for shadow logging; NFP premium, post-auction relief, mid-month coupon and quarter-end amplification rejected. 2026 YTD EOM on ZROZ -8%. |
| [NDX dip ladder into TQQQ: 100% S&P, 2% into 3x Nasdaq at -5% and each further -3%, back to 100% S&P at the old NDX peak](studies/ndx_dip_ladder_tqqq_study.md) | diagnostic | drawdown_ladder_leveraged_dip_buying | 2026-10-16T06:38:22+00:00 | Rejected. 1988-2026: B0 12.3%/yr, Sharpe 0.50, max DD -70% vs SPY 11.4%, 0.52, -55%. The extra return is leverage (average 7% TQQQ, ~1.2x): SPY levered to B0's risk earns 12.7% and constant 7% TQQQ without timing 12.9%/0.54/-64%. H1-H3 all fail (Holm). 56 of 57 dip episodes win small; the 2000-2015 episode (NDX -83%, 26 tranches, 15.6 years held) loses 54% of the book vs SPY. Wins 1988-99 and 2010-26, loses 2000-09; 10-year windows starting 1990-99 trail SPY 89% of the time. 0/36 grid cells beat SPY's Sharpe in all three eras. |
| [Learning to rank for cross-sectional momentum (Poh-Lim-Zohren-Roberts 2020 / Aligrithm 10.23) on US PIT equities](studies/ltr_momentum_study.md) | diagnostic | cross_sectional_momentum_machine_learning_ranking | 2026-10-14T19:30:00+00:00 | Not replicated. On Russell 3000 1996-2020 (Open->Open, no costs) LambdaMART gets Sharpe 0.99 vs the paper's 2.16, ListNet and rank:ndcg are negative, and the regress-then-rank MLP the paper reports at 0.27 gets 1.34. With identical trees/features/labels, squared error beats the pairwise ranking loss by 0.81 Sharpe (SE 0.20): the paper's central mechanism reverses. The working models learn momentum plus a strong low-volatility tilt; top-weighted list losses learn volatility (who lands in the extreme decile) and lose money while winning NDCG. LambdaMART's 20 hyperparameter trials span -1.2..+2.3 Sharpe within a block; the no-selection ensemble is 0.67 before and 0.00 after 2020. After costs and T-bill financing of the net-long dollars that inverse-vol weights create, no ML model beats a two-line rank(12m) - rank(vol) rule in tradable universes, and neither beats plain 12-month momentum by more than random-book noise after 2021 (financed net Sharpe 0.31-0.50 vs JT 0.33-0.37). |
| [Whole-book picture: true G3 core plus every paper/shadow sleeve, marginal effects and an illustrative ladder](studies/book_picture_20261014.md) | diagnostic | portfolio_construction_book_level | 2026-10-14T19:01:27+00:00 | G3 alone 2012-10..2026-07: 20.0%/yr, excess Sharpe 1.18 (rf0 1.30), max DD -14.1%; its two legs carry ~100% of risk. EOM C_main is the only sleeve that adds in both eras and ex-COVID (+0.07 Sharpe at 15%, DD to -11.9%). The crisis pod (5+5%) is insurance (COVID -5% -> 0%) with ~0 Sharpe effect ex-COVID. eVRP's +0.04 is mostly March 2020 (+0.01 ex). Equity sleeves (NDX tilt, TBMC-05, CORE5, Connors, IBS) add <= 0.02 by diluting vol and lose with the book in crises. Illustrative B3 (65 G3 / 15 EOM / 5 core / 5 VIXM / 10 eVRP): 15.5%/yr, excess Sharpe 1.32 (rf0 1.49), DD -9.1%; 21.9% at G3's vol with 1.5x financing. Still ~95% of risk in the G3 legs. |
| [RangerZ (Pardo Ranger on Zorro) range-breakout strategy generator: public-rule rebuild and Evaluation-Shell selection-bias audit](studies/rangerz_breakout_generator_study.md) | diagnostic | range_breakout_strategy_generator_and_walk_forward_selection | 2026-10-12T21:12:28+00:00 | The public Ranger architecture (3,744 job-market cells) plus a mechanical Evaluation Shell selected 11 algos with discovery Sharpe 2.50; on the locked 2022-2026 holdout they lost money (Sharpe -0.22, 19th percentile of random 11-job picks, below classic Donchian 0.32, generator average 0.28 and buy-and-hold 0.95). The same pipeline on 30 bootstrapped random-price worlds gives discovery Sharpe 1.8-4.4 (p 0.90); the shell's own Montecarlo check still reports p 0.01. RangerZ's 35-350% 'annual return' is profit / (normalized drawdown + margin) on 1-lot CFDs. |
| [CORE5 equity sleeve on QQQ / UPRO / TQQQ instead of SPY: own vs SPY signal, synthetic 3x through 2008, equal-risk whole-book leverage, 2000-2026 proxy book](studies/core5_equity_vehicle_v2.md) | forward_hypothesis | adaptive_trend_macro_sleeves_vehicle_swap | 2026-10-11T00:00:00+03:00 | QQQ with its own QQQ signal passes the frozen rule (Sharpe 1.02 vs 0.95, CAGR 8.0% vs 7.2%, DD -7.9% vs -6.8%; not significant after Holm) and should replace the August QQQ/SPY-signal forward line, which fails the 2000-02 check. UPRO with the SPY signal passes as a leverage dial (11.1%, Sharpe 0.99, DD -14.1%; beats whole-book leverage at equal vol, survives 2000-02 at -7%). TQQQ (13-14% CAGR 2007-26) is downgraded: the 2000-02 proxy book loses 19-31% vs -6% for equal-risk whole-book leverage. LETF-price signals are worse than underlying signals. Ladder (post hoc): S&P Sharpe flat 2x-3x; Nasdaq peaks at 1.5-2x, so QLD beats TQQQ. |
| [Antonacci (Concretum interview 2026-10): GEM replication and public-ingredient reconstructions of SMURF / GLTR / BADASS / FIRM / Snap Back](studies/antonacci_models_study.md) | diagnostic | cross_asset_dual_momentum_and_channel_breakout_with_reversal_overlays | 2026-10-10T00:00:00+03:00 | GEM reproduces on proxies (1991-2026 CAGR 12.7%, excess Sharpe 0.82, MDD -21.4%); post-publication 2014-11..2026-08 it trails the S&P 500 (Sharpe 0.56 vs 0.82, bootstrap P(GEM>SP)=0.02). The proprietary claims (SMURF Sharpe 1.52, GLTR 2.03, BADASS 2.01) are not reachable with the described public ingredients: best grid cells 0.67 / 0.90 / 0.88; primaries 0.62 / 0.77. 50/30/20 of the primaries: Sharpe 0.81 vs site 2.35. Snap Back primary win rate 51.2% vs claimed 70-80%. |
| [Mega-cap concentration as a beta boost: biggest-N S&P 500 / Nasdaq-100 books with trend gates and leverage against the S&P at matched beta, plus a century of CRSP size deciles](studies/megacap_concentration_beta_boost.md) | diagnostic | size_concentration_beta_overlay | 2026-10-10T00:00:00+03:00 | REJECTED AS ALPHA, ACCEPTED AS BETA. Against the S&P 500 held at the same on-day beta, the biggest-10 S&P 500 book adds 1.9 pp/yr (t 1.0, Holm p 0.80); no primary member passes the frozen rule. The gain is a 2009-2026 regime (biggest-10 gated: -0.7 pp/yr vs control in 1991-2008, +4.5 pp/yr in 2009-2026); over 1926-2026 the largest CRSP decile trailed the market by 0.4 pp/yr and concentration is at a century high. A levered gated S&P (2x + SMA200: 16.3%/yr, DD -48%) delivers the boost with less risk than the 2x gated biggest-10 (20.7%/yr, DD -72%, beta 2.7). Regime timing adds nothing (Holm p 1). Forward hypothesis only: a 3-year turnover size proxy (t 2.7) and the Nasdaq-100 universe. |
| [Levered gated Nasdaq-100 with a half-sized mega-cap tilt (biggest 10 by 3-year turnover, CORE5 per stock, SMA200 gate, 1.5x): pre-registered sleeve with walk-forward](studies/ndx_megacap_tilt_sleeve.md) | forward_hypothesis | levered_trend_gated_index_with_size_tilt | 2026-10-10T00:00:00+03:00 | PROMOTED TO PAPER SHADOW by the pre-registered rule. Sleeve 1994-2026: CAGR 23.7%, Sharpe 0.79, max DD -43% vs the levered gated index alone 21.7% / 0.72 / -51%. Sharpe gain bootstrap P 0.95; CAGR difference t 1.3 (Holm p 0.35, not significant). The tilt did not hurt in 1995-2008 (Sharpe 0.71 vs 0.68); walk-forward OOS Sharpe 0.68 vs index leg 0.62. Guardrails (weekly exit, drawdown de-lever) rejected. Leverage is a clean dial (Sharpe flat 0.79-0.80 from 1x to 2x). |

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
| Cross Asset Dual Momentum And Channel Breakout With Reversal Overlays | 1 |
| Dual Momentum | 1 |
| Cross Market Sentiment And Short Horizon Equity Reversal | 1 |
| Calendar Bond Index Flow | 1 |
| Portfolio Construction Book Level | 1 |
| Valuation Yield Curve Regime Router | 1 |
| Cross Market Sentiment | 1 |
| Cross Sectional Momentum | 1 |
| Time Series Trend Donchian Ensemble Vol Scaled | 1 |
| Adaptive Trend Macro Sleeves Vehicle Swap | 1 |
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
| Cross Sectional Momentum Machine Learning Ranking | 1 |
| Cross Asset Calendar Reversal | 1 |
| Market Regime Allocation | 1 |
| Size Concentration Beta Overlay | 1 |
| Cross Asset Momentum Mean Variance Allocation | 1 |
| Cross Sectional Momentum With Orthogonal Companions | 1 |
| Cross Sectional Momentum Trend | 1 |
| Drawdown Ladder Leveraged Dip Buying | 1 |
| Levered Trend Gated Index With Size Tilt | 1 |
| Index Daily Mean Reversion Variance Ratio | 1 |
| Long Only Momentum Trend And Tactical Asset Allocation | 1 |
| Cross Asset Pca Risk Regime | 1 |
| Tail Risk Overlay | 1 |
| Factor Momentum | 1 |
| Closing Auction Basis | 1 |
| Short Horizon Cross Sectional Reversal | 1 |
| Momentum And Trend Overlays | 1 |
| Volatility Term Structure Timing And Crisis Hedging | 1 |
| Cross Sectional Momentum Rotation | 1 |
| Cross Asset Momentum Rotation | 1 |
| Cross Sectional Momentum Long Only Rotation | 1 |
| Range Breakout Strategy Generator And Walk Forward Selection | 1 |
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
| Calendar Intraday Fx Time Of Day | 1 |
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
