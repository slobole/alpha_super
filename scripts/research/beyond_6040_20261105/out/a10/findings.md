# A10 — Beyond 60/40 for an ILS investor (1993-02 → 2022-12, dev sample only, nominal; fixed engine 2026-11-05)

| Policy | CAGR | Vol | Excess Sharpe* | Max DD | Worst 12m | Avg hedge | FX leg /yr |
|---|---|---|---|---|---|---|---|
| USD investor (B0, ref) | 7.6% | 7.3% | 0.71 | -16.2% | -13.0% | – | 0 |
| ILS unhedged | 8.4% | 9.3% | 0.33 | -17.5% | -15.8% | 0 | +1.0% |
| ILS full hedge (carry = IL − FF) | 10.7% | 7.3% | 0.70 | -17.0% | -13.9% | 1.0 | +3.0% |
| ILS static 50% hedge | 9.6% | 7.5% | 0.54 | -12.6% | -10.5% | 0.5 | +2.0% |
| ILS active binary (from 1998-02) | 6.4% | 8.7% | 0.34 | -18.6% | -15.0% | 0.24 | -0.7% |
| ILS active 0.5-band (from 1998-02) | 8.4% | 7.7% | 0.60 | -12.6% | -10.5% | 0.50 | +1.0% |
*Excess over the investor's own cash: Fed funds for USD, Israel short rate (5.5%/yr avg 1993-2022, 1.6% 2005-2022) for ILS.

1. Full hedging gives the ILS investor the same excess Sharpe as the USD investor (0.70 vs 0.71), as it must; it cost nothing on average because IL rates exceeded US rates in 72% of months (carry received +3.0%/yr 1993-2022, +0.3%/yr 2005-2022). 2005-2022: hedged 7.4% CAGR / Sharpe 0.78 vs unhedged 6.1% / 0.50.
2. Unhedged, the shekel's appreciation ate the strategy: USD/ILS fell 26% in 2002-06→2008-07 (strategy +85% in USD, +34% in ILS, +112% hedged) and 17% in 2012-01→2022-01 (+90% USD, +58% ILS unhedged, +86% hedged). Excess Sharpe halves (0.71 → 0.33); the bootstrap gap hedged−unhedged is +0.36 (90% band +0.12 to +0.61), 2005-2022 +0.25 (−0.10 to +0.60).
3. The unhedged investor's worst drawdowns are FX drawdowns, not strategy drawdowns: 2007-10→2008-07 -17.5% while USD/ILS fell 19% and the USD strategy rose 3%; 2012-08→2013-12 -17.1% with USD/ILS -13%; 2020-08→2022-08 -12.1% (USD -4%, not recovered by 2022-12); 2015 -11.6% (USD -6%). Underwater up to 2.4 years vs 1.4 hedged.
4. The dollar is a crisis asset for an Israeli: monthly corr(strategy, USD/ILS) = -0.24; in the strategy's worst-decile months USD/ILS rose 1.2% on average (up in 64% of them). That is why the 50% hedge has the shallowest drawdown (-12.6%, worst 12m -10.5%) in every window, at a cost of ~1.1% CAGR vs full hedge; its Sharpe gain over unhedged is +0.21 (90% band +0.10 to +0.32).
5. The author's active trend hedge fails here. The expanding OLS of next-month USD/ILS on the 1/3/6/12m z-score trend signal has no forecast power (beta ≈ 0, sign hit rate 46%, corr(pred, realised) -0.07), so the rule collapses to "hedge when carry > the historical USD drift". It hedged 24% of months, mostly 1998-2003 and 2008-2013, and was unhedged through almost all of 2005-2008 and 2014-2022, exactly when the shekel rose. Result: worst of all policies (Sharpe 0.34, DD -18.6%, FX leg -0.7%/yr).
6. The 0.5-band variant degenerates: |expected value| never exceeds 0.41 × residual sd (median 0.08), so every month is in the band and it is the static 50% hedge under another name.
7. By decade (excess Sharpe unhedged / hedged / 50%): 1993-99 -0.05 / 0.37 / 0.13 (IL cash 14.7%); 2000s 0.23 / 0.73 / 0.49; 2010s 0.81 / 1.12 / 1.04; 2020-22 0.16 / 0.02 / 0.11 (the only block where unhedged won: USD +13% in 2022 while the strategy lost 16%).

Practical answer for a fund reporting in ILS: hedge the NAV. A full passive hedge reproduces the USD track (Sharpe 0.70, excess CAGR +5.2% over Israeli cash) and has historically been paid for by the carry; the active trend-hedge rule has no edge and should not be used. If the mandate values drawdown over return, a static 50% hedge is the robust middle: lowest max DD and worst-12m in every window, ~1.1% CAGR behind full hedge, and it keeps half the dollar's crisis cushion. Rebalance the hedge monthly with NAV; a daily full-notional hedge was assumed here and the cross term (r_usd × fx) is immaterial. Caveats: forward bid-ask and roll costs are not charged (ILS 1m forwards roughly 5-10 bp/yr for a fund); the IL rate is the OECD monthly series lagged one month; 1993-2002 carried 7-10% IL-US differentials that will not repeat, so 2005-2022 (carry ≈ 0) is the relevant window for forward planning, and there hedging still added +1.3% CAGR and +0.27 Sharpe.

Files: policies_1993_2022.csv, policies_2005_2022.csv, policies_common_window.csv, policies_by_decade.csv, drawdown_episodes.csv, shekel_runs.csv, fx_by_year.csv, active_decisions_*.csv, active_hit_rate.csv, diagnostics.json.
