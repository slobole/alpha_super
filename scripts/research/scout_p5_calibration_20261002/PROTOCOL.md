# Scout P5 calibration: the MCPT score for timing and volatility-sized ETF families

Written 2026-10-02 before any result below was computed.

**Question.** A5 left this open: the S5 MCPT score for families that size or gate by volatility.
- The P2 review showed that raw Sharpe scoring passes volatility-sized families with drift and no signal 38% of
  the time, because the date shuffle destroys volatility clustering, which any volatility-timed rule exploits.
- Volatility timing is a real effect, but a generic one. It is not the edge such a family claims (momentum, trend,
  a regime filter).
- So the score must net out a baseline that captures the generic effects: drift and volatility timing.

**Panel.** 6 ETF-like assets, 3,780 sessions (15 years).
- **Returns:** r_i = 0.0003 + β_i × f + e_i.
  - f: a GARCH(1,1) Student-t factor, 16% a year.
  - β: (0.3, 0.5, 0.7, 0.9, 1.1) for assets 1-5, and 3.0 for asset 6 (a leveraged-like fallback).
  - e_i: GARCH idiosyncratic, 10% a year.
- **Implied-volatility proxy V_t:** 1.15 × the factor's conditional volatility × exp(N(0, 0.1²)), annualised in
  percent. It is an exogenous column that moves with its date in the shuffle.
- **No return predictability in the null.** Planted edge for power: asset i's daily drift gains 0.05 × its own
  mean return over the last 126 sessions, lagged one session (time-series momentum).

**Families.** Plateau selection; decisions at the close of month-end T, held from the close of T+1; gross.
- **F1, TAA-like:**
  - **Rule:** score the assets 1-5 by the mean of their k-month returns; ranks get weights (5, 4, 3, 2, 1)/15 if
    the score is > 0, otherwise that slot goes to asset 6.
  - **Gate:** asset 6 is held only while the realised volatility of f over W sessions (annualised %) is below V_T;
    otherwise that weight sits in cash.
  - **Grid:** k-set ∈ {(1,3), (1,3,6), (1,3,6,12)} × W ∈ {10, 20, 40}: 9 configurations.
- **F2, volatility-scaled trend on asset 6:**
  - **Rule:** hold asset 6 when its price is above SMA(L), at weight min(1, max(0.25, R / V_T)).
  - **Grid:** L ∈ {100, 150, 200, 250} × R ∈ {16, 20, 24}: 12 configurations.

**Scores.**
- **SA:** Sharpe(selected) − Sharpe(EW buy-and-hold of the family's traded assets). This is A5 as written.
- **SB:** Sharpe(selected) − Sharpe(volatility-targeted EW): the EW portfolio scaled daily by
  min(1, 10% / its trailing 20-session realised volatility), lagged one session.
- **SC:** the Sharpe of the daily active return, selected − EW buy-and-hold.
- **SD:** the Sharpe of the daily active return, selected − volatility-targeted EW.

**Null:** plain date-row shuffle of the 7-column matrix (6 returns and V), 200 permutations.

**Runs:** 200 null seeds and 100 planted-edge seeds per family.

**Decision:**
- **Size:** a score qualifies if its false-pass rate at p ≤ 0.05 is ≤ 7.0% for BOTH families.
- **Choice:** among qualifying scores, the one with the highest mean power over the two families becomes the S5
  score for ETF timing families.
- **If none qualifies:** ETF timing families cannot pass S5 (WATCHLIST), and that is reported.
- **Bound:** 7.0% is 5% plus about 1.4 binomial standard errors at 200 seeds. It is stricter than P4b's 8%,
  because these families run on a plain shuffle, which is exact up to the volatility structure.
