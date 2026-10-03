# Adaptive stress measures for the DV2 gate (frozen 2026-10-03, before any run of these measures)

Owner (2026-10-03): test the CSS Analytics (D. Varadi) adaptive ideas for robustness:
- "Adaptive VIX moving average with Ehlers alpha" (2019-12-04): D = percentrank(1 / SMA10(VIX), 2000 days); the post
  notes buying on down days is far more profitable when this measure says VIX is high.
- "Adaptive volatility" (2017-11-15): EMA of squared returns with SC = min(exp(-10 x (1 - R^2(price vs time, 20))), 0.5).
- "Adaptive volatility: a robustness test" (2017-11-29): judge a method by its whole parameter grid, not one point.
Only the parts that measure market stress are used here. The SPY trend rule and risk parity are not.

## Reference and controls
Reference = C3 (VIX > 20 opens the gate, stays open >= 10 sessions; MR_GATE_DEFINITION_20261003). Controls: ANY_OFF,
ungated DV2, T-bills. Setup identical to the gate study (DV2 wired replica, next-open entries, exits never gated,
idle cash at BIL, engine costs and +5 bps, book {TAA .5, L .25, X .25}, blocks G-P1/2/3, G-FULL, G-LONG).

## Measures (known at close t)
- PR: VIX percentile. D_t = percentrank of 1 / SMA10(VIX) within the last 2000 sessions (min 1000; before 2000 rows
  the window is what exists). Stress = D_t < d (VIX high versus its own ~8-year history).
- AV: adaptive S&P 500 volatility. R^2_t = squared correlation of log(SPX) with time over 20 sessions;
  SC_t = min(exp(-10 (1 - R^2_t)), 0.5); v_t = SC_t r_t^2 + (1 - SC_t) v_{t-1}; AV_t = sqrt(252 v_t).
  Stress = AV_t > a.
- RV: plain 20-day realized volatility of the S&P 500 (the control the AV post beats). Stress = RV_t > a.

## Variants (all reported)
- PR, d in {0.3, 0.4, 0.5}, without memory and with 10-session memory.
- AV, a in {14%, 16%, 18%, 20%}, without memory and with 10-session memory.
- RV, a in {14%, 16%, 18%, 20%}, with 10-session memory (same-footing control for AV).
- VOTE: open if >= 2 of {VIX > 20, PR d = 0.4, AV a = 16%}, with 10-session memory.

## Decision rule
A variant replaces C3 only if ALL hold:
1. book Sharpe >= C3 in G-FULL and G-LONG, and >= C3 - 0.03 in each of G-P1, G-P2, G-P3;
2. the same at +5 bps;
3. every grid neighbour of its family (same memory setting) is >= C3 - 0.03 in G-FULL;
4. DV2 standalone 1991–1999 (locked holdout) >= C3 - 0.05.
Otherwise C3 stays. Also reported: does AV beat RV at every threshold (the post's claim), switches per year, open share.

## Amendment 1 (before any run)
PR needs >= 1000 VIX sessions, so it is undefined before ~1994. The locked holdout for rule 4 is therefore
1995-01-03 -> 1999-12-31 for every variant, C3 included.
Memory semantics are C3's:
- the gate opens on the first close where the condition holds;
- it closes on the first close where the condition fails, provided >= 10 sessions have passed since it opened.
