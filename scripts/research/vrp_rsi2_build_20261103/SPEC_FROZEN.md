# VRP RSI2 build: frozen rule, checks and decision rule (2026-11-03)

Research only. Nothing here touches `alpha/live`, a broker route, a schedule or capital. Live wiring is a separate
owner decision.

## What was already seen

Everything below was chosen after seeing 2011-2026 results, so this is a retro registration:

- The parent study `evrp_dual_signal_vol_sleeve_retro_20261017` (Concretum / SSRN 5316487, ledger rows 310-311).
- Antigravity's scout of 2026-11-01 and its clean rebuild `vrp_multipod_scout_retro_20261102` (ledger row 588):
  36 short-vol cells, the RSI2 boost, 27 macro-collateral cells, a 6-point threshold grid, 150 random baskets.
- The owner's lock of 2026-11-03 (`docs/research/VRP_RSI2_LOCKED_SPEC_AND_GROWTH_CONSTRUCTION_20261103.md`):
  RSI2 boost kept, VIX/VIX3M entry threshold 0.98, idle cash in T-bills, no gold, no macro basket.
- Claude's reviews of 2026-11-01 to 2026-11-03: RSI2 placebo (0 of 400 shifts reach the rule), threshold grid
  0.90-1.05, VX futures 2006-2011, fund-harness book tests (10 points from CORE5).

The row-588 registration froze tau = 1.00 and no RSI2. The locked form differs on both, so it is a new registration
(`vrp_rsi2_locked_retro_20261103`, parent row 588).

## The rule (decided after the close of T, filled at the open of T+1)

Inputs, all official closes stamped T: `$VIX`, `$VIX3M`, `$SPX` (price index), SPY total-return close.

    ERV10_T = sample std (ddof 1) of the last 10 SPY total-return close-to-close returns x sqrt(252) x 100
    eVRP_T  = VIX_T - ERV10_T
    TS_T    = VIX_T / VIX3M_T
    RSI2_T  = Wilder RSI(2) of the $SPX close (exponential smoothing, alpha 1/2)

    R1  TS < 0.98 and eVRP > 0:   SVXY = min(2.0 VIX/100, 1); x1.25 (cap 1) when RSI2 < 25
    R2  TS < 0.98 and eVRP <= 0:  SVXY = min(1.0 VIX/100, 1); min(1.5 VIX/100, 1) when RSI2 < 20
    R3  0.98 <= TS <= 1.00:       cash
    R4  TS > 1.00 and eVRP <= 0:  VIXM = min(1.25 VIX/100, 1)
    R5  TS > 1.00 and eVRP > 0:   cash

Before 2018-02-28 SVXY was a -1x product, so its target is halved for decision dates before that day (same short-VIX
exposure across the history). Live trading has no halving.

Band: a decision trades when a target turns on or off, or when either target moved 5 points or more since the last
traded target. Otherwise nothing is traded and positions drift. The band compares targets with targets, so the
schedule depends on the signals only.

## Execution model (the engine module)

- Module: `strategies/vix_stuff/strategy_vrp_svxy_vixm_rsi2.py`.
- Shares: trunc(prior-close NAV x target / Close_T) for both assets on a trading day. Fill at Open(T+1).
- Costs: 5 bp slippage per side, IBKR fixed commission (USD 0.005 a share, USD 1 minimum). Idle cash earns 0% in the
  engine, as every house pod; books add fair cash separately.
- Missing input: fails loud. No forward fill beyond Norgate's market-day padding (the last close carried on a session an
  index skipped).

Scout spec: `alpha/scout/specs/vrp_rsi2.py`, the same rule through the Scout weights engine. Gate: exact tier
against the engine (`python -m alpha.scout gate vrp_rsi2 --fresh`).

## Checks written before the engine run

1. Engine = Scout spec at the exact tier.
2. Engine vs the independent research replica (Claude's weight simulator of 2026-11-03): daily return correlation of
   at least 0.99 from 2012 on; Sharpe within 0.03. The replica holds weights constant between trades (free
   rebalancing); the engine lets them drift and trades whole shares.
3. Prefix causality: the targets at T do not change when data after T are removed.
4. One extra session of delay, VIXM leg costs (+15 and +40 bp a side), cost sweep, threshold neighbours.

## Adoption rule (fixed now, judged later)

The pod enters a fund book only when all hold:

1. 10 points come from CORE5, not pro rata (pro rata breaks the return floors).
2. Measured fill cost (paper or small live) at or below 15 bp a side on SVXY and VIXM.
3. Twelve months of shadow without a RED card (Cold Blood Index or monthly CUSUM).
4. An exit failsafe exists: a missed or cancelled exit order is retried the same session (IBKR cancelled two MOO
   orders on 2026-11-02).
5. An accountant has confirmed the K-1 treatment for the fund and its clients.

Plan Sharpe for the pod: about 0.70 (three quarters of the backtest edge), not 0.93.

## Shadow

S8 shadow of the frozen rule from 2026-11-04, provider `alpha.scout.specs.vrp_rsi2:shadow_daily_return_ser`,
expected process = the engine-equivalent Scout series before the start, benchmark Sharpe 0.
