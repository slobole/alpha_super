# Russell 1000 momentum leg with a liquidity filter (2026-09-26)

Research only. Frozen plan: [R1000_LIQUID_MOMENTUM_PREREG_20260926.md](R1000_LIQUID_MOMENTUM_PREREG_20260926.md)
(SHA-256 `05a9c3b6...`, frozen 23:16 before any filtered run). Code: `scripts/research/run_r1000_liquid_momentum_check.py`
(uses the NDX-study replica, parity gate re-passed). Numbers: `results/research/ndx_param_robustness_20260926/r1000_liquid_check.json`.

## Verdict: keep L (the NDX leg)

- **W-REL25 fails R1 by a small margin.** Its G3 Sharpe margin versus L is +0.17 in 2008-11, **-0.04** in 2012-21
  and +0.03 in 2022-26. W-REL25 is the frozen walk-forward Russell 1000 configuration with the least liquid 25% of
  members removed.
- **R2 (drawdown) passes easily.** G3 max DD is -11.8% against L's -14.1% on 2012-26.
- **R3 fails.** The same pattern holds with +5 bps per side.
- **R5 (tradability) passes.** In 2021-26 the largest order reaches 5% of ADV20 only at a $9.6M pod.
- **In short, it is a tie in Sharpe with less drawdown and less return.** G3 2012-26 makes 16.0%/yr at Sharpe 1.28,
  against L's 20.0% at 1.30. The paired bootstrap gives a difference of -0.02, 90% interval [-0.20, +0.15], p = 0.56.

## Results (engine costs; G3 = 0.5 TAA + 0.5 leg)

| Leg | Standalone 2000-26 CAGR / Sharpe / MaxDD | G3 Sharpe 08-11 / 12-21 / 22-26 | G3 2012-26 CAGR / Sharpe / MaxDD | Turnover | Pod $ at which the largest order = 1% / 5% of ADV20 (2021-26) |
|---|---|---|---|---|---|
| **NDX L (live)** | 11.9% / 0.76 / -29.7% | 0.70 / 1.30 / 1.29 | 20.0% / 1.30 / -14.1% | 7.9x | $3.6M / $18.1M |
| W unfiltered | 7.0% / 0.82 / -16.7% | 0.92 / 1.26 / 1.35 | 15.7% / 1.29 / -11.8% | 8.5x | $0.3M / $1.3M |
| **W-REL25 (primary)** | 7.2% / 0.77 / -21.4% | 0.87 / 1.27 / 1.31 | 16.0% / 1.28 / -11.8% | 8.8x | $1.9M / $9.6M |
| W-REL50 | 7.8% / 0.74 / -19.6% | 0.90 / 1.24 / 1.21 | 16.2% / 1.23 / -11.8% | 8.7x | $4.6M / $23.2M |
| W-ABS5M | 7.0% / 0.81 / -16.7% | 0.91 / 1.26 / 1.35 | 15.6% / 1.28 / -11.8% | 8.5x | $1.3M / $6.6M |
| A0-R (NDX design on R1000), REL25 | 11.3% / 0.73 / -25.7% | 0.94 / 1.06 / 1.37 | 18.1% / 1.17 / -15.4% | 10.4x | $2.9M / $14.7M |

- **W's low standalone CAGR is by design.** A VXN target of 18 and inverse-vol weights give an average exposure well
  below 1, and uninvested cash earns nothing in the engine.
- **The filter's cost is small.** REL25 changes about 12% of W's names and costs about 0.01 G3 Sharpe.
- **REL50 costs more.** It changes about a third of the names and loses the 2022-26 block (-0.08).
- **A nominal $5M floor does almost nothing today.** It removes 14% of members in 2000-01 but only 0.4% in the last
  two years. That is why a relative filter was chosen.
- **The filter fixes the recent tail, not the early one.** Over the full history REL25's largest order still reaches
  1% of ADV20 at a $0.6M pod (early-2000s names).

## Descriptive only: a three-leg book

0.5 TAA + 0.25 NDX L + 0.25 W-REL25, rebalanced daily:
- G3 Sharpe by block is 0.79 / 1.30 / 1.34, against G3's 0.70 / 1.30 / 1.29.
- Over 2012-26 it makes 18.0%/yr at Sharpe 1.31, with a -12.9% max drawdown (G3: 20.0%, 1.30, -14.1%).
- The correlation of W-REL25 with NDX L is 0.75, so the second momentum leg mainly trades return for drawdown.
- This was not part of the rule. It would need its own frozen test.

## Caveats

- W was chosen by an exploration in which its unfiltered 2012-26 results were seen, and Russell 1000 itself was
  picked after results. A forward shadow line is the only clean evidence.
- The filter percentile (25%) was fixed before the run; REL50 and ABS5M are the declared sensitivities.
- The capacity figures are a proxy, and they overstate market-on-open capacity.
