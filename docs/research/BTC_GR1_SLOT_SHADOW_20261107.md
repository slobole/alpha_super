# A 5% Bitcoin slot beside GR1: forward shadow (2026-11-07)

Research-only. This is a simulated shadow: no orders, no capital, nothing wired.

| What | Where |
| --- | --- |
| Spec | `alpha/scout/specs/btc_vol_sleeve.py` |
| Shadow provider | `alpha/scout/shadow.py`, `SHADOW_PROVIDER_DICT` |
| Start script | `scripts/research/btc_gr1_slot_shadow_20261107/start_shadow.py` |
| Tests | `tests/test_scout_btc_vol_sleeve.py`, data lane |
| Registration | `btc_vol_sleeve_gr1_slot_20261107` (retro, WATCHLIST) |
| Shadow start | 2026-11-09 |

## The rule

The sleeve holds IBIT plus BIL.

- **Decision:** at the close of the last session of each week, set the IBIT target to `min(1, 20% / sigma60)`. sigma60 is the volatility of IBIT's last 60 daily returns.
- **Fill:** at the next open.
- **No-trade band:** while the IBIT weight is within ±10% of the target, nothing trades. Outside the band, trade only to the nearest edge.
- **BIL holds the rest.**

The candidate allocation is 5% of GR1 in this sleeve, which is about 1.7% of GR1 in Bitcoin.

## Why a shadow and not an allocation

The evidence comes from Pakal `btc_sleeve_candidates_study`, plus a recheck on 2026-11-07. The table compares GR1 alone with GR1 holding 5% in the sleeve.

| | GR1 alone | GR1 + 5% sleeve |
| --- | --- | --- |
| Excess Sharpe, 2011-08..2026-08 | 1.35 | 1.47 (P 1.00) |
| Excess Sharpe, 2021-26 | 1.27 | 1.28 (P 0.46) |
| Excess Sharpe, since IBIT (2024-01) | 1.35 | 1.33 |
| Vol | 12.5% | 12.1% |
| Max drawdown | -12.0% | -12.3% |
| Chance of a -20% drawdown (2,000 bootstrap paths) | 3.1% | 2.2% |

The whole historical gain was made in 2011-20. The slot costs almost nothing in risk, but it has not paid since 2021. The sleeve alone made 2.6% a year at 20% vol over the IBIT era (2024-04..2026-11). The shadow answers only the forward question.

## Monitoring and review

**Monitoring.** The S8 card compares the forward returns with the expected process: the same rule over the IBIT era, all of which has already been seen. The card turns RED on the calibrated Cold Blood Index or the monthly CUSUM. Retiring the shadow stays the owner's decision.

**Review at 12 months.** The fund-products Lab that builds GR1 ends on 2026-08-19. At the review:

1. Rebuild GR1 on the newest data.
2. Blend it with the sleeve's forward returns: 95/5, rebalanced monthly.
3. Adopt the slot only if GR1 + 5% beats GR1 and its drawdown is no deeper.

**One place only.** If CORE6 (`docs/research/CORE6_20261107.md`) is ever adopted in MONTHLY or DEF, do not also hold this slot in a book that contains CORE6. That would double the Bitcoin.
