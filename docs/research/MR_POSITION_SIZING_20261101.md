# Can DV2 and HPI size positions better than 1/N? (2026-11-01)

**Question (owner):** leaving the gate aside, can the stock positions in DV2 and HPI be sized more cleverly than 1/N?

**Answer:** one rule out of 13 works: give more money to the more liquid stocks. Every other idea loses to 1/N or adds nothing. It is a real but modest improvement, not a new engine. Nothing live changed. The recommendation and the honest reading of the gain moved to [GROWTH_MR_PRODUCTS_20261101](GROWTH_MR_PRODUCTS_20261101.md): in the gated book about half of the gain is bigger entries, because the rule is not normalised, and half is liquidity.

Frozen spec: `scripts/research/mr_position_sizing_20261101/SPEC_FROZEN.md`, ledger row 552. Post-result checks: `SPEC_CHECKS_L1.md`. Outputs: `results/scout/mr_position_sizing_20261101/`.

## Results

DV2 + HPI, 50 / 50, no gate, 2004-2026, net of costs, idle cash at the T-bill rate. Selection and exits as wired. Sharpe is over T-bills.

| Rule | CAGR | Sharpe | Max DD | P(beats 1/N) |
|---|---|---|---|---|
| 1/N, 10 names (today) | 18.3% | 0.98 | -23.3% | |
| **More to liquid stocks** | **21.8%** | **1.09** | **-19.9%** | **0.99** |
| Half now, half if it keeps falling | 16.6% | 1.01 | -20.8% | 0.89 |
| More when VIX is high | 18.5% | 0.98 | -29.4% | 0.57 |
| More to deeper signal | 17.3% | 0.97 | -22.9% | 0.39 |
| Use idle cash (up to 20% a name) | 19.2% | 0.96 | -31.3% | 0.36 |
| More on busy signal days | 17.8% | 0.94 | -19.6% | 0.22 |
| Less to volatile stocks, by range (NATR) | 13.5% | 0.92 | -15.8% | 0.09 |
| 15 names | 16.1% | 0.92 | -22.2% | 0.06 |
| More to volatile stocks | 18.6% | 0.90 | -27.6% | 0.05 |
| Less to volatile stocks, by 63-day vol | 13.2% | 0.89 | -16.9% | 0.02 |
| More after a harder 3-day fall | 15.9% | 0.88 | -25.9% | 0.03 |
| 20 names | 14.2% | 0.85 | -21.4% | 0.01 |
| 5 names | 17.3% | 0.83 | -30.9% | 0.01 |

Calmer-stock sizing does lower the drop, but it costs more than that: 16.7% CAGR at 1/N's risk. Random sizes with liquidity sizing's spread give a median Sharpe of 0.95, so spreading sizes without a reason hurts. That is why 1/N is hard to beat.

**The rule:** size = 1/10 x sqrt(the stock's 63-day dollar volume / typical candidate's), between half and double, total exposure capped at 100%. Known at the close of T. Average exposure is the same as 1/N (DV2 85%, HPI 71-72%).

**Why it may work, not proven:** trades in the most liquid third earn about 0.7% against 0.25% in the least liquid third, in both strategies (same-month difference t about 4), with no matching rise in risk. Volatile stocks also earn more per trade, but their risk rises as much, so vol sizing gains nothing.

## Checks of liquidity sizing

| Check | Result | Read |
|---|---|---|
| Frozen rules R1-R6 | all pass | |
| Across all 13 tries (family-wise) | p 0.07 | **just misses 5%** |
| Data leak | Turnover = split-adjusted close x split-adjusted volume = true dollars | clean |
| Random sizes, 300 draws | above all 300; top 1.04 | not luck in the weights |
| Real engine, $1M, both strategies | 22.0% / 1.10 / -19.7% vs 18.7% / 0.99 / -23.1%, P 0.98, better in all 3 eras | confirmed |
| Engine harness at 1/N | identical to sign-off baselines (4e-16) | harness proven |
| Nearby settings | window 21 / 126: +0.12 / +0.13; linear: +0.14; wider limits: +0.12; power 0.25: +0.06; **narrow limits 0.75-1.33: +0.044** | steady dose response, but misses my +0.05 bar once |
| Without its 10 best names | +0.08 (they carry 36% of the gain in DV2, 29% in HPI) | survives, partly mega caps |
| Years better | 19 of 23 | |
| Excess over 1/N against the S&P 500 | beta 0.02, 2.8% a year, t 3.1 | not hidden market risk |
| **2023-26** | **tie (0.00)**: 2023 and 2024 up about 7 points, 2026 to date -9 points | **recent edge not shown** |
| With the VIX gate | see the last section (real engine) | same answer, and 2023-26 is better |

Growth portfolio, TAA 3x 1N 50 / gated capsule 30 / CORE5 20, 2008-2026, replica:

| | CAGR | Sharpe | Max DD |
|---|---|---|---|
| 1/N | 21.0% | 1.26 | -15.6% |
| Liquidity sizing | 22.0% | 1.30 | -14.0% |

P 0.9965, 2016+ Sharpe 1.42 to 1.47.

Capacity, USD 100M in the capsule, order as a share of the median day, trades since 2023:

| | Typical | 1 in 10 | 1 in 100 | Worst |
|---|---|---|---|---|
| DV2 1/N | 1.5% | 4.2% | 8.9% | 25% |
| DV2 liquidity | 1.7% | 3.1% | 4.6% | 13% |
| HPI 1/N | 1.2% | 3.8% | 8.7% | 41% |
| HPI liquidity | 1.5% | 2.9% | 4.5% | 20% |

## Reading

- This differs from the rejected liquid DV2 pod (2026-10-04, 2026-10-05). That one changed which stocks are bought and held stocks 76% of the time. This keeps the signal and rank rules at the same average exposure and moves dollars between the names. It does not hold exactly the same names: larger entries fill the book sooner, and about a third of entry days buy a different set (`GROWTH_MR_PRODUCTS_20261101`).
- A forward test cannot settle it soon. Excess return 2.8% a year on 4% tracking error is about t 0.7 a year, so about 8 years to reach t 2. The decision is a judgment on this evidence, plus the capacity gain, which is mechanical rather than statistical.
- Selected from 13 on history already studied. Plan on less than the full gain.
- Flat costs. The capacity table suggests real costs favour this rule, but no fund-scale impact model was run.

## With the gate, in the real engine (added after the owner asked)

**Corrected 2026-11-01.** The first version of this section came from runs with a defect in `engine_l1.py run_g`: BIL was targeted as if every entry were 1/N, so cash fell to -25% and -30% of NAV with no financing charged. The tables below are from the fixed pods of `growth_mr_products_20261101` (BIL on the true entry dollars, stocks capped at 99% of NAV, `gated_corrected.json`). The fix moved the capsule by 0.01 of Sharpe and 0.2 points of Max DD, so the picture stands. Do not use `engine_gated_compare.json`.

The final pods, VIX gate and BIL parking, $1M, 2004-01 to 2026-10, research-only subclasses. At 1/N the harness matches the sign-off baselines to 6e-16. Post-result: the rule was frozen on the ungated pods, and these windows were chosen after that result.

| | CAGR | Sharpe over T-bills | Max DD |
|---|---|---|---|
| DV2 gated, 1/N | 16.0% | 0.85 | -28.8% |
| DV2 gated, liquidity | 19.0% | 0.95 | -24.0% |
| HPI gated, 1/N | 12.4% | 0.83 | -16.8% |
| HPI gated, liquidity | 14.3% | 0.91 | -19.7% |
| Capsule, 1/N | 14.3% | 0.90 | -21.4% |
| Capsule, liquidity | 16.8% | 1.00 | -18.5% |

| Capsule, 1/N against liquidity | CAGR | Sharpe over T-bills | Max DD |
|---|---|---|---|
| 2004-11 | 8.0% / 9.1% | 0.45 / 0.51 | -20.1% / -18.2% |
| 2012-19 | 11.1% / 12.1% | 1.15 / 1.19 | -11.1% / -12.6% |
| 2020-21 | 58.6% / 71.0% | 2.24 / 2.42 | -21.4% / -18.5% |
| 2022 | 15.7% / 14.4% | 0.76 / 0.66 | -13.1% / -15.7% |
| 2023-26 | 15.0% / 20.8% | 0.75 / 1.00 | -12.1% / -14.8% |
| Without 2020 | 12.4% / 14.3% | 0.81 / 0.89 | -20.1% / -18.2% |

P(better): 0.989 full, 0.980 from 2016, 0.990 from 2023 (a short sample). 14 years better by more than 0.1 points, 6 worse, 3 flat; 2020 alone is +23 points, 2026 to date +6.

Growth portfolio, TAA 3x 1N 50 / capsule 30 / CORE5 20, 2008-2026, fund Lab (fair cash): 20.9% / 1.26 / -15.6% against 21.9% / 1.30 / -14.0%, paired share 0.998; from 2023 Sharpe 1.55 against 1.63.

**Correction:** the "tie since 2023" above is the ungated capsule. The gated capsule, which is what would trade, gained in 2023-26. The same rule in 2026 to date is -9 points ungated and +6 gated, so the recent period is noisy and should not carry the case either way.

**Is it liquidity or volatility?** Trades ranked within their year, base sizing, no gate: liquid minus illiquid third is +0.57% per trade in DV2 (t 4.4, 22 of 23 years) and +0.61% in HPI (t 3.3, 19 of 23 years). Liquidity and volatility ranks correlate only 0.24 and 0.14. Among the wildest third of DV2 trades, liquid earns 1.14% and illiquid 0.30%, at the same trade risk (table in `double_sort.csv`). So it is not a volatility tilt in disguise.

## Recommendation

Superseded by `GROWTH_MR_PRODUCTS_20261101`: adopt it as the MR form of the Growth products, plan on about half the backtest gain, and build it before go-live only if that does not move the go-live date. NDX and TAA are not touched. Close the other sizing ideas on this measurement.

The average exposure figures above hide one thing the independent review found: real entries are larger than 1/N's (mean m 1.18 in DV2, 1.35 in HPI) and fewer. A constant size at those levels gets about half the book's gain with no liquidity input.
