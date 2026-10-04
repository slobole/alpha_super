# Scout: where does DV2's edge come from? A size ladder of 12 universes

Date: 2026-10-02. Research only; nothing here changes live trading.
- **Scripts:** `scripts/research/scout_dv2_size_ladder_20261002/`.
- **Helper:** `alpha/scout/universes.py` (one superset price panel plus exact membership matrices).
- **Results:** `results/scout/dv2_size_ladder/` (not in git).
- **Ledger:** `dv2_size_ladder_20261002`, registered before any result, with 212 prior trials. The universes were
  chosen after results (the 2026-09-25 study had already run MidCap, SmallCap and Russell), and the registration
  says so.

**Design.**
- **The rule is frozen:** the WIRED DV2 live configuration, with no parameter search.
- **Universes:** 12 Norgate indexes with exact point-in-time membership.
- **Window:** in sample 2004 to 2022.
- **The price panel:** one superset of 10,116 symbols. Russell 3000 equals Russell 1000 ∪ Russell 2000 exactly, and
  S&P 1500 equals 500 ∪ 400 ∪ 600 exactly.
- **Check against P7:** on the S&P 500 the panel reproduces P7 exactly (membership and gross replica identical;
  costed Sharpe 0.937 vs the engine's 0.942).

## The answer

**1. Gross of costs, the edge is in the smallest, least liquid stocks.** The cleanest view is one rule on one
universe: Russell 3000 plus Micro Cap, events bucketed by size on the event date. Excess at three days is measured
over same-date members of the same bucket.

| Bucket | Events | Excess (bp) | t | Placebo p | Round trip (bp, pooled spread) | Event ADV |
|---|---|---|---|---|---|---|
| Top 200 (mega) | 40,216 | +4.0 | 1.50 | 0.22 | 10 | $281M |
| Russell 1000 ex Top 200 | 152,982 | +3.5 | 1.49 | 0.21 | 16 | $54M |
| Russell 2000, upper half | 174,565 | +1.9 | 0.69 | 0.27 | 31 | $11M |
| Russell 2000, lower half (in Micro) | 153,837 | **+15.4** | **4.22** | **0.005** | 78 | $2.2M |
| Micro Cap ex Russell 2000 | 95,080 | +13.0 | 3.32 | 0.065 | 178 | $0.5M |
| ADV63 low / middle / high tercile | | +17.3 / +5.3 / +0.6 | 5.10 / 1.69 / 0.23 | | 89 / 32 / 14 | |

**2. Net of costs, only the large-cap end survives.** The frozen rule as a pod, with Sharpe under engine costs, a
stress case, and two liquidity-aware spread models (AR and Pooled, which bracket the true cost):

| Universe | Gross | Engine | Stress | AR / Pooled | Max DD (engine) | Capacity (1% ADV) |
|---|---|---|---|---|---|---|
| S&P 100 | 1.00 | 0.83 | 0.18 | 0.28 / 0.71 | −22% | $27.7M |
| Russell Top 200 | 1.02 | 0.85 | 0.19 | 0.23 / 0.72 | −29% | $20.2M |
| **S&P 500** | **1.11** | **0.94** | **0.30** | 0.26 / 0.77 | −31% | $7.9M |
| Russell 1000 | 0.83 | 0.66 | 0.05 | −0.03 / 0.44 | −38% | $2.4M |
| Russell Mid Cap | 0.87 | 0.70 | 0.10 | 0.03 / 0.48 | −39% | $2.3M |
| MidCap 400 | 0.60 | 0.43 | −0.17 | −0.22 / 0.20 | −55% | $1.9M |
| SmallCap 600 | 0.59 | 0.41 | −0.10 | −0.39 / −0.18 | −74% | $0.22M |
| Russell 2000 | 0.34 | 0.13 | ruined | ruined | −94% | $0.08M |
| Micro Cap | 0.68 | 0.39 | −0.14 | ruined | −80% | $0.02M |
| Russell 3000 | 0.34 | 0.13 | ruined | ruined | −93% | $0.09M |
| S&P 1500 | 0.50 | 0.33 | −0.23 | −0.57 / −0.17 | −77% | $0.29M |

- **Why small caps fail as a pod although their event edge is the largest:**
  - The round trip costs more than the edge (78-178 bp vs +13-15 bp).
  - DV2 ranks by NATR, so it picks the most volatile names (41% of Micro Cap entries are below $5).
  - Capacity is tens of thousands of dollars.
- **Broad universes behave like their small-cap tail** for the same reason. The Russell 3000 pod is essentially the
  Russell 2000 pod; the S&P 1500 (0.33) is far below the S&P 500 (0.94).

**3. The registered hypothesis holds in sample.** The edge is larger gross in smaller, less liquid stocks, but net of
realistic costs it concentrates in large caps. **The S&P 500 is the right universe for this rule.** The per-asset
MCPT passes for the S&P 100 (p 0.003), Russell 1000 (0.001) and S&P 500 (0.001).

## The more important finding: the event edge has decayed

The DV2 S&P 500 event edge at three days, by era (verified on P7's own S3 inputs):

| Era | Excess | t |
|---|---|---|
| 1998-2007 | +18.2 bp | 4.84 |
| 2008-2015 | +5.4 bp | 1.27 |
| 2016-2022 | +0.8 bp | 0.2 |

- **Correction to P7:** P7's S3 headline (+9.1 bp, t 3.96) includes 1998-2003. From 2004, the pod's own window, it is
  +3.8 bp, t 1.54, placebo p 0.16.
- **Every universe and bucket is about zero in 2016-2022,** micro caps included (+3.5 bp, t 0.76). This matches the
  2026-09-28 Alpha101 finding ("edge real to 2011, then below costs").
- **Yet the pods keep earning.** Net Sharpe by era, live configurations:

  | Era | DV2 S&P 500 | HPI vote | DV2 industry ETF |
  |---|---|---|---|
  | 2004-2007 | 1.04 | 1.22 | (not trading) |
  | 2008-2015 | 0.82 | 0.80 | 1.27 |
  | 2016-2022 | 1.03 | 1.28 | 1.23 |

  - DV2's gross active Sharpe over the equal-weight members is 0.87 in 2016-2022.
  - **The reading:** the average entry event no longer beats its peers. The pods' recent returns come from what the
    S3 event does not measure: the NATR choice (high-volatility, higher-beta names), the exit rule, and being
    invested after dips in uptrends. Part of that is beta, which S6's factor regression already discounts (DV2 alpha
    t 1.80).
  - **Consequence:** the "stock-picking" story behind short-term reversal is weak in this decade. HPI and DV2 should
    be read as timing-and-exposure pods.
  - **DV2 industry ETF** is the exception worth watching: its event edge covers costs (1.62) over 2012-2022, and its
    MCPT, DSR and alpha all pass.

## Caveats

- **Neither cost model is a true spread.** The per-stock Abdi-Ranaldo estimate is contaminated by volatility; the
  pooled one undercharges mid and small caps. A TAQ-calibrated or EDGE estimator is needed before any small-cap claim.
- **Russell Micro Cap:** membership thins in 2009-2020, and before June 2005 it is Russell's back-calculated history.
- **The long-only reversal MCPT null is centred well below zero** (−0.34 for the S&P 100). The p-values are valid,
  but beating that null is a low bar.
- **Fees:** the engine's split-adjusted fees overcharge slightly (S&P 500 Sharpe 0.893 vs 0.937 on nominal shares),
  as the `split_adjusted_share_units` deviation records.
