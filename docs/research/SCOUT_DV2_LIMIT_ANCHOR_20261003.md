# Scout: where should DV2's limit sit? Anchor and offset measure

Date: 2026-10-03. Research only; nothing here changes live trading.
- **Scripts:** `scripts/research/scout_dv2_limit_anchor_20261003/`.
- **Results:** `results/scout/dv2_limit_anchor/` (not in git).
- **Ledger:** `dv2_limit_anchor_20261003`, registered before any result. Grid of 144 (3 universes × 2 anchors ×
  4 measures × 3 fill targets × 2 exits); prior trials now 456.

**The question (owner).** Would the limit be better placed:
- at today's open instead of yesterday's close;
- using last month's movement instead of NATR;
- using the stock's typical downside move?

**Design: equal fill rates.** A deeper limit is not better, only different. So every variant is calibrated to fill
25%, 40% or 60% of orders:
- the multiplier is calibrated on 2004-2012 only, then run unchanged to 2022;
- realized fill rates land within about 2 points of target;
- fills and costs follow the limit study (trade-through fills, AR and pooled half-spreads);
- the exit is the limit exit.

## Results

**S&P 500, net Sharpe AR / pooled:**

| Fill | Close × NATR14 (today's study) | Close × downside excursion | Open × NATR14 | Open × downside excursion |
|---|---|---|---|---|
| 25% | 0.74 / 0.83 | 0.88 / 0.98 | 0.71 / 0.80 | 0.79 / 0.88 |
| 40% | 0.70 / 0.82 | 0.77 / 0.89 | 0.69 / 0.80 | 0.77 / 0.87 |
| 60% | 0.62 / 0.78 | 0.69 / 0.85 | 0.73 / 0.85 | 0.71 / 0.83 |

- **Reference:** market-on-open in and out, 0.26 / 0.77.
- **Downside excursion:** the 21-day mean of (Open − Low) / Open. The 21-day std and the 63-day quantile variants fall
  in between.

**Replication** (pooled Sharpe at 25 / 40 / 60%):

| Cell | S&P 100 | Russell 1000 |
|---|---|---|
| Close × NATR14 | 0.81 / 0.85 / 0.80 | 0.95 / 0.78 / 0.68 |
| Close × downside excursion | 0.82 / 0.89 / 0.83 | 0.92 / 0.76 / 0.71 |
| Open × downside excursion | 0.89 / 0.89 / 0.87 | 0.78 / 0.63 / 0.61 |

## What this means

1. **The registered hypothesis fails.** No variant beats Close − k × NATR14 once the comparisons are counted.
   - **The best cell** (close × downside excursion, S&P 500, 25%) is +0.15 Sharpe, CI [+0.03, +0.26].
     - The gain sits in 2004-2007; 2016-2022 is a tie (0.90 vs 0.90).
     - It does not replicate on the Russell 1000.
     - Adjusted for all 21 comparisons (Romano-Wolf), p is 0.13.
   - **The open anchor** helps on the S&P 100 and hurts on the Russell 1000 (several CIs entirely below zero). It also
     loses more under strict fills: 0.45-0.63 against 0.64-0.70 at 40%.
2. **"Last month's movement" is NATR in another form.** On signal days NATR14 correlates 0.85 with the 21-day std
   and 0.88 with the downside excursion. The measure barely changes which names fill (84-96% overlap), so it moves
   Sharpe by only about ±0.05.
3. **Keep the close anchor.** It can be sent overnight with the rest of the orders. The open anchor needs an order just
   after the open, and its fills are the ones daily bars flatter most: the day's low often prints before such an order
   could rest.
4. **A correction to the limit study's headline.** Moving the multiplier by ±10% shifts Sharpe by about ±0.04, the size
   of most differences here. k 0.5 sits on a lucky point: k 0.48 gives 0.82 pooled, against 0.86. Plan on about
   0.70 / 0.82 (AR / pooled) at a 40% fill rate.
   - **What does not change:** the conclusion of the limit study. Limit entries beat market-on-open (0.26 / 0.77) by
     a wide margin, under both cost models.
5. **MCPT p 0.001** for the plateau cell. It tests DV2 with limits against no edge, which any anchor passes. It does not
   separate the variants; the paired bootstrap and Romano-Wolf do.

## Recommendation

No change to the limit design: Close − k × NATR14, with k around 0.5 and a fill rate of about 40%.
- **For the paper shadow:** record fills by fill rate, not by k. If paper fills come out near 40% at k 0.5, the
  backtest's fill model is roughly right.
- **The downside-excursion offset** is a reasonable alternative with no proven advantage.
