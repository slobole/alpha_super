# Current Strategies

TL;DR: The current focus is three chosen strategies: `strategy_mr_qpi_ibs_rsi_exit`, `strategy_mr_dv2_price_adv`, and `strategy_taa_df`. Only `strategy_taa_df` remains from the TAA branch; the other TAA variants are no longer part of the current active set.

Math note: formulas are intentionally written in plain text instead of LaTeX because the current Markdown preview does not render block math reliably.

This file explains what is current now. Use `QUANT_PHILOSOPHY.md` for house doctrine and `ASSUMPTIONS_AND_GAPS.md` for realism limits.

## Common Execution Contract

All current strategies are causal:

```text
signal_t = f(information available up to previous_bar)
decision at t -> execution at next open
```

For the daily stock pods, the strategy reads information through `previous_bar` and sends orders that fill at the next daily open.

For `strategy_taa_df`, the decision is made at month-end and the rebalance happens at the first tradable open of the next month:

```text
month_end_decision_t -> first_tradable_open_of_next_month
```

Common data rules:

- Stock universes use point-in-time membership.
- Tradable stocks and ETFs use `CAPITALSPECIAL` for fills and valuation.
- `TOTALRETURN` is used only where the signal really needs it, such as benchmark comparison or the TAA ranking close series.

## Chosen Strategies

This is the current chosen snapshot:

| Strategy | Sharpe | Type | Role in the book |
|---|---:|---|---|
| `strategy_mr_qpi_ibs_rsi_exit` | 1.02 | Daily stock mean reversion | Fast oversold-bounce pod |
| `strategy_mr_dv2_price_adv` | 0.92 | Daily stock mean reversion | Broader liquid pullback pod |
| `strategy_taa_df` | 0.82 | Monthly ETF allocation | Slow diversifying allocation pod |

## `strategy_mr_qpi_ibs_rsi_exit`

Simple rule:

```text
entry = (QPI < 30)
     and (Close > SMA200)
     and (3d_return < 0)
     and (IBS < 0.10)

exit = (IBS > 0.90) or (RSI2 > 90)
```

What it is doing:

- It buys stocks that are still in a long-term uptrend, but have just had a sharp short-term washout.
- `QPI` is the "how unusually weak was the recent move" filter.
- `IBS < 0.10` means the stock closed very near the day's low, which is a strong exhaustion-style entry.
- `RSI2` and high `IBS` are used as fast rebound exits, so this pod is designed to get in on weakness and get out quickly once the bounce is stretched.
- Candidates are ranked by `Turnover`, so the pod prefers the more liquid names first.

Why it is in the chosen set:

- It is the cleanest fast mean-reversion expression in the repo right now.
- The logic is easy to explain: buy panic inside strength, then exit once the rebound becomes crowded.

## `strategy_mr_dv2_price_adv`

Simple rule:

```text
entry = (Close > 10)
     and (ADV20 >= 20,000,000)
     and (DV2 < 10)
     and (Close > SMA200)
     and (126d_return > 0)

exit = Close > yesterday_high
rank = highest NATR first
```

What it is doing:

- It also buys short-term weakness inside longer-term strength, but the expression is different from QPI.
- `DV2 < 10` is the oversold trigger.
- `Close > SMA200` and positive `126d_return` keep the pod on names that are still structurally strong.
- `Close > 10` and `ADV20 >= 20M` remove low-price and weaker-liquidity names.
- Ranking by `NATR` pushes the strategy toward names that are actually moving enough to matter.

Why it is in the chosen set:

- It is a robust liquid-stock mean-reversion pod.
- Compared with the QPI pod, it is less about one extreme washout bar and more about a broader oversold-in-strength setup.

## `strategy_taa_df`

Current asset set:

- Defensive sleeve: `GLD`, `UUP`, `TLT`, `DBC`
- Fallback asset: `SPY`

Simple rule:

```text
momentum_score = average(1m_return, 3m_return, 6m_return, 12m_return)
rank_weight_vec = [0.40, 0.30, 0.20, 0.10]

if asset_momentum_score > cash_return:
    keep that rank weight on the defensive asset
else:
    redirect that slot weight to SPY
```

Execution mapping:

```text
month-end signal -> first tradable open of next month
```

What it is doing:

- This is the slow monthly allocator in the repo.
- Every month it ranks the defensive assets by momentum.
- Better-ranked assets get bigger weights: `40%`, `30%`, `20%`, `10%`.
- If one of those defensive assets is too weak versus the cash hurdle, its weight is not kept there; it is redirected to `SPY`.
- Signal formation uses `TOTALRETURN` closes, but actual fills and valuation use tradable `CAPITALSPECIAL` prices.

Why it is in the chosen set:

- It gives the book a different tempo from the daily stock pods.
- It is lower turnover, easier to hold operationally, and helps diversify the equity mean-reversion sleeves.
- It is the only TAA strategy that is still current. The other TAA strategies were removed from the active set.

## Current Research Landscape

The repo still contains multiple research branches, but they are not all equal in status.

- `QPI` family: short-term mean reversion after unusual weakness. Variants add IBS filters, RSI exits, price and liquidity filters, different universes, and short or long-short versions. The chosen version is `strategy_mr_qpi_ibs_rsi_exit`.
- `DV2` family: short-term mean reversion using the DV2 stretch signal. Variants change liquidity filters, exits, and universe definitions. The chosen version is `strategy_mr_dv2_price_adv`.
- `Alpha19` family: pullback-in-winners logic. This is still a valid research branch, but it is not in the chosen set right now.
- `Momentum` family: `strategy_mo_squeeze` is the breakout branch. It is directionally different from the mean-reversion pods, but it is still research, not a chosen pod.
- `TAA` family: only `strategy_taa_df` is current. Do not treat older TAA variants as active strategies.

## How To Think About The Current Book

- `strategy_mr_qpi_ibs_rsi_exit` is the fastest and most reactive pod.
- `strategy_mr_dv2_price_adv` is still a daily pullback pod, but with a more liquid and slightly broader profile.
- `strategy_taa_df` is the slow monthly allocator that gives the current book diversification by speed, instrument type, and behavior.

## Interpretation Rule

- The current chosen set is two daily single-stock mean-reversion pods plus one monthly ETF allocation pod.
- None of these strategies should be described as exact live replication.
- Only `strategy_taa_df` should be described as current TAA.
