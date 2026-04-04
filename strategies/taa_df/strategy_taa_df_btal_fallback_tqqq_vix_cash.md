# `strategy_taa_df_btal_fallback_tqqq_vix_cash`

## Short Description

This is a monthly tactical allocation strategy.

It starts with a defensive asset basket:

- `GLD`
- `UUP`
- `TLT`
- `DBC`
- `BTAL`

Its fallback asset is `TQQQ`.

The main idea is:

1. Rank the defensive assets by momentum at each month-end.
2. Keep the assets whose momentum is strong enough versus cash.
3. Send the weak slots to `TQQQ`.
4. Before allowing the `TQQQ` fallback, check whether market volatility looks supportive.
5. If not, keep that fallback part in cash instead.

So the strategy is trying to:

- stay invested in defensive assets when they are behaving well
- use `TQQQ` only for the part that failed the defensive filter
- avoid using `TQQQ` when short-term realized volatility is already too high relative to the VIX

## What Is Calculated

### 1. Defensive momentum

For each defensive asset, the strategy computes:

- 1-month return
- 3-month return
- 6-month return
- 12-month return

Then it averages them:

```text
momentum_score = average(1m, 3m, 6m, 12m returns)
```

This is done on **month-end closes**.

### 2. Cash hurdle from `DTB3`

The strategy loads `DTB3`, which is the 3-month T-bill yield, and converts it into an approximate 1-month cash return:

```text
cash_return = (1 + DTB3 / 100)^(1/12) - 1
```

This is used only as a **hurdle**.

Important:

- `DTB3` is **not** traded
- cash in the backtest does **not** earn that rate automatically
- it is only used to decide whether a defensive asset is strong enough

### 3. Rank weights

The five defensive assets are ranked from strongest to weakest by momentum.

They get these slot weights:

- rank 1: `5 / 15 = 33.33%`
- rank 2: `4 / 15 = 26.67%`
- rank 3: `3 / 15 = 20.00%`
- rank 4: `2 / 15 = 13.33%`
- rank 5: `1 / 15 = 6.67%`

## Monthly Decision Logic

At each month-end:

1. Rank the five defensive assets by momentum.
2. For each ranked asset, compare:
   - `momentum_score`
   - `cash_return`
3. If:

```text
momentum_score > cash_return
```

the asset keeps its slot weight.

4. If:

```text
momentum_score <= cash_return
```

that slot weight is redirected to `TQQQ`.

So before the VIX filter, the portfolio is always split between:

- the defensive assets that passed
- `TQQQ` for the failed slots

## VIX Cash Overlay

After the normal month-end weights are built, the strategy checks whether the `TQQQ` fallback should actually stay invested.

It uses:

- `SPY` daily closes
- `$VIX` daily closes

First it computes SPY daily returns:

```text
ret_spy_t = SPY_t / SPY_{t-1} - 1
```

Then it computes 20-day realized volatility:

```text
rv20_t = std(last 20 daily SPY returns) * sqrt(252) * 100
```

At month-end:

- if `rv20 < VIX`, the `TQQQ` fallback stays invested
- if `rv20 >= VIX`, the `TQQQ` fallback weight is set to `0`

The removed `TQQQ` weight becomes **cash**.

Important:

- this VIX rule affects **only the fallback sleeve**
- it does **not** change the weights of `GLD`, `UUP`, `TLT`, `DBC`, or `BTAL`

## When Trades Happen

The strategy makes its decision at the **end of the month**.

It does **not** trade at that same close.

Instead:

- month-end data is used for the signal
- the rebalance happens at the **first trading day of the next month**
- the fill is modeled at the **next open**

So the sequence is:

1. Month-end close: compute scores and weights
2. First trading day of next month: rebalance at the open

This is important because it avoids look-ahead bias.

## Data Used

The strategy intentionally uses two different price views:

### Signal data

For ranking and momentum:

- `TOTALRETURN` close data

Reason:

- this gives cleaner total-return signal measurement

### Execution data

For fills and valuation:

- `CAPITALSPECIAL` OHLC data

Reason:

- this is more realistic for trading and account valuation

## Practical Meaning

In plain English, the strategy says:

- own the best defensive assets if they are doing better than cash
- for the weak defensive slots, use `TQQQ`
- but only keep `TQQQ` if the VIX is still above recent realized SPY volatility
- otherwise, keep that fallback part in cash

This makes the strategy:

- more aggressive than the original SPY fallback version
- but more conservative than always holding `TQQQ` whenever defensive assets fail

## Important Assumptions

- Rebalance frequency is **monthly**.
- There is no intramonth signal update.
- The VIX filter uses **SPY realized volatility**, not `TQQQ` realized volatility.
- The VIX filter uses **20 trading days**.
- The comparison is strict:

```text
rv20 < VIX   -> allow TQQQ
rv20 >= VIX  -> go to cash for the fallback sleeve
```

- Position sizing is based on the portfolio value available before the rebalance and next-open prices, so realized weights can drift slightly because of:
  - overnight gaps
  - slippage
  - rounding
- Cash is literal residual cash in the backtest engine.

## What Can Happen In Practice

### Case 1: all defensive assets are strong

Then the portfolio stays in the defensive basket and `TQQQ` gets little or no weight.

### Case 2: some defensive assets are weak

Those failed slots go to `TQQQ`.

Then the VIX rule decides:

- keep that `TQQQ` part
- or replace that part with cash

### Case 3: all defensive assets are weak

Before the VIX rule, the portfolio would be `100% TQQQ`.

After the VIX rule:

- if `rv20 < VIX`, it can stay `100% TQQQ`
- if `rv20 >= VIX`, it becomes `100% cash`

## Effective Start Date

The strategy needs:

- `BTAL` history
- `TQQQ` history
- `VIX` history

So the effective start date is the latest required inception date.

In practice, this variant starts from the BTAL regime, because BTAL is the limiting asset.

## One-Sentence Summary

This strategy is a monthly ranked defensive allocator with a `TQQQ` fallback, where the fallback is allowed only when recent realized SPY volatility is below the VIX; otherwise the fallback sleeve stays in cash.
