---
title: HPI 3-Day Mean Reversion
description: Daily S&P 500 mean-reversion strategy using a historical percentile of negative 3-day returns.
document_type: reference
authority: guide
risk_scope: live
source_paths:
  - alpha/strategy_registry.py
  - strategies/hpi/strategy_mr_hpi_sp500_ibs_rsi_exit.py
  - strategies/hpi/stateful_long.py
---

# HPI 3-Day Mean Reversion

!!! abstract "Plain-English summary"
    Find S&P 500 stocks whose negative 3-day move is unusually deep relative to their own negative history. Enter only when the stock is still above its long-term trend and closes near the day’s low.

<div class="grid cards" markdown>

-   :material-calendar-today: **Decision**

    Every trading-day close

-   :material-clock-outline: **Execution**

    Modeled at the next trading-day open

-   :material-history: **HPI history**

    Previous 1,260 observations, excluding today

-   :material-connection: **Maturity**

    `WIRED` — connected to a LIVE account route

</div>

## Decision flow

```mermaid
flowchart LR
    A["Point-in-time S&P 500"] --> B["Compare 3-day return with<br/>1,260 prior observations"]
    B --> C{"Return < 0<br/>HPI < 30<br/>IBS < 0.10<br/>Close > SMA200?"}
    C -->|"Pass"| D["Rank by turnover<br/>fill up to 10 slots"]
    C -->|"Fail"| E["No entry"]
    D --> F{"IBS > 0.90<br/>RSI2 > 90<br/>or membership lost?"}
    F -->|"Yes"| G["Exit next tradable open"]
```

## Exact rules

| Item | Rule |
|---|---|
| Universe | Point-in-time S&P 500 membership |
| HPI | Percentile-like rank of the current 3-day return within the same-sign tail of the previous 1,260 observations |
| Entry | 3-day return `< 0`, `HPI < 30`, `IBS < 0.10`, and `Close > SMA200` |
| Ranking | Turnover descending; symbol ascending on ties |
| Sizing | Up to 10 equal 10% slots |
| Exit | `IBS > 0.90`, `RSI(2) > 90`, or loss of point-in-time index membership |
| Data | Stocks use `CAPITALSPECIAL`; performance benchmark uses total-return S&P 500 |
| Modeled costs | 2.5 bps slippage; $0.005/share commission; $1 minimum |

!!! danger "Timing boundary"
    The HPI reference set excludes the current observation. Decisions use `Close T`; orders execute at the modeled `Open T+1`.

!!! warning "What WIRED does — and does not — mean"
    `WIRED` confirms a LIVE route exists. It does not establish research quality, release enablement, or current runtime health.

## Sources of truth

- Maturity: `alpha/strategy_registry.py`
- Variant configuration: `strategies/hpi/strategy_mr_hpi_sp500_ibs_rsi_exit.py`
- Shared HPI rules and lifecycle: `strategies/hpi/stateful_long.py`
