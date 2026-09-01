---
title: Strategy Catalog
description: Simple, source-linked specifications for every WIRED strategy.
document_type: reference
authority: guide
risk_scope: live
source_paths:
  - alpha/strategy_registry.py
---

# Strategy Catalog

!!! abstract "What is included"
    This catalog contains all **9 strategies currently marked `WIRED`** in the central registry. Each page explains the mechanics, timing, sizing, and governing source without making a performance claim.

!!! warning "WIRED is plumbing, not proof"
    `WIRED` means a strategy is connected to a LIVE account route. It does **not** mean the strategy is profitable, its release is enabled, its data is healthy, or an order is currently due. Verify releases and runtime evidence separately.

## Daily mean reversion

<div class="grid cards" markdown>

-   :material-chart-bell-curve: **DV2 Mean Reversion**

    Oversold S&P 500 stocks above SMA200, ranked by NATR14.

    [:octicons-arrow-right-24: Open](dv2-mean-reversion.md)

-   :material-chart-timeline-variant: **QPI + IBS Mean Reversion**

    QPI and IBS pullback entry with IBS or RSI2 exits.

    [:octicons-arrow-right-24: Open](qpi-ibs-mean-reversion.md)

-   :material-history: **HPI 3-Day Mean Reversion**

    Negative 3-day moves ranked against 1,260 prior observations.

    [:octicons-arrow-right-24: Open](hpi-3d-mean-reversion.md)

-   :material-vote-outline: **HPI 2/3/5-Day Vote**

    Requires agreement from at least two HPI horizons.

    [:octicons-arrow-right-24: Open](hpi-235-vote.md)

</div>

## Monthly allocation and momentum

<div class="grid cards" markdown>

-   :material-chart-timeline-variant-shimmer: **TAA Defensive + TQQQ**

    Rank-weighted defensive allocation with a TQQQ-or-cash fallback.

    [:octicons-arrow-right-24: Open](taa-defensive-tqqq.md)

-   :material-view-grid-plus-outline: **TAA Equal Slots + TQQQ**

    Five equal defensive slots with a TQQQ-or-cash fallback.

    [:octicons-arrow-right-24: Open](taa-equal-slots-tqqq.md)

-   :material-chart-bell-curve-cumulative: **TAA Linearity + QQQ**

    Regression-linearity signal with equal slots and a QQQ-or-cash fallback.

    [:octicons-arrow-right-24: Open](taa-linearity-qqq.md)

-   :material-numeric-10-box-multiple-outline: **NDX ATR-Normalized Momentum**

    Top Nasdaq-100 momentum names after ATR and trend filters.

    [:octicons-arrow-right-24: Open](ndx-atr-momentum.md)

-   :material-tune-vertical-variant: **NDX ATR Momentum + VXN Scaling**

    The same NDX selection with VXN-based exposure reduction.

    [:octicons-arrow-right-24: Open](ndx-atr-vxn-scaled.md)

</div>

## Publication rule

Every page must name its governing source and state the exact universe, data, rules, timing, sizing, cash behavior, costs, and limitations. Use the [strategy template](../governance/templates/strategy.md) for future additions.
