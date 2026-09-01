---
title: Strategy Page Template
description: Reusable structure for a governed strategy specification.
document_type: reference
authority: canonical
risk_scope: research
---

# Strategy Page Template

Copy the structure below into the canonical strategy document. Remove sections only when they genuinely do not apply.

```markdown
---
title: Strategy name
description: One-sentence description
document_type: reference
authority: canonical
risk_scope: research | live
source_paths:
  - strategy/code/path.py
  - test/or/artifact/path
---

# Strategy name

## Executive summary
What the strategy does, in plain language.

## Status
- Research maturity:
- Deployment status:
- Last verified evidence:

## Intuition
Why the edge may exist and when it may fail.

## Instruments and data
Universe, adjustment type, point-in-time rules, and dependencies.

## Exact rules
Signals, parameters, ranking, filters, and formulas.

## Timing
Decision time, information cutoff, order time, and assumed fill.

## Portfolio construction
Selection, weights, cash behavior, limits, and rebalance rules.

## Flow diagram
One Mermaid diagram showing data -> signal -> sizing -> execution.

## Costs and execution assumptions
Commissions, slippage, borrow, capacity, and fill assumptions.

## Evidence
Verified results and exact artifact paths. No unsupported performance claims.

## Limitations and failure modes
Known gaps, regime dependence, sample limits, and live/backtest divergence.

## Source map
Code, tests, configuration, research artifacts, and related documents.
```
