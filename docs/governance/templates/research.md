---
title: Research Page Template
description: Reusable structure for evidence-backed quantitative research.
document_type: reference
authority: canonical
risk_scope: research
---

# Research Page Template

```markdown
---
title: Study name
description: The research question in one sentence
document_type: research
authority: canonical
risk_scope: research
source_paths:
  - study/script/path.py
  - saved/artifact/path
---

# Study name

## Question
The falsifiable claim being tested.

## Frozen specification
Signal, parameters, universe, benchmark, and pass/fail gates fixed before results.

## Data and timing
Point-in-time treatment, adjustments, decision time, and execution time.

## Search space
Every variant tested and the multiple-comparison treatment.

## Costs and capacity
Commissions, slippage, borrow, liquidity, and participation assumptions.

## Results
In-sample and out-of-sample tables tied to saved artifacts.

## Robustness
Subperiods, regimes, parameter stability, and transfer tests.

## Limitations
Sample size, data gaps, proxy assumptions, and live/backtest divergence.

## Verdict
Promote, retain as diagnostic, reject, or require more evidence — with reasons.

## Reproduction
Exact commands, environment assumptions, and artifact paths.
```
