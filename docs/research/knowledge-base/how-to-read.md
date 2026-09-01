---
title: "How to Read the Research Knowledge Base"
description: "Evidence hierarchy, timing semantics, and status definitions."
document_type: research
authority: guide
risk_scope: research
source_paths:
  - "pakal-research/AGENTS.md"
  - "pakal-research/knowledge/STRATEGY_FEATURE_CATALOG.md"
---
<!-- GENERATED FILE. -->

# How to read this knowledge base

## Evidence hierarchy

```text
source claim
-> literal local replication
-> executable timing translation
-> signal diagnostics
-> portfolio economics after costs
-> locked validation / confirmation
-> research verdict
```

Do not collapse these layers. A replicated source is not automatically an executable strategy, and an attractive backtest is not deployment proof.

## Timing

When a signal uses final `Close_T` data, the primary executable assumption is normally `Open_(T+1)`. A `Close_T` fill can remain a useful timing diagnostic for overnight attribution, but it is not causal execution unless a real pre-close signal and order cutoff are modeled.

For close-derived signals, compare two clearly labelled paths when the data support it:

```text
diagnostic only: Close_T entry -> later exit
causal primary:  Open_(T+1) entry -> same later exit
```

For a next-close horizon, use the compounded identity:

```text
r_overnight = Open_(T+1) / Close_T - 1
r_intraday  = Close_(T+1) / Open_(T+1) - 1
1 + r_close_to_close = (1 + r_overnight) * (1 + r_intraday)
```

The diagnostic answers **where the historical return occurred**. The causal path answers **what the strategy could have captured**. If the edge disappears at `Open_(T+1)`, record that the apparent alpha was concentrated in the untradeable overnight leg; do not silently keep the same-close result.

## Statuses

| Status | Interpretation |
| --- | --- |
| `diagnostic` | Useful evidence, failed strategy, or state variable; no advancement claim. |
| `forward_hypothesis` | Frozen rule awaiting data it has not already seen. |
| `research_candidate` | Passed the declared research gate, still research-only. |
| `deployment_candidate` | Research evidence is sufficient to begin a separate operational-readiness process. |

## What to inspect first

1. Verdict and replication outcome.
2. Decision and fill timing.
3. Validation/confirmation evidence, not only full-history metrics.
4. Search breadth and multiple comparisons.
5. Costs, turnover, capacity, borrow, and implementation gaps.
6. Limitations and next frozen gate.

## Boundary

Nothing in this knowledge base changes LIVE, broker, scheduler, allocation, release, or pod state.
