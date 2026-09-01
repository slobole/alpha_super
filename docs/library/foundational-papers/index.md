---
title: Foundational Papers
description: Important research papers and serious commentary that shape the project's quantitative reasoning.
document_type: reference
authority: guide
risk_scope: research
---

# Foundational Papers

!!! abstract "Purpose"
    This collection preserves important research papers and serious research commentary, with a governed reading note for each source. A source may shape the research agenda without becoming a strategy specification or evidence that Alpha Super has independently reproduced its results.

## Equity and multi-factor foundations

| Paper | Why it matters | Internal status |
|---|---|---|
| [Momentum Factor Construction and Signal Orthogonality](momentum-factor-construction.md) | Connects 12-1 momentum, information coefficients, effective breadth, signal orthogonality, risk scaling, costs, and statistical testing in one framework. | Foundational source; independent replication not yet performed |

## Crypto

Crypto sources live in a separate collection because their data quality, 24/7 timing, shorting, exchange, liquidity, and market-definition problems require different controls.

[Open the Crypto collection](crypto/index.md){ .md-button }

## Reading rule

```mermaid
flowchart LR
    A["Original paper"] --> B["Governed reading note"]
    B --> C["Frozen replication specification"]
    C --> D["Saved internal evidence"]
    D --> E["Strategy decision"]
```

The original PDF is preserved as the source. The reading note separates the author's claims from the project's own evidence and identifies what must be frozen before any replication or implementation.
