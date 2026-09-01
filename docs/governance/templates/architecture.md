---
title: Architecture Page Template
description: Reusable structure for a system explanation or architecture contract.
document_type: reference
authority: canonical
risk_scope: none
---

# Architecture Page Template

```markdown
---
title: Component or flow
description: One-sentence system boundary
document_type: explanation | reference
authority: canonical | guide
risk_scope: none | research | live
source_paths:
  - implementation/path
---

# Component or flow

## Problem
What this part of the system is responsible for.

## Mental model
A short explanation before internal detail.

## System boundary
Inputs, outputs, owners, and explicit non-responsibilities.

## Diagram
Flowchart, sequence diagram, or state machine.

## Components and data flow
Responsibilities, stored state, and interfaces.

## Timing and state transitions
Exact timing boundaries and legal transitions.

## Invariants
Rules that must always remain true.

## Failure modes
How failures surface, where they stop, and what remains recoverable.

## Limitations
Known gaps and intentionally unsupported cases.

## Source map
Code, tests, schemas, configuration, and related contracts.
```
