---
title: Operations Page Template
description: Reusable structure for an auditable operating procedure.
document_type: reference
authority: canonical
risk_scope: live
---

# Operations Page Template

```markdown
---
title: Procedure name
description: What outcome this procedure safely produces
document_type: how-to
authority: canonical | guide
risk_scope: live
source_paths:
  - relevant/code/or/runbook
---

# Procedure name

## Purpose
The exact outcome and when to use this procedure.

## Safety boundary
PAPER or LIVE, affected Pod/account, allowed mutations, and required approval.

## Preconditions
Access, configuration, backups, runtime state, and expected versions.

## Flow
One diagram showing checks, actions, branches, and stop points.

## Procedure
Numbered steps. Each step includes the action and expected result.

## Stop conditions
Exact output or state that means do not continue.

## Recovery or rollback
How to return to a known state without hiding partial effects.

## Verification checklist
Observable proof that the intended outcome was achieved.

## Sources
Current code, configuration, contracts, and runtime evidence used for verification.
```
