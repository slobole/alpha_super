---
title: Documentation Standard
description: Minimal rules for accurate, readable, maintainable documentation.
document_type: reference
authority: canonical
risk_scope: none
---

# Documentation Standard

## Goal

Documentation must be simple enough for a human to scan, precise enough for an operator or engineer to trust, and structured enough for an AI to update without guessing.

## Non-negotiable rules

1. **One fact, one canonical source.** A guide may summarize a contract, but it must name the governing source and must not copy large rule sets that can drift.
2. **Plain language first.** Explain the intuition before formulas or implementation detail.
3. **Exact semantics second.** State instruments, parameters, data source, timing, sizing, cash behavior, costs, and failure conditions explicitly.
4. **Evidence for claims.** Do not publish performance, readiness, health, or deployment claims without current artifact paths or runtime evidence.
5. **Separate statuses.** Document authority, strategy maturity, and LIVE status are independent fields.
6. **Show the flow.** Use Mermaid for multi-stage pipelines, decision trees, state machines, and timing-sensitive behavior.
7. **Mark boundaries.** Research, PAPER, LIVE, historical, and derived material must be visibly labeled.
8. **Prefer stable Markdown.** Avoid application-specific syntax that does not render in GitHub or MkDocs.
9. **English by default.** Create Hebrew versions only when requested; preserve exact identifiers, formulas, and timing.
10. **Archive; do not disguise.** Old reviews and superseded guidance remain historical and must not appear as current instructions.

## Required page metadata

```yaml
---
title: Clear page title
description: One-sentence purpose
document_type: tutorial | how-to | reference | explanation | research | historical
authority: canonical | guide | historical
risk_scope: none | research | live
source_paths:
  - path/to/source
---
```

`source_paths` is required whenever the page derives facts from code, configuration, artifacts, or another document.

## Visual rules

- Use one diagram when it materially shortens the explanation.
- Keep a diagram focused on one question and normally below 25 lines.
- Label branches and timing boundaries.
- Store durable screenshots and images under `docs/assets/<subject>/`.
- Give every image meaningful alternative text.
- Treat presentations as derived communication artifacts, not canonical specifications.

## Update workflow

```text
identify changed behavior
-> locate canonical source
-> update source and affected guide
-> verify timing, links, and diagrams
-> build with strict checks
-> review the rendered page
```

If a contradiction is found, stop and surface it. Do not silently choose the more convenient version.
