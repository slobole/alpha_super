---
title: Maintain the Base
description: Rules and templates for keeping project knowledge trustworthy.
document_type: reference
authority: canonical
risk_scope: none
---

# Maintain the Base

!!! abstract "Who this section is for"
    Humans or AI adding, reviewing, or reorganizing documentation. Normal readers can ignore this section.

<div class="grid cards" markdown>

-   :material-ruler-square: **Documentation Standard**

    The minimum rules for readable, precise, source-backed pages.

    [:octicons-arrow-right-24: Open the standard](documentation-standard.md)

-   :material-map-legend: **Content Map**

    What exists, what is published, what needs review, and what is historical.

    [:octicons-arrow-right-24: Open the map](content-map.md)

-   :material-file-document-multiple-outline: **Templates**

    Start a strategy, operations, architecture, or research page without inventing a new structure.

    [:octicons-arrow-right-24: Open the strategy template](templates/strategy.md)

</div>

## Update flow

```mermaid
flowchart TD
    A["Behavior or evidence changes"] --> B["Identify the canonical source"]
    B --> C["Update source and affected guide"]
    C --> D["Verify facts, timing, links, and diagrams"]
    D --> E["Build and inspect the portal"]
```
