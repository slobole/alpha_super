---
title: Content Map
description: Inventory and migration plan for existing project knowledge.
document_type: reference
authority: canonical
risk_scope: none
---

# Content Map

This is a routing map, not a claim that every document has been audited. “Publish later” means the source stays available in the repository but is not yet presented as current portal guidance.

## Core and doctrine

| Current source | Authority | Portal destination | Action |
|---|---|---|---|
| `README.md` | Guide | Start Here / Reference | Keep as repository entry point; do not duplicate |
| `QUANT_PHILOSOPHY.md` | Canonical | Reference | Keep path; add audited wrapper later |
| `ASSUMPTIONS_AND_GAPS.md` | Canonical | Reference | Keep path; add audited wrapper later |
| `LIVE_START_HERE.md` | Canonical LIVE entry | Operations / Reference | Publish only after LIVE audit |
| `COMMANDS.md` | Reference candidate | Reference | Audit commands before publication |
| `AGENTS.md`, `CLAUDE.md` | Canonical agent guidance | Reference | Keep paths; do not present as user tutorials |
| `docs/ai/KARPATHY_GUIDELINES.md` | Canonical engineering guidance | Reference | Keep path |

## LIVE and operations

| Current source group | Authority | Portal destination | Action |
|---|---|---|---|
| `docs/live/LIVE_RUNBOOK.md` | Canonical candidate | Operations | Full code and runtime audit before publication |
| `docs/live/LIVE_USER_SETUP_QUICK.md` | Guide candidate | Operations | Reconcile with the main runbook |
| `docs/live/DEBUGGING_RUNBOOK.md` | Guide candidate | Operations | Verify current diagnostics and stop conditions |
| `docs/live/DASHBOARD_V3_RUNBOOK.md` | Guide candidate | Operations | Verify current dashboard contract |
| `docs/live/NORGATE_SNAPSHOT_V1.md` | Canonical specialized contract | Operations / Reference | Publish after data-source audit |
| `docs/live/LIVE_TECHNICAL_REFERENCE.md` | Canonical candidate | Reference | Audit and expose through a wrapper |
| `docs/live/LIVE_TRADING_ARCHITECTURE.md` | Canonical candidate | System / Reference | Audit; keep simplified guide separate |
| `docs/live/INSPECTOR_CONTRACT.md` | Canonical candidate | Reference | Audit current consumers |
| `docs/live/LIVE_RELEASES_FIELDS.md` | Canonical candidate | Reference | Audit against released schema |
| `docs/live/INCUBATION_FLOW.md` | Guide candidate | Operations / System | Classify current versus historical content |
| `docs/live/release_templates/README.md` | Reference candidate | Reference | Keep beside templates; audit links |
| Owner new-client Word notes | Unverified source notes | Operations / New Client Onboarding | Master checklist published; private credentials and chat links excluded |
| `docs/operations/client-onboarding.md` | Draft master guide | Operations | Publish detailed procedures one at a time; keep LIVE activation gated |
| `docs/operations/ibkr-flex-performance-setup.md` | Audited guide | Operations | Published; re-audit when the Flex CLI, task script, or IBKR portal contract changes |
| `docs/operations/vps-runtime-checklist.md` | Draft guide | Operations | Generic runtime checklist; keep Draft until the full check passes on each production VPS |

## Strategies

| Current source | Authority | Portal destination | Action |
|---|---|---|---|
| `strategies/taa_df/strategy_taa_df_btal_fallback_tqqq_vix_cash.md` | Canonical strategy specification | Strategies | Published directly through a wrapper |
| Other strategy code without a governed page | Code is current behavior | Strategies | Create one canonical specification per selected strategy |

## Research

| Current source group | Authority | Portal destination | Action |
|---|---|---|---|
| `docs/assets/papers/momentum-factor-construction-and-signal-orthogonality.pdf` | External foundational source | Library / Foundational Papers | Preserve the original PDF; use the governed reading note to separate author claims from internal evidence |
| `docs/assets/papers/crypto/crypto-tsmom-lives-in-volume-weighted-returns.pdf` | External crypto research commentary | Library / Foundational Papers / Crypto | Preserve the original PDF; keep the underlying study, timing audit, and internal replication status explicit |
| `docs/research/CAPACITY_ANALYSIS_GUIDE.md` | Research guide | Research | Audit formulas, evidence, and artifact paths |
| `docs/research/TRANSACTION_COSTS_RESEARCH.md` | Research | Research | Publish with evidence and verdict metadata |
| `docs/research/ENGINE_REALISM_*.md` | Research / decision records | Research | Separate current decisions from historical analysis |
| `docs/research/SECTOR_DISPERSION_*.md` | Preregistrations | Research | Preserve frozen specifications; link resulting artifacts |
| Pakal `pakal-research/reports/**/knowledge_record.json` | Canonical research metadata and artifact lineage | Research / Knowledge Base | Generate validated, searchable study pages; quarantine invalid records and never infer LIVE authority |
| Pakal `pakal-research/knowledge/STRATEGY_FEATURE_CATALOG.md` | Canonical cross-study feature catalog | Research / Knowledge Base / Feature Map | Preserve as source; publish a derived cross-study index |

## Historical reviews and visual artifacts

| Current source group | Authority | Portal destination | Action |
|---|---|---|---|
| `docs/reviews/**` | Historical | Archive | Preserve dates; never present as current truth without revalidation |
| `docs/live_flow_images/**` | Derived images | Archive / System assets | Keep only where an audited page still needs them |
| `docs/investor/**` | Derived presentation artifacts | Presentations / Archive | Keep separate from canonical strategy and performance facts |
| `docs/presentations/**` | Derived presentation artifacts | Presentations / Archive | Link to source evidence; do not use as governing specifications |

## Migration order

1. Foundation, standards, templates, and content map.
2. Core mental model and governing references.
3. Strategy catalog, one strategy at a time.
4. Operations, only after current code and runtime verification.
5. Research collections and derived presentations.
6. Historical review cleanup and archive labeling.
