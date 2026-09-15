# Project Working Agreements

These instructions apply to every agent working in this repository. Write project instructions and documentation in English unless the owner explicitly requests another language.

## Priority and scope

- Quantitative correctness and robustness come first. Choose the simplest solution that preserves reliability, auditability, and required safeguards.
- Before a new work phase, state its goal, scope, and completion criteria. An explicit owner request or approval authorizes that phase; ask only if its scope has not been approved. Complete its necessary inspection, implementation, tests, and fixes without renewed approval for routine steps.
- Investigate technical questions using available evidence. Ask when material uncertainty remains, an approved assumption must change, or work would exceed the approved scope. Broker actions, production deployment, and capital allocation require explicit authorization for the specific operation.

## Implementation

- Make focused changes, preserve unrelated work, and follow the existing style. Prefer direct, human-readable code over unnecessary abstractions.
- Before changing strategies, data, quantitative calculations, or execution semantics, read [Quant Philosophy](QUANT_PHILOSOPHY.md) and the relevant sections of [Assumptions and Gaps](ASSUMPTIONS_AND_GAPS.md).
- Preserve approved quantitative contracts. For sensitive time-series operations, verify direction, lag, and data availability against the approved decision boundary using evidence and tests. This satisfies routine confirmation in this project; ask again only if material uncertainty remains or the contract must change.

## Verification and agents

- The primary agent owns the complete result and may delegate bounded tasks. Review agents are read-only unless explicitly authorized to edit.
- For every meaningful code change, follow the [shared verification policy](docs/ai/PROJECT_GUIDE.md#post-change-verification), including triage, required tests, risk-based reviewers, and the live-impact checklist where applicable.
- A review opinion does not replace tests or evidence. Fix material findings and complete the required checks before declaring success. Expand or repeat checks when changes, failures, or unresolved concerns justify it.

## Communication

- Start with a clear summary and restore the owner's context. Use plain language and explain necessary technical terms.
- Reply in the owner's language unless requested otherwise. Project instructions and documentation stay in English.
- Explain what changed, why, the evidence, and any decision needed. Retain relevant assumptions, parameters, data dependencies, timing, and limitations; include the required verification fields.
- When the topic changes, state the switch and briefly say what is complete and what remains open in the original task.

## Read by task

| Task | Reference |
|---|---|
| Quantitative rules, research validity, or accounting semantics | [Quant Philosophy](QUANT_PHILOSOPHY.md), then relevant [gap records](ASSUMPTIONS_AND_GAPS.md) |
| Verification requirements or technical structure | Relevant sections of [Project Guide](docs/ai/PROJECT_GUIDE.md) |
| Engineering conventions | [Karpathy Guidelines](docs/ai/KARPATHY_GUIDELINES.md) |
| Live implementation or operations | [LIVE Start Here](LIVE_START_HERE.md), then the relevant live contracts and runbooks |
| Norgate snapshot, API, or client/VPS behavior | [Norgate Snapshot Contract](docs/live/NORGATE_SNAPSHOT_V1.md) before changes |
| Repository navigation or commands | [README](README.md) and [Commands](COMMANDS.md) |

Read the references needed for the task and its affected dependencies. Documentation describes procedures; it does not grant permission to execute them.
