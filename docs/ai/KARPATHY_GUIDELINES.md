# Karpathy Guidelines

Source: adapted from `forrestchang/andrej-karpathy-skills`
Original repository: `https://github.com/forrestchang/andrej-karpathy-skills`
License: MIT

Purpose: keep Andrej Karpathy's coding observations available inside this repository without requiring external plugin files, `.cursor` rules, `.claude-plugin` metadata, or any other upstream repository layout.

## Guiding Objective

Use these guardrails as a simple optimization target:

```text
engineering_quality_score
= correctness
+ auditability
+ clarity
- silent_assumptions
- unnecessary_complexity
```

For code changes, prefer the smallest valid delta:

```text
minimal_change_set = { line_i | line_i directly supports the requested outcome }
```

These guidelines are additive. They do not replace the house quant doctrine. If there is tension, causal correctness, execution realism, and the repo's quantitative rules win.

## 1. Think Before Coding

Do not silently assume semantics.

- State assumptions explicitly before implementation when the behavior is ambiguous.
- If multiple interpretations exist, surface them instead of choosing one silently.
- If a simpler approach exists, say so.
- If something is unclear, stop and name the uncertainty.

## 2. Simplicity First

Prefer the minimum code that solves the actual problem.

- Do not add features beyond the request.
- Do not add abstractions for single-use code.
- Do not add configurability that is not needed now.
- Do not add defensive branches for scenarios the system cannot actually reach.

Simple code is a control on quant risk:

```text
implementation_risk increases as hidden_state + hidden_assumptions + unnecessary_branches increase
```

## 3. Surgical Changes

Touch only the code required by the task.

- Do not refactor adjacent code unless the task requires it.
- Match the existing style unless the task explicitly changes it.
- If you notice unrelated dead code, mention it instead of deleting it.
- Remove only the orphans created by your own edits.

Audit rule:

```text
for each changed_line:
    changed_line must trace to the user request or a required verification step
```

## 4. Goal-Driven Execution

Translate vague requests into verifiable success criteria.

- "Fix the bug" -> write or identify a failing reproduction, then make it pass.
- "Add validation" -> define invalid inputs and prove they are rejected.
- "Refactor X" -> preserve behavior and verify before/after equivalence where practical.

Operational loop:

```text
goal -> implementation -> verification -> iterate until verification passes
```

## Project Merge Notes

- Keep these guidelines local to this repository.
- Do not require the upstream repository structure to use them here.
- Do not assume Claude- or Cursor-specific files are present.
- When working on quant logic, combine these guardrails with strict `Domain_Type` naming, explicit formulas where they improve auditability, and `*** CRITICAL***` comments on sensitive time-series operations.
