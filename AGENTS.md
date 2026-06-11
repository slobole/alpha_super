# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Read First

Before changing code, read these doctrine documents in this exact order:

1. `QUANT_PHILOSOPHY.md`
2. `ASSUMPTIONS_AND_GAPS.md`

These documents are authoritative for house philosophy and realism assumptions. This file remains the operational entrypoint for repo-specific coding behavior.

After the doctrine documents above, also read `docs/ai/KARPATHY_GUIDELINES.md`. This project keeps a self-contained local adaptation of `forrestchang/andrej-karpathy-skills` so the guidance is available here without requiring external plugin files, `.cursor` rules, or `.claude-plugin` metadata.

For Norgate snapshot/API/client-VPS work, also read
`docs/live/NORGATE_SNAPSHOT_V1.md` before changing live data-source,
scheduler, or dashboard behavior.

## Karpathy-Derived Engineering Guardrails

These guardrails are additive to the quant doctrine:

- **Think before coding** - state assumptions explicitly, surface ambiguity, and do not silently choose semantics when they matter.
- **Simplicity first** - prefer the minimum code that solves the actual problem; avoid speculative flexibility and unnecessary abstractions.
- **Surgical changes** - change only what the task requires and avoid drive-by cleanup in unrelated areas.
- **Goal-driven execution** - define a verifiable success condition, implement against it, and verify before declaring completion.

### Human-Readable Code Rule

Code is accepted only if a human owner can read it later without needing the agent who wrote it.

Before meaningful code changes:

- state the smallest change needed;
- name the behavior that must stay the same;
- avoid new abstractions unless they remove more complexity than they add.

Prefer plain functions, explicit steps, and familiar names. Do not write "agent architecture" that only another agent can comfortably understand.

Use the local reference file for the full text and attribution details.

## Post-Change Verification

After every meaningful code change, classify the diff by blast radius before
the final response. The user should not need to run this manually; agents must
use the local helper and then apply the matching review depth.

Run:

```bash
uv run python scripts/review/triage.py
```

By default the helper reads `git diff --name-only` and also includes untracked
non-ignored files, so brand-new scripts and tests are not missed.

To classify a proposed or explicit path set:

```bash
uv run python scripts/review/triage.py --name-only alpha/live/runner.py
uv run python scripts/review/triage.py --base HEAD
```

Tests are the hard gate. Agents are a soft review gate. If a review agent finds
a real issue that tests missed, prefer adding or tightening a regression test.
The main agent patches; review agents are read-only unless the user explicitly
requests otherwise.

### Tiers

**Tier 0 - docs, comments, isolated tooling**

- Scope: docs, comments, presentation artifacts, isolated tooling.
- Required verification: relevant tests/checks only.
- Required agents: none.
- Escalate live runbooks, operator docs, or docs that change live behavior to
  Tier 3.

**Tier 1 - research and backtest-only work**

- Scope: `strategies/**`, notebooks, research scripts, backtest-only
  experiments.
- Required verification: tests plus one quant-pitfalls agent.

**Tier 2 - engine, shared utilities, indicators, metrics**

- Scope: `alpha/engine/**`, shared data utilities, indicators, metrics,
  portfolio utilities, and execution-sensitive shared helpers.
- Required verification: tests plus quant-pitfalls, parity, and coverage
  agents.

**Tier 3 - live execution, orders, sizing, reconcile, released configs**

- Scope: `alpha/live/**`, `alpha/live/releases/**`, live runner/scheduler,
  order, reconcile, reference-price, sizing, dashboard/logging consumed by
  live, and released pod YAML/config/state contract changes.
- Required verification: tests, mandatory live-impact checklist, and parity,
  failure-modes, and coverage agents.
- Add a quant-pitfalls agent too when the live change also touches strategy,
  backtest, sizing math, reference-price semantics, or quantitative behavior.

If multiple tiers match, choose the highest tier and say which lower-tier
surfaces were also touched.

### Quant-Pitfalls Agent

This is a full quant review, not just a lookahead check. It must explicitly
check lookahead, survivorship, data mining, multiple comparisons, in-sample
contamination, target leakage, regime dependence, sample size, corporate
actions, adjustment type, cost/slippage realism, and live/backtest divergence.

### Live-Impact Checklist

For Tier 3 changes, explicitly answer:

- Order timing semantics unchanged, especially next-open execution.
- Sizing math unchanged, including `amount`, `target=True`, percent versus
  value semantics.
- Reference price source unchanged, with no silent close/open substitution.
- State files, pickle files, SQLite schemas, and config formats backward
  compatible unless intentionally migrated.
- Logging fields consumed by dashboards or runbooks still present.
- No new Windows VPS failure mode around paths, encodings, file locks,
  idempotency, or process restarts.
- Released pod YAMLs still parse and produce the same intended route/intent
  unless the change explicitly targets those semantics.

### Required Final Response Fields

After code changes, the final response must state:

- tier
- agents used
- findings fixed
- tests run
- residual risk

## Quantitative Correctness Standards

The full standards live in [CLAUDE.md](CLAUDE.md) — Quantitative Correctness
Standards, the non-negotiable rules, the architecture overview, and the command
reference. They apply to Codex identically; this file does not maintain a
second copy.

The non-negotiables, one line each (full text in CLAUDE.md):

- **Lookahead bias** — features and decisions may use only information available before the decision point; `iterate()` sees data up to `previous_bar` only.
- **Survivorship bias** — universes must use point-in-time constituent membership, never today's composition.
- **Data-mining / overfitting** — few parameters, an explainable edge, no tuning on the full period without out-of-sample validation.
- **Execution realism** — orders placed in `iterate()` fill at the next bar's open; never bypass with same-bar prices.
- **Price adjustment** — `CAPITALSPECIAL` for individual stocks; `TOTALRETURN` only for benchmark indices.
- **Statistical honesty** — full out-of-sample metrics, no cherry-picked windows; Sharpe uses a 0 risk-free rate.
- **Simplicity over complexity** — fewer justified rules beat parameter-heavy rulesets.
- **Explicit semantics** — any change to signal/order timing, execution assumptions, rebalance mapping, aggregation math, or cost modeling must state the old behavior, the new behavior, and the quantitative consequence.
- **Live pod-account mapping** — one live pod = one strategy = one linked IBKR account/subaccount route = one ledger; never two live strategies on one account route, and no pod-as-label inside a shared raw account without a first-class pod ledger.
- **Domain naming** — strict `Domain_Type` names (`price_vec`, `return_ser`, `signal_df`, `target_weight_ser`).
- **Sensitive time-series auditability** — `*** CRITICAL***` comments next to `shift()`, rolling windows, forward returns, and rebalance-date mapping.
- **Human-readable explanations** — plain-language intuition first, exact rule second, formulas only where they materially improve auditability.
- **Uncertainty handling** — when uncertain, choose the more explicit implementation and state the uncertainty.
