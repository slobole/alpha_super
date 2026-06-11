# alpha_super

A live-first quantitative research and trading stack for a multi-strategy, pod-based book: an event-driven backtest engine built to make lookahead and survivorship bias hard to express, a set of research analyzers (friction, timing, risk, stress), and a deterministic live execution layer for IBKR where each pod is one strategy trading one real account.

This README is the index. It answers "where do I find X" — the content lives in the documents below.

## Start Here, By Goal

| You want to... | Go to |
|---|---|
| Understand the house rules before touching code | [QUANT_PHILOSOPHY.md](QUANT_PHILOSOPHY.md), then [ASSUMPTIONS_AND_GAPS.md](ASSUMPTIONS_AND_GAPS.md) |
| Run research: backtests, analyses, portfolios | [COMMANDS.md](COMMANDS.md), or the Bench web UI (`uv run python -m alpha.bench`) |
| Operate live trading | [LIVE_START_HERE.md](LIVE_START_HERE.md) |
| Diagnose a live red alert | [docs/live/DEBUGGING_RUNBOOK.md](docs/live/DEBUGGING_RUNBOOK.md) |
| Work on the code as an AI agent | [CLAUDE.md](CLAUDE.md) (Claude Code) / [AGENTS.md](AGENTS.md) (Codex) |

## Repository Map

```text
alpha/engine/      backtest engine: Strategy base, runner, orders, metrics, portfolio,
                   friction/timing/risk/stress analyzers
alpha/live/        live trading layer: runner, scheduler, broker adapters, state stores,
                   reconciliation, dashboard_v3, ops watchdog
alpha/bench/       Bench — local research control panel (web UI)
alpha/data/        auxiliary data loaders (FRED, Kenneth French)
data/              Norgate loaders (production data source) + snapshot store
strategies/        concrete Strategy subclasses by family (dv2, taa_df, momentum, ...)
                   + run_* entry scripts
portfolios/        multi-pod portfolio YAML configs
scripts/           Norgate server/client tooling, live ops watchdog, research utilities
tests/             pytest suite (engine semantics, live layer, strategies, data)
docs/              runbooks, references, research notes, dated reviews
results/           run artifacts (gitignored)
```

## Document Index

### Doctrine — read first

| Document | Purpose |
|---|---|
| [QUANT_PHILOSOPHY.md](QUANT_PHILOSOPHY.md) | House doctrine: causality, execution realism, pod model, what good quant work looks like. |
| [ASSUMPTIONS_AND_GAPS.md](ASSUMPTIONS_AND_GAPS.md) | The gaps register: every known realism limit, its impact, mitigation, and status. |
| [docs/ai/KARPATHY_GUIDELINES.md](docs/ai/KARPATHY_GUIDELINES.md) | Engineering guardrails: think before coding, simplicity first, surgical changes. |

### Operations

| Document | Purpose |
|---|---|
| [COMMANDS.md](COMMANDS.md) | CLI cheat sheet for everything: strategy runs, analyses, portfolios, web panels, Norgate, live ops. |
| [LIVE_START_HERE.md](LIVE_START_HERE.md) | Live trading TL;DR: the mental model, the five rules, safe first commands. |
| [docs/live/LIVE_RUNBOOK.md](docs/live/LIVE_RUNBOOK.md) | Full live operator guide: Norgate server/client setup, command semantics, manual vs automatic flow, recovery. |
| [docs/live/DEBUGGING_RUNBOOK.md](docs/live/DEBUGGING_RUNBOOK.md) | Red-alert funnel: Discord → dashboard → doctor → artifacts → logs. |
| [docs/live/DASHBOARD_V3_RUNBOOK.md](docs/live/DASHBOARD_V3_RUNBOOK.md) | Dashboard V3 deploy recipe. |
| [docs/live/LIVE_USER_SETUP_QUICK.md](docs/live/LIVE_USER_SETUP_QUICK.md) | Create a new pod manifest (one pod = one strategy = one IBKR account route). |
| [docs/live/release_templates/README.md](docs/live/release_templates/README.md) | Tracked example pod YAMLs to copy from. |

### Reference

| Document | Purpose |
|---|---|
| [docs/live/LIVE_TECHNICAL_REFERENCE.md](docs/live/LIVE_TECHNICAL_REFERENCE.md) | Implementation truth: core objects (DecisionPlan/VPlan), timing, broker contract, state stores. |
| [docs/live/LIVE_TRADING_ARCHITECTURE.md](docs/live/LIVE_TRADING_ARCHITECTURE.md) | Design rationale: the decision vs execution split. |
| [docs/live/LIVE_RELEASES_FIELDS.md](docs/live/LIVE_RELEASES_FIELDS.md) | Release YAML field-by-field reference and validation rules. |
| [docs/live/INCUBATION_FLOW.md](docs/live/INCUBATION_FLOW.md) | SIM-ledger rehearsal mode: what incubation proves and what it does not. |
| [docs/live/INSPECTOR_CONTRACT.md](docs/live/INSPECTOR_CONTRACT.md) | Read-only health-check contract behind the watchdog and dashboard verdicts. |
| [docs/live/NORGATE_SNAPSHOT_V1.md](docs/live/NORGATE_SNAPSHOT_V1.md) | Norgate artifact server and client snapshot sync architecture. |
| [alpha/bench/README.md](alpha/bench/README.md) | Bench research control panel: what it does and deliberately does not do. |

### Research notes and dated reviews

| Document | Purpose |
|---|---|
| [docs/research/TRANSACTION_COSTS_RESEARCH.md](docs/research/TRANSACTION_COSTS_RESEARCH.md) | Slippage and commission model analysis behind the current cost defaults. |
| `docs/reviews/` | Dated architecture/research review snapshots. Historical records of past analysis — not current guidance. |

Strategy-specific notes live next to the strategy file (for example `strategies/taa_df/strategy_taa_df_btal_fallback_tqqq_vix_cash.md`).

## Architecture Sketch

![System design](system_design.png)

The sketch above is the original high-level design (2025). The current authoritative architecture description is the [Architecture section of CLAUDE.md](CLAUDE.md#architecture); for the live layer, [docs/live/LIVE_TECHNICAL_REFERENCE.md](docs/live/LIVE_TECHNICAL_REFERENCE.md).
