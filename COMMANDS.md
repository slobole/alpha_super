# COMMANDS — CLI Cheat Sheet

Quick reference for the main shell commands in this repo. All commands run from
the repo root and use `uv` (Python 3.12). Flags shown are the useful ones — add
`--help` to any script for the full list.

**Jump to:**
[0. Setup](#0-setup--basics) ·
[1. Run one strategy](#1-run-one-strategy) ·
[2. Research (all-in-one)](#2-research-orchestration--the-main-one) ·
[3. Single analyses](#3-single-analyses-frictiontimingriskcrisisstress) ·
[4. Variant suites](#4-variant-suites-compare-many-configs) ·
[5. Portfolios](#5-portfolios-multi-pod) ·
[6. Control panels (web UI)](#6-control-panels-web-ui) ·
[7. Norgate data snapshots](#7-norgate-data-snapshots) ·
[8. Live trading ops](#8-live-trading-ops) ·
[9. Dev / utility](#9-dev--utility)

> **Naming:** `<name>` = strategy module name (e.g. `strategy_mr_dv2`), a `.py`
> path, or a dotted import path. `<key>` = a *supported* strategy key — run the
> script with `--help` to list valid keys. `<yaml>` = a file under `portfolios/`.

---

## 0. Setup & basics

| Command | What / when |
|---|---|
| `uv sync` | Install/refresh all dependencies. Run once after clone or after `pyproject.toml` changes. |
| `uv run jupyter notebook` | Launch Jupyter for interactive research. |
| `uv run pytest` | Run the test suite. Use before committing. |
| `uv run python <path>` | Generic way to run any script below. |

---

## 1. Run one strategy

**`strategies/run_strategy.py`** — single backtest of one strategy, saves the report.

```bash
uv run python strategies/run_strategy.py strategy_mr_dv2
```
Useful flags: `--no-save` (don't write artifacts), `--dry-run` (just import-check
the module), `--output-dir results`, `--strategy-kwarg KEY=VALUE` (repeatable).

---

## 2. Research orchestration — *the main one*

**`scripts/research/run_strategy_analysis.py`** — runs a strategy through several
analyses in one shot. This is the command Bench's run buttons call. **Start here**
for a full picture of a strategy.

```bash
# Default = vanilla + friction + timing
uv run python scripts/research/run_strategy_analysis.py strategy_mr_dv2

# Pick a subset (repeat --analysis)
uv run python scripts/research/run_strategy_analysis.py strategy_mr_dv2 \
  --analysis vanilla --analysis risk

# Everything, and don't stop on a failure
uv run python scripts/research/run_strategy_analysis.py strategy_mr_dv2 \
  --analysis vanilla --analysis friction --analysis timing \
  --analysis risk --analysis stress --keep-going
```
`--analysis` choices: `vanilla` · `friction` · `timing` · `risk` · `stress`
(default: the first three). Other flags: `--no-save`, `--keep-going`,
`--output-dir results`.

---

## 3. Single analyses (friction / timing / risk / crisis / stress)

Use these when you want just one lens (or finer control than section 2 gives).

| Command | What / when |
|---|---|
| `uv run python strategies/run_friction_analysis.py <name>` | Execution friction / slippage / auction cost for one strategy. |
| `uv run python scripts/research/execution_timing_analyzer.py <dotted.module>` | Entry/exit timing matrix. Takes a **dotted module path**, not a bare name. Tune with `--entry-timing` / `--exit-timing` (repeatable). |
| `uv run python strategies/run_risk_analysis.py <name>` | Stationary-bootstrap risk / confidence intervals. Flags: `--simulation-count 500`, `--block-length` (repeatable, default 21/63/126), `--confidence-level 0.95`. |
| `uv run python strategies/run_crisis_replay.py <key>` | Replay a strategy across historical crisis windows. Flags: `--show-progress`, `--no-save`. |
| `uv run python strategies/run_stress_test.py <key>` | Performance in the run-up *before* crises. Flag: `--launch-offset` (repeatable, default 5/21/42/63 bars). |

Examples:
```bash
uv run python strategies/run_risk_analysis.py strategy_mr_dv2 --simulation-count 500
uv run python scripts/research/execution_timing_analyzer.py \
  strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash
uv run python strategies/run_stress_test.py <key> --launch-offset 5 --launch-offset 21
```

---

## 4. Variant suites (compare many configs)

Pre-wired sweeps that compare a family of variants and write a comparison report.
Most take **no required args** — just run them.

| Command | Compares |
|---|---|
| `uv run python strategies/momentum/run_smooth_trend_variant_suite.py` | Smooth-trend variants across universes / score modes / VIX scaling. Flags: `--universes sp500,mid400`, `--capital-base`, `--include-russell`. |
| `uv run python strategies/momentum/run_ndx_vxn_roc_variant_suite.py` | NDX momentum ROC windows + VIX scaling. Flags: `--focused-atr-grid`, `--atr-windows 20,63,126`. |
| `uv run python strategies/taa_df/run_taa_df_fallback_variant_suite.py` | TAA "Defense First" across fallback assets (SPY/QQQ/TQQQ…). Flag: `--save-results`. |
| `uv run python strategies/taa_df/run_taa_df_fallback_vix_cash_variant_suite.py` | …with VIX-cash defensive fallbacks. |
| `uv run python strategies/taa_df/run_taa_df_fallback_vix_cash_multi_rv_variant_suite.py` | …with multi-lookback (regime-dependent) VIX-cash fallbacks. |

There are also dedicated crisis-replay runners for specific strategies, e.g.
`strategies/momentum/run_crisis_replay_strategy_mo_atr_normalized_ndx.py` and
`strategies/taa_df/run_crisis_replay_taa_df_btal_fallback_tqqq_vix_cash.py`
(flags: `--show-progress`, `--no-save`).

---

## 5. Portfolios (multi-pod)

Two schemas, two runners. Bench's Build button routes to the right one.

```bash
# A) Combine already-computed strategy pickles into a book (fast, read-only math)
uv run python strategies/run_portfolio.py portfolios/multipod.yaml
uv run python strategies/run_portfolio.py portfolios/multipod.yaml --capital 200000

# B) Fresh multi-pod backtest from scratch (runs each pod, can parallelize)
uv run --python 3.12 python strategies/run_portfolio_manager.py portfolios/current_book_fresh.yaml
```
`run_portfolio.py` flags: `--name`, `--capital`.
`run_portfolio_manager.py` flags: `--max-workers 1` (serial/debug), `--no-save`,
`--show-display`.

---

## 6. Control panels (web UI)

Local Flask consoles. Both bind to `127.0.0.1` only (single operator).

| Command | What |
|---|---|
| `uv run python -m alpha.bench` | **Bench** research console → http://127.0.0.1:8765 . Lists strategies/portfolios, one-click runs, job logs. Flags: `--port 9000`, `--skip-env-file`. |
| `uv run python -m alpha.live.dashboard_v3` | **Dashboard V3** live operator console → http://127.0.0.1:8080 . Flags: `--port`, `--skip-env-file`. |

---

## 7. Norgate data snapshots

Server exports point-in-time snapshots; clients sync them. "Doctor" scripts
diagnose setup. Most paths/IDs default from env (`config.env`).

| Command | Role |
|---|---|
| `uv run python scripts/export_norgate_snapshot.py --snapshot-root <path> --profile <profile>` | Export a snapshot locally (on the Windows Norgate node). `--profile` is **required** (e.g. `norgate_eod_sp500_pit`). |
| `.\scripts\start_norgate_server.cmd` | Start the private Norgate snapshot API on the Windows Norgate node and run the server doctor. |
| `uv run python scripts/serve_norgate_snapshot_api.py --host <ip> --port 8787` | Serve snapshots over HTTP to clients. |
| `uv run python scripts/sync_norgate_snapshots_api.py --api-url http://<host>:8787` | Download required snapshots on a client. Flag: `--overwrite`. |
| `uv run python scripts/doctor_norgate_server.py` | Health-check the server (disk, export, API). |
| `uv run python scripts/doctor_norgate_client.py` | Health-check a client (env, API, sync, reads). |

---

## 8. Live trading ops

| Command | What |
|---|---|
| `.\scripts\collect_vps_debug_bundle.ps1 -Mode live -PodId <pod_id>` | Collect a redacted baseline VPS debug bundle under `results/vps_debug_bundles/` and zip it for trusted review. Does not tick, submit, reconcile, or EOD snapshot. Add `-IncludeRunnerDetails` for status/plan views that can write diagnostic job metadata, `-IncludeDoctor -DoctorBrokerClientId <unused_id>` for doctor, or `-ReleaseManifestPath <yaml> -IbkrProbeClientId <unused_id>` for the separate IBKR probe. |
| `uv run python scripts/live_debug/ibkr_connectivity_probe.py` | Test the IBKR API connection + dump account snapshot (cash/positions/orders). Flags: `--release-manifest-path <yaml>` (auto-configures), `--port 7497`, `--json`. |
| `uv run python -m alpha.live.dashboard_v3` | Live operator console (see [section 6](#6-control-panels-web-ui)). |
| `uv run python -m alpha.live.runner doctor --mode paper --pod-id <pod_id>` | **First check when something feels wrong.** Produces the live health verdict: release, config, data gate, DecisionPlan/VPlan, broker/reconcile state. Add `--json` for raw detail. |
| `uv run python -m alpha.live.runner status --mode paper --pod-id <pod_id>` | Read current sleeve state without submitting orders. |
| `uv run python -m alpha.live.scheduler_service next_due --mode paper --pod-id <pod_id>` | Show what the scheduler expects to do next, and when. |
| `uv run python -m alpha.live.runner tick --mode paper --pod-id <pod_id>` | Run one live-cycle tick manually: snapshot gate -> DecisionPlan -> VPlan -> optional submit/reconcile depending on state/config. |
| `uv run python -m alpha.live.scheduler_service run_once --mode paper --pod-id <pod_id>` | Scheduler-aware single pass. Useful when debugging due-time logic. |
| `uv run python -m alpha.live.scheduler_service serve --mode paper --pod-id <pod_id>` | Keep one pod's scheduler running. Normal VPS service command. |
| `uv run python -m alpha.live.runner show_decision_plan --mode paper --pod-id <pod_id>` | Inspect latest DecisionPlan before trusting the order plan. |
| `uv run python -m alpha.live.runner show_vplan --mode paper --pod-id <pod_id>` | Inspect latest VPlan before submission. |
| `uv run python -m alpha.live.runner submit_vplan --mode paper --pod-id <pod_id> --vplan-id <id>` | Submit a reviewed VPlan manually. Keep first live/paper cycles manual unless explicitly enabling auto-submit. |
| `uv run python -m alpha.live.runner post_execution_reconcile --mode paper --pod-id <pod_id>` | Reconcile broker fills/positions after submission. |
| `uv run python -m alpha.live.runner eod_snapshot --mode paper --pod-id <pod_id>` | Record end-of-day broker cash, positions, and NetLiq. |
| `uv run python -m alpha.live.runner execution_report --mode paper --pod-id <pod_id>` | Summarize execution/fill evidence for the current sleeve. |
| `uv run python -m alpha.live.runner compare_reference --mode paper --pod-id <pod_id>` | Compare live/paper/incubation state to a reference backtest/pickle. Flags: `--reference-strategy-pickle <path>`, `--html`, `--output-dir <dir>`. |
| `uv run python -m alpha.live.runner export_trade_sheet --mode paper --pod-id <pod_id>` | Export the latest DecisionPlan + VPlan as an xlsx trade sheet (Orders / Decision / Context tabs) for manual execution or pre-trade review. Read-only. Writes `results/trade_sheets/<mode>/<pod_id>/`. Flags: `--vplan-id <id>`, `--output-path <file>`. Also available as a download link in the Dashboard V3 pod panel. |
| `uv run python scripts/live_ops_watchdog.py --json` | Scheduled VPS watchdog: build the Inspector report, persist `alpha/live/logs/ops_report_latest.json`, fire red-transition Discord alerts, ping the dead-man switch. Register every 5 min via `.\scripts\setup_live_ops_watchdog_task.ps1`. |

For incubation rehearsal, replace `--mode paper` with `--mode incubation`.
Omit `--pod-id` only when you intentionally want the all-pod fan-out supported
by that command. If you use `--db-path`, keep the same DB path across `serve`,
`status`, `next_due`, `tick`, `show_decision_plan`, `show_vplan`, and reconcile.

Manual review flow:

```bash
uv run python -m alpha.live.runner doctor --mode paper --pod-id <pod_id>
uv run python -m alpha.live.runner tick --mode paper --pod-id <pod_id>
uv run python -m alpha.live.runner show_decision_plan --mode paper --pod-id <pod_id>
uv run python -m alpha.live.runner show_vplan --mode paper --pod-id <pod_id>
uv run python -m alpha.live.runner submit_vplan --mode paper --pod-id <pod_id> --vplan-id <id>
uv run python -m alpha.live.runner post_execution_reconcile --mode paper --pod-id <pod_id>
uv run python -m alpha.live.runner eod_snapshot --mode paper --pod-id <pod_id>
```

### Live OPS Watchdog (24/7 monitoring)

Runs on each VPS via Windows Task Scheduler: every 5 min it builds the Inspector
report, persists it, fires Discord on a red transition, and pings the
healthchecks.io dead-man switch last. Full walkthrough:
`docs/live/LIVE_RUNBOOK.md`. Debugging a red alert: `docs/live/DEBUGGING_RUNBOOK.md`.

**One-time setup (per VPS)**

```powershell
# Prereqs (Discord webhook, healthchecks.io check, config.env URLs): see the
# runbook's "Live OPS Watchdog (scheduled)" section. Then register the task:
.\scripts\setup_live_ops_watchdog_task.ps1 -Mode live
```

**Operate & verify**

```powershell
uv run python scripts\live_ops_watchdog.py --mode live --json  # run by hand, full result
Start-ScheduledTask   -TaskName AlphaLiveOpsWatchdog           # run the task now
Get-ScheduledTaskInfo -TaskName AlphaLiveOpsWatchdog           # LastTaskResult: 0 ok / 1 red / 2 fatal
Get-Content alpha\live\logs\ops_report_latest.json            # latest persisted report
```

**Dead-man drill (prove silence alerts you)**

```powershell
# Instant alert-delivery check: fail then recover (paste your real ping URL)
Invoke-RestMethod -Uri "https://hc-ping.com/<uuid>/fail" -Method Post
Invoke-RestMethod -Uri "https://hc-ping.com/<uuid>"      -Method Post
# Real silence test:
Disable-ScheduledTask -TaskName AlphaLiveOpsWatchdog          # stop pinging
#   wait ~20 min (period 5 + grace 15) -> healthchecks alerts you
Enable-ScheduledTask  -TaskName AlphaLiveOpsWatchdog          # MUST re-enable
Start-ScheduledTask   -TaskName AlphaLiveOpsWatchdog
```

**Remove / re-scope**

```powershell
.\scripts\setup_live_ops_watchdog_task.ps1 -Unregister        # remove the task
.\scripts\setup_live_ops_watchdog_task.ps1 -Mode live         # re-register (live only)
```

Severity → behavior: `green`/`yellow`/`gray` = success ping, no page · `red` =
Discord + `/fail` ping · silence = healthchecks alerts. Why `-Mode live` scoping
matters: [LIVE_RUNBOOK.md](docs/live/LIVE_RUNBOOK.md#live-ops-watchdog-scheduled).

---

## 9. Dev / utility

| Command | What / when |
|---|---|
| `uv run python scripts/review/triage.py --base origin/main` | Classify changed files by impact tier; suggests tests/review agents. Use before review. |
| `uv run python scripts/archive_research_results.py --dry-run` | Move old `results/` folders into a timestamped archive. `--dry-run` previews first. |
| `uv run python scripts/benchmark_fast_indicators.py` | Benchmark reference vs Numba indicators (DV2/QPI). No args. |
| `uv run python scripts/research/export_finhacker_sp500_top20_market_cap.py` | Scrape S&P 500 top-20 market-cap history. Flags: `--refresh`, `--annual-only`. |
