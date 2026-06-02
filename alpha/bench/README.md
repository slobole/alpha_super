# Bench — research control panel

Bench is a small local web UI that centralizes the strategy research loop you
otherwise drive from the command line. It is **read-mostly and back-end light**:
every heavy operation is delegated to a script that already exists, so Bench only
discovers, reads, and launches.

```bash
uv run python -m alpha.bench            # http://127.0.0.1:8765
uv run python -m alpha.bench --port 9000
uv run python -m alpha.bench --skip-env-file   # don't auto-load config.env
```

It binds to `127.0.0.1` only — a single-operator console, not a service.

## What it does

- **Strategies** — every `strategies/**/strategy_*.py`, with a ★ WIRED badge for
  the live/supported pods (read from `SUPPORTED_STRATEGY_IMPORT_TUPLE` in
  `alpha/live/release_manifest.py`). Search, filter by family, and see the latest
  vanilla CAGR / Sharpe / Max DD per strategy. Drop a new strategy file in and it
  appears automatically.
- **Strategy detail** — one-click run buttons (Vanilla / Friction / Timing / Risk
  / Stress, plus *Standard* = V+F+T and *Full* = all five), the full run history
  from `results/`, and the latest report embedded inline.
- **Portfolios** — the books under `portfolios/*.yaml` (both the simple
  `run_portfolio.py` schema and the richer `run_portfolio_manager.py` schema),
  with a Build button routed to the correct runner.
- **Jobs** — a live view of the background runs Bench launched, with status,
  elapsed time, exit code, and streaming logs.

## How a run button maps to a command

| Button | Command |
|---|---|
| Vanilla | `python scripts/research/run_strategy_analysis.py <module> --analysis vanilla` |
| Full | `… --analysis vanilla --analysis friction --analysis timing --analysis risk --analysis stress --keep-going` |
| Build (simple) | `python strategies/run_portfolio.py <yaml>` |
| Build (manager) | `python strategies/run_portfolio_manager.py <yaml>` |

Jobs run as subprocesses with `cwd` = repo root and the inherited environment, so
they behave exactly like the same command typed in the terminal. Output streams
to `results/_bench/jobs/<job_id>.log`.

## Layout

| File | Responsibility |
|---|---|
| `catalog.py` | discover strategies (+ wired flag) and portfolios |
| `runs.py` | read the `results/` tree; link runs to strategies via metadata |
| `jobs.py` | the background job runner + status persistence |
| `app.py` | Flask routes (pages, run API, artifact serving) |
| `__main__.py` | `python -m alpha.bench` entry point |

Tests live in `tests/test_bench.py`.
