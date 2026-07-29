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

- **Strategies** — every `strategies/**/strategy_*.py`, with a WIRED badge for
  the live/supported pods (read from `SUPPORTED_STRATEGY_IMPORT_TUPLE` in
  `alpha/live/release_manifest.py`). The catalog separates evidence from absence
  of evidence: strategies with at least one mapped run render as dense table
  rows sortable by CAGR / Sharpe / Max DD / trades / last run (a strategy
  missing a figure always sinks to the bottom — no number is not a score of
  zero), while the majority that have never been run are folded into a
  collapsed section, since "never run" is a different state from "ran and
  failed". Family filters stack (OR within families, AND with the Wired/Recent
  toggles), search covers both sections and opens the fold when its only hits
  are in there, and Momentum is split into focused sub-families. Drop a new
  strategy file in and it appears automatically.
- **Strategy detail** — one-click run buttons (Vanilla / Capacity / Timing / Risk
  / Stress, plus *Standard* = V+C+T and *Full* = all five), the full run history
  from `results/`, and the latest report embedded inline. Each run row shows the
  backtest window and capital **as the runner recorded them** in `run_info.json`;
  an analysis that wrote no window shows `—` rather than implying full history.
  Without this, two runs of the same strategy over different windows are
  indistinguishable in the table.
- **Portfolios** — the books under `portfolios/*.yaml` (both the simple
  `run_portfolio.py` schema and the richer `run_portfolio_manager.py` schema),
  with a Build button routed to the correct runner.
- **Jobs** — a live view of the background runs Bench launched, with status,
  elapsed time, exit code, queue position, and streaming logs. Click a row for
  its log; a job that passed links straight to the report it produced. **Cancel**
  tears down the whole process tree — the analysis runner spawns the actual
  backtest as its own child, so killing only the launched parent would leave the
  real work running with nothing left to reach it. A cancelled job is recorded
  as `cancelled`, never `failed`: the kill's exit code is not a verdict on the
  strategy, and whatever artifacts the run had already written stay on disk.

### Running with parameters

The strategy page can override `run_variant` keyword arguments, forwarded as
`--strategy-kwarg KEY=VALUE`. Fields are read from each strategy's **own**
`run_variant` signature (parsed with `ast`, never assumed), because the kwargs
are not uniform across the catalog and `run_strategy.py` raises on one the
target does not declare. Only scalar parameters are offered — a value has to
survive a command line, so a `pricing_data_df` cannot be a form field.

> **The parameters reach Vanilla and Risk only.** Capacity, Timing and Stress
> build their commands without forwarding `--strategy-kwarg`, so they always run
> the strategy's own defaults. Choosing a custom window together with those
> produces a windowed run and a full-history run written side by side under one
> job and one timestamp, with nothing in either artifact recording that they
> disagree. Bench states this at the point of choice, and a test asserts the
> warning against the runner's actual command builder rather than a
> hand-maintained list — so if forwarding ever changes, the test fails instead
> of the warning quietly becoming false.

Overrides are stamped into the job label, so the Jobs table never shows two runs
of one strategy over different windows as identical.

## How a run button maps to a command

| Button | Command |
|---|---|
| Vanilla | `python scripts/research/run_strategy_analysis.py <module> --analysis vanilla` |
| Full | `… --analysis vanilla --analysis capacity --analysis timing --analysis risk --analysis stress --keep-going` |
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

Tests live in `tests/test_bench.py`, and the palette contract below is guarded
by `tests/test_theme_no_hardcoded_colors.py`.

## Look and feel

Bench renders in the `swiss` signature variant: white page, one grotesque sans
for everything, heavy black rules, and a single red. The red is the page's only
colour and is deliberately shared by the active nav item, the WIRED badge, and
negative figures — gains stay ink, because only the thing that demands
attention gets colour. Code-shaped content (logs, commands, paths) keeps a
monospace face regardless of the variant.

Strategies are identified by their file stem (`strategy_mr_dv2`), not a
prettified rename — the stem is the identity the results tree, the logs, and
the YAML pods already use, so the console uses it verbatim in cards, page
titles, and job labels.

The page opens with a masthead rather than a toolbar: the title set large over
a standing strapline, closed by the heavy rule. Only the navigation line below
it is sticky, so a 230-entry catalog keeps its navigation without a tall header
trailing down the page. Table column headers are sticky too, parked directly
under that rule — without them you lose which figure is Sharpe and which is
Max DD after the first screen of a long run history.

There is a print stylesheet: controls, toolbars and embedded report frames drop
out, column headers repeat on every page, and rows and cards do not break across
a page boundary, so any catalog or run history prints as a sheet you can mark up.
It also redeclares the palette tokens to plain black-on-white — paper is a
physical medium, not a signature variant, and without that override a dark
variant would print its light foreground onto white and come out near-blank.

## Switching style

The **Style** control in the footer switches the console between signature
variants — `swiss` (default), `blueprint` (a dark cyanotype drawing sheet,
monospace throughout), and `journal` (warm paper, serif). It writes one
display-only cookie; nothing reaches the server or a launched job.

It restyles **Bench only**. Reports are baked at render time by
`alpha.engine.report`, so an already-generated `report.html` keeps whatever
variant produced it — switching here can leave the console and an embedded
report disagreeing until that report is re-rendered. The report variant is set
separately by `ALPHA_REPORT_VARIANT_STR`.

There is **one palette**, and it lives in `alpha/engine/theme.py`. `bench.css`
is written directly against the report's own token names (`--color-ink`,
`--font-figure`, …), and `build_bench_theme_css()` emits the active signature
variant using the exact same `:root` builder the report layout uses. Switching
signature variant therefore moves both surfaces at once.

`bench.css` keeps a `:root` of its own holding the swiss values, purely so the
console still renders correctly with no theme block injected. That is a second
copy of the palette, so a test pins every token in it to the generated value —
add a colour to the signature palette, never to the stylesheet.

For the complete Capacity workflow and report-reading guide, see
[`docs/research/CAPACITY_ANALYSIS_GUIDE.md`](../../docs/research/CAPACITY_ANALYSIS_GUIDE.md).
