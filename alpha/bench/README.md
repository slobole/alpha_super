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
  from `results/`, and the latest saved report rendered natively inside the
  unified BENCH workspace. Static charts, tables and prose are preserved while
  active HTML is stripped; **Open artifact** remains the exact source document.
  Each run row shows the
  backtest window and capital **as the runner recorded them** in `run_info.json`;
  an analysis that wrote no window shows `—` rather than implying full history.
  Without this, two runs of the same strategy over different windows are
  indistinguishable in the table.
- **Compare** — tick 2–5 rows in the Tested table to see their latest vanilla
  metrics side by side, read straight from each run's `summary.json` with
  nothing recomputed. When the runs' backtest windows differ, the page says so
  before the numbers: metrics measured over different periods are not directly
  comparable, and that disagreement must never be left for the operator to
  notice on their own.
- **Research** — result folders under `results/research/strategy/` that no
  strategy page can reach: sweeps, universe comparisons, and diagnostics
  written under names that match no `strategy_*.py` file. The work already
  ran; this page makes it visible instead of leaving it stranded on disk.
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

Bench renders in the `desk` signature variant: pure white, mono figures against
a grotesque for prose, hairline containers, and colour spent only on
machine-readable state — a link, an active tab, a PASS, a SKIP, a FAIL. The
equity curve stays ink and the benchmark stays grey. The rule that keeps this
from drifting into decoration: if a value cannot be PASS/SKIP/FAIL or
navigable, it renders in ink.

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

## One style, two densities

There is no style switcher. Bench previously offered `swiss`, `blueprint` and
`journal` alongside the house look — a design search kept alive past its
usefulness. A single-operator research console gains nothing from a menu of
identities, and the switch could leave the console and the report embedded
inside it disagreeing about which palette was active.

What remains is a **Density** control in the top bar: `Working` and
`Presentation`. It scales type and row height only, for showing the console on
a projector, and writes one display-only cookie; nothing reaches the server or
a launched job. It never gates content — both densities render identical
markup, so what you rehearse is what the room sees.

There is **one palette**, and it lives in `alpha/engine/theme.py`. `bench.css`
is written directly against the report's own token names (`--color-ink`,
`--font-figure`, …), and `build_bench_theme_css()` emits it using the exact
same `:root` builder the report layout uses. `alpha.engine.report` renders
under the same `desk` variant by default, so the console and the artifacts
embedded in it agree.

Charts are rasterised at render time, so the variant is baked into every saved
PNG. Artifacts written under a previous variant keep it for good; only a fresh
run adopts a change. Override with `ALPHA_REPORT_VARIANT_STR` if you need the
legacy `current` dashboard.

`bench.css` keeps a `:root` of its own holding the desk values, purely so the
console still renders correctly with no theme block injected. That is a second
copy of the palette, so a test pins every token in it to the generated value —
add a colour to the signature palette, never to the stylesheet.

For the complete Capacity workflow and report-reading guide, see
[`docs/research/CAPACITY_ANALYSIS_GUIDE.md`](../../docs/research/CAPACITY_ANALYSIS_GUIDE.md).
