# Scout — research pipeline design (v1)

Date: 2026-09-29. Base: `main` @ `cb29d4f`. Author: Claude. The owner delegated every design decision
in this document to Claude ("you decide the catalog / pipe in full"). No code was changed.

**Amendment A1 (2026-09-30)** is folded into the text below. It follows an independent critique of v1
(Pakal session "Z Systems research", 2026-09-30) and the Zorro / Financial Hacker concept review from the
same session. Section 16 lists what changed and why.

This is a plan. It grants no permission to change live code, strategies, the VPS, or capital allocation.
`AGENTS.md`, `QUANT_PHILOSOPHY.md` and the verification policy in `docs/ai/PROJECT_GUIDE.md` apply to all
work here. Where this document and `QUANT_PHILOSOPHY.md` disagree, `QUANT_PHILOSOPHY.md` wins, and this
document must be fixed.

## 1. What Scout is, in one paragraph

Scout is a research pipeline that sits **in front of** the real engine. It takes a written hypothesis and
runs it through a fixed sequence of stations: registration, causality checks, indicator quality, edge
study, fast strategy backtest, overfitting and multiple-testing tests, book value, a one-time sealed
holdout ("the vault"), and forward shadow tracking. Every evaluation is written to an append-only trial
ledger, so the statistics know how many things were tried. Scout treats new ideas and existing
strategies identically: **no strategy is grandfathered**. Scout never trades, never feeds published
books or clients, and never produces a final number. Final numbers come from the real engine, after the
Scout version of the strategy has passed an identity gate against it.

The governing equation from `QUANT_PHILOSOPHY.md` is the design brief:

$$
\text{Observed performance} = \text{edge} + \text{luck} + \text{bias} + \text{implementation error}
$$

Stations S1 and the identity gate attack **bias** and **implementation error**. Stations S3–S5 and the
vault attack **luck**. Speed exists only so the luck tests (the permutation test of the whole search,
walk-forward) are affordable. Speed is never the goal.

### Non-goals (v1)

- Intraday strategies, options, crypto, ML models (see section 12).
- Shorts as a promotable result. Short legs may be *studied* with a flat borrow model and are labelled
  `NOT_EXECUTABLE_BORROW_UNMODELLED` until real borrow data exists.
- Portfolio construction research (HRP, NCO, risk budgeting). Scout judges pods; book construction is a
  separate project. Scout does judge a pod's *value to a book* (S6).
- Replacing the real engine, the Bench, or `RiskAnalysis`. Scout reuses them.

## 2. Decisions at a glance

| # | Decision | Why |
|---|---|---|
| D1 | Scout lives in `alpha/scout/` inside alpha_super | Identity gate must run against the same commit as the engine |
| D2 | `alpha/live` may never import `alpha.scout` (enforced by a test) | Research code cannot reach real money |
| D3 | One pipeline for new ideas and existing strategies ("re-audition") | No grandfathering; existing pods earn their place the same way |
| D4 | Three hypothesis classes: Event (E), Cross-sectional rebalance (X), Weights/TAA (W) | Covers every LIVE and PM_READY strategy and every study of the last month |
| D5 | Two engines: a weights engine (X, W) and a path engine (E) | X and W reduce to target weights; E needs slots, exits and stops |
| D6 | Parity mode by default: Scout reproduces the real engine, known biases included | The gate is only meaningful against the engine as it is; biases are measured separately |
| D7 | Identity gate before any Scout number is quoted for an existing strategy | Two implementations drift; the gate catches drift in CI |
| D8 | Vault = 2023-01-01 to today, sealed at the data layer, **clean only for families with no prior research on post-2022 data**; every registration also gets a true forward period that starts at its ledger timestamp | The owner has already seen 2023–2026 for most mainstream families; only data that did not exist yet is truly unseen (A1) |
| D9 | Unit of inference in edge studies = the date, not the event | Overlapping holds and same-day clustering make events far from independent |
| D10 | Append-only, hash-chained trial ledger, committed to git | Every trial counts; tampering is visible |
| D11 | Trials are counted per mechanism family, with an effective-N correction | 100 variants of one idea are not 100 independent trials |
| D12 | Multiple testing is corrected **once**, by the ledger (BY-FDR across families at S3, DSR with family N_eff at S5), not again by a fixed t ≥ 3 bar | t ≥ 3 is itself a multiple-testing haircut; stacking it on top of DSR and FDR double-counts and rejects real edges (A1) |
| D13 | Parameter choice = centre of the best plateau, never the peak | Peaks are where luck lives |
| D14 | Reported strategy numbers are the median of the rebalance-day luck band | The best offset is luck |
| D15 | Every candidate must beat T-bills in its own book slot (S6) | This is the bar every recent study used; it is the real opportunity cost |
| D16 | Capacity is judged on today's volumes, separately from small-account friction | Owner rule (tradability from today) |
| D17 | Shadow (S8) kill and confirm rules are written before shadow starts | Otherwise the forward test becomes another tuning loop |
| D18 | The statistics live in a shared `alpha/stats/` package, not inside `alpha/scout/` | Pakal, Scout and the live pod-health report use one implementation; `alpha/live` may import `alpha.stats` but never `alpha.scout` (A1) |
| D19 | S5 gates on **MCPT alone** (plain date shuffle, p ≤ 0.05; full search over the union of the family's registered grids; score = selected Sharpe minus a registered baseline on the same path; not runnable on point-in-time panels until P4, so they WATCHLIST); DSR as an exact correlation-aware null p-value is a WARN; walk-forward, PBO, CPCV and the haircut are printed diagnostics | P2 calibration (amendment 1): MCPT held 4.4-4.5% false pass with 93-95% power at Sharpe 0.79 in a varied and a near-duplicate grid; the clustered-N_eff DSR reached 11.8% false pass; requiring MCPT, DSR and walk-forward together rejected 16-26% of Sharpe-0.79 edges and 43-61% of Sharpe-0.52 edges (A1, A5) |
| D20 | A calibration study on known cases decides which tests gate, before the pipeline is built | A test earns a gate only if it separates noise and dead strategies from planted edges better than the gates already in place (A1) |
| D21 | A pod-health monitor for the live pods (Cold Blood Index with block bootstrap, plus CUSUM) is built first | It protects real money now and needs only the shared stats package; it is the S8 machinery pointed at LIVE pods (A1) |
| D22 | **Reject on evidence against, not on missing proof.** Only causality violations (S1), a wrong-sign or absent edge (S3), and a negative net result at double costs (S4) end a family. A borderline statistical result goes to `WATCHLIST` with cheap forward tracking, not to `REJECTED` | Our history has one sample; strict conjunctions of tests on it throw away real edges. Forward data after registration is clean, so borderline ideas can earn their way in over time (A2) |

## 3. Package layout

```text
alpha/scout/
  data/        panel.py (cached PIT panels), vault.py (seal), snapshot.py (data hash)
  features/    library.py (registered features), contracts.py (declared lag/basis/lookback)
  spec.py      ScoutSpec dataclass + hashing
  families/    one module per mechanism family (specs live here, with their registration)
  engines/     weights.py (X, W), path.py (E, numba), costs.py, execution.py
  gate/        identity.py (Scout vs real engine), deviations.py (known-engine-bias registry)
  ledger.py    append-only JSONL with hash chain (P0, built)
  registration.py, families.py, trials.py   S0, mechanism families, effective N (P0, built)
  stations/    s0_register.py ... s8_shadow.py
  card/        research card (HTML), reused by a Bench page
  cli.py       python -m alpha.scout <command>
alpha/stats/   shared statistics, no dependency on alpha.scout (D18). Built in P0:
               newey_west.py, bootstrap.py (same algorithm as the RiskAnalysis stationary
               bootstrap, vectorised; RiskAnalysis unchanged), permutation.py, psr_dsr.py
               (PSR, DSR, MinTRL), mcpt.py, walk_forward.py, fdr.py, health.py (Cold Blood
               Index, CUSUM). Later, as diagnostics: pbo_cscv.py, cpcv.py, spa.py
tests/test_stats_*.py, tests/test_scout_*.py   flat, like the rest of tests/
research_ledger/scout_ledger.jsonl    committed ledger (small rows only)
results/scout/                        heavy artefacts (return series, cards), gitignored, backed up
```

Reuse, do not rebuild: `alpha/engine/risk_analysis.py` (stationary bootstrap), `capacity_analysis.py`
(capacity v2 numbers), `crisis.py` and `stress_test.py` (crisis windows), `metrics.py` (metric
definitions, benchmark regression), `dv2_indicator_fast.py` and `qp_indicator_fast.py` (fast
indicators), `alpha/data/alfred_snapshot.py` (macro vintages), `alpha/data/kenneth_french_loader.py`
(factors), and the DV2 numba replica pattern in `scripts/research/dv2_deep_20260925/replica.py`.

## 4. Data layer and the vault

### 4.1 Panels

A panel is a cached, memory-mapped set of `(date × symbol)` arrays for one universe, built once from
Norgate and hashed. The hash (`data_snapshot_id`) goes into every ledger row.

| Array | Basis | Use |
|---|---|---|
| Open, High, Low, Close | `CAPITALSPECIAL` | fills, marks, level/scale-sensitive features |
| Close | `TOTALRETURN` | return-space signals and benchmarks only |
| Unadjusted close | raw | restoring nominal units at each decision (dollar ATR, share counts) |
| Volume, `Turnover` | as Norgate | liquidity; dollar liquidity comes from native `Turnover` |
| Dividend | per share | cash dividends in parity with the engine |
| Membership mask | point-in-time | `index_constituent_timeseries`; `True` only on days the symbol was a member |
| Delisting | last bar + reason | exit handling for names that disappear |

Macro inputs come only through ALFRED vintages (the value known on the decision date), never the
current FRED vintage. T-bill returns and Kenneth French factors are separate series with their own
provenance.

*** CRITICAL*** The engine's `norgate_loader` drops the last five member days of ex-members (see
`project_membership_trim_lookahead`). **Parity mode** reproduces that trim so the identity gate can pass.
**Truth mode** uses the exact membership. The trim is registered as a known engine deviation (section 7.3),
and every card shows its measured effect.

### 4.2 The vault

- **Sealed period:** 2023-01-01 to the latest data date. The seal lives in `data/vault.py`. A panel
  request for any date on or after the seal returns data truncated at the last session before the seal,
  unless the caller passes a vault-opening token.
- **Forward labels respect the seal.** A forward return that would read a bar inside the vault is `NaN`,
  so labels near the boundary cannot leak (purge at the seal).
- **Opening tokens** are issued only by station S7, once per family, and are written to the ledger
  before any data is returned. A second opening for the same family is refused.
- **Contamination flag.** The vault test of a family is marked `CONTAMINATED` when either:
  - the idea's source (paper, book, talk) was published on or after the seal date, so its author may
    have known the period; or
  - **the owner has already looked at that family on post-2022 data.** Evidence: prior ledger rows,
    `docs/research/`, `scripts/research/`, Pakal reports. When in doubt, it is contaminated.

  As of 2026-09-30 this covers most mainstream families: short-term mean reversion, momentum and
  rotation, TAA, trend and breakout, month-end flow, the Alpha101 set, IPO/ATH and the Zorro Z systems.
  A contaminated vault result is printed as a diagnostic only, and promotion requires forward evidence.
- **Forward period from registration (A1).** Data dated after a registration's ledger timestamp did not
  exist when the hypothesis was frozen. For every family it is the one period that is clean by
  construction. S7 and S8 read it automatically from the ledger. This makes the vault honest about our
  own eyes: from today, every day of new data is a clean holdout for everything registered before it.
- **In-sample windows:** US equities from 1998-01-01 (reliable PIT membership), ETFs from inception, macro
  from the first ALFRED vintage. All end on 2022-12-30.
- **Power warning.** The vault holds about 3.75 years. For a monthly strategy that is about 45
  observations, which is too few to prove anything. The vault therefore tests for *collapse*, not for
  greatness (S7 criteria), and S8 does the confirming.

## 5. Hypothesis classes and the spec

| Class | Shape | Examples today |
|---|---|---|
| **E: event** | Condition on (date, symbol) → enter next session → exit by rule, fixed horizon or stop; limited slots | DV2, QPI, HPI, sector-ETF IBS, breakouts, IPO/ATH, EOM flow |
| **X: cross-sectional rebalance** | Rank a universe on a schedule → hold top k with a weighting rule | NDX VXN, MOSAIC, sector rotation, residual momentum |
| **W: weights / TAA** | Small asset set → signals → target weights on a schedule | TAA 3x, TAA 1/N, CORE5, Compass, TFI, BTAL_QQQ, Trinity |

A strategy is written as a `ScoutSpec`: a plain, frozen dataclass.

```text
ScoutSpec
  family_id         mechanism family (section 8.3)
  hclass            E | X | W
  universe          panel name + membership rule
  features          names from the feature library (each with declared lookback, basis, lag)
  signal_fn         pure function: feature arrays -> signal arrays (no data access of its own)
  rule              class-specific: entry/exit/slots (E), rank/k/weighting (X), weight map (W)
  schedule          daily | weekly(day) | monthly(offset) | calendar rule
  execution         next_open (default) | close_moc (flagged, needs S4 timing evidence)
  sizing            equal slot | inverse vol | target weights; whole shares on/off
  costs             profile name (section 6.2)
  param_grid        the registered search space (finite, declared in S0)
```

The signal function receives only features, never raw panel access. Features are the single place where
time-series operations happen, and every feature carries a declared contract that S1 tests
automatically.

## 6. Engines and the execution contract

### 6.1 Contract (identical to the real engine, see `QUANT_PHILOSOPHY.md` "Engine Order Is Part Of The Model")

1. Features are computed from information up to the close of session `p`.
2. The decision for session `t = p + 1` is taken after the close of `p`.
3. Orders fill at `Open_t` with slippage. MOC fills at `Close_t` are available only in `close_moc` mode,
   which requires a decision-time price and is flagged on the card.
4. Dividends are credited before the open of the ex-date session, with 25% withholding, in parity with the
   engine.
5. A held symbol with no `Open_t` or `Close_t` is sold at the last valid close, commission charged.
6. `NAV_t = cash_t + Σ shares × Close_t`.

### 6.2 Costs

- **Commission:** IBKR tiered in parity mode, as the engine computes it: `max(1, 0.005 × |shares|)`.
- **Slippage:** basis points by point-in-time liquidity bucket (Turnover tercile inside the universe).
  Default 2.5 bps per side for ETFs and the top tercile, 5 bps for the middle tercile, 10 bps for the
  bottom tercile.
- **Borrow** (study only): 50 bps a year general collateral. Anything else is `NOT_EXECUTABLE`.
- **Two account modes:** `small` ($30K, whole shares, minimum commissions) and `institutional`
  ($1M and $10M, fractional sizing, participation checks). Both are always reported.

### 6.3 Engines

- **Weights engine** (`engines/weights.py`, vectorised numpy): target weights per rebalance date →
  drifting holdings between rebalances → turnover → costs → daily NAV. Serves classes X and W.
- **Path engine** (`engines/path.py`, numba): a day loop with slots, entries, exits, stops and cash,
  built from the DV2 replica pattern. Serves class E.

Speed targets on the workstation:

| Case | Target |
|---|---|
| A 25-year monthly W strategy | ≤ 50 ms per run |
| A 25-year monthly X strategy on the NDX 100 | ≤ 200 ms per run |
| A 25-year S&P 500 E strategy | ≤ 1 s per run |
| MCPT for a 100-config family with 500 permutations | overnight, parallel across cores |

## 7. The identity gate

### 7.1 What it compares

**A6 supersedes the thresholds below for parity mode:** in parity mode the gate is exact. Daily returns and daily
weights must agree within 1e-9 on every date, the trade dates must be identical, and the comparison must cover the
engine's whole run. The table below is the informational tolerance tier, used for truth mode.

For a strategy that exists in the real engine, the gate runs the real engine and the Scout spec on the
same data snapshot over full history, in parity mode, and compares:

| Check | Pass threshold |
|---|---|
| Decisions: W/X target weights per rebalance | \|Δw\| ≤ 0.5 pp on ≥ 99.5% of (rebalance, asset) cells |
| Decisions: E entries and exits | same (date, symbol, side) on ≥ 99.5% of trades |
| Daily return correlation | ≥ 0.999 |
| Annualised return difference | ≤ 5 bps |
| Max drawdown difference | ≤ 0.25 pp |
| Every mismatch | listed and explained in the gate report (no unexplained rows) |

### 7.2 When it runs

- **In CI** for every gated strategy, on any change under `strategies/`, `alpha/engine/` or `alpha/scout/`.
  A failing gate blocks the merge.
- **For new ideas at promotion (S7):** the idea is implemented in the real engine, and that
  implementation must match the Scout spec.
- **Ungated results:** a Scout result for a strategy without a passing gate is labelled `UNGATED` and
  cannot move past S6.

### 7.3 Known engine deviations

`gate/deviations.py` lists known biases in the real engine that parity mode must reproduce. Each entry
records `issue_description_str`, `expected_bias_direction_str`, `impact_level_str` and `mitigation_str`,
per `QUANT_PHILOSOPHY.md`. The first two entries are:

1. **Membership trim:** the last five member days of ex-members are dropped.
2. **Commission on split-adjusted shares:** commission is computed on split-adjusted share counts, not on
   the shares actually held at the time.

Each card runs truth mode once and prints the effect of every deviation on that strategy. A deviation
that moves a strategy's result materially goes to the engine fix list. Scout never silently "fixes" the
engine.

## 8. Registration, ledger and trial counting

### 8.1 Registration (station S0)

Nothing runs without a frozen registration. The registration is written **before any data is looked at**,
hashed, and written to the ledger:

- `hypothesis`: one sentence, falsifiable.
- `mechanism`: why this should work. Who is on the other side, and why they keep paying.
- `expected_sign_and_location`: for example, "bottom decile of the indicator beats the regime baseline".
- `hclass`, `universe`, `horizon`, `schedule`, `execution`.
- `param_grid`: finite. More than 200 configurations requires a written justification in the
  registration.
- `primary_metric`: for example, the date-clustered mean excess return at the registered horizon (S3), or
  the median-offset Sharpe (S4).
- `kill_criteria`: which station failure ends the family.
- `source`: paper, book, talk or own idea, with publication date (used for the vault contamination flag).
- `prior_trials`: required in re-audition mode (section 10).

A registration may not be edited. A change after seeing any result is a **new registration** with
`parent_id`, and its trials count toward the same family.

### 8.2 The ledger

- **Format:** `research_ledger/scout_ledger.jsonl`, append-only and committed. One row per evaluation of
  a configuration on real (non-permuted) data, plus one row per registration, station verdict, vault
  opening and gate result.
- **Row fields:** `row_id`, `prev_row_hash`, `row_hash`, `utc_ts`, `family_id`, `registration_id`,
  `spec_hash`, `code_commit`, `data_snapshot_id`, `mode` (parity or truth), `station`, `params`,
  `window`, headline metrics, `vault_touched` (bool), and the path to the return series under
  `results/scout/`.
- **Hash chain:** each row includes the hash of the previous row, and a test verifies the chain. Git
  history is the second line of tamper evidence.
- **Not trials:** permutation, bootstrap and CSCV resamples. Each of those tests is logged once, as a
  single row with its result.

### 8.3 Families and effective N

- **Families are defined by mechanism, not by indicator.** For example, "short-term mean reversion in US
  large-cap stocks" is one family whether it uses DV2, QPI, RSI(2) or IBS. The family taxonomy is a
  maintained list in `families/__init__.py`.
- **Effective number of trials:** the number of correlation clusters among the family's trial return
  series: distance `d = √((1 − ρ)/2)`, average-linkage hierarchical clustering, cut at ρ = 0.5.
  N identical trials give 1, N uncorrelated trials give N, and lopsided grids count correctly
  (80 + 5 + 5 + 5 + 5 near-identical trials give 5). This replaces the v1 silhouette-chosen cut (one fixed
  cut, no tuning) and a participation-ratio estimate tried during P0, which under-counted lopsided grids
  (90 + 10 gave 1.23) and so under-deflated. Without stored return series, every recorded trial counts as
  independent (the conservative fallback). Retro `prior_trials` are added as independent.
- **Trial variance fallback:** with fewer than two recorded trials the family's Sharpe dispersion is
  unknown; the DSR then uses the null sampling variance of a Sharpe estimate, 1 / (T − 1), so prior
  trials still deflate.
- **Frozen grid:** a trial whose configuration is not in its registration's grid is refused.
- **Global count:** a ledger report applies Benjamini–Yekutieli FDR at q = 0.05 across all families' S3
  p-values, so the whole research program is controlled, not only each family.

## 9. The pipeline: stations S0–S8

Each station returns **PASS**, **WARN** or **FAIL** per check. A WARN is printed on the card and needs a
written owner note before promotion.

**Two kinds of error cost money (A2, D22).** Passing a lucky rule loses capital; rejecting a real edge
loses the business. So a FAIL has two meanings:
- **Hard FAIL, the family stops:** only the S1 causality checks, an S3 edge with the wrong sign or none at
  all (mean excess return ≤ 0 or q > 0.5), and S4 net Sharpe ≤ 0 at double costs.
- **Soft FAIL, the family goes to `WATCHLIST`:** any other S3, S5 or S6 threshold missed. The frozen
  configuration is tracked forward at no cost from its registration timestamp. It returns to S5–S6 when
  the forward period reaches its MinTRL, or earlier if the forward evidence is strong, and it is judged
  again on in-sample plus forward data.

P2 calibration measures both error rates: the false-pass rate on noise and dead strategies, and the
false-reject rate on planted edges. A gate set that rejects more than 30% of planted Sharpe-0.8 edges is
too strict and must be loosened. The station sequence is
enforced by the runner: a later station refuses to run until the earlier station has a ledger verdict.

### S0 — Registration

Covered in section 8.1. Hard gate: no registration, no run.

### S1 — Data and causality (automatic, all hard)

| Check | Method | Pass |
|---|---|---|
| Prefix invariance | Recompute every feature on data truncated at 50 random dates; compare with the full-history value on those dates | exact equality (float tolerance 1e-12) |
| Future corporate-action invariance | Rescale one symbol's history after date t (synthetic split); features at dates ≤ t must not change | exact equality, for every scale-sensitive feature |
| Membership PIT | Every symbol-date used is a member on that date (truth mode) | 0 violations |
| Macro vintage | Every macro value used was published on or before the decision date | 0 violations |
| Price basis role | Fills and marks use `CAPITALSPECIAL`; `TOTALRETURN` only in return-space signals | declared roles match usage |
| Coverage | Symbols, delisted share, missing-bar share per year | printed; WARN if more than 2% of a year's decision rows are missing |

### S2 — Indicator quality (Masters, *Statistically Sound Indicators*; diagnostic, WARN only)

- **Stability over time.** Quantiles (1, 5, 25, 50, 75, 95, 99%) per 5-year block, plus a
  two-sample Kolmogorov–Smirnov test between blocks. The card also prints **threshold drift**: what share
  of observations the registered threshold selects in each block. For example, "QPI < 15" should select a
  similar share in 2003 and in 2020.
- **Tails.** Share of values beyond Q1 − 3·IQR and Q3 + 3·IQR. Heavy tails get a recommendation to use a
  rank or compressed form.
- **Information content.** Mutual information between the indicator and the forward return in each block,
  against a permutation baseline.
- **Novelty.** Rank correlation with a fixed reference set of known features: 1-, 3- and 5-day return,
  12-1 momentum, 20-day volatility, NATR, size (Turnover rank) and distance from the 200-day average.
  Above 0.8 against any of them: WARN "this is probably X in disguise", and the family is re-assigned to
  that mechanism.

### S3 — Edge study (the owner's notebook method, upgraded)

**Goal:** decide whether the signal carries information *before* any strategy, cost or slot logic
exists. This station replaces the Pakal edge notebooks. It keeps their structure: regime baseline,
events against non-events inside the regime, win/loss profile, deciles, interaction heatmap and IC. It
fixes their three known problems: no PIT membership, events treated as independent, and no time
breakdown.

**Labels.** The forward return is measured from the entry price (`Open_{t+1}`) to the exit at each
horizon h ∈ {1, 2, 3, 5, 10, 20}. The registered h is the primary one. Every label is expressed as an
**excess return** over the equal-weighted return of the same regime on the same date:

$$
x_{i,t} = r_{i,t\to t+h} - \frac{1}{|R_t|}\sum_{j\in R_t} r_{j,t\to t+h}
$$

where $R_t$ is the set of regime-eligible members on date $t$. Subtracting the same-date regime mean
removes the market move of that day from the comparison.

**Unit of inference = the date.** First average the event excess returns within each date,
$\bar{x}_t = \text{mean}_i\, x_{i,t}$, and then test the time series $\{\bar{x}_t\}$:

- Newey–West t-statistic with lag = h − 1 (handles overlapping holds).
- Stationary block bootstrap 95% confidence interval (mean block length = 2h).
- **Two permutation tests** (10,000 draws):
  - *Within-date shuffle* of event labels among that date's regime members. This tests stock
    selection and keeps every market-day effect.
  - *Block-of-dates shuffle.* This tests timing.

**Diagnostics.** All are printed on the card. Some are hard gates (see the table below).

- **Deciles** of the indicator within the regime: mean excess return, t-statistic and count per decile,
  and a monotonicity score (Spearman correlation between decile index and decile mean).
- **Lag decay:** the edge when entry is delayed by 0, 1, 2, 3 or 5 sessions. A real edge decays smoothly.
  A cliff after day 0 is a warning sign (fragile or microstructure-driven).
- **Placebo:** random events matched on date count and volatility bucket, and the reversed signal. The
  placebo must be inside its own null; the real signal must be outside it.
- **Stability:** by year (share of positive years), by era (1998–2007, 2008–2015, 2016–2022), by
  volatility regime (VIX terciles), in up and down markets, and before and after the source's publication
  date.
- **Replication:** the same spec on sibling universes (S&P 500, NDX 100, Russell 1000, sector ETFs where
  relevant) and on neighbouring parameters.
- **Concentration:** the edge with the top 1% of event-dates removed, and with the crisis windows
  (2000–02, 2008–09, Feb–Apr 2020) removed.
- **Liquidity:** the edge by point-in-time Turnover tercile.
- **Cost coverage:** mean gross edge per trade divided by the round-trip cost in the liquidity bucket
  the strategy would trade.

**Hard criteria for class E and X** (all must pass):

| Criterion | Threshold |
|---|---|
| Newey–West t, date-clustered, registered horizon | ≥ 2.0, and its p-value enters the ledger-level Benjamini–Yekutieli FDR at q ≤ 0.05 (D12; A5: the only calibrated S3 test) |
| Within-date permutation p (E and X), block-of-dates p (W) | Diagnostic only (A5: the within-date shuffle over-rejects when signals persist and cluster by sector, 11% at nominal 5%). A8: the persistence-preserving placebo (event mask shifted ≥ 1 year) holds 4.3-4.5% and is reported as a diagnostic |
| Era sign | correct sign in ≥ 2 of 3 eras |
| Positive years | ≥ 60% |
| Concentration | t ≥ 2.0 with the top 1% of event-dates removed |
| Replication | same sign in ≥ 1 sibling universe, where a sibling exists |
| Cost coverage | ≥ 2.0 in the tradeable liquidity bucket |
| Liquidity | the edge must not exist only in the bottom Turnover tercile |

**Class W** (few assets, monthly) has too little data for an event study with this power. For W, S3
runs per-asset predictive regressions and the signal-on against signal-off spread, with the bar at
t ≥ 2.0. The burden moves to S5 and S6.

### S4 — Strategy build (fast engine)

- **Full reality.** Costs (section 6.2), both account modes, whole shares in the small mode, cash and
  margin limits, next-open execution.
- **Parameter surface** over the registered grid. For each configuration, compute the
  **neighbourhood median**: the median Sharpe of the configuration and its one-step grid neighbours.
  The **chosen configuration** is the one with the highest neighbourhood median (the centre of the best
  plateau), never the raw peak.
  - **Plateau ratio** = neighbourhood median of the chosen configuration ÷ peak Sharpe of the grid.
    PASS ≥ 0.7, WARN 0.5–0.7, FAIL < 0.5.
- **Luck band.** Run every rebalance offset: 21 for monthly, 5 for weekly. The card shows min, median and
  max. **The reported number is the median.** WARN if the median is below 70% of the best offset. The
  **worst** offset is the planning case for drawdown expectations and for the S8 health monitor. This is
  the Zorro "execution-day luck" point: in the Z9 audit the rebalance day alone moved Sharpe from 0.62
  to 0.89.
- **Losing-streak test.** The longest run of losing trades or months, compared with the distribution
  expected if outcomes were independent (runs test). A much longer streak means losses cluster, and the
  drawdown estimates are too kind. WARN only.
- **Timing variants.** Next open (default); MOC only if the spec registered it and a decision-time price
  model exists (the DV2 study showed 15:45 decisions lose; do not assume MOC).
- **Cost stress.** Twice the costs, plus 10 bps slippage per side, and the **breakeven cost** (the cost
  per side that takes net Sharpe to zero). FAIL if net Sharpe at twice the costs is ≤ 0.
- **Metric set** (from `metrics.py`, same definitions as the engine): CAGR, volatility, Sharpe (zero
  risk-free rate, per `QUANT_PHILOSOPHY.md`), excess Sharpe over T-bills, Sortino, max drawdown, Calmar,
  longest time underwater, turnover, exposure, average hold, win rate, payoff ratio, skew, kurtosis, 95%
  expected shortfall, and a per-year table.

### S5 — Overfitting and multiple testing (the heart of Scout)

All tests run on the family's registered grid. The whole selection procedure from S4 (plateau rule) is
re-applied inside every resample, so the tests judge the *process*, not a single picked configuration.

**A5 (P2 calibration) supersedes the gate list below.**
- **The only S5 gate is MCPT:** plain date shuffle, p ≤ 0.05, re-running the full search including plateau selection,
  under three conditions:
  - **Search space:** it re-runs the union of every registered grid of the family.
  - **Score (A9):** the Sharpe of the daily ACTIVE return of each configuration over the volatility-targeted
    equal weight of the family's traded assets (10% target, 20-session realised volatility, lagged one session);
    plateau selection runs on these active Sharpes. P5 calibration: 2.0% / 7.0% false passes on volatility-timed
    families. "Selected Sharpe minus baseline Sharpe" failed (8.5%), and so did the active return over plain equal
    weight (8.5%).
  - **Point-in-time panels (A8):** they use the per-asset null (`mcpt_live_spans` / `alpha.scout.null.mcpt_panel`,
    membership strata). The score is the Sharpe of the daily ACTIVE return over the baseline, not a difference of
    Sharpes. Calibrated at 8.0% (momentum) and 2.0% (reversal) false passes. A cross-sectional ranking family with
    0.025 < p ≤ 0.05 passes with a "marginal" flag on its card.
  - Its 4.4-4.5% size is empirical, not exact, on GARCH data.
- **DSR is a WARN,** not a gate. It is an exact null p-value (`null_selected_sharpe_draws` +
  `null_selected_sharpe_p_value`, p ≤ 0.05): where the selected Sharpe falls among the Sharpes the search would select
  with no edge, simulated from the correlation matrix of this grid and the family's earlier grids. Earlier trials
  without stored series count as independent draws. A miss needs a written owner note.
- **Printed diagnostics:** walk-forward (efficiency and the 8-design strip) and PBO.
- **The clustered-N_eff SR*_0 formula below is retired** for gating: its deflation vanishes when N_eff collapses.

Evidence: `docs/research/SCOUT_P2_CALIBRATION_20260930.md`. The text below is kept as the pre-calibration design.

S5 asks two questions, and each question has its own gate (D19, pre-calibration):

| Question | Gate | Why this test |
|---|---|---|
| Is the search manufacturing winners? | **MCPT** of the whole search + **DSR** with family N_eff | MCPT covers this registration's grid exactly; DSR is the only test that also counts the family's *earlier* studies from the ledger |
| Does the chosen rule hold up out of sample? | **Walk-forward** with design sensitivity | Tests re-selection through time, the way the rule would actually be run |

Everything else in this station is a printed diagnostic. A diagnostic becomes a gate only if the
calibration study (P2, section 13) shows it catches failures that the three gates miss.

**Gate 1. Monte Carlo permutation test of the whole process (Masters, *Permutation and Randomization Tests*).**
- **How:** permute the price changes and rebuild prices, then re-run the **full** grid search and
  plateau selection on each permuted history. Compare the real selected result with the distribution of
  permuted winners.
- **Null construction:** for multi-asset panels, permute whole dates' cross-sectional return vectors, so
  the correlation between assets survives. Exogenous inputs (VIX, VXN, T5YIE) are passed as extra columns
  so they move with their dates. **Plain date shuffling is the default.** Shuffling within volatility
  strata keeps volatility clustering, but it also keeps any volatility-timing edge in place (P0 review:
  power 0.60 plain vs 0.33 stratified on a planted regime edge), which matters for TAA, trend and
  VXN-scaled pods. P2 decides between them on planted cases.
- **Point-in-time panels:** a panel whose NaN pattern changes over time (membership, late listings) is
  refused by the P0 harness; shuffling it scatters NaN and weakens the null (review: 10.7% false positives
  at p ≤ 0.05 on noise). Such panels use the per-asset null built in P4b (A8).
- **What it answers:** "Could this search procedure have found something this good in data with no
  temporal structure?"
- **Draws:** 500 minimum, 1,000 for promotion.
- **Thresholds:** PASS p ≤ 0.01. WARN 0.01–0.05. FAIL > 0.05.

**Gate 2. Probabilistic and Deflated Sharpe (Bailey & López de Prado).** Here $\widehat{SR}$ is
per-period (not annualised), $T$ is the number of return observations, and $\gamma_3$, $\gamma_4$ are the
skewness and kurtosis of returns.

$$
\text{PSR}(SR^{*}) = \Phi\!\left(\frac{(\widehat{SR}-SR^{*})\sqrt{T-1}}{\sqrt{1-\gamma_3\widehat{SR}+\frac{\gamma_4-1}{4}\widehat{SR}^{2}}}\right)
$$

$$
SR^{*}_0 = \sqrt{V[\widehat{SR}_n]}\left((1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac{1}{N_{\text{eff}}}\right)+\gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{N_{\text{eff}}\,e}\right)\right),\quad \gamma\approx0.5772
$$

$$
\text{DSR} = \text{PSR}(SR^{*}_0)
$$

$V[\widehat{SR}_n]$ is the variance of the Sharpe ratios across the family's trials, and $N_{\text{eff}}$
comes from section 8.3. The bracket is the paper's approximation of E[max of N standard normals]; Scout
computes that expectation exactly by numerical integration (A3), because the approximation turns negative
for N < 1.28. **PASS: DSR ≥ 0.95. WARN: 0.90–0.95. FAIL: < 0.90.**

**Gate 3. Walk-forward (anchored), with design sensitivity.**
- **Registered design.** Refit the plateau choice on the first trading day of every January on all data
  up to the prior December (first refit once 5 years of history exist), trade the following year, and
  stitch the out-of-sample years together. A configuration needs 252 finite training returns to be
  selectable, and a NaN inside a chosen configuration's test window is an error, not a silent drop.
- **Walk-forward efficiency** = OOS Sharpe ÷ mean in-sample Sharpe of the chosen configurations.
- **Design sensitivity (from Lotter / Financial Hacker).** Re-run the walk-forward under a small, fixed
  grid of 8 designs: anchored (5-year minimum history) and rolling 3-, 5- and 8-year windows, each
  refitting every 6 or 12 months (January / July). Anchored designs with different minimum histories were
  dropped at build time: they make the same selections, so they would count one piece of evidence
  several times. The card shows the share of designs with positive stitched OOS Sharpe.
  (Lotter built a placebo strategy with no edge that looked excellent at exactly 9 walk-forward cycles
  and fell apart at every other count.)
- **PASS** if the registered design has efficiency ≥ 0.5 and stitched OOS Sharpe > 0 after costs,
  **and** at least 6 of the 8 designs are positive.

**Diagnostics (printed, not gated, until P2 says otherwise):**
- **PBO by CSCV (Bailey, Borwein, López de Prado, Zhu).** Split the in-sample period into S blocks. For
  every choice of half the blocks as training, pick the best configuration by the plateau rule and find
  its relative rank ω̄ on the other half. λ = ln(ω̄ / (1 − ω̄)), and PBO is the share of splits with
  λ ≤ 0. Use S = 10 (252 splits) by default. S = 16 (12,870 splits) is optional; it is heavy and adds
  little.
- **CPCV (López de Prado).** N = 10 groups, k = 2 test groups: 45 splits and 9 OOS paths. Purge = max
  holding period + label horizon; embargo = 1% of observations after each test group. The card shows the
  5th percentile of OOS Sharpe and the share of negative paths.
- **Haircut Sharpe (Harvey & Liu).** One "what the Sharpe is really worth after the search" number.

**Conditional test: best-of-family claims.** This runs only when the claim is "the chosen variant beats
benchmark B": Hansen's SPA test (stationary bootstrap, 5,000 draws) over all family configurations
against B, or Romano–Wolf stepdown when several variants are promoted together. Pass: p ≤ 0.05.

### S6 — Value to the book

- **Spanning regression.** Monthly returns on a fixed factor set with Newey–West errors:
  - market (SPY), QQQ, bonds (IEF), T-bills;
  - an in-house trend factor: 12-1 time-series momentum on SPY, EFA, EEM, IEF, TLT, GLD, DBC and UUP;
  - Fama–French 5 plus momentum;
  - every LIVE and PM_READY pod.

  PASS if alpha t ≥ 2.0. Defensive pods are also judged on crisis-window returns (below), not on
  alpha alone.
- **T-bill slot test.** Take the reference books (the current live book and the G3 book of
  2026-09-27), put the candidate into its intended slot, and put T-bills into the same slot as the
  alternative. Run a paired stationary bootstrap of the book Sharpe difference.
  **PASS if P(candidate > T-bills) ≥ 0.80** and the point difference is > 0.
- **Diversification.** Correlation to the book and to each pod, conditional returns in the crisis
  windows (2008, 2020, 2022, using `crisis.py`), and tail co-movement (the candidate's return on the
  book's worst 5% days).
- **Capacity** (from `capacity_analysis.py`, capacity v2 conventions). Recommended maximum AUM on
  **today's** volumes, at participation ≤ 1% of ADV for MOO or MOC. Both the full-history and the
  recent-window figures are shown.
- **Small-account friction,** reported separately: the $30K run against the institutional run, whole
  shares and minimum commissions.

### S7 — Vault and promotion

1. **Freeze.** The chosen configuration and every threshold are frozen and written to the ledger.
2. **Open the vault once.** Run the frozen configuration from 2023-01-01 to today.
3. **Vault verdict:**
   - PASS if net return > T-bills over the vault period, **and** the vault Sharpe is at or above the
     5th percentile of the in-sample stationary-bootstrap distribution of Sharpe over windows of the
     same length.
   - FAIL is final for that family when the vault is clean. A new family needs a genuinely new mechanism.
   - `CONTAMINATED` (section 4.2): the result is printed as a diagnostic, and promotion waits for S8
     forward evidence from the registration timestamp onward. In practice this is the normal path for
     most families today.
4. **Real engine and gate.** The idea is implemented in the real engine (Codex) and must pass the
   identity gate against the Scout spec.
5. **Final numbers** are produced by the real engine, and the research card is issued.
6. **Grade.** Scout grades are `REJECTED` (hard FAIL only, D22), `WATCHLIST` (soft FAIL anywhere; tracked
   forward and re-judged), `CANDIDATE` (passed S0–S6) and `PROMOTED` (vault or forward pass plus gate). A `PROMOTED` strategy enters the existing maturity
   tiers in `alpha/strategy_registry.py`. Scout grades do not replace those tiers; they are a separate
   claim about edge, not plumbing.

### S8 — Shadow with pre-registered rules

Written before shadow starts:

- **Expected distribution:** the vault-inclusive bootstrap distribution of monthly returns for the frozen
  configuration.
- **Minimum track record length** to confirm the claimed Sharpe at 95% confidence:

$$
\text{MinTRL} = 1 + \left(1-\gamma_3\widehat{SR}+\frac{\gamma_4-1}{4}\widehat{SR}^{2}\right)\left(\frac{z_{0.95}}{\widehat{SR}-SR^{*}}\right)^{2}
$$

  with $SR^{*}$ = the T-bill-slot Sharpe. The card states MinTRL in months, honestly, even if it is
  years.
- **Kill rule, two detectors** (both in `alpha/stats/health.py`; either one turns the card red):
  - **Mean shift: CUSUM.** A one-sided CUSUM on (realised − expected) monthly returns, with the
    threshold set so that the false-kill rate is 5% a year under the expected distribution.
  - **Abnormal drawdown: Cold Blood Index (Lotter / Zorro), block version.** When the pod is in a
    drawdown of depth D that has lasted L sessions, estimate how often a drawdown at least that deep
    occurs within L sessions in the expected return process. Zorro counts overlapping backtest windows
    as independent samples, which overstates confidence. Here the probability comes from a stationary
    block bootstrap of the worst-offset (S4) daily returns. A single evaluation is a probability, not an
    alert: re-evaluated every day on the drawdown the data chose, a fixed "RED below 5%" goes RED for a
    healthy pod about a third of the time in a year (P0 review simulation). The RED and AMBER cuts are
    therefore calibrated on the monitoring procedure itself: simulate healthy paths, run the same daily
    evaluation, and set RED so that P(any RED within 252 sessions) = 5% (AMBER at 15%), with a minimum
    observation length before the first evaluation. P1 does this calibration.
  - The CUSUM alarm latches (it is the minimum over the whole history), so a healthy pod's chance of ever
    alarming grows with time (about 9% by 24 months, 24% by 60 months). The card says so.

  The kill is automatic only in the sense that the card turns red. Retiring the pod stays the owner's
  decision.
- **Parity check:** every month, Scout re-runs the frozen spec on the latest data and compares its
  decisions with the shadow or live decisions. Any difference is an implementation incident, not
  "noise".

## 10. Re-audition mode (existing strategies)

Every LIVE and PM_READY strategy goes through S0–S6 and S8, with three differences:

1. **Retro-registration.** S0 is written from the strategy's docs and code and marked `RETRO`. Its
   `prior_trials` field is required: the auditor counts the documented configurations in
   `scripts/research/`, `docs/research/` and Pakal for that family. If the count cannot be established,
   N = max(documented, 50). That N is added to the family's `N_eff` before DSR.
2. **No clean vault.** The period since 2023 has been seen. S7's vault test is replaced by the
   **post-adoption period**: from the date the strategy's current rule first entered `main` (git history)
   to today. That period is the only honest out-of-sample evidence, and the card says how short it is.
3. **No automatic demotion.** The re-audition card gives a verdict per station. Demotion or retirement
   is the owner's decision.

**Order** (the gate needs each strategy's post-audit-fix version, see section 13):
- **Money today:** TAA 3x, NDX VXN.
- **Wired pods:** TAA 1/N, BTAL_QQQ, CORE5. (QPI was demoted to RESEARCH on 2026-09-29 and is re-run
  in P4 as an S3 edge study, not a re-audition.)
- **Book pods:** Compass (both variants), TFI, EOM, sector IBS, NDX NATR20.
- **Remaining PM_READY:** hedges, 2x variants, Trinity.
- **Class E strategies (need the path engine, phase P7):** DV2 (and the ETF-DV2 candidate), HPI.

## 11. Research card and Bench page

One HTML page per family and run, stored under `results/scout/cards/`, with the ledger row id in the
header.

1. **Verdict strip:** grade, one line per station (PASS, WARN or FAIL plus the key number).
2. **Registration** as frozen, including `source` and `prior_trials`.
3. **S3 charts:** decile bars with CIs, lag-decay curve, per-year bars, era table, replication table,
   liquidity-tercile bars.
4. **S4 charts:** parameter surface heat map with the chosen plateau outlined, luck band, cost-stress
   table.
5. **S5 charts:** the three gates first (MCPT null histogram with the real value marked, DSR with its
   inputs, walk-forward stitched curve with the 8-design sensitivity strip), then the diagnostics (PBO
   logit histogram, CPCV Sharpe distribution, haircut).
6. **S6:** spanning table, T-bill slot result, crisis table, capacity table.
7. **Known engine deviations,** with the measured effect of each.
8. **What was not tested and why.**

A later Bench page lists families, their grades and cards, and gives a ledger explorer. The card follows
the explanation order of `QUANT_PHILOSOPHY.md`: conclusion first, then intuition, then detail.

## 12. Deliberately excluded (decided noise for this book)

| Excluded | Source | Why |
|---|---|---|
| Fractional differentiation | López de Prado, AFML ch. 5 | For ML feature stationarity; our features are already stationary oscillators or ranks |
| Microstructure features (VPIN, Kyle λ), dollar and volume bars | AFML ch. 2, 19 | Intraday data and horizons we do not trade |
| Random forests and MDI/MDA feature importance | AFML ch. 8, MLAM | Rule-based pods with 1–3 features; importance ranking adds search without adding evidence |
| Triple-barrier labels, meta-labeling | AFML ch. 3 | Parked: revisit only as a filter layer on DV2 after P7 |
| Structural-break tests (SADF, CUSUM on prices) | AFML ch. 17 | Era and regime splits in S3 answer our question more simply |
| Synthetic OU "optimal trading rules" | AFML ch. 13 | Produces more tunable parameters, the opposite of what we need |
| HRP, NCO, bet sizing | AFML ch. 10, 16; MLAM | Portfolio construction, a separate project |
| Genetic or automated rule search | various | A mining engine by design; S5 would reject its output anyway |
| Masters' prediction-model machinery (committees, neural nets) | *Assessing and Improving Prediction* | ML models are a non-goal for v1 |
| Zorro as a platform (engine, IB bridge) | Zorro project | C/C++ second codebase, no Norgate support, unstable IB reconnects per its own forum; our stack already does PIT data, deterministic IB execution and reconciliation |
| Zorro Evaluation Shell, RangerZ | Zorro project, Pardo | Mass combination search with eyeball curve filtering and a final-result-only reality check; the vendor itself warns of selection bias |
| OptimalF sizing | Zorro | Computed on the full sample (look-ahead), as the author acknowledges |
| Equity-curve on/off switching | Zorro "phantom trading" | Adds a rule tuned on the same history; only as a pre-registered S0 hypothesis of its own |
| Inverted-price test | Zorro | Meaningful only for symmetric long/short rules; our pods are long-only |
| Zorro performance metrics (AR, in-market Sharpe) | Zorro | Not comparable with our metric definitions in `metrics.py` |

Kept from each source:
- **López de Prado:** registration-first research, PSR, DSR, MinTRL, CSCV/PBO, CPCV with purge and
  embargo, effective N by clustering.
- **Masters:** indicator quality (S2), MCPT of the whole process, walk-forward, and the idea that the
  selection bias of the search is part of the result.
- **Aronson:** placebo and detrending logic, data-mining bias.
- **Harvey–Liu–Zhu:** the multiple-testing logic behind the ledger-level FDR, and haircut Sharpe (the
  fixed t ≥ 3 bar itself was dropped in A1, D12).
- **Lotter (Zorro, Financial Hacker):** the Cold Blood Index (in block form), walk-forward design
  sensitivity, execution-day luck as a planning case, the losing-streak test, and Z8 / Alpha101 as known
  dead controls for calibration.
- **White, Hansen, Romano–Wolf:** best-of-family tests.
- **McLean–Pontiff:** before and after publication.
- **Carver:** replication across instruments and preference for few rules.
- **Pardo:** the plateau and walk-forward efficiency.

## 13. Build phases

Each phase has a goal, scope and completion criteria, per `AGENTS.md`. The order was changed in A1 so
that the cheapest parts, and the ones that protect real money, come first, and so that the calibration
study decides the gate set before the expensive parts are built. Phases P0–P2 and P4 do not touch
strategies and can start now. P3 gates each strategy only after the readiness-audit fix list
(2026-09-28) has landed for that strategy, so no gate is built twice.

| Phase | Scope | Done when |
|---|---|---|
| **P0 Stats + ledger** | `alpha/stats/` (Newey–West, bootstrap wrapper, permutation, PSR/DSR, MinTRL, MCPT harness, walk-forward, FDR, health), the ledger with hash chain, registration (S0), families, effective N, import-boundary tests | Each statistic matches a published worked example or an independent implementation; chain tamper tested; `alpha/live` imports of `alpha.scout` fail the test suite |
| **P1 Live pod health (report only)** | Cold Blood Index (block) + CUSUM for TAA 3x and NDX VXN from their reference backtests and live ledgers; a daily research-side report | False-alarm rate measured on history (target ≤ 5% a year); detection delay measured on a synthetic "edge died" path; **wiring into the watchdog is a separate live change that needs the owner's explicit authorisation** |
| **P2 Calibration study** | Run the candidate S3 and S5 tests on known cases: pure noise (many seeds), planted edges of known Sharpe (0.3, 0.5, 0.8) inside noise, known dead strategies (Z8 after July 2016, Alpha101 after 2011, IPO/ATH after 1999, MOSAIC), and plain vs volatility-stratified MCPT nulls | A table of false-pass rate and detection power per test; the gate set of S3/S5 is confirmed or changed from that table (D20), and this document is amended |
| **P3 Weights engine + gate** | `engines/weights.py`, costs, both account modes, identity gate, deviations registry, CI job | TAA 3x and NDX VXN (then CORE5) pass the gate on full history; CI blocks a deliberately broken strategy |
| **P4b Null + calibration** | per-asset MCPT null, S3 per-event and placebo calibration, Masters' S2 battery, Nasdaq-100 panel + replication, MCPT on Z9 | Done 2026-10-01 (A8): the null passes its frozen calibration; stock families can pass S5 |
| **P4 Edge toolkit** | panels + snapshot hash, vault seal and contamination record, feature library with contracts, S1, S2 and S3, CLI | S1 catches seeded leaks (future-peeking feature, split-sensitive feature, non-PIT member); the Pakal DV2-S&P 500 and QPI studies re-run through S3 with PIT membership, and the card shows the old-vs-new difference |
| **P5 Strategy, luck, book and card** | S4, the three S5 gates and their diagnostics, S6, the research card, re-audition cards for TAA 3x and NDX VXN | Done 2026-10-02 (A9): cards issued; TAA 3x CANDIDATE (S3 pending), NDX VXN WATCHLIST |
| **P6 Re-audition, classes W and X** | Retro registrations, prior-trial counts, cards for the remaining W and X strategies in section 10 order, Bench page | One card per strategy |
| **P7 Path engine** | `engines/path.py`, gates for DV2 and HPI, then their re-audition | DV2 and HPI pass the gate; cards issued |
| **P8 Discovery and shadow** | S7 vault opening with contamination rules, S8 monitor for new candidates, BY-FDR ledger report | First new family run end-to-end; the vault is opened only through S7 |

## 14. Known limitations of this design

- **Parity with a biased engine.** The gate proves Scout equals the engine, not that the engine is
  right. Truth mode and the deviations registry bound this gap; they do not close it.
- **Permutation nulls destroy all temporal structure.** MCPT answers "is there any structure the process
  exploits", not "is it the structure we believe". S3's mechanism checks and replication carry that
  second question.
- **Retro-registered strategies are optimistic by construction.** The `prior_trials` estimate is a floor,
  not the true number of things the owner once looked at.
- **The vault is short, and mostly already seen.** It can catch a collapse; it cannot confirm an edge,
  and for families the owner has studied on 2023+ data it is only a diagnostic. Only forward time after
  registration confirms.
- **The gate set is provisional until P2.** The thresholds in S3 and S5 are reasoned defaults. The
  calibration study may tighten, loosen or drop them, and that change is recorded here as an amendment,
  never as a quiet edit.
- **Our history is one history.** Every test here resamples the same 25 years. Replication across
  universes is the only partial remedy.

## 15. Sources

- M. López de Prado, *Advances in Financial Machine Learning* (2018); *Machine Learning for Asset
  Managers* (2020).
- D. Bailey and M. López de Prado, "The Deflated Sharpe Ratio" (2014); "The Sharpe Ratio Efficient
  Frontier" (2012).
- D. Bailey, J. Borwein, M. López de Prado and Q. Zhu, "The Probability of Backtest Overfitting" (2017).
- T. Masters, *Statistically Sound Indicators for Financial Market Prediction* (2013); *Testing and
  Tuning Market Trading Systems* (2018); *Permutation and Randomization Tests for Trading System
  Development* (2020).
- D. Aronson, *Evidence-Based Technical Analysis* (2006).
- C. Harvey, Y. Liu and H. Zhu, "… and the Cross-Section of Expected Returns" (2016); C. Harvey and
  Y. Liu, "Backtesting" (2015).
- H. White, "A Reality Check for Data Snooping" (2000); P. Hansen, "A Test for Superior Predictive
  Ability" (2005); J. Romano and M. Wolf, "Stepwise Multiple Testing as Formalized Data Snooping" (2005).
- R. D. McLean and J. Pontiff, "Does Academic Research Destroy Stock Return Predictability?" (2016).
- R. Carver, *Systematic Trading* (2015). R. Pardo, *The Evaluation and Optimization of Trading
  Strategies* (2008).
- Y. Benjamini and D. Yekutieli, "The Control of the False Discovery Rate under Dependency" (2001).
- J. C. Lotter, Financial Hacker: "The Cold Blood Index" (2015), "White's Reality Check" (2016), "Why 90%
  of Backtests Fail" (2019); Zorro manual (zorro-project.com/manual).
- Pakal report `reports/zorro_zsystems_daily_audit/REPORT.md` (2026-09-29): the Z8/Z9/Z13 audit used as
  dead and fragile controls.

## 16. Amendment log

**A1 (2026-09-30).** Source: an independent critique of v1 and a Zorro / Financial Hacker concept review
(Pakal session "Z Systems research", 2026-09-30). Decided by Claude under the owner's delegation.

| Change | v1 | A1 | Why |
|---|---|---|---|
| Luck tests | Seven tests, most gated | Three gates (MCPT, DSR, walk-forward); PBO, CPCV, haircut as diagnostics; SPA conditional | Overlapping tests on one history; conjunction rejects real edges; each threshold is a knob (D19) |
| Significance bar | t ≥ 3 at S3 plus DSR plus FDR | Ledger-level BY-FDR at S3, DSR at S5; t ≥ 2 as sanity | t ≥ 3 is already a multiple-testing haircut; stacking double-counts (D12) |
| Vault | Contaminated only if the source was published after 2023 | Also contaminated when the owner studied the family on post-2022 data; forward data after each registration is the clean holdout | Most families were already studied on 2023–2026 (D8) |
| Statistics location | `alpha/scout/stats/` | `alpha/stats/`, shared with Pakal and the live health report | One implementation instead of two or three (D18) |
| Build order | Kill rule last (P7) | Stats + ledger, then live pod health, then calibration, then gates | Cheapest, money-protecting parts first; calibration before building expensive tests (D20, D21) |
| Calibration | None | P2 study on noise, planted edges and dead strategies | A test earns a gate by evidence, not by reputation |
| Zorro ideas | Not considered | Cold Blood Index (block version) in S8 and P1; walk-forward design sensitivity; worst rebalance offset as planning case; losing-streak test; volatility-stratified MCPT null | The useful Zorro ideas are monitoring and robustness checks, not its search tools |
| PBO cost | S = 16 (12,870 splits) | S = 10 (252 splits) by default | Same information at a fraction of the cost |

**A2 (2026-09-30).** Owner direction: lean, and do not reject too eagerly. Hard FAIL narrowed to
evidence against the idea; every other miss sends the family to `WATCHLIST` with forward tracking; P2
calibration must also measure the false-reject rate (D22).

**A3 (2026-09-30, P0 build).** Three independent reviews of the P0 code (quant pitfalls, parity against
reference implementations, test coverage with mutation experiments) led to these design changes:
- **Exact E[max] in the DSR.** The paper's closed-form approximation is negative for N < 1.28, which would
  turn a deflation into a bonus for highly correlated families; E[max] is now integrated exactly (the
  paper example is still reproduced with the approximation in the tests).
- **Effective N by clustering at ρ = 0.5** (section 8.3), after a participation-ratio estimate
  under-counted lopsided grids.
- **DSR variance fallback** 1 / (T − 1) when fewer than two trials are recorded.
- **Plain MCPT null by default;** stratified is a P2 option. Point-in-time panels refused until P4.
- **Walk-forward:** January / July calendar refits, 252-observation minimum, NaN in test windows is an
  error, 8 distinct designs with a 6-of-8 rule.
- **Cold Blood Index thresholds** are calibrated on the daily monitoring procedure in P1, not fixed.
- **Ledger:** non-string keys refused (they would have broken the chain permanently), duplicate checks
  run inside the lock, the default ledger is always the main checkout's (worktree writes would vanish),
  trials outside the frozen grid are refused, and every trial row carries station, mode and code commit.

**A4 (2026-09-30, P1 build: live pod health).** `python -m alpha.scout health` (alpha/scout/pod_health.py, maths in
alpha/stats/pod_monitor.py). Choices and findings:
- **Inputs.** Realised = IBKR Flex daily TWR (flow-neutral, dividends included), read through the pure
  `client_reporting` parser or the `mode=ro` store reader, from a pinned monitoring start (2026-07-01 for both pods);
  non-session postings fold into the next session, and a missing session stops the report instead of hiding returns.
  Expected = backtest returns that END before the monitoring start. NDX VXN: worst of the 21 rebalance offsets of the
  parity-checked replica, chosen on pre-live data (Sharpe 0.69); TAA 3x: latest vanilla backtest (no offset study yet).
  Both include dividends, like the TWR; each report records the source file and its sha256.
- **Calibration.** CBI table of 20,000 reference paths (block length 20); drawdowns within 1e-7 of a reference path
  count as ties (a float32 tie-break had pushed TAA's RED cut to the resolution floor, found in review). RED / AMBER
  cuts set so a healthy pod evaluated every session from session 21 crosses them within 252 sessions with 5% / 15%
  probability. Simulated pods (calibration and detection power) carry a 1e-4 relative jitter: exact bootstrap
  copies tie with reference paths far more than continuous live data do, which had understated TAA's real false
  alarms by half (second review). CUSUM on completed calendar months (a month counts only when its first and last sessions are in the
  data), k = 0.5, h for 5% per 12 months. The combined RED rate of both detectors is 8-10% a year and is printed.
- **Status.** RED > STALE (data more than 5 sessions behind the report date) > TOO_EARLY > AMBER > GREEN; an
  earlier RED is kept. Lookups beyond the table length raise instead of clamping.
- **Finding: returns see a dead edge only slowly.** Within one year the monitor flags "edge died (zero drift)"
  38% of the time for TAA 3x and 17% for NDX VXN, against 10% and 8% false alarms. Confirming it at 80% power needs
  about 2.5 years (TAA 3x) and 12 years (NDX VXN) of data. Large breaks are caught within months. The fast guard
  against implementation breaks is the decision-level live-vs-backtest comparison, which the report points to.
- **Model limits (printed).** The rates hold for a pod that behaves like the bootstrap of its backtest. Higher
  volatility than the backtest, or crash clustering longer than 20 sessions, raises real false alarms (1-year
  historical windows containing the 2020 crash went RED often). The CBI is blunter after year one.
- **Gap recorded:** G-034 in `ASSUMPTIONS_AND_GAPS.md`. Wiring into the watchdog stays an owner decision.

**A5 (2026-09-30, P2 calibration).** Two pre-registered runs (`PROTOCOL.md` 5660232; `PROTOCOL_AMENDMENT_1.md`
52e961b, written after an independent review of the first run), in `docs/research/SCOUT_P2_CALIBRATION_20260930.md`.

- **Decisive run:** 7,000 synthetic histories, two grid families (varied; near-duplicates), plateau selection.
  - **MCPT plain at p ≤ 0.05:** 4.4 / 4.5% false pass and 62 / 75% power at Sharpe 0.52 (93 / 95% at 0.79). The frozen
    rule chose it as the only S5 gate.
  - **DSR with a correlation-aware benchmark:** 2.2 / 3.5% false pass, power 53 / 73%. It is a WARN, because it alone
    counts the ledger's earlier trials.
  - **The old clustered-N_eff DSR:** 11.8% false pass when N_eff collapsed to 1. It is retired.
  - **Walk-forward (21-25% false pass) and PBO (blind to shared edges):** diagnostics.
- **S3:** with sector factors and a persistent signal, the naive per-event t-test rejected a true null 19% of the time
  and the within-date permutation 11%. Only the date-level Newey-West t held 5%, so it is S3's significance test and
  the permutation is a diagnostic until P4 calibrates a persistence-preserving version (the panel is exploratory and
  labelled so).
- **Real dead cases:**
  - The correlation-aware DSR (now a WARN) flags Zorro Z9 on its hindsight-picked list, which the old DSR passed.
  - MCPT, the gate, was not run on it; the reviewer's rough estimate is p ≈ 0.07. MCPT on Z9 is to be run in P4.
  - The correlation-aware DSR passes Alpha101 gross, whose edge was real and was killed by costs (S4's job).
- **The first run's "DSR alone" is withdrawn.** It sat on the sampling boundary, used the flawed DSR, and had edge
  labels that were too high.
- **S0** must record how a universe or asset list was chosen and whether results had been seen (P4).

**A6 (2026-09-30, P3 build: weights engine and identity gate).**
- **Layout:** specs live in `alpha/scout/specs/` (`taa_3x.py`, `ndx_vxn.py`), not `families/`, because
  `alpha/scout/families.py` is the mechanism taxonomy. The shared execution engine is `alpha/scout/engines/weights.py`;
  the gate is `alpha/scout/gate/` (`identity.py` comparator, `run.py` runner, `deviations.py` registry).
- **The engine contract was mapped line by line** for the full-target monthly pods:
  - sizing from the previous close's total value and T prices, truncated whole shares;
  - fills at Open(t) with slippage, and a per-share fee with a minimum;
  - no cash check;
  - dividends net of 25% withholding, credited before the next open;
  - missing-price liquidation at the last close;
  - two share-unit modes: adjusted (TAA 3x) and historical raw shares (NDX VXN, after fix fb81e86).
- **Gate results against the saved engine runs of 2026-09-30:**

  | Pod | Return difference | Daily correlation | Drawdown difference | Holdings cells matched | Scout time | Engine time |
  |---|---|---|---|---|---|---|
  | TAA 3x | 0.00 bps | 1.000 | 0.000 pp | 610 of 610 | 3 s | 22 s |
  | NDX VXN | 0.00 bps | 1.000 | 0.000 pp | 2,329 of 2,329 | 13 s | 274 s |

- **CI substitute:** there is no CI in the repo, so the gate runs in the local test suite
  (`tests/test_scout_weights_engine.py`). It includes a deliberately broken spec (reversed rank weights) that the gate
  must fail. `python -m alpha.scout gate <spec> [--fresh]` exits 1 on failure.
- **Deviations registry:** the membership tail trim is retired (fix #7). The split-adjusted share units deviation
  applies to TAA 3x only. The DTB3 publication lag (0 of 168 decisions) is registered.
- **Review of P3 → exact parity tier.** The independent review showed that the section 7.1 tolerances (5 bps,
  0.999, 99.5% of cells) let real bugs pass while parity is exact to about 1e-16:
  - membership read one session early (4 bps, 99.7% of cells);
  - the $1 minimum fee dropped;
  - dividends credited a day early;
  - a Scout run shortened by years.

  The gate is now `identity.compare_exact`: in parity mode every one of these must hold, or the gate fails —
  - daily returns and daily weights within 1e-9 on every date;
  - identical trade dates;
  - the same first date;
  - no engine date missing inside the span;
  - at most 5 sessions missing at the end.

  The 7.1 tolerances remain an informational tier for truth mode.
  - Result: TAA 3x 4e-16 / 3e-16 over 3,518 days; NDX VXN 4e-16 / 1e-16 over 6,725 days.
  - The tests include two near-miss mutants that must fail (membership one session early; no minimum fee).
  - The engine now runs before Scout: fresh mode refreshes DTB3 first, and capital comes from the engine run.
  - Scout is cut at the engine's last date.
  - DTB3 is read from the main checkout and refused if older than the last executed decision.
  - Real-data tests skip only when `norgatedata.status()` is false.
  - Limitation: saved mode compares today's data with an older run, so a future split can fail it falsely in
    TAA's split-adjusted units; `--fresh` is authoritative.
- **Deferred:**
  - an institutional (fractional-share) account mode, which belongs to S4 in P5; parity mode's capital parameter covers
    the $30K case;
  - CORE5's spec, which goes to P6 with the other re-auditions;
  - truth-mode runs, which come with the research card in P5.

**A7 (2026-10-01, P4 build: edge toolkit).**
- **Built:**
  - `alpha/scout/panel.py`: point-in-time panels with exact membership, a parquet cache kept per snapshot (never
    overwritten), a content snapshot id, and the vault seal applied at load. Only the id of a `vault_opening` row
    for the family, found in the verified ledger chain, opens it.
  - `alpha/scout/features.py`: features with declared lookback and basis; QPI and DV2 reuse the engine's fast
    indicators.
  - `stations/s1_causality.py`: prefix invariance at random cutoffs and a simulated future 2:1 split, on a symbol
    sample (about 8 s per feature), plus a panel-level membership integrity check (no tail trim).
  - `stations/s2_indicator.py`: per-block quantiles, threshold drift and Spearman novelty.
  - `stations/s3_edge.py`: excess over the same-date eligible mean, date-level Newey-West, eras, years, volatility
    regimes, liquidity terciles, lag decay, concentration, crisis windows, cost coverage and per-date deciles;
    D22 verdicts.
  - A `universe_choice_str` field in S0 registration, and `python -m alpha.scout panel`.
- **Evidence:** S1 catches seeded leaks (next-day close, a centred window, a full-sample z-score, a mislabelled
  dollar threshold) and passes the whole feature library; the membership check fails a tail-trimmed mask (a fixed gap). S3 finds
  a planted reversal, stays quiet on noise, rejects a planted wrong-sign effect, and puts an edge planted in liquid
  names in the top liquidity tercile.
- **First real use** (`docs/research/SCOUT_P4_EDGE_RERUN_20261001.md`): the owner's Pakal notebooks re-run.
  - **QPI pullback:** notebook method t 10.0, Scout −1.3 (per event +1.8). WATCHLIST by the rule, no edge of its
    own: most of the apparent edge came from stocks outside the index on the event date, and QPI is the 3-day
    return in disguise (Spearman 0.94). QPI is not studied again as its own family; reversal ideas go to the
    3-day-return family with these runs as prior trials.
  - **DV2 oversold:** notebook method t 27, Scout 3.3 (per event 5.5), WATCHLIST. The per-date deciles fall in
    perfect order, but the edge sits in stress, has faded from 19 bp (1998-2007) to 0.6 bp (2016-2022), and does
    not cover costs (0.9 against 2).
- **Changes from the independent review (2026-10-01):**
  - **S3 concentration check:** trims the most extreme 1% of event-dates on BOTH sides. Trimming only the favourable
    tail gives t ≈ −3 on pure noise, because date means are fat-tailed.
  - **S3 estimators and hard fail:** a per-event estimator (ratio of sums, date-level Newey-West error) is reported
    beside the date-level one. A hard fail needs the sign wrong under both, or a date-level t of −2 or lower. The
    per-event estimator is not calibrated yet, so it can save an idea from a hard fail but not cause one.
  - **Point-in-time membership:** the old "no PIT violation" check could never fail, because S3 filters non-members
    first. It is replaced by the reported non-member share and the panel-level tail-trim check.
  - **Snapshots:** the id also hashes symbol and field names; each snapshot keeps its own cache folder.
- **Deviations from section 9, recorded:**
  - S1 uses 20 cutoffs (design: 50) on a 40-symbol sample, at a relative tolerance of 1e-12. A check with fewer
    than 200 finite comparisons counts as not tested (fail). S1 cannot see leaks in the data itself (revised
    Turnover, a symbol list holding future joiners); the membership check covers the one known case.
  - S3 does not yet have: the replication criterion in a sibling universe (needs the Nasdaq-100 panel), bootstrap
    intervals, placebo dates, up/down-market splits, extra horizons, or VIX terciles (it uses the realised
    volatility of the member universe). Volatility terciles and deciles use full-sample cut points (diagnostics).
  - `universe_chosen_after_results_bool` is recorded at S0 but read only from P5, where S5 requires MCPT for such a
    family.
- **Deferred to P4b, before S5 runs on stock families:**
  - the per-asset MCPT null for point-in-time panels (until then, stock families cannot pass S5);
  - a persistence-preserving permutation for S3 diagnostics;
  - MCPT on the Zorro Z9 grid;
  - a Nasdaq-100 panel cache (built on first use) and the S3 replication criterion that needs it;
  - calibration of the per-event S3 estimator on the P2 synthetic panels;
  - Masters' per-indicator battery in S2 (range/IQR, relative entropy, mutual information against a shuffled
    baseline, a serially-correlated mean-break test). Threshold optimisation stays out of S2: it is a search, so it
    belongs in S4/S5 where trials are counted.

**A8 (2026-10-01, P4b: the per-asset null and S2/S3 calibration).** Report:
`docs/research/SCOUT_P4B_CALIBRATION_20261001.md`; protocol frozen in df76c34, with amendments 1-2 recorded before
any Part B result.

- **The per-asset MCPT null, adopted as S5's null for point-in-time panels:**
  - **How it works:** one global date permutation. Each stock follows it wherever the source date is its own
    listed date in the same membership state; its leftover dates fill the remaining slots in the order of a global
    key. Bars move whole (Masters), chained from each stock's first bar. Membership, Volume and Turnover stay on
    their real dates.
  - **Score:** the Sharpe of the daily active return over the baseline. A "Sharpe minus Sharpe" score was biased
    toward false passes, because the null keeps only part of the co-movement (independent review).
  - **Calibration** on an S&P-like synthetic panel (tenure 0.41, drift +0.04% a day):
    - momentum: 8.0% false passes, 67% power;
    - reversal: 2.0% false passes, 40% power;
    - the frozen bar was ≤ 8.0% and power ≥ twice the false passes.
  - **Flag:** momentum-like families with 0.025 < p ≤ 0.05 pass as "marginal".
  - **Not calibrated:** market-timing families inside stock panels.
- **S3:**
  - The per-event estimator over-rejects with sector factors and a persistent signal (9.5%). It stays
    informational: it can save an idea from a hard fail, never cause one.
  - The shift placebo (4.3-4.5%) is added as a diagnostic.
  - The replication check in a sibling universe is a soft check. The Nasdaq-100 panel was built for it: DV2
    replicates (+17 bp, t 3.3, cost coverage 1.7); QPI fails it with a hard fail on the Nasdaq-100.
- **S2:** Masters' battery (tails, relative entropy, mutual information against within-date shuffles, a
  single-break test with Andrews' automatic lag: 0-3.8% false warnings up to autocorrelation 0.95). Diagnostic.
- **S1:** the membership check now looks for the fingerprint of a fixed tail trim (many stocks sharing one gap of 2
  or more sessions). A share bar wrongly flagged the Nasdaq-100, where real removals before acquisitions are
  scattered and mostly one session.
- **Zorro Z9, a known dead case:** the MCPT stops it in sample (p 0.11 on its 2017 list, 0.15 on a neutral list).
- **Still open after P4b:**
  - **The persistence-preserving permutation for S3:** replaced by the shift placebo, which keeps persistence and
    clustering. No further permutation is planned.
  - **The weights engine:** reads Close and Open only, so liquidity filters in a search must come from the real
    panel, like membership.

**A9 (2026-10-02, P5: S4-S6, the research card, the live pods re-audited).**
Report: `docs/research/SCOUT_P5_REAUDITION_20261002.md`. Calibration protocol: `scout_p5_calibration_20261002`
(frozen 9ab5bb9, amendment 1 before any result).

- **MCPT score for ETF and timing families:**
  - **The score:** the Sharpe of the daily active return over the volatility-targeted equal weight of the
    family's traded assets.
  - **Calibration:** 2.0% / 7.0% false passes; power 6% (TAA-like) and 36% (trend).
  - **Rejected:** "Sharpe minus Sharpe" (8.5%) and the active return over plain equal weight (8.5%).
  - **Composite pods** are split into components. For NDX VXN these are stock selection (per-asset null, active
    return over the overlay × equal-weight members) and the timing overlay (date shuffle). The gate needs every
    component.
- **Built:**
  - `alpha/scout/family.py`: specs as parameter families; the default is the live pod, and the identity gate still
    passes.
  - `metrics.py`.
  - `searches.py`: fast MCPT replicas, checked against the engine.
  - The stations `s4_strategy.py`, `s5_overfit.py` and `s6_book.py`.
  - `card.py`: an HTML research card.
  - Two RETRO ledger registrations.
- **S4 deviations, recorded:**
  - **Luck band:** decision offsets 0-15 (21 would skip short months).
  - **Common window:** one in-sample window shared by every configuration.
  - **Planning number:** the card quotes the median decision day.
- **S6 as built:**
  - **Spanning:** ETF factors (QQQ; SPY QQQ IEF GLD; + 12-1 trend; + the other live pod) stand in for
    Fama-French, and months before a factor exists are dropped.
  - **T-bill slot:** on the live book only (60% TAA 3x / 40% NDX VXN).
  - **Capacity:** a simplified estimate (95th-percentile order at 1% of the 63-session ADV).
- **Results:**
  - **TAA 3x: CANDIDATE (S3 pending).**
    - MCPT p 0.010.
    - Net alpha +13.2% a year, t 2.71.
    - T-bill slot P 0.92.
    - Plan with Sharpe about 0.9: the live month-end day is the best of 16.
    - Capacity about $0.8M (BTAL).
  - **NDX VXN: WATCHLIST.**
    - MCPT FAIL (selection p 0.18, overlay p 0.09).
    - Alpha t 1.25 once TAA 3x is a factor.
    - T-bill slot P 0.47.
    - No demotion (section 10): the owner decides.

**A10 (2026-10-02, P6a: S3 for classes W and X; the live families' siblings).**
Report: `docs/research/SCOUT_P6A_REAUDITION_20261002.md`.

- **S3 for class W:**
  - **Tests:** Fama-MacBeth slope, signal-on minus signal-off, and per-asset slopes (monthly, Newey-West lag 2);
    `gate_split` for risk gates (next-month volatility on vs off, Levene test).
  - **Status:** diagnostics. The design's own phrase applies: "the burden moves to S5 and S6".
- **S3 for class X:** the monthly rank IC and the top-N spread over the eligible mean. These are soft checks.
- **Re-audition runner:** `alpha/scout/reaudit.py`.
- **Siblings gated exactly:** the TAA 1/N, linearity and 2x variants, and NDX ATR / NATR20 / NATR20 VXN.
- **Findings:**
  - **TAA:** the edge is the gated leveraged fallback. The defensive rotation predicts nothing.
  - **NDX, the live score:** it divides by dollar ATR, so it ranks partly by share price (Spearman −0.46). The
    NATR20 ranking passes the selection MCPT at p 0.001.
  - **NDX in the book:** no NDX rule adds value to the live book (2012-2022).

**A11 (2026-10-02, P6b: macro and allocation pods).**
Report: `docs/research/SCOUT_P6B_REAUDITION_20261002.md`.

- **Gated exactly:** Compass, Compass QQQ, CORE5, TFI and Trinity.
- **Weights engine:** opt-in shorts, borrow fee, sign-flip orders, and a path-dependent `decision_fn` hook.
- **Runner:** MCPT kind "spec", where the spec module carries its validated replica.
- **Family placement:** CORE5 moves to the trend family, Trinity to low-risk allocation.
- **Grades:** CORE5 CANDIDATE; Compass and Compass QQQ WATCHLIST (real timing, MCPT p 0.003-0.005, but no
  independent alpha after TAA 3x); TFI and Trinity WATCHLIST.
- **In NDX VXN's book slot:** CORE5 (P 0.85) and Compass QQQ (P 0.81) beat T-bills; NDX variants, TFI and Trinity
  do not.

**A12 (2026-10-02, P6c: EOM, sector IBS, and a first discovery family).**
Report: `docs/research/SCOUT_P6C_REAUDITION_20261002.md`.

- **Weights engine, opt-in:** MOC fills, close-and-reopen orders, fractional shares, and hold-NaN.
- **Gated exactly:** EOM and the four sector IBS pods.
- **Grade rule:** an S3 hard fail grades REJECTED (D22).
- **Walk-forward:** not runnable on short histories, and said so.
- **Grades:**
  - EOM: CANDIDATE (MCPT p 0.001, alpha t 3.50, book 1.21 → 1.46).
  - Sector IBS VOX IYR and Dispersion KIE IHI XLC: WATCHLIST.
  - The two SMA200 dispersion variants: REJECTED. Their relative edge is negative, although their timing value
    passes S5 and S6.
- **Discovery: Bitcoin + gold** (`specs/btc_gold.py`, the GQResearch rule; registered before results, with 866
  prior trials).
  - The trend rule passes the MCPT (p 0.03); a fixed 50/50 does not.
  - A 10% sleeve nudges the book (P 0.77 in sample, 0.95 over a contaminated full period).
  - Grade: WATCHLIST.

**A13 (2026-10-02, P7: point-in-time stock pods).**
Report: `docs/research/SCOUT_P7_REAUDITION_20261002.md`.

- **Gated exactly:** DV2 S&P 500 (WIRED), DV2 Nasdaq-100, HPI 2/3/5 vote (WIRED) and HPI IBS RSI exit.
  - The DV2 Nasdaq-100 module is an empty stub, so its engine side is the restored 300ca70 code
    (`gate/legacy_dv2_ndx.py`), a gate harness only.
- **Weights engine, opt-in:** `missing_open_hold_df` (HPI keeps a held member through a missing open).
- **Runner:** MCPT kind "panel" applies the A8 per-asset null to a whole stock-pod search, with a per-pod worker count.
- **Gate practice:** a saved engine run goes stale when Norgate revises history. A gate failure is re-checked with
  `--fresh` before it is read as a Scout defect, and a fresh run must not overlap the nightly Norgate update.
- **Grades:** DV2 S&P 500 and DV2 Nasdaq-100: WATCHLIST (MCPT p 0.001 each; book 1.21 → 1.37 / 1.36). HPI vote and HPI IBS RSI: WATCHLIST, on soft S3 criteria only.
  The whole rule passes S4-S6 strongly (vote: MCPT p 0.021, DSR p 0.000, alpha t 2.64, book 1.21 → 1.37), but the
  entry event barely beats the same-date members (+3.6 bp, t 0.79). DV2 and HPI correlate 0.75 daily, so the book
  needs one stock reversal pod, not two.

**A14 (2026-10-02, P7b: DV2 on industry ETFs).**
Report: `docs/research/SCOUT_P7B_DV2_INDUSTRY_ETF_20261002.md`.

- **Gated exactly:** `dv2_industry_etf`, with no engine change.
- **MCPT:** the ETF date shuffle keeps eligibility (history, Turnover, raw price) on real dates (A8).
- **Family:** `etf_short_term_reversal`. A registration cannot have a parent in another family, so it has no parent.
- **Grade:** WATCHLIST on a single WARN (S3 cost coverage 1.62).
  - MCPT p 0.001, DSR p 0.002, alpha t 3.05, slot P 0.99 (book 1.21 → 1.32).
  - Correlation about 0.5 with DV2 and HPI.
- **Idle cash:** a cash-heavy pod's slot test is also shown with idle cash credited at the T-bill rate (information).

**A15 (2026-10-02, DV2 size ladder).**
Report: `docs/research/SCOUT_DV2_SIZE_LADDER_20261002.md`.

- **The study:** the frozen DV2 rule on 12 point-in-time universes, built from one superset panel
  (`alpha/scout/universes.py`).
- **Result:** the gross edge sits in small and micro caps; net of costs, only the large-cap end survives. The S&P 500
  stays the universe for the rule.
- **Correction to P7:** DV2's S3 headline included 1998-2003. The event edge decays by era: +18.2, +5.4, then +0.8 bp.
- **S3 practice:** an S3 window must equal the pod's own backtest window, and the era table is read with the headline.

**Not adopted from the critique.** One correction: the critique said the kill rule closes gap G-006.
It does not. G-006's missing circuit breaker is about **repeated reconciliation failures**, not about
performance. A performance kill rule is a separate, currently unrecorded gap, and P1 should add it to
`ASSUMPTIONS_AND_GAPS.md`.
