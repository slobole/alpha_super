# Scout — research pipeline design (v1)

Date: 2026-09-29. Base: `main` @ `cb29d4f`. Author: Claude. The owner delegated every design decision
in this document to Claude ("you decide the catalog / pipe in full"). No code was changed.

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
vault attack **luck**. Speed exists only so the luck tests (permutation, CSCV, walk-forward) are
affordable. Speed is never the goal.

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
| D8 | Vault = 2023-01-01 to today, sealed at the data layer | A single honest out-of-sample test per family |
| D9 | Unit of inference in edge studies = the date, not the event | Overlapping holds and same-day clustering make events far from independent |
| D10 | Append-only, hash-chained trial ledger, committed to git | Every trial counts; tampering is visible |
| D11 | Trials are counted per mechanism family, with an effective-N correction | 100 variants of one idea are not 100 independent trials |
| D12 | Default significance bar t ≥ 3 for new ideas | Harvey–Liu–Zhu; our own search history shows how many ideas die |
| D13 | Parameter choice = centre of the best plateau, never the peak | Peaks are where luck lives |
| D14 | Reported strategy numbers are the median of the rebalance-day luck band | The best offset is luck |
| D15 | Every candidate must beat T-bills in its own book slot (S6) | This is the bar every recent study used; it is the real opportunity cost |
| D16 | Capacity is judged on today's volumes, separately from small-account friction | Owner rule (tradability from today) |
| D17 | Shadow (S8) kill and confirm rules are written before shadow starts | Otherwise the forward test becomes another tuning loop |

## 3. Package layout

```text
alpha/scout/
  data/        panel.py (cached PIT panels), vault.py (seal), snapshot.py (data hash)
  features/    library.py (registered features), contracts.py (declared lag/basis/lookback)
  spec.py      ScoutSpec dataclass + hashing
  families/    one module per mechanism family (specs live here, with their registration)
  engines/     weights.py (X, W), path.py (E, numba), costs.py, execution.py
  gate/        identity.py (Scout vs real engine), deviations.py (known-engine-bias registry)
  ledger/      ledger.py (append-only JSONL, hash chain), trials.py (effective N)
  stats/       newey_west.py, bootstrap.py, permutation.py, psr_dsr.py, pbo_cscv.py,
               cpcv.py, walk_forward.py, spa.py, mcpt.py, fdr.py, mintrl.py
  stations/    s0_register.py ... s8_shadow.py
  card/        research card (HTML), reused by a Bench page
  cli.py       python -m alpha.scout <command>
tests/scout/   unit tests per module + the import-boundary test + gate tests
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
- **Contamination flag.** If the idea's source (paper, book, talk) was published on or after the seal
  date, its vault test is marked `CONTAMINATED`: the author may have known the period. Such a family
  needs shadow evidence (S8) before promotion.
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
- **Effective number of trials:** trials in a family are clustered on the correlation distance
  `d = √(½(1 − ρ))` of their daily return series (hierarchical clustering, number of clusters chosen by
  silhouette score, as in López de Prado's ONC). The number of clusters is `N_eff`, used by DSR.
- **Global count:** a ledger report applies Benjamini–Yekutieli FDR at q = 0.05 across all families' S3
  p-values, so the whole research program is controlled, not only each family.

## 9. The pipeline: stations S0–S8

Each station returns **PASS**, **WARN** or **FAIL** per check. A FAIL on a hard check stops the family.
A WARN is printed on the card and needs a written owner note before promotion. The station sequence is
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
| Newey–West t, date-clustered, registered horizon | ≥ 3.0 |
| Within-date permutation p (E and X), block-of-dates p (W) | ≤ 0.01 |
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
  max. **The reported number is the median.** WARN if the median is below 70% of the best offset.
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

**1. Probabilistic and Deflated Sharpe (Bailey & López de Prado).** Here $\widehat{SR}$ is per-period
(not annualised), $T$ is the number of return observations, and $\gamma_3$, $\gamma_4$ are the skewness
and kurtosis of returns.

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
comes from section 8.3. **PASS: DSR ≥ 0.95. WARN: 0.90–0.95. FAIL: < 0.90.**

**2. Probability of Backtest Overfitting, CSCV (Bailey, Borwein, López de Prado, Zhu).**
1. Split the in-sample period into S = 16 contiguous blocks.
2. For each of the C(16, 8) = 12,870 ways to choose half of the blocks as the training set, pick the
   best configuration on the training half (by the plateau rule).
3. Find that configuration's relative rank ω̄ on the other half, and compute λ = ln(ω̄ / (1 − ω̄)).
4. PBO is the share of splits with λ ≤ 0.

**PASS: ≤ 0.20. WARN: 0.20–0.50. FAIL: > 0.50.**

**3. Monte Carlo permutation test of the whole process (Masters, *Permutation and Randomization Tests*).**
- **How:** permute the price changes and rebuild prices. For multi-asset panels, permute whole dates'
  cross-sectional return vectors, so the correlation between assets survives. Then re-run the **full**
  grid search and plateau selection on each permuted history. Compare the real selected result with the
  distribution of permuted winners.
- **What it answers:** "Could this search procedure have found something this good in data with no
  temporal structure?"
- **Draws:** 500 minimum, 1,000 for promotion.
- **Thresholds:** PASS p ≤ 0.01. WARN 0.01–0.05. FAIL > 0.05.

**4. Walk-forward (anchored).** Refit the plateau choice every January on all data up to the prior
December, trade the following year, and stitch the out-of-sample years together.
- Walk-forward efficiency = OOS Sharpe ÷ mean in-sample Sharpe of the chosen configurations.
- PASS if efficiency is ≥ 0.5 and the stitched OOS Sharpe is > 0 after costs.

**5. Combinatorial purged cross-validation (López de Prado).**
- **Splits:** N = 10 groups, k = 2 test groups. That gives 45 splits and 9 full OOS paths.
- **Leak control:** purge = max holding period + label horizon; embargo = 1% of observations after each
  test group.
- **Output:** a distribution of OOS Sharpe. The card shows its 5th percentile and the share of paths
  with Sharpe below zero.
- **Thresholds:** WARN if more than 20% of paths are negative, FAIL if more than 40%.

**6. Best-of-family claims.** When the claim is "the chosen variant beats benchmark B", run Hansen's SPA
test (stationary bootstrap, 5,000 draws) over all family configurations against B. When several variants
are promoted together, use Romano–Wolf stepdown. Pass: p ≤ 0.05.

**7. Haircut Sharpe (Harvey & Liu).** Printed, not gated. It gives the owner a single "what the Sharpe is
really worth after the search" number.

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
   - FAIL is final for that family. A new family needs a genuinely new mechanism.
   - `CONTAMINATED` if the source was published after the seal (section 4.2); this goes to S8 before
     any promotion.
4. **Real engine and gate.** The idea is implemented in the real engine (Codex) and must pass the
   identity gate against the Scout spec.
5. **Final numbers** are produced by the real engine, and the research card is issued.
6. **Grade.** Scout grades are `REJECTED`, `WATCHLIST` (passed S3, failed later), `CANDIDATE` (passed
   S0–S6) and `PROMOTED` (vault pass plus gate). A `PROMOTED` strategy enters the existing maturity
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
- **Kill rule:** a one-sided CUSUM on (realised − expected) monthly returns, with the threshold set so
  that the false-kill rate is 5% a year under the expected distribution. The kill is automatic in the
  sense that the card turns red; retiring the pod stays the owner's decision.
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
- **Wired pods:** TAA 1/N, BTAL_QQQ, CORE5, QPI.
- **Book pods:** Compass (both variants), TFI, EOM, sector IBS, NDX NATR20.
- **Remaining PM_READY:** hedges, 2x variants, Trinity.
- **Class E strategies (need the path engine, phase P6):** DV2 (and the ETF-DV2 candidate), HPI.

## 11. Research card and Bench page

One HTML page per family and run, stored under `results/scout/cards/`, with the ledger row id in the
header.

1. **Verdict strip:** grade, one line per station (PASS, WARN or FAIL plus the key number).
2. **Registration** as frozen, including `source` and `prior_trials`.
3. **S3 charts:** decile bars with CIs, lag-decay curve, per-year bars, era table, replication table,
   liquidity-tercile bars.
4. **S4 charts:** parameter surface heat map with the chosen plateau outlined, luck band, cost-stress
   table.
5. **S5 charts:** DSR with its inputs, PBO logit histogram, MCPT null histogram with the real value
   marked, walk-forward stitched curve, CPCV Sharpe distribution.
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
| Triple-barrier labels, meta-labeling | AFML ch. 3 | Parked: revisit only as a filter layer on DV2 after P6 |
| Structural-break tests (SADF, CUSUM on prices) | AFML ch. 17 | Era and regime splits in S3 answer our question more simply |
| Synthetic OU "optimal trading rules" | AFML ch. 13 | Produces more tunable parameters, the opposite of what we need |
| HRP, NCO, bet sizing | AFML ch. 10, 16; MLAM | Portfolio construction, a separate project |
| Genetic or automated rule search | various | A mining engine by design; S5 would reject its output anyway |
| Masters' prediction-model machinery (committees, neural nets) | *Assessing and Improving Prediction* | ML models are a non-goal for v1 |

Kept from each source:
- **López de Prado:** registration-first research, PSR, DSR, MinTRL, CSCV/PBO, CPCV with purge and
  embargo, effective N by clustering.
- **Masters:** indicator quality (S2), MCPT of the whole process, walk-forward, and the idea that the
  selection bias of the search is part of the result.
- **Aronson:** placebo and detrending logic, data-mining bias.
- **Harvey–Liu–Zhu:** the t ≥ 3 bar and haircut Sharpe.
- **White, Hansen, Romano–Wolf:** best-of-family tests.
- **McLean–Pontiff:** before and after publication.
- **Carver:** replication across instruments and preference for few rules.
- **Pardo:** the plateau and walk-forward efficiency.

## 13. Build phases

Each phase has a goal, scope and completion criteria, per `AGENTS.md`. Phases P0 and P1 do not touch
strategies and can start now. P2 gates each strategy only after the readiness-audit fix list
(2026-09-28) has landed for that strategy, so no gate is built twice.

| Phase | Scope | Done when |
|---|---|---|
| **P0 Foundation** | panels + snapshot hash, vault seal, ledger with hash chain, `ScoutSpec`, feature library with contracts, S0 and S1 stations, import-boundary test | S1 catches seeded leaks in tests (a future-peeking feature, a split-sensitive feature, a non-PIT member); vault refusal tested; chain tamper tested |
| **P1 Edge toolkit** | S2 and S3 in full, card sections for them, CLI | The Pakal DV2-S&P 500 and QPI studies re-run through S3 with PIT membership; the card shows the old-vs-new difference |
| **P2 Weights engine + gate** | `engines/weights.py`, costs, both account modes, identity gate, deviations registry, CI job | TAA 3x, NDX VXN and CORE5 pass the gate on full history; CI blocks a deliberately broken strategy |
| **P3 Overfitting toolbox** | S4 and S5 (surface, plateau, luck band, cost stress, DSR, PBO, MCPT, WFA, CPCV, SPA, haircut) | Each statistic matches a published worked example or an independent implementation; MCPT on a pure-noise strategy gives a uniform p-value over 200 seeds |
| **P4 Book and card** | S6, full research card, Bench page | Full cards for the three P2 strategies |
| **P5 Re-audition, classes W and X** | Retro registrations, prior-trial counts, cards for all W and X strategies in section 10 order | One card per strategy, owner summary in Hebrew |
| **P6 Path engine** | `engines/path.py`, gates for DV2 and HPI, then their re-audition | DV2 and HPI pass the gate; cards issued |
| **P7 Discovery and shadow** | S7 vault opening, S8 shadow monitor, BY-FDR ledger report | First new family run end-to-end; the vault is opened only through S7 |

## 14. Known limitations of this design

- **Parity with a biased engine.** The gate proves Scout equals the engine, not that the engine is
  right. Truth mode and the deviations registry bound this gap; they do not close it.
- **Permutation nulls destroy all temporal structure.** MCPT answers "is there any structure the process
  exploits", not "is it the structure we believe". S3's mechanism checks and replication carry that
  second question.
- **Retro-registered strategies are optimistic by construction.** The `prior_trials` estimate is a floor,
  not the true number of things the owner once looked at.
- **The vault is short.** It can catch a collapse; it cannot confirm an edge. Only S8 time confirms.
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
