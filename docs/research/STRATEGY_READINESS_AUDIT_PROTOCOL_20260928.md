# Strategy readiness audit — frozen protocol (2026-09-28)

Frozen before any new result of this audit was inspected. The SHA-256 of this file is recorded in
`STRATEGY_READINESS_AUDIT_PROTOCOL_20260928.sha256` next to it. Any later change goes into
`STRATEGY_READINESS_AUDIT_PROTOCOL_20260928_AMENDMENTS.md` with a date, a reason and whether it was made before or
after seeing a result. This file is never edited after freezing.

Context read before freezing (prior results, not this audit's): the 2026-09-27 leakage hunt verdict and its findings
files, the paused share-units handoff, the registry, the release manifest and `alpha/live/strategy_host.py`.

## 1. Goal and question

For every in-scope strategy answer three separate questions, each with its own verdict:

- **BC — backtest correctness.** Is the reported backtest free of look-ahead, data artefacts and accounting errors?
- **LP — live parity.** For wired strategies: does the live host make the same decisions as the backtest on the same
  information? For unwired strategies: `NOT WIRED` plus a one-line note on what a live route would need.
- **TR — tradability.** Can it be traded as modelled at USD 30K (owner), USD 1M and USD 10M?

A strategy is guilty until the evidence says otherwise: a mandatory check that was not run caps that dimension at
`READY WITH CAVEATS` and is listed under "not tested".

## 2. Code under audit

- Commit `cb29d4f` (HEAD of `main` and of this worktree at freeze time). Only committed code is audited.
- Data: the local Norgate installation on this workstation (one vintage, last bar 2026-09-25) and FRED/ALFRED where a
  strategy uses macro data. The VPS snapshot store is not accessed; snapshot-profile differences are audited by code
  read of `data/norgate_snapshot_store.py` and `data/norgate_loader.py`.
- No strategy, engine, data-loader, live, release-manifest, registry-tier or portfolio-YAML file is modified. New files
  only under `tests/` (audit tests), `scripts/research/strategy_readiness_audit_20260928/` and
  `results/research/strategy_readiness_audit_20260928/`, plus the two audit documents.

## 3. Scope and order

Tier A (wired, finish first and report as its own milestone): TAA 3x; NDX ATR and NDX ATR VXN; TAA 1/N; BTAL_QQQ;
HPI 2/3/5 vote and HPI IBS/RSI exit; DV2; CORE5; QPI.
Tier B: Inflation Compass and its QQQ variant; Tactical FI; EOM flow; Industry-ETF DV2; sector VOX/IYR and the three
KIE/IHI dispersion variants; NDX NATR20 VXN.
Tier C: crisis_trend_core, vixm_backwardation, TAA 2x QLD/SSO/BTAL-QLD, linearity no-BTAL, Trinity vol control.
Excluded: MOSAIC (one-line recommendation on its PM_READY tier only).

## 4. Checklist per strategy

### A. Engine correctness (BC)

| ID | Check | Method | Mandatory for |
|---|---|---|---|
| A1 | Feature code read | Every feature: formula, file:line, input field and adjustment, window endpoints, shift direction, resample/month-end rule, joins (`merge_asof` direction and `allow_exact_matches`), fills (`ffill`/`fillna`), universe filter. Recorded as a feature table. | all |
| A2 | Future corporate-action invariance | Real data. One symbol's full loaded history rescaled as if a k:1 split happened after the last date, k in {40, 0.1, 1.5}; OHLC and Dividend divided by k, Volume multiplied by k, `Unadjusted Close` and `Turnover` kept nominal. At least 3 symbols per strategy, including one that the strategy actually held. Compare decisions (selected set, rank, target weights, or engine trade list). | all |
| A3 | Truncation invariance | Decisions computed from data ending at T equal full-history decisions at T. At least 8 cut-offs: mid-month, a month-end on a weekend, a month-end before a holiday, the first session of a month, the last completed month-end, and the current partial month. | all |
| A4 | Positive control | One injected leak (for example the signal reading Close_(T+1)) that the harness of A2/A3 must catch. At least once per harness family. | per harness |
| A5 | Macro and helper timing | For every non-price input (FRED, VIX/VXN, breakevens): observation date vs publication time vs decision time. Causal replay with one extra session of lag; ALFRED vintages where they exist. | strategies with such inputs |
| A6 | Universe point-in-time | Membership source, the 5-session trim (`data/norgate_loader.py:107`), delisted-name handling, liquidation price. | index-universe strategies |
| A7 | Padded/stale bars | Count decisions or fills on padded or zero-volume bars. | all |
| A8 | Accounting | Price adjustment role (fills on CAPITALSPECIAL, TR only for return signals/benchmarks), dividends, commission units (adjusted vs raw shares), slippage, borrow for shorts, negative cash and its financing, rounding, fill timing vs data availability. | all |
| A9 | Capital scaling | Same run at two capitals; return paths must agree up to rounding and minimum-fee effects. | all (reuse PM-readiness evidence if on record and code unchanged) |
| A10 | Determinism | Two identical runs give bit-identical equity. | all (reuse if on record) |

### B. Live parity (LP), wired strategies only

| ID | Check | Method |
|---|---|---|
| B1 | Decision replay | Call the live host builder (`alpha/live/strategy_host.py` / `core5_adapter.py`) with data available as of each historical decision date and a pod state equal to the backtest's state; compare decision plans to the backtest decisions exactly (sets, target weights to 1e-6, entry/exit intents). Monthly strategies: at least 24 month-ends. Daily strategies: at least 20 sessions that include entries and exits. |
| B2 | Invocation timing | What the host does when invoked mid-month, on a holiday, after a missed month-end, or on the first session of the month (catch-up). |
| B3 | Data profile | Differences between the backtest loader and the live snapshot profile (symbols, fields, padding, adjustment, start date, membership). |
| B4 | Guards | Missing price, stale data, halted or delisted holding, zero or negative price: fail closed or silent last price? |
| B5 | Execution path | Target-to-order conversion (reference price, rounding, cash reserve, sell-before-buy), partial fills and residual handling, corporate actions between signal and fill, order type vs backtest fill assumption. |

### C. Tradability (TR)

| ID | Check | Method |
|---|---|---|
| C1 | Order size vs ADV | For each historical order: order notional / 20-session median dollar volume (native `Turnover` where available), at USD 30K, 1M, 10M. Report the median and the 99th percentile, and the last 3 years separately. |
| C2 | Whole shares at small size | Simulated whole-share holdings at USD 30K vs target weights: weight error and cash drag; names priced above the per-name budget. |
| C3 | Instruments | Inception, current assets, closure/merger risk for ETFs; leveraged-ETF daily reset (already inside the ETF price); short/borrow availability and cost. |
| C4 | Order type | MOO/MOC feasibility on IBKR for every traded symbol (listing venue, auction), and account-type needs (margin vs cash, settlement, negative cash in the backtest). |

### D. Selection and robustness (noted, not re-done)

Record the known trial count and prior robustness evidence; add nothing new unless a finding requires it.

## 5. Pass/fail rules

Materiality: an issue is **material** if its measured or bounded effect on the strategy's own backtest is at least
0.25 pp CAGR, or 0.02 Sharpe, or 1 pp max drawdown, or if it can make a live order differ from the backtest intent
on any realistic date, or if it cannot be bounded.

| Verdict | BC | LP | TR |
|---|---|---|---|
| READY | All mandatory checks run; no material issue; residual issues are conservative or immaterial. | Replay matches exactly on all tested dates; guards fail closed; no invocation path produces a different decision. | At USD 30K and 1M: median order ≤ 1% ADV and 99th percentile ≤ 5% ADV; whole-share weight error ≤ 2% of NAV per name; no instrument at material closure risk; order type available. |
| READY WITH CAVEATS | No optimistic material issue; conservative material issues, or a mandatory check not run, or bounded non-material optimistic issues. | Differences exist but are intended, documented and bounded (for example a known lag), or a mandatory check not run, or a failure mode that is loud (crash/park) rather than silent. | Passes at USD 30K and 1M but fails at 10M, or whole-share error 2–5% of NAV, or a documented account-type need. |
| NOT READY | Any optimistic material issue (look-ahead, artefact, accounting) that is unfixed. | Any replay mismatch without a documented intended cause, or a silent failure mode (stale data or last price used without a block) that can produce a wrong order. | Fails at USD 1M, or at USD 30K the model cannot be held (whole-share error > 5% of NAV per name or a required instrument/order type unavailable). |

A finding that could affect a live account today (TAA 3x or NDX VXN) is escalated to the owner immediately, before the
audit continues.

## 6. Evidence format

Every finding is recorded as:

```text
ID            e.g. A-TAA3X-03
strategy      import string
dimension     BC | LP | TR
severity      S1 live money at risk now | S2 optimistic/material | S3 conservative, immaterial or documentation
issue         issue_description_str
bias          expected_bias_direction_str (optimistic / conservative / neutral / unknown)
impact        quantified (pp CAGR, Sharpe, MaxDD, flips) or an explicit bound
evidence      file:line, test name, script, output path
mitigation    mitigation_str, written so Codex can act on it
```

Each scorecard also lists what was not tested and why.

## 7. Reuse of prior work

Prior results (the 2026-09-27 leakage hunt, the HPI and Tactical FI handoffs, the Inflation Compass deep research, the
DV2 deep research, the NDX parameter-robustness report, the share-units handoff) are inputs, not evidence. Every key
claim reused in a verdict is re-verified independently (re-run or re-derived); a claim that is only cited is marked
"cited, not re-verified". Where prior work stopped (live parity, tradability, strategies not covered, such as QPI),
this audit extends it.

## 8. Independent review

After each tier, at least two read-only reviewers with different lenses (quant pitfalls and look-ahead; live parity
and failure modes; tradability) try to break the conclusions. Confirmed findings are folded in; rejected findings
are recorded with the reason.

## 9. Outputs

- Report: `docs/research/STRATEGY_READINESS_AUDIT_20260928.md` (scorecards, ranked fix list, owner summary in Hebrew).
- Study code: `scripts/research/strategy_readiness_audit_20260928/`.
- Outputs: `results/research/strategy_readiness_audit_20260928/` (gitignored, local).
- Audit tests: `tests/test_strategy_readiness_audit_*.py`.
- Nothing is committed or pushed without the owner's explicit approval.
