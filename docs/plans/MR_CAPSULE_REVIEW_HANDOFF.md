# MR capsule — review handoff (for Codex)

Date: 2026-10-04. Base: `main` @ `4b6da58` plus the commit that carries this file. Author: Claude, with the owner.

**The ask.** The owner sees the four MR capsule strategies as PM_READY now, and as WIRED later. Before that tier is
relied on, review the build independently.
- The review is read-only unless the owner says otherwise.
- Report each finding with:
  - file:line;
  - a concrete failure scenario (inputs or state, then the wrong result);
  - severity;
  - a suggested fix.

This is a plan. It grants no permission to deploy, to run broker operations, or to change the live engine.
`AGENTS.md` and `docs/ai/PROJECT_GUIDE.md` apply. Tier 1 covers the strategy code; any live wiring is Tier 3.

## 1. What the capsule is

Two mean-reversion pods behind one VIX stress gate, 50/50 capital, reset once a year. The frozen specification and the
research record are in [MR_CAPSULE_20261003.md](../research/MR_CAPSULE_20261003.md): "Final capsule specification
(for the build)" and "Build record (2026-10-04)".

- **Gate (shared):**
  - It opens after a close where VIX is above the expanding mean of all VIX closes since 1990-01-02 (minimum 500 closes).
  - Once open, it stays open at least 15 sessions, counted from the opening close.
  - It closes at the first close at or below the threshold after that.
- **DV2-G:** the wired DV2 rules (`DVO2Strategy`). New entries only while the gate is open; exits never gated.
- **HPI-G:** the wired HPI 2/3/5 vote rules (`HPIStatefulLongStrategy`, vote mode). Same gate.
- **Idle cash, per pod, two variants:**
  - `_spmo` (the owner's 2026-10-04 spec):
    - while the gate is closed, SPMO at min(1, 8% / 20-day realised volatility) of the idle value, set weekly and at gate switches;
    - this applies only once SPMO has traded on each of the last 20 sessions (amendment B1);
    - the rest of the idle cash sits in BIL, and all of it while the gate is open.
  - `_bil`: all idle cash in BIL.
  - In both variants, BIL is bought on re-target closes only and sold whenever the day's orders need the cash (amendment B2).

## 2. Files

| File | Role |
|---|---|
| `strategies/mr_capsule/vix_stress_gate.py` | Gate threshold and state machine, SPMO weight, `gate_state_at` |
| `strategies/mr_capsule/parking.py` | `plan_parking_orders`: pure function, SPMO/BIL whole-share targets |
| `strategies/mr_capsule/capsule_pod.py` | `CapsulePodMixin`: gate read at the decision close, parking after the stock orders, stock-only trade statistics |
| `strategies/mr_capsule/dv2_vix_gated.py` | `DV2VixGatedStrategy` (subclasses `DVO2Strategy`), data load, `run_dv2_capsule_pod` |
| `strategies/mr_capsule/hpi_vote_vix_gated.py` | `HPIVoteVixGatedStrategy` (subclasses `HPIStatefulLongStrategy`), `append_parking_prices`, `run_hpi_capsule_pod` |
| `strategies/mr_capsule/strategy_mr_{dv2_vix_gated,hpi_vote_vix_gated}_{spmo,bil}.py` | The four Bench entry points. Each fixes its parking and its results name |
| `tests/test_strategy_mr_capsule_gate.py`, `tests/test_strategy_mr_capsule_pods.py` | 47 tests |
| `scripts/research/mr_capsule_build_20261004/run_engine.py`, `compare.py` | Engine runs and the reconciliation with the research record |

The wired parents (`strategies/dv2/strategy_mr_dv2.py`, `strategies/hpi/stateful_long.py`) were **not modified**: the
live host imports them. The two pods copy their parent's `iterate` with three changes:
- parking symbols are kept out of the slots and the exit rules;
- the gate wraps the entries;
- the parking call is added at the end.

## 3. Checklist

1. **Timing and lookahead.** Every decision input must be known at Close(previous_bar):
   - the gate (`gate_state_at`, searchsorted `side="right"`);
   - gate switches, read from the gate series (`_gate_switched_at_decision`);
   - the SPMO weight, computed per decision from the engine's `data`, which already ends at previous_bar;
   - the ISO-week end, from the next session's date (a calendar fact);
   - the stock value at Close_T.
2. **Parent drift.**
   - Diff both copied `iterate` bodies against the parents.
   - The tests pin gate-open, no-parking parity trade for trade (including 10 full slots with SPMO/BIL in the frame), but only on synthetic data.
   - Is copying acceptable, or should the parents expose a hook?
3. **The parking plan** (`parking.py`):
   - the 1% buffer and the 1% BIL band;
   - mid-week behaviour (sell BIL only) and the funding of same-open entries;
   - a fully invested pod;
   - missing prices.
4. **Engine contract.**
   - `order_target(shares)` for SPMO/BIL in the same session as the stock `order_value` orders: FIFO, no cash check, cash settles after all fills.
   - Negative cash is the parents' own (DV2-G 136 sessions with parking, 133 without; minimum −8.8% of NAV) and is not financed (G-023).
5. **Data.**
   - SPMO/BIL are CAPITALSPECIAL and never benchmarks.
   - HPI re-declares `attrs["norgate_adjustment_by_symbol_dict"]` after `pd.concat` and keeps the other attrs.
   - The parking ETFs keep Norgate's market-day padding in both pods.
   - Dividends carry the house 25% withholding.
   - The pods enable the dividend ledger explicitly (`configure_dividend_cash_ledger(enabled_bool=True)` in the mixin's `__init__`), as the repo's other BIL holders do. Pricing without a Dividend field fails loud instead of silently earning about 0% on parked cash.
6. **Live replay.**
   - The live host builds a fresh strategy object for each decision (`strategy_host._seed_strategy_state`, then one `iterate`).
   - Confirm the pods keep no behavioural state between `iterate` calls. Re-target triggers come from the calendar and the gate series.
   - Test: `test_a_fresh_object_retargets_only_on_the_calendar_and_gate_triggers`.
7. **Registry.**
   - The four entry points are PM_READY by module path, and absent from the live release allowlist and the host sets.
   - `tests/test_strategy_registry.py` should pass.

## 4. How to reproduce

Run Norgate-loading commands one at a time: concurrent loads have hung on this machine.

```bash
uv run pytest tests/test_strategy_mr_capsule_gate.py tests/test_strategy_mr_capsule_pods.py tests/test_strategy_registry.py -q
uv run python scripts/research/mr_capsule_build_20261004/run_engine.py dv2 parked
uv run python scripts/research/mr_capsule_build_20261004/run_engine.py hpi parked
uv run python scripts/research/mr_capsule_build_20261004/run_engine.py dv2 cash
uv run python scripts/research/mr_capsule_build_20261004/run_engine.py hpi cash
uv run python scripts/research/mr_capsule_build_20261004/run_engine.py dv2 bil
uv run python scripts/research/mr_capsule_build_20261004/run_engine.py hpi bil
uv run python scripts/research/mr_capsule_build_20261004/compare.py
uv run python scripts/research/check_pm_readiness.py strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil --check-determinism
```

Each engine run takes about 7–10 minutes. `compare.py` writes `results/research/mr_capsule_build_20261004/compare.json`.
`check_pm_readiness.py` writes to `results/research/pm_readiness/<timestamp>/pm_readiness.json`.

## 5. Evidence so far

| Check | Result |
|---|---|
| HPI-G, cash at 0%, vs the research engine run | Identical: 5,998 of 5,998 trade events, NAV max difference 0.0 |
| DV2-G, cash at 0%, vs the research replica | 98.7% of trade events in common; daily corr 0.9988; Sharpe 0.951 vs 0.955 |
| Gate vs the research gate | Identical on real VIX 2004–2026 |
| Capsule 2004–2026, `_spmo` / `_bil` / cash at 0% | 15.01% / 1.034 / −23.5%; 14.31% / 1.019 / −21.4%; 14.18% / 1.010 / −21.6% |
| PM_READY checks (capital ×2, total-return benchmark, determinism) | All four PASS (section 7) |
| Independent quant-pitfalls review (Claude agent) | No lookahead, leakage or parent drift. Findings fixed: BIL churn, a vacuous gate test, no-volume ETF days, signal-audit state, trade statistics, test gaps |

## 6. Open owner decisions and known caveats

- **Parking (final owner decision, 2026-10-04):** BIL is the main version, and SPMO is kept as a ready alternative. Review both variants, and wire BIL first. The two tie on book Sharpe since 2017-11 (1.484 BIL vs 1.464 SPMO); BIL is simpler and scales further (shared-gate SPMO switches hit one auction).
  - Since SPMO traded every day (2017-11), SPMO adds about 2 pp/yr to the capsule (0.4 to the book) at a statistically equal Sharpe (P 0.54 capsule, 0.30 book).
  - It loses more in sell-offs: Q4 2018 −12.8% vs −6.3%.
  - Both variants need the same new live order shape, so wiring BIL first keeps SPMO one configuration away.
- **Selection.** About 100 gate variants and about 10 parking forks were tried. The DSR is 0.966 (N = 110, engine capsule, 2004–26) but 0.78 from 2018.
- **Conservative costs.** BIL dividends carry the house 25% withholding, and BIL trades pay 2.5 bps; BIL's spread is about 1 bp.
- **Small pods.** About 52–58 parking orders a year at the USD 1 minimum commission.
- **Trade statistics.** Bench trade statistics and exposure time count stock trades only; NAV and costs include parking.

Caveat table: [book-strategy-caveats.md](../strategies/book-strategy-caveats.md).

## 7. PM_READY check results

`scripts/research/check_pm_readiness.py --check-determinism`, 2026-10-04, on the final code. Three runs each: capital
C, capital 2C, and C again.

| Strategy | Capital: $100K → / $200K → (scale) | Benchmark | Determinism |
|---|---|---|---|
| `strategy_mr_dv2_vix_gated_bil` | $2,941,577 / $5,901,662 (2.006) | PASS, gap 0.00% | PASS |
| `strategy_mr_hpi_vote_vix_gated_bil` | $1,403,312 / $2,816,111 (2.007) | PASS, gap 0.00% | PASS |
| `strategy_mr_dv2_vix_gated_spmo` | $3,424,962 / $6,871,179 (2.006) | PASS, gap 0.00% | PASS |
| `strategy_mr_hpi_vote_vix_gated_spmo` | $1,595,112 / $3,199,270 (2.006) | PASS, gap 0.00% | PASS |

Each $100K final value equals, to the cent, the build-check engine run of the same variant. So the two late changes
(gate switches read from the gate series; the dividend ledger enabled explicitly) changed no result.
The results are in `results/research/pm_readiness/` (gitignored). Changes after the checks change no behaviour:
- module docstrings;
- the order of class attributes;
- a loader guard requiring `(BIL|SPMO, "Dividend")` columns. It passes on the data; smoke runs of DV2 and HPI succeeded through `run_strategy.py`. The registry entries and the books
`portfolios/mr_capsule_{spmo,bil}.yaml` are in the commit that carries this section.

## 8. Not built yet: Bench capacity and timing hooks

The four entry points have no `build_capacity_analysis_inputs` or `build_execution_timing_analysis_inputs`, so Bench's
Capacity and Timing buttons are absent for them. The owner will ask a separate chat to add them.

- **Precedents:**
  - DV2: `strategies/dv2/strategy_mr_dv2.py:23` and `:100`;
  - HPI: `strategies/hpi/stateful_long.py:876` and `:924`, wrapped by `strategy_mr_hpi_sp500_2_3_5_vote.py`.
- **Requirement:** the hooks must build the capsule pod classes (gate and parking), not the parents.
- **Tests that move:**
  - `tests/test_bench.py` pins the count of capacity-hook modules (48 since 635f2f4);
  - `tests/test_run_capacity_analysis.py` imports every hook module.

## 9. WIRED prerequisites (Tier 3, not in scope now)

1. **Owner decisions:**
   - which parking variant;
   - a paper run first;
   - margin accounts (gate-opening entries are funded by the same auction's sales);
   - one account per pod (both pods hold BIL).
2. **`alpha/live/strategy_host.py`:** add a decision-plan builder per pod, modelled on `_run_dv2_strategy_for_live_decision` and the HPI builder. Each builder must:
   - load S&P 500 plus SPMO and BIL;
   - set `vix_close_ser` from $VIX since 1990;
   - seed the state, including SPMO/BIL positions;
   - run one `iterate`.
3. **Live order contract (the main new work).** The incremental contract for DV2/HPI (`_classify_incremental_order_shape` in `alpha/live/strategy_host.py`) maps only two order shapes:
   - `entry_value`: MarketOrder, `target=False`, unit `value`, amount > 0;
   - `exit_to_zero`: MarketOrder, `target=True`, amount 0.

   The parking emits a third shape: `order_target(SPMO/BIL, N shares)` with N > 0, a resize. Today that raises `NotImplementedError`. Wiring the parking needs one of:
   - a new intent kind (target shares), with its sizing, broker quantity and reconciliation semantics;
   - a hybrid of the incremental family and the full-target family.

   Either way this is live-impact checklist territory.

   **A smaller first stage is possible:** the gate only, with idle cash left as cash in the account (the backtest's `cash` mode, 0% by house convention). Its orders fit the existing two shapes.
4. **Pod state:** parking trade ids (from 900,000,000, `_parking_trade_id_map`) are not part of `_extract_strategy_state_dict`. A fresh live object would reuse ids across days. Decide whether parking needs persisted ids, or whether the live intent ids replace them.
5. **Data:** the snapshot data profiles need SPMO, BIL and $VIX from 1990.
   - **Stale VIX.** `_gate_switched_at_decision` compares the gate row before the decision row with the decision row. If VIX arrives a session late live, a switch can be missed until the week end.
   - **Fix to wire with the host:** use the gate state at the previous pricing session (`data_df.index[-2]`), and fail loud when the gate series has no row for the decision date. On Norgate history both forms are identical: the VIX and S&P calendars match.
   - **Reference runs.** Reference-compare runs build the strategy inside `dividend_cash_ledger_disabled_context()`. Because the capsule pods enable the ledger in `__init__`, the contract check reports mode "enabled", and the V3 dashboard hides its "disabled or unproven" warning (`alpha/live/reference_compare.py:137-162`). This is an engine-level issue for any strategy that enables the ledger explicitly.
6. **Release surfaces:** the release manifest allowlist, a release template, the host import sets, the registry tier WIRED, and the `tests/test_live_*` consistency tests.
7. **Verification:** the live-impact checklist, then parity, failure-mode and coverage review agents.
