# Architecture Review — 2026-04-17

**Reviewer stance:** Senior quant researcher + infrastructure engineer. Reviewing a pre-capital platform transitioning from paper to personal capital, then external capital.

**Scope:** Architecture, infrastructure, execution. **Not** alpha quality or parameter choices.

**Primary concern:** Live / backtest parity.

**Operating principle:** Speed-to-live beats completeness, **but never at the cost of parity or safety rails**.

---

## 1. What's Actually Good

A pre-capital system rarely gets this many things right. Call it out so it doesn't get refactored away under pressure.

### 1.1 Doctrine is written down and specific

`QUANT_PHILOSOPHY.md` names eight pitfalls and the operational defense for each. `ASSUMPTIONS_AND_GAPS.md` logs thirteen known gaps (G-001 through G-013) with severity and impact. This is rarer than it should be. Most solo quants carry this in their heads; you've externalized it, which means it survives context loss and can be enforced by code review.

### 1.2 Decision / Execution split is the right shape

The `DecisionPlan → VPlan → BrokerOrder` pipeline ([alpha/live/models.py:46-141](alpha/live/models.py:46)) cleanly separates "what the strategy wants" from "what we can actually do at submit time." This is the deterministic order-clerk pattern the doctrine mandates, and it's implemented correctly: overnight freezes intent, pre-submit freezes shares. The two-stage model in [LIVE_START_HERE.md](LIVE_START_HERE.md) is faithful to the code.

### 1.3 Strategy contract is narrow and enforced

[alpha/live/strategy_host.py:188-213](alpha/live/strategy_host.py:188) rejects any order shape other than the two canonical forms (`entry_value` and `exit_to_zero`) with an explicit `NotImplementedError`. A strategy that drifts from the contract fails loud, immediately, at the host boundary — not silently at submit time. This is the single most important safety rail in the live layer.

### 1.4 Submission idempotency is real

`submission_key_str = f"vplan:{decision_plan_id}"` plus `count_broker_orders_for_vplan` check at [alpha/live/runner.py:1250-1329](alpha/live/runner.py:1250) means a re-run of `submit_vplan` does not create duplicate orders. Combined with the 5-minute lease at [alpha/live/runner.py:1941-2024](alpha/live/runner.py:1941), tick serialization is robust against overlapping cron invocations. This is worth real money the first time a scheduler misfires.

### 1.5 Paper/live gating via account prefix

Paper accounts start with `DU`, live accounts don't. Gating at the adapter rather than a config flag is the right choice — config flags get flipped by accident, account numbers don't. Pod manifests in `alpha/live/releases/excelence_trade_paper_001/*.yaml` pin account strings per pod, so a live manifest accidentally run in paper mode fails the adapter check rather than routing to the wrong book.

### 1.5a One pod = one sub-account = one independent NLV

Each pod manifest pins a distinct IBKR account (`DUK322077`, `DU5566778`, `DU2233445`, `DU3344556`). `broker_snapshot.net_liq_float` is therefore the pod's own sub-account NLV, and each sub-account compounds independently of the others. This is the correct mapping from the backtest portfolio model ("sum of independent pod equities") onto IBKR's account primitives. The design cleanly avoids the daily-rebalancing anti-pattern that a single shared account would create. Funding, allocation, and risk budgeting can be decided per sub-account without cross-pod coupling.

### 1.6 State store has V1 archive + V2 active

`state_store_v2.py` preserves the V1 schema for cutover audit. When something post-cutover looks wrong, you can diff against pre-cutover state rather than reconstructing from logs. This is a mature choice that most solo platforms skip.

### 1.7 Share drift is a warning, not a block

The design explicitly prefers "size from broker truth and warn" over "reject on drift" ([LIVE_START_HERE.md:70-78](LIVE_START_HERE.md:70)). This is correct for equity strategies at your scale: rejecting because the strategy thinks you hold 100 shares but IBKR shows 99 would strand the system on every split, dividend-reinvestment event, or previous-day partial fill. Warn-and-reconcile is the right trade.

### 1.8 Reconcile semantics are honest

[LIVE_START_HERE.md:155-168](LIVE_START_HERE.md:155): the VPlan does not get marked complete until residual shares are zero, and exit-to-zero with non-zero broker shares raises a `critical` event. This is the anti-"stuck exit" rail that most solo systems discover they needed after their first outage.

---

## 2. Top 5 Concerns, Ranked by Severity

Severity = probability × blast radius at real capital. I'm ignoring "things that would be nice" and focusing on things that actually cost money or blow up silently.

### HIGH — #1: Sizing price differs between backtest and live

**Where:** [alpha/engine/strategy.py:277-293](alpha/engine/strategy.py:277) (`_get_order_sizing_price_float`) vs [alpha/live/execution_engine.py:69-73](alpha/live/execution_engine.py:69).

**Backtest:** `order_value` / `order_target_value` size shares from **previous_bar close**, marked `*** CRITICAL***`. Execution then prints at current_bar open × slippage penalty.

**Live:** Shares computed from `live_reference_price` snapshot at VPlan build time, which is ~10-15 minutes before execution (pre_close_15m clock).

**Divergence mechanics:**
- Backtest: share_count uses close_t-1; fill uses open_t. Gap between the two is the overnight move.
- Live: share_count uses live_quote at 15:45; fill happens at close. Gap is intraday drift, typically smaller but different in sign distribution.
- On high-gap days (macro news, earnings clustered in the universe), backtest and live shares for the same target_weight diverge by overnight gap magnitude. Median S&P500 |overnight gap| is ~30-50bps, tail days can be 2-5%.

**Blast radius:** For DV2 (daily turnover, 10 positions, ~10% per name) a 2% sizing price error = ~20bps notional drift per name per day. Across 10 names compounded, that's material against the strategy edge.

**Probability:** 100%. Every single trade.

**Note:** this is a design choice, not a bug — live *should* size from broker truth and current price. The problem is that the **backtest does not match this choice**. Either add a backtest mode that mimics "size from next-bar-open" (changes backtest semantics), or accept this is irreducible and measure its impact via a parity test harness (see §7).

---

### HIGH — #2: No parity test harness exists

**Where:** Not present. Searched 76 test files in `tests/`; nothing compares `DecisionPlan → shares` output between backtest and live code paths.

**Why this is HIGH:**
- Every other concern on this list is detectable with a parity harness that feeds the same strategy, same prices, same account state through backtest and live sizing, and asserts share counts match within a declared tolerance.
- Without it, you cannot regress-test any change to either side. A refactor to `execution_engine.py` can silently introduce drift that nobody notices until a live fill looks weird.
- The doctrine says *"Live trading must preserve backtest semantics up to irreducible market frictions. If the live implementation changes strategy meaning, then it is not the same strategy."* ([CLAUDE.md](CLAUDE.md)). Parity is not verified, only asserted.

**Blast radius:** Every time you change either side, you risk unnoticed divergence. This compounds with time.

**Probability of divergence in next 6 months:** very high. You will keep iterating on execution_engine and strategy.

---

### MEDIUM — #3: Signal audit is off by default

**Where:** [alpha/engine/backtester.py:72-74](alpha/engine/backtester.py:72): `audit_enabled_bool` defaults to False.

**What it does when on:** recomputes signals on truncated histories at random bars and checks for divergence — the canonical lookahead detector.

**Why it matters:** DV2's `shift(126)` and `rolling(200).mean()` at [strategies/dv2/strategy_mr_dv2.py:36-39](strategies/dv2/strategy_mr_dv2.py:36) are **not marked `*** CRITICAL***`** per doctrine. The audit is the backstop for missing markers. Defaulting it off means the backstop only runs when someone remembers to flip it.

**Blast radius:** A lookahead bug shipped to live quietly inflates backtest results. It does not directly lose money (live is point-in-time), but it means the backtest you used to size capital is wrong, which means your risk budget is wrong.

**Probability:** Unknown without running the audit. Should be the first thing turned on before any further live capital.

---

### MEDIUM — #4: No per-pod notional cap beyond fraction × net_liq

**Where:** [alpha/live/execution_engine.py:51](alpha/live/execution_engine.py:51). The only sizing bound is `pod_budget_fraction × net_liq`.

**Failure modes not guarded:**
- If `net_liq` is mis-reported by IBKR (stale snapshot, FX conversion glitch, margin loan artifact), the pod sizes against a wrong base.
- If a pod's target_weight_sum exceeds 1.0 (bug in strategy code — possible given `full_target_weight_book` math), the pod can overshoot its own budget.
- Non_positive_net_liq is guarded at [alpha/live/runner.py:1184](alpha/live/runner.py:1184), but absurdly_high_net_liq is not.

**What's missing:** An absolute dollar cap per pod, per symbol, and per order, independent of net_liq. Even a crude `max_dollars_per_pod_float: 50000` in the manifest would prevent a single bad net_liq read from wiping the account.

**Blast radius:** Tail risk. Low probability, catastrophic outcome. This is exactly the kind of rail that costs nothing to add and saves you the one time it matters.

---

### MEDIUM — #5: `pod_budget_fraction_float` is a scaling throttle, not a weight — its live value is undeclared

**Where:** All four pod manifests in [alpha/live/releases/excelence_trade_paper_001/](alpha/live/releases/excelence_trade_paper_001/) set `pod_budget_fraction_float: 0.03`. Used at [alpha/live/execution_engine.py:51](alpha/live/execution_engine.py:51) and `:145`.

**Architecture (corrected):** Each pod maps 1:1 to its own IBKR sub-account (`DUK322077`, `DU5566778`, `DU2233445`, `DU3344556`). `broker_snapshot.net_liq_float` is the **sub-account's** own NLV. Each sub-account compounds independently — this matches the backtest "independent pod compounding" model. So the shape is correct.

**What the fraction actually does:** Scales the deployed notional down from sub-account NLV. With `fraction = 0.03`, only 3% of the sub-account's NLV is deployed per VPlan; the remaining 97% sits as idle cash. This is paper-trading safety sizing.

**The concern is operational, not semantic:**
- Backtest uses `previous_total_value` (full pod equity, i.e., effectively fraction = 1.0) to compute order sizes.
- Live uses `fraction × NLV` — at 0.03, live is trading at 3% of what the backtest assumes.
- If live fraction stays at 0.03 after personal-capital deployment, live P&L ≈ 0.03 × backtest P&L (plus idle-cash drag).
- There is no documented contract on what fraction *should* be in live, nor whether it changes across the paper → personal → external-capital transitions.

**Related: sub-account initial funding.** Parity between backtest and live also requires the sub-account to be funded at (or proportional to) the backtest's `capital_base`. If the backtest simulates with $100k pod capital and the sub-account is funded with $50k, compounded P&L diverges by 2× even at `fraction = 1.0`. This funding contract is not documented.

**Blast radius:** Not catastrophic — it is a systematic scaling of live return vs backtest. But it means "backtest Sharpe of X translates to live Sharpe of X" is only true when (a) sub-account NLV matches backtest capital_base, and (b) fraction is set to 1.0 (or whatever fraction matches the backtest's deployment assumption). Currently neither is declared.

**Probability:** 100%. The value 0.03 is in every manifest and will be inherited unchanged unless someone explicitly changes it.

---

## 3. Live / Backtest Parity — Detailed Map

Each row: one axis of semantic fidelity. Columns: live behavior → backtest behavior → probability of divergence → expected impact.

| Axis | Live Behavior | Backtest Behavior | Divergence Probability | Divergence Impact |
|------|---------------|-------------------|-----------------------|-------------------|
| **Pod capital base** | `fraction × sub_account.net_liq` — each pod = own IBKR sub-account, compounds independently ([execution_engine.py:51](alpha/live/execution_engine.py:51)) | Independent compounding from `capital × weight` ([portfolio.py](alpha/engine/portfolio.py)) | Shape matches; scale differs by `fraction` | LOW-MEDIUM — if `fraction ≠ 1.0`, live deploys a scaled-down version of the backtest; semantics match but magnitudes don't |
| **Sizing price** | `live_reference_price` at pre_close_15m ([execution_engine.py:69-73](alpha/live/execution_engine.py:69)) | `previous_bar.Close` ([strategy.py:277-293](alpha/engine/strategy.py:277)) | 100% | MEDIUM — typical ~30-50bps, tail 2-5% |
| **Execution price** | Market close (TWS MOC or similar) | `current_bar.Open × (1 + sign × slippage)` ([strategy.py:360](alpha/engine/strategy.py:360)) | 100% | MEDIUM — different bar, different slippage distribution |
| **Slippage model** | Real fills, no penalty | Fixed 0.00025 per share, always unfavorable ([strategy.py:43](alpha/engine/strategy.py:43)) | 100% | LOW-MEDIUM — model is conservative, real slippage at your size should be lower |
| **Commission** | IBKR tiered (per-share or per-value) | `max(1.0, 0.005 × shares)` ([strategy.py:110-114](alpha/engine/strategy.py:110)) | 20% | LOW — IBKR is roughly at this level, not exact |
| **Share rounding** | `floor((weight × budget) / price)` ([execution_engine.py](alpha/live/execution_engine.py)) | `int()` truncation ([order.py:82,84,91,93](alpha/engine/order.py:82)) | HIGH on short sales | MEDIUM — `int()` truncates toward zero, `floor` toward -∞; for negative shares these disagree |
| **Cash check** | IBKR rejects if insufficient buying power | None — backtest can overdraw | Rare in practice | MEDIUM when it happens — live rejects order, backtest books phantom fill |
| **Share drift (strategy vs broker)** | Warn, size from broker ([LIVE_START_HERE.md:70-78](LIVE_START_HERE.md:70)) | Not possible — simulator is single source of truth | 100% some drift | LOW — warning, not block |
| **Dividends** | Cash arrives at broker, position unchanged | CAPITALSPECIAL pre-adjusts price ([data/norgate_loader.py](data/norgate_loader.py)) | 100% | LOW if reconciler handles cash correctly; UNKNOWN whether it does |
| **Splits** | IBKR rebases position (100 → 200 shares at half price) overnight | CAPITALSPECIAL pre-adjusts price series | 100% on split days | MEDIUM — reconciler must accept that broker share count changed overnight without a trade |
| **Order shape** | Strict: `entry_value` or `exit_to_zero` only ([strategy_host.py:188-213](alpha/live/strategy_host.py:188)) | Any order method on Strategy base class | N/A at contract level | Contract enforcement prevents divergence |
| **Timing / clock** | `pre_close_15m` (actual wall clock) | Bar index (no wall clock) | Intraday timing varies | LOW-MEDIUM — if close moves fast, live fills differ from an "assume close" backtest |
| **Leverage** | IBKR buying power | No check | 100% if strategy overshoots 1.0 | HIGH when it happens — see §2.5 |

**Bottom-line parity score:** the pipeline shape is right; the sizing math is wrong in two important ways (pod base, sizing price). A parity harness would catch both.

---

## 4. Architectural Tensions

Reasonable choices in conflict. The code currently picks one without documenting why. Flag for awareness; do not necessarily change.

### 4.1 "Warn and reconcile" vs "refuse to trade on drift"

Current choice: warn and reconcile. Correct for equities, wrong for options / futures where tiny drifts compound.

**Tension:** If you ever add an options pod, the reconcile semantics need a per-asset-class override. Hard-coding warn-and-reconcile at the framework level will cost you the first time you add a non-equity book.

**Recommendation:** Fine for now. Document in `ASSUMPTIONS_AND_GAPS.md` that this is an equity-only rail.

### 4.2 Sizing price: live reference vs previous close

Current choice: live reference in live, previous close in backtest.

**Tension:** Two defensible semantics, but only one is "the strategy's definition." Right now, neither side is declared canonical, so you can't say which side is "wrong" when they disagree.

**Recommendation:** Declare the canonical semantics in `QUANT_PHILOSOPHY.md`. My vote: live is canonical (it's what actually executes), and the backtest should sim-reference at next-bar open. This is a bigger change than accepting the drift, so defer until you have the parity harness to measure the impact.

### 4.3 Pod budget fraction: safety throttle vs production knob

Current choice: `pod_budget_fraction_float = 0.03` across all manifests. Each pod has its own IBKR sub-account, so compounding is independent and matches backtest semantics.

**Tension:** The fraction serves two conflicting purposes:
1. **Paper-testing safety throttle** — 0.03 means bugs cost pennies, not real dollars.
2. **Production deployment knob** — at live, this needs to be the real deployment ratio. Leaving it at 0.03 means trading 3% of sub-account NLV forever.

There is no documented place where the intended live value is declared, no lifecycle stage mapping (paper → personal → external), and no validation that the fraction was consciously set rather than inherited.

**Recommendation:** Add a per-pod explicit declaration in the manifest of the intended live fraction and the paper-test fraction — e.g., `pod_budget_fraction_paper_float: 0.03` and `pod_budget_fraction_live_float: 1.0`. Require the deploy script to pick the correct one based on `deployment.mode`. Forces conscious decision at the cutover boundary.

### 4.4 Scheduler-service serve vs tick cron

Current choice: both exist. `tick` can run standalone via cron; `scheduler_service` wraps it with wall-clock signal clocks.

**Tension:** Two entry points with overlapping concerns. A bug in one doesn't imply a bug in the other. Operator confusion is a real risk — which is canonical?

**Recommendation:** Pick one for real capital operation and mark the other experimental. The 5-minute tick lease protects you from double-execution, but operator cognitive load is a separate cost.

### 4.5 Manifest per pod vs per release

Current choice: per-pod YAML inside a release directory.

**Tension:** At three pods this is fine. At ten pods, per-pod manifests become fragile — same account, same budget fraction, duplicated. The release-level grouping exists but isn't exploited.

**Recommendation:** Fine for now. Revisit when pod count hits ~5.

---

## 5. Implicit Assumptions

Things the code assumes without checking. Each is a probability-weighted risk.

### 5.1 IBKR snapshot is internally consistent

The code assumes `broker_snapshot.net_liq_float` and `broker_snapshot.positions` are captured atomically (or close enough). [alpha/live/ibkr_socket_client.py](alpha/live/ibkr_socket_client.py) does not enforce atomicity; if positions update mid-snapshot, net_liq and positions can disagree. Low probability, high blast if it happens during sizing.

### 5.2 IBKR `net_liq` includes locked/pending cash correctly

There is no cross-check against `available_funds` or `buying_power`. If a large unsettled trade is pending, `net_liq` may be right but `buying_power` is lower — and the sizing math uses `net_liq`. Orders can be rejected by IBKR for insufficient buying power even though the sizing math looked fine.

### 5.3 Norgate is available at build_decision_plan time

[alpha/live/scheduler_utils.py:48-52](alpha/live/scheduler_utils.py:48) uses `$SPX` as the heartbeat. If Norgate is down, the heartbeat fails and `build_decision_plan` is skipped. Good — this is explicit. But the dependency on Norgate for overnight decision building means any Norgate outage = no trades tomorrow. No fallback data source.

### 5.4 Strategy state survives pickle round-trip

[alpha/live/strategy_host.py:88-132](alpha/live/strategy_host.py:88) seeds strategy state from `PodState` and extracts back. Assumes the strategy's internal state (e.g., `self.current_trade` defaultdict in DV2) pickles cleanly. Most things do; defaultdicts with lambdas do not pickle in older Python. Would fail loudly if it fails, which is the right direction.

### 5.5 Clock source is `datetime.now(tz=UTC)` at CLI invocation

[alpha/live/runner.py:45-48](alpha/live/runner.py:45). Assumes the machine clock is accurate. NTP drift on a Windows box can be seconds to minutes. If `pre_close_15m` fires because the local clock is 20 minutes fast, the VPlan uses stale prices. Low probability on a modern Windows 11 box; worth documenting.

### 5.6 VPlan expiry window is long enough for manual re-runs

[alpha/live/runner.py:1131](alpha/live/runner.py:1131): expired if `target_execution_timestamp <= as_of`. If you're manually reviewing a VPlan and the window closes, you cannot submit. The window length is pod-specific; no globally documented minimum.

### 5.7 Exceptions during tick do not leave the lease held

Lease is acquired at [alpha/live/runner.py:1941-2024](alpha/live/runner.py:1941). If an uncaught exception bypasses the release, the next tick is blocked for 5 minutes. Needs a try/finally or context manager — would have to re-read the exact implementation to confirm. Flag for audit.

### 5.8 Floor vs int truncation on negative shares

[alpha/engine/order.py:82,84,91,93](alpha/engine/order.py:82) uses `int()`. For `value=-100, price=10`, `int(-10.5) = -10` (toward zero), `floor(-10.5) = -11`. Live uses `floor` ([execution_engine.py](alpha/live/execution_engine.py)). On short sales with fractional shares, the two disagree by one share in the short direction. None of the current strategies short, so dormant — but will activate the first time you add a short book.

---

## 6. Missing-Features Verdict (a–h)

Per your brief: you know these are missing. You do **not** want them flagged as gaps. You **do** want an architectural-fit verdict on whether the current design accommodates them cleanly when you get to them.

### (a) Parameter robustness (grid sensitivity)

**Fits:** Yes. Strategy classes have parameters as class attributes; a grid runner can instantiate variants and call `run_daily` with shared `pricing_data`. No architectural change needed.

**Watch:** Don't add it inside `compute_signals` — that's a cache boundary. Build a runner that wraps `VanillaBacktester` with variant loops.

### (b) Parameter reduction (fewer knobs)

**Fits:** Architectural no-op. Strategies are free to expose as few or as many parameters as they want. The engine doesn't care.

**Watch:** The more you hardcode, the harder it is to run robustness checks later. Pick 2-3 "anchor" parameters per strategy that stay grid-testable.

### (c) Walk-forward / out-of-sample

**Fits:** Partially. `run_daily(strategy, pricing_data, calendar)` takes a calendar subset, so you can split train/test by slicing. No native OOS harness exists.

**Gap:** No Purged K-Fold or similar for overlapping windows. For your timeframes (DV2 = 1 day hold, TAA = monthly) this doesn't matter much; for anything with long holds, it would.

**Verdict:** Slice-based walk-forward works today. Don't build complex purging machinery you don't need yet.

### (d) Stress testing

**Fits:** Architecturally clean to add. A stress runner generates synthetic price series (bootstrap, block bootstrap, synthetic gap days) and feeds them through `run_daily`.

**Gap:** No stress scenario library. The framework allows it; nothing is written.

**Verdict:** Moderate effort to add a stress module. Do it **after** parity harness, not before.

### (e) Fee / commission robustness

**Fits:** Trivial. Commission is a `Strategy` constructor argument. Run the same calendar with 2×, 5× commission and compare drawdown.

**Verdict:** Half a day to wire a script. Low priority pre-capital; high priority the day before you allocate $1M.

### (f) Execution-time robustness

**Fits:** Partially. Backtest executes at `current_bar.Open`. To sim "execute at +5 min intraday," you need intraday data, which the current framework doesn't support (`pricing_data` is daily OHLC).

**Gap:** Material. The live system executes near close, not at open. Current backtest execution price (open × slippage) is not "the close." Doctrine allows it because the slippage model is conservative, but sensitivity to execution timing is architecturally blocked today.

**Verdict:** Do not solve with intraday data. Solve with a **second execution model in the backtest** that prints at `current_bar.Close × (1 + slippage)` and compare. Cheap, directly informative.

### (g) Intra-strategy diversification

**Fits:** N/A at architecture level. This is a strategy-design concern, not an infra concern.

**Verdict:** Out of scope for this review.

### (h) Portfolio optimization (mean-variance / risk parity / etc.)

**Fits:** The `Portfolio` class currently supports fixed weights with optional periodic rebalance. Adding an optimizer means: compute covariance from pod returns, solve for weights, apply at rebalance.

**Architectural concern:** Live pod capital is per-sub-account NLV × fixed fraction. A portfolio optimizer that wants to dynamically shift weights across pods must either (a) rebalance cash between IBKR sub-accounts (which is a real operational action — wire or journal), or (b) change each pod's fraction dynamically. Neither path has existing plumbing. The backtest `Portfolio` aggregator can simulate either; only the cross-sub-account rebalance maps to live reality.

**Verdict:** Architecturally workable but non-trivial at the live layer. When you add optimization, decide early whether weight changes are executed as IBKR cash journals (slow, manual) or as dynamic fraction changes (fast, automated but only affects deployed notional, not idle cash). Current design assumes static fractions per deployment.

---

## 7. Minimum Viable Path to Live

Fewer than 10 items. Each: what, why, where, effort estimate.

### 7.1 Turn on signal audit and run it end-to-end

- **What:** Set `audit_enabled_bool=True` in `VanillaBacktester` invocations. Run all 5 strategies. Verify no divergence.
- **Why:** Backstop for missing `*** CRITICAL***` markers. DV2 is known to lack them.
- **Where:** [alpha/engine/backtester.py:72-74](alpha/engine/backtester.py:72).
- **Effort:** 1 day (includes fixing anything it finds).

### 7.2 Build a parity test harness

- **What:** Given (strategy, date, prices, broker_snapshot), assert `backtest_shares ≈ live_shares` within declared tolerance. Fail the CI build if it regresses.
- **Why:** Every other parity concern becomes measurable. Changes to either side are regress-tested.
- **Where:** New file: `tests/parity/test_sizing_parity.py`. Exercises `alpha.engine.strategy._get_order_sizing_price_float` vs `alpha.live.execution_engine` share computation.
- **Effort:** 2-3 days.

### 7.3 Add absolute per-pod and per-order dollar caps

- **What:** `max_dollars_per_pod_float` and `max_dollars_per_order_float` in pod manifest. Enforce in `execution_engine` before VPlan emission.
- **Why:** Absolute backstop against bad net_liq snapshot or strategy bug. §2.5.
- **Where:** [alpha/live/models.py:42](alpha/live/models.py:42) (manifest), [alpha/live/execution_engine.py](alpha/live/execution_engine.py) (check).
- **Effort:** 1 day.

### 7.4 Declare pod_budget_fraction semantics and live value per pod

- **What:** Add `pod_budget_fraction_live_float` (separate from the current `pod_budget_fraction_float`, which is treated as the paper value) to each manifest. Deploy script selects based on `deployment.mode`. Add `sub_account_funding_target_float` to document the intended funding of each sub-account so it can be cross-checked against IBKR at startup.
- **Why:** §2.5. The scaling throttle is currently conflated with the production knob. Without an explicit live value, the paper safety value (0.03) will silently persist into personal capital.
- **Where:** [alpha/live/models.py](alpha/live/models.py) (`LiveRelease` dataclass), manifests in [alpha/live/releases/](alpha/live/releases/), deploy script logic.
- **Effort:** 1 day.

### 7.5 Add `*** CRITICAL***` markers to strategy time-series ops

- **What:** Audit all 5 strategies for `shift`, `rolling`, `resample`, `pct_change` usage. Add inline marker per doctrine.
- **Why:** DV2 confirmed missing; others unverified.
- **Where:** [strategies/dv2/strategy_mr_dv2.py:36-39](strategies/dv2/strategy_mr_dv2.py:36) and the four other strategy files.
- **Effort:** 0.5 day.

### 7.6 Add second backtest execution model: close-based

- **What:** Flag `execution_price_source: 'open' | 'close'`. Close mode sizes at prev_close, fills at current_bar.Close × slippage.
- **Why:** Live executes near close, not open. Without a close-mode backtest, you cannot measure sensitivity to execution timing. Cheapest way to close §3 row "Execution price."
- **Where:** [alpha/engine/strategy.py:295-438](alpha/engine/strategy.py:295) (`process_orders`).
- **Effort:** 1-2 days.

### 7.7 Define and test a lease-failure recovery runbook

- **What:** If tick crashes mid-lease, what happens at minute +5? Test by killing tick during a VPlan submit and observing recovery.
- **Why:** §5.7. A stuck lease blocks all trading for its duration.
- **Where:** [alpha/live/runner.py:1941-2024](alpha/live/runner.py:1941) — verify try/finally release; [LIVE_RUNBOOK.md](LIVE_RUNBOOK.md) — add recovery steps.
- **Effort:** 1 day.

### 7.8 IBKR reconnect policy

- **What:** When `ibkr_socket_client` disconnects mid-tick, policy is undefined. Define: retry N times with backoff, then fail the tick and log critical.
- **Why:** IBKR TWS drops connections daily. Without explicit reconnect, trades are missed silently.
- **Where:** [alpha/live/ibkr_socket_client.py](alpha/live/ibkr_socket_client.py).
- **Effort:** 1-2 days.

**Total:** ~8-10 engineering days. This is the bar to cross before personal capital.

---

## 8. What Can Wait

Things that look important but don't pay at your fund size.

### 8.1 Real-time P&L dashboard

The state store has everything you need; a query is enough for daily review. A live dashboard is pure cost until you're managing someone else's money.

### 8.2 Multiple broker support

IBKR is fine for both personal and early external capital. Broker abstraction before you need it is ceremony.

### 8.3 Intraday data infrastructure

Daily OHLC is sufficient for all five current strategies. Intraday feeds, storage, alignment — multi-week effort for zero current benefit. The one place it would help (§6f, execution-time robustness) has a cheaper solution (§7.6).

### 8.4 Kubernetes / distributed scheduler

A Windows box running scheduler_service is enough. Moving to containers before you've hit scheduler_service's real limits is infra theater.

### 8.5 Automated strategy code review / CI gates

Your CI should run `pytest` and the parity harness (once built). Anything more — lint rules for `*** CRITICAL***`, auto-PR on doctrine violation, etc. — is worth doing at 5-developer scale, not 1.

### 8.6 Walk-forward cross-validation framework

Slice-based walk-forward works. Purging, combinatorial CV, fractional differentiation — academic elegance for a system with 5 strategies you already understand.

### 8.7 Options / futures / FX extensions

You don't trade them. Architecting for them now freezes the wrong abstractions.

### 8.8 Cloud backups beyond the state store file

`live_state.sqlite3` is small. A nightly Windows scheduled task copying to OneDrive is sufficient. S3 lifecycle rules and point-in-time recovery are for a system running money you can't afford to lose, which isn't this one yet.

---

## 9. Suggested Durable Artifacts

Markdown files the platform is missing. One line each on purpose. These are cheap to write and pay forever.

- **`LIVE_BACKTEST_PARITY.md`** — canonical declaration of what is identical, what is "irreducible frictions," and what divergences are measured by the parity harness.
- **`EXECUTION_SEMANTICS.md`** — precise contract: when does a VPlan build, when does it submit, what clock, what broker snapshot, what failure modes block vs warn.
- **`POD_CAPITAL_MODEL.md`** — declares the one-pod-one-sub-account contract, the intended paper vs live `pod_budget_fraction_float` values, the sub-account funding targets, and the cutover checklist that enforces them.
- **`STRATEGY_CONTRACT.md`** — documents the `entry_value` and `exit_to_zero` contract enforced in `strategy_host.py`, with examples and the error you get when you violate it.
- **`INCIDENT_RUNBOOK.md`** — step-by-step: IBKR disconnect, stale lease, bad net_liq snapshot, stuck exit residual, Norgate down at overnight build time.
- **`CAPITAL_GATE_CHECKLIST.md`** — the pre-deploy checklist: audit passes, parity harness passes, dollar caps set, reconnect tested, runbook rehearsed. One file, one list, one sign-off per live deploy.
- **`DATA_DEPENDENCIES.md`** — what depends on Norgate, what would survive its outage, what is the fallback (currently: nothing — and that's worth writing down).
- **`SCALING_TRIGGERS.md`** — the thresholds at which §8's "can wait" items become "must do." E.g., "add virtual sub-account NLV tracking when total AUM > $X."

---

## Closing Note

This is a rare pre-capital state: the skeleton is right, most of the safety rails exist, and the doctrine is externalized. The work to cross the "personal capital" gate is focused and small (§7, ~8-10 days). The work to cross the "external capital" gate is a superset: parity harness as CI, pod capital model resolved, incident runbook rehearsed, absolute caps in place.

Do not let the scope of the external-capital gate delay the personal-capital gate. Get §7 done, then run paper → personal for three months, then re-review this document with live data.

---

## Revision Note

Initial draft (2026-04-17) claimed pod budget shared a single account, which would have been a HIGH-severity semantic divergence. That was wrong. Each pod is mapped 1:1 to its own IBKR sub-account (DUK322077 / DU5566778 / DU2233445 / DU3344556), so compounding is independent and matches the backtest `Portfolio` pod model. The remaining concern in that area is operational, not semantic: `pod_budget_fraction_float` is currently a paper-safety throttle (0.03) with no declared live value. Sections 2, 3 (parity table), 4.3, 6(h), and 7.4 updated accordingly.

— End of review.
