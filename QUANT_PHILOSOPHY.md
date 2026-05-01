# Quant Philosophy

This document is the house doctrine for this repository. If you are an AI agent or an engineer, read this before changing the engine, adding strategies, or modifying research logic.

## Operating Context

This repository is not a toy backtester and not an academic notebook collection. It represents a hedge fund quant research and engineering stack for a multi-strategy, pod-based platform.

Today:

- We research and backtest systematic strategies.
- We combine them as pods inside a multi-strategy portfolio.
- We care about institutional quantitative correctness, not just good-looking charts.

Target state:

- A live multi-strategy stack with pod-level capital allocation.
- A future execution layer connected to IBKR.
- A research environment that makes bad quant work hard to express.

## Governing Equations

Backtest performance is not proof of edge:

\[
\text{Backtest} \neq \text{proof of edge}
\]

Observed performance is a mixture of truth and distortion:

\[
\text{Observed performance} = \text{edge} + \text{luck} + \text{bias} + \text{implementation error}
\]

The first duty of the engine is to reduce the last three terms as aggressively as possible.

Backtest quality is judged by credibility and robustness, not by cosmetic performance:

\[
\text{Backtest credibility}
=
\text{causality}
+
\text{execution realism}
+
\text{robustness}
+
\text{auditability}
-
\text{hidden assumptions}
\]

Signals must be causal:

\[
\text{signal}_t = f(\mathcal{I}_{t-1})
\]

not

\[
\text{signal}_t = f(\mathcal{I}_{t}, \mathcal{I}_{t+1}, \dots)
\]

Live trading should preserve backtest semantics up to unavoidable market frictions:

\[
\text{live behavior}_t \approx \text{backtest behavior}_t + \epsilon^{friction}_t
\]

where \(\epsilon^{friction}_t\) is limited to irreducible effects such as slippage, partial fills, latency, broker constraints, and rounding.

Live order intent must be deterministic:

\[
\text{order intent}_t = g(\mathcal{I}_{t-1}, \text{state}_{t-1}, \text{config})
\]

The same information, state, and configuration should produce the same intended orders.

Portfolio equity in the pod model is the sum of independently compounded pod equities:

\[
E^{portfolio}_t = \sum_{i=1}^{N} E^{pod_i}_t
\]

not a hidden daily-rebalanced weighted return shortcut unless daily rebalancing is explicitly the model.

## Live-First Simulation

The simulator exists to approximate the live trading problem as closely as the current data and engine allow. It is not here to manufacture optimistic research results under assumptions we would reject in production.

\[
\text{usable edge} = \text{raw edge} - \text{costs} - \text{slippage} - \text{liquidity friction} - \text{operational constraints}
\]

A backtest is interesting only if the edge survives that conversion under assumptions that are plausible for the intended live deployment. The backtest should be as credible, conservative, and robust as the current data and engine allow.

- Treat tradability as part of the strategy, not as a post-hoc implementation detail.
- Judge tradability at the intended capital scale and operational workflow, not with vague claims that something is "easy to trade."
- Prefer assumptions that understate readiness rather than overstate it when uncertainty cannot yet be resolved.
- If an assumption is unrealistic but temporarily unavoidable, record it explicitly in `ASSUMPTIONS_AND_GAPS.md` with its likely consequence.
- If a result depends mainly on unrealistic execution, impossible liquidity, or undocumented simplifications, treat it as non-actionable until that gap is closed or explicitly bounded.

## Live Must Preserve Backtest Semantics

Live trading is not a place to "improve" a strategy with discretionary overrides, hidden execution intelligence, or ad hoc rule changes after seeing the backtest.

- The live system should trade the same strategy semantics that were tested.
- Differences between backtest and live should come from irreducible real-world frictions, not from silent logic drift.
- If live execution requires a semantic change, the change must be treated as a strategy change and documented as such.
- If the live implementation cannot preserve the tested semantics, then the live strategy and the backtest are not the same strategy.

## Deterministic Order-Clerk Execution

For live deployment, prefer a deterministic order-clerk model: simple, explicit, and auditable.

- The strategy generates clear order intent from prior-available information.
- The execution layer transmits, tracks, reconciles, and reports that intent.
- The execution layer should not add opaque optimization logic that changes strategy meaning.
- When the same inputs arrive, the intended order set should be the same unless an explicitly modeled external constraint prevents it.

## What Good Quant Work Looks Like

### Simplicity and readability are quantitative controls

Simple code is not just style. It is a defense against hidden assumptions and hidden bugs.

- Prefer direct implementations over clever abstractions.
- Prefer plain explanations for orchestration, control flow, and glue code.
- Prefer explicit formulas when they materially improve auditability of a quantitative calculation.
- Prefer readable intermediate variables over compressed one-liners.
- Prefer a strategy with a small number of justified rules over a parameter-heavy ruleset.

If a reader cannot explain the logic quickly, the implementation is too complicated.

Code is accepted only if a human owner can read it later without needing the
agent or engineer who wrote it. Prefer code that looks like the obvious next
step a careful human would have written.

### Mathematical clarity where it matters, human clarity by default

Keep the logic rigorous, but keep the explanation simple, precise, and easy for a human to follow quickly.

Default explanation order:

- plain-language intuition
- exact rule
- formula only if it materially improves correctness, auditability, or semantic precision

Write the formula when the code implements a non-trivial quantitative transformation that a reviewer must audit.

That includes:

- indicators
- returns, compounding, and drawdown math
- portfolio aggregation math
- sizing logic
- sensitive time-series transforms

Do not add formula ceremony to trivial plumbing, orchestration, or obvious control flow. In those areas, a plain explanation and direct code are the clearer choice.

Do not confuse explanation complexity with quantitative rigor. The goal is to preserve correctness while making decisions, assumptions, and tradeoffs easy for a human operator to understand.

Examples:

\[
r_t = \frac{P_t}{P_{t-1}} - 1
\]

\[
\text{annualized return}^{(L)}_t = \left(\frac{P_t}{P_{t-L}}\right)^{252 / L} - 1
\]

\[
\text{drawdown}_t = \frac{E_t}{\max(E_1, \dots, E_t)} - 1
\]

The code should make the math easier to audit, not harder.

### Naming must expose domain intent

Use strict `Domain_Type` naming whenever you add or refactor quantitative logic.

Examples:

- `price_vec`
- `return_ser`
- `signal_df`
- `target_weight_ser`
- `portfolio_value_float`

Do not hide domain meaning behind vague names like `x`, `tmp`, `data2`, or `thing`.

### Sensitive time-series operations must be flagged

Any operation that can create look-ahead bias or other quant leakage must be visibly marked with a `*** CRITICAL***` comment.

Examples:

- `shift()`
- rolling windows
- forward returns
- rebalance-date mapping
- feature sampling at month-end or quarter-end

The purpose of the comment is to force a second audit pass by future readers and AI agents.

### Dangerous assumptions must fail loud

Any dangerous action, realism gap, hidden assumption, operational ambiguity, or result-distorting simplification that could mislead us must be surfaced explicitly and discussed.

Do not leave potentially misleading behavior implicit.

When a risk is identified, document at minimum:

- `issue_description_str`
- `expected_bias_direction_str`
- `impact_level_str`
- `mitigation_str`

If the impact cannot yet be quantified exactly, bound it qualitatively and say so plainly. When uncertain, do not silently choose the prettier implementation. Choose the more explicit implementation and state the uncertainty.

## What Backtests Are For

Backtests are for falsification, sanity checks, and implementation validation. They are not a machine for proving alpha.

Useful backtest questions:

- Does the implementation obey causality?
- Does the strategy collapse under realistic costs?
- Could this actually be traded at the intended capital scale and operational workflow?
- Is turnover operationally acceptable?
- Does the idea survive simple perturbations?
- Does the pod behave coherently inside a portfolio?

Dangerous backtest misuse:

- repeated tweaking after reading the results
- selecting only the winning variant
- building an ex post narrative around random historical luck
- reporting the final winner without reporting the search

The correct mental model is:

\[
P(\text{false discovery}) \uparrow \text{ as } N_{\text{trials}} \uparrow
\]

If you run enough variants, some of them will look brilliant by chance.

## Non-Negotiable Quant Pitfalls

### 1. Look-ahead bias

This is the most important failure mode.

- Features must use only past information.
- Decisions taken for bar \(t\) must not use close, high, low, or other information from after the decision point.
- Same-bar fill fantasy is forbidden unless the model explicitly supports it and the data are aligned correctly.

### 2. Survivorship bias

Universe membership must be point-in-time correct.

\[
\mathcal{U}_t \neq \mathcal{U}_{today}
\]

Never backtest on a static list of current survivors when the live strategy would have traded delisted or removed names historically.

### 3. Storytelling bias

Do not create an economic story after the result is already known and then confuse that story with evidence.

### 4. Overfitting and data snooping

Do not keep adding filters and parameters until the historical Sharpe looks attractive.

\[
\hat{\theta} = \arg\max_{\theta \in \Theta} \text{BacktestMetric}(\theta)
\]

becomes statistically meaningless when \(\Theta\) has been searched aggressively on the same data.

### 5. Turnover and transaction costs

Gross returns are not enough.

\[
r^{net}_t = r^{gross}_t - c^{trade}_t - c^{slippage}_t - c^{borrow}_t
\]

If turnover is high, costs are part of the strategy, not a footnote.

### 6. Outliers

A small number of extreme observations can dominate averages, rankings, regressions, and conclusions.

### 7. Asymmetric shorting reality

Shorting is not simply the long trade with a minus sign.

- borrow fees matter
- locate availability matters
- recalls matter
- squeezes and gap risk are asymmetric

### 8. Live-trading fantasy

Do not assume frictionless liquidity, effortless execution, or operational simplicity that the intended deployment does not have.

- Capacity is strategy logic, not an afterthought.
- Execution difficulty is part of the expected return distribution.
- If the live workflow would be materially harder than the simulation implies, the result is overstated.

## Engine Order Is Part Of The Model

The engine lifecycle is a quantitative contract:

1. `compute_signals(pricing_data)`
2. restrict data to information available up to `previous_bar`
3. `iterate(data, close, open_prices)`
4. `process_orders(pricing_data)`
5. `update_metrics(pricing_data, start)`

This order is not an implementation detail. It is part of the model.

If you change this order, you have changed the meaning of the backtest.

In particular:

- `compute_signals()` may run on the full dataset, so every derived feature inside it must still be causal.
- `iterate()` is the decision step.
- orders placed in `iterate()` are executed at the next bar open under the current engine contract.
- metrics are updated after execution and end-of-day valuation.

Any change to this sequence must be treated as a model change, not a refactor.

## Pod Model And Multi-Strategy Reality

Each strategy is a pod.

A pod is:

- an independent capital sleeve
- its own state machine
- its own positions, logic, and results
- a building block for the firm-level portfolio

For live trading, keep the pod idea concrete and human:

```text
one pod = one independent strategy sleeve
```

The intended production pattern is:

```text
one live pod = one strategy = one linked IBKR account/subaccount route = one ledger
```

That broker account must have its own cash, positions, and account value that
can be read and reconciled independently.

The portfolio math stays simple:

\[
\text{client equity}_t = \sum_i \text{pod equity}_{i,t}
\]

A pod must not be just a label inside one shared raw broker account. That only
works if the system later has a real pod ledger that assigns orders, fills,
positions, cash, and corporate actions to each pod.

In an IBKR Family Account, Friends and Family Advisor, Family Office, or Advisor
setup, this means one strategy sleeve should map to one real subaccount/account
route by default.

The portfolio is not a bag of signals. It is a combination of independently compounded pods.

That means:

- pod equity curves compound independently
- portfolio equity is the sum of pod equities
- weights drift unless an explicit rebalance policy is applied
- cross-pod diversification is part of the edge of the overall system

When designing code, think in terms of pod isolation first and aggregation second.

## AI And Engineer Change Rules

If you change any of the following, you must explicitly state the old behavior, the new behavior, and the quantitative consequence:

- signal timing
- order timing
- execution price assumptions
- price adjustment settings
- universe construction
- rebalance mapping
- portfolio aggregation math
- cost modeling

Do not silently change semantics by renaming variables, moving code blocks, or cleaning up "duplicate" logic that is actually enforcing timing correctness.

If you encounter a dangerous assumption, ambiguous semantic choice, or realism gap that could mislead future research or live deployment, raise the flag explicitly instead of smoothing it over.

If a change touches any sensitive time-series or execution path, add or preserve:

- formulas for non-trivial quantitative logic when they materially improve auditability
- tests
- `*** CRITICAL***` comments

## Final Rule

This repository should help us reject bad research, preserve good causal logic, and move toward a live institutional stack with as little self-deception as possible.

If a proposed change improves aesthetics but weakens causality, realism, live tradability, auditability, or simplicity, reject the change.
