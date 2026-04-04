# Assumptions And Gaps

This file is the truth source for the current realism limits, modeling assumptions, and known gaps between this backtest environment and the intended live trading stack.

The goal is not to pretend the backtest is identical to live trading. The goal is to state clearly where it differs and what still needs to improve.

## Register

| ID | Area | Backtest Assumption / Current Behavior | Live Gap / Risk | Impact | Current Mitigation | Desired End State | Status |
|---|---|---|---|---|---|---|---|
| G-001 | TAA target sizing | `strategy_taa_df` sizes target shares from stored prior-bar portfolio value and prior closes, then fills at the next open. | Realized rebalance weights can drift from intended weights after overnight gaps, slippage, and rounding. | Medium for the current TAA sleeve; High if a future sleeve requires exact open weights. | Treat TAA weights as intended overnight allocations, not exact open marks. | Configurable rebalance sizing modes plus broker-aware reconciliation and optional reserve cash. | Intentional approximation |
| G-002 | Slippage model | Slippage is a simple signed price penalty. | Real slippage depends on liquidity, order size, volatility, queue position, and market regime. | Medium to High. | Use conservative slippage settings and treat results skeptically. | Regime-aware or liquidity-aware execution-cost model. | Known gap |
| G-003 | Commission model | Commission is simplified to an IBKR-like per-share rule with a minimum. | Real commissions, fees, rebates, and routing effects can differ by instrument and venue. | Medium. | Keep the model explicit and conservative. | Instrument-aware live fee model. | Known gap |
| G-004 | Fill mechanics | Orders either fill under the model rules or do not fill; no partial fills are modeled. | Live execution can split, miss, queue, or partially fill orders. | High for larger or less liquid trades. | Keep position sizing realistic and avoid pretending large orders are frictionless. | Order-book-aware or broker-fill-aware live execution layer. | Known gap |
| G-005 | Market microstructure | No latency, queue position, venue selection, or routing microstructure is modeled. | Live trading can experience delays, price drift, missed fills, and routing variance. | Medium to High. | Keep the engine honest about next-open execution and avoid intraday precision claims it cannot support. | Execution simulator or live broker integration with timestamps and reconciliation. | Known gap |
| G-006 | Broker state | No live broker state, cash reconciliation, trade reconciliation, or portfolio sync exists yet. | Live accounts can drift from model state because of rejected orders, partial fills, corporate actions, and broker-side events. | High. | None beyond explicit acknowledgment. | A real broker integration and reconciliation layer, likely through IBKR. | Planned gap |
| G-007 | Shorting realism | Shorting constraints are not modeled as a first-class system feature. | Live shorting depends on borrow availability, borrow cost, recalls, and operational constraints. | High for any future short strategy. | Treat short-side results as optimistic unless explicitly modeled. | Borrow-aware and fee-aware short simulation. | Known gap |
| G-008 | Calendar cleanliness | Backtest calendars and market data are cleaner than live operations. | Live systems face holidays, half-days, symbol events, stale prices, and broker-specific calendar quirks. | Medium. | Use real trading calendars where possible and avoid overclaiming precision. | Production calendar and session model aligned with the live broker. | Known gap |
| G-009 | Point-in-time safety | Current engine structure helps prevent leakage, but future strategies can still be written incorrectly. | A new strategy can misuse `compute_signals()`, `shift()`, rolling windows, or data joins and still create leakage if written carelessly. | High. | Signal audit exists for derived columns and the repo enforces explicit reviews of time-series logic. | Broader invariant testing and continued doctrine enforcement. | Ongoing control |
| G-010 | Data dependence | Point-in-time correctness depends on the data source and how a strategy uses it. | A strategy may still be wrong if the underlying data or joins are not point-in-time correct. | High. | Prefer Norgate point-in-time datasets and explicit feature construction. | Fully documented point-in-time data contracts per strategy family. | Known gap |
| G-011 | Stress realism | Historical backtests mainly replay the realized path. | Live trading can fail under regimes or shocks not represented well in the historical sample. | Medium to High. | Use backtests as filters, not proof. | Scenario testing, stress testing, and walk-forward style evaluation. | Planned gap |

## Detailed Note: Phase-1 TAA Target Sizing

The current house choice for `strategy_taa_df` is an overnight OPG-style approximation:

\[
\text{target\_shares}^{taa}_t
\ =
\left\lfloor \frac{V^{close}_{t-1} \cdot w_t}{P^{size}_{t-1}} \right\rfloor
\]

with:

\[
P^{size}_{t-1} = close_{t-1}
\]

for target-share sizing, while execution still occurs at the next open. This is
an intentional phase-1 simplification. It treats target weights as intended
overnight allocations and accepts small drift in realized weights.

A more exact open-aware rebalance would be:

\[
\text{target\_shares}^{open}_t
=
\left\lfloor \frac{V^{open}_{t} \cdot w_t}{open_t} \right\rfloor
\]

where

\[
V^{open}_{t}
=
cash_{t-1} + \sum_i q_{i,t-1} \cdot open_{i,t}
\]

That more exact formulation is not the chosen default for the current TAA sleeve, but it remains a useful reference if a future strategy requires exact open-weight rebalancing.

## Interpretation Rule

If a result looks attractive only because one of the known gaps is favorable to the backtest, the result should be treated as overstated until the gap is closed or explicitly bounded.

