# Assumptions And Gaps

This file is the truth source for the current realism limits, modeling assumptions, and known gaps between this backtest environment and the intended live trading stack.

The goal is not to pretend the backtest is identical to live trading. The goal is to state clearly where it differs and what still needs to improve.

This file is part of the live-first control system. Recording a gap does not make it acceptable. It prevents silent self-deception and makes the remaining distance to the intended live workflow explicit.

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
| G-012 | Capacity and tradability | Research can still look acceptable without an explicit check that order size, turnover, and liquidity are practical at the intended capital scale. | A strategy may look tradable in the simulator but be operationally difficult or impossible to run live. | High. | Favor liquid universes, conservative sizing, and explicit liquidity filters. | Strategy-level capacity and tradability checks tied to intended capital scale and workflow. | Ongoing control |
| G-013 | Close-exit execution | Some research-only monthly strategies model entry at the open and exit at the same month's close via a custom backtest path outside `run_daily()`. | The main engine and future live stack do not yet support explicit market-on-close or close-auction order intent with broker-aware reconciliation. | Medium. | Mark these paths as research-only and state the exact execution difference in the strategy file. | First-class close/auction order support in the engine and live broker layer. | Known gap |
| G-014 | Symbol continuity | When a held equity loses current-bar prices because of a delisting, merger, or ticker retirement, the engine now force-liquidates the stale symbol at the last available prior close. | Live corporate actions can convert the position into cash, successor shares, or a more complex event path. A simple prior-close liquidation can misstate PnL versus the exact corporate-action outcome. | Medium, and potentially High around large merger gaps or ratio changes. | Fail loud in the engine, close the stale symbol deterministically, and treat the result as conservative bookkeeping rather than exact corporate-action replay. | Corporate-action-aware successor mapping and broker/live reconciliation for symbol changes, cash mergers, and share conversions. | Known gap |
| G-015 | FRED DTB3 publication timing | TAA backtests now auto-refresh `DTB3` from `FRED` and then use the revised daily history that is currently available, not a strict replay of what `FRED` had published at each historical timestamp. | Live TAA decisions may use a one-business-day-old `DTB3` observation when that is the latest published `FRED` value available at decision time. If `DTB3` is older than one business day in live, the pod now fails loud and skips decision-plan creation for that cycle. | Low to Medium for the current monthly TAA sleeve because `DTB3` moves slowly, but non-zero near absolute-momentum thresholds. | Auto-refresh from `FRED`, local CSV cache fallback, live freshness gate `freshness_business_days_int <= 1`, and explicit decision metadata recording of the `DTB3` observation date and download status. | Point-in-time macro publication replay only if a future strategy requires archival publication semantics rather than the current `FRED`-only policy. | Intentional approximation |
| G-016 | MOM proxy continuity | `strategy_mo_mtum_timed_by_mom` uses the delisted Norgate ETF symbol `MOM-202103` as the article-faithful momentum proxy, so the shared `MTUM`/`MOM` sample ends on `2021-03-12`. | Any extension past `2021-03-12` requires changing the signal source. A future switch to Kenneth French preserves the strategy idea but changes the data definition and therefore the exact backtest lineage. | Medium. | The strategy loader fails loud whenever the requested date range lies outside the supported shared overlap. | Replace the ETF proxy with the intended Kenneth French momentum series under an explicit v2 data contract and separately documented sample history. | Known gap |
| G-017 | Kenneth French publication timing | `strategy_mo_pdp_timed_by_kf_mom` uses the current Kenneth French daily momentum factor history as the signal source for historical daily decisions. The Data Library explicitly reconstructs the full return history when it updates the portfolios, so this is not a point-in-time archival feed of what a live trader could have observed each historical day. | A backtest that treats the current Kenneth French daily factor values as if they were available at each past close can materially overstate live replicability of the daily timing rule. | High for live claims; Medium for research idea validation. | Mark the strategy as research-only, keep the caveat explicit in code, and separate it from the live-first Norgate proxy implementation. | Use an archival point-in-time factor publication feed or a tradeable live proxy that preserves the same signal semantics without publication-lag leakage. | Known gap |
| G-018 | MOM to PDP bridge continuity | `strategy_mo_mtum_timed_by_mom_pdp_bridge` extends the `MOM` proxy beyond `2021-03-12` by compounding `MOM-202103` daily returns through the delisting date and `PDP` daily returns thereafter into one synthetic bridge index. This preserves causal daily timing semantics better than Kenneth French, but the post-switch signal is no longer the same instrument as the pre-switch signal. | Backtest behavior after the switch depends on how well `PDP` continues the economic behavior of the delisted `MOM` ETF. If that continuity is weak, post-2021 results can drift materially from the original proxy's intent. | Medium. | Keep the bridge variant separate from the article-faithful `MOM`-only strategy and expose the switch explicitly in code through `signal_switch_date_str` and `use_fallback_bool_ser`. | Replace the bridge only if a better point-in-time, tradeable continuation proxy is found and separately validated against the `MOM` overlap sample. | Known gap |

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

## Detailed Note: Future Work For Symbol Continuity

`G-014` is intentionally not considered a finished solution. The current
engine behavior is a conservative bookkeeping fallback, not the desired final
economic model.

Today the backtest fallback is:

\[
\Delta q^{liq}_{i,t} = -q^{open}_{i,t}
\]

\[
P^{exit}_{i,t} = P^{last\_avail}_{i,\le t-1}
\]

This prevents zombie positions, but it does not represent the true economics
of a merger, rename, cash acquisition, spin-off, or share-ratio conversion.

The desired future state is an explicit symbol-continuity policy with separate
semantics for research and live trading:

\[
\text{policy}^{backtest} \in \{\text{raise}, \text{liquidate\_last\_close}, \text{corporate\_action\_map}\}
\]

\[
\text{policy}^{live} = \text{block\_and\_reconcile}
\]

Under a fuller corporate-action-aware model, the preferred state transition is
not synthetic liquidation but explicit successor mapping:

\[
q^{succ}_{t} = q^{old}_{t-1} \cdot r
\]

\[
cash_t = cash_{t-1} + cash^{deal}_t + cash^{fractional}_t
\]

where \(r\) is the deal share-conversion ratio and the cash terms capture cash
consideration and fractional-share settlement.

This work should be implemented in a later phase, not folded silently into the
current engine behavior. Until then, any result materially affected by symbol
discontinuity should be interpreted as using a conservative approximation
rather than exact corporate-action replay.

## Interpretation Rule

If a result looks attractive only because one of the known gaps is favorable to the backtest, or because the strategy would be materially harder to trade live than the simulation implies, the result should be treated as overstated until the gap is closed or explicitly bounded.

