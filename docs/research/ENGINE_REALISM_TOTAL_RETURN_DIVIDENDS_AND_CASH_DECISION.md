# Engine Realism Decision: Execution Prices, Signal Returns, Dividends, and Cash

**Decision status:** OWNER APPROVED - Codex and Claude aligned; the owner authorized the engine-only net-dividend ledger on a dedicated branch

**Implementation status:** Phase 1 provenance/data-contract work, gross/net shadow studies, the engine-only `net_dividend_cash_ledger_v2` implementation, and the seven WIRED Vanilla artifact regenerations are complete on the feature branch; releases, incubation, and LIVE remain unchanged

**Repository snapshot reviewed:** 2026-07-25

**Primary question:** How should `alpha_super` separate executable prices, signal returns, and broker-account economics so that BENCH, incubation, and live trading describe the same strategy?

## TL;DR

The current engine mixes three different concepts:

1. the price at which a security can be traded;
2. the return representation used to form a signal;
3. the economic value of the brokerage account.

They must be separated.

The preliminary Codex position is:

- Never use `TOTALRETURN` prices as order-fill prices or position marks.
- Keep a non-total-return execution basis. The current engine uses Norgate `CAPITALSPECIAL`, but this is an adjusted price proxy, not literally the historical exchange quote.
- Add explicit dividend events. Keep positive cash at the owner-approved `0%`, and report negative cash as a disclosed diagnostic under the current owner-approved research policy.
- Do not globally change every signal to `TOTALRETURN`. Choose the signal basis feature by feature and strategy by strategy.
- Treat Norgate `TOTALRETURN` as a signal or benchmark representation and as a parity check, not as the account ledger.
- Correct the NDX/SPY benchmark provenance mismatch before relying on benchmark-relative analyzer output.

The highest-priority gap is the economic ledger. The signal question is important but separate: changing a signal from price return to total return changes the strategy, while crediting a dividend that the account actually earned corrects missing accounting.

## 1. Purpose and review protocol

This file is a decision document for Codex, Claude, and the human owner. It is not an implementation specification yet.

Each reviewer should:

1. distinguish verified current behavior from recommendations;
2. cite the exact code or external documentation behind any factual disagreement;
3. state whether a proposal changes only accounting or also changes strategy semantics;
4. state the expected bias direction and affected WIRED strategies;
5. finish with a direct verdict in the review table near the end of this document.

Do not silently rewrite the verified-facts sections. If a fact is disputed, add the conflicting evidence to the review log.

## 2. The central problem

A realistic daily trading engine needs three separate data and accounting channels:

```text
Market observations
    |
    +--> Execution and valuation prices
    |       Actual/non-total-return price basis
    |       Used for fills, marks, gaps, stops, and order sizing
    |
    +--> Signal representations
    |       Price return, economic total return, volatility, index level, etc.
    |       Selected explicitly for each feature
    |
    +--> Economic events and declared policies
            Dividends, the explicit 0% positive-cash policy, borrow, taxes,
            splits, mergers, and settlements
            Posted to or enforced by the account ledger
```

The present engine has the first channel in an approximate form and has several signal representations, but it does not have a complete shared economic-event ledger.

That omission affects more than the final performance report. Account equity is reused for later position sizing, so a missing cash flow changes:

- subsequent share quantities;
- portfolio exposure;
- compounding;
- drawdowns;
- capacity analysis;
- risk analysis;
- portfolio aggregation;
- reference-to-live tracking error.

## 3. Precise definitions

### 3.1 True historical as-traded prices

A true as-traded price is the quote that existed on that historical date. It includes visible price discontinuities from splits, special distributions, and other corporate actions. A broker account then needs corresponding share and cash events.

This is the cleanest economic model, but it requires a corporate-action-aware position ledger.

### 3.2 Norgate `CAPITALSPECIAL`

The current repository normally uses Norgate `CAPITALSPECIAL` for tradeable securities. It adjusts for capital reconstructions and special distributions while leaving ordinary cash dividends outside the price return.

This is not literally an as-traded price series. For example, a direct local Norgate check of AAPL on 2020-08-28 returned an approximately `126.01` `CAPITALSPECIAL` open versus an approximately `504.05` unadjusted historical open before the four-for-one split.

Therefore the accurate description of current behavior is:

> `CAPITALSPECIAL` is the repository's corporate-action-normalized, non-total-return execution and valuation proxy.

It preserves return continuity without requiring the common engine to change historical share quantities for splits. That is useful, but it must not be confused with a literal broker fill record.

### 3.3 Norgate `TOTALRETURN`

`TOTALRETURN` incorporates distributions into the price history as split-like adjustments. It represents the economic growth of a reinvested position.

It is useful for:

- economic momentum;
- relative-return ranking;
- benchmark wealth;
- parity checks against an explicit dividend ledger.

It is not suitable for:

- historical order fills;
- stop or limit prices;
- broker cash;
- literal share quantities;
- exact dividend payment timing.

Using `TOTALRETURN` for position marks and also crediting explicit dividends would double-count the same economic benefit.

### 3.4 Explicit economic ledger

The target account equation is:

```text
account equity
= settled cash
+ marked position value
+ dividend receivables
+ interest receivables
- short dividend liabilities
- financing and borrow liabilities
- other accrued fees and taxes
```

The ledger, not a synthetic price, should determine what cash is available to size future orders. Under the owner-approved current policy, the positive-cash interest term is explicitly zero.

## 4. Simple examples

### 4.1 Long dividend

Assume an account holds 100 shares at a close of `$100`. The security has a `$1` dividend and opens ex-dividend at `$99`, with no other market movement.

Current price-only ledger:

```text
position value = 100 x $99 = $9,900
dividend value = $0
economic change = -$100
```

Correct gross economic ledger:

```text
position value      = 100 x $99 = $9,900
dividend receivable = 100 x $1  =   $100
economic equity                    $10,000
```

The dividend should not automatically buy more shares. It becomes a receivable and later cash, unless the strategy explicitly reinvests it.

### 4.2 Short dividend

If the account is short 100 shares through the same entitlement boundary, the account owes the equivalent dividend:

```text
short dividend liability = 100 x $1 = $100
```

Ignoring this liability makes a short backtest optimistic.

### 4.3 Signal ranking

Suppose:

- Asset A gains 5% in price and distributes approximately 4%.
- Asset B gains 8% in price and distributes nothing.

A price-return momentum signal ranks B above A. An economic-return momentum signal may rank A above B. Neither representation is universally correct; the strategy hypothesis must decide what is being measured.

### 4.4 Ex-dividend oversold signal

A stock falling from `$100` to `$99` only because of a `$1` distribution has approximately:

```text
price return    = -1%
economic return =  0%
```

A short-horizon mean-reversion signal can incorrectly interpret the mechanical ex-dividend move as selling pressure. This is especially relevant to QPI and, to a lesser degree, DV2.

### 4.5 VXN cash sleeve

The WIRED VXN scaler uses:

```text
exposure = clip(22 / VXN, 0.25, 1.00)
```

At `VXN = 44`, exposure is 50%. A `$100,000` pod therefore carries approximately `$50,000` as residual cash before rounding. The current backtest and incubation model give that cash a zero return.

The real broker rate is not simply DTB3. It depends on settled cash, currency, account NAV, account segment, thresholds, and the broker's historical rate schedule.

The owner has chosen to retain the zero-return treatment as an explicit, intentionally pessimistic policy rather than model those broker details. This is conservative for absolute returns but not neutral across strategies: it penalizes cash-heavy pods such as VXN-scaled NDX and TAA more than fully invested pods.

## 5. Verified current repository behavior

### 5.1 Loader

[`data/norgate_loader.py`](../../data/norgate_loader.py) `:105-143` loads:

- ordinary symbols with `CAPITALSPECIAL`;
- symbols passed in the benchmark list with `TOTALRETURN`.

Snapshot mode preserves the same selection in [`data/norgate_snapshot_store.py`](../../data/norgate_snapshot_store.py) `:396-436`.

The snapshot exporter retains the Norgate fields returned by the API, including `Dividend`, rather than reducing the data to OHLC only. See [`scripts/export_norgate_snapshot.py`](../../scripts/export_norgate_snapshot.py) `:124-139`. The snapshot validation contract does not currently require a `Dividend` column, so this must be made explicit before relying on it universally.

### 5.2 Shared backtest ledger

[`alpha/engine/strategy.py`](../../alpha/engine/strategy.py) `:653-823`:

- executes orders;
- deducts trade notional;
- deducts commission;
- marks positions at the current close;
- calculates `total_value = cash + portfolio_value`.

The common engine does not post:

- ordinary long dividends;
- short dividend liabilities;
- dividend receivables;
- positive-cash interest;
- negative-cash margin interest;
- borrow fees;
- taxes or withholding;
- corporate-action cash or successor securities.

The engine can create negative cash without applying a shared financing charge. Missing-price positions use a synthetic last-available-close liquidation rather than a complete corporate-action replay.

### 5.3 BENCH and analyzers

Vanilla reports and the standard analyzers consume the same `Strategy.results` equity path:

- RiskAnalysis uses `daily_returns` or reconstructs returns from `total_value`: [`alpha/engine/risk_analysis.py`](../../alpha/engine/risk_analysis.py) `:129-152`.
- CapacityAnalysis consumes the completed strategy equity and transaction path: [`alpha/engine/capacity_analysis.py`](../../alpha/engine/capacity_analysis.py) `:1041-1043`.
- ExecutionTiming builds its own timing paths but still uses the same strategy cash and marked-position concepts: [`alpha/engine/execution_timing.py`](../../alpha/engine/execution_timing.py) `:1100-1264`.

No analyzer can reconstruct a dividend or interest event that is absent from the underlying ledger. Existing results are useful price-return research, but they are not complete broker-economic returns.

### 5.4 Incubation

The currently enabled release manifests in this checkout are incubation manifests.

[`alpha/live/incubation.py`](../../alpha/live/incubation.py) `:701-897` records:

- trade notional;
- commission;
- updated positions;
- `total_value = cash + marked positions`.

It does not accrue dividends or interest. Incubation therefore reproduces the same economic omission as the research backtester.

### 5.5 Real IBKR mode

The real broker path reads:

- `TotalCashValue`;
- `NetLiquidation`;
- positions;
- available funds and related account fields.

See [`alpha/live/ibkr_socket_client.py`](../../alpha/live/ibkr_socket_client.py) `:395-428`.

IBKR NetLiq includes dividend accruals as balance-sheet items. Broker cash and NetLiq therefore absorb posted dividends, interest, fees, and taxes at the aggregate account level.

However, the local system does not persist dividend or interest line items. Cash mismatch is intentionally non-blocking in [`alpha/live/reconcile.py`](../../alpha/live/reconcile.py) `:25-46`. Real live equity may therefore be economically correct because IBKR is the source of truth, while the system cannot independently explain why its research/reference cash differs.

### 5.6 Current doctrine conflict

[`CLAUDE.md`](../../CLAUDE.md) `:53` says:

> Use `CAPITALSPECIAL` for individual stocks and `TOTALRETURN` only for benchmark indices.

The WIRED TAA implementation intentionally uses `TOTALRETURN` ETF closes for signals. This is an existing doctrine/code conflict. It must be resolved explicitly rather than hidden by a global loader change.

## 6. WIRED scope

BENCH defines a module as WIRED when it appears in
`SUPPORTED_STRATEGY_IMPORT_TUPLE`. The current imports are listed in
[`alpha/live/release_manifest.py`](../../alpha/live/release_manifest.py):

1. DV2;
2. QPI IBS/RSI exit;
3. HPI S&P 500 2/3/5 vote;
4. HPI S&P 500 IBS/RSI exit;
5. TAA BTAL fallback-TQQQ VIX-cash;
6. TAA BTAL 1/n fallback-TQQQ VIX-cash;
7. TAA BTAL linearity 1/n fallback-QQQ VIX-cash;
8. NDX ATR-normalized momentum;
9. NDX ATR-normalized VXN-scaled momentum.

WIRED does not mean currently submitting broker orders. In this local checkout:

- seven manifest instances are enabled, all in `mode: incubation`;
- the broker-facing QPI paper manifest is disabled;
- the broker-facing NDX VXN and TAA live manifests are disabled;
- two of the WIRED TAA variants have host/template support but no current release manifest.

Runtime VPS state can differ from the local checkout. This document is not an operational live-status report.

## 7. WIRED strategy impact table

| WIRED module | Current signal basis | Execution and valuation | Cash behavior | Main realism consequence |
|---|---|---|---|---|
| `strategy_mr_dv2` | Stock `CAPITALSPECIAL` OHLC; 126-day return, NATR, DV2, SMA200 | Same stock series; next open | Empty slots remain cash | Held dividends and cash yield disappear; ex-dividend signal effects are non-directional |
| `strategy_mr_qpi_ibs_rsi_exit` | Stock `CAPITALSPECIAL` OHLC; 3-day return, SMA200, IBS, QPI, RSI2 | Same stock series; next open | Empty slots remain cash; enabled manifest reserves 5% of account budget | Missing dividends/cash yield depress equity; ex-dividend moves can create artificial oversold states |
| HPI S&P 500 variants | Stock `CAPITALSPECIAL` OHLC on the dedicated no-padding/exact-PIT profile; prior-only 1,260-observation HPI, SMA200, IBS, RSI2 | Same stock series; next open | Empty slots remain cash | Separate profile preserves strict HPI feature timing; missing dividends/cash yield still affect equity realism |
| `strategy_mo_atr_normalized_ndx` | NDX stocks and SPY all use `CAPITALSPECIAL`; momentum, ATR20, stock SMA100, SPY SMA200 | Same stock series; first open of next month | Regime failure can produce 100% cash; incomplete selection leaves residual cash | Long return usually understated, but price-only signals can change holdings and are not simply conservative |
| `strategy_mo_atr_normalized_ndx_vxn_scaled` | Same NDX signals plus observed VXN level | Same | Deliberate 25%-100% exposure, hence up to 75% cash | The intentional `0%` cash policy depresses its absolute result and penalizes it versus the base NDX strategy; stock dividends are also missing |
| `strategy_taa_df_btal_fallback_tqqq_vix_cash` | Defensive ETF momentum uses `TOTALRETURN`; SPY/VIX gate uses `CAPITALSPECIAL` | Tradeable ETFs use `CAPITALSPECIAL`; next-month open | VIX gate can turn the TQQQ fallback allocation into real cash | Signals are distribution-aware, but ETF distributions are missing and the intentional `0%` cash policy penalizes cash-heavy periods |
| `strategy_taa_df_btal_1n_fallback_tqqq_vix_cash` | Same, with equal defensive slots | Same | Same VIX cash mechanism | Same accounting gap |
| `strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash` | `TOTALRETURN` defensive ETF closes feed the linearity signal; SPY/VIX gate uses `CAPITALSPECIAL` | Same, with QQQ fallback | Same VIX cash mechanism | Same accounting gap |

## 8. Strategy-specific examples

### 8.1 DV2

DV2 loads S&P 500 point-in-time stocks as `CAPITALSPECIAL` and `$SPX` as a true `TOTALRETURN` benchmark. Its price-based features are constructed in [`strategies/dv2/strategy_mr_dv2.py`](../../strategies/dv2/strategy_mr_dv2.py) `:158-183`.

The strategy is long-only with up to ten equal slots. Its holding periods are often short, so dividend drag may be smaller than for a monthly ETF allocator, but it is not zero.

Two separate effects exist:

- A position held through entitlement loses the ex-dividend price amount without receiving the dividend in the ledger. This is pessimistic.
- An ex-dividend move can change DV2, NATR, SMA, and entry eligibility. The direction of that strategy change is not known in advance.

### 8.2 QPI

QPI uses price-based 3-day return, SMA200, IBS, QPI, and RSI2 in [`strategies/qpi/strategy_mr_qpi_ibs_rsi_exit.py`](../../strategies/qpi/strategy_mr_qpi_ibs_rsi_exit.py) `:281-360`.

QPI is particularly sensitive to ex-dividend mechanics because a mechanical price drop can look like short-horizon weakness. IBS is an intraday range location and may be less directly affected, but 3-day return, RSI, QPI, and the trend filter can change.

Blindly replacing the strategy's OHLC with synthetic `TOTALRETURN` OHLC would also be wrong because order-relevant ranges must remain tied to the traded market. A dividend-neutral return or reference-close treatment should be tested as a separate signal change.

The enabled local QPI incubation manifest also uses a `0.95` pod budget fraction, leaving an additional account reserve that may or may not earn broker interest.

### 8.3 NDX ATR-normalized momentum

The NDX loader creates:

```text
symbols = point-in-time NDX stocks + SPY
benchmarks = []
```

See [`strategies/momentum/strategy_mo_atr_normalized_ndx.py`](../../strategies/momentum/strategy_mo_atr_normalized_ndx.py) `:344-367`.

Therefore:

- NDX stocks are `CAPITALSPECIAL`;
- SPY regime data is `CAPITALSPECIAL`;
- the benchmark curve produced from the loaded SPY close is also `CAPITALSPECIAL`.

Some helper paths later label the benchmark adjustment as `TOTALRETURN`, while Vanilla artifacts may leave it undeclared. The data curve itself is not total return. This is a concrete provenance defect, not a philosophical signal question.

NDX uses price-only 12-month momentum, ATR20, stock SMA100, and SPY SMA200. Missing dividends normally depress the long ledger, but a change to dividend-aware signals could alter rankings, eligibility, and regime decisions. The impact is not merely an additive return correction.

SPY needs two explicit roles:

```text
SPY_PRICE  -> regime signal, if that is the chosen strategy definition
SPY_TR     -> performance benchmark
```

One loaded column should not silently serve both roles.

### 8.4 VXN-scaled NDX

The VXN variant multiplies the base NDX target weights by:

```text
clip(22 / VXN, 0.25, 1.00)
```

See [`strategies/momentum/strategy_mo_atr_normalized_ndx_vxn_scaled.py`](../../strategies/momentum/strategy_mo_atr_normalized_ndx_vxn_scaled.py) `:123-159`.

The strategy is explicitly no-leverage at the brokerage-account level. VXN itself has no dividend issue. The important gaps are:

- missing dividends on the selected stocks;
- zero interest on deliberate residual cash;
- the same price-only momentum/ATR/SMA question as base NDX;
- the same SPY benchmark provenance problem.

### 8.5 TAA Defense First family

TAA already has the cleanest data separation in the WIRED set:

- `TOTALRETURN` closes form defensive ETF signals;
- `CAPITALSPECIAL` OHLC is used for tradeable ETF execution and valuation;
- `$SPX TOTALRETURN` is used as the benchmark.

See [`strategies/taa_df/strategy_taa_df.py`](../../strategies/taa_df/strategy_taa_df.py) `:187-250`.

The standard BTAL/TQQQ family ranks `GLD`, `UUP`, `TLT`, `DBC`, and `BTAL`. Failed defensive slots flow to the fallback ETF. The VIX overlay can remove the fallback allocation and leave the residual as cash. See [`strategies/taa_df/strategy_taa_df_fallback_vix_cash_variant_utils.py`](../../strategies/taa_df/strategy_taa_df_fallback_vix_cash_variant_utils.py) `:170-229`.

The inconsistencies are:

- DTB3 is used as an economic hurdle, but residual cash earns zero.
- TLT, BTAL, QQQ/TQQQ, and other ETF distributions are absent from account P&L.
- SPY realized volatility for the VIX gate uses `CAPITALSPECIAL`; an ETF distribution can slightly contaminate the 20-day volatility estimate.

TQQQ is an internally leveraged ETF. Its fund financing, fees, and daily reset are already embedded in the traded ETF price. The engine should not add a second brokerage margin charge merely because TQQQ is leveraged. A separate margin charge is required only if the pod itself creates negative broker cash.

### 8.6 Levered All-Weather research variant

The source PDF at `C:/Users/User/Downloads/4weather.pdf` describes a leveraged portfolio of SPY, TLT, DBC, and GLD. It is a useful motivation for this audit because the portfolio combines:

- distribution-paying ETFs;
- explicit leverage;
- periodic rebalancing;
- potentially material cash financing.

The local research-only variant [`strategies/all_weather/strategy_taa_levered_all_weather.py`](../../strategies/all_weather/strategy_taa_levered_all_weather.py):

- uses `CAPITALSPECIAL` ETF prices;
- charges a fixed 2.4% annual rate on negative cash;
- does not receive a shared ordinary-dividend ledger;
- is not WIRED.

Its bias is mixed: omitted distributions are pessimistic, while a fixed 2.4% financing assumption can be optimistic or pessimistic depending on the historical broker-rate regime.

The article does not settle the engine design question. A published equity curve is not enough unless its price adjustment, dividend, financing, tax, and rebalance accounting contracts are explicit.

## 9. Signal-basis decision matrix

There should be no global `use_total_return_bool` that silently changes every feature.

| Feature | Preliminary preferred basis | Reason | Decision status |
|---|---|---|---|
| Order fills and position marks | Non-total-return execution basis | Must correspond to tradeable economics | Strong agreement expected |
| Stop, limit, and gap levels | As-traded or corporate-action-safe price levels | Synthetic total-return levels are not executable | Strong agreement expected |
| Cross-sectional momentum | Causal economic-return index when the hypothesis is investor wealth | Dividends are part of economic performance | Needs controlled validation |
| SMA trend | Price or economic index, explicitly chosen | The choice changes regime and eligibility | OPEN |
| ATR / realized volatility | Dividend-neutral range/return construction | Ordinary distributions should not masquerade as risk | OPEN |
| IBS | Tradeable daily OHLC | It measures location within the traded daily range | Likely price based |
| QPI / RSI / short returns | Price data plus explicit dividend-neutralization where appropriate | Ex-dividend drops can create false oversold signals | OPEN |
| VIX / VXN | Observed index level | These are state variables, not dividend-paying holdings | Strong agreement expected |
| Performance benchmark | True `TOTALRETURN` series with asserted provenance | Benchmark should represent investor wealth | Strong agreement expected |

A safe causal economic-return index can be built from `CAPITALSPECIAL` prices and dividend events known by each date. That avoids relying on a globally back-adjusted price level where scale-sensitive cross-sectional features could accidentally use future adjustment information.

## 10. Dividend timing

Norgate's `Dividend` indicator is attached to the entitlement session: the trading day before the ex-dividend date. The holder at that session's close is entitled.

For a raw Norgate series, let `D^N_t` be the `Dividend` value stamped on
entitlement session `t`. The economic ex-date return on the next trading
session `t+1` is:

`r^(economic)_(t+1) = (Close_(t+1) + D^N_t) / Close_t - 1`.

Equivalently, an ex-date-aligned event series shifts the raw Norgate dividend
forward by one trading session. This alignment is mandatory; adding `D^N_t`
to the entitlement-session close-to-close return would post the event one day
too early.

Therefore a correct event order must ensure:

- a position held at entitlement close receives the dividend;
- a buyer at the next ex-dividend open does not receive it;
- a seller at the ex-dividend open retains the entitlement;
- a short held at entitlement close owes the dividend;
- the event is posted exactly once.

Two accounting models are possible:

### Model A - RealTest-like research accrual

Credit or debit account equity on the ex-dividend session. This is practical with Norgate daily data and closely matches economic total return. RealTest uses this type of mark-to-market dividend credit and does not automatically reinvest the cash into the same position.

### Model B - Broker-style receivable and pay-date cash

Create a dividend receivable on the ex-date, include it in NAV, and convert it to settled cash on pay date. This is closer to IBKR statements but requires reliable pay-date, tax, fee, correction, and currency data.

The joint verdict must decide whether phase 1 targets research-economic parity or exact broker statement timing. A reasonable staged approach is Model A first, followed by Model B when the data contract exists.

## 11. Cash, financing, and borrow

The ledger must distinguish:

- positive settled cash interest;
- negative cash or margin interest;
- short-stock borrow fees;
- interest on short proceeds;
- ETF-internal leverage;
- unsettled trade cash;
- account-specific broker thresholds and tiers.

For IBKR, positive cash interest depends on account NAV, cash balance, currency, account segment, and current rate tiers. The rate changes over time. A constant current rate or a constant DTB3 proxy is not a historically exact broker model.

The owner-approved policy for the current engine is:

- Positive cash earns exactly `0%`. This is an intentional pessimistic assumption, not an unmodelled promise of broker parity.
- No historical broker-rate, threshold, or tier model will be added for positive cash.
- Cross-strategy comparisons must disclose that this policy penalizes cash-heavy strategies asymmetrically, especially VXN-scaled NDX and the TAA cash sleeves.
- The current WIRED strategies are not designed to use account-level leverage. Negative-cash days are reported as a known sizing and execution diagnostic, but do not block the current dividend-accounting research.
- A future strategy that intentionally uses negative cash must introduce and validate a separate financing-cost policy before it can be accepted.

## 12. Issue register

| ID | Issue | Expected bias direction | Impact | Proposed mitigation |
|---|---|---|---|---|
| ER-001 | No shared dividend ledger | Long pessimistic; short optimistic | High | Explicit long credit, short debit, and entitlement tests |
| ER-002 | Positive residual cash earns zero by explicit owner policy | Pessimistic for absolute return; asymmetrically penalizes cash-heavy pods | High for VXN/TAA cash; potentially material for DV2/QPI | Declare `0%` in artifacts and allocation reviews; report or inspect cash exposure when comparing pods |
| ER-003 | The engine can permit free negative cash even though current WIRED strategies are non-levered | Optimistic if it occurs | High if triggered; expected to be absent | Report day count, episodes, minimum dollars, minimum NAV weight, and average deficit; revisit sizing or financing only if the owner later chooses to close the gap |
| ER-004 | Price-only signals react to ex-dividend moves | Non-directional strategy change | High for QPI; Medium for DV2/NDX | Feature-specific dividend-neutral signal research |
| ER-005 | `CAPITALSPECIAL` described as literal trade-as | Audit/provenance error | Medium | Use precise terminology; decide whether true raw corporate-action replay is required |
| ER-006 | NDX SPY benchmark data and metadata disagree or are undeclared | Benchmark comparison unreliable | Medium to High | Load separate SPY price and SPY TR roles; assert provenance |
| ER-007 | Incubation omits events that live IBKR NetLiq includes | Systematic reference drift | High | Add the dividend ledger and declare actual broker cash interest as expected policy-driven drift from the `0%` reference |
| ER-008 | Saved artifacts omit accounting policy metadata | Results can be misinterpreted | Medium | Persist signal, execution, dividend, interest, and tax contracts |
| ER-009 | Ordinary-dividend and cash-policy assumptions were not previously explicit in `ASSUMPTIONS_AND_GAPS.md` | Hidden known gap | Medium | Record the owner-approved zero-positive-cash policy and negative-cash diagnostic contract in the formal register |

## 13. Preliminary Codex verdict

### Decision 1 - execution and valuation

`TOTALRETURN` must never be used for order fills.

Continue using a non-total-return execution basis in the near term. Describe current `CAPITALSPECIAL` accurately as a normalized proxy. True historical as-traded prices plus explicit split/corporate-action share events are the more exact long-term design.

### Decision 2 - dividends

Dividends must be explicit ledger events.

For phase 1, use prior-entitlement-close positions and recognize the economic credit/debit on the ex-dividend transition. Do not automatically reinvest it in the same security.

### Decision 3 - cash and financing

Positive cash earns `0%` by explicit owner decision. This deliberately understates absolute performance and must be disclosed when comparing pods because it penalizes cash-heavy strategies more heavily.

For the current non-levered WIRED book, negative cash remains an optimistic free-financing gap. Report it explicitly without blocking the current research. Add a financing-cost model only when the owner chooses to close this gap or a future strategy is intentionally authorized to use account-level leverage.

Do not double-charge leveraged ETFs for leverage already embedded in their NAV.

### Decision 4 - signals

Do not make a global signal adjustment change.

- Preserve current WIRED strategy semantics until controlled comparisons exist.
- Test dividend-aware NDX momentum separately from ATR and SMA.
- Test dividend-neutral short-horizon features for QPI and DV2.
- Keep observed VIX/VXN levels.
- Keep TAA's signal/execution separation as the leading architectural example, while explicitly resolving its conflict with `CLAUDE.md`.

### Decision 5 - benchmark truth

Fix data provenance so the adjustment label is generated and asserted by the loader. Give SPY separate regime and benchmark series where necessary.

### Decision 6 - release policy

Do not silently overwrite historical BENCH results. Version the accounting contract, regenerate the WIRED baselines, and show old versus corrected results.

## 14. Proposed implementation order after joint approval

No implementation is authorized by this document alone.

If the joint verdict accepts the direction, the lowest-risk sequence is:

1. Add explicit accounting-policy metadata and the formal gap register entry.
2. Add a shadow dividend ledger and reconciliation diagnostics without changing saved headline equity.
3. Validate entitlement timing, long/short signs, and CS-plus-dividend parity against Norgate TR.
4. Activate corrected equity under a versioned engine/accounting contract.
5. Persist the intentional `0%` positive-cash policy and the negative-cash diagnostic contract; do not add a positive-cash rate model.
6. Extend incubation and its cash-ledger schema to non-order economic events.
7. Fix benchmark provenance and separate SPY signal/benchmark roles.
8. Run controlled signal-basis variants; do not mutate WIRED signals in place.
9. Regenerate Vanilla, Risk, Capacity, Stress, and reference artifacts for every WIRED strategy.
10. Compare corrected reference equity against IBKR EOD NetLiq, explicitly classifying actual broker cash interest as expected drift from the intentional `0%` reference policy.

## 15. Minimum acceptance tests

A corrected implementation should not be accepted until all of these are demonstrated:

- A long held at entitlement close receives exactly one dividend.
- A same-open ex-date buyer receives no dividend.
- A same-open ex-date seller retains entitlement.
- A short held at entitlement close pays the dividend.
- A zero-market-move ex-dividend example leaves gross economic NAV unchanged.
- `CAPITALSPECIAL + explicit eligible dividends` matches Norgate `TOTALRETURN` buy-and-hold wealth within a documented tolerance.
- No future dividend information enters a historical signal.
- No dividend is counted both in a price series and in the ledger.
- Positive cash earns exactly `0%`, and saved artifacts identify this as an intentional pessimistic policy.
- Every current WIRED run reports negative-cash day count, episodes, minimum dollars, minimum NAV weight, and average deficit without blocking the research run.
- TAA and VXN residual cash receive the same zero-return treatment, and cross-pod allocation reviews disclose the asymmetric penalty.
- Leveraged ETF financing is not double-counted.
- NDX has separate, correctly labelled SPY regime and SPY benchmark series.
- Incubation and Vanilla produce the same accounting events under the same price path.
- IBKR comparisons require position and dividend reconciliation while explicitly labeling actual broker cash interest as accepted policy-driven drift from the `0%` reference.
- Saved artifacts declare signal adjustment, execution adjustment, dividend policy, cash policy, tax policy, and engine version.
- All WIRED analyzer outputs are regenerated and compared before a deployment decision.

## 16. Blast radius of a future implementation

This is not a small strategy edit.

At minimum, a shared dividend or interest implementation would touch Tier 2 engine behavior:

- loader and snapshot contracts;
- `Strategy` cash and equity;
- alternate execution timing;
- reports, metrics, capacity, risk, and portfolio aggregation;
- saved artifact schemas.

If extended to incubation, released configs, broker reconciliation, or live state, it becomes Tier 3:

- incubation cash ledger;
- state-store schema;
- EOD snapshots;
- reconciliation and reference comparison;
- dashboard/operator fields;
- backward compatibility of existing pod state.

Changing any WIRED signal basis is also a separate quantitative strategy change and requires the quant-pitfalls, parity, and coverage review stack.

## 17. Questions for Claude

Claude should answer each question directly:

1. Is the verified description of `CAPITALSPECIAL`, `TOTALRETURN`, and the current ledger correct?
2. Should phase 1 recognize dividends as RealTest-like ex-date equity events, or should it wait for a full receivable/pay-date model?
3. Should ordinary dividends be gross, net of withholding, or policy-configurable?
4. Which exact NDX features should use economic returns: momentum, SMA, ATR, or some subset?
5. How should QPI and DV2 neutralize ex-dividend moves without replacing executable OHLC?
6. Should TAA retain `TOTALRETURN` ETF signals despite the current `CLAUDE.md` rule?
7. Is zero interest an acceptable declared conservative mode for positive cash? What is the minimum acceptable negative-cash model?
8. Should current BENCH artifacts be labelled `price-return ledger` until they are regenerated?
9. Is fixing NDX benchmark provenance a prerequisite before any further performance verdict?
10. Does the proposed phased implementation minimize live/research parity risk, or should the order change?

## 18. Review and joint-verdict table

| Decision | Codex position | Claude position | Joint verdict |
|---|---|---|---|
| Fill and mark price | Never `TOTALRETURN`; keep non-TR basis | AGREE. Verified ledger in `alpha/engine/strategy.py:653-823`; TR fills plus an explicit dividend ledger would double-count | AGREED (Codex + Claude); owner sign-off pending |
| Meaning of current `CAPITALSPECIAL` | Adjusted execution proxy, not literal as-traded | AGREE. Split-adjusted levels confirmed (AAPL 2020 unadjusted ~504 vs CAPITALSPECIAL ~126); terminology fix only, no behavior change required now | AGREED; owner sign-off pending |
| Dividend accounting | Explicit long credit and short debit | AGREE. Gross amount computed first, with a configurable withholding rate applied as policy (see Claude answers, Q3) | AGREED; owner sign-off pending |
| Dividend timing | Ex-date economic event first; receivable/pay-date later | AGREE (Model A first). Norgate entitlement-session stamping verified empirically on 2026-07-25: AAPL `Dividend` 0.205 on 2020-08-06 (ex 2020-08-07), SPY 1.3392 on 2020-09-17 (ex 2020-09-18) | AGREED; owner sign-off pending |
| Positive cash | Explicit policy; zero allowed if declared | AGREE. Zero is acceptable as a declared pessimistic mode; materiality is highest for VXN-scaled and TAA cash sleeves | OWNER DECISION: `0%` by design; no broker-rate model; disclose the asymmetric cross-strategy penalty |
| Negative cash | Must never remain an invisible free-financing path | AGREE that every negative-cash episode must be visible because no WIRED strategy is meant to lever | OWNER DECISION: report the defined diagnostics without blocking current research; defer sizing and financing changes |
| Signals | Feature-specific; no global TR switch | AGREE. Controlled A/B variants only; strongest candidate is NDX cross-sectional momentum; IBS stays on traded OHLC; VIX/VXN stay observed levels | AGREED; owner sign-off pending |
| TAA TR signals | Defensible existing exception, doctrine must be clarified | AGREE retain. Return-space TR signal formation is causal; amend `CLAUDE.md` to state the precise rule (TR allowed for return-space signals and benchmarks; never for fills, marks, or level/scale-sensitive features) | AGREED; owner sign-off pending |
| NDX benchmark provenance | Must be corrected | AGREE. Confirmed defect: `strategy_mo_atr_normalized_ndx.py:916` and the VXN variant `:409` stamp `TOTALRETURN` while the loader fetched SPY with `benchmarks=[]`, i.e. `CAPITALSPECIAL`. Fix early — it is small and independent | AGREED; owner sign-off pending |
| Existing results | Preserve and relabel/version; regenerate after change | AGREE. Label existing artifacts `price-return ledger` immediately; regenerate under a versioned accounting contract | AGREED; owner sign-off pending |
| Implementation authorization | None until joint approval | AGREE. This document authorizes nothing; owner must approve scope and sequence | AGREED that no implementation is authorized; OPEN until owner approves |

## 19. Evidence and external references

### Repository sources

- [`QUANT_PHILOSOPHY.md`](../../QUANT_PHILOSOPHY.md)
- [`ASSUMPTIONS_AND_GAPS.md`](../../ASSUMPTIONS_AND_GAPS.md)
- [`CLAUDE.md`](../../CLAUDE.md)
- [`data/norgate_loader.py`](../../data/norgate_loader.py)
- [`alpha/engine/strategy.py`](../../alpha/engine/strategy.py)
- [`alpha/live/release_manifest.py`](../../alpha/live/release_manifest.py)
- [`alpha/live/incubation.py`](../../alpha/live/incubation.py)
- [`alpha/live/ibkr_socket_client.py`](../../alpha/live/ibkr_socket_client.py)
- [`alpha/live/reconcile.py`](../../alpha/live/reconcile.py)
- [`docs/research/ENGINE_REALISM_DIVIDENDS_AND_REALTEST_HE.md`](ENGINE_REALISM_DIVIDENDS_AND_REALTEST_HE.md)

### External references

- Norgate Dividend indicator and entitlement timing: <https://norgatedata.com/data-content-tables.php>
- RealTest adjustment modes: <https://mhptrading.com/docs/topics/idh-topic1390.htm>
- RealTest dividend handling: <https://mhptrading.com/docs/topics/idh-topic1100.htm>
- RealTest `IgnoreDividends`: <https://mhptrading.com/docs/topics/idh-topic10807.htm>
- IBKR dividend accruals and NAV: <https://www.ibkrguides.com/reportingreference/reportguide/changeindividendaccruals_realized.htm>
- IBKR cash-interest calculations: <https://www.interactivebrokers.com/en/pricing/pricing-calculations-int.php>

## 20. Review log

### Codex - 2026-07-25

- Completed a read-only audit of the shared engine, the original PDF, official Norgate/RealTest/IBKR documentation, all seven WIRED modules, local release manifests, incubation accounting, and broker snapshot accounting.
- Recorded the preliminary verdict above.
- Made no implementation or release change.

### Claude - 2026-07-25

**Status:** COMPLETE

**Reviewer:** Claude (Fable 5), independent read-only verification against the local checkout plus a live Norgate data check.

**Date:** 2026-07-25

**Facts verified independently:**

- Loader adjustment split (`CAPITALSPECIAL` for symbols, `TOTALRETURN` for benchmarks): confirmed at `data/norgate_loader.py:125-129`.
- Shared ledger posts only trade notional, commission, and close marks; `total_value = cash + portfolio_value`; no dividend, interest, borrow, or tax events; negative cash possible with no financing charge: confirmed at `alpha/engine/strategy.py:653-823`.
- Incubation cash ledger contains only `trade_notional` and `commission` entry types; `total_value_float = cash + marked positions`: confirmed at `alpha/live/incubation.py:796-897`.
- Cash mismatch intentionally non-blocking in reconciliation: confirmed at `alpha/live/reconcile.py:25-46`.
- TAA signal/execution separation (`TOTALRETURN` signal closes, `CAPITALSPECIAL` execution OHLC): confirmed at `strategies/taa_df/strategy_taa_df.py:187-250`.
- NDX benchmark provenance defect: `strategies/momentum/strategy_mo_atr_normalized_ndx.py:916` and `strategy_mo_atr_normalized_ndx_vxn_scaled.py:409` hard-code `_performance_benchmark_adjustment_str = "TOTALRETURN"` while the loader is called with `benchmarks=[]` at `:362-367`, so SPY is actually `CAPITALSPECIAL`. Metadata and data disagree exactly as claimed.
- Norgate dividend timing: verified empirically on 2026-07-25 with the local Norgate install. AAPL shows `Dividend = 0.205` (0.82 pre-split / 4) on 2020-08-06, the session **before** the 2020-08-07 ex-date; SPY shows `Dividend = 1.3392` on 2020-09-17, the session before the 2020-09-18 ex-date. The entitlement-session convention in section 10 is correct as stated, and the amount is expressed in the split-adjusted price scale, which is the correct scale to pair with `CAPITALSPECIAL` share quantities.

**Facts disputed:** None. All checked factual claims in sections 5-8 and 10 are accurate.

**Decisions accepted:** All six preliminary Codex decisions (execution basis, dividends, cash/financing, signals, benchmark truth, release policy).

**Decisions rejected:** None.

**Alternative proposal (amendments, not rejections):**

1. **Reorder the implementation sequence.** Move benchmark-provenance repair (step 7) to immediately after step 1. It is small, independent of the ledger work, and the current state is not merely incomplete but actively mislabeled — the only place in this audit where an artifact claims something untrue rather than omitting something.
2. **Make the snapshot `Dividend` column a validation-contract requirement in step 1.** Section 5.1 notes the exporter retains `Dividend` but the validation contract does not require it. The dividend ledger must not silently degrade when running in snapshot mode on the client VPS; a missing `Dividend` column must fail loud.
3. **Add withholding as a first-class policy field from day one** (see Q3 below), even if phase 1 defaults it to zero for the TR-parity test.

**Answers to section 17 questions:**

1. **Correct?** Yes. Every claim I checked is verified above, including the two that matter most: the ledger omission and the entitlement-session dividend stamping.
2. **Model A or wait for Model B?** Model A now. It is implementable from data already in the snapshot, it matches the RealTest research convention, and the CS-plus-dividends vs TR parity test gives it an objective acceptance gate. Model B adds pay-date/receivable realism that only matters for IBKR statement-level reconciliation; build it when that reconciliation is actually being enabled. Waiting for Model B means keeping a known-wrong ledger longer for no research benefit.
3. **Gross, net, or configurable?** Policy-configurable with an explicit rate, computed as gross first. The parity test against Norgate `TOTALRETURN` must run at withholding = 0 (TR is a gross-reinvestment series). The live-first default for this operator should then be net of actual withholding (for an Israeli tax resident holding US listings via IBKR, the US treaty rate is 25%; verify against actual IBKR statements before hard-coding). A gross-only ledger would overstate live-account economics by roughly a quarter of the dividend stream.
4. **Which NDX features get economic returns?** Only cross-sectional momentum is a strong candidate, because its hypothesis is relative investor wealth. ATR should remain a traded-price range measure (a mechanical ex-dividend gap is not volatility, but ordinary NDX dividends are small relative to ATR20; test a dividend-neutral variant before adopting it). Stock SMA100 and SPY SMA200 should stay price-based initially — changing a regime filter's basis changes regime dates, which is a strategy redefinition, not a correction. All of this via controlled A/B runs; never mutate the WIRED signal in place.
5. **QPI/DV2 ex-dividend neutralization?** Compute short-horizon returns as dividend-neutral returns: `r = (close_t + dividend_known_at_t) / close_{t-1} - 1`, using the entitlement-aligned `Dividend` field (causally safe: Norgate stamps it at the entitlement session, after public declaration). For level-based indicators (RSI2, DV2, SMA200) build a *causal forward-compounded economic close index* anchored at the strategy start — never a globally back-adjusted series, whose scale embeds future adjustment factors. IBS stays on raw traded OHLC; it measures location in the actual traded range. All variants are separate research strategies until validated.
6. **TAA TR signals despite `CLAUDE.md`?** Retain them. Return-space signals from a TR series over past windows use only adjustment factors from dividends that occurred inside those past windows — causal. The `CLAUDE.md:53` rule exists to block forward-looking bias in *level/scale-sensitive* usage of back-adjusted stock series and to keep fills honest. Resolution: amend the doctrine to the precise rule — `TOTALRETURN` is permitted for return-space signal formation and benchmarks, and forbidden for fills, marks, stops/limits, and any level- or scale-sensitive feature — rather than forcing TAA back to price-return signals, which would change a WIRED strategy's meaning.
7. **Zero positive-cash interest acceptable? Minimum negative-cash model?** Zero is acceptable only as a *declared* pessimistic mode, and the declaration must note materiality: near-zero drag over 2009-2021, but on the order of 1-3%/yr of understatement for a pod holding 25-75% cash during 2022-2025 rate regimes. The recommended default is a dated benchmark-rate series (e.g. daily fed funds or DTB3) minus a broker haircut, applied to positive settled cash above a threshold. Minimum negative-cash model: benchmark rate plus a spread (IBKR-style ~1.5%) charged daily on negative settled cash. This was Claude's reviewer recommendation; the later owner amendment supersedes it for the current research and requires diagnostics only.
8. **Label BENCH artifacts `price-return ledger`?** Yes, immediately. It costs nothing, changes no numbers, and converts a hidden assumption into a declared one — exactly the fail-loud doctrine.
9. **NDX provenance fix a prerequisite?** Yes, for any *benchmark-relative* verdict (alpha, relative drawdown, analyzer comparisons). The strategy's own equity curve is unaffected, so absolute-return conclusions stand. This is also the item I would fix first (see amendment 1).
10. **Does the phased order minimize parity risk?** Broadly yes. Three changes: pull the provenance fix forward (amendment 1); fold the snapshot `Dividend` contract into step 1 (amendment 2); and treat step 3's parity validation as the hard gate for everything after it — no corrected-equity activation (step 4) until CS + dividends reproduces TR buy-and-hold wealth within the documented tolerance on a multi-symbol, multi-year sample including split-plus-dividend overlaps (the AAPL 2020-08-06/07 window is a good canonical test case, since it has a dividend two days after a 4:1 split).

**Bias-direction summary for the owner:** For the current all-long, no-leverage, no-short WIRED book, the ledger omissions make backtests and incubation *pessimistic* (missing long dividends, missing cash interest), not inflated. The genuinely misleading items are narrower: (a) the NDX benchmark label claims TR while the data is price-return, so benchmark-relative conclusions there are unreliable; (b) incubation/reference equity will drift from IBKR NetLiq for reasons the system cannot itemize; (c) signal-level ex-dividend contamination in QPI/DV2 is non-directional — it adds noise trades, not systematic optimism. Free negative cash and short-dividend omission are real engine defects but currently latent for this strategy set.

**Final verdict:** ACCEPT all six Codex decisions with the three amendments above. No implementation is authorized by this review; owner sign-off on scope and sequence is the remaining gate.

### Owner cash-policy decision - 2026-07-25

- Positive cash earns `0%` in research, backtests, and incubation. This is an intentional pessimistic assumption.
- No positive-cash broker-rate, threshold, or tier model is requested.
- The owner accepts that this policy is not neutral when comparing pods: it penalizes cash-heavy strategies such as VXN-scaled NDX and TAA more than fully invested strategies.
- Current WIRED strategies are not intended to use account-level leverage. Negative cash is reported through the defined diagnostics and does not block the current dividend-accounting research.
- Financing-rate modeling is deferred until a future strategy is intentionally authorized to use negative cash.
- This decision does not authorize the remaining engine or live implementation.

### Joint verdict

**Status:** ALIGNED (Codex + Claude, 2026-07-25), with the owner cash-policy amendment above. The owner authorized Phase 1 truth/provenance work; dividend-ledger activation remains gated.

**Approved accounting contract (proposed):**

- Execution and marks stay on the non-TR `CAPITALSPECIAL` basis, documented as a corporate-action-normalized proxy, never `TOTALRETURN`.
- Model A dividend events: long credit / short debit recognized on the ex-dividend transition from prior-entitlement-close positions, using Norgate's entitlement-session `Dividend` field (empirically verified convention); no automatic reinvestment; posted exactly once.
- Withholding is a first-class configurable policy (gross computed first; parity tests at 0; live-first default set to the operator's actual treaty rate after verification against IBKR statements).
- Positive cash earns `0%` by intentional owner policy. Negative cash is a disclosed diagnostic-only gap for the current research; financing-rate modeling remains deferred.
- No double-charging of leveraged-ETF internal financing.
- Accounting contract is versioned; existing artifacts relabeled `price-return ledger` and preserved; corrected baselines regenerated alongside, never overwritten silently.

**Approved signal contracts (proposed):**

- No global TR switch. WIRED signal semantics frozen until controlled A/B comparisons exist.
- TAA keeps `TOTALRETURN` return-space signals; `CLAUDE.md` doctrine amended to the precise rule (TR for return-space signals and benchmarks only; never fills, marks, or level/scale-sensitive features).
- IBS stays on traded OHLC; VIX/VXN stay observed levels; SPY gets separate regime (price) and benchmark (TR) series with loader-asserted provenance.

**Required experiments:**

- CS + explicit dividends vs Norgate TR buy-and-hold parity across multiple symbols/years, including split-plus-dividend overlaps (AAPL 2020-08 canonical case).
- Dividend-aware NDX momentum A/B (momentum only; ATR and SMA unchanged).
- Dividend-neutral short-horizon return variants for QPI and DV2 as separate research strategies.

**Implementation scope:** Codex's section 14 sequence with Claude's amendments and the owner cash-policy amendment: benchmark-provenance fix moved to immediately after step 1; snapshot validation contract must require the `Dividend` column in step 1; step 3 parity is a hard gate for steps 4+; positive-cash rate modeling is removed; negative cash remains a reported, non-blocking gap for the current research.

**Deployment gate:** All section 15 acceptance tests pass; all WIRED analyzer artifacts regenerated and old-vs-corrected compared; IBKR position/dividend reconciliation demonstrated; any NetLiq difference caused by actual broker cash interest is explicitly labeled as expected drift from the `0%` reference policy.

### Phase 1 implementation - 2026-07-25

Implemented scope:

- Direct-Norgate NDX and VXN-NDX research/analyzer runs keep `SPY`
  `CAPITALSPECIAL` for the regime signal and use the explicit Norgate `$SPXTR`
  series for the reported `$SPX` total-return benchmark. Vanilla, Capacity,
  Execution Timing, and NDX Crisis replay all resolve the public `$SPX` label
  through this mapping.
- The benchmark display label is `$SPX`; saved metadata records the actual
  `$SPX -> $SPXTR` data-symbol mapping and adjustment roles.
- This benchmark amendment does not change the snapshot exporter, client
  schema, VPS rollout, or live DecisionPlan. Existing NDX snapshots do not
  contain `$SPXTR`, so snapshot-mode analyzers remain unsupported until a
  separately approved rollout; live trading remains `CAPITALSPECIAL`-only.
- Strategy artifacts declare `price_return_ledger_v1`, dividends not credited,
  intentional `0%` positive-cash return, and no modeled negative-cash financing.
- Existing snapshot schema v1 remains readable. New exporter output is schema
  v2 and fails validation if the `Dividend` field is absent, nonnumeric, or
  null. Norgate index/helper rows that cannot distribute cash (`$SPX`, `$VIX`,
  `$VXN`) are explicitly normalized to `Dividend = 0`; missing ETF/stock
  dividend data still fails loudly. NDX v2 snapshots must contain both SPY
  `CAPITALSPECIAL` and SPY `TOTALRETURN` rows.
- Deployment is reader-first: every client receives the dual v1/v2 reader
  before the producer publishes v2. Same-date analyzer regeneration uses the
  validated `--overwrite` sync path when a client already has v1.
- TAA and all other WIRED signal rules remain unchanged.

Not activated in Phase 1:

- no dividend cash or receivable events;
- no withholding calculation;
- no positive-cash interest model;
- no financing-rate model;
- no corrected-equity baseline replacement;
- no live order, sizing, fill, or reconciliation change.

### Owner amendment - negative cash diagnostics - 2026-07-25

The owner has deferred cash-constrained sizing and negative-cash financing.
For the current dividend-accounting research, negative cash is therefore a
disclosed diagnostic rather than a blocking gate. Reports must include its day
count, episode count, minimum dollar balance, minimum NAV weight, and average
deficit. Positive cash continues to earn the intentional `0%`.

This amendment does not change LIVE, VPS, release YAML, order sizing, or the
existing engine.

### Full WIRED dividend-ledger A/B - 2026-07-25

The approved gross-parity candidate was run once against the unchanged
price-return baseline for all seven WIRED strategy modules. This was one
predefined accounting comparison, not a parameter sweep. Costs, slippage,
signal definitions, execution timing, universes, and positive-cash return
remained unchanged. Withholding was `0%` so that this run measures gross
dividend accounting before any owner-specific tax policy.

| WIRED strategy | Baseline CAGR | Dividend CAGR | Delta | Baseline Sharpe | Dividend Sharpe |
|---|---:|---:|---:|---:|---:|
| DV2 | 17.836% | 19.236% | +1.400 pp | 0.911 | 0.969 |
| QPI | 14.302% | 15.743% | +1.441 pp | 0.944 | 1.026 |
| TAA BTAL fallback | 22.809% | 23.740% | +0.931 pp | 1.277 | 1.321 |
| TAA BTAL 1/N | 29.158% | 30.043% | +0.885 pp | 1.219 | 1.249 |
| TAA linearity 1/N | 11.834% | 12.709% | +0.875 pp | 1.236 | 1.320 |
| NDX ATR-normalized | 18.922% | 19.493% | +0.571 pp | 1.119 | 1.147 |
| NDX VXN-scaled | 17.367% | 17.911% | +0.544 pp | 1.194 | 1.226 |

All seven pairs completed with identical pricing inputs, calendars, and stored
signal diagnostics. DV2 and QPI also retained identical executed
date/asset/direction skeletons. The five target-sized monthly strategies
developed small execution-skeleton differences because the credited
dividends raised NAV and therefore changed integer target-share rounding.
Examples include an additional one-share rebalance or a tiny target correction
changing from a two-share sale to a three-share purchase. This is an intended
accounting consequence, not a signal change, but it means the result is not a
fixed-quantity attribution.

Negative cash remained the owner-approved non-blocking diagnostic and was
reported separately for baseline and candidate ledgers. The study did not add
cash interest, financing charges, a reserve-cash rule, or any
deployment-budget feature.

Saved artifact:
`results/research/accounting/wired_dividend_cash_ledger/ab_study/2026-07-25_213011`.

**Research verdict:** Gross dividend omission is material and pessimistic for
every current long-only WIRED strategy. This gross run established the upper
accounting bound and justified the owner-approved `25%` withholding run below.
It did not authorize an engine, LIVE, VPS, release, or reconciliation change.

### Full WIRED net-dividend A/B - 2026-07-25

The same seven baseline/candidate pairs were rerun with the owner-approved
`25%` withholding rate. Every positive dividend event therefore credited
`75%` of gross cash; a future short manufactured-dividend event would still
debit the full gross amount. Positive cash continued to earn `0%`, and negative
cash remained diagnostic-only.

| WIRED strategy | Baseline CAGR | Net-dividend CAGR | Delta | Baseline Sharpe | Net-dividend Sharpe |
|---|---:|---:|---:|---:|---:|
| DV2 | 17.836% | 18.886% | +1.050 pp | 0.911 | 0.955 |
| QPI | 14.302% | 15.384% | +1.082 pp | 0.944 | 1.005 |
| TAA BTAL fallback | 22.809% | 23.508% | +0.699 pp | 1.277 | 1.310 |
| TAA BTAL 1/N | 29.158% | 29.821% | +0.664 pp | 1.219 | 1.241 |
| TAA linearity 1/N | 11.834% | 12.492% | +0.658 pp | 1.236 | 1.299 |
| NDX ATR-normalized | 18.922% | 19.350% | +0.429 pp | 1.119 | 1.140 |
| NDX VXN-scaled | 17.367% | 17.775% | +0.408 pp | 1.194 | 1.218 |

Validation results:

- all seven runs completed;
- every baseline terminal value exactly matched the prior gross-study baseline;
- pricing inputs, calendars, and stored signal diagnostics matched within every
  baseline/candidate pair;
- `net_dividend_cash / gross_dividend_cash = 0.75` for every current long-only
  WIRED strategy;
- every net-dividend CAGR was strictly above its no-dividend baseline and
  strictly below its gross-dividend counterpart.

Saved artifact:
`results/research/accounting/wired_dividend_cash_ledger/ab_study/2026-07-25_224817`.

**Net research verdict:** The `25%` withholding candidate is the recommended
backtest-accounting policy for the current owner context. It materially reduces
the known pessimistic dividend omission without using gross cash that the
account does not retain. The next gated step is an engine-only implementation
under a new accounting-contract version, followed by regeneration and
comparison of all WIRED analyzers. LIVE, VPS, release YAML, signal definitions,
order timing, and execution remain out of scope unless separately authorized.

### Engine-only activation - 2026-07-26

The owner authorized the next gated step on the dedicated
`codex/dividend-ledger-v2` branch.

Implemented behavior:

- Vanilla engine inputs containing a `Dividend` field activate
  `net_dividend_cash_ledger_v2` automatically.
- Fills and end-of-day marks remain on the existing non-TR execution basis.
- The engine samples positions held before the current open, reads Norgate's
  `Dividend` from the prior entitlement session, and posts the event before
  processing current-open orders.
- Positive gross dividend cash uses the owner default `25%` withholding; short
  manufactured dividends are debited at full gross.
- Dividends remain cash. There is no automatic share purchase and no change to
  strategy signals, order timing, or sizing policy. A later normal sizing
  decision may use the higher recorded NAV.
- Positive cash still earns `0%`. Negative cash remains a reported,
  non-blocking diagnostic with no financing charge.
- Saved strategy artifacts include versioned accounting metadata and a separate
  `dividend_ledger.csv`.
- Norgate execution inputs carry per-symbol adjustment provenance. The ledger
  fails before execution if a traded asset is marked `TOTALRETURN`; verified
  traded assets are recorded as `CAPITALSPECIAL` in artifact metadata.
- Legacy inputs with no `Dividend` field remain readable and are explicitly
  labeled `price_return_ledger_v1`; callers can force validation when v2 is
  required.

Semantics explicitly unchanged:

- incubation, broker reconciliation, VPS procedures, release YAML, and pod
  sizing;
- automatic `compare_reference` remains explicitly pinned to
  `price_return_ledger_v1`; a small fail-loud guard prevents the new research
  ledger from entering this LIVE diagnostic before a separate authorization;
- WIRED signals, regimes, universes, costs, slippage, and next-open execution;
- existing saved artifacts, which are not overwritten.

### WIRED Vanilla regeneration - 2026-07-26

All seven WIRED Vanilla artifacts were regenerated from the engine
implementation and compared with the frozen 2026-07-25 paired baseline and
approved net-dividend shadow candidate.

| WIRED strategy | Old CAGR | Engine v2 CAGR | Delta | Old Sharpe | Engine v2 Sharpe |
|---|---:|---:|---:|---:|---:|
| DV2 | 17.836% | 18.886% | +1.050 pp | 0.911 | 0.955 |
| QPI | 14.302% | 15.384% | +1.082 pp | 0.944 | 1.005 |
| TAA BTAL fallback | 22.809% | 23.508% | +0.699 pp | 1.277 | 1.310 |
| TAA BTAL 1/N | 29.158% | 29.821% | +0.664 pp | 1.219 | 1.241 |
| TAA linearity 1/N | 11.834% | 12.492% | +0.658 pp | 1.236 | 1.299 |
| NDX ATR-normalized | 18.922% | 19.350% | +0.429 pp | 1.119 | 1.140 |
| NDX VXN-scaled | 17.367% | 17.775% | +0.408 pp | 1.194 | 1.218 |

Validation results:

- all seven engine-v2 artifacts reproduced the approved shadow candidate;
- transactions matched exactly after excluding the process-global `order_id`;
- dividend-ledger rows matched exactly;
- headline metrics and terminal NAV matched within recorded CSV/JSON
  serialization tolerances;
- pricing inputs and stored signal diagnostics remained equal in every pair;
- every saved artifact declares `net_dividend_cash_ledger_v2`, `25%`
  withholding, `0%` positive-cash return, `CAPITALSPECIAL` execution/marks,
  and a `TOTALRETURN` reporting benchmark;
- the regeneration exposed and fixed stale DV2/QPI/TAA regression-benchmark
  metadata. This reporting-only fix initializes benchmark provenance before
  summarization and did not change any economic result.

Saved comparison package:
`results/dividend_ledger_v2_wired_regeneration_20260726/comparison`.

The seven-strategy Vanilla regeneration gate is complete. Risk, Capacity,
Stress, live-reference, incubation, and broker-reconciliation rollout remain
separate tasks; any LIVE or incubation dividend event still requires explicit
authorization.
