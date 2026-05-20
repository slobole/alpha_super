TL;DR: For this repo, the most defensible simple model is:

```text
Baseline:
exact_fees + fixed_slippage

Capacity / scaling check:
exact_fees + fixed_slippage + ADV_addon
```

This note records the source-backed facts, the current repo implementation snapshot, and the practical house heuristic that follows from them.

# Transaction Costs Research

As of date: `2026-04-16`

## Purpose

This file exists so future research and live-trading work can reuse one audited summary of:

- explicit broker and regulatory fees
- slippage and opening-auction caveats
- IBKR PaperTrader limitations
- the current repo implementation
- the recommended simple house model

The note separates:

- `direct_source_str`: directly supported by an external or local source
- `house_inference_str`: a practical conclusion inferred from the sources
- `house_heuristic_str`: a deliberately simple modeling rule chosen for robustness and ease of use

## Current Repo Snapshot

As of `2026-04-16`, the local repo implements:

- Engine default slippage:

```text
slippage_float = 0.00025 = 2.5 bps one-way
```

Source:
- `alpha/engine/strategy.py:42-54`

- Commission model:

```text
commission_cash_float
=
max(commission_minimum_float,
    commission_per_share_float * abs(qty_float))
```

Source:
- `alpha/engine/strategy.py:110-114`

- Live MOO submission path:

```text
broker_order_type_str == "MOO" -> orderType="MKT", tif="OPG"
```

Source:
- `alpha/live/ibkr_socket_client.py:136-142`

- Live execution-quality reference convention currently uses the frozen sizing reference price, which is taken from the prior close:

```text
sizing_reference_price_float = previous_close_price_float
```

Sources:
- `alpha/live/strategy_host.py:145-159`
- `alpha/live/state_store.py:652-670`

Important consequence:

```text
current_live_slippage_snapshot
=
fill_px_float - prior_close_reference_px_float
```

not:

```text
fill_px_float - official_open_px_float
```

That means any future opening-auction analysis must keep the two benchmarks separate.

## Core Formulas

Order notional:

```text
order_notional_float
=
abs(qty_float) * reference_price_float
```

Explicit fee burden:

```text
explicit_fee_bps_float
=
10_000 * explicit_fee_cash_float / order_notional_float
```

One-way total cost:

```text
one_way_total_cost_bps_float
=
explicit_fee_bps_float
+ implicit_slippage_bps_float
```

Round-trip total cost:

```text
round_trip_total_cost_bps_float
=
2 * one_way_total_cost_bps_float
```

Capacity proxy:

```text
participation_rate_float
=
order_notional_float / adv20_dollar_float
```

## Source-Backed Findings

| Claim | Type | Source |
|---|---|---|
| IBKR US stock fixed commission is `USD 0.005` per share with `USD 1.00` minimum per order. | `direct_source_str` | [IBKR commissions](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php) |
| IBKR US stock pricing page explicitly lists third-party fees: regulatory, exchange, clearing, and pass-through fees. | `direct_source_str` | [IBKR commissions](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php) |
| IBKR currently lists `SEC Transaction Fee = USD 0.0000206 * aggregate sales value`. | `direct_source_str` | [IBKR commissions](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php) |
| IBKR currently lists `FINRA Trading Activity Fee = USD 0.000195 * quantity sold`. | `direct_source_str` | [IBKR commissions](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php), [FINRA fee schedule](https://www.finra.org/rules-guidance/rule-filings/sr-finra-2024-019/fee-adjustment-schedule) |
| IBKR currently lists `NSCC/DTC clearing fee = USD 0.00020 per share`. | `direct_source_str` | [IBKR commissions](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php) |
| SEC Section 31 fee was `USD 0.00 per million` for covered sales from `2025-05-14` through `2026-04-03`. | `direct_source_str` | [SEC FY2025 Section 31 advisory](https://www.sec.gov/rules-regulations/fee-rate-advisories/2025-2), [SEC FY2026 Section 31 advisory](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2) |
| SEC Section 31 fee became `USD 20.60 per million` for covered sales on or after `2026-04-04`. | `direct_source_str` | [SEC FY2026 Section 31 advisory](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2) |
| FINRA's 2026 TAF schedule for covered equity securities is `USD 0.000195` per share, capped per trade. | `direct_source_str` | [FINRA fee schedule](https://www.finra.org/rules-guidance/rule-filings/sr-finra-2024-019/fee-adjustment-schedule) |
| IBKR PaperTrader has no execution or clearing ability because it is a simulator. | `direct_source_str` | [IBKR Paper Trading Account glossary](https://www.interactivebrokers.com/campus/glossary-terms/paper-trading-account/), [IBKR paper vs live lesson](https://www.interactivebrokers.com/campus/trading-lessons/paper-trading-vs-live-trading-whats-the-difference/) |
| IBKR PaperTrader does not support some order types including auction orders. | `direct_source_str` | [IBKR Paper Trading Account glossary](https://www.interactivebrokers.com/campus/glossary-terms/paper-trading-account/), [IBKR paper vs live lesson](https://www.interactivebrokers.com/campus/trading-lessons/paper-trading-vs-live-trading-whats-the-difference/) |
| IBKR PaperTrader fills are simulated from the top of the book and have no deep book access. | `direct_source_str` | [IBKR Paper Trading Account glossary](https://www.interactivebrokers.com/campus/glossary-terms/paper-trading-account/), [IBKR paper vs live lesson](https://www.interactivebrokers.com/campus/trading-lessons/paper-trading-vs-live-trading-whats-the-difference/) |
| IBKR says smart-routed `MOO` orders are designed to route to and execute on the primary listing exchange. | `direct_source_str` | [IBKR Market-on-Open order type](https://www.interactivebrokers.com/en/trading/orders/balance-impact-risk.php) |
| IBKR says the official opening price may differ from the `09:30:00` opening print, and in volatile markets the official opening print may be delayed. | `direct_source_str` | [IBKR Market-on-Open order type](https://www.interactivebrokers.com/en/trading/orders/balance-impact-risk.php) |
| NYSE opening auctions are single-price auctions and the opening auction executes within auction collars. | `direct_source_str` | [NYSE auctions](https://www.nyse.com/trade/auctions), [NYSE American trading info](https://www.nyse.com/markets/nyse-mkt/trading-info) |
| Nasdaq describes the opening and closing crosses as price-discovery facilities that cross orders at a single price. | `direct_source_str` | [Nasdaq opening and closing crosses FAQ](https://nasdaqtrader.com/content/productsservices/trading/crosses/openclose_faqs.pdf) |
| IBKR may simulate some market orders with marketable limit orders to manage price risk. | `direct_source_str` | [IBKR simulated market orders](https://www.interactivebrokers.com/en/trading/simulated-market-orders.php) |
| High-turnover anomaly portfolios are much more vulnerable to trading costs than slower-turnover strategies. | `direct_source_str` | [AQR: Trading Costs of Asset Pricing Anomalies](https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs-of-Asset-Pricing-Anomalies), [Novy-Marx and Velikov, RFS](https://academic.oup.com/rfs/article/29/1/104/1844518) |
| Trading costs depend materially on implementation and capital scale; they should not be modeled as a universal fixed all-in number across all ticket sizes. | `house_inference_str` | Inference from the fee formulas above plus [AQR](https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs-of-Asset-Pricing-Anomalies) and [Novy-Marx and Velikov, RFS](https://academic.oup.com/rfs/article/29/1/104/1844518) |
| IBKR PaperTrader `MOO` fills should not be used as the primary calibrator for permanent production slippage assumptions. | `house_inference_str` | Inference from IBKR PaperTrader limitations, top-of-book simulation, lack of auction support, and the official-open caveat in the IBKR MOO documentation |

## Practical Interpretation

### 1. Fees and slippage should not be collapsed into one permanent fixed number

Reason:

```text
explicit_fee_bps_float
falls as order_notional_float rises
```

while:

```text
impact_bps_float
usually rises as participation_rate_float rises
```

So one all-in fixed number is convenient, but it mixes two effects that move in opposite directions.

### 2. PaperTrader is good for workflow testing, not for slippage calibration

This is the key practical point from the official broker documentation.

The external sources directly support:

- simulator, not live execution
- no clearing
- top-of-book fill simulation
- no deep book access
- no support for some auction order types

Therefore:

```text
paper_fill_gap_bps_float
!=
clean_live_opening_auction_slippage_bps_float
```

### 3. Opening-auction execution is conceptually different from regular-session spread crossing

For regular-session market orders, spread costs matter in the familiar bid/ask way.

For `MOO/OPG`, the correct benchmark is the auction execution price, not an arbitrary `09:30:00` print and not necessarily the prior close.

Therefore:

```text
opening_auction_benchmark_px_float
should be analyzed separately from
prior_close_reference_px_float
```

## Recommended House Model

### Baseline model

Use:

```text
explicit_fee_cash_float = broker_commission_cash_float + optional_regulatory_fee_cash_float
implicit_slippage_bps_float = fixed_asset_type_slippage_bps_float
```

Recommended defaults:

| Asset bucket | `fixed_asset_type_slippage_bps_float` | Type |
|---|---:|---|
| Liquid US stocks | `2.5` | `house_heuristic_str` |
| Broad liquid ETFs | `1.0` | `house_heuristic_str` |
| Other liquid ETFs | `2.0` | `house_heuristic_str` |

This is the model to use for normal research and backtest reporting.

### Capacity / scaling model

Keep the same exact fees and fixed slippage, then add one simple `%ADV` tier:

| `participation_rate_float` | `adv_addon_bps_float` | Type |
|---:|---:|---|
| `<= 0.10%` | `0` | `house_heuristic_str` |
| `0.10% - 0.50%` | `1` | `house_heuristic_str` |
| `0.50% - 1.00%` | `2` | `house_heuristic_str` |
| `1.00% - 5.00%` | `5` | `house_heuristic_str` |
| `> 5.00%` | `10` | `house_heuristic_str` |

Then:

```text
implicit_slippage_bps_float
=
fixed_asset_type_slippage_bps_float + adv_addon_bps_float
```

This is not a claim that the world behaves in these exact step functions. It is a deliberate simplicity choice for robust capacity testing.

## Why This House Model Is Reasonable

It is:

- simple enough to explain and audit
- anchored to exact broker fees where possible
- conservative without pretending to model microstructure exactly
- robust to the known weakness of PaperTrader as a fill-quality proxy
- sensitive to scale through a single `%ADV` control

It avoids:

- overfitting a slippage model to noisy paper fills
- pretending the same all-in bps number works for every account size
- forcing a complex market-impact model that the repo cannot truly calibrate yet

## Practical Rules For Future Use

When evaluating a strategy:

1. Report `gross`, `net_of_fees`, and `net_all_in`.
2. Keep fees explicit.
3. Use fixed slippage for baseline research.
4. Use the `%ADV` addon only for capacity / scale checks.
5. Do not calibrate permanent house slippage from IBKR PaperTrader `MOO` fills alone.

## Source List

Official and primary sources used:

- Interactive Brokers US stock commissions: <https://www.interactivebrokers.com/en/pricing/commissions-stocks.php>
- Interactive Brokers Paper Trading Account glossary: <https://www.interactivebrokers.com/campus/glossary-terms/paper-trading-account/>
- Interactive Brokers paper trading vs live trading lesson: <https://www.interactivebrokers.com/campus/trading-lessons/paper-trading-vs-live-trading-whats-the-difference/>
- Interactive Brokers Market-on-Open order type page: <https://www.interactivebrokers.com/en/trading/orders/balance-impact-risk.php>
- Interactive Brokers simulated market order handling: <https://www.interactivebrokers.com/en/trading/simulated-market-orders.php>
- SEC FY2025 Section 31 advisory: <https://www.sec.gov/rules-regulations/fee-rate-advisories/2025-2>
- SEC FY2026 Section 31 advisory: <https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2>
- FINRA fee adjustment schedule: <https://www.finra.org/rules-guidance/rule-filings/sr-finra-2024-019/fee-adjustment-schedule>
- NYSE auctions page: <https://www.nyse.com/trade/auctions>
- NYSE American trading info: <https://www.nyse.com/markets/nyse-mkt/trading-info>
- Nasdaq opening and closing crosses FAQ: <https://nasdaqtrader.com/content/productsservices/trading/crosses/openclose_faqs.pdf>
- AQR working paper: <https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs-of-Asset-Pricing-Anomalies>
- Novy-Marx and Velikov, Review of Financial Studies: <https://academic.oup.com/rfs/article/29/1/104/1844518>

Local repo files referenced:

- `alpha/engine/strategy.py`
- `alpha/live/ibkr_socket_client.py`
- `alpha/live/strategy_host.py`
- `alpha/live/state_store.py`

## Update Checklist

If this file is revisited in the future, re-check:

- IBKR commission page
- SEC Section 31 rate
- FINRA TAF rate
- any IBKR PaperTrader limitation changes
- the repo's local execution-quality reference convention

