# NYSE Opening Auction Imbalance — Deep Research

**Purpose:** Understand what the NYSE opening imbalance feed is, how it's delivered through IBKR, and how it could improve `live_reference_price` for your MOO workflow. No code changes — research only.

**Correction to prior note:** The correct IBKR generic tick ID is **225**, not 56. I misremembered earlier.

---

## 1. What your code does today

Verified in [alpha/live/ibkr_socket_client.py:243-257](alpha/live/ibkr_socket_client.py:243):

- Calls `ib_obj.reqTickers(*contracts)` with **no `genericTickList`**.
- Reads `ticker_obj.marketPrice()` per symbol.
- `marketPrice()` returns:
  - last trade price if it falls inside the current bid/ask spread, else
  - midpoint `(bid + ask) / 2`, else
  - last trade price if no bid/ask available.
- Snapshot timestamped `datetime.now(tz=UTC)`, no caching, fresh per VPlan build.

**What this means in the pre-open window:**
- Before 9:30 ET, the NYSE continuous book is not trading. `marketPrice()` falls back to whatever quote is available — usually stale post-close bid/ask from yesterday, or a sparse pre-market print, depending on whether a Nasdaq or ARCA venue has a live quote. For liquid S&P 500 names there's usually *something*, but it is **not a forecast of the opening auction price**.
- Auction-specific data (indicative match price, unmatched imbalance, paired volume) is **never requested** because no generic ticks are subscribed. The feed exists; your client simply isn't listening to it.

---

## 2. NYSE opening auction — how it actually works

This is the mechanism your MOO orders participate in. Understanding the timeline is the prerequisite to using the feed.

### 2.1 Timeline (all times ET)

| Time | Event |
|------|-------|
| 04:00 | Pre-market trading begins on NYSE Arca |
| 06:30 | MOO / LOO order entry eligible |
| **08:00** | **Opening imbalance messages begin publishing, every second** |
| 09:25 | "Enhanced" imbalance history data begins (what NYSE publishes graphically) |
| 09:28 | **Hard deadline**: cancel / cancel-and-replace of MOO / LOO orders rejected after this |
| 09:29:55 | **Imbalance Freeze** — no new MOO / LOO orders accepted; the book is locked |
| 09:30:00 | **Auction cross** — single-price clearing |

### 2.2 What the auction does

The Core Open Auction is a **single-price auction** that matches buy and sell orders at the **Indicative Match Price** — the price that maximizes the quantity of stock tradable within the Auction Collars (pre-set % bands around a reference price). Any unmatched MOO interest at the cross is filled at that single print.

### 2.3 What an imbalance message contains (per spec)

Each message carries:

- **Reference Price** — the pivot price around which the auction is evaluated.
- **Indicative Match Price** — "if we crossed the auction right now, this is the print."
- **Total Imbalance Qty** — unmatched volume at the reference price. Signed (buy-side vs sell-side).
- **Market Imbalance Qty** — unmatched MARKET orders (MOO), separate from limit-on-open (LOO) imbalance.
- **Paired Qty** — volume that WOULD execute at the indicative match price.
- **Book Clearing Price** / **Far Clearing Price** / **Auction Interest Clearing Price** — alternative clearings used by sophisticated models.
- **Auction Collar (upper / lower)** — price band limits.
- **Imbalance Freeze Indicator** — true after 9:29:55.
- **Auction Indicator** — auction type (opening / closing / reopening / IPO).

Update frequency: every second, only when something changes. No change → no message.

---

## 3. How this data reaches you via IBKR

### 3.1 Generic tick 225

To receive auction data, you call:

```python
ticker = ib.reqMktData(contract, genericTickList='225')
```

(Note: `reqMktData`, not `reqTickers`. `reqTickers` is a one-shot helper that doesn't pass `genericTickList` through cleanly — see §5.)

Generic tick 225 enables four underlying tick types:

| Tick ID | Field on `Ticker` (ib_insync) | Meaning |
|---------|-------------------------------|---------|
| 34 | `auctionVolume` | Shares that would trade if auction crossed now (≈ Paired Qty) |
| 35 | `auctionPrice` | Indicative match price |
| 36 | `auctionImbalance` | Unmatched shares, signed |
| 61 | (not exposed by ib_insync — see GH issue #404) | Regulatory imbalance |

So on `ib_insync.Ticker`, three auction fields are populated. The fourth (Regulatory Imbalance) would require a raw `ib_async` or `ibapi` subscription.

### 3.2 Subscription prerequisites

- Base subscription needed: **NYSE real-time** (for NYSE-listed) or **ARCA Book** (for Arca-listed). Non-pro accounts get this bundled with the $10/month waivable fee if you trade; pro accounts pay ~$45/month for NYSE Level 1.
- Whether generic tick 225 requires an **additional OpenBook subscription** is not perfectly documented and has changed over time. Operationally, the safe answer is: **subscribe to OpenBook on your IBKR account ($20-50/month depending on pro status) and the ticks will flow**. IBKR will silently return `NaN` on auction ticks if the subscription is missing — no error, no warning.
- Paper account: **opening imbalance data is usually NOT delivered on paper trading accounts**. Paper feeds are delayed/synthetic for most subscription-gated data. Plan on validating with a live account once you're past paper.

### 3.3 ib_insync mechanics

```python
from ib_insync import IB, Stock
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

contract = Stock('XOM', 'SMART', 'USD', primaryExchange='NYSE')
ticker = ib.reqMktData(contract, genericTickList='225', snapshot=False)

ib.sleep(2)   # give the feed time to populate

print(ticker.auctionPrice)       # indicative match
print(ticker.auctionImbalance)   # signed shares
print(ticker.auctionVolume)      # paired qty
```

Key detail: auction ticks arrive **as updates**. A snapshot call (`snapshot=True`) may not include them; you need a streaming subscription. The NYSE feed pushes a new message each second only when values change, so your `Ticker` object updates opportunistically.

---

## 4. How this helps your MOO workflow

### 4.1 Better `live_reference_price`

Today, at VPlan build time (morning tick before open), `marketPrice()` returns a stale post-close or sparse pre-market quote. If you're building the VPlan at 9:15 ET, the NYSE has already been publishing the indicative match price for 1 hour 15 minutes. **Between 9:20 and 9:28**, this price converges toward what actually prints at 9:30 — typically within 10-30 bps for liquid S&P 500 names.

Concrete benefit: overnight-gap drift between backtest-sized shares and live-sized shares shrinks because you're pricing against a live forecast of the opening print, not against a stale quote.

### 4.2 Fill-probability signal

The `auctionImbalance` field tells you whether the auction has excess buy or sell interest, signed. Uses:

- If you're buying 100 shares of XOM and `auctionImbalance = -50,000` (sell-side), your order will fill with high confidence and at-or-below the indicative match.
- If `auctionImbalance = +50,000` (buy-side) and you're also buying, the auction is competitive on your side. Still fills (MOOs are unconditional), but the indicative match may keep climbing until 9:28.
- If `auctionImbalance` is microscopic relative to `auctionVolume`, the auction is balanced and the match price will be stable.

At your scale (3-pod × 10 names × small notional), you are never a meaningful fraction of the auction imbalance. But **this data becomes essential later** at 7-figure per-name orders.

### 4.3 Auction-failure early warning

If the `Imbalance Freeze Indicator` shows the freeze at 9:29:55 but the indicative match is **outside the Auction Collar**, the auction may not cross and your MOO may be held until a continuous-book fill. Rare, but happens on news days (halts, circuit breakers). Having this field visible in state lets you detect and log these events rather than being surprised by a missing fill.

### 4.4 Post-trade TCA

Store per order:

- `reference_price_preopen` (now `marketPrice()`, later `auctionPrice` at VPlan build)
- `indicative_match_at_freeze` (`auctionPrice` at 9:29:55)
- `realized_fill_price` (from broker)
- `auction_imbalance_at_freeze` (size)

With these four fields, you can attribute slippage:

```
overnight_gap_drift     = indicative_at_freeze - reference_at_submit
auction_price_impact    = realized_fill - indicative_at_freeze
total_slippage          = realized_fill - reference_at_submit
```

This is the canonical TCA decomposition for auction orders. Over 3-6 months you build an empirical slippage distribution, which replaces the hardcoded `0.00025` in the backtest.

### 4.5 Size-aware order splitting (future)

At higher capital, if your buy-side share need is > some % of `auctionImbalance` or `auctionVolume`, you would split: part MOO (take the auction print), part VWAP into first 30 min (avoid moving the auction). The data enables the policy — implementation is later.

---

## 5. Practical notes & gotchas

### 5.1 `reqTickers` vs `reqMktData`

Your code uses `reqTickers` which is a convenience wrapper around `reqMktData` with `snapshot=True`. **Snapshot mode does not reliably deliver generic-tick data** — especially auction ticks which are stream-updates, not snapshot values. If you want auction data, you need streaming `reqMktData(..., genericTickList='225', snapshot=False)` held open long enough to receive at least one auction message (typically 1-2 seconds after subscription).

This is the single biggest "why isn't it working?" trap. The subscription returns immediately; the data arrives 1-2 seconds later via event callbacks. `ib_insync` handles this with `ib.sleep(2)` after the request, or by awaiting the ticker.

### 5.2 Delivery window

Imbalance data is published **08:00 – 09:30 ET for opening**. If your VPlan builds outside this window (e.g. the night before at 22:00 ET), you will get no auction data — the feed is silent. So the feed is only useful if VPlan build moves to morning. If you prefer to build overnight, you can't get a pre-auction price.

For your stack, this means: if you want to exploit auction data, `build_vplans` should be scheduled to fire **once** sometime between 09:10-09:25 ET, pick up auction data, then submit the MOO before 09:28. Right now your scheduler timing is driven by `pre_close_15m` / `eod_snapshot_ready` / `month_end_snapshot_ready` — none of which obviously map to "morning of the open." You'd need a new clock, say `pre_open_15m` = 09:15 ET.

### 5.3 Closing auction has its own feed

Symmetric mechanism: NYSE Closing Cross publishes imbalance 15:50-16:00 ET (15:45-16:00 starting around 2024 when rules changed). Same `genericTickList='225'` subscription delivers both — the data is tagged by auction type. If you ever add a `same_day_moc` execution policy (you have one in the enum at [alpha/live/models.py](alpha/live/models.py)), this becomes directly relevant. The `pre_close_15m` clock you already have is perfectly aligned for a closing-imbalance-aware VPlan build.

### 5.4 Nasdaq has its own

Nasdaq Opening Cross and Closing Cross publish imbalance via the **Net Order Imbalance Indicator (NOII)** feed. Different subscription (Nasdaq TotalView), different IBKR tick type delivery (generally the same `genericTickList='225'` on Nasdaq-listed symbols). Your S&P 500 universe includes both NYSE and Nasdaq listings, so in practice you'd need both data sources for full coverage. The mechanism is analogous but NYSE's opening imbalance is richer (more fields, longer pre-auction publication).

### 5.5 Paper account vs live account

Paper accounts usually deliver **delayed market data** (15 min) or **synthetic data**, not the real imbalance feed. This is an IBKR policy, not a TWS limit. Consequence: you cannot meaningfully validate the auction-data integration in paper. It must be tested on a funded live account with actual market data subscriptions active.

### 5.6 Cost

Rough order-of-magnitude (check current IBKR fee schedule — prices change):

- NYSE Level 1 (non-pro): ~$1-5/month depending on trading activity waiver
- NYSE OpenBook (non-pro): ~$20-30/month
- Nasdaq TotalView (non-pro): ~$30-40/month
- Pro rates are roughly 3-10× higher

Not a barrier at your scale but plan for ~$50-70/month to get full opening + closing imbalance coverage across both exchanges, non-pro.

### 5.7 The "it's quiet at 08:00" problem

Imbalance messages are only published when values change. Between 08:00 and 09:00 the data is often sparse and volatile — early orders can swing the indicative price wildly because book depth is thin. The **useful window is 09:15 – 09:28**, when real order flow is arriving and the indicative price is stabilizing. Treat data before 09:15 as indicative only, not a reference price.

---

## 6. The causality question — does using auction data break causality?

Short answer: **no**. The auction data is public, published by the exchange, available to all participants simultaneously. Using the indicative match at 09:20 to size an order you submit before 09:28 is entirely causal — you're using information available to you at decision time, just like you use prior-day close in the backtest.

The thing that would break causality is using the **realized 09:30 auction print** to size the order. That you cannot do — by the time that price exists, your order has already been queued.

So backtest parity concern: the backtest uses `prev_close` as the sizing price. The live system currently uses `marketPrice()` (stale quote). If you switch live to `auctionPrice` at 09:20-09:25, the live sizing price moves *away* from the backtest sizing price (prev_close), which could be seen as a bigger divergence.

**Counter-argument:** the backtest's use of `prev_close` is a *necessity*, not a *preference*. The backtest has no forward-looking auction feed; the closest causal proxy is prev_close. In live, you have a strictly better causal estimate of the actual fill price. Using the better estimate is improving live, not breaking it — but it does widen the gap between backtest-sized shares and live-sized shares.

The right framing: **backtest and live will always disagree by roughly the overnight gap.** Using `auctionPrice` reduces that disagreement because `auctionPrice` converges to the realized open. Using `marketPrice()` (stale quote) leaves the full overnight gap unexploited. Either way, a parity harness (see architecture review §7.2) that asserts share-count equality within a tolerance absorbs this correctly.

---

## 7. What this unlocks long-term

Beyond the immediate "better reference price" benefit, once your system records auction data per order, you get:

1. **Empirical slippage model** — regress realized fill - auction price at freeze against (order size / auction volume), recover a real market-impact curve. Replaces the backtest's constant-slippage assumption.
2. **Auction timing analysis** — what's the mean convergence of indicative price from 09:15 to 09:30? Does your strategy's universe have names where the auction is noisy and a limit-on-open would be safer?
3. **Imbalance as a signal input** — pure research, but academic literature (Stoikov, Cartea) shows opening imbalance has short-term predictive value. At your scale this is overkill; at $10M+ with intraday strategies, it becomes a real edge source.
4. **Fail-fast on auction halts** — news halts freeze the auction with freeze indicator = true and imbalance stays huge. Your reconciler can detect this and log a critical event rather than discovering a missing fill 10 minutes later.

---

## 8. Summary — what you actually learned

1. Your code today requests **zero auction data**. `marketPrice()` is a stale/continuous-book fallback, not an auction-aware reference.
2. The correct IBKR generic tick ID is **225**, unlocking `auctionPrice`, `auctionVolume`, `auctionImbalance` on the ib_insync `Ticker`.
3. NYSE opening imbalance is **live from 08:00 ET, reliable from 09:15 ET, frozen at 09:29:55, crossed at 09:30:00**.
4. Using it as `live_reference_price` improves the sizing price from "stale overnight quote" to "exchange-published forecast of the actual fill price" — materially tighter, still causal.
5. The subscription path is OpenBook (~$20-30/month non-pro). Paper accounts do not deliver this data reliably; plan to validate on a funded live account.
6. The integration is minor: `reqMktData(contract, '225', snapshot=False)` + `ib.sleep(2)` + read `ticker.auctionPrice`. The orchestration change is bigger: VPlan build must move to the 09:15-09:25 ET window, which means a new scheduler clock.
7. The real long-term value isn't the better reference price — it's the **data capture for TCA**. Recording `auction_price_at_submit`, `auction_imbalance_at_freeze`, `realized_fill` per order builds the empirical slippage distribution that lets you replace the backtest's constant-slippage assumption with a calibrated model.

---

## Sources

- [IBKR TWS API — Available Tick Types](https://interactivebrokers.github.io/tws-api/tick_types.html)
- [IBKR TWS API — Requesting Watchlist Data (reqMktData)](https://interactivebrokers.github.io/tws-api/md_request.html)
- [Databento — Introducing real-time NYSE imbalance data](https://databento.com/blog/NYSE-imbalance-feeds)
- [NYSE — Opening and Closing Auctions Fact Sheet (PDF)](https://www.nyse.com/publicdocs/nyse/markets/nyse/NYSE_Opening_and_Closing_Auctions_Fact_Sheet.pdf)
- [NYSE — XDP Imbalances Feed Client Specification (PDF)](https://www.nyse.com/publicdocs/nyse/data/XDP_Imbalances_Feed_Client_Specification_v2.2a.pdf)
- [NYSE — Opening Auction Tool Enhancement, 2024](https://www.nyse.com/data-insights/nyse-introduces-the-enhanced-nyse-auction-tool-with-opening-imbalance-history)
- [ib_insync — API docs](https://ib-insync.readthedocs.io/api.html)
- [ib_insync GitHub issue #404 — missing regulatoryImbalance attribute](https://github.com/erdewit/ib_insync/issues/404)
- [IBKR — Market Data Subscription Pricing](https://www.interactivebrokers.com/en/pricing/market-data-pricing.php)
- [IBKR Campus — NYSE OpenBook glossary](https://www.interactivebrokers.com/campus/glossary-terms/nyse-openbook/)
- [IBKR Campus — Market Data: Professional vs Non-Professional](https://www.ibkrguides.com/kb/en-us/market-data-prof-vs-non-prof-usage.htm)
