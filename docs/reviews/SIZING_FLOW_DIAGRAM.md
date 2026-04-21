# Sizing Flow — Backtest vs Live

Three views of the same thing, each for a different way of thinking about it.

---

## View 1 — Timeline (when is each quantity sampled?)

Time flows left → right. Boxes show *when* each input is frozen.

```
                t (close)              overnight             pre-submit window          t+1 open
                ═════════              ═════════             ═════════════════          ════════

                ┌─────────────┐                                                         ┌──────┐
   BACKTEST     │ decide here │─────────────────────────────────────────────────────────▶ FILL │
   incremental  │ entry_value │                                                         │ open │
                │ prev_close  │                                                         └──────┘
                │ prev_total  │
                └─────────────┘
                   sizing uses: prev_total_value, prev_close

                ┌─────────────┐                              ┌──────────────────┐       ┌──────┐
   LIVE         │ decide here │                              │ size here        │       │ MOO  │
   incremental  │ entry_value │──── freeze as weight ────────▶ pod_budget =     │───────▶ open │
                │ prev_close  │     entry_value /            │   net_liq × frac │       │ fill │
                │ prev_total  │     prev_total_value         │ live_reference   │       └──────┘
                └─────────────┘                              │ shares = ...     │
                                                             └──────────────────┘
                   sizing uses: sub_account.net_liq, live_reference (pre_close_15m)

                ┌─────────────┐                                                         ┌──────┐
   BACKTEST     │ decide here │─────────────────────────────────────────────────────────▶ FILL │
   full-weight  │ weight=0.10 │                                                         │ open │
                │ prev_close  │                                                         └──────┘
                │ prev_total  │
                └─────────────┘
                   sizing uses: prev_total_value, prev_close

                ┌─────────────┐                              ┌──────────────────┐       ┌──────┐
   LIVE         │ decide here │                              │ size here        │       │ MOO  │
   full-weight  │ weight=0.10 │──── freeze as weight ────────▶ pod_budget =     │───────▶ open │
                │ prev_close  │     (already a weight)       │   net_liq × frac │       │ fill │
                │ prev_total  │                              │ live_reference   │       └──────┘
                └─────────────┘                              │ shares = ...     │
                                                             └──────────────────┘
                   sizing uses: sub_account.net_liq, live_reference (pre_close_15m)
```

**Takeaway:** Backtest freezes budget + price at `t`. Live freezes *intent* at `t`, but samples budget + price near the open. That's the core timing divergence.

---

## View 2 — The Divergence Map (what exactly differs?)

Every row is a quantity. Columns = the four paths. Colors show where divergence lives.

```
                           BACKTEST                     LIVE
                    ┌──────────────┬──────────────┬──────────────┬──────────────┐
                    │ Incremental  │ Full-Weight  │ Incremental  │ Full-Weight  │
┌───────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Intent emitted    │ entry_value  │ target_weight│ entry_value  │ target_weight│   ← SAME
│                   │   ($)        │   (fraction) │   ($)        │   (fraction) │
├───────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Budget base       │ prev_total_  │ prev_total_  │ net_liq ×    │ net_liq ×    │   ← DIFFERS
│                   │ value        │ value        │ fraction     │ fraction     │      (timing + source)
├───────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Reference price   │ prev_close   │ prev_close   │ live_ref     │ live_ref     │   ← DIFFERS
│                   │   (at t)     │   (at t)     │ (pre_close15)│ (pre_close15)│      (timing)
├───────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Dollar target     │ entry_value  │ weight ×     │ entry_       │ weight ×     │
│ (target_dollar)   │ directly     │ prev_total   │ weight ×     │ pod_budget   │
│                   │              │              │ pod_budget   │              │
├───────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Shares formula    │ floor(       │ floor(       │ floor(       │ floor(       │
│                   │  $ /         │  $ /         │  $ /         │  $ /         │
│                   │  prev_close) │  prev_close) │  live_ref)   │  live_ref)   │
├───────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Order type        │ value order  │ target-based │ MOO delta    │ MOO delta    │
├───────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Fill price        │ open_{t+1}   │ open_{t+1}   │ MOO auction  │ MOO auction  │   ← SAME (approx)
│                   │ × slippage   │ × slippage   │ at open_{t+1}│ at open_{t+1}│
└───────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

                    ↑                                ↑
                    │                                │
           Intent is the same.             Two quantities are re-sampled:
                                           - budget base (prev_total → net_liq × frac)
                                           - reference price (prev_close → live_ref)
```

---

## View 3 — A Concrete Example (XOM, target 10%)

Assume: strategy wants 10% of pod in XOM. Pod just closed day at `prev_total_value = $100,000`. `prev_close = $95`. Next morning gaps up to `live_reference = $97`. Sub-account NLV = $100,000. Fraction = 1.0.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    Full-Weight path, concrete numbers                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   Intent:  target_weight = 0.10                                                │
│                                                                                │
│   BACKTEST                             LIVE                                    │
│   ──────────                           ──────────                              │
│   budget  = prev_total  = $100,000     budget  = NLV × frac = $100,000 × 1.0   │
│                                                = $100,000                      │
│                                                                                │
│   dollar  = 0.10 × $100,000            dollar  = 0.10 × $100,000               │
│           = $10,000                            = $10,000                       │
│                                                                                │
│   price   = prev_close  = $95          price   = live_ref    = $97             │
│                                                                                │
│   shares  = floor(10000/95)            shares  = floor(10000/97)               │
│           = floor(105.26)                      = floor(103.09)                 │
│           = 105 shares                         = 103 shares                    │
│                                                                                │
│   fill    = open_{t+1} × slippage      fill    = MOO auction at open_{t+1}     │
│           (simulator)                           (real)                         │
│                                                                                │
│   notional≈ 105 × open_{t+1}           notional≈ 103 × open_{t+1}              │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

   Share count drift this day:  105 vs 103  = 2 shares  ≈ 1.9% of notional

   Source of drift:  the overnight gap from $95 → $97  (+2.1%)

                   ┌─────────────────────────────────────────────┐
                   │ Rule of thumb:                              │
                   │                                             │
                   │   |share_drift|  ≈  |overnight_gap|         │
                   │                                             │
                   │ Median S&P500 |overnight gap| ≈ 30–50 bps   │
                   │ Tail (earnings, macro)        up to 2–5%    │
                   └─────────────────────────────────────────────┘
```

---

## View 4 — Shortest Mental Model

```
INCREMENTAL BACKTEST:
  signal ──▶ entry_value ──▶ prev_close ──▶ shares ──▶ OPEN fill

INCREMENTAL LIVE:
  signal ──▶ entry_value
             ↓ freeze
             entry_target_weight ──▶ pod_budget ──▶ live_ref ──▶ shares ──▶ MOO OPEN fill

WEIGHT BACKTEST:
  signal ──▶ target_weight ──▶ prev_total ──▶ prev_close ──▶ shares ──▶ OPEN fill

WEIGHT LIVE:
  signal ──▶ target_weight ──▶ pod_budget ──▶ live_ref ──▶ shares ──▶ MOO OPEN fill
```

**One-line summary:** Backtest sizes at close. Live sizes at the open-window sample. Same intent, different sampling clock → systematic drift equal to the overnight gap.

---

## Code Trace

- Backtest sizing: [alpha/engine/strategy.py:277-293](../../alpha/engine/strategy.py) (`_get_order_sizing_price_float`)
- Live weight freeze: [alpha/live/strategy_host.py:188-213](../../alpha/live/strategy_host.py)
- Live sizing: [alpha/live/execution_engine.py:51-73](../../alpha/live/execution_engine.py)
- Live orchestration: [alpha/live/runner.py:1155-1247](../../alpha/live/runner.py)
