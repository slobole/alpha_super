# Scout: do limit orders make DV2 cheaper to trade?

Date: 2026-10-02. Research only; nothing here changes live trading.
- **Scripts:** `scripts/research/scout_dv2_limit_entry_20261002/`.
- **Results:** `results/scout/dv2_limit_entry/` (not in git).
- **Ledger:** `dv2_limit_entry_20261002`, registered before any result. Grid of 50 (5 universes × 5 entries × 2
  exits), 262 prior trials. The universes were chosen after results.

**The question.** The size ladder showed that DV2's gross edge is largest in small caps (Russell 2000 lower half
+15.4 bp, t 4.2), but a market-on-open round trip there costs about 78 bp. Can passive limit orders save or earn the
spread?

## The model (daily bars, made conservative)

- **Entry:** after Close_T, a day limit buy at L = Close_T × (1 − k × NATR14_T), with k ∈ {0, 0.25, 0.5, 1}, against
  today's market-on-open.
  - **Open ≤ L:** fills at the Open and pays the half-spread.
  - **Trade-through, Low ≤ L × (1 − m):** fills at L with no spread. The margin m is the larger of one tick and 0.1 of
    the half-spread, so a touch is not a fill.
  - **Otherwise:** no fill. The slot stays empty, and the name is re-ordered only if its signal still holds.
- **Exit:** today's market-on-open, or a sell limit at the previous close (market-on-open after 5 days).
- **Costs:** commissions on nominal shares. Two half-spread models are charged on marketable fills only, and they
  bracket the truth: AR overcharges large caps, pooled undercharges small caps.
- **Check:** with market-on-open orders, the book equals the size ladder's costed replica exactly.

## Results, 2004-2022 (net Sharpe, AR / pooled)

| Universe | Market-on-open (today) | k 0.25, limit exit | k 0.5, limit exit | k 1, limit exit | k 0.5 with strict fills (f = 1.0) |
|---|---|---|---|---|---|
| **S&P 500** | 0.26 / 0.77 | 0.62 / 0.79 | **0.75 / 0.86** | 0.81 / 0.89 | 0.53 / 0.64 |
| S&P 100 | 0.28 / 0.71 | 0.67 / 0.81 | 0.78 / 0.87 | 0.73 / 0.79 | |
| Russell 1000 | −0.03 / 0.44 | 0.53 / 0.67 | 0.69 / 0.80 | 0.79 / 0.86 | 0.61 / 0.68 (k 1) |
| Russell 2000 lower half | ruined | 0.21 / 0.22 | 0.39 / 0.39 | 0.68 / 0.68 (k 1, moo exit 0.78 / 0.75) | 0.02 / 0.02 |
| SmallCap 600 | −0.39 / −0.18 | 0.21 / 0.26 | 0.33 / 0.37 | 0.46 / 0.49 | 0.06 / 0.10 |

**S&P 500 detail** (pooled cost model):

| Entry / exit | Fill rate | Trades/yr | CAGR | Max DD | Round trip (AR / pooled) |
|---|---|---|---|---|---|
| Market-on-open / market-on-open (today) | 1.00 | 464 | 14.8% | −32% | 37 / 13 bp |
| k 0.5 / limit | 0.38 | 264 | 13.8% | −33% | 14 / 6 bp |
| k 1 / limit | 0.13 | 143 | 10.9% | −18% | 13 / 6 bp |

## What this means

1. **Large caps: limit orders help, and the gain is robust.**
   - On the S&P 500 the cost per round trip falls by about 60%.
   - Net Sharpe under the strict cost model rises from 0.26 to 0.75 (k 0.5). Under the lenient one it moves from
     0.77 to 0.86.
   - With strict fills (the price must move the mid through the limit), k 0.5 still gives 0.53 / 0.64 against
     0.26 / 0.77 for market-on-open.
   - It holds in every era (S&P 500, k 1 / limit, pooled: 1.08 / 0.79 / 0.91).
   - **Adverse selection:** after the fill day, filled names continue as well as unfilled ones at k ≤ 0.5 (S&P 500
     +5.4 vs +3.0 bp). Only at k 1 do filled small-cap names do worse, by about 5-7 bp, at small t.
   - **The cost:** fewer trades and an under-invested book (about 3.5 of 10 slots at k 1). So CAGR falls (14.8% to
     13.8% at k 0.5) while Sharpe rises. A shallow limit (k 0.25-0.5) is the balance.
2. **Small caps: deep limits turn DV2 net-positive on paper, but it is not a pod.**
   - **The case for:** the Russell 2000 lower half goes from ruined to Sharpe about 0.7 at k 1. The whole-grid MCPT
     passes (p 0.005) for both small-cap universes.
   - **The case against:**
     - **The edge has faded:** in 2016-2022 every small-cap cell earns about zero (Sharpe ≤ 0.1), the same decay
       the size ladder found.
     - **Fill optimism:** under strict fills, 40-95% of the gain disappears (lower half k 0.5: 0.39 → 0.02).
       Small-cap daily lows are often a single small print, which daily bars cannot separate from a real fill. The
       MCPT shares the fill model, so it cannot test this.
     - **Peers:** SmallCap 600 still loses to its equal-weight members in every cell. Its MCPT "passes" with a
       negative active Sharpe, because the long-only reversal null is centred far below zero.
     - **Capacity:** $0.07-0.24M, and that is optimistic for passive fills.
3. **Only real fills can settle the fill model.** The open question is how often a resting order really fills when
   the daily low trades through it. Neither intraday nor quote data is available here; paper fills can answer it.

## Recommendation

- **DV2 S&P 500 (and HPI by the same logic):** a shallow limit entry (k 0.25-0.5) with a limit exit is the best
  evidence-based change Scout has found for the stock reversal pods. It is a live-execution change, so it is the
  owner's decision.
  - **The safe first step** is a paper shadow next to today's orders. It records the real fill rate and the price
    against the model's, for a few months.
  - **Promote it** only if the real fill rate and slippage are close to this model's (the S&P 500 at k 0.5: 38% fills,
    6-14 bp round trip).
- **Small caps:** not a pod. Revisit only with intraday data, and only for a small account.

## Known optimism in the fill model

The most likely issues come first.
1. **Full fills:** a trade-through fills the whole order, with no queue position and no partial fills.
2. **Phantom lows:** a bad print can make a low; strict fills show how sensitive the result is.
3. **The open:** any size fills at the open, even when the open equals the limit; a micro-cap "open" may be a first
   print rather than the cross.
4. **Volume:** no fill is capped by its share of volume.
5. **Delistings:** a delisting exits at the last close with no spread.

The fills are identical under both cost models (the margin uses the larger spread), so the AR/pooled bracket isolates
cost, not fills.
