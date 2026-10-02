# Scout P7: the point-in-time stock pods (DV2 S&P 500, DV2 Nasdaq-100, HPI)

> **Correction (2026-10-02, size ladder):** DV2's S3 headline below (+9.1 bp, t 3.96) includes 1998-2003. From 2004,
> the pod's own window, it is +3.8 bp, t 1.54, placebo p 0.16; by era +18.2 bp (1998-2007), +5.4 bp (2008-2015),
> +0.8 bp (2016-2022). See `SCOUT_DV2_SIZE_LADDER_20261002.md`. The grades are unchanged.

Date: 2026-10-02. Research only; nothing here changes live trading.
- **Script:** `scripts/research/scout_p7_reaudition_20261002/run_p7.py`.
- **Cards:** `results/scout/cards/` (not in git).
- **Ledger:** four RETRO registrations (`dv2`, `dv2_ndx`, `hpi_vote`, `hpi_ibs_rsi` `_reaudition_20261002`), written
  before any Scout result. Prior trials counted: DV2 200, HPI 100.

**Gates.** All four pods pass the exact identity gate against a fresh engine run (largest daily return difference
4.4e-16; same trade dates; 5,723 weight days for DV2 S&P 500):
- **DV2 S&P 500** (WIRED) and **DV2 Nasdaq-100** (variant): `alpha/scout/specs/dv2.py`.
- **HPI 2/3/5 vote** (WIRED, live account) and **HPI IBS RSI exit**: `alpha/scout/specs/hpi.py`. The weights engine
  gained one opt-in, `missing_open_hold_df` (a held member with no open is kept, marked at the close).
- All earlier gated specs still pass after the merge.
- **The saved-run gate tests:** after the 2026-10-02 nightly Norgate revision, the two HPI tests that compare with the
  engine runs saved at 2026-10-01 23:18 fail (5.5e-5 from 2024-06-11). `gate hpi_vote --fresh` passes at 4.4e-16, so
  the saved runs are stale, not Scout. Re-saving the engine runs clears them.

**MCPT kind "panel".** The per-asset null of P4b (A8) on the sealed Scout panel: one global date permutation
followed per stock and membership stratum, liquidity left on real dates. Score: the Sharpe of the daily active
return over the equal-weight members, after the plateau choice of the whole grid. Calibrated for short-term
reversal at 2.0% false passes. HPI runs it with 6 workers instead of 14 (each worker holds a full panel and HPI's
five-year features; 14 ran out of memory).

## Verdict

Each pod is tested in NDX VXN's slot of the live 60/40 book, as in P6b and P6c.

| Pod | Grade | MCPT p | Alpha t (fullest, net) | Book Sharpe with it vs T-bills | Slot P | DSR p (warn) | Capacity |
|---|---|---|---|---|---|---|---|
| **DV2 S&P 500** (WIRED) | WATCHLIST (alpha t) | **0.001** | 1.80 (+7.8%/yr) | 1.37 vs 1.21 | 0.85 | **0.004** | $11.3M |
| **DV2 Nasdaq-100** | WATCHLIST (S3 placebo, DSR) | **0.001** | **2.55** (+9.2%/yr) | 1.36 vs 1.21 | 0.91 | 0.406 | $28.9M |
| **HPI 2/3/5 vote** (WIRED, live account) | WATCHLIST (S3 weak) | 0.021 | **2.64** (+9.6%/yr) | 1.37 vs 1.21 | 0.90 | **0.000** | $12.0M |
| HPI IBS RSI exit | WATCHLIST (S3 weak) | 0.032 | 2.27 (+7.4%/yr) | 1.34 vs 1.21 | 0.86 | **0.001** | $12.8M |

## What this means

**1. DV2's edge is real; what is unproven is that it adds to the book beyond its factors.**
- **The evidence for:** the search beats 999 of 1,000 per-asset shuffles in both universes. The S&P 500 pod's DSR
  passes with 200 prior trials (p 0.004). S3: the raw entry signal earns +9.1 bp over the same-date members at three
  days (S&P 500: t 3.96, placebo p 0.005, 120,965 events). Walk-forward: out-of-sample Sharpe 0.72, 100% of designs
  positive.
- **The evidence against:**
  - **Costs:** the event excess covers about 0.91 of the round trip (S3 WARN in both universes). At twice the costs
    plus 10 bp the S&P 500 pod keeps a Sharpe of 0.26 and the Nasdaq-100 one 0.04.
  - **Independent alpha:** after SPY, QQQ, IEF, GLD, trend and TAA 3x, the S&P 500 pod's alpha is +7.8%/yr at t 1.80,
    short of 2. The long-only pod also earns the market's return in the regimes it trades.
- **In the book:** either DV2 pod in NDX VXN's slot lifts the live book's Sharpe from 1.21 to about 1.37 (P 0.85 and
  0.91). That is close to EOM's slot result (1.46, P 1.00) and above CORE5's (P 0.85).

**2. S&P 500 or Nasdaq-100?** (the owner asked for the comparison)

| | DV2 S&P 500 | DV2 Nasdaq-100 |
|---|---|---|
| S3 event edge (h 3) | +9.1 bp, t 3.96, placebo p 0.005, 120,965 events | +9.1 bp, t 2.37, **placebo p 0.368**, 23,588 events; positive years < 60% |
| DSR (prior trials 200) | **p 0.004** | p 0.406 |
| PBO (diagnostic) | 0.22 | 0.62 |
| Stress Sharpe (2x costs + 10 bp) | 0.26 | 0.04 |
| Net alpha t | 1.80 | **2.55** |
| Slot P / book Sharpe | 0.85 / 1.37 | 0.91 / 1.36 |
| Capacity (1% ADV) | $11.3M | $28.9M |

- **The reading:** the S&P 500 version has the stronger statistical case. Its signal is broad (five times the events),
  survives the placebo, and its DSR passes. The Nasdaq-100 version shows more alpha after factors and more capacity,
  but on a fifth of the events: its placebo fails, its DSR does not pass, and its in-sample best configuration rarely
  stays best (PBO 0.62).
- **The universe was chosen after results:** P4b's S3 found the signal stronger in the Nasdaq-100 before this run, so
  its registration says so (`universe_chosen_after_results_bool`). That is one more reason its higher alpha should be
  read with care.
- **Recommendation:** keep the S&P 500 pod as the DV2 pod of record. The Nasdaq-100 variant stays WATCHLIST; it is
  not a replacement.

**3. HPI: the strongest whole rule of the three, and the weakest entry signal.**
- **The whole rule passes everything:** MCPT p 0.021 (vote) and 0.032 (IBS RSI), below the 0.05 bar, calibrated at
  2.0% false passes. DSR p 0.000 with 100 prior trials. Net alpha +9.6%/yr at t 2.64 (vote) after SPY, QQQ, IEF, GLD,
  trend and TAA 3x. Stress Sharpe 0.57 at twice the costs plus 10 bp, the most cost-robust stock pod audited.
  Walk-forward out-of-sample Sharpe 0.97 (vote), 100% of designs positive.
- **The entry event alone barely beats the same-date members:** +3.6 bp at five days, t 0.79, placebo p 0.36,
  covering a third of costs. These are soft S3 criteria, so the grade is WATCHLIST, not REJECTED.
- **The reading, as with the sector IBS pods in P6c:** HPI does not earn its money by picking the right pullback
  name. It earns it from being invested in uptrending large caps right after pullbacks, and from its exit and slot
  rules. The selection claim is unproven; the rule's value is real in sample.
- **PBO 0.62-0.71 (diagnostic):** the grid's in-sample best rarely stays best, but the grid is a plateau
  (ratio 0.95-0.96). Any nearby configuration does about as well, so the live one is not a lucky peak.
- **The vote and the IBS RSI exit are one strategy** (daily correlation 0.95). The vote is slightly better on every
  row.

**4. DV2 and HPI are largely the same bet.** In-sample daily return correlations (2005 to 2022-12-30, live
configurations):

| | DV2 S&P 500 | DV2 Nasdaq-100 | HPI vote | TAA 3x | NDX VXN |
|---|---|---|---|---|---|
| DV2 S&P 500 | 1 | 0.63 | 0.75 | 0.37 | 0.52 |
| DV2 Nasdaq-100 | | 1 | 0.57 | 0.48 | 0.65 |
| HPI vote | | | 1 | 0.38 | 0.52 |

- Each lifts the live book from 1.21 to about 1.37 in NDX VXN's slot. Two of them in the book would mostly double one
  short-term reversal exposure in uptrending large caps.
- **If the book holds one stock reversal pod, HPI vote has the better case:** more alpha after factors (t 2.64 vs
  1.80), twice the stress Sharpe, the same book lift. Its weaker MCPT (0.021 vs 0.001) is still a pass. DV2 has the
  stronger entry signal (S3 t 3.96), so the two share an edge: HPI's exit and slot rules turn it into more money.
- HPI vote already trades in the live account. Scout does not ask for a change. It records that the pod is
  WATCHLIST for one reason only (its entry event is not a stock-picking edge) and that no Scout evidence argues
  against it.

## Findings outside the verdict

- **The DV2 Nasdaq-100 strategy module is an empty stub.** `strategies/dv2/strategy_mr_dv2_nasdaq100.py` has
  star-imported itself since the 2026-04-04 refactor (bf1a334), so it defines nothing and has no `run_variant`. The gate's
  engine side is its last implementation (300ca70), restored verbatim into `alpha/scout/gate/legacy_dv2_ndx.py` and run
  on the real engine. A follow-up task to restore the module was offered separately; when it lands, the gate entry
  points back at it and the legacy file is deleted.
- **A gate false alarm from the nightly Norgate update.** A `--fresh` DV2 gate failed once (5.5e-5) after the HPI
  merge. The HPI engine change was checked to be opt-in; the cause was the Norgate database updating at 00:55 during
  the run, so the engine and Scout read different prices. The re-run passed (4.4e-16). Rule: a fresh gate run should
  not overlap the nightly Norgate update.
- **New deviations:**
  - `float32_momentum_threshold` (DV2): Norgate closes are float32, so the momentum bar is tested in float32. S&P 500:
    0 of 21,452 trades change. Nasdaq-100: 6 of 13,530 trades change, +3.3 bp/yr. Immaterial.
  - `hpi_open_known_slot_refill` (HPI): the backtest frees a slot only when the held name's next open prints; the live
    host assumes it does. Zero sessions differ on 2004-2026. Immaterial.

## Engine and runner changes

- **`engines/weights.py`:** opt-in `missing_open_hold_df` (HPI). Every gated spec is unchanged.
- **`reaudit.py`:** MCPT kind "panel" for point-in-time stock pods, with a per-pod worker count.
- **New specs:** `specs/dv2.py` (S&P 500 and Nasdaq-100) and `specs/hpi.py` (vote and IBS RSI exit), each with a
  fast panel replica for the MCPT and S3 event inputs.
