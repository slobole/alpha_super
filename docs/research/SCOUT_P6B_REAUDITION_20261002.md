# Scout P6b: the macro and allocation pods re-audited

Date: 2026-10-02. Research only; nothing here changes live trading.
- **Script:** `scripts/research/scout_p6b_reaudition_20261002/run_p6b.py`.
- **Runner:** `alpha/scout/reaudit.py`, with MCPT kind "spec": each spec module carries its own validated fast
  replica.
- **Cards:** `results/scout/cards/` (not in git).
- **Ledger:** five RETRO registrations.

**Gates.** Three agents specified the pods. Every pod passes the exact identity gate against a fresh engine run
(differences of 2e-16 to 4e-16):
- **Compass** (both variants);
- **CORE5:** the weights engine gained opt-in shorts and a borrow fee; the borrow fees agree to the cent;
- **TFI**, which needed no internet: its FRED files are hash-locked in the repo;
- **Trinity:** the weights engine gained a `decision_fn` hook for its no-trade band.

All 15 gated specs were re-gated after the merges.

**The question for these pods.** Each is tested in NDX VXN's slot of the live 60/40 book, the slot P5 and P6a found
adds nothing. Is a macro pod a better use of that slot than T-bills?

## Verdict

| Pod | Family | Grade | Sharpe, live / median day | MCPT p | Alpha t (fullest, net) | Book Sharpe with it vs T-bills | Slot P | Capacity |
|---|---|---|---|---|---|---|---|---|
| **CORE5** | trend | **CANDIDATE** | 1.05 / 1.05 | 0.048 | 2.08 | **1.26 vs 1.21** | **0.85** | $1.8M (DBC) |
| Compass QQQ | macro regime | WATCHLIST (alpha t) | 1.12 / 0.98 | **0.003** | 1.55 | **1.33 vs 1.21** | **0.81** | $11.5M |
| Compass | macro regime | WATCHLIST (alpha t, slot) | 1.07 / 0.93 | **0.005** | 1.40 | 1.31 vs 1.21 | 0.77 | $10.6M |
| TFI | macro regime | WATCHLIST (MCPT, alpha, slot) | 0.64 / 0.60 | 0.080 | 0.65 | 1.23 vs 1.21 | 0.74 | large |
| Trinity | low-risk allocation | WATCHLIST (alpha, slot) | 0.82 / 0.83 | 0.033 | −0.24 | 1.14 vs 1.21 | 0.17 | $92M |

## What this means

1. **CORE5 is the first macro pod to pass every gate.**
   - **Strengths:** the MCPT passes (just), there is alpha after the ETF factors, trend and TAA 3x, and it improves
     the book in the slot (P 0.85).
   - **Its character:** a low-risk trend book. CAGR 6.2%, max drawdown −6.8%, Sharpe about 1.05 on every decision
     day (the luck band is flat: the decision day doesn't matter).
   - **Its limits:** capacity is about $1.8M, because the DBC short binds. The MCPT p (0.048) is close to the bar.
2. **Compass has real timing, but most of its return is exposure the factors explain.**
   - **Timing:** both variants pass the MCPT strongly (p 0.003-0.005). The DSR stays significant after counting
     the 826 trials of the 2026-09-28 study.
   - **Every period positive:** Sharpe 1.62 / 0.42 / 1.25 / 0.97 / 1.40 over 2003-07 / 2008-12 / 2013-17 /
     2018-21 / 2022.
   - **But:** after SPY, QQQ, IEF, GLD, trend and TAA 3x, its alpha is not significant (t 1.40-1.55).
   - **For the book:** the QQQ variant lifts the live book's Sharpe from 1.21 to 1.33 (P 0.81, a pass); the
     original reaches 1.31 (P 0.77, just under the bar).
   - **Reconciling with the 2026-09-28 study** ("edge in 2003-07 and 2022, lucky peak, holdout fail"):
     - Scout finds no lucky peak in its own grid (plateau 0.95).
     - The return holds in every era.
     - What the study called the edge, the value beyond the alternatives, is weak here too.
     - **Read Compass as a good diversifying sleeve whose independent alpha is not proven**, rather than as an edge.
3. **TFI and Trinity do not earn a slot.**
   - **TFI:** misses the gate (p 0.08); its alpha is t 0.65. Its very high Sharpe in some eras comes from mostly
     holding BIL; near-zero volatility makes Sharpe meaningless there.
   - **Trinity:** passes the MCPT, but its alpha is zero and the book is better with T-bills in its slot (P 0.17).
     Its 2022 (Sharpe −1.5, stocks and bonds falling together) is the risk-parity weakness.

## For the open slot (owner decision; no live change made)

The slot P5 and P6a left open now has evidence-based candidates:
- **CORE5:** passes everything; the most defensive.
- **Compass QQQ:** the largest book improvement, but no proven independent alpha.

A combination is plausible, and so is keeping T-bills. Scout's role ends at the evidence; the allocation is a book
decision. The 2026-09-30 defensive study ("CORE5 + BTAL_QQQ + DV2-IND") and these cards should be read together.

## Engine and spec changes

- **`engines/weights.py`:**
  - `allow_short_bool`: shorts; without it a negative weight raises instead of being dropped silently.
  - `split_sign_flip_bool`.
  - `BorrowModel`.
  - `decision_fn`: path-dependent rules. Its decided rows replay exactly through `simulate`.
- **New deviations:**
  - **T5YIE evening publication** (Compass): reading one session later changes 6 of 282 decisions, and the Sharpe
    rises slightly (not flattering).
  - **T5YIE current vintage** (G-027, Compass).
  - **TFI current-vintage FRED:** the engine's point-in-time mode gives Sharpe 0.694-0.711 against the frozen
    0.694 (not flattering).
  - **Split-adjusted units:** CORE5 at −0.013 pp a year (optimistic, BIL's 2017 consolidation); Trinity measured,
    Compass and TFI immaterial.
- **Family decisions:**
  - CORE5 counts in `time_series_trend_and_breakout`, because it is a per-asset trend rule, not a macro map.
  - Trinity counts in `optimized_low_risk_allocation` (inverse-volatility weights with a volatility target).
- **Prior trials:** Compass 826 (the 2026-09-28 study); CORE5, TFI and Trinity 50 (the rule's floor; their
  documented variants were not counted).
- **Luck-band fixes:** two edge cases, both fixed without changing the live (offset 0) path.
  - TFI: an offset decision on the data's first session.
  - Trinity: an offset decision before BIL listed.
- **Costs:** S4 runs every pod at the house costs, for comparable cards. The engines' own cost models (Compass and
  TFI 5 bp, no commission; Trinity 1 bp plus IBKR fees) are what the identity gate uses.

## Not done

- S3 for Compass and TFI uses their gate or score structure (diagnostics, class W). Trinity has no score; its S3 is
  its risk gate.
- The G3 reference book, capacity v2, and Fama-French factors.
- P6c (EOM, sector IBS) and P7 (DV2, HPI) remain.
