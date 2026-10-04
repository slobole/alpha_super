# Scout P6c: EOM, the sector IBS pods, and a Bitcoin + gold sleeve

Date: 2026-10-02. Research only; nothing here changes live trading.
- **Scripts:** `scripts/research/scout_p6c_reaudition_20261002/run_p6c.py` and
  `scripts/research/scout_btc_gold_20261002/run.py`.
- **Cards:** `results/scout/cards/` (not in git).
- **Ledger:** eight registrations: six RETRO re-auditions, plus the two Bitcoin + gold registrations, written before
  any Scout result.

**Gates.** Two agents specified the pods. All pass the exact identity gate against a fresh engine run (differences
of 4e-16 to 6.7e-16):
- **EOM:** the weights engine gained opt-in same-session close (MOC) fills and EOM's close-and-reopen orders.
- **The four sector IBS pods:** the engine gained fractional-share entries and "leave this asset untouched".

All 18 previously gated specs still pass after the merge.

## Verdict

Each pod is tested in NDX VXN's slot of the live 60/40 book, as in P6b.

| Pod | Grade | MCPT p | Alpha t (fullest, net) | Book Sharpe with it vs T-bills | Slot P | Capacity |
|---|---|---|---|---|---|---|
| **EOM** (month-end flow, MOC) | **CANDIDATE** | **0.001** | **3.50** (+9.0%/yr) | **1.46 vs 1.21** | **1.00** | $23M (TLT) |
| **Sector IBS VOX IYR** | WATCHLIST (S3 cost coverage 0.98) | 0.036 | 3.01 | 1.31 vs 1.21 | 0.96 | $2.1M |
| Dispersion IBS KIE IHI XLC | WATCHLIST (DSR live p 0.076) | 0.005 | 2.94 | 1.42 vs 1.12 | 0.99 | $2.6M |
| Dispersion IBS KIE IHI XLC SMA200 | **REJECTED** (S3 hard fail) | 0.004 | 2.03 | 1.43 vs 1.23 | 0.99 | $2.4M |
| Dispersion IBS KIE IHI SMA200 | **REJECTED** (S3 hard fail) | 0.136 | 2.47 | 1.32 vs 1.21 | 0.97 | $1.9M |

## What this means

**1. EOM is the strongest pod Scout has audited.**
- **The evidence:** the month-end flow passes every gate with margin.
  - The search beats all 1,000 shuffled histories but one.
  - Net alpha is +9% a year after SPY, QQQ, IEF, GLD, trend and TAA 3x.
  - It lifts the live book's Sharpe from 1.21 to 1.46.
  - S3 (two windows, labels only): the pressure percentile predicts both legs with the expected sign (t 2.58 and 2.15).
- **The cautions:**
  - It is cost-sensitive: Sharpe about 1.15 net falls to 0.59 at twice the costs plus 10 bp. It trades both legs at
    the close several times a month, so execution quality decides it.
  - PBO is 0.90 (a diagnostic): the in-sample best configuration rarely stays best.
  - MOC fills are assumed at the official close.
- **Earlier work:** the 2026-09-26 search called EOM "a forward-test candidate". Scout's evidence is stronger than
  that, and the forward test is still the right next step.

**2. The sector IBS pods earn their money from timing, not from picking the right ETF.**
- **The tension:** S3 measures the event's excess over the same-date basket, the relative edge. S5 and S6 measure the
  whole rule.
  - **VOX IYR:** small but positive relative edge (+9.8 bp, t 2.13, placebo p 0.01), short of covering costs.
  - **SMA200 dispersion variants:** the relative edge is NEGATIVE under both estimators, which is a hard fail by D22.
    Yet their MCPT passes (one of them at p 0.004), and they improve the book with P 0.97-0.99.
- **The reading:** the rule makes money by being invested in sector ETFs right after market-wide down shocks. The
  "buy the most oversold ETF" selection adds nothing, or subtracts. The REJECTED grade is the pre-registered rule
  applied as written: it rejects the selection claim. The timing value is real in sample and is the owner's call.
- **History:** the XLC variants have only 2018-07 to 2022-12 in sample (4.5 years; walk-forward not runnable).
- **Capacity:** about $2M for every IBS pod.

**3. Bitcoin + gold (the GQResearch article; see below).**

## The Bitcoin + gold sleeve: should the book hold a little Bitcoin?

**Rule:** the article's "hold both". Bitcoin and gold are held while their 21/42/63-session trends are up: 50/50
when both are, 100% in one when only one is, otherwise T-bills. Each is capped at 20% volatility.
- **Replica:** CAGR 17.7%, drawdown −17.7% on 2018-2025, against Pakal's 18.4% / −17.6%.
- **Control:** a fixed 50/50 with the same cap (14.0% / −23.5%; Pakal 14.3% / −23.3%).
- **The question:** a 10% sleeve (54% TAA 3x / 36% NDX VXN / 10% sleeve) against the same book with 10% T-bills.

| | Trend rule | Static 50/50 with cap |
|---|---|---|
| MCPT (does the rule beat a volatility-targeted Bitcoin + gold mix?) | **p 0.030, PASS** | p 0.50, FAIL |
| DSR with Pakal's 866 prior trials (warn) | p 1.00 | p 1.00 |
| Net alpha, fullest model | +3.1%/yr, t 0.59 | −2.6%/yr, t −0.51 |
| 10% sleeve, in sample (2017-10 to 2022-12) | book 1.11 vs 1.08, P 0.77 | book 1.07 vs 1.08, P 0.38 |
| 10% sleeve, full 2017-10 to 2026-09 (contaminated) | book 1.40 vs 1.35, **P 0.95** | book 1.37 vs 1.35, P 0.75 |
| Grade | WATCHLIST | WATCHLIST |

**The answer to "hold a little Bitcoin?":**
- **If at all, then with the trend rule, not as a fixed holding.**
  - **The trend version:** its signal beats a volatility-targeted mix (MCPT p 0.03). As a 10% sleeve it nudges the
    book up (+0.03 Sharpe in sample, +0.05 over the full period).
  - **The fixed version:** adds nothing.
- **But the evidence is thin:**
  - only about five years in sample;
  - an 11-month losing streak (a WARN);
  - no alpha after the factors;
  - with the 866 trials already run on this idea, the DSR cannot pass.
- **Where it agrees with Pakal:** "risk management, not alpha".
- **Where Scout adds:** the trend signal does carry some timing information, and in a small sleeve the book is
  slightly better with it than with T-bills.
- **Recommendation:** if the owner wants Bitcoin exposure, a small trend-ruled Bitcoin + gold sleeve is defensible
  as a forward shadow. It is not an allocation Scout can certify.

## Engine and runner changes

- **`engines/weights.py`** (all opt-in; every gated spec unchanged):
  - `fill_at_close_bool` and `close_and_reopen_bool` (EOM's MOC adapter);
  - `fractional_shares_bool` and `hold_nan_bool` (sector IBS).
- **`reaudit.py`:** an optional custom reference book per pod (`book_weight_dict`, used for the 10% sleeve).
- **`s5_overfit.py`:** walk-forward reports "not runnable" when the history is shorter than the registered design's
  training window.
- **`card.py`:** an S3 hard fail (class E/X) grades REJECTED (D22); it was graded WATCHLIST before.
- **New spec:** `alpha/scout/specs/btc_gold.py`, the first discovery family with no engine counterpart. It reads
  Pakal's local panel and does not download.
- **New deviation:** `xnys_closure_hindsight` (EOM: today's exchange calendar knows about the 2012 Sandy closures;
  Sharpe 1.087 → 1.086).
