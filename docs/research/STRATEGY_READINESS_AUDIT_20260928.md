# Strategy readiness audit — Tier A milestone (2026-09-28)

Audit only. No strategy, engine, data-loader, live, release, registry or portfolio file was changed, and nothing
was committed. Code audited: commit `cb29d4f` (HEAD of `main`). Data: the local Norgate installation, whose last
bar is 2026-09-25.

- Frozen protocol: [`STRATEGY_READINESS_AUDIT_PROTOCOL_20260928.md`](STRATEGY_READINESS_AUDIT_PROTOCOL_20260928.md)
  (SHA-256 `9e4bc29f…6b41b`).
- Post-result clarifications: [`…_AMENDMENTS.md`](STRATEGY_READINESS_AUDIT_PROTOCOL_20260928_AMENDMENTS.md).
- Study code: `scripts/research/strategy_readiness_audit_20260928/`.
- Outputs (local, gitignored): `results/research/strategy_readiness_audit_20260928/`.

Tiers B and C are in sections 10–12. Corrections made after the Tier A review are in section 7b.

## 1. Summary

**The strategy logic is sound, and live makes the same decisions as the backtest.** Across every wired strategy the
live host was replayed on historical dates with data cut at each date. It reproduced the backtest's decisions
exactly:

| Strategy | Month-ends or sessions matched |
|---|---|
| TAA 3x, TAA 1/N, BTAL_QQQ | 167/167 each |
| NDX ATR VXN | 51/51 |
| NDX ATR | 23/23 |
| HPI | 31/31 per variant |
| DV2 | 44/44 |
| QPI | 46/46 |
| CORE5 | 626/626 |

- **No new look-ahead of the NDX kind** (split-adjusted prices leaking into the backtest) was found. Future-split,
  truncation and planted-leak tests pass everywhere.
- **The one remaining look-ahead is the known 5-session membership trim.** It is immaterial over the full history,
  but it is optimistic by about 0.5 pp/yr for NDX over the last 3 years.

**What is not ready is the live plumbing around those decisions, and size.**

1. **Account-level checks are missing.** Currency, security type, unrelated holdings, a missing broker read and
   duplicate processes are not checked. The owner confirmed on 2026-09-28 that the live accounts are USD, margin,
   hold strategy positions only, and run one process per pod. That closes these doors for now, but only by
   operating discipline.
2. **Stale helper data passes silently.** $VIX, $VXN and SPY one day stale would have flipped the TAA 3x TQQQ gate
   in 4 of 169 months, and moved NDX exposure by up to 12% of NAV in 60 of 169 months. This has never happened since
   2008, but nothing would block it.
3. **Live order sizing does not follow the backtest's sizing rule.** The effect is about 0.16 pp/yr for TAA 3x.
4. **Size.**
   - At the owner's likely NDX pod size (about USD 12–18K), the USD 1 minimum commission and whole shares cost about
     1.3–2.1 pp/yr against the backtest's 12% CAGR, and high-priced names are skipped without a warning.
   - DV2 and QPI lose 4.5 and 1.8 pp/yr at USD 30K.
   - At today's BTAL volume (USD 8.6M/day over 252 sessions) the TAA family's order-size cap is about USD 1.5–2.2M.
     The earlier USD 0.9M figure came from 3–4 BTAL orders in 2023-10, when BTAL was briefly thin (see the
     correction in section 7b).

Under the frozen rules a silent failure mode means **live parity NOT READY**, even when the replay is exact. That is
why most live-parity verdicts below are NOT READY. Every one of them has a small, specific fix (section 5).

## 2. Scorecard

Verdicts follow protocol section 5 literally. Abbreviations: **R** = READY, **RC** = READY WITH CAVEATS,
**NR** = NOT READY.

| Strategy | Money | BC | LP | TR | Why LP / TR are not READY |
|---|---|---|---|---|---|
| TAA 3x `strategy_taa_df_btal_fallback_tqqq_vix_cash` | LIVE | RC | NR | RC (C1 cap about USD 1.8M at today's BTAL volume) | Stale-helper silent path; sizing semantics; account guards · house auction model +0.87 pp/yr at USD 1M |
| TAA 1/N `strategy_taa_df_btal_1n_fallback_tqqq_vix_cash` | wired | RC | NR | RC | Same as TAA 3x |
| BTAL_QQQ `strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash` | wired | RC | NR | RC | Same as TAA 3x |
| NDX ATR VXN `strategy_mo_atr_normalized_ndx_vxn_scaled` | LIVE | RC | NR | RC at 30K; NR below about 20K | Stale VXN silent; trim divergence; no stale-plan guard · whole shares and minimum fee at pod size |
| NDX ATR `strategy_mo_atr_normalized_ndx` | wired | RC | NR | same as VXN | Same, plus only 12 informative replay dates |
| HPI 2/3/5 vote | wired, not live | RC | NR | NR at 30K (literal) | Missed day loses exits silently; reconcile memory · zero-share names; margin up to 60% of NAV |
| HPI IBS/RSI exit | wired, not live | RC | NR | NR at 30K (literal) | Same as vote |
| DV2 `strategy_mr_dv2:DVO2Strategy` | wired | RC | NR | NR at 30K | Missed day; held name without a bar; reconcile memory · −4.5 pp/yr at 30K |
| QPI `strategy_mr_qpi_ibs_rsi_exit` | wired | RC | NR | NR at 30K | Held name without a bar is never exited (silent) · −1.8 pp/yr at 30K |
| CORE5 `strategy_taa_adaptive_macro_core5` | PM_READY, LIVE refused in code | **R** | NR | RC | A missed evening halts the pod permanently with only a WARN; short has no locate or margin check |

"Money": TAA 3x (account U21192795) and NDX ATR VXN (U25384771) trade real money. The VPS README lists only these
two live. Whether DV2, QPI or HPI run anywhere in paper mode was not checked (no VPS access).

## 3. Per-strategy scorecards

### 3.1 TAA 3x — and TAA 1/N and BTAL_QQQ, which share its code path

**Rule.** At each month-end, rank GLD, UUP, TLT, DBC and BTAL by the average of their 1/3/6/12-month total returns,
using TOTALRETURN closes. They get rank weights 5/15 … 1/15. A slot whose score is not above the DTB3 1-month hurdle
goes to TQQQ. TQQQ's weight goes to cash if SPY 20-day realized vol is not below $VIX. Orders are MOO at the first
session of the next month. 1/N uses equal slots; BTAL_QQQ uses a linearity score, QQQ as fallback, and no DTB3.

**Backtest correctness: RC.**

| Check | Result | Evidence |
|---|---|---|
| A1 code read | Momentum = ratios of TR closes (scale-free); `resample("ME").last()` on the full padded daily index; hurdle `resample("ME").last()` of DTB3; the VIX gate uses an inner join of SPY and $VIX; the partial-month row is never traded in the backtest and is refused live (`scheduler_utils.py:271-346`) | `strategy_taa_df.py:280-363`, `…vix_cash_variant_utils.py:112-227` |
| A2 future-split invariance | 21/21 per variant (7 symbols × k ∈ {40, 0.1, 1.5}) | `taa/bc_checks_*.json` |
| A3/A4 truncation and one-session leaks | Multi-month: 9/9. One-session: shown by the 167/167 live replay, which catches a planted Close_(T+1) (AM-02) | `taa/live_parity_*.csv`, `review_quant/rq_taa_positive_controls.json` |
| A5 DTB3 publication lag | 0/169 flips with a one-session lag. The margin can be thin (2.5e-5 on 2025-09-30) | `review_quant/rq_taa_checks.json` |
| A7 padded bars | BTAL fills on 3 zero-volume bars (2014-12-01, 2015-04-01, 2017-07-03) | `taa/bc_checks_taa3x.json` |
| A8 accounting | Dividends net of 25% withholding; cash earns 0%; negative cash unfinanced (min −1.8% of NAV, 39% of days, < 0.01 pp) | `taa/bc_checks_*.json` |
| A9/A10 | Capital scaling: daily returns correlate 0.99999997. Two runs bit-identical | `taa/bc_checks_*.json` |

- **Material but conservative.** Fees are charged per split-adjusted share: TQQQ has split 8 times. Raw-share fees
  would add +0.32 pp CAGR (3x) and +0.38 pp (1/N) (`review_quant/rq_taa_fee_and_hurdle_margin.json`).
- **Sample window.** CAGR is 23.9% over the full window and 22.2% from 2014. 2012–13 was a strong period, and BTAL
  then traded about USD 30K a day, so those fills were not executable. From 2019 the results are stronger: 26.2%,
  Sharpe 1.42 (`taa/window_sensitivity.json`, `review_trade/r5_*.json`).
- **Selection (cited).** TAA 3x ranked #3 of 48 sibling variants. BTAL and the TQQQ fallback were chosen with
  hindsight.

**Live parity: NR.**

- **B1 replay:** 167/167 month-ends exact; weights diff 0.0; execution dates match (`taa/live_parity_summary_*.json`).
  - Pod state has no effect: a hostile PodState gives identical weights.
  - Snapshot mode gives the same weights as direct Norgate (`review_live/snapshot_vs_direct_host_probe.json`).
- **Why NR:**
  - **A-TAA-01 = A-LIVE-13 (upgraded to S2).** The profile has no per-symbol month-end freshness check. A one-day-stale
    $VIX would flip the TQQQ gate in 4/169 months (2015-08, 2018-10, 2018-11, 2026-06); SPY and $VIX both stale, in 9.
    The plan metadata does not record the $VIX date (`review_live/helper_one_day_stale_sensitivity.json`).
  - **A-TAA-02 = A-LIVE-06.** The backtest fixes shares at `int(NAV_close × w / Close_T)` and trades only non-zero
    deltas (`strategy_taa_df.py:473-506`). Live re-prices every target at the auction indicative price
    (`execution_engine.py:182-213`), so it trades against the overnight gap every month.
    - Return effect: −0.16 pp CAGR (TAA 3x), −0.20 (1/N), −0.01 (BTAL_QQQ).
    - Needs margin up to 2.8% of NAV (`taa/live_sizing_semantics.json`).
  - **Account-level guards** (A-LIVE-01…04, section 4). They are closed today only by the owner's confirmation.
- **Not replayed:** the VPS copy of the exporter, and the real month-end night timing.

**Tradability: RC. Passes at owner size; order-size cap about USD 1.5–2.2M at today's volume (corrected in §7b).**

| Pod size | p99 order, % of ADV (last 3y) | Result |
|---|---|---|
| USD 30K | 0.17% | passes |
| USD 1M | 5.67% (1/N 5.62%, BTAL_QQQ 5.82%) | fails; p99 reaches 5% at about USD 0.88M |

- **Binding name: BTAL**, which trades about USD 7.2M a day (`tradability/*/participation_summary.csv`).
- **Opening-auction effect.** MOO trades only the auction, not the whole day. The house guardrail is 0.10% of ADV
  (`alpha/engine/capacity_analysis.py:91-92`). At USD 1M, 30% of orders exceed it. Modelled impact is about
  +1.05 pp/yr at 1M and 6–13 bp/yr at owner size (`review_trade/r4_auction_participation.json`).
- **Whole shares at USD 15K:** mean rounding cash 1.1% of NAV, max 3.3%.
- **Minimum fee at USD 12–18K:** −0.2 to −0.3 pp/yr on IBKR Fixed.
- **Open question — withholding on DBC/UUP sales (unverified).** These are US partnership ETFs. IBKR's §1446(f)
  withholding policy for non-US holders is unknown. If IBKR withholds 10% of gross sale proceeds, about 7.7% of NAV a
  year would be withheld. Check an IBKR statement for a past UUP/DBC sale.

### 3.2 NDX ATR VXN (live) and NDX ATR (wired)

**Rule.**
- **Decision:** at the actual last session of each month, among point-in-time Nasdaq-100 members with Close > SMA100,
  rank by `ROC12 / ATR20$`. ATR20$ is the adjusted ATR rebased to nominal dollars at T.
- **Regime:** hold only if SPY > SMA200; otherwise all cash.
- **Positions:** top 10 at 1/10 each.
- **VXN variant:** scale all weights by clip(22/VXN_T, 0.25, 1).
- **Orders:** MOO at the next open.

**Backtest correctness: RC.**

| Check | Result | Evidence |
|---|---|---|
| A2 split invariance | 33/33. The pre-fb81e86 formula is caught in 28/33 | `ndx/split_invariance_vxn.json` |
| Month-end detection on partial data | Drops a month whose last bar is not the XNYS month-end (`strategy_mo_atr_normalized_ndx.py:277-313`) | code read and probe |
| A6 membership trim | Full history +0.004 pp (VXN) and −0.006 pp (ATR). Last 3 years **−0.47 pp (VXN)** and −0.59 pp (ATR): the backtest is optimistic. One decision drives it: 2026-07-31, where live would hold EA-202608 and the backtest MRVL | `review_quant/rq_ndx_untrimmed.json` |
| Determinism and accounting | Raw historical share units are already on. Fills at the open, sized from Close_T | `strategy_mo_atr_normalized_ndx.py:718-720` |

**Design findings for the owner: not leaks, and live equals backtest.**

- **A-NDX-01 — the score is mostly a nominal-share-price tilt.** Because ATR20$ is in dollars,
  `log score = log ROC − log NATR − log price`.
  - Over the last 60 decisions, log price carries about 68% of the score's cross-sectional variance.
  - Picks have a median price of USD 85, against USD 185 for the eligible pool.
  - Only about 4 of 10 picks match the price-free (NATR) ranking.
  - A stock split raises a name's rank mechanically (`ndx/nominal_price_dependence.csv`).
- **A-NDX-02 — the score favours pinned takeover targets.** A cash deal pins the price, so ATR collapses and the score
  inflates.
  - The production backtest picked EA in 5 of the last 11 months; it also picked AZN-202601 and WBD.
  - On the 2026-09-25 bar, WBD ranks #1 with ATR at 1.36% of price.
  - The live account therefore holds merger-arbitrage names (`review_quant/rq_ndx_untrimmed_detail.json`).

**Live parity: NR.**

- **B1 replay:**
  - VXN: 51/51 exact, 37 of them non-empty. ATR: 23/23 exact, but only 12 non-empty, below the 24 required.
  - These replays prove the host code is identical; they cannot see the universe difference (AM-04).
- **Why NR:**
  - **A-NDX-03 — trim divergence.** Live membership is untrimmed, so live picks differ from the backtest in 8 of 320
    month-ends, one name each time. The cause is not recorded in `ASSUMPTIONS_AND_GAPS.md`.
    - One is checkable now: on 2026-08-03 live should have bought EA, not MRVL.
    - Evidence: `review_live/ndx_trim_live_divergence_summary.json`.
  - **A-NDX-04 — stale $VXN is silent.** A one-day-stale VXN changes exposure in 60/169 months, by up to 11.8 pp of
    NAV. Operator check for 2026-10-01: the plan's `vxn_reference_date_str` must be `2026-09-30`.
  - **A-LIVE-15 — no stale-plan guard in the NDX host.** Invoked mid-month, it returns last month's plan with a target
    in the past (`ndx/ndx_invocation_timing_probe.json`). Scheduled `serve` never does this, and the runner then
    expires such a plan, so no order results.
  - **A-LIVE-06 and the account-level guards.**
- **Snapshot mode equals direct mode** (`review_live/snapshot_vs_direct_host_probe.json`).

**Tradability: RC at USD 30K; NR below about USD 20K.**

- **ADV participation is negligible:** p99 3.3% of ADV even at USD 10M; last 3y 1.1%.
- **Whole shares (since 2016)** (`tradability/ndx_vxn/whole_share_summary.csv`):

  | Pod size | Max per-name error, % of NAV | Mean rounding cash, % of NAV |
  |---|---|---|
  | USD 30K | 4.8% | 1.5% |
  | USD 12–15K | 10% (a whole slot) | 3.2% |

  - SNDK (USD 1,567), selected on 2026-08-31, gets 0 shares at ≤ USD 15K, with no warning (A-LIVE-08).
- **Minimum fee (R-01).** At USD 12K every order pays the USD 1 minimum: 100 bp/yr on IBKR Fixed, 40 bp/yr on
  Tiered, against 12.7 bp/yr in the backtest (`review_trade/r2_commission_owner_size.json`).
- **Total drag vs the backtest's 12.0% CAGR:**

  | Pod size | IBKR Fixed | IBKR Tiered |
  |---|---|---|
  | USD 12K | about −2.1 pp/yr | about −1.4 pp/yr |
  | USD 18K | about −1.3 pp/yr | about −0.9 pp/yr |
  | USD 30K | about −1.0 pp/yr | about −0.7 pp/yr |

- **Idle cash earns 0%.** Mean cash is 32% of NAV (13% in the last 3y). That is realistic at owner size.

### 3.3 HPI 2/3/5 vote and HPI IBS/RSI exit (wired, not live)

Full findings: `results/research/strategy_readiness_audit_20260928/hpi/HPI_FINDINGS.md` (A-HPI-01…13).

**Backtest correctness: RC.**
- Split invariance 16/16; row-T truncation 20/20; planted one-day leak caught 9/9.
- The headline reproduces: vote 16.18% CAGR, Sharpe 1.041.
- Conservative, material:
  - adjusted-unit fees: +0.33 / +0.37 pp;
  - 0% on 28–30% idle cash: about +0.4 pp.
- Liquidation fee in adjusted units (A-HPI-07), one event.
- Method finding (A-HPI-09): the engine prefix test used by the leakage hunt cannot see a one-bar leak. A row-T
  feature test is needed.

**Live parity: NR.**
- Replay 31/31 per variant, including 13–16 same-open refill sessions, so the refill fix matches.
- **Why NR:**
  - A-HPI-10: a missed daily cycle loses exits silently. Example: state from 2025-06-12 invoked on 06-13 loses the
    AJG, BRK.B and KMB exits.
  - A-LIVE-16: reconcile commits strategy memory even when reconciliation failed.
  - A halted member is treated as removed (A-HPI-03, zero historical cases).

**Tradability: NR at USD 30K (literal rule).**
- Zero-share names (AZO, BKNG, NVR) give a 10% per-name error on 0.13% of entries. The return effect is small:
  +0.06/+0.09 pp.
- **Margin (A-HPI-06, S2):** refill buys need buying power beyond cash of median 9.4% of NAV, up to 60%
  (`hpi/scans_*.json`).
- USD 10M p99 is 5.16% of ADV (baseline).

### 3.4 DV2 and QPI (wired)

Full findings: `results/research/strategy_readiness_audit_20260928/mr_dv2_qpi/DV2_QPI_FINDINGS.md`
(A-DV2-01…09, A-QPI-01…08). This is QPI's first audit.

**Backtest correctness: RC.**

| Check | QPI | DV2 |
|---|---|---|
| Split invariance | 18/18 | 17/18; the failure is the adjusted-unit zero-share cancel, which passes 9/9 in raw units |
| Feature truncation | bit-identical at 9 cut-offs | bit-identical at 9 cut-offs |
| Planted leak | caught 18/18 | caught 18/18 |

- **QPI's quantile window includes Close_T.** That is causal, since T is known at the decision.
- **Adjusted-unit fees understate returns (conservative):** DV2 +0.87 pp, QPI +0.36 pp.
- **The trim is optimistic over the last 3 years:** DV2 0.22 pp, QPI 0.24 pp.
- **Fast vs reference indicators differ on float32 near-ties.** This is a test-oracle issue; backtest and live both
  use the fast kernel.

**Live parity: NR.**
- Replay 44/44 (DV2) and 46/46 (QPI), including 36 and 24 same-auction refill sessions. There is no HPI-type refill
  bug.
- **Why NR:**
  - QPI never exits a held name that has no bar or a NaN IBS, and says nothing (A-QPI-05).
  - DV2 raises `KeyError` in that case: loud, but it blocks the pod (A-DV2-04).
  - Missed-day loss (A-HPI-10 applies to both).
  - A-LIVE-16.
  - Class-share tickers such as BRK.B go to IBKR unmapped (A-DV2-05, code read).

**Tradability: NR at USD 30K.**
- Last 3 years at USD 30K vs USD 10M: DV2 23.7% vs 28.3% CAGR; QPI 10.1% vs 11.9%.
- The USD 1 minimum fee explains −3.5 / −2.0 pp of that.
- Margin is needed on 78% / 46% of entry days, up to 97% of NAV.

### 3.5 CORE5 (PM_READY; live adapter exists; LIVE mode refused at `alpha/live/core5_adapter.py:37-38`)

Full findings: `results/research/strategy_readiness_audit_20260928/core5/CORE5_FINDINGS.md`.

**Backtest correctness: R.**
- Split invariance 10/10 full-engine runs; truncation 14/14; planted leak caught 8/8.
- Deterministic.
- Capital scaling: 6.92% CAGR at 30K up to 7.17% at 10M.
- Dividend stamping verified on 135/135 SPY and 285/285 IEF events.
- All accounting items are below 0.02 pp, or conservative.

**Live parity: NR.**
- Replay 626/626 decisions: 229 month-ends, 211 with a DBC short target, 72 two-leg sign flips.
- **Why NR:**
  - A-CORE5-10: a missed or duplicate session raises, and the runner only WARNs and continues
    (`runner.py:2664-2673`). The snapshot window (`runner.py:3636-3651`) then halts the pod permanently, with no
    catch-up.
  - A-CORE5-12: the DBC short is a plain SELL MOO with no locate or margin check.
  - Incubation credits no dividends and charges no borrow (A-CORE5-11).

**Tradability: RC.**
- Fails 5% of ADV at USD 10M (DBC/UUP); the edge is at about USD 3.3–3.6M.
- One SPY share is 2.6% of NAV at USD 30K.
- Needs a margin account for the short.
- DBC/UUP withholding is the same open question as TAA.

## 4. Shared live execution path (all pods)

Full findings: `results/research/strategy_readiness_audit_20260928/live_path/LIVE_PATH_FINDINGS.md` (A-LIVE-01…17)
and `review_live/REVIEW_LIVE.md`. Tests: `tests/test_strategy_readiness_audit_live_path_20260928.py` (40) and
`tests/test_strategy_readiness_audit_review_live_20260928.py` (6), all synthetic.

**Verified correct:**
- **Timing chain:** month-end gate, next-month first session, holidays 2026–27, early closes, idempotent plan build.
- **Snapshot equals direct:** snapshot fields and dtypes match the direct loader, and the exporter output gives the
  same decisions.
- **Price guards:** zero, negative and NaN prices are refused.
- **Order failures park the pod:** rejected or cancelled orders and residuals park it.

**Gaps**, ranked by money at risk in the fix list below: A-LIVE-01…04, 06, 07, 08, 09, 13, 15 and 16.

## 5. Ranked fix list for Codex (money at risk first)

Each item says what to change, where, and how to test it. Tier-3 items need the project's live-impact checklist and
the parity, failure-modes and coverage reviewers.

| # | Fix | Where | Test to add | Affects |
|---|---|---|---|---|
| 1 | **Account-level guards in the trading snapshot.** Read NetLiquidation, TotalCashValue and AvailableFunds by (tag, currency) and require USD. Build positions only from STK / USD / multiplier-1 rows and raise on any other row that shares a traded symbol. Raise on a fractional position. | `alpha/live/ibkr_socket_client.py:395-432` | Fakes: ILS NetLiq raises; an OPT row on TQQQ raises; a fractional position raises | TAA 3x, NDX VXN (live), all pods |
| 2 | **Make the pre-VPlan position reconciliation blocking for full-target books**, not only CORE5. Add an allow-list so a full-target VPlan can sell only symbols in the strategy's tradeable universe. Generalize CORE5's fresh pre-submit broker check (positions and open orders equal the VPlan) to every pod. | `runner.py:2925-2955`, `:3273-3285`; `models.py:137-144` | Empty broker read raises (no 2× rebuy); stray AAPL is not sold; second DB/process sends no second basket | TAA 3x, NDX VXN |
| 3 | **Per-symbol month-end freshness check for helpers and ETFs** in `norgate_eod_etf_plus_vix_helper` and `norgate_eod_ndx_pit_plus_vxn_helper`. Extend CORE5's observed-endpoint check to $VIX, $VXN, SPY, the TAA ETFs and held NDX names, failing closed. Record the $VIX/SPY observation dates in TAA plan metadata. | `scripts/export_norgate_snapshot.py:309-328`; `strategy_host.py:838-1098`, `:1262-1285` | A padded (stale) $VIX/$VXN month-end bar raises | TAA 3x, NDX VXN |
| 4 | **Size full-target books like the backtest.** Freeze target shares at `trunc(NAV_close × w / Close_T)` in the DecisionPlan and trade only non-zero deltas, as CORE5 already does. Any other sizing is a semantic change and must be recorded as one. | `execution_engine.py:150-230`, `strategy_host.py` TAA/NDX builders | 8% gap-down: 0 shares traded for an unchanged target | TAA 3x, NDX VXN |
| 5 | **NDX host guards:** copy TAA's snapshot-month = signal-month and target-not-in-past checks. | `strategy_host.py:1101-1295` (copy from `:1072-1097`) | Mid-month as-of raises | NDX VXN |
| 6 | **Warn on zero-share targets:** add a VPlan warning row and a trace WARN whenever target weight > 0 and target shares == 0. | `execution_engine.py` VPlan builder | BKNG at a USD 750 budget gives a warning | NDX VXN at owner size |
| 7 | **Remove the 5-session membership trim** in the loader and the exporter, or apply it only to real removals. Record G-rows in `ASSUMPTIONS_AND_GAPS.md` until done. Give replays an as-of universe (as `mr_dv2_qpi/aud_data.py:60-74` does). | `data/norgate_loader.py:106-107`, `scripts/export_norgate_snapshot.py:134-139` | Membership as of T equals index membership on T | NDX, DV2, QPI (backtests); live universe already correct |
| 8 | **Missed-cycle detection for daily incremental pods:** alert and park when the last completed plan's signal is older than the previous session. Retry a blocked VPlan (missing live price, account not visible) until 09:27:30 instead of giving up. | `runner.py:2990`, `:3059`; `dashboard.py:1353` | Skipped session raises an alert; transient price miss retried | DV2, QPI, HPI; TAA/NDX month skip |
| 9 | **Fail closed on held names without a bar or finite exit features** (QPI silent, DV2 KeyError). Add a Norgate→IBKR class-share symbol map (BRK.B → "BRK B"), verified in paper. | `strategy_host.py:430-583`; `ibkr_socket_client.py:143-152` | Held name with NaN IBS parks the pod; BRK.B qualifies | DV2, QPI |
| 10 | **Commit strategy memory only after successful reconciliation.** | `runner.py:3927-3940` | Failed reconcile keeps the prior state | DV2, QPI, HPI |
| 11 | **CORE5:** park and alert on the first `core5_decision_blocked`; add a catch-up or resume path; wrap `complete_core5_cycle`; add a pre-trade account-type, shortable and borrow check for DBC before any paper use. | `runner.py:2664-2673`, `:2695-2697`, `:3636-3651`; `ibkr_socket_client.py:696` | Missed evening parks the pod; short without shortable flag raises | CORE5 (before paper) |
| 12 | **Reject `--as-of-ts` on live and paper mutating commands**, or require it within 60 s of now. | `runner.py:395-398`, `:5641` | Backdated submit refused | all |
| 13 | **Expire MOO/OPG plans at 09:28:00, not at the open**, and alert loudly when a plan expires unsubmitted. | `scheduler_utils.py:465-480` | 09:29 submit refused with an alert | manual-submit pods |
| 14 | **fred_loader: compare the as-of date in New York time, not UTC.** Lag DTB3 by one trading session in the TAA backtest, for parity with live (0 flips measured). | `alpha/data/fred_loader.py:61-66, 130-131`; `strategy_taa_df.py:296-299` | 20:00 NY as-of excludes T+1 observations | TAA 3x, 1/N (replays; parity) |
| 15 | **Research accounting: raw historical share units** for the TAA family, DV2, QPI, HPI and CORE5 (paused share-units handoff, option A). Fix HPI's liquidation fee units. | `alpha/engine/strategy.py`, `strategies/hpi/stateful_long.py:673` | Existing share-units tests | Reported numbers only (conservative today) |
| 16 | **Docs:** `CORE5_ADAPTER_QUALIFICATION.md:87-88`; the snapshot contract (last-row Dividend is provisional); G-033 margin wording (60% of NAV, not −2.3%). | docs | — | — |

## 5b. NDX: live ATR rule vs NATR20 (closed 2026-09-28)

Study: `scripts/research/strategy_readiness_audit_20260928/ndx/natr20_vs_atr_decision.py`.
Outputs: `results/research/strategy_readiness_audit_20260928/ndx/natr20_decision/`.

The two rules differ in one line: score = ROC12 / ATR20 in dollars (live) vs ROC12 / ATR20 in percent of price
(NATR20). Both use the live (untrimmed) universe below.

**NATR20 backtest correctness: RC (corrected after review).**
- Split invariance: 18/18.
- Truncation: 8/8, but only 4 cut-offs are informative; the other 4 are regime-off months where both sides are empty.
- The planted-leak control was mis-specified: it compared leaky full-history code with honest truncated code.
- A5 and A7–A10 were not run for NATR20.
- Substance holds: NATR20's own loader is bit-identical to the ATR-VXN loader used here
  (`review_bc_quant/rbq_natr20_loader_parity.json`), and its month-end helper drops partial months correctly.

| Window | Live ATR rule: CAGR / Sharpe / MaxDD | NATR20: CAGR / Sharpe / MaxDD |
|---|---|---|
| 2000–2026 | 12.1% / 0.77 / −29.3% | 15.0% / 0.83 / −29.3% |
| 2016+ | 14.7% / 0.88 / −20.5% | 16.8% / 0.82 / −29.3% |
| Last 3y | 28.8% / 1.23 / −20.5% | 24.2% / 0.88 / −29.3% |

**At the real pod size.** USD 12K, raw whole shares; the pod compounds from USD 12K.

| Start | Live ATR: Fixed / Tiered / USD 1M | NATR20: Fixed / Tiered / USD 1M |
|---|---|---|
| 2016 | 13.75% / 14.18% / 14.74% | 14.87% / 15.34% / 16.82% |
| 2021 | 12.65% / 13.18% / 14.04% | 12.01% / 12.49% / 15.01% |

**Findings.**
- **Price tilt confirmed.** The median nominal price of buys since 2016 is USD 68 for the live rule and USD 198 for
  NATR20.
- **Takeover tilt: the earlier claim was wrong.** Section 3.2 (A-NDX-02) said the live score favours pinned takeover
  targets. This comparison does not support it. The share of buys delisted within 3 months since 2016 is 2.15% for
  the live rule and 2.57% for NATR20, because a pinned price lowers percent ATR as much as dollar ATR.
- **Small-account drag, realistic path.** The growing-pod path is smaller than the reviewer's constant-USD-12K
  bound. The live rule gives up about 1.0–1.4 pp/yr on Fixed and 0.6–0.9 pp/yr on Tiered.
- **NATR20 at a small pod.** It gives up 2.0–3.0 pp/yr, because it buys higher-priced names.
- **Wiring.** NATR20 has no live route today: it is not in the release manifest, and the host's NDX builder calls
  `compute_atr_normalized_signal_tables`, which NATR20 names differently.

**Conclusion.**
- NATR20 shows no leak (BC RC), earns more over 2000–2026 with more volatility, and has a similar full-history Sharpe.
- It is worse on risk over 2016+ and on the last 3 years.
- At a USD 12K pod it is not better risk-adjusted than the live rule. From 2016 on IBKR Fixed its CAGR is 14.87% vs 13.75%, but its Sharpe is 0.79 vs 0.86 and its max drawdown −27.5% vs −19.0%. These small-pod runs compound (to USD 48–53K by the end), so a pod kept at a constant USD 12K would lose more.
- **Recommendation: keep the live rule, and move the NDX pod account to IBKR Tiered pricing.** Revisit NATR20 only if
  the pod grows well above USD 100K, and only as a declared strategy change with forward shadowing.

## 6. Owner decisions

1. **NDX rule semantics (A-NDX-01, A-NDX-02).** The live rule is mostly a low-share-price and pinned-takeover tilt.
   Keep it as is, or move to the price-free NATR20 variant (Tier B item, which also needs a live route).
2. **NDX pod size and IBKR pricing plan.** At about USD 12–18K the pod gives up about 1.3–2.1 pp/yr to fees and
   rounding.
   - IBKR Tiered pricing roughly halves the fee part.
   - A no-trade band for trims under 1% of NAV saves about 0.3 pp/yr. That is a strategy change.
3. **TAA capacity.** With MOO the family is capped at about USD 0.9M, because BTAL is the binding name. Above that,
   BTAL needs worked orders or a cap.
4. **DV2 / QPI / HPI minimum pod size.** At USD 30K the edge loses 1.8–4.5 pp/yr (DV2 and QPI). Running at
   ≥ USD 100K per pod, or on Tiered pricing, recovers most of it. HPI needs margin headroom of ≥ 0.6× pod NAV.
5. **DBC/UUP withholding.** Check an IBKR statement for a UUP or DBC sale to see whether §1446(f) withholding
   occurs.

## 7. Independent review (protocol section 8)

Three read-only reviewers, each with a different lens:

- look-ahead and quant pitfalls: `review_quant/REVIEW_QUANT.md`;
- live parity and failure modes: `review_live/REVIEW_LIVE.md`;
- tradability: `review_trade/REVIEW_TRADE.md`.

**Adopted (confirmed):**

| Finding | Effect on this report |
|---|---|
| NDX replay blind to the trim; −0.47 pp last-3y optimism | A-NDX-03; AM-04 |
| TAA A3 cannot catch one-session leaks; the replay does | AM-02 |
| DTB3 lag method flaw; 0 flips re-verified | AM-03 |
| fred_loader UTC-date bug | Fix #14 |
| TAA adjusted-unit fees +0.32/+0.38 pp | §3.1 |
| Missing positive controls for TAA/linearity, supplied by the reviewer | AM-02 |
| NDX ATR only 12 informative replays | LP stays capped |
| Stale helper data flips decisions (A-LIVE-13 upgraded to S2) | Fix #3 |
| Empty broker read gives a 2× rebuy | Fix #2 |
| A-LIVE-06 mechanism and its fix corrected | Fix #4 |
| NDX owner-size drag of 1.3–2.1 pp/yr | §3.2; owner decision 2 |
| TAA fails C1 at USD 1M literally | TR verdict |
| DBC/UUP withholding | Owner decision 5 |
| Market-data subscription cost | See below |
| Opening-auction share vs ADV | §3.1 |

The market-data subscription is about USD 120/yr, which is about 0.4% of a USD 30K account and is not in any
backtest.

**Rejected, with reasons:**
- **Hostile PodState changes TAA/NDX decisions:** it does not. The state holds only trade ids.
- **Snapshot mode differs from direct mode:** identical output.
- **Future dividends leak through TOTALRETURN closes:** all TR features are scale-free.
- **Turnover is not nominal:** it matches nominal dollar volume within 0.8%.
- **Slippage sign, ex-date dividend timing and VXN timing are wrong:** all correct.
- **BTAL 2012–18 illiquidity inflates the record:** the 2019+ subsample is stronger.
- **BTAL spread cost is material at owner size:** under 1 bp/yr.
- **TQQQ path risk is a tradability failure:** it is in the price series.

**Rejected from my own earlier claims:**
- "TAA mid-month truncation 9/9" as evidence against one-session leaks (replaced by the replay).
- "DTB3 +1 calendar-day lag, 0/169" as the lag test (replaced by the session lag).
- "NDX 51/51" as evidence of universe parity (it shows code parity only).

## 7b. Corrections to Tier A after the Tier B/C reviews (2026-09-28)

- **TAA capacity.** Following the owner's rule to judge from today's volumes, BTAL's 252-session ADV is USD 8.6M. TAA
  3x's p99 order at USD 1M is then 2.7% of ADV, and the C1 cap is about USD 1.8M
  (`review_bc_trade/rbt04_capacity_at_todays_adv.json`). The 0.88M figure came from 3–4 BTAL orders in 2023-10, when
  BTAL's 20-day ADV was USD 3.06M. TAA TR moves from NR to RC.
- **House auction model for TAA 3x.** Extra cost is about +0.1 pp/yr at USD 30K, +0.23 pp/yr at USD 100K and
  +0.87 pp/yr at USD 1M.
- **Live calendar window (new, all live TAA pods).**
  - `alpha/live/scheduler_utils.py:68-75` builds XNYS with the library's default 20-year-back / 1-year-ahead
    window. Today that window is 2006-09-28 → 2027-09-28.
  - The TAA month-end resolver checks every loaded date, so TAA 3x, 1/N and BTAL_QQQ, whose data starts 2011-09-13,
    will raise from about 2031-09-13. It fails loudly and sends no wrong order.
  - A `serve` process running for about a year also reaches the forward end of the window.
  - Evidence: `review_bc_trade/rbt01_calendar_window_probe.json`. Fix #17.
- **Small-account friction.** It was measured on compounding pods. At a constant pod size it is larger; see
  section 12.

## 8. What was not tested, and why

- **The IBKR side:** real IBKR behaviour, the account statements and the VPS. There is no broker or VPS access, by
  the audit's limits. Specifically:
  - the opening-auction indicative-price availability at 09:23:30;
  - the OPG cutoff;
  - fractional or class-share handling;
  - PTP withholding;
  - the pricing plan;
  - the real release YAMLs, the auto-submit setting and the state databases.
- **Opening-auction volumes and prints.** They are not in the Norgate data.
- **The Norgate server's exporter copy.** It may carry a hotfix (`docs/live/NORGATE_SERVER_DEPLOYMENT.md:52-65`).
- **Data vintages.** One Norgate vintage; a vendor restatement cannot be detected.
- **Deflated Sharpe / selection statistics.** They were cited, not recomputed. Protocol D asks only to note them.
- **The 2000–02 dot-com month-ends for NDX.** The live calendar horizon starts in 2006, so those could not be
  replayed through the live host.

## 9. Verification fields

- **Tier:** 1 (research scripts, audit tests and docs only; no production code changed).
- **Agents used:**
  - eight auditors: live path, HPI, DV2/QPI, CORE5, macro (Compass and Tactical FI), ETF mean-reversion and EOM,
    Industry-ETF DV2 and TAA 2x, and hedges (CTC, VIXM, Trinity);
  - five independent reviewers: three for Tier A (quant, live parity, tradability) and two for Tiers B/C (quant,
    tradability).
- **Findings fixed:** none in code (audit only). Reviewer findings were adopted into this report, as listed in
  section 7.
- **Tests run:** 133 new synthetic audit tests in `tests/test_strategy_readiness_audit_*.py`, all pass.
  Existing HPI, QPI and CORE5 suites were re-run read-only and pass.
- **Real-data checks:** about 1,100 live-host replay decisions, over 200 invariance and truncation cases, and planted
  leak controls.
- **Residual risk:** as listed in section 8.

## 10. Tier B and Tier C scorecards (book strategies, none wired)

**Detailed findings files** (all under `results/research/strategy_readiness_audit_20260928/`):

| Scope | Findings file |
|---|---|
| Compass, Compass QQQ, Tactical FI | `tierb_macro/TIERB_MACRO_FINDINGS.md` |
| EOM flow, VOX/IYR, KIE/IHI | `tierb_etf_mr/TIERB_ETF_MR_FINDINGS.md` |
| Industry-ETF DV2, TAA 2x, linearity no-BTAL | `tierbc_dv2etf_taa2x/TIERBC_DV2ETF_TAA2X_FINDINGS.md` |
| CTC, VIXM, Trinity | `tierc_hedge/TIERC_HEDGE_FINDINGS.md` |

**Reviews:** `review_bc_quant/REVIEW_BC_QUANT.md` and `review_bc_trade/REVIEW_BC_TRADE.md`.

**Grading rules applied:** AM-01 (last-3-year window), AM-05 (0% idle cash is the house convention) and AM-06 (house
auction model under A8).

**Columns:**
- *LP:* "not wired" for every row.
- *House-model capacity:* the pod size at which extra opening- or closing-auction cost reaches 0.25 pp/yr at today's
  volume.

| Strategy | BC | TR | House-model capacity | Main reason |
|---|---|---|---|---|
| Inflation Compass | RC | RC | about USD 0.6-0.7M | Clean timing: the strict T5YIE rule is point-in-time on 152/152 ALFRED vintages. Selection is lucky-peak (826 trials, cited). |
| Inflation Compass QQQ | RC | RC | about USD 0.6M | As above. One QQQ share is 6% of NAV at USD 12K. |
| Tactical FI | **NR at cb29d4f; fixed 2026-09-28 (uncommitted), see section 14** | RC | - | At cb29d4f cash was credited at the full DGS3MO rate (under the house 0% convention the book window falls from 2.71% / Sharpe 1.06 to 0.90% / 0.36), withholding was 0%, and the strategy has been 100% cash since 2022-05. Fixed by owner decision: the cash sleeve is held in BIL, with 25% withholding. |
| EOM flow | R | RC | about USD 4M (MOC) | No look-ahead. Sandy 2012 is +0.018 pp. It trails T-bills over the last 3 years (2.5% vs 4.5%). The live route needs MOC timing (fix #23), a TLT borrow and margin. |
| Sector VOX/IYR | **NR** (full history); RC if quoted from 2013 | RC | about USD 0.39M | The house auction model adds +0.51 pp/yr to the full-history record (+0.12 last 3y). |
| KIE/IHI/XLC | **NR** | RC | about USD 64-93K | Auction cost +0.98 / +0.64 pp/yr (full / last 3y). The capacity is below the published USD 100K record. |
| KIE/IHI/XLC SMA200 | **NR** | RC | about USD 64-93K | +0.66 / +0.50 pp/yr. |
| KIE/IHI SMA200 | **NR** | RC | about USD 64-93K | +3.19 / +0.70 pp/yr. The 2007-12 record was not executable at the modelled cost. |
| Industry-ETF DV2 (research tier) | R | RC | about USD 193K | Clean. Hindsight: best of 4 ETF groups. The docstring figures are stale. At a constant USD 12K pod it loses -2.1 pp/yr to IBKR Fixed fees. |
| NDX NATR20 | RC | as NDX ATR | as NDX ATR | Section 5b. It has no live route. |
| TAA 2x QLD 1/N | RC | RC | about USD 170-220K | Adjusted-unit fees understate returns by 0.44 pp (conservative). The live route is blocked by the calendar window and the snapshot lacks QLD. |
| TAA 2x SSO 1/N | R | RC | about USD 170-220K | As above. |
| TAA 2x BTAL-QLD 1/N | R | RC | about USD 127K | BTAL-bound. |
| Linearity no-BTAL (QQQ) | R | RC | about USD 170-220K | Calendar-window blocker. One QQQ share is 6.2% of NAV at USD 12K. |
| Crisis Trend Core | **NR** | **NR** | about USD 48K | It does not run at HEAD (SHY starts 2002-07-26; loader starts 2002-01-01). Withholding is 0%. Gross exposure reaches 2.5x NAV, which a Reg T account cannot open. It shorts 11 ETFs. Corrected CAGR 1.05%, Sharpe 0.16. |
| VIXM backwardation | **NR** | **NR** (even at owner size) | about USD 3K | Withholding is 0% (-0.35 pp against a 2.1% CAGR). 71% of the edge comes from the first opening print. A one-session delay turns CAGR to -3.6%. The auction cost is 1.6-3.1 pp/yr at USD 12-30K. The fund has about USD 28M AUM and issues a K-1. |
| Trinity vol control 8% BIL | RC | R | above USD 11M | Clean (98/98 truncation). Only adjusted-unit fees are sensitive to future splits. The claim of a dividend double-count in the timing adapter was rejected: 0 of 4,861 decisions differ. Small-pod fee drag is -0.66 pp/yr at USD 12K. |
| MOSAIC | not audited | - | - | **Recommend revoking PM_READY to RESEARCH.** Its own docstring marks the validation invalidated; the corrected Sharpe is about 0.62; the Russell-1000 trim is unmeasured; it is in no book since 2026-09-28. This is an owner decision (registry tier). |

## 11. Book-number direction: which published numbers are optimistic and which conservative

**Optimistic** — the published number is higher than a house-consistent one:
- **Tactical FI:** cash credited at DGS3MO, and 0% withholding.
- **CTC and VIXM:** 0% withholding.
- **The three KIE/IHI pods and VOX/IYR:** auction cost.
- **NDX over the last 3 years:** the membership trim, -0.47 pp.

**Conservative** — the published number is lower than a house-consistent one:
- **Adjusted-unit fees:** TAA 3x/1N +0.32/+0.38 pp, DV2 +0.87, HPI +0.33/+0.37, QPI +0.36, KIE/IHI +0.36, TAA 2x QLD +0.44.
- **0% idle cash** for fund-size accounts.

At owner size (USD 12-30K), IBKR pays little or nothing on idle cash. So the 0% convention is close to realistic
there, and the conservative cushion mostly disappears.

## 12. Small-account friction at a constant pod size (owner-like pods)

These numbers re-price IBKR Fixed fees on each strategy's published orders, over the last 3 years, at a constant pod
size (`review_bc_trade/rbt05_published_commission_vs_owner_size.json`, about ±0.1 pp). Earlier figures were
measured on pods that compounded to USD 53-336K.

| Pod | USD 12K | USD 30K |
|---|---|---|
| Industry-ETF DV2 | -2.06 pp/yr | -0.76 |
| KIE/IHI (3 variants) | -0.89 to -1.42 | -0.29 to -0.48 |
| CTC | -0.95 | -0.45 |
| VOX/IYR | -0.91 | -0.31 |
| Trinity | -0.66 | -0.27 |

- For comparison, the live NDX pod is -1.0 to -1.4 pp/yr on a compounding path.
- IBKR Tiered cuts this by about 60-65%. VIXM is the exception, because the per-share rate binds there.
- **Whole shares at USD 12K:** one share is more than 5% of NAV for SPY in EOM (6.4%), QQQ (6.0-6.2%), SMH (5.6%) and
  the dispersion ETFs (5.3%).

## 13. Fix list additions from Tiers B and C (continue section 5)

| # | Fix | Where | Test | Affects |
|---|---|---|---|---|
| 17 | Build the live XNYS calendar with an explicit start (for example 1990-01-01) and end (today + 2 years). Check only the signal month's dates in the month-end resolver. | `alpha/live/scheduler_utils.py:68-75`, `:271-346` | A price frame starting 2000 resolves; a process-age test at +13 months | Live TAA pods from about 2031; any `serve` running more than 1 year; the TAA 2x / linearity routes today |
| 18 | CTC: set `history_start_date_str` to 2002-07-26 and add a real-data smoke test. Re-run the PM-readiness gate. | `strategies/tail_hedge/strategy_crisis_trend_core.py:116` | `run_variant()` completes | CTC tier |
| 19 | Use the house 25% withholding (or document an exemption) in CTC `:436`, VIXM `:190` and Tactical FI `:1033-1036`. Restate their records. | the three modules | Accounting policy asserts 25% | CTC, VIXM, Tactical FI numbers |
| 20 | Tactical FI cash convention: hold the cash leg as a BIL position (with withholding), or credit 0% as the house ledger does. Restate book numbers. The owner decides which. | `strategy_taa_tactical_fixed_income_ief_lqd.py:871-906` | - | Defensive books holding Tactical FI |
| 21 | Restate the dispersion pods and VOX/IYR net of the house MOO cost model, or add that model to their engine cost as a declared change. Add a house-model capacity line to capacity reports. | `alpha/engine/capacity_analysis.py`; the four modules | - | Fund-menu books holding them |
| 22 | Revoke MOSAIC's PM_READY tier (owner decision). | `alpha/strategy_registry.py:113-118` | registry test | Tier table |
| 23 | EOM MOC live route, if ever wired: submit on the fill day before the venue cutoff and expire at that cutoff (the close-auction version of fix #13). | `alpha/live/scheduler_utils.py:36, 416-421, 449-480` | 15:51 submit refused | EOM |
| 24 | Documentation: the Compass docstring `:17-23` and G-027 (T5YIE has been published the same evening since 2019-06; the strict rule is still correct); the Industry-ETF DV2 docstring figures `:22-32`; the G-033 margin wording. | docs and docstrings | - | - |
| 25 | If the TAA 2x variants are ever wired: add QLD and SSO to the `norgate_eod_etf_plus_vix_helper` export. | `scripts/export_norgate_snapshot.py:63-67` | - | TAA 2x |
| 26 | Compass or Trinity live routes: exclude the partial-month row, and derive month-end from the exchange calendar, not from "next session exists". | module decision helpers | - | Future routes |

## 14. Change made after the audit: Tactical FI cash vehicle (owner decision, 2026-09-28, uncommitted)

**Old behavior.** Positive cash was credited daily with the causal DGS3MO rate (ACT/365). This paid the full T-bill
yield with no fund fee, no execution cost and no tax, and it was about 71% of P&L since 2012-10. Dividends were
withheld at 0%.

**New default.**
- The frozen "Cash" sleeve is held as a real BIL position, traded like IEF and LQD: shares sized from Close_T, fill at
  Open_(T+1), 5 bp slippage.
- Residual cash earns 0%, as in the house ledger.
- Dividends are withheld at 25%.
- Before BIL's first bar (2007-05-30) the sleeve stays in 0% cash. This is conservative.
- The monthly IEF/LQD/Cash decisions and the frozen signal hash are unchanged.
- The legacy ledger is still selectable (`cash_vehicle_str="dgs3mo_accrual"`, withholding 0.0). It reproduces the HEAD
  module bit for bit (final NAV 273,841.58).

**Result.** From `results/research/strategy_readiness_audit_20260928/tierb_macro/tfi_bil_before_after.json`:

| Ledger | Book 2012-10 to 2026-08: CAGR / Sharpe / MaxDD | Full 2002-08 to 2026-08: CAGR / Sharpe |
|---|---|---|
| Before (DGS3MO accrual, 0% withholding) | 2.71% / 1.06 / -6.9% | 4.27% / 0.97 |
| Withholding 25% only | 2.55% / 1.00 / -7.1% | 3.86% / 0.88 |
| BIL only | 2.49% / 0.97 / -7.0% | 3.70% / 0.84 |
| **After (BIL + 25%, new default)** | **1.92% / 0.75 / -7.2%** | **3.03% / 0.69** |
| After, ALFRED point-in-time | 1.92% / 0.76 | 3.03% / 0.70 |

**Checks.**
- PM-readiness gate passes: capital ×2.000 exactly, total-return benchmark, deterministic
  (`results/research/pm_readiness/2026-09-28_235348`).
- Tactical FI suites 51/51 pass. They include new tests for the BIL mapping, zero-rate pre-inception cash and 25%
  withholding; the two legacy-number tests now pin the legacy ledger explicitly.
- Crisis and stress suites pass 36/36.
- `ASSUMPTIONS_AND_GAPS.md` G-029 is updated.

**Direction.** The 25% withholding on BIL is conservative if IBKR applies the US "interest-related dividend"
exemption for non-residents. Under that exemption the book result would sit between 2.5% and 1.9%.

**Books.** Defensive books that hold Tactical FI are restated downward when re-run. They are not re-run here.
