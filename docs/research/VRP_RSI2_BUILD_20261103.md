# VRP RSI2 sleeve: build, checks and shadow (2026-11-03)

Research only. Nothing here touches `alpha/live`, a broker route, a schedule or capital.

## Answer

The owner's locked rule (short SVXY in VIX contango with an RSI(2) dip boost, long VIXM in a crash, T-bills otherwise)
is built as an engine module, matches its Scout spec exactly, and is in forward shadow from 2026-11-04. It is not in a
fund book. It enters one only under the adoption rule in
[SPEC_FROZEN.md](../../scripts/research/vrp_rsi2_build_20261103/SPEC_FROZEN.md): 10 points from CORE5, measured fills
at or below 15 bp a side, twelve months of shadow without RED, an exit failsafe, and the K-1 question answered.

## What was built

| Piece | Where | Status |
|---|---|---|
| Frozen rule, checks, adoption rule | `scripts/research/vrp_rsi2_build_20261103/SPEC_FROZEN.md` | written before the engine run |
| Scout registration | ledger row 594, `vrp_rsi2_locked_retro_20261103`, parent row 588 | retro, 429-cell grid of what was already seen, 460 prior trials |
| Engine module | `strategies/vix_stuff/strategy_vrp_svxy_vixm_rsi2.py` | PM_READY (Candidate); PM plumbing only, adoption gates unchanged |
| Scout spec | `alpha/scout/specs/vrp_rsi2.py` | gate `vrp_rsi2` PASS, largest daily return difference 4.4e-16 |
| Unit tests | `tests/test_strategy_vrp_svxy_vixm_rsi2.py` | 16 pass: regimes, exact edges, halving, band (switches, VIXM alone, exact band), causality under truncation |
| Grade | ledger row 595 | WATCHLIST (no card; forward shadow): card stations S3-S6 not run |
| Shadow | ledger row 596, start 2026-11-04 | provider `alpha.scout.specs.vrp_rsi2:shadow_daily_return_ser`, MinTRL 35 months |

## The rule

After the close of T: ERV10 = 10-day SPY total-return volatility, eVRP = VIX - ERV10, TS = VIX / VIX3M, RSI2 on the
$SPX price close.

| State | Holding |
|---|---|
| TS < 0.98, eVRP > 0 | SVXY min(2 VIX/100, 1), x1.25 when RSI2 < 25 |
| TS < 0.98, eVRP <= 0 | SVXY min(VIX/100, 1), or min(1.5 VIX/100, 1) when RSI2 < 20 |
| 0.98 <= TS <= 1.00 | cash |
| TS > 1.00, eVRP <= 0 | VIXM min(1.25 VIX/100, 1) |
| TS > 1.00, eVRP > 0 | cash |

SVXY's target is halved for decision dates before 2018-02-28 (it was -1x then). A decision trades when a target turns on
or off or moves 5 points from the last traded target. Shares from the prior-close NAV and Close_T, filled at the next
open, 5 bp slippage and the IBKR fixed fee. Idle cash earns 0% in the engine.

## Numbers

2011-10-10 to 2026-08-19 unless stated. Research replica = Claude's weight simulator (weights held between trades, 5 bp
all-in, T-bills on idle cash).

| Measure | Value |
|---|---|
| Research replica | 15.5% / Sharpe 0.93 / Max DD -23.2% |
| Engine (whole shares, fees, drift), raw, 2012-01 to 2026-10-30 | Sharpe 0.932 vs replica 0.954 at 0% cash; daily correlation 0.9987 |
| Engine with T-bills on the decided cash share (approximation) | 15.0% / 0.90 / -23.1% |
| Plan Sharpe | about 0.70 (three quarters of the edge) |
| Average cash | 75% (67% since 2018) |

Review checks (`research_checks.py`, `vixm_checks.py`, `book_*.py` in the same folder):

| Check | Result |
|---|---|
| RSI2 boost vs 400 circular-shift placebos | boost 0.932, placebo median 0.868, p95 0.902, none reach it; gain in every era |
| Same exposure without the boost, sized up (2.4 x VIX) | 0.889 |
| Contango threshold 0.96 to 1.05 | 0.82 to 1.01, no plateau; 0.97 gives 0.82; Feb 2018 -22% at 1.02 and above |
| 0.98 vs 1.00 in the crises | identical Feb 2018 (-3.1%) and Aug 2024 (-5.8%); exit margins 0.62 and 0.52 VIX points |
| One session late | 0.62, Feb 2018 -22.7% |
| Costs a side 10 / 15 / 20 / 35 bp | 0.89 / 0.86 / 0.82 / 0.70 |
| VIXM legs at 20 / 45 bp (SVXY at 5) | 0.92 / 0.91 |
| VIXM filled at the T+1 close instead of the open | 0.89; Covid window +35% instead of +44% |
| VIXM P&L on the first day of an episode | 18% of the leg (the old VIXM pod: 71%) |
| No long-vol leg | 0.92 alone, but the book gain mostly disappears (below) |
| VX futures proxy 2006-07 to 2011-10 | +134%; 2008 +56% (one session late +55%), Oct 2008 +37%; May-Jul 2010 -15% to -19% |

Fund harness (`book_core5.py`, `book_doses.py`), current picks of
[FUND_PRODUCTS_OPT_20261102](FUND_PRODUCTS_OPT_20261102.md), 10 points moved from CORE5, 2008-03 to 2026-08, pod in
T-bills before 2011-10:

| Growth Plus | CAGR | Sharpe | Max DD | Share all / 2011+ / 2023+ | House checks |
|---|---:|---:|---:|---:|---|
| Pick | 20.3% | 1.48 | -11.3% | | |
| 5 bp | 20.9% | 1.51 | -11.2% | 0.88 / 0.83 / 0.46 | pass |
| 15 bp | 20.8% | 1.51 | -11.2% | 0.81 / 0.75 / 0.39 | pass |
| 20 bp | 20.7% | 1.50 | -11.2% | 0.77 / 0.72 / 0.35 | fail (share) |
| Half VIXY for VIXM | 20.7% | 1.50 | -11.2% | 0.80 / 0.74 / 0.42 | pass |
| 2020 crash in T-bills | 20.6% | 1.49 | -11.2% | 0.73 / 0.64 / 0.46 | fail |
| One session late | 20.5% | 1.48 | -11.2% | 0.56 / 0.49 / 0.37 | below 0.80 |
| No long-vol leg | 20.6% | 1.48 | -11.3% | 0.57 / 0.45 / 0.45 | below 0.80 |
| VX proxy before 2011 | 21.5% | 1.54 | -11.6% | 0.96 / 0.83 / 0.46 | not run |

Growth: 18.1% / 1.53 / -10.0% to 18.7% / 1.58 / -9.9%, share 0.88 / 0.82 / 0.49, all checks pass at 5 bp. Pro rata
funding falls under the return floors (Growth 17.6%, Growth Plus 19.6%).

## Corrections to the locked-spec document

- 0.98 does not change the crises it was meant to fix (above); it is harmless, not a fix.
- Average cash is 75%, not 46%. The MR capsule has no account, so it is not a "live DV2/HPI" precedent for delay.
- Capacity: a VIXM entry at VIX 30 is 11% of VIXM's daily dollar volume at a $10M fund and 29% at $25M (10% pod).
  Above about $5M of fund, half-size VIXY in place of VIXM keeps the book checks.
- Tax: ProShares' qualified notice of 2026-02-26 covers SVXY and VIXM for the 1446(f) ten-percent exception; brokers
  may rely on it but need not, and it is renewed every 92 days. "Fully exempt" and "no ECI" were not verified.
- Section 4 of that document predates the Flow Vote picks; the fund books are in FUND_PRODUCTS_OPT_20261102.
- Its RSI, gold and growth tables come from a script outside the repo; this folder's scripts replace them.

## Independent review (2026-11-03)

A read-only review found no look-ahead and no dropped trade; it re-ran engine and spec (4.4e-16) and checked fill
timing, prior-close sizing and the cancelled-fill guard. Fixed: the 0.98 edge test now uses an exact 0.98 ratio; band
tests for VIXM alone, a direct switch and a move equal to the band; the shadow provider now raises on a missing input
(the band alone would skip it silently) and drops today's row (Norgate can hold a partial bar during market hours).
Kept, with reasons:

- SVXY halving is keyed on the decision date, as frozen. ProShares' change took effect at the open of 2018-02-28, so
  the decision of 2018-02-27 (no trade that day) and the position bought 2018-02-23 ran at half the intended exposure for
  one session (-0.20% that day). History only; live never halves; changing it would change a frozen rule for nothing.
- Signals use Norgate's market-day padding, a causal carry of the last close on a session an index skipped (e.g.
  2012-12-24); there is no other fill.
- Fees are charged on adjusted shares in both engine and spec (SVXY split factors 0.5-4, VIXM 0.25-1); immaterial at
  $100K, shared, so the gate cannot see it.
- The engine's own signal-truncation audit is off in `run_daily` by default; causality is pinned by the unit test.

## Caveats

See the section in [book-strategy-caveats.md](../strategies/book-strategy-caveats.md). The main ones: the book gain
leans on the 2020 crash; one missed session costs a third of the Sharpe; VIXM is small (about $28-38M of assets in the
2026-09 audit) and could close; SVXY can lose about half its value overnight if VIX futures double; the 2006-2011 test
uses a same-close futures proxy.

## PM_READY integration (2026-11-03)

Owner-requested promotion to Portfolio Manager eligibility. The registry now includes the frozen engine module at
`MaturityTier.PM_READY` (Bench label `Candidate`). No strategy rule, fund weight, live route, scheduler or broker
configuration changed. The adoption rule above remains required before adding it to an actual fund book.

The common `run_variant` accepts the PM start/end dates and allocated capital. The earliest supported start is
2011-10-05, after SVXY's first close; an earlier book start fails loudly on missing ETF bars. A 2008-start fund book
requires a separately approved pre-inception cash policy before using this pod.

The PM review identified a missing-SPX validation gap. It was fixed on 2026-11-04 with owner authorization:
all supplied SPX closes, including RSI warmup history, must be finite and positive before smoothing. Invalid data
raises a dated ValueError. The frozen valid-input RSI arithmetic and zero-loss fallback remain unchanged.
See the validation section below for the source audit and replay evidence.

Verification: `tests/test_strategy_registry.py` checks exact PM_READY status, PM config acceptance and allocated
capital, and exclusion from the live release allowlist. The required expensive promotion gate and actual PM smoke
run passed (completed 2026-11-04, Israel local time).

| Promotion check | Result |
|---|---|
| Full-history capital | PASS: USD 100,000 ends at 661,594; USD 200,000 at 1,323,702; ratio 2.001 |
| Benchmark | PASS: stored SPX total-return CAGR 15.42%, mapped SPXTR 15.42% |
| Determinism | PASS: repeated terminal equity 661,594.2924, identical |
| Actual Portfolio Manager smoke | PASS: one pod, 459 sessions, 2025-01-02 through 2026-10-30, no artifacts or fund config written |
| Regression suite | 187 passed: strategy rules, registry, Portfolio Manager and analyzer tests |
| Bench catalog | tier 2, label `candidate`, common runner discovered |
| Independent quant-pitfalls review | No PM registration blocker for valid dates; history and missing-SPX limitations disclosed above |
| Verification triage | Tier 1; quant-pitfalls reviewer used; task-owned tracked diff whitespace check clean |

Machine-readable promotion evidence:
[`pm_readiness.json`](../../results/research/pm_readiness/2026-11-04_000324/pm_readiness.json).
These checks certify PM integration, not an investment edge or satisfaction of the fund adoption rule.

## SPX source-validation fix (2026-11-04)

Scope: the local VRP RSI helper, regression tests and research verification only. Scout calls the same feature
function and inherits the rejection. No shared indicator, live route, capital allocation or strategy threshold changed.
Old behavior accepted invalid source closes into smoothing; new behavior rejects the entire supplied frame at its
first invalid SPX date. This includes warmup because recursive smoothing retains earlier history. A bad later source
row aborts the precomputed run; it is not imputed or used to select earlier trades.

For every supplied SPX close P_t, require isfinite(P_t) and P_t > 0. After validation the existing formula is unchanged:
Delta_t = P_t - P_(t-1); G_t = EWM(max(Delta_t, 0), alpha=1/2, adjust=False);
L_t = EWM(max(-Delta_t, 0), alpha=1/2, adjust=False); RSI_t = 100 - 100/(1 + G_t/L_t).
The frozen convention sets RSI to 50 when this valid-input formula is undefined, including zero loss and initialization.
This convention is preserved; it is separate from rejecting an invalid source price.
Close_T decisions, prior-close NAV/price share sizing and Open_(T+1) execution are unchanged.

Tests first reproduced 15 failures: NaN, positive/negative infinity, zero and negative SPX closes at the first,
warmup and final supplied rows were accepted. All now raise with the source date. A separate test preserves the
valid rising-price zero-loss fallback. The four-suite regression run passed 203 tests; the independent quant-pitfalls
review found no material issue and independently passed all 32 strategy tests. Verification triage is Tier 1.

Source audit: all 3,981 supplied sessions from 2011-01-03 through 2026-10-30 have valid SPX closes. A separate
Norgate request with PaddingType.NONE found no missing/invalid SPX bars on those dates either. This is evidence
for the saved data snapshot through that end date, not a guarantee about future downloads.
The original source and exact pricing snapshot are retained under `results/research/vrp_spx_validation_20261104/`.
The reproducible checker is `scripts/research/vrp_rsi2_build_20261103/spx_validation.py`.
Full historical feature tables, engine result frames (including daily NAV) and decision frames before/after are
exactly equal: 3,790 scoring sessions, 1,010 decision rows, terminal NAV USD 659,684.5383377969 from USD 100,000.
This replay ends on 2026-10-30; its terminal value is not comparable to the earlier PM gate's later end date.
No separate fill-ledger comparison is claimed. Machine-readable evidence:
[`verification.json`](../../results/research/vrp_spx_validation_20261104/verification.json).
The checker reuses the retained pricing/SPX snapshots on reruns. Review's minor reproducibility concern was resolved
with that reuse policy; saved snapshots were reloaded and SPX coverage checked on all 3,981 dates.

Residual limitations: start dates before 2011-10-05 remain unsupported, and fund adoption/forward execution gates
remain unchanged. This fix does not audit every other source field or change Norgate padding policy.

## Bench analyzer hooks (2026-11-04)

The module now exposes the four analyzer hooks the other PM_READY pods have: `build_capacity_analysis_inputs` (MOO,
`MOO_ETF_PROXY`), `build_execution_timing_analysis_inputs` (prior-close decisions, `taa_rebalance` risk model, default
cell T+1 Open), `build_stress_test_context_dict` and `build_stress_test_strategy_obj` (registered in
`alpha/engine/crisis.py`, full-history replay because the band is stateful).

Two engine-side changes, both generic and both identical on the main run:

- The band now runs inside `iterate` on the run's own traded targets (`band_trade_bool`, shared with
  `rebalance_flag_ser`). A run that starts later, such as a stress launch, starts flat there and trades straight to its
  target. From the first decision this is the precomputed `rebalance_bool`; the only date that differs is the final
  close, which has no next session to fill.
- `VrpRsi2TimingStrategy` (Timing analyzer only) orders the change from the last traded shares and skips the holding
  check, because the analyzer queues delayed fills outside the strategy. The vanilla class keeps the holding check.

Proof: fresh `run_variant` vs the saved engine run, largest daily return difference 0.0 over 3,790 sessions; Scout
identity gate PASS 4.44e-16, same trade dates; 36 strategy tests, 122 registry tests and 127 Bench tests pass (the
Bench capacity-hook count moves from 57 to 58). PM readiness rerun: capital x2 = 2.001, benchmark gap 0.00%,
determinism identical (`results/research/pm_readiness/2026-11-04_084716/`).

Timing (entry rows, exit columns; Sharpe as the analyzer computes it, base 13.38% a year = the engine run):

| Entry \ Exit | T+1 Open | T+1 Close | T+2 Open | T+2 Close |
|---|---:|---:|---:|---:|
| T+1 Open (base) | 0.92 | 0.87 | 0.81 | 0.82 |
| T+1 Close | 0.84 | 0.79 | 0.74 | 0.76 |
| T+2 Open | 0.80 | 0.73 | 0.66 | 0.65 |
| T+2 Close | 0.74 | 0.67 | 0.61 | 0.60 |

A full session late costs about 0.26 Sharpe (0.92 to 0.66) and 4.2 points a year; the closing auction of the same day
costs 0.13. Max DD moves from -23.1% to -25.6%. This repeats the earlier one-session-late check (0.62).

Stress (fresh launch 5-63 sessions before each event; event return median across launches):

| Event | Pod | $SPX TR | Pod event DD | Worst day |
|---|---:|---:|---:|---:|
| China/oil 2015-08 to 2016-02 | -4.4% | -12.1% | -9.0% | -3.1% |
| Volmageddon 2018-02-02 to 02-09 | -3.4% | -7.1% | -3.4% | -2.5% |
| COVID 2020-02-20 to 04-07 | +41.5% | -21.2% | -23.1% | -11.4% |
| Inflation bear 2022 | -3.9% | -24.0% | -11.1% | -3.4% |
| Tariffs 2025-02 to 04 | -7.5% | -7.5% | -8.7% | -2.1% |

Launch date barely matters (spread at most 7 points, COVID). In 2018 the rule left SVXY at the 2018-02-05 open,
before that evening's collapse: real, but one session from -23% (see the caveats). The COVID gain comes with a -23%
drawdown inside the event and an -11% day. 2025 is the one event where the pod did no better than the index.

Capacity (house MOO model, `MOO_ETF_PROXY`: 40 bp of impact at 1% of ADV, square root; judged on the recent five
years, as current volumes are what a fund trades into):

| Pod size | SVXY cost a side (5 bp + impact) | Pod return a year (recent 5y) | Sharpe |
|---:|---:|---:|---:|
| $0 (engine) | 5 bp | 8.2% | 0.73 |
| $250K | 13.5 bp | 6.5% | 0.60 |
| $500K | 18 bp | 5.6% | 0.52 |
| $1M | 24.5 bp | 4.3% | 0.42 |
| $2M | 34 bp | 2.3% | 0.25 |

- The cost is SVXY, not VIXM: 84 SVXY orders a year, turnover about 19 times NAV a year, against ADV about $106M.
  VIXM trades about once a year and costs 0.3% a year at a $1M pod.
- Against the adoption line of 15 bp a side, the pod holds to about $300-400K, a $3-4M fund at a 10-point sleeve. At
  the fund plan's $10M (a $1M pod) the model reads about 24 bp a side, over the line.
- The analyzer's Recommended and Break-even read "none" and "below $50K" for a different reason: its rules measure cost
  against the excess return over $SPX TR, and the pod (13.4%) is below $SPX (15.5%) before any cost. That framing
  suits stock pods, not a diversifier; the table above is the usable number.
- The model assumes the opening auction takes the whole order. Worked orders in the first minutes could do better;
  that is not measured for SVXY, and it changes the fill timing that the rule was tested on.

Artifacts: `results/research/vrp_rsi2_analyzers_20261104/` (gitignored).
