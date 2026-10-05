# Fund products: close the DEFENSIVE and GROWTH products from all strategy families (frozen plan)

Written 2026-09-30 before any product book of this study was built or read. Amendments go to the log at the end, dated,
with reason and whether before or after a result. SHA-256 in
`results/research/portfolio/fund_products_20260930/experiment_ledger.jsonl` (this worktree).

## 0. Owner task and what is already known

Owner (Hebrew, 2026-09-30): close two products, DEFENSIVE and one offensive product (GROWTH; the earlier "growth"
and "aggressive" are one product), each with options; weigh ease of trading and AUM; integrate all earlier work
(fund product menu 2026-09-23, growth shelf 2026-09-24/26, shelf rebuild and defensive v2 2026-09-29, growth /
aggressive study with extension and verdict 2026-09-30, client-split study 2026-09-30); many strategies are variants of
one engine (NDX-VXN vs NDX-NATR20, DV2 vs its floor/ADV variants, HPI variants, the 12/12/12 capsule): decide, per
family, one variant or a split for strategy diversification; more CAGR is preferred for a future 2/20 fund but must stay
optimal for investors; deliver bench portfolios named fund_*; the owner delegated the quant design.

Known before this freeze (all in-sample; this is not a blind test): Compass/Inflation Compass is a lucky peak (demoted);
MOSAIC dead; QPI demoted; NDX-RM post hoc (fails its twin test without 2020 and 2026); stock MR adds <= +0.05 Sharpe
before costs and nothing after +5 bps per side (fund menu follow-up); ETF MR (sector, dispersion) is a de-risker; growth
books hit a Sharpe ceiling ~1.4-1.5 and extra CAGR above G3 is mostly leverage on the same factor; the defensive core
CORE5 + BTAL_QQQ is the deployable champion, the four-pod core (+ EOM + DV2-IND) the target; TAA and BTAL_QQQ share the
Defense First engine (corr 0.76 between the growth pod and the core); Tactical FI is not live-ready.

## 1. Inputs

As the growth study: shelf-rebuild sleeve runs (HEAD f9ad358, $1M each, end 2026-08-19); LONG 2008-03-04 -> end with the
validated 2008 TQQQ/BTAL proxy before 2012-10-02 and the A2 DV2-IND fill; EXACT 2012-10-02 -> end; FAIR CASH main frame;
BIL as T-bills; gross is primary (no fee is charged today), 2/20 net reported (daily accrual, yearly payment, HWM).

## 2. Families (every sleeve of the inventory)

| family | variants (tier) | route |
|---|---|---|
| TAA (Defense First, Nasdaq fallback) | taa3x (W), taa3x_1n (W), taa2x_1n (P), taa_1n_qld (P), taa_1n_sso (P) | monthly |
| NDX momentum | ndx_vxn (W), ndx_atr (W), ndx_natr20 (S) | monthly |
| Stock mean reversion | dv2 (W), dv2_adv (S), dv2_floor (S), hpi_vote (W), hpi_ibs_rsi (W) | daily |
| ETF mean reversion | etf_dv2 (S), downshock (P), disp (P) | daily |
| Defensive macro | core5 (P), btal_qqq (W), trinity (P, daily band) | monthly (trinity daily) |
| Flows | eom_flow (P; month-end MOC + TLT short, not live-tradable) | month-end |
| T-bills | BIL | hold |

Excluded, reported only: compass, compass_qqq (lucky peak), tactical_fi (not live-ready; frozen/ALFRED issues),
taa_lin_qqq (BTAL_QQQ's no-BTAL twin, dominated by it in the BTAL check), disp_xlc and disp_xlc_sma (no LONG history),
NDX-RM (post hoc). W = WIRED, P = PM_READY, S = shadow.

## 3. Stage 1: the family block (one variant or a split)

For each family the variants are first screened: a variant is viable when its LONG excess CAGR over BIL is > 0 in blocks
B, C and RECENT. Two viable variants with daily return correlation >= 0.97 count as one (keep the more mature tier, then
the higher LONG Sharpe). A viable variant is dropped as dominated when another viable variant beats it on >= 90% of the
paired bootstrap paths (2,000 paths, block 63, seed 20260929) by Sharpe. The family block is the EQUAL-capital split of
the remaining variants, reset annually (strategy diversification by default; a single variant only when the data says
the others are clearly worse). Two blocks per family: DEPLOYABLE (only W/P variants) and TARGET (shadow allowed).
Reported per family: each variant, the split, pairwise correlations (all days and S&P 500 worst-5% days), and the split's
drawdown relative to the worst and best single variant (the model-risk insurance).

## 4. Stage 2: products from family blocks (resampled optimisation)

Blocks: TAA, NDX, SMR (stock MR), EMR (ETF MR), DEF (defensive macro), EOM, CASH (BIL).
Lines (each a complete construction):
- LOW-TOUCH (monthly pods only): TAA, NDX, DEF (deployable, monthly variants only), CASH.
- MAIN (daily MR allowed, deployable): + SMR, EMR.
- TARGET (what becomes possible once shadow pods and EOM are wired): all blocks, TARGET variants.
Products (risk rules = the owner's rungs of the earlier studies, built one step below the hard limit):
- DEFENSIVE: maximise LONG Sharpe s.t. LONG max DD >= -7% and bootstrap P(max DD < -10%) <= 10%.
- GROWTH: maximise LONG CAGR (gross) s.t. LONG max DD >= -17% and P(max DD < -20%) <= 15%.
- GROWTH PLUS (the "more growth" option): maximise LONG CAGR s.t. LONG max DD >= -22% and P(max DD < -25%) <= 15%.
Policy caps: TAA block <= 70% of a product; every weight on a 5% grid (10% for MAIN and TARGET lines, to keep the grid
tractable); the grid includes zero weights (a family earns its place or gets 0).
Resampled weights: on each of 200 bootstrap paths (block 63, seeds 20260930+) the grid optimum of the product's rule is
found using the path's metrics (the breach constraint is replaced on a path by its hist-DD rule applied to that path,
i.e. path max DD >= the product's build limit); the product's ROBUST weights are the average of the path optima
(paths with no feasible mix are counted and reported). The historical grid optimum and its plateau (all grid mixes
within 0.3 pp CAGR / 0.03 Sharpe of it) are reported beside it. In the optimisation stage the mixes use constant daily
weights (a disclosed approximation); every reported product figure uses the house pod model with an annual reset.
The robust weights are rounded to 1% (largest remainder) and then must pass the product's own rule on the historical
LONG window with the 10-seed mean breach; if they fail, they are scaled toward CASH (GROWTH) or toward DEF (DEFENSIVE)
in 5% steps until they pass (reported).

## 5. Champions and the decision

Champions: GROWTH vs G3 (TAA3x 50 / NDX-VXN 50) and the growth verdict (TAA3x-1N 38.4 / NDX-VXN 25.6 / CORE5 18 /
BTAL_QQQ 18); GROWTH PLUS vs the aggressive verdict (57.4 / 24.6 / 9 / 9); DEFENSIVE vs CORE5 60 / BTAL_QQQ 40 and the
four-pod core (TARGET line). A new product replaces its champion only if it beats it on >= 80% of the paired paths by the
product's objective, with a 10-seed breach no worse than the champion's and a better EXACT-window objective; otherwise the
champion is the product (and the new book is shown as the alternative).
The final recommendation per product (one line to launch now, one target line) is judgement on top of these results,
weighing ease (monthly vs daily, pods, trading days), AUM capacity ($10M/$25M/$50M, growth-shelf route model), readiness
(what is wired) and the robustness checks; it is labelled as judgement.

## 6. Robustness (reported for every final product and champion)

10 seeds; frames: house 0% cash, +5 bps per side, unscaled BTAL proxy, HPI live gap, EXACT window, block 21 and 126 (each
re-running stage 2 for the LOW-TOUCH and MAIN lines of GROWTH and DEFENSIVE); forward split (resampled weights from paths
of 2008-03..2017-06 only, evaluated on 2017-07..end, and the reverse); factor alpha gross and net (QQQ alone; QQQ/IEF/GLD/
DBC/UUP; + QQQ 200-day rule; weekly, Newey-West 4 lags); crises and co-falls; correlation to S&P on its worst days;
capacity and fee income; year by year.

## 7. Deliverables

Bench portfolios `portfolios/fund_defensive.yaml`, `fund_growth.yaml`, `fund_growth_plus.yaml` (and `*_target.yaml` for
the target lines when they differ); a Hebrew report that integrates all earlier studies (what stays, what goes, and why),
lists what was tested and did not help, and states the decision; an independent review before the report is final.

## Amendment log

- A0 (2026-09-30, BEFORE any product book was built; owner request mid-run): the report shows, per product, a MENU of
  named options with their trade-offs and requirements (pods, monthly/daily, trading days, what must be wired, AUM
  capacity), and then the recommendation. The options are defined here, each by the same resampled construction of
  section 4 with one change:
  DEFENSIVE menu: D-LAUNCH (LOW-TOUCH line, the section-4 rule); D-MAIN (MAIN line); D-TARGET (TARGET line);
  D-CALMER (LOW-TOUCH, build limit -5% instead of -7%, P(< -7%) <= 10%); D-RICHER (LOW-TOUCH, maximise CAGR instead of
  Sharpe, build limit -7%, P(< -10%) <= 10%); D-SIMPLEST (CORE5 60 / BTAL_QQQ 40, the defensive champion).
  GROWTH menu: G-LAUNCH (LOW-TOUCH, GROWTH rule); G-PLUS (LOW-TOUCH, GROWTH PLUS rule); G-MAIN (MAIN, GROWTH rule);
  G-TARGET (TARGET, GROWTH rule); G-SHARPE (LOW-TOUCH, maximise Sharpe with the GROWTH risk limits and CAGR >= G3's
  LONG CAGR); G-SIMPLEST (G3). Every option gets the full metric set (CAGR gross and net, Sharpe, Sortino, max DD,
  Calmar, CVaR 5% daily and monthly, worst month/year/12 months, beta and crisis correlation, 10-seed breach, crises,
  capacity, ease). The recommendation picks one option per product to launch now and names the target.
- A1 (2026-09-30, AFTER the first stage-2 run of the main frame; post-result correction of a spec error): the SPEC's
  "Sharpe" objective (house convention: zero risk-free rate) is degenerate once T-bills are a block: BIL alone has a
  zero-rate Sharpe of ~2.9, so DEFENSIVE and D-CALMER came out 95-97% T-bills. Every Sharpe objective and Sharpe
  constraint in this study (DEFENSIVE, D-CALMER, G-SHARPE) now uses the EXCESS Sharpe over BIL (mean of daily returns
  minus BIL, over their standard deviation), the metric the defensive-v2 study already found ranking-stable. CAGR
  objectives are unchanged. The first run is kept as stage2_A0run.json.
- A2 (2026-09-30, AFTER the first forward-split results; post-result, labelled, nothing replaces the frozen products):
  the forward split showed the section-4 GROWTH product (whose TAA block is the equal split of four non-dominated TAA
  variants: taa3x, taa3x_1n, taa_1n_qld, taa_1n_sso) losing to G3 and the growth verdict out of sample in both halves.
  Two of those four are 2x-leveraged no-BTAL cousins, so that split mixes leverage levels rather than versions of one
  strategy. Added as a labelled comparison: the same stage-2 construction (LOW-TOUCH and MAIN lines, GROWTH, GROWTH PLUS
  and DEFENSIVE) with the TAA block replaced by (a) TAA-3X = equal split of the two 3x BTAL variants {taa3x, taa3x_1n},
  and (b) TAA-1N = taa3x_1n alone. Reported beside the frozen products; any preference for them is judgement.
- P2 (2026-10-01, implementation note, no rule change): section 4's "scale toward CASH/DEF until the rule passes" was
  missing from the first evaluate.py run (which crashed before writing results on an alias bug); implemented as written
  (DEF = the monthly CORE5 + BTAL_QQQ split). The alias map now skips the sensitivity copy tactical_fi_frozen.
- A3 (2026-10-01, AFTER all results; judgement, labelled): the final bench portfolios are clean versions of the chosen
  options, because resampled averages leave 1-3% slivers that add pods without effect: slivers below 3% are folded into
  the nearest wired pod of the same family (or dropped and the rest scaled up), the four-variant TAA slivers of D-RICHER
  become the two wired 3x variants. Each clean version is re-checked with the full rule (hist build limit, 10-seed
  breach), the champion test, +5 bps and the forward halves (simplify.py); a clean version that fails is not used.
  A3 outcome: the first clean fund_defensive failed its rule (P(-10%) 11.5%) and was scaled toward cash to 30% BIL (passes,
  4.4%); the first clean growth-with-MR book dropped the 3% defensive slice and sat at 14.3% breach, so the slice is kept
  (9.5%). Names: fund_growth (monthly), fund_growth_mr (adds daily stock MR, same risk rung), fund_growth_plus (the more
  risk rung = the aggressive verdict), fund_defensive, fund_defensive_target (the four-pod core, once EOM and DV2-IND are
  wired).
- A4 (2026-10-01, AFTER the Tier-1 review REVIEW_TIER1.md; decisions follow the frozen rules): (1) GROWTH launch = the
  growth verdict (TAA3x-1N 38.4 / NDX-VXN 25.6 / CORE5 18 / BTAL_QQQ 18): the A2/A3 book "G-LAUNCH-3X" fails the SPEC-5
  champion test (breach 13.5% vs 11.4%; excess Sharpe equal; it is the verdict with more leverage) and is dropped from
  the products (kept in the menu as "tested"). (2) The growth book with stock MR (fund_growth_mr) is a TARGET line, not a
  launch product: its edge is gone at ~8-10 bps per side extra cost and equals the monthly book with the HPI gap on both
  HPI variants plus 5 bps; only the 2017-26 forward half is a clean out-of-sample win; the reality check over the growth
  candidates gives p 0.06 before costs, 0.20-0.44 at +5 bps. Gate: measured live slippage <= 3-4 bps per side and the HPI
  align-live fix landed. (3) DEFENSIVE launch = CORE5 60 / BTAL_QQQ 40 scaled with 20% BIL (the simplest book that
  passes the rule; the earlier "fund_defensive" = D0 + 22% growth sleeve + 30% cash is kept as the labelled "more return"
  defensive option; its gain equals a naive DEF + G3 + cash mix). Corrections to earlier statements: the 69% beat share
  was the CAGR share (76% by excess Sharpe); moving the defensive book toward cash was a rule change (SPEC 4 said toward
  DEF, which cannot pass). (4) Engine concentration: the Defense First engine (TAA3x, TAA3x-1N, BTAL_QQQ) carries ~70-77%
  of the growth books' risk; the two 3x TAA variants (corr 0.92) are not a diversification of it.

## A5 (2026-10-01, owner questions; post-result exploration, labelled)
Recorded before running round2.py. Owner: only the three wired Defense First strategies (TAA3x, TAA3x-1N, BTAL_QQQ) count as the TAA family; the 2x QLD/SSO variants leave the menus, and rule outputs that sprinkle slices of every TAA variant into the defensive book (D-LAUNCH, D-CALMER, D-MAIN) leave the menu. New descriptive checks, none of which replaces an A4 product by itself:
- Defensive third pod: D0 (CORE5 60 / BTAL_QQQ 40), then CORE5 + BTAL_QQQ + DV2-IND, + downshock, + both, + EOM with each, and CORE5 + TAA3x + (DV2-IND | downshock); equal weights, then the smallest T-bill share (5% steps, max 50%) that passes the DEFENSIVE rule (DD >= -7%, P(DD < -10%) <= 10%); champion test vs D0 by excess Sharpe (80%).
- Growth at ~22%: compare three routes at the same gross CAGR (20/22/24%): more TAA3x-1N inside the verdict structure; daily margin on fund_growth; margin on fund_growth_mr; margin on fund_defensive_target. Margin borrows at T-bill + 1.5%. Judge by DD, P(DD < -20/-25/-30%), +5 bps, the EXACT window, and since-2023.

## A6 (2026-10-01, owner request: "סדר בהגנתי", streamlined menus, a fresh pass over everything)
Recorded before a6.py runs. Supersedes the A4/A5 menus; the A4 champions stay the defaults to beat. Post-result in spirit (A5 results and the stand-alone engine profile were seen first), so every rule below is fixed now and applied mechanically.

Engine profile seen before freezing (LONG 2008-03..2026-08, main frame): downshock lost 15.5% in the GFC window, max DD -20%, correlation +0.49 with the S&P 500 on its worst 5% days (a dip buyer); DV2-IND ~0 in the GFC window (pre-2010 = DV2 deep study research run, idle Dec 2008-Mar 2009, earns 0%), corr -0.01 on bad days; EOM +19% GFC, +15% 2022; stock DV2 -30% COVID, HPI -15% COVID. BTAL_QQQ/TAA before 2012-10 are the synthetic proxy.

Rules (bootstrap: 10 seeds x 2,000 paths, block 63; a rule passes only if both the 10-seed mean and the worst seed are within the cap):
- DEF: full-sample max DD >= -7% and P(DD < -10%) <= 10%.
- CALM: max DD >= -5% and P(DD < -7%) <= 10%.
- Crisis floor (owner concern, all defensive slots): GFC window >= -1%, 2022 bear window >= -1%, worst of the five named crises >= -5%.
- GROWTH limits reported for growth rows: GROWTH (DD >= -17%, P(<-20%) <= 15%), GROWTH PLUS (DD >= -22%, P(<-25%) <= 15%).

Cores: C_L = CORE5 60 / BTAL_QQQ 40 (needs CORE5 only); C_N = CORE5 / BTAL_QQQ / DV2-IND thirds (one more wiring); C_T = CORE5 / BTAL_QQQ / EOM / DV2-IND quarters. Cash = BIL, 5% steps up to 60%.

Defensive slots (the owner's five):
1. Launch: default C_L + the smallest cash passing DEF + floor. Challengers: CORE5 50/BTAL_QQQ 50; C_L 90 + stock DV2 10; C_L 90 + HPI 10 (all wired at launch).
2. Next step (one new wiring): default C_N + smallest passing cash. Challengers: CORE5 40/BTAL_QQQ 27/DV2-IND 33; C_L 75 + DV2-IND 25; C_L 80 + DV2-IND 20; downshock in place of DV2-IND (thirds; C_L 80 + 20; C_L 90 + 10).
3. Very defensive: the launch winner's core + the smallest cash passing CALM + floor (the next-core version is reported).
4. More return: the launch winner's core (no cash) blended with the growth launch book as one unit at g in {5..50%} (5% steps), plus cash 0..50%; pick the highest-CAGR pair passing DEF + floor. If its CAGR gain over the launch slot is under 1.0 pp, build it on C_N instead and mark it as needing DV2-IND.
5. Ideal target: default C_T + smallest passing cash. Challengers: C_T with downshock 5% / 10% taken pro rata; CORE5 30/BTAL_QQQ 20/EOM 25/DV2-IND 25; CORE5/BTAL_QQQ/EOM/downshock quarters; CORE5/BTAL_QQQ/EOM thirds.
Challenge test (replaces a default only if ALL hold): passes the slot rule + floor; excess Sharpe above the default on >= 80% of 2,000 paired paths; slot breach (10-seed mean) no worse; EXACT-window (2012-10+) excess Sharpe higher; +5 bps excess Sharpe not lower; excess Sharpe higher in both halves (split 2017-06-30). If several pass, the highest paired share wins.
Downshock sweep (reported, decides the owner's question): downshock at 5/10/15/20/25% pro rata inside C_L, C_N and C_T. Downshock enters a product only through a challenge above.

Growth (streamlined; every levered and unlevered option kept):
1. Launch = fund_growth (unchanged). 2. More return without leverage = fund_growth_plus (unchanged). 3. MR target = fund_growth_mr (unchanged, gated), plus its 22% levered variant.
4-6. 22% routes, one per stage: s x fund_growth + (1-s) x core, levered to 22.0% gross with daily margin at T-bill + 1.5%. Default s = 1/2, challengers s = 1/3 and 2/3 (same challenge test, with P(DD < -20%) as the breach, EXACT excess Sharpe and +5 bps CAGR). Launch-stage default is fund_growth levered alone, and the C_L mixes are its challengers. Next stage uses C_N, target stage uses C_T. Cores enter the mixes without cash (no borrowing to hold cash).
7. Max-leverage route: the target-slot book levered to 22% (reported, no challenge).
Leverage L = the smallest value on a 0.01 grid that reaches 22.0% full-sample gross CAGR. Reported: financing at T-bill + 0.5% / 2.5%, a worst-case Reg-T initial requirement (3x ETF sleeves at 75%, everything else 50%), and capacity divided by L.
Outputs: report/a6.json. A Tier-1 review follows before publication.

### A6-b (post-result, labelled; reported only, replaces nothing)
Recorded after a6.py ran. Four questions the A6 results raised: (1) next-step book vs the launch winner head to head; (2) HPI kept alongside DV2-IND (C_N + HPI 10%; launch book + DV2-IND 25% / 33%), challenge-tested against C_N; (3) the HPI launch under the HPI live-gap frame vs 60/40 + 5% cash and vs the DV2 alternative; (4) the target's downshock 5% paired share (0.801) re-drawn on seeds 1-4.

### A6-c (post-review, labelled; recorded before the re-run)
Tier-1 review of A6 (phase 1) accepted. Changes:
1. Consistency with A4: stock mean-reversion pods (DV2, HPI, HPI-RSI) carry the A4 gate everywhere, including the defensive launch slot: they may not take a launch slot until live slippage on their routes is measured at <= 3-4 bps per side (the HPI same-open refill fix landed on main 2026-09-28, 8b21a2e; HPI-RSI was demoted to PM_READY on 2026-09-30, 693ec8e). Their launch challenges are still run and reported. The best passing one is reported as the gated launch upgrade, with its paired share at +5 and +10 bps. Until the gate is met, the launch is the default (C_L + smallest passing cash). Very defensive and more return then build on C_L, per the A6 rule "the launch winner's core".
2. Bug fix (spec deviation): the 22% mixes use the literal cores C_L, C_N and C_T, as A6 states, not the slot winners. The max-leverage route stays the target-slot book.
3. Bug fix (house convention): margin is financed at DTB3 (prior observation) + spread, ACT/360, as negative cash in the frames, not BIL + spread.
4. The tail cache key includes a fingerprint of the main frame and the bootstrap parameters, plus the financing convention for levered books. Unlevered entries computed this session are adopted under the fingerprint. The start-up check is always recomputed.
5. Capacity of a levered book uses its unlevered excess return (excess / L) before dividing by L. Values at the top of the capacity grid are reported as "at least".
6. Reported, not decided: in-window drawdowns next to the crisis-floor numbers; Reg-T above 90% flagged as needing portfolio margin; the financing ranking at 2.5%; the more-return feasible frontier (next passing points and slack to each limit); the next-step vs launch head-to-head; DV2-IND's pre-2010 fill predating the liquidity-unit audit; the HPI-gap frame now being a historical scenario; no minimum effect size for adding a pod (downshock +0.013 excess Sharpe).

### A6-d (post-review 2, labelled; recorded before a6d.py runs)
Phase-2 review of A6-c accepted. Changes:
1. Cost robustness. Every defensive slot must pass its rule (DEF or CALM, 10-seed mean and worst seed) both at model costs and in the +5 bps frame. Cash levels and the more-return point are re-scanned with that double test. The launch, next and target picks (defaults and challenges, including the A6-c gate) are re-run with it. Very defensive builds on the launch winner's core. More return keeps the 1.0 pp fallback, measured against the re-scanned launch. The re-scan also gives the next-step vs launch head-to-head and the gated HPI shares (0 / +5 / +10 bps) against the new launch.
2. Downshock as a launch upgrade (reported, not a slot): 60/40 + downshock 5% and 10% (pro rata), with the smallest cash passing the double test. It is challenge-tested against the launch and against the next step. It is PM_READY but research-only and has no live route, so it is marked "upgrade once wired". The owner's question gets this answer: downshock is crash-prone alone, but at 5-10% inside 60/40 it is a diversifier, at the cost of about 1 pt of start-to-end 2008 return per 5%.
3. Evidence flags: rows needing DV2-IND (RESEARCH tier; pre-2010 fill invalidated pending a native-Turnover rerun; "forward-test first") or EOM (research/bench only; modeled MOC fills; cost-sensitive) are marked conditional on a forward test (EOM also needs a tradable non-MOC route). The 22% message becomes conditional.
4. Report corrections: alpha significance restated (net t >= 2: target, 22% next and target, max leverage, MR target), with the caveats that leverage raises net t mechanically and that all t-stats are in-sample; the 2012+ claim is limited to the challenge tests; capacity grid labels (60/40 about $250M, IEF-bound); more-return expectation about 9.8%; the cash-fee claim; the downshock floor wording per core; the max-leverage wording; engine shares 66-81%; HPI/DV2 cost wording; which A6-b checks still stand (Q2, Q4); the one-third-growth target mix (lower P(DD<-20%), loses only +5 bps CAGR); stale YAMLs removed; the in-window drawdown excludes the window's start day, like the window return.

### A7 (2026-10-01, owner question "what about more than 22%?"; descriptive, selects nothing)
Recorded before a7.py runs. Targets 24 / 26 / 28 / 30 / 35% gross CAGR (LONG window, main frame, A6 conventions). Routes:
- No leverage: TAA3x-1N share t (0.40 ... 1.00, step 0.01) inside the verdict structure (the rest pro rata), plus pure TAA3x and pure TAA3x-1N as the unlevered ceiling.
- Leverage (daily, DTB3 + 1.5% ACT/360), with L the smallest on a 0.01 grid reaching the target:
  - fund_growth x L, and fund_growth_plus x L;
  - 50/50 growth + C_N x L;
  - 50/50 and 1/3 growth + 2/3 C_T x L;
  - the target slot book x L;
  - fund_growth_mr x L.
Reported per point:
- L, CAGR at +5 and +10 bps, and at a 2.5% financing spread;
- max DD and P(DD < -20 / -25 / -30 / -35 / -40%) (10 seeds x 2,000 paths, mean and worst seed);
- worst year, the 2008 and 2022 windows, Reg-T (worst case) and capacity / L.
Annotations, not gates: GROWTH PLUS (DD >= -22%, P(<-25%) <= 15%), and an illustrative AGGRESSIVE limit (DD >= -30%, P(<-30%) <= 15%). EOM / DV2-IND routes carry the A6-d conditions. Separate tail cache (extended limits).
