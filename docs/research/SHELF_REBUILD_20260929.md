# Shelf rebuild after the 22-29 Sep 2026 fixes (defensive, growth, portfolio map)

Date: 2026-09-29. Owner request (Hebrew): rebuild the portfolios rigorously after the week's fixes, defensive first,
then growth (drawdown matters more than Sharpe), with the 2008 proxy, ease of operation and AUM, and update the
portfolio map. The full Hebrew report is the published artifact "מפת התיקים"
(https://claude.ai/artifact/Ti65jbjiAUSsig2xfeax7V, private; copy in `results/.../report/shelf_map.html`). This file is
the English record.

- Plan: `scripts/research/shelf_rebuild_20260929/SPEC_FROZEN.md` (frozen before any run; amendments A1-A6 at the end;
  A1-A2 before results, A3-A6 after results and labelled; A2 and A4 corrected after the independent review).
- Code: `scripts/research/shelf_rebuild_20260929/` (run_sleeves, run_proxies, lib, family, part_d, part_d_prime, part_g,
  part_m, breach_frontier, selection_checks, friction_runs, report_data). Tests: `tests/test_research_shelf_rebuild_20260929.py` (13).
- Outputs: `results/research/portfolio/shelf_rebuild_20260929/` (sources, proxy_runs, part_d, part_g, part_m, checks,
  friction, report, experiment_ledger.jsonl).

## Inputs

27 sleeves re-run at HEAD f9ad358 ($1M each, to 2026-08-19, one Norgate vintage during the runs): every PM_READY and
WIRED strategy plus four shadow candidates (NDX NATR20, DV2 floor, DV2 floor + ADV rank, industry-ETF DV2). The 2008
proxy for the four BTAL TAA sleeves (synthetic TQQQ / BTAL spliced before the real funds) re-passes the
strategy-level check at HEAD (monthly corr 0.96-0.99, CAGR gap <= 1.8 pp, drawdown gap <= 0.9 pp). T-bills = BIL
total return. Norgate updated at 13:58 local after the sleeve runs; benchmark values through 2026-08-19 are unchanged.

## What the fixes changed (2012-10-02 -> 2026-08-19, before -> after)

NDX-VXN (live) Sharpe 1.18 -> 0.93, CAGR 22.2% -> 15.5%; NDX-ATR 1.15 -> 0.90; Tactical FI 1.06 -> 0.76 (cash held as
BIL); Compass 1.10 -> 1.08; DV2 1.083 -> 1.080. CORE5, the TAA sleeves, EOM flow and HPI are unchanged in backtest.

## Results

- Part D (83 defensive books, objective LONG excess Calmar): 26 pass the gates; the binding gate is the T-bill slot
  test. The rules pick D* = CORE5 + EOM (equal): 9.7% CAGR, -8.3% max DD (LONG, 2008-03 -> 2026-08).
- Cash convention (found by the independent review): sleeves credit idle cash at 0% while every hurdle and slot
  replacement earns the full BIL rate (4.5% a year in the last three years). EOM holds 89% idle cash, so its RECENT
  excess is -1.1%/yr under the house convention but +2.7%/yr with realistic cash interest; under the cash-realism
  frame the full rule picks CORE5 + BTAL_QQQ + EOM + DOWNSHOCK [IV]. EOM is therefore excluded from the recommended
  map only for operability (month-end MOC and a TLT short, gap G-032), not for performance.
- A3 (post-result): the same rules without EOM give D' = CORE5 + BTAL_QQQ, inverse vol (CORE5 50-75%, mean 63%):
  8.9% / -6.2% / Sharpe 1.31, two monthly pods. IV and EQ are tied (IV beats EQ on 39% of paths; +5 bps flips the pick
  to EQ); the core is the same under every sensitivity.
- Part G (72 growth books, objective LONG CAGR at a -20% budget): 36 pass; all 36 Compass books fail the twin gate
  (27-64% of paths vs 90% required). The bootstrap cannot separate the other 36. The ease tie-break picks
  G* = TAA3x-1N + NDX-ATR: 21.2% / -19.6% (at the budget), stable under every sensitivity. Before the window NDX-ATR fell
  -47% in Mar-May 2000 vs -29% (2004-05) for NDX-VXN. G' (A4, re-derived with the ease tie-break at the -16% design
  point) = TAA3x-1N + NDX-VXN: 20.9% / -17.1%. The live pair G3 = TAA3x + NDX-VXN: 17.6% / -15.3%.
- PBO 0.21 (D) / 0.22 (G) describes an argmax-over-all-books rule. Fixed-book median OOS rank over 12,870 CSCV splits:
  D* 0.64, D' 0.63, G* 0.96, G' 0.93, G3 0.41 (G3 gives up return for distance from the limit), G3 + DV2/HPI 0.52.
- Bootstrap breach lens (A5, descriptive): a book fitted to exactly -10% breaks -10% on 75-93% of resampled histories;
  fitted to -7% it breaks -10% on 14-23%. Growth: probability of breaking -20% is 13.5% for G3, 60% for G', 6% for G3
  plus the DV2 + HPI pair (all WIRED, daily). The defensive and balanced products' binding drawdown is Jan-Feb 2010
  (synthetic TQQQ/BTAL era); on real data only their worst drawdowns are -6.4% and -9.0%.

## Recommended map (judgement, labelled in the report)

One defensive core D', one growth core G3 (live today), T-bills; each product built one step below its hard limit:

| Product | Weights (defensive / growth / T-bills) | LONG CAGR | LONG max DD | P(break hard limit) | Friction at $30K / $100K |
|---|---|---|---|---|---|
| Defensive (hard -10%) | 55 / 30 / 15 | 10.5% | -6.9% | 14% | 2.3% / 0.6% a year |
| Balanced (hard -15%) | 45 / 55 / 0 | 13.8% | -9.6% | 7% | 2.0% / 0.6% |
| Growth (hard -20%) | 0 / 100 / 0 (G3) | 17.6% | -15.3% | 13.5% | 1.3% / 0.5% |
| Growth+ (option) | G' | 20.9% | -17.1% | 60% | 1.3% / 0.5% |

Friction = CAGR loss from whole shares and the $1 minimum at today's prices (runs from 2023-08-21), mostly the NDX pod.
Four monthly components (CORE5, BTAL_QQQ, TAA 3x, NDX-VXN) plus BIL. Annual rebalancing is required (without it the
defensive product's drawdown reaches -10.3%). Pre-declared sensitivities move product CAGRs by less than 1 pp.
Capacity (growth-shelf route model, last three years of fills): MOO today about $1-5M per product, worked + blocks
$100-250M. PortfolioManager accepts only PM_READY+ pods, so the T-bill slot cannot be a YAML pod today.

## Independent review (Tier 1 quant-pitfalls agent)

No code bug changes D*, D', G* or the rung weights; every selection, gate, objective, bootstrap array, PBO and Part M
grid reproduced independently. Findings addressed: cash-convention asymmetry (H1, reported and A3 re-grounded on
operability), selection re-run under every sensitivity (H2, `selection_checks.py`), G* noise and pre-2008 NDX evidence
(M1), PBO labelling and fixed-book OOS ranks (M2), D' code and ledger (M3, `part_d_prime.py`), post-result forks
logged (M4, A4 timing, A5), budgets as historical with the 2010 binding drawdown (M5), A2's false "ETF splits are
rare" corrected and capsule books flagged (M6), friction leading the ops section (M7), tests for the slot-test IV
weight, tie-break, rung pick and cash lag (L3), base-date guard (L6).

## Open owner decisions

Approve the map (or Growth+ with 1/N); wire CORE5 (the only component without a live route); EOM: wire it (MOC + TLT
short) as the defensive upgrade path or keep it research-only; how to hold the T-bill slot (register the passive BIL
control or hold BIL outside the PM); minimum account (about $100K for the defensive and balanced products); archive
fund_menu_* and ladder_1..3 and write map YAMLs; growth upgrade path G3 + DV2/HPI (daily, margin for HPI).
