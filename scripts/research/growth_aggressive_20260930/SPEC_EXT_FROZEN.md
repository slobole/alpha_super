# Extension E (2026-09-30): gross objective, a MAX rung with heavier TAA, and NDX-RM (frozen plan)

Written 2026-09-30 after the main study (SPEC_FROZEN.md, amendments A1-A2) and its report, at the owner's request
(Hebrew, 2026-09-30): "run all three" of (1) the gross return as the objective, because no fee is charged today;
(2) loosen the risk rung and allow a heavier TAA weight; (3) add the NDX-RM momentum sleeve from the PAKAL research.
No extended book has been built or read when this is frozen. Its SHA-256 goes to the same ledger. Everything not
changed here is as in SPEC_FROZEN.md (inputs, fee model, book model, windows, bootstrap, statistics).

## E0. What is already known

The main study's results (G3 at its GROWTH breach limit; TAA3x-1N + NDX-VXN cores win the AGGRESSIVE rung; Compass
failed every twin test; the family's gross ceiling was ~22%). NDX-RM (PAKAL `momentum_trend_universe_search`, 27 Sep
2026): found post hoc AFTER its pre-registered design winner failed the holdout; ~2,300 configurations were tried; its
record recommends shadow beside NDX-L, not a replacement; the beside-L book's edge over G3 rests on 2026 (Sharpe 1.36
with 2026, 1.31 through 2025; p 0.18). This extension cannot remove that selection history; it applies a strict twin
test instead.

## E1. NDX-RM input (validated before this freeze)

- Series: `pakal/pakal-research/reports/momentum_trend_universe_search/final/ndx_rm_daily_returns.csv`, column
  `ndx_rm` (10 bps per side, the PAKAL central cost layer; 10 NDX names, residual momentum, sqrt-dollar-turnover
  weights with a 15% cap, dual-signal half brake, 4 tranches at month end +0/5/10/15 sessions, decision Close_T, fill
  Open_(T+1)). Idle cash earns 0% in that engine; no fair-cash add is possible (no cash column): conservative, disclosed.
- Checks done: no missing or extra dates against the house calendar on LONG; PAKAL's copy of the live NDX pod matches the
  house `ndx_vxn` sleeve (daily corr 0.9997, CAGR 15.4% vs 15.8% on 2012-10..2026-07) and its TAA copy matches `taa3x`
  (corr 1.0000); so PAKAL's engine is comparable and slightly conservative. NDX-RM LONG: CAGR 22.1%, max DD -31.2%, corr
  0.71 with ndx_vxn. Tier: shadow (research, no alpha_super module). Execution: 4 tranche dates a month, no daily signal.
- It enters as sleeve `ndx_rm` in every frame (the same series in every cash frame).

## E2. Family (1,395 books)

TAA leg {taa3x, taa3x_1n, taa2x_1n} x NDX leg {ndx_vxn, ndx_atr, ndx_natr20, ndx_rm, vxn_rm} x TAA:NDX ratio
{50:50, 60:40, 70:30} x satellite (none, or the main study's 15 types at 18% or 36%; the satellite takes its share from
the core pro rata). `vxn_rm` = the NDX share split equally between ndx_vxn and ndx_rm ("beside L"). Compass is dropped
(it failed every twin test in the main study; disclosed as a post-result family choice). LOW-TOUCH = satellite in
{none, def2, tbill}; NDX-RM is allowed there (flagged: 4 tranche dates a month, no daily signal).

## E3. Two objective modes, each a complete selection

- GROSS (primary now, the owner charges no fee today): objective LONG gross CAGR; R1 and R2 on the gross book only;
  R3 gross excess over T-bills > 0 in blocks B, C, RECENT; R4 slot test on gross CAGR; tie band and champion tests on
  gross CAGR per path.
- NET (the fund): exactly the main study's rules (2/20; R1 and R2 on gross and net).

## E4. Rungs

GROWTH (hist >= -17%, P(< -20%) <= 15%), AGGRESSIVE (hist >= -22%, P(< -25%) <= 15%), and new MAX (hist >= -27%,
P(< -30%) <= 15%).

## E5. Gates, tie band, tie-break, champion chain

As SPEC_FROZEN sections 5-6 in the mode's currency, with R5 replaced by R6: a book holding `ndx_rm` must beat its twin
with `ndx_vxn` in the same place on >= 90% of the bootstrap paths; a `vxn_rm` book must beat its `ndx_vxn` twin on
>= 90%. Champion chain per line and mode: GROWTH pick vs G3 (>= 80% of paths and breach <= G3's, else G3; if G3 fails a
GROWTH gate, the pick by default); AGGRESSIVE pick vs that line's GROWTH product (>= 80%); MAX pick vs that line's
AGGRESSIVE product (>= 80%); a rung whose pick fails its test has no separate product.

## E6. Seed check (lesson of A2, reported, not a gate)

For every product and pick: R2 on 10 seeds (frozen seed + 9); a product whose 10-seed mean breach exceeds 15% is
flagged "at the limit".

## E7. Sensitivities (each re-runs the complete selection in both modes)

S1 house 0% cash; S3 +5 bps per side on house sleeves AND ndx_rm at its PAKAL 20 bps layer (+10 bps, harsher);
S5 HPI live gap; S6 EXACT window; S7 block 126; S8 block 21; S9 ndx_rm at 20 bps only. (S2 and S4 of the main study are
dropped: they changed nothing there.)

## E8. Reported, not selected

PBO (argmax rule) per rung-line in GROSS mode; capacity and fee income for the new products (NDX-RM is capacity-rich per
its record; its orders are not in the house route model, so its capacity is taken from its record and labelled);
the margin what-if for comparison.

## E9. Review

An independent Tier-1 review of this extension before the report is updated; findings go to the amendment log below.

## Amendment log

- EA1 (2026-09-30, AFTER all extension results and the Tier-1 review `results/.../REVIEW_TIER1_EXT.md`; post-result,
  nothing re-selected): (1) the 70:30 TAA:NDX ratio is the edge of the pre-registered grid, not an optimum; on a
  read-only extended grid (50..100) the rule keeps adding TAA up to TAA-only books, braked only by TAA's 2008-09
  drawdown, which lies in the synthetic TQQQ/BTAL proxy; a TAA cap is therefore an owner concentration policy and is
  reported as such. (2) The GROSS GROWTH product's gross hist DD (-16.6%) is 0.4 pp inside the limit, in the proxy era,
  and its net DD (-17.1%) breaks it: it is a gross-only product. (3) R6 is in-sample and selection-biased for NDX-RM:
  without 2020 and 2026 no NDX-RM book passes it; NDX-RM stays shadow-only and no product is built on it. (4) Reality
  checks: GROWTH pick vs G3 p ~0.2 (not significant); AGGRESSIVE LOW-TOUCH pick p ~0.03-0.05 but only 72% of paths vs
  G3 levered to the same volatility; MAX Sharpe below G3's: the higher rungs are mostly leverage on the same factor.
  (5) Products are reported as families; MAX is not separately supported (chain test 82% / 74% / 67%). (6) NDX sleeves
  are in drawdown at the window end (NDX-VXN -16.8%, NDX-NATR20 -24.1% on 2026-08-19); disclosed. (7) Correction of
  E1's text: the NDX-RM series used is the SEL_CFG variant with sqrt-ADV weights capped at 3x equal weight (30% at 10
  names), not the 15%-cap shadow variant PAKAL recommended (book Sharpe 1.359 vs 1.355).
- EA2 (2026-09-30, owner request after the extension report: "decide the open questions yourself; I plan a fund, not
  the current account"; post-result, a JUDGEMENT rule, written before any alpha regression was run): the fund verdict
  per rung (GROWTH, AGGRESSIVE) is the book with the highest main-frame NET CAGR among books that (a) pass the rung in
  NET mode in >= 7 of the 8 extension frames, (b) are LOW-TOUCH (monthly pods only, so capacity is not capped by stock
  auctions near $10M), (c) hold no shadow sleeve (so no NDX-RM, NDX-NATR20, DV2 variants or DV2-IND), (d) hold at most
  70% TAA (concentration policy, EA1), and (e) keep a 10-seed mean breach (gross and net) <= 15% for the rung. This rule
  also settles the TAA leg (TAA3x vs TAA3x-1N) and the TAA:NDX ratio. Reported beside it, not selecting: a factor
  regression of the verdict books, G3 and ladder_4 (weekly excess returns on QQQ, IEF, GLD, DBC, UUP excess returns,
  model M2 adding the QQQ 200-day trend rule; Newey-West 4 lags; as `fund_menu_20260923/allocator_first_look.py`),
  gross and net of 2/20, on LONG, EXACT and each half of EXACT.
