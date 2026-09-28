# Strategy readiness audit protocol — amendments

The frozen protocol is `STRATEGY_READINESS_AUDIT_PROTOCOL_20260928.md` (SHA-256 `9e4bc29f…6b41b`, see the `.sha256` file).
It is unchanged. Everything below was written **after** Tier A results were seen and is labelled that way.
No amendment changes a pass/fail threshold.

## AM-01 (2026-09-28, after results): which window governs check C1

**Problem.** Section 5 says "median order ≤ 1% ADV and 99th percentile ≤ 5% ADV" but does not say over which window.
Section 4C asks for the full history and the last 3 years separately. For the TAA family the two windows disagree
because BTAL's turnover was about USD 30K/day in 2011–13 against about USD 7.2M/day in the last 3 years.

**Decision.** The last-3-year window governs current tradability, and the full-history figure is reported next to it.

**Why this is acceptable after the fact.** It softens only the USD 30K verdict for the TAA family: full-history
p99 is 67% of ADV and last-3-year p99 is 0.17%. It does not change any USD 1M verdict, which fails for the TAA family
in both windows. The owner stated independently, before seeing this amendment, that 2012–2018 BTAL liquidity is not
relevant to current trading.

## AM-02 (2026-09-28, after results): evidence for one-session leaks in the TAA family

The primary TAA truncation check (A3) compared only months strictly before the cut-off month. The independent
reviewer showed that it cannot catch a planted Close_(T+1) read (0/9 caught). The same planted leak is caught by the
live-host replay (TAA 3x 16/45 decisions differ, BTAL_QQQ 4/45).

For one-session leaks the report therefore cites the 167/167 live-host replay, not the A3 result. The A3 result is
kept only as evidence against multi-month leaks. The reviewer's positive controls
(`results/research/strategy_readiness_audit_20260928/review_quant/rq_taa_positive_controls.json`) are adopted as the
A4 record for the TAA split harness and the linearity family.

## AM-03 (2026-09-28, after results): DTB3 publication-lag method

The primary A5 check shifted DTB3 by one calendar day, which leaves 50–51 of 169 month-ends unlagged, namely those
ending on a weekend or holiday. Two independent re-runs with a one-trading-session lag (`review_quant/rq_taa_checks.json`,
`review_live/dtb3_session_lag_probe.json`) give the result the report cites: 0/169 flips for TAA 3x and TAA 1/N. The
smallest score-to-hurdle margin was 2.5e-5, on 2025-09-30.

## AM-04 (2026-09-28, after results): NDX replay coverage

The NDX replays handed the same universe to both sides: one full-history universe build, trimmed with today's
knowledge. They therefore prove that the host code is identical, not that the live universe equals the backtest
universe. The universe difference is measured separately: 8 of 320 decisions differ
(`review_live/ndx_trim_live_divergence_summary.json`, `review_quant/rq_ndx_untrimmed.json`).

Of the NDX ATR replay's 23 exact matches, 11 are empty-versus-empty (regime off). That leaves 12 informative
month-ends, below the 24 required by B1, so NDX ATR live parity stays capped at READY WITH CAVEATS or lower.

## AM-05 (2026-09-28, after results): how 0% idle cash is graded

**Problem.** The Tier B/C auditors counted 0% interest on idle cash as a "material conservative issue" that blocks
READY. Tier A did not count it (NDX, CORE5), so the grading was inconsistent.

**Rule.** The owner-approved house ledger (G-024) credits 0% on positive cash. It is a declared convention, not an
issue, so on its own it does not cap BC at READY WITH CAVEATS. Its size is still reported beside each result. A
strategy that departs from the convention is graded on that departure, when the departure is optimistic and
material. Example: Tactical FI credits the full DGS3MO rate.

**Direction.** This can only raise a BC grade whose sole caveat was 0% cash (EOM, Industry-ETF DV2). It never
lowers one. At owner size (USD 12–30K), IBKR pays little or nothing on idle cash, so the 0% convention is close to
realistic there.

## AM-06 (2026-09-28, after results): house opening-auction cost model in BC and TR

The protocol lists slippage under accounting (A8). Where the house opening-auction (MOO) cost model
(`alpha/engine/capacity_analysis.py`) shows that the published record's 2.5 bp slippage understates execution cost
by a material amount at the published size, this is an optimistic accounting issue, and BC is NOT READY. Every TR
scorecard also carries a "house-model capacity" line. No threshold changes.
