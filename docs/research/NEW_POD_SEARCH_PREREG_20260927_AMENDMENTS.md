# Amendments to NEW_POD_SEARCH_PREREG_20260927.md (lead, 2026-09-27, after the results)

The frozen file is unchanged (SHA-256 `69fafb7b...`). The research agent's implementation notes N1-N9 are in
`results/research/new_pod_search_20260927/amendments_and_bugs.json`. The items below come from the independent
quant-pitfalls review. None hides a tradeable edge.

- **A1 - R4 halves: the liquidity pool was ambiguous, and it decides the formal verdict of candidate M1.**
  - Section 4 defines the event liquidity filter (REL25) against "that day's members" and the R4 halves as "members of
    the S&P 500 at d" versus "members not in the S&P 500 at d". It does not say whether each half recomputes the REL25
    percentile inside the half, or keeps the Russell 1000 percentile used by the primary run.
  - **Implemented reading (per half).** The implementation recomputed the percentile inside each half (note N6),
    `new_pod_search_20260927/data.py:112` and `features.py:59-72`. The halves then no longer split the primary run's
    events.
    - M1 centre: 62 primary entries become 12 + 58 in the halves.
    - The mid-cap half trades names that fail the Russell 1000 filter.
    - Result: M1 fails R4 on the mid-cap half, 1.3650 against C_BIL 1.3652.
  - **Pooled reading (one Russell 1000 threshold).** This splits the events exactly (24 + 38 = 62). It is the reading
    Study 2 used with one Russell 3000 threshold.
    - The independent reviewer's in-memory re-run gives 1.3674 (S&P half) and 1.3654 (mid-cap half) against C_BIL
      1.3652, so M1 would pass R1-R5.
    - Its minimum block margin is +0.0020 Sharpe; M2 still fails (1.3643).
  - **Both readings are reported.** Under the pooled reading M1 is a formal pass. Economically it is T-bills plus about
    three deals a year:
    - exposure about 3%;
    - book gain +0.002 Sharpe;
    - Reality Check p = 0.92;
    - the paired p of 0.02 is on a +0.002 difference.
  - The report therefore says: "formal pass depends on the REL25 pool; not significant; economically nil; not a pod to
    trade."
- **A2 - Deflated Sharpe of family M is uninformative.** It was computed on returns that include the idle-cash sweep,
  so its standalone Sharpe of 2.2-2.45 is T-bill carry. Without the sweep M1's FULL Sharpe is 0.70. Future studies
  compute the DSR on returns in excess of BIL, or without the sweep.
- **A3 - R5 was evaluated on the centre cell, not the neighbourhood median.** There is no effect: every M cell clears
  $50M.
- **A4 - The HEDGED form gives up part of the T-bill carry.** SH's own T-bill distributions are credited net of the 25%
  withholding in the engine, while BIL's total return is gross. Adding back 0.5 x BIL moves the HEDGED 2022-26 margin
  from -0.122 to -0.084. It still fails.
