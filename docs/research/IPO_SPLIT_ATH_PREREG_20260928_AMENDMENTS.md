# Amendments to IPO_SPLIT_ATH_PREREG_20260928 (dated; none changes a computed verdict)

Frozen PREREG SHA-256 `2309fcaac69e5fedda2e3fa512bcb623eaa6ef8ece3279a3c760b8ed129f15c4` (see
`results/research/ipo_split_ath_20260928/prereg_freeze.json`).

## A1 - 2026-09-28, implementation facts fixed before any result was read

- **Security types.** Norgate has no REIT subtype. REITs are classed as "Operating/Holding Company", so the allowed
  subtype2 list is `["Operating/Holding Company"]`. Excluded: "Special Purpose Company" (SPACs that never merged,
  shells) and "Investment Company" (closed-end funds, BDCs). S-SPAC adds operating companies whose first quoted day was
  blank-check, which are de-SPACs such as DWAC/DJT. Recorded in `symbol_meta_summary.json`.
- **Bug found and fixed before any result.** Norgate returns no last-quoted date for active symbols. The first
  build's filter `last_quoted >= 1992-11-02` therefore dropped all 6,508 active stocks, leaving only delisted names.
  The split detector's post-2010 counts gave it away: AAPL, TSLA and NVDA were missing. It is fixed: NaT means active.
  A guard now fails the build unless both databases are represented. The rebuilt universe has 21,259 symbols
  (6,508 active, 14,751 delisted) and 8,954 forward splits. The known 2020-2024 splits are all detected.
- **Engine alignment of the replica.**
  - Halted sessions are padded (O = H = L = C = last close, Dividend 0), as the engine's ALLMARKETDAYS data are.
  - Terminal liquidation is at the last close with commission and no slippage, as in
    `_liquidate_missing_price_positions`.
  - Dividend cash is posted before the next open, as in `_credit_dividend_cash_before_open`.
- **Parity comparison fix.** The first parity run matched returns exactly (correlation 1.000000, max daily
  difference 7e-16). It flagged two entries only because the replica's closed-trade list omits positions still open at
  the window end. Entries are now taken from the intent log. Final: identical 578 transactions; passed.

## A2 - 2026-09-28, the rule for R3

C_BIL for R3 uses the stressed L series (`returns_A_NDX_stress`), so the candidate and its control carry the same
stress on L. The unstressed C_BIL is also reported; both readings fail.

## A3 - 2026-09-28, POST-HOC diagnostic added after the verdict (cannot change it)

- **Why.** Our point-in-time results (IPO rule ≈ 0%/yr since 2001) are far from the talk's (≈ 18%/yr, Sharpe 1.4). One
  hypothesis: Sharadar's TICKERS table sets its market-cap tier (`scalemarketcap`) from the most recent market cap. A
  "mid, large and mega cap" filter built from that field keeps the IPOs that later became large, which is look-ahead.
- **Test.** `posthoc_lookahead.py` rebuilds that filter:
  - last_cap = last known Norgate shares outstanding x Unadjusted Close on or before that date;
  - the universe is IPO-window ATH events with last_cap >= $2B;
  - no liquidity or price filter; SPACs allowed.
- **Runs.** A0 in E1 and E2 form. This adds 2 labelled post-hoc runs; they are not candidates.

## Results log

- Unit tests: 30 pass.
- V1: 685 checks, 0 failures. V2: 300 checks, 0 failures.
- Engine parity (IPO-A0, 2015-2019): passed.
- Stage A "edge present": IPO no (t −1.37, 1 of 4 blocks positive); split no (t −1.98, 1 of 4).
- Rule:
  - IPO-A0: R1 no, R2 yes, R3 no, R4 no, R5 yes, so it FAILS.
  - SPLIT-A0: R1 no, R2 yes, R3 no, R4 no, R5 yes, so it FAILS.
- Reality Check p 0.9985; 0 of 30 cells beat C_BIL on G-FULL.

## Independent review (2026-09-28, read-only)

- **No bugs found.** Every headline number was reproduced, and 8 random trades were rebuilt from raw Norgate bars.
- **Caveats added to the report:**
  - event-weighted clustered t +1.47 beside the frozen month-weighted t -1.37;
  - the look-ahead diagnostic is confounded with dropping the liquidity and $5 filters, so point-in-time
    widening runs were added;
  - the 63% cash-use figure is the 1993-2026 average;
  - the close-entry book ties C_BIL on G-FULL but fails R1;
  - the Stage A dividend window is one bar early (immaterial).
- **Verdict:** cannot flip.
