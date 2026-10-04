# Scout P7b: DV2 on industry ETFs

Date: 2026-10-02. Research only; nothing here changes live trading. The pod's registry tier stays RESEARCH.
- **Script:** `scripts/research/scout_p7b_reaudition_20261002/run_p7b.py`.
- **Spec:** `alpha/scout/specs/dv2_industry_etf.py` (the DV2 spec's features, exits and decision function, plus the
  ETF eligibility).
- **Ledger:** one RETRO registration, `dv2_industry_etf_reaudition_20261002`, family `etf_short_term_reversal`,
  50 prior trials. The universe was chosen after results (the 2026-09-25 research tried 23 universes and found this
  one), and the registration says so.

**The pod.** The WIRED DV2 rule, unchanged, on 19 industry ETFs (XBI, IBB, SMH, SOXX, KRE, KBE, XHB, ITB, XRT, XOP, OIH,
XME, GDX, IYT, IGV, ITA, IHI, XSD, XPH). An ETF is eligible after 252 sessions of history and while its 63-day average
native Turnover is above $50M. The backtest starts on 2012-01-03, so the in-sample period is 2012 to 2022.

**Gate.** `gate dv2_industry_etf --fresh` passes at the exact tier (4.4e-16, 0 trade dates differ) with no engine
change. `dv2` and `sector_ibs_vox_iyr` were re-run fresh and still pass.

## Verdict: WATCHLIST, with a single WARN (S3 cost coverage 1.62 < 2)

| Station | Result |
|---|---|
| S3 event edge (h 3) | +16.2 bp over the same-date eligible ETFs, NW t 2.01, placebo p 0.025, 1,736 events; cost coverage 1.62 (WARN) |
| S4 | plateau 0.85; Sharpe 0.83 at twice the costs + 10 bp; luck band and losing streak pass |
| S5 MCPT (date shuffle, eligibility on real dates) | **p 0.001** |
| S5 DSR (50 prior trials, warn) | **p 0.002** |
| S5 diagnostics | walk-forward OOS Sharpe 0.84, 100% of designs positive; PBO 0.35 |
| S6 net alpha (ETF mix + trend + TAA 3x) | **+4.9%/yr, t 3.05** |
| S6 T-bill slot (NDX VXN's slot) | **P 0.99**, book Sharpe 1.32 vs 1.21 |
| Same, idle cash at the T-bill rate (information) | P 1.00, book Sharpe 1.35 vs 1.21 |
| Capacity (1% ADV) | $6.9M (binding: IYT) |

**Net, 2012 to 2022:** CAGR 7.2%, Sharpe 1.22, max drawdown −8.3%. The pod is idle on 55% of days, and its average
gross exposure is 12%.

## What this means

1. **This is the cleanest reversal pod Scout has audited.**
   - Every gate passes. The one WARN is cost coverage, and at 1.62 it is the best of the reversal pods: DV2 stocks
     0.91, HPI 0.36, sector IBS 0.98.
   - Unlike HPI and the sector IBS pods, its entry event also beats the same-date basket (t 2.01, placebo p 0.025).
     So DV2's selection works on ETFs too, not only its timing.
   - The weak points:
     - **Short history:** 11 years in sample, with the eligible list growing from 6-10 ETFs in 2012 to 13-17 since 2018.
     - **Survivorship:** the list is today's ETFs.
     - **Capacity:** small, about $7M.
2. **It is a different bet from the stock reversal pods.** In-sample daily correlations, 2013 to 2022:

   | | DV2 S&P 500 | HPI vote | TAA 3x | NDX VXN | CORE5 | EOM |
   |---|---|---|---|---|---|---|
   | DV2 industry ETF | 0.47 | 0.50 | 0.35 | 0.43 | 0.33 | 0.10 |

   - DV2 and HPI correlate 0.75 with each other; this pod correlates about 0.5 with either. It can sit next to one
     stock reversal pod rather than replace it.
3. **The idle-cash convention understates it.** It holds cash on most days, and the engine pays nothing on cash.
   Credited at the T-bill rate, the book lift grows from +0.11 to +0.14 Sharpe.
4. **The 2026-09-30 defensive v2 study** (CORE5 + BTAL_QQQ + DV2-IND as the owner's leaning) now has Scout evidence
   for its third leg. The first evidence-based step is a forward shadow (S8), not a live route.

## Notes

- **The strategy docstring's warning is out of date for the code.** It says the legacy results used raw Close times
  split-adjusted Volume. The module already uses native Turnover, so the gated engine is the corrected version. Only
  the legacy figures it quotes (CAGR 5.2%, Sharpe 0.92, 2000-2026) should not be cited.
- **Norgate data quirks:**
  - SMH and OIH start on 2011-12-21 (the VanEck relaunch; the older history is not linked), so they are eligible only
    from 2012-12-21.
  - XPH is never eligible; XSD becomes eligible only in 2026-06.
- **Deviations:** the pod shares two existing ones, each measured as immaterial here.
  - `float32_momentum_threshold`: 0 trades change.
  - `split_adjusted_share_units`: −1.4 bp/yr, conservative.
- **Grid:** max positions 5/10/15, not 20, because at most 18 ETFs are ever eligible.
