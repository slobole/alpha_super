# Amendments to ALPHA101_PREREG_20260928.md (lead, 2026-09-28, after the results)

The frozen file is unchanged (SHA-256 `74af502a...`). The research agent's implementation notes N1-N22 are in
`results/research/alpha101_20260928/amendments_and_bugs.json`. The items below are the ones a reader needs. None
changes the verdict.

- **A1 - Which cells R1-R3 are judged on (note N8).**
  - The PREREG defines the plateau as a neighbourhood median but writes "candidate-book Sharpe" in R1-R3.
  - The implementation judged the rule on the neighbourhood median, the convention of the earlier new-pod studies. It
    reports the centre cell alongside.
  - The verdict is the same under both readings. In G-P3 (C_BIL 1.437):

    | Candidate | Neighbourhood median | Centre cell |
    |---|---|---|
    | C_EQ long | 1.212 | 1.220 |
    | C_WF long | 1.173 | 1.165 |
- **A2 - Norgate's Turnover is synthetic before 2014 (note N21).**
  - Native Turnover equals Close x Volume on these shares of S&P 500 member-days:
    - 100% in 2006-2013;
    - 66-85% in 2000-2005;
    - 28% in 2014;
    - 0% from 2015.
  - So the frozen `vwap` = Turnover / Volume equals the close on those days. For the 43 alphas that use `vwap`,
    results before 2014 are results of different formulas (vwap replaced by close), and alpha 83 is missing on about
    half the member-days.
  - The frozen definition was kept. P3 (2022-26) and everything from 2015 use a true daily VWAP, so the P3 failure is
    not a data artifact.
  - The lead confirmed the pattern independently on AAPL, XOM, JPM and KO (Norgate NONE bars):
    - synthetic on 100% of days in 2008 and 2012;
    - synthetic on 0% of days in 2016, 2020 and 2025.
  - Uses of Turnover as dollar volume (ADV, liquidity filters) are unaffected. Only VWAP derived from it is.
- **A3 - Tie tolerance (note N6).**
  - The cache's adjusted prices carry float32 precision, so ties decided on them depended on the download date. V1
    found 327k mismatched cells.
  - Fix: comparisons, rank ties and constant windows are decided at 1e-6 relative, and near-zero sums and differences
    are snapped to zero.
  - Chronology (corrected after the independent review): the main fix came before any Stage A number was read. A last
    residual (170 of 530 million V2 cells, in alphas 63, 73, 82 and 87) was fixed only after Stage A and the grid had
    first run. Everything was then rebuilt: Stage A medians were identical to 4 decimals, and grid NAVs moved by under
    1%.
  - After the fix, V1 and V2 show 0 mismatches on all 100 alphas.
- **A4 - Alphas 96 and 97 carry no signal (note N22).** Under the frozen window rules (all values in a window must be
  finite; a constant input gives NaN):
  - 96 is never finite and 97 almost never.
  - The family is effectively 98 alphas; C_EQ averages 96 alphas per stock-day.
- **A5 - Turnover convention (note N5).** tau counts buys plus sells per dollar of gross, about twice the paper's
  "1 / holding days" number for the same behaviour. The break-even cost c* is convention-free.
- **A6 - Readings kept from earlier studies:**
  - the sweep earns 0 before BIL's first return (2007-05-31) (N10);
  - the DSR window (N13);
  - ADV20 = the 20-day median of dollar Turnover (N14);
  - the PIT conversion applied literally on split days for the IndNeutralize alphas and alphas 29 and 101 (N19).
- **A7 - Small-account runs compound (lead).**
  - The $10k / $25k / $1M runs start at that capital and compound, so their later years describe larger accounts.
  - The report therefore also gives the constant-size commission arithmetic. C_EQ long makes about 2,190 orders a year:
    - at $10k, about 22% of NAV a year;
    - at $25k, about 9%;
    - at $100k, about 2%.
- **A8 - The plateau sits on the grid's edge (lead).**
  - Every candidate uses the stickiest exit band, B = 8.
  - Standalone Sharpe rises with B in every row: the less the pod trades, the better.
  - The grid was not extended after seeing this, because that would be a post-hoc variant.
- **A9 - Membership is trimmed before a former member's exit (found by the independent review).**
  - The PREREG says "point-in-time membership; delisted names are included". The house loader (`data/norgate_loader.py`
    lines 106-107, and the snapshot exporter) drops the last 5 membership rows of every symbol that later leaves the
    index.
  - That is look-ahead: the exit date is known in advance. The trimmed windows hold the index's collapses. The lead
    confirmed five on Norgate:
    - Lehman -96%;
    - WaMu -99%;
    - Enron -93%;
    - First Republic -98%;
    - SVB -62%.
  - V1 cannot detect it, because it reuses the membership panel.
  - Effect here, from the reviewer's re-run with membership restored: every candidate gets worse. The C_EQ N20 B8 G-P3
    margin goes from -0.217 to -0.235, and REV5's FULL Sharpe from 0.407 to 0.364. The verdict is unchanged.
  - The trim is repo-wide, so the DV2 and NDX backtests used in the slot tables carry it too.
