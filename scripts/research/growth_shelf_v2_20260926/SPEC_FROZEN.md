# Growth shelf v2 and a rebuilt 2008 proxy: frozen plan (2026-09-26)

Written before any result of this study was computed. Later changes go to the amendment log at the end, dated,
with the reason; nothing above the log is edited after the first run.

## Owner request (Hebrew, 2026-09-26, paraphrased)

After the mean-reversion (MR) work, put order into the two heavy shelves. Defensive feels locked (LT_DEF A1, the
main-line Defensive, and CORE5 + BTAL_QQQ). Growth probably changes because of what we learned about MR: G3, G4,
ladder_4_1n, ladder_4 with MR and the combinations. For each book: how easy it is to trade, and capacity. A 2008
proxy if possible, but a correct one.

## What is wrong with the current 2008 proxy

Before 2012-10-02 the BTAL TAA sleeves were replaced by `taa_1n_qld`: the same Defense First 1/N rules on four
defensive assets (no BTAL) with a 2x fallback (QLD) instead of 3x (TQQQ). A later check scaled the QLD leg to 3x.
Two errors remain: the rank-weighted sleeve (`taa_btal_tqqq`) is proxied by a 1/N rule, and BTAL's slot and its
momentum vote are simply missing. The fix below runs the real strategy code over 2008 with synthetic TQQQ and
synthetic BTAL, and validates both instruments and the strategy against reality where they overlap.

## Part A: synthetic instruments

Common rules. All inputs are Norgate daily bars. Returns of day t use closes t-1 and t only. Anything used to
build weights at a rebalance uses data up to that decision close only. Each synthetic series is used only before
the real fund's first bar; from the first real bar on, the real series is used unchanged (splice).

### A1. Synthetic leveraged Nasdaq/S&P ETFs

    r_syn(t) = L * r_und(t) - (L - 1) * f(t) - d / 252

- r_und: the underlying ETF's CAPITALSPECIAL close-to-close return (price return; leveraged ETFs track a price
  index). L = 3 for TQQQ (underlying QQQ), 2 for QLD (QQQ), 3 for SPXL (SPY).
- f(t): financing on the borrowed notional = FRED DTB3, prior observation only, times calendar days / 360
  (the same lagged T-bill accrual as the menu study).
- d: one constant annual drag (expense ratio + swap spread), calibrated per instrument ONLY on 2010-02-11 ->
  2026-08-19 so the synthetic total return matches the real fund's TOTALRETURN over that window.
- Bars: Open(t) = C_syn(t-1) * (1 + L * (Open_und(t) / Close_und(t-1) - 1)); High and Low map the same way;
  Close from r_syn; Dividend 0; Volume = underlying volume; Unadjusted Close = Close.

Validation (pre-declared acceptance):
- TQQQ in-sample fit, 2010-02-11 -> 2026-08-19: daily return correlation >= 0.995; median absolute calendar-year
  tracking difference <= 2 pp.
- Out-of-sample crisis test with d frozen from the post-2010 fit:
  - QLD (real from 2006-06-21) on 2006-06-21 -> 2010-02-10: daily corr >= 0.99; the GFC window
    2008-05-19 -> 2009-03-09 cumulative return within 3 pp of real QLD.
  - SPXL (real from 2008-11-05) on 2008-11-05 -> 2010-02-10: daily corr >= 0.99; cumulative return within 5 pp.
- If A1 fails, the proxy is reported as not credible and the old stand-in stays in the tables.

### A2. Synthetic BTAL (anti-beta, dollar-neutral long/short)

BTAL tracks a long/short index: long the lowest-beta fifth and short the highest-beta fifth of about the 1,000
largest US stocks, equal-weighted, sector-neutral, rebalanced monthly. Replica on Norgate:

- Universe at decision date m (last session of each month): Russell 1000 members as of m (the MR study's cache,
  engine as-of membership semantics), price > $5 at m, and >= 200 valid daily returns in the beta window.
- Beta_i(m) = cov(r_i, r_SPY) / var(r_SPY) over the window ending at m (daily total returns; SPY TOTALRETURN).
- Selection: within each current GICS level-1 sector (Norgate, known look-ahead in labels; UNKNOWN is its own
  bucket), long the lowest 20% and short the highest 20% by beta (at least one name per side when a sector has
  >= 5 names). Equal dollars per name within each leg; each leg = 100% of NAV at m.
- Holding: from close m to close of the next decision; positions drift with prices (buy and hold between
  rebalances); a name that stops trading keeps its last value (return 0 after the last bar).
- Daily return: r(t) = [sum_long w_i(t-1) r_i(t) - sum_short w_j(t-1) r_j(t)] + f(t) - d / 252, with w the
  drifted leg weights per unit of NAV at t-1 and f(t) the lagged T-bill accrual on the collateral. Stock returns
  are total returns: r_i(t) = (Close_t + Dividend_{t-1}) / Close_{t-1} - 1 (Norgate stamps a dividend on the
  last cum session, as the engine books it).

Variants (the only ones allowed):
- V1: sector-neutral quintiles, 252-day beta.
- V2: no sector split (quintiles over the whole universe), 252-day beta.
- V3: sector-neutral quintiles, 126-day beta.

Selection and calibration (pre-declared): pick the variant with the highest correlation of monthly returns with
real BTAL (TOTALRETURN) over 2011-10 -> 2026-08; ties (difference < 0.01) go to V1. Then calibrate d so the
synthetic CAGR equals real BTAL's over 2011-09-13 -> 2026-08-19. Acceptance: monthly corr >= 0.80 and daily
corr >= 0.70; beta to SPY within 0.15 of BTAL's; reported beside it: calendar-year returns, 2020 crash and 2022.

### A3. Strategy-level validation (the decisive test)

Run each BTAL TAA sleeve through the real engine with the synthetic instruments used for the WHOLE history
(no real TQQQ/BTAL bars at all), and compare with the real inventory run on 2012-10-02 -> 2026-08-19:
- sleeves: `taa_btal_tqqq` (rank, 3x), `taa_btal_1n_tqqq` (1/N, 3x), `taa_btal_lin_qqq` (BTAL_QQQ, 1x QQQ);
- acceptance per sleeve: monthly return corr >= 0.90; |CAGR difference| <= 2.0 pp; |max drawdown difference|
  <= 5 pp. Also reported: mean monthly allocation overlap, 1 - 0.5 * sum|w_syn - w_real|.

If a sleeve fails A3, its 2008 numbers are shown with an explicit "approximate" flag, and the old stand-in beside.

### A4. The long-window series used in the tables

Engine run of each BTAL TAA sleeve with spliced instruments (synthetic before the fund's first bar, real after),
config start moved to 2006-01-01 so the first decision is 2008-02-29 (UUP needs 12 months of history) and the
first trade is at the 2008-03 open. The book tables take this run's daily returns only before 2012-10-02, and the
real inventory returns from 2012-10-02 on (same rule as before: the proxy fills only the dates before the real
sleeve exists). Every other sleeve uses its own real history. Industry-ETF DV2 before 2012-01-03 uses the DV2
study's research run (same rules; replica equal to the engine).

## Part B: defensive status check

Books (weights as in the YAMLs / earlier studies; annual reset): CORE5 alone; CORE5 + BTAL_QQQ 50/50; LT_DEF A1
(CORE5 55 / Tactical FI 27 / NDX 6 / MOSAIC 6 / TAA 3x 6); main-line Defensive (CORE5 33 / TFI 17 / EOM 17 /
HPI 9 / sector ETF dips 8 / NDX 8 / TAA 3x 8); and main-line Defensive without EOM (EOM's 17% spread pro rata
over the other pods). Reported: exact-window metrics (2012-10-02 -> 2026-08-19), long-window metrics and max
drawdown including 2008 with the new proxy, next to the old stand-in's figures. Nothing is re-selected here.

## Part C: growth shelf v2

### Books (fixed list; annual reset unless marked drift)

Engine cores (weights before the MR sleeve):
- G3: TAA 3x rank 50 / NDX-VXN 50 (the live pair).
- G3-1N: TAA 3x 1/N 50 / NDX-VXN 50.
- G4: TAA 3x 1/N 1/3 / NDX-VXN 1/3 / MOSAIC 1/3.

MR options (36% of the book, cores scaled to 64%; 36% is the owner-approved capsule size of 2026-09-26):
- none;
- stock pair: DV2 (WIRED) 18 / HPI vote 18;
- capsule: DV2 (WIRED) 12 / HPI vote 12 / industry-ETF DV2 12.

That gives nine books (3 cores x 3 MR options). The owner's ladders are added as they exist in `portfolios/`:
ladder_4 and ladder_4_1n (DV2 16 / HPI 17 / NDX 25 / MOSAIC 8 / TAA 34), each annual and drift. Thirteen books.

### Metrics

- Exact window 2012-10-02 -> 2026-08-19: CAGR, volatility, Sharpe (rf 0), max drawdown, Calmar, worst 12 months,
  worst calendar year, share of positive months, beta to the S&P 500 TR.
- Long window 2008-03-04 -> 2026-08-19 (Part A series): CAGR, Sharpe, max drawdown including 2008, Calmar,
  GFC return (2008-05-19 -> 2009-03-09).
- Stability: Sharpe 2012-10 -> 2021-12 and 2022-01 -> 2026-08.
- Crises: GFC, 2018 Q4, 2020 crash, 2022 bear, 2025 tariffs (episode windows of the growth dossier).
- Cost sensitivity (not a gate): +5 bps per side on every traded dollar: CAGR, Sharpe, Calmar.
- Ease of trading: pods, WIRED share of capital, book-level trading days per year, turnover x NAV per year,
  whether any pod trades daily, and what is missing before the book can run live.
- Capacity: last three years of fills, the growth-shelf route model (house auction limits for stocks, one-day
  worked ETF orders in the auction routes; worked up to 5 days and blocks for BTAL/UUP/DBC in the fund route).
  One fix, declared here: the short-hold industry-ETF DV2 orders cannot wait, so in the worked routes they are
  worked within one day (the old route function left urgent ETF orders uncosted). Reported: recommended AUM for
  MOO (today), MOC (planned) and worked+blocks, the first failing gate, and cost per year at $25M.

### Gates (owner's growth rules of 2026-09-24)

Sharpe >= 1.35 (exact window); Sharpe >= 1.20 in each of the two stability periods; max drawdown including 2008
(new proxy) no worse than -20%.

### Ordering

Among gate-passers, books are ordered by long-window Calmar (return per unit of the worst drawdown, 2008
included). Ease of trading and capacity are shown beside it and are not folded into one score: the final
recommendation weighs them in words and is labelled as judgement.

## Amendment log

- A1 (2026-09-26, before the first run): the universe rule ">= 200 valid daily returns in the beta window" cannot
  hold for V3's 126-day window. For V3 the minimum is 100 of 126 (the same ~80% share). V1 and V2 keep 200 of 252.
- A2 (2026-09-26, AFTER the A1 results; post-result decision, labelled as such): the daily-correlation criterion
  failed narrowly out of sample (QLD 0.987, SPXL 0.990 vs >= 0.99) and SPXL's cumulative gap was 5.6 pp (> 5).
  Diagnosis: the misses sit on 2008 crisis days where residuals flip sign on consecutive days (closing-price
  asynchrony, e.g. 2008-09-29 / 09-30); weekly corr 0.998 / 0.996, monthly 0.999 / 0.996; QLD calendar-year
  differences 0.2-1.8 pp; the GFC window (the thing the proxy is for) matches within 1 pp (-77.2% vs -78.1%).
  SPXL's early history likely tracked a different large-cap index (2009 gap 8.7 pp in one year), so it is a weak
  test. Decision: the synthetic TQQQ is used in the tables, with the failed criterion disclosed and the old
  stand-in's figures shown beside the new ones.
- A3 (2026-09-26, AFTER the A2 results of synthetic BTAL; post-result, labelled): V1 passes the correlation
  criteria (monthly 0.935, daily 0.821) but fails the beta criterion (-0.69 vs BTAL -0.50): the replica has about
  1.4x BTAL's amplitude (2022 +40% vs +20%). Fix: one exposure scale k = OLS slope of BTAL's daily excess return
  (r - f) on the replica's gross long/short return over 2011-09-13 -> 2026-08-19, then d recalibrated on the same
  window: r_syn = k * gross + f - d / 252 (the overnight part scaled by k too). Both the scaled (main) and the
  unscaled (sensitivity) versions go through A3; A3's acceptance rules are unchanged. This is fitted only on the
  overlap; nothing before 2011-09 is looked at to set it.
- A4 (2026-09-26, after the independent quant-pitfalls review; post-result, labelled):
  - H1: the engine charges max($1, $0.005 x shares) on split-adjusted share counts (pre-existing, conservative).
    TQQQ's split factor is 384 in 2010 and 96 in 2013, so its commissions ran at ~86 bps of notional in 2013 and up
    to ~5% of NAV a year in the proxy period. `commission_fix.py` re-prices every fill on the real share count
    (research only). All tables are shown twice: "engine costs" (matches every earlier study and Bench) and
    "commission-fixed". Verdicts are read on both.
  - M1: A3's k came from daily data, which stale BTAL closes attenuate (BTAL monthly beta -0.66 vs daily -0.50; a
    monthly fit gives k ~0.85). No new k is chosen: scaled (k 0.734, the conservative end for 2008) and unscaled
    (k 1) are reported as a bracket. An independent S&P 500 Low Volatility minus High Beta series corroborates the
    2008-09 path (GFC +44% vs +35% scaled / +51% unscaled).
  - M2: A2 overrode the pre-declared consequence of an A1 failure; the synthetic QLD was 0.2-1.8 pp a year
    optimistic out of sample. A +3%/yr financing stress on synthetic TQQQ moves book drawdowns by <= 0.2 pp.
  - M3: capacity attribution corrected: in the MR books both DV2 and HPI break the close-auction limit at $25M.
  - M4 and a correction to Part C's text: the Sharpe >= 1.20 stability gate and its 2012-21 / 2022-26 split were
    this study's choice (carried from the growth study's half-period rule), not the owner's rule, which is Sharpe
    >= 1.35 and drawdown <= 20% including 2008. The ordering's top is reported as a tie band.
