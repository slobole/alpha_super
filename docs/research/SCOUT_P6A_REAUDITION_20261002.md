# Scout P6a: the siblings of the two live families re-audited, and S3 for allocation and ranking families

Date: 2026-10-02. Research only; nothing here changes live trading.

**Scripts:** `scripts/research/scout_p6_reaudition_20261002/`, which hold:
- `run_s3.py` and `attach_s3.py`: S3 for the two live pods;
- `run_p6a.py`: the eight siblings.

**Where things live:**
- **Runner:** `alpha/scout/reaudit.py`.
- **Cards:** `results/scout/cards/` (not in git).
- **Ledger:** eight RETRO registrations, each the child of its family's P5 registration.

**Gates:** two agents extended the specs. Every sibling passes the exact identity gate against a fresh engine run,
with differences of 2e-16 to 7e-16:
- TAA 1/N, linearity and the 2x variants;
- NDX without VXN, NATR20, and NATR20 with VXN.

## Verdict

| Pod | Registry | Grade | Sharpe, live day / median day | MCPT p | Alpha t (fullest) | T-bill slot P |
|---|---|---|---|---|---|---|
| **TAA 3x** | LIVE | **CANDIDATE** | 1.18 / 0.92 | 0.010 | 2.71 | 0.92 |
| **TAA 3x 1/N** | WIRED | **CANDIDATE** | 1.08 / 0.84 | 0.032 | 2.09 | 0.82 |
| TAA linearity 1/N QQQ | WIRED | WATCHLIST (DSR live p 0.058) | 1.02 / 0.86 | 0.037 | 2.09 | 0.83 |
| **TAA 2x 1/N QLD** | PM_READY | **CANDIDATE** | 1.11 / 0.89 | 0.022 | 2.32 | 0.87 |
| TAA no-BTAL 2x 1/N QLD | PM_READY | WATCHLIST (alpha t 1.66) | 0.96 / 0.81 | 0.034 | 1.66 | 0.88 |
| TAA no-BTAL 2x 1/N SSO | PM_READY | WATCHLIST (alpha t 1.94) | 0.89 / 0.76 | 0.023 | 1.94 | 0.83 |
| NDX VXN | LIVE | WATCHLIST | 0.70 / 0.59 | 0.18 / 0.09 | 1.25 | 0.47 |
| NDX ATR | WIRED | WATCHLIST | 0.65 / 0.54 | 0.25 / 0.11 | 0.88 | 0.39 |
| **NDX NATR20** | research | WATCHLIST | 0.81 / 0.66 | **0.001** / 0.11 | 0.79 | 0.39 |
| **NDX NATR20 VXN** | research | WATCHLIST | 0.83 / 0.70 | **0.001** / 0.09 | 1.12 | 0.48 |

For NDX the MCPT column shows two numbers: stock selection / timing overlay.

## Three findings

**1. The TAA family's edge is the gated leveraged fallback, not the defensive rotation.**
- **What S3 found:** over the ETFs' full histories (to 2022), the momentum ranking of the defensive ETFs does not
  predict their next-month returns.
  - Fama-MacBeth slope: t 1.47.
  - Signal on minus signal off: t 0.70.
  - Per asset: none is significant.
- **Same for linearity:** t 1.56 and 1.24.
- **The VIX gate does predict risk:** next-month Nasdaq volatility is 1.75 times higher when the gate is off
  (p < 0.001).
- **Yet every TAA sibling passes the MCPT (p 0.010-0.037):**
  - Its score nets out generic volatility timing (A9).
  - What remains is holding the leveraged Nasdaq fallback only while realised volatility is below implied
    volatility.
- **Without BTAL:** the siblings lose their alpha after the trend factor and NDX. BTAL is part of the edge.

**2. The live NDX rule ranks by a number that depends on the share price.**
- **The rule:** its score is ROC12 divided by the 20-day ATR in DOLLARS.
- **Price dependence:** among rising members, that score has a median Spearman correlation of **−0.46 with the
  share price**, so cheaper shares rank higher for no economic reason.
- **NATR20 instead:** dividing by ATR as a percentage of price (NATR20) removes the price.
  - Only half of the two rules' monthly top 10 are the same stocks.
  - The NATR20 ranking carries real information: MCPT selection p **0.001**, rank IC t 3.33, top 10 over the
    eligible mean +0.79% a month (t 2.93), PBO 0.18, DSR p 0.005.
  - The dollar-ATR rules fail: p 0.18-0.25, PBO 0.77-0.92.
- **Calibration:** the per-asset null was calibrated for exactly this family (8% false passes at p ≤ 0.05), so
  p 0.001 is not a borderline result.

**3. No NDX rule adds value to this book (2012-2022).**
- **Overlay:** the timing overlay (SPY regime, with or without VXN) misses the gate (p 0.09-0.11).
- **Alpha:** once TAA 3x is a factor, the alpha is t 0.8-1.25.
- **T-bill slot:** replacing any NDX pod with T-bills leaves the book's Sharpe unchanged (1.18-1.20 against 1.21).
- **Why:** TAA 3x already holds leveraged Nasdaq exposure in calm markets, which is most of what an NDX momentum
  pod adds.

## What this means (owner decisions; no live change made)

- **TAA 3x and TAA 3x 1/N: CANDIDATE.**
  - Plan with the median decision day: Sharpe about 0.85-0.9.
  - Capacity is about $0.8M at 1% of ADV, because of BTAL.
- **NDX VXN (live).**
  - **The dollar-ATR score should be replaced by NATR20.** That rule has the same structure, its stock selection
    passes the gate decisively, and its in-sample Sharpe is higher (0.83 against 0.70).
  - **This is a live change and the owner's decision.** The recommended path is a paper or shadow run first.
    That agrees with the 2026-09-26 robustness study's "shadow = NATR20 + SMA200".
  - **The bigger question stays open:** whether this book needs an NDX pod at all, given TAA 3x. Scout's book
    test says the slot adds nothing in 2012-2022.
- **TAA linearity 1/N QQQ:** WATCHLIST only on the DSR warning for the live configuration (p 0.058). It is close
  to CANDIDATE, and unleveraged (CAGR 9.5%, max drawdown −11%).

## Method notes

- **Fast replicas** (gross, constant weights within the month) were checked against the engine on every grid:
  - TAA siblings: daily correlation 0.987-0.993, configuration ranking 0.85-0.96;
  - NDX: as in P5.
- **NDX timing overlay:** its MCPT uses the SPY regime alone when the pod has no VXN scaling.
- **Shared S3 rows:** the TAA siblings that share a score and universe share their S3 rows (the same signal is
  tested once).
- **Not done:**
  - capacity v2;
  - Fama-French factors;
  - the G3 reference book;
  - the post-adoption period for the siblings (they were never live).
