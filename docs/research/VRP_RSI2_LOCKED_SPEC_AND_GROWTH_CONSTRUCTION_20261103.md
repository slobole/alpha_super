# Locked `VRP_RSI2` Volatility Sleeve Specification & Updated Fund Portfolio Construction (`GROWTH` & `DEFENSIVE`)

**Date:** 2026-11-03  
**Prepared for:** Owner & Claude (Implementation / Strategy Build Handoff)  
**Related Scripts & Tests:**
- Peer-Review Rebuild Script: [`scripts/research/vrp_multipod_scout_20261101/run_clean_vrp_rebuild_20261102.py`](../../scripts/research/vrp_multipod_scout_20261101/run_clean_vrp_rebuild_20261102.py)
- Automated Unit Tests: [`tests/test_vrp_multipod_scout_20261102.py`](../../tests/test_vrp_multipod_scout_20261102.py) (`3 passed`)
- Detailed Peer-Review Audit Report: [`docs/research/VRP_MULTIPOD_SCOUT_20261101.md`](VRP_MULTIPOD_SCOUT_20261101.md) (Scout Registry ID `588`)

---

## Executive Summary & Owner Decisions

Following Claude's 14-point peer review of the initial VRP + Macro Collateral scout, we rebuilt the entire test suite from scratch, audited every parameter and sub-period, and reviewed the trade-offs with the owner. **Here are the final decisions locked by the owner:**

1. **We Drop the 5-Asset Macro Basket (`{GLD, DBC, UUP, TBT, BTAL}`):** Claude correctly showed that this 5-asset subset sat at the 98th percentile of 13 non-equity candidates (`0.819` median random 5-of-13 Sharpe) and acted primarily as post-hoc 2022 insurance.
2. **We Drop Gold (`{GLD}`) from the Volatility Sleeve's Idle Cash (Choosing Robustness Over Higher In-Sample Sharpe):**
   - While adding Gold Trend (`{GLD}`) to idle cash sat on a flat parameter plateau (`16.4%–16.7% CAGR | 0.981–0.993 Excess Sharpe` across all 9 `SMA × VolTarget` grid cells) and boosted `2019–2026` Excess Sharpe to `1.21` (`1.35` in `2023–2026`), a sub-period split revealed that Gold dragged the sleeve's Excess Sharpe down in **`2011–2018`** from **`0.846` to `0.667–0.710`**.
   - Furthermore, `TAA_1N` (`50%` of the Growth book) and `CORE5` (`20%` of the Growth book) already hold and trend-follow `GLD`.
   - **Owner Decision:** Prioritize pure regime robustness and zero cross-pod overlap. Idle cash inside the volatility sleeve stays in **`T-Bills`**.
3. **We KEEP the `$SPX RSI(2)` Boost Inside the Volatility Sleeve:**
   - Claude's review initially set `RSI(2)` aside on procedural grounds (it was added after observing the `0.856` base result).
   - We tested `RSI(n)` across **15 parameter combinations** (`n ∈ {2, 3, 4}` × thresholds `{15, 20, 25, 30, 35}`) on the clean `c = 0.98` contango gate. Every single cell beat the no-RSI baseline (`0.869` Excess Sharpe), forming a flat plateau of **`0.927–0.939` Excess Sharpe** (`15.0%–15.5% CAGR`) with the **exact same `-23.2% MaxDD`**, while improving **both** `2011–2018` (`0.809 -> 0.846`) and `2019–2026` (`0.969 -> 1.044`).
4. **We Lock the Contango Entry Threshold at `c = 0.98` (with a `0.98–1.00` Cash Buffer):**
   - Fixes the `c = 1.00` knife-edge fragility identified by Claude in Feb 2018 (`VIX 17.31 vs VIX3M 17.03`, `1.0164`) and Aug 2024 (`23.39 vs 23.34`, `1.0021`).
5. **Why `T+2` Execution Delay Is Not an Operational Blocker:**
   - Degrading when delayed by a full session (`T+2 Open`) is an expected property of a fast 1–3 day volatility/mean-reversion edge (identical to our live `DV2` and `HPI` pods). In our automated IBKR runner, signals are computed after `Close[t]` and submitted automatically for `Open[t+1]` (`MOO`).
6. **Updated Fund Portfolio Construction (`GROWTH` & `DEFENSIVE`):**
   - Section 5 below documents the complete replacement of the outdated `1/N MR` and `E2+cap Momentum` capsules with the **Liquidity-Sized MR Capsule (`50% dv2_g_l + 50% hpi_g_l`)** and **Closed 3-Leg Momentum Capsule (`A, B, C` across `k0 + k10`)**, along with the updated `GROWTH` ladder and `DEFENSIVE` verification.

---

## 1. Locked Strategy Specification: `VRP_RSI2` (`Option A` — Pure Volatility + `RSI(2)`, `c = 0.98`)

### 1.1 Universe & Instruments
| Leg | Ticker | Instrument Description | Pre-Feb 2018 Normalization | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Short Volatility** | `SVXY` | ProShares Short VIX Short-Term Futures ETF | Multiply target weight by `0.50` prior to `2018-02-28` (`SVXY` was `-1.0x` pre-Feb 2018 and `-0.5x` post-Feb 2018); remainder in `T-Bills` | Harvests VIX roll yield + volatility mean-reversion when term structure is in contango (`VIX / VIX3M < 0.98`) |
| **Long Volatility** | `VIXM` | ProShares VIX Mid-Term Futures ETF | `1.0x` throughout (unchanged leverage) | Convex crisis hedge when VIX term structure flips to backwardation (`VIX / VIX3M > 1.00`) and realized vol exceeds implied vol (`eVRP <= 0`) |
| **Idle Cash** | `BIL` / `T-Bills` | 3-Month U.S. Treasury Bill rate | Accrues daily on `1.0 - w_SVXY - w_VIXM` | Holds uninvested collateral (`~46%` average cash share) |
| **Signal Indices** | `$VIX`, `$VIX3M`, `$SPX`, `SPY` | CBOE VIX, CBOE 3-Month VIX, S&P 500 Index, SPDR S&P 500 ETF | Read-only signal inputs at `Close[t]` | Used exclusively to compute `VIX / VIX3M`, `eVRP`, and `RSI(2)` |

### 1.2 Exact Signal Equations (Evaluated Daily at `Close[t]`)

1. **10-Day Empirical Realized Volatility (`ERV_10`):**
   $$\text{ERV}_{10, t} = \text{StdDev}_{10}\left(r_{\text{SPY}, t-9 \dots t},\ \text{ddof}=1\right) \times \sqrt{252} \times 100$$
2. **Empirical Volatility Risk Premium (`eVRP`):**
   $$\text{eVRP}_t = \text{VIX}_t - \text{ERV}_{10, t}$$
3. **Term-Structure Ratio (`TS`):**
   $$\text{TS}_t = \frac{\text{VIX}_t}{\text{VIX3M}_t}$$
4. **Wilder's 2-Period RSI on `$SPX` (`RSI(2)`):**
   - Standard Wilder smoothing (`alpha = 1 / 2 = 0.50`) on daily `$SPX` close changes $\Delta P_t = P_t - P_{t-1}$:
     $$\text{AvgGain}_t = 0.5 \cdot \text{AvgGain}_{t-1} + 0.5 \cdot \max(\Delta P_t, 0), \quad \text{AvgLoss}_t = 0.5 \cdot \text{AvgLoss}_{t-1} + 0.5 \cdot \max(-\Delta P_t, 0)$$
     $$\text{RSI}(2)_t = 100 - \frac{100}{1 + \text{AvgGain}_t / \text{AvgLoss}_t}$$

### 1.3 Target Weight Rules at `Close[t]` (Executed at `Open[t+1]`)

Let $L_t = 0.50$ if $t < \text{2018-02-28}$ else $1.00$ (historical `SVXY` leverage normalization; in live trading $L_t = 1.00$):

* **Regime 1 — Contango + Positive eVRP ($\text{TS}_t < 0.98$ and $\text{eVRP}_t > 0$):**
  $$w_{\text{base}, t} = \min\left(2.0 \times \frac{\text{VIX}_t}{100},\ 1.0\right)$$
  $$w_{\text{SVXY}, t} = L_t \times \begin{cases} \min(1.25 \times w_{\text{base}, t},\ 1.0) & \text{if } \text{RSI}(2)_t < 25.0 \\ w_{\text{base}, t} & \text{otherwise} \end{cases}, \qquad w_{\text{VIXM}, t} = 0$$

* **Regime 2 — Contango + Negative eVRP ($\text{TS}_t < 0.98$ and $\text{eVRP}_t \le 0$):**
  $$w_{\text{SVXY}, t} = L_t \times \begin{cases} \min\left(1.50 \times \frac{\text{VIX}_t}{100},\ 1.0\right) & \text{if } \text{RSI}(2)_t < 20.0 \\ \min\left(1.00 \times \frac{\text{VIX}_t}{100},\ 1.0\right) & \text{otherwise} \end{cases}, \qquad w_{\text{VIXM}, t} = 0$$

* **Regime 3 — Neutral Buffer Zone ($0.98 \le \text{TS}_t \le 1.00$):**
  $$w_{\text{SVXY}, t} = 0, \qquad w_{\text{VIXM}, t} = 0 \quad (\text{100\% Cash in T-Bills})$$

* **Regime 4 — Backwardation + Negative eVRP ($\text{TS}_t > 1.00$ and $\text{eVRP}_t \le 0$):**
  $$w_{\text{SVXY}, t} = 0, \qquad w_{\text{VIXM}, t} = \min\left(1.25 \times \frac{\text{VIX}_t}{100},\ 1.0\right)$$

* **Regime 5 — Backwardation + Positive eVRP ($\text{TS}_t > 1.00$ and $\text{eVRP}_t > 0$):**
  $$w_{\text{SVXY}, t} = 0, \qquad w_{\text{VIXM}, t} = 0 \quad (\text{100\% Cash in T-Bills})$$

### 1.4 Execution & Rebalance Band Accounting (`Open[t+1]`)
- **Rebalance Band (`5%`):** At `Open[t+1]`, trade asset $i \in \{\text{SVXY}, \text{VIXM}\}$ from current weight $w_{i, \text{curr}}$ to target weight $w_{i, t}$ **if and only if**:
  1. $|w_{i, t} - w_{i, \text{curr}}| > 0.05$, **OR**
  2. $w_{i, t} = 0$ and $w_{i, \text{curr}} > 0$ (full exit Always executes immediately), **OR**
  3. $w_{i, \text{curr}} = 0$ and $w_{i, t} > 0$ (new entry Always executes immediately).
- **Daily Return Formula (`Close[t] -> Close[t+1]`):**
  - From `Close[t]` to `Open[t+1]`, the portfolio earns the overnight return on the **pre-trade** weights $w_{\text{pre}}$.
  - At `Open[t+1]`, trades execute to $w_{\text{post}}$, paying one-way transaction cost $\sum_i |w_{i, \text{post}} - w_{i, \text{pre}}| \times c_i$.
  - From `Open[t+1]` to `Close[t+1]`, the portfolio earns the intraday return on $w_{\text{post}}$, plus the daily risk-free rate $r_{f, t+1}$ on residual cash $(1 - \sum_i w_{i, \text{post}})$.

---

## 2. Quantitative Robustness Evidence for the Locked Decisions

### 2.1 Why `RSI(2)` Is Kept: Full 15-Cell `RSI(n)` Parameter Plateau (`c = 0.98`, `2011–2026`)
Every single lookback (`RSI(2)`, `RSI(3)`, `RSI(4)`) and every threshold (`15` through `35`) beats the no-RSI baseline (`th = 0`), with zero increase in Max Drawdown (`-23.2%`) and consistent gains in both halves of the sample:

| RSI Lookback | Threshold (`th` / `th-5`) | Full CAGR (`5bp`) | Full Excess Sharpe | MaxDD | Annual Turnover | `2011–2018` Excess Sharpe | `2019–2026` Excess Sharpe |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No RSI (`Base`)** | `0` (Disabled) | `13.7%` | `0.869` | `-23.2%` | `4.23x` | `0.809` | `0.969` |
| **`RSI(2)`** | `15 / 10` | `15.0%` | `0.927` | `-23.2%` | `5.21x` | `0.865` | `1.026` |
| **`RSI(2)`** | `20 / 15` | `15.3%` | `0.939` | `-23.2%` | `5.55x` | `0.846` | `1.053` |
| **`RSI(2)` (Locked)** | **`25 / 20`** | **`15.3%`** | **`0.933`** | **`-23.2%`** | **`5.80x`** | **`0.846`** | **`1.044`** |
| **`RSI(2)`** | `30 / 25` | `15.4%` | `0.935` | `-23.2%` | `6.01x` | `0.834` | `1.052` |
| **`RSI(2)`** | `35 / 30` | `15.5%` | `0.934` | `-23.2%` | `6.21x` | `0.814` | `1.061` |
| **`RSI(3)`** | `20 / 15` | `15.1%` | `0.940` | `-23.2%` | `4.97x` | `0.911` | `1.026` |
| **`RSI(3)`** | `25 / 20` | `15.2%` | `0.937` | `-23.2%` | `5.27x` | `0.871` | `1.038` |
| **`RSI(3)`** | `30 / 25` | `15.1%` | `0.927` | `-23.2%` | `5.47x` | `0.841` | `1.037` |
| **`RSI(4)`** | `25 / 20` | `14.9%` | `0.931` | `-23.2%` | `4.87x` | `0.907` | `1.013` |
| **`RSI(4)`** | `30 / 25` | `15.4%` | `0.949` | `-23.2%` | `5.18x` | `0.926` | `1.030` |

### 2.2 Why Gold (`{GLD}`) Was Excluded from Idle Cash: Option A vs Option B (`c = 0.98`)
We tested a 9-cell grid for the Gold Trend overlay (`SMA(20,100)`, `SMA(50,126)`, `SMA(50,200)` × `VolTarget ∈ {6%, 8%, 10%}`) on top of `Boost Short-Vol (c = 0.98, WITH RSI2)`:

| Variant (`c = 0.98`, `WITH RSI2`) | Full CAGR | Full Excess Sharpe | MaxDD | Turnover | `2011–2018` XS | `2019–2026` XS | `2023–2026` CAGR / XS | Avg `GLD` Weight |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Option A: NO GOLD (`T-Bills` only — Locked)** | **`15.3%`** | **`0.933`** | **`-23.2%`** | **`5.80x`** | **`0.846`** | **`1.044`** | **`14.3% / 0.89`** | `0.0%` |
| **Option B1: Gold `SMA(20,100)` `VT=6%`** | `16.5%` | `0.987` | `-23.2%` | `7.50x` | `0.704` | `1.198` | `21.2% / 1.34` | `11.5%` |
| **Option B2: Gold `SMA(50,126)` `VT=6%`** | `16.5%` | `0.982` | `-23.2%` | `7.05x` | `0.667` | `1.209` | `21.4% / 1.35` | `13.2%` |
| **Option B3: Gold `SMA(50,200)` `VT=8%`** | `16.6%` | `0.987` | `-23.2%` | `7.04x` | `0.703` | `1.199` | `21.5% / 1.35` | `13.2%` |

* **Conclusion:** Option A is balanced across both halves of history (`0.846` and `1.044`), trades only 2 tickers (`SVXY`, `VIXM`), has 3 fewer parameters, and avoids duplicating the `GLD` trend exposure already held in `TAA_1N` (`50%` of the fund) and `CORE5`.

### 2.3 Standalone Stress Matrix (`c = 1.00` vs `c = 0.98`, Base vs `RSI(2)`)
| Variant (`2011-10-10` to `2026-08-19`) | CAGR (`5bp`) | XS (`5bp`) | MaxDD | XS (`10bp`) | XS (`15/35bp`) | No-COVID XS | No-2022 XS | `2023–2026` CAGR / XS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1. Base Short-Vol (c=1.00, No RSI2)` | `14.1%` | `0.872` | `-23.2%` | `0.844` | `0.703` | `0.857` | `0.897` | `11.6% / 0.67` |
| `2. Boost Short-Vol (c=1.00, WITH RSI2)` | `15.8%` | `0.935` | `-23.2%` | `0.898` | `0.709` | `0.923` | `0.963` | `14.1% / 0.82` |
| `3. Base Short-Vol (c=0.98, No RSI2)` | `13.7%` | `0.869` | `-23.2%` | `0.839` | `0.691` | `0.863` | `0.876` | `12.0% / 0.76` |
| **`4. Locked VRP_RSI2 (c=0.98, WITH RSI2)`** ⭐ | **`15.3%`** | **`0.933`** | **`-23.2%`** | **`0.894`** | **`0.699`** | **`0.931`** | **`0.941`** | **`14.3% / 0.89`** |

---

## 3. What `VRP_RSI2` Adds Relative to the Existing `GROWTH` Pods

### 3.1 Daily Excess-Return Correlations (`2011–2026` Full vs Excluding `2020`)

| Daily Excess Return Correlation (`2011–2026` Full) | `VRP_RSI2` (`c=0.98`) | `TAA_1N` | `MR_Sized` | `CORE5` | `MOM_3Leg` | `SPY` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`VRP_RSI2` (`c=0.98`)** | **`1.000`** | **`0.282`** | **`0.080`** | **`0.239`** | **`0.293`** | **`0.247`** |
| **`TAA_1N`** | `0.282` | `1.000` | `0.382` | `0.458` | **`0.686`** | `0.628` |
| **`MR_Sized` (`dv2_g_l + hpi_g_l`)** | **`0.080`** | `0.382` | `1.000` | `0.220` | `0.462` | `0.657` |
| **`CORE5`** | `0.239` | `0.458` | `0.220` | `1.000` | `0.369` | `0.329` |
| **`MOM_3Leg` (`A,B,C x k0,k10`)** | `0.293` | **`0.686`** | `0.462` | `0.369` | `1.000` | `0.640` |

* **Excluding `2020` (`Non-COVID` correlation):** `VRP_RSI2` correlates **`0.336`** with `MR_Sized`, **`0.286`** with `CORE5`, **`0.475`** with `TAA_1N`, **`0.516`** with `MOM_3Leg`, and **`0.605`** with `SPY`.

### 3.2 Crisis & Regime Window Comparison (`Total Return %`)

| Crisis / Regime Window | **`VRP_RSI2` (`c=0.98`)** | `TAA_1N` | `MR_Sized` | `CORE5` | `MOM_3Leg` | `SPY` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2018 Volmageddon` (`Jan 26 – Feb 12, 2018`)** | **`-3.1%`** | `-9.3%` | `+4.0%` | `-2.7%` | `-7.5%` | `-6.3%` |
| **`2018 Q4 Selloff` (`Oct 1 – Dec 24, 2018`)** | **`-6.3%`** | `-9.5%` | `-6.8%` | `-0.1%` | `-7.2%` | `-18.9%` |
| **`2020 COVID Crash` (`Feb 19 – Mar 23, 2020`)** | **`+44.6%`** | `-13.5%` | `-18.2%` | `+1.5%` | `-22.2%` | `-33.4%` |
| **`2022 Rate/Bear Year` (`Jan 3 – Dec 30, 2022`)** | **`+5.2%`** | `-9.9%` | `+15.1%` | `+4.0%` | `-13.9%` | `-18.2%` |
| **`Aug 2024 Yen Unwind` (`Jul 31 – Aug 8, 2024`)** | **`-5.8%`** | `-0.1%` | `+8.2%` | `-0.2%` | `+0.4%` | `-2.1%` |
| **`Apr 2025 Tariff Crash` (`Feb 19 – Apr 8, 2025`)** | **`-7.6%`** | `-13.3%` | `-6.6%` | `-3.7%` | `-16.5%` | `-18.6%` |

### 3.3 Why This Complements `GROWTH`:
1. **`TAA_1N` and `MOM_3Leg` are slow/medium price-trend engines (`0.686` correlation)** that get caught together in sudden V-shaped crashes (`2020 COVID`, `Apr 2025`) and choppy bear markets (`2022`).
2. **`MR_Sized` buys short-term equity dips**, which works in normal pullbacks (`2022`, `Aug 2024`) but loses `-18.2%` in a liquidation crash (`2020 COVID`).
3. **`VRP_RSI2` reads the VIX futures term structure (`VIX / VIX3M`) daily:** when a crash is systemic (`VIX > VIX3M`), it immediately exits `SVXY` and buys `VIXM` (`+44.6%` in COVID), while in normal or grinding markets it harvests volatility carry (`15.3% CAGR`, nearly double `CORE5`'s `8.4% CAGR`).

### 3.4 Capacity & Tax Guardrails (`SVXY` & `VIXM`)
- **Liquidity Cap (`VIXM`):** `60-day` median daily dollar volume (`ADV60`) is **`$72.7M` for `SVXY`** and **`$3.2M` for `VIXM`**. At a `1%` ADV participation limit (`~$32k` single-session trade in `VIXM`), a **`10%` satellite weight** supports up to **`~$1.3M` total fund AUM** with standard `MOO` orders, or **`$3M–$5M` AUM** if `VIXM` entries/exits are split across `Open` + `TWAP`/`VWAP`.
- **Tax / K-1 Status:** Both `SVXY` and `VIXM` are series of **ProShares Trust II** (a Delaware statutory commodity pool) and issue a **Schedule K-1**. Per ProShares Trust II's official Qualified Notice under Treas. Reg. §1.1446-4(b)(4), neither fund generates U.S. Effectively Connected Income (`ECI`), making them **100% exempt from the 10% IRC Section 1446(f) PTP gross-proceeds withholding tax** for non-U.S. investors.

---

## 4. Updated Portfolio Construction (`GROWTH` & `DEFENSIVE`)

### 4.1 Capsule Upgrades Applied (`מוצרי הקרן.html` Update)
1. **Updated `MR_Sized` Capsule (`50% dv2_g_l + 50% hpi_g_l` from `results/scout/growth_mr_products_20261101/pods/`):**
   - Replaces old `1/N MR` (`50% dv2_g_n + 50% hpi_g_n`).
   - Standalone (`2008–2026`): **`19.91% CAGR | 1.139 Excess Sharpe | -18.5% MaxDD`** (vs old `1/N MR`: `18.18% CAGR | 1.077 Excess Sharpe | -20.5% MaxDD`).
   - Improves `CAGR` by `+1.73%/yr`, `Excess Sharpe` by `+0.062`, and `MaxDD` by `+2.0%`, while solving small-end R1000 capacity via `$ADV20` liquidity-proportional sizing (`5%` stock cap).
2. **Updated `MOM_3Leg` Capsule (`1/6` each across `A|k0, B|k0, C|k0, A|k10, B|k10, C|k10` from `results/scout/residual_momentum_20261031/report_close/legs.parquet`):**
   - Replaces old 2-leg `E2+cap` (`50% atr_cap_s + 50% natr_cap_s`).
   - Combines **Leg A** (`NDX NATR + cap + buf18`), **Leg B** (`NDX Residual Mom + cap + buf18`), and **Leg C** (`R1000 Industry-Adjusted Residual Mom + cap4 + buf35 + inv-vol`), staggered across two 21-day rebalance schedules 10 sessions apart (`k0` and `k10`).
   - Standalone (`2008–2026`): **`14.33% CAGR | 0.870 Excess Sharpe | -22.9% MaxDD`** (vs old `E2+cap`: `16.43% CAGR | 0.830 Excess Sharpe | -25.1% MaxDD`), and **cuts the `Jul–Oct 2026` momentum crash in half (`-9.2%` vs `-18.4%`)**.

### 4.2 Complete Updated `GROWTH` Portfolio Matrix (Annual Rebalancing, `2008-03-04` to `2026-08-19` + Extension to `2026-10-14`)

All portfolios use **annual January rebalancing** (`hunt.book_annual`). Drawdown tail probabilities `P(DD < L)` use the project's 63-day stationary bootstrap (`2,000` paths, seed `20260929`).

| Portfolio / Candidate | Weights (`TAA / MR_Sized / MOM_3Leg / CORE5 / VRP_RSI2`) | CAGR (`08–26`) | Excess Sharpe | Vol | MaxDD | `P(DD < -22%)` | Rung Passed | `+5 bps` CAGR / XS | Plan `3/4 Edge` CAGR / XS | Stress `1/2 Edge +5bp` | `Jul–Oct 2026` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Old `REC` (Baseline Reference)** | `50 TAA_1N / 30 MR_1N / 0 / 20 CORE5 / 0` | `20.87%` | `1.258` | `14.5%` | `-15.6%` | `3.9%` | `GROWTH` | `20.51% / 1.239` | `16.01% / 0.944` | `10.97% / 0.610` | `-4.6%` |
| **1. `Sized REC` (3-Pillar Default)** ⭐ | **`50 TAA_1N / 30 MR_Sized / 0 / 20 CORE5 / 0`** | **`21.88%`** | **`1.304`** | **`14.6%`** | **`-14.0%`** | **`2.7%`** | **`GROWTH`** | **`21.52% / 1.285`** | **`16.77% / 0.978`** | **`11.47% / 0.633`** | **`-4.8%`** |
| **2. `Sized REC + 10% VRP_RSI2`** ⭐ | **`45 TAA_1N / 27 MR_Sized / 0 / 18 CORE5 / 10 VRP`** | **`21.01%`** | **`1.346`** | **`13.7%`** | **`-12.6%`** | **`0.8%`** | **`MOD`** | **`20.66% / 1.326`** | **`16.12% / 1.010`** | **`11.05% / 0.654`** | **`-4.3%`** |
| **3. `Balanced 4-Pillar` (With `MOM_3Leg`)** ⭐ | **`45 TAA_1N / 25 MR_Sized / 10 MOM3 / 20 CORE5 / 0`** | **`20.99%`** | **`1.287`** | **`14.2%`** | **`-13.9%`** | **`2.4%`** | **`GROWTH`** | **`20.67% / 1.269`** | **`16.10% / 0.965`** | **`11.05% / 0.626`** | **`-4.9%`** |
| **4. `5-Pod Complete` (`+MOM_3Leg +VRP`)** ⭐ | **`40 TAA_1N / 25 MR_Sized / 10 MOM3 / 15 CORE5 / 10 VRP`** | **`20.68%`** | **`1.332`** | **`13.6%`** | **`-12.8%`** | **`1.1%`** | **`MOD`** | **`20.36% / 1.313`** | **`15.87% / 0.999`** | **`10.90% / 0.647`** | **`-4.4%`** |
| **5. `Growth Core` (`45 / 35 / 10 / 10`)** | `45 TAA_1N / 35 MR_Sized / 10 MOM3 / 10 CORE5 / 0` | `22.58%` | `1.309` | `15.1%` | `-14.6%` | `3.9%` | `GROWTH` | `22.21% / 1.289` | `17.29% / 0.982` | `11.81% / 0.635` | `-5.3%` |
| **6. `Growth Core 3x` (`TAA 3x` instead of `1N`)** | `45 TAA_3x / 35 MR_Sized / 10 MOM3 / 10 CORE5 / 0` | `23.43%` | `1.318` | `15.5%` | `-16.0%` | `5.2%` | `GROWTH` | `23.05% / 1.298` | `17.92% / 0.988` | `12.23% / 0.639` | `-5.0%` |
| **7. `Growth Aggressive` (`No CORE5`, `45/35/20`)** | `45 TAA_3x / 35 MR_Sized / 20 MOM3 / 0 / 0` | `24.47%` | `1.301` | `16.4%` | `-17.4%` | `10.1%` | `AGG_25` | `24.11% / 1.284` | `18.70% / 0.976` | `12.75% / 0.633` | `-5.9%` |
| **8. `Max Octane` (`50 TAA_3x / 35 MR / 15 MOM3`)** | `50 TAA_3x / 35 MR_Sized / 15 MOM3 / 0 / 0` | `25.13%` | `1.320` | `16.6%` | `-16.9%` | `9.0%` | `AGG_25` | `24.75% / 1.301` | `19.19% / 0.990` | `13.07% / 0.641` | `-5.4%` |

### 4.3 Annual Rebalance Robustness Across All 12 Reset Months & Frequencies
A critical requirement is that annual rebalancing (`January`) is not a lucky calendar artifact:
- Across all **12 possible annual rebalance months (`Jan` through `Dec`)**:
  - **`Sized REC (50/30/20)`:** Excess Sharpe ranges from **`1.289` to `1.316`** (mean **`1.303`**; January is `1.304`, right at the median), CAGR **`21.7%–22.1%`**, MaxDD **`-13.9% to -14.8%`**.
  - **`Balanced 4-Pillar (45/25/10/20)`:** Excess Sharpe ranges from **`1.274` to `1.298`** (mean **`1.286`**; January is `1.287`), CAGR **`20.8%–21.2%`**, MaxDD **`-13.8% to -14.7%`**.
  - **`Growth Core (45/35/10/10)`:** Excess Sharpe ranges from **`1.295` to `1.321`** (mean **`1.308`**; January is `1.309`), CAGR **`22.4%–22.8%`**, MaxDD **`-14.5% to -15.4%`**.
- **Monthly / Quarterly / Annual / Drift:** Annual rebalancing beats pure drift by `+0.05` to `+0.07` Sharpe while adding zero extra turnover vs Monthly/Quarterly.

### 4.4 `DEFENSIVE` Product Verification (`מוצרי הקרן.html`)
We also verified the `DEFENSIVE` product tier from `מוצרי הקרן.html`:
- **Live Next Step (`הצעד הבא`: `1/3 CORE5 + 1/3 BTAL_QQQ + 1/3 DV2-IND`):**
  - Full (`2008–2026`): **`8.8% CAGR | 1.26 Excess Sharpe | 5.7% Vol | -6.5% MaxDD`**, `P(DD < -17%) = 0.0%` (`MOD` rung).
- **Ideal Target (`יעד אידיאלי`: `25% CORE5 + 25% BTAL_QQQ + 25% DV2-IND + 25% EOM Flow`):**
  - On Lab `EOM Flow`: **`10.2% CAGR | 1.60 Excess Sharpe | 5.3% Vol | -5.3% MaxDD`**.
  - On conservative **Plan `EOM Flow` (`0.95` Sharpe haircut)**: **`9.6% CAGR | 1.50 Excess Sharpe | 5.3% Vol | -5.3% MaxDD`**.
- **Verdict on `DEFENSIVE`:** Completely solid and requires no structural changes (neither `MR_Sized` nor `MOM_3Leg` belongs inside `DEFENSIVE`, as `DEFENSIVE` uses `DV2-IND` rather than `Growth MR`).
