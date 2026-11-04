# VRP Multi-Pod Scout, Macro Collateral & Inflation Compass Audit (2026-11-01, Peer-Review Rebuild 2026-11-02)

> [!IMPORTANT]
> **Status & Governance (`Tier 1 — Research Only`):**
> - **Owner Decision & Locked Specification (2026-11-03):** Following review of this rebuild, the owner locked **Option A (`VRP_RSI2`: `SVXY + VIXM + RSI(2)` at `c = 0.98`, idle cash in `T-Bills`, no Gold/Macro collateral)** for strategy handoff, alongside the updated `GROWTH` & `DEFENSIVE` portfolio construction. See the complete handoff document: [`docs/research/VRP_RSI2_LOCKED_SPEC_AND_GROWTH_CONSTRUCTION_20261103.md`](VRP_RSI2_LOCKED_SPEC_AND_GROWTH_CONSTRUCTION_20261103.md).
> - **Not locked as Pod #4.** Per the peer-review protocol (2026-11-02), this strategy is **not** locked into the production book, no master 5-pod adoption is enacted, and the search for additional families is paused until the owner reviews this clean rebuild.
> - **Scout Registration:** Registered in [`research_ledger/scout_ledger.jsonl`](../../research_ledger/scout_ledger.jsonl) as row `588` (`id = vrp_multipod_scout_retro_20261102`, parent `evrp_dual_signal_vol_sleeve_retro_20261017`, family `volatility_premium_and_hedges`, hash `0cf53b96214b2ed65e3373a891397c2564fe8889a87aa5d9b962a2750eb0ebb2`).
> - **Frozen Spec & Scripts:** Frozen prior to the clean rebuild in [`scripts/research/vrp_multipod_scout_20261101/SPEC_FROZEN.md`](../../scripts/research/vrp_multipod_scout_20261101/SPEC_FROZEN.md), executed via [`run_clean_vrp_rebuild_20261102.py`](../../scripts/research/vrp_multipod_scout_20261101/run_clean_vrp_rebuild_20261102.py), and verified by [`tests/test_vrp_multipod_scout_20261102.py`](../../tests/test_vrp_multipod_scout_20261102.py).

---

## 1. Executive Summary & Attribution to Prior Literature

### 1.1 Citation & Prior Internal Work (Fixing Item 8)
This volatility sleeve is **not** a new discovery. It is an adaptation of:
- **Academic / Practitioner Source:** Ali Aziz & Carlo Zarattini, *"Expected Variance Risk Premium: A Dual-Signal Approach to Selling Volatility"* (Concretum Group, SSRN `5316487`, June 14, 2026).
- **Prior Internal Study:** [`docs/research/knowledge-base/studies/evrp_dual_signal_vol_sleeve_study.md`](knowledge-base/studies/evrp_dual_signal_vol_sleeve_study.md) (Scout ledger row `525`, `evrp_dual_signal_vol_sleeve_retro_20261017`), which evaluated the original `SVXY + VIXY + SHY` strategy over `2007-01` to `2026-06` and placed it on the **Watchlist** at **`7.7% CAGR / 0.632 raw Sharpe (0.530 excess Sharpe) / -32.8% MaxDD`**, explicitly noting that the February 2018 exit relied on thin contango-gate luck (`VIX 17.31` vs `VIX3M 17.03` on Feb 2, 2018).
- **Prior Volatility Futures Study:** [`docs/research/knowledge-base/studies/quantseeker_volatility_study.md`](knowledge-base/studies/quantseeker_volatility_study.md) (Scout ledger row `482`), which constructed Cboe constant-maturity VX futures (`vx_st_ret` for 30-day M1/M2 and `vx_mt_ret` for 150-day M4–M7 back to `2004-06`) and identified `VIXM` liquidity as the binding ETP capacity bottleneck.

### 1.2 Step-by-Step Bridge from `0.632` (Pakal Study) to `0.856 / 0.872` (Base Pod)
Why did the prior internal study report `7.7% CAGR / 0.632 raw Sharpe (0.530 excess Sharpe) / -32.8% MaxDD` (`2011-10-10` to `2026-06-12` gives `7.6% CAGR / 0.632 Sharpe0 / 0.530 xs`), whereas our base short-vol pod reports **`13.9% CAGR / 0.856 excess Sharpe / -23.2% MaxDD`** (`2011-10-10` to `2026-10-30`, or **`14.1% CAGR / 0.872 excess Sharpe`** on the `2011-10-10` to `2026-08-19` lab window)?

| Step | Configuration (`2011-10-10` to `2026-08-19` common window) | CAGR | Excess Sharpe (`xs`) | Raw Sharpe (`Sharpe0`) | MaxDD | What Changed? |
|---|---|---:|---:|---:|---:|---|
| **1. Pakal `S4_full_P`** | Synthetic `SVXYH` (`-0.5x`), `1.0x VIX/100` sizing, `VIXY` (`1.0x`), `SHY` cash (`15/35 bps`) | **7.6%** | **0.530** | **0.632** | **-32.8%** | Paper's literal `VIX/100` formula on `-0.5x` `SVXYH` (halves notional vs pre-2018 `-1x` ETP). |
| **2. Pakal `S4_full_A`** | Synthetic `SVXYH` (`-0.5x`), `2.0x VIX/100` sizing, `VIXY` (`1.0x`), `SHY` cash (`15/35 bps`) | **12.5%** | **0.703** | **0.779** | **-32.8%** | Restores Concretum's economic `-1x * VIX/100` notional via `2.0 * VIX/100` on `-0.5x` `SVXYH` (+0.173 `xs`). |
| **3. Real `SVXY` + `VIXM` (15/35 bps)** | Actual Norgate `SVXY` (`0.50` pre-2018-02-28), `VIXM` (`1.25x`) instead of `VIXY`, T-bill cash, `5%` band, `15/35 bps` | **11.3%** | **0.703** | **0.779** | **-23.2%** | Replacing front-month `VIXY` with mid-curve `VIXM` cuts March 2020 whipsaw MaxDD from `-32.8%` to `-23.2%`. |
| **4. Base Pod (5 bps)** | Same as Step 3, evaluated at `5 bps` transaction cost | **14.1%** | **0.872** (`0.856` to `2026-10-30`) | **0.941** | **-23.2%** | Lowering vol ETP cost assumption from `35 bps` to `5 bps` adds `+0.169 xs` (`+2.8%` CAGR). |

*(Source: [`results/research/vrp_multipod_scout_20261101/evrp_ablation_and_bridge_20261102.csv`](../../results/research/vrp_multipod_scout_20261101/evrp_ablation_and_bridge_20261102.csv))*

---

## 2. Sleeve 1: Base Short-Vol Pod (`EVRP_SVXY_VIXM_BASE`, No RSI2)

### 2.1 Frozen Signal Rules & Execution Contract
Per peer review, the `RSI(2)` dip-boost is **dropped** from the clean baseline, leaving the pure Base Short-Vol Sleeve (`EVRP_SVXY_VIXM_BASE`):
1. **Observables at `Close[t]`:**
   - `VIX[t]`: Cboe VIX Index close (`$VIX`).
   - `VIX3M[t]`: Cboe 3-Month Volatility Index close (`$VIX3M`).
   - `eRV10[t]`: 10-session sample standard deviation of `SPY` total-return close-to-close percent changes, annualized (`* sqrt(252) * 100`).
   - `eVRP[t] = VIX[t] - eRV10[t]`.
2. **Target Weights at `Close[t]` (Executed at `Open[t+1]`):**
   - Leverage normalization: `lev_adj = 0.50` for `t < 2018-02-28` (when `SVXY` was `-1.0x`), `1.00` for `t >= 2018-02-28` (when `SVXY` is `-0.5x`).
   - **Regime 1 — Full Contango Harvest (`VIX[t] < VIX3M[t]` and `eVRP[t] > 0`):**
     `w_SVXY[t] = min(2.0 * (VIX[t] / 100), 1.0) * lev_adj`, `w_VIXM[t] = 0`.
   - **Regime 2 — Half-Size Contango (`VIX[t] < VIX3M[t]` and `eVRP[t] <= 0`):**
     `w_SVXY[t] = min(1.0 * (VIX[t] / 100), 1.0) * lev_adj`, `w_VIXM[t] = 0`.
   - **Regime 3 — Backwardation + Realized Vol Explosion (`VIX[t] > VIX3M[t]` and `eVRP[t] <= 0`):**
     `w_SVXY[t] = 0`, `w_VIXM[t] = min(1.25 * (VIX[t] / 100), 1.0)`.
   - **Regime 4 — Backwardation with Positive `eVRP` (`VIX[t] >= VIX3M[t]` and `eVRP[t] > 0`):**
     `100%` Cash (T-bills).
   - **Rebalance Band:** `5%` weight-change threshold on sizing adjustments within the same regime; immediate execution (`0%` band) on any regime switch.

### 2.2 Component Ablation, Cost Sensitivity, Delay & Crisis Blanking (Items 1, 7)
What actually drives the Base Short-Vol sleeve (`2011-10-10` to `2026-08-19`, one-way annual turnover `sum(|dw|)/2 * 252`)?

| Variant (`2011-10-10` to `2026-08-19`) | CAGR | Excess Sharpe (`xs`) | MaxDD | 2022–26 `xs` | 2018 Volmageddon | 2020 COVID (`Feb 19–Apr 30`) | 2022 Bear | SPY Corr | 1-Way Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. `SVXY` Always (`VIX`-scaled, no gates, 5 bps)** | 9.9% | 0.552 | -50.5% | 0.675 | -21.5% | -42.1% | -4.9% | +0.77 | 1.1x |
| **2. Contango Gate Only (`VIX < VIX3M`, 5 bps)** | 11.6% | 0.850 | -16.3% | 0.393 | -2.4% | -3.2% | -10.2% | +0.61 | 2.7x |
| **3. Gate + `eVRP` Half-Size (`No VIXM`, 5 bps)** | 10.7% | 0.845 | **-14.4%** | **0.509** | -2.6% | -2.9% | -7.6% | +0.60 | 3.6x |
| **4. Base Pod (`Gate + eVRP + VIXM 1.25x`, 5 bps)** | **14.1%** | **0.872** (`0.856` to `10/30`) | -23.2% | 0.500 | -2.6% | **+44.5%** | -7.6% | **+0.26** | **4.1x** |
| **5. Base Pod (`Drifted Weights`, 5 bps)** | 13.8% | 0.850 | -23.4% | 0.523 | -3.1% | +43.8% | -6.9% | +0.26 | 4.3x |
| **6. Base Pod (`+5 bps` = `10 bps` cost)** | 13.6% | 0.844 | -23.2% | 0.452 | -2.6% | +44.4% | -8.4% | +0.26 | 4.1x |
| **7. Base Pod (`15/35 bps` Vol ETP cost)** | 11.3% | **0.703** | -23.3% | 0.212 | -2.6% | +43.6% | -12.4% | +0.26 | 4.1x |
| **8. Base Pod (`1 Session Late: T+2 Open`, 5 bps)** | 9.6% | **0.612** (`0.605` to `10/30`) | -25.3% | 0.412 | **-14.6%** | +22.8% | -11.2% | +0.31 | 4.1x |
| **9. Base Pod (`COVID Blanked`: Feb–Apr 2020 = RF)** | 10.9% | **0.857** (`0.79` raw) | -14.4% | 0.500 | -2.6% | +0.3% | -7.6% | +0.57 | 4.1x |
| **10. Base Pod (`2022 Blanked`: 2022 = RF)** | 14.1% | **0.897** | -23.2% | 0.591 | -2.6% | +44.5% | +0.6% | +0.20 | 4.1x |
| *Ref: Base + RSI2 (`5 bps`, dropped)* | *15.8%* | *0.935* | *-23.2%* | *0.617* | *-3.7%* | *+44.6%* | *-7.7%* | *+0.28* | *5.8x* |
| *Ref: Base + RSI2 + 5-Asset Macro (`15/35 bps`)* | *12.3%* | **0.688** (`0.67` to `10/30`) | *-23.4%* | *0.690* | *-2.3%* | *+43.0%* | *-3.6%* | *+0.21* | *11.8x* |

**Key Structural Takeaways (Verifying Item 7):**
- **Contango Gate (`VIX < VIX3M`)** does the heavy lifting on Sharpe (`0.552 -> 0.850`).
- **The `eVRP <= 0` half-size rule** adds zero full-sample Sharpe (`0.850 -> 0.845`), but trims `MaxDD` (`-16.3% -> -14.4%`) and improves `2022–2026` `xs` (`0.393 -> 0.509`).
- **`VIXM` backwardation** raises standalone `MaxDD` from `-14.4%` to `-23.2%` (due to the violent `VIXM` reversal on March 18–24, 2020 after its `+60%` surge), but cuts `SPY` correlation from `+0.60` to `+0.26` by turning the Feb–Apr 2020 COVID window into a `+44.5%` gain (`+26.4%` through the March 23 equity trough).
- **One session of execution delay (`T+2 Open` instead of `T+1 Open`)** drops `xs` from `0.872` to `0.612` (`0.856` to `0.605` through `2026-10-30`) because missing the Feb 5, 2018 exit open turns `2018 Volmageddon` from `-2.6%` into `-14.6%`.

---

### 2.3 `VIX / VIX3M` Contango Threshold Sensitivity & Feb 2018 / Aug 2024 Knife-Edge (Item 6)
In our earlier draft, we claimed that the `eVRP` half-size rule saved the pod in February 2018. **That claim was wrong.** On Friday, February 2, 2018, `VIX` closed at `17.31` and `VIX3M` closed at `17.03` (`VIX / VIX3M = 1.0164`, backwardation by just `0.28` VIX points), while `eVRP` was still positive (`17.31 - 10.66 = +6.65`). Thus the pod held full size through Feb 2 and exited to `100%` cash at the Monday, Feb 5, 2018 open solely because `VIX / VIX3M` crossed `1.000`. Similarly, prior to the August 5, 2024 Yen-carry spike, on August 2, 2024 `VIX` closed at `23.39` vs `VIX3M` `23.34` (`ratio = 1.0021`, just `0.05` points apart).

Varying the `SVXY` contango exit threshold `VIX / VIX3M < c` across `c in [0.95, 0.98, 1.00, 1.01, 1.02, 1.05, 1.10]` (while keeping the `VIXM` backwardation trigger at `ratio > 1.00` and `ratio >= c`) reveals the exact cliff:

| Contango Gate Threshold (`VIX / VIX3M < c`) | CAGR | Excess Sharpe (`xs`) | MaxDD | **2018 Volmageddon (`Jan 26 – Feb 9`)** | **2024 Aug Yen Carry (`Jul 16 – Aug 15`)** | 2020 COVID (`Feb 19 – Apr 30`) | 2025 Tariff Shock (`Feb 19 – Apr 15`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`c = 0.95`** (Conservative) | 11.3% | 0.747 | -23.2% | **-2.6%** | **-2.7%** | +44.4% | **-2.8%** |
| **`c = 0.98`** (2% Contango Buffer) | **13.7%** | **0.869** | -23.2% | **-2.6%** | -8.2% | +44.5% | -5.9% |
| **`c = 1.00` (Standard Contango)** | **14.1%** | **0.872** | -23.2% | **-2.6%** (Exited Feb 5 Open) | -7.6% (Exited Aug 5 Open) | +44.5% | -5.6% |
| **`c = 1.01`** | 16.0% | 1.004 | -12.2% | **-2.6%** (Exited Feb 5 Open) | -7.6% (Stayed Long Aug 5!) | +57.5% | -10.9% |
| **`c = 1.02`** | **14.6%** | **0.881** | **-25.6%** | **-20.8%** (Stayed Long Feb 5!) | -7.6% (Stayed Long Aug 5!) | +57.5% | -11.3% |
| **`c = 1.05`** | 16.7% | 0.983 | -24.4% | **-20.8%** | -7.6% | +93.7% | -11.3% |
| **`c = 1.10`** | 18.1% | 1.060 | -24.6% | **-20.8%** | -7.6% | +93.7% | -14.8% |

*(Source: [`results/research/vrp_multipod_scout_20261101/evrp_vix_vix3m_threshold_grid_20261102.csv`](../../results/research/vrp_multipod_scout_20261101/evrp_vix_vix3m_threshold_grid_20261102.csv))*

- At `c = 1.02`, the pod remains long `SVXY` on Monday, February 5, 2018 and suffers **`-20.8%`** during Volmageddon (deepening `MaxDD` to `-25.6%`; note that `c >= 1.01` happens to look higher on full-sample `xs` only because delaying `VIXM` entry in March 2020 sidestepped an early whipsaw and boosted the single March 2020 `VIXM` spike from `+44.5%` to `+57.5%–93.7%`, while worsening 2018 Volmageddon to `-20.8%` and 2025 Tariff to `-11.3%.. -14.8%`).
- Conversely, tightening the contango exit gate to `c = 0.98` (requiring a `2%` contango buffer before selling volatility) removes the Feb 2018 / Aug 2024 knife-edge dependency while preserving `0.869` excess Sharpe (`13.7%` CAGR).

---

### 2.4 Pre-2011 & 2008 GFC Stress Test Using Constant-Maturity VX Futures (Item 14)
Because `SVXY` and `VIXM` launched in `2011`, the ETP backtest was T-bills before `2011-10-10` and never tested `2008`. To test `2008` without synthetic model assumptions, we load the actual Cboe constant-maturity VX futures series built in [`quantseeker_volatility_study.md`](knowledge-base/studies/quantseeker_volatility_study.md) (`vx_st_ret` = 30-day M1/M2 constant maturity, `vx_mt_ret` = 150-day M4–M7 constant maturity) from `2006-07-18` (when `$VIX3M` history begins in Norgate) through `2026-10-01`, applying the exact same frozen `EVRP` signal (`-0.5x * vx_st_ret` in contango, `+1.0x * vx_mt_ret` in backwardation + `eVRP <= 0`, `95/85 bps/yr` ETP expense drag, `10 bps` trade cost):

| Window (`VX Constant-Maturity Futures`, `10 bps` cost) | Base Short-Vol (`SVXY_synth + VIXM_synth`) Return / CAGR | Excess Sharpe (`xs`) | MaxDD | `SPY` Return / CAGR |
|---|---:|---:|---:|---:|
| **Full 20-Year Sample (`2006-07-18` to `2026-10-01`)** | **15.0% CAGR** (`+1591.8%` total) | **0.927** | **-21.9%** | `+792.2%` total |
| **Pre-ETP / GFC Era (`2006-07-18` to `2011-10-07`)** | **23.8% CAGR** (`+205.6%` total) | **1.517** | **-15.0%** | `+4.1%` total |
| **2008 Calendar Year (`2008-01-01` to `2008-12-31`)** | **+53.9%** (`53.6%` CAGR) | **2.266** | **-8.6%** | **-36.8%** |
| **GFC Bear Market (`2007-10-09` to `2009-03-09`)** | **+36.1%** (`24.4%` CAGR) | **1.149** | **-15.0%** | **-54.8%** |
| **ETP Era (`2011-10-10` to `2026-10-01`, VX Proxy)** | **12.1% CAGR** (`+453.7%` total) | **0.737** | **-21.9%** | `+757.4%` total |

*(Source: [`results/research/vrp_multipod_scout_20261101/evrp_2004_2026_vx_futures_test_20261102.csv`](../../results/research/vrp_multipod_scout_20261101/evrp_2004_2026_vx_futures_test_20261102.csv))*

- During the **2008 GFC**, `$VIX > $VIX3M` backwardation kept the strategy out of short-vol and rotated into mid-curve VX futures (`vx_mt_ret`), generating **`+53.9%` in calendar year 2008** (`MaxDD -8.6%`) while `SPY` fell `-36.8%`.
- When spliced into the `2008–2026` book (`VX futures` for `2008-01` to `2011-10`, `SVXY/VIXM` ETPs from `2011-10` onward), a `+10%` allocation to Base Short-Vol in `REC` lifts `2008–2026` Excess Sharpe from `1.258` to **`1.326`**, cuts `MaxDD` from `-15.6%` to **`-12.5%`**, and achieves bootstrap win shares of **`0.99 (all) / 0.88 (2011+) / 0.30 (2023+)`**.

---

## 3. Sleeve 2: Macro Trend Collateral & Selection-Bias Audit (Items 3, 4, 5, 10)

### 3.1 Separating the Two Sleeves & Comparing `Gold-Only`, `5-Asset`, and `Broad 13-Asset` (Items 4, 10)
The Base Short-Vol pod holds **`47.9%` average idle cash** (`2011–2026`). Instead of fusing short-vol and macro trend into a single opaque pod, we evaluate them both as **two separate sleeves** and as a **cash-collateral overlay** (uniform `6%` vol target per asset, `35%` per-asset cap, Calendar Friday rebalance `rebal_weekday_int = 4` per [`SPEC_FROZEN.md`](../../scripts/research/vrp_multipod_scout_20261101/SPEC_FROZEN.md)), comparing:
1. **`Gold Trend Only ({GLD})`**: Single macro reserve asset.
2. **`5-Asset Stagflation Basket ({GLD, DBC, UUP, TBT, BTAL})`**: Explicitly disclosed as **post-hoc / 2022-motivated**.
3. **`5-Asset Ex-Gold ({DBC, UUP, TBT, BTAL})`**: Isolates what the non-gold legs do on their own.
4. **`Broad 13-Asset Non-Equity Pool ({GLD, SLV, DBC, USO, UUP, UDN, FXE, FXY, TBT, TLT, IEF, TIP, BTAL})`**: Unfiltered non-equity trend pool.

| Macro Universe (`2011-10-10` to `2026-08-19`, Calendar Fri Rebal) | Standalone `xs` (Full) | Standalone `xs` (**Ex-2022**) | Standalone 2022 Return | Overlay on Base Short-Vol (`5 bps` `xs`) | Overlay (`15/35 bps` `xs`) | Overlay `xs` (`COVID Blanked`) | Overlay `xs` (`2022 Blanked`) | Overlay 2022 Return | 1-Way Ann. Turnover | Corr(Unified, Plain Sum) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0. No Macro (Base Short-Vol Only)** | — | — | — | **0.861** (`0.872` w/ 15d warmup) | **0.692** (`0.703`) | 0.840 | 0.885 | -7.6% | **4.1x** | 1.000 |
| **1. `Gold Trend Only ({GLD})`** | **+0.369** | **+0.391** | -0.4% | **0.920** (`0.948` Tue) | **0.738** | **0.908** | **0.941** | -7.0% | **5.4x** | **0.999** |
| **2. `5-Asset Stagflation ({GLD,DBC,UUP,TBT,BTAL})`** | +0.300 | +0.142 | **+22.4%** | **0.918** (`0.952` Tue) | 0.662 | 0.902 | 0.896 | **+4.0%** | **11.3x** | **0.993** |
| **3. `5-Asset Ex-Gold ({DBC,UUP,TBT,BTAL})`** | +0.167 | **-0.024** | **+22.5%** | 0.854 | 0.615 | 0.833 | 0.825 | **+5.6%** | 9.7x | 0.991 |
| **4. `Broad 13-Asset Non-Equity Pool`** | +0.204 | +0.051 | +24.4% | 0.822 | 0.498 | 0.773 | 0.792 | +5.4% | 16.8x | **0.985** |

*(Source: [`results/research/vrp_multipod_scout_20261101/evrp_macro_sleeves_comparison_20261102.csv`](../../results/research/vrp_multipod_scout_20261101/evrp_macro_sleeves_comparison_20261102.csv))*

### 3.2 150 Random 5-of-13 Non-Equity Baskets Audit (Item 3)
Drawing `150` random 5-asset subsets from the 13 non-equity ETFs (`GLD, SLV, DBC, USO, UUP, UDN, FXE, FXY, TBT, TLT, IEF, TIP, BTAL`) on top of Base Short-Vol (`5 bps`):
- **No-Macro Baseline `xs`:** `0.861` (`2011-10-10` to `2026-10-30`).
- **Median Random 5-of-13 Basket `xs`:** `0.858` (below No-Macro; under the `20%` per-asset cap without `GLD`, the median falls to `0.837`, and under equal unfiltered sizing only `19%`–`48%` beat No-Macro).
- **Random Baskets Containing `GLD` vs Excluding `GLD`:** Median `xs = 0.890` with `GLD` vs `0.837` without `GLD`.
- **Our Chosen 5-Asset Basket (`{GLD, DBC, UUP, TBT, BTAL}`):** Sits at the **`87.3rd–98.0th` percentile** (`xs = 0.918` on Base Short-Vol; `98th` percentile under the `run_varadi_compass_and_evrp_ablation_20261101.py` settings).
- **Conclusion on Items 3, 4 & 10:**
  1. **Gold (`GLD`) does almost all the structural work outside 2022:** Excluding 2022, `Gold Trend Only` has standalone `xs = +0.391`, whereas `5-Asset Ex-Gold ({DBC, UUP, TBT, BTAL})` has standalone `xs = -0.024` (`+0.142` with `GLD` included, `+0.051` for all 13 assets). The four non-gold legs (`DBC, UUP, TBT, BTAL`) are purely **2022 rate/inflation shock insurance** (`+4.0%` in 2022 vs `-7.0%` for Gold-only), and they more than double annual one-way turnover (`11.3x` vs `5.4x` for Gold-only and `4.1x` for Base Short-Vol).
  2. **1-Pod vs 2-Pod:** The unified cash-collateral pod has **`0.993–0.999` daily return correlation** with a plain sum of Base Short-Vol excess return and Macro Trend excess return. Merging them adds no structural magic; keeping **Base Short-Vol** and **Gold Trend** as separate, independently monitored rules is cleaner and more transparent.

### 3.3 Fixing the `t % 5` Index-Offset Artifact (Item 5)
In `run_varadi_compass_and_evrp_ablation_20261101.py`, the baseline series was built on `master_index` (`2008-01-02` start) while the 27-cell grid was built on `svxy_idx` (`2011-10-04` start). Because `len(master_index < 2011-10-04) = 946` (`946 % 5 = 1`), `t % 5 == 0` landed on different trading days in the baseline (`0.983`) vs the `(50, 126, 6%)` grid cell (`1.016`), and the labels `"Monday" .. "Friday"` were `t % 5` array offsets rather than actual calendar weekdays.
- In [`run_clean_vrp_rebuild_20261102.py`](../../scripts/research/vrp_multipod_scout_20261101/run_clean_vrp_rebuild_20261102.py), rebalancing is anchored to **true calendar weekdays (`index.dayofweek == d`)**, verified by unit test `test_macro_trend_calendar_weekday_index_invariance` in [`tests/test_vrp_multipod_scout_20261102.py`](../../tests/test_vrp_multipod_scout_20261102.py) to produce **identical** weights (`max diff = 0.0`) regardless of whether the DataFrame starts in `2008` or `2011`.
- Across true calendar weekdays (`Mon..Fri`), `Gold Trend Only` is rock-solid (`xs = [0.935, 0.948, 0.936, 0.928, 0.938]`, spread `0.020`), whereas the 5-asset basket has a `0.089` weekday spread (`[0.950, 0.952, 0.864, 0.941, 0.948]`).

---

## 4. Orthogonality & Ex-2020 Correlation Audit (Items 12, 13)

### 4.1 Exact Pod Definitions (Fixing Item 13)
- **TAA 1N (`taa3x_1n`):** 5-Asset 1/N Macro Trend (`SPY, QQQ, IWM, EFA, EEM` routed to `IEF/TLT/GLD/Cash`).
- **MR Capsule (`mr`):** `50% DV2 + 50% HPI` on S&P 500 stocks (not `IBS + RSI(2)`).
- **CORE5 (`core5`):** Equal-weight defensive diversifier (`dbmf, gld, btal_qqq, downshock, eom_flow`).
- **Three-Leg Momentum (`mom3`):** Equal mix (`33.3% / 33.3% / 33.3%`) of Leg A (`Residual Mom`), Leg B (` Frog-in-the-Pan Raw Mom`), and Leg C (`52-Week High`), averaged `50/50` across `k=0` and `k=10` (or `40/30/30` in `Mom 40/30/30`).

### 4.2 Full-Sample vs Ex-2020 Correlations & Spanning Alphas (Item 12)
Headline low correlations (`+0.19` to `+0.28`) and statistically significant spanning alphas lean heavily on **March 2020** (when `VIXM` surged `+26.4%` while equities crashed). Dropping calendar year `2020` entirely reveals the normal-times co-movement:

| Sleeve Variant (`2011-10-10` to `2026-08-19`) | Window | Corr vs `TAA 1N` | Corr vs `MR (DV2+HPI)` | Corr vs `CORE5` | Corr vs `Mom 40/30/30` | Corr vs `SPY` | Spanning Alpha vs `REC` (`TAA + MR + CORE5`) | NW `t`-stat |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Base Short-Vol (No Macro)** | Full (`2011–2026`) | +0.28 | +0.11 | +0.22 | +0.29 | +0.27 | **+7.02%/yr** | +1.49 |
| **Base Short-Vol (No Macro)** | **Ex-2020 (Drop 2020)** | **+0.47** | **+0.39** | **+0.27** | **+0.51** | **+0.64** | **+0.81%/yr** | **+0.33** |
| **Base Short-Vol + Gold Only** | Full (`2011–2026`) | +0.30 | +0.12 | +0.35 | +0.29 | +0.26 | **+6.82%/yr** | +1.49 |
| **Base Short-Vol + Gold Only** | **Ex-2020 (Drop 2020)** | **+0.49** | **+0.37** | **+0.42** | **+0.50** | **+0.62** | **+0.72%/yr** | **+0.29** |
| **Base Short-Vol + 5-Asset Macro** | Full (`2011–2026`) | +0.27 | +0.08 | +0.40 | +0.25 | +0.19 | **+7.37%/yr** | +1.62 |
| **Base Short-Vol + 5-Asset Macro** | **Ex-2020 (Drop 2020)** | **+0.43** | **+0.30** | **+0.48** | **+0.43** | **+0.49** | **+2.04%/yr** | **+0.78** |

*(Source: [`results/research/vrp_multipod_scout_20261101/evrp_orthogonality_ex2020_20261102.csv`](../../results/research/vrp_multipod_scout_20261101/evrp_orthogonality_ex2020_20261102.csv))*

- Outside of `2020`, `SVXY` contango harvesting is positively correlated (`+0.43` to `+0.49` with `TAA`, `CORE5`, and `Momentum`, and `+0.49` to `+0.64` with `SPY`), and the spanning alpha over the `REC` sleeves shrinks from `+6.8%–7.4%` (`t ~ 1.5–1.6`) to `+0.7%–2.0%` (`t ~ 0.3–0.8`).

---

## 5. Book Integration Against Official `REC` (`50/30/20`) and `Growth` (`40/30/30`) (Items 1, 9, 10, 11)

Instead of benchmarking against the rejected 3-pod momentum book, we evaluate all variants against the **official Recommended Book (`REC`: `50% TAA 1N / 30% MR Capsule / 20% CORE5`)** and **Growth Book (`40% TAA 1N / 30% MR Capsule / 30% CORE5`)**, reporting paired stationary block-bootstrap win shares across **`all (2008+) / 2011+ / 2023+`**:

| Portfolio Configuration (`2008-01-02` to `2026-08-19`) | 2008–26 CAGR | 2008–26 `xs` | 2008–26 MaxDD | 2011–26 CAGR | 2011–26 `xs` | 2011–26 MaxDD | Bootstrap Share **`all / 2011+ / 2023+`** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline `REC` (`50% TAA / 30% MR / 20% CORE5`)** | **20.9%** | **1.258** | **-15.6%** | **22.3%** | **1.372** | **-14.2%** | — (Reference) |
| **`REC` + 10% Short Vol Only (`5 bps`, T-bills pre-2011)** | 20.0% | 1.294 | -14.1% | 21.5% | 1.409 | -12.7% | **0.91 / 0.87 / 0.30** |
| **`REC` + 10% Short Vol Only (`+5 bps` = `10 bps`)** | 19.9% | 1.292 | -14.1% | 21.5% | 1.406 | -12.7% | **0.88 / 0.84 / 0.28** |
| **`REC` + 10% Short Vol Only (`15/35 bps` Vol ETP cost)** | 19.8% | 1.280 | -14.1% | 21.3% | 1.391 | -12.7% | 0.74 / 0.69 / 0.15 |
| **`REC` + 10% Short Vol Only (`COVID Blanked`)** | 19.7% | 1.268 | -14.1% | 21.2% | 1.375 | -12.7% | 0.81 / 0.72 / 0.30 |
| **`REC` + 10% Short Vol Only (`With 2008 VX Futures`, 10 bps)** | **20.7%** | **1.326** | **-12.5%** | 21.5% | 1.409 | -12.5% | **0.99 / 0.88 / 0.30** |
| **`REC` + 10% Short Vol + Gold Only (`5 bps`)** | **20.1%** | **1.297** | **-14.1%** | **21.7%** | **1.415** | **-12.7%** | **0.94 / 0.89 / 0.72** |
| **`REC` + 10% Short Vol + Gold Only (`+5 bps` = `10 bps`)** | **20.0%** | **1.294** | **-14.1%** | **21.6%** | **1.411** | **-12.7%** | **0.91 / 0.85 / 0.68** |
| **`REC` + 10% Short Vol + Gold Only (`15/35 bps`)** | 19.8% | 1.281 | -14.1% | 21.3% | 1.394 | -12.7% | 0.75 / 0.70 / 0.49 |
| **`REC` + 10% Short Vol + Gold Only (`COVID Blanked`)** | 19.8% | 1.272 | -14.1% | 21.3% | 1.383 | -12.7% | **0.86 / 0.79 / 0.72** |
| **`REC` + 10% Short Vol + 5 Assets (`5 bps`)** | **20.1%** | **1.300** | **-14.1%** | **21.7%** | **1.417** | **-12.7%** | **0.97 / 0.93 / 0.76** |
| **`REC` + 10% Short Vol + 5 Assets (`+5 bps` = `10 bps`)** | **20.0%** | **1.294** | **-14.1%** | **21.6%** | **1.409** | **-12.7%** | **0.93 / 0.88 / 0.66** |
| **`REC` + 10% Short Vol + 5 Assets (`15/35 bps`)** | 19.7% | 1.277 | -14.1% | 21.2% | 1.386 | -12.7% | 0.74 / 0.69 / 0.41 |
| **`REC` + 10% Short Vol + 5 Assets (`COVID Blanked`)** | 19.9% | 1.277 | -14.1% | 21.4% | 1.386 | -12.7% | **0.93 / 0.86 / 0.76** |
| **`REC` + 10% Short Vol + Broad 13 Assets (`5 bps`)** | 20.0% | 1.293 | -14.1% | 21.5% | 1.406 | -12.7% | 0.90 / 0.84 / 0.37 |
| **`REC` + 2 Separate Sleeves (`7% Short Vol + 3% Gold Trend`)** | 19.7% | 1.286 | -14.1% | 21.1% | 1.402 | -12.7% | **0.95 / 0.89 / 0.47** |
| **`REC` + 2 Separate Sleeves (`7% Short Vol + 3% 5-Asset Macro`)** | 19.8% | 1.287 | -14.1% | 21.2% | 1.401 | -12.7% | **0.96 / 0.93 / 0.50** |
| **5-Pod (`TAA 50 / MR 25 / Mom 15 / EOM_Tradable 10`, No VRP)** | 22.3% | 1.273 | -16.5% | 22.5% | 1.314 | -15.3% | 0.69 / 0.26 / 0.37 |
| **5-Pod (`TAA 40 / MR 25 / Mom 15 / SV+Gold 10 / EOM_Tradable 10`)** | 20.8% | 1.331 | -12.0% | 21.2% | 1.375 | -12.0% | 0.97 / 0.86 / 0.40 |
| **Baseline `Growth` (`40% TAA / 30% MR / 30% CORE5`)** | **18.7%** | **1.283** | **-13.1%** | **20.0%** | **1.410** | **-12.7%** | — (Reference) |
| **`Growth` + 10% Short Vol Only (`5 bps`)** | 18.1% | 1.322 | -11.8% | 19.4% | 1.448 | -11.4% | **0.90 / 0.84 / 0.32** |
| **`Growth` + 10% Short Vol + Gold Only (`5 bps`)** | 18.2% | 1.325 | -11.8% | 19.6% | 1.454 | -11.4% | **0.92 / 0.85 / 0.71** |
| **`Growth` + 10% Short Vol + Gold Only (`10 bps`)** | 18.1% | 1.322 | -11.8% | 19.5% | 1.450 | -11.4% | **0.88 / 0.82 / 0.68** |
| **`Growth` + 10% Short Vol + Gold Only (`COVID Blanked`)** | 17.9% | 1.297 | -11.8% | 19.2% | 1.419 | -11.4% | 0.83 / 0.71 / 0.71 |
| **`Growth` + 10% Short Vol + 5 Assets (`10 bps`)** | 18.1% | 1.321 | -11.8% | 19.5% | 1.448 | -11.4% | **0.91 / 0.84 / 0.66** |
| **`Growth` + 10% Short Vol + 5 Assets (`COVID Blanked`)** | 17.9% | 1.301 | -11.8% | 19.3% | 1.422 | -11.4% | **0.90 / 0.81 / 0.74** |

*(Source: [`results/research/vrp_multipod_scout_20261101/evrp_book_integration_rec_growth_20261102.csv`](../../results/research/vrp_multipod_scout_20261101/evrp_book_integration_rec_growth_20261102.csv))*

---

## 6. Capacity & Tax / K-1 / IRC §1446(f) Operational Audit

### 6.1 Trailing 60-Session Dollar Volume Capacity Audit
Using Norgate unadjusted price and volume through `2026-10-30`, the trailing 60-session median daily dollar volume (`Close * Volume`) and `1% ADV` single-order limit are:

| Ticker | Asset Description | 60-Session Median Daily `$` Volume | `1% ADV` Max Single Order | Max Pod Capital at Max Target Weight | Tax Form | Non-US / Israel Withholding & PTP Status |
|---|---|---:|---:|---:|---|---|
| **`VIXM`** | ProShares VIX Mid-Term Futures ETF | **`$3.2M / day`** | **`$32K`** | **`$32K`–`$64K` pod** (at `50%–100%` weight) | **Schedule K-1** | **Commodity Pool (PTP).** `0%` dividend; **IRC §1446(f) 10% gross sale withholding** is currently **EXEMPT** via ProShares' quarterly 92-day Qualified Notice. |
| **`BTAL`** | AGFiQ US Market Neutral Anti-Beta | **`$6.5M / day`** | **`$65K`** | `$185K` pod (at `35%` cap) | Form 1099-DIV | Standard 1940-Act ETF (`25%` div withholding; no K-1, no 1446(f)). |
| **`TBT`** | ProShares UltraShort 20+ Year Treasury | **`$10.4M / day`** | **`$104K`** | `$297K` pod (at `35%` cap) | Form 1099-DIV | Standard 1940-Act ETF (no K-1, no 1446(f)). |
| **`DBC`** | Invesco DB Commodity Index Tracking | **`$21.3M / day`** | **`$213K`** | `$608K` pod (at `35%` cap) | **Schedule K-1** | **Commodity Pool (PTP).** Exempt via Qualified Notice; **can be replaced 1-for-1 by `PDBC`** (`$86.8M/day`, 1940-Act ETF, no K-1, no 1446(f)). |
| **`UUP`** | Invesco DB US Dollar Index Bullish | **`$41.1M / day`** | **`$411K`** | `$1.17M` pod (at `35%` cap) | **Schedule K-1** | **Commodity Pool (PTP).** Exempt via Qualified Notice; or substitute `USDU` (1940-Act ETF, no K-1). |
| **`SVXY`** | ProShares Short VIX Short-Term Futures | **`$72.7M / day`** | **`$727K`** | `$1.45M` pod (at `50%` weight) | **Schedule K-1** | **Commodity Pool (PTP).** `0%` dividend; **IRC §1446(f)** currently **EXEMPT** via ProShares' quarterly 92-day Qualified Notice. |
| **`GLD`** | SPDR Gold Shares | **`$3,467.0M / day`** | **`$34.67M`** | `>$34M` pod | 1099 Grantor Trust | **Not a PTP.** `0%` dividend, no K-1, no 1446(f) withholding. |

*(Source: [`results/research/vrp_multipod_scout_20261101/evrp_capacity_and_tax_audit_20261102.csv`](../../results/research/vrp_multipod_scout_20261101/evrp_capacity_and_tax_audit_20261102.csv))*

> [!WARNING]
> **Capacity & PTP / K-1 Operational Findings:**
> 1. **`VIXM` Liquidity Bottleneck (`$3.2M/day` ADV):** At `1%` of median daily dollar volume (`$32K` per order), a `10%` pod allocation in a total portfolio above **`$320K–$640K`** (or `$1.6M` at `5%` of ADV) will experience elevated market impact in `VIXM` (`15–35 bps` per side, which reduces standalone `xs` from `0.872` to `0.703`). Above `~$35K–$50K` in `VIXM`, execution must either step into `VIXM` over multiple TWAP slices or use direct Cboe VX months 4–7 futures (`quantseeker_volatility_study`).
> 2. **Schedule K-1 & IRC §1446(f) PTP Risk:** Both `SVXY` and `VIXM` are series of **ProShares Trust II**, a Delaware statutory trust taxed as a partnership (Commodity Pool / Publicly Traded Partnership). For US taxable accounts, they issue **Schedule K-1** (IRC §1256 `60/40` mark-to-market futures treatment). For non-US / Israeli accounts at Interactive Brokers, PTP sales are subject to **IRC §1446(f) `10%` gross-proceeds withholding** *unless* the issuer publishes a quarterly **92-day Qualified Notice** certifying that the fund is not engaged in a US trade or business (`ECI = 0%`). While ProShares Trust II currently publishes valid Qualified Notices (so IBKR applies `0%` withholding today), any administrative lapse or broker rule change would trigger a devastating `10%` tax on gross sales.

---

## 7. David Varadi Inflation Compass Audit (Fixing Item 2)

Our audit of David Varadi's *Inflation Compass* (`compass_qqq` and de-peaked T+1 Open variants, saved to [`results/research/vrp_multipod_scout_20261101/varadi_vs_evrp_comparison_20261101.csv`](../../results/research/vrp_multipod_scout_20261101/varadi_vs_evrp_comparison_20261101.csv)) confirms the prior [`inflation_compass_deep_20260928`](../research/) conclusion:
- **Reconciliation of the `0.984` vs `1.258` `REC` baseline discrepancy (Item 2):** In our earlier console printout, `varadi_vs_evrp_comparison` accidentally passed `15 bps ETF / 35 bps Vol ETP` into the `simulate_t1_open_weights` call and compared against an unweighted slice, whereas `REC` (`50% TAA / 30% MR / 20% CORE5`) has **`20.9% CAGR / 1.258 xs / -15.6% MaxDD`** (`2008–2026`). Both numbers are now reconciled in [`varadi_vs_evrp_comparison_20261101.csv`](../../results/research/vrp_multipod_scout_20261101/varadi_vs_evrp_comparison_20261101.csv).
- **Compass in the `20% CORE5` Slot of `REC`:** Replacing `CORE5` (`20%`) with `Varadi Compass QQQ` raises `2008–2026` CAGR from `20.9%` to **`23.2%`**, leaves Excess Sharpe virtually unchanged (`1.257` vs `1.258`, bootstrap win share **`0.50`**), and deepens `MaxDD` from `-15.6%` to **`-18.3%`** because `Compass` holds `100% QQQ` in Goldilocks (`+0.55` correlation with `TAA 1N`).
- **De-peaking the 60-day `T5YIE` lookback:** Replacing Varadi's fragile 60-day lookback (`98.7th` percentile of its parameter neighborhood) with a smooth ensemble (`21, 42, 63, 84, 126` days, `T+1` lagged `T5YIE`) drops standalone `2011–2026` `xs` from `1.015` to **`0.720`** (`+4.0%` in `2022` instead of `+31.7%`).

---

## 8. Summary of All 14 Peer-Review Corrections & Adoption Rule

| # | Peer-Review Finding | Verification & Resolution in This Rebuild |
|---|---|---|
| **1** | **Costs (`5 bps` vs `15/35 bps`) & Turnover** | Verified. Earlier summary mislabeled a `5 bps` run as `15/35 bps`. At true `15/35 bps`, `Base+RSI2+5Macro` is `0.688` (`0.67` to `10/30`), `Base Short-Vol` is `0.703`, and `+10%` book shares drop to `0.74 / 0.69 / 0.15` (Short-Vol) or `0.75 / 0.70 / 0.49` (Short-Vol + Gold). Macro raises one-way annual turnover from `4.1x` to `5.4x` (Gold-only) or `11.3x` (5-Asset). |
| **2** | **Missing `varadi_vs_evrp_comparison_20261101.csv` & Section 7** | Saved to [`results/research/vrp_multipod_scout_20261101/varadi_vs_evrp_comparison_20261101.csv`](../../results/research/vrp_multipod_scout_20261101/varadi_vs_evrp_comparison_20261101.csv), reconciled with `REC = 1.258`, and documented in Section 7. |
| **3** | **5-Asset Basket Selection Bias (98th percentile)** | Verified across 150 random 5-of-13 non-equity baskets ([`evrp_150_random_baskets_20261102.csv`](../../results/research/vrp_multipod_scout_20261101/evrp_150_random_baskets_20261102.csv)). The median random basket without `GLD` lowers the pod to `0.837` (below `0.861` No-Macro). |
| **4** | **Gold Does Most of the Work** | Verified. Outside 2022, standalone `Gold Trend` is `+0.391 xs` (`+0.32` through `10/30`), whereas `5-Asset Ex-Gold` is `-0.024 xs`. The non-gold legs only pay off in `2022` (`+4.0%` vs `-7.0%`). |
| **5** | **`t % 5` Index-Offset Artifact in 27-Cell Grid** | Verified (`946 % 5 = 1` offset between `master_index` and `svxy_idx`). Fixed to use true calendar weekdays (`index.dayofweek == d`) and verified by unit test. |
| **6** | **Feb 2018 Exit Was `VIX/VIX3M = 1.0164`, Not Half-Size** | Verified (`17.31 vs 17.03`). Added 7-point `VIX/VIX3M` threshold grid showing `-20.8%` in 2018 Volmageddon (`MaxDD -25.6%`) at `c >= 1.02` vs `-2.6%` at `c <= 1.01` (`xs = 0.869` at `c = 0.98`). |
| **7** | **What Drives Short Vol (`Gate`, `eVRP`, `VIXM`, `T+2` delay)** | Verified in Section 2.2: Gate gives `0.850`, `eVRP` trims DD (`-16.3% -> -14.4%`), `VIXM` cuts SPY corr (`+0.60 -> +0.26`) at `-23.2%` DD, and `T+2` delay drops `xs` to `0.612` (`0.605`). |
| **8** | **Cite Aziz/Zarattini (SSRN 5316487) & Bridge from `0.63`** | Cited in Section 1.1 and bridged step-by-step in Section 1.2 (`1x -> 2x` sizing `+0.173 xs`, `VIXM` vs `VIXY` cuts MaxDD `-32.8% -> -23.2%`, `5 bps` vs `35 bps` `+0.169 xs`). |
| **9** | **Benchmark Against `REC` and `Growth`, Not 3-Pod** | All book tables in Section 5 now benchmark against `REC (50/30/20)` and `Growth (40/30/30)` with `all / 2011+ / 2023+` shares. |
| **10** | **1 Pod vs 2 Separate Sleeves (`0.99` correlation)** | Verified (`corr = 0.993–0.999`). Evaluated as two separate sleeves (`7% Short Vol + 3% Gold Trend`). |
| **11** | **5-Pod Result Used Non-Tradable Lab `eom_flow`** | Replaced with tradable `Month-End Flow Vote` (`eom_v2`, plan `xs = 0.95`); without VRP, the 5-pod book share vs `REC` is `0.69 / 0.26 / 0.37` (or `0.50 / 0.38 / 0.28` with 10% VRP at 4-pod weights). |
| **12** | **Orthogonality Leans on 2020** | Verified in Section 4.2: Ex-2020 correlations rise to `0.43–0.49` with `TAA/CORE5/Mom` (`0.49–0.64` with `SPY`) and spanning alpha drops to `+0.7%–2.0%` (`t = 0.29–0.78`). |
| **13** | **Pod Descriptions (`MR = DV2 + HPI`, `Mom = 1/3 each`)** | Corrected in Section 4.1. |
| **14** | **2008 Test, Repo Scripts & Scout Registration** | Added `2006–2026` Cboe VX constant-maturity futures test (`+53.9%` in `2008`), moved all scripts to `scripts/research/vrp_multipod_scout_20261101/`, and registered row `588` in `scout_ledger.jsonl`. |

### Suggested Adoption & Shadow Rule (For Owner Decision)
Per the peer review's suggested adoption gate (**`2011+` share `> 0.80` under both `+5 bps` (`10 bps` total) and `COVID blanked`, followed by 12 months of shadow monitoring without a break**):
1. **Expected Portfolio Trade-off at `+10%` Pro-Rata in `REC` (`45% TAA / 27% MR / 18% CORE5 / 10% Sleeve`):**
   - **CAGR:** `~ -0.7` to `-0.8` pts (`20.9% -> 20.1%`).
   - **Excess Sharpe (`xs`):** `+0.04` to `+0.045` (`1.258 -> 1.297–1.300`, or `1.326` with `2008` VX futures).
   - **MaxDD:** `1.5` pts shallower (`-15.6% -> -14.1%`, or `-12.5%` with `2008` VX futures).
2. **Gate Check (`2011+` Bootstrap Share vs `REC`):**
   - **`Short Vol Only`:** `0.84` at `+5 bps` (**Pass**), `0.72` with `COVID blanked` (**Fail** — below `0.80`; `2023+` share is `0.30`).
   - **`Short Vol + Gold Trend Only`:** `0.85` at `+5 bps` (**Pass**), `0.79` with `COVID blanked` (**Borderline `0.79` vs `0.80`**; `2023+` share is `0.72`, zero basket-selection bias, no K-1 on `GLD`).
   - **`Short Vol + 5-Asset Macro`:** `0.88` at `+5 bps` (**Pass**), `0.86` with `COVID blanked` (**Pass**, `2023+` share `0.76`, but benefits from `2022` in-sample basket selection).
3. **Recommendation:** Do **not** deploy live capital now. If the owner wishes to pursue this family, place **`Base Short-Vol` (with tightened `VIX/VIX3M < 0.98` contango buffer)** and **`Gold Trend ({GLD})`** into **12-month shadow tracking** while verifying IBKR K-1 / IRC §1446(f) handling and `VIXM` slippage.
