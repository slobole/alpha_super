# CapacityAnalysis v2.1 User Guide

## TL;DR

`CapacityAnalysis` estimates how much capital a strategy can deploy before its
orders become too large relative to market liquidity and modeled execution
costs materially damage performance.

It does **not** change the strategy or the normal backtest. It reruns the
completed strategy at several AUM levels, measures every executed order against
lagged dollar ADV, applies an MOO- or MOC-specific square-root impact model, and
creates one self-contained HTML report.

The main number to use is **Recommended Max Capacity** from the recent
five-year window. Treat **Outer Capacity** as a stretch estimate and
**Break-even** as a theoretical boundary, not an operating target.

The result is a **pre-TCA model estimate**. It is suitable for research and an
initial institutional conversation, but it is not proof that live execution
will support the stated AUM.

---

## 1. The question this feature answers

The feature answers:

> If I run this exact strategy with more capital, when do its orders become too
> large relative to available liquidity, and how does that affect the equity
> curve and Sharpe ratio?

Capacity is presented as a band rather than one magical number:

| Result | Plain-English meaning |
|---|---|
| **Optimal Capacity** | The supported tested AUM with the highest Central Sharpe. |
| **Recommended Max** | The highest contiguous tested AUM that passes all normal deployment gates. This is the main operating estimate. |
| **Outer Capacity** | The highest contiguous tested AUM that still survives the Stress model and outer limits. This is a stretch estimate. |
| **Break-even bracket** | Adjacent tested AUM points where Central benchmark excess annual return crosses zero. |

Capacity is not necessarily monotonic in theory. Small accounts can suffer from
minimum commissions, while large accounts suffer from market impact. The
current classifications are intentionally **contiguous**: once an AUM point
fails, a later passing point does not restore the classification.

---

## 2. What stays unchanged

The normal strategy backtest remains the source of truth for:

- signals;
- point-in-time universe membership;
- sizing and leverage;
- order timing;
- fills at the strategy's declared execution point;
- IBKR Fixed commissions;
- the existing 2.5 bps per-side slippage assumption;
- the baseline equity curve.

CapacityAnalysis does not change live execution, released pods, portfolio
weights, order routing, or strategy logic.

The Capacity overlay only subtracts additional size-dependent cost above the
2.5 bps already present in the baseline:

```text
incremental capacity bps = max(0, modeled impact bps - 2.5 bps)
```

This prevents double counting. It also guarantees:

```text
Baseline equity >= Central equity >= Stress equity
```

The Capacity model can preserve the baseline result when modeled impact is
below 2.5 bps. It can never improve the baseline.

---

## 3. End-to-end flow

```text
  +----------------------------+
  | Strategy Capacity hook     |
  | MOO/MOC + MOO profile      |
  +-------------+--------------+
                |
                v
  +----------------------------+
  | Full strategy rerun at     |
  | every tested AUM           |
  +-------------+--------------+
                |
         +------+------+
         |             |
         v             v
  +-------------+  +----------------+
  | Full history|  | Recent five    |
  | window      |  | years          |
  +------+------+  +-------+--------+
         |                 |
         +--------+--------+
                  v
  +---------------------------------+
  | Aggregate actual order deltas   |
  | by date + asset + side          |
  +----------------+----------------+
                   |
                   v
  +---------------------------------+
  | Robust lagged dollar ADV        |
  | min(10d mean, 20d median)       |
  +----------------+----------------+
                   |
                   v
  +---------------------------------+
  | MOO/MOC square-root impact      |
  | Central + Stress                |
  +----------------+----------------+
                   |
                   v
  +---------------------------------+
  | Adjusted equity + liquidity     |
  | gates + capacity classifications|
  +----------------+----------------+
                   |
                   v
  +---------------------------------+
  | One HTML report + four data     |
  | artifacts                       |
  +---------------------------------+
```

Every AUM is a full strategy rerun. The analyzer does not simply multiply an
existing P&L curve.

---

## 4. How the analyzer knows MOO or MOC

The engine does not guess from the strategy name or ticker list. A supported
strategy exposes:

```python
def build_capacity_analysis_inputs(
    capital_base_float: float,
    backtest_start_date_str: str | None,
    end_date_str: str | None,
    ...,
) -> dict[str, object]:
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",  # or "MOC"
        "impact_profile_str": "MOO_ETF_PROXY",  # required only for MOO
    }
```

Rules:

- `execution_policy_str` must be exactly `MOO` or `MOC`.
- Every MOO strategy must declare one supported impact profile.
- MOC uses the shared MOC coefficients and requires no profile.
- Mixed MOO/MOC strategies are not supported in v2.1.
- The builder must accept AUM, start date, and end date so both windows can be
  rerun honestly.

If the hook is missing, Bench disables the Capacity button and displays:

```text
Capacity unavailable — missing capacity hook
```

If the hook exists but its policy or profile is invalid, the job fails. It is
not silently skipped.

---

## 5. MOO and MOC models

### 5.1 Shared liquidity calculation

For each asset and day:

```text
10d ADV = mean dollar volume over the previous 10 completed sessions
20d ADV = median dollar volume over the previous 20 completed sessions

robust ADV = min(10d ADV, 20d ADV)
```

Both rolling measures are shifted by one day. An order on day `T` only uses
volume through `T-1`.

Norgate `Turnover` is preferred when available. Otherwise:

```text
dollar volume = Close x Volume
```

The lower ADV estimate is used so that a few unusually high-volume days do not
artificially inflate capacity.

For each completed order:

```text
q = absolute order notional / robust lagged dollar ADV
```

The order is the actual position change, not the final target position.
Same-date, same-asset, same-side transactions inside the strategy are combined
before impact is calculated.

### 5.2 Square-root impact

```text
impact bps = lambda x sqrt(q / 0.01)
```

`lambda` means the modeled impact when one order equals 1% of ADV. It is not a
fixed cost charged to every order.

Example with `lambda = 40`:

```text
Order/ADV = 0.05%
Impact = 40 x sqrt(0.05% / 1.00%)
Impact = 8.94 bps
Incremental cost above baseline = 8.94 - 2.50 = 6.44 bps
```

### 5.3 MOC defaults

| Scenario | Lambda at 1% ADV |
|---|---:|
| Central | 8.2 bps |
| Stress | 17.8 bps |

MOC liquidity guardrails:

| Limit | Order/ADV |
|---|---:|
| Soft | 0.25% |
| Hard | 0.50% |

MOC is the higher-confidence branch because Norgate's US `Close` normally
represents the listing exchange's closing-auction price.

### 5.4 MOO profiles

| Profile | Central lambda | Stress lambda | Intended use | Confidence |
|---|---:|---:|---|---|
| `MOO_LARGE_MIXED` | 40.0 | 66.4 | DV2 and QPI common-stock strategies | Medium, pre-TCA |
| `MOO_NASDAQ_LARGE` | 66.4 | 114.0 | Nasdaq-100 strategies | Medium, pre-TCA |
| `MOO_ETF_PROXY` | 40.0 | 66.4 | TAA and sector-dispersion ETF strategies | Low, proxy |

MOO liquidity guardrails:

| Limit | Order/ADV |
|---|---:|
| Soft | 0.05% |
| Hard | 0.10% |

Every MOO order above 1% ADV is flagged as extrapolation:

- common stock: academic extrapolation;
- ETF: low-confidence proxy extrapolation.

Norgate `Open` is the first reported trade from a venue or ECN and is not
guaranteed to equal the listing exchange's official opening-auction print.
That makes MOO capacity less certain than MOC capacity.

The ETF profile uses common-stock coefficients. Its bias direction is unknown:
ETF creation/redemption may add liquidity, but a thin opening auction may also
be worse than the proxy.

---

## 6. Tested AUM grid

The default grid is:

```text
$50K
$100K
$250K
$500K
$1M
$2M
$5M
$10M
$25M
$50M
$100M
```

If the highest tested point passes, the report does not pretend it found an
exact ceiling. It displays:

```text
>= $100M
```

This means “at least $100M under the tested model,” not “exactly $100M.” The
grid is not automatically extended.

---

## 7. Why there are two time windows

Every AUM is analyzed over:

1. **Full history** — the entire completed strategy history.
2. **Recent five years** — five years ending at the exact full-history endpoint.

The recent five-year result is the headline because it better represents
current deployability and current liquidity. Full history answers a different
question:

> Could the historical track record have supported the same AUM using the
> liquidity that existed at the time?

If recent Recommended Max is higher than full-history Recommended Max, the
report displays a historical-feasibility warning. It does not silently replace
the recent number with the lower historical number.

If the full history is shorter than five years, the same completed runs are
reused instead of rerunning identical dates.

The runner fails loudly if a builder ignores the requested recent boundary. It
also requires at least 20 pricing observations before the recent performance
window so lagged ADV has causal warmup history.

These windows are diagnostics. They are not independent out-of-sample tests.

---

## 8. Exact classification rules

### 8.1 Optimal Capacity

Optimal Capacity is the supported grid point with the highest Central Sharpe.

“Supported” here means:

```text
complete liquidity data
P95 Order/ADV <= Soft limit
P99 Order/ADV <= Hard limit
```

If two supported points have the same Central Sharpe, the higher AUM wins.

Optimal Capacity is not automatically the recommended deployment level. A
point can have the best Central Sharpe but fail another Recommended rule.

### 8.2 Recommended Max Capacity

Every rule must pass:

```text
1. All orders have both required lagged ADV measures.
2. P95 Order/ADV <= Soft limit.
3. P99 Order/ADV <= Hard limit.
4. Central Sharpe erosion <= 20% versus Baseline.
5. Incremental Central cost <= 25% of benchmark excess annual return.
6. Rolling three-year Sharpe erosion <= 20% in every eligible window.
```

A rolling three-year window is eligible only when Baseline rolling Sharpe is at
least `0.30`. If no eligible rolling window exists, Recommended Max is `N/A`.

### 8.3 Outer Capacity

Every rule must pass under Stress:

```text
1. Liquidity coverage is complete.
2. Stress benchmark excess annual return remains positive.
3. Fewer than 5% of assessed orders breach the Hard limit.
4. Stress cost consumes less than 50% of benchmark excess annual return.
```

Outer Capacity is a research stretch boundary. Do not present it as the normal
operating recommendation.

### 8.4 Break-even bracket

Break-even uses Central **benchmark excess annual return**:

```text
strategy annual return - declared benchmark annual return
```

It is not beta-adjusted alpha.

The report gives the adjacent tested grid points surrounding the zero crossing:

```text
$5M to $10M
```

It does not interpolate a falsely precise number. Other possible outputs are:

```text
Above $100M
Below $50K
Not estimable
Not estimable from adjacent finite grid points
```

### 8.5 Contiguous classification

Recommended and Outer are evaluated from the lowest AUM upward:

```text
pass -> pass -> fail -> pass
```

The classified capacity stops at the second point. The final pass is diagnostic
only. This prevents a noisy later result from reopening capacity after an
earlier failure.

---

## 9. How to run it from Bench

Start Bench:

```powershell
uv run python -m alpha.bench
```

Open:

```text
http://127.0.0.1:8765
```

Then:

1. Open the strategy page.
2. Find the **Run** section.
3. Click **Capacity**.
4. Open **Jobs** to follow progress and logs.
5. When complete, return to the strategy page and open the Capacity report.

The Capacity button is enabled only when the strategy exposes a Capacity hook.

The **Standard** preset runs:

```text
Vanilla + Capacity + Timing
```

The **Full** preset runs:

```text
Vanilla + Capacity + Timing + Risk + Stress
```

For unsupported strategies, combined presets record Capacity as `SKIP` and
continue with supported analyses. Clicking Capacity alone is rejected.

Bench displays:

- `Capacity · v2.1` for current results;
- `Capacity · Legacy v1` for old or missing-version artifacts;
- recent and full-history dates on the run row.

Historical artifacts are kept readable and are never rewritten.

---

## 10. How to run it from the command line

Default AUM grid:

```powershell
uv run python strategies/run_capacity_analysis.py `
  strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi
```

Custom grid:

```powershell
uv run python strategies/run_capacity_analysis.py `
  strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi `
  --aum 100000 `
  --aum 500000 `
  --aum 1000000 `
  --aum 5000000
```

Resolve a strategy without running it:

```powershell
uv run python strategies/run_capacity_analysis.py `
  strategies/dv2/strategy_mr_dv2.py `
  --dry-run
```

Explicit full-history boundaries:

```powershell
uv run python strategies/run_capacity_analysis.py `
  strategies.dv2.strategy_mr_dv2 `
  --backtest-start-date 2008-01-01 `
  --end-date 2026-06-30
```

Do not save artifacts:

```powershell
uv run python strategies/run_capacity_analysis.py `
  strategies.dv2.strategy_mr_dv2 `
  --no-save
```

Combined research runner:

```powershell
uv run python scripts/research/run_strategy_analysis.py `
  strategies.dv2.strategy_mr_dv2 `
  --analysis capacity
```

The CLI prints:

```text
Strategy
Execution policy
Impact profile
Optimal capacity
Recommended capacity
Outer capacity
Break-even bracket
Report folder
```

---

## 11. Output folder and files

Results are saved under:

```text
results/research/strategy/<strategy>/capacity_analysis/<timestamp>/
```

Exactly five files are written:

| File | Purpose |
|---|---|
| `report.html` | Self-contained report intended for normal reading. |
| `capacity_curve.csv` | One summary row per AUM and time window. |
| `capacity_order_diagnostics.csv` | Order-level liquidity, impact, cost, breach, and extrapolation evidence. |
| `summary.json` | Headline recent result plus nested full/recent summaries and classification flags. |
| `metadata.json` | Model version, execution/profile declarations, actual window dates, and save metadata. |

Both CSV files contain `window_str`:

```text
full_history
recent_5y
```

Flat headline fields in `summary.json` are the recent-five-year results. The
full-history results are stored in `window_summary_dict`.

---

## 12. How to read the HTML report

### 12.1 Read this first

This section tells you:

- the actual recent window;
- the strategy's declared MOO or MOC policy;
- Recommended and Outer capacity;
- whether current capacity exceeds historical feasibility;
- whether later AUM points passed after an earlier failure.

Read this section before looking at any chart.

### 12.2 Four headline cards

- **Optimal Capacity** — best Central Sharpe inside the liquidity-supported area.
- **Recommended Max** — main current deployment estimate.
- **Outer Capacity** — stretch boundary.
- **Break-even bracket** — Central benchmark-excess-return zero crossing.

`>=` means the top tested grid point still passed. It is a censored lower
bound, not a measured ceiling.

### 12.3 Performance versus AUM

This chart shows:

```text
Baseline Sharpe
Central Sharpe
Stress Sharpe
```

Expected behavior is flat or declining adjusted performance as AUM grows.

If all lines overlap at low AUM, that can be correct: modeled impact is still
below the 2.5 bps already charged by the baseline. If they remain identical at
large AUM, inspect the Order/ADV chart and declared impact profile.

### 12.4 Normalized equity at Recommended Max

This chart compares Baseline, Central, and Stress equity, all starting at 1.0.

It is shown at Recommended Max. If no AUM qualifies, the lowest grid point is
shown and clearly labeled diagnostic only.

### 12.5 Liquidity usage versus AUM

This chart compares:

```text
P95 Order/ADV
P99 Order/ADV
Soft limit
Hard limit
```

P95 means 95% of assessed orders are no larger than that ratio. P99 focuses on
the tail and is the more sensitive indicator of isolated liquidity problems.

### 12.6 Share of orders beyond limits

This chart shows the percentage of assessed orders above Soft and Hard limits.
It tells you whether the problem is one isolated order or a broad execution
problem.

### 12.7 Current-window AUM results

Use this table to see exactly why one AUM passed and the next failed:

- Baseline, Central, and Stress Sharpe;
- P95 and P99 Order/ADV;
- Central cost as a share of benchmark excess return;
- Hard-breach percentage;
- Recommended pass status.

### 12.8 Full-history feasibility

This section repeats the capacity comparison over the full historical record.
It answers historical feasibility, not current deployability.

### 12.9 Assumptions and limitations

This section records the exact lambdas, confidence label, rolling-window count,
missing-liquidity share, extrapolation share, complete-fill assumption, and TCA
warning. Do not remove this section when sharing the report.

### 12.10 Largest liquidity bottlenecks

This table lists the worst asset/date/AUM combinations by Order/ADV. Use it to
identify:

- thin assets;
- unusual historical dates;
- positions that dominate the capacity result;
- candidates for position caps or different execution methods.

Do not change strategy rules merely to make this table look better without a
separate controlled research study.

---

## 13. Common interpretations

### “Performance versus AUM looks identical.”

This is usually normal at low AUM. The baseline already charges 2.5 bps, so no
additional cost is subtracted until modeled impact exceeds 2.5 bps.

Check:

1. the declared impact profile;
2. P95/P99 Order/ADV;
3. Central implicit cost in the bottleneck table;
4. whether the chart is showing genuinely different SVG lines on top of one
   another.

### “Recommended Capacity is N/A.”

Common causes:

- no eligible rolling three-year window with Baseline Sharpe at least 0.30;
- missing declared performance benchmark or benchmark data;
- missing lagged ADV for one or more orders;
- the lowest tested AUM already fails a Recommended rule;
- recent window did not satisfy the classification gates.

`N/A` does not mean infinite capacity. It means the analyzer cannot award a
defensible Recommended figure under the current evidence.

### “Optimal is above Recommended.”

This can happen. Optimal uses Central Sharpe inside the liquidity-supported
region. Recommended also requires benchmark-excess-return cost limits and the
rolling three-year gate.

Use Recommended for deployment discussions.

### “Outer is much higher than Recommended.”

That means the strategy survives the Stress model at higher AUM but no longer
passes the stricter normal-deployment rules. It is a stretch range, not free
capacity.

### “Current capacity is above historical capacity.”

Recent liquidity or strategy behavior supports more capital than earlier
history did. Use the current number as the headline but disclose the historical
warning. Do not claim the full track record was achievable at the current AUM.

### “The report says extrapolation.”

At least one MOO order exceeded 1% ADV. The math can still produce a curve, but
the result is outside the intended empirical support and must be treated as
diagnostic only.

---

## 14. Common failures and what to do

| Message or symptom | Meaning | Action |
|---|---|---|
| `Capacity unavailable — missing capacity hook` | Strategy has not been reviewed and enabled for Capacity. | Add a strategy-specific hook only after execution policy/profile review. |
| `requires impact_profile_str` | MOO strategy did not declare a supported profile. | Choose the profile explicitly; do not infer it from tickers in the engine. |
| `must accept backtest_start_date_str and end_date_str` | Builder cannot perform honest dual-window reruns. | Extend the builder while preserving its original warmup and semantics. |
| `did not honor the requested trailing-five-year start` | Builder ignored or mishandled the recent boundary. | Fix the builder; do not relabel full history as recent. |
| `must retain at least 20 pre-start observations` | Recent pricing data lacks causal ADV warmup. | Load history before the execution window. |
| Recommended is `N/A` | Required evidence or one of the gates is unavailable. | Read the AUM table, rolling-window count, benchmark declaration, and unavailable-order share. |
| Job shows `SKIP` inside Standard/Full | Strategy lacks a Capacity hook. | Expected behavior; other analyses continue. |

---

## 15. What to say to an institutional client

Recommended wording:

> The reported capacity is a pre-TCA, model-estimated capacity. The current
> headline uses a trailing five-year window ending at the latest completed
> backtest date. The method fully reruns the strategy across an AUM grid, uses
> robust lagged dollar ADV, applies auction-specific square-root impact above
> the existing 2.5 bps execution-cost floor, and reports current deployability
> separately from full-history feasibility. Recommended, Outer, and Break-even
> capacity are reported separately.

When quoting a number, include:

```text
Recommended Max Capacity: $X
Outer Capacity: $Y
Break-even bracket: $A to $B
Execution policy/profile: MOO/MOC + profile
Model version: capacity_v2_1
Window: recent five years ending YYYY-MM-DD
Status: pre-TCA model estimate
```

Do not say:

- “This is proven live capacity.”
- “Every order will fill completely at this cost.”
- “Outer Capacity is the amount we recommend deploying.”
- “The ETF coefficient is calibrated on ETFs.”
- “The five-year and full-history windows are independent out-of-sample tests.”
- “Break-even is a precise dollar estimate.”

---

## 16. Known limitations

The current model does not include:

- live TCA calibration;
- actual opening or closing auction volume;
- imbalance feeds;
- daily volatility scaling of lambda;
- partial fills;
- queue position or routing quality;
- book-level aggregation across pods or clients;
- mixed MOO/MOC strategies;
- automatic AUM-grid extension;
- beta-adjusted alpha;
- ETF-specific empirical lambda calibration.

Rolling three-year windows overlap heavily. Their count is a stability
diagnostic, not a count of independent samples.

The recent-window start validation assumes the 16 currently supported daily
strategy calendars. A future genuinely weekly or monthly equity calendar will
need a separately reviewed boundary rule.

---

## 17. Supported strategies in v2.1

At the time of this guide, Capacity is enabled for 16 strategies:

### Common-stock strategies

- `strategies.dv2.strategy_mr_dv2`
- `strategies.momentum.strategy_mo_atr_normalized_ndx`
- `strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled`
- `strategies.qpi.strategy_mr_qpi_ibs_rsi_exit`

### TAA and ETF strategies

- `strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash`
- `strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash`
- `strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash`
- `strategies.taa_df.strategy_taa_df_btal_fallback_spy`
- `strategies.taa_df.strategy_taa_df_dual_momentum_pivot5`
- `strategies.taa_df.strategy_taa_df_dual_momentum_pivot5_no_bndx`

### Sector-dispersion ETF strategies

- `strategies.mean_reversion.strategy_mr_sector_dispersion_ibs`
- `strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie`
- `strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi`
- `strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200`
- `strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc`
- `strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200`

The catalog is the live source of truth. If code changes later, Bench may show a
different supported set.

---

## 18. Practical reading checklist

Use this order every time:

1. Confirm the strategy and MOO/MOC declaration.
2. Confirm the impact profile and confidence label.
3. Read Recommended Max, not just Optimal or Outer.
4. Check whether the value is censored with `>=`.
5. Read the historical-feasibility warning.
6. Compare Baseline, Central, and Stress Sharpe.
7. Inspect P95/P99 Order/ADV against Soft/Hard limits.
8. Check hard-breach and extrapolation shares.
9. Read the normalized equity chart at Recommended Max.
10. Inspect the largest liquidity bottlenecks.
11. Confirm benchmark data and eligible rolling-window count.
12. Label the result pre-TCA in every external discussion.

---

## 19. Source documents

- [Transaction Costs Research](TRANSACTION_COSTS_RESEARCH.md)
- [Assumptions and Gaps](../../ASSUMPTIONS_AND_GAPS.md)
- [Goyal, Jegadeesh, and Wu — open-access JFQA paper](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/price-impact-in-closing-auctions-opening-auctions-and-continuous-markets-a-benchmark-for-cost-of-trading-on-anomalies/0F72910A79C5B42CF6E85F55164CE846)
- [Interactive Brokers US stock commissions](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php)
- [Norgate data content and price definitions](https://norgatedata.com/data-content-tables.php)

Implementation references:

- `alpha/engine/capacity_analysis.py`
- `strategies/run_capacity_analysis.py`
- `scripts/research/run_strategy_analysis.py`
- `alpha/bench/`

