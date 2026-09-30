# Portfolio Vanilla refresh — 2026-09-30

The raw run artifacts are local under `results/research/portfolio/` and are gitignored; this tracked report records their outcomes and lineage. The local HTML paths below are not available in a fresh Git checkout.

## Scope and method

- Bench displayed 30 `portfolios/*.yaml` configurations at the start of this run. All 30 were preflighted with `PortfolioManager.from_yaml`.
- 26 configurations were valid and completed a fresh PortfolioManager Vanilla run. Four VIXM tail research configurations were rejected before execution because their strategy is not in the Portfolio Manager allowlist.
- Three portfolio jobs ran concurrently with `--max-workers 1` per job. A transient Norgate SPY read error interrupted the first `fund_menu_balanced` attempt; its isolated retry passed. Both logs are preserved.
- No portfolio YAML, strategy rules, capital weights, LIVE release, scheduler, broker state, or allocation was changed by this refresh.
- Validation: each passing portfolio has `summary.json`, `manager_metadata.json`, `metadata.json`, `run_info.json`, `report.html`, the configured pod count, and Bench `Current config` status.

## All displayed portfolios

| Portfolio | Tier | Result | Actual common window | CAGR | Sharpe | Max DD | Report |
|---|---|---|---|---:|---:|---:|---|
| `00_0` | RESEARCH | Passed | 2012-10-01 to 2026-08-19 | 7.24% | 1.53 | -5.16% | local `results/research/portfolio/00_0/vanilla_backtest/2026-09-30_143250/report.html` |
| `0_1` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 12.90% | 1.54 | -7.83% | local `results/research/portfolio/0_1/vanilla_backtest/2026-09-30_143448/report.html` |
| `0_3` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 17.32% | 1.41 | -11.34% | local `results/research/portfolio/0_3/vanilla_backtest/2026-09-30_143638/report.html` |
| `0_4` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 21.52% | 1.34 | -16.28% | local `results/research/portfolio/0_4/vanilla_backtest/2026-09-30_143646/report.html` |
| `0_allin` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 17.50% | 1.44 | -12.65% | local `results/research/portfolio/0_allin/vanilla_backtest/2026-09-30_112114/report.html` |
| `fund_menu_aggressive` | RESEARCH | Passed | 2012-10-02 to 2026-08-19 | 20.65% | 1.37 | -13.29% | local `results/research/portfolio/fund_menu_aggressive/vanilla_backtest/2026-09-30_112114/report.html` |
| `fund_menu_balanced` | RESEARCH | Passed on retry | 2012-10-03 to 2026-08-19 | 13.16% | 1.60 | -7.58% | local `results/research/portfolio/fund_menu_balanced/vanilla_backtest/2026-09-30_144741/report.html` |
| `fund_menu_defensive` | RESEARCH | Passed | 2012-10-03 to 2026-08-19 | 9.86% | 1.80 | -4.39% | local `results/research/portfolio/fund_menu_defensive/vanilla_backtest/2026-09-30_143652/report.html` |
| `fund_menu_growth` | RESEARCH | Passed | 2012-10-03 to 2026-08-19 | 15.93% | 1.46 | -11.02% | local `results/research/portfolio/fund_menu_growth/vanilla_backtest/2026-09-30_112840/report.html` |
| `fund_menu_low_touch_balanced` | RESEARCH | Passed | 2012-10-02 to 2026-08-19 | 11.86% | 1.52 | -6.91% | local `results/research/portfolio/fund_menu_low_touch_balanced/vanilla_backtest/2026-09-30_120146/report.html` |
| `fund_menu_low_touch_defensive` | RESEARCH | Passed | 2012-10-02 to 2026-08-19 | 7.14% | 1.57 | -3.87% | local `results/research/portfolio/fund_menu_low_touch_defensive/vanilla_backtest/2026-09-30_120615/report.html` |
| `fund_menu_low_touch_growth` | RESEARCH | Passed | 2012-10-02 to 2026-08-19 | 15.55% | 1.44 | -9.66% | local `results/research/portfolio/fund_menu_low_touch_growth/vanilla_backtest/2026-09-30_120909/report.html` |
| `ladder_1_defensive` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 10.68% | 1.46 | -8.45% | local `results/research/portfolio/ladder_1_defensive/vanilla_backtest/2026-09-30_143802/report.html` |
| `ladder_1_defensive_proxy_2008` | RESEARCH | Passed | 2008-03-03 to 2026-09-29 | 10.56% | 1.28 | -12.31% | local `results/research/portfolio/ladder_1_defensive_proxy_2008/vanilla_backtest/2026-09-30_121257/report.html` |
| `ladder_2_balanced` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 12.62% | 1.22 | -10.02% | local `results/research/portfolio/ladder_2_balanced/vanilla_backtest/2026-09-30_121518/report.html` |
| `ladder_2_balanced_proxy_2008` | RESEARCH | Passed | 2008-03-03 to 2026-09-29 | 11.74% | 1.17 | -9.92% | local `results/research/portfolio/ladder_2_balanced_proxy_2008/vanilla_backtest/2026-09-30_121536/report.html` |
| `ladder_3_growth` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 20.25% | 1.42 | -13.26% | local `results/research/portfolio/ladder_3_growth/vanilla_backtest/2026-09-30_121628/report.html` |
| `ladder_3_growth_tail_vixm_10_research` | RESEARCH | Blocked: VIXM eligibility | — | — | — | — | — |
| `ladder_3b_growth_2x` | RESEARCH | Passed | 2008-03-03 to 2026-09-29 | 17.38% | 1.21 | -17.47% | local `results/research/portfolio/ladder_3b_growth_2x/vanilla_backtest/2026-09-30_122328/report.html` |
| `ladder_3b_growth_2x_tail_vixm_10_research` | RESEARCH | Blocked: VIXM eligibility | — | — | — | — | — |
| `ladder_3c_growth_2x_btal` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 19.23% | 1.34 | -15.20% | local `results/research/portfolio/ladder_3c_growth_2x_btal/vanilla_backtest/2026-09-30_122403/report.html` |
| `ladder_3c_growth_2x_btal_tail_vixm_10_research` | RESEARCH | Blocked: VIXM eligibility | — | — | — | — | — |
| `ladder_4_growth` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 20.62% | 1.44 | -12.70% | local `results/research/portfolio/ladder_4_growth/vanilla_backtest/2026-09-30_131824/report.html` |
| `ladder_4_growth_1n` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 24.54% | 1.35 | -17.06% | local `results/research/portfolio/ladder_4_growth_1n/vanilla_backtest/2026-09-30_132638/report.html` |
| `ladder_4_growth_1n_rebalance` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 23.11% | 1.36 | -17.26% | local `results/research/portfolio/ladder_4_growth_1n_rebalance/vanilla_backtest/2026-09-30_132648/report.html` |
| `ladder_4_growth_inflation_compass_05_rebalance` | RESEARCH | Passed | 2012-10-01 to 2026-08-14 | 20.77% | 1.46 | -13.66% | local `results/research/portfolio/ladder_4_growth_inflation_compass_05_rebalance/vanilla_backtest/2026-09-30_140346/report.html` |
| `ladder_4_growth_inflation_compass_10_rebalance` | RESEARCH | Passed | 2012-10-01 to 2026-08-14 | 20.79% | 1.46 | -13.98% | local `results/research/portfolio/ladder_4_growth_inflation_compass_10_rebalance/vanilla_backtest/2026-09-30_140742/report.html` |
| `ladder_4_growth_rebalance` | RESEARCH | Passed | 2012-10-01 to 2026-09-29 | 20.42% | 1.43 | -13.35% | local `results/research/portfolio/ladder_4_growth_rebalance/vanilla_backtest/2026-09-30_140749/report.html` |
| `ladder_4_growth_tail_vixm_10_research` | RESEARCH | Blocked: VIXM eligibility | — | — | — | — | — |
| `loren` | WIRED | Passed | 2012-10-01 to 2026-09-29 | 21.14% | 1.35 | -15.33% | local `results/research/portfolio/Loren/vanilla_backtest/2026-09-30_112114/report.html` |

Actual end dates among the 26 completed runs: 16 end on 2026-09-29, 8 on 2026-08-19, and 2 on 2026-08-14. These are actual common pod endpoints, not merely YAML request dates.

## Newly added portfolio configuration files

The latest portfolio configuration commit (`36b6349`, 2026-09-29) added the following 12 YAML files. Some had older research-preview artifacts before the YAML entered the tracked `portfolios/` catalog.

| New portfolio | Rebalance | Configured pods and target weights | Actual Vanilla end |
|---|---|---|---|
| [00_0](../../portfolios/00_0.yaml) | annually / fixed | `pod_taa_df_btal_linearity_1n_fallback_qqq_vix_cash` 33.34%; `pod_taa_adaptive_macro_core5` 33.33%; `pod_taa_tactical_fixed_income_ief_lqd` 33.33% | 2026-08-19 |
| [0_1](../../portfolios/0_1.yaml) | annually / fixed | `pod_taa_df_btal_fallback_tqqq_vix_cash` 33.34%; `pod_mr_us_sector_etf_ibs_downshock_vox_iyr` 33.33%; `pod_taa_adaptive_macro_core5` 33.33% | 2026-09-29 |
| [0_3](../../portfolios/0_3.yaml) | annually / fixed | `pod_taa_df_btal_fallback_tqqq_vix_cash` 60%; `pod_taa_adaptive_macro_core5` 40% | 2026-09-29 |
| [0_4](../../portfolios/0_4.yaml) | annually / fixed | `pod_taa_df_btal_1n_fallback_tqqq_vix_cash` 60%; `pod_taa_adaptive_macro_core5` 40% | 2026-09-29 |
| [0_allin](../../portfolios/0_allin.yaml) | annually / equal | `pod_taa_df_btal_fallback_tqqq_vix_cash` 20%; `pod_taa_adaptive_macro_core5` 20%; `pod_mr_hpi_sp500_2_3_5_vote` 20%; `pod_mo_atr_normalized_ndx_vxn_scaled` 20%; `pod_mr_dv2` 20% | 2026-09-29 |
| [fund_menu_aggressive](../../portfolios/fund_menu_aggressive.yaml) | annually / fixed | `pod_ndx_vxn` 33.33%; `pod_taa_btal_tqqq` 33.34%; `pod_infl_compass` 33.33% | 2026-08-19 |
| [fund_menu_balanced](../../portfolios/fund_menu_balanced.yaml) | annually / fixed | `pod_core5` 19%; `pod_tactical_fi` 9%; `pod_eom_flow` 9%; `pod_sector_vox_iyr` 9%; `pod_dv2` 9%; `pod_hpi_vote` 9%; `pod_disp_kie_ihi_sma` 9%; `pod_taa_btal_tqqq` 9%; `pod_ndx_vxn` 9%; `pod_infl_compass` 9% | 2026-08-19 |
| [fund_menu_defensive](../../portfolios/fund_menu_defensive.yaml) | annually / fixed | `pod_core5` 33%; `pod_tactical_fi` 17%; `pod_eom_flow` 17%; `pod_hpi_vote` 9%; `pod_sector_vox_iyr` 8%; `pod_ndx_vxn` 8%; `pod_taa_btal_tqqq` 8% | 2026-08-19 |
| [fund_menu_growth](../../portfolios/fund_menu_growth.yaml) | annually / fixed | `pod_hpi_vote` 14%; `pod_dv2` 14%; `pod_disp_kie_ihi_sma` 14%; `pod_sector_vox_iyr` 12%; `pod_ndx_vxn` 12%; `pod_taa_btal_tqqq` 12%; `pod_infl_compass` 12%; `pod_core5` 10% | 2026-08-19 |
| [fund_menu_low_touch_balanced](../../portfolios/fund_menu_low_touch_balanced.yaml) | annually / fixed | `pod_core5` 39%; `pod_tactical_fi` 19%; `pod_ndx_vxn` 14%; `pod_infl_compass` 14%; `pod_taa_btal_tqqq` 14% | 2026-08-19 |
| [fund_menu_low_touch_defensive](../../portfolios/fund_menu_low_touch_defensive.yaml) | annually / fixed | `pod_core5` 59%; `pod_tactical_fi` 29%; `pod_ndx_vxn` 6%; `pod_taa_btal_tqqq` 6% | 2026-08-19 |
| [fund_menu_low_touch_growth](../../portfolios/fund_menu_low_touch_growth.yaml) | annually / fixed | `pod_core5` 23%; `pod_ndx_vxn` 22%; `pod_taa_btal_tqqq` 22%; `pod_infl_compass` 22%; `pod_tactical_fi` 11% | 2026-08-19 |

### Strategy import for each new portfolio pod

| Pod ID | Strategy import |
|---|---|
| `pod_core5` | `strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5` |
| `pod_disp_kie_ihi_sma` | `strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200` |
| `pod_dv2` | `strategies.dv2.strategy_mr_dv2:DVO2Strategy` |
| `pod_eom_flow` | `strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow` |
| `pod_hpi_vote` | `strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote` |
| `pod_infl_compass` | `strategies.taa_df.strategy_taa_inflation_compass` |
| `pod_mo_atr_normalized_ndx_vxn_scaled` | `strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy` |
| `pod_mr_dv2` | `strategies.dv2.strategy_mr_dv2:DVO2Strategy` |
| `pod_mr_hpi_sp500_2_3_5_vote` | `strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote` |
| `pod_mr_us_sector_etf_ibs_downshock_vox_iyr` | `strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr` |
| `pod_ndx_vxn` | `strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy` |
| `pod_sector_vox_iyr` | `strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr` |
| `pod_taa_adaptive_macro_core5` | `strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5` |
| `pod_taa_btal_tqqq` | `strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash` |
| `pod_taa_df_btal_1n_fallback_tqqq_vix_cash` | `strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash` |
| `pod_taa_df_btal_fallback_tqqq_vix_cash` | `strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash` |
| `pod_taa_df_btal_linearity_1n_fallback_qqq_vix_cash` | `strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash` |
| `pod_taa_tactical_fixed_income_ief_lqd` | `strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd` |
| `pod_tactical_fi` | `strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd` |

## Existing portfolio configurations changed in the same commit

Seven Ladder 4 YAML files were modified. Each removes its MOSAIC pod and redistributes weights across the other pods; the table shows every affected target weight. Rebalance policies and configured date bounds were retained.

| Portfolio | Removed pod | Remaining pod weights, old → current | Refresh |
|---|---|---|---|
| [ladder_4_growth](../../portfolios/ladder_4_growth.yaml) | `pod_mo_mosaic_russell1000` 8% | `pod_mr_dv2` 16% → 17%; `pod_mr_hpi_sp500_2_3_5_vote` 17% → 19%; `pod_mo_atr_normalized_ndx_vxn_scaled` 25% → 27%; `pod_taa_df_btal_fallback_tqqq_vix_cash` 34% → 37% | Passed |
| [ladder_4_growth_1n](../../portfolios/ladder_4_growth_1n.yaml) | `pod_mo_mosaic_russell1000` 8% | `pod_mr_dv2` 16% → 17%; `pod_mr_hpi_sp500_2_3_5_vote` 17% → 19%; `pod_mo_atr_normalized_ndx_vxn_scaled` 25% → 27%; `pod_taa_df_btal_1n_fallback_tqqq_vix_cash` 34% → 37% | Passed |
| [ladder_4_growth_1n_rebalance](../../portfolios/ladder_4_growth_1n_rebalance.yaml) | `pod_mo_mosaic_russell1000` 8% | `pod_mr_dv2` 16% → 17%; `pod_mr_hpi_sp500_2_3_5_vote` 17% → 19%; `pod_mo_atr_normalized_ndx_vxn_scaled` 25% → 27%; `pod_taa_df_btal_1n_fallback_tqqq_vix_cash` 34% → 37% | Passed |
| [ladder_4_growth_inflation_compass_05_rebalance](../../portfolios/ladder_4_growth_inflation_compass_05_rebalance.yaml) | `pod_mo_mosaic_russell1000` 7.403% | `pod_mr_dv2` 16% → 17%; `pod_mr_hpi_sp500_2_3_5_vote` 17% → 19%; `pod_mo_atr_normalized_ndx_vxn_scaled` 23.1343% → 24.890625%; `pod_taa_df_btal_fallback_tqqq_vix_cash` 31.4627% → 34.109375% | Passed |
| [ladder_4_growth_inflation_compass_10_rebalance](../../portfolios/ladder_4_growth_inflation_compass_10_rebalance.yaml) | `pod_mo_mosaic_russell1000` 6.8059% | `pod_mr_dv2` 16% → 17%; `pod_mr_hpi_sp500_2_3_5_vote` 17% → 19%; `pod_mo_atr_normalized_ndx_vxn_scaled` 21.2687% → 22.78125%; `pod_taa_df_btal_fallback_tqqq_vix_cash` 28.9254% → 31.21875% | Passed |
| [ladder_4_growth_rebalance](../../portfolios/ladder_4_growth_rebalance.yaml) | `pod_mo_mosaic_russell1000` 8% | `pod_mr_dv2` 16% → 17%; `pod_mr_hpi_sp500_2_3_5_vote` 17% → 19%; `pod_mo_atr_normalized_ndx_vxn_scaled` 25% → 27%; `pod_taa_df_btal_fallback_tqqq_vix_cash` 34% → 37% | Passed |
| [ladder_4_growth_tail_vixm_10_research](../../portfolios/ladder_4_growth_tail_vixm_10_research.yaml) | `pod_mo_mosaic_russell1000` 7.2% | `pod_mr_dv2` 14.4% → 15.3%; `pod_mr_hpi_sp500_2_3_5_vote` 15.3% → 17.1%; `pod_mo_atr_normalized_ndx_vxn_scaled` 22.5% → 24.3%; `pod_taa_df_btal_fallback_tqqq_vix_cash` 30.6% → 33.3% | Blocked: VIXM |

### Old saved run versus refreshed Ladder 4 run

These are descriptive comparisons across run dates. They do not isolate the MOSAIC removal.

| Portfolio | Previous → refreshed end | CAGR, old → refreshed | Sharpe, old → refreshed | Max DD, old → refreshed |
|---|---|---:|---:|---:|
| `ladder_4_growth` | 2026-09-11 → 2026-09-29 | 21.65% → 20.62% | 1.49 → 1.44 | -12.90% → -12.70% |
| `ladder_4_growth_1n` | 2026-08-14 → 2026-09-29 | 25.40% → 24.54% | 1.43 → 1.35 | -16.43% → -17.06% |
| `ladder_4_growth_1n_rebalance` | 2026-08-14 → 2026-09-29 | 24.71% → 23.11% | 1.45 → 1.36 | -16.92% → -17.26% |
| `ladder_4_growth_inflation_compass_05_rebalance` | 2026-08-14 → 2026-08-14 | 22.16% → 20.77% | 1.53 → 1.46 | -13.67% → -13.66% |
| `ladder_4_growth_inflation_compass_10_rebalance` | 2026-08-14 → 2026-08-14 | 22.11% → 20.79% | 1.53 → 1.46 | -13.96% → -13.98% |
| `ladder_4_growth_rebalance` | 2026-09-04 → 2026-09-29 | 21.87% → 20.42% | 1.51 → 1.43 | -13.39% → -13.35% |

## Changes relative to earlier saved previews

These seven files were new to the tracked catalog on 2026-09-29, but Bench had older preview runs under the same names. Their current YAML differs from those saved artifacts as follows. The metric pairs are descriptive comparisons across run dates and must not be read as an isolated MOSAIC effect.

### `0_allin`

- Previous saved preview: `2026-09-21_213343`. Removed: `pod_mo_mosaic_russell1000` 16.67%.
- Reweighted: `pod_taa_df_btal_fallback_tqqq_vix_cash` 16.67% → 20%; `pod_taa_adaptive_macro_core5` 16.67% → 20%; `pod_mr_hpi_sp500_2_3_5_vote` 16.67% → 20%; `pod_mo_atr_normalized_ndx_vxn_scaled` 16.67% → 20%; `pod_mr_dv2` 16.65% → 20%.
- CAGR 18.76% → 17.50%; Sharpe 1.51 → 1.44; Max DD -12.55% → -12.65%.

### `fund_menu_aggressive`

- Previous saved preview: `2026-09-24_011427`. Removed: `pod_mosaic` 25%.
- Reweighted: `pod_ndx_vxn` 25% → 33.33%; `pod_taa_btal_tqqq` 25% → 33.34%; `pod_infl_compass` 25% → 33.33%.
- CAGR 22.03% → 20.65%; Sharpe 1.49 → 1.37; Max DD -11.92% → -13.29%.

### `fund_menu_balanced`

- Previous saved preview: `2026-09-24_105620`. Removed: `pod_mosaic` 8%.
- Reweighted: `pod_core5` 18% → 19%; `pod_sector_vox_iyr` 8% → 9%; `pod_dv2` 8% → 9%; `pod_hpi_vote` 8% → 9%; `pod_disp_kie_ihi_sma` 8% → 9%; `pod_taa_btal_tqqq` 8% → 9%; `pod_ndx_vxn` 8% → 9%; `pod_infl_compass` 8% → 9%.
- CAGR 14.08% → 13.16%; Sharpe 1.66 → 1.60; Max DD -7.75% → -7.58%.

### `fund_menu_growth`

- Previous saved preview: `2026-09-24_005652`. Removed: `pod_mosaic` 11%.
- Reweighted: `pod_hpi_vote` 12% → 14%; `pod_dv2` 12% → 14%; `pod_disp_kie_ihi_sma` 12% → 14%; `pod_sector_vox_iyr` 11% → 12%; `pod_ndx_vxn` 11% → 12%; `pod_taa_btal_tqqq` 11% → 12%; `pod_infl_compass` 11% → 12%; `pod_core5` 9% → 10%.
- CAGR 17.07% → 15.93%; Sharpe 1.52 → 1.46; Max DD -11.08% → -11.02%.

### `fund_menu_low_touch_balanced`

- Previous saved preview: `2026-09-24_111310`. Removed: `pod_mosaic` 12%.
- Reweighted: `pod_core5` 35% → 39%; `pod_tactical_fi` 17% → 19%; `pod_ndx_vxn` 12% → 14%; `pod_infl_compass` 12% → 14%; `pod_taa_btal_tqqq` 12% → 14%.
- CAGR 13.61% → 11.86%; Sharpe 1.62 → 1.52; Max DD -6.19% → -6.91%.

### `fund_menu_low_touch_defensive`

- Previous saved preview: `2026-09-24_111637`. Removed: `pod_mosaic` 6%, `pod_infl_compass` 6%.
- Reweighted: `pod_core5` 51% → 59%; `pod_tactical_fi` 25% → 29%.
- CAGR 9.60% → 7.14%; Sharpe 1.70 → 1.57; Max DD -4.32% → -3.87%.

### `fund_menu_low_touch_growth`

- Previous saved preview: `2026-09-24_112427`. Removed: `pod_mosaic` 18%.
- Reweighted: `pod_core5` 19% → 23%; `pod_ndx_vxn` 18% → 22%; `pod_taa_btal_tqqq` 18% → 22%; `pod_infl_compass` 18% → 22%; `pod_tactical_fi` 9% → 11%.
- CAGR 17.54% → 15.55%; Sharpe 1.54 → 1.44; Max DD -8.87% → -9.66%.

## Four blocked VIXM research configurations

The following displayed YAMLs reference `strategies.tail_hedge.strategy_vixm_backwardation`, which is not in the Portfolio Manager allowlist. This is a maturity/eligibility boundary, so the configurations were not rewritten and no substitute portfolio result was invented:

- [`ladder_3_growth_tail_vixm_10_research`](../../portfolios/ladder_3_growth_tail_vixm_10_research.yaml)
- [`ladder_3b_growth_2x_tail_vixm_10_research`](../../portfolios/ladder_3b_growth_2x_tail_vixm_10_research.yaml)
- [`ladder_3c_growth_2x_btal_tail_vixm_10_research`](../../portfolios/ladder_3c_growth_2x_btal_tail_vixm_10_research.yaml)
- [`ladder_4_growth_tail_vixm_10_research`](../../portfolios/ladder_4_growth_tail_vixm_10_research.yaml)

## Interpretation limits

- A fresh run date and Bench `Current config` status do not mean that every portfolio contains market data through 2026-09-29. Eight completed runs end on 2026-08-19 and two on 2026-08-14 because of configured or effective pod cutoffs. `00_0` has no YAML end date but its common pod window ends on 2026-08-19.
- Old-vs-new headline metrics above mix portfolio composition changes with later runs, data updates, and in some cases different end dates. They do not isolate the effect of removing MOSAIC or establish independent edge.
- All 30 displayed portfolios are research artifacts here. The one `WIRED` tier (`loren`) denotes plumbing maturity only; this refresh did not approve capital allocation, PAPER, or LIVE deployment.
- The failed first `fund_menu_balanced` attempt hit a Norgate `price_timeseries` access error while loading SPY in the TAA pod. A later isolated full retry passed, including that pod. See the two local logs in `results/research/portfolio_refresh_20260930/logs/`.
