"""Station S0, retro: the nine studies of 2026-10-16 that ran outside Scout (design A21, gap 1 of the Bench review).

Nine knowledge-base records were added on 2026-10-16 (pakal-research, frozen specs) with no Scout ledger row, so their
trials were missing from every family's N the same day A19 asked for "registration in the same week". This script
registers them retroactively (retro_bool True, prior_trials_int = evaluations the study ran on real data), one row per
knowledge record. No trial rows are written; nothing is re-run.

Counting rules (as A19; prior trials count as independent, overlaps are double counted rather than dropped):
- prior_trials_int = the study's own `adaptive_search.actual_total_variants` (research_state.json / knowledge_record
  adaptive_lineage), which counts every evaluation including holdout re-evaluations and cost-layer re-runs.
- three_momentum_directions_20261016 holds two lanes. Its commodity lane is its own knowledge record
  (commodity_momentum_20261016, 259 scenarios, tactical_asset_allocation); the parent row counts the equity lanes only
  (1,080 - 259 = 821, equity_cross_sectional_momentum), so the 259 are counted once.
- The two data-blocked intakes (curve carry, reversal speed) evaluated nothing: prior 0. They are registered so the
  record exists and a later real test is a child row.
- mr_smart_reversal_proxy attempted 16 cells and completed none (missing delisted holdings): counted as 16, because the
  setups were run on real data even though no path finished.

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/scout_ledger_completeness_20261016/register.py [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import Ledger
from alpha.scout.registration import Registration, register, registration_rows

PAKAL_REPORTS_STR = "pakal-research/reports"
KILL_RETRO_STR = "Retro registration of a closed study: its verdict stands; a forward line, if any, is killed by its S8 CUSUM / Cold Blood Index."
KILL_INTAKE_STR = "Intake only (no performance seen): a real test is a child registration written before its first result, with its own kill rule."


def registration_list() -> list[Registration]:
    return [
        Registration(
            registration_id_str="commodity_carry_contract_intake_retro_20261016",
            family_id_str="carry_premium",
            hypothesis_str="Commodity futures sorted by curve carry (front vs second contract, annualised) earn a positive long-short premium on the 24-market Koijen-Moskowitz-Pedersen-Vrugt universe.",
            mechanism_str="Carry: backwardated curves pay the holder for supplying insurance to hedgers; contango curves charge storage.",
            expected_sign_and_location_str="High-carry minus low-carry monthly excess return positive, with the source's sign in each era; not tested.",
            hypothesis_class_str="X",
            universe_str="24 source commodity markets with their historical entry dates (source Appendix D)",
            horizon_str="one month",
            schedule_str="after all relevant month-end settlements are published",
            execution_str="earliest next tradable session open; needs synchronized executable contract prices",
            param_grid_dict={"signal_str": ("current", "1-12", "2-13"), "timing_str": ("settlement_diagnostic", "next_open"), "cost_bps_round_trip_int": (0, 2, 10)},
            primary_metric_str="Long-short carry portfolio excess return and Sharpe after next-open fills and costs (not run: data blocked)",
            kill_criteria_str=KILL_INTAKE_STR,
            source_str=f"Koijen, Moskowitz, Pedersen, Vrugt (2018) 'Carry', JFE 127(2); {PAKAL_REPORTS_STR}/mr_other_discovery_20261016/other_families/carry_contract_intake (18 declared, 0 executed: the exact contract panel is not available locally). Verdict: inconclusive, intake only.",
            source_published_date_str="2018-02-01",
            retro_bool=True,
            prior_trials_int=0,
            universe_choice_str="The source's 24 markets; no local result exists.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="commodity_dual_momentum_retro_20261016",
            family_id_str="tactical_asset_allocation",
            hypothesis_str="Adding an absolute-trend filter to relative momentum across four commodity sector ETFs (dual momentum) improves risk-adjusted return over relative momentum alone, after next-open fills and costs.",
            mechanism_str="Cross-asset relative momentum plus a time-series trend exit to cash in commodity bear markets.",
            expected_sign_and_location_str="Dual minus single momentum Sharpe positive across the source grid (lookback x breadth), at 10 and 25 bp a side.",
            hypothesis_class_str="W",
            universe_str="DBA, DBB, DBE, DBP (fixed); cash reserve 1%",
            horizon_str="one month",
            schedule_str="monthly Close_T",
            execution_str="Open_(T+1), quantities fixed after Close_T",
            param_grid_dict={"lookback_month_int": (3, 6, 9, 12), "breadth_int": (1, 2, 3), "framework_str": ("single", "dual")},
            primary_metric_str="Paired dual-vs-single Sharpe difference per cell, incremental regression on existing momentum, ADV capacity",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"Quantpedia 'Dual vs Single Momentum in Commodities' (SSRN 6953821 blocked); {PAKAL_REPORTS_STR}/three_momentum_directions_20261016/commodity_lane (259 scenarios = 256 frozen + 3 post-result capacity). Verdict: promising risk-control component; independent alpha unproven (control regression p 0.115); ETF capacity severe.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=259,
            universe_choice_str="The source's four DB sector ETFs, fixed in the frozen spec.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="cross_asset_regime_gate_taa_retro_20261016",
            family_id_str="tactical_asset_allocation",
            parent_id_str="taa_3x_reaudition_20261002",
            hypothesis_str="A literature gate (PSV cross-asset trend, TIP canary, Zakamulin 2/4-state regime) improves the live TAA pod's TQQQ-fallback decision versus its rv20 < VIX gate, on excess Sharpe and drawdown after costs.",
            mechanism_str="Cross-asset and regime trend states carry information on next-month equity returns and volatility.",
            expected_sign_and_location_str="At least one F1 gate Holm-significant over the live gate in 1994-2020 dev, not worse in the 2021-2026 holdout, and better in the G3 book.",
            hypothesis_class_str="W",
            universe_str="TAA pod list GLD UUP TLT DBC BTAL TQQQ BIL (+ SPY QQQ IEF TLT TIP signals), proxies before listing",
            horizon_str="one month",
            schedule_str="close of the last session of the month",
            execution_str="Open of the first session of the next month",
            param_grid_dict={"gate_str": ("PSV_stack", "PSV_solo", "TIP_stack", "TIP_solo", "ZK2_stack", "ZK2_solo", "ZK4_stack", "ZK4_solo")},
            primary_metric_str="One-sided bootstrap of the excess Sharpe difference vs the live gate, Holm over the 8 F1 gates",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"Pitkajarvi-Suominen-Vaittinen (2020), Keller-Keuning HAA (2023), Zakamulin (2026); {PAKAL_REPORTS_STR}/cross_asset_regime_gate_taa_study (58 evaluations). Verdict: rejected; no gate beats the live rule on dev, holdout or the G3 book.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=58,
            universe_choice_str="The live TAA pod's ETF list (fixed by the live strategy).",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="intramonth_momentum_cycle_ndx_retro_20261016",
            family_id_str="equity_cross_sectional_momentum",
            parent_id_str="ndx_vxn_reaudition_20261002",
            hypothesis_str="Momentum returns concentrate before the turn of the month (PreTOM D-9..D-4, Nathan-Suominen-Tasa 2026) in point-in-time S&P 500 / NDX members, and re-timing the live NDX pod's rebalance around it helps.",
            mechanism_str="Predictable pre-month-end selling of losers by funds (window dressing / flows) concentrates the momentum spread.",
            expected_sign_and_location_str="PreTOM loser-leg return below the rest of the month in both universes (Holm over 6); at least one pod re-timing beats D0 in 2000-2020 (Holm over 7) and holds in 2021-2026.",
            hypothesis_class_str="X",
            universe_str="S&P 500 and Nasdaq-100 point-in-time members (Norgate)",
            horizon_str="one month",
            schedule_str="Close_T of the decision row (D0 live; D-k variants)",
            execution_str="Open_(T+1)",
            param_grid_dict={"variant_str": ("P1_10", "P1_9", "P1_8", "P2_10", "P2_9", "P2_8", "P3")},
            primary_metric_str="One-sided bootstrap p of the Sharpe difference vs the live D0 schedule, Holm over 7",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"Nathan, Suominen, Tasa (2026) 'The Intramonth Momentum Cycle', SSRN 6426026 (2026; exact date unknown, the year start is the published date; internet appendix read); {PAKAL_REPORTS_STR}/intramonth_momentum_cycle_ndx_study (87 evaluations, 44 distinct). Verdict: rejected for the pod; offset ranks flip between periods (rebalance-day luck).",
            source_published_date_str="2026-01-01",
            retro_bool=True,
            prior_trials_int=87,
            universe_choice_str="The live NDX pod's universe plus the S&P 500 for replication.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="mr_liquidity_speed_intake_retro_20261016",
            family_id_str="us_equity_short_term_reversal",
            hypothesis_str="Short-term reversal is faster in high-volatility stocks and more persistent in high share-turnover stocks (Dai et al. 2023), which a reversal pod could use to size or time exits.",
            mechanism_str="Liquidity provision: the price impact of uninformed selling decays at a speed set by volatility and turnover.",
            expected_sign_and_location_str="Reversal response curve steeper for high-volatility groups and flatter for high-turnover groups; not tested.",
            hypothesis_class_str="X",
            universe_str="Source: US common stocks NYSE/AMEX/NASDAQ (CRSP/Compustat); no local tested universe",
            horizon_str="1-63 sessions",
            schedule_str="after Close_T",
            execution_str="Open_(T+1) prospective; not tested",
            param_grid_dict={"formation_window_int": (1, 5, 21), "conditioning_str": ("volatility", "turnover", "size"), "group_int": (1, 2, 3, 4, 5)},
            primary_metric_str="Group response curves of the source (not run: PIT inputs and strategy definition unavailable)",
            kill_criteria_str=KILL_INTAKE_STR,
            source_str=f"Dai et al. (2023), NBER 30917 / SSRN 4349550 (full read); {PAKAL_REPORTS_STR}/mr_other_discovery_20261016/liquidity_speed (45 declared, 0 executed). Verdict: NOT_TESTED, intake only.",
            source_published_date_str="2023-02-01",
            retro_bool=True,
            prior_trials_int=0,
            universe_choice_str="The source's universe; no local result exists.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="mr_smart_reversal_proxy_retro_20261016",
            family_id_str="us_equity_short_term_reversal",
            hypothesis_str="Weekly five-day reversal among the largest US stocks, with half-rank retention ('smart' turnover control, de Groot-Huij-Zhou 2011), keeps positive net returns after a skipped session and costs.",
            mechanism_str="Liquidity provision to impatient sellers in large caps; retention cuts the turnover that costs eat.",
            expected_sign_and_location_str="Smart constructor net Sharpe above zero and above the standard constructor in 1995-2015 discovery, in both proxies.",
            hypothesis_class_str="X",
            universe_str="S&P 100 PIT proxy and Russell Top 200 PIT proxy (not the source's exact top-100 cap universe)",
            horizon_str="one week",
            schedule_str="Close_T",
            execution_str="Open_(T+2) after a skipped T+1",
            param_grid_dict={"universe_str": ("sp100", "rtop200"), "constructor_str": ("standard", "smart"), "cost_str": ("no_cost", "paper_like", "central_research", "conservative_survival")},
            primary_metric_str="Discovery-period net Sharpe and smart-minus-standard difference (no path completed)",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"de Groot, Huij, Zhou (2011) 'Another look at trading costs and short-term reversal profits' (full read); {PAKAL_REPORTS_STR}/mr_other_discovery_20261016/reversal (16 cells attempted, 0 completed: unresolved delisted holdings). Verdict: inconclusive, data-blocked.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=16,
            universe_choice_str="Two declared proxies for the source universe, frozen before the run.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="ndx_pod_multi_tranche_retro_20261016",
            family_id_str="equity_cross_sectional_momentum",
            parent_id_str="intramonth_momentum_cycle_ndx_retro_20261016",
            hypothesis_str="Splitting the live NDX momentum pod into K fixed-capital sub-books rebuilt on different days of the month cuts drawdown and rebalance-day luck without losing CAGR versus the single D0 / D+1-open pod.",
            mechanism_str="No new signal: diversification across rebalance days averages away the luck of one day's ranking.",
            expected_sign_and_location_str="K2 or K4 Ulcer index below the live day (Holm over 6) with CAGR non-inferior, in NDX 2000-2026, NDX 1995-1999 and S&P 500 transfer windows.",
            hypothesis_class_str="X",
            universe_str="Nasdaq-100 point-in-time (primary), S&P 500 point-in-time (transfer)",
            horizon_str="one month per sub-book",
            schedule_str="Close_T of each sub-book's own decision row",
            execution_str="Open_(T+1) per sub-book",
            param_grid_dict={"tranche_count_int": (2, 4), "window_str": ("ndx_seen", "ndx_pre2000", "sp500_transfer")},
            primary_metric_str="Ulcer bootstrap and CAGR non-inferiority vs the live day, Holm over 6 each; G3 book Sharpe",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"Post-hoc finding IMC-03 of intramonth_momentum_cycle_ndx_study; {PAKAL_REPORTS_STR}/ndx_pod_multi_tranche_study (129 evaluations). Verdict: rejected as an upgrade; K4 fails the frozen rule, no Holm survivor, G3 1.294 -> 1.310 below the 1.35 gate.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=129,
            universe_choice_str="The live pod's universe; NDX 2000-2026 and all 20 offsets had been seen in the parent study (disclosed).",
            universe_chosen_after_results_bool=True,
        ),
        Registration(
            registration_id_str="three_momentum_directions_equity_retro_20261016",
            family_id_str="equity_cross_sectional_momentum",
            hypothesis_str="Two non-ML momentum translations, nearness to the 52-week high (George-Hwang 2004, dynamic anchoring 2024) and return acceleration (AFS grid), keep positive net returns after next-open fills and add alpha beyond ordinary 12-1 momentum.",
            mechanism_str="Anchoring on the 52-week high and under-reaction to accelerating trends delay price discovery.",
            expected_sign_and_location_str="Net long-short return positive across the published grid, and a positive incremental alpha over existing momentum after correction (96 tests).",
            hypothesis_class_str="X",
            universe_str="Point-in-time Russell 1000 with raw price >= 5, ADV63 >= $5M, 240 valid 52-week closes",
            horizon_str="1-12 months",
            schedule_str="completed monthly Close_T with source trailing / skip rules",
            execution_str="Open_(T+1), quantities fixed at Close_T",
            param_grid_dict={"afs_f_int": (3, 6, 12), "afs_h_int": (1, 3, 6, 12), "afs_s_int": (1, 3, 6), "equity_cost_side_bps_int": (0, 10, 25)},
            primary_metric_str="Net return per cell and corrected incremental alpha over existing momentum",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"George and Hwang (2004) JF; dynamic 52-week high (2024); AFS acceleration grid; {PAKAL_REPORTS_STR}/three_momentum_directions_20261016 (1,080 evaluations, of which 259 are the commodity lane registered separately: 821 here). Verdict: positive net returns, no corrected incremental alpha among 96 tests.",
            source_published_date_str=None,
            retro_bool=True,
            prior_trials_int=821,
            universe_choice_str="A liquid PIT Russell 1000 proxy for the CRSP source universe, frozen before the run.",
            universe_chosen_after_results_bool=False,
        ),
        Registration(
            registration_id_str="vxn_trend_double_count_ndx_retro_20261016",
            family_id_str="equity_cross_sectional_momentum",
            parent_id_str="ndx_vxn_reaudition_20261002",
            hypothesis_str="The live NDX-VXN pod's VXN exposure scale double-counts an index trend gate (Nasdaq-100 12-month excess return > 0), so dropping one of the two, or adding the gate, would not hurt.",
            mechanism_str="Volatility scaling and trend gates both cut exposure in bear markets; the question is whether they carry the same information.",
            expected_sign_and_location_str="2x2 (VXN on/off x trend on/off): a live-change variant beats the live pod in 2000-2020 (Holm over 3) and holds in the 2021-2026 holdout.",
            hypothesis_class_str="X",
            universe_str="Nasdaq-100 point-in-time (Norgate)",
            horizon_str="one month",
            schedule_str="close of the last session of the month (D0)",
            execution_str="Open_(T+1)",
            param_grid_dict={"vxn_scale_bool": (True, False), "trend_gate_bool": (True, False)},
            primary_metric_str="One-sided bootstrap of the Sharpe difference vs the live pod, Holm over 3; interaction of the 2x2",
            kill_criteria_str=KILL_RETRO_STR,
            source_str=f"Zakamulin (2026) SSRN 7318318 / 6910478 (exact date unknown, the year start is the published date) via catalog LSM-04; {PAKAL_REPORTS_STR}/vxn_trend_double_count_ndx_study (16 evaluations, 6 distinct). Verdict: no double count; the trend gate lowers the pod's Sharpe; keep the live pod.",
            source_published_date_str="2026-01-01",
            retro_bool=True,
            prior_trials_int=16,
            universe_choice_str="The live pod's universe.",
            universe_chosen_after_results_bool=False,
        ),
    ]


def main() -> None:
    dry_run_bool = "--dry-run" in sys.argv
    ledger = Ledger()
    existing_dict = registration_rows(ledger)
    for reg in registration_list():
        reg.validate()
        if reg.registration_id_str in existing_dict:
            print(f"skip (exists)  {reg.registration_id_str}")
            continue
        if dry_run_bool:
            print(f"would register {reg.registration_id_str}  family={reg.family_id_str}  prior={reg.prior_trials_int}  grid={reg.grid_size_int}  parent={reg.parent_id_str}")
            continue
        row_dict = register(ledger, reg)
        existing_dict[reg.registration_id_str] = row_dict
        print(f"registered row {row_dict['row_id_int']}  {reg.registration_id_str}  prior={reg.prior_trials_int}")
    print(f"ledger chain OK: {ledger.verify()} rows")


if __name__ == "__main__":
    main()
