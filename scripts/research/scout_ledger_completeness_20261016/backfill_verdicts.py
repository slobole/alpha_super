"""Station verdict back-fill: every registration gets machine-readable verdict rows (design A21, gap 3).

Before this script only the dividend month premium (row 52) had `station_verdict` rows. The grades of the 27
re-auditions lived in their HTML cards and pickled bundles, and the other registrations' outcomes lived in reports.
The Bench page needs them in the ledger. Two sources, both retro (`retro_bool` True, `source_str` = the file):

1. **Re-auditions with a saved bundle** (results/scout/reaudition/<pod>/bundle.pkl, found through the registration's
   back-filled test_result rows): one row per station the bundle holds (S3 if run, S4, S5, S6), with the card's own
   check lists, and the grade from `alpha.scout.card.grade_str`, the function that printed the card. Nothing is re-run.
2. **Registrations without a card** (Scout studies that stopped before a card, and studies run outside Scout): one
   `grade` row, read from the study's registered rule and report. The qualifier says how far the study got:
   "(no card)" = a Scout study that ran its own registered rule but not S4-S6 as a card; "(outside Scout)" = a retro
   registration of a study run elsewhere. None of them is graded CANDIDATE, because none passed S0-S6.

A registration that already has a grade row is skipped (the dividend month premium, row 52).

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/scout_ledger_completeness_20261016/backfill_verdicts.py [--dry-run]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_PATH))

from alpha.scout.card import grade_str, s3_check_list
from alpha.scout.fdr_report import record_station_verdict
from alpha.scout.ledger import Ledger
from alpha.scout.registration import registration_rows

NO_CARD_STR = "no card"
OUTSIDE_STR = "outside Scout"
ROBUSTNESS_REPORT_STR = "docs/research/SCOUT_ROBUSTNESS_20261002.md"
MEGACAP_REPORT_STR = "docs/research/MEGACAP_MOMENTUM_SCOUT_20261005.md"
PAKAL_STR = "pakal-research/reports"

# registration id -> (grade word, qualifier, reason, source)
NO_CARD_GRADE_DICT: dict[str, tuple[str, str, str, str]] = {
    "dv2_size_ladder_20261002": ("DIAGNOSTIC", NO_CARD_STR, "Hypothesis held in sample: the gross edge sits in small caps, net of costs only large caps survive. No new pod; the S&P 500 stays DV2's universe. S3 era decay +18.2 / +5.4 / +0.8 bp (A16).", "docs/research/SCOUT_DV2_SIZE_LADDER_20261002.md"),
    "dv2_limit_entry_20261002": ("WATCHLIST", NO_CARD_STR, "Large caps: limit entries cut cost per round trip about 60%, robust to strict fills; plan on S&P 500 net Sharpe about 0.70 / 0.82 at a 40% fill rate (A18). Small-cap variant not a pod. Promotion needs paper or real fill evidence (A17).", "docs/research/SCOUT_DV2_LIMIT_ENTRY_20261002.md"),
    "ndx_natr20_vxn_plateau_filter_20261002": ("DIAGNOSTIC", NO_CARD_STR, "Sensitivity map (not a promotion candidate by its registration): flat ROC x filter surface; the live ROC 12 / SMA100 / top 10 sits inside the plateau; short filters hurt.", ROBUSTNESS_REPORT_STR),
    "ndx_natr20_vxn_ensemble_cmma_20261002": ("DIAGNOSTIC", NO_CARD_STR, "Not a promotion candidate by its registration: the E1 plateau ensemble does not narrow the luck band; the E2 ranking blend is the balanced design (later the design of record); CMMA gives no robust improvement.", ROBUSTNESS_REPORT_STR),
    "ndx_vxn_linear_trend_ranking_20261003": ("REJECTED", NO_CARD_STR, "No replacement for NATR20: the linear-trend ranking is the best-shaped signal (rank IC 0.023, t 1.9) but not a better strategy; it lags after 2022 (0.71).", ROBUSTNESS_REPORT_STR),
    "dv2_limit_anchor_20261003": ("REJECTED", NO_CARD_STR, "Registered hypothesis fails: no anchor / offset variant beats Close - k x NATR14 at equal fill rate after Romano-Wolf (best adjusted p 0.13); the open anchor does not replicate (A18).", "docs/research/SCOUT_DV2_LIMIT_ANCHOR_20261003.md"),
    "ndx_natr20_vxn_filter_bakeoff_20261003": ("REJECTED", NO_CARD_STR, "F4 passed the per-study rule, but the family-wide Romano-Wolf over all 38 filters gives adjusted p 0.19: no filter beats Close > SMA100; registered fallback, keep F0.", ROBUSTNESS_REPORT_STR),
    "ndx_rank_horizon_blend_candidate_20261003": ("WATCHLIST", NO_CARD_STR, "C passed its registered rule (luck median and 97-98% of paired draws in sample) but trailed the live pod after 2022 on 91% of draws; decision 2026-10-03: no shadow (corr 0.94, about 50 years to resolve 0.1 Sharpe); superseded by E2 as the fund design.", ROBUSTNESS_REPORT_STR),
    "ndx_e2_sector_cap_20261004": ("WATCHLIST", NO_CARD_STR, "The 40% GICS-sector cap passed its registered rule (Max DD -24.8% -> -21.5%, Sharpe and luck median within 0.05); design of record, WIRED 2026-10-11. No S5 MCPT of its own; current GICS labels are a mild look-ahead.", ROBUSTNESS_REPORT_STR),
    "ndx_momentum_decision_controls_20261004": ("WATCHLIST", NO_CARD_STR, "Registered verdict UNCLEAR: the top-10 selection beats 200 random books (99.5th percentile) and equal weight (P 0.98), but not QQQ held at the pod's own exposure (P 0.78 < 0.80).", "docs/research/MOMENTUM_DECISION_20261004.md"),
    "sp500_megacap_momentum_voltarget_20261005": ("WATCHLIST", NO_CARD_STR, "Report status WATCHLIST / diagnostic: the overlays fix the drawdown (-76% -> about -20%); the selection passes the in-sample rule only on 1999-2008; S5 MCPT not run.", MEGACAP_REPORT_STR),
    "sp500_megacap_momentum_stock_filter_vix_20261005": ("REJECTED", NO_CARD_STR, "Both registered additions dropped: the adaptive stock filter (mean P 0.26 vs the 0.80 bar) and both VIX-level gates (the VIX level is not a trend signal: -65% in the dot-com bear at VIX < 25).", MEGACAP_REPORT_STR),
    "sp500_biggest_n_adaptive_entry_gate_20261005": ("REJECTED", NO_CARD_STR, "Registered verdict: not a candidate. Mean P(adaptive > no filter) 0.42, P(> SPY) 0.58 (0.80 needed); the adaptive filter does not beat the SMA100 benchmark filter: a brake, not an edge.", MEGACAP_REPORT_STR),
    "antonacci_gem_retro_20261010": ("REJECTED", OUTSIDE_STR, "Post-publication GEM trails the S&P 500 (P 0.02).", f"{PAKAL_STR}/antonacci_models_study"),
    "antonacci_trend_models_retro_20261010": ("REJECTED", OUTSIDE_STR, "Claimed Sharpe 1.5-2.2 not reachable from public ingredients (best 0.67 / 0.90 / 0.88).", f"{PAKAL_STR}/antonacci_models_study"),
    "antonacci_reversal_overlays_retro_20261010": ("REJECTED", OUTSIDE_STR, "Snap Back win rate 51.2% against the claimed 70-80%.", f"{PAKAL_STR}/antonacci_models_study"),
    "bond_calendar_flow_family_retro_20261016": ("WATCHLIST", OUTSIDE_STR, "Effect real and index-driven, but the composite gate passed no window; forward hypotheses only.", f"{PAKAL_STR}/bond_calendar_flow_family_study"),
    "core5_equity_vehicle_v2_retro_20261011": ("WATCHLIST", OUTSIDE_STR, "QQQ own-signal and UPRO SPY-signal are forward hypotheses; TQQQ downgraded by 2000-02. The QQQ line is in S8 shadow since 2026-10-17 (row 119).", f"{PAKAL_STR}/core5_equity_vehicle_v2"),
    "ltr_momentum_retro_20261014": ("REJECTED", OUTSIDE_STR, "Learning-to-rank momentum not replicated on Russell 3000 / 1000.", f"{PAKAL_STR}/ltr_momentum_study"),
    "megacap_concentration_beta_boost_retro_20261010": ("REJECTED", OUTSIDE_STR, "Rejected as alpha (no Holm survivor vs the beta-matched control); accepted as beta.", f"{PAKAL_STR}/megacap_concentration_beta_boost"),
    "ndx_megacap_tilt_sleeve_retro_20261010": ("WATCHLIST", OUTSIDE_STR, "Promoted to paper shadow by its pre-registered rule although not significant (Holm p 0.35); no Scout spec, so no S8 shadow yet.", f"{PAKAL_STR}/ndx_megacap_tilt_sleeve"),
    "ndx_dip_ladder_tqqq_retro_20261016": ("REJECTED", OUTSIDE_STR, "The excess over SPY is leverage, not timing (random-timing placebo).", f"{PAKAL_STR}/ndx_dip_ladder_tqqq_study"),
    "rangerz_breakout_generator_retro_20261012": ("REJECTED", OUTSIDE_STR, "The selection process fails its locked holdout (Sharpe -0.22).", f"{PAKAL_STR}/rangerz_breakout_generator_study"),
    "valuelytica_fx_us_hours_retro_20261016": ("REJECTED", OUTSIDE_STR, "Costs exceed the drift after 2020.", f"{PAKAL_STR}/valuelytica_fx_us_hours_study"),
    "commodity_carry_contract_intake_retro_20261016": ("NOT_TESTED", OUTSIDE_STR, "Data blocked: no exact simultaneous contract panel; 0 of 18 declared variants run.", f"{PAKAL_STR}/mr_other_discovery_20261016/other_families/carry_contract_intake"),
    "commodity_dual_momentum_retro_20261016": ("WATCHLIST", OUTSIDE_STR, "Risk-adjusted improvement across the source grid, but independent alpha unproven (control regression p 0.115) and ETF capacity severe.", f"{PAKAL_STR}/three_momentum_directions_20261016/commodity_lane"),
    "cross_asset_regime_gate_taa_retro_20261016": ("REJECTED", OUTSIDE_STR, "No gate beats the live rv20 < VIX rule on dev, holdout or the G3 book (dev Sharpe -0.02 .. -0.22).", f"{PAKAL_STR}/cross_asset_regime_gate_taa_study"),
    "intramonth_momentum_cycle_ndx_retro_20261016": ("REJECTED", OUTSIDE_STR, "0/7 pod re-timings pass discovery (Holm p 1.0); offset ranks flip between periods (rebalance-day luck).", f"{PAKAL_STR}/intramonth_momentum_cycle_ndx_study"),
    "mr_liquidity_speed_intake_retro_20261016": ("NOT_TESTED", OUTSIDE_STR, "Data blocked: point-in-time inputs and the strategy definition unavailable; 0 of 45 declared variants run.", f"{PAKAL_STR}/mr_other_discovery_20261016/liquidity_speed"),
    "mr_smart_reversal_proxy_retro_20261016": ("NOT_TESTED", OUTSIDE_STR, "Data blocked: 16 cells attempted, none completed (unresolved delisted holdings).", f"{PAKAL_STR}/mr_other_discovery_20261016/reversal"),
    "ndx_pod_multi_tranche_retro_20261016": ("REJECTED", OUTSIDE_STR, "K4 fails the frozen rule vs the live day (Ulcer higher, -0.46 pp CAGR); no Holm survivor; G3 1.294 -> 1.310, below the 1.35 gate.", f"{PAKAL_STR}/ndx_pod_multi_tranche_study"),
    "three_momentum_directions_equity_retro_20261016": ("REJECTED", OUTSIDE_STR, "The registered claim (alpha beyond ordinary momentum) fails: no corrected incremental alpha among 96 tests; net returns positive but redundant with existing momentum.", f"{PAKAL_STR}/three_momentum_directions_20261016"),
    "vxn_trend_double_count_ndx_retro_20261016": ("REJECTED", OUTSIDE_STR, "No double count: VXN holds all of the trend flag's volatility information; the trend gate lowers the pod's Sharpe; 0/3 live-change variants pass. Keep the live pod.", f"{PAKAL_STR}/vxn_trend_double_count_ndx_study"),
}


def bundle_folder_dict(ledger: Ledger) -> dict[str, Path]:
    """registration id -> re-audition folder, from the paths in its back-filled test_result rows."""
    folder_dict: dict[str, Path] = {}
    for row_dict in ledger.rows("test_result"):
        folder_path = ROOT_PATH / Path(row_dict.get("source_str", "").replace("\\", "/")).parent
        if (folder_path / "bundle.pkl").exists():
            folder_dict.setdefault(row_dict["registration_id_str"], folder_path)
    return folder_dict


def graded_id_set(ledger: Ledger) -> set[str]:
    return {row["registration_id_str"] for row in ledger.rows("station_verdict") if row.get("station_str") == "grade"}


def bundle_station_list(bundle: dict) -> list[tuple[str, list]]:
    station_list = []
    s3_list = s3_check_list(bundle)
    if s3_list:
        station_list.append(("S3", s3_list))
    for station_str in ("s4", "s5", "s6"):
        station_list.append((station_str.upper(), list(bundle[station_str].check_list)))
    return station_list


def main() -> None:
    dry_run_bool = "--dry-run" in sys.argv
    ledger = Ledger()
    registration_dict = registration_rows(ledger)
    folder_dict = bundle_folder_dict(ledger)
    done_set = graded_id_set(ledger)
    unknown_list = [rid for rid in registration_dict if rid not in folder_dict and rid not in NO_CARD_GRADE_DICT and rid not in done_set]
    if unknown_list:
        raise SystemExit(f"No verdict source for: {unknown_list}")
    stale_list = [rid for rid in NO_CARD_GRADE_DICT if rid not in registration_dict]
    if stale_list:
        raise SystemExit(f"Not registered: {stale_list}")

    count_dict = {"bundle": 0, "no_card": 0, "skipped": 0}
    for rid in registration_dict:
        if rid in done_set:
            count_dict["skipped"] += 1
            print(f"skip (graded)  {rid}")
            continue
        if rid in folder_dict:
            bundle_path = folder_dict[rid] / "bundle.pkl"
            with bundle_path.open("rb") as file_obj:
                bundle = pickle.load(file_obj)
            source_str = str(bundle_path.relative_to(ROOT_PATH)).replace("\\", "/")
            grade = grade_str(bundle)
            station_list = bundle_station_list(bundle)
            print(f"{'would write' if dry_run_bool else 'write'} {rid}: {' '.join(s for s, _ in station_list)} + grade {grade}  ({source_str})")
            if not dry_run_bool:
                for station_str, check_list in station_list:
                    record_station_verdict(ledger, rid, station_str, check_list, retro_bool=True, source_str=source_str)
                record_station_verdict(ledger, rid, "grade", [("grade", grade, "S0-S6 re-audition card (grade_str of the saved bundle)")],
                                       retro_bool=True, source_str=source_str)
            count_dict["bundle"] += 1
        else:
            word_str, qualifier_str, reason_str, source_str = NO_CARD_GRADE_DICT[rid]
            grade = f"{word_str} ({qualifier_str})"
            print(f"{'would write' if dry_run_bool else 'write'} {rid}: grade {grade}")
            if not dry_run_bool:
                record_station_verdict(ledger, rid, "grade", [("grade", grade, reason_str)], retro_bool=True, source_str=source_str)
            count_dict["no_card"] += 1
    print(f"{count_dict}; ledger chain OK: {ledger.verify()} rows")


if __name__ == "__main__":
    main()
