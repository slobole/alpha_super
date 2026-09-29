"""Freeze a mechanism-led portfolio comparison before reading its new results."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT_PATH = Path(__file__).resolve().parents[3]
STUDY_PATH = ROOT_PATH / "results/research/portfolio_family_20260923"

ALIAS_DICT = {
    "core5": "strategy_taa_adaptive_macro_core5",
    "ndx": "strategy_mo_atr_normalized_ndx_vxn_scaled",
    "mosaic": "strategy_mo_mosaic_russell1000",
    "dv2": "strategy_mr_dv2",
    "hpi": "strategy_mr_hpi_sp500_2_3_5_vote",
    "sector": "strategy_mr_us_sector_etf_ibs_downshock_vox_iyr",
    "taa_def": "strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
    "taa_growth": "strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "trinity": "strategy_taa_trinity_vol_control_8_bil",
    "crisis": "strategy_crisis_trend_core",
    "vixm": "strategy_vixm_backwardation",
    "flow": "strategy_taa_month_end_rebalancing_flow",
    "tactical_fi": "strategy_taa_tactical_fixed_income_ief_lqd",
    "bil": "benchmark_bil",
}


def sha256_str(file_path: Path) -> str:
    hash_obj = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk_bytes in iter(lambda: file_obj.read(1 << 20), b""):
            hash_obj.update(chunk_bytes)
    return hash_obj.hexdigest()


def candidate_list() -> list[dict]:
    candidate_dict_list = []

    def add(candidate_id_str: str, family_str: str, weight_dict: dict,
            hypothesis_str: str, role_str: str = "candidate") -> None:
        clean_weight_dict = {ALIAS_DICT.get(key_str, key_str): float(value_float)
                             for key_str, value_float in weight_dict.items()
                             if value_float > 0}
        if abs(sum(clean_weight_dict.values()) - 1) > 1e-12:
            raise ValueError(candidate_id_str)
        candidate_dict_list.append({"candidate_id": candidate_id_str,
                                    "family": family_str, "role": role_str,
                                    "weights": clean_weight_dict,
                                    "mechanism": hypothesis_str})

    growth_weight_dict = {"ndx": .25, "mosaic": .25, "dv2": .25, "hpi": .25}
    for core_weight_float in (1, .75, .5, .25, 0):
        weight_dict = {"core5": core_weight_float, **{
            alias_str: (1 - core_weight_float) * sleeve_weight_float
            for alias_str, sleeve_weight_float in growth_weight_dict.items()}}
        add(f"CORE_{int(core_weight_float * 100):03d}", "core_ladder", weight_dict,
            "Owner-selected macro anchor plus equal momentum/rebound budgets; equal representatives within each growth family.")

    for taa_alias_str in ("taa_def", "taa_growth"):
        for core_weight_float in (.75, .5, .25):
            weight_dict = {"core5": core_weight_float,
                           taa_alias_str: (1 - core_weight_float) / 3}
            weight_dict.update({alias_str: (1-core_weight_float)/6
                                for alias_str in growth_weight_dict})
            add(f"{taa_alias_str.upper()}_CORE_{int(core_weight_float*100)}",
                "taa_increment", weight_dict,
                "Reallocate the growth budget equally across momentum, rebound and TAA; compare unlevered versus embedded leveraged equity fallback.")

    for taa_weight_float in (1, .75, .5, .25, 0):
        add(f"MONTHLY_TAA_{int(taa_weight_float*100):03d}", "monthly_only",
            {"taa_def": taa_weight_float, "ndx": (1-taa_weight_float)/2,
             "mosaic": (1-taa_weight_float)/2},
            "Strict month-end strategy orders, unlevered TAA anchor and equal momentum representatives.")

    for satellite_str in ("trinity", "crisis", "vixm"):
        for satellite_weight_float in (.10, .20):
            add(f"DEF_{satellite_str.upper()}_{int(satellite_weight_float*100)}",
                "defensive_satellite",
                {"core5": 1-satellite_weight_float, satellite_str: satellite_weight_float},
                "Test whether a different risk-control mechanism improves CORE5 downside enough to justify carry, complexity and overlap.")

    ladder4_weight_dict = {"dv2": .16, "hpi": .17, "ndx": .25, "mosaic": .08, "taa_growth": .34}
    for core_weight_float in (.25, .50, .75):
        for anchor_str in ("core5", "bil"):
            add(f"L4_{anchor_str.upper()}_{int(core_weight_float*100)}", "ladder_anchor",
                {anchor_str: core_weight_float, **{alias_str: (1-core_weight_float)*weight_float
                 for alias_str, weight_float in ladder4_weight_dict.items()}},
                "Reduce every original Ladder4 growth sleeve proportionally; compare CORE5 with equal capital in short Treasury bills.",
                "matched_control" if anchor_str == "bil" else "candidate")

    for ndx_weight_float in (0, .08, .165, .33):
        add(f"L4_NDX_{int(round(ndx_weight_float*1000)):03d}", "momentum_split",
            {**ladder4_weight_dict, "ndx": ndx_weight_float, "mosaic": .33-ndx_weight_float},
            "Keep total momentum capital at 33%; test broad stock universe versus Nasdaq concentration at coarse, interpretable splits.")

    for portfolio_name_str in ("ladder_1_defensive", "ladder_2_balanced", "ladder_3_growth", "ladder_4_growth"):
        config_dict = yaml.safe_load((ROOT_PATH / "portfolios" / f"{portfolio_name_str}.yaml").read_text(encoding="utf-8"))
        weight_dict = {pod_dict["strategy_import_str"].split(":")[0].split(".")[-1]:
                       pod_dict["weight_float"] for pod_dict in config_dict["pods"]}
        add(portfolio_name_str.upper(), "existing_ladder", weight_dict,
            "Existing owner weight template at USD1m synthetic scale; report original no-rebalance and annual-rebalance scenarios separately, not an exact original account replay.", "reference")
    return candidate_dict_list


def main() -> None:
    spec_path = STUDY_PATH / "research_spec_frozen.json"
    if spec_path.exists():
        raise FileExistsError("A frozen specification cannot be silently replaced.")
    timestamp_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
    portfolio_list = candidate_list()
    source_manifest_path = STUDY_PATH / "source_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError("Finish the source extraction and provenance audit first.")
    source_manifest_dict = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest_dict.get("status_str") != "complete":
        raise ValueError("Source extraction must be complete before freeze.")
    benchmark_input_path = ROOT_PATH / "results/research/strategy/strategy_taa_adaptive_macro_core5/snapshot_data_qualification/2026-09-15_step2_final/direct_prices.parquet"
    marginal_test_list = []
    for candidate_dict in portfolio_list:
        family_str = candidate_dict["family"]
        candidate_id_str = candidate_dict["candidate_id"]
        if family_str == "defensive_satellite":
            comparator_id_str = "CORE_100"
        elif family_str == "taa_increment":
            comparator_id_str = "CORE_" + candidate_id_str.rsplit("_", 1)[-1].zfill(3)
        elif family_str == "ladder_anchor" and "CORE5" in candidate_id_str:
            comparator_id_str = candidate_id_str.replace("CORE5", "BIL")
        elif family_str == "momentum_split":
            comparator_id_str = "LADDER_4_GROWTH"
        else:
            continue
        marginal_test_list.append({"candidate": candidate_id_str, "comparator": comparator_id_str,
                                   "metric": "annualized arithmetic mean of daily paired excess returns",
                                   "alternative": "candidate_greater_than_comparator"})
    spec_dict = {
        "schema_version": "quant-research-spec-v1", "study_id": "portfolio_family_20260923",
        "status": "frozen_before_new_portfolio_results", "initial_frozen_at": timestamp_str,
        "research_only": True,
        "objective": "Which simple CORE5-centered and monthly-only portfolio structures have credible economic roles and stable historical tradeoffs after explicitly modeled frictions?",
        "sources": [{"path": "alpha/strategy_registry.py", "role": "eligible_universe", "complete_read": True},
                    {"path": "INTAKE_CONTRACT.md", "role": "owner_scope_and_constraints", "complete_read": True}],
        "data": {"currency": "USD", "reference_capital_usd": 1000000,
                 "source_manifest_sha256": sha256_str(source_manifest_path),
                 "source_type": "immutable_saved_standalone_native_artifacts",
                 "code_at_creation_verified": False,
                 "primary_anchor_close": "2012-10-01", "primary_last_close": "2026-07-31",
                 "all_25_anchor_close": "2019-04-04", "all_25_last_close": "2026-07-31",
                 "long_history": "pairwise exact coverage; diagnostics only; never extend an ETF before inception",
                 "membership": "native PIT for stock strategies; native observed ETF histories",
                 "adjustment": "native CAPITALSPECIAL executions, explicit dividend ledger v2; source-specific signal metadata",
                 "missing_returns": "forbidden; assert exact calendar equality; no fill or internal intersection deletion",
                 "benchmark": "independently frozen BIL and SPY native price/dividend snapshot; after same 25% dividend assumption",
                 "benchmark_input_path": str(benchmark_input_path.relative_to(ROOT_PATH)),
                 "benchmark_input_sha256": sha256_str(benchmark_input_path),
                 "benchmark_policy": "USD1m each;2007-05-30 close anchors cash; next-open initial buy; monthly residual-cash reinvest using prior-close cash/price and known fee reserve; standard2.5bps+$0.005/share min$1;25% long dividend withholding; native funding omitted and disclosed"},
        "timing": {"native_signals": "unchanged; usually Close_T to Open_T+1; flow has declared MOC schedule",
                   "allocation": "target set before applying the first session return of each calendar year",
                   "critical_boundary": "outer weights use prior-close book NAV, never current-session performance",
                   "entry": "saved ongoing-strategy unit allocation at anchor close;10bps initial unit-allocation charge; client inception orders excluded here and must use known-cash anchor in selected native replay",
                   "exit": "mark-to-market at final observed close; no invented liquidation",
                   "outer_execution_limit": "synthetic sleeve-unit rebalance, not physical open-order replay; full close-close return approximation and capital-scale effects disclosed"},
        "signal": {"formula": "native strategy rules remain fixed; only initial capital fractions and declared outer rebalance are compared"},
        "feature_roles": {"correlations": "diagnostic only, never optimizer input", "risk_contribution": "explain composition", "cost_scenarios": "survival diagnostics"},
        "portfolio": {"candidates": portfolio_list, "primary_rebalance": "annual_fixed",
                      "sensitivity_rebalance": "none_drift", "outer_added_cost_bps_per_absolute_weight_change": 10,
                      "formula": "E_i,t=E_i,t-1*(1+r_i,t); at scheduled boundary reset E_i,t-1=w_i*sum(E_i,t-1), net of declared rebalance charge; E_p,t=sum(E_i,t)",
                      "new_leverage": False, "idle_cash": "native policy; typically zero interest", "short_proceeds": "preserve source treatment"},
        "evaluation": {"selection": "mechanism-first preferred core ladder fixed before results; alternatives must justify each added component",
                       "metrics": ["CAGR", "annual_volatility", "Sharpe_zero", "Sharpe_excess_BIL", "max_drawdown", "daily_ES5", "worst_month", "worst_rolling_12m", "underwater_days", "beta_SPY", "daily_and_monthly_correlation", "turnover", "funding_cost", "gross_short_exposure"],
                       "annualization": 252, "risk_free_convention": "zero-rate Sharpe explicitly labelled; excess-BIL companion on exact shared dates",
                       "partitions": [["2012-10-02", "2016-12-30"], ["2017-01-03", "2020-12-31"], ["2021-01-04", "2026-07-31"]],
                       "crises": [["2015-08-18", "2016-02-11"], ["2018-02-02", "2018-02-09"], ["2018-10-01", "2018-12-24"], ["2020-02-20", "2020-03-23"], ["2022-01-03", "2022-10-12"], ["2025-02-03", "2025-04-30"]],
                       "long_history_crises": [["2008-09-15", "2009-03-09"], ["2011-08-01", "2011-10-04"]],
                       "rolling_window_sessions": 126, "inference_unit": "synchronized date blocks",
                       "bootstrap": {"replicates": 2000, "block_sessions": 63, "seed": 20260923,
                                     "method": "circular moving-block bootstrap; identical sampled dates for all books",
                                     "primary_scenario": "common_account/annual_fixed",
                                     "test_statistic": "mean daily paired difference; centered one-sided bootstrap p with plus-one correction",
                                     "alpha": .05, "confidence_interval": "95% percentile interval for annualized paired mean",
                                     "family_correction": "Holm across this exact 19-test family; does not correct prior source/portfolio research",
                                     "tests": marginal_test_list},
                       "holdout": "No untouched historical holdout; all chronological and bootstrap checks remain diagnostics."},
        "costs": {"native": "saved slippage, commission, native borrow and tax; never called uniform client tax",
                  "common_account": {"long_dividend_withholding": .25, "funding_rate": .05, "extra_slippage_bps": 0, "extra_short_borrow": 0},
                  "conservative": {"long_dividend_withholding": .25, "funding_rate": .08, "extra_slippage_bps": 10, "annual_short_borrow_target": .05},
                  "overlay_formula": "r_adjusted,t = r_native,t - incremental_tax_t/NAV_t-1 - funding_t/NAV_t-1 - extra_slippage*abs_traded_notional_t/NAV_t-1 - incremental_borrow_t/NAV_t-1",
                  "funding_base": "max(0, short_collateral - cash); use exact saved collateral where available, otherwise explicit 102% short-value proxy; ordinary positive cash earns no extra yield",
                  "day_count": "ACT/360 calendar interval until next known session; no fee after final mark",
                  "posting": "funding and incremental borrow charged in current close return for the known calendar interval to next session; tax on ledger ex_date (actual saved cash posting date); slippage on transaction bar",
                  "incremental_borrow": "max(0,target_rate-native_rate)*saved_native_fee/native_rate when complete exact native fee ledger exists; source's own day count/collateral retained. No second debit of native fees. Funding is separate from securities borrow.",
                  "limit": "Fixed-holdings sensitivity, not stateful re-sizing or broker-calibrated rates. Selected native replay is separately identified.",
                  "excluded": ["client management fees", "capital gains tax", "FX conversion", "borrow recalls", "auction impact and capacity calibration"]},
        "search_space": {"new_portfolio_definitions": len(portfolio_list), "rebalance_modes": 2, "cost_scenarios": 3,
                         "composition_cost_rebalance_cells": len(portfolio_list)*6,
                         "standalone_diagnostic_series": 25,
                         "full_sample_optimizer": False, "max_total_evaluation_cells": 320,
                         "prior_search": "Foundry 64 reported portfolio tests; CORE5 source nominal 441 paths plus subsequent studies; overlap unknown, counts cannot be added as independent trials.",
                         "pre_result_exposure": "Prior Foundry full report and source-rule summaries inspected during intake; memory includes older Ladder and Compass outcomes."},
        "promotion_rule": {"maximum_verdict": "forward_hypothesis",
                           "family": "Five CORE5 weights100/75/50/25/0 are preselected descriptive models, not fitted. Each adjacent reduction in CORE5 must weakly increase volatility and ES5 on full common sample and in at least2/3 partitions in common-account and conservative scenarios; failed adjacency leaves that pair unclassified. Drawdown reversals are displayed, not renamed after the result.",
                           "satellite": "versus CORE5: max drawdown must improve by at least 1 percentage point, daily ES5 by at least 5%, CAGR sacrifice at most 0.5 percentage point; direction must hold in at least 2 of 3 partitions and common/conservative costs",
                           "core_vs_bil": "Dominance requires CAGR at least0.5pp higher, ES5 no more than10% worse, and max drawdown no more than1pp worse in both cost scenarios and2/3 partitions. Otherwise report a tradeoff, not superiority, without changing weights.",
                           "momentum_split": "prefer a broad stable range; reject a precise optimal ratio claim when period/cost rankings change",
                           "portfolio_adoption": "No production changes or allocation authority; exact implementation, account suitability and genuinely prospective evidence remain separate."},
        "known_limits": ["all past strategy and portfolio development creates selection bias", "saved runs have heterogeneous vendor vintages and incomplete source-at-creation hashes", "synthetic units and outer rebalance", "whole-share capital effects", "current-vintage macro series", "native tax differences", "fixed funding/borrow rates are scenarios", "no live capacity or loss guarantee"],
        "outputs": ["REPORT.html", "REPORT.md", "REPORT_FULL.md", "decision_notebook.ipynb", "tables", "charts", "run_manifest.json", "knowledge_record.json", "tests_and_review"],
        "evidence_waivers": {"forecasting_IC": "not_applicable: fixed existing strategy composition, no new forecasting feature",
                             "same_close_timing_attribution": "not_applicable to allocation comparison; underlying execution contracts remain audited",
                             "capacity_clearance": "not_tested: current work can report order overlap and unresolved capacity, not certify client AUM"},
        "adaptive_workflow": {"profile": "standard", "max_rounds": 2, "max_total_evaluation_cells": 320,
                              "budget_override_reason": "36 predeclared compositions x2 rebalance modes x3 accounting-cost scenarios=216 cells, plus25 standalone x3=75, plus at most29 benchmark/replay cells; no optimization. Total limit320.",
                              "post_result_rule": "append an amendment and preserve all original definitions/results"},
    }
    spec_path.write_text(json.dumps(spec_dict, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"spec": str(spec_path), "sha256": sha256_str(spec_path),
                      "compositions": len(portfolio_list), "cells": len(portfolio_list)*6}))


if __name__ == "__main__":
    main()
