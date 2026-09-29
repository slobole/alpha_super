"""Independent mandate-first portfolio hypotheses; no outcome-fitted weights."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[3]
STUDY_PATH = ROOT_PATH / "results/research/client_menu_independent_20260923"
SOURCE_PATH = ROOT_PATH / "results/research/portfolio_family_20260923"


def digest_str(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def write_json(file_path: Path, payload_obj: object) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload_obj, ensure_ascii=False, indent=2, allow_nan=False, default=str)+"\n", encoding="utf-8")


def architecture_dict() -> dict:
    return {
        "S0": {"TRINITY": 1.},
        "S1": {"TRINITY": .5, "FIXED_INCOME": .5},
        "S2": {"TRINITY": .5, "CORE5": .5},
        "S3": {"TRINITY": 1/3, "FIXED_INCOME": 1/3, "CORE5": 1/3},
        "S4": {"TRINITY": .9, "VIXM": .1},
        "S5": {"TRINITY": .75, "CRISIS_TREND": .25},
        "M0": {"MOSAIC": .5, "DF_BTAL_QQQ_LINEAR": .5},
        "M1": {"NDX_VXN": .5, "DF_BTAL_QQQ_LINEAR": .5},
        "M2": {"MOSAIC": .25, "NDX_VXN": .25, "DF_BTAL_QQQ_LINEAR": .5},
        "A0": {"MOSAIC": 1/3, "HPI": 1/3, "DF_BTAL_QQQ_LINEAR": 1/3},
        "A1": {"MOSAIC": 1/3, "SECTOR_DOWNSHOCK": 1/3, "DF_BTAL_QQQ_LINEAR": 1/3},
        "A2": {"MOSAIC": .25, "HPI": .25, "DF_BTAL_QQQ_LINEAR": .25, "CRISIS_TREND": .25},
        "G0": {"MOSAIC": .5, "HPI": .5},
        "G1": {"MOSAIC": 1/3, "HPI": 1/3, "DF_BTAL_QLD": 1/3},
        "G2": {"MOSAIC": 1/3, "HPI": 1/3, "DF_BTAL_TQQQ_EQUAL": 1/3},
    }


def all_candidate_dict() -> dict:
    candidate_dict = {key_str: {"weights": weight_dict, "role": "architecture", "parent": None}
                      for key_str, weight_dict in architecture_dict().items()}
    swap_dict = {
        "M0": {"MOSAIC": ["NDX", "COMPASS"], "DF_BTAL_QQQ_LINEAR": ["DF_QQQ_LINEAR", "DF_QLD", "DF_SSO", "DF_BTAL_QLD", "DF_BTAL_TQQQ_EQUAL", "DF_BTAL_TQQQ_RANK"]},
        "A0": {"HPI": ["DV2", "QPI", "HPI_VOTE", "SECTOR_5_TREND", "SECTOR_6", "SECTOR_6_TREND"], "DF_BTAL_QQQ_LINEAR": ["CORE5", "TRINITY", "FIXED_INCOME"]},
    }
    for parent_str, slot_dict in swap_dict.items():
        for old_str, replacement_list in slot_dict.items():
            for replacement_str in replacement_list:
                weight_dict = dict(candidate_dict[parent_str]["weights"])
                weight_float = weight_dict.pop(old_str)
                weight_dict[replacement_str] = weight_float
                candidate_dict[f"{parent_str}_swap_{replacement_str}"] = {"weights": weight_dict, "role": "mechanism_challenge", "parent": parent_str}
    for overlay_str in ["VIXM", "CRISIS_TREND", "MONTH_END_FLOW", "BIL"]:
        weight_dict = {alias_str: .9*weight_float for alias_str, weight_float in architecture_dict()["A0"].items()}
        weight_dict[overlay_str] = .1
        candidate_dict[f"A0_overlay_{overlay_str}"] = {"weights": weight_dict, "role": "mechanism_challenge", "parent": "A0"}
    for parent_str, allocation_float in [("S4", .1), ("S5", .25)]:
        candidate_dict[f"{parent_str}_cash"] = {"weights": {"TRINITY": 1-allocation_float, "BIL": allocation_float}, "role": "cash_matched_control", "parent": parent_str}
    for parent_str, base_dict in architecture_dict().items():
        alias_list = list(base_dict)
        if len(alias_list) < 2:
            continue
        for sign_int in [-1, 1]:
            weight_dict = dict(base_dict)
            weight_dict[alias_list[0]] += sign_int*.1
            weight_dict[alias_list[1]] -= sign_int*.1
            if min(weight_dict.values()) < -1e-12:
                continue
            weight_dict = {alias_str: weight_float for alias_str, weight_float in weight_dict.items() if weight_float > 1e-12}
            candidate_dict[f"{parent_str}_weight_{sign_int:+d}"] = {"weights": weight_dict, "role": "weight_neighborhood", "parent": parent_str}
    for parent_str in ["M0", "A0", "G0"]:
        for alias_str, weight_float in architecture_dict()[parent_str].items():
            weight_dict = dict(architecture_dict()[parent_str])
            del weight_dict[alias_str]
            weight_dict["BIL"] = weight_float
            candidate_dict[f"{parent_str}_without_{alias_str}"] = {"weights": weight_dict, "role": "cash_ablation", "parent": parent_str}
    return candidate_dict


def freeze() -> dict:
    spec_path = STUDY_PATH / "research_spec_frozen.json"
    if spec_path.exists():
        return json.loads(spec_path.read_text(encoding="utf-8"))
    catalog_list = json.loads((SOURCE_PATH / "catalog_complete.json").read_text(encoding="utf-8"))
    input_path_list = [SOURCE_PATH / relative_str for relative_str in ["source_manifest.json", "source_audit_addendum.json", "catalog_complete.json", "benchmarks/benchmark_manifest.json"]]
    input_path_list += [ROOT_PATH / "alpha/strategy_registry.py", ROOT_PATH / "scripts/research/portfolio_family_20260923/analyze.py", Path(__file__)]
    candidate_dict = all_candidate_dict()
    spec_dict = {
        "schema_version": "quant-research-spec-v1", "study_id": "client_menu_independent_20260923",
        "status": "frozen_before_new_portfolio_results_on_previously_seen_history", "initial_frozen_at": datetime.now(timezone.utc).isoformat(), "research_only": True,
        "objective": "Can distinct simple mandate-driven portfolios earn a place in a small client menu after costs, loss-budget falsification, role ablation and weight sensitivity, without optimizing historical returns?",
        "prior_exposure": "Root read Claude report before user forbade using its ideas. Exclude that report, scripts and portfolio YAMLs from this study. Root also saw rejected CORE5-centric results. No period is an untouched holdout. Fresh independent design agent received source-only context and did not read either product study.",
        "excluded_paths": ["scripts/research/fund_menu_20260923", "results/research/portfolio/fund_product_menu_20260923", "portfolios/fund_menu_*.yaml"],
        "sources": [{"id": row_dict["alias"], "path": row_dict["strategy_import"], "role": row_dict["family"], "complete_read": "rules audited in own source catalog; source-code identity rechecked"} for row_dict in catalog_list],
        "input_files": [{"path_str": str(file_path), "sha256_str": digest_str(file_path)} for file_path in input_path_list],
        "data": {"vendor": "Norgate PIT stocks and CAPITALSPECIAL ETF accounting; explicit native dividends; FRED current-vintage where used", "as_of": "frozen saved native exports, assorted run endpoints; common endpoint 2026-07-31", "primary_anchor_close": "2012-10-01", "all25_anchor_close": "2019-04-04", "end_close": "2026-07-31", "benchmark_input_sha256": "d355567b94b60f3349f3e6eb42341ac9c7a808de552c494abcc9584256ef4160", "missing_returns": "forbidden; no filling or pre-inception proxies"},
        "timing": {"decision": "native Close_T signals, except independently lagged calendar-flow MOC rule", "entry": "native Open_T+1 or declared calendar MOC", "critical_boundary": "synthetic sleeve allocations use only prior-close weights, never current-return information", "terminal": "common marked close; refund only modeled prepaid future financing/borrow in adjusted scenarios"},
        "signal": {"baseline": "unchanged existing strategy rules; policy portfolio weights independent of expected-return estimates", "rationale": "equal capital among economic mechanisms as transparent prior; split equity-persistence capital only when explicitly testing two implementations; riskier products remove defensive allocation or test existing leveraged fallback"},
        "feature_roles": {"returns": "retrospective falsification, never optimizer input", "correlations": "diagnose common loss and duplicates, not select a historical optimum", "holdings": "same-date descriptive concentration and exposure only"},
        "portfolio": {"candidates": candidate_dict, "unit_equation": "E_t=sum_i E_i,t; each sleeve compounds independently between annual transfers", "rebalance": "annual_fixed from prior close, synthetic fund units; none_drift robustness", "outer_cost_per_abs_weight": .001, "cash": "BIL only in explicit controls; no invented yield on idle source cash", "currency": "USD; no ILS hedge", "client_scale": "no deployable capital promise; saved sources at100000USD, benchmark at1000000USD"},
        "evaluation": {"primary": "2012-10-02 to2026-07-31", "partitions": [["2012-10-02", "2017-12-29"], ["2018-01-02", "2021-12-31"], ["2022-01-03", "2026-07-31"]], "all25": "2019-04-05 to2026-07-31", "metrics": ["CAGR", "volatility", "Sharpe_zero", "Sharpe_excess_BIL", "max_drawdown", "worst_12m", "long_horizon_loss", "underwater", "ES5", "market_beta", "tail_correlation", "turnover", "subperiods"], "inference": "paired circular63day block bootstrap2000, Holm adjustment; descriptive due prior research exposure", "benchmarks": ["BIL", "SPY", "Ladder1", "Ladder4"]},
        "costs": {"native": "source's disclosed existing commissions, slippage, withholding and borrow", "central": "25% long dividend withholding and5% annual financing charged on collateral minus cash if positive, native transaction costs retained", "conservative": "central plus10bps per-side incremental slippage,8% financing, target5% short borrow; fixed native holdings", "fees": "additional1% annual management-fee sensitivity on primary architectures", "cash_policy_bridge": "remove FIXED_INCOME native positive cash interest as an explicit comparison, not its actual rule", "impact": "not calibrated; aggregate order-dollar scale diagnostics do not prove auction capacity"},
        "search_space": {"composition_count": len(candidate_dict), "candidate_ids": list(candidate_dict), "maximum_evaluation_cells": 400, "reason_for_default_override": "fixed portfolio/cost/control/neighborhood diagnostics, not400 return-optimized candidates", "adaptive_rounds": 1, "max_new_hypotheses": 5},
        "promotion_rule": {"meaning": "historical mandate fit only, not deployment or guaranteed risk limits", "S": {"max_dd": .15, "worst12m_loss": .10, "max_underwater_sessions": 756, "horizon_sessions": 756}, "M": {"max_dd": .25, "worst12m_loss": .20, "max_underwater_sessions": 1260, "horizon_sessions": 1260}, "A": {"max_dd": .25, "worst12m_loss": .20, "max_underwater_sessions": 1008, "horizon_sessions": 1260, "versus_M0": "DD improves>=2pp OR ES5 loss improves>=10%; CAGR sacrifice<=1.5pp"}, "G": {"max_dd": .40, "worst12m_loss": .30, "max_underwater_sessions": 1764, "horizon_sessions": 1764, "versus_A0": "CAGR uplift>=2pp, positive uplift>=2/3 chronological blocks", "leveraged_versus_G0": "additional CAGR uplift>=2pp, positive uplift>=2/3blocks"}, "all": "no negative rolling mandate-horizon compounded return; central and conservative pass risk gates; no capital guarantee; prefer predeclared simpler architecture before more complex alternatives; challengers diagnose mechanisms and cannot silently become new products", "hedges": "versus cash-matched control improve DD>=2pp OR ES5>=10%, with CAGR sacrifice<=1pp; otherwise decline permanent allocation", "neighborhood": "both +/-10pp transfer neighbors must retain mandate risk gates in central scenario; otherwise fragile flag blocks strong recommendation", "cash_substitute": "not claimed; investigate BIL-relative drawdown and rolling12m loss explicitly"},
        "known_limits": ["all history seen; portfolio weights are policy hypotheses, not uniquely estimated truths", "fixed-holdings cash/cost adjustments do not rerun future source sizing", "saved code identities incomplete at historic run time", "TRINITY saved holdings absent; exact security look-through unknown", "FRED vintages and hypothetical FI cash rate", "US dollar research cannot establish ILS capital stability", "same-symbol netting, borrow availability, account-specific fees and capacity not validated"],
        "outputs": ["REPORT.html", "REPORT.md", "REPORT_FULL.md", "tables", "charts", "executed decision notebook", "source manifest", "research state and hypothesis ledger", "knowledge record", "verification", "run manifest"],
        "evidence_waivers": [{"layer": "paper_replication_and_IC", "status": "not_applicable", "reason": "portfolio product design from existing sleeves, no paper or new prediction signal"}, {"layer": "untouched_holdout", "status": "unavailable", "reason": "all history previously researched; historical falsification only"}, {"layer": "full_account_execution_and_capacity", "status": "not_tested", "reason": "saved-unit product design; necessary separate gate before client allocation"}],
        "adaptive_workflow": {"profile": "standard", "target_active_minutes": 90, "hard_cap_minutes": 180, "post_result_rule": "new hypotheses require separate timestamped amendment; never overwrite this freeze"},
    }
    write_json(spec_path, spec_dict)
    write_json(STUDY_PATH / "research_state.json", {"phase": "frozen", "frozen_spec_sha256": digest_str(spec_path), "completed_cells": 0, "holdout": "none"})
    return spec_dict


if __name__ == "__main__":
    frozen_dict = freeze()
    print(json.dumps({"composition_count": len(frozen_dict["portfolio"]["candidates"]), "frozen_sha256": digest_str(STUDY_PATH / "research_spec_frozen.json")}))
