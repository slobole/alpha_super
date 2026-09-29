"""Evaluate frozen mandate hypotheses using independently compounded sleeve units.

Research only. Imports only calculation plumbing from the earlier owned study;
its portfolio definitions and conclusions are never called.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.research.portfolio_family_20260923 import analyze as ledger
from scripts.research.client_menu_independent_20260923.protocol import ROOT_PATH, STUDY_PATH, SOURCE_PATH, digest_str, write_json


def horizon_metrics(return_series: pd.Series) -> dict:
    result_dict = {}
    for year_int in [3, 5, 7]:
        # *** CRITICAL *** Trailing realized returns for diagnosis only. No
        # future observation selects a position or changes a policy weight.
        horizon_series = np.expm1(np.log1p(return_series).rolling(year_int*252, min_periods=year_int*252).sum()).dropna()
        result_dict[f"worst_{year_int}y"] = float(horizon_series.min()) if len(horizon_series) else np.nan
    relative_nav_vec = np.r_[1., np.cumprod(1+return_series.to_numpy())]
    drawdown_vec = relative_nav_vec/np.maximum.accumulate(relative_nav_vec)-1
    last_peak_int = np.flatnonzero(drawdown_vec >= -1e-12)[-1]
    result_dict["terminal_underwater_sessions"] = len(drawdown_vec)-1-int(last_peak_int)
    result_dict["terminal_recovery_censored"] = bool(drawdown_vec[-1] < -1e-12)
    return result_dict


def risk_gate(metric_dict: dict, mandate_dict: dict) -> tuple[bool, list[str]]:
    reason_list = []
    comparison_list = [
        (metric_dict["max_drawdown"] >= -mandate_dict["max_dd"], "drawdown"),
        (metric_dict["worst_rolling252"] >= -mandate_dict["worst12m_loss"], "worst12m"),
        (metric_dict["max_underwater_sessions"] <= mandate_dict["max_underwater_sessions"], "underwater"),
        (metric_dict[f"worst_{mandate_dict['horizon_sessions']//252}y"] >= 0, "long_horizon_loss"),
    ]
    for passed_bool, label_str in comparison_list:
        if not passed_bool:
            reason_list.append(label_str)
    return not reason_list, reason_list


def main() -> None:
    spec_path = STUDY_PATH / "research_spec_frozen.json"
    spec_dict = json.loads(spec_path.read_text(encoding="utf-8"))
    for input_dict in spec_dict["input_files"]:
        ledger.verify_file(input_dict)
    addendum_dict = json.loads((SOURCE_PATH/"source_audit_addendum.json").read_text())
    ledger.preflight_source_ids(spec_dict, addendum_dict)
    catalog_list = json.loads((SOURCE_PATH/"catalog_complete.json").read_text(encoding="utf-8"))
    alias_id_dict = {row_dict["alias"]: row_dict["strategy_import"].split(":")[0].split(".")[-1] for row_dict in catalog_list}
    alias_id_dict.update({"BIL": "benchmark_bil", "SPY": "benchmark_spy"})
    component_dict, metadata_dict = {}, {}
    source_record_list = []
    for alias_str, source_id_str in alias_id_dict.items():
        folder_path = SOURCE_PATH / ("benchmarks" if alias_str in {"SPY", "BIL"} else "data") / source_id_str
        component_dict[alias_str], metadata_dict[alias_str] = ledger.source_components(folder_path)
        source_record_list.append({"alias": alias_str, "source_id": source_id_str, "metadata_path": str(folder_path/"source_metadata.json"), "metadata_sha256": digest_str(folder_path/"source_metadata.json"), "start": str(component_dict[alias_str].index[0].date()), "end": str(component_dict[alias_str].index[-1].date()), "capital": metadata_dict[alias_str]["native_capital_float"]})
    candidate_dict = spec_dict["portfolio"]["candidates"]
    control_dict = {"BIL": {"BIL": 1.}, "SPY": {"SPY": 1.}, "Ladder1": {"DF_BTAL_QQQ_LINEAR": .55, "SECTOR_DOWNSHOCK": .45}, "Ladder4": {"DV2": .16, "HPI_VOTE": .17, "NDX_VXN": .25, "MOSAIC": .08, "DF_BTAL_TQQQ_RANK": .34}}
    end_str = spec_dict["data"]["end_close"]
    metric_row_list, period_row_list, exposure_row_list, contribution_row_list = [], [], [], []
    return_series_dict, case_return_dict, gate_row_list = {}, {}, []
    cell_list = []
    (STUDY_PATH/"tables").mkdir(parents=True, exist_ok=True)
    (STUDY_PATH/"data").mkdir(parents=True, exist_ok=True)

    def evaluate_case(candidate_str: str, weight_dict: dict, scenario_str: str, window_str: str, role_str: str, rebalance_str: str = "annual_fixed") -> dict:
        anchor_str = spec_dict["data"]["primary_anchor_close" if window_str == "primary" else "all25_anchor_close"]
        if window_str == "crisis2008":
            anchor_str = "2007-12-31"
        scenario_base_str = "common_account" if scenario_str in {"fee1pct", "no_fi_cash_interest"} else scenario_str
        panel_df = ledger.strict_return_panel(component_dict, list(dict.fromkeys([*weight_dict, "SPY", "BIL"])), anchor_str, end_str, scenario_base_str)
        if scenario_str == "no_fi_cash_interest":
            interest_df = ledger.read_frame(SOURCE_PATH/"data"/alias_id_dict["FIXED_INCOME"]/"cash_interest.csv.gz")
            native_df = ledger.read_frame(SOURCE_PATH/"data"/alias_id_dict["FIXED_INCOME"]/"nav.csv.gz", True)
            date_str = "accrual_start_date_ts" if "accrual_start_date_ts" in interest_df else next(column_str for column_str in interest_df if "date" in column_str)
            amount_str = next(column_str for column_str in interest_df if "interest" in column_str and "rate" not in column_str)
            interest_series = ledger.event_totals(interest_df, date_str, interest_df[amount_str], native_df.index)
            # *** CRITICAL *** Remove only interest actually embedded at each
            # source date, divided by that day's PRIOR NAV. Fixed-holdings bridge.
            interest_return_series = interest_series/native_df.total_value.shift(1)
            panel_df["FIXED_INCOME"] -= interest_return_series.loc[panel_df.index]
        nav_series, start_weight_df, contribution_df, outer_cost_series = ledger.simulate_book(panel_df, weight_dict, pd.Timestamp(anchor_str), rebalance_str, .001)
        # *** CRITICAL *** Consecutive marked NAV return, never padded; initial
        # observed close retained so allocation costs enter the first return.
        return_series = nav_series.pct_change(fill_method=None).iloc[1:]
        if scenario_str == "fee1pct":
            return_series = (1+return_series)*math.exp(-.01/252)-1
        metric_dict = {**ledger.metrics_dict(return_series, panel_df.SPY, panel_df.BIL), **horizon_metrics(return_series)}
        metric_dict.update({"candidate": candidate_str, "scenario": scenario_str, "window": window_str, "role": role_str, "rebalance": rebalance_str})
        metric_row_list.append(metric_dict)
        key_str = f"{candidate_str}|{scenario_str}|{window_str}|{rebalance_str}"
        case_return_dict[key_str] = return_series
        cell_list.append({"candidate": candidate_str, "scenario": scenario_str, "window": window_str, "role": role_str, "rebalance": rebalance_str})
        if window_str == "primary" and scenario_str == "common_account" and rebalance_str == "annual_fixed" and role_str in {"architecture", "control"}:
            return_series_dict[candidate_str] = return_series
            for partition_int, (start_str, stop_str) in enumerate(spec_dict["evaluation"]["partitions"]):
                partition_series = return_series.loc[start_str:stop_str]
                period_row_list.append({"candidate": candidate_str, "partition": partition_int+1, **ledger.metrics_dict(partition_series, panel_df.SPY.loc[partition_series.index], panel_df.BIL.loc[partition_series.index])})
            for event_str, start_str, stop_str in [("COVID", "2020-02-19", "2020-03-23"), ("Inflation2022", "2022-01-03", "2022-12-30"), ("Q42018", "2018-10-01", "2018-12-31")]:
                event_series = return_series.loc[start_str:stop_str]
                period_row_list.append({"candidate": candidate_str, "partition": event_str, **ledger.metrics_dict(event_series)})
            # EOD source exposures are DESCRIPTIVE, never same-day decisions.
            eod_weight_df = start_weight_df.mul(1+panel_df[list(weight_dict)])
            eod_weight_df = eod_weight_df.div(eod_weight_df.sum(axis=1), axis=0)
            exposure_df = pd.DataFrame(index=return_series.index)
            for field_str in ["gross", "short", "embedded_equity_leverage_extra", "cash_weight", "funding_base_weight"]:
                field_df = pd.concat({alias_str: component_dict[alias_str][field_str].loc[return_series.index] for alias_str in weight_dict}, axis=1)
                exposure_df[field_str] = (eod_weight_df*field_df).sum(axis=1)
            turnover_df = pd.concat({alias_str: component_dict[alias_str].turnover.loc[return_series.index] for alias_str in weight_dict}, axis=1)
            exposure_row_list.append({"candidate": candidate_str, **{f"{field_str}_{stat_str}": float(getattr(exposure_df[field_str], stat_str)()) for field_str in exposure_df for stat_str in ["mean", "max"]}, "annual_oneway_turnover": float((start_weight_df*turnover_df).sum(axis=1).mean()*252), "outer_annual_cost": float(outer_cost_series.mean()*252), "max_sleeve_weight": float(eod_weight_df.max().max())})
            tail_mask = return_series <= return_series.quantile(.05)
            for alias_str in weight_dict:
                contribution_row_list.append({"candidate": candidate_str, "alias": alias_str, "starting_policy_weight": weight_dict[alias_str], "mean_drifted_weight": float(start_weight_df[alias_str].mean()), "variance_share": float(contribution_df[alias_str].cov(return_series)/return_series.var()), "worst5pct_loss_share": float(contribution_df.loc[tail_mask, alias_str].sum()/return_series.loc[tail_mask].sum())})
        return metric_dict

    for candidate_str, candidate_info_dict in candidate_dict.items():
        role_str = candidate_info_dict["role"]
        window_str = "all25" if role_str == "mechanism_challenge" else "primary"
        scenario_list = ["native", "common_account", "conservative"] if role_str in {"architecture", "mechanism_challenge", "cash_matched_control"} else ["common_account"]
        for scenario_str in scenario_list:
            evaluate_case(candidate_str, candidate_info_dict["weights"], scenario_str, window_str, role_str)
        if role_str == "architecture":
            evaluate_case(candidate_str, candidate_info_dict["weights"], "common_account", "all25", role_str)
            evaluate_case(candidate_str, candidate_info_dict["weights"], "common_account", "primary", "rebalance_sensitivity", "none_drift")
            evaluate_case(candidate_str, candidate_info_dict["weights"], "fee1pct", "primary", "management_fee")
    for candidate_str, weight_dict in control_dict.items():
        for scenario_str in ["native", "common_account", "conservative"]:
            evaluate_case(candidate_str, weight_dict, scenario_str, "primary", "control")
    for candidate_str in ["S1", "S3"]:
        evaluate_case(candidate_str, candidate_dict[candidate_str]["weights"], "no_fi_cash_interest", "primary", "cash_policy_bridge")
    # Pre-result amendment adds available real2008 histories; no invented proxies.
    for candidate_str in ["S0", "S1", "S2", "S3", "S5", "G0"]:
        for scenario_str in ["common_account", "conservative"]:
            evaluate_case(candidate_str, candidate_dict[candidate_str]["weights"], scenario_str, "crisis2008", "older_real_history")
    all25_panel_df = ledger.strict_return_panel(component_dict, list(alias_id_dict), spec_dict["data"]["all25_anchor_close"], end_str, "common_account")
    all25_panel_df.to_csv(STUDY_PATH/"data/all25_returns.csv.gz")
    pd.DataFrame(return_series_dict).to_csv(STUDY_PATH/"data/architecture_returns.csv.gz")
    standalone_list = [{"alias": alias_str, **ledger.metrics_dict(all25_panel_df[alias_str], all25_panel_df.SPY, all25_panel_df.BIL), **horizon_metrics(all25_panel_df[alias_str])} for alias_str in alias_id_dict]
    pd.DataFrame(standalone_list).to_csv(STUDY_PATH/"tables/universe.csv", index=False)
    all25_panel_df.corr().to_csv(STUDY_PATH/"tables/correlation.csv")
    all25_panel_df.loc[all25_panel_df.SPY <= all25_panel_df.SPY.quantile(.05)].corr().to_csv(STUDY_PATH/"tables/tail_correlation.csv")
    metric_df, period_df = pd.DataFrame(metric_row_list), pd.DataFrame(period_row_list)
    metric_df.to_csv(STUDY_PATH/"tables/results.csv", index=False)
    period_df.to_csv(STUDY_PATH/"tables/periods.csv", index=False)
    pd.DataFrame(exposure_row_list).to_csv(STUDY_PATH/"tables/exposures.csv", index=False)
    pd.DataFrame(contribution_row_list).to_csv(STUDY_PATH/"tables/risk_contributions.csv", index=False)

    def lookup(candidate_str: str, scenario_str: str = "common_account", window_str: str = "primary") -> dict:
        selected_df = metric_df.loc[(metric_df.candidate == candidate_str)&(metric_df.scenario == scenario_str)&(metric_df.window == window_str)&(metric_df.rebalance == "annual_fixed")]
        if len(selected_df) != 1:
            raise ValueError(f"Ambiguous metric lookup {candidate_str} {scenario_str} {window_str}")
        return selected_df.iloc[0].to_dict()

    for candidate_str, candidate_info_dict in candidate_dict.items():
        if candidate_info_dict["role"] != "architecture":
            continue
        mandate_str = candidate_str[0]
        failure_list = []
        for scenario_str in ["common_account", "conservative"]:
            passed_bool, reason_list = risk_gate(lookup(candidate_str, scenario_str), spec_dict["promotion_rule"][mandate_str])
            failure_list.extend(f"{scenario_str}:{reason_str}" for reason_str in reason_list)
            if candidate_str in {"S0", "S1", "S2", "S3", "S5", "G0"}:
                passed_bool, reason_list = risk_gate(lookup(candidate_str, scenario_str, "crisis2008"), spec_dict["promotion_rule"][mandate_str])
                failure_list.extend(f"older_history_{scenario_str}:{reason_str}" for reason_str in reason_list)
        for neighbor_str, neighbor_dict in candidate_dict.items():
            if neighbor_dict["role"] == "weight_neighborhood" and neighbor_dict["parent"] == candidate_str:
                passed_bool, reason_list = risk_gate(lookup(neighbor_str), spec_dict["promotion_rule"][mandate_str])
                failure_list.extend(f"{neighbor_str}:{reason_str}" for reason_str in reason_list)
        central_dict = lookup(candidate_str)
        if mandate_str == "A":
            base_dict = lookup("M0")
            if not (central_dict["max_drawdown"]-base_dict["max_drawdown"] >= .02 or central_dict["es5_loss"] <= .9*base_dict["es5_loss"]):
                failure_list.append("activity_does_not_earn_downside_improvement")
            if central_dict["cagr"] < base_dict["cagr"]-.015:
                failure_list.append("activity_return_sacrifice")
        if mandate_str == "G":
            for comparator_str in (["A0"] if candidate_str == "G0" else ["A0", "G0"]):
                if central_dict["cagr"] < lookup(comparator_str)["cagr"]+.02:
                    failure_list.append(f"less_than_2pp_uplift_vs_{comparator_str}")
                comparison_df = period_df.loc[period_df.partition.isin([1, 2, 3]) & period_df.candidate.isin([candidate_str, comparator_str])].pivot(index="partition", columns="candidate", values="cagr")
                if (comparison_df[candidate_str] > comparison_df[comparator_str]).sum() < 2:
                    failure_list.append(f"uplift_not_in_two_blocks_vs_{comparator_str}")
        if candidate_str in {"S4", "S5"}:
            cash_dict = lookup(f"{candidate_str}_cash")
            if not (central_dict["max_drawdown"]-cash_dict["max_drawdown"] >= .02 or central_dict["es5_loss"] <= .9*cash_dict["es5_loss"]):
                failure_list.append("hedge_no_cash_matched_improvement")
            if central_dict["cagr"] < cash_dict["cagr"]-.01:
                failure_list.append("hedge_excessive_carry")
        gate_row_list.append({"candidate": candidate_str, "historical_fit": not failure_list, "failures": ";".join(failure_list)})
    gate_df = pd.DataFrame(gate_row_list)
    gate_df.to_csv(STUDY_PATH/"tables/mandate_gates.csv", index=False)
    selected_dict = {mandate_str: next((row_dict["candidate"] for row_dict in gate_row_list if row_dict["candidate"].startswith(mandate_str) and row_dict["historical_fit"]), None) for mandate_str in "SMAG"}
    inference_list = [{"candidate": "M0", "comparator": "BIL"}, {"candidate": "A0", "comparator": "M0"}, {"candidate": "G0", "comparator": "A0"}, {"candidate": "M1", "comparator": "M0"}, {"candidate": "M2", "comparator": "M0"}, {"candidate": "S2", "comparator": "S0"}]
    inference_df = ledger.paired_bootstrap(pd.DataFrame(return_series_dict), inference_list, 20260923, 2000, 63)
    inference_df.to_csv(STUDY_PATH/"tables/inference.csv", index=False)
    ledger_row_list = []
    for cell_int, cell_dict in enumerate(cell_list):
        ledger_row_list.append({"experiment_id": f"E{cell_int+1:04d}", "study_id": spec_dict["study_id"], "recorded_at": datetime.now(timezone.utc).isoformat(), "phase": "diagnosis", "hypothesis_ids": [cell_dict["candidate"]], "data_periods_seen": [cell_dict["window"]], "declared_variant_count": 1, "selection_role": cell_dict["role"], "status": "complete", "spec_content_id": digest_str(spec_path), "code_content_id": digest_str(Path(__file__)), "evidence_paths": ["tables/results.csv"], **cell_dict})
    if len(cell_list) + len(standalone_list) > spec_dict["search_space"]["maximum_evaluation_cells"]:
        raise ValueError("Frozen evaluation budget exceeded")
    (STUDY_PATH/"experiment_ledger.jsonl").write_text("".join(json.dumps(row_dict)+"\n" for row_dict in ledger_row_list), encoding="utf-8")
    write_json(STUDY_PATH/"input_sources.json", source_record_list)
    write_json(STUDY_PATH/"selection.json", {"policy_priority_selection": selected_dict, "interpretation": "historical policy fit; old-history and implementation limits can still weaken or block recommendation", "architecture_count": 15, "composition_count": len(candidate_dict), "portfolio_evaluation_cells": len(cell_list), "standalone_diagnostic_cells": len(standalone_list), "frozen_sha256": digest_str(spec_path)})
    write_json(STUDY_PATH/"research_state.json", {"phase": "diagnosis", "completed_portfolio_cells": len(cell_list), "source_diagnostics": len(standalone_list), "holdout": "none", "frozen_spec_sha256": digest_str(spec_path)})
    print(json.dumps({"selected": selected_dict, "cells": len(cell_list), "source_diagnostics": len(standalone_list)}))
    print(metric_df.loc[(metric_df.role == "architecture")&(metric_df.window == "primary")&(metric_df.scenario == "common_account"), ["candidate", "cagr", "volatility", "max_drawdown", "worst_rolling252", "max_underwater_sessions"]].to_string(index=False))


if __name__ == "__main__":
    main()
