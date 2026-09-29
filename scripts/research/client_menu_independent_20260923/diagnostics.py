"""Post-result fragility checks. May veto a menu choice, never select a new one."""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.research.portfolio_family_20260923 import analyze as ledger
from scripts.research.client_menu_independent_20260923.analyze import horizon_metrics, risk_gate
from scripts.research.client_menu_independent_20260923.protocol import ROOT_PATH, SOURCE_PATH, STUDY_PATH, digest_str, write_json


def freeze_diagnostics() -> dict:
    amendment_path = STUDY_PATH/"amendment_002_fragility.json"
    if amendment_path.exists():
        return json.loads(amendment_path.read_text(encoding="utf-8"))
    spec_dict = json.loads((STUDY_PATH/"research_spec_frozen.json").read_text(encoding="utf-8"))
    selection_dict = json.loads((STUDY_PATH/"selection.json").read_text())
    selected_list = [value_str for value_str in selection_dict["policy_priority_selection"].values() if value_str]
    candidate_dict = spec_dict["portfolio"]["candidates"]
    extra_dict = {}
    for parent_str, candidate_info_dict in candidate_dict.items():
        if candidate_info_dict["role"] != "architecture":
            continue
        base_dict = candidate_info_dict["weights"]
        pair_list = list(combinations(list(base_dict), 2))
        for pair_int, (left_str, right_str) in enumerate(pair_list):
            for sign_int in [-1, 1]:
                if pair_int == 0 and parent_str not in selected_list:
                    continue
                weight_dict = dict(base_dict)
                weight_dict[left_str] += sign_int*.1
                weight_dict[right_str] -= sign_int*.1
                if min(weight_dict.values()) < -1e-12:
                    continue
                weight_dict = {alias_str: weight_float for alias_str, weight_float in weight_dict.items() if weight_float > 1e-12}
                scenario_list = (["conservative"] if pair_int == 0 else ["common_account", "conservative"]) if parent_str in selected_list else ["common_account"]
                extra_dict[f"{parent_str}_pair{pair_int}_{sign_int:+d}"] = {"parent": parent_str, "weights": weight_dict, "scenarios": scenario_list, "role": "all_pair_fragility"}
    for parent_str in selected_list:
        if parent_str in {"M0", "A0", "G0"}:
            continue  # Their cash removals already exist in the original freeze.
        for alias_str, weight_float in candidate_dict[parent_str]["weights"].items():
            weight_dict = dict(candidate_dict[parent_str]["weights"])
            del weight_dict[alias_str]
            weight_dict["BIL"] = weight_float
            extra_dict[f"{parent_str}_remove_{alias_str}"] = {"parent": parent_str, "weights": weight_dict, "scenarios": ["native", "common_account", "conservative"], "role": "cash_ablation"}
    amendment_dict = {"amended_at": datetime.now(timezone.utc).isoformat(), "result_seen_before_amendment": True, "reason": "Initial two-leg weight check leaves leveraged/macro leg unperturbed in3+pod books. Test every remaining pair, plus selected books under conservative costs, without selecting a better result. Audit source trading cadence, risk concentration, and fixed overnight-shock scenarios.", "selected_before": selected_list, "frozen_spec_sha256": digest_str(STUDY_PATH/"research_spec_frozen.json"), "selection_sha256": digest_str(STUDY_PATH/"selection.json"), "extra_candidates": extra_dict, "additional_evaluation_cells": sum(len(value_dict["scenarios"]) for value_dict in extra_dict.values()), "decision_rule": "Any selected-book neighbor failing its frozen loss/horizon gate marks the product fragile; original allocations remain unchanged. Cash ablations explain purpose, do not optimize weights. No new winner or significance claim.", "shocks": {"liquidation": {"equity": -.20, "TLT": -.15, "IEF": -.07, "LQD": -.12, "GLD": -.10, "DBC": -.20, "UUP": .05, "BIL": 0., "SHY": 0., "BTAL": -.10, "VIXM": .30}, "inflation": {"equity": -.20, "TLT": -.20, "IEF": -.10, "LQD": -.15, "GLD": .10, "DBC": .20, "UUP": .05, "BIL": 0., "SHY": -.01, "BTAL": .10, "VIXM": .10}}, "shock_convention": "Hypothetical simultaneous one-session asset returns; all individual equities use equity shock. QLD/SSO=2x,TQQQ=3x equity shock; no forecasts/probabilities, responses/trades/fees not modeled. Apply to each historical saved EOD holding state. TRINITY has no actual security panel: conservative envelope assigns its whole long market value to the worst of VTI/GLD/TLT/BIL in each scenario. This is a labeled lower P&L envelope, not measured allocation.", "funding_and_exposure": "native within-pod holdings blended with adjusted sleeve weights are descriptive, not stateful physical account simulation"}
    write_json(amendment_path, amendment_dict)
    return amendment_dict


def main() -> None:
    amendment_dict = freeze_diagnostics()
    spec_dict = json.loads((STUDY_PATH/"research_spec_frozen.json").read_text(encoding="utf-8"))
    if digest_str(STUDY_PATH/"research_spec_frozen.json") != amendment_dict["frozen_spec_sha256"]:
        raise ValueError("Original protocol changed")
    if digest_str(STUDY_PATH/"selection.json") != amendment_dict["selection_sha256"]:
        raise ValueError("Original selection changed")
    for input_dict in spec_dict["input_files"]:
        ledger.verify_file(input_dict)
    ledger.preflight_source_ids(spec_dict, json.loads((SOURCE_PATH/"source_audit_addendum.json").read_text()))
    catalog_list = json.loads((SOURCE_PATH/"catalog_complete.json").read_text(encoding="utf-8"))
    alias_id_dict = {row_dict["alias"]: row_dict["strategy_import"].split(":")[0].split(".")[-1] for row_dict in catalog_list}
    alias_id_dict.update({"BIL": "benchmark_bil", "SPY": "benchmark_spy"})
    component_dict, holdings_dict = {}, {}
    for alias_str, source_id_str in alias_id_dict.items():
        source_path = SOURCE_PATH/("benchmarks" if alias_str in {"BIL", "SPY"} else "data")/source_id_str
        component_dict[alias_str], metadata_dict = ledger.source_components(source_path)
        holdings_dict[alias_str] = ledger.read_frame(source_path/"realized_weights.csv.gz", True)
    recovery_path = STUDY_PATH/"trinity_holdings_recovery/manifest.json"
    if recovery_path.exists():
        recovery_dict = json.loads(recovery_path.read_text())
        if recovery_dict["status"] == "reconciled":
            ledger.verify_file({"path_str": recovery_dict["holdings_path"], "sha256_str": recovery_dict["holdings_sha256"]})
            if recovery_dict["max_relative_nav_error"] > 1e-8:
                raise ValueError("Trinity recovery did not reconcile")
            holdings_dict["TRINITY"] = ledger.read_frame(Path(recovery_dict["holdings_path"]), True)
    anchor_str = spec_dict["data"]["primary_anchor_close"]
    end_str = spec_dict["data"]["end_close"]
    row_list = []
    for candidate_str, candidate_info_dict in amendment_dict["extra_candidates"].items():
        weight_dict = candidate_info_dict["weights"]
        for scenario_str in candidate_info_dict["scenarios"]:
            panel_df = ledger.strict_return_panel(component_dict, list(dict.fromkeys([*weight_dict, "SPY", "BIL"])), anchor_str, end_str, scenario_str)
            nav_series, weight_df, contribution_df, cost_series = ledger.simulate_book(panel_df, weight_dict, pd.Timestamp(anchor_str), "annual_fixed", .001)
            return_series = nav_series.pct_change(fill_method=None).iloc[1:]
            metric_dict = {**ledger.metrics_dict(return_series, panel_df.SPY, panel_df.BIL), **horizon_metrics(return_series)}
            passed_bool, reason_list = risk_gate(metric_dict, spec_dict["promotion_rule"][candidate_info_dict["parent"][0]])
            row_list.append({"candidate": candidate_str, "parent": candidate_info_dict["parent"], "role": candidate_info_dict["role"], "scenario": scenario_str, "risk_gate_pass": passed_bool, "failures": ";".join(reason_list), **metric_dict})
    pd.DataFrame(row_list).to_csv(STUDY_PATH/"tables/fragility.csv", index=False)
    stress_row_list, concentration_row_list, cadence_row_list = [], [], []
    for candidate_str in amendment_dict["selected_before"]:
        weight_dict = spec_dict["portfolio"]["candidates"][candidate_str]["weights"]
        panel_df = ledger.strict_return_panel(component_dict, list(weight_dict), anchor_str, end_str, "common_account")
        nav_series, start_weight_df, contribution_df, cost_series = ledger.simulate_book(panel_df, weight_dict, pd.Timestamp(anchor_str), "annual_fixed", .001)
        eod_weight_df = start_weight_df.mul(1+panel_df)
        eod_weight_df = eod_weight_df.div(eod_weight_df.sum(axis=1), axis=0)
        asset_book_df = pd.DataFrame(index=panel_df.index)
        unknown_long_series = pd.Series(0., index=panel_df.index)
        for alias_str in weight_dict:
            holding_df = holdings_dict[alias_str]
            if holding_df.empty:
                if alias_str != "TRINITY":
                    raise ValueError("Unknown missing holdings")
                unknown_long_series += eod_weight_df[alias_str]*component_dict[alias_str].gross.loc[panel_df.index]
            else:
                selected_df = holding_df.loc[panel_df.index].drop(columns=["Cash"])
                # *** CRITICAL *** Only native sparse unheld ASSET cells ->0.
                # No price, return or calendar filling. Same-date EOD diagnostic.
                selected_df = selected_df.fillna(0.)
                selected_df = selected_df.mul(eod_weight_df[alias_str], axis=0)
                asset_book_df = asset_book_df.add(selected_df, fill_value=0.)
            source_path = SOURCE_PATH/"data"/alias_id_dict[alias_str]
            transaction_df = ledger.read_frame(source_path/"transactions.csv.gz")
            transaction_df["bar"] = pd.to_datetime(transaction_df.bar)
            transaction_df = transaction_df.loc[(transaction_df.bar > pd.Timestamp(anchor_str)) & (transaction_df.bar <= pd.Timestamp(end_str))]
            date_series = transaction_df.bar.drop_duplicates()
            month_count_series = date_series.groupby(date_series.dt.to_period("M")).size()
            ordinary_df = transaction_df.loc[transaction_df.order_id >= 0]
            ordinary_date_series = ordinary_df.bar.drop_duplicates()
            ordinary_month_count_series = ordinary_date_series.groupby(ordinary_date_series.dt.to_period("M")).size()
            cadence_row_list.append({"candidate": candidate_str, "alias": alias_str, "trade_days": len(date_series), "max_trade_days_in_month": int(month_count_series.max()), "months_more_than_one_trade_day": int((month_count_series>1).sum()), "ordinary_trade_days": len(ordinary_date_series), "max_ordinary_trade_days_in_month": int(ordinary_month_count_series.max()), "forced_exit_rows": int((transaction_df.order_id < 0).sum()), "orders_per_year": len(transaction_df)/(len(panel_df)/252)})
        long_df = asset_book_df.clip(lower=0.)
        long_total_series = long_df.sum(axis=1)
        normalized_df = long_df.div(long_total_series.replace(0., np.nan), axis=0)
        concentration_row_list.append({"candidate": candidate_str, "holdings_complete": bool(unknown_long_series.max() == 0), "unknown_long_weight_max": float(unknown_long_series.max()), "top_instrument_weight_max_known_only": float(long_df.max(axis=1).max()), "top5_instrument_weight_mean_known_only": float(np.sort(long_df.to_numpy(), axis=1)[:, -5:].sum(axis=1).mean()), "effective_instruments_mean_known_only": float((1/normalized_df.pow(2).sum(axis=1).replace(0., np.nan)).mean()), "note": "ETF instruments not underlying constituents; incomplete book names explicitly marked"})
        asset_book_df.to_csv(STUDY_PATH/f"data/{candidate_str}_known_asset_weights.csv.gz")
        for shock_str, shock_dict in amendment_dict["shocks"].items():
            shock_series = pd.Series({asset_str: shock_dict.get(asset_str, shock_dict["equity"]*{"TQQQ": 3., "QLD": 2., "SSO": 2.}.get(asset_str, 1.)) for asset_str in asset_book_df.columns})
            stress_series = asset_book_df.mul(shock_series).sum(axis=1)
            trinity_envelope_float = min(shock_dict["equity"], shock_dict["GLD"], shock_dict["TLT"], shock_dict["BIL"])
            stress_series += unknown_long_series*trinity_envelope_float
            stress_row_list.append({"candidate": candidate_str, "scenario": shock_str, "worst_historical_state_shock": float(stress_series.min()), "worst_state_date": str(stress_series.idxmin().date()), "median_state_shock": float(stress_series.median()), "trinity_conservative_envelope": bool(unknown_long_series.max()>0), "not_forecast": True})
    pd.DataFrame(stress_row_list).to_csv(STUDY_PATH/"tables/shock_scenarios.csv", index=False)
    pd.DataFrame(concentration_row_list).to_csv(STUDY_PATH/"tables/concentration.csv", index=False)
    pd.DataFrame(cadence_row_list).to_csv(STUDY_PATH/"tables/cadence.csv", index=False)
    selected_failure_df = pd.DataFrame(row_list).loc[lambda frame_df: frame_df.parent.isin(amendment_dict["selected_before"]) & (frame_df.role == "all_pair_fragility") & ~frame_df.risk_gate_pass]
    write_json(STUDY_PATH/"fragility_verdict.json", {"additional_cells": len(row_list), "selected_failures": selected_failure_df[["candidate", "scenario", "failures"]].to_dict("records"), "post_result": True, "selection_changed": False})
    print(json.dumps({"additional_cells": len(row_list), "selected_failed_neighbors": len(selected_failure_df)}))
    print(pd.DataFrame(stress_row_list).to_string(index=False))


if __name__ == "__main__":
    main()
