"""Long-history, fixed-rule portfolio hedge diagnostics from saved Vanilla NAVs.

Research only: no strategy, registry, portfolio configuration or live changes.
All resets below redistribute NAV, not next-open cash transfers or new fills.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from strategies.tail_hedge.run_ladder_tail_hedge_study import (
    MIX_DICT, GATE_DICT, exact_common_calendar, funded_weights, gate_pass,
    validate_saved_config,
)
from strategies.tail_hedge.run_tail_hedge_vanilla_study import (
    CRISIS_WINDOW_DICT, crisis_tables, drawdown_ser, metric_dict,
    nav_return_ser, require_returns,
)

OUTPUT_PATH = Path("results/research/proxy_tail_architecture_20260905")
PREVIOUS_PATH = Path("results/research/ladder_tail_hedge_20260905_verified")
HEDGE_PATH = Path("results/research/tail_hedge_vanilla_20260904/data")
LONG_BOOK_TUPLE = ("ladder_1_defensive_proxy_2008", "ladder_2_balanced_proxy_2008", "ladder_3b_growth_2x")
BOOK_TUPLE = LONG_BOOK_TUPLE + ("ladder_1_defensive", "ladder_2_balanced", "ladder_4_growth")
POLICY_TUPLE = ("drift", "annual_all", "annual_hedge_only")
OLD_POD_DICT = {
    "NDX_2000": "strategy_mo_atr_normalized_ndx_vxn_scaled/vanilla_backtest/2026-08-24_203439/strategy_mo_atr_normalized_ndx_vxn_scaled.pkl",
    "MOSAIC_2000": "strategy_mo_mosaic_russell1000/vanilla_backtest/2026-08-15_122354/strategy_mo_mosaic_russell1000.pkl",
}


def policy_path(return_df: pd.DataFrame, weight_ser: pd.Series, policy_str: str,
                hedge_list: list[str], transfer_bps_float: float = 10.) -> tuple[pd.Series, pd.DataFrame]:
    """Independent sleeve NAV compounding, with three explicit reset policies.

    For hedge-only: h_i'=target_i; b_i'=(1-sum(h'))*b_i/sum(b).
    Transfer cost/NAV = bps/10000 * sum(abs(w_target-w_prior_close)).
    """
    selected_df = return_df.loc[:, weight_ser.index]
    require_returns(selected_df)
    if policy_str not in POLICY_TUPLE or not set(hedge_list).issubset(weight_ser.index):
        raise ValueError("Invalid policy or hedge identity.")
    if (weight_ser < 0).any() or not np.isclose(weight_ser.sum(), 1.) or not np.allclose(selected_df.iloc[0], 0.):
        raise ValueError("Invalid allocation or nonzero capital anchor.")
    target_vec = weight_ser.to_numpy()
    hedge_mask_vec = weight_ser.index.isin(hedge_list)
    base_mask_vec = ~hedge_mask_vec
    if not base_mask_vec.any():
        raise ValueError("A funded base sleeve is required.")
    sleeve_vec = target_vec.copy()
    equity_mat = np.empty(selected_df.shape)
    year_vec = selected_df.index.year.to_numpy()
    for position_int, return_vec in enumerate(selected_df.to_numpy()):
        # *** CRITICAL*** Calendar-only reset before today's return; only prior
        # close NAV enters sizing. This is not a same-open fill simulation.
        if position_int and year_vec[position_int] != year_vec[position_int-1] and policy_str != "drift":
            prior_weight_vec = sleeve_vec / sleeve_vec.sum()
            if policy_str == "annual_all":
                reset_vec = target_vec.copy()
            else:
                reset_vec = prior_weight_vec.copy()
                reset_vec[hedge_mask_vec] = target_vec[hedge_mask_vec]
                reset_vec[base_mask_vec] *= (1.-target_vec[hedge_mask_vec].sum()) / prior_weight_vec[base_mask_vec].sum()
            transfer_float = transfer_bps_float / 10_000 * np.abs(reset_vec-prior_weight_vec).sum()
            sleeve_vec = reset_vec * sleeve_vec.sum() * (1.-transfer_float)
        sleeve_vec *= 1.+return_vec
        equity_mat[position_int] = sleeve_vec
    equity_df = pd.DataFrame(equity_mat, index=return_df.index, columns=weight_ser.index)
    total_ser = equity_df.sum(axis=1)
    return nav_return_ser(total_ser), equity_df.div(total_ser, axis=0)


def longest_underwater_int(return_ser: pd.Series) -> int:
    run_int, longest_int = 0, 0
    for underwater_bool in drawdown_ser(return_ser).lt(-1e-12):
        run_int = run_int+1 if underwater_bool else 0
        longest_int = max(longest_int, run_int)
    return longest_int


def bounded_event_dict(return_ser: pd.Series, start_str: str, end_str: str) -> dict:
    if return_ser.empty or return_ser.index.min() > pd.Timestamp(start_str) or return_ser.index.max() < pd.Timestamp(end_str):
        return {"status": "unavailable"}
    event_ser = return_ser.loc[(return_ser.index > start_str) & (return_ser.index <= end_str)]
    if event_ser.empty:
        return {"status": "unavailable"}
    require_returns(event_ser.to_frame())
    return {"status": "complete", "start_close": start_str, "end_close": end_str,
        "sessions": len(event_ser), "return": float((1+event_ser).prod()-1),
        "mdd": float(drawdown_ser(event_ser).min())}


def compare_dict(candidate_ser: pd.Series, baseline_ser: pd.Series, market_ser: pd.Series) -> dict:
    candidate_dict = metric_dict(candidate_ser, market_ser)
    baseline_dict = metric_dict(baseline_ser, market_ser)
    post_idx = candidate_ser.index[candidate_ser.index >= "2021-01-01"]
    post_candidate_dict = metric_dict(candidate_ser.loc[post_idx], market_ser.loc[post_idx])
    post_base_dict = metric_dict(baseline_ser.loc[post_idx], market_ser.loc[post_idx])
    count_int = max(1, int(np.ceil(len(candidate_ser)*.01)))
    market_tail_idx = market_ser.nsmallest(count_int).index
    own_tail_idx = baseline_ser.nsmallest(count_int).index
    crisis_mask_ser = pd.Series(False, index=market_ser.index)
    for start_str, end_str in CRISIS_WINDOW_DICT.values():
        crisis_mask_ser |= (crisis_mask_ser.index > start_str) & (crisis_mask_ser.index <= end_str)
    normal_idx = market_ser.index[~crisis_mask_ser]
    relative_log_ser = np.log1p(candidate_ser)-np.log1p(baseline_ser)
    # *** CRITICAL*** Trailing diagnostics only, never inputs to allocation.
    candidate_5y_ser = np.expm1(np.log1p(candidate_ser).rolling(1260, min_periods=1260).sum()/5.)
    base_5y_ser = np.expm1(np.log1p(baseline_ser).rolling(1260, min_periods=1260).sum()/5.)
    drag_5y_ser = (candidate_5y_ser-base_5y_ser).dropna()
    result_dict = {**candidate_dict,
        "delta_cagr": candidate_dict["cagr"]-baseline_dict["cagr"],
        "delta_mdd": candidate_dict["mdd"]-baseline_dict["mdd"],
        "cvar_relative_gain": (candidate_dict["cvar5_daily"]-baseline_dict["cvar5_daily"])/abs(baseline_dict["cvar5_daily"]),
        "post2020_delta_cagr": post_candidate_dict["cagr"]-post_base_dict["cagr"],
        "post2020_delta_cvar": post_candidate_dict["cvar5_daily"]-post_base_dict["cvar5_daily"],
        "market_tail_delta": (candidate_ser.loc[market_tail_idx]-baseline_ser.loc[market_tail_idx]).mean(),
        "own_tail_delta": (candidate_ser.loc[own_tail_idx]-baseline_ser.loc[own_tail_idx]).mean(),
        "up_market_daily_delta": (candidate_ser-baseline_ser).loc[market_ser > 0].mean(),
        "noncrisis_relative_cagr": float(np.expm1(relative_log_ser.loc[normal_idx].mean()*252)),
        "rolling5y_delta_cagr_min": float(drag_5y_ser.min()),
        "rolling5y_delta_cagr_median": float(drag_5y_ser.median()),
        "rolling5y_drag_exceeds1pp_fraction": float((drag_5y_ser < -.01).mean()),
        "longest_underwater_sessions": longest_underwater_int(candidate_ser),
        "terminal_wealth_relative": float(np.exp(relative_log_ser.sum())-1),
    }
    result_dict["gate_pass"] = gate_pass(pd.Series(result_dict))
    return result_dict


def hedge_stress_df(hedge_df: pd.DataFrame) -> pd.DataFrame:
    result_df = hedge_df.copy()
    for hedge_str in ("Core", "VIXM"):
        nav_df = pd.read_csv(HEDGE_PATH / f"{hedge_str}_results.csv", index_col=0, parse_dates=True)
        turnover_ser = pd.read_csv(HEDGE_PATH / f"{hedge_str}_turnover.csv", index_col=0, parse_dates=True).iloc[:, 0]
        # *** CRITICAL*** Ex-post cost stress normalized by original prior NAV.
        # All actual transaction sides count. No signal/fill or cash rerun.
        prior_nav_ser = nav_df.total_value.shift(1)
        prior_nav_ser.iloc[0] = 100_000.
        penalty_ser = .0015*turnover_ser + .05/252*(-nav_df.cash).clip(lower=0.)/prior_nav_ser
        available_idx = hedge_df[hedge_str].dropna().index
        if not available_idx.isin(penalty_ser.index).all():
            raise ValueError("Missing cost evidence.")
        result_df.loc[available_idx, hedge_str] -= penalty_ser.loc[available_idx]
    return result_df


def run_study() -> None:
    table_path, input_path, path_path = OUTPUT_PATH / "tables_verified", OUTPUT_PATH / "inputs_verified", OUTPUT_PATH / "paths_verified"
    for directory_path in (table_path, input_path, path_path):
        directory_path.mkdir(exist_ok=True)
    previous_dict = json.loads((PREVIOUS_PATH / "manifest.json").read_text())
    hedge_df = pd.read_csv(HEDGE_PATH / "full_returns.csv", index_col=0, parse_dates=True)[["Core", "VIXM", "SHY", "SPY"]]
    stress_df = hedge_stress_df(hedge_df)
    provenance_dict = {"scope": "saved_NAV_research_only", "books": BOOK_TUPLE,
        "policies": POLICY_TUPLE, "mixes": MIX_DICT, "gates": GATE_DICT,
        "frozen_design_sha256": hashlib.sha256((OUTPUT_PATH / "FROZEN_DESIGN.md").read_bytes()).hexdigest(),
        "sources": {}, "new_strategy_parameters_tried": 0}
    metric_row_list, crisis_row_list, subperiod_row_list = [], [], []
    pod_row_list, pod_crisis_list, cohort_row_list, weight_row_list = [], [], [], []
    pod_series_dict = {}
    pod_market_dict = {}
    pod_provenance_dict = {}
    for book_str in BOOK_TUPLE:
        source_dict = previous_dict["inputs"][book_str]
        pickle_path = Path(source_dict["path"])
        source_hash_str = hashlib.sha256(pickle_path.read_bytes()).hexdigest()
        if source_hash_str != source_dict["sha256"]:
            raise ValueError("Prior screen source changed.")
        with pickle_path.open("rb") as source_file:
            portfolio_obj = pickle.load(source_file)
        config_path = Path("portfolios") / f"{book_str}.yaml"
        config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate_saved_config(portfolio_obj, config_dict)
        capital_float = float(config_dict["capital_base_float"])
        provenance_dict["sources"][book_str] = {"path": str(pickle_path.resolve()), "sha256": source_hash_str, "config": config_dict}
        pod_df = portfolio_obj._daily_rets.copy()
        pod_df.columns = [pod_dict["pod_id_str"] for pod_dict in portfolio_obj.pod_info_list]
        weight_ser = pd.Series(portfolio_obj.weights, index=pod_df.columns)
        import_dict = {pod_dict["pod_id_str"]: pod_dict["strategy_import_str"] for pod_dict in portfolio_obj.pod_info_list}
        for pod_dict, strategy_obj in zip(portfolio_obj.pod_info_list, portfolio_obj.strategies):
            identity_str = pod_dict["strategy_import_str"]
            if identity_str not in pod_series_dict:
                pod_series_dict[identity_str] = nav_return_ser(strategy_obj.results.total_value, strategy_obj._capital_base)
                # Embedded POD benchmark labels differ. Use the same observed
                # SPY total-return reference as the portfolio/hedge comparison.
                pod_market_dict[identity_str] = hedge_df.loc[strategy_obj.results.index, "SPY"]
                pod_provenance_dict[identity_str] = {"representative_source_book": book_str,
                    "original_capital": strategy_obj._capital_base,
                    "selection": "first_pinned_book_instance_not_longest_or_best", "market": "SPY_TOTALRETURN"}
        del portfolio_obj
        panel_tuple = ("long_core", "common") if book_str in LONG_BOOK_TUPLE else ("common",)
        for panel_str in panel_tuple:
            hedge_list = ["Core", "SHY", "SPY"] if panel_str == "long_core" else list(hedge_df.columns)
            available_df = hedge_df[hedge_list].dropna(how="all")
            if panel_str == "common":
                available_df = available_df.loc[hedge_df.VIXM.first_valid_index():]
            common_idx = exact_common_calendar(pod_df.index, available_df.index)
            combined_df = pd.concat([pod_df.loc[common_idx], available_df.loc[common_idx]], axis=1)
            require_returns(combined_df)
            combined_df.iloc[0] = 0.  # Explicit common-close anchor; no P&L filled.
            combined_df.to_csv(input_path / f"{book_str}_{panel_str}.csv")
            mix_name_list = list(MIX_DICT) if panel_str == "common" else ["Baseline", "Core05", "Core10", "SHY05", "SHY10"]
            for cost_str in ("vanilla", "stress25bps_funding5pct"):
                scenario_df = combined_df.copy()
                if cost_str != "vanilla":
                    for hedge_str in set(hedge_list) & {"Core", "VIXM"}:
                        scenario_df[hedge_str] = stress_df.loc[common_idx, hedge_str]
                    scenario_df.iloc[0] = 0.
                for policy_str in POLICY_TUPLE:
                    path_dict = {}
                    for mix_str in mix_name_list:
                        target_ser = funded_weights(weight_ser, import_dict, capital_float, MIX_DICT[mix_str])
                        return_ser, drift_df = policy_path(scenario_df, target_ser, policy_str, list(MIX_DICT[mix_str]))
                        path_dict[mix_str] = return_ser
                        if cost_str == "vanilla":
                            hedge_weight_ser = drift_df.loc[:, list(MIX_DICT[mix_str])].sum(axis=1)
                            weight_row_list.append({"book": book_str, "panel": panel_str, "policy": policy_str,
                                "mix": mix_str, "initial": sum(MIX_DICT[mix_str].values()),
                                "mean": hedge_weight_ser.mean(), "final": hedge_weight_ser.iloc[-1],
                                "minimum": hedge_weight_ser.min(), "maximum": hedge_weight_ser.max()})
                    path_df = pd.DataFrame(path_dict)
                    scored_df = path_df.iloc[1:]
                    market_ser = scenario_df.SPY.iloc[1:]
                    identity_dict = {"book": book_str, "panel": panel_str, "policy": policy_str, "cost": cost_str}
                    for mix_str in path_df:
                        metric_row_list.append({**identity_dict, "mix": mix_str,
                            **compare_dict(scored_df[mix_str], scored_df.Baseline, market_ser)})
                        if cost_str == "vanilla":
                            for period_str, period_mask_vec in {
                                "pre2020": scored_df.index < "2020-01-01",
                                "2020": scored_df.index.year == 2020,
                                "post2020": scored_df.index >= "2021-01-01",
                                "excluding2008and2020": ~scored_df.index.year.isin([2008, 2020]),
                            }.items():
                                subperiod_row_list.append({**identity_dict, "mix": mix_str, "period": period_str,
                                    **metric_dict(scored_df.loc[period_mask_vec, mix_str], market_ser.loc[period_mask_vec])})
                    if cost_str == "vanilla":
                        path_df.assign(SPY=scenario_df.SPY).to_csv(path_path / f"{book_str}_{panel_str}_{policy_str}.csv")
                        event_df = crisis_tables(path_df.assign(SPY=scenario_df.SPY), scenario_df.SPY)
                        event_df = event_df.assign(**identity_dict)
                        crisis_row_list.extend(event_df.to_dict("records"))
                        if panel_str == "long_core":
                            for mix_str in list(path_df)+["SPY"]:
                                event_ser = scenario_df.SPY if mix_str == "SPY" else path_df[mix_str]
                                crisis_row_list.append({**identity_dict, "crisis": "2008_GFC_partial_March_anchor",
                                    "series": mix_str, "partial_GFC_label": True,
                                    **bounded_event_dict(event_ser, "2008-03-03", "2009-03-09")})
            # Five-year year-start cohorts, fixed base states; not a new signal fit.
            for year_int in range(common_idx[0].year+1, common_idx[-1].year-4):
                cohort_df = combined_df.loc[(combined_df.index.year >= year_int) & (combined_df.index.year < year_int+5)].copy()
                if len(cohort_df) < 1200:
                    continue
                cohort_df.iloc[0] = 0.
                baseline_ser, _ = policy_path(cohort_df, weight_ser, "annual_hedge_only", [])
                baseline_dict = metric_dict(baseline_ser.iloc[1:], cohort_df.SPY.iloc[1:])
                for mix_str in mix_name_list[1:]:
                    target_ser = funded_weights(weight_ser, import_dict, capital_float, MIX_DICT[mix_str])
                    cohort_ser, _ = policy_path(cohort_df, target_ser, "annual_hedge_only", list(MIX_DICT[mix_str]))
                    cohort_dict = metric_dict(cohort_ser.iloc[1:], cohort_df.SPY.iloc[1:])
                    cohort_row_list.append({"book": book_str, "panel": panel_str, "mix": mix_str, "year": year_int,
                        **cohort_dict, "delta_cagr": cohort_dict["cagr"]-baseline_dict["cagr"],
                        "delta_mdd": cohort_dict["mdd"]-baseline_dict["mdd"]})
            print(f"Completed {book_str} {panel_str}: {common_idx[0].date()}..{common_idx[-1].date()}", flush=True)
    for name_str, relative_str in OLD_POD_DICT.items():
        pickle_path = Path("results/research/strategy") / relative_str
        with pickle_path.open("rb") as source_file:
            strategy_obj = pickle.load(source_file)
        pod_series_dict[name_str] = nav_return_ser(strategy_obj.results.total_value, strategy_obj._capital_base)
        pod_market_dict[name_str] = nav_return_ser(strategy_obj.results["$SPX"], strategy_obj._capital_base)
        pod_provenance_dict[name_str] = {"representative_source_book": "standalone",
            "original_capital": strategy_obj._capital_base, "selection": "pinned_oldest_saved_history", "market": "SPXTR"}
        provenance_dict["sources"][name_str] = {"path": str(pickle_path.resolve()), "sha256": hashlib.sha256(pickle_path.read_bytes()).hexdigest()}
        del strategy_obj
    for pod_str, return_ser in pod_series_dict.items():
        market_ser = pod_market_dict[pod_str]
        file_str = pod_str.split(":")[0].split(".")[-1]
        pd.DataFrame({"pod": return_ser, "market": market_ser}).to_csv(input_path / f"pod_{file_str}.csv")
        pod_row_list.append({"pod": pod_str, "panel": "full_saved", **pod_provenance_dict[pod_str], **metric_dict(return_ser, market_ser)})
        for hedge_str in ("Core", "VIXM"):
            hedge_ser = hedge_df[hedge_str].loc[hedge_df[hedge_str].first_valid_index():]
            overlap_idx = exact_common_calendar(return_ser.index, hedge_ser.index)
            pod_row_list.append({"pod": pod_str, "panel": f"overlap_{hedge_str}",
                **pod_provenance_dict[pod_str], "market": "SPY_TOTALRETURN",
                **metric_dict(return_ser.loc[overlap_idx], hedge_df.loc[overlap_idx, "SPY"]),
                "hedge_correlation": return_ser.loc[overlap_idx].corr(hedge_ser.loc[overlap_idx])})
        event_df = crisis_tables(pd.DataFrame({"pod": return_ser}), market_ser).assign(pod=pod_str)
        pod_crisis_list.extend(event_df.to_dict("records"))
        if return_ser.index.min() <= pd.Timestamp("2000-03-24"):
            pod_crisis_list.append({"pod": pod_str, "crisis": "2000_Dotcom", "status": "complete",
                **bounded_event_dict(return_ser, "2000-03-24", "2002-10-09")})
    for name_str, row_list in {"metrics": metric_row_list, "crises": crisis_row_list,
        "subperiods": subperiod_row_list, "pod_metrics": pod_row_list, "pod_crises": pod_crisis_list,
        "cohorts5y": cohort_row_list, "hedge_budget": weight_row_list}.items():
        pd.DataFrame(row_list).to_csv(table_path / f"{name_str}.csv", index=False)
    provenance_dict["allocation_cells_including_stress_and_baselines"] = len(metric_row_list)
    provenance_dict["nonbaseline_allocation_cells"] = sum(row_dict["mix"] != "Baseline" for row_dict in metric_row_list)
    provenance_dict["cohort_cells"] = len(cohort_row_list)
    cost_path_list = [HEDGE_PATH / f"{hedge_str}_{file_str}.csv"
        for hedge_str in ("Core", "VIXM") for file_str in ("results", "turnover")]
    helper_path_list = [Path(__file__).with_name(file_str) for file_str in
        ("run_ladder_tail_hedge_study.py", "run_tail_hedge_vanilla_study.py")]
    for file_path in list(input_path.glob("*.csv"))+[HEDGE_PATH / "full_returns.csv", Path(__file__)]+cost_path_list+helper_path_list:
        provenance_dict["sources"][str(file_path)] = {"sha256": hashlib.sha256(file_path.read_bytes()).hexdigest()}
    (OUTPUT_PATH / "manifest.json").write_text(json.dumps(provenance_dict, indent=2), encoding="utf-8")
    print("Study complete", provenance_dict["nonbaseline_allocation_cells"], flush=True)


if __name__ == "__main__":
    run_study()
