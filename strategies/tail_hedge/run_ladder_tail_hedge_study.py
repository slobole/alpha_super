"""Frozen LADDER allocation screen using saved Vanilla pod NAVs, not new fills.

Run as a module. The screen never modifies existing portfolio configurations.
Capital is anchored at the common close; signals/positions retain their history.
No forward filling, daily constant-weight mixing, or retrospective crisis resets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from strategies.tail_hedge.run_tail_hedge_vanilla_study import (
    CRISIS_WINDOW_DICT, crisis_tables, metric_dict, nav_return_ser, require_returns,
)

BOOK_TUPLE = (
    "ladder_1_defensive", "ladder_2_balanced", "ladder_3_growth",
    "ladder_3b_growth_2x", "ladder_3c_growth_2x_btal", "ladder_4_growth",
    "ladder_4_growth_rebalance", "ladder_4_growth_1n",
    "ladder_4_growth_1n_rebalance", "ladder_1_defensive_proxy_2008",
    "ladder_2_balanced_proxy_2008",
)
MIX_DICT = {
    "Baseline": {}, "Core05": {"Core": .05}, "Core10": {"Core": .10},
    "VIXM05": {"VIXM": .05}, "VIXM10": {"VIXM": .10},
    "Blend05": {"Core": .025, "VIXM": .025},
    "Blend10": {"Core": .05, "VIXM": .05},
    "SHY05": {"SHY": .05}, "SHY10": {"SHY": .10},
}
GATE_DICT = {
    "full_cagr_loss_max": .01, "mdd_improvement_min": .01,
    "cvar5_relative_improvement_min": .05,
    "post2020_cagr_loss_max": .01, "post2020_cvar_must_not_worsen": True,
    "market_worst1_mean_must_improve": True,
}


def exact_common_calendar(left_idx: pd.DatetimeIndex, right_idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    start_ts, end_ts = max(left_idx.min(), right_idx.min()), min(left_idx.max(), right_idx.max())
    left_window_idx = left_idx[(left_idx >= start_ts) & (left_idx <= end_ts)]
    right_window_idx = right_idx[(right_idx >= start_ts) & (right_idx <= end_ts)]
    if len(left_window_idx) < 2 or not left_window_idx.equals(right_window_idx):
        raise ValueError("Internal session mismatch: do not intersect away missing P&L.")
    return left_window_idx


def validate_saved_config(portfolio_obj, config_dict: dict) -> None:
    saved_list = [(pod_dict["pod_id_str"], pod_dict["strategy_import_str"])
                  for pod_dict in portfolio_obj.pod_info_list]
    current_list = [(pod_dict["pod_id_str"], pod_dict["strategy_import_str"])
                    for pod_dict in config_dict["pods"]]
    if (saved_list != current_list
        or not np.allclose(portfolio_obj.weights, [pod_dict["weight_float"] for pod_dict in config_dict["pods"]])
        or not np.isclose(portfolio_obj._capital_base, float(config_dict["capital_base_float"]))):
        raise ValueError("Saved/current allocation, strategy identity or capital mismatch.")


def funded_weights(weight_ser: pd.Series, import_dict: dict, capital_float: float,
                   mix_dict: dict[str, float]) -> pd.Series:
    """Preserve DV2/HPI weights when proportional cuts breach their $25k floor."""
    budget_float = sum(mix_dict.values())
    protected_list = [pod_str for pod_str in weight_ser.index
                      if ("strategies.dv2." in import_dict[pod_str]
                          or "strategies.hpi." in import_dict[pod_str])
                      and capital_float * weight_ser[pod_str] * (1-budget_float) < 25_000]
    if any(capital_float * weight_ser[pod_str] < 25_000-1e-8 for pod_str in protected_list):
        raise ValueError("Baseline already breaches its capital floor.")
    funding_list = [pod_str for pod_str in weight_ser.index if pod_str not in protected_list]
    funding_float = float(weight_ser.loc[funding_list].sum())
    if funding_float <= budget_float:
        raise ValueError("Insufficient unprotected allocation.")
    target_ser = weight_ser.copy()
    target_ser.loc[funding_list] *= (funding_float-budget_float)/funding_float
    return pd.concat([target_ser, pd.Series(mix_dict, dtype=float)])


def allocate_path(return_df: pd.DataFrame, weight_ser: pd.Series,
                  annual_bool: bool, transfer_bps_float: float = 10.0
                  ) -> tuple[pd.Series, pd.DataFrame]:
    """E_i,t = E_i,t-1*(1+r_i,t); annual reset uses only prior-close E.

    Transfer stress charges bps * sum_i |new_weight_i - drift_weight_i|.
    This is NAV reallocation, not next-open broker execution or resized shares.
    """
    require_returns(return_df.loc[:, weight_ser.index])
    if (weight_ser < 0).any() or not np.isclose(weight_ser.sum(), 1):
        raise ValueError("Weights must be nonnegative and sum to one.")
    if not np.allclose(return_df.iloc[0].loc[weight_ser.index], 0):
        raise ValueError("First common close must be a zero-return capital anchor.")
    sleeve_vec = weight_ser.to_numpy().copy()
    equity_mat = np.empty((len(return_df), len(weight_ser)))
    previous_year_int = int(return_df.index[0].year)
    for position_int, (date_ts, return_row_ser) in enumerate(return_df.loc[:, weight_ser.index].iterrows()):
        # *** CRITICAL*** Calendar-only schedule. Prior-close capital, before
        # this day's return. No return from T is used to size the reset at T.
        if annual_bool and date_ts.year != previous_year_int:
            total_float = float(sleeve_vec.sum())
            turnover_float = float(np.abs(weight_ser.to_numpy()-sleeve_vec/total_float).sum())
            sleeve_vec = weight_ser.to_numpy() * total_float * (1-transfer_bps_float/10_000*turnover_float)
        sleeve_vec *= 1 + return_row_ser.to_numpy()
        equity_mat[position_int] = sleeve_vec
        previous_year_int = int(date_ts.year)
    equity_df = pd.DataFrame(equity_mat, index=return_df.index, columns=weight_ser.index)
    total_ser = equity_df.sum(axis=1)
    return nav_return_ser(total_ser), equity_df.div(total_ser, axis=0)


def gate_pass(row_ser: pd.Series) -> bool:
    return bool(row_ser["delta_cagr"] >= -.01
                and row_ser["delta_mdd"] >= .01
                and row_ser["cvar_relative_gain"] >= .05
                and row_ser["post2020_delta_cagr"] >= -.01
                and row_ser["post2020_delta_cvar"] >= 0
                and row_ser["market_tail_delta"] > 0)


def run_screen(output_path: Path, hedge_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=False)
    input_path = output_path / "inputs"
    input_path.mkdir()
    # Freeze before loading results. Eight fixed allocations, two policies;
    # no threshold or signal tuning. Previously seen history is NOT a holdout.
    manifest_dict = {"stage": "saved_artifact_screen", "books": BOOK_TUPLE,
                     "mixes": MIX_DICT, "gates": GATE_DICT,
                     "policies": ["drift", "annual_10bps_transfer_stress"],
                     "scope": "research_only_no_live", "inputs": {},
                     "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
    hedge_df = pd.read_csv(hedge_path, index_col=0, parse_dates=True)
    manifest_dict["hedge_source"] = {"path": str(hedge_path.resolve()),
        "sha256": hashlib.sha256(hedge_path.read_bytes()).hexdigest()}
    metric_row_list, crisis_df_list, weight_row_list, subperiod_row_list = [], [], [], []
    for book_str in BOOK_TUPLE:
        config_path = Path("portfolios") / f"{book_str}.yaml"
        config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        artifact_root_path = Path("results/research/portfolio") / book_str / "vanilla_backtest"
        pickle_list = sorted(artifact_root_path.glob(f"*/{book_str}.pkl"))
        if not pickle_list:
            manifest_dict["inputs"][book_str] = {"status": "unavailable"}
            continue
        pickle_path = pickle_list[-1]
        with pickle_path.open("rb") as source_file:
            portfolio_obj = pickle.load(source_file)
        pod_name_list = [pod_dict["pod_id_str"] for pod_dict in portfolio_obj.pod_info_list]
        pod_df = portfolio_obj._daily_rets.copy()
        pod_df.columns = pod_name_list
        common_idx = exact_common_calendar(pod_df.index, hedge_df.index)
        combined_df = pd.concat([pod_df.loc[common_idx], hedge_df.loc[common_idx, ["Core", "VIXM", "SHY", "SPY"]]], axis=1)
        require_returns(combined_df)
        # *** CRITICAL*** Explicit single-row capital anchor at the common close.
        # No missing observation is filled. The first day's return is not scored.
        combined_df.iloc[0] = 0.0
        weight_ser = pd.Series(portfolio_obj.weights, index=pod_name_list)
        import_dict = {pod_dict["pod_id_str"]: pod_dict["strategy_import_str"] for pod_dict in portfolio_obj.pod_info_list}
        validate_saved_config(portfolio_obj, config_dict)
        combined_df.to_csv(input_path / f"{book_str}.csv")
        manifest_dict["inputs"][book_str] = {"path": str(pickle_path.resolve()),
            "sha256": hashlib.sha256(pickle_path.read_bytes()).hexdigest(),
            "config": config_dict, "start_anchor": str(common_idx[0].date()),
            "end": str(common_idx[-1].date()), "original_policy": portfolio_obj._rebalance}
        for annual_bool in (False, True):
            policy_str = "annual" if annual_bool else "drift"
            path_dict = {}
            for mix_str, mix_dict in MIX_DICT.items():
                target_ser = funded_weights(weight_ser, import_dict, portfolio_obj._capital_base, mix_dict)
                return_ser, drift_df = allocate_path(combined_df, target_ser, annual_bool)
                path_dict[mix_str] = return_ser
                for pod_str, target_float in target_ser.items():
                    weight_row_list.append({"book": book_str, "policy": policy_str,
                        "mix": mix_str, "pod": pod_str, "initial_weight": target_float,
                        "capital": target_float*portfolio_obj._capital_base,
                        "final_weight": drift_df[pod_str].iloc[-1]})
            path_df = pd.DataFrame(path_dict)
            path_df.to_csv(output_path / f"returns_{book_str}_{policy_str}.csv")
            scored_df = path_df.iloc[1:]
            market_ser = combined_df["SPY"].iloc[1:]
            baseline_ser = scored_df["Baseline"]
            baseline_dict = metric_dict(baseline_ser, market_ser)
            post_idx = scored_df.index[scored_df.index >= "2021-01-01"]
            post_base_dict = metric_dict(baseline_ser.loc[post_idx], market_ser.loc[post_idx])
            tail_idx = market_ser.nsmallest(max(1, int(np.ceil(len(market_ser)*.01)))).index
            for mix_str in MIX_DICT:
                metric_result_dict = metric_dict(scored_df[mix_str], market_ser)
                post_dict = metric_dict(scored_df.loc[post_idx, mix_str], market_ser.loc[post_idx])
                metric_row_list.append({"book": book_str, "policy": policy_str,
                    "mix": mix_str, **metric_result_dict,
                    "delta_cagr": metric_result_dict["cagr"]-baseline_dict["cagr"],
                    "delta_mdd": metric_result_dict["mdd"]-baseline_dict["mdd"],
                    "cvar_relative_gain": (metric_result_dict["cvar5_daily"]-baseline_dict["cvar5_daily"])/abs(baseline_dict["cvar5_daily"]),
                    "post2020_delta_cagr": post_dict["cagr"]-post_base_dict["cagr"],
                    "post2020_delta_cvar": post_dict["cvar5_daily"]-post_base_dict["cvar5_daily"],
                    "market_tail_mean": scored_df.loc[tail_idx, mix_str].mean(),
                    "market_tail_delta": (scored_df.loc[tail_idx, mix_str]-baseline_ser.loc[tail_idx]).mean(),
                    "corr_core": baseline_ser.corr(combined_df.loc[baseline_ser.index, "Core"]),
                    "corr_vixm": baseline_ser.corr(combined_df.loc[baseline_ser.index, "VIXM"])})
                for period_str, period_idx in {
                    "pre2020": scored_df.index[scored_df.index < "2020-01-01"],
                    "2020": scored_df.index[scored_df.index.year == 2020],
                    "post2020": post_idx,
                    "excluding2020": scored_df.index[scored_df.index.year != 2020],
                }.items():
                    subperiod_row_list.append({"book": book_str, "policy": policy_str,
                        "mix": mix_str, "period": period_str,
                        **metric_dict(scored_df.loc[period_idx, mix_str], market_ser.loc[period_idx])})
            crisis_df = crisis_tables(path_df, combined_df["SPY"])
            crisis_df["book"], crisis_df["policy"] = book_str, policy_str
            crisis_df_list.append(crisis_df)
        del portfolio_obj
        print(f"Screened {book_str}: {common_idx[0].date()} to {common_idx[-1].date()}", flush=True)
    metric_df = pd.DataFrame(metric_row_list)
    metric_df["gate_pass"] = metric_df.apply(gate_pass, axis=1)
    metric_df.to_csv(output_path / "metrics.csv", index=False)
    pd.concat(crisis_df_list, ignore_index=True).to_csv(output_path / "crises.csv", index=False)
    pd.DataFrame(weight_row_list).to_csv(output_path / "weights.csv", index=False)
    pd.DataFrame(subperiod_row_list).to_csv(output_path / "subperiods.csv", index=False)
    manifest_path.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
    print(metric_df.loc[metric_df["gate_pass"], ["book", "policy", "mix", "delta_cagr", "delta_mdd", "cvar_relative_gain"]].to_string(index=False))


if __name__ == "__main__":
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("--output", type=Path, default=Path("results/research/ladder_tail_hedge_20260905"))
    parser_obj.add_argument("--hedge-returns", type=Path, required=True)
    argument_obj = parser_obj.parse_args()
    run_screen(argument_obj.output, argument_obj.hedge_returns)
