"""Reproducible report tables and four figures from the frozen hedge study."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

from strategies.tail_hedge.run_ladder_tail_hedge_study import (
    BOOK_TUPLE, MIX_DICT, exact_common_calendar, allocate_path, gate_pass,
)
from strategies.tail_hedge.run_tail_hedge_vanilla_study import (
    CRISIS_WINDOW_DICT, crisis_tables, drawdown_ser, metric_dict, require_returns,
)


def dividend_tax_penalty(ledger_df: pd.DataFrame, nav_ser: pd.Series,
                         capital_float: float, tax_rate_float: float) -> pd.Series:
    # *** CRITICAL*** Approximate return penalty on engine-modeled ex-date,
    # not a broker payment date or an exact fixed-share cash-ledger replay.
    # Zero means no dividend event, never a filled missing market return.
    tax_ledger_df = ledger_df.copy()
    tax_ledger_df["ex_date"] = pd.to_datetime(tax_ledger_df.ex_date)
    tax_ledger_df["tax_debit_float"] = tax_rate_float*tax_ledger_df.gross_dividend_cash_float.clip(lower=0.)
    debit_ser = tax_ledger_df.groupby("ex_date").tax_debit_float.sum().reindex(nav_ser.index, fill_value=0.)
    prior_nav_ser = nav_ser.shift(1)
    prior_nav_ser.iloc[0] = capital_float
    return debit_ser/prior_nav_ser


def plot_tradeoff(screen_path: Path) -> None:
    metric_df = pd.read_csv(screen_path / "metrics.csv")
    chart_path = screen_path / "charts"
    chart_path.mkdir(exist_ok=True)
    selected_df = metric_df.loc[metric_df.book.isin(BOOK_TUPLE[:6]) & metric_df.policy.eq("drift") & metric_df.mix.isin(["Core10", "VIXM10", "Blend10"])].copy()
    figure_obj, axis_vec = plt.subplots(1, 2, figsize=(13, 5))
    for axis_obj, field_str, title_str in zip(axis_vec, ["delta_cagr", "delta_mdd"], ["CAGR change (pp/year)", "Max drawdown improvement (pp)"]):
        pivot_df = selected_df.pivot(index="book", columns="mix", values=field_str).reindex(BOOK_TUPLE[:6]) * 100
        pivot_df.index = ["L1", "L2", "L3", "L3b", "L3c", "L4"]
        pivot_df.plot.bar(ax=axis_obj, rot=0, color=["#59a14f", "#1f77b4", "#e17c05"])
        axis_obj.set(title=title_str, xlabel="Saved-artifact screen; not new fills")
        axis_obj.axhline(0, color="gray", linewidth=.7); axis_obj.grid(axis="y", alpha=.2)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path / "ladder_tradeoff.png", dpi=150); plt.close(figure_obj)


def build_evidence(screen_path: Path, fresh_path: Path, hedge_path: Path) -> None:
    chart_path = screen_path / "charts"
    chart_path.mkdir(exist_ok=True)
    crisis_df = pd.read_csv(screen_path / "crises.csv")
    baseline_df = pd.read_csv(fresh_path / "baseline_results.csv", index_col=0, parse_dates=True)
    candidate_df = pd.read_csv(fresh_path / "candidate_results.csv", index_col=0, parse_dates=True)
    if not baseline_df.index.equals(candidate_df.index):
        raise ValueError("Fresh realized baseline/candidate calendars differ.")
    hedge_df = pd.read_csv(hedge_path, index_col=0, parse_dates=True)
    common_idx = exact_common_calendar(baseline_df.index, hedge_df.index)
    if not common_idx.equals(baseline_df.index):
        raise ValueError("Benchmark must cover the entire fresh pair.")
    fresh_df = pd.DataFrame({"LADDER4": baseline_df.daily_returns,
        "LADDER4_VIXM10": candidate_df.daily_returns, "SPY": hedge_df.loc[common_idx, "SPY"]})
    require_returns(fresh_df)
    # *** CRITICAL*** First common close is the Portfolio capital anchor, not P&L.
    fresh_df.iloc[0] = 0.
    fresh_df.to_csv(screen_path / "fresh_common_returns.csv")
    scored_df = fresh_df.iloc[1:]
    fresh_metric_df = pd.DataFrame({series_str: metric_dict(scored_df[series_str], scored_df.SPY)
                                   for series_str in fresh_df}).T
    fresh_metric_df.to_csv(screen_path / "fresh_metrics.csv")
    old_control_ser = pd.read_csv(screen_path / "returns_ladder_4_growth_drift.csv", index_col=0, parse_dates=True).Baseline
    parity_idx = exact_common_calendar(old_control_ser.index, fresh_df.index)
    parity_df = pd.DataFrame({"saved_control": old_control_ser.loc[parity_idx],
                             "fresh_control": fresh_df.loc[parity_idx, "LADDER4"]})
    parity_df.to_csv(screen_path / "control_refresh_comparison.csv")
    parity_dict = {"max_abs_daily_difference": float((parity_df.fresh_control-parity_df.saved_control).abs().max()),
        "saved_common_cagr": metric_dict(parity_df.saved_control.iloc[1:], fresh_df.loc[parity_idx, "SPY"].iloc[1:])["cagr"],
        "fresh_common_cagr": metric_dict(parity_df.fresh_control.iloc[1:], fresh_df.loc[parity_idx, "SPY"].iloc[1:])["cagr"],
        "same_snapshot_replay": False}
    (screen_path / "control_refresh_comparison.json").write_text(json.dumps(parity_dict, indent=2), encoding="utf-8")
    fresh_crisis_df = crisis_tables(fresh_df, fresh_df.SPY)
    fresh_crisis_df.to_csv(screen_path / "fresh_crises.csv", index=False)
    period_row_list = []
    for period_str, period_idx in {
        "pre2020": scored_df.index[scored_df.index < "2020-01-01"],
        "2020": scored_df.index[scored_df.index.year == 2020],
        "post2020": scored_df.index[scored_df.index >= "2021-01-01"],
        "excluding2020": scored_df.index[scored_df.index.year != 2020],
    }.items():
        for series_str in scored_df:
            period_row_list.append({"period": period_str, "series": series_str,
                **metric_dict(scored_df.loc[period_idx, series_str], scored_df.loc[period_idx, "SPY"])})
    period_df = pd.DataFrame(period_row_list)
    period_df.to_csv(screen_path / "fresh_subperiods.csv", index=False)
    baseline_metric_ser = fresh_metric_df.loc["LADDER4"]
    candidate_metric_ser = fresh_metric_df.loc["LADDER4_VIXM10"]
    post_period_df = period_df.loc[period_df.period.eq("post2020")].set_index("series")
    market_tail_idx = scored_df.SPY.nsmallest(int(np.ceil(len(scored_df)*.01))).index
    gate_result_ser = pd.Series({
        "delta_cagr": candidate_metric_ser.cagr-baseline_metric_ser.cagr,
        "delta_mdd": candidate_metric_ser.mdd-baseline_metric_ser.mdd,
        "cvar_relative_gain": (candidate_metric_ser.cvar5_daily-baseline_metric_ser.cvar5_daily)/abs(baseline_metric_ser.cvar5_daily),
        "post2020_delta_cagr": post_period_df.loc["LADDER4_VIXM10", "cagr"]-post_period_df.loc["LADDER4", "cagr"],
        "post2020_delta_cvar": post_period_df.loc["LADDER4_VIXM10", "cvar5_daily"]-post_period_df.loc["LADDER4", "cvar5_daily"],
        "market_tail_delta": (scored_df.loc[market_tail_idx, "LADDER4_VIXM10"]-scored_df.loc[market_tail_idx, "LADDER4"]).mean(),
    })
    gate_result_ser["gate_pass"] = gate_pass(gate_result_ser)
    gate_result_ser.to_frame().T.to_csv(screen_path / "fresh_gate.csv", index=False)
    tail_row_list = []
    for selector_str in ("SPY", "LADDER4"):
        for fraction_float in (.01, .05):
            tail_idx = scored_df[selector_str].nsmallest(int(np.ceil(len(scored_df)*fraction_float))).index
            for series_str in scored_df:
                tail_row_list.append({"selector": selector_str, "fraction": fraction_float,
                    "series": series_str, "sessions": len(tail_idx),
                    "mean": scored_df.loc[tail_idx, series_str].mean(),
                    "hit_rate": (scored_df.loc[tail_idx, series_str] > 0).mean()})
    pd.DataFrame(tail_row_list).to_csv(screen_path / "fresh_tail_days.csv", index=False)
    # Diagnostic multiplicity penalty, not a claim of independent crisis trials.
    # H0: positive/negative crisis improvements are equally likely; episodes
    # may remain dependent, so even this conservative result is only descriptive.
    comparison_count_int = len(BOOK_TUPLE)*2*(len(MIX_DICT)-1)
    sign_row_list = []
    for (book_str, policy_str), group_df in crisis_df.loc[crisis_df.status.eq("complete")].groupby(["book", "policy"]):
        pivot_df = group_df.pivot(index="crisis", columns="series", values="return")
        for mix_str in MIX_DICT:
            if mix_str == "Baseline":
                continue
            delta_ser = pivot_df[mix_str]-pivot_df.Baseline
            nonzero_ser = delta_ser.loc[delta_ser.abs() > 1e-12]
            win_int, count_int = int((nonzero_ser > 0).sum()), len(nonzero_ser)
            probability_float = float(binomtest(win_int, count_int, .5, alternative="greater").pvalue)
            sign_row_list.append({"book": book_str, "policy": policy_str, "mix": mix_str,
                "wins": win_int, "crises": count_int, "p_raw_diagnostic": probability_float,
                "p_bonferroni_176": min(1., comparison_count_int*probability_float)})
    pd.DataFrame(sign_row_list).to_csv(screen_path / "crisis_multiplicity.csv", index=False)
    # Rebound after each trough: full positions continue; no favorable reset.
    rebound_row_list = []
    for crisis_str, group_df in fresh_crisis_df.loc[fresh_crisis_df.status.eq("complete")].groupby("crisis"):
        end_ts = pd.Timestamp(group_df.end_close.iloc[0])
        rebound_idx = fresh_df.index[fresh_df.index > end_ts][:5]
        if len(rebound_idx) != 5:
            continue
        for series_str in fresh_df:
            rebound_row_list.append({"crisis": crisis_str, "series": series_str,
                "return_next5": (1+fresh_df.loc[rebound_idx, series_str]).prod()-1})
    pd.DataFrame(rebound_row_list).to_csv(screen_path / "fresh_rebounds.csv", index=False)
    fresh_drift_df = pd.read_csv(fresh_path / "candidate_drift.csv", index_col=0, parse_dates=True)
    hedge_column_str = "strategy_vixm_backwardation"
    # *** CRITICAL*** Descriptive trailing126 sessions, no signal input.
    rolling_df = scored_df.rolling(126, min_periods=126).corr(scored_df.SPY)
    rolling_df.to_csv(screen_path / "fresh_rolling126.csv")
    colors_dict = {"LADDER4": "#1f77b4", "LADDER4_VIXM10": "#e17c05", "SPY": "#777777"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure_obj, axis_vec = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for series_str, color_str in colors_dict.items():
        axis_vec[0].plot(fresh_df.index, 100*(1+fresh_df[series_str]).cumprod(), color=color_str, label=series_str)
        axis_vec[1].plot(fresh_df.index, 100*drawdown_ser(fresh_df[series_str]), color=color_str)
    axis_vec[0].set(yscale="log", title="Fresh Vanilla pods + PM reconstruction (same dates)", ylabel="Wealth (100, log)")
    axis_vec[0].legend(ncol=3); axis_vec[1].set_ylabel("Drawdown (%)")
    for axis_obj in axis_vec: axis_obj.grid(alpha=.2)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path / "fresh_equity_drawdown.png", dpi=150); plt.close(figure_obj)
    plot_tradeoff(screen_path)
    pivot_df = fresh_crisis_df.loc[fresh_crisis_df.status.eq("complete")].pivot(index="crisis", columns="series", values="return")
    pivot_df = pivot_df.reindex([crisis_str for crisis_str in CRISIS_WINDOW_DICT if crisis_str in pivot_df.index])
    axis_obj = (pivot_df[list(colors_dict)]*100).plot.bar(figsize=(12, 5), color=list(colors_dict.values()), width=.8)
    axis_obj.set(title="Fresh LADDER4 crisis windows: peak close to trough close", ylabel="Return (%)", xlabel="")
    axis_obj.axhline(0, color="gray", linewidth=.7); axis_obj.grid(axis="y", alpha=.2)
    axis_obj.figure.tight_layout(); axis_obj.figure.savefig(chart_path / "fresh_crises.png", dpi=150); plt.close(axis_obj.figure)
    figure_obj, axis_vec = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for series_str in ("LADDER4", "LADDER4_VIXM10"):
        axis_vec[0].plot(rolling_df.index, rolling_df[series_str], label=series_str, color=colors_dict[series_str])
    axis_vec[0].set(title="Market dependence and hedge-budget erosion", ylabel="126-session SPY correlation", ylim=(-1, 1))
    axis_vec[0].legend(ncol=2)
    axis_vec[1].plot(fresh_drift_df.index, 100*fresh_drift_df[hedge_column_str], color="#e17c05", label="Fresh VIXM sleeve (drift)")
    screen_input_df = pd.read_csv(screen_path / "inputs/ladder_4_growth.csv", index_col=0, parse_dates=True)
    screen_weight_df = pd.read_csv(screen_path / "weights.csv")
    chosen_weight_ser = screen_weight_df.loc[screen_weight_df.book.eq("ladder_4_growth") & screen_weight_df.policy.eq("annual") & screen_weight_df.mix.eq("VIXM10")].set_index("pod").initial_weight
    _, annual_drift_df = allocate_path(screen_input_df, chosen_weight_ser, True)
    axis_vec[1].plot(annual_drift_df.index, 100*annual_drift_df.VIXM, color="#59a14f", alpha=.65, label="Annual reset: saved-NAV diagnostic")
    axis_vec[1].axhline(10, color="gray", linestyle="--", linewidth=.7)
    axis_vec[1].set_ylabel("VIXM sleeve / portfolio (%)"); axis_vec[1].legend()
    for axis_obj in axis_vec: axis_obj.grid(alpha=.2)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path / "correlation_and_budget.png", dpi=150); plt.close(figure_obj)
    file_list = list(Path("strategies/tail_hedge").glob("*.py")) + list(Path("portfolios").glob("*tail_vixm_10_research.yaml")) + [Path("alpha/strategy_registry.py")]
    file_list += list(fresh_path.glob("*.csv")) + [fresh_path / "manifest.json", screen_path / "manifest.json"]
    file_list += [hedge_path] + [hedge_path.parent / f"{hedge_str}_{kind_str}.csv"
        for hedge_str in ("Core", "VIXM") for kind_str in ("results", "turnover")]
    provenance_dict = {str(file_path.resolve()): hashlib.sha256(file_path.read_bytes()).hexdigest() for file_path in file_list}
    (screen_path / "final_provenance.json").write_text(json.dumps(provenance_dict, indent=2), encoding="utf-8")
    cost_row_list = []
    for policy_str in ("drift", "annual"):
        for mix_str in ("VIXM10", "Blend10"):
            target_ser = screen_weight_df.loc[screen_weight_df.book.eq("ladder_4_growth") & screen_weight_df.policy.eq(policy_str) & screen_weight_df.mix.eq(mix_str)].set_index("pod").initial_weight
            for slippage_bps_float, funding_rate_float in ((10., 0.), (25., 0.), (10., .05), (25., .05)):
                stress_input_df = screen_input_df.copy()
                for hedge_str in ("Core", "VIXM"):
                    turnover_ser = pd.read_csv(hedge_path.parent / f"{hedge_str}_turnover.csv", index_col=0, parse_dates=True).iloc[:, 0]
                    result_df = pd.read_csv(hedge_path.parent / f"{hedge_str}_results.csv", index_col=0, parse_dates=True)
                    # *** CRITICAL*** Ex-post fixed-holdings funding sensitivity;
                    # previous NAV normalizes the day debit; not a trading input.
                    prior_nav_ser = result_df.total_value.shift(1)
                    prior_nav_ser.iloc[0] = 100_000.
                    debit_ser = (-result_df.cash).clip(lower=0.)/prior_nav_ser
                    stress_input_df[hedge_str] -= (slippage_bps_float-10.)/10_000*turnover_ser.loc[stress_input_df.index]
                    stress_input_df[hedge_str] -= funding_rate_float/252*debit_ser.loc[stress_input_df.index]
                stress_input_df.iloc[0] = 0.
                stress_ser, _ = allocate_path(stress_input_df, target_ser, policy_str == "annual")
                cost_row_list.append({"policy": policy_str, "mix": mix_str,
                    "hedge_slippage_bps": slippage_bps_float, "hedge_funding_rate": funding_rate_float,
                    **metric_dict(stress_ser.iloc[1:], stress_input_df.SPY.iloc[1:])})
    pd.DataFrame(cost_row_list).to_csv(screen_path / "portfolio_cost_stress.csv", index=False)
    fresh_manifest_dict = json.loads((fresh_path / "manifest.json").read_text(encoding="utf-8"))
    hedge_pod_path = Path(fresh_manifest_dict["candidate"]["artifact"]) / "pods/pod_tail_vixm"
    ledger_path = hedge_pod_path / "dividend_ledger.csv"
    pickle_path = hedge_pod_path / "strategy_vixm_backwardation.pkl"
    with pickle_path.open("rb") as source_file:
        hedge_strategy_obj = pickle.load(source_file)
    tax_penalty_ser = dividend_tax_penalty(pd.read_csv(ledger_path),
        hedge_strategy_obj.results.total_value, hedge_strategy_obj._capital_base, .25)
    candidate_pod_df = pd.read_csv(fresh_path / "candidate_pod_returns.csv", index_col=0, parse_dates=True)
    candidate_weight_ser = fresh_drift_df.iloc[0]
    candidate_pod_df[hedge_column_str] -= tax_penalty_ser.loc[candidate_pod_df.index]
    candidate_pod_df.iloc[0] = 0.
    tax_stress_ser, _ = allocate_path(candidate_pod_df, candidate_weight_ser, False)
    pd.DataFrame([{"scenario": "hedge_dividend_withholding_25pct_original_holdings_approx_posthoc",
        **metric_dict(tax_stress_ser.iloc[1:], scored_df.SPY)}]).to_csv(screen_path / "fresh_tax_sensitivity.csv", index=False)
    for source_path in (ledger_path, pickle_path):
        provenance_dict[str(source_path.resolve())] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    for artifact_dict in fresh_manifest_dict.values():
        config_path = Path(artifact_dict["config"])
        provenance_dict[str(config_path.resolve())] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for pod_dict in config_dict["pods"]:
            source_path = Path(*pod_dict["strategy_import_str"].split(":")[0].split(".")).with_suffix(".py")
            provenance_dict[str(source_path.resolve())] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    (screen_path / "final_provenance.json").write_text(json.dumps(provenance_dict, indent=2), encoding="utf-8")
    print(fresh_metric_df.to_string())
    print("Final hedge weight:", fresh_drift_df[hedge_column_str].iloc[-1])


if __name__ == "__main__":
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("--screen", type=Path, required=True)
    parser_obj.add_argument("--fresh", type=Path, required=True)
    parser_obj.add_argument("--hedge-returns", type=Path, required=True)
    argument_obj = parser_obj.parse_args()
    build_evidence(argument_obj.screen, argument_obj.fresh, argument_obj.hedge_returns)
