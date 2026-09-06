"""Vanilla hedge diagnostics; no parameter search or deployment changes.

Run with ``uv run python -m strategies.tail_hedge.run_tail_hedge_vanilla_study``.
Standalone engines keep full-history positions through each crisis. Combined
curves sum normalized sleeve NAVs with initial allocations and no rebalancing:
E_t = sum_i w_i,0 * prod_{s<=t}(1+r_i,s). These are allocation diagnostics,
not a fresh share-level simulation at every allocated dollar amount.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.norgate_loader import load_price_timeseries
from strategies.tail_hedge import strategy_crisis_trend_core as core_module
from strategies.tail_hedge import strategy_vixm_backwardation as vixm_module


CRISIS_WINDOW_DICT = {
    "2008_GFC": ("2007-10-09", "2009-03-09"),
    "2008_Lehman": ("2008-09-12", "2008-11-20"),
    "2010_Flash": ("2010-04-23", "2010-07-02"),
    "2011_Euro": ("2011-07-25", "2011-10-03"),
    "2015_China": ("2015-08-17", "2015-09-29"),
    "2018_Volmageddon": ("2018-02-01", "2018-02-09"),
    "2018_Q4": ("2018-10-01", "2018-12-24"),
    "2020_Covid": ("2020-02-19", "2020-03-23"),
    "2022_Bear": ("2022-01-03", "2022-10-12"),
    "2024_Yen": ("2024-07-16", "2024-08-05"),
    "2025_Tariff": ("2025-02-19", "2025-04-08"),
}
ALLOCATIONS_DICT = {
    "Hedge_50_50": {"Core": 0.5, "VIXM": 0.5},
    "SPY90_Core10": {"SPY": 0.9, "Core": 0.1},
    "SPY90_VIXM10": {"SPY": 0.9, "VIXM": 0.1},
    "SPY90_Core5_VIXM5": {"SPY": 0.9, "Core": 0.05, "VIXM": 0.05},
    "SPY90_SHY10": {"SPY": 0.9, "SHY": 0.1},
}
COLOR_DICT = {"Core": "#2171b5", "VIXM": "#d94841", "Hedge_50_50": "#6a51a3", "SPY": "#333333"}


def require_returns(return_df: pd.DataFrame) -> None:
    if return_df.empty or not return_df.index.is_unique or not return_df.index.is_monotonic_increasing:
        raise ValueError("Returns require a nonempty, unique, ordered calendar.")
    if not np.isfinite(return_df.to_numpy()).all() or (return_df <= -1.0).any().any():
        raise ValueError("Missing, nonfinite, or insolvent returns cannot be filled.")


def nav_return_ser(nav_ser: pd.Series, initial_nav_float: float = 1.0) -> pd.Series:
    # *** CRITICAL*** Report-only: use prior NAV; preserve the first day's P&L.
    prior_nav_ser = nav_ser.shift(1)
    prior_nav_ser.iloc[0] = initial_nav_float
    return nav_ser / prior_nav_ser - 1.0


def drawdown_ser(return_ser: pd.Series) -> pd.Series:
    nav_ser = (1.0 + return_ser).cumprod()
    # Include the initial capital peak, including losses on the first day.
    return nav_ser / nav_ser.cummax().clip(lower=1.0) - 1.0


def combine_sleeves(return_df: pd.DataFrame, allocation_dict: dict[str, float]) -> tuple[pd.Series, pd.DataFrame]:
    require_returns(return_df.loc[:, list(allocation_dict)])
    if any(weight_float < 0 for weight_float in allocation_dict.values()) or not np.isclose(sum(allocation_dict.values()), 1.0):
        raise ValueError("Initial weights must be nonnegative and sum to one.")
    sleeve_nav_df = (1.0 + return_df.loc[:, list(allocation_dict)]).cumprod().mul(pd.Series(allocation_dict))
    portfolio_nav_ser = sleeve_nav_df.sum(axis=1)
    return nav_return_ser(portfolio_nav_ser), sleeve_nav_df.div(portfolio_nav_ser, axis=0)


def metric_dict(return_ser: pd.Series, market_ser: pd.Series) -> dict:
    if not return_ser.index.equals(market_ser.index):
        raise ValueError("Metrics require the exact market calendar.")
    require_returns(pd.DataFrame({"strategy": return_ser, "market": market_ser}))
    std_float = float(return_ser.std(ddof=1))
    market_variance_float = float(market_ser.var(ddof=1))
    count_int = max(1, math.ceil(len(return_ser) * 0.05))
    # Calendar-month products are diagnostic; no monthly information feeds signals.
    monthly_ser = (1.0 + return_ser).resample("ME").prod(min_count=1).dropna() - 1.0
    market_monthly_ser = (1.0 + market_ser).resample("ME").prod(min_count=1).dropna() - 1.0
    return {
        "start": str(return_ser.index[0].date()), "end": str(return_ser.index[-1].date()),
        "sessions": len(return_ser), "total_return": float((1.0 + return_ser).prod() - 1.0),
        "cagr": float((1.0 + return_ser).prod() ** (252.0 / len(return_ser)) - 1.0),
        "vol": std_float * np.sqrt(252.0),
        "sharpe": float(return_ser.mean() / std_float * np.sqrt(252.0)) if std_float else np.nan,
        "mdd": float(drawdown_ser(return_ser).min()),
        "corr_daily": float(return_ser.corr(market_ser)),
        "corr_monthly": float(monthly_ser.corr(market_monthly_ser)),
        "beta": float(return_ser.cov(market_ser) / market_variance_float) if market_variance_float else np.nan,
        "cvar5_daily": float(return_ser.nsmallest(count_int).mean()),
        "worst_day": float(return_ser.min()), "best_day": float(return_ser.max()),
    }


def crisis_tables(return_df: pd.DataFrame, market_ser: pd.Series) -> pd.DataFrame:
    row_list = []
    for crisis_str, (start_str, end_str) in CRISIS_WINDOW_DICT.items():
        # *** CRITICAL*** Ex-post scoring only. Peak close -> trough close:
        # exclude the return ending at the named start close. Never reset holdings.
        calendar_idx = market_ser.loc[(market_ser.index > start_str) & (market_ser.index <= end_str)].index
        for series_str in return_df:
            available_ser = return_df[series_str].dropna()
            boundary_complete_bool = bool(
                len(available_ser) > 0
                and market_ser.index.min() <= pd.Timestamp(start_str)
                and market_ser.index.max() >= pd.Timestamp(end_str)
                and available_ser.index.min() <= pd.Timestamp(start_str)
                and available_ser.index.max() >= pd.Timestamp(end_str)
            )
            if not boundary_complete_bool or len(calendar_idx) == 0 or not calendar_idx.isin(available_ser.index).all():
                row_list.append({"crisis": crisis_str, "series": series_str, "status": "unavailable"})
                continue
            window_ser = available_ser.loc[calendar_idx]
            market_window_ser = market_ser.loc[calendar_idx]
            first_count_int = min(5, len(calendar_idx))
            row_list.append({
                "crisis": crisis_str, "series": series_str, "status": "complete",
                "start_close": start_str, "end_close": end_str, "sessions": len(window_ser),
                "return": float((1.0 + window_ser).prod() - 1.0),
                "mdd": float(drawdown_ser(window_ser).min()),
                "first5_return": float((1.0 + window_ser.iloc[:first_count_int]).prod() - 1.0),
                "first5_spy_return": float((1.0 + market_window_ser.iloc[:first_count_int]).prod() - 1.0),
            })
    return pd.DataFrame(row_list)


def save_figures(return_df: pd.DataFrame, crisis_df: pd.DataFrame, chart_path: Path) -> None:
    chart_path.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure_obj, axis_vec = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for series_str, color_str in COLOR_DICT.items():
        axis_vec[0].plot(return_df.index, 100 * (1 + return_df[series_str]).cumprod(), label=series_str, color=color_str)
        axis_vec[1].plot(return_df.index, 100 * drawdown_ser(return_df[series_str]), color=color_str)
    axis_vec[0].set(yscale="log", ylabel="Wealth (start=100, log scale)", title="Vanilla: full-history sleeve returns, common dates")
    axis_vec[0].legend(ncol=4); axis_vec[1].set_ylabel("Drawdown (%)")
    for axis_obj in axis_vec: axis_obj.grid(alpha=0.2)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path / "equity_drawdown.png", dpi=150); plt.close(figure_obj)

    figure_obj, axis_obj = plt.subplots(figsize=(12, 4.5))
    for series_str in ("Core", "VIXM", "Hedge_50_50"):
        # *** CRITICAL*** trailing 126 observations; descriptive, not a trading input.
        rolling_ser = return_df[series_str].rolling(126, min_periods=126).corr(return_df["SPY"])
        axis_obj.plot(rolling_ser.index, rolling_ser, label=series_str, color=COLOR_DICT[series_str], linewidth=1)
    axis_obj.set(title="Trailing 126-session correlation to SPY", ylabel="Pearson correlation", ylim=(-1, 1))
    axis_obj.axhline(0, color="gray", linewidth=0.7); axis_obj.legend(ncol=3); axis_obj.grid(alpha=0.2)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path / "rolling_correlation.png", dpi=150); plt.close(figure_obj)

    crisis_pivot_df = crisis_df.loc[crisis_df["status"] == "complete"].pivot(index="crisis", columns="series", values="return")
    crisis_pivot_df = crisis_pivot_df.reindex(CRISIS_WINDOW_DICT).loc[:, list(COLOR_DICT)].dropna()
    axis_obj = (crisis_pivot_df * 100).plot.bar(figsize=(12, 5), color=list(COLOR_DICT.values()), width=0.82)
    axis_obj.set(title="Crisis return: peak close to trough close; no restart", ylabel="Return (%)")
    axis_obj.axhline(0, color="gray", linewidth=0.7); axis_obj.legend(ncol=4); axis_obj.grid(axis="y", alpha=0.2)
    axis_obj.figure.tight_layout(); axis_obj.figure.savefig(chart_path / "crises.png", dpi=150); plt.close(axis_obj.figure)

    figure_obj, axis_vec = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for series_str in ("SPY", "SPY90_Core10", "SPY90_VIXM10", "SPY90_Core5_VIXM5", "SPY90_SHY10"):
        axis_vec[0].plot(return_df.index, 100 * (1 + return_df[series_str]).cumprod(), label=series_str)
        axis_vec[1].plot(return_df.index, 100 * drawdown_ser(return_df[series_str]))
    axis_vec[0].set(title="Funded allocation diagnostic: initial weights, then drift", ylabel="Wealth (start=100)")
    axis_vec[0].legend(ncol=2); axis_vec[1].set_ylabel("Drawdown (%)")
    for axis_obj in axis_vec: axis_obj.grid(alpha=0.2)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path / "market_portfolios.png", dpi=150); plt.close(figure_obj)


def run_study(output_path: Path, end_date_str: str) -> None:
    output_path.mkdir(parents=True, exist_ok=False)
    (output_path / "data").mkdir(); (output_path / "tables").mkdir()
    strategy_by_name_dict = {}
    run_path_dict = {}
    diagnostics_list = []
    for label_str, module_obj, class_obj, loader_fn in (
        ("Core", core_module, core_module.CrisisTrendCoreStrategy, core_module.get_crisis_trend_core_data),
        ("VIXM", vixm_module, vixm_module.VixmBackwardationStrategy, vixm_module.get_vixm_backwardation_data),
    ):
        print(f"START {label_str}", flush=True)
        config_obj = replace(module_obj.DEFAULT_CONFIG, end_date_str=end_date_str)
        pricing_df = loader_fn(config_obj)
        pricing_df.to_csv(output_path / "data" / f"{label_str}_input.csv.gz")
        calendar_idx = module_obj.build_execution_calendar_idx(pricing_df, config_obj.backtest_start_date_str)
        strategy_obj = class_obj(config_obj)
        run_daily(strategy_obj, pricing_df, calendar=calendar_idx, show_progress=False,
                  show_signal_progress_bool=False, audit_override_bool=True, audit_sample_size_int=10)
        run_path_dict[label_str] = str(save_results(strategy_obj).resolve())
        strategy_obj.results.to_csv(output_path / "data" / f"{label_str}_results.csv")
        strategy_obj.daily_target_weights.to_csv(output_path / "data" / f"{label_str}_targets.csv")
        strategy_obj.realized_weight_df.to_csv(output_path / "data" / f"{label_str}_realized_weights.csv")
        transaction_df = strategy_obj.get_transactions().copy()
        transaction_df.to_csv(output_path / "data" / f"{label_str}_transactions.csv", index=False)
        result_df = strategy_obj.results.copy(); result_df.index = pd.to_datetime(result_df.index)
        # *** CRITICAL*** Post-run cost diagnostic, prior close NAV on the execution date.
        prior_nav_ser = result_df["total_value"].shift(1); prior_nav_ser.iloc[0] = config_obj.capital_base_float
        turnover_ser = pd.Series(0.0, index=result_df.index)
        for transaction_obj in transaction_df.itertuples():
            bar_ts = pd.Timestamp(transaction_obj.bar)
            raw_open_float = float(pricing_df.loc[bar_ts, (transaction_obj.asset, "Open")])
            turnover_ser.loc[bar_ts] += abs(float(transaction_obj.amount)) * raw_open_float / prior_nav_ser.loc[bar_ts]
        turnover_ser.to_csv(output_path / "data" / f"{label_str}_turnover.csv")
        cash_weight_ser = result_df["cash"] / result_df["total_value"]
        realized_df = strategy_obj.realized_weight_df
        active_fraction_float = float(realized_df["VIXM"].notna().mul(realized_df["VIXM"] > 0).mean()) if label_str == "VIXM" else np.nan
        diagnostics_list.append({
            "series": label_str, "sessions": len(result_df), "transactions": len(transaction_df),
            "turnover_per_year": float(turnover_ser.sum() * 252 / len(turnover_ser)),
            "slippage_arithmetic_per_year": float(turnover_ser.sum() * 0.001 * 252 / len(turnover_ser)),
            "borrow_dollars": float(getattr(strategy_obj, "borrow_fee_total_float", 0)),
            "negative_cash_days": int((cash_weight_ser < -1e-10).sum()),
            "min_cash_weight": float(cash_weight_ser.min()), "vixm_active_fraction": active_fraction_float,
        })
        strategy_by_name_dict[label_str] = strategy_obj
        print(f"COMPLETE {label_str}", flush=True)

    analyze_results(output_path, end_date_str, strategy_by_name_dict, run_path_dict, diagnostics_list)


def analyze_results(output_path: Path, end_date_str: str, strategy_by_name_dict: dict,
                    run_path_dict: dict, diagnostics_list: list, reuse_bool: bool = False) -> None:
    benchmark_df = pd.DataFrame()
    for asset_str in ("SPY", "SHY"):
        benchmark_path = output_path / "data" / f"{asset_str}_benchmark.csv.gz"
        if reuse_bool:
            price_df = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
        else:
            price_df = load_price_timeseries(asset_str, adjustment_str="TOTALRETURN", start_date_str="2003-12-01", end_date_str=end_date_str)
            price_df.to_csv(benchmark_path)
        # *** CRITICAL*** Benchmark close-to-close total returns aligned to Vanilla NAV marks.
        benchmark_df[asset_str] = price_df["Close"].astype(float).pct_change(fill_method=None)
    full_return_df = pd.DataFrame({label_str: strategy_obj.results["daily_returns"].astype(float)
                                  for label_str, strategy_obj in strategy_by_name_dict.items()})
    full_return_df.index = pd.to_datetime(full_return_df.index)
    full_return_df = full_return_df.join(benchmark_df)
    common_start_ts = max(strategy_obj.results.index[0] for strategy_obj in strategy_by_name_dict.values())
    common_return_df = full_return_df.loc[pd.Timestamp(common_start_ts):].copy()
    require_returns(common_return_df)
    for portfolio_str, allocation_dict in ALLOCATIONS_DICT.items():
        portfolio_return_ser, weight_df = combine_sleeves(common_return_df, allocation_dict)
        common_return_df[portfolio_str] = portfolio_return_ser
        weight_df.to_csv(output_path / "data" / f"{portfolio_str}_weights.csv")
        full_return_df[portfolio_str] = portfolio_return_ser
    common_return_df.to_csv(output_path / "data" / "common_returns.csv")
    full_return_df.to_csv(output_path / "data" / "full_returns.csv")
    summary_df = pd.DataFrame({series_str: metric_dict(common_return_df[series_str], common_return_df["SPY"])
                               for series_str in common_return_df}).T
    summary_df.to_csv(output_path / "tables" / "common_summary.csv")
    full_core_df = full_return_df.loc[full_return_df["Core"].notna(), ["Core", "SPY", "SHY"]]
    require_returns(full_core_df)
    pd.DataFrame({series_str: metric_dict(full_core_df[series_str], full_core_df["SPY"])
                  for series_str in full_core_df}).T.to_csv(output_path / "tables" / "full_core_summary.csv")
    crisis_df = crisis_tables(full_return_df, full_return_df["SPY"])
    crisis_df.to_csv(output_path / "tables" / "crises.csv", index=False)
    aftermath_list = []
    for crisis_str, (_, end_str) in CRISIS_WINDOW_DICT.items():
        # *** CRITICAL*** Forward window is ex-post reporting only; it never
        # changes the position path. Show whipsaw losses after the chosen trough.
        aftermath_df = common_return_df.loc[common_return_df.index > end_str].iloc[:5]
        if pd.Timestamp(end_str) < common_return_df.index[0] or len(aftermath_df) < 5:
            continue
        for series_str in COLOR_DICT:
            aftermath_list.append({"crisis": crisis_str, "series": series_str,
                                   "start": str(aftermath_df.index[0].date()), "end": str(aftermath_df.index[-1].date()),
                                   "return": float((1 + aftermath_df[series_str]).prod() - 1)})
    pd.DataFrame(aftermath_list).to_csv(output_path / "tables" / "crisis_next5_sessions.csv", index=False)
    common_return_df.corr().to_csv(output_path / "tables" / "correlation_daily.csv")
    monthly_df = (1 + common_return_df).resample("ME").prod() - 1
    monthly_df.corr().to_csv(output_path / "tables" / "correlation_monthly.csv")
    annual_df = (1 + common_return_df).resample("YE").prod() - 1
    annual_df.to_csv(output_path / "tables" / "annual_returns.csv")
    # *** CRITICAL*** Ex-post tail selection, same-day score; never a forecast or signal.
    tail_list = []
    for fraction_float in (0.01, 0.05):
        count_int = max(1, math.ceil(len(common_return_df) * fraction_float))
        tail_idx = common_return_df["SPY"].nsmallest(count_int).index
        for series_str in common_return_df:
            tail_ser = common_return_df.loc[tail_idx, series_str]
            tail_list.append({"fraction": fraction_float, "n": count_int, "series": series_str,
                              "mean_return": float(tail_ser.mean()), "positive_fraction": float((tail_ser > 0).mean()),
                              "worst_return": float(tail_ser.min())})
    pd.DataFrame(tail_list).to_csv(output_path / "tables" / "market_tail_days.csv", index=False)
    common_return_df.loc[common_return_df["SPY"].nsmallest(20).index].sort_index().to_csv(output_path / "tables" / "worst20_market_days.csv")
    subperiod_list = []
    for start_str, end_str in (("2011-03-01", "2019-12-31"), ("2020-01-01", "2020-12-31"), ("2021-01-01", end_date_str)):
        period_df = common_return_df.loc[start_str:end_str]
        for series_str in COLOR_DICT:
            subperiod_list.append({"series": series_str, **metric_dict(period_df[series_str], period_df["SPY"])})
    pd.DataFrame(subperiod_list).to_csv(output_path / "tables" / "subperiods.csv", index=False)
    # Omit 2020 returns without recomputing signals: descriptive concentration test only.
    without_2020_df = common_return_df.loc[common_return_df.index.year != 2020]
    pd.DataFrame({series_str: metric_dict(without_2020_df[series_str], without_2020_df["SPY"])
                  for series_str in COLOR_DICT}).T.to_csv(output_path / "tables" / "excluding_2020_diagnostic.csv")
    cost_list = []
    for label_str in strategy_by_name_dict:
        turnover_ser = pd.read_csv(output_path / "data" / f"{label_str}_turnover.csv", index_col=0, parse_dates=True).iloc[:, 0]
        stress_ser = common_return_df[label_str] - 0.0015 * turnover_ser.loc[common_return_df.index]
        cost_list.append({"series": label_str, "description": "25bps_fixed_holdings_diagnostic",
                          **metric_dict(stress_ser, common_return_df["SPY"])})
        result_df = strategy_by_name_dict[label_str].results.copy()
        result_df.index = pd.to_datetime(result_df.index)
        # *** CRITICAL*** Post-run sensitivity only: debit on end-close negative
        # cash / prior NAV. No new signals or financed-position replay is implied.
        prior_nav_ser = result_df["total_value"].shift(1)
        prior_nav_ser.iloc[0] = strategy_by_name_dict[label_str].config_obj.capital_base_float
        debit_ser = (-result_df["cash"].clip(upper=0) / prior_nav_ser).loc[common_return_df.index]
        funding_stress_ser = common_return_df[label_str] - debit_ser * 0.05 / 252
        cost_list.append({"series": label_str, "description": "10bps_plus_5pct_debit_funding_fixed_holdings",
                          **metric_dict(funding_stress_ser, common_return_df["SPY"])})
    pd.DataFrame(cost_list).to_csv(output_path / "tables" / "cost_stress.csv", index=False)
    pd.DataFrame(diagnostics_list).to_csv(output_path / "tables" / "execution_diagnostics.csv", index=False)
    # *** CRITICAL*** The rolling series ends at each current close and is report-only.
    rolling_df = pd.DataFrame({series_str: common_return_df[series_str].rolling(126).corr(common_return_df["SPY"])
                              for series_str in ("Core", "VIXM", "Hedge_50_50")})
    rolling_df.to_csv(output_path / "data" / "rolling126_correlation.csv")
    save_figures(common_return_df, crisis_df, output_path / "charts")
    manifest_dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "end_date": end_date_str,
        "scope": "research_only", "engine": "VanillaBacktester", "signal_audit": True,
        "capital_per_engine": 100000, "slippage_bps": 10, "core_borrow_bps": 50,
        "withholding": 0, "market": "SPY_TOTALRETURN_close_to_close", "annualization": 252,
        "risk_free_rate": 0, "allocations": ALLOCATIONS_DICT,
        "aggregation": "normalized_full_history_NAVs; initial_weights; no_rebalance; not_dollar_resized_engines",
        "crisis_boundary": "start_close_exclusive_end_close_inclusive; no_position_reset",
        "crisis_windows": CRISIS_WINDOW_DICT, "crises_overlap": "Lehman is inside GFC; not independent trials",
        "new_signal_variants": 0,
        "cost_stress": {"mode": "fixed_holdings_diagnostics_not_engine_replays",
                        "slippage_stress_bps": 25, "incremental_slippage_bps": 15,
                        "funding_annual_rate": 0.05,
                        "funding_basis": "max(-end_close_cash,0)/prior_NAV/252; baseline_10bps_slippage"},
        "vanilla_artifacts": run_path_dict,
        "source_hashes": {str(Path(module_obj.__file__).resolve()): hashlib.sha256(Path(module_obj.__file__).read_bytes()).hexdigest()
                          for module_obj in (core_module, vixm_module)},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": {str(file_path.relative_to(output_path)): hashlib.sha256(file_path.read_bytes()).hexdigest()
                        for file_path in (output_path / "data").iterdir()},
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
    print(summary_df.to_string(), flush=True)
    print(f"STUDY_COMPLETE {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("--end-date", default="2026-08-31")
    parser_obj.add_argument("--output", type=Path, default=Path("results/research/tail_hedge_vanilla_20260904"))
    parser_obj.add_argument("--analyze-existing", action="store_true")
    argument_obj = parser_obj.parse_args()
    if argument_obj.analyze_existing:
        manifest_dict = json.loads((argument_obj.output / "manifest.json").read_text(encoding="utf-8"))
        run_path_dict = manifest_dict["vanilla_artifacts"]
        strategy_by_name_dict = {}
        for label_str, run_path_str in run_path_dict.items():
            pickle_path_list = list(Path(run_path_str).glob("*.pkl"))
            if len(pickle_path_list) != 1:
                raise ValueError("Expected exactly one locally generated Vanilla pickle.")
            strategy_by_name_dict[label_str] = core_module.Strategy.read_pickle(pickle_path_list[0])
        diagnostics_list = pd.read_csv(argument_obj.output / "tables" / "execution_diagnostics.csv").to_dict("records")
        analyze_results(argument_obj.output, manifest_dict["end_date"], strategy_by_name_dict, run_path_dict, diagnostics_list, reuse_bool=True)
    else:
        run_study(argument_obj.output, argument_obj.end_date)
