"""Post-run evidence tables and scientific charts; never feeds selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_DICT = {"legacy_vintage_diagnostic": "Legacy (invalid)",
    "asof_atr_signal_only_attribution": "ATR signal correction only",
    "asof_atr_turnover_only_attribution": "ATR + turnover correction",
    "asof_atr_corrected": "Corrected nominal ATR", "natr20_corrected": "NATR20 candidate"}
MAIN_VARIANT_LIST = ["legacy_vintage_diagnostic", "asof_atr_corrected", "natr20_corrected"]
COLOR_LIST = ["#c64b48", "#2865a8", "#268b62"]


def main(input_path, output_path):
    metric_path_list = sorted(input_path.glob("*/*/metrics.json"))
    if len(metric_path_list) != 13:
        raise RuntimeError(f"Expected all 13 frozen runs, found {len(metric_path_list)}")
    output_path.mkdir(parents=True, exist_ok=True)
    chart_path = output_path / "charts"
    chart_path.mkdir(exist_ok=True)
    metric_list, period_list, selection_list = [], [], []
    daily_dict = {}
    for metric_path in metric_path_list:
        metric_dict = json.loads(metric_path.read_text(encoding="utf-8"))
        family_str, variant_str = metric_dict["family"], metric_dict["variant"]
        result_df = pd.read_csv(metric_path.parent / "daily_results.csv", index_col=0, parse_dates=True)
        daily_dict[(family_str, variant_str)] = result_df
        equity_ser = result_df["total_value"].astype(float)
        # *** CRITICAL *** These are retrospective report statistics only.
        # They never enter signals, fitted transforms, or position decisions.
        return_ser = equity_ser.pct_change(fill_method=None)
        return_ser.iloc[0] = equity_ser.iloc[0] / 100_000.0 - 1.0
        benchmark_ser = result_df["$SPX"].astype(float)
        benchmark_return_ser = benchmark_ser.pct_change(fill_method=None)
        benchmark_return_ser.iloc[0] = 0.0
        if return_ser.isna().any() or benchmark_return_ser.isna().any():
            raise ValueError("No filling permitted in report return alignment")
        metric_dict["market_daily_correlation"] = float(return_ser.corr(benchmark_return_ser))
        metric_dict["market_beta"] = float(return_ser.cov(benchmark_return_ser) / benchmark_return_ser.var())
        monthly_return_ser = (1.0 + return_ser).resample("ME").prod() - 1.0
        market_monthly_ser = (1.0 + benchmark_return_ser).resample("ME").prod() - 1.0
        metric_dict["market_monthly_correlation"] = float(monthly_return_ser.corr(market_monthly_ser))
        annual_return_ser = (1.0 + return_ser).resample("YE").prod() - 1.0
        annual_return_ser.rename("return").to_csv(output_path / f"{family_str}_{variant_str}_annual.csv")
        rolling_correlation_ser = return_ser.rolling(126, min_periods=126).corr(benchmark_return_ser)
        rolling_correlation_ser.rename("rolling126_market_corr").to_csv(output_path / f"{family_str}_{variant_str}_rolling126.csv")
        metric_dict["rolling126_corr_min"] = float(rolling_correlation_ser.min())
        metric_dict["rolling126_corr_max"] = float(rolling_correlation_ser.max())
        transaction_df = pd.read_csv(metric_path.parent / "transactions.csv")
        previous_equity_ser = equity_ser.shift(1)
        previous_equity_ser.iloc[0] = 100_000.0
        prior_nav_vec = previous_equity_ser.reindex(pd.to_datetime(transaction_df["bar"])).to_numpy()
        years_float = len(result_df) / 252.0
        metric_dict["annual_commission_bps"] = float((transaction_df["commission"].to_numpy() / prior_nav_vec).sum() / years_float * 10000.0)
        metric_dict["annual_slippage_bps_proxy"] = float((transaction_df["total_value"].abs().to_numpy() / prior_nav_vec).sum() * 0.00025 / years_float * 10000.0)
        metric_dict["extra_10bps_per_side_annual_drag_proxy"] = float((transaction_df["total_value"].abs().to_numpy() / prior_nav_vec).sum() * 0.001 / years_float)
        metric_dict["negative_cash_days"] = int((result_df["cash"] < 0.0).sum())
        metric_dict["min_cash_fraction"] = float((result_df["cash"] / equity_ser).min())
        for period_dict in metric_dict.pop("subperiods"):
            period_list.append(dict(family=family_str, variant=variant_str, **period_dict))
        metric_list.append({key_str: value_obj for key_str, value_obj in metric_dict.items() if not isinstance(value_obj, dict)})
        decisions_list = json.loads((metric_path.parent / "decisions.json").read_text(encoding="utf-8"))
        legacy_list = json.loads((input_path / family_str / "legacy_vintage_diagnostic" / "decisions.json").read_text(encoding="utf-8"))
        legacy_map = {row_dict["decision_date"]: row_dict["weights"] for row_dict in legacy_list}
        for row_dict in decisions_list:
            old_dict, new_dict = legacy_map[row_dict["decision_date"]], row_dict["weights"]
            old_set, new_set = set(old_dict), set(new_dict)
            union_set = old_set | new_set
            selection_list.append(dict(family=family_str, variant=variant_str,
                decision_date=row_dict["decision_date"], changed=old_set != new_set,
                active=bool(union_set), jaccard=len(old_set & new_set) / len(union_set) if union_set else 1.0,
                old_count=len(old_set), new_count=len(new_set),
                weight_l1=sum(abs(old_dict.get(symbol_str, 0.0) - new_dict.get(symbol_str, 0.0)) for symbol_str in union_set)))
    metric_df = pd.DataFrame(metric_list)
    metric_df.to_csv(output_path / "metrics_all_13_runs.csv", index=False)
    pd.DataFrame(period_list).to_csv(output_path / "subperiods.csv", index=False)
    selection_df = pd.DataFrame(selection_list)
    selection_df.to_csv(output_path / "selection_changes_by_month.csv", index=False)
    selection_df.groupby(["family", "variant"]).agg(months=("changed", "size"),
        changed_months=("changed", "sum"), active_months=("active", "sum"),
        mean_jaccard=("jaccard", "mean"), mean_weight_l1=("weight_l1", "mean")).to_csv(output_path / "selection_damage.csv")
    benchmark_df = daily_dict[("ndx", "asof_atr_corrected")]
    benchmark_ser = benchmark_df["$SPX"].astype(float)
    benchmark_return_ser = benchmark_ser.pct_change(fill_method=None)
    benchmark_return_ser.iloc[0] = 0.0
    benchmark_dict = dict(label="$SPXTR total return", start=str(benchmark_ser.index.min().date()),
        end=str(benchmark_ser.index.max().date()), cagr=float((benchmark_ser.iloc[-1] / 100000.0) ** (252.0/len(benchmark_ser))-1.0),
        volatility=float(benchmark_return_ser.std()*np.sqrt(252.0)),
        sharpe=float(benchmark_return_ser.mean()/benchmark_return_ser.std()*np.sqrt(252.0)),
        max_drawdown=float((benchmark_ser/benchmark_ser.cummax()-1.0).min()))
    (output_path / "benchmark.json").write_text(json.dumps(benchmark_dict, indent=2), encoding="utf-8")
    for chart_kind_str in ["equity", "drawdown", "correlation"]:
        figure_obj, axes_vec = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        for axis_obj, family_str in zip(axes_vec, ["ndx", "vxn", "mosaic"]):
            for variant_str, color_str in zip(MAIN_VARIANT_LIST, COLOR_LIST):
                result_df = daily_dict[(family_str, variant_str)]
                equity_ser = result_df["total_value"].astype(float)
                if chart_kind_str == "equity":
                    plotted_ser = equity_ser / 100_000.0
                    axis_obj.set_yscale("log")
                elif chart_kind_str == "drawdown":
                    plotted_ser = 100.0 * (equity_ser / equity_ser.cummax().clip(lower=100_000.0) - 1.0)
                else:
                    plotted_ser = equity_ser.pct_change(fill_method=None).rolling(126, min_periods=126).corr(result_df["$SPX"].pct_change(fill_method=None))
                axis_obj.plot(plotted_ser.index, plotted_ser.values, label=LABEL_DICT[variant_str], color=color_str, linewidth=1.15)
            if chart_kind_str == "equity":
                axis_obj.plot(benchmark_ser.index, benchmark_ser/100000.0, label="S&P 500 total return", color="#7b8089", linewidth=1.0, linestyle="--")
            axis_obj.set_title(family_str.upper(), loc="left", fontweight="bold")
            axis_obj.grid(alpha=0.2)
            axis_obj.set_ylabel({"equity": "Growth of $1 (log)", "drawdown": "Drawdown (%)", "correlation": "126-session correlation"}[chart_kind_str])
        axes_vec[0].legend(fontsize=9, ncol=2)
        figure_obj.suptitle("Corporate-action leakage audit | 2000-01-03 to 2026-09-25", fontsize=15)
        figure_obj.tight_layout(rect=(0, 0, 1, 0.96))
        figure_obj.savefig(chart_path / f"{chart_kind_str}.png", dpi=170)
        plt.close(figure_obj)
    print(metric_df[["family", "variant", "cagr", "sharpe", "max_drawdown", "market_beta"]].to_string(index=False))


if __name__ == "__main__":
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("--inputs", type=Path, required=True)
    parser_obj.add_argument("--outputs", type=Path, required=True)
    argument_obj = parser_obj.parse_args()
    main(argument_obj.inputs, argument_obj.outputs)
