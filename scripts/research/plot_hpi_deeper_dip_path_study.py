"""Four standalone figures for the frozen HPI path-sensitivity study."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

REPO_PATH = Path(__file__).resolve().parents[2]
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))
from scripts.research.run_hpi_deeper_dip_path_study import STUDY_PATH
from scripts.research.run_mr_deeper_dip_study import STUDY_PATH as PARENT_PATH, write_json
from scripts.research.analyze_mr_deeper_dip_study import nav_returns

BLUE_STR, ORANGE_STR, GREEN_STR, RED_STR = "#3869a5", "#df8b26", "#238575", "#c65257"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                    "axes.spines.right": False, "axes.titleweight": "bold",
                    "axes.labelcolor": "#334155", "text.color": "#1e293b", "svg.fonttype": "none"})


def save_figure(figure_obj, name_str):
    chart_path = STUDY_PATH / "charts"
    chart_path.mkdir(exist_ok=True)
    figure_obj.savefig(chart_path / f"{name_str}.png", dpi=170, bbox_inches="tight", facecolor="white")
    figure_obj.savefig(chart_path / f"{name_str}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure_obj)


def main(attribution_only_bool=False):
    table_path = STUDY_PATH / "tables"
    group_df = pd.read_csv(table_path / "attribution_groups.csv")
    figure_obj, axis_obj = plt.subplots(figsize=(10, 5.7))
    central_ser = group_df[group_df["layer"] == "central"].set_index("group")["delta_pnl_contribution"]
    value_vec = central_ser.loc[["shared", "moo_only", "limit_only"]].to_numpy() / 1000.
    total_float = float(value_vec.sum())
    running_float = 0.
    for index_int, value_float in enumerate(value_vec):
        next_float = running_float + value_float
        axis_obj.bar(index_int, abs(value_float), bottom=min(running_float, next_float),
                     color=GREEN_STR if value_float >= 0 else RED_STR, width=.62)
        axis_obj.text(index_int, max(running_float, next_float) + 27,
                      f"{value_float:+,.0f}k", ha="center", weight="bold", fontsize=13)
        axis_obj.plot([index_int + .31, index_int + .69], [next_float, next_float], color="#94a3b8", lw=1)
        running_float = next_float
    axis_obj.bar(3, total_float, color=BLUE_STR, width=.62)
    axis_obj.text(3, total_float + 27, f"{total_float:+,.0f}k", ha="center", weight="bold", fontsize=13)
    axis_obj.set_xticks(range(4), ["Same asset / entry date\nChange in trade P&L", "Only MOO trades\nTheir P&L is subtracted",
                                  "Only limit trades\nTheir P&L is added", "Final account\nvalue difference"])
    axis_obj.set_ylabel("Difference in dollars (thousands)")
    figure_obj.suptitle("Where did HPI's additional account value come from?", x=.125, y=.99,
                       ha="left", weight="bold", fontsize=16)
    axis_obj.set_ylim(0, max(np.cumsum(value_vec)) * 1.22)
    axis_obj.grid(axis="y", alpha=.17)
    axis_obj.set_axisbelow(True)
    figure_obj.text(.125, .925, "Limit -0.5% minus MOO | 2010-01-05 to 2026-09-14 | central costs", color="#64748b")
    figure_obj.text(.125, .005, "Includes dividends, fees and final open-position marks. Accounting bridge; not causal attribution.\nMOO-only trades are not all missed limits. Position sizes and accumulated capital differ.", fontsize=9, color="#64748b")
    figure_obj.subplots_adjust(bottom=.22, top=.84)
    save_figure(figure_obj, "01_accounting_bridge")
    if attribution_only_bool:
        return

    difference_df = pd.read_csv(table_path / "window_differences.csv")
    figure_obj, axis_vec = plt.subplots(2, 1, figsize=(11, 8.3), sharex=True)
    for axis_obj, layer_str in zip(axis_vec, ("central", "stress")):
        layer_df = difference_df[difference_df["layer"] == layer_str]
        restart_df = layer_df[layer_df["mode"] == "restart"].sort_values("start_year")
        continuous_df = layer_df[layer_df["mode"] == "continuous"].sort_values("start_year")
        value_vec = restart_df["cagr"].to_numpy() * 100.
        continuous_vec = continuous_df["cagr"].to_numpy() * 100.
        position_vec = np.arange(len(value_vec))
        axis_obj.bar(position_vec, value_vec, color=[GREEN_STR if value_float >= 0 else RED_STR for value_float in value_vec],
                     width=.62, label="Fresh $100k account")
        axis_obj.scatter(position_vec, continuous_vec, facecolor="white", edgecolor="#26394f", s=40,
                         linewidth=1.5, zorder=5, label="Continuous accounts, same dates")
        axis_obj.axhline(0, color="#334155", lw=1)
        axis_obj.set_ylabel("Annual return difference\n(percentage points)")
        axis_obj.set_title("Central costs" if layer_str == "central" else "Stress costs + stricter limit fills", loc="left", fontsize=12)
        axis_obj.grid(axis="y", alpha=.17)
        axis_obj.set_axisbelow(True)
        axis_obj.set_xticks(position_vec, restart_df["start_year"].astype(str))
        axis_obj.margins(y=.20)
        for index_int, value_float in enumerate(value_vec):
            anchor_float = (max(value_float, continuous_vec[index_int]) if value_float >= 0
                            else min(value_float, continuous_vec[index_int]))
            axis_obj.annotate(f"{value_float:+.1f}", (index_int, anchor_float), xytext=(0, 7 if value_float >= 0 else -13),
                              textcoords="offset points", ha="center", fontsize=9)
    axis_vec[0].legend(loc="upper left", fontsize=9, frameon=False)
    axis_vec[-1].set_xlabel("Starting year of the 3-calendar-year window")
    figure_obj.suptitle("Does the limit advantage survive a fresh start?", x=.125, ha="left", weight="bold", fontsize=16)
    figure_obj.text(.125, .015, "Above zero: limit -0.5% earned more. Below zero: MOO earned more.\nOverlapping historical windows; not independent trials or unseen validation.", fontsize=10, color="#64748b")
    figure_obj.subplots_adjust(top=.89, bottom=.13, hspace=.30)
    save_figure(figure_obj, "02_restart_return_difference")

    figure_obj, axis_vec = plt.subplots(2, 1, figsize=(11, 7.3), sharex=True)
    for axis_obj, metric_str, title_str in zip(axis_vec, ("max_drawdown", "exposure"),
            ("Drawdown difference: above zero means a smaller loss", "Average exposure difference: below zero means more cash")):
        for offset_float, layer_str, color_str in ((-.18, "central", BLUE_STR), (.18, "stress", ORANGE_STR)):
            selected_df = difference_df[(difference_df["mode"] == "restart") & (difference_df["layer"] == layer_str)].sort_values("start_year")
            axis_obj.bar(np.arange(len(selected_df)) + offset_float, selected_df[metric_str] * 100., width=.34,
                         label=layer_str.capitalize(), color=color_str)
        axis_obj.axhline(0, color="#334155", lw=1)
        axis_obj.set_title(title_str, loc="left", fontsize=12)
        axis_obj.set_ylabel("Limit minus MOO\n(percentage points)")
        axis_obj.grid(axis="y", alpha=.17)
        axis_obj.set_axisbelow(True)
    axis_vec[0].legend(frameon=False, fontsize=9)
    axis_vec[-1].set_xticks(np.arange(14), range(2010, 2024))
    axis_vec[-1].set_xlabel("Starting year of the 3-calendar-year window")
    figure_obj.suptitle("What changes in risk and invested capital?", x=.125, ha="left", weight="bold", fontsize=16)
    figure_obj.subplots_adjust(top=.89, bottom=.10, hspace=.35)
    save_figure(figure_obj, "03_restart_risk_exposure")

    benchmark_ser = pd.read_csv(table_path / "benchmark_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
    figure_obj, axis_vec = plt.subplots(3, 1, figsize=(11, 9.3), sharex=True)
    for policy_str, label_str, color_str in (("moo", "MOO", BLUE_STR), ("limit_0.5pct", "Limit -0.5%", ORANGE_STR),
                                             ("benchmark", "S&P 500 total return", "#8694a7")):
        if policy_str == "benchmark":
            return_ser = benchmark_ser
        else:
            daily_df = pd.read_csv(PARENT_PATH / f"runs/hpi235/central_{policy_str}/daily.csv", index_col="date", parse_dates=True)
            return_ser = nav_returns(daily_df["nav"])
        wealth_ser = (1. + return_ser).cumprod()
        # *** CRITICAL*** Running peak includes initialNAV=1; backward accounting only.
        peak_vec = np.maximum.accumulate(np.r_[1., wealth_ser.to_numpy()])[1:]
        axis_vec[0].plot(wealth_ser.index, wealth_ser * 100000., label=label_str, color=color_str, lw=1.5)
        axis_vec[1].plot(wealth_ser.index, wealth_ser / peak_vec - 1., color=color_str, lw=1)
        if policy_str != "benchmark":
            rolling_ser = pd.read_csv(table_path / f"central_{policy_str}_rolling_correlation.csv", index_col=0, parse_dates=True).iloc[:, 0]
            axis_vec[2].plot(rolling_ser.index, rolling_ser, color=color_str, lw=1)
    axis_vec[0].set_yscale("log")
    axis_vec[0].yaxis.set_major_formatter(FuncFormatter(lambda value_float, position_int: f"${value_float / 1000:,.0f}k"))
    axis_vec[0].set_ylabel("Account value\n(log scale)")
    axis_vec[0].legend(frameon=False, fontsize=9, loc="upper left")
    axis_vec[1].yaxis.set_major_formatter(PercentFormatter(1.))
    axis_vec[1].set_ylabel("Drawdown")
    axis_vec[2].set_ylabel("126-session\nmarket correlation")
    axis_vec[2].set_ylim(min(-.25, axis_vec[2].get_ylim()[0]), 1.)
    for axis_obj in axis_vec:
        axis_obj.grid(alpha=.17)
    figure_obj.suptitle("Original continuous accounts: return, risk and market exposure", x=.125, ha="left", weight="bold", fontsize=15)
    figure_obj.text(.125, .015, "Central costs | Same observed sessions | Benchmark is S&P 500 TOTALRETURN index, without trading fees.", fontsize=9, color="#64748b")
    figure_obj.subplots_adjust(top=.93, bottom=.07, hspace=.13)
    save_figure(figure_obj, "04_continuous_context")
    write_json(STUDY_PATH / "charts/chart_data_receipt.json", {"figures": 4, "restart_windows": 14,
               "return_difference_unit": "percentage_points", "accounting_unit": "USD", "context": "central original paths"})
    print("PASS four PNG and SVG figures", flush=True)


if __name__ == "__main__":
    import argparse
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("--attribution-only", action="store_true")
    args_obj = parser_obj.parse_args()
    main(args_obj.attribution_only)
