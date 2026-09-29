"""Charts for the IPO / split all-time-high report (static PNG, light surface).

Usage: python charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE_PATH = Path(__file__).resolve().parent
if str(HERE_PATH) not in sys.path:
    sys.path.insert(0, str(HERE_PATH))

import common  # noqa: E402
import run_pods  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side_str in ["top", "right"]:
        ax.spines[side_str].set_visible(False)
    for side_str in ["left", "bottom"]:
        ax.spines[side_str].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_stage_a() -> None:
    year_df = pd.read_csv(common.RESULTS_DIR_PATH / "stage_a_by_year.csv")
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    style(ax)
    label_dict = {"IPO_ATH": "IPO at all-time high", "SPLIT_ATH": "Post-split at all-time high", "BASE_ATH": "Other stocks at all-time high"}
    for color_str, pop_str in zip(SERIES, label_dict):
        sub_df = year_df[(year_df["population"] == pop_str) & (year_df["date"] <= 2025)]
        ax.plot(sub_df["date"], sub_df["X20_mean"] * 100, color=color_str, linewidth=2, marker="o", markersize=4, label=label_dict[pop_str])
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_ylabel("20-day return minus SPY, % (mean per year)", color=INK_2, fontsize=9)
    ax.set_title("After 1999 the IPO all-time-high edge is gone", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(common.CHART_DIR_PATH / "stage_a_excess_by_year.png", dpi=150)
    plt.close(fig)


def chart_equity() -> None:
    sweep_df = pd.read_parquet(common.RESULTS_DIR_PATH / "pod_returns_sweep.parquet")
    la_df = pd.read_parquet(common.RESULTS_DIR_PATH / "posthoc_lookahead_returns_sweep.parquet")
    spy_ser = pd.read_parquet(common.CACHE_DIR_PATH / "spy_tr.parquet")["Close"].pct_change()
    frame_dict = {
        "IPO rule, point-in-time universe": sweep_df["IPO_ATH|N20|P20L10|base"],
        "Split rule, point-in-time universe": sweep_df["SPLIT_ATH|N20|P20L10|base"],
        "IPO rule, LOOK-AHEAD universe (latest market cap)": la_df["LA_BIG|E2"],
        "SPY total return": spy_ser,
    }
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    style(ax)
    for (label_str, ret_ser), color_str in zip(frame_dict.items(), SERIES):
        ret_ser = ret_ser.loc["2001-01-01":"2026-08-19"].fillna(0.0)
        wealth_ser = (1 + ret_ser).cumprod()
        ax.plot(wealth_ser.index, wealth_ser, color=color_str, linewidth=2 if "LOOK" not in label_str else 1.6,
                linestyle="--" if "LOOK" in label_str else "-", label=label_str)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 since 2001 (log)", color=INK_2, fontsize=9)
    ax.set_title("The talk's result needs a universe chosen with hindsight", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(common.CHART_DIR_PATH / "equity_2001_2026.png", dpi=150)
    plt.close(fig)


def chart_grid() -> None:
    stage_b_dict = json.loads((common.RESULTS_DIR_PATH / "stage_b.json").read_text())
    c_bil_float = stage_b_dict["controls"]["C_BIL"]["G-FULL"]["sharpe"]
    cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    col_list = [f"P{round(p * 100)}L{round(l_ * 100)}" for p, l_ in run_pods.PAIR_LIST]
    for ax, pop_str, title_str in zip(axes, run_pods.GRID_POPULATION_LIST, ["IPO", "Split"]):
        value_arr = np.array([[stage_b_dict["books"][f"{pop_str}|N{n}|{c}|base"]["G-FULL"]["sharpe"] - c_bil_float for c in col_list]
                              for n in run_pods.SLOT_LIST])
        ax.imshow(value_arr, cmap=cmap, vmin=-0.15, vmax=0.15, aspect="auto")
        for i in range(value_arr.shape[0]):
            for j in range(value_arr.shape[1]):
                ax.text(j, i, f"{value_arr[i, j]:+.3f}", ha="center", va="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(col_list)), [c.replace("P", "+").replace("L", "% / -") + "%" for c in col_list], fontsize=8, color=INK_2)
        ax.set_yticks(range(len(run_pods.SLOT_LIST)), [f"{n} slots" for n in run_pods.SLOT_LIST], fontsize=8, color=INK_2)
        ax.set_title(f"{title_str}: book Sharpe minus T-bill slot, 2012-26", color=INK, fontsize=10, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(common.CHART_DIR_PATH / "grid_book_vs_tbills.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    common.CHART_DIR_PATH.mkdir(parents=True, exist_ok=True)
    chart_stage_a()
    chart_equity()
    chart_grid()
    common.log_progress("charts: done")
