"""Decision tables and figures for the frozen proxy architecture study."""
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.tail_hedge.run_ladder_tail_hedge_study import MIX_DICT, funded_weights, exact_common_calendar
from strategies.tail_hedge.run_proxy_tail_architecture_study import OUTPUT_PATH, HEDGE_PATH, policy_path, bounded_event_dict
from strategies.tail_hedge.run_tail_hedge_vanilla_study import metric_dict, drawdown_ser, crisis_tables

BOOK_LABEL_DICT = {"ladder_1_defensive_proxy_2008": "L1 proxy", "ladder_2_balanced_proxy_2008": "L2 proxy",
    "ladder_3b_growth_2x": "L3b 2x", "ladder_1_defensive": "L1 actual", "ladder_2_balanced": "L2 actual", "ladder_4_growth": "L4 actual"}
COLOR_DICT = {"Baseline": "#1f2937", "Core05": "#4292c6", "Core10": "#08519c", "VIXM05": "#e6550d", "VIXM10": "#a63603", "Blend05": "#756bb1", "Blend10": "#54278f", "SHY05": "#74c476", "SHY10": "#238b45", "SPY": "#aaaaaa"}


def load_path_df(book_str: str, panel_str: str, policy_str: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_PATH / "paths_verified" / f"{book_str}_{panel_str}_{policy_str}.csv", index_col=0, parse_dates=True)


def build_evidence() -> None:
    table_path = OUTPUT_PATH / "tables_verified"
    chart_path = OUTPUT_PATH / "charts"
    chart_path.mkdir(exist_ok=True)
    metric_df = pd.read_csv(table_path / "metrics.csv")
    crisis_df = pd.read_csv(table_path / "crises.csv")
    budget_df = pd.read_csv(table_path / "hedge_budget.csv")
    cohort_df = pd.read_csv(table_path / "cohorts5y.csv")
    vanilla_df = metric_df.loc[metric_df.cost == "vanilla"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    long_df = load_path_df("ladder_3b_growth_2x", "long_core", "annual_hedge_only")
    figure_obj, axis_vec = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for mix_str in ("Baseline", "Core05", "Core10", "SHY10", "SPY"):
        axis_vec[0].plot(long_df.index, 100*(1+long_df[mix_str]).cumprod(), color=COLOR_DICT[mix_str], label=mix_str)
        axis_vec[1].plot(long_df.index, 100*drawdown_ser(long_df[mix_str]), color=COLOR_DICT[mix_str])
    axis_vec[0].set(title="L3b real-QLD book: annual hedge-only budget, 2008-03 to 2026-08", ylabel="Wealth (100, log scale)", yscale="log")
    axis_vec[0].legend(ncol=5); axis_vec[1].set_ylabel("Drawdown (%)")
    for axis_obj in axis_vec: axis_obj.grid(alpha=.2)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path / "long_history_equity_drawdown.png", dpi=150); plt.close(figure_obj)

    figure_obj, axis_vec = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for axis_obj, (book_str, book_label_str) in zip(axis_vec.flat, BOOK_LABEL_DICT.items()):
        selected_df = vanilla_df.loc[(vanilla_df.book == book_str) & (vanilla_df.panel == "common") & (vanilla_df.policy == "annual_hedge_only") & (vanilla_df.mix != "Baseline")]
        for row_obj in selected_df.itertuples():
            axis_obj.scatter(-100*row_obj.delta_cagr, 100*row_obj.delta_mdd, color=COLOR_DICT[row_obj.mix], s=55,
                marker="o" if row_obj.mix.endswith("05") else "s", label=row_obj.mix)
        axis_obj.axvline(1, color="gray", linestyle="--", linewidth=.8)
        axis_obj.axhline(1, color="gray", linestyle="--", linewidth=.8)
        axis_obj.set_title(book_label_str); axis_obj.grid(alpha=.2)
    figure_obj.suptitle("Maintained hedge budget: annual return cost vs drawdown improvement\nCommon real-VIXM period; exact start/end differ by book", fontsize=13)
    figure_obj.supxlabel("Annual CAGR drag (percentage points; lower is better)")
    figure_obj.supylabel("Max drawdown improvement (percentage points)")
    handle_list, label_list = axis_vec[0, 0].get_legend_handles_labels()
    figure_obj.legend(handle_list, label_list, loc="upper center", ncol=8, bbox_to_anchor=(.5, .94), fontsize=9)
    figure_obj.tight_layout(rect=(0, 0, 1, .9)); figure_obj.savefig(chart_path / "maintained_budget_tradeoff.png", dpi=150); plt.close(figure_obj)

    selected_crisis_list = ["2008_Lehman", "2011_Euro", "2018_Q4", "2020_Covid", "2022_Bear", "2024_Yen", "2025_Tariff"]
    heat_row_list, heat_label_list = [], []
    for book_str in ("ladder_2_balanced_proxy_2008", "ladder_3b_growth_2x", "ladder_4_growth"):
        for mix_str in ("Core05", "VIXM05", "Blend05"):
            heat_vec = []
            for crisis_str in selected_crisis_list:
                panel_str = "long_core" if crisis_str == "2008_Lehman" else "common"
                selected_df = crisis_df.loc[(crisis_df.book == book_str) & (crisis_df.panel == panel_str) & (crisis_df.policy == "annual_hedge_only") & (crisis_df.crisis == crisis_str) & (crisis_df.status == "complete")].set_index("series")
                heat_vec.append(100*(selected_df.loc[mix_str, "return"]-selected_df.loc["Baseline", "return"]) if mix_str in selected_df.index else np.nan)
            heat_row_list.append(heat_vec); heat_label_list.append(f"{BOOK_LABEL_DICT[book_str]} / {mix_str}")
    heat_mat = np.array(heat_row_list)
    figure_obj, axis_obj = plt.subplots(figsize=(12, 6))
    limit_float = max(1., np.nanmax(np.abs(heat_mat)))
    plot_obj = axis_obj.imshow(heat_mat, cmap="RdYlGn", vmin=-limit_float, vmax=limit_float, aspect="auto")
    axis_obj.set_xticks(range(len(selected_crisis_list)), [label_str.replace("_", " ") for label_str in selected_crisis_list], rotation=25, ha="right")
    axis_obj.set_yticks(range(len(heat_label_list)), heat_label_list)
    for row_int in range(heat_mat.shape[0]):
        for col_int in range(heat_mat.shape[1]):
            value_float = heat_mat[row_int, col_int]
            axis_obj.text(col_int, row_int, "N/A" if np.isnan(value_float) else f"{value_float:+.2f}", ha="center", va="center", fontsize=9)
    axis_obj.set_title("Crisis return improvement vs matched base (percentage points)\n5% total hedge budget, annual hedge-only reset; no crisis restart")
    figure_obj.colorbar(plot_obj, ax=axis_obj, shrink=.8)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path / "crisis_value_add.png", dpi=150); plt.close(figure_obj)

    provenance_dict = json.loads((OUTPUT_PATH / "manifest.json").read_text())
    input_df = pd.read_csv(OUTPUT_PATH / "inputs_verified/ladder_4_growth_common.csv", index_col=0, parse_dates=True)
    config_dict = provenance_dict["sources"]["ladder_4_growth"]["config"]
    weight_ser = pd.Series({pod_dict["pod_id_str"]: pod_dict["weight_float"] for pod_dict in config_dict["pods"]})
    import_dict = {pod_dict["pod_id_str"]: pod_dict["strategy_import_str"] for pod_dict in config_dict["pods"]}
    target_ser = funded_weights(weight_ser, import_dict, float(config_dict["capital_base_float"]), MIX_DICT["VIXM10"])
    figure_obj, axis_vec = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    rolling_df = pd.DataFrame(index=input_df.index)
    for policy_str in ("drift", "annual_all", "annual_hedge_only"):
        return_ser, drift_df = policy_path(input_df, target_ser, policy_str, ["VIXM"])
        axis_vec[0].plot(input_df.index, 100*drift_df.VIXM, label=policy_str)
    path_df = load_path_df("ladder_4_growth", "common", "annual_hedge_only")
    for mix_str in ("Baseline", "VIXM05", "Blend05"):
        # *** CRITICAL*** Trailing 126-session descriptive correlation only.
        rolling_df[mix_str] = path_df[mix_str].rolling(126, min_periods=126).corr(path_df.SPY)
        axis_vec[1].plot(rolling_df.index, rolling_df[mix_str], label=mix_str, color=COLOR_DICT[mix_str], linewidth=.8)
    rolling_df.to_csv(table_path / "l4_rolling126_market_correlation.csv")
    axis_vec[0].set(title="L4: a 10% initial allocation is not a maintained 10% hedge", ylabel="VIXM sleeve NAV weight (%)")
    axis_vec[1].set(ylabel="126-session SPY correlation", ylim=(-1, 1))
    for axis_obj in axis_vec: axis_obj.legend(ncol=3); axis_obj.grid(alpha=.2)
    figure_obj.tight_layout(); figure_obj.savefig(chart_path / "budget_and_market_correlation.png", dpi=150); plt.close(figure_obj)

    # Cohorts are overlapping allocation-start sensitivities, not OOS trials.
    cohort_summary_df = cohort_df.groupby(["book", "panel", "mix"]).agg(
        cohorts=("year", "size"), median_drag=("delta_cagr", "median"), worst_drag=("delta_cagr", "min"),
        fraction_drag_within1pp=("delta_cagr", lambda value_ser: float((value_ser >= -.01).mean())),
        median_mdd_gain=("delta_mdd", "median"), fraction_mdd_gain1pp=("delta_mdd", lambda value_ser: float((value_ser >= .01).mean())))
    cohort_summary_df.to_csv(table_path / "cohort_summary.csv")
    gates_df = metric_df.loc[metric_df.mix != "Baseline"].groupby(["cost", "policy"]).gate_pass.agg(["sum", "count"])
    gates_df.to_csv(table_path / "gate_counts.csv")

    # Authentic original ETN proxy remains separate; no return stitching.
    vxz_ser = pd.read_csv(OUTPUT_PATH / "vxz_proxy_verified/returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
    hedge_df = pd.read_csv(HEDGE_PATH / "full_returns.csv", index_col=0, parse_dates=True)
    proxy_row_list = [{"series": "VXZ_proxy", "sample": "full_2009_2019", **metric_dict(vxz_ser, hedge_df.loc[vxz_ser.index, "SPY"])}]
    actual_ser = hedge_df.VIXM.loc[hedge_df.VIXM.first_valid_index():]
    overlap_idx = exact_common_calendar(vxz_ser.index, actual_ser.index)
    for label_str, return_ser in (("VXZ_proxy", vxz_ser), ("VIXM", actual_ser)):
        proxy_row_list.append({"series": label_str, "sample": "exact_overlap", **metric_dict(return_ser.loc[overlap_idx], hedge_df.loc[overlap_idx, "SPY"]),
            "proxy_to_actual_correlation": vxz_ser.loc[overlap_idx].corr(actual_ser.loc[overlap_idx])})
    for period_str, start_str, end_str in (("pre_vixm", "2009-02-02", "2011-02-28"), ("2010_flash", "2010-04-23", "2010-07-02")):
        proxy_row_list.append({"series": "VXZ_proxy", "sample": period_str, **bounded_event_dict(vxz_ser, start_str, end_str)})
    pd.DataFrame(proxy_row_list).to_csv(table_path / "vxz_proxy_evidence.csv", index=False)
    crisis_tables(pd.DataFrame({"VXZ_proxy": vxz_ser}), hedge_df.SPY).to_csv(table_path / "vxz_proxy_crises.csv", index=False)
    relation_row_list = []
    for book_str in BOOK_LABEL_DICT:
        path_df = load_path_df(book_str, "common", "annual_hedge_only").iloc[1:]
        input_df = pd.read_csv(OUTPUT_PATH / "inputs_verified" / f"{book_str}_common.csv", index_col=0, parse_dates=True).loc[path_df.index]
        tail_idx = path_df.Baseline.nsmallest(int(np.ceil(.01*len(path_df)))).index
        for hedge_str in ("Core", "VIXM"):
            relation_row_list.append({"book": book_str, "hedge": hedge_str,
                "start": str(path_df.index[0].date()), "end": str(path_df.index[-1].date()),
                "corr_baseline": input_df[hedge_str].corr(path_df.Baseline),
                "beta_to_baseline": input_df[hedge_str].cov(path_df.Baseline)/path_df.Baseline.var(),
                "baseline_worst1_hedge_mean": input_df.loc[tail_idx, hedge_str].mean(),
                "baseline_worst1_hedge_positive_fraction": (input_df.loc[tail_idx, hedge_str] > 0).mean()})
    pd.DataFrame(relation_row_list).to_csv(table_path / "book_hedge_relationships.csv", index=False)
    fingerprint_dict = {}
    for file_path in list(table_path.glob("*.csv"))+list(chart_path.glob("*.png"))+[Path(__file__), OUTPUT_PATH / "manifest.json", OUTPUT_PATH / "vxz_proxy_verified/manifest.json"]:
        fingerprint_dict[str(file_path)] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    (OUTPUT_PATH / "evidence_fingerprints.json").write_text(json.dumps(fingerprint_dict, indent=2), encoding="utf-8")
    print(gates_df.to_string())
    print(pd.DataFrame(proxy_row_list).to_string(index=False))
    print(vanilla_df.loc[(vanilla_df.policy == "annual_hedge_only") & (vanilla_df.mix.isin(["Baseline", "Core05", "VIXM05", "Blend05"]))][["book", "panel", "mix", "cagr", "mdd", "delta_cagr", "delta_mdd", "post2020_delta_cagr", "gate_pass"]].to_string(index=False))


if __name__ == "__main__":
    build_evidence()
