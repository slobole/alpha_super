"""Numbers for the report's defensive chapter (fund product menu), in one JSON file (2026-09-24).

Everything here re-derives from saved sleeve runs; nothing is tuned. It combines the simple-defensive study
(simple_defensive_study.py), its crisis / CVaR / correlation dossier (crisis_cvar_corr.py) and the owner's
amendment A1 (Low-touch Defensive without Inflation Compass), reading the final product weights from the
rebuilt menu (books/product_weights.csv) and the frozen ones from books_before_A1/.
Long window: 2008-03-04 on, BTAL sleeves replaced by their no-BTAL twins ONLY before 2012-10-02 (the menu's
own tables hold the stand-in for the whole period, so 'incl. 2008' figures here differ slightly).
Output: results/research/portfolio/simple_defensive_20260924/report_section.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fund_menu_20260923"))
import common  # noqa: E402
import simple_defensive_study as study  # noqa: E402

FUND_MENU_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "fund_menu_20260923"
LONG_STAND_IN_DICT = {"taa_btal_lin_qqq": "taa_lin_qqq", "btal_lin_spy": "nobtal_lin_spy", "taa_btal_tqqq": "taa_1n_qld"}
RATE_SHOCK_LIST = [("Taper tantrum (bonds)", "2013-05-02", "2013-09-05"), ("Reflation sell-off (bonds)", "2016-07-08", "2016-12-15"),
                   ("2022 rate shock", "2022-01-03", "2022-10-24"), ("Long-bond rout", "2023-07-31", "2023-10-19")]
TIER_DICT = {**study.TIER_BY_ALIAS_DICT, "ndx_vxn": "wired", "taa_btal_tqqq": "wired", "mosaic": "pm-ready", "infl_compass": "pm-ready",
             "eom_flow": "pm-ready", "hpi_vote": "wired", "dv2": "wired"}
POD_LABEL_DICT = {**study.LABEL_BY_ALIAS_DICT, "infl_compass": "Inflation Compass"}


def floor_usd(alias_str: str, floor_dict: dict) -> float:
    return float(floor_dict.get(alias_str, floor_dict["default"]))


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:study.END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:study.END_TS]
    for alias_str in ("btal_lin_spy", "nobtal_lin_spy"):
        sleeve_df[alias_str] = study.extra_return_ser(alias_str, sleeve_df.index)
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in LONG_STAND_IN_DICT.items():
        # *** CRITICAL*** the stand-in fills only the dates before the real sleeve exists.
        long_df.loc[long_df.index < study.CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < study.CUT_TS, stand_in_str]
    floor_dict = yaml.safe_load((FUND_MENU_SCRIPT_PATH / "report_text_static.yaml").read_text(encoding="utf-8"))["pod_floor_usd"]

    def weights_for(csv_path: Path, product_str: str) -> dict:
        frame_df = pd.read_csv(csv_path)
        rows_df = frame_df[(frame_df["product_id_str"] == product_str) & frame_df["weight_float"].notna()]
        return dict(zip(rows_df["alias_str"], rows_df["weight_float"].astype(float)))

    lt_def_new_dict = weights_for(common.STUDY_DIR_PATH / "books" / "product_weights.csv", "LT_DEF")
    lt_def_old_dict = weights_for(common.STUDY_DIR_PATH / "books_before_A1" / "product_weights.csv", "LT_DEF")
    def_dict = weights_for(common.STUDY_DIR_PATH / "books" / "product_weights.csv", "DEF")
    if "infl_compass" in lt_def_new_dict or "infl_compass" not in lt_def_old_dict:
        raise RuntimeError("Expected the rebuilt LT_DEF without Inflation Compass and the frozen one with it.")
    third_float = 1.0 / 3.0
    book_list = [
        {"id": "core5", "name": "CORE5 alone", "w": {"core5": 1.0}, "policy": "annual", "group": "simple"},
        {"id": "core5_btal", "name": "CORE5 + BTAL_QQQ", "w": {"core5": 0.5, "taa_btal_lin_qqq": 0.5}, "policy": "annual", "group": "simple"},
        {"id": "core5_btal_fi", "name": "CORE5 + BTAL_QQQ + Tactical FI", "w": {"core5": third_float, "taa_btal_lin_qqq": third_float, "tactical_fi": third_float},
         "policy": "annual", "group": "simple"},
        {"id": "core5_btal_disp", "name": "CORE5 + BTAL_QQQ + Dispersion", "w": {"core5": third_float, "taa_btal_lin_qqq": third_float, "disp_kie_ihi_sma": third_float},
         "policy": "annual", "group": "simple"},
        {"id": "core5_btal_down", "name": "CORE5 + BTAL_QQQ + Sector downshock", "w": {"core5": third_float, "taa_btal_lin_qqq": third_float, "sector_vox_iyr": third_float},
         "policy": "annual", "group": "simple"},
        {"id": "core5_btalspy", "name": "CORE5 + BTAL_SPY", "w": {"core5": 0.5, "btal_lin_spy": 0.5}, "policy": "annual", "group": "simple"},
        {"id": "lt_def", "name": "Low-touch Defensive (amended)", "w": lt_def_new_dict, "policy": "annual", "group": "menu"},
        {"id": "lt_def_frozen", "name": "Low-touch Defensive (as frozen)", "w": lt_def_old_dict, "policy": "annual", "group": "menu"},
        {"id": "def", "name": "Defensive (main line)", "w": def_dict, "policy": "annual", "group": "menu"},
        {"id": "ladder_1", "name": "ladder_1 (current)", "w": {"taa_btal_lin_qqq": 0.55, "sector_vox_iyr": 0.45}, "policy": "none", "group": "reference"},
    ]
    lt_variant_list = [
        {"id": "frozen", "name": "As frozen (with Inflation Compass)", "w": lt_def_old_dict},
        {"id": "amended", "name": "Amended: removed, the rest pro rata (chosen)", "w": lt_def_new_dict},
        {"id": "template", "name": "Removed by the template rule (slot to the 3x TAA)",
         "w": {"core5": 0.513, "tactical_fi": 0.256, "taa_btal_tqqq": 0.116, "ndx_vxn": 0.058, "mosaic": 0.057}},
        {"id": "swap", "name": "Swapped for BTAL_QQQ", "w": {("taa_btal_lin_qqq" if a == "infl_compass" else a): w for a, w in lt_def_old_dict.items()}},
    ]

    spx_long_ser = bench_df["SPXTR"].loc[study.LONG_TS:]
    episode_list = []
    for e in common.equity_drawdown_episode_list(spx_long_ser):
        peak_ts, trough_ts = pd.Timestamp(e["peak_date_str"]), pd.Timestamp(e["trough_date_str"])
        episode_list.append({"label": f"S&P 500 −{abs(e['spx_drawdown_float']) * 100:.0f}%: {peak_ts:%b %Y} to {trough_ts:%b %Y}",
                             "start": e["peak_date_str"], "end": e["trough_date_str"], "kind": "equity"})
    for label_str, start_str, end_str in RATE_SHOCK_LIST:
        episode_list.append({"label": label_str, "start": start_str, "end": end_str, "kind": "rates"})
    episode_list.sort(key=lambda d: d["start"])
    crisis_mask_ser = pd.Series(False, index=spx_long_ser.index)
    for e in episode_list:
        if e["kind"] == "equity":
            crisis_mask_ser.loc[e["start"]:e["end"]] = True
    stress_mask_ser = spx_long_ser <= spx_long_ser.quantile(0.10)

    trade_date_cache_dict: dict[str, set] = {}
    stressed_df = sleeve_df.copy()

    def trade_dates(alias_str: str) -> set:
        if alias_str not in trade_date_cache_dict:
            transaction_df = study.transaction_df_for(alias_str)
            trade_date_cache_dict[alias_str] = set(transaction_df.loc[(transaction_df["date"] >= study.CUT_TS) & (transaction_df["date"] <= study.END_TS), "date"])
            from evaluation import extra_slippage_cost_ser  # noqa: PLC0415 - fund-menu helper on the same path
            drag_ser = extra_slippage_cost_ser(transaction_df, study.nav_for(alias_str).loc[:study.END_TS], study.EXTRA_SLIPPAGE_PER_SIDE_FLOAT)
            live_mask = sleeve_df[alias_str].notna()
            stressed_df.loc[live_mask, alias_str] = sleeve_df.loc[live_mask, alias_str] - drag_ser.reindex(sleeve_df.index).fillna(0.0)[live_mask]
        return trade_date_cache_dict[alias_str]

    exact_years_float = len(sleeve_df.loc[study.CUT_TS:]) / 252.0

    def metrics(ser: pd.Series) -> dict:
        base_ts = sleeve_df.index[sleeve_df.index.get_loc(ser.index[0]) - 1]
        return common.metric_dict(ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)

    def evaluate(weight_dict: dict, policy_str: str) -> tuple[dict, pd.Series]:
        union_set = set().union(*(trade_dates(a) for a in weight_dict))
        exact_ser = common.book_return_ser(sleeve_df.loc[study.CUT_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        stress_ser = common.book_return_ser(stressed_df.loc[study.CUT_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        long_ser = common.book_return_ser(long_df.loc[study.LONG_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        me, ms, ml = metrics(exact_ser), metrics(stress_ser), metrics(long_ser)
        crisis_dict = {e["label"]: common.window_return_float(long_ser, e["start"], e["end"]) for e in episode_list}
        nav_ser = common.nav_from_return_ser(exact_ser)
        out = {"pods": len(weight_dict), "trade_days": len(union_set) / exact_years_float,
               "min_account": max(floor_usd(a, floor_dict) / w for a, w in weight_dict.items()),
               "wired_share": sum(w for a, w in weight_dict.items() if TIER_DICT.get(a) == "wired"),
               "cagr": me["cagr_float"], "vol": me["volatility_float"], "sharpe": me["sharpe_rf0_float"], "sharpe_excess": me["sharpe_excess_float"],
               "maxdd": me["max_drawdown_float"], "worst_12m": float((nav_ser / nav_ser.shift(252) - 1).dropna().min()),
               "sharpe_stress": ms["sharpe_rf0_float"],
               "cagr_long": ml["cagr_float"], "sharpe_long": ml["sharpe_rf0_float"], "maxdd_long": ml["max_drawdown_float"],
               "maxdd_long_trough": ml["max_dd_trough_date_str"], "cvar5_daily": ml["es95_daily_float"], "cvar5_21d": ml["es95_21d_float"],
               "corr_spx": float(long_ser.corr(spx_long_ser)), "corr_spx_crisis": float(long_ser[crisis_mask_ser].corr(spx_long_ser[crisis_mask_ser])),
               "worst_crisis": float(min(crisis_dict.values())), "crises": crisis_dict,
               "weights": {POD_LABEL_DICT.get(a, a): round(float(w), 4) for a, w in sorted(weight_dict.items(), key=lambda kv: -kv[1])}}
        return out, long_ser

    book_out_list = []
    for book_dict in book_list:
        result_dict, _ = evaluate(book_dict["w"], book_dict["policy"])
        book_out_list.append({"id": book_dict["id"], "name": book_dict["name"], "group": book_dict["group"], **result_dict})
    variant_out_list = []
    for variant_dict in lt_variant_list:
        result_dict, _ = evaluate(variant_dict["w"], "annual")
        variant_out_list.append({"id": variant_dict["id"], "name": variant_dict["name"], **result_dict})
    spx_crisis_dict = {e["label"]: common.window_return_float(spx_long_ser, e["start"], e["end"]) for e in episode_list}

    pod_alias_list = ["core5", "taa_btal_lin_qqq", "disp_kie_ihi_sma", "tactical_fi", "sector_vox_iyr", "infl_compass"]
    pod_df = long_df.loc[study.LONG_TS:, pod_alias_list]
    corr_all_df = pod_df.corr()
    corr_stress_df = pod_df[stress_mask_ser.reindex(pod_df.index).fillna(False)].corr()
    sleeve_row_list = []
    for alias_str in ["core5", "taa_btal_lin_qqq", "btal_lin_spy", "sector_vox_iyr", "tactical_fi", "disp_kie_ihi_sma", "infl_compass"]:
        ser = long_df[alias_str].loc[study.LONG_TS:].dropna()
        detail_dict = common.max_drawdown_detail_dict(common.nav_from_return_ser(ser))
        sleeve_row_list.append({"alias": alias_str, "name": POD_LABEL_DICT.get(alias_str, alias_str), "tier": TIER_DICT.get(alias_str, ""),
                                "proxy_before_2012": alias_str in LONG_STAND_IN_DICT,
                                "y2008": float((1.0 + ser.loc["2008"]).prod() - 1.0),
                                "gfc": common.window_return_float(ser, episode_list[0]["start"], episode_list[0]["end"]),
                                "maxdd": detail_dict["max_drawdown_float"], "maxdd_peak": detail_dict["max_dd_peak_date_str"],
                                "maxdd_trough": detail_dict["max_dd_trough_date_str"]})
    payload_dict = {
        "windows": {"exact": [study.CUT_TS.date().isoformat(), study.END_TS.date().isoformat()],
                    "long": [study.LONG_TS.date().isoformat(), study.END_TS.date().isoformat()]},
        "books": book_out_list, "lt_def_variants": variant_out_list,
        "episodes": episode_list, "spx_crises": spx_crisis_dict,
        "pod_corr": {"labels": [POD_LABEL_DICT[a] for a in pod_alias_list],
                     "all": np.round(corr_all_df.to_numpy(), 3).tolist(), "stress": np.round(corr_stress_df.to_numpy(), 3).tolist()},
        "sleeves_2008": sleeve_row_list,
        "search": pd.read_csv(study.STUDY_DIR_PATH / "books_equal_qualification.csv", index_col=0)[
            ["pods", "cagr", "sharpe", "sharpe_excess", "maxdd_long", "qualifies"]].reset_index().to_dict(orient="records"),
    }
    output_path = study.STUDY_DIR_PATH / "report_section.json"
    output_path.write_text(json.dumps(payload_dict, indent=1, default=float), encoding="utf-8")
    print(f"wrote {output_path}")
    for row in book_out_list + variant_out_list:
        print(f"{row['name']:<52} pods {row['pods']}  min ${row['min_account']:>9,.0f}  days {row['trade_days']:5.1f}  CAGR {row['cagr']:.3f}  "
              f"Sh {row['sharpe']:.2f}/{row['sharpe_excess']:.2f}  DD {row['maxdd']:.3f}  DD08 {row['maxdd_long']:.3f}  CVaR21 {row['cvar5_21d']:.3f}  "
              f"worst {row['worst_crisis']:.3f}  corrCrisis {row['corr_spx_crisis']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
