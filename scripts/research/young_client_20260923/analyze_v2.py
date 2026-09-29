"""Part 2 analysis: TAA choice (rank vs 1/N) and whether any second sleeve earns its place.

Window: 2012-10-02 -> 2026-08-19 (plan_v2_frozen.yaml). Pods at their real dollar sizes.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_PATH.parents[0] / "fund_menu_20260923"))
import common  # noqa: E402

STUDY_DIR_PATH = common.REPO_ROOT_PATH / "results" / "research" / "portfolio" / "young_client_taa_core5_20260923"
SOURCE_DIR_PATH = STUDY_DIR_PATH / "sources"
MENU_SOURCE_DIR_PATH = common.STUDY_DIR_PATH / "sources"   # fund-menu study (Tactical FI governed run)
START_TS, END_TS = pd.Timestamp("2012-10-02"), pd.Timestamp("2026-08-19")
ACCOUNT_FLOAT = 26645.0
TAA_LIST = ["taa_btal_tqqq", "taa_btal_1n_tqqq"]
TAA_LABEL = {"taa_btal_tqqq": "TAA rank-weighted", "taa_btal_1n_tqqq": "TAA 1/N"}
PARTNER_LIST = ["core5", "trinity", "tactical_fi", "infl_compass", "sector_vox_iyr", "disp_kie_ihi_sma", "vixm", "eom_flow", "taa_btal_lin_qqq", "ndx_vxn"]
SHARE_TO_CAPITAL = {0.20: (21316, 5329), 0.35: (17319, 9326), 0.50: (13322, 13322)}


def path_df_for(alias_str: str, capital_float: float) -> pd.DataFrame:
    if alias_str == "tactical_fi":   # governed saved run (fresh runs blocked by the frozen fingerprint); ETF sleeve, scale-free
        return pd.read_csv(MENU_SOURCE_DIR_PATH / "tactical_fi__path.csv.gz", index_col="date", parse_dates=True)
    return pd.read_csv(SOURCE_DIR_PATH / f"{alias_str}__{int(round(capital_float))}__path.csv.gz", index_col="date", parse_dates=True)


def return_ser_for(alias_str: str, capital_float: float) -> pd.Series:
    path_df = path_df_for(alias_str, capital_float)
    nav_ser = path_df["total_value_float"].astype(float)
    invested_ser = path_df["portfolio_value_float"].abs() > 1e-9
    first_int = nav_ser.index.get_loc(invested_ser[invested_ser].index[0])
    return nav_ser.iloc[max(first_int - 1, 0):].pct_change()


def main() -> int:  # noqa: C901
    part_str = sys.argv[1] if len(sys.argv) > 1 else "all"
    session_index = return_ser_for("taa_btal_tqqq", ACCOUNT_FLOAT).index
    session_index = session_index[(session_index >= START_TS - pd.Timedelta(days=10)) & (session_index <= END_TS)]
    bench_df = common.build_benchmark_return_df(session_index, session_index[0].date().isoformat(), END_TS.date().isoformat())
    window_index = session_index[session_index >= START_TS]
    base_ts = session_index[session_index.get_loc(window_index[0]) - 1]
    tbill_ser = bench_df["TBILL"].reindex(window_index).fillna(0.0)

    def in_window(ser: pd.Series) -> pd.Series:
        return ser.reindex(window_index)

    def metrics(ser: pd.Series, index=None) -> dict:
        index = window_index if index is None else index
        sliced_ser = ser.reindex(index)
        local_base_ts = session_index[session_index.get_loc(index[0]) - 1]
        out = common.metric_dict(sliced_ser, bench_df["SPXTR"], bench_df["TBILL"], local_base_ts)
        nav_ser = common.nav_from_return_ser(sliced_ser)
        out["worst_12m_float"] = float((nav_ser / nav_ser.shift(252) - 1).dropna().min())
        weekly_ser = (1 + sliced_ser).resample("W-FRI").prod() - 1
        out["worst_week_float"] = float(weekly_ser.min())
        return out

    def cash_mix_ser(ser: pd.Series, risky_share_float: float) -> pd.Series:
        pair_df = pd.DataFrame({"risky": in_window(ser), "tbill": tbill_ser})
        mixed_ser, _ = common.book_return_ser(pair_df, {"risky": risky_share_float, "tbill": 1 - risky_share_float}, "annual")
        return mixed_ser

    def cash_mix_at_vol(ser: pd.Series, target_vol_float: float) -> pd.Series:
        low_float, high_float = 0.0, 1.0
        for _ in range(40):
            mid_float = (low_float + high_float) / 2
            if cash_mix_ser(ser, mid_float).std() * np.sqrt(252) < target_vol_float:
                low_float = mid_float
            else:
                high_float = mid_float
        return cash_mix_ser(ser, (low_float + high_float) / 2)

    report_dict: dict = {}
    taa_ret = {a: return_ser_for(a, ACCOUNT_FLOAT) for a in TAA_LIST}

    if part_str in ("a", "all"):
        # A1 metrics
        rows = []
        for alias_str in TAA_LIST:
            m = metrics(taa_ret[alias_str])
            y = in_window(taa_ret[alias_str])
            x = in_window(bench_df["QQQ"])
            beta_float = float(np.cov(y, x)[0, 1] / np.var(x, ddof=1))
            alpha_float = float((y.mean() - beta_float * x.mean()) * 252)
            rows.append({"variant": TAA_LABEL[alias_str], "cagr": m["cagr_float"], "vol": m["volatility_float"], "sharpe": m["sharpe_rf0_float"],
                         "sharpe_excess": m["sharpe_excess_float"], "maxdd": m["max_drawdown_float"], "worst_12m": m["worst_12m_float"],
                         "worst_week": m["worst_week_float"], "beta_qqq": beta_float, "alpha_vs_qqq_ann": alpha_float,
                         "longest_underwater_days": m["longest_underwater_days_int"]})
        a_df = pd.DataFrame(rows).set_index("variant")
        # A2 equal-risk: 1/N mixed with T-bills down to the rank variant's volatility
        rank_vol_float = a_df.loc["TAA rank-weighted", "vol"]
        derisked_ser = cash_mix_at_vol(taa_ret["taa_btal_1n_tqqq"], rank_vol_float)
        derisked_m = metrics(derisked_ser)
        # A3 TQQQ weight after each rebalance
        tqqq_rows = []
        for alias_str in TAA_LIST:
            path_df = path_df_for(alias_str, ACCOUNT_FLOAT)
            tx_df = pd.read_csv(SOURCE_DIR_PATH / f"{alias_str}__{int(ACCOUNT_FLOAT)}__transactions.csv.gz", parse_dates=["date"])
            tx_df = tx_df[tx_df["date"] >= START_TS - pd.Timedelta(days=5)]
            holdings = {}
            weight_list = []
            for date_ts, day_df in tx_df.groupby("date"):
                for r in day_df.itertuples():
                    holdings[r.asset_str] = holdings.get(r.asset_str, 0.0) + r.amount_float
                last_price = day_df[day_df["asset_str"] == "TQQQ"]["fill_price_float"]
                if holdings.get("TQQQ", 0) > 0 and len(last_price):
                    nav_float = float(path_df["total_value_float"].asof(date_ts))
                    weight_list.append(holdings["TQQQ"] * float(last_price.iloc[-1]) / nav_float)
                elif holdings.get("TQQQ", 0) <= 0:
                    weight_list.append(0.0)
            weight_arr = np.array(weight_list)
            tqqq_rows.append({"variant": TAA_LABEL[alias_str], "mean_tqqq_weight_after_trades": float(weight_arr.mean()),
                              "median": float(np.median(weight_arr)), "max": float(weight_arr.max()),
                              "share_of_rebalances_with_tqqq_ge_50pct": float((weight_arr >= 0.5).mean())})
        tqqq_df = pd.DataFrame(tqqq_rows).set_index("variant")
        # A4 crises and calendar years
        episode_list = common.equity_drawdown_episode_list(in_window(bench_df["SPXTR"]), -0.10)
        crisis_rows = []
        for e in episode_list:
            row = {"episode": f"{e['peak_date_str']} to {e['trough_date_str']}", "S&P": e["spx_drawdown_float"]}
            for alias_str in TAA_LIST:
                row[TAA_LABEL[alias_str]] = common.window_return_float(in_window(taa_ret[alias_str]), e["peak_date_str"], e["trough_date_str"])
            row["QQQ"] = common.window_return_float(in_window(bench_df["QQQ"]), e["peak_date_str"], e["trough_date_str"])
            crisis_rows.append(row)
        years_df = pd.DataFrame({TAA_LABEL[a]: (1 + in_window(taa_ret[a])).resample("YE").prod() - 1 for a in TAA_LIST}
                                | {"QQQ": (1 + in_window(bench_df["QQQ"])).resample("YE").prod() - 1}).rename(index=lambda t: t.year)
        pd.set_option("display.width", 250)
        print(a_df.round(3).to_string())
        print(f"\n1/N mixed with T-bills to the rank variant's volatility ({rank_vol_float:.3f}): CAGR {derisked_m['cagr_float']:.3f}, maxDD {derisked_m['max_drawdown_float']:.3f}, sharpe {derisked_m['sharpe_rf0_float']:.3f}")
        print(tqqq_df.round(3).to_string())
        print(pd.DataFrame(crisis_rows).round(3).to_string(index=False))
        print(years_df.round(3).to_string())
        report_dict["part_a"] = {"metrics": a_df.to_dict(), "derisked_1n": {k: derisked_m[k] for k in ("cagr_float", "volatility_float", "max_drawdown_float", "sharpe_rf0_float")},
                                 "tqqq": tqqq_df.to_dict(), "crises": crisis_rows, "years": years_df.to_dict()}
        pd.concat([a_df, tqqq_df], axis=1).to_csv(STUDY_DIR_PATH / "v2_taa_choice.csv", float_format="%.6g")
        pd.DataFrame(crisis_rows).to_csv(STUDY_DIR_PATH / "v2_taa_crises.csv", index=False, float_format="%.6g")
        years_df.to_csv(STUDY_DIR_PATH / "v2_taa_years.csv", float_format="%.6g")

    if part_str in ("b", "all"):
        half_int = len(window_index) // 2
        halves = {"H1": window_index[:half_int], "H2": window_index[half_int:]}
        partner_rows, blend_rows = [], []
        for base_str in TAA_LIST:
            base_ser = taa_ret[base_str]
            base_m = metrics(base_ser)
            for partner_str in PARTNER_LIST:
                partner_full_ser = return_ser_for(partner_str, ACCOUNT_FLOAT)
                if in_window(partner_full_ser).isna().any():
                    continue
                pm = metrics(partner_full_ser)
                corr_float = float(in_window(partner_full_ser).corr(in_window(base_ser)))
                corr_m_float = float(((1 + in_window(partner_full_ser)).resample("ME").prod()).corr((1 + in_window(base_ser)).resample("ME").prod()))
                hurdle_float = corr_float * base_m["sharpe_excess_float"]
                partner_rows.append({"base": TAA_LABEL[base_str], "partner": partner_str, "partner_cagr": pm["cagr_float"], "partner_vol": pm["volatility_float"],
                                     "partner_sharpe_excess": pm["sharpe_excess_float"], "corr_daily": corr_float, "corr_monthly": corr_m_float,
                                     "hurdle": hurdle_float, "passes_marginal_test": pm["sharpe_excess_float"] > hurdle_float})
                for share_float, (taa_cap, partner_cap) in SHARE_TO_CAPITAL.items():
                    pair_df = pd.DataFrame({"taa": in_window(return_ser_for(base_str, taa_cap)), "partner": in_window(return_ser_for(partner_str, partner_cap))})
                    blend_ser, _ = common.book_return_ser(pair_df, {"taa": 1 - share_float, "partner": share_float}, "annual")
                    bm = metrics(blend_ser)
                    cash_ser = cash_mix_at_vol(base_ser, bm["volatility_float"])
                    cm = metrics(cash_ser)
                    half_value = {}
                    for half_str, half_index in halves.items():
                        half_value[half_str] = metrics(blend_ser, half_index)["cagr_float"] - metrics(cash_ser, half_index)["cagr_float"]
                    pod_drag_float = metrics(return_ser_for(partner_str, ACCOUNT_FLOAT))["cagr_float"] - metrics(return_ser_for(partner_str, partner_cap))["cagr_float"]
                    blend_rows.append({"base": TAA_LABEL[base_str], "partner": partner_str, "share": share_float, "partner_pod_usd": partner_cap,
                                       "cagr": bm["cagr_float"], "vol": bm["volatility_float"], "sharpe": bm["sharpe_rf0_float"], "maxdd": bm["max_drawdown_float"],
                                       "worst_12m": bm["worst_12m_float"], "cash_mix_cagr_same_vol": cm["cagr_float"], "cash_mix_maxdd": cm["max_drawdown_float"],
                                       "value_add_vs_tbills": bm["cagr_float"] - cm["cagr_float"], "value_add_H1": half_value["H1"], "value_add_H2": half_value["H2"],
                                       "partner_small_pod_drag": pod_drag_float})
        partner_df = pd.DataFrame(partner_rows)
        blend_df = pd.DataFrame(blend_rows)
        partner_df.to_csv(STUDY_DIR_PATH / "v2_partners.csv", index=False, float_format="%.6g")
        blend_df.to_csv(STUDY_DIR_PATH / "v2_blends.csv", index=False, float_format="%.6g")
        pd.set_option("display.width", 260)
        print(partner_df.round(3).to_string(index=False))
        print(blend_df[blend_df["share"] == 0.35].round(3).sort_values(["base", "value_add_vs_tbills"], ascending=[True, False]).to_string(index=False))
        rule_df = blend_df[(blend_df["share"] == 0.35) & (blend_df["value_add_vs_tbills"] >= 0.01) & (blend_df["value_add_H1"] > 0) &
                           (blend_df["value_add_H2"] > 0) & (blend_df["partner_small_pod_drag"] < 0.005)]
        print("\npartners passing the pre-declared rule at 35%:\n", rule_df[["base", "partner", "value_add_vs_tbills", "value_add_H1", "value_add_H2", "partner_small_pod_drag"]].round(4).to_string(index=False))
        report_dict["part_b_pass"] = rule_df[["base", "partner"]].to_dict(orient="records")
    (STUDY_DIR_PATH / f"v2_report_{part_str}.json").write_text(json.dumps(report_dict, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
