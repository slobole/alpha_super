"""What an allocator's analyst would check first: is it just cheap exposure? (owner question, 2026-09-24)

Descriptive, nothing selected or tuned. Three checks on the menu products, the current ladders and the
main sleeves:
  1. Naive public rules an allocator can buy for a few basis points, same windows:
       QQQ buy and hold; 60/40 SPY/AGG (monthly); QQQ 200-day trend filter (QQQ above its 200-day
       average at the prior close, else IEF); GEM dual momentum (month-end 12-month total return:
       SPY or EFA when SPY beats T-bills, else AGG). No costs charged to the naive rules (favours them).
  2. Weekly excess-return regressions (Newey-West, 4 lags):
       M1 = QQQ, IEF, GLD, DBC, UUP (the TAA universe held statically)
       M2 = M1 + the QQQ 200-day trend rule (cheap timing)
     alpha = what is left after static exposure (M1) and after cheap trend timing (M2); by half as well.
  3. Sampling error of the backtest Sharpe: SE ~ sqrt((1 + SR^2 / 2) / years).
Windows as in ladders_vs_menu.py: exact 2012-10-02 -> 2026-08-19; long 2008-03-04 onward with the
BTAL-based TAA replaced by its no-BTAL sibling ONLY before 2012-10-02.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import ladders_vs_menu  # noqa: E402

END_TS, CUT_TS, LONG_TS = ladders_vs_menu.END_TS, ladders_vs_menu.CUT_TS, ladders_vs_menu.LONG_TS
ETF_LIST = ["QQQ", "SPY", "EFA", "AGG", "IEF", "GLD", "DBC", "UUP"]
FACTOR_M1_LIST = ["QQQ", "IEF", "GLD", "DBC", "UUP"]
NW_LAG_INT = 4
LADDER_LIST = ["ladder_1_defensive", "ladder_2_balanced", "ladder_3_growth", "ladder_4_growth", "ladder_4_growth_1n"]
SLEEVE_LIST = ["taa_btal_tqqq", "ndx_vxn", "mosaic", "infl_compass", "dv2", "hpi_vote", "sector_vox_iyr", "core5", "eom_flow", "tactical_fi"]


def naive_rule_return_df(etf_close_df: pd.DataFrame, tbill_ser: pd.Series) -> pd.DataFrame:
    etf_return_df = etf_close_df.pct_change(fill_method=None)
    # QQQ 200-day trend filter: *** CRITICAL*** the close-t signal sets the holding for session t+1.
    qqq_close_ser = etf_close_df["QQQ"]
    in_qqq_ser = (qqq_close_ser > qqq_close_ser.rolling(200).mean()).astype(float).shift(1)
    trend_ser = in_qqq_ser * etf_return_df["QQQ"] + (1.0 - in_qqq_ser) * etf_return_df["IEF"]
    trend_ser[in_qqq_ser.isna()] = np.nan
    # GEM dual momentum: decided on month-end closes, held from the next session to the next month-end.
    session_ser = etf_close_df.index.to_series()
    month_end_index = etf_close_df.index[session_ser.dt.month.ne(session_ser.shift(-1).dt.month).to_numpy()]
    month_close_df = etf_close_df.loc[month_end_index, ["SPY", "EFA"]]
    bill_nav_ser = (1.0 + tbill_ser.reindex(etf_close_df.index).fillna(0.0)).cumprod().loc[month_end_index]
    momentum_df = month_close_df / month_close_df.shift(12) - 1.0
    bill_momentum_ser = bill_nav_ser / bill_nav_ser.shift(12) - 1.0
    choice_ser = pd.Series(np.where(momentum_df["SPY"] > bill_momentum_ser,
                                    np.where(momentum_df["SPY"] >= momentum_df["EFA"], "SPY", "EFA"), "AGG"), index=month_end_index)
    choice_ser[momentum_df["SPY"].isna() | bill_momentum_ser.isna()] = None
    held_ser = choice_ser.reindex(etf_close_df.index).ffill().shift(1)  # *** CRITICAL*** month-end decision applies next session
    gem_ser = pd.Series(np.nan, index=etf_close_df.index)
    for asset_str in ("SPY", "EFA", "AGG"):
        mask = held_ser == asset_str
        gem_ser[mask] = etf_return_df.loc[mask, asset_str]
    # 60/40 SPY/AGG with month-end resets.
    sixty_forty_list, value_float, weight_float = [], 1.0, 0.6
    for position_int, session_ts in enumerate(etf_close_df.index):
        spy_float, agg_float = etf_return_df["SPY"].iloc[position_int], etf_return_df["AGG"].iloc[position_int]
        if position_int == 0 or np.isnan(spy_float) or np.isnan(agg_float):
            sixty_forty_list.append(np.nan)
            continue
        spy_leg_float, agg_leg_float = value_float * weight_float * (1 + spy_float), value_float * (1 - weight_float) * (1 + agg_float)
        new_value_float = spy_leg_float + agg_leg_float
        sixty_forty_list.append(new_value_float / value_float - 1.0)
        value_float, weight_float = new_value_float, spy_leg_float / new_value_float
        if session_ts in month_end_index:
            weight_float = 0.6
    return pd.DataFrame({"NAIVE_QQQ": etf_return_df["QQQ"], "NAIVE_60_40": sixty_forty_list, "NAIVE_QQQ_TREND200": trend_ser,
                         "NAIVE_GEM": gem_ser}, index=etf_close_df.index)


def weekly_ser(daily_ser: pd.Series) -> pd.Series:
    return (1.0 + daily_ser).resample("W-FRI").prod(min_count=1) - 1.0


def newey_west_ols(y_ser: pd.Series, x_df: pd.DataFrame, lag_int: int) -> dict:
    frame_df = pd.concat([y_ser.rename("y"), x_df], axis=1).dropna()
    y_arr = frame_df["y"].to_numpy()
    x_arr = np.column_stack([np.ones(len(frame_df)), frame_df[x_df.columns].to_numpy()])
    xtx_inv = np.linalg.inv(x_arr.T @ x_arr)
    beta_arr = xtx_inv @ x_arr.T @ y_arr
    resid_arr = y_arr - x_arr @ beta_arr
    score_arr = x_arr * resid_arr[:, None]
    s_mat = score_arr.T @ score_arr
    for lag in range(1, lag_int + 1):
        gamma_mat = score_arr[lag:].T @ score_arr[:-lag]
        s_mat += (1.0 - lag / (lag_int + 1.0)) * (gamma_mat + gamma_mat.T)
    se_arr = np.sqrt(np.diag(xtx_inv @ s_mat @ xtx_inv))
    r2_float = 1.0 - resid_arr.var() / y_arr.var()
    out = {"alpha_ann": beta_arr[0] * 52.0, "alpha_t": beta_arr[0] / se_arr[0], "r2": r2_float, "weeks": len(frame_df),
           "mean_excess_ann": y_arr.mean() * 52.0}
    out.update({f"b_{name}": b for name, b in zip(x_df.columns, beta_arr[1:])})
    return out


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in ladders_vs_menu.STAND_IN_DICT.items():
        # *** CRITICAL*** the stand-in fills only the dates before the real sleeve exists.
        long_df.loc[long_df.index < CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < CUT_TS, stand_in_str]

    etf_close_df = pd.concat([common.load_total_return_close_ser(s, "2005-01-01", END_TS.strftime("%Y-%m-%d")) for s in ETF_LIST], axis=1)
    etf_close_df = etf_close_df.reindex(sleeve_df.index).loc["2005-01-01":]
    tbill_ser = bench_df["TBILL"].reindex(etf_close_df.index).fillna(0.0)
    naive_df = naive_rule_return_df(etf_close_df, tbill_ser)
    etf_return_df = etf_close_df.pct_change(fill_method=None)

    book_dict = {}  # name -> (weights, policy)
    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    for product_str, group_df in weight_df.groupby("product_id_str", sort=False):
        book_dict[product_str] = (dict(zip(group_df["alias_str"], group_df["weight_float"])), "annual")
    for ladder_str in LADDER_LIST:
        book_dict[ladder_str] = ladders_vs_menu.ladder_book(ladder_str)
    series_dict = {}  # name -> (exact_ser, long_ser)
    for name_str, (weight_dict, policy_str) in book_dict.items():
        series_dict[name_str] = tuple(common.book_return_ser(frame_df.loc[start_ts:, list(weight_dict)], weight_dict, policy_str)[0]
                                      for frame_df, start_ts in ((sleeve_df, CUT_TS), (long_df, LONG_TS)))
    for alias_str in SLEEVE_LIST:
        series_dict[f"sleeve:{alias_str}"] = (sleeve_df[alias_str].loc[CUT_TS:], long_df[alias_str].loc[LONG_TS:])
    for rule_str in naive_df.columns:
        series_dict[rule_str] = (naive_df[rule_str].loc[CUT_TS:], naive_df[rule_str].loc[LONG_TS:])

    def base_ts_for(ser: pd.Series) -> pd.Timestamp:
        return sleeve_df.index[sleeve_df.index.get_loc(ser.index[0]) - 1]

    factor_weekly_df = pd.DataFrame({s: weekly_ser(etf_return_df[s]) for s in FACTOR_M1_LIST})
    trend_weekly_ser = weekly_ser(naive_df["NAIVE_QQQ_TREND200"])
    tbill_weekly_ser = weekly_ser(tbill_ser)
    metric_row_list, alpha_row_list = [], []
    for name_str, (exact_ser, long_ser) in series_dict.items():
        row = {"series": name_str}
        for label_str, ser in (("exact", exact_ser), ("long", long_ser)):
            ser = ser.dropna()
            m = common.metric_dict(ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts_for(ser))
            years_float = len(ser) / 252.0
            row.update({f"cagr_{label_str}": m["cagr_float"], f"vol_{label_str}": m["volatility_float"], f"sharpe_{label_str}": m["sharpe_rf0_float"],
                        f"sharpe_excess_{label_str}": m["sharpe_excess_float"], f"maxdd_{label_str}": m["max_drawdown_float"],
                        f"sharpe_se_{label_str}": float(np.sqrt((1.0 + m["sharpe_rf0_float"] ** 2 / 2.0) / years_float))})
        metric_row_list.append(row)
        if name_str.startswith("NAIVE_"):
            continue
        exact_ser = exact_ser.dropna()
        halves = np.array_split(exact_ser.index, 2)
        for window_str, ser in (("exact", exact_ser), ("first_half", exact_ser.loc[halves[0]]), ("second_half", exact_ser.loc[halves[1]]),
                                ("long", long_ser.dropna())):
            y_ser = weekly_ser(ser)
            y_ser = (y_ser - tbill_weekly_ser).iloc[1:-1]  # drop the partial first and last weeks
            x1_df = factor_weekly_df.sub(tbill_weekly_ser, axis=0).reindex(y_ser.index)
            x2_df = x1_df.assign(TREND200=(trend_weekly_ser - tbill_weekly_ser).reindex(y_ser.index))
            for model_str, x_df in (("M1", x1_df), ("M2", x2_df)):
                alpha_row_list.append({"series": name_str, "window": window_str, "model": model_str, **newey_west_ols(y_ser, x_df, NW_LAG_INT)})
    metric_df = pd.DataFrame(metric_row_list).set_index("series")
    alpha_df = pd.DataFrame(alpha_row_list)
    output_dir_path = common.STUDY_DIR_PATH / "books"
    metric_df.to_csv(output_dir_path / "allocator_first_look_metrics.csv", float_format="%.6g")
    alpha_df.to_csv(output_dir_path / "allocator_first_look_alpha.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    print(metric_df[["cagr_exact", "vol_exact", "sharpe_exact", "sharpe_se_exact", "maxdd_exact", "cagr_long", "sharpe_long", "maxdd_long"]].round(3).to_string())
    print()
    pivot_df = alpha_df.pivot_table(index="series", columns=["model", "window"], values=["alpha_ann", "alpha_t"], sort=False)
    print(pivot_df.round(3).to_string())
    print()
    print(alpha_df[(alpha_df.window == "exact") & (alpha_df.model == "M2")].set_index("series")[
        ["mean_excess_ann", "alpha_ann", "alpha_t", "r2", "b_QQQ", "b_IEF", "b_GLD", "b_DBC", "b_UUP", "b_TREND200"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
