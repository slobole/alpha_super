"""Why does the 1/N aggressive book hold no mean reversion, and what happens if it does? (owner question, 2026-09-24)

AGG_1N inherited the menu's Aggressive definition: the low-touch line only (<= 36 trade days/yr, no
closing-auction orders), which excludes every mean-reversion sleeve by rule (73-199 trade days/yr).
That was a line rule, not a finding. This script tests MR inside the 1/N book directly.

Pre-declared (written before any of these books were built; AGG_1N, AGG_MAX and RET_ALL_1N were already
seen in aggressive_24_check.py and are carried as references):
  Engines exactly as in frozen_spec.yaml, with the 1/N TAA in the TAA slot:
    MOM  = ndx_vxn + mosaic          TAA  = taa_btal_1n_tqqq + infl_compass
    MR_S = hpi_vote + dv2            MR_X = sector_vox_iyr + disp_kie_ihi_sma
  Books, equal capital per engine, 50/50 inside an engine:
    AGG_1N        = MOM + TAA                    (reference)
    AGG_1N+MR_S   = MOM + TAA + MR_S
    AGG_1N+MR_X   = MOM + TAA + MR_X
    RET_ALL_1N    = MOM + TAA + MR_S + MR_X      (reference)
    AGG_1N+HPI    = MOM + TAA + MR_S with DV2 off (the frozen spec flags DV2 with a negative verdict)
  One descriptive curve per MR engine (not a search - nothing is picked from it):
    (1 - m) * AGG_1N + m * MR bucket, m in {0.1, 0.2, 0.3, 0.4, 0.5}, for MR_S, MR_X and MR_ALL.
  Diagnostics: each MR sleeve's correlation with the AGG_1N book and the marginal-Sharpe test
  (adding a little of sleeve i raises the book's Sharpe iff Sharpe_i > rho(i, book) * Sharpe_book).
  Stress on every book: +5 bps per side on every traded dollar (the menu's stress), sleeve by sleeve.
Window 2012-10-02 -> 2026-08-19; long window 2008-03-04 onward with the 2x no-BTAL TAA standing in
ONLY before 2012-10-02. Annual reset.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import evaluation  # noqa: E402

END_TS, CUT_TS, LONG_TS = pd.Timestamp("2026-08-19"), pd.Timestamp("2012-10-02"), pd.Timestamp("2008-03-04")
EXTRA_SLIPPAGE_PER_SIDE_FLOAT = 0.0005
STAND_IN_DICT = {"taa_btal_1n_tqqq": "taa_1n_qld"}
ENGINE_DICT = {
    "MOM": ["ndx_vxn", "mosaic"],
    "TAA": ["taa_btal_1n_tqqq", "infl_compass"],
    "MR_S": ["hpi_vote", "dv2"],
    "MR_X": ["sector_vox_iyr", "disp_kie_ihi_sma"],
    "MR_S_HPI": ["hpi_vote"],
}
MR_ALIAS_LIST = ["hpi_vote", "dv2", "sector_vox_iyr", "disp_kie_ihi_sma"]


def engine_book_dict(engine_list: list[str]) -> dict[str, float]:
    """Equal capital per engine, equal capital per sleeve inside an engine."""
    weight_dict: dict[str, float] = {}
    for engine_str in engine_list:
        member_list = ENGINE_DICT[engine_str]
        for alias_str in member_list:
            weight_dict[alias_str] = weight_dict.get(alias_str, 0.0) + 1.0 / len(engine_list) / len(member_list)
    return weight_dict


def blend_dict(base_dict: dict[str, float], bucket_list: list[str], share_float: float) -> dict[str, float]:
    weight_dict = {k: v * (1.0 - share_float) for k, v in base_dict.items()}
    for alias_str in bucket_list:
        weight_dict[alias_str] = weight_dict.get(alias_str, 0.0) + share_float / len(bucket_list)
    return weight_dict


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in STAND_IN_DICT.items():
        # *** CRITICAL*** the stand-in fills only the dates before the real sleeve exists.
        long_df.loc[long_df.index < CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < CUT_TS, stand_in_str]

    base_dict = engine_book_dict(["MOM", "TAA"])
    book_by_name_dict = {
        "AGG_1N": base_dict,
        "AGG_1N+MR_S": engine_book_dict(["MOM", "TAA", "MR_S"]),
        "AGG_1N+MR_X": engine_book_dict(["MOM", "TAA", "MR_X"]),
        "RET_ALL_1N": engine_book_dict(["MOM", "TAA", "MR_S", "MR_X"]),
        "AGG_1N+HPI": engine_book_dict(["MOM", "TAA", "MR_S_HPI"]),
    }
    for bucket_str, bucket_list in (("MR_S", ENGINE_DICT["MR_S"]), ("MR_X", ENGINE_DICT["MR_X"]), ("MR_ALL", MR_ALIAS_LIST)):
        for share_float in (0.1, 0.2, 0.3, 0.4, 0.5):
            book_by_name_dict[f"curve:{bucket_str}@{share_float:.0%}"] = blend_dict(base_dict, bucket_list, share_float)

    # Stressed sleeve returns: +5 bps per side on every traded dollar, from each sleeve's own fills.
    path_by_alias_dict = common.load_sleeve_path_dict()
    used_alias_set = {a for d in book_by_name_dict.values() for a in d}
    stressed_df = sleeve_df.copy()
    for alias_str in sorted(used_alias_set):
        path_df = path_by_alias_dict[alias_str].loc[:END_TS]
        transaction_df = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        drag_ser = evaluation.extra_slippage_cost_ser(transaction_df, path_df["total_value_float"], EXTRA_SLIPPAGE_PER_SIDE_FLOAT)
        live_mask = sleeve_df[alias_str].notna()
        stressed_df.loc[live_mask, alias_str] = sleeve_df.loc[live_mask, alias_str] - drag_ser.reindex(sleeve_df.index).fillna(0.0)[live_mask]

    def book(frame_df: pd.DataFrame, weight_dict: dict, start_ts: pd.Timestamp) -> pd.Series:
        return common.book_return_ser(frame_df.loc[start_ts:, list(weight_dict)], weight_dict, "annual")[0]

    def base_ts_for(ser: pd.Series) -> pd.Timestamp:
        return sleeve_df.index[sleeve_df.index.get_loc(ser.index[0]) - 1]

    row_list = []
    exact_ser_dict = {}
    for name_str, weight_dict in book_by_name_dict.items():
        exact_ser = book(sleeve_df, weight_dict, CUT_TS)
        long_ser = book(long_df, weight_dict, LONG_TS)
        stress_ser = book(stressed_df, weight_dict, CUT_TS)
        exact_ser_dict[name_str] = exact_ser
        m = common.metric_dict(exact_ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts_for(exact_ser))
        ml = common.metric_dict(long_ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts_for(long_ser))
        ms = common.metric_dict(stress_ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts_for(stress_ser))
        nav_ser = common.nav_from_return_ser(exact_ser)
        out = {"book": name_str, "mr_share": sum(v for k, v in weight_dict.items() if k in MR_ALIAS_LIST),
               "cagr": m["cagr_float"], "vol": m["volatility_float"], "sharpe": m["sharpe_rf0_float"], "maxdd": m["max_drawdown_float"],
               "worst12m": float((nav_ser / nav_ser.shift(252) - 1).dropna().min()), "beta": m["beta_spx_float"],
               "cagr_stress": ms["cagr_float"], "sharpe_stress": ms["sharpe_rf0_float"],
               "cagr_long": ml["cagr_float"], "sharpe_long": ml["sharpe_rf0_float"], "maxdd_long": ml["max_drawdown_float"]}
        for i, part in enumerate(np.array_split(exact_ser.index, 3), 1):
            part_nav = common.nav_from_return_ser(exact_ser.loc[part])
            out[f"cagr_third{i}"] = float(part_nav.iloc[-1] ** (365.25 / (part[-1] - part[0]).days) - 1)
        for ep_str, start_str, end_str in (("gfc", "2008-05-19", "2009-03-09"), ("covid", "2020-02-19", "2020-03-23"),
                                           ("y2022", "2022-01-03", "2022-10-12"), ("y2025", "2025-02-19", "2025-04-08")):
            out[ep_str] = common.window_return_float(long_ser, start_str, end_str)
        row_list.append(out)
    result_df = pd.DataFrame(row_list).set_index("book")

    # Marginal-Sharpe test of each MR sleeve against the AGG_1N book (exact window, daily).
    agg_ser = exact_ser_dict["AGG_1N"]
    agg_sharpe_float = float(result_df.loc["AGG_1N", "sharpe"])
    diag_row_list = []
    for alias_str in MR_ALIAS_LIST:
        sleeve_ser = sleeve_df.loc[agg_ser.index, alias_str]
        sm = common.metric_dict(sleeve_ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts_for(sleeve_ser))
        rho_float = float(sleeve_ser.corr(agg_ser))
        monthly_df = pd.concat([sleeve_ser, agg_ser], axis=1).add(1).resample("ME").prod().sub(1)
        diag_row_list.append({"alias": alias_str, "cagr": sm["cagr_float"], "vol": sm["volatility_float"], "sharpe": sm["sharpe_rf0_float"],
                              "maxdd": sm["max_drawdown_float"], "rho_daily_vs_agg": rho_float,
                              "rho_monthly_vs_agg": float(monthly_df.corr().iloc[0, 1]),
                              "hurdle_rho_x_sharpe": rho_float * agg_sharpe_float,
                              "raises_sharpe": sm["sharpe_rf0_float"] > rho_float * agg_sharpe_float,
                              "covid": common.window_return_float(sleeve_df[alias_str], "2020-02-19", "2020-03-23"),
                              "gfc": common.window_return_float(sleeve_df[alias_str], "2008-05-19", "2009-03-09")})
    diag_df = pd.DataFrame(diag_row_list).set_index("alias")

    output_dir_path = common.STUDY_DIR_PATH / "books"
    result_df.to_csv(output_dir_path / "mr_in_aggressive_check.csv", float_format="%.6g")
    diag_df.to_csv(output_dir_path / "mr_in_aggressive_marginal.csv", float_format="%.6g")
    pd.set_option("display.width", 280)
    print(result_df[["mr_share", "cagr", "vol", "sharpe", "maxdd", "worst12m", "beta", "cagr_stress", "sharpe_stress"]].round(3).to_string())
    print()
    print(result_df[["cagr_third1", "cagr_third2", "cagr_third3", "cagr_long", "sharpe_long", "maxdd_long", "gfc", "covid", "y2022", "y2025"]].round(3).to_string())
    print()
    print(diag_df.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
