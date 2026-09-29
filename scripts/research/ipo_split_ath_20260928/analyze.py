"""Stage B analysis: standalone metrics, books, the frozen rule R1-R5, multiplicity, capacity, charts (PREREG 6, 8).

Usage: python analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
if str(HERE_PATH) not in sys.path:
    sys.path.insert(0, str(HERE_PATH))

import common  # noqa: E402
import run_pods  # noqa: E402
import stage_a  # noqa: E402
from new_pod_search_20260927 import common as npc  # noqa: E402
from trend_breakout_20260927 import analyze as trend_analyze  # noqa: E402

SEED_INT = 20260928
DSR_TRIALS_INT = 535
CAPACITY_ADV_SHARE_FLOAT = 0.05
CAPACITY_MIN_AUM_FLOAT = 5_000_000.0
CAPACITY_START_TS = pd.Timestamp("2021-01-01")
RULE_BLOCK_TUPLE = ("G-P1", "G-P2", "G-P3")
DD_BLOCK_TUPLE = ("G-FULL", "G-LONG")
ANCHOR_SUFFIX_STR = "N20|P20L10"


def anchor_key(pop_str: str, label_str: str = "base") -> str:
    return f"{pop_str}|{ANCHOR_SUFFIX_STR}|{label_str}"


def spy_tr_ret_ser() -> pd.Series:
    spy_df = pd.read_parquet(common.CACHE_DIR_PATH / "spy_tr.parquet")
    return spy_df["Close"].astype(float).pct_change().rename("SPY_TR")


def ols_alpha(x_ser: pd.Series, m_ser: pd.Series, lag_int: int = 5) -> dict:
    frame_df = pd.concat([x_ser, m_ser], axis=1).dropna()
    if len(frame_df) < 60:
        return {"alpha_ann": float("nan"), "alpha_t_nw": float("nan"), "beta": float("nan")}
    y_vec = frame_df.iloc[:, 0].to_numpy()
    m_vec = frame_df.iloc[:, 1].to_numpy()
    design_arr = np.column_stack([np.ones(len(m_vec)), m_vec])
    coef_vec, *_ = np.linalg.lstsq(design_arr, y_vec, rcond=None)
    resid_vec = y_vec - design_arr @ coef_vec
    xtx_inv_arr = np.linalg.inv(design_arr.T @ design_arr)
    score_arr = design_arr * resid_vec[:, None]
    meat_arr = score_arr.T @ score_arr
    for lag in range(1, lag_int + 1):
        weight_float = 1.0 - lag / (lag_int + 1)
        gamma_arr = score_arr[lag:].T @ score_arr[:-lag]
        meat_arr += weight_float * (gamma_arr + gamma_arr.T)
    cov_arr = xtx_inv_arr @ meat_arr @ xtx_inv_arr
    return {"alpha_ann": float(coef_vec[0] * 252), "alpha_t_nw": float(coef_vec[0] / np.sqrt(cov_arr[0, 0])), "beta": float(coef_vec[1])}


def trade_stats(trade_df: pd.DataFrame, years_float: float) -> dict:
    if len(trade_df) == 0:
        return {"trades": 0}
    ret_vec = trade_df["ret"].to_numpy()
    hold_vec = (trade_df["exit_pos"] - trade_df["entry_pos"]).to_numpy()
    reason_ser = trade_df["reason"].value_counts(normalize=True)
    return {
        "trades": int(len(trade_df)), "trades_per_year": float(len(trade_df) / years_float),
        "mean_ret": float(ret_vec.mean()), "median_ret": float(np.median(ret_vec)), "win_rate": float((ret_vec > 0).mean()),
        "avg_win": float(ret_vec[ret_vec > 0].mean()) if (ret_vec > 0).any() else float("nan"),
        "avg_loss": float(ret_vec[ret_vec <= 0].mean()) if (ret_vec <= 0).any() else float("nan"),
        "mean_hold_sessions": float(hold_vec.mean()), "exit_reason_share": {k: float(v) for k, v in reason_ser.items()},
    }


def capacity(trade_df: pd.DataFrame, return_ser: pd.Series, capital_float: float, start_ts: pd.Timestamp | None) -> dict:
    nav_ser = capital_float * (1.0 + return_ser.fillna(0.0)).cumprod()
    calendar_idx = pd.DatetimeIndex(np.load(common.CACHE_DIR_PATH / "calendar.npy"))
    sub_df = trade_df if start_ts is None else trade_df[trade_df["entry_date"] >= start_ts]
    if len(sub_df) == 0:
        return {"trades": 0}
    decision_date_idx = calendar_idx[sub_df["entry_pos"].to_numpy() - 1]
    nav_at_decision_vec = nav_ser.reindex(decision_date_idx).ffill().fillna(capital_float).to_numpy()
    weight_vec = sub_df["order_value"].to_numpy() / nav_at_decision_vec
    adv_vec = sub_df["entry_adv"].to_numpy()
    share_at_1m_vec = weight_vec * 1_000_000.0 / adv_vec
    aum_limit_vec = CAPACITY_ADV_SHARE_FLOAT * adv_vec / weight_vec
    return {"trades": int(len(sub_df)), "max_share_of_adv_at_1m": float(share_at_1m_vec.max()),
            "p95_share_of_adv_at_1m": float(np.quantile(share_at_1m_vec, 0.95)),
            "aum_at_largest_order_5pct_adv": float(aum_limit_vec.min()),
            "median_entry_adv": float(np.median(adv_vec))}


def main() -> None:
    ret_df = pd.read_parquet(common.RESULTS_DIR_PATH / "pod_returns.parquet")
    cash_df = pd.read_parquet(common.RESULTS_DIR_PATH / "pod_cash_weight.parquet")
    trades_df = pd.read_parquet(common.RESULTS_DIR_PATH / "pod_trades.parquet")
    cost_dict = json.loads((common.RESULTS_DIR_PATH / "pod_costs.json").read_text())
    bil_ser = npc.load_bil_ret_ser()
    spy_ser = spy_tr_ret_ser()
    taa_ser = npc.load_taa_ser()
    l_ser = npc.load_l_ret_ser("engine")
    l_stress_ser = npc.load_l_ret_ser("stress")
    spy_book_ser = npc.load_spy_tr_ret_ser()

    sweep_df = pd.DataFrame({k: npc.sweep_return_ser(ret_df[k], cash_df[k], bil_ser) for k in ret_df.columns})
    sweep_df.to_parquet(common.RESULTS_DIR_PATH / "pod_returns_sweep.parquet")

    # ---- standalone -------------------------------------------------------------------------------------------------
    standalone_dict = {}
    for key in ret_df.columns:
        entry_dict = {"sweep": trend_analyze.common.window_metrics(sweep_df[key], stage_a.BLOCK_DICT),
                      "no_sweep": trend_analyze.common.window_metrics(ret_df[key], stage_a.BLOCK_DICT)}
        entry_dict["alpha_vs_spy"] = {b: ols_alpha(sweep_df[key].loc[s:e], spy_ser.loc[s:e]) for b, (s, e) in stage_a.BLOCK_DICT.items()}
        overlap_df = pd.concat([sweep_df[key], taa_ser, l_ser], axis=1).dropna()
        entry_dict["corr_taa"] = float(overlap_df.iloc[:, 0].corr(overlap_df.iloc[:, 1]))
        entry_dict["corr_L"] = float(overlap_df.iloc[:, 0].corr(overlap_df.iloc[:, 2]))
        entry_dict["cash_utilization"] = float(1.0 - cash_df[key].mean())
        key_trades_df = trades_df[trades_df["key"] == key]
        years_float = len(ret_df) / 252.0
        entry_dict["trades"] = trade_stats(key_trades_df, years_float)
        entry_dict["trades_by_block"] = {b: trade_stats(key_trades_df[(key_trades_df["entry_date"] >= s) & (key_trades_df["entry_date"] <= e)],
                                                        len(ret_df.loc[s:e]) / 252.0) for b, (s, e) in stage_a.BLOCK_DICT.items()}
        nav_mean_float = float((cost_dict[key]["capital"] * (1.0 + ret_df[key].fillna(0.0)).cumprod()).mean())
        entry_dict["cost_pct_nav_per_year"] = {"commission": cost_dict[key]["commission_total"] / nav_mean_float / years_float,
                                               "slippage": cost_dict[key]["slippage_total"] / nav_mean_float / years_float}
        standalone_dict[key] = entry_dict

    # ---- books ------------------------------------------------------------------------------------------------------
    controls_dict = npc.control_books(taa_ser, l_ser, bil_ser, spy_book_ser)
    controls_stress_dict = {"C_BIL": npc.candidate_book_blocks(taa_ser, l_stress_ser, bil_ser)}
    book_dict = {key: npc.candidate_book_blocks(taa_ser, l_ser, sweep_df[key]) for key in ret_df.columns}
    book_no_sweep_dict = {key: npc.candidate_book_blocks(taa_ser, l_ser, ret_df[key]) for key in ret_df.columns}
    c_bil_dict = controls_dict["C_BIL"]

    # ---- rule -------------------------------------------------------------------------------------------------------
    rule_dict = {}
    for pop_str in run_pods.GRID_POPULATION_LIST:
        cand_key = anchor_key(pop_str)
        book = book_dict[cand_key]
        stress_book = npc.candidate_book_blocks(taa_ser, l_stress_ser, sweep_df[anchor_key(pop_str, "stress")])
        c_bil_stress = controls_stress_dict["C_BIL"]
        r1_bool = all(book[b]["sharpe"] > c_bil_dict[b]["sharpe"] for b in RULE_BLOCK_TUPLE)
        r2_bool = all(book[b]["max_dd"] >= c_bil_dict[b]["max_dd"] - npc.DD_TOLERANCE_FLOAT for b in DD_BLOCK_TUPLE)
        r3_bool = (all(stress_book[b]["sharpe"] > c_bil_stress[b]["sharpe"] for b in RULE_BLOCK_TUPLE)
                   and all(stress_book[b]["max_dd"] >= c_bil_stress[b]["max_dd"] - npc.DD_TOLERANCE_FLOAT for b in DD_BLOCK_TUPLE))
        r4_bool = book_dict[anchor_key(pop_str, "u500")]["G-FULL"]["sharpe"] > c_bil_dict["G-FULL"]["sharpe"]
        cap_dict = capacity(trades_df[trades_df["key"] == cand_key], ret_df[cand_key], 100_000.0, CAPACITY_START_TS)
        r5_bool = bool(cap_dict.get("aum_at_largest_order_5pct_adv", 0.0) >= CAPACITY_MIN_AUM_FLOAT)
        e2_book = book_dict[anchor_key(pop_str, "e2")]
        rule_dict[pop_str] = {
            "candidate": cand_key, "R1": r1_bool, "R2": r2_bool, "R3": r3_bool, "R4": r4_bool, "R5": r5_bool,
            "passes_bool": bool(r1_bool and r2_bool and r3_bool and r4_bool and r5_bool),
            "book": book, "book_stress": stress_book, "capacity_2021_26": cap_dict,
            "capacity_full": capacity(trades_df[trades_df["key"] == cand_key], ret_df[cand_key], 100_000.0, None),
            "labels": {
                "beats_G3_all_rule_blocks": all(book[b]["sharpe"] > controls_dict["G3"][b]["sharpe"] for b in RULE_BLOCK_TUPLE),
                "beats_C_SPY_all_rule_blocks": all(book[b]["sharpe"] > controls_dict["C_SPY"][b]["sharpe"] for b in RULE_BLOCK_TUPLE),
                "owner_gates_g_full": bool(book["G-FULL"]["sharpe"] >= npc.OWNER_GATE_SHARPE_FLOAT and book["G-FULL"]["max_dd"] >= npc.OWNER_GATE_MAX_DD_FLOAT),
                "e2_R1": all(e2_book[b]["sharpe"] > c_bil_dict[b]["sharpe"] for b in RULE_BLOCK_TUPLE),
                "e2_book": e2_book,
            },
        }

    # ---- multiplicity -----------------------------------------------------------------------------------------------
    grid_key_list = [k for k in ret_df.columns if k.endswith("|base") and k.split("|")[0] in run_pods.GRID_POPULATION_LIST]
    book_full_df = pd.DataFrame({k: npc.candidate_book_series(taa_ser, l_ser, sweep_df[k], "G-FULL") for k in grid_key_list})
    book_full_df["C_BIL"] = npc.candidate_book_series(taa_ser, l_ser, bil_ser, "G-FULL")
    book_full_df = book_full_df.dropna()
    candidate_key_list = [anchor_key(p) for p in run_pods.GRID_POPULATION_LIST]
    rc_dict = trend_analyze.reality_check(book_full_df, "C_BIL", candidate_key_list, SEED_INT)
    excess_df = sweep_df[grid_key_list].sub(bil_ser.reindex(sweep_df.index), axis=0).loc["2007-06-01":"2026-08-19"].dropna()
    daily_sharpe_vec = (excess_df.mean() / excess_df.std(ddof=1)).to_numpy()
    dsr_dict = {k: trend_analyze.deflated_sharpe(excess_df[k], daily_sharpe_vec, DSR_TRIALS_INT) for k in candidate_key_list}

    out_dict = {"controls": controls_dict, "controls_stress": controls_stress_dict, "rule": rule_dict,
                "standalone": standalone_dict, "books": book_dict, "books_no_sweep": book_no_sweep_dict,
                "reality_check": rc_dict, "deflated_sharpe": dsr_dict}
    common.write_json("stage_b.json", out_dict)
    common.log_progress("analyze: done")


if __name__ == "__main__":
    main()
