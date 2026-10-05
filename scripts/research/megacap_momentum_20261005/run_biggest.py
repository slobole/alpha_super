"""Biggest-N S&P 500 names, each held only while its own CORE5 adaptive trend filter is on (owner clarification
2026-10-05: "do not use momentum; hold the N biggest stocks, and the entry gate is the adaptive momentum").

Second child of `sp500_megacap_momentum_voltarget_20261005` (run.py: data, engine, costs, eligibility, size proxy;
run_filters.py: the filters). No cross-sectional momentum ranking anywhere.

At decision T (month end; every input reads rows <= T; execution Open(T+1)):
    biggest      eligible members sorted by size = 252-session median dollar Turnover, largest first
    stock filter "none" | "sma100" (Close > SMA100) | "adaptive_ama" (CORE5 rule per stock, CORE5's own parameters:
                 SMA10(T) > AMA(T), alpha/scout/specs/core5.py asset_signal_df)
    slot rule    "cash"  the top_count_int biggest names; a name whose filter is off leaves its slot in cash
                 "next"  the top_count_int biggest names whose filter is on (the next biggest takes the slot)
    weight       1 / top_count_int per held name
    market gate  "none" | "spy_sma200" (off -> all cash); volatility target none | 15% (as run.py)

    uv run python scripts/research/megacap_momentum_20261005/run_biggest.py --register
    uv run python scripts/research/megacap_momentum_20261005/run_biggest.py
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import run as base
import run_filters as filters
from alpha.scout.ledger import Ledger
from alpha.scout.metrics import performance_dict
from alpha.scout.registration import register
from alpha.scout.stations.robustness import paired_sharpe_probability

REGISTRATION_ID_STR = "sp500_biggest_n_adaptive_entry_gate_20261005"
OUTPUT_DIR_PATH = base.OUTPUT_DIR_PATH / "biggest"
GRID_DICT = {
    "top_count_int": (5, 10, 20, 50),
    "stock_filter_str": ("none", "sma100", "adaptive_ama"),
    "slot_str": ("cash", "next"),
    "market_gate_str": ("none", "spy_sma200"),
    "vol_target_float": (0.0, 0.15),
}
ERA_DICT = {"1999_2008": ("1999", "2008"), "2009_2022": ("2009", "2022")}
_WORKER: dict = {}

REGISTRATION = dataclasses.replace(
    base.REGISTRATION,
    registration_id_str=REGISTRATION_ID_STR,
    family_id_str="time_series_trend_and_breakout",
    parent_id_str=None,
    hypothesis_str=(
        "Holding the 5 to 50 most-traded S&P 500 members, each only while its own CORE5 adaptive trend filter is on "
        "(no cross-sectional momentum ranking), earns a higher net Sharpe and a shallower drawdown than holding the "
        "same names with no filter, and a higher net Sharpe than SPY."
    ),
    mechanism_str=(
        "Per-asset time-series trend: a big stock is held while its own trend is up and sold when it turns, so a "
        "falling leader is dropped without a market-wide signal. The filter is a risk overlay; whether it adds return "
        "beyond the cut in exposure is the question."
    ),
    expected_sign_and_location_str=(
        "Paired stationary-bootstrap P(adaptive filter > no filter) of at least 0.80 on in-sample net Sharpe, averaged "
        "over top_count_int, with a shallower drawdown; P(> SPY) of at least 0.80; the same sign in 1999-2008 and "
        "2009-2022."
    ),
    param_grid_dict=GRID_DICT,
    primary_metric_str=(
        "In-sample (1999-02 to 2022-12-30) net Sharpe, CAGR and maximum drawdown, idle cash at T-bills; paired P "
        "against no filter (same N, gate, target) and against SPY; both eras."
    ),
    kill_criteria_str=(
        "Not a candidate if mean P(adaptive > no filter) < 0.80, or mean P(adaptive > SPY) < 0.80, or the Sharpe gain "
        "over no filter is negative in 2009-2022. The plain SMA100 is the benchmark filter: an adaptive filter that does "
        "not beat it is reported as 'any trend filter', not as an adaptive edge."
    ),
    source_str=(
        "Owner clarification 2026-10-05, after the results of ledger rows 38 and 39 were seen (those rows ranked on "
        "momentum; row 39's unranked 50-name control with the adaptive filter is the N = 50 'cash' cell here). Recorded "
        "under the trend family because nothing is ranked; rows 38 and 39 stay in the momentum family."
    ),
    prior_trials_int=24,
)


def weight_df(inputs: dict, signal: dict, config_dict: dict) -> pd.DataFrame:
    panel = inputs["panel"]
    symbol_arr = np.array(panel.symbol_list)
    count_int = config_dict["top_count_int"]
    filter_mat = signal[f"filter_{config_dict['stock_filter_str']}"]
    row_dict = {}
    for row_int in signal["decision_row_vec"]:
        with np.errstate(invalid="ignore"):
            eligible_vec = (
                signal["member_mat"][row_int] & np.isfinite(signal["close_mat"][row_int]) & np.isfinite(signal["m12_mat"][row_int])
                & np.isfinite(signal["sigma_mat"][row_int]) & (signal["sigma_mat"][row_int] > 0) & np.isfinite(signal["size_mat"][row_int])
                & (signal["raw_close_mat"][row_int] >= base.PRICE_FLOOR_FLOAT)
            )
        target_ser = pd.Series(dtype=float)
        idx_vec = np.flatnonzero(eligible_vec)
        if len(idx_vec) >= count_int and signal[f"gate_{config_dict['market_gate_str']}"][row_int]:
            sorted_vec = idx_vec[np.argsort(-signal["size_mat"][row_int, idx_vec], kind="stable")]
            if config_dict["slot_str"] == "cash":
                pick_vec = sorted_vec[:count_int]
                pick_vec = pick_vec[filter_mat[row_int, pick_vec]]
            else:
                pick_vec = sorted_vec[filter_mat[row_int, sorted_vec]][:count_int]
            if len(pick_vec):
                weight_vec = np.full(len(pick_vec), 1.0 / count_int)
                if config_dict["vol_target_float"] > 0:
                    block_mat = np.nan_to_num(signal["return_mat"][row_int - base.VOL_WINDOW_INT + 1: row_int + 1][:, pick_vec])
                    vol_float = float(np.std(block_mat @ weight_vec, ddof=1) * np.sqrt(252.0))
                    weight_vec = weight_vec * min(1.0, config_dict["vol_target_float"] / vol_float) if vol_float > 0 else weight_vec
                target_ser = pd.Series(weight_vec, index=symbol_arr[pick_vec])
        row_dict[panel.date_index[row_int + 1]] = target_ser
    return pd.DataFrame(row_dict).T.fillna(0.0)


def label_str(config_dict: dict) -> str:
    return (f"N{config_dict['top_count_int']}_{config_dict['stock_filter_str']}_{config_dict['slot_str']}_{config_dict['market_gate_str']}"
            f"_vt{int(config_dict['vol_target_float'] * 100)}")


def _init() -> None:
    _WORKER["inputs"] = base.load_inputs()
    _WORKER["signal"] = filters.signal_dict(_WORKER["inputs"])
    base.weight_df = weight_df  # run_config resolves weight_df at call time


def _task(config_dict: dict):
    daily_ser = base.run_config(_WORKER["inputs"], _WORKER["signal"], config_dict)
    return label_str(config_dict), daily_ser, daily_ser.attrs["exposure_float"]


def main() -> None:
    if "--register" in sys.argv:
        row_dict = register(Ledger(), REGISTRATION)
        print(f"Registered {REGISTRATION_ID_STR} as ledger row {row_dict['row_id_int']} ({row_dict['utc_ts_str']}).")
        return
    if REGISTRATION_ID_STR not in {r["registration_id_str"] for r in Ledger().rows("registration")}:
        raise SystemExit("Register first: run with --register.")

    name_list = sorted(GRID_DICT)
    config_list = [dict(zip(name_list, combo)) for combo in itertools.product(*[GRID_DICT[n] for n in name_list])]
    with Pool(8, initializer=_init) as pool_obj:
        out_list = pool_obj.map(_task, config_list, chunksize=2)
    daily_df = pd.DataFrame({label: ser for label, ser, _ in out_list})
    exposure_dict = {label: exposure for label, _, exposure in out_list}
    inputs = base.load_inputs()
    tbill_ser = inputs["tbill_ser"]
    daily_df["SPY"] = inputs["spy_ser"].reindex(daily_df.index)
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    daily_df.to_parquet(OUTPUT_DIR_PATH / "daily_returns.parquet")

    row_list = []
    for config_dict in config_list:
        label, none_label = label_str(config_dict), label_str({**config_dict, "stock_filter_str": "none"})
        sma_label = label_str({**config_dict, "stock_filter_str": "sma100"})
        stat = performance_dict(daily_df[label], tbill_ser)
        row = {**config_dict, "label": label, "sharpe": stat["sharpe_float"], "cagr": stat["cagr_float"], "vol": stat["volatility_float"],
               "max_dd": stat["max_drawdown_float"], "exposure": exposure_dict[label],
               "p_gt_none": float("nan") if label == none_label else paired_sharpe_probability(daily_df[label], daily_df[none_label]),
               "p_gt_sma100": float("nan") if config_dict["stock_filter_str"] != "adaptive_ama" else paired_sharpe_probability(daily_df[label], daily_df[sma_label]),
               "p_gt_spy": paired_sharpe_probability(daily_df[label], daily_df["SPY"])}
        for era_str, (a, b) in ERA_DICT.items():
            era_stat, none_stat = performance_dict(daily_df[label].loc[a:b]), performance_dict(daily_df[none_label].loc[a:b])
            row.update({f"sharpe_{era_str}": era_stat["sharpe_float"], f"cagr_{era_str}": era_stat["cagr_float"],
                        f"gain_{era_str}": era_stat["sharpe_float"] - none_stat["sharpe_float"]})
        row_list.append(row)
    grid_df = pd.DataFrame(row_list)
    grid_df.to_csv(OUTPUT_DIR_PATH / "grid.csv", index=False)

    spy_dict = {"full": performance_dict(daily_df["SPY"], tbill_ser), **{k: performance_dict(daily_df["SPY"].loc[a:b]) for k, (a, b) in ERA_DICT.items()}}
    spy_dict = {k: {m: v[m] for m in ("sharpe_float", "cagr_float", "volatility_float", "max_drawdown_float")} for k, v in spy_dict.items()}
    value_list = ["sharpe", "cagr", "vol", "max_dd", "exposure", "p_gt_none", "p_gt_sma100", "p_gt_spy", "sharpe_1999_2008", "sharpe_2009_2022",
                  "cagr_1999_2008", "cagr_2009_2022", "gain_1999_2008", "gain_2009_2022"]
    summary_dict = {
        "registration_id_str": REGISTRATION_ID_STR, "panel_snapshot_id_str": inputs["panel"].snapshot_id_str, "spy": spy_dict,
        "filter_mean": grid_df.groupby("stock_filter_str")[value_list].mean().to_dict("index"),
    }
    (OUTPUT_DIR_PATH / "summary.json").write_text(json.dumps(summary_dict, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 280, "display.max_columns", 40, "display.max_rows", 200)
    print("SPY", json.dumps(spy_dict, default=float))
    print("== mean by filter ==")
    print(grid_df.groupby("stock_filter_str")[value_list].mean().round(3).to_string())
    print("== mean by gate x target x filter ==")
    print(grid_df.groupby(["market_gate_str", "vol_target_float", "stock_filter_str"])[value_list].mean().round(3).to_string())
    print("== mean by N x slot x filter (no market gate, no target) ==")
    plain_df = grid_df[(grid_df["market_gate_str"] == "none") & (grid_df["vol_target_float"] == 0.0)]
    print(plain_df.set_index(["top_count_int", "slot_str", "stock_filter_str"])[value_list].round(3).sort_index().to_string())
    print("== N x slot x filter, SPY gate + 15% target ==")
    full_df = grid_df[(grid_df["market_gate_str"] == "spy_sma200") & (grid_df["vol_target_float"] == 0.15)]
    print(full_df.set_index(["top_count_int", "slot_str", "stock_filter_str"])[value_list].round(3).sort_index().to_string())


if __name__ == "__main__":
    main()
