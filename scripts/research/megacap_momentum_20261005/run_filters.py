"""Mega-cap momentum, follow-up (owner question 2026-10-05): hold a stock only while its own CORE5-style adaptive trend
filter is on, and only when VIX is low, instead of (or on top of) the SPY 200-day gate and the volatility target.

Child of `sp500_megacap_momentum_voltarget_20261005` (see run.py for the base rule, data, engine and costs). Fixed from
the parent: pool = the 50 most-traded eligible S&P 500 members, score = m12 / sigma, monthly decision, next open.

New at decision T (every input reads rows <= T):
    stock filter  "none"          every pool name is a candidate
                  "sma100"        Close(T) > SMA100(T)                                    (the NDX pod's filter)
                  "adaptive_ama"  CORE5 rule per stock, CORE5's own parameters (alpha/scout/specs/core5.py
                                  asset_signal_df): SMA10(T) > AMA(T), the AMA's speed blended between EMA50 and
                                  EMA200 by the 126-session percentile of the drawdown from the running high, squared
    market gate   "none" | "spy_sma200" (SPY total-return index > its 200-session average)
                  | "vix20" | "vix25" ($VIX close(T) < 20 / 25); gate off -> all cash
    picks         the top_count_int candidates by score; each gets a fixed slot of 1 / top_count_int and an empty slot
                  stays in cash (as the NDX pod), then the 15% volatility target scales the book when it is on.

Matched control: every pool candidate (top_count_int = 50, slot 1 / 50) under the same filter, gate and target. With
the adaptive filter it is the owner's idea without the ranking: each big stock held only while its own filter is on.

    uv run python scripts/research/megacap_momentum_20261005/run_filters.py --register
    uv run python scripts/research/megacap_momentum_20261005/run_filters.py
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
from alpha.scout.ledger import Ledger
from alpha.scout.metrics import performance_dict
from alpha.scout.registration import register
from alpha.scout.stations.robustness import paired_sharpe_probability

REGISTRATION_ID_STR = "sp500_megacap_momentum_stock_filter_vix_20261005"
OUTPUT_DIR_PATH = base.OUTPUT_DIR_PATH / "filters"
POOL_INT = 50
GRID_DICT = {
    "top_count_int": (5, 10, 20),
    "stock_filter_str": ("none", "sma100", "adaptive_ama"),
    "market_gate_str": ("none", "spy_sma200", "vix20", "vix25"),
    "vol_target_float": (0.0, 0.15),
}
_WORKER: dict = {}

REGISTRATION = dataclasses.replace(
    base.REGISTRATION,
    registration_id_str=REGISTRATION_ID_STR,
    parent_id_str=base.REGISTRATION_ID_STR,
    hypothesis_str=(
        "On the parent's book (50 most-traded S&P 500 members, top 5 to 20 by 12-1 momentum per unit of volatility), a "
        "per-stock CORE5 adaptive trend filter and a VIX-level market gate (VIX < 20 or < 25) give a net Sharpe at least "
        "as high, and a drawdown no deeper, than the parent's SPY 200-day gate with no stock filter."
    ),
    mechanism_str=(
        "Risk overlay, not alpha: a stock-level trend filter exits single losers that a market gate cannot see, and a "
        "VIX-level gate steps aside in stressed markets. Prior evidence on the Nasdaq-100 pod (38 stock filters, "
        "Romano-Wolf p 0.19; the adaptive filter 0.89 against 0.87 with no filter) says stock filters add little."
    ),
    expected_sign_and_location_str=(
        "Paired stationary-bootstrap P(variant > parent overlay) of at least 0.80 on in-sample net Sharpe for the "
        "adaptive filter (vs no stock filter, same gate) and for the VIX gates (vs spy_sma200, same filter), averaged "
        "over top_count_int, with a maximum drawdown no deeper than the comparison."
    ),
    param_grid_dict=GRID_DICT,
    primary_metric_str=(
        "In-sample (1999-02 to 2022-12-30) net Sharpe and maximum drawdown, idle cash at T-bills; marginal effect of "
        "each stock filter and each market gate across the grid; paired P against the parent overlay and against the "
        "matched control."
    ),
    kill_criteria_str=(
        "The adaptive filter is dropped if its mean P against 'none' at the same gate is below 0.80. A VIX gate is "
        "dropped if its mean P against spy_sma200 is below 0.80 or its drawdown is deeper. No threshold is tuned after."
    ),
    source_str=(
        "Owner question 2026-10-05 ('hold a stock like CORE5 with the adaptive filter, and only with VIX below 20, as an "
        "example'), after the parent run's results were seen. VIX 25 added as the one neighbour of the owner's example."
    ),
    prior_trials_int=29,
)


def signal_dict(inputs: dict) -> dict:
    from alpha.scout.specs import core5

    signal = base.signal_dict(inputs)
    panel = inputs["panel"]
    close_df = panel.field("Close")
    # *** CRITICAL*** both filters end at their own row; the VIX is the last close dated <= T.
    signal["filter_none"] = np.ones(close_df.shape, dtype=bool)
    signal["filter_sma100"] = (close_df > close_df.rolling(100, min_periods=100).mean()).to_numpy()
    config = core5.Core5Config()
    signal["filter_adaptive_ama"] = pd.DataFrame(
        {s: core5.asset_signal_df(close_df[s], config)["long"] for s in close_df.columns}, index=close_df.index
    ).eq(1.0).to_numpy()
    # Loaded from 1998 directly: taa_3x.load_inputs() starts at BTAL's inception (2011-09), which left every VIX gate
    # off before 2011 in the first run of this script (found and fixed 2026-10-05, before any result was reported).
    from data.norgate_loader import load_price_timeseries

    vix_ser = load_price_timeseries("$VIX", adjustment_str="CAPITALSPECIAL", start_date_str="1998-01-01")["Close"].astype(float)
    vix_ser = vix_ser.reindex(vix_ser.index.union(close_df.index)).ffill().reindex(close_df.index)
    if vix_ser.loc[close_df.index[base.LOOKBACK_INT]:].isna().any():
        raise ValueError("VIX has gaps inside the decision window.")
    signal["gate_none"] = np.ones(len(close_df), dtype=bool)
    signal["gate_spy_sma200"] = signal["regime_vec"]
    signal["gate_vix20"] = (vix_ser < 20.0).to_numpy()
    signal["gate_vix25"] = (vix_ser < 25.0).to_numpy()
    return signal


def weight_df(inputs: dict, signal: dict, config_dict: dict) -> pd.DataFrame:
    panel = inputs["panel"]
    symbol_arr = np.array(panel.symbol_list)
    count_int = config_dict["top_count_int"]
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
        if len(idx_vec) >= POOL_INT and signal[f"gate_{config_dict['market_gate_str']}"][row_int]:
            pool_vec = idx_vec[np.argsort(-signal["size_mat"][row_int, idx_vec], kind="stable")[:POOL_INT]]
            pool_vec = pool_vec[signal[f"filter_{config_dict['stock_filter_str']}"][row_int, pool_vec]]
            score_vec = signal["m12_mat"][row_int, pool_vec] / signal["sigma_mat"][row_int, pool_vec]
            pick_vec = pool_vec[np.argsort(-score_vec, kind="stable")[:count_int]]
            if len(pick_vec):
                weight_vec = np.full(len(pick_vec), 1.0 / count_int)  # fixed slots; an empty slot is cash
                if config_dict["vol_target_float"] > 0:
                    block_mat = np.nan_to_num(signal["return_mat"][row_int - base.VOL_WINDOW_INT + 1: row_int + 1][:, pick_vec])
                    vol_float = float(np.std(block_mat @ weight_vec, ddof=1) * np.sqrt(252.0))
                    weight_vec = weight_vec * min(1.0, config_dict["vol_target_float"] / vol_float) if vol_float > 0 else weight_vec
                target_ser = pd.Series(weight_vec, index=symbol_arr[pick_vec])
        row_dict[panel.date_index[row_int + 1]] = target_ser
    return pd.DataFrame(row_dict).T.fillna(0.0)


def label_str(config_dict: dict) -> str:
    return f"N{config_dict['top_count_int']}_{config_dict['stock_filter_str']}_{config_dict['market_gate_str']}_vt{int(config_dict['vol_target_float'] * 100)}"


def _init() -> None:
    _WORKER["inputs"] = base.load_inputs()
    _WORKER["signal"] = signal_dict(_WORKER["inputs"])
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
    control_list = [c for c in ({**c, "top_count_int": POOL_INT} for c in config_list if c["top_count_int"] == GRID_DICT["top_count_int"][0])]
    with Pool(8, initializer=_init) as pool_obj:
        out_list = pool_obj.map(_task, config_list + control_list, chunksize=2)
    daily_df = pd.DataFrame({label: ser for label, ser, _ in out_list})
    exposure_dict = {label: exposure for label, _, exposure in out_list}
    inputs = base.load_inputs()
    tbill_ser = inputs["tbill_ser"]
    daily_df["SPY"] = inputs["spy_ser"].reindex(daily_df.index)
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    daily_df.to_parquet(OUTPUT_DIR_PATH / "daily_returns.parquet")

    def row(config_dict: dict) -> dict:
        label = label_str(config_dict)
        stat = performance_dict(daily_df[label], tbill_ser)
        out = {**config_dict, "label": label, "sharpe": stat["sharpe_float"], "cagr": stat["cagr_float"], "vol": stat["volatility_float"],
               "max_dd": stat["max_drawdown_float"], "exposure": exposure_dict[label],
               **{f"sharpe_{k}": performance_dict(daily_df[label].loc[a:b])["sharpe_float"] for k, (a, b) in
                  {"1999_2008": ("1999", "2008"), "2009_2022": ("2009", "2022")}.items()},
               **{f"cagr_{k}": performance_dict(daily_df[label].loc[a:b])["cagr_float"] for k, (a, b) in
                  {"1999_2008": ("1999", "2008"), "2009_2022": ("2009", "2022")}.items()}}
        for name_str, ref_dict in {"p_gt_no_filter": {**config_dict, "stock_filter_str": "none"},
                                   "p_gt_spy_gate": {**config_dict, "market_gate_str": "spy_sma200"},
                                   "p_gt_control": {**config_dict, "top_count_int": POOL_INT}}.items():
            out[name_str] = float("nan") if ref_dict == config_dict else paired_sharpe_probability(daily_df[label], daily_df[label_str(ref_dict)])
        return out

    grid_df = pd.DataFrame([row(c) for c in config_list + control_list])
    grid_df.to_csv(OUTPUT_DIR_PATH / "grid.csv", index=False)
    pick_df, control_df = grid_df[grid_df["top_count_int"] != POOL_INT], grid_df[grid_df["top_count_int"] == POOL_INT]

    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200)
    value_list = ["sharpe", "cagr", "vol", "max_dd", "exposure", "sharpe_1999_2008", "sharpe_2009_2022", "cagr_2009_2022"]
    print("== picks: mean over top_count_int, by gate x filter x target ==")
    print(pick_df.groupby(["vol_target_float", "market_gate_str", "stock_filter_str"])[value_list + ["p_gt_no_filter", "p_gt_spy_gate", "p_gt_control"]].mean().round(3).to_string())
    print("== controls (whole filtered pool, slot 1/50) ==")
    print(control_df.set_index(["vol_target_float", "market_gate_str", "stock_filter_str"])[value_list + ["p_gt_no_filter", "p_gt_spy_gate"]].round(3).sort_index().to_string())
    print("== every pick configuration ==")
    print(pick_df[["label"] + value_list + ["p_gt_no_filter", "p_gt_spy_gate", "p_gt_control"]].round(3).to_string(index=False))
    spy_stat = performance_dict(daily_df["SPY"], tbill_ser)
    summary_dict = {
        "registration_id_str": REGISTRATION_ID_STR, "panel_snapshot_id_str": inputs["panel"].snapshot_id_str,
        "filter_marginal": pick_df.groupby("stock_filter_str")[value_list + ["p_gt_no_filter"]].mean().to_dict("index"),
        "gate_marginal": pick_df.groupby("market_gate_str")[value_list + ["p_gt_spy_gate"]].mean().to_dict("index"),
        "gate_on_share": {k: float(np.mean(signal_dict(inputs)[f"gate_{k}"][base.signal_dict(inputs)["decision_row_vec"]])) for k in GRID_DICT["market_gate_str"]},
        "spy": {k: v for k, v in spy_stat.items() if k != "year_return_dict"},
    }
    (OUTPUT_DIR_PATH / "summary.json").write_text(json.dumps(summary_dict, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: summary_dict[k] for k in ("filter_marginal", "gate_marginal", "gate_on_share")}, indent=1, default=float))


if __name__ == "__main__":
    main()
