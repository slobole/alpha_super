"""S&P 500 mega-cap momentum with volatility normalisation (owner question 2026-10-05), on the Scout stack.

Owner idea: market leadership changes, so each month hold the momentum leaders among the biggest S&P 500 names.
A first screen outside Scout (Pakal `megacap_momentum_scout`, 29 books, full history to 2026-09, fully invested,
equal weight) showed drawdowns of 77% to 88%. The owner asked why, and whether volatility-normalised position
sizing fixes it. This script answers on the sealed Scout panel (in sample to 2022-12-30) with the house engine.

Decision date T: the last panel session of each calendar month. Execution: Open(T+1), Scout weights engine,
house parity costs (2.5 bps slippage, USD 0.005 a share, USD 1 minimum), USD 100K, whole shares. Idle cash is
credited at the T-bill rate (as the A15 scripts). Prices CAPITALSPECIAL.

For each stock i at T (every input reads rows <= T):
    size     = median Turnover over the last 252 sessions (at least 200 present)     # no point-in-time market cap
    m12      = Close(T - 21) / Close(T - 252) - 1
    sigma    = std of daily Close returns over the last 63 sessions (at least 50) x sqrt(252)
    eligible = member(T), Close(T), m12, sigma and size finite, sigma > 0, Unadjusted Close(T) >= 5
    pool     = the pool_int eligible names with the largest size
    score    = m12 (rank_str "m12") or m12 / sigma (rank_str "m12_over_vol")
    picks    = the top_count_int pool names by score (ties: symbol ascending)
    weights  = 1 / n (weight_str "equal") or proportional to 1 / sigma, capped at 2 / n ("inverse_vol")
    exposure = min(1, vol_target_float / vol_p) when vol_target_float > 0, where vol_p is the annualised std of the
               weighted daily returns of the picks over the last 63 sessions; else 1
    regime   = SPY total-return index(T) > its 200-session average when regime_bool; off -> all cash

Matched control of a configuration: the whole pool (top_count_int = pool_int) under the same weight rule, the same
volatility target and the same regime gate. It removes the selection and keeps every risk overlay.

    uv run python scripts/research/megacap_momentum_20261005/run.py --register   # S0, once, before any run
    uv run python scripts/research/megacap_momentum_20261005/run.py
"""

from __future__ import annotations

import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import simulate
from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger
from alpha.scout.metrics import performance_dict, sharpe_float, tbill_daily_ser
from alpha.scout.panel import load_panel
from alpha.scout.registration import Registration, register
from alpha.scout.stations.robustness import paired_sharpe_probability

REGISTRATION_ID_STR = "sp500_megacap_momentum_voltarget_20261005"
OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "megacap_momentum_20261005"
PANEL_NAME_STR = "S&P 500"
GRID_DICT = {
    "pool_int": (50, 100),
    "top_count_int": (5, 10, 20),
    "rank_str": ("m12", "m12_over_vol"),
    "weight_str": ("equal", "inverse_vol"),
    "vol_target_float": (0.0, 0.15),
    "regime_bool": (False, True),
}
SIZE_WINDOW_INT, SIZE_MIN_INT = 252, 200
VOL_WINDOW_INT, VOL_MIN_INT = 63, 50
SKIP_INT, LOOKBACK_INT = 21, 252
REGIME_SMA_INT = 200
PRICE_FLOOR_FLOAT = 5.0
CRISIS_DICT = {
    "Dot-com bear 2000-03 to 2002-10": ("2000-03-24", "2002-10-09"),
    "GFC 2007-10 to 2009-03": ("2007-10-09", "2009-03-09"),
    "COVID crash 2020": ("2020-02-19", "2020-03-23"),
    "2022 bear": ("2022-01-03", "2022-10-12"),
}
_WORKER: dict = {}

REGISTRATION = Registration(
    registration_id_str=REGISTRATION_ID_STR,
    family_id_str="equity_cross_sectional_momentum",
    hypothesis_str=(
        "Among the 50 or 100 most-traded S&P 500 members, the top 5 to 20 names by 12-1 momentum, held monthly with "
        "volatility-normalised sizing (inverse-volatility weights and/or a 15% portfolio volatility target, with or "
        "without an SPY 200-day regime gate), earn a higher net Sharpe than the whole pool held under the same sizing "
        "and gate, with a maximum drawdown no deeper than 35%."
    ),
    mechanism_str=(
        "Cross-sectional momentum (under-reaction) inside the most liquid large caps; leadership rotation is followed "
        "by the monthly re-ranking. Volatility normalisation is a risk overlay, not alpha: it is expected to cut the "
        "drawdown, and the matched control receives the same overlay so that the selection is judged alone."
    ),
    expected_sign_and_location_str=(
        "Paired stationary-bootstrap P(configuration > matched control) of at least 0.80 on in-sample net Sharpe at the "
        "plateau configuration; the volatility target and the regime gate cut the maximum drawdown of both the "
        "configuration and its control; the selection effect is concentrated in the smallest top_count_int."
    ),
    hypothesis_class_str="X",
    universe_str="S&P 500 point-in-time members (Scout panel 'S&P 500', sealed at 2022-12-30)",
    horizon_str="one month",
    schedule_str="month-end decision",
    execution_str="next_open",
    param_grid_dict=GRID_DICT,
    primary_metric_str=(
        "In-sample (first decision 1999 to 2022-12-30) net Sharpe, idle cash at T-bills, of the configuration with the "
        "highest neighbourhood-median Sharpe; paired stationary-bootstrap P(> matched control); maximum drawdown."
    ),
    kill_criteria_str=(
        "Not a candidate if P(plateau configuration > matched control) < 0.80, or if fewer than half of the 96 "
        "configurations beat their matched control on Sharpe, or if the plateau configuration's maximum drawdown is "
        "deeper than 35%. A miss is WATCHLIST or diagnostic, not a new search."
    ),
    source_str=(
        "Owner idea 2026-10-05. Prior look outside Scout: Pakal reports/megacap_momentum_scout (29 books on Norgate "
        "1991-01 to 2026-09, fully invested, equal weight), whose results the author of this registration has seen, "
        "including 2023 to 2026. The family's vault is CONTAMINATED; this run stays inside the seal."
    ),
    prior_trials_int=29,
    universe_choice_str=(
        "The owner named the S&P 500 before any result. 'Biggest' is the 252-session median dollar Turnover because "
        "no point-in-time market capitalisation exists in the panel; that proxy was fixed in the Pakal screen before "
        "its results and is kept unchanged."
    ),
    universe_chosen_after_results_bool=False,
)


# ---------------------------------------------------------------- inputs and signals
def load_inputs() -> dict:
    from alpha.scout.reaudit import factor_daily_df
    from alpha.scout.specs import taa_3x

    panel = load_panel(PANEL_NAME_STR)  # sealed: no bar on or after 2023-01-01
    date_index = panel.date_index
    tbill_ser = tbill_daily_ser(taa_3x.load_inputs().dtb3_ser, date_index)
    spy_ser = factor_daily_df(pd.bdate_range("1998-01-01", date_index[-1]), tbill_ser)["SPY"].reindex(date_index).fillna(0.0)
    return {"panel": panel, "tbill_ser": tbill_ser, "spy_ser": spy_ser}


def signal_dict(inputs: dict) -> dict:
    panel = inputs["panel"]
    close_df = panel.field("Close")
    # *** CRITICAL*** every rolling window ends at its own row; a decision at T reads row T only.
    return_df = close_df.pct_change(fill_method=None)
    spy_index_ser = (1.0 + inputs["spy_ser"]).cumprod()
    period_ser = pd.Series(panel.date_index.to_period("M"), index=panel.date_index)
    month_end_mask = (period_ser != period_ser.shift(-1)).to_numpy()
    month_end_mask[-1] = False  # the last panel session has no next open
    return {
        "return_mat": return_df.to_numpy(dtype=float),
        "size_mat": panel.field("Turnover").rolling(SIZE_WINDOW_INT, min_periods=SIZE_MIN_INT).median().to_numpy(dtype=float),
        "m12_mat": (close_df.shift(SKIP_INT) / close_df.shift(LOOKBACK_INT) - 1.0).to_numpy(dtype=float),
        "sigma_mat": (return_df.rolling(VOL_WINDOW_INT, min_periods=VOL_MIN_INT).std() * np.sqrt(252.0)).to_numpy(dtype=float),
        "member_mat": panel.member_df.to_numpy() > 0,
        "close_mat": close_df.to_numpy(dtype=float),
        "raw_close_mat": panel.field("Unadjusted Close").to_numpy(dtype=float),
        "regime_vec": (spy_index_ser > spy_index_ser.rolling(REGIME_SMA_INT).mean()).to_numpy(),
        "decision_row_vec": np.flatnonzero(month_end_mask & (np.arange(len(panel.date_index)) >= LOOKBACK_INT + 1)),
    }


def weight_df(inputs: dict, signal: dict, config_dict: dict) -> pd.DataFrame:
    """Target weights indexed by execution date (the session after each decision)."""
    panel = inputs["panel"]
    symbol_arr = np.array(panel.symbol_list)
    row_dict = {}
    for row_int in signal["decision_row_vec"]:
        with np.errstate(invalid="ignore"):
            eligible_vec = (
                signal["member_mat"][row_int] & np.isfinite(signal["close_mat"][row_int]) & np.isfinite(signal["m12_mat"][row_int])
                & np.isfinite(signal["sigma_mat"][row_int]) & (signal["sigma_mat"][row_int] > 0) & np.isfinite(signal["size_mat"][row_int])
                & (signal["raw_close_mat"][row_int] >= PRICE_FLOOR_FLOAT)
            )
        target_ser = pd.Series(dtype=float)
        idx_vec = np.flatnonzero(eligible_vec)
        if len(idx_vec) >= config_dict["pool_int"] and (signal["regime_vec"][row_int] or not config_dict["regime_bool"]):
            pool_vec = idx_vec[np.argsort(-signal["size_mat"][row_int, idx_vec], kind="stable")[: config_dict["pool_int"]]]
            score_vec = signal["m12_mat"][row_int, pool_vec]
            if config_dict["rank_str"] == "m12_over_vol":
                score_vec = score_vec / signal["sigma_mat"][row_int, pool_vec]
            pick_vec = pool_vec[np.argsort(-score_vec, kind="stable")[: config_dict["top_count_int"]]]
            count_int = len(pick_vec)
            if config_dict["weight_str"] == "inverse_vol":
                raw_vec = 1.0 / signal["sigma_mat"][row_int, pick_vec]
                weight_vec = np.minimum(raw_vec / raw_vec.sum(), 2.0 / count_int)
                weight_vec = weight_vec / weight_vec.sum() if weight_vec.sum() > 1.0 else weight_vec
            else:
                weight_vec = np.full(count_int, 1.0 / count_int)
            if config_dict["vol_target_float"] > 0:
                block_mat = np.nan_to_num(signal["return_mat"][row_int - VOL_WINDOW_INT + 1: row_int + 1][:, pick_vec])
                vol_float = float(np.std(block_mat @ weight_vec, ddof=1) * np.sqrt(252.0))
                weight_vec = weight_vec * min(1.0, config_dict["vol_target_float"] / vol_float) if vol_float > 0 else weight_vec
            target_ser = pd.Series(weight_vec, index=symbol_arr[pick_vec])
        row_dict[panel.date_index[row_int + 1]] = target_ser
    return pd.DataFrame(row_dict).T.fillna(0.0)


def run_config(inputs: dict, signal: dict, config_dict: dict) -> pd.Series:
    panel = inputs["panel"]
    target_df = weight_df(inputs, signal, config_dict)
    symbol_list = [s for s in target_df.columns if (target_df[s] != 0).any()]
    target_df = target_df[symbol_list]
    close_df = panel.field("Close")[symbol_list]
    result = simulate(panel.field("Open")[symbol_list], close_df, panel.field("Dividend")[symbol_list].fillna(0.0), target_df,
                      start_date=target_df.index[0])
    # idle cash at the T-bill rate (scripts/research/scout_robustness_20261002/run.py cash_credited)
    position_df = result.daily_position_df
    long_value_ser = (position_df * close_df.reindex(index=position_df.index, columns=position_df.columns)).clip(lower=0.0).sum(axis=1, min_count=0)
    idle_ser = (1.0 - long_value_ser / result.total_value_ser).clip(0.0, 1.0)
    daily_ser = result.daily_return_ser + idle_ser.shift(1).fillna(0.0) * inputs["tbill_ser"].reindex(result.daily_return_ser.index).fillna(0.0)
    daily_ser.attrs["exposure_float"] = float(1.0 - idle_ser.mean())
    return daily_ser


def label_str(config_dict: dict) -> str:
    return (f"top{config_dict['pool_int']}_N{config_dict['top_count_int']}_{config_dict['rank_str']}_{config_dict['weight_str']}"
            f"_vt{int(config_dict['vol_target_float'] * 100)}_reg{int(config_dict['regime_bool'])}")


def control_config(config_dict: dict) -> dict:
    return {**config_dict, "top_count_int": config_dict["pool_int"], "rank_str": "m12"}


def _init() -> None:
    _WORKER["inputs"] = load_inputs()
    _WORKER["signal"] = signal_dict(_WORKER["inputs"])


def _task(config_dict: dict):
    daily_ser = run_config(_WORKER["inputs"], _WORKER["signal"], config_dict)
    return label_str(config_dict), daily_ser, daily_ser.attrs["exposure_float"]


def window_return(daily_ser: pd.Series, start_str: str, end_str: str) -> float:
    part_ser = daily_ser.loc[start_str:end_str]
    return float((1.0 + part_ser).prod() - 1.0) if len(part_ser) else float("nan")


def drawdown_dict(daily_ser: pd.Series) -> dict:
    value_ser = (1.0 + daily_ser).cumprod()
    drawdown_ser = value_ser / value_ser.cummax() - 1.0
    trough_ts = drawdown_ser.idxmin()
    return {"peak_str": str(value_ser.loc[:trough_ts].idxmax().date()), "trough_str": str(trough_ts.date()), "depth_float": float(drawdown_ser.min())}


def main() -> None:
    if "--register" in sys.argv:
        row_dict = register(Ledger(), REGISTRATION)
        print(f"Registered {REGISTRATION_ID_STR} as ledger row {row_dict['row_id_int']} ({row_dict['utc_ts_str']}).")
        return
    if REGISTRATION_ID_STR not in {r["registration_id_str"] for r in Ledger().rows("registration")}:
        raise SystemExit("Register first: run with --register.")

    name_list = sorted(GRID_DICT)
    config_list = [dict(zip(name_list, combo)) for combo in itertools.product(*[GRID_DICT[n] for n in name_list])]
    control_list = []
    for config_dict in config_list:
        if control_config(config_dict) not in control_list:
            control_list.append(control_config(config_dict))
    with Pool(8, initializer=_init) as pool_obj:
        out_list = pool_obj.map(_task, config_list + control_list, chunksize=2)
    daily_df = pd.DataFrame({label: ser for label, ser, _ in out_list})
    exposure_dict = {label: exposure for label, _, exposure in out_list}
    inputs = load_inputs()
    tbill_ser, spy_ser = inputs["tbill_ser"], inputs["spy_ser"].reindex(daily_df.index)
    daily_df["SPY"] = spy_ser
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    daily_df.to_parquet(OUTPUT_DIR_PATH / "daily_returns.parquet")

    row_list = []
    for config_dict in config_list:
        label, control_label = label_str(config_dict), label_str(control_config(config_dict))
        stat, control_stat = performance_dict(daily_df[label], tbill_ser), performance_dict(daily_df[control_label], tbill_ser)
        row_list.append({
            **config_dict, "label": label, "sharpe": stat["sharpe_float"], "excess_sharpe": stat["excess_sharpe_float"], "cagr": stat["cagr_float"],
            "vol": stat["volatility_float"], "max_dd": stat["max_drawdown_float"], "exposure": exposure_dict[label],
            "control_sharpe": control_stat["sharpe_float"], "control_cagr": control_stat["cagr_float"], "control_max_dd": control_stat["max_drawdown_float"],
            "p_gt_control": paired_sharpe_probability(daily_df[label], daily_df[control_label]),
            **{f"crisis_{k}": window_return(daily_df[label], a, b) for k, (a, b) in CRISIS_DICT.items()},
        })
    grid_df = pd.DataFrame(row_list)

    # neighbourhood median (D13): the configuration and its one-step grid neighbours
    index_df = pd.DataFrame({n: grid_df[n].map({v: i for i, v in enumerate(GRID_DICT[n])}) for n in name_list})
    index_mat = index_df.to_numpy()
    grid_df["neighbourhood_median_sharpe"] = [
        float(np.median(grid_df["sharpe"].to_numpy()[np.abs(index_mat - index_mat[i]).sum(axis=1) <= 1])) for i in range(len(grid_df))
    ]
    grid_df.to_csv(OUTPUT_DIR_PATH / "grid.csv", index=False)
    chosen_row = grid_df.loc[grid_df["neighbourhood_median_sharpe"].idxmax()]

    switch_dict = {}
    for name_str in name_list:
        switch_dict[name_str] = {str(v): {k: float(grid_df.loc[grid_df[name_str] == v, k].mean()) for k in ("sharpe", "cagr", "vol", "max_dd", "p_gt_control", "exposure")}
                                 for v in GRID_DICT[name_str]}
    control_stat_dict = {label_str(c): {**{k: v for k, v in performance_dict(daily_df[label_str(c)], tbill_ser).items() if k != "year_return_dict"},
                                        "exposure": exposure_dict[label_str(c)]} for c in control_list}
    summary_dict = {
        "registration_id_str": REGISTRATION_ID_STR, "panel_snapshot_id_str": inputs["panel"].snapshot_id_str,
        "window_str": f"{daily_df.index[0].date()} to {daily_df.index[-1].date()}",
        "chosen": chosen_row.to_dict(), "chosen_drawdown": drawdown_dict(daily_df[chosen_row["label"]]),
        "peak_config": grid_df.loc[grid_df["sharpe"].idxmax()].to_dict(),
        "share_beating_control_float": float((grid_df["sharpe"] > grid_df["control_sharpe"]).mean()),
        "share_p80_float": float((grid_df["p_gt_control"] >= 0.80).mean()),
        "switch_mean": switch_dict, "controls": control_stat_dict,
        "spy": {k: v for k, v in performance_dict(spy_ser, tbill_ser).items() if k != "year_return_dict"},
        "raw_drawdowns": {label: drawdown_dict(daily_df[label]) for label in
                          ("top50_N10_m12_equal_vt0_reg0", "top50_N50_m12_equal_vt0_reg0", "SPY")},
        "chosen_year_returns": performance_dict(daily_df[chosen_row["label"]], tbill_ser)["year_return_dict"],
    }
    (OUTPUT_DIR_PATH / "summary.json").write_text(json.dumps(summary_dict, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200)
    show_list = ["label", "sharpe", "cagr", "vol", "max_dd", "exposure", "control_sharpe", "control_max_dd", "p_gt_control", "neighbourhood_median_sharpe"]
    print(grid_df[show_list].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in summary_dict.items() if k not in ("chosen_year_returns",)}, indent=1, default=float))
    print({y: round(v, 3) for y, v in summary_dict["chosen_year_returns"].items()})


if __name__ == "__main__":
    main()
