"""Part C (PROTOCOL.md): MCPT of the Zorro Z9 search on its own ETF lists, in sample (before publication, 2017-10-01).

A vectorised, gross, close-to-close replica of the Pakal audit's Z9 rules (`Pakal/pakal-research/
zorro_zsystems_daily_audit.py`, z9_weights_at / z9_schedule): rank the list by momentum at the close of T, hold the
top third with positive momentum (bonds ranked with the rest; with the crash filter on, only type-1 bonds when SPY
is below its 200-day average), equal or momentum weights, from the close of T+1 until the next decision; cash earns
zero. All configurations are scored from one common start, so they share one return window. Score = the best
configuration's Sharpe minus the list's equal-weight Sharpe. Null: plain date-row shuffle of the list + SPY returns.

    uv run python scripts/research/scout_p4b_calibration_20261001/run_z9_mcpt.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.stats.mcpt import mcpt

PAKAL_AUDIT_PATH = Path(r"C:\Users\User\Documents\workspace\Pakal\pakal-research\reports\zorro_zsystems_daily_audit")
OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "p4b_calibration"
PUBLICATION_STR, PERMUTATION_COUNT_INT = "2017-10-01", 1000
TYPE_DICT_DICT = {
    "L2017": {"XBI": 0, "ITB": 0, "SMH": 0, "XLV": 0, "VOO": 1, "AGG": 1, "HYG": 1, "IGSB": 1, "TLT": 1},
    "N14": {"XLB": 0, "XLE": 0, "XLF": 0, "XLI": 0, "XLK": 0, "XLP": 0, "XLU": 0, "XLV": 0, "XLY": 0,
            "EFA": 2, "EEM": 2, "IEF": 1, "TLT": 1, "GLD": 1},
}
GRID_LIST = list(itertools.product(("mean200", "roc252", "roc126"), (25, 35), ("off", "sma200"), ("equal", "momentum")))
LOOKBACK_DICT = {"mean200": 200, "roc252": 252, "roc126": 126}
WARM_UP_INT = 253  # rows before the first decision: the longest lookback (252) plus one


def _momentum_mat(return_mat: np.ndarray, price_mat: np.ndarray, momentum_str: str) -> np.ndarray:
    # *** CRITICAL*** every momentum ends at Close_T.
    if momentum_str == "mean200":
        cumulative_mat = np.vstack([np.zeros(return_mat.shape[1]), np.cumsum(return_mat, axis=0)])
        out_mat = np.full_like(return_mat, np.nan)
        out_mat[199:] = (cumulative_mat[200:] - cumulative_mat[:-200]) / 200.0
        return out_mat
    lookback_int = LOOKBACK_DICT[momentum_str]
    out_mat = np.full_like(price_mat, np.nan)
    out_mat[lookback_int:] = price_mat[lookback_int:] / price_mat[:-lookback_int] - 1.0
    return out_mat


def config_daily_vec(return_mat: np.ndarray, type_vec: np.ndarray, config_tuple: tuple) -> np.ndarray:
    """Daily gross return of one configuration; column 0 of return_mat is SPY, the rest the list."""
    momentum_str, rebalance_int, crash_str, weighting_str = config_tuple
    price_mat = np.cumprod(1.0 + return_mat, axis=0)
    list_return_mat, spy_price_vec = return_mat[:, 1:], price_mat[:, 0]
    momentum_mat = _momentum_mat(list_return_mat, price_mat[:, 1:], momentum_str)
    sma_vec = pd.Series(spy_price_vec).rolling(200, min_periods=200).mean().to_numpy()
    top_int = max(1, round(type_vec.size / 3.0))
    date_count_int = return_mat.shape[0]
    daily_vec = np.zeros(date_count_int)
    decision_vec = np.arange(WARM_UP_INT, date_count_int - 1, rebalance_int)
    for decision_idx_int, t_int in enumerate(decision_vec):
        crash_bool = crash_str == "sma200" and spy_price_vec[t_int] < sma_vec[t_int]
        eligible_vec = (type_vec == 1) if crash_bool else np.ones(type_vec.size, dtype=bool)
        score_vec = np.where(eligible_vec & np.isfinite(momentum_mat[t_int]), momentum_mat[t_int], -np.inf)
        chosen_vec = np.argsort(-score_vec, kind="stable")[:top_int]
        chosen_vec = chosen_vec[score_vec[chosen_vec] > 0]
        if chosen_vec.size == 0:
            continue
        if weighting_str == "equal":
            weight_vec = np.full(chosen_vec.size, 1.0 / top_int)
        else:
            weight_vec = score_vec[chosen_vec] / score_vec[chosen_vec].sum() * (chosen_vec.size / top_int)
        start_int = t_int + 2  # filled at the close of T+1, first return on T+2
        end_int = decision_vec[decision_idx_int + 1] + 2 if decision_idx_int + 1 < decision_vec.size else date_count_int
        if start_int >= end_int:
            continue
        growth_mat = np.cumprod(1.0 + list_return_mat[start_int:end_int, chosen_vec], axis=0)
        value_vec = np.concatenate([[weight_vec.sum() + (1.0 - weight_vec.sum())], (growth_mat * weight_vec).sum(axis=1) + (1.0 - weight_vec.sum())])
        daily_vec[start_int:end_int] = value_vec[1:] / value_vec[:-1] - 1.0
    return daily_vec


def _sharpe(daily_vec: np.ndarray) -> float:
    window_vec = daily_vec[WARM_UP_INT + 2 :]
    return float(window_vec.mean() / window_vec.std(ddof=1) * np.sqrt(252.0))


def make_search(type_vec: np.ndarray):
    def search(return_mat: np.ndarray) -> float:
        best_float = max(_sharpe(config_daily_vec(return_mat, type_vec, config_tuple)) for config_tuple in GRID_LIST)
        return best_float - _sharpe(return_mat[:, 1:].mean(axis=1))
    return search


def main() -> None:
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    long_df = pd.read_parquet(PAKAL_AUDIT_PATH / "data" / "norgate_zsystems_panel.parquet")
    long_df["Date"] = pd.to_datetime(long_df["Date"])
    close_df = long_df.pivot(index="Date", columns="Symbol", values="Close")
    audit_df = pd.read_parquet(PAKAL_AUDIT_PATH / "tables" / "daily_returns_all_runs.parquet")
    result_list = []
    for list_str, type_dict in TYPE_DICT_DICT.items():
        symbol_list = ["SPY", *type_dict]
        list_close_df = close_df[symbol_list].loc[: pd.Timestamp(PUBLICATION_STR) - pd.Timedelta(days=1)]
        list_close_df = list_close_df.loc[list_close_df.notna().all(axis=1).idxmax() :].ffill()
        return_df = list_close_df.pct_change().iloc[1:]
        return_mat = return_df.to_numpy()
        type_vec = np.array(list(type_dict.values()))
        search_fn = make_search(type_vec)

        baseline_config = ("mean200", 25, "off", "equal")
        replica_ser = pd.Series(config_daily_vec(return_mat, type_vec, baseline_config), index=return_df.index).iloc[WARM_UP_INT + 2 :]
        audit_ser = audit_df[f"Z9|{list_str}|mean200|r25|off|equal"].reindex(replica_ser.index).dropna()
        replica_on_audit_ser = replica_ser.reindex(audit_ser.index)
        sharpe_by_config_dict = {"|".join(map(str, c)): _sharpe(config_daily_vec(return_mat, type_vec, c)) for c in GRID_LIST}

        started_float = time.time()
        result = mcpt(search_fn, return_mat, PERMUTATION_COUNT_INT, 4_000_000 + len(result_list))
        result_list.append({
            "list_str": list_str,
            "in_sample_str": f"{return_df.index[WARM_UP_INT + 2].date()} to {return_df.index[-1].date()}",
            "best_config_str": max(sharpe_by_config_dict, key=sharpe_by_config_dict.get),
            "best_sharpe_float": max(sharpe_by_config_dict.values()),
            "equal_weight_sharpe_float": _sharpe(return_mat[:, 1:].mean(axis=1)),
            "observed_score_float": result.observed_score_float,
            "null_median_float": float(np.median(result.null_score_vec)),
            "null_95_float": float(np.quantile(result.null_score_vec, 0.95)),
            "mcpt_p_float": result.p_value_float,
            "replica_vs_audit_baseline": {
                "days_int": int(audit_ser.size),
                "correlation_float": float(np.corrcoef(replica_on_audit_ser, audit_ser)[0, 1]),
                "replica_sharpe_float": float(replica_on_audit_ser.mean() / replica_on_audit_ser.std() * np.sqrt(252)),
                "audit_sharpe_float": float(audit_ser.mean() / audit_ser.std() * np.sqrt(252)),
            },
            "sharpe_by_config": sharpe_by_config_dict,
            "seconds_float": time.time() - started_float,
        })
        print(json.dumps({k: v for k, v in result_list[-1].items() if k != "sharpe_by_config"}, indent=1, default=str), flush=True)
    (OUTPUT_DIR_PATH / "z9_mcpt.json").write_text(json.dumps(result_list, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
