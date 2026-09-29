"""Core library for the NDX momentum pod parameter-robustness study (research only).

Frozen plan: docs/research/NDX_PARAM_ROBUSTNESS_PREREG_20260926.md. Nothing here is imported by live code.

Three parts:
1. Data: point-in-time universe panels loaded through the repo's own NDX data function (read-only).
2. Features and selection: scale-free scores, filters, rank buffer, rebalance offsets, VXN scaler.
3. A fast replica of the repo engine for monthly top-N rotations (same order semantics, costs and dividend
   ledger as `alpha/engine/strategy.py`), checked against real engine runs before any grid is read.

Core formulas (decision date t = a month's last session shifted by k sessions; fills at the next open):

    ROC_m(t)       = Close(t) / Close(anchor_{t-m}) - 1            anchors = the same shifted monthly schedule
    ROC12-1(t)     = Close(anchor_{t-1}) / Close(anchor_{t-12}) - 1
    NATR_n(t)      = mean(TR over the last n sessions) / Close(t)
    sigma63(t)     = std of the last 63 daily close-to-close returns (>= 60 valid)
    score(t)       = numerator(t) / denominator(t)                 (denominator "none" -> numerator)
    eligible_i(t)  = member_i(t) and filter_i(t) and finite(score_i(t)), and regime(t) passes
    exposure(t)    = clip(target / VXN(t), floor, 1)               (1 when the scaler is off)
    target_shares  = int(V(t) * w_i / Close_i(t))                  V(t) = total value at the decision close

Every feature above is unchanged if a stock's prices are all multiplied by one constant, so no selection depends
on how far back splits were adjusted. The incumbent L (ROC12 / ATR20 in decision-day dollars) and the biased B
(ROC12 / ATR20 in today's CAPITALSPECIAL dollars) are computed only as references.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

RESULTS_DIR_PATH = Path(
    os.environ.get(
        "NDX_PARAM_ROBUSTNESS_OUT",
        str(REPO_ROOT_PATH / "results" / "research" / "ndx_param_robustness_20260926"),
    )
)
CACHE_DIR_PATH = RESULTS_DIR_PATH / "_cache"

UNIVERSE_INDEXNAME_DICT = {"NDX": "Nasdaq 100", "SP500": "S&P 500", "R1000": "Russell 1000"}
TRADING_START_TS = pd.Timestamp("2000-01-01")
FIRST_DECISION_MONTH_STR = "2000-01"
DIVIDEND_NET_RATE_FLOAT = 0.75  # engine: 25% withholding on long dividends, no reinvestment
COMMISSION_PER_SHARE_FLOAT = 0.005
COMMISSION_MINIMUM_FLOAT = 1.0
ENGINE_SLIPPAGE_FLOAT = 0.00025
STRESS_SLIPPAGE_FLOAT = 0.00075  # engine 2.5 bps + 5 bps stress, per side
CAPITAL_BASE_FLOAT = 100_000.0


# ======================================================================================================================
# 1. data
# ======================================================================================================================
def universe_cache_path(universe_str: str) -> Path:
    return CACHE_DIR_PATH / f"universe_{universe_str}.pkl"


def prepare_universe(universe_str: str) -> dict:
    """Load one PIT universe through the repo's own NDX data function (only the index name changes)."""
    from data.norgate_loader import CAPITALSPECIAL_ADJUSTMENT_STR, load_price_timeseries
    from strategies.momentum.strategy_mo_atr_normalized_ndx import get_monthly_decision_close_df
    from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
        DEFAULT_CONFIG,
        get_vxn_scaled_atr_normalized_ndx_data,
    )

    config_obj = dataclasses.replace(DEFAULT_CONFIG, indexname_str=UNIVERSE_INDEXNAME_DICT[universe_str])
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = get_vxn_scaled_atr_normalized_ndx_data(
        config_obj,
        include_total_return_benchmark_bool=False,
    )
    date_index = pd.DatetimeIndex(pricing_data_df.index)
    symbol_list = [str(symbol_str) for symbol_str in universe_df.columns]

    def panel_arr(field_str: str, dtype_obj=np.float64) -> np.ndarray:
        return np.column_stack(
            [pricing_data_df[(symbol_str, field_str)].to_numpy(dtype=np.float64) for symbol_str in symbol_list]
        ).astype(dtype_obj)

    panel_dtype_obj = np.float64 if universe_str == "NDX" else np.float32
    close_arr = panel_arr("Close")
    universe_dict = {
        "universe_str": universe_str,
        "date_index": date_index,
        "symbol_list": symbol_list,
        "open_arr": panel_arr("Open", panel_dtype_obj),
        "high_arr": panel_arr("High", panel_dtype_obj),
        "low_arr": panel_arr("Low", panel_dtype_obj),
        "close_arr": close_arr.astype(panel_dtype_obj),
        "volume_arr": panel_arr("Volume", panel_dtype_obj),
        "dividend_arr": panel_arr("Dividend", panel_dtype_obj),
        "unadjusted_close_arr": panel_arr("Unadjusted Close", panel_dtype_obj),
        # *** CRITICAL *** audited PIT membership is already forward-filled onto the price index (never back-filled);
        # row t = the latest constituent row on or before t, exactly what the engine's as-of lookup returns.
        "member_arr": universe_df.reindex(date_index).fillna(0).to_numpy(dtype=np.int8),
        "spy_close_ser": pricing_data_df[(config_obj.regime_symbol_str, "Close")].astype(float),
        "vxn_close_ser": vxn_scale_signal_df["vxn_close"].astype(float),
        "repo_schedule_df": rebalance_schedule_df.copy(),
        "month_end_index": pd.DatetimeIndex(
            get_monthly_decision_close_df(pd.DataFrame({"x": close_arr[:, 0]}, index=date_index)).index
        ),
    }
    qqq_df = load_price_timeseries(
        "QQQ",
        adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR,
        start_date_str=config_obj.history_start_date_str,
        end_date_str=None,
    )
    universe_dict["qqq_close_ser"] = qqq_df["Close"].astype(float).reindex(date_index)
    del pricing_data_df
    CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    with open(universe_cache_path(universe_str), "wb") as file_obj:
        pickle.dump(universe_dict, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    return universe_dict


def load_universe(universe_str: str) -> dict:
    with open(universe_cache_path(universe_str), "rb") as file_obj:
        return pickle.load(file_obj)


# ======================================================================================================================
# 2. features
# ======================================================================================================================
def rolling_mean_arr(value_arr: np.ndarray, window_int: int, min_periods_int: int | None = None) -> np.ndarray:
    # *** CRITICAL *** trailing window ending at row t (inclusive); never centred, never forward-looking.
    return (
        pd.DataFrame(value_arr)
        .rolling(window=window_int, min_periods=window_int if min_periods_int is None else min_periods_int)
        .mean()
        .to_numpy()
    )


def true_range_arr(high_arr: np.ndarray, low_arr: np.ndarray, close_arr: np.ndarray) -> np.ndarray:
    """Repo TR: max(H-L, |H-Cprev|, |L-Cprev|), NaN propagates."""
    # *** CRITICAL *** prior close = row t-1, so TR_t uses nothing after the close of t.
    prior_close_arr = np.vstack([np.full((1, close_arr.shape[1]), np.nan), close_arr[:-1]])
    return np.maximum(
        np.maximum(high_arr - low_arr, np.abs(high_arr - prior_close_arr)),
        np.abs(low_arr - prior_close_arr),
    )


class FeatureBook:
    """Lazily computed daily feature panels (T x S) for one universe."""

    def __init__(self, universe_dict: dict):
        self.u = universe_dict
        self.close_arr = universe_dict["close_arr"].astype(np.float64)
        self._cache_dict: dict = {}
        # Invariance check V2 only: restated numerator tables {numerator_str: months x symbols} for the k = 0 schedule.
        self.numerator_override_dict: dict[str, np.ndarray] = {}
        self.date_count_int, self.symbol_count_int = self.close_arr.shape
        # symbol tie-break rank: ascending symbol string, as the repo's mergesort on (score desc, symbol asc)
        self.symbol_rank_vec = np.argsort(np.argsort(np.array(universe_dict["symbol_list"], dtype=object)))

    def _get(self, key_tuple, builder_fn):
        if key_tuple not in self._cache_dict:
            self._cache_dict[key_tuple] = builder_fn()
        return self._cache_dict[key_tuple]

    def atr_cs(self, window_int: int) -> np.ndarray:
        def build():
            tr_arr = true_range_arr(
                self.u["high_arr"].astype(np.float64), self.u["low_arr"].astype(np.float64), self.close_arr
            )
            return rolling_mean_arr(tr_arr, window_int)

        return self._get(("atr_cs", window_int), build)

    def natr(self, window_int: int) -> np.ndarray:
        # ATR and close are in the same (any) price units on day t, so the ratio is scale-free.
        return self._get(("natr", window_int), lambda: self.atr_cs(window_int) / self.close_arr)

    def sigma63(self) -> np.ndarray:
        def build():
            # *** CRITICAL *** close-to-close return of day t uses closes t-1 and t only.
            return_arr = self.close_arr[1:] / self.close_arr[:-1] - 1.0
            return_arr = np.vstack([np.full((1, self.symbol_count_int), np.nan), return_arr])
            return pd.DataFrame(return_arr).rolling(window=63, min_periods=60).std().to_numpy()

        return self._get(("sigma63",), build)

    def trend_pass(self, window_int: int) -> np.ndarray:
        if window_int == 0:
            return np.ones_like(self.close_arr, dtype=bool)

        def build():
            sma_arr = rolling_mean_arr(self.close_arr, window_int)
            with np.errstate(invalid="ignore"):
                return self.close_arr > sma_arr  # NaN compares False -> not eligible, as in the repo

        return self._get(("trend", window_int), build)

    def regime_pass(self, regime_str: str) -> np.ndarray:
        if regime_str == "none":
            return np.ones(self.date_count_int, dtype=bool)

        def build():
            regime_close_ser = self.u["spy_close_ser"] if regime_str == "SPY" else self.u["qqq_close_ser"]
            close_vec = regime_close_ser.to_numpy(dtype=np.float64)
            # *** CRITICAL *** trailing 200-session SMA of the regime symbol, known at the decision close.
            sma_vec = pd.Series(close_vec).rolling(window=200, min_periods=200).mean().to_numpy()
            with np.errstate(invalid="ignore"):
                return close_vec > sma_vec

        return self._get(("regime", regime_str), build)

    def split_factor(self) -> np.ndarray:
        """R(t) = Unadjusted Close / Close: capital events AFTER t. Used only for the L/B references and the
        commission-fixed sensitivity, never inside a scale-free score."""

        def build():
            factor_arr = self.u["unadjusted_close_arr"].astype(np.float64) / self.close_arr
            return np.where(np.isfinite(factor_arr) & (factor_arr > 0), factor_arr, np.nan)

        return self._get(("split_factor",), build)

    def adv20_dollar(self) -> np.ndarray:
        def build():
            # Close_CS x Volume_CS: split factors cancel, so this is the traded dollar value of the day.
            dollar_arr = self.close_arr * self.u["volume_arr"].astype(np.float64)
            # *** CRITICAL *** trailing 20-session median ending at the decision close.
            return pd.DataFrame(dollar_arr).rolling(window=20, min_periods=20).median().to_numpy()

        return self._get(("adv20",), build)

    def vxn_scale_vec(self, decision_pos_vec: np.ndarray, target_float: float | None, floor_float: float) -> np.ndarray:
        if target_float is None:
            return np.ones(len(decision_pos_vec))
        vxn_close_ser = self.u["vxn_close_ser"].dropna().sort_index()
        decision_ts_index = self.u["date_index"][decision_pos_vec]
        # *** CRITICAL *** as-of lookup: latest VXN close on or before the decision date, never a later one.
        row_vec = vxn_close_ser.index.searchsorted(decision_ts_index, side="right") - 1
        if (row_vec < 0).any():
            raise RuntimeError("VXN has no close on or before a decision date.")
        vxn_vec = vxn_close_ser.to_numpy()[row_vec]
        return np.clip(float(target_float) / vxn_vec, float(floor_float), 1.0)


# ======================================================================================================================
# 3. schedule and scores
# ======================================================================================================================
def build_schedule(universe_dict: dict, offset_int: int) -> dict:
    """Shifted monthly schedule.

    anchor_j = position of month j's last session in the repo's month-end list (history from 1999-01);
    decision_j = anchor_j + offset; execution_j = decision_j + 1 (next session open).
    Decisions are traded from the January-2000 month on (the repo's first decision is 2000-01-31).
    """
    date_index = universe_dict["date_index"]
    anchor_pos_vec = date_index.get_indexer(universe_dict["month_end_index"])
    if (anchor_pos_vec < 0).any():
        raise RuntimeError("month-end anchor missing from the price index")
    # *** CRITICAL *** shifting the whole schedule by k sessions; the ROC anchors shift with it, so every
    # look-back still ends at the decision close and starts on the same shifted schedule m months earlier.
    decision_pos_vec = anchor_pos_vec + int(offset_int)
    valid_vec = (decision_pos_vec >= 0) & (decision_pos_vec + 1 < len(date_index))
    month_period_vec = universe_dict["month_end_index"].to_period("M")
    trade_vec = valid_vec & (month_period_vec >= pd.Period(FIRST_DECISION_MONTH_STR, "M"))
    trade_vec &= date_index[np.clip(decision_pos_vec + 1, 0, len(date_index) - 1)] >= TRADING_START_TS
    return {
        "offset_int": int(offset_int),
        "decision_pos_vec": decision_pos_vec,  # all months (look-back anchors)
        "valid_vec": valid_vec,
        "trade_vec": trade_vec,  # months that are actually traded
    }


def roc_table(close_arr: np.ndarray, schedule_dict: dict, numerator_str: str) -> np.ndarray:
    """Numerator per schedule row (months x symbols), NaN where any needed close is missing."""
    decision_pos_vec = schedule_dict["decision_pos_vec"]
    month_count_int = len(decision_pos_vec)
    anchor_close_arr = np.full((month_count_int, close_arr.shape[1]), np.nan)
    valid_vec = schedule_dict["valid_vec"]
    anchor_close_arr[valid_vec] = close_arr[decision_pos_vec[valid_vec]]

    def roc(back_int: int, skip_int: int = 0) -> np.ndarray:
        # *** CRITICAL *** close at schedule row j - skip over close at row j - back: both at or before row j.
        out_arr = np.full_like(anchor_close_arr, np.nan)
        out_arr[back_int:] = anchor_close_arr[back_int - skip_int: month_count_int - skip_int] / anchor_close_arr[: month_count_int - back_int] - 1.0
        return out_arr

    if numerator_str.startswith("ROC") and "-" not in numerator_str:
        return roc(int(numerator_str[3:]))
    if numerator_str == "ROC12-1":
        return roc(12, 1)
    if numerator_str == "B3612":
        return (roc(3) + roc(6) + roc(12)) / 3.0
    if numerator_str == "B612":
        return (roc(6) + roc(12)) / 2.0
    raise ValueError(numerator_str)


@dataclasses.dataclass(frozen=True)
class Cell:
    numerator_str: str = "ROC12"
    denominator_str: str = "NATR20"  # none | NATR20 | NATR63 | ATR20_LIVE (L, reference) | ATR20_CS (B, reference)
    n_int: int = 10
    weight_str: str = "EW"  # EW | IV
    stock_filter_int: int = 100  # 0 = none
    regime_str: str = "SPY"  # none | SPY | QQQ
    buffer_int: int = 0
    offset_int: int = 0
    vxn_target_float: float | None = 22.0  # None = no scaler
    vxn_floor_float: float = 0.25
    liquidity_str: str = "none"  # none | REL25 | REL50 (bottom x% of PIT members by ADV20 excluded) | ABS5M

    @property
    def key_str(self) -> str:
        vxn_str = "off" if self.vxn_target_float is None else f"{self.vxn_target_float:g}-{self.vxn_floor_float:g}"
        liquidity_suffix_str = "" if self.liquidity_str == "none" else f"|Q{self.liquidity_str}"
        return (
            f"{self.numerator_str}/{self.denominator_str}|N{self.n_int}{self.weight_str}|F{self.stock_filter_int}"
            f"|G{self.regime_str}|b{self.buffer_int}|k{self.offset_int:+d}|V{vxn_str}{liquidity_suffix_str}"
        )

    @property
    def scale_free_bool(self) -> bool:
        return self.denominator_str in ("none", "NATR20", "NATR63")


ANCHOR_CELL = Cell()
L_CELL = Cell(denominator_str="ATR20_LIVE")
B_CELL = Cell(denominator_str="ATR20_CS")


def score_table(feature_obj: FeatureBook, schedule_dict: dict, cell: Cell) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return _score_table(feature_obj, schedule_dict, cell)


def _score_table(feature_obj: FeatureBook, schedule_dict: dict, cell: Cell) -> np.ndarray:
    if cell.numerator_str in feature_obj.numerator_override_dict and schedule_dict["offset_int"] == 0:
        numerator_arr = feature_obj.numerator_override_dict[cell.numerator_str]
    else:
        numerator_arr = roc_table(feature_obj.close_arr, schedule_dict, cell.numerator_str)
    decision_pos_vec = np.where(schedule_dict["valid_vec"], schedule_dict["decision_pos_vec"], 0)
    if cell.denominator_str == "none":
        score_arr = numerator_arr
    elif cell.denominator_str in ("NATR20", "NATR63"):
        score_arr = numerator_arr / feature_obj.natr(int(cell.denominator_str[4:]))[decision_pos_vec]
    elif cell.denominator_str == "ATR20_CS":
        score_arr = numerator_arr / feature_obj.atr_cs(20)[decision_pos_vec]
    elif cell.denominator_str == "ATR20_LIVE":
        # *** CRITICAL *** L: ATR20 in the dollars of decision day t = R(t) x ATR20_CS(t) (stage-2 identity).
        # R(t) contains events after t, but the product restores day-t units exactly, which a day-t snapshot shows.
        score_arr = numerator_arr / (feature_obj.atr_cs(20)[decision_pos_vec] * feature_obj.split_factor()[decision_pos_vec])
    else:
        raise ValueError(cell.denominator_str)
    score_arr = np.where(np.isfinite(score_arr), score_arr, np.nan)
    score_arr[~schedule_dict["valid_vec"]] = np.nan
    return score_arr


# ======================================================================================================================
# 4. selection
# ======================================================================================================================
def liquidity_pass_vec(feature_obj: FeatureBook, decision_pos_int: int, member_vec: np.ndarray, liquidity_str: str) -> np.ndarray:
    """Eligibility by 20-session median traded dollars (Close_CS x Volume_CS = actual dollars, split invariant).

    REL25 / REL50: drop PIT members below the 25th / 50th percentile of ADV20 among that day's PIT members.
    ABS5M: drop names below $5M a day. A name without ADV20 fails every filter.
    """
    if liquidity_str == "none":
        return np.ones(len(member_vec), dtype=bool)
    # *** CRITICAL *** ADV20 row = trailing 20 sessions ending at the decision close.
    adv_vec = feature_obj.adv20_dollar()[decision_pos_int]
    finite_vec = np.isfinite(adv_vec) & (adv_vec > 0)
    if liquidity_str == "ABS5M":
        return finite_vec & (np.nan_to_num(adv_vec) >= 5_000_000.0)
    member_adv_vec = adv_vec[(member_vec == 1) & finite_vec]
    if len(member_adv_vec) == 0:
        return np.zeros(len(member_vec), dtype=bool)
    cut_float = float(np.quantile(member_adv_vec, {"REL25": 0.25, "REL50": 0.50}[liquidity_str]))
    return finite_vec & (np.nan_to_num(adv_vec) >= cut_float)


def build_target_list(feature_obj: FeatureBook, cell: Cell, schedule_cache_dict: dict | None = None) -> list[dict]:
    """Target weights per traded decision: [{decision_pos, execution_pos, symbol_idx_vec, weight_vec}]."""
    u = feature_obj.u
    schedule_key = ("schedule", cell.offset_int)
    if schedule_cache_dict is not None and schedule_key in schedule_cache_dict:
        schedule_dict = schedule_cache_dict[schedule_key]
    else:
        schedule_dict = build_schedule(u, cell.offset_int)
        if schedule_cache_dict is not None:
            schedule_cache_dict[schedule_key] = schedule_dict
    score_arr = score_table(feature_obj, schedule_dict, cell)
    trend_arr = feature_obj.trend_pass(cell.stock_filter_int)
    regime_vec = feature_obj.regime_pass(cell.regime_str)
    member_arr = u["member_arr"]
    sigma_arr = feature_obj.sigma63() if cell.weight_str == "IV" else None
    traded_row_vec = np.flatnonzero(schedule_dict["trade_vec"])
    decision_pos_vec = schedule_dict["decision_pos_vec"][traded_row_vec]
    scale_vec = feature_obj.vxn_scale_vec(decision_pos_vec, cell.vxn_target_float, cell.vxn_floor_float)

    target_list: list[dict] = []
    previous_selected_vec = np.array([], dtype=np.int64)
    for row_int, decision_pos_int, scale_float in zip(traded_row_vec, decision_pos_vec, scale_vec):
        # *** CRITICAL *** everything below is read at the decision close (row decision_pos); fills are next open.
        score_vec = score_arr[row_int]
        eligible_vec = (member_arr[decision_pos_int] == 1) & trend_arr[decision_pos_int] & np.isfinite(score_vec)
        eligible_vec &= liquidity_pass_vec(feature_obj, decision_pos_int, member_arr[decision_pos_int], cell.liquidity_str)
        if not regime_vec[decision_pos_int]:
            eligible_vec[:] = False
        eligible_idx_vec = np.flatnonzero(eligible_vec)
        # rank: score descending, then symbol ascending (repo mergesort order)
        order_vec = np.lexsort((feature_obj.symbol_rank_vec[eligible_idx_vec], -score_vec[eligible_idx_vec]))
        ranked_idx_vec = eligible_idx_vec[order_vec]
        if cell.buffer_int > 0 and len(previous_selected_vec) > 0 and len(ranked_idx_vec) > 0:
            rank_pos_dict = {int(symbol_idx): pos_int for pos_int, symbol_idx in enumerate(ranked_idx_vec)}
            keep_list = [
                int(symbol_idx)
                for symbol_idx in previous_selected_vec
                if rank_pos_dict.get(int(symbol_idx), 10**9) < cell.n_int + cell.buffer_int
            ]
            keep_set = set(keep_list)
            fill_list = [int(s) for s in ranked_idx_vec if int(s) not in keep_set][: max(cell.n_int - len(keep_list), 0)]
            selected_vec = np.array(keep_list + fill_list, dtype=np.int64)
        else:
            selected_vec = ranked_idx_vec[: cell.n_int].astype(np.int64)

        count_int = len(selected_vec)
        if count_int == 0:
            weight_vec = np.array([], dtype=np.float64)
        elif cell.weight_str == "EW":
            weight_vec = np.full(count_int, float(scale_float) / cell.n_int)
        else:
            sigma_vec = sigma_arr[decision_pos_int, selected_vec].astype(np.float64)
            finite_vec = np.isfinite(sigma_vec) & (sigma_vec > 0)
            if not finite_vec.any():
                sigma_vec = np.ones(count_int)
            else:
                sigma_vec = np.where(finite_vec, sigma_vec, np.median(sigma_vec[finite_vec]))
            inverse_vec = 1.0 / sigma_vec
            # budget = what EW would invest (count/N slots of the scaled exposure), split by inverse volatility
            weight_vec = float(scale_float) * (count_int / cell.n_int) * inverse_vec / inverse_vec.sum()
        target_list.append(
            {
                "decision_pos": int(decision_pos_int),
                "execution_pos": int(decision_pos_int) + 1,
                "symbol_idx_vec": selected_vec,
                "weight_vec": weight_vec,
                "exposure_float": float(scale_float),
            }
        )
        previous_selected_vec = selected_vec
    return target_list


# ======================================================================================================================
# 5. engine replica
# ======================================================================================================================
def simulate(
    universe_dict: dict,
    target_list: list[dict],
    slippage_float: float = ENGINE_SLIPPAGE_FLOAT,
    commission_fixed_bool: bool = False,
    split_factor_arr: np.ndarray | None = None,
    adv_arr: np.ndarray | None = None,
) -> dict:
    """Replica of alpha/engine/strategy.py for this strategy family.

    Per session t (trading calendar from 2000-01-03):
        1. dividend cash: cash += 0.75 * sum_i shares_i * Dividend_i(t-1)          (entitlement t-1, before open t)
        2. positions without an open or close today are sold at the last close <= t-1 (commission, no slippage)
        3. on an execution day (t = decision + 1):
               target_i = int(V(t-1) * w_i / Close_i(t-1)); names outside the target set are sold in full;
               delta_i = target_i - shares_i, filled at Open_i(t) * (1 + sign(delta) * slippage)
               commission = max($1, $0.005 * |delta|)  (engine share counts; or real share counts if commission_fixed)
        4. total(t) = cash + sum_i shares_i * Close_i(t)
    """
    date_index = universe_dict["date_index"]
    open_arr = universe_dict["open_arr"]
    close_arr = universe_dict["close_arr"]
    dividend_arr = universe_dict["dividend_arr"]
    symbol_count_int = close_arr.shape[1]
    start_pos_int = int(date_index.searchsorted(TRADING_START_TS, side="left"))
    date_count_int = len(date_index)

    target_by_execution_dict = {target_dict["execution_pos"]: target_dict for target_dict in target_list}
    shares_vec = np.zeros(symbol_count_int)
    cash_float = CAPITAL_BASE_FLOAT
    total_vec = np.full(date_count_int, np.nan)
    traded_notional_vec = np.zeros(date_count_int)
    commission_vec = np.zeros(date_count_int)
    order_frac_list: list[tuple[int, float, float]] = []  # (execution_pos, |notional| / V, ADV20 dollar)
    previous_total_float = CAPITAL_BASE_FLOAT
    missing_dividend_count_int = 0

    def commission_float(delta_float: float, pos_int: int, symbol_idx: int) -> float:
        share_float = abs(delta_float)
        if commission_fixed_bool:
            factor_float = split_factor_arr[pos_int, symbol_idx]
            if np.isfinite(factor_float) and factor_float > 0:
                share_float = share_float / factor_float
        return max(COMMISSION_MINIMUM_FLOAT, COMMISSION_PER_SHARE_FLOAT * share_float)

    for pos_int in range(start_pos_int, date_count_int):
        held_idx_vec = np.flatnonzero(shares_vec != 0.0)
        if pos_int > start_pos_int and len(held_idx_vec) > 0:
            # *** CRITICAL *** Norgate stamps Dividend on entitlement session t-1; credited before the open of t.
            dividend_vec = dividend_arr[pos_int - 1, held_idx_vec].astype(np.float64)
            missing_dividend_count_int += int(np.isnan(dividend_vec).sum())
            gross_vec = shares_vec[held_idx_vec] * np.nan_to_num(dividend_vec)
            cash_float += float(np.sum(np.where(gross_vec > 0, gross_vec * DIVIDEND_NET_RATE_FLOAT, gross_vec)))

        # missing-price liquidation at the last available close no later than t-1
        liquidated_set: set[int] = set()
        if len(held_idx_vec) > 0:
            missing_vec = ~np.isfinite(open_arr[pos_int, held_idx_vec]) | ~np.isfinite(close_arr[pos_int, held_idx_vec])
            for symbol_idx in held_idx_vec[missing_vec]:
                history_vec = close_arr[:pos_int, symbol_idx]
                finite_pos_vec = np.flatnonzero(np.isfinite(history_vec))
                price_float = float(history_vec[finite_pos_vec[-1]])
                delta_float = -shares_vec[symbol_idx]
                fee_float = commission_float(delta_float, pos_int, symbol_idx)
                cash_float -= delta_float * price_float + fee_float
                traded_notional_vec[pos_int] += abs(delta_float * price_float)
                commission_vec[pos_int] += fee_float
                shares_vec[symbol_idx] = 0.0
                liquidated_set.add(int(symbol_idx))  # engine clears this asset's pending orders

        target_dict = target_by_execution_dict.get(pos_int)
        if target_dict is not None:
            decision_pos_int = target_dict["decision_pos"]
            if decision_pos_int != pos_int - 1:
                raise RuntimeError("decision must be the previous session")
            budget_float = previous_total_float  # engine: previous_total_value (before today's dividend credit)
            target_share_dict: dict[int, float] = {}
            for symbol_idx, weight_float in zip(target_dict["symbol_idx_vec"], target_dict["weight_vec"]):
                sizing_price_float = float(close_arr[decision_pos_int, symbol_idx])
                if not np.isfinite(sizing_price_float) or sizing_price_float <= 0:
                    raise RuntimeError("invalid decision close for a selected name")
                # *** CRITICAL *** share count fixed from the decision close, not the execution open.
                target_share_int = int(budget_float * float(weight_float) / sizing_price_float)
                if target_share_int > 0:
                    target_share_dict[int(symbol_idx)] = float(target_share_int)
            order_dict: dict[int, float] = {}
            for symbol_idx in np.flatnonzero(shares_vec > 0):
                if int(symbol_idx) not in target_share_dict:
                    order_dict[int(symbol_idx)] = -shares_vec[symbol_idx]
            for symbol_idx, target_share_float in target_share_dict.items():
                if shares_vec[symbol_idx] != target_share_float:
                    order_dict[symbol_idx] = target_share_float - shares_vec[symbol_idx]
            for symbol_idx, delta_float in order_dict.items():
                if symbol_idx in liquidated_set:
                    continue
                open_float = float(open_arr[pos_int, symbol_idx])
                if not np.isfinite(open_float) or delta_float == 0.0:
                    continue  # engine cancels orders without a tradable open, and zero-share fills
                price_float = open_float * (1.0 + np.sign(delta_float) * slippage_float)
                fee_float = commission_float(delta_float, pos_int, symbol_idx)
                cash_float -= delta_float * price_float + fee_float
                traded_notional_vec[pos_int] += abs(delta_float * price_float)
                commission_vec[pos_int] += fee_float
                shares_vec[symbol_idx] += delta_float
                if adv_arr is not None:
                    order_frac_list.append(
                        (pos_int, abs(delta_float * price_float) / budget_float, float(adv_arr[decision_pos_int, symbol_idx]))
                    )

        held_idx_vec = np.flatnonzero(shares_vec != 0.0)
        portfolio_value_float = float(np.dot(shares_vec[held_idx_vec], close_arr[pos_int, held_idx_vec].astype(np.float64)))
        total_float = cash_float + portfolio_value_float
        total_vec[pos_int] = total_float
        previous_total_float = total_float

    total_ser = pd.Series(total_vec[start_pos_int:], index=date_index[start_pos_int:], name="total_value")
    return {
        "total_ser": total_ser,
        "return_ser": total_ser.pct_change().iloc[1:],
        "traded_notional_ser": pd.Series(traded_notional_vec[start_pos_int:], index=total_ser.index),
        "commission_ser": pd.Series(commission_vec[start_pos_int:], index=total_ser.index),
        "order_frac_arr": np.array(order_frac_list, dtype=np.float64).reshape(-1, 3),
        "missing_dividend_count_int": missing_dividend_count_int,
    }


# ======================================================================================================================
# 6. grids (PREREG section 4)
# ======================================================================================================================
STAGE1_NUMERATOR_TUPLE = ("ROC3", "ROC6", "B3612", "ROC9", "B612", "ROC12", "ROC12-1")
STAGE1_DENOMINATOR_TUPLE = ("none", "NATR63", "NATR20")
STAGE2_N_TUPLE = (5, 8, 10, 15, 20, 30)
STAGE2_WEIGHT_TUPLE = ("EW", "IV")
STAGE3_FILTER_TUPLE = (0, 50, 100, 200)
STAGE3_REGIME_TUPLE = ("none", "SPY", "QQQ")
STAGE4_BUFFER_TUPLE = (0, 2, 5, 10)
STAGE4_OFFSET_TUPLE = tuple(range(-10, 11))
STAGE5_TARGET_TUPLE = (18.0, 20.0, 22.0, 24.0, 26.0)
STAGE5_FLOOR_TUPLE = (0.0, 0.125, 0.25, 0.375, 0.5)


def stage_grid_dict() -> dict[str, list[tuple[tuple, Cell]]]:
    """Each stage: list of ((row_label, col_label), cell). Rows/cols are the heatmap axes."""
    a = ANCHOR_CELL
    grid_dict: dict[str, list[tuple[tuple, Cell]]] = {
        "S1_score": [
            ((numerator_str, denominator_str), dataclasses.replace(a, numerator_str=numerator_str, denominator_str=denominator_str))
            for numerator_str in STAGE1_NUMERATOR_TUPLE
            for denominator_str in STAGE1_DENOMINATOR_TUPLE
        ],
        "S2_n_weight": [
            ((weight_str, n_int), dataclasses.replace(a, n_int=n_int, weight_str=weight_str))
            for weight_str in STAGE2_WEIGHT_TUPLE
            for n_int in STAGE2_N_TUPLE
        ],
        "S3_filters": [
            ((regime_str, filter_int), dataclasses.replace(a, stock_filter_int=filter_int, regime_str=regime_str))
            for regime_str in STAGE3_REGIME_TUPLE
            for filter_int in STAGE3_FILTER_TUPLE
        ],
        "S4_buffer_offset": [
            ((buffer_int, offset_int), dataclasses.replace(a, buffer_int=buffer_int, offset_int=offset_int))
            for buffer_int in STAGE4_BUFFER_TUPLE
            for offset_int in STAGE4_OFFSET_TUPLE
        ],
        "S5_vxn": [
            ((target_float, floor_float), dataclasses.replace(a, vxn_target_float=target_float, vxn_floor_float=floor_float))
            for target_float in STAGE5_TARGET_TUPLE
            for floor_float in STAGE5_FLOOR_TUPLE
        ],
        "S5_reference": [(("off", "off"), dataclasses.replace(a, vxn_target_float=None))],
    }
    return grid_dict


def reference_cell_list() -> list[Cell]:
    """Incumbent L (all offsets, a diagnostic of its timing luck) and the biased B (contrast only)."""
    return [dataclasses.replace(L_CELL, offset_int=offset_int) for offset_int in STAGE4_OFFSET_TUPLE] + [B_CELL]


def all_cell_list() -> list[Cell]:
    seen_dict: dict[str, Cell] = {}
    for cell_list in stage_grid_dict().values():
        for _, cell in cell_list:
            seen_dict.setdefault(cell.key_str, cell)
    for cell in reference_cell_list():
        seen_dict.setdefault(cell.key_str, cell)
    return list(seen_dict.values())
