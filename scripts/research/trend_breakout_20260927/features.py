"""Daily feature panels for the trend / breakout study (research only).

Extends the 26 Sep FeatureBook (imported, not modified). Every feature below is either invariant to a constant
rescaling of one stock's price history or expressed in decision-date units (PREREG section 2):

    ATR20_t   = mean(TR over the 20 sessions ending t)                 (adjusted units, compared with adjusted prices)
    NATR20_t  = ATR20_t / Close_t
    ATR20$_t  = ATR20_t x U_t / Close_t                                 (L's decision-day dollar ATR)
    HH_N(t)   = max(Close_{t-N}, ..., Close_{t-1})                      (breakout reference, excludes t)
    ROC252_t  = Close_t / Close_{t-252} - 1
    ADV20_t   = median(Turnover over the 20 sessions ending t)          (native dollars, nominal)
    score_d   = (Close_t / Close_{ME12(t)} - 1) / ATR20$_t              (daily refill score of family A)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import ndx_param_robustness_core as core  # noqa: E402


class DailyFeatureBook(core.FeatureBook):
    """Lazily computed daily panels (T x S) for one universe, plus per-session helpers."""

    def __init__(self, universe_dict: dict):
        super().__init__(universe_dict)
        self.date_index = pd.DatetimeIndex(universe_dict["date_index"])
        self.symbol_list = list(universe_dict["symbol_list"])

    # ------------------------------------------------------------------ overrides
    def adv20_dollar(self) -> np.ndarray:
        """ADV20 from native Turnover (PREREG section 3); overrides the 26 Sep Close x Volume product."""

        def build():
            turnover_arr = self.u["turnover_arr"].astype(np.float64)
            with np.errstate(invalid="ignore"):
                turnover_arr = np.where(np.isfinite(turnover_arr) & (turnover_arr > 0), turnover_arr, np.nan)
            # *** CRITICAL *** trailing 20-session median ending at the decision close; 20 valid sessions required.
            return pd.DataFrame(turnover_arr).rolling(window=20, min_periods=20).median().to_numpy()

        return self._get(("adv20_turnover",), build)

    # ------------------------------------------------------------------ panels
    def unadjusted_close(self) -> np.ndarray:
        return self._get(("unadj64",), lambda: self.u["unadjusted_close_arr"].astype(np.float64))

    def open64(self) -> np.ndarray:
        return self._get(("open64",), lambda: self.u["open_arr"].astype(np.float64))

    def dividend64(self) -> np.ndarray:
        return self._get(("dividend64",), lambda: self.u["dividend_arr"].astype(np.float64))

    def atr_dollar(self) -> np.ndarray:
        # ATR20 in decision-day dollars = adjusted ATR x U_t / Close_t (stage-2 identity; invariant to rescaling).
        return self._get(("atr_dollar",), lambda: self.atr_cs(20) * self.split_factor())

    def hh_close(self, window_int: int) -> np.ndarray:
        def build():
            # *** CRITICAL *** highest close of the PREVIOUS window_int sessions (shift 1): the breakout day itself is
            # excluded, so Close_t > HH_N(t) is a genuine new high; all window_int closes must be valid.
            return pd.DataFrame(self.close_arr).rolling(window=window_int, min_periods=window_int).max().shift(1).to_numpy()

        return self._get(("hh_close", window_int), build)

    def breakout_pass(self, window_int: int) -> np.ndarray:
        def build():
            with np.errstate(invalid="ignore"):
                return self.close_arr > self.hh_close(window_int)  # NaN compares False -> no signal

        return self._get(("breakout", window_int), build)

    def roc_daily(self, window_int: int) -> np.ndarray:
        def build():
            out_arr = np.full_like(self.close_arr, np.nan)
            # *** CRITICAL *** Close_t / Close_{t-window} - 1: both at or before the decision close.
            out_arr[window_int:] = self.close_arr[window_int:] / self.close_arr[:-window_int] - 1.0
            return out_arr

        return self._get(("roc_daily", window_int), build)

    def rank_score(self, rank_str: str) -> np.ndarray:
        """Family B admission score: R1 = ROC252 / NATR20 (higher first), R2 = -NATR20 (lower volatility first)."""

        def build():
            with np.errstate(divide="ignore", invalid="ignore"):
                if rank_str == "R1":
                    score_arr = self.roc_daily(252) / self.natr(20)
                elif rank_str == "R2":
                    natr_arr = self.natr(20)
                    score_arr = np.where(natr_arr > 0, -natr_arr, np.nan)
                else:
                    raise ValueError(rank_str)
            return np.where(np.isfinite(score_arr), score_arr, np.nan)

        return self._get(("rank_score", rank_str), build)

    def rel25_pass(self) -> np.ndarray:
        """Bool panel: PIT member with ADV20 at or above the 25th percentile of that day's members' ADV20."""

        def build():
            adv_arr = self.adv20_dollar()
            member_arr = self.u["member_arr"]
            out_arr = np.zeros(self.close_arr.shape, dtype=bool)
            for pos_int in range(self.date_count_int):
                adv_vec = adv_arr[pos_int]
                finite_vec = np.isfinite(adv_vec) & (adv_vec > 0)
                pool_vec = adv_vec[(member_arr[pos_int] == 1) & finite_vec]
                if len(pool_vec) == 0:
                    continue
                cut_float = float(np.quantile(pool_vec, 0.25))
                out_arr[pos_int] = finite_vec & (np.nan_to_num(adv_vec) >= cut_float)
            return out_arr

        return self._get(("rel25",), build)

    def vxn_scale_all(self, target_float: float = 22.0, floor_float: float = 0.25) -> np.ndarray:
        """VXN exposure scale as-of every session (latest VXN close on or before the session)."""

        def build():
            vxn_close_ser = self.u["vxn_close_ser"].dropna().sort_index()
            # *** CRITICAL *** as-of lookup: never a later VXN close.
            row_vec = vxn_close_ser.index.searchsorted(self.date_index, side="right") - 1
            scale_vec = np.full(self.date_count_int, np.nan)
            ok_vec = row_vec >= 0
            scale_vec[ok_vec] = np.clip(float(target_float) / vxn_close_ser.to_numpy()[row_vec[ok_vec]], float(floor_float), 1.0)
            return scale_vec

        return self._get(("vxn_scale_all", target_float, floor_float), build)

    # ------------------------------------------------------------------ family A daily refill score
    def me12_anchor_pos(self, schedule_dict: dict) -> np.ndarray:
        """For each session t: the position of the month-end decision close 12 schedule rows before the coming
        decision (the anchor the coming month-end's ROC12 uses); -1 when unavailable."""
        key_tuple = ("me12_anchor", schedule_dict["offset_int"])

        def build():
            decision_pos_vec = schedule_dict["decision_pos_vec"]
            valid_vec = schedule_dict["valid_vec"]
            anchor_vec = np.full(self.date_count_int, -1, dtype=np.int64)
            for pos_int in range(self.date_count_int):
                # *** CRITICAL *** the coming decision row j has decision_pos >= t; its ROC12 anchor is row j - 12.
                row_int = int(np.searchsorted(decision_pos_vec, pos_int, side="left"))
                anchor_row_int = row_int - 12
                if 0 <= anchor_row_int < len(decision_pos_vec) and row_int < len(decision_pos_vec) and valid_vec[anchor_row_int]:
                    anchor_vec[pos_int] = int(decision_pos_vec[anchor_row_int])
            return anchor_vec

        return self._get(key_tuple, build)

    def daily_l_score(self, schedule_dict: dict) -> np.ndarray:
        key_tuple = ("daily_l_score", schedule_dict["offset_int"])

        def build():
            anchor_vec = self.me12_anchor_pos(schedule_dict)
            out_arr = np.full_like(self.close_arr, np.nan)
            atr_dollar_arr = self.atr_dollar()
            ok_vec = anchor_vec >= 0
            with np.errstate(divide="ignore", invalid="ignore"):
                out_arr[ok_vec] = (self.close_arr[ok_vec] / self.close_arr[anchor_vec[ok_vec]] - 1.0) / atr_dollar_arr[ok_vec]
            return np.where(np.isfinite(out_arr), out_arr, np.nan)

        return self._get(key_tuple, build)


def last_finite_close_before(close_arr: np.ndarray, pos_int: int, symbol_idx: int) -> tuple[int, float]:
    """Engine rule: the latest available close no later than the previous bar (positions < pos_int)."""
    history_vec = close_arr[:pos_int, symbol_idx]
    finite_pos_vec = np.flatnonzero(np.isfinite(history_vec))
    if len(finite_pos_vec) == 0:
        raise RuntimeError("no prior close available for a terminal liquidation")
    anchor_int = int(finite_pos_vec[-1])
    return anchor_int, float(history_vec[anchor_int])
