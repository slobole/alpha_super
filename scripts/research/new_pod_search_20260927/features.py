"""Features for the new-pod search (research only). Every feature is scale-free or uses native Turnover.

Family M (event at session d, all data through d; decision at the close of d + W):
    jump_d          = Close_d / Close_{d-1} - 1 >= J
    volume shock    = Turnover_d >= 5 x median(Turnover_{d-60..d-1})            (60 valid, positive)
    liquidity       = ADV20_{d-1} >= 25th percentile of ADV20_{d-1} among the PIT members at d
                      ADV20_{d-1} = median(Turnover_{d-20..d-1})
    pin window s = d+1..d+W: every Turnover_s > 0 (no halted / padded bar)
    pin_vol_d       = mean_s(TR_s / Close_s) <= theta,  TR_s = max(H_s - L_s, |H_s - C_{s-1}|, |L_s - C_{s-1}|)
    hold_d          = min_s Close_s >= 0.97 x Close_d
    pin_ref_d       = median_s Close_s                                             (fixed for the whole position)
    confirmation at t = d + W: event_d and pin_ok_d (the decision row carries pin_vol_d and pin_ref_d)

Family S (decision at the repo month-end of month m, year Y; target month m+1):
    r_{i}(m+1, y)   = Close_ME(m+1, y) / Close_ME(m, y) - 1      (both month-end closes must exist)
    score_i(t)      = mean over the horizon's past years y of r_i(m+1, y), if at least the minimum count is valid
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import ndx_param_robustness_core as core  # noqa: E402

from new_pod_search_20260927.cells import HORIZON_DICT
from trend_breakout_20260927.features import DailyFeatureBook

VOLUME_SHOCK_MULT_FLOAT = 5.0
VOLUME_MEDIAN_WINDOW_INT = 60
HOLD_FLOOR_FLOAT = 0.97


class PodFeatureBook(DailyFeatureBook):
    # ------------------------------------------------------------------ family M
    def turnover64(self) -> np.ndarray:
        def build():
            turnover_arr = self.u["turnover_arr"].astype(np.float64)
            with np.errstate(invalid="ignore"):
                return np.where(np.isfinite(turnover_arr) & (turnover_arr > 0), turnover_arr, np.nan)

        return self._get(("turnover64",), build)

    def turnover_median60_prev(self) -> np.ndarray:
        def build():
            # *** CRITICAL *** median of the 60 sessions BEFORE d (shift 1); all 60 must be valid.
            return pd.DataFrame(self.turnover64()).rolling(window=VOLUME_MEDIAN_WINDOW_INT, min_periods=VOLUME_MEDIAN_WINDOW_INT).median().shift(1).to_numpy()

        return self._get(("turnover_median60_prev",), build)

    def adv20_prev(self) -> np.ndarray:
        def build():
            out_arr = np.full_like(self.close_arr, np.nan)
            out_arr[1:] = self.adv20_dollar()[:-1]  # median of Turnover over d-20..d-1
            return out_arr

        return self._get(("adv20_prev",), build)

    def rel25_prev_pass(self) -> np.ndarray:
        """ADV20 before the event at or above the 25th percentile among that day's PIT members."""

        def build():
            adv_arr = self.adv20_prev()
            member_arr = self.u["member_arr"]
            out_arr = np.zeros(self.close_arr.shape, dtype=bool)
            for pos_int in range(self.date_count_int):
                adv_vec = adv_arr[pos_int]
                finite_vec = np.isfinite(adv_vec) & (adv_vec > 0)
                pool_vec = adv_vec[(member_arr[pos_int] == 1) & finite_vec]
                if len(pool_vec) == 0:
                    continue
                out_arr[pos_int] = finite_vec & (np.nan_to_num(adv_vec) >= float(np.quantile(pool_vec, 0.25)))
            return out_arr

        return self._get(("rel25_prev",), build)

    def jump_ret(self) -> np.ndarray:
        def build():
            out_arr = np.full_like(self.close_arr, np.nan)
            out_arr[1:] = self.close_arr[1:] / self.close_arr[:-1] - 1.0
            return out_arr

        return self._get(("jump_ret",), build)

    def volume_shock(self) -> np.ndarray:
        def build():
            with np.errstate(invalid="ignore"):
                return self.turnover64() >= VOLUME_SHOCK_MULT_FLOAT * self.turnover_median60_prev()

        return self._get(("volume_shock",), build)

    def event(self, jump_float: float) -> np.ndarray:
        def build():
            with np.errstate(invalid="ignore"):
                return (self.u["member_arr"] == 1) & (self.jump_ret() >= jump_float) & self.volume_shock() & self.rel25_prev_pass()

        return self._get(("event", jump_float), build)

    def tr_ratio(self) -> np.ndarray:
        def build():
            tr_arr = core.true_range_arr(self.u["high_arr"].astype(np.float64), self.u["low_arr"].astype(np.float64), self.close_arr)
            with np.errstate(invalid="ignore", divide="ignore"):
                return tr_arr / self.close_arr

        return self._get(("tr_ratio",), build)

    def pin_features(self, window_int: int) -> dict[str, np.ndarray]:
        """At the event row d: pin_vol, pin_ref, min-close ratio and the all-bars-traded flag over d+1..d+W."""
        key_tuple = ("pin", window_int)

        def build():
            date_count_int, symbol_count_int = self.close_arr.shape
            ratio_arr = self.tr_ratio()
            turnover_arr = self.turnover64()
            pin_sum_arr = np.zeros((date_count_int, symbol_count_int))
            min_close_arr = np.full((date_count_int, symbol_count_int), np.inf)
            traded_arr = np.ones((date_count_int, symbol_count_int), dtype=bool)
            close_stack_list = []
            for k_int in range(1, window_int + 1):
                shifted_ratio_arr = np.full((date_count_int, symbol_count_int), np.nan)
                shifted_ratio_arr[:-k_int] = ratio_arr[k_int:]  # value at d+k placed on row d
                shifted_close_arr = np.full((date_count_int, symbol_count_int), np.nan)
                shifted_close_arr[:-k_int] = self.close_arr[k_int:]
                shifted_turnover_arr = np.full((date_count_int, symbol_count_int), np.nan)
                shifted_turnover_arr[:-k_int] = turnover_arr[k_int:]
                pin_sum_arr = pin_sum_arr + shifted_ratio_arr
                with np.errstate(invalid="ignore"):
                    min_close_arr = np.fmin(min_close_arr, np.where(np.isfinite(shifted_close_arr), shifted_close_arr, np.inf))
                    traded_arr &= np.isfinite(shifted_turnover_arr) & (shifted_turnover_arr > 0)
                close_stack_list.append(shifted_close_arr)
            pin_vol_arr = pin_sum_arr / window_int  # NaN if any TR ratio in the window is NaN
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices on the last rows
                pin_ref_arr = np.nanmedian(np.stack(close_stack_list, axis=0), axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                hold_arr = min_close_arr / self.close_arr >= HOLD_FLOOR_FLOAT
            pin_ok_arr = traded_arr & np.isfinite(pin_vol_arr) & hold_arr & np.isfinite(pin_ref_arr)
            return {"pin_vol": pin_vol_arr, "pin_ref": pin_ref_arr, "traded": traded_arr, "hold": hold_arr, "pin_ok": pin_ok_arr}

        return self._get(key_tuple, build)

    def confirmation(self, jump_float: float, theta_float: float, window_int: int) -> dict[str, np.ndarray]:
        """Confirmation flags at the decision row t = d + W, with pin_vol and pin_ref of the event carried to row t."""
        key_tuple = ("confirmation", jump_float, theta_float, window_int)

        def build():
            pin_dict = self.pin_features(window_int)
            with np.errstate(invalid="ignore"):
                confirmed_at_event_arr = self.event(jump_float) & pin_dict["pin_ok"] & (pin_dict["pin_vol"] <= theta_float)
            date_count_int, symbol_count_int = self.close_arr.shape
            conf_arr = np.zeros((date_count_int, symbol_count_int), dtype=bool)
            pin_vol_arr = np.full((date_count_int, symbol_count_int), np.nan)
            pin_ref_arr = np.full((date_count_int, symbol_count_int), np.nan)
            event_pos_arr = np.full((date_count_int, symbol_count_int), -1, dtype=np.int64)
            # *** CRITICAL *** the event at d is confirmed at the close of d + W and traded at the open of d + W + 1.
            conf_arr[window_int:] = confirmed_at_event_arr[:-window_int]
            pin_vol_arr[window_int:] = np.where(confirmed_at_event_arr[:-window_int], pin_dict["pin_vol"][:-window_int], np.nan)
            pin_ref_arr[window_int:] = np.where(confirmed_at_event_arr[:-window_int], pin_dict["pin_ref"][:-window_int], np.nan)
            event_pos_arr[window_int:] = np.where(confirmed_at_event_arr[:-window_int], np.arange(date_count_int - window_int)[:, None], -1)
            return {"conf": conf_arr, "pin_vol": pin_vol_arr, "pin_ref": pin_ref_arr, "event_pos": event_pos_arr, "events_at_event_row": self.event(jump_float)}

        return self._get(key_tuple, build)


# ----------------------------------------------------------------------------------------------------------------------
# family S scores
# ----------------------------------------------------------------------------------------------------------------------
def monthly_return_table(monthly_close_df: pd.DataFrame, symbol_list: list[str]) -> tuple[pd.PeriodIndex, np.ndarray]:
    """Calendar-month returns (rows = contiguous monthly periods, cols = symbols); NaN unless both month-ends exist."""
    period_index = pd.PeriodIndex(pd.DatetimeIndex(monthly_close_df.index).to_period("M"))
    if not (np.diff(period_index.asi8) == 1).all():
        raise RuntimeError("month-end close panel is not contiguous in calendar months")
    close_arr = monthly_close_df.reindex(columns=symbol_list).to_numpy(dtype=np.float64)
    return_arr = np.full_like(close_arr, np.nan)
    # *** CRITICAL *** return of calendar month p = close at the end of p over close at the end of p - 1.
    with np.errstate(invalid="ignore", divide="ignore"):
        return_arr[1:] = close_arr[1:] / close_arr[:-1] - 1.0
    return period_index, return_arr


def seasonality_score_tables(universe_dict: dict, monthly_close_df: pd.DataFrame, horizon_list: list[str]) -> dict[str, np.ndarray]:
    """Score tables (schedule rows x symbols) keyed by horizon, aligned to the cache's month-end index.

    For the schedule row of month m (period P): the target month is P + 1; past-year returns are the calendar-month
    returns of periods P + 1 - 12 k, k in the horizon's year offsets. Nothing at or after P + 1 enters a score.
    """
    symbol_list = list(universe_dict["symbol_list"])
    period_index, return_arr = monthly_return_table(monthly_close_df, symbol_list)
    period_pos_dict = {int(p.ordinal): i for i, p in enumerate(period_index)}
    month_end_index = pd.DatetimeIndex(universe_dict["month_end_index"])
    table_dict: dict[str, np.ndarray] = {}
    for horizon_str in horizon_list:
        offset_tuple, min_count_int = HORIZON_DICT[horizon_str]
        table_arr = np.full((len(month_end_index), len(symbol_list)), np.nan)
        for row_int, month_end_ts in enumerate(month_end_index):
            target_period = pd.Timestamp(month_end_ts).to_period("M") + 1
            row_list = []
            for k_int in offset_tuple:
                past_period = target_period - 12 * k_int
                pos_int = period_pos_dict.get(int(past_period.ordinal))
                if pos_int is not None and pos_int >= 1:
                    row_list.append(return_arr[pos_int])
            if not row_list:
                continue
            stack_arr = np.stack(row_list, axis=0)
            count_vec = np.isfinite(stack_arr).sum(axis=0)
            with np.errstate(invalid="ignore"):
                mean_vec = np.nanmean(np.where(np.isfinite(stack_arr), stack_arr, np.nan), axis=0)
            table_arr[row_int] = np.where(count_vec >= min_count_int, mean_vec, np.nan)
        table_dict[horizon_str] = table_arr
    return table_dict
