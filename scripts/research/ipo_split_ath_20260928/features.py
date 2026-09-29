"""Per-symbol causal features for the IPO / split all-time-high study (PREREG sections 3-4).

Every function here takes one symbol's own daily bars (CAPITALSPECIAL, padding NONE) mapped onto the market session
calendar and returns per-bar values known at that bar's close. Nothing here looks at another symbol.

    age_j            = cal_pos_j - cal_pos_first                          (listing session has age 0)
    k_j              = Unadjusted Close_j / Close_j                       (adjustment factor at bar j)
    split_ratio_j    = k_{j-1} / k_j                                      (4.0 on a 4-for-1 ex-date)
    ath_j            = Close_j >= max(Close_0 .. Close_{j-1}),  j >= 1
    adv_j            = median(Turnover_{j-19} .. Turnover_j), listing bar excluded, >= 1 value
    R_h(j)           = (Open_{j+1+h} + sum Div_{j+1..j+h}) / Open_{j+1} - 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IPO_WINDOW_MAX_AGE_INT = 89
SPLIT_WINDOW_MAX_SESSIONS_INT = 89
SPLIT_MIN_AGE_INT = 252
SPLIT_MIN_RATIO_FLOAT = 1.24
SPLIT_RATIO_TOLERANCE_FLOAT = 0.005
ADV_WINDOW_INT = 20
HORIZON_LIST = [1, 5, 10, 20]
NEW_LISTING_MIN_DATE_TS = pd.Timestamp("1991-01-01")
PRICE_FIELD_LIST = ["Open", "High", "Low", "Close"]


def is_forward_split_ratio_bool(ratio_float: float) -> bool:
    """True when ratio >= 1.24 and within 0.5% of a/b with 1 <= b <= 4 and b < a <= 50."""
    if not np.isfinite(ratio_float) or ratio_float < SPLIT_MIN_RATIO_FLOAT:
        return False
    for denominator_int in range(1, 5):
        numerator_int = int(round(ratio_float * denominator_int))
        if denominator_int < numerator_int <= 50:
            if abs(ratio_float - numerator_int / denominator_int) / ratio_float < SPLIT_RATIO_TOLERANCE_FLOAT:
                return True
    return False


def clean_bars_df(bars_df: pd.DataFrame, calendar_idx: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Keep bars on market sessions with finite, positive OHLC and a positive adjustment factor."""
    count_dict = {"raw": int(len(bars_df))}
    bars_df = bars_df[bars_df.index.isin(calendar_idx)]
    count_dict["off_calendar"] = count_dict["raw"] - int(len(bars_df))
    price_ok_arr = np.isfinite(bars_df[PRICE_FIELD_LIST].to_numpy(dtype=float)).all(axis=1)
    price_ok_arr &= (bars_df[PRICE_FIELD_LIST].to_numpy(dtype=float) > 0).all(axis=1)
    unadjusted_arr = bars_df["Unadjusted Close"].to_numpy(dtype=float)
    price_ok_arr &= np.isfinite(unadjusted_arr) & (unadjusted_arr > 0)
    count_dict["bad_price"] = int((~price_ok_arr).sum())
    return bars_df[price_ok_arr], count_dict


def symbol_feature_df(bars_df: pd.DataFrame, calendar_idx: pd.DatetimeIndex, first_quoted_ts: pd.Timestamp,
                      new_listing_bool: bool) -> pd.DataFrame:
    """Per-bar causal features of one symbol. bars_df must already be cleaned (clean_bars_df)."""
    close_arr = bars_df["Close"].to_numpy(dtype=float)
    unadjusted_arr = bars_df["Unadjusted Close"].to_numpy(dtype=float)
    turnover_arr = bars_df["Turnover"].to_numpy(dtype=float).copy()
    cal_pos_arr = calendar_idx.get_indexer(bars_df.index)
    n_int = len(close_arr)

    # Listing session = the first quoted session (normally the first bar). A data-start symbol keeps age from its
    # first bar too; its "listing" is only the data start and it is never a new listing.
    first_cal_pos_int = int(calendar_idx.searchsorted(first_quoted_ts)) if pd.notna(first_quoted_ts) else int(cal_pos_arr[0])
    first_cal_pos_int = min(first_cal_pos_int, int(cal_pos_arr[0]))
    age_arr = cal_pos_arr - first_cal_pos_int

    # *** CRITICAL*** k_j uses only bar j; the split ratio compares bar j with the previous bar of the same symbol,
    # both known at the close of j (the ex-date). A future split rescales Close and Unadjusted Close/Close together,
    # so ratios between consecutive k are unchanged by later adjustments.
    k_arr = unadjusted_arr / close_arr
    split_ratio_arr = np.full(n_int, np.nan)
    split_ratio_arr[1:] = k_arr[:-1] / k_arr[1:]
    is_split_arr = np.array([is_forward_split_ratio_bool(r) for r in split_ratio_arr], dtype=bool)

    # Sessions since the latest forward-split ex-date at or before j (market sessions).
    last_split_cal_pos_arr = np.where(is_split_arr, cal_pos_arr, -1)
    last_split_cal_pos_arr = np.maximum.accumulate(last_split_cal_pos_arr) if n_int else last_split_cal_pos_arr
    sessions_since_split_arr = np.where(last_split_cal_pos_arr >= 0, cal_pos_arr - last_split_cal_pos_arr, -1)
    in_split_window_arr = (sessions_since_split_arr >= 0) & (sessions_since_split_arr <= SPLIT_WINDOW_MAX_SESSIONS_INT)

    # *** CRITICAL*** all-time high compares today's close with the maximum of STRICTLY EARLIER closes (shifted
    # running max), so the decision at the close of j uses closes 0..j only. Bar 0 is never an ATH.
    prior_max_arr = np.full(n_int, np.nan)
    if n_int > 1:
        prior_max_arr[1:] = np.maximum.accumulate(close_arr)[:-1]
    ath_arr = np.zeros(n_int, dtype=bool)
    ath_arr[1:] = close_arr[1:] >= prior_max_arr[1:]

    # *** CRITICAL*** ADV is a trailing median of native Turnover ending at j (inclusive); the listing bar's
    # abnormal first-day turnover is excluded for every symbol.
    turnover_arr[0] = np.nan
    turnover_arr[~np.isfinite(turnover_arr) | (turnover_arr < 0)] = np.nan
    adv_arr = pd.Series(turnover_arr).rolling(ADV_WINDOW_INT, min_periods=1).median().to_numpy()

    in_ipo_window_arr = bool(new_listing_bool) & (age_arr >= 1) & (age_arr <= IPO_WINDOW_MAX_AGE_INT)

    return pd.DataFrame({
        "cal_pos": cal_pos_arr.astype(np.int32),
        "age": age_arr.astype(np.int32),
        "k": k_arr,
        "is_split": is_split_arr,
        "split_ratio": split_ratio_arr,
        "sessions_since_split": sessions_since_split_arr.astype(np.int32),
        "in_split_window": in_split_window_arr,
        "ath": ath_arr,
        "adv": adv_arr,
        "uclose": unadjusted_arr,
        "in_ipo_window": in_ipo_window_arr,
    }, index=bars_df.index)


def forward_return_df(bars_df: pd.DataFrame, row_pos_arr: np.ndarray, cal_pos_arr: np.ndarray) -> pd.DataFrame:
    """Forward returns from the next bar's open for the requested bar positions (PREREG section 4).

    *** CRITICAL*** these are OUTCOMES, never features: entry is Open_{j+1}, exit Open_{j+1+h}; dividends paid on
    bars j+1..j+h are added. When bar j+1+h does not exist (delisting or data end), the last available close is the
    exit and exit_is_close is set. Rows with no bar j+1 get NaN.
    """
    open_arr = bars_df["Open"].to_numpy(dtype=float)
    close_arr = bars_df["Close"].to_numpy(dtype=float)
    dividend_arr = np.nan_to_num(bars_df["Dividend"].to_numpy(dtype=float), nan=0.0)
    dividend_cum_arr = np.concatenate([[0.0], np.cumsum(dividend_arr)])
    n_int = len(open_arr)
    out_dict = {}
    entry_row_arr = row_pos_arr + 1
    has_entry_arr = entry_row_arr < n_int
    safe_entry_arr = np.minimum(entry_row_arr, n_int - 1)
    entry_open_arr = np.where(has_entry_arr, open_arr[safe_entry_arr], np.nan)
    out_dict["entry_cal_pos"] = np.where(has_entry_arr, cal_pos_arr[safe_entry_arr], -1).astype(np.int32)
    for horizon_int in HORIZON_LIST:
        exit_row_arr = entry_row_arr + horizon_int
        exit_is_close_arr = exit_row_arr >= n_int
        safe_exit_arr = np.minimum(exit_row_arr, n_int - 1)
        exit_price_arr = np.where(exit_is_close_arr, close_arr[n_int - 1], open_arr[safe_exit_arr])
        # dividends on bars j+1 .. min(j+h, last bar)
        last_div_row_arr = np.minimum(row_pos_arr + horizon_int, n_int - 1)
        dividend_sum_arr = dividend_cum_arr[last_div_row_arr + 1] - dividend_cum_arr[np.minimum(entry_row_arr, n_int)]
        out_dict[f"R{horizon_int}"] = np.where(has_entry_arr, (exit_price_arr + dividend_sum_arr) / entry_open_arr - 1.0, np.nan)
        out_dict[f"exit_cal_pos{horizon_int}"] = np.where(exit_is_close_arr, cal_pos_arr[n_int - 1], cal_pos_arr[safe_exit_arr]).astype(np.int32)
        out_dict[f"exit_is_close{horizon_int}"] = exit_is_close_arr
        # Carlos's close-to-close label (not tradable): Close_{j+h} / Close_j - 1
        cc_row_arr = row_pos_arr + horizon_int
        out_dict[f"CC{horizon_int}"] = np.where(cc_row_arr < n_int, close_arr[np.minimum(cc_row_arr, n_int - 1)] / close_arr[row_pos_arr] - 1.0, np.nan)
    return pd.DataFrame(out_dict)
