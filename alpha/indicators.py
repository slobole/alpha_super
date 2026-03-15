"""
Shared public indicator entry point for research and strategy code.

Core formulas
-------------

    ADV_t^{(L)} = (1 / L) * sum_{k=0}^{L-1} Turnover_{t-k}

    IBS_t = (C_t - L_t) / (H_t - L_t)

    HL2_t = (H_t + L_t) / 2

    DV1_t = (C_t / HL2_t) - 1

    DV_t = (DV1_t + DV1_{t-1}) / 2

    DV2_t = 100 * PctRank(DV_{t-L+1}, ..., DV_t)

    r_t^{(w)} = (C_t / C_{t-w}) - 1

    p_down,t = (1 / L) * sum_{k=t-L+1}^{t} 1[r_k <= 0]

    p_up,t = 1 - p_down,t
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.engine.dv2_indicator_fast import dv2_indicator_fast as _dv2_indicator_fast_backend
from alpha.engine.indicators import dv2_indicator as _dv2_indicator_reference_backend
from alpha.engine.indicators import qp_indicator as _qp_indicator_reference_backend
from alpha.engine.qp_indicator_fast import qp_indicator_fast as _qp_indicator_fast_backend

__all__ = [
    "dv2_indicator",
    "qp_indicator",
    "adv_dollar_indicator",
    "ibs_indicator",
    "dv2_indicator_fast",
    "qp_indicator_fast",
    "dv2_indicator_reference",
    "qp_indicator_reference",
]


PriceData_Type = pd.Series | pd.DataFrame


def _validate_positive_window_int(window_int: int, parameter_name_str: str) -> None:
    if window_int <= 0:
        raise ValueError(f"{parameter_name_str} must be positive.")


def dv2_indicator_reference(
    close_price_ser: pd.Series,
    high_price_ser: pd.Series,
    low_price_ser: pd.Series,
    length_int: int = 126,
) -> pd.Series:
    _validate_positive_window_int(length_int, "length_int")
    return _dv2_indicator_reference_backend(
        close_price_ser,
        high_price_ser,
        low_price_ser,
        length=length_int,
    )


def qp_indicator_reference(
    close_price_ser: pd.Series,
    window_int: int = 3,
    lookback_years_int: int = 5,
) -> pd.Series:
    _validate_positive_window_int(window_int, "window_int")
    _validate_positive_window_int(lookback_years_int, "lookback_years_int")
    return _qp_indicator_reference_backend(
        close_price_ser,
        window=window_int,
        lookback_years=lookback_years_int,
    )


def dv2_indicator_fast(
    close_price_ser: pd.Series,
    high_price_ser: pd.Series,
    low_price_ser: pd.Series,
    length_int: int = 126,
) -> pd.Series:
    return _dv2_indicator_fast_backend(
        close_price_ser,
        high_price_ser,
        low_price_ser,
        length_int=length_int,
    )


def qp_indicator_fast(
    close_price_ser: pd.Series,
    window_int: int = 3,
    lookback_years_int: int = 5,
) -> pd.Series:
    return _qp_indicator_fast_backend(
        close_price_ser,
        window_int=window_int,
        lookback_years_int=lookback_years_int,
    )


def dv2_indicator(
    close_price_ser: pd.Series,
    high_price_ser: pd.Series,
    low_price_ser: pd.Series,
    length_int: int = 126,
) -> pd.Series:
    return dv2_indicator_fast(
        close_price_ser,
        high_price_ser,
        low_price_ser,
        length_int=length_int,
    )


def qp_indicator(
    close_price_ser: pd.Series,
    window_int: int = 3,
    lookback_years_int: int = 5,
) -> pd.Series:
    return qp_indicator_fast(
        close_price_ser,
        window_int=window_int,
        lookback_years_int=lookback_years_int,
    )


def adv_dollar_indicator(
    turnover_dollar_vec: PriceData_Type,
    window_int: int = 20,
) -> PriceData_Type:
    _validate_positive_window_int(window_int, "window_int")
    # *** CRITICAL*** rolling(window_int) must remain a trailing turnover average.
    adv_dollar_vec = turnover_dollar_vec.rolling(
        window=window_int,
        min_periods=window_int,
    ).mean()
    return adv_dollar_vec


def ibs_indicator(
    close_price_vec: PriceData_Type,
    high_price_vec: PriceData_Type,
    low_price_vec: PriceData_Type,
) -> PriceData_Type:
    # *** CRITICAL*** zero high-low ranges must become NaN rather than leaking infinities.
    range_price_vec = (high_price_vec - low_price_vec).replace(0.0, np.nan)
    ibs_value_vec = (close_price_vec - low_price_vec) / range_price_vec
    return ibs_value_vec
