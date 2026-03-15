"""
Benchmark reference vs fast indicator implementations on synthetic data.

Usage
-----

    uv run python scripts/benchmark_fast_indicators.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from alpha.engine.dv2_indicator_fast import dv2_indicator_fast
from alpha.engine.indicators import dv2_indicator, qp_indicator
from alpha.engine.qp_indicator_fast import qp_indicator_fast


def build_price_panel(
    symbol_count_int: int = 100,
    bar_count_int: int = 1500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng_obj = np.random.default_rng(7)
    date_index = pd.bdate_range("2018-01-01", periods=bar_count_int)
    symbol_list = [f"S{idx_int:03d}" for idx_int in range(symbol_count_int)]

    close_return_arr = rng_obj.normal(0.0, 0.01, size=(bar_count_int, symbol_count_int))
    close_price_arr = 100.0 * np.exp(np.cumsum(close_return_arr, axis=0))
    high_price_arr = close_price_arr * (1.0 + np.abs(rng_obj.normal(0.0, 0.003, size=close_price_arr.shape)))
    low_price_arr = close_price_arr * (1.0 - np.abs(rng_obj.normal(0.0, 0.003, size=close_price_arr.shape)))

    close_price_df = pd.DataFrame(close_price_arr, index=date_index, columns=symbol_list)
    high_price_df = pd.DataFrame(high_price_arr, index=date_index, columns=symbol_list)
    low_price_df = pd.DataFrame(low_price_arr, index=date_index, columns=symbol_list)
    return close_price_df, high_price_df, low_price_df


def benchmark_dv2(
    close_price_df: pd.DataFrame,
    high_price_df: pd.DataFrame,
    low_price_df: pd.DataFrame,
) -> tuple[float, float]:
    symbol_list = close_price_df.columns.tolist()

    reference_start_float = perf_counter()
    for symbol_str in symbol_list:
        dv2_indicator(
            close_price_df[symbol_str],
            high_price_df[symbol_str],
            low_price_df[symbol_str],
            length=126,
        )
    reference_elapsed_float = perf_counter() - reference_start_float

    fast_start_float = perf_counter()
    for symbol_str in symbol_list:
        dv2_indicator_fast(
            close_price_df[symbol_str],
            high_price_df[symbol_str],
            low_price_df[symbol_str],
            length_int=126,
        )
    fast_elapsed_float = perf_counter() - fast_start_float
    return reference_elapsed_float, fast_elapsed_float


def benchmark_qpi(close_price_df: pd.DataFrame) -> tuple[float, float]:
    symbol_list = close_price_df.columns.tolist()

    reference_start_float = perf_counter()
    for symbol_str in symbol_list:
        qp_indicator(
            close_price_df[symbol_str],
            window=3,
            lookback_years=1,
        )
    reference_elapsed_float = perf_counter() - reference_start_float

    fast_start_float = perf_counter()
    for symbol_str in symbol_list:
        qp_indicator_fast(
            close_price_df[symbol_str],
            window_int=3,
            lookback_years_int=1,
        )
    fast_elapsed_float = perf_counter() - fast_start_float
    return reference_elapsed_float, fast_elapsed_float


def main():
    close_price_df, high_price_df, low_price_df = build_price_panel()

    dv2_reference_seconds_float, dv2_fast_seconds_float = benchmark_dv2(
        close_price_df,
        high_price_df,
        low_price_df,
    )
    qpi_reference_seconds_float, qpi_fast_seconds_float = benchmark_qpi(close_price_df)

    print("DV2 benchmark")
    print(f"  reference_seconds = {dv2_reference_seconds_float:.3f}")
    print(f"  fast_seconds      = {dv2_fast_seconds_float:.3f}")
    print(f"  speedup_x         = {dv2_reference_seconds_float / dv2_fast_seconds_float:.2f}")
    print("QPI benchmark")
    print(f"  reference_seconds = {qpi_reference_seconds_float:.3f}")
    print(f"  fast_seconds      = {qpi_fast_seconds_float:.3f}")
    print(f"  speedup_x         = {qpi_reference_seconds_float / qpi_fast_seconds_float:.2f}")


if __name__ == "__main__":
    main()
