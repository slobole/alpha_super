from __future__ import annotations

import os

import pandas as pd
import pytest


RUN_PARITY_BOOL = os.getenv("ALPHA_RUN_NORGATE_SNAPSHOT_PARITY_BOOL", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

pytestmark = pytest.mark.skipif(
    not RUN_PARITY_BOOL,
    reason=(
        "Direct-vs-snapshot parity requires a Windows Norgate install and matching "
        "NORGATE_SNAPSHOT_ROOT. Set ALPHA_RUN_NORGATE_SNAPSHOT_PARITY_BOOL=true to run."
    ),
)


def _set_snapshot_mode_bool(monkeypatch, enabled_bool: bool) -> None:
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true" if enabled_bool else "false")


def _assert_frame_equal(left_df: pd.DataFrame, right_df: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        left_df.sort_index().sort_index(axis=1),
        right_df.sort_index().sort_index(axis=1),
        check_dtype=False,
        check_freq=False,
    )


def test_dv2_qpi_sp500_boundary_parity(monkeypatch):
    from data.norgate_loader import build_index_constituent_matrix, load_raw_prices

    start_date_str = os.getenv("NORGATE_PARITY_START_DATE_STR", "2024-01-01")
    end_date_str = os.getenv("NORGATE_PARITY_END_DATE_STR", "2024-01-31")

    _set_snapshot_mode_bool(monkeypatch, False)
    _, direct_universe_df = build_index_constituent_matrix(indexname="S&P 500")
    direct_price_df = load_raw_prices(
        symbols=["SPY"],
        benchmarks=["$SPX"],
        start_date=start_date_str,
        end_date=end_date_str,
    )

    _set_snapshot_mode_bool(monkeypatch, True)
    _, snapshot_universe_df = build_index_constituent_matrix(indexname="S&P 500")
    snapshot_price_df = load_raw_prices(
        symbols=["SPY"],
        benchmarks=["$SPX"],
        start_date=start_date_str,
        end_date=end_date_str,
    )

    _assert_frame_equal(direct_universe_df, snapshot_universe_df)
    _assert_frame_equal(direct_price_df, snapshot_price_df)


def test_taa_helper_boundary_parity(monkeypatch):
    from strategies.taa_df.strategy_taa_df import load_execution_price_df, load_signal_close_df
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import load_helper_close_ser

    start_date_str = os.getenv("NORGATE_PARITY_START_DATE_STR", "2024-01-01")
    end_date_str = os.getenv("NORGATE_PARITY_END_DATE_STR", "2024-01-31")

    _set_snapshot_mode_bool(monkeypatch, False)
    direct_signal_close_df = load_signal_close_df(["GLD", "UUP", "TLT", "DBC"], start_date_str, end_date_str)
    direct_execution_price_df = load_execution_price_df(["GLD", "UUP", "TLT", "DBC", "SPY"], ["$SPX"], start_date_str, end_date_str)
    direct_vix_close_ser = load_helper_close_ser("$VIX", start_date_str, end_date_str)

    _set_snapshot_mode_bool(monkeypatch, True)
    snapshot_signal_close_df = load_signal_close_df(["GLD", "UUP", "TLT", "DBC"], start_date_str, end_date_str)
    snapshot_execution_price_df = load_execution_price_df(["GLD", "UUP", "TLT", "DBC", "SPY"], ["$SPX"], start_date_str, end_date_str)
    snapshot_vix_close_ser = load_helper_close_ser("$VIX", start_date_str, end_date_str)

    _assert_frame_equal(direct_signal_close_df, snapshot_signal_close_df)
    _assert_frame_equal(direct_execution_price_df, snapshot_execution_price_df)
    pd.testing.assert_series_equal(direct_vix_close_ser, snapshot_vix_close_ser, check_dtype=False, check_freq=False)


def test_ndx_and_vxn_boundary_parity(monkeypatch):
    from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
    from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import load_vxn_close_ser

    start_date_str = os.getenv("NORGATE_PARITY_START_DATE_STR", "2024-01-01")
    end_date_str = os.getenv("NORGATE_PARITY_END_DATE_STR", "2024-01-31")

    _set_snapshot_mode_bool(monkeypatch, False)
    _, direct_universe_df = build_index_constituent_matrix(indexname="Nasdaq 100")
    direct_price_df = load_raw_prices(
        symbols=["SPY"],
        benchmarks=[],
        start_date=start_date_str,
        end_date=end_date_str,
    )
    direct_vxn_close_ser = load_vxn_close_ser("$VXN", start_date_str, end_date_str)

    _set_snapshot_mode_bool(monkeypatch, True)
    _, snapshot_universe_df = build_index_constituent_matrix(indexname="Nasdaq 100")
    snapshot_price_df = load_raw_prices(
        symbols=["SPY"],
        benchmarks=[],
        start_date=start_date_str,
        end_date=end_date_str,
    )
    snapshot_vxn_close_ser = load_vxn_close_ser("$VXN", start_date_str, end_date_str)

    _assert_frame_equal(direct_universe_df, snapshot_universe_df)
    _assert_frame_equal(direct_price_df, snapshot_price_df)
    pd.testing.assert_series_equal(direct_vxn_close_ser, snapshot_vxn_close_ser, check_dtype=False, check_freq=False)
