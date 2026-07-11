import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from scripts.research.run_sector_dispersion_short_sleeve_study import (
    SHORT_MODE_MIRROR_STR,
    SHORT_MODE_SPX_SMA200_STR,
    SectorDispersionIbsShortSleeveStrategy,
    SectorDispersionShortSleeveConfig,
    _combined_total_value_ser,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import DEFAULT_CONFIG


class SectorDispersionShortSleeveStudyTests(unittest.TestCase):
    def make_symbol_ohlc_map(
        self,
        log_range_list: list[float],
        ibs_list: list[float],
        open_list: list[float] | None = None,
    ) -> dict[str, list[float]]:
        low_vec = np.full(len(log_range_list), 100.0)
        high_vec = low_vec * np.exp(np.array(log_range_list, dtype=float))
        close_vec = low_vec + np.array(ibs_list, dtype=float) * (high_vec - low_vec)
        if open_list is None:
            open_vec = close_vec.copy()
        else:
            open_vec = np.array(open_list, dtype=float)
        return {
            "Open": open_vec.tolist(),
            "High": high_vec.tolist(),
            "Low": low_vec.tolist(),
            "Close": close_vec.tolist(),
        }

    def make_pricing_data_df(
        self,
        spx_close_list: list[float] | None = None,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex, int, int]:
        date_index = pd.bdate_range("2024-01-02", periods=8)
        signal_day_int = 4
        fill_day_int = 5
        log_range_list = [0.010, 0.011, 0.012, 0.013, 0.120, 0.010, 0.011, 0.012]
        neutral_ibs_list = [0.50] * len(date_index)
        short_entry_ibs_list = [0.50] * len(date_index)
        short_entry_ibs_list[signal_day_int] = 0.95
        aaa_open_list = [100.0] * len(date_index)
        aaa_open_list[fill_day_int] = 111.0
        if spx_close_list is None:
            spx_close_list = [100.0 + index_int for index_int in range(len(date_index))]

        column_map: dict[tuple[str, str], pd.Series] = {}
        for symbol_str, field_map_dict in {
            "AAA": self.make_symbol_ohlc_map(
                log_range_list=log_range_list,
                ibs_list=short_entry_ibs_list,
                open_list=aaa_open_list,
            ),
            "BBB": self.make_symbol_ohlc_map(
                log_range_list=log_range_list,
                ibs_list=neutral_ibs_list,
            ),
        }.items():
            for field_str, value_list in field_map_dict.items():
                column_map[(symbol_str, field_str)] = pd.Series(value_list, index=date_index, dtype=float)
        column_map[("$SPX", "Close")] = pd.Series(spx_close_list, index=date_index, dtype=float)
        pricing_data_df = pd.DataFrame(column_map, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df, date_index, signal_day_int, fill_day_int

    def make_strategy(
        self,
        short_mode_str: str,
        spx_sma_lookback_day_int: int = 3,
    ) -> SectorDispersionIbsShortSleeveStrategy:
        base_config_obj = replace(
            DEFAULT_CONFIG,
            symbol_tuple=("AAA", "BBB"),
            history_start_date_str="2023-12-01",
            backtest_start_date_str="2024-01-02",
            range_vol_lookback_day_int=3,
            capital_base_float=100_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )
        short_config_obj = SectorDispersionShortSleeveConfig(
            base_config_obj=base_config_obj,
            short_mode_str=short_mode_str,
            spx_sma_lookback_day_int=spx_sma_lookback_day_int,
        )
        return SectorDispersionIbsShortSleeveStrategy(
            name="ShortSleeveTest",
            benchmarks=["$SPX"],
            short_config_obj=short_config_obj,
        )

    def test_mirror_short_enters_negative_target_at_next_open(self):
        pricing_data_df, date_index, signal_day_int, fill_day_int = self.make_pricing_data_df()
        strategy_obj = self.make_strategy(short_mode_str=SHORT_MODE_MIRROR_STR)

        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=date_index,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=False,
        )

        transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
        close_price_float = float(pricing_data_df.loc[date_index[signal_day_int], ("AAA", "Close")])
        expected_target_share_float = -100_000.0 * 0.5 / close_price_float

        self.assertEqual(len(transaction_df), 1)
        entry_row_ser = transaction_df.iloc[0]
        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), date_index[fill_day_int])
        self.assertEqual(entry_row_ser["asset"], "AAA")
        self.assertAlmostEqual(float(entry_row_ser["price"]), 111.0)
        self.assertAlmostEqual(float(entry_row_ser["amount"]), expected_target_share_float)

    def test_spx_sma_gate_blocks_short_entry_when_market_is_above_sma(self):
        pricing_data_df, date_index, _signal_day_int, _fill_day_int = self.make_pricing_data_df(
            spx_close_list=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        )
        strategy_obj = self.make_strategy(short_mode_str=SHORT_MODE_SPX_SMA200_STR)

        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=date_index,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=False,
        )

        self.assertEqual(len(strategy_obj.get_transactions()), 0)

    def test_combined_total_value_ser_uses_independent_sleeve_weights(self):
        date_index = pd.bdate_range("2024-01-02", periods=3)
        long_total_value_ser = pd.Series([100.0, 120.0, 150.0], index=date_index, dtype=float)
        short_total_value_ser = pd.Series([100.0, 90.0, 130.0], index=date_index, dtype=float)

        combined_total_value_ser = _combined_total_value_ser(
            long_total_value_ser=long_total_value_ser,
            short_total_value_ser=short_total_value_ser,
            short_allocation_float=0.25,
            capital_base_float=100_000.0,
        )

        expected_final_float = (0.75 * 1.5 + 0.25 * 1.3) * 100_000.0
        self.assertAlmostEqual(float(combined_total_value_ser.iloc[0]), 100_000.0)
        self.assertAlmostEqual(float(combined_total_value_ser.iloc[-1]), expected_final_float)


if __name__ == "__main__":
    unittest.main()
