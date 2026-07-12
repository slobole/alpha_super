import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from scripts.research.run_sector_dispersion_volatility_rank_filter_study import (
    BULLISH_FILTER_ASSET_SMA_STR,
    BULLISH_FILTER_NONE_STR,
    BULLISH_FILTER_SPX_SMA_STR,
    SectorDispersionVolatilityRankFilterConfig,
    SectorDispersionVolatilityRankFilterStrategy,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import DEFAULT_CONFIG


class SectorDispersionVolatilityRankFilterStudyTests(unittest.TestCase):
    def make_symbol_ohlc_map(
        self,
        log_range_list: list[float],
        ibs_list: list[float],
        low_list: list[float] | None = None,
        open_list: list[float] | None = None,
    ) -> dict[str, list[float]]:
        if low_list is None:
            low_vec = np.full(len(log_range_list), 100.0)
        else:
            low_vec = np.array(low_list, dtype=float)
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
        aaa_low_list: list[float] | None = None,
        bbb_low_list: list[float] | None = None,
        spx_close_list: list[float] | None = None,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex, int, int]:
        date_index = pd.bdate_range("2024-01-02", periods=8)
        signal_day_int = 4
        fill_day_int = 5
        aaa_log_range_list = [0.010, 0.020, 0.080, 0.010, 0.120, 0.010, 0.011, 0.012]
        bbb_log_range_list = [0.010, 0.011, 0.012, 0.013, 0.120, 0.010, 0.011, 0.012]
        ibs_list = [0.50] * len(date_index)
        ibs_list[signal_day_int] = 0.05
        aaa_open_list = [100.0] * len(date_index)
        bbb_open_list = [200.0] * len(date_index)
        aaa_open_list[fill_day_int] = 111.0
        bbb_open_list[fill_day_int] = 222.0
        if spx_close_list is None:
            spx_close_list = [100.0 + index_int for index_int in range(len(date_index))]

        column_map: dict[tuple[str, str], pd.Series] = {}
        for symbol_str, field_map_dict in {
            "AAA": self.make_symbol_ohlc_map(
                log_range_list=aaa_log_range_list,
                ibs_list=ibs_list,
                low_list=aaa_low_list,
                open_list=aaa_open_list,
            ),
            "BBB": self.make_symbol_ohlc_map(
                log_range_list=bbb_log_range_list,
                ibs_list=ibs_list,
                low_list=bbb_low_list,
                open_list=bbb_open_list,
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
        vol_rank_top_n_int: int | None,
        bullish_filter_str: str,
    ) -> SectorDispersionVolatilityRankFilterStrategy:
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
        filter_config_obj = SectorDispersionVolatilityRankFilterConfig(
            base_config_obj=base_config_obj,
            vol_rank_top_n_int=vol_rank_top_n_int,
            bullish_filter_str=bullish_filter_str,
            bullish_sma_lookback_day_int=3,
        )
        return SectorDispersionVolatilityRankFilterStrategy(
            name="VolRankFilterTest",
            benchmarks=["$SPX"],
            filter_config_obj=filter_config_obj,
        )

    def test_volatility_rank_top1_enters_only_highest_vol_asset_at_next_open(self):
        pricing_data_df, date_index, signal_day_int, fill_day_int = self.make_pricing_data_df()
        strategy_obj = self.make_strategy(
            vol_rank_top_n_int=1,
            bullish_filter_str=BULLISH_FILTER_NONE_STR,
        )

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
        expected_target_share_float = (
            100_000.0 * strategy_obj.target_weight_float / close_price_float
        )

        self.assertEqual(len(transaction_df), 1)
        entry_row_ser = transaction_df.iloc[0]
        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), date_index[fill_day_int])
        self.assertEqual(entry_row_ser["asset"], "AAA")
        self.assertAlmostEqual(float(entry_row_ser["price"]), 111.0)
        self.assertAlmostEqual(float(entry_row_ser["amount"]), expected_target_share_float)

    def test_spx_bullish_filter_blocks_entry_when_market_is_below_sma(self):
        pricing_data_df, date_index, _signal_day_int, _fill_day_int = self.make_pricing_data_df(
            spx_close_list=[108.0, 107.0, 106.0, 105.0, 100.0, 101.0, 102.0, 103.0],
        )
        strategy_obj = self.make_strategy(
            vol_rank_top_n_int=None,
            bullish_filter_str=BULLISH_FILTER_SPX_SMA_STR,
        )

        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=date_index,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=False,
        )

        self.assertEqual(len(strategy_obj.get_transactions()), 0)

    def test_asset_bullish_filter_allows_only_asset_above_own_sma(self):
        pricing_data_df, date_index, signal_day_int, fill_day_int = self.make_pricing_data_df(
            aaa_low_list=[80.0, 82.0, 84.0, 86.0, 100.0, 100.0, 100.0, 100.0],
            bbb_low_list=[130.0, 128.0, 126.0, 124.0, 100.0, 100.0, 100.0, 100.0],
        )
        strategy_obj = self.make_strategy(
            vol_rank_top_n_int=None,
            bullish_filter_str=BULLISH_FILTER_ASSET_SMA_STR,
        )

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
        expected_target_share_float = (
            100_000.0 * strategy_obj.target_weight_float / close_price_float
        )

        self.assertEqual(len(transaction_df), 1)
        entry_row_ser = transaction_df.iloc[0]
        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), date_index[fill_day_int])
        self.assertEqual(entry_row_ser["asset"], "AAA")
        self.assertAlmostEqual(float(entry_row_ser["amount"]), expected_target_share_float)


if __name__ == "__main__":
    unittest.main()
