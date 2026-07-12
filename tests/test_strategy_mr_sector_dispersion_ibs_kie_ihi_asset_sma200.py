import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.bench import catalog
from strategies.mean_reversion import (
    strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200 as asset_sma_module,
)


class SectorDispersionIbsKieIhiAssetSma200Tests(unittest.TestCase):
    def make_symbol_ohlc_map(
        self,
        log_range_list: list[float],
        ibs_list: list[float],
        low_list: list[float],
        open_list: list[float] | None = None,
    ) -> dict[str, list[float]]:
        low_vec = np.array(low_list, dtype=float)
        high_vec = low_vec * np.exp(np.array(log_range_list, dtype=float))
        close_vec = low_vec + np.array(ibs_list, dtype=float) * (high_vec - low_vec)
        open_vec = close_vec.copy() if open_list is None else np.array(open_list, dtype=float)
        high_vec = np.maximum(high_vec, open_vec)
        low_vec = np.minimum(low_vec, open_vec)
        return {
            "Open": open_vec.tolist(),
            "High": high_vec.tolist(),
            "Low": low_vec.tolist(),
            "Close": close_vec.tolist(),
        }

    def make_pricing_data_df(
        self,
        signal_low_list: list[float],
        date_index: pd.DatetimeIndex,
        signal_day_int: int,
        fill_day_int: int,
    ) -> pd.DataFrame:
        log_range_list = [
            0.010 + 0.001 * (index_int % 5)
            for index_int in range(len(date_index))
        ]
        log_range_list[signal_day_int] = 0.120
        neutral_ibs_list = [0.50] * len(date_index)
        signal_ibs_list = [0.50] * len(date_index)
        signal_ibs_list[signal_day_int] = 0.05
        neutral_low_list = [100.0 + 0.01 * index_int for index_int in range(len(date_index))]
        signal_open_list = signal_low_list.copy()
        signal_open_list[fill_day_int] = signal_low_list[fill_day_int] + 0.5

        column_map: dict[tuple[str, str], pd.Series] = {}
        for symbol_str in asset_sma_module.STRATEGY_SYMBOL_TUPLE:
            if symbol_str == "IHI":
                field_map_dict = self.make_symbol_ohlc_map(
                    log_range_list=log_range_list,
                    ibs_list=signal_ibs_list,
                    low_list=signal_low_list,
                    open_list=signal_open_list,
                )
            else:
                field_map_dict = self.make_symbol_ohlc_map(
                    log_range_list=log_range_list,
                    ibs_list=neutral_ibs_list,
                    low_list=neutral_low_list,
                )
            for field_str, value_list in field_map_dict.items():
                column_map[(symbol_str, field_str)] = pd.Series(
                    value_list,
                    index=date_index,
                    dtype=float,
                )

        column_map[("$SPX", "Close")] = pd.Series(
            [5000.0 + index_int for index_int in range(len(date_index))],
            index=date_index,
            dtype=float,
        )
        pricing_data_df = pd.DataFrame(column_map, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_default_config_and_bench_discovery(self):
        self.assertEqual(
            asset_sma_module.STRATEGY_SYMBOL_TUPLE,
            ("SOXX", "IGV", "IBB", "KIE", "IHI"),
        )
        self.assertAlmostEqual(asset_sma_module.DEFAULT_CONFIG.portfolio_leverage_float, 1.0)
        self.assertEqual(asset_sma_module.ASSET_SMA_LOOKBACK_DAY_INT, 200)

        entry_obj = catalog.get_strategy_by_module(asset_sma_module.__name__)

        self.assertIsNotNone(entry_obj)
        assert entry_obj is not None
        self.assertTrue(entry_obj.has_run_variant_bool)
        self.assertFalse(entry_obj.is_wired_bool)

    def test_bearish_asset_keeps_raw_entry_but_blocks_final_entry(self):
        date_index = pd.bdate_range("2023-01-02", periods=205)
        signal_day_int = 202
        pricing_data_df = self.make_pricing_data_df(
            signal_low_list=[200.0 - 0.45 * index_int for index_int in range(len(date_index))],
            date_index=date_index,
            signal_day_int=signal_day_int,
            fill_day_int=203,
        )

        signal_data_df = asset_sma_module.compute_asset_sma200_filtered_signal_df(
            pricing_data_df=pricing_data_df,
        )

        self.assertTrue(
            bool(signal_data_df.loc[date_index[signal_day_int], ("IHI", asset_sma_module.RAW_ENTRY_SIGNAL_FIELD_STR)])
        )
        self.assertFalse(
            bool(signal_data_df.loc[date_index[signal_day_int], ("IHI", asset_sma_module.ASSET_BULLISH_FIELD_STR)])
        )
        self.assertFalse(bool(signal_data_df.loc[date_index[signal_day_int], ("IHI", "entry_signal_bool")]))

    def test_bullish_asset_enters_at_next_open(self):
        date_index = pd.bdate_range("2023-01-02", periods=205)
        signal_day_int = 202
        fill_day_int = 203
        pricing_data_df = self.make_pricing_data_df(
            signal_low_list=[80.0 + 0.45 * index_int for index_int in range(len(date_index))],
            date_index=date_index,
            signal_day_int=signal_day_int,
            fill_day_int=fill_day_int,
        )

        strategy_obj = asset_sma_module.run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str=date_index[0].date().isoformat(),
            capital_base_float=100_000.0,
            pricing_data_df=pricing_data_df,
            audit_override_bool=False,
        )

        transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
        signal_close_float = float(pricing_data_df.loc[date_index[signal_day_int], ("IHI", "Close")])
        expected_amount_float = (
            100_000.0
            * (asset_sma_module.DEFAULT_CONFIG.portfolio_leverage_float / len(asset_sma_module.STRATEGY_SYMBOL_TUPLE))
            / signal_close_float
        )
        fill_open_float = float(pricing_data_df.loc[date_index[fill_day_int], ("IHI", "Open")])
        expected_fill_float = fill_open_float * (1.0 + asset_sma_module.DEFAULT_CONFIG.slippage_float)

        self.assertEqual(strategy_obj.results.index[0], date_index[199])
        self.assertEqual(len(transaction_df), 1)
        entry_row_ser = transaction_df.iloc[0]
        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), date_index[fill_day_int])
        self.assertEqual(entry_row_ser["asset"], "IHI")
        self.assertAlmostEqual(float(entry_row_ser["price"]), expected_fill_float)
        self.assertAlmostEqual(float(entry_row_ser["amount"]), expected_amount_float)

    def test_readiness_waits_for_200_delayed_ihi_closes(self):
        date_index = pd.bdate_range("2023-01-02", periods=210)
        pricing_data_df = self.make_pricing_data_df(
            signal_low_list=[80.0 + 0.45 * index_int for index_int in range(len(date_index))],
            date_index=date_index,
            signal_day_int=205,
            fill_day_int=206,
        )
        for field_str in ("Open", "High", "Low", "Close"):
            pricing_data_df.loc[date_index[:4], ("IHI", field_str)] = np.nan

        strategy_obj = asset_sma_module.run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str=date_index[0].date().isoformat(),
            pricing_data_df=pricing_data_df,
            audit_override_bool=False,
        )

        self.assertEqual(strategy_obj.results.index[0], date_index[203])

    def test_saved_metadata_points_to_new_wrapper(self):
        date_index = pd.bdate_range("2023-01-02", periods=205)
        pricing_data_df = self.make_pricing_data_df(
            signal_low_list=[80.0 + 0.45 * index_int for index_int in range(len(date_index))],
            date_index=date_index,
            signal_day_int=202,
            fill_day_int=203,
        )

        with tempfile.TemporaryDirectory() as temp_dir_str:
            asset_sma_module.run_variant(
                show_display_bool=False,
                save_results_bool=True,
                output_dir_str=temp_dir_str,
                backtest_start_date_str=date_index[0].date().isoformat(),
                pricing_data_df=pricing_data_df,
                audit_override_bool=False,
            )
            metadata_path_list = sorted(
                (
                    Path(temp_dir_str)
                    / "research"
                    / "strategy"
                    / asset_sma_module.STRATEGY_NAME_STR
                    / "vanilla_backtest"
                ).glob("*/metadata.json")
            )
            self.assertEqual(len(metadata_path_list), 1)
            metadata_dict = json.loads(metadata_path_list[0].read_text(encoding="utf-8"))

        self.assertEqual(metadata_dict["class_module"], asset_sma_module.__name__)
        self.assertEqual(
            metadata_dict["class_name"],
            "SectorDispersionIbsKieIhiAssetSma200Strategy",
        )


if __name__ == "__main__":
    unittest.main()
