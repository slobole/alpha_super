import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.bench import catalog
from strategies.mean_reversion import strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200 as asset_sma_module


class SectorDispersionIbsKieIhiXlcAssetSma200Tests(unittest.TestCase):
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
        signal_symbol_str: str,
        signal_low_list: list[float],
        date_index: pd.DatetimeIndex,
        signal_day_int: int,
        fill_day_int: int,
    ) -> pd.DataFrame:
        log_range_list = [0.010 + 0.001 * (index_int % 5) for index_int in range(len(date_index))]
        log_range_list[signal_day_int] = 0.120
        neutral_ibs_list = [0.50] * len(date_index)
        signal_ibs_list = [0.50] * len(date_index)
        signal_ibs_list[signal_day_int] = 0.05
        neutral_low_list = [100.0 + 0.01 * index_int for index_int in range(len(date_index))]
        signal_open_list = signal_low_list.copy()
        signal_open_list[fill_day_int] = 111.0

        column_map: dict[tuple[str, str], pd.Series] = {}
        for symbol_str in asset_sma_module.STRATEGY_SYMBOL_TUPLE:
            if symbol_str == signal_symbol_str:
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
                column_map[(symbol_str, field_str)] = pd.Series(value_list, index=date_index, dtype=float)

        column_map[("$SPX", "Close")] = pd.Series(
            [5000.0 + index_int for index_int in range(len(date_index))],
            index=date_index,
            dtype=float,
        )
        pricing_data_df = pd.DataFrame(column_map, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_default_config_uses_fixed_defensive_basket(self):
        self.assertEqual(
            asset_sma_module.STRATEGY_SYMBOL_TUPLE,
            ("SOXX", "IGV", "IBB", "KIE", "IHI", "XLC"),
        )
        self.assertEqual(asset_sma_module.DEFAULT_CONFIG.symbol_tuple, asset_sma_module.STRATEGY_SYMBOL_TUPLE)
        self.assertAlmostEqual(asset_sma_module.DEFAULT_CONFIG.slippage_float, 0.00025)
        self.assertEqual(asset_sma_module.ASSET_SMA_LOOKBACK_DAY_INT, 200)

    def test_bench_catalog_discovers_asset_sma200_variant(self):
        entry_obj = catalog.get_strategy_by_module(asset_sma_module.__name__)

        self.assertIsNotNone(entry_obj)
        assert entry_obj is not None
        self.assertTrue(entry_obj.has_run_variant_bool)
        self.assertFalse(entry_obj.is_wired_bool)

    def test_signal_layer_preserves_raw_entry_and_gates_entry_with_asset_sma200(self):
        date_index = pd.bdate_range("2023-01-02", periods=205)
        signal_day_int = 202
        fill_day_int = 203
        signal_low_list = [200.0 - 0.45 * index_int for index_int in range(len(date_index))]
        pricing_data_df = self.make_pricing_data_df(
            signal_symbol_str="XLC",
            signal_low_list=signal_low_list,
            date_index=date_index,
            signal_day_int=signal_day_int,
            fill_day_int=fill_day_int,
        )

        signal_data_df = asset_sma_module.compute_asset_sma200_filtered_signal_df(
            pricing_data_df=pricing_data_df,
            config_obj=asset_sma_module.DEFAULT_CONFIG,
        )

        self.assertTrue(
            bool(signal_data_df.loc[date_index[signal_day_int], ("XLC", asset_sma_module.RAW_ENTRY_SIGNAL_FIELD_STR)])
        )
        self.assertFalse(
            bool(signal_data_df.loc[date_index[signal_day_int], ("XLC", asset_sma_module.ASSET_BULLISH_FIELD_STR)])
        )
        self.assertFalse(bool(signal_data_df.loc[date_index[signal_day_int], ("XLC", "entry_signal_bool")]))

    def test_run_variant_enters_only_when_asset_is_above_sma200_and_fills_next_open(self):
        date_index = pd.bdate_range("2023-01-02", periods=205)
        signal_day_int = 202
        fill_day_int = 203
        signal_low_list = [80.0 + 0.45 * index_int for index_int in range(len(date_index))]
        pricing_data_df = self.make_pricing_data_df(
            signal_symbol_str="XLC",
            signal_low_list=signal_low_list,
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
        close_price_float = float(pricing_data_df.loc[date_index[signal_day_int], ("XLC", "Close")])
        expected_target_share_float = (
            100_000.0
            * (asset_sma_module.DEFAULT_CONFIG.portfolio_leverage_float / len(asset_sma_module.STRATEGY_SYMBOL_TUPLE))
            / close_price_float
        )
        expected_fill_price_float = 111.0 * (
            1.0 + asset_sma_module.DEFAULT_CONFIG.slippage_float
        )

        self.assertEqual(strategy_obj.name, asset_sma_module.STRATEGY_NAME_STR)
        self.assertEqual(strategy_obj.symbol_tuple, asset_sma_module.STRATEGY_SYMBOL_TUPLE)
        self.assertEqual(len(transaction_df), 1)
        entry_row_ser = transaction_df.iloc[0]
        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), date_index[fill_day_int])
        self.assertEqual(entry_row_ser["asset"], "XLC")
        self.assertAlmostEqual(float(entry_row_ser["price"]), expected_fill_price_float)
        self.assertAlmostEqual(float(entry_row_ser["amount"]), expected_target_share_float)

    def test_run_variant_accepts_portfolio_manager_start_before_default_history_start(self):
        date_index = pd.bdate_range("2004-01-01", periods=205)
        signal_low_list = [80.0 + 0.45 * index_int for index_int in range(len(date_index))]
        pricing_data_df = self.make_pricing_data_df(
            signal_symbol_str="XLC",
            signal_low_list=signal_low_list,
            date_index=date_index,
            signal_day_int=202,
            fill_day_int=203,
        )

        strategy_obj = asset_sma_module.run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str="2004-01-01",
            capital_base_float=100_000.0,
            pricing_data_df=pricing_data_df,
            audit_override_bool=False,
        )

        self.assertLess(
            pd.Timestamp(strategy_obj.config_obj.history_start_date_str),
            pd.Timestamp(strategy_obj.config_obj.backtest_start_date_str),
        )
        self.assertEqual(strategy_obj.config_obj.backtest_start_date_str, "2004-01-01")

    def test_saved_metadata_points_to_asset_sma200_wrapper_module(self):
        date_index = pd.bdate_range("2023-01-02", periods=205)
        signal_low_list = [80.0 + 0.45 * index_int for index_int in range(len(date_index))]
        pricing_data_df = self.make_pricing_data_df(
            signal_symbol_str="XLC",
            signal_low_list=signal_low_list,
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
                capital_base_float=100_000.0,
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
        self.assertEqual(metadata_dict["class_name"], "SectorDispersionIbsKieIhiXlcAssetSma200Strategy")


if __name__ == "__main__":
    unittest.main()
