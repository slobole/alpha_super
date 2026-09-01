import unittest
from dataclasses import replace
from inspect import signature
from unittest.mock import patch

import numpy as np
import pandas as pd

from alpha.engine.order import MarketOrder
from alpha.engine.strategy import Strategy
from strategies.momentum.strategy_mo_ibit_adaptive_momentum_regime import (
    IBIT_CONFIG,
    IbitAdaptiveMomentumRegimeStrategy,
    get_ibit_adaptive_momentum_regime_data,
    run_variant as run_ibit_variant,
)
from strategies.momentum.strategy_mo_qqq_adaptive_momentum_regime import (
    QQQ_CONFIG,
    QqqAdaptiveMomentumRegimeStrategy,
    get_qqq_adaptive_momentum_regime_data,
    run_variant as run_qqq_variant,
)
from strategies.momentum.strategy_mo_spy_adaptive_momentum_regime import (
    SIGNAL_NAMESPACE_STR,
)


class QqqIbitAdaptiveMomentumRegimeTests(unittest.TestCase):
    def variant_case_dict(self) -> dict[str, dict[str, object]]:
        return {
            "QQQ": {
                "config_obj": QQQ_CONFIG,
                "signal_symbol_str": "QQQ_TR_SIGNAL",
                "strategy_class_obj": QqqAdaptiveMomentumRegimeStrategy,
                "loader_function_obj": get_qqq_adaptive_momentum_regime_data,
                "run_function_obj": run_qqq_variant,
                "run_daily_patch_str": (
                    "strategies.momentum.strategy_mo_qqq_adaptive_momentum_regime.run_daily"
                ),
            },
            "IBIT": {
                "config_obj": IBIT_CONFIG,
                "signal_symbol_str": "IBIT_TR_SIGNAL",
                "strategy_class_obj": IbitAdaptiveMomentumRegimeStrategy,
                "loader_function_obj": get_ibit_adaptive_momentum_regime_data,
                "run_function_obj": run_ibit_variant,
                "run_daily_patch_str": (
                    "strategies.momentum.strategy_mo_ibit_adaptive_momentum_regime.run_daily"
                ),
            },
        }

    def make_config(self, asset_symbol_str: str):
        base_config_obj = self.variant_case_dict()[asset_symbol_str]["config_obj"]
        return replace(
            base_config_obj,
            percentile_lookback_int=3,
            fast_lookback_int=2,
            slow_lookback_int=4,
            price_filter_lookback_int=2,
            history_start_date_str="2023-12-01",
            backtest_start_date_str="2024-01-02",
        )

    def make_pricing_data_df(self, asset_symbol_str: str) -> pd.DataFrame:
        case_dict = self.variant_case_dict()[asset_symbol_str]
        signal_symbol_str = str(case_dict["signal_symbol_str"])
        date_index = pd.date_range("2024-01-02", periods=8, freq="B")
        execution_close_vec = np.array(
            [100.0, 101.0, 99.0, 98.0, 103.0, 104.0, 105.0, 106.0]
        )
        total_return_signal_close_vec = np.array(
            [200.0, 204.0, 192.0, 198.0, 208.0, 202.0, 212.0, 218.0]
        )
        benchmark_close_vec = np.arange(5000.0, 5008.0)
        column_dict = {
            (asset_symbol_str, "Open"): execution_close_vec - 0.5,
            (asset_symbol_str, "High"): execution_close_vec + 1.0,
            (asset_symbol_str, "Low"): execution_close_vec - 1.0,
            (asset_symbol_str, "Close"): execution_close_vec,
            (asset_symbol_str, "Unadjusted Close"): execution_close_vec,
            (asset_symbol_str, "Dividend"): np.zeros(len(date_index)),
            ("$SPX", "Open"): benchmark_close_vec - 1.0,
            ("$SPX", "High"): benchmark_close_vec + 1.0,
            ("$SPX", "Low"): benchmark_close_vec - 1.0,
            ("$SPX", "Close"): benchmark_close_vec,
            ("$SPX", "Unadjusted Close"): benchmark_close_vec,
            ("$SPX", "Dividend"): np.zeros(len(date_index)),
            (signal_symbol_str, "Open"): total_return_signal_close_vec - 0.5,
            (signal_symbol_str, "Close"): total_return_signal_close_vec,
        }
        return pd.DataFrame(column_dict, index=date_index)

    def test_loader_uses_each_asset_for_execution_and_total_return_signal(self):
        date_index = pd.date_range("2024-01-11", periods=2, freq="B")
        for asset_symbol_str, case_dict in self.variant_case_dict().items():
            with self.subTest(asset_symbol_str=asset_symbol_str):
                signal_symbol_str = str(case_dict["signal_symbol_str"])
                execution_df = pd.DataFrame(
                    {
                        (asset_symbol_str, "Open"): [100.0, 101.0],
                        (asset_symbol_str, "Close"): [101.0, 102.0],
                        ("$SPX", "Open"): [5000.0, 5010.0],
                        ("$SPX", "Close"): [5005.0, 5015.0],
                    },
                    index=date_index,
                )
                execution_df.attrs["norgate_adjustment_by_symbol_dict"] = {
                    asset_symbol_str: "CAPITALSPECIAL",
                    "$SPX": "TOTALRETURN",
                }
                total_return_signal_df = pd.DataFrame(
                    {
                        (asset_symbol_str, "Open"): [200.0, 202.0],
                        (asset_symbol_str, "Close"): [201.0, 203.0],
                    },
                    index=date_index,
                )

                with patch(
                    "strategies.momentum.strategy_mo_spy_adaptive_momentum_regime.load_raw_prices",
                    side_effect=[execution_df, total_return_signal_df],
                ) as load_mock:
                    pricing_data_df = case_dict["loader_function_obj"](
                        self.make_config(asset_symbol_str)
                    )

                self.assertEqual(
                    load_mock.call_args_list[0].kwargs["symbols"],
                    [asset_symbol_str],
                )
                self.assertEqual(
                    load_mock.call_args_list[1].kwargs["benchmarks"],
                    [asset_symbol_str],
                )
                self.assertIn((asset_symbol_str, "Close"), pricing_data_df.columns)
                self.assertIn((signal_symbol_str, "Close"), pricing_data_df.columns)
                self.assertEqual(
                    pricing_data_df.attrs["signal_data_symbol_by_alias_dict"],
                    {signal_symbol_str: asset_symbol_str},
                )

    def test_signal_and_order_both_use_the_same_asset(self):
        for asset_symbol_str, case_dict in self.variant_case_dict().items():
            with self.subTest(asset_symbol_str=asset_symbol_str):
                config_obj = self.make_config(asset_symbol_str)
                strategy_obj = case_dict["strategy_class_obj"](
                    name=f"{asset_symbol_str.lower()}_self_signal_test",
                    benchmarks=[config_obj.benchmark_symbol_str],
                    config=config_obj,
                )
                pricing_data_df = self.make_pricing_data_df(asset_symbol_str)
                signal_data_df = strategy_obj.compute_signals(pricing_data_df)
                self.assertEqual(
                    float(strategy_obj.regime_signal_df.iloc[2]["signal_price_close_ser"]),
                    192.0,
                )
                strategy_obj.previous_bar = pricing_data_df.index[-2]
                strategy_obj.current_bar = pricing_data_df.index[-1]
                close_row_ser = signal_data_df.loc[strategy_obj.previous_bar].copy()
                close_row_ser.loc[(SIGNAL_NAMESPACE_STR, "target_weight_ser")] = 1.0

                strategy_obj.iterate(
                    signal_data_df.loc[: strategy_obj.previous_bar],
                    close_row_ser,
                    pd.Series({asset_symbol_str: 100.0}, dtype=float),
                )

                order_list = strategy_obj.get_orders()
                self.assertEqual(len(order_list), 1)
                self.assertIsInstance(order_list[0], MarketOrder)
                self.assertEqual(order_list[0].asset, asset_symbol_str)

    def test_run_variants_match_engine_cost_defaults(self):
        engine_signature_obj = signature(Strategy.__init__)
        for asset_symbol_str, case_dict in self.variant_case_dict().items():
            with self.subTest(asset_symbol_str=asset_symbol_str):
                strategy_obj = case_dict["run_function_obj"](
                    show_display_bool=False,
                    save_results_bool=False,
                    pricing_data_df=self.make_pricing_data_df(asset_symbol_str),
                    config_obj=self.make_config(asset_symbol_str),
                )

                self.assertEqual(
                    strategy_obj.config.slippage_float,
                    engine_signature_obj.parameters["slippage"].default,
                )
                self.assertEqual(
                    strategy_obj.config.commission_per_share_float,
                    engine_signature_obj.parameters["commission_per_share"].default,
                )
                self.assertEqual(
                    strategy_obj.config.commission_minimum_float,
                    engine_signature_obj.parameters["commission_minimum"].default,
                )
                self.assertIsNotNone(strategy_obj.summary)

    def test_run_variants_exclude_pre_inception_rows(self):
        pre_inception_index = pd.date_range("2023-12-28", periods=2, freq="B")
        for asset_symbol_str, case_dict in self.variant_case_dict().items():
            with self.subTest(asset_symbol_str=asset_symbol_str):
                signal_symbol_str = str(case_dict["signal_symbol_str"])
                pricing_data_df = self.make_pricing_data_df(asset_symbol_str)
                pre_inception_df = pd.DataFrame(
                    np.nan,
                    index=pre_inception_index,
                    columns=pricing_data_df.columns,
                )
                pre_inception_df.loc[:, (signal_symbol_str, "Close")] = [98.0, 99.0]
                combined_pricing_df = pd.concat([pre_inception_df, pricing_data_df])

                with patch(str(case_dict["run_daily_patch_str"])) as run_daily_mock:
                    case_dict["run_function_obj"](
                        show_display_bool=False,
                        save_results_bool=False,
                        pricing_data_df=combined_pricing_df,
                        config_obj=self.make_config(asset_symbol_str),
                    )

                calendar_index = run_daily_mock.call_args.kwargs["calendar"]
                self.assertTrue(calendar_index.equals(pricing_data_df.index))

    def test_run_variants_reject_cross_asset_configs(self):
        with self.assertRaisesRegex(ValueError, "trade_symbol_str='QQQ'"):
            run_qqq_variant(
                show_display_bool=False,
                save_results_bool=False,
                pricing_data_df=self.make_pricing_data_df("QQQ"),
                config_obj=replace(QQQ_CONFIG, trade_symbol_str="SPY"),
            )
        with self.assertRaisesRegex(ValueError, "trade_symbol_str='IBIT'"):
            run_ibit_variant(
                show_display_bool=False,
                save_results_bool=False,
                pricing_data_df=self.make_pricing_data_df("IBIT"),
                config_obj=replace(IBIT_CONFIG, trade_symbol_str="SPY"),
            )

    def test_strategy_classes_reject_cross_asset_configs(self):
        with self.assertRaisesRegex(ValueError, "trade_symbol_str='QQQ'"):
            QqqAdaptiveMomentumRegimeStrategy(
                name="invalid_qqq_strategy",
                benchmarks=[QQQ_CONFIG.benchmark_symbol_str],
                config=IBIT_CONFIG,
            )
        with self.assertRaisesRegex(ValueError, "trade_symbol_str='IBIT'"):
            IbitAdaptiveMomentumRegimeStrategy(
                name="invalid_ibit_strategy",
                benchmarks=[IBIT_CONFIG.benchmark_symbol_str],
                config=QQQ_CONFIG,
            )


if __name__ == "__main__":
    unittest.main()
