import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.strategy_mo_arp_core9 import (
    ArpCore9Config,
    ArpCore9Strategy,
    _compute_single_asset_trend_signal_df,
    compute_arp_core9_signal_tables,
    compute_matrix_inverse_sqrt_df,
)


class ArpCore9StrategyTests(unittest.TestCase):
    def make_config(self, **kwargs) -> ArpCore9Config:
        base_kwargs = dict(
            risk_asset_list=("SPY", "VEA", "VWO", "IEF", "TLT", "GLD", "DBC", "VNQ"),
            cash_proxy_str="BIL",
            benchmark_list=("$SPX",),
            eta_float=0.25,
            portfolio_smoothing_float=0.50,
            vol_lookback_day_int=3,
            corr_lookback_week_int=4,
            corr_min_period_week_int=2,
            target_volatility_float=0.10,
            max_gross_weight_float=1.0,
            eigenvalue_floor_float=1e-6,
            capital_base_float=100_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )
        base_kwargs.update(kwargs)
        return ArpCore9Config(**base_kwargs)

    def make_strategy(self, **kwargs) -> ArpCore9Strategy:
        config = self.make_config()
        base_kwargs = dict(
            name="ArpCore9Test",
            benchmarks=["$SPX"],
            risk_asset_list=config.risk_asset_list,
            cash_proxy_str=config.cash_proxy_str,
            eta_float=config.eta_float,
            portfolio_smoothing_float=config.portfolio_smoothing_float,
            vol_lookback_day_int=config.vol_lookback_day_int,
            corr_lookback_week_int=config.corr_lookback_week_int,
            corr_min_period_week_int=config.corr_min_period_week_int,
            target_volatility_float=config.target_volatility_float,
            max_gross_weight_float=config.max_gross_weight_float,
            eigenvalue_floor_float=config.eigenvalue_floor_float,
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )
        base_kwargs.update(kwargs)
        return ArpCore9Strategy(**base_kwargs)

    def make_pricing_data_df(self, downward_risk_bool: bool = False) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=80, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        asset_param_map: dict[str, tuple[float, float, float, float]] = {
            "SPY": (100.0, 1.4, 0.10, 0.0),
            "VEA": (90.0, 1.2, 0.09, 0.6),
            "VWO": (80.0, 1.6, 0.11, 1.2),
            "IEF": (105.0, 0.8, 0.08, 1.8),
            "TLT": (110.0, 0.7, 0.07, 2.4),
            "GLD": (120.0, 1.0, 0.09, 3.0),
            "DBC": (70.0, 1.5, 0.10, 3.6),
            "VNQ": (95.0, 1.1, 0.09, 4.2),
        }

        first_up_vec = np.minimum(step_vec, 25.0)
        middle_down_vec = np.clip(step_vec - 25.0, 0.0, 25.0)
        final_up_vec = np.clip(step_vec - 50.0, 0.0, None)

        data_map: dict[tuple[str, str], np.ndarray] = {}
        for asset_str, (base_float, amplitude_float, frequency_float, phase_float) in asset_param_map.items():
            if downward_risk_bool:
                signal_close_vec = (
                    base_float
                    - (0.28 * step_vec)
                    + (0.15 * amplitude_float * np.sin(step_vec * frequency_float + phase_float))
                )
            else:
                signal_close_vec = (
                    base_float
                    + (0.45 * first_up_vec)
                    - (0.60 * middle_down_vec)
                    + (0.35 * final_up_vec)
                    + (amplitude_float * np.sin(step_vec * frequency_float + phase_float))
                )

            close_vec = signal_close_vec.copy()
            data_map[(asset_str, "Open")] = close_vec - 0.25
            data_map[(asset_str, "High")] = close_vec + 0.75
            data_map[(asset_str, "Low")] = close_vec - 0.75
            data_map[(asset_str, "Close")] = close_vec
            data_map[(asset_str, "SignalClose")] = signal_close_vec

        bil_signal_close_vec = 50.0 + (0.01 * step_vec)
        data_map[("BIL", "Open")] = bil_signal_close_vec - 0.02
        data_map[("BIL", "High")] = bil_signal_close_vec + 0.05
        data_map[("BIL", "Low")] = bil_signal_close_vec - 0.05
        data_map[("BIL", "Close")] = bil_signal_close_vec
        data_map[("BIL", "SignalClose")] = bil_signal_close_vec

        benchmark_close_vec = 4_000.0 + (1.5 * step_vec) + (12.0 * np.sin(step_vec * 0.05))
        data_map[("$SPX", "Open")] = benchmark_close_vec - 2.0
        data_map[("$SPX", "High")] = benchmark_close_vec + 4.0
        data_map[("$SPX", "Low")] = benchmark_close_vec - 4.0
        data_map[("$SPX", "Close")] = benchmark_close_vec

        pricing_data_df = pd.DataFrame(data_map, index=date_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map, dtype=float)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_compute_matrix_inverse_sqrt_df_identity_matrix(self):
        matrix_df = pd.DataFrame(np.eye(3), index=["A", "B", "C"], columns=["A", "B", "C"], dtype=float)
        inverse_sqrt_df = compute_matrix_inverse_sqrt_df(matrix_df, eigenvalue_floor_float=1e-6)
        np.testing.assert_allclose(inverse_sqrt_df.to_numpy(dtype=float), np.eye(3), atol=1e-12)

    def test_compute_matrix_inverse_sqrt_df_diagonal_matrix(self):
        matrix_df = pd.DataFrame(np.diag([4.0, 9.0]), index=["A", "B"], columns=["A", "B"], dtype=float)
        inverse_sqrt_df = compute_matrix_inverse_sqrt_df(matrix_df, eigenvalue_floor_float=1e-6)
        expected_arr = np.diag([0.5, 1.0 / 3.0])
        np.testing.assert_allclose(inverse_sqrt_df.to_numpy(dtype=float), expected_arr, atol=1e-12)

    def test_compute_matrix_inverse_sqrt_df_nearly_singular_stays_finite_with_floor(self):
        matrix_df = pd.DataFrame(
            [[1.0, 0.999999], [0.999999, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
            dtype=float,
        )
        inverse_sqrt_df = compute_matrix_inverse_sqrt_df(matrix_df, eigenvalue_floor_float=1e-3)
        inverse_sqrt_arr = inverse_sqrt_df.to_numpy(dtype=float)

        self.assertTrue(np.isfinite(inverse_sqrt_arr).all())
        np.testing.assert_allclose(inverse_sqrt_arr, inverse_sqrt_arr.T, atol=1e-12)

        max_eigenvalue_float = float(np.max(np.linalg.eigvalsh(inverse_sqrt_arr)))
        self.assertLessEqual(max_eigenvalue_float, 1.0 / np.sqrt(1e-3) + 1e-9)

    def test_single_asset_trend_signal_matches_manual_recursion(self):
        date_index = pd.date_range("2024-01-02", periods=4, freq="B")
        signal_close_ser = pd.Series([100.0, 110.0, 99.0, 138.6], index=date_index, dtype=float)

        signal_df = _compute_single_asset_trend_signal_df(signal_close_ser=signal_close_ser, eta_float=0.25)

        self.assertAlmostEqual(float(signal_df.loc[date_index[1], "return_ser"]), 0.10, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[1], "signal_vol_ser"]), 0.10, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[2], "normalized_return_ser"]), -1.0, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[2], "trend_signal_ser"]), -0.5, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "normalized_return_ser"]), 4.0, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "trend_signal_ser"]), 1.625, places=12)

    def test_compute_arp_core9_signal_tables_keeps_weights_non_negative_and_residual_to_bil(self):
        pricing_data_df = self.make_pricing_data_df()
        signal_close_df = pricing_data_df.xs("SignalClose", axis=1, level=1)[
            ["SPY", "VEA", "VWO", "IEF", "TLT", "GLD", "DBC", "VNQ", "BIL"]
        ]
        config = self.make_config()

        _, _, _, _, _, _, target_weight_df = compute_arp_core9_signal_tables(
            signal_close_df=signal_close_df,
            config=config,
        )

        target_weight_ser = target_weight_df.iloc[-1]
        risk_weight_sum_float = float(target_weight_ser.loc[list(config.risk_asset_list)].sum())

        self.assertTrue((target_weight_ser >= -1e-12).all())
        self.assertLessEqual(float(target_weight_ser.sum()), 1.0 + 1e-12)
        self.assertAlmostEqual(
            float(target_weight_ser.loc["BIL"]),
            1.0 - risk_weight_sum_float,
            places=12,
        )
        self.assertGreater(risk_weight_sum_float, 0.0)

    def test_compute_arp_core9_signal_tables_sends_all_weight_to_bil_when_all_trends_are_negative(self):
        pricing_data_df = self.make_pricing_data_df(downward_risk_bool=True)
        signal_close_df = pricing_data_df.xs("SignalClose", axis=1, level=1)[
            ["SPY", "VEA", "VWO", "IEF", "TLT", "GLD", "DBC", "VNQ", "BIL"]
        ]
        config = self.make_config(corr_lookback_week_int=999, corr_min_period_week_int=999)

        _, _, _, _, _, _, target_weight_df = compute_arp_core9_signal_tables(
            signal_close_df=signal_close_df,
            config=config,
        )

        target_weight_ser = target_weight_df.iloc[-1]
        risk_weight_sum_float = float(target_weight_ser.loc[list(config.risk_asset_list)].sum())

        self.assertAlmostEqual(float(target_weight_ser.loc["BIL"]), 1.0, places=12)
        self.assertAlmostEqual(risk_weight_sum_float, 0.0, places=12)

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("SPY", "return_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "signal_vol_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "normalized_return_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "trend_signal_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "raw_arp_exposure_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "smoothed_arp_exposure_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "target_weight_ser"), signal_data_df.columns)
        self.assertIn(("BIL", "target_weight_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_submits_no_orders_when_target_shares_are_unchanged(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-02-09")
        strategy.current_bar = pd.Timestamp("2024-02-12")
        strategy.add_transaction(10, strategy.previous_bar, "SPY", 200, 100.0, 20_000.0, 1, 0.0)
        strategy.add_transaction(11, strategy.previous_bar, "BIL", 1600, 50.0, 80_000.0, 2, 0.0)
        strategy.current_trade_map["SPY"] = 10
        strategy.current_trade_map["BIL"] = 11

        close_row_ser = self.make_close_row_ser(
            {
                ("SPY", "target_weight_ser"): 0.20,
                ("BIL", "target_weight_ser"): 0.80,
            }
        )
        open_price_ser = pd.Series({"SPY": 100.0, "BIL": 50.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_liquidates_reductions_before_increases_and_reuses_trade_ids(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-02-09")
        strategy.current_bar = pd.Timestamp("2024-02-12")
        strategy.trade_id_int = 20

        strategy.add_transaction(7, strategy.previous_bar, "SPY", 400, 100.0, 40_000.0, 1, 0.0)
        strategy.add_transaction(8, strategy.previous_bar, "VEA", 100, 50.0, 5_000.0, 2, 0.0)
        strategy.current_trade_map["SPY"] = 7
        strategy.current_trade_map["VEA"] = 8

        close_row_ser = self.make_close_row_ser(
            {
                ("SPY", "target_weight_ser"): 0.20,
                ("VEA", "target_weight_ser"): 0.30,
                ("BIL", "target_weight_ser"): 0.50,
            }
        )
        open_price_ser = pd.Series({"SPY": 100.0, "VEA": 50.0, "BIL": 25.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual([order.asset for order in order_list], ["SPY", "VEA", "BIL"])

        spy_order = order_list[0]
        self.assertIsInstance(spy_order, MarketOrder)
        self.assertTrue(spy_order.target)
        self.assertEqual(spy_order.unit, "shares")
        self.assertEqual(spy_order.amount, 200)
        self.assertEqual(spy_order.trade_id, 7)

        vea_order = order_list[1]
        self.assertEqual(vea_order.amount, 600)
        self.assertEqual(vea_order.trade_id, 8)

        bil_order = order_list[2]
        self.assertEqual(bil_order.amount, 2000)
        self.assertEqual(bil_order.trade_id, 21)

    def test_run_daily_smoke_generates_summary(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()
        calendar_idx = pricing_data_df.index[10:]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in scalar divide",
                category=RuntimeWarning,
            )
            run_daily(
                strategy,
                pricing_data_df,
                calendar=calendar_idx,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=None,
            )

        self.assertIsNotNone(strategy.summary)
        self.assertIn("Strategy", strategy.summary.columns)
        self.assertGreater(len(strategy.results), 0)
        self.assertGreater(len(strategy.get_transactions()), 0)


if __name__ == "__main__":
    unittest.main()
