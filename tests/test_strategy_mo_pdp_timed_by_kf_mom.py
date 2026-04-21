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
from strategies.momentum.strategy_mo_pdp_timed_by_kf_mom import (
    DEFAULT_KENNETH_FRENCH_CACHE_ZIP_PATH_STR,
    PdpTimedByKfMomConfig,
    PdpTimedByKfMomStrategy,
    SUPPORTED_SHARED_START_DATE_STR,
    compute_kf_mom_proxy_signal_df,
    get_pdp_timed_by_kf_mom_data,
)


class PdpTimedByKfMomStrategyTests(unittest.TestCase):
    def make_config(self, **kwargs) -> PdpTimedByKfMomConfig:
        base_kwargs = dict(
            trade_symbol_str="PDP",
            signal_symbol_str="KF_MOM",
            benchmark_list=("$SPX",),
            start_date_str=SUPPORTED_SHARED_START_DATE_STR,
            end_date_str="2007-03-30",
            signal_return_lookback_day_int=10,
            signal_smoothing_day_int=5,
            oversold_rank_threshold_float=0.50,
            kenneth_french_cache_zip_path_str=DEFAULT_KENNETH_FRENCH_CACHE_ZIP_PATH_STR,
            capital_base_float=100_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )
        base_kwargs.update(kwargs)
        return PdpTimedByKfMomConfig(**base_kwargs)

    def make_strategy(self, **kwargs) -> PdpTimedByKfMomStrategy:
        base_kwargs = dict(
            name="PdpTimedByKfMomTest",
            benchmarks=["$SPX"],
            trade_symbol_str="PDP",
            signal_symbol_str="KF_MOM",
            signal_return_lookback_day_int=10,
            signal_smoothing_day_int=5,
            oversold_rank_threshold_float=0.50,
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return PdpTimedByKfMomStrategy(**base_kwargs)

    def make_factor_return_ser(self) -> pd.Series:
        date_index = pd.date_range("2024-01-02", periods=20, freq="B")
        factor_return_vec = np.array(
            [
                0.0100, -0.0040, 0.0060, -0.0030, 0.0110,
                0.0050, -0.0080, 0.0070, -0.0020, 0.0040,
                -0.0150, -0.0100, -0.0050, -0.0060, -0.0040,
                0.0030, 0.0080, 0.0120, 0.0090, 0.0100,
            ],
            dtype=float,
        )
        return pd.Series(factor_return_vec, index=date_index, dtype=float, name="KF_MOM")

    def make_pricing_data_df(self) -> pd.DataFrame:
        factor_return_ser = self.make_factor_return_ser()
        date_index = factor_return_ser.index
        step_vec = np.arange(len(date_index), dtype=float)

        trade_close_vec = 30.0 + 0.60 * step_vec + 0.75 * np.sin(step_vec * 0.30)
        benchmark_close_vec = 4_000.0 + 5.0 * step_vec + 12.0 * np.cos(step_vec * 0.10)

        pricing_data_df = pd.DataFrame(
            {
                ("PDP", "Open"): trade_close_vec - 0.4,
                ("PDP", "High"): trade_close_vec + 0.8,
                ("PDP", "Low"): trade_close_vec - 0.9,
                ("PDP", "Close"): trade_close_vec,
                ("KF_MOM", "SignalReturn"): factor_return_ser.to_numpy(dtype=float),
                ("$SPX", "Open"): benchmark_close_vec - 5.0,
                ("$SPX", "High"): benchmark_close_vec + 10.0,
                ("$SPX", "Low"): benchmark_close_vec - 10.0,
                ("$SPX", "Close"): benchmark_close_vec,
            },
            index=date_index,
            dtype=float,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map, dtype=float)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_compute_kf_mom_proxy_signal_df_matches_explicit_formula(self):
        config = self.make_config()
        factor_return_ser = self.make_factor_return_ser()

        signal_df = compute_kf_mom_proxy_signal_df(
            factor_return_ser=factor_return_ser,
            config=config,
        )

        expected_factor_index_float = float((1.0 + factor_return_ser).prod() * 100.0)
        expected_proxy_return_10d_float = float(
            np.prod(1.0 + factor_return_ser.iloc[-config.signal_return_lookback_day_int:]) - 1.0
        )

        trailing_proxy_return_list: list[float] = []
        for position_int in range(
            len(factor_return_ser) - config.signal_smoothing_day_int,
            len(factor_return_ser),
        ):
            trailing_proxy_return_float = float(
                np.prod(
                    1.0
                    + factor_return_ser.iloc[
                        position_int - config.signal_return_lookback_day_int + 1: position_int + 1
                    ]
                ) - 1.0
            )
            trailing_proxy_return_list.append(trailing_proxy_return_float)
        expected_proxy_return_10d_sma5_float = float(np.mean(trailing_proxy_return_list))

        historical_proxy_smooth_value_list: list[float] = []
        first_valid_position_int = (
            config.signal_return_lookback_day_int
            + config.signal_smoothing_day_int
            - 1
        )
        for position_int in range(first_valid_position_int, len(factor_return_ser)):
            rolling_proxy_return_list: list[float] = []
            for smooth_position_int in range(
                position_int - config.signal_smoothing_day_int + 1,
                position_int + 1,
            ):
                proxy_return_float = float(
                    np.prod(
                        1.0
                        + factor_return_ser.iloc[
                            smooth_position_int - config.signal_return_lookback_day_int + 1: smooth_position_int + 1
                        ]
                    ) - 1.0
                )
                rolling_proxy_return_list.append(proxy_return_float)
            proxy_smooth_float = float(np.mean(rolling_proxy_return_list))
            historical_proxy_smooth_value_list.append(proxy_smooth_float)

        historical_proxy_smooth_vec = np.asarray(historical_proxy_smooth_value_list, dtype=float)
        expected_proxy_rank_pct_float = float(
            np.mean(historical_proxy_smooth_vec <= expected_proxy_return_10d_sma5_float)
        )
        expected_target_weight_float = float(
            expected_proxy_rank_pct_float < config.oversold_rank_threshold_float
        )

        self.assertAlmostEqual(
            float(signal_df.iloc[-1]["factor_index_ser"]),
            expected_factor_index_float,
            places=12,
        )
        self.assertAlmostEqual(
            float(signal_df.iloc[-1]["proxy_return_10d_ser"]),
            expected_proxy_return_10d_float,
            places=12,
        )
        self.assertAlmostEqual(
            float(signal_df.iloc[-1]["proxy_return_10d_sma5_ser"]),
            expected_proxy_return_10d_sma5_float,
            places=12,
        )
        self.assertAlmostEqual(
            float(signal_df.iloc[-1]["proxy_rank_pct_ser"]),
            expected_proxy_rank_pct_float,
            places=12,
        )
        self.assertAlmostEqual(
            float(signal_df.iloc[-1]["target_weight_ser"]),
            expected_target_weight_float,
            places=12,
        )

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("KF_MOM", "factor_index_ser"), signal_data_df.columns)
        self.assertIn(("KF_MOM", "proxy_return_10d_ser"), signal_data_df.columns)
        self.assertIn(("KF_MOM", "proxy_return_10d_sma5_ser"), signal_data_df.columns)
        self.assertIn(("KF_MOM", "proxy_rank_pct_ser"), signal_data_df.columns)
        self.assertIn(("PDP", "target_weight_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_submits_no_order_when_target_shares_match_current_position(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-22")
        strategy.current_bar = pd.Timestamp("2024-01-23")
        strategy.trade_id_int = 7
        strategy.current_trade_id_int = 7
        strategy.add_transaction(7, strategy.previous_bar, "PDP", 1_000, 100.0, 100_000.0, 1, 0.0)

        close_row_ser = self.make_close_row_ser({("PDP", "target_weight_ser"): 1.0})
        open_price_ser = pd.Series({"PDP": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_enters_when_proxy_is_oversold(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-22")
        strategy.current_bar = pd.Timestamp("2024-01-23")

        close_row_ser = self.make_close_row_ser({("PDP", "target_weight_ser"): 1.0})
        open_price_ser = pd.Series({"PDP": 125.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "PDP")
        self.assertEqual(entry_order.unit, "shares")
        self.assertTrue(entry_order.target)
        self.assertEqual(entry_order.amount, 800)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_exits_when_proxy_is_not_oversold(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-22")
        strategy.current_bar = pd.Timestamp("2024-01-23")
        strategy.trade_id_int = 5
        strategy.current_trade_id_int = 5
        strategy.add_transaction(5, strategy.previous_bar, "PDP", 250, 100.0, 25_000.0, 1, 0.0)

        close_row_ser = self.make_close_row_ser({("PDP", "target_weight_ser"): 0.0})
        open_price_ser = pd.Series({"PDP": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, "PDP")
        self.assertEqual(exit_order.unit, "shares")
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 5)
        self.assertEqual(strategy.current_trade_id_int, -1)

    def test_get_pdp_timed_by_kf_mom_data_rejects_requested_start_outside_supported_overlap(self):
        early_config = self.make_config(start_date_str="2007-02-28")

        with self.assertRaisesRegex(ValueError, SUPPORTED_SHARED_START_DATE_STR):
            get_pdp_timed_by_kf_mom_data(config=early_config)

    def test_run_daily_smoke_generates_summary_and_transactions(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()
        calendar_idx = pricing_data_df.index[15:]

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
