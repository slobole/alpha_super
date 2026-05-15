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
from strategies.momentum.strategy_mo_spy_sphb_splv_canary import (
    SphbSplvCanaryStrategy,
    build_backtest_calendar_idx,
    compute_sphb_splv_canary_signal_df,
)
from strategies.momentum.strategy_mo_spy_sphb_splv_canary_btal import (
    DEFAULT_CONFIG as BTAL_CANARY_CONFIG,
)
from strategies.momentum.strategy_mo_spy_sphb_splv_canary_iau import (
    DEFAULT_CONFIG as IAU_CANARY_CONFIG,
)
from strategies.momentum.strategy_mo_spy_sphb_splv_canary_tlt import (
    DEFAULT_CONFIG as TLT_CANARY_CONFIG,
)
from strategies.momentum.strategy_mo_sso_sphb_splv_canary import (
    DEFAULT_CONFIG as SSO_CASH_CANARY_CONFIG,
)
from strategies.momentum.strategy_mo_sso_sphb_splv_canary_iau import (
    DEFAULT_CONFIG as SSO_IAU_CANARY_CONFIG,
)
from strategies.momentum.strategy_mo_upro_sphb_splv_canary_iau import (
    DEFAULT_CONFIG as UPRO_IAU_CANARY_CONFIG,
)


class SphbSplvCanaryStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> SphbSplvCanaryStrategy:
        base_kwargs = dict(
            name="SphbSplvCanaryTest",
            benchmarks=["$SPXTR"],
            trade_symbol_str="SPY",
            high_beta_symbol_str="SPHB",
            low_vol_symbol_str="SPLV",
            sma_lookback_day_int=3,
            confirmation_day_int=2,
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return SphbSplvCanaryStrategy(**base_kwargs)

    def make_pricing_data_df(self, include_btal_bool: bool = False) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=9, freq="B")
        canary_ratio_vec = np.array([1.0, 1.0, 1.0, 1.1, 1.2, 0.9, 1.2, 1.3, 0.8], dtype=float)
        low_vol_close_vec = np.full(len(date_index), 100.0, dtype=float)
        high_beta_close_vec = low_vol_close_vec * canary_ratio_vec
        spy_close_vec = np.array([50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0], dtype=float)
        btal_close_vec = np.array([20.0, 20.5, 20.3, 20.4, 20.8, 21.0, 20.7, 20.9, 21.1], dtype=float)
        benchmark_close_vec = np.linspace(4_000.0, 4_080.0, len(date_index))

        price_col_map = {
            ("SPY", "Open"): spy_close_vec - 0.5,
            ("SPY", "High"): spy_close_vec + 1.0,
            ("SPY", "Low"): spy_close_vec - 1.0,
            ("SPY", "Close"): spy_close_vec,
            ("SPHB", "Open"): high_beta_close_vec - 0.5,
            ("SPHB", "High"): high_beta_close_vec + 1.0,
            ("SPHB", "Low"): high_beta_close_vec - 1.0,
            ("SPHB", "Close"): high_beta_close_vec,
            ("SPLV", "Open"): low_vol_close_vec - 0.5,
            ("SPLV", "High"): low_vol_close_vec + 1.0,
            ("SPLV", "Low"): low_vol_close_vec - 1.0,
            ("SPLV", "Close"): low_vol_close_vec,
            ("$SPXTR", "Open"): benchmark_close_vec - 5.0,
            ("$SPXTR", "High"): benchmark_close_vec + 10.0,
            ("$SPXTR", "Low"): benchmark_close_vec - 10.0,
            ("$SPXTR", "Close"): benchmark_close_vec,
        }
        if include_btal_bool:
            price_col_map.update(
                {
                    ("BTAL", "Open"): btal_close_vec - 0.1,
                    ("BTAL", "High"): btal_close_vec + 0.2,
                    ("BTAL", "Low"): btal_close_vec - 0.2,
                    ("BTAL", "Close"): btal_close_vec,
                }
            )

        pricing_data_df = pd.DataFrame(price_col_map, index=date_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, target_weight_float: float) -> pd.Series:
        close_row_ser = pd.Series({("SPY", "target_weight_ser"): target_weight_float}, dtype=float)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_compute_sphb_splv_canary_signal_df_matches_formula(self):
        pricing_data_df = self.make_pricing_data_df()
        signal_df = compute_sphb_splv_canary_signal_df(
            high_beta_close_ser=pricing_data_df[("SPHB", "Close")],
            low_vol_close_ser=pricing_data_df[("SPLV", "Close")],
            sma_lookback_day_int=3,
            confirmation_day_int=2,
        )
        date_index = pricing_data_df.index

        expected_sma_float = float(np.mean([1.0, 1.1, 1.2]))
        expected_momentum_float = 1.2 / expected_sma_float - 1.0

        self.assertAlmostEqual(float(signal_df.loc[date_index[4], "canary_ratio_ser"]), 1.2, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[4], "canary_sma_ser"]), expected_sma_float, places=12)
        self.assertAlmostEqual(
            float(signal_df.loc[date_index[4], "canary_momentum_ser"]),
            expected_momentum_float,
            places=12,
        )
        self.assertTrue(pd.isna(signal_df.loc[date_index[2], "target_weight_ser"]))
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "target_weight_ser"]), 0.0, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[4], "target_weight_ser"]), 1.0, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[5], "target_weight_ser"]), 0.0, places=12)

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("SPHB", "canary_ratio_ser"), signal_data_df.columns)
        self.assertIn(("SPHB", "canary_sma_ser"), signal_data_df.columns)
        self.assertIn(("SPHB", "canary_momentum_ser"), signal_data_df.columns)
        self.assertIn(("SPHB", "positive_momentum_ser"), signal_data_df.columns)
        self.assertIn(("SPHB", "confirmation_count_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "target_weight_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_enters_spy_when_canary_is_confirmed(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-08")
        strategy.current_bar = pd.Timestamp("2024-01-09")

        close_row_ser = self.make_close_row_ser(1.0)
        open_price_ser = pd.Series({"SPY": 60.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "SPY")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertEqual(entry_order.amount, 1.0)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_exits_spy_when_canary_turns_off(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-08")
        strategy.current_bar = pd.Timestamp("2024-01-09")
        strategy.trade_id_int = 3
        strategy.current_trade_id_int = 3
        strategy.add_transaction(3, strategy.previous_bar, "SPY", 100, 50.0, 5_000.0, 1, 0.0)

        close_row_ser = self.make_close_row_ser(0.0)
        open_price_ser = pd.Series({"SPY": 60.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, "SPY")
        self.assertEqual(exit_order.unit, "shares")
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 3)
        self.assertEqual(strategy.current_trade_id_int, -1)

    def test_btal_wrapper_defaults_to_btal_risk_off_asset(self):
        self.assertEqual(BTAL_CANARY_CONFIG.risk_off_symbol_str, "BTAL")

    def test_iau_wrapper_defaults_to_iau_risk_off_asset(self):
        self.assertEqual(IAU_CANARY_CONFIG.risk_off_symbol_str, "IAU")

    def test_tlt_wrapper_defaults_to_tlt_risk_off_asset(self):
        self.assertEqual(TLT_CANARY_CONFIG.risk_off_symbol_str, "TLT")

    def test_sso_cash_wrapper_sets_sso_risk_on_and_no_risk_off_asset(self):
        self.assertEqual(SSO_CASH_CANARY_CONFIG.trade_symbol_str, "SSO")
        self.assertIsNone(SSO_CASH_CANARY_CONFIG.risk_off_symbol_str)

    def test_sso_iau_wrapper_sets_sso_risk_on_and_iau_risk_off(self):
        self.assertEqual(SSO_IAU_CANARY_CONFIG.trade_symbol_str, "SSO")
        self.assertEqual(SSO_IAU_CANARY_CONFIG.risk_off_symbol_str, "IAU")

    def test_upro_iau_wrapper_sets_upro_risk_on_and_iau_risk_off(self):
        self.assertEqual(UPRO_IAU_CANARY_CONFIG.trade_symbol_str, "UPRO")
        self.assertEqual(UPRO_IAU_CANARY_CONFIG.risk_off_symbol_str, "IAU")

    def test_iterate_enters_btal_when_canary_is_off_and_risk_off_asset_is_set(self):
        strategy = self.make_strategy(risk_off_symbol_str="BTAL")
        strategy.previous_bar = pd.Timestamp("2024-01-08")
        strategy.current_bar = pd.Timestamp("2024-01-09")

        close_row_ser = self.make_close_row_ser(0.0)
        open_price_ser = pd.Series({"SPY": 60.0, "BTAL": 20.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "BTAL")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertEqual(entry_order.amount, 1.0)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_switches_from_btal_to_spy_when_canary_turns_on(self):
        strategy = self.make_strategy(risk_off_symbol_str="BTAL")
        strategy.previous_bar = pd.Timestamp("2024-01-08")
        strategy.current_bar = pd.Timestamp("2024-01-09")
        strategy.trade_id_int = 5
        strategy.current_trade_id_map["BTAL"] = 5
        strategy.add_transaction(5, strategy.previous_bar, "BTAL", 1_000, 20.0, 20_000.0, 1, 0.0)

        close_row_ser = self.make_close_row_ser(1.0)
        open_price_ser = pd.Series({"SPY": 60.0, "BTAL": 20.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 2)
        exit_order = order_list[0]
        entry_order = order_list[1]
        self.assertEqual(exit_order.asset, "BTAL")
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 5)
        self.assertEqual(entry_order.asset, "SPY")
        self.assertEqual(entry_order.amount, 1.0)
        self.assertEqual(entry_order.trade_id, 6)

    def test_run_daily_smoke_generates_summary_and_transactions(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()
        signal_feature_df = compute_sphb_splv_canary_signal_df(
            high_beta_close_ser=pricing_data_df[("SPHB", "Close")],
            low_vol_close_ser=pricing_data_df[("SPLV", "Close")],
            sma_lookback_day_int=3,
            confirmation_day_int=2,
        )
        calendar_idx = build_backtest_calendar_idx(
            pricing_data_df=pricing_data_df,
            signal_feature_df=signal_feature_df,
        )

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

    def test_run_daily_btal_risk_off_smoke_generates_both_assets(self):
        strategy = self.make_strategy(risk_off_symbol_str="BTAL")
        pricing_data_df = self.make_pricing_data_df(include_btal_bool=True)
        signal_feature_df = compute_sphb_splv_canary_signal_df(
            high_beta_close_ser=pricing_data_df[("SPHB", "Close")],
            low_vol_close_ser=pricing_data_df[("SPLV", "Close")],
            sma_lookback_day_int=3,
            confirmation_day_int=2,
        )
        calendar_idx = build_backtest_calendar_idx(
            pricing_data_df=pricing_data_df,
            signal_feature_df=signal_feature_df,
        )

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

        traded_asset_set = set(strategy.get_transactions()["asset"].astype(str))
        self.assertIn("SPY", traded_asset_set)
        self.assertIn("BTAL", traded_asset_set)


if __name__ == "__main__":
    unittest.main()
