import os
import unittest
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.momentum.strategy_mo_radge_ndx import RadgeMomentumNdxStrategy
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    AtrNormalizedNdxStrategy,
    DEFAULT_CONFIG as NDX_DEFAULT_CONFIG,
)
from strategies.momentum.strategy_mo_atr_normalized_sp500 import (
    AtrNormalizedSp500Strategy,
    DEFAULT_CONFIG as SP500_DEFAULT_CONFIG,
)
from strategies.momentum.strategy_mo_atr_normalized_russell1000 import (
    AtrNormalizedRussell1000Strategy,
    DEFAULT_CONFIG as RUSSELL1000_DEFAULT_CONFIG,
)


class AtrNormalizedVariantWrapperTests(unittest.TestCase):
    def make_rebalance_schedule_df(self) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-28")]},
            index=pd.to_datetime(["2024-04-01"]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def test_ndx_default_config_points_to_nasdaq_100(self):
        self.assertEqual(NDX_DEFAULT_CONFIG.indexname_str, "Nasdaq 100")
        self.assertEqual(NDX_DEFAULT_CONFIG.regime_symbol_str, "SPY")
        self.assertEqual(NDX_DEFAULT_CONFIG.max_positions_int, 10)

    def test_sp500_default_config_points_to_sp500(self):
        self.assertEqual(SP500_DEFAULT_CONFIG.indexname_str, "S&P 500")
        self.assertEqual(SP500_DEFAULT_CONFIG.regime_symbol_str, "SPY")
        self.assertEqual(SP500_DEFAULT_CONFIG.max_positions_int, 10)

    def test_russell1000_default_config_points_to_russell1000(self):
        self.assertEqual(RUSSELL1000_DEFAULT_CONFIG.indexname_str, "Russell 1000")
        self.assertEqual(RUSSELL1000_DEFAULT_CONFIG.regime_symbol_str, "SPY")
        self.assertEqual(RUSSELL1000_DEFAULT_CONFIG.max_positions_int, 10)

    def test_ndx_wrapper_strategy_can_be_built(self):
        strategy = AtrNormalizedNdxStrategy(
            name="AtrNormalizedNdxTest",
            benchmarks=[NDX_DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str=NDX_DEFAULT_CONFIG.regime_symbol_str,
            capital_base=NDX_DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            lookback_month_int=NDX_DEFAULT_CONFIG.lookback_month_int,
            index_trend_window_int=NDX_DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=NDX_DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=NDX_DEFAULT_CONFIG.max_positions_int,
        )

        self.assertIsInstance(strategy, RadgeMomentumNdxStrategy)
        self.assertEqual(strategy.max_positions_int, 10)

    def test_sp500_wrapper_strategy_can_be_built(self):
        strategy = AtrNormalizedSp500Strategy(
            name="AtrNormalizedSp500Test",
            benchmarks=[SP500_DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str=SP500_DEFAULT_CONFIG.regime_symbol_str,
            capital_base=SP500_DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            lookback_month_int=SP500_DEFAULT_CONFIG.lookback_month_int,
            index_trend_window_int=SP500_DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=SP500_DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=SP500_DEFAULT_CONFIG.max_positions_int,
        )

        self.assertIsInstance(strategy, RadgeMomentumNdxStrategy)
        self.assertEqual(strategy.max_positions_int, 10)

    def test_russell1000_wrapper_strategy_can_be_built(self):
        strategy = AtrNormalizedRussell1000Strategy(
            name="AtrNormalizedRussell1000Test",
            benchmarks=[RUSSELL1000_DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str=RUSSELL1000_DEFAULT_CONFIG.regime_symbol_str,
            capital_base=RUSSELL1000_DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            lookback_month_int=RUSSELL1000_DEFAULT_CONFIG.lookback_month_int,
            index_trend_window_int=RUSSELL1000_DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=RUSSELL1000_DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=RUSSELL1000_DEFAULT_CONFIG.max_positions_int,
        )

        self.assertIsInstance(strategy, RadgeMomentumNdxStrategy)
        self.assertEqual(strategy.max_positions_int, 10)


if __name__ == "__main__":
    unittest.main()
