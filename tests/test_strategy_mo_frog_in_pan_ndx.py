import os
import unittest
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.strategy_mo_frog_in_pan import FrogInPanStrategy
from strategies.strategy_mo_frog_in_pan_ndx import DEFAULT_CONFIG


class FrogInPanNdxPresetTests(unittest.TestCase):
    def test_default_config_uses_ndx_universe_and_qqq_regime(self):
        self.assertEqual(DEFAULT_CONFIG.indexname_str, "Nasdaq 100")
        self.assertEqual(DEFAULT_CONFIG.regime_symbol_str, "QQQ")
        self.assertEqual(DEFAULT_CONFIG.benchmark_symbol_list, ("QQQ",))
        self.assertEqual(DEFAULT_CONFIG.max_holdings_int, 10)

    def test_strategy_can_be_built_from_ndx_preset(self):
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-28")]},
            index=pd.to_datetime(["2024-04-01"]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"

        strategy = FrogInPanStrategy(
            name="FrogInPanNdxPresetTest",
            benchmarks=DEFAULT_CONFIG.benchmark_symbol_list,
            rebalance_schedule_df=rebalance_schedule_df,
            regime_symbol_str=DEFAULT_CONFIG.regime_symbol_str,
            capital_base=DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            momentum_lookback_month_int=DEFAULT_CONFIG.momentum_lookback_month_int,
            momentum_skip_month_int=DEFAULT_CONFIG.momentum_skip_month_int,
            regime_momentum_lookback_month_int=DEFAULT_CONFIG.regime_momentum_lookback_month_int,
            regime_momentum_skip_month_int=DEFAULT_CONFIG.regime_momentum_skip_month_int,
            rebalance_month_interval_int=DEFAULT_CONFIG.rebalance_month_interval_int,
            top_momentum_quantile_float=DEFAULT_CONFIG.top_momentum_quantile_float,
            top_fip_fraction_float=DEFAULT_CONFIG.top_fip_fraction_float,
            max_holdings_int=DEFAULT_CONFIG.max_holdings_int,
        )

        self.assertEqual(strategy.regime_symbol_str, "QQQ")
        self.assertEqual(strategy.max_holdings_int, 10)


if __name__ == "__main__":
    unittest.main()
