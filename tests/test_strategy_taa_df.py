import os
import unittest
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / '.tmp_norgatedata'
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault('NORGATEDATA_ROOT', str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.taa_df.strategy_taa_df import DefenseFirstStrategy, map_month_end_weights_to_rebalance_open_df


class DefenseFirstStrategyTests(unittest.TestCase):
    def make_strategy(self, rebalance_weight_df: pd.DataFrame, tradeable_asset_list: list[str]) -> DefenseFirstStrategy:
        return DefenseFirstStrategy(
            name='DefenseFirstTest',
            benchmarks=[],
            rebalance_weight_df=rebalance_weight_df,
            tradeable_asset_list=tradeable_asset_list,
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_map_month_end_weights_to_rebalance_open_df_uses_first_trading_day_of_next_month(self):
        month_end_weight_df = pd.DataFrame(
            {'GLD': [0.40, 0.20], 'SPY': [0.60, 0.80]},
            index=pd.to_datetime(['2024-01-31', '2024-02-29']),
        )
        execution_index = pd.to_datetime(['2024-02-01', '2024-02-02', '2024-03-01', '2024-03-04'])

        rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(month_end_weight_df, execution_index)

        expected_weight_df = pd.DataFrame(
            {'GLD': [0.40, 0.20], 'SPY': [0.60, 0.80]},
            index=pd.to_datetime(['2024-02-01', '2024-03-01']),
        )
        expected_weight_df.index.name = 'rebalance_date'

        pd.testing.assert_frame_equal(rebalance_weight_df, expected_weight_df)

    def test_iterate_submits_no_orders_on_non_rebalance_dates(self):
        rebalance_weight_df = pd.DataFrame(
            {'GLD': [0.25], 'SPY': [0.75]},
            index=pd.to_datetime(['2024-02-01']),
        )
        strategy = self.make_strategy(rebalance_weight_df, ['GLD', 'SPY'])
        strategy.current_bar = pd.Timestamp('2024-02-02')
        strategy.previous_bar = pd.Timestamp('2024-02-01')

        open_price_ser = pd.Series({'GLD': 100.0, 'SPY': 200.0})

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), open_price_ser)

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_uses_generic_target_helpers_on_rebalance_dates(self):
        rebalance_date = pd.Timestamp('2024-02-01')
        rebalance_weight_df = pd.DataFrame(
            {'GLD': [0.25], 'UUP': [0.0], 'SPY': [0.75], 'DBC': [0.0]},
            index=pd.to_datetime([rebalance_date]),
        )
        strategy = self.make_strategy(rebalance_weight_df, ['GLD', 'UUP', 'SPY', 'DBC'])
        strategy.current_bar = rebalance_date
        strategy.previous_bar = pd.Timestamp('2024-01-31')
        strategy.trade_id_int = 22

        strategy.add_transaction(21, strategy.previous_bar, 'GLD', 10, 100.0, 1000.0, 1, 0.0)
        strategy.add_transaction(22, strategy.previous_bar, 'UUP', 5, 50.0, 250.0, 2, 0.0)
        strategy.current_trade_map['GLD'] = 21
        strategy.current_trade_map['UUP'] = 22

        open_price_ser = pd.Series({'GLD': 100.0, 'UUP': 50.0, 'SPY': 200.0, 'DBC': 25.0})

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 3)
        self.assertEqual([order.asset for order in order_list], ['UUP', 'GLD', 'SPY'])

        liquidation_order = order_list[0]
        self.assertIsInstance(liquidation_order, MarketOrder)
        self.assertEqual(liquidation_order.unit, 'shares')
        self.assertTrue(liquidation_order.target)
        self.assertEqual(liquidation_order.amount, 0)
        self.assertEqual(liquidation_order.trade_id, 22)

        resize_order = order_list[1]
        self.assertIsInstance(resize_order, MarketOrder)
        self.assertEqual(resize_order.unit, 'percent')
        self.assertTrue(resize_order.target)
        self.assertEqual(resize_order.amount, 0.25)
        self.assertEqual(resize_order.trade_id, 21)

        new_position_order = order_list[2]
        self.assertIsInstance(new_position_order, MarketOrder)
        self.assertEqual(new_position_order.unit, 'percent')
        self.assertTrue(new_position_order.target)
        self.assertEqual(new_position_order.amount, 0.75)
        self.assertEqual(new_position_order.trade_id, 23)

        self.assertEqual(strategy.current_trade_map['SPY'], 23)
        self.assertEqual(strategy.trade_id_int, 23)
        self.assertNotIn('DBC', [order.asset for order in order_list])


if __name__ == '__main__':
    unittest.main()


