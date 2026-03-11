import unittest

import pandas as pd

from alpha.engine.portfolio import Portfolio
from alpha.engine.strategy import Strategy


class DummyStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


def make_strategy(name: str, dates_index: pd.DatetimeIndex, daily_returns_list: list[float], capital_base: float = 100.0):
    strategy = DummyStrategy(
        name=name,
        benchmarks=[],
        capital_base=capital_base,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    daily_returns_ser = pd.Series(daily_returns_list, index=dates_index, dtype=float)
    total_value_ser = capital_base * (1 + daily_returns_ser).cumprod()
    strategy.results = pd.DataFrame({
        'daily_returns': daily_returns_ser,
        'total_value': total_value_ser,
        'portfolio_value': total_value_ser,
    }, index=dates_index)
    strategy.summary = pd.DataFrame()
    strategy.summary_trades = pd.DataFrame()
    return strategy


class PortfolioTests(unittest.TestCase):
    def test_no_rebalance_compounds_pods_independently(self):
        dates_index = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-03', '2024-02-04'])
        strategy_a = make_strategy('StrategyA', dates_index, [0.0, 1.0, 0.0, 0.0])
        strategy_b = make_strategy('StrategyB', dates_index, [0.0, 0.0, 0.0, 1.0])

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        self.assertAlmostEqual(portfolio.results.iloc[-1]['total_value'], 200.0)

    def test_monthly_rebalance_redistributes_capital(self):
        dates_index = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-03', '2024-02-04'])
        strategy_a = make_strategy('StrategyA', dates_index, [0.0, 1.0, 0.0, 0.0])
        strategy_b = make_strategy('StrategyB', dates_index, [0.0, 0.0, 0.0, 1.0])

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
            rebalance='monthly',
        )

        self.assertAlmostEqual(portfolio.results.iloc[-1]['total_value'], 225.0)

    def test_common_date_range_uses_overlap_only(self):
        strategy_a = make_strategy(
            'StrategyA',
            pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            [0.0, 0.01, 0.02],
        )
        strategy_b = make_strategy(
            'StrategyB',
            pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04']),
            [0.0, -0.01, 0.01],
        )

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        self.assertEqual(list(portfolio.results.index), list(pd.to_datetime(['2024-01-02', '2024-01-03'])))

    def test_weight_sum_validation(self):
        dates_index = pd.to_datetime(['2024-01-01', '2024-01-02'])
        strategy_a = make_strategy('StrategyA', dates_index, [0.0, 0.01])
        strategy_b = make_strategy('StrategyB', dates_index, [0.0, 0.01])

        with self.assertRaisesRegex(ValueError, 'Weights must sum to 1.0'):
            Portfolio(
                strategies=[strategy_a, strategy_b],
                weights=[0.6, 0.3],
                capital_base=100.0,
            )
