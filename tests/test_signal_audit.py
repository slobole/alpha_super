import unittest

import pandas as pd

from alpha.engine.strategy import Strategy


class SafeFeatureStrategy(Strategy):
    signal_audit_sample_size = 5
    enable_signal_audit = True

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        signal_data[('TEST', 'lagged_close')] = signal_data[('TEST', 'Close')].shift(1)
        return signal_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


class LeakyFeatureStrategy(Strategy):
    signal_audit_sample_size = 5
    enable_signal_audit = True

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        signal_data[('TEST', 'future_close')] = signal_data[('TEST', 'Close')].shift(-1)
        return signal_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


class SignalAuditTests(unittest.TestCase):
    def make_pricing_data(self):
        dates_index = pd.date_range('2024-01-01', periods=8, freq='D')
        columns = pd.MultiIndex.from_product([['TEST'], ['Open', 'High', 'Low', 'Close']])
        values = []
        for close in range(10, 18):
            values.append([close - 0.5, close + 0.5, close - 1.0, float(close)])
        return pd.DataFrame(values, index=dates_index, columns=columns)

    def test_audit_accepts_past_only_feature(self):
        strategy = SafeFeatureStrategy(
            name='Safe',
            benchmarks=[],
            commission_per_share=0.0,
            commission_minimum=0.0,
            slippage=0.0,
        )
        pricing_data = self.make_pricing_data()
        signal_data = strategy.compute_signals(pricing_data)

        strategy.audit_signals(pricing_data, signal_data)

    def test_audit_rejects_future_looking_feature(self):
        strategy = LeakyFeatureStrategy(
            name='Leaky',
            benchmarks=[],
            commission_per_share=0.0,
            commission_minimum=0.0,
            slippage=0.0,
        )
        pricing_data = self.make_pricing_data()
        signal_data = strategy.compute_signals(pricing_data)

        with self.assertRaisesRegex(ValueError, 'possible lookahead leakage'):
            strategy.audit_signals(pricing_data, signal_data)


if __name__ == '__main__':
    unittest.main()