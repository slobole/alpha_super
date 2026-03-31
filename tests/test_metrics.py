import unittest

import pandas as pd

from alpha.engine.metrics import generate_trades


class GenerateTradesTests(unittest.TestCase):
    def test_generate_trades_preserves_long_trade_return_math(self):
        transaction_df = pd.DataFrame(
            [
                {
                    "trade_id": 1,
                    "bar": pd.Timestamp("2024-01-02"),
                    "asset": "AAA",
                    "amount": 10,
                    "price": 100.0,
                    "total_value": 1_000.0,
                    "order_id": 1,
                    "commission": 0.0,
                },
                {
                    "trade_id": 1,
                    "bar": pd.Timestamp("2024-01-10"),
                    "asset": "AAA",
                    "amount": -10,
                    "price": 110.0,
                    "total_value": -1_100.0,
                    "order_id": 2,
                    "commission": 0.0,
                },
            ]
        )

        trade_df = generate_trades(transaction_df)

        self.assertAlmostEqual(float(trade_df.loc[1, "capital"]), 1_000.0)
        self.assertAlmostEqual(float(trade_df.loc[1, "profit"]), 100.0)
        self.assertAlmostEqual(float(trade_df.loc[1, "return"]), 0.10)

    def test_generate_trades_uses_absolute_entry_notional_for_short_trade(self):
        transaction_df = pd.DataFrame(
            [
                {
                    "trade_id": 2,
                    "bar": pd.Timestamp("2024-01-02"),
                    "asset": "BBB",
                    "amount": -10,
                    "price": 100.0,
                    "total_value": -1_000.0,
                    "order_id": 1,
                    "commission": 0.0,
                },
                {
                    "trade_id": 2,
                    "bar": pd.Timestamp("2024-01-10"),
                    "asset": "BBB",
                    "amount": 10,
                    "price": 90.0,
                    "total_value": 900.0,
                    "order_id": 2,
                    "commission": 0.0,
                },
            ]
        )

        trade_df = generate_trades(transaction_df)

        self.assertAlmostEqual(float(trade_df.loc[2, "capital"]), 1_000.0)
        self.assertAlmostEqual(float(trade_df.loc[2, "profit"]), 100.0)
        self.assertAlmostEqual(float(trade_df.loc[2, "return"]), 0.10)


if __name__ == "__main__":
    unittest.main()
