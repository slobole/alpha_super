import unittest

import numpy as np
import pandas as pd

from alpha.engine.metrics import generate_overall_metrics, generate_trades


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


class GenerateOverallMetricsTests(unittest.TestCase):
    def test_generate_overall_metrics_adds_l1_mar_and_underwater_metrics(self):
        date_index = pd.date_range("2024-01-01", periods=5, freq="D")
        total_value_ser = pd.Series(
            [100.0, 110.0, 105.0, 115.0, 120.0],
            index=date_index,
            dtype=float,
        )

        summary_ser = generate_overall_metrics(
            total_value_ser,
            capital_base=100.0,
            days_in_year=5,
        )

        daily_return_ser = total_value_ser.pct_change(fill_method=None).dropna()
        drawdown_ser = total_value_ser / total_value_ser.cummax() - 1.0
        max_drawdown_pct_float = float(drawdown_ser.min() * 100)
        annual_return_pct_float = float(((120.0 / 100.0) ** (5 / 5) - 1.0) * 100)

        self.assertAlmostEqual(
            float(summary_ser.loc["AAR [%]"]),
            float(daily_return_ser.abs().mean() * 100),
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Downside L1 [%]"]),
            float(np.maximum(-daily_return_ser, 0.0).mean() * 100),
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Avg. Loss Day [%]"]),
            float((-daily_return_ser[daily_return_ser < 0.0]).mean() * 100),
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Time Under Water [%]"]),
            float((drawdown_ser < 0.0).mean() * 100),
        )
        self.assertAlmostEqual(float(summary_ser.loc["Max. Drawdown [%]"]), max_drawdown_pct_float)
        self.assertAlmostEqual(
            float(summary_ser.loc["MAR Ratio"]),
            annual_return_pct_float / abs(max_drawdown_pct_float),
        )

    def test_generate_overall_metrics_adds_turnover_and_cost_drag_from_transactions(self):
        date_index = pd.date_range("2024-01-01", periods=5, freq="D")
        total_value_ser = pd.Series(
            [10_000.0, 10_010.0, 10_000.0, 10_020.0, 10_030.0],
            index=date_index,
            dtype=float,
        )
        transaction_df = pd.DataFrame(
            [
                {
                    "bar": date_index[1],
                    "amount": 10.0,
                    "total_value": 1_001.0,
                    "commission": 2.0,
                },
                {
                    "bar": date_index[3],
                    "amount": -10.0,
                    "total_value": -999.0,
                    "commission": 3.0,
                },
            ]
        )

        summary_ser = generate_overall_metrics(
            total_value_ser,
            capital_base=10_000.0,
            days_in_year=5,
            transactions_df=transaction_df,
            slippage_float=0.001,
        )

        average_equity_float = float(total_value_ser.mean())
        gross_trade_notional_float = 2_000.0
        expected_slippage_float = 2.0
        expected_total_cost_float = 7.0

        self.assertAlmostEqual(
            float(summary_ser.loc["Turnover (Ann.) [%]"]),
            gross_trade_notional_float / average_equity_float * 100,
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Estimated Slippage [$]"]),
            expected_slippage_float,
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Total Trading Costs [$]"]),
            expected_total_cost_float,
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Cost Drag (Ann.) [%]"]),
            expected_total_cost_float / average_equity_float * 100,
        )


if __name__ == "__main__":
    unittest.main()
