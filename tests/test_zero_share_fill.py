"""Guards for the zero-share fill rule.

An order whose target allocation rounds down to zero shares must not execute:
nothing is bought or sold, so no transaction is recorded and no commission is
charged. Executing it anyway billed the commission minimum for an unfilled
order and produced a phantom round-trip trade whose return is profit / 0.
"""

import unittest

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.strategy import Strategy


def make_pricing_data(close_price_float: float) -> pd.DataFrame:
    """Two assets: one normally priced, one priced beyond the capital base."""
    date_index = pd.date_range("2024-01-01", periods=4, freq="D")
    columns = pd.MultiIndex.from_product(
        [["CHEAP", "PRICEY"], ["Open", "High", "Low", "Close"]]
    )
    row_list = []
    for _ in range(4):
        row_list.append(
            [10.0, 10.5, 9.5, 10.0]
            + [close_price_float, close_price_float, close_price_float, close_price_float]
        )
    return pd.DataFrame(row_list, index=date_index, columns=columns)


class BuyBothStrategy(Strategy):
    """Allocate half the book to each asset, once, on the first tradable bar."""

    def compute_signals(self, pricing_data):
        self._has_ordered_bool = False
        return pricing_data

    def iterate(self, data, close, open_prices):
        if getattr(self, '_has_ordered_bool', False):
            return
        self._has_ordered_bool = True
        self.order_target_percent("CHEAP", 0.5)
        self.order_target_percent("PRICEY", 0.5)


class ZeroShareFillTests(unittest.TestCase):
    def _run(self, pricey_close_float: float) -> Strategy:
        pricing_data_df = make_pricing_data(pricey_close_float)
        strategy_obj = BuyBothStrategy(
            name="zero_share_fill",
            benchmarks=[],
            capital_base=10_000.0,
            commission_per_share=0.005,
            commission_minimum=1.0,
        )
        run_daily(strategy_obj, pricing_data_df, show_progress=False)
        return strategy_obj

    def test_unaffordable_asset_records_no_transaction_and_no_commission(self):
        """A share priced above the whole book must not produce a filled order."""
        skipped_strategy_obj = self._run(pricey_close_float=50_000.0)
        filled_strategy_obj = self._run(pricey_close_float=100.0)

        skipped_transaction_df = skipped_strategy_obj._transactions
        self.assertEqual(len(skipped_transaction_df[skipped_transaction_df["asset"] == "PRICEY"]), 0)
        # The affordable leg still trades normally.
        self.assertGreater(len(skipped_transaction_df[skipped_transaction_df["asset"] == "CHEAP"]), 0)

        # The unfilled order costs exactly nothing: total commission equals the
        # filled run's total minus that order's own commission.
        skipped_commission_float = float(skipped_transaction_df["commission"].sum())
        filled_transaction_df = filled_strategy_obj._transactions
        pricey_commission_float = float(
            filled_transaction_df[filled_transaction_df["asset"] == "PRICEY"]["commission"].sum()
        )
        filled_commission_float = float(filled_transaction_df["commission"].sum())
        self.assertAlmostEqual(
            skipped_commission_float, filled_commission_float - pricey_commission_float
        )

    def test_no_zero_amount_transactions_are_ever_recorded(self):
        strategy_obj = self._run(pricey_close_float=50_000.0)
        amount_vec = strategy_obj._transactions["amount"].astype(float).to_numpy()
        self.assertFalse(np.isclose(amount_vec, 0.0).any())

    def test_trade_returns_stay_finite(self):
        """The phantom trade is what produced -inf trade returns."""
        strategy_obj = self._run(pricey_close_float=50_000.0)
        trade_df = strategy_obj._trades
        if len(trade_df) > 0:
            return_vec = trade_df["return"].astype(float).to_numpy()
            self.assertFalse(np.isinf(return_vec).any())

    def test_affordable_asset_is_unaffected(self):
        """The guard must not suppress orders that can actually fill."""
        strategy_obj = self._run(pricey_close_float=100.0)
        transaction_df = strategy_obj._transactions

        pricey_transaction_df = transaction_df[transaction_df["asset"] == "PRICEY"]
        self.assertGreater(len(pricey_transaction_df), 0)
        self.assertGreater(float(pricey_transaction_df["amount"].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
