import contextlib
import io
import unittest

import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.strategy import Strategy


def make_pricing_data() -> pd.DataFrame:
    date_index = pd.date_range("2024-01-01", periods=6, freq="D")
    columns = pd.MultiIndex.from_product([["TEST"], ["Open", "High", "Low", "Close"]])
    value_list = [
        [10.0, 10.5, 9.5, 10.0],
        [10.2, 10.8, 9.9, 10.4],
        [10.5, 11.0, 10.1, 10.8],
        [11.0, 11.4, 10.8, 11.2],
        [11.4, 11.9, 11.1, 11.7],
        [11.8, 12.2, 11.5, 12.0],
    ]
    return pd.DataFrame(value_list, index=date_index, columns=columns)


class AuditTrackingStrategy(Strategy):
    enable_signal_audit = True
    signal_audit_sample_size = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audit_call_count_int = 0
        self.audit_sample_size_arg = None

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        # *** CRITICAL*** The lagged close feature must remain trailing only.
        signal_data[("TEST", "lagged_close")] = signal_data[("TEST", "Close")].shift(1)
        return signal_data

    def audit_signals(self, pricing_data: pd.DataFrame, signal_data: pd.DataFrame, sample_size: int | None = None):
        self.audit_call_count_int += 1
        self.audit_sample_size_arg = sample_size
        return super().audit_signals(pricing_data, signal_data, sample_size=sample_size)

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None

    def finalize(self, current_data: pd.DataFrame):
        return None

    def summarize(self, include_benchmarks=True):
        self.summary = pd.DataFrame()
        self.summary_trades = pd.DataFrame()


class ProgressSmokeStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        for symbol_str in self.signal_progress(["TEST"], desc_str="test signal", total_int=1):
            # *** CRITICAL*** The lagged close feature must remain trailing only.
            signal_data[(symbol_str, "lagged_close")] = signal_data[(symbol_str, "Close")].shift(1)
        return signal_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        if close is None:
            return

        close_price_float = float(close[("TEST", "Close")])
        if self.get_position("TEST") == 0 and close_price_float <= 10.4:
            self.order("TEST", 1, trade_id=1)
        elif self.get_position("TEST") > 0 and close_price_float >= 11.2:
            self.order_target("TEST", 0, trade_id=1)

    def finalize(self, current_data: pd.DataFrame):
        return None


class RunDailyRuntimeControlTests(unittest.TestCase):
    def make_audit_strategy(self) -> AuditTrackingStrategy:
        return AuditTrackingStrategy(
            name="AuditTracking",
            benchmarks=[],
            capital_base=1000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def make_progress_strategy(self) -> ProgressSmokeStrategy:
        return ProgressSmokeStrategy(
            name="ProgressSmoke",
            benchmarks=[],
            capital_base=1000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_run_daily_skips_audit_when_override_is_false(self):
        pricing_data = make_pricing_data()
        strategy = self.make_audit_strategy()

        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                strategy,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=False,
            )

        self.assertEqual(strategy.audit_call_count_int, 0)

    def test_run_daily_passes_audit_sample_size_override(self):
        pricing_data = make_pricing_data()
        strategy = self.make_audit_strategy()

        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                strategy,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=None,
                audit_sample_size_int=3,
            )

        self.assertEqual(strategy.audit_call_count_int, 1)
        self.assertEqual(strategy.audit_sample_size_arg, 3)

    def test_progress_flags_do_not_change_trades_or_equity(self):
        pricing_data = make_pricing_data()
        strategy_without_progress = self.make_progress_strategy()
        strategy_with_progress = self.make_progress_strategy()

        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                strategy_without_progress,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=False,
            )
        with contextlib.redirect_stdout(io.StringIO()):
            run_daily(
                strategy_with_progress,
                pricing_data,
                show_progress=False,
                show_signal_progress_bool=True,
                audit_override_bool=False,
            )

        pd.testing.assert_frame_equal(
            strategy_without_progress.results,
            strategy_with_progress.results,
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            strategy_without_progress.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
            strategy_with_progress.get_transactions().drop(columns=["order_id"]).reset_index(drop=True),
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            strategy_without_progress.summary,
            strategy_with_progress.summary,
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            strategy_without_progress.summary_trades,
            strategy_with_progress.summary_trades,
            check_like=True,
        )


if __name__ == "__main__":
    unittest.main()
