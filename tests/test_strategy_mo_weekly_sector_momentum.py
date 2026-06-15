import unittest

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from strategies.momentum.strategy_mo_weekly_sector_momentum import (
    CASH_COLUMN_STR,
    MODEL_SYMBOL_STR,
    WeeklyRiskManagedSectorMomentumStrategy,
    WeeklySectorMomentumConfig,
    build_daily_target_weight_df,
    compute_weekly_sector_momentum_signal_tables,
    get_monday_decision_close_df,
    map_monday_decision_dates_to_rebalance_schedule_df,
)


class WeeklySectorMomentumStrategyTests(unittest.TestCase):
    def make_config(
        self,
        sector_symbol_tuple: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"),
        buy_rank_int: int = 3,
        hold_rank_int: int = 5,
        risk_off_exposure_float: float = 0.5,
    ) -> WeeklySectorMomentumConfig:
        return WeeklySectorMomentumConfig(
            sector_symbol_tuple=sector_symbol_tuple,
            regime_symbol_str="SPY",
            benchmark_list=("SPY",),
            history_start_date_str="2023-12-01",
            backtest_start_date_str="2024-01-01",
            return_short_window_int=2,
            return_mid_window_int=3,
            return_long_window_int=4,
            volatility_window_int=2,
            regime_sma_window_int=3,
            buy_rank_int=buy_rank_int,
            hold_rank_int=hold_rank_int,
            risk_off_exposure_float=risk_off_exposure_float,
            capital_base_float=100_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )

    def make_strategy(
        self,
        config: WeeklySectorMomentumConfig,
        decision_date_ts: pd.Timestamp = pd.Timestamp("2024-01-08"),
        execution_date_ts: pd.Timestamp = pd.Timestamp("2024-01-09"),
    ) -> WeeklyRiskManagedSectorMomentumStrategy:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [decision_date_ts]},
            index=pd.DatetimeIndex([execution_date_ts], name="execution_date_ts"),
        )
        return WeeklyRiskManagedSectorMomentumStrategy(
            name="WeeklySectorMomentumTest",
            benchmarks=list(config.benchmark_list),
            rebalance_schedule_df=rebalance_schedule_df,
            config=config,
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )

    def make_close_row_ser(
        self,
        config: WeeklySectorMomentumConfig,
        rank_by_symbol_dict: dict[str, float],
        volatility_by_symbol_dict: dict[str, float],
        exposure_multiplier_float: float = 1.0,
    ) -> pd.Series:
        row_dict: dict[tuple[str, str], float] = {
            (MODEL_SYMBOL_STR, "exposure_multiplier_float"): exposure_multiplier_float,
        }
        for sector_symbol_str in config.sector_symbol_tuple:
            rank_float = float(rank_by_symbol_dict.get(sector_symbol_str, np.nan))
            volatility_float = float(volatility_by_symbol_dict.get(sector_symbol_str, np.nan))
            row_dict[(sector_symbol_str, "rank_float")] = rank_float
            row_dict[(sector_symbol_str, "score_ser")] = (
                100.0 - rank_float if np.isfinite(rank_float) else np.nan
            )
            row_dict[(sector_symbol_str, "volatility_2d_ser")] = volatility_float

        close_row_ser = pd.Series(row_dict, dtype=float)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def make_pricing_data_df(
        self,
        config: WeeklySectorMomentumConfig,
        periods_int: int = 40,
    ) -> pd.DataFrame:
        trading_index = pd.date_range("2024-01-01", periods=periods_int, freq="B")
        bar_idx_vec = np.arange(periods_int, dtype=float)
        pricing_data_map: dict[tuple[str, str], np.ndarray] = {}

        for sector_position_int, sector_symbol_str in enumerate(config.sector_symbol_tuple, start=1):
            drift_float = 0.10 + sector_position_int * 0.015
            wave_vec = 0.5 * np.sin(bar_idx_vec / (2.0 + sector_position_int))
            close_vec = 50.0 + sector_position_int * 3.0 + drift_float * bar_idx_vec + wave_vec
            open_vec = close_vec * 0.999
            high_vec = np.maximum(open_vec, close_vec) * 1.002
            low_vec = np.minimum(open_vec, close_vec) * 0.998
            pricing_data_map[(sector_symbol_str, "Open")] = open_vec
            pricing_data_map[(sector_symbol_str, "High")] = high_vec
            pricing_data_map[(sector_symbol_str, "Low")] = low_vec
            pricing_data_map[(sector_symbol_str, "Close")] = close_vec
            pricing_data_map[(sector_symbol_str, "SignalClose")] = close_vec * (
                1.0 + 0.0001 * bar_idx_vec
            )

        spy_close_vec = 100.0 + 0.2 * bar_idx_vec
        pricing_data_map[("SPY", "Open")] = spy_close_vec * 0.999
        pricing_data_map[("SPY", "High")] = spy_close_vec * 1.002
        pricing_data_map[("SPY", "Low")] = spy_close_vec * 0.998
        pricing_data_map[("SPY", "Close")] = spy_close_vec

        pricing_data_df = pd.DataFrame(pricing_data_map, index=trading_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_signal_tables_match_formula_on_monday_decision(self):
        config = self.make_config(sector_symbol_tuple=("AAA", "BBB", "CCC"))
        trading_index = pd.date_range("2024-01-01", periods=6, freq="B")
        sector_signal_close_df = pd.DataFrame(
            {
                "AAA": [100.0, 101.0, 102.0, 103.0, 104.0, 112.0],
                "BBB": [100.0, 99.5, 99.0, 98.5, 98.0, 97.5],
                "CCC": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0],
            },
            index=trading_index,
            dtype=float,
        )
        regime_close_ser = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            index=trading_index,
            dtype=float,
        )

        (
            return_short_df,
            return_mid_df,
            return_long_df,
            volatility_df,
            score_df,
            rank_df,
            _regime_sma_ser,
            exposure_multiplier_ser,
        ) = compute_weekly_sector_momentum_signal_tables(
            sector_signal_close_df=sector_signal_close_df,
            regime_close_ser=regime_close_ser,
            config=config,
        )

        decision_date_ts = pd.Timestamp("2024-01-08")
        expected_short_float = 112.0 / 103.0 - 1.0
        expected_mid_float = 112.0 / 102.0 - 1.0
        expected_long_float = 112.0 / 101.0 - 1.0
        aaa_daily_return_ser = sector_signal_close_df["AAA"] / sector_signal_close_df["AAA"].shift(1) - 1.0
        expected_volatility_float = float(aaa_daily_return_ser.rolling(2, min_periods=2).std().loc[decision_date_ts])
        expected_score_float = (
            0.5 * expected_short_float + 0.3 * expected_mid_float + 0.2 * expected_long_float
        ) / expected_volatility_float

        self.assertEqual(list(score_df.index), [decision_date_ts])
        self.assertAlmostEqual(float(return_short_df.loc[decision_date_ts, "AAA"]), expected_short_float)
        self.assertAlmostEqual(float(return_mid_df.loc[decision_date_ts, "AAA"]), expected_mid_float)
        self.assertAlmostEqual(float(return_long_df.loc[decision_date_ts, "AAA"]), expected_long_float)
        self.assertAlmostEqual(float(volatility_df.loc[decision_date_ts, "AAA"]), expected_volatility_float)
        self.assertAlmostEqual(float(score_df.loc[decision_date_ts, "AAA"]), expected_score_float)
        self.assertEqual(float(rank_df.loc[decision_date_ts, "AAA"]), 1.0)
        self.assertTrue(pd.isna(score_df.loc[decision_date_ts, "CCC"]))
        self.assertEqual(float(exposure_multiplier_ser.loc[decision_date_ts]), 1.0)

    def test_signal_tables_compute_risk_off_from_spy_sma(self):
        config = self.make_config(sector_symbol_tuple=("AAA", "BBB", "CCC"))
        trading_index = pd.date_range("2024-01-01", periods=6, freq="B")
        sector_signal_close_df = pd.DataFrame(
            {
                "AAA": [100.0, 101.0, 102.0, 103.0, 104.0, 112.0],
                "BBB": [100.0, 99.5, 99.0, 98.5, 98.0, 97.5],
                "CCC": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0],
            },
            index=trading_index,
            dtype=float,
        )
        regime_close_ser = pd.Series(
            [100.0, 110.0, 110.0, 110.0, 110.0, 90.0],
            index=trading_index,
            dtype=float,
        )

        (
            _return_short_df,
            _return_mid_df,
            _return_long_df,
            _volatility_df,
            _score_df,
            _rank_df,
            regime_sma_ser,
            exposure_multiplier_ser,
        ) = compute_weekly_sector_momentum_signal_tables(
            sector_signal_close_df=sector_signal_close_df,
            regime_close_ser=regime_close_ser,
            config=config,
        )

        decision_date_ts = pd.Timestamp("2024-01-08")
        self.assertGreater(float(regime_sma_ser.loc[decision_date_ts]), float(regime_close_ser.loc[decision_date_ts]))
        self.assertAlmostEqual(float(exposure_multiplier_ser.loc[decision_date_ts]), 0.5, places=12)

    def test_monday_decisions_map_to_next_tradable_open(self):
        trading_index = pd.date_range("2024-01-01", periods=8, freq="B")
        price_close_df = pd.DataFrame({"AAA": np.arange(8, dtype=float) + 100.0}, index=trading_index)

        monday_decision_close_df = get_monday_decision_close_df(price_close_df=price_close_df)
        rebalance_schedule_df = map_monday_decision_dates_to_rebalance_schedule_df(
            decision_date_index=pd.DatetimeIndex(monday_decision_close_df.index),
            execution_index=trading_index,
        )

        self.assertIn(pd.Timestamp("2024-01-01"), monday_decision_close_df.index)
        self.assertIn(pd.Timestamp("2024-01-08"), monday_decision_close_df.index)
        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-01-02"), "decision_date_ts"]),
            pd.Timestamp("2024-01-01"),
        )
        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-01-09"), "decision_date_ts"]),
            pd.Timestamp("2024-01-08"),
        )

    def test_turnover_buffer_keeps_held_sector_still_in_top_five(self):
        config = self.make_config()
        strategy = self.make_strategy(config=config)
        strategy.previous_bar = pd.Timestamp("2024-01-08")
        strategy.current_bar = pd.Timestamp("2024-01-09")
        strategy.add_transaction(
            7,
            strategy.previous_bar,
            "DDD",
            100,
            100.0,
            10_000.0,
            1,
            0.0,
        )
        strategy.current_trade_id_map["DDD"] = 7

        close_row_ser = self.make_close_row_ser(
            config=config,
            rank_by_symbol_dict={
                "AAA": 1.0,
                "BBB": 2.0,
                "CCC": 3.0,
                "EEE": 4.0,
                "DDD": 5.0,
                "FFF": 6.0,
            },
            volatility_by_symbol_dict={
                "AAA": 0.02,
                "BBB": 0.01,
                "CCC": 0.04,
                "DDD": 0.05,
                "EEE": 0.03,
                "FFF": 0.02,
            },
            exposure_multiplier_float=1.0,
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertIn("DDD", target_weight_ser.index)
        self.assertIn("AAA", target_weight_ser.index)
        self.assertIn("BBB", target_weight_ser.index)
        self.assertNotIn("CCC", target_weight_ser.index)
        self.assertAlmostEqual(float(target_weight_ser.sum()), 1.0, places=12)

    def test_all_ineligible_mature_monday_liquidates_existing_position(self):
        config = self.make_config(sector_symbol_tuple=("AAA", "BBB"))
        trading_index = pd.date_range("2024-01-01", periods=7, freq="B")
        sector_signal_close_df = pd.DataFrame(
            {
                "AAA": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0],
                "BBB": [100.0, 99.2, 98.4, 97.6, 96.8, 96.0, 95.2],
            },
            index=trading_index,
            dtype=float,
        )
        regime_close_ser = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            index=trading_index,
            dtype=float,
        )
        (
            _return_short_df,
            _return_mid_df,
            _return_long_df,
            _volatility_df,
            score_df,
            _rank_df,
            _regime_sma_ser,
            _exposure_multiplier_ser,
        ) = compute_weekly_sector_momentum_signal_tables(
            sector_signal_close_df=sector_signal_close_df,
            regime_close_ser=regime_close_ser,
            config=config,
        )
        self.assertIn(pd.Timestamp("2024-01-08"), score_df.index)
        self.assertTrue(score_df.loc[pd.Timestamp("2024-01-08")].isna().all())

        rebalance_schedule_df = map_monday_decision_dates_to_rebalance_schedule_df(
            decision_date_index=pd.DatetimeIndex(score_df.index),
            execution_index=trading_index,
        )
        strategy = WeeklyRiskManagedSectorMomentumStrategy(
            name="WeeklySectorMomentumAllCashTest",
            benchmarks=list(config.benchmark_list),
            rebalance_schedule_df=rebalance_schedule_df,
            config=config,
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )
        strategy.previous_bar = pd.Timestamp("2024-01-08")
        strategy.current_bar = pd.Timestamp("2024-01-09")
        strategy.add_transaction(9, strategy.previous_bar, "AAA", 100, 95.0, 9_500.0, 1, 0.0)
        strategy.current_trade_id_map["AAA"] = 9

        pricing_data_map: dict[tuple[str, str], pd.Series] = {}
        for sector_symbol_str in config.sector_symbol_tuple:
            close_ser = sector_signal_close_df[sector_symbol_str]
            pricing_data_map[(sector_symbol_str, "Close")] = close_ser
            pricing_data_map[(sector_symbol_str, "SignalClose")] = close_ser
        pricing_data_map[("SPY", "Close")] = regime_close_ser
        pricing_data_df = pd.DataFrame(pricing_data_map, index=trading_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        signal_data_df = strategy.compute_signals(pricing_data_df)

        strategy.iterate(
            data_df=signal_data_df.loc[: strategy.previous_bar],
            close_row_ser=signal_data_df.loc[strategy.previous_bar],
            open_price_ser=pd.Series({"AAA": 94.0, "BBB": 95.2}, dtype=float),
        )

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        self.assertEqual(order_list[0].asset, "AAA")
        self.assertEqual(float(order_list[0].amount), 0.0)

    def test_risk_off_filter_scales_sector_weights_to_half_cash(self):
        config = self.make_config(sector_symbol_tuple=("AAA", "BBB", "CCC"))
        strategy = self.make_strategy(config=config)
        close_row_ser = self.make_close_row_ser(
            config=config,
            rank_by_symbol_dict={"AAA": 1.0, "BBB": 2.0, "CCC": 3.0},
            volatility_by_symbol_dict={"AAA": 0.02, "BBB": 0.01, "CCC": 0.04},
            exposure_multiplier_float=0.5,
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)
        rebalance_weight_df = target_weight_ser.to_frame().T
        rebalance_weight_df.index = pd.DatetimeIndex([pd.Timestamp("2024-01-09")])
        daily_target_weight_df = build_daily_target_weight_df(
            rebalance_weight_df=rebalance_weight_df,
            execution_index=pd.DatetimeIndex([pd.Timestamp("2024-01-09"), pd.Timestamp("2024-01-10")]),
            sector_symbol_tuple=config.sector_symbol_tuple,
        )

        self.assertAlmostEqual(float(target_weight_ser.sum()), 0.5, places=12)
        self.assertAlmostEqual(float(daily_target_weight_df.loc[pd.Timestamp("2024-01-09"), CASH_COLUMN_STR]), 0.5)

    def test_run_daily_smoke_generates_summary_and_daily_target_weights(self):
        config = self.make_config(
            sector_symbol_tuple=("AAA", "BBB", "CCC", "DDD"),
            buy_rank_int=2,
            hold_rank_int=3,
        )
        pricing_data_df = self.make_pricing_data_df(config=config, periods_int=45)
        sector_signal_close_df = pd.DataFrame(
            {
                sector_symbol_str: pricing_data_df[(sector_symbol_str, "SignalClose")]
                for sector_symbol_str in config.sector_symbol_tuple
            },
            index=pricing_data_df.index,
            dtype=float,
        )
        regime_close_ser = pricing_data_df[("SPY", "Close")].astype(float)
        (
            _return_short_df,
            _return_mid_df,
            _return_long_df,
            _volatility_df,
            score_df,
            _rank_df,
            _regime_sma_ser,
            _exposure_multiplier_ser,
        ) = compute_weekly_sector_momentum_signal_tables(
            sector_signal_close_df=sector_signal_close_df,
            regime_close_ser=regime_close_ser,
            config=config,
        )
        rebalance_schedule_df = map_monday_decision_dates_to_rebalance_schedule_df(
            decision_date_index=pd.DatetimeIndex(score_df.index),
            execution_index=pd.DatetimeIndex(pricing_data_df.index),
        )
        strategy = WeeklyRiskManagedSectorMomentumStrategy(
            name="WeeklySectorMomentumSmokeTest",
            benchmarks=list(config.benchmark_list),
            rebalance_schedule_df=rebalance_schedule_df,
            config=config,
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )

        run_daily(
            strategy,
            pricing_data_df,
            calendar=pricing_data_df.index,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=None,
            audit_sample_size_int=4,
        )

        self.assertIsNotNone(strategy.summary)
        self.assertIn("Strategy", strategy.summary.columns)
        self.assertGreater(len(strategy.results), 0)
        self.assertGreater(len(strategy.get_transactions()), 0)
        self.assertGreater(len(strategy.daily_target_weights), 0)
        self.assertTrue(set(config.sector_symbol_tuple).issubset(strategy.daily_target_weights.columns))
        self.assertIn(CASH_COLUMN_STR, strategy.daily_target_weights.columns)
        weight_sum_ser = strategy.daily_target_weights.sum(axis=1)
        self.assertTrue(np.allclose(weight_sum_ser.to_numpy(dtype=float), 1.0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
