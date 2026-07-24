from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alpha.bench import catalog
from alpha.engine.backtest import run_daily
from strategies.momentum.strategy_mo_paper_b_russell3000_vol10 import (
    DEFAULT_CONFIG,
    PaperBRussell3000Strategy,
    _get_base_calendar_idx,
    build_exposure_schedule_df,
    build_paper_b_selection_df,
    compound_daily_returns_to_calendar_month_ser,
    compute_paper_b_signal_tables,
)


def _small_config():
    return replace(
        DEFAULT_CONFIG,
        max_long_positions_int=2,
        max_short_positions_int=2,
    )


def _single_rebalance_schedule_df() -> pd.DataFrame:
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": [pd.Timestamp("2024-03-29")]},
        index=pd.to_datetime(["2024-04-01"]),
    )
    rebalance_schedule_df.index.name = "execution_date_ts"
    return rebalance_schedule_df


def _selection_fixture_df() -> pd.DataFrame:
    row_list: list[dict[str, object]] = []
    for side_str, symbol_list, base_target_weight_float in (
        ("long", ["A", "B"], 0.5),
        ("short", ["E", "F"], -0.5),
    ):
        for rank_int, symbol_str in enumerate(symbol_list, start=1):
            row_list.append(
                {
                    "decision_date_ts": pd.Timestamp("2024-03-29"),
                    "execution_date_ts": pd.Timestamp("2024-04-01"),
                    "side_str": side_str,
                    "rank_int": rank_int,
                    "symbol_str": symbol_str,
                    "paper_b_score_float": float(3 - rank_int)
                    if side_str == "long"
                    else float(-2 - rank_int),
                    "classic_momentum_float": 0.20 if side_str == "long" else -0.20,
                    "last_month_return_float": 0.05,
                    "unadjusted_close_float": 20.0,
                    "adv63_dollar_float": 10_000_000.0,
                    "eligible_count_int": 6,
                    "base_target_weight_float": base_target_weight_float,
                }
            )
    return pd.DataFrame(row_list)


def _close_row_ser(close_price_by_symbol_dict: dict[str, float] | None = None) -> pd.Series:
    if close_price_by_symbol_dict is None:
        close_price_by_symbol_dict = {symbol_str: 20.0 for symbol_str in ["A", "B", "E", "F"]}
    close_row_ser = pd.Series(
        {
            (symbol_str, "Close"): close_price_float
            for symbol_str, close_price_float in close_price_by_symbol_dict.items()
        },
        dtype=float,
    )
    close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
    return close_row_ser


def test_default_config_matches_requested_paper_b_specification():
    assert DEFAULT_CONFIG.indexname_str == "Russell 3000"
    assert DEFAULT_CONFIG.max_long_positions_int == 50
    assert DEFAULT_CONFIG.max_short_positions_int == 50
    assert DEFAULT_CONFIG.minimum_unadjusted_close_float == 1.0
    assert DEFAULT_CONFIG.minimum_adv_dollar_float == 1_000_000.0
    assert DEFAULT_CONFIG.adv_lookback_day_int == 63
    assert DEFAULT_CONFIG.volatility_lookback_month_int == 12
    assert DEFAULT_CONFIG.target_annualized_volatility_float == 0.10
    assert DEFAULT_CONFIG.maximum_exposure_multiplier_float == 1.0


def test_compute_paper_b_signal_tables_preserves_loser_rebound_math():
    decision_date_idx = pd.date_range("2020-01-31", periods=14, freq="BME")
    close_price_vec = np.full(len(decision_date_idx), 100.0)
    close_price_vec[1] = 100.0
    close_price_vec[12] = 70.0
    close_price_vec[13] = 77.0
    price_close_df = pd.DataFrame({"LOSER": close_price_vec}, index=decision_date_idx)

    (
        _monthly_decision_close_df,
        classic_momentum_df,
        last_month_return_df,
        paper_b_score_df,
    ) = compute_paper_b_signal_tables(price_close_df=price_close_df)

    final_decision_ts = decision_date_idx[-1]
    assert classic_momentum_df.loc[final_decision_ts, "LOSER"] == pytest.approx(-0.30)
    assert last_month_return_df.loc[final_decision_ts, "LOSER"] == pytest.approx(0.10)
    assert paper_b_score_df.loc[final_decision_ts, "LOSER"] == pytest.approx(-0.33)


def test_build_selection_uses_pit_raw_price_adv_and_equal_side_weights():
    config_obj = _small_config()
    decision_date_ts = pd.Timestamp("2024-03-29")
    symbol_list = ["A", "B", "C", "D", "E", "F", "OUT"]
    score_ser = pd.Series(
        {"A": 3.0, "B": 2.0, "C": 1.0, "D": -1.0, "E": -2.0, "F": -3.0, "OUT": 99.0}
    )
    classic_ser = score_ser / 10.0
    last_month_ser = pd.Series(0.05, index=symbol_list)
    raw_close_ser = pd.Series(20.0, index=symbol_list)
    raw_close_ser["A"] = 1.0
    raw_close_ser["C"] = 0.99
    adv_ser = pd.Series(10_000_000.0, index=symbol_list)
    adv_ser["D"] = 999_999.0
    adv_ser["E"] = 1_000_000.0

    universe_df = pd.DataFrame(
        {symbol_str: [0 if symbol_str == "OUT" else 1] for symbol_str in symbol_list},
        index=[decision_date_ts],
    )
    paper_b_score_df = pd.DataFrame([score_ser], index=[decision_date_ts])
    classic_momentum_df = pd.DataFrame([classic_ser], index=[decision_date_ts])
    last_month_return_df = pd.DataFrame([last_month_ser], index=[decision_date_ts])
    unadjusted_close_decision_df = pd.DataFrame([raw_close_ser], index=[decision_date_ts])
    adv_dollar_decision_df = pd.DataFrame([adv_ser], index=[decision_date_ts])

    selection_df = build_paper_b_selection_df(
        rebalance_schedule_df=_single_rebalance_schedule_df(),
        universe_df=universe_df,
        paper_b_score_df=paper_b_score_df,
        classic_momentum_df=classic_momentum_df,
        last_month_return_df=last_month_return_df,
        unadjusted_close_decision_df=unadjusted_close_decision_df,
        adv_dollar_decision_df=adv_dollar_decision_df,
        config=config_obj,
    )

    long_selection_df = selection_df.loc[selection_df["side_str"] == "long"]
    short_selection_df = selection_df.loc[selection_df["side_str"] == "short"]
    assert long_selection_df["symbol_str"].tolist() == ["A", "B"]
    assert short_selection_df["symbol_str"].tolist() == ["F", "E"]
    assert long_selection_df["base_target_weight_float"].tolist() == [0.5, 0.5]
    assert short_selection_df["base_target_weight_float"].tolist() == [-0.5, -0.5]
    assert set(selection_df["symbol_str"]) == {"A", "B", "E", "F"}
    assert (selection_df["eligible_count_int"] == 4).all()


def test_compound_daily_returns_uses_non_overlapping_calendar_months():
    daily_return_ser = pd.Series(
        [0.10, -0.10, 0.20, -0.20],
        index=pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29"]),
    )

    monthly_return_ser = compound_daily_returns_to_calendar_month_ser(daily_return_ser)

    assert monthly_return_ser.loc[pd.Period("2024-01", freq="M")] == pytest.approx(-0.01)
    assert monthly_return_ser.loc[pd.Period("2024-02", freq="M")] == pytest.approx(-0.04)


def test_hidden_base_calendar_excludes_mid_month_pre_rebalance_stub():
    pricing_date_idx = pd.bdate_range("2024-01-15", "2024-03-29")
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]},
        index=pd.to_datetime(["2024-02-01", "2024-03-01"]),
    )

    base_calendar_idx = _get_base_calendar_idx(
        pricing_date_idx=pricing_date_idx,
        rebalance_schedule_df=rebalance_schedule_df,
    )

    assert base_calendar_idx[0] == pd.Timestamp("2024-02-01")
    assert not (base_calendar_idx.to_period("M") == pd.Period("2024-01", freq="M")).any()


def test_exposure_requires_exactly_12_completed_base_returns_and_uses_unscaled_path():
    holding_month_idx = pd.period_range("2023-01", periods=12, freq="M")
    base_monthly_return_ser = pd.Series(
        [0.10, -0.10] * 6,
        index=holding_month_idx,
        dtype=float,
    )
    decision_date_idx = pd.date_range("2023-01-31", periods=12, freq="BME")
    execution_date_idx = pd.date_range("2023-02-01", periods=12, freq="BMS")
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": decision_date_idx},
        index=execution_date_idx,
    )
    rebalance_schedule_df.index.name = "execution_date_ts"

    exposure_schedule_df = build_exposure_schedule_df(
        base_monthly_return_ser=base_monthly_return_ser,
        rebalance_schedule_df=rebalance_schedule_df,
    )

    assert (exposure_schedule_df.iloc[:11]["exposure_multiplier_float"] == 0.0).all()
    assert not exposure_schedule_df.iloc[:11]["warmup_complete_bool"].any()
    expected_annualized_volatility_float = float(base_monthly_return_ser.std(ddof=1) * np.sqrt(12.0))
    expected_exposure_float = min(1.0, 0.10 / expected_annualized_volatility_float)
    final_row_ser = exposure_schedule_df.iloc[-1]
    assert bool(final_row_ser["warmup_complete_bool"]) is True
    assert final_row_ser["completed_base_return_count_int"] == 12
    assert final_row_ser["annualized_base_volatility_float"] == pytest.approx(
        expected_annualized_volatility_float
    )
    assert final_row_ser["exposure_multiplier_float"] == pytest.approx(expected_exposure_float)
    assert final_row_ser["gross_target_float"] == pytest.approx(2.0 * expected_exposure_float)


def test_exposure_never_leverages_above_the_base_portfolio():
    base_monthly_return_ser = pd.Series(
        np.linspace(-0.002, 0.002, 12),
        index=pd.period_range("2023-01", periods=12, freq="M"),
    )
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": [pd.Timestamp("2023-12-29")]},
        index=pd.to_datetime(["2024-01-02"]),
    )

    exposure_schedule_df = build_exposure_schedule_df(
        base_monthly_return_ser=base_monthly_return_ser,
        rebalance_schedule_df=rebalance_schedule_df,
    )

    assert exposure_schedule_df.iloc[0]["exposure_multiplier_float"] == 1.0
    assert exposure_schedule_df.iloc[0]["gross_target_float"] == 2.0


def test_strategy_uses_repo_default_costs_and_submits_scaled_dollar_neutral_targets():
    exposure_schedule_df = pd.DataFrame(
        {
            "decision_date_ts": [pd.Timestamp("2024-03-29")],
            "exposure_multiplier_float": [0.5],
        },
        index=pd.to_datetime(["2024-04-01"]),
    )
    strategy_obj = PaperBRussell3000Strategy(
        name="PaperBTest",
        benchmarks=["SPY"],
        rebalance_schedule_df=_single_rebalance_schedule_df(),
        selection_df=_selection_fixture_df(),
        exposure_schedule_df=exposure_schedule_df,
        config=_small_config(),
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
    strategy_obj.current_bar = pd.Timestamp("2024-04-01")

    strategy_obj.iterate(
        data_df=pd.DataFrame(index=[strategy_obj.previous_bar]),
        close_row_ser=_close_row_ser(),
        open_price_ser=pd.Series(dtype=float),
    )

    target_order_map = {
        order_obj.asset: order_obj
        for order_obj in strategy_obj.get_orders()
        if order_obj.unit == "percent" and order_obj.target
    }
    assert strategy_obj._slippage == 0.00025
    assert strategy_obj._commission_per_share == 0.005
    assert strategy_obj._commission_minimum == 1.0
    assert target_order_map["A"].amount == pytest.approx(0.25)
    assert target_order_map["B"].amount == pytest.approx(0.25)
    assert target_order_map["E"].amount == pytest.approx(-0.25)
    assert target_order_map["F"].amount == pytest.approx(-0.25)
    assert sum(float(order_obj.amount) for order_obj in target_order_map.values()) == pytest.approx(0.0)
    assert sum(abs(float(order_obj.amount)) for order_obj in target_order_map.values()) == pytest.approx(
        1.0
    )


def test_strategy_suppresses_zero_share_target_orders_and_minimum_commissions():
    exposure_schedule_df = pd.DataFrame(
        {
            "decision_date_ts": [pd.Timestamp("2024-03-29")],
            "exposure_multiplier_float": [0.5],
        },
        index=pd.to_datetime(["2024-04-01"]),
    )
    strategy_obj = PaperBRussell3000Strategy(
        name="PaperBNoopTest",
        benchmarks=["SPY"],
        rebalance_schedule_df=_single_rebalance_schedule_df(),
        selection_df=_selection_fixture_df(),
        exposure_schedule_df=exposure_schedule_df,
        config=_small_config(),
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
    strategy_obj.current_bar = pd.Timestamp("2024-04-01")

    strategy_obj.iterate(
        data_df=pd.DataFrame(index=[strategy_obj.previous_bar]),
        close_row_ser=_close_row_ser(
            {symbol_str: 1_000_000_000.0 for symbol_str in ["A", "B", "E", "F"]}
        ),
        open_price_ser=pd.Series(dtype=float),
    )

    assert strategy_obj.get_orders() == []


def test_strategy_keeps_an_exact_one_share_target_delta():
    exposure_schedule_df = pd.DataFrame(
        {
            "decision_date_ts": [pd.Timestamp("2024-03-29")],
            "exposure_multiplier_float": [0.5],
        },
        index=pd.to_datetime(["2024-04-01"]),
    )
    strategy_obj = PaperBRussell3000Strategy(
        name="PaperBOneShareTest",
        benchmarks=["SPY"],
        rebalance_schedule_df=_single_rebalance_schedule_df(),
        selection_df=_selection_fixture_df(),
        exposure_schedule_df=exposure_schedule_df,
        config=_small_config(),
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
    strategy_obj.current_bar = pd.Timestamp("2024-04-01")
    strategy_obj.add_transaction(
        trade_id=1,
        bar=strategy_obj.previous_bar,
        asset="A",
        amount=100_000,
        price=0.25,
        total_value=25_000.0,
        order_id=1,
        commission=0.0,
    )
    strategy_obj.current_trade_map["A"] = 1
    one_share_delta_close_float = 25_000.0 / 100_001.0

    strategy_obj.iterate(
        data_df=pd.DataFrame(index=[strategy_obj.previous_bar]),
        close_row_ser=_close_row_ser(
            {
                "A": one_share_delta_close_float,
                "B": 20.0,
                "E": 20.0,
                "F": 20.0,
            }
        ),
        open_price_ser=pd.Series(dtype=float),
    )

    target_order_map = {
        order_obj.asset: order_obj
        for order_obj in strategy_obj.get_orders()
        if order_obj.unit == "percent" and order_obj.target
    }
    assert "A" in target_order_map


def test_vanilla_engine_smoke_fills_long_and_short_targets_with_full_ohlc():
    date_idx = pd.to_datetime(["2024-03-29", "2024-04-01", "2024-04-02", "2024-04-03"])
    pricing_data_map: dict[tuple[str, str], list[float]] = {}
    for symbol_int, symbol_str in enumerate(["A", "B", "E", "F"], start=1):
        close_price_float = 20.0 + symbol_int
        close_price_list = [
            close_price_float,
            close_price_float,
            close_price_float * (1.0 + 0.002 * symbol_int),
            close_price_float * (1.0 - 0.001 * symbol_int),
        ]
        pricing_data_map[(symbol_str, "Open")] = close_price_list
        pricing_data_map[(symbol_str, "High")] = [price_float * 1.01 for price_float in close_price_list]
        pricing_data_map[(symbol_str, "Low")] = [price_float * 0.99 for price_float in close_price_list]
        pricing_data_map[(symbol_str, "Close")] = close_price_list
    pricing_data_map[("SPY", "Open")] = [500.0, 500.0, 501.0, 499.0]
    pricing_data_map[("SPY", "Close")] = [500.0, 500.0, 501.0, 499.0]
    pricing_data_df = pd.DataFrame(pricing_data_map, index=date_idx)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    exposure_schedule_df = pd.DataFrame(
        {
            "decision_date_ts": [pd.Timestamp("2024-03-29")],
            "exposure_multiplier_float": [0.5],
        },
        index=pd.to_datetime(["2024-04-01"]),
    )
    strategy_obj = PaperBRussell3000Strategy(
        name="PaperBVanillaSmoke",
        benchmarks=["SPY"],
        rebalance_schedule_df=_single_rebalance_schedule_df(),
        selection_df=_selection_fixture_df(),
        exposure_schedule_df=exposure_schedule_df,
        config=_small_config(),
    )

    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=date_idx,
        show_progress=False,
        show_signal_progress_bool=False,
        audit_override_bool=False,
    )

    transaction_df = strategy_obj.get_transactions()
    assert set(transaction_df["asset"]) == {"A", "B", "E", "F"}
    assert (transaction_df.loc[transaction_df["asset"].isin(["A", "B"]), "amount"] > 0).all()
    assert (transaction_df.loc[transaction_df["asset"].isin(["E", "F"]), "amount"] < 0).all()


def test_bench_discovers_paper_b_as_runnable_cross_sectional_momentum():
    entry_obj = next(
        entry_obj
        for entry_obj in catalog.list_strategies()
        if entry_obj.stem_str == "strategy_mo_paper_b_russell3000_vol10"
    )

    assert entry_obj.has_run_variant_bool is True
    assert entry_obj.category_str == "momentum"
    assert entry_obj.subcategory_str == "cross_sectional"
