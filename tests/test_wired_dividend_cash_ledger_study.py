from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.engine.backtest import run_daily
from alpha.engine.strategy import Strategy
from scripts.research.run_wired_dividend_cash_ledger_study import (
    _assert_pair_inputs_equal,
    _cash_diagnostic_dict,
    _execution_skeleton_df,
    _strategy_result_row_dict,
    credit_dividend_cash_before_open,
    get_dividend_ledger_df,
    research_dividend_cash_ledger_context,
)


class PassiveTestStrategy(Strategy):
    def iterate(
        self,
        data: pd.DataFrame,
        close: pd.DataFrame,
        open_prices: pd.Series,
    ):
        return None


class EntitlementRoundTripTestStrategy(Strategy):
    def iterate(
        self,
        data: pd.DataFrame,
        close: pd.DataFrame,
        open_prices: pd.Series,
    ):
        if self.current_bar == pd.Timestamp("2024-01-02"):
            self.order_target("AAA", 10.0)
        elif self.current_bar == pd.Timestamp("2024-01-03"):
            self.order_target("AAA", 0.0)


def _strategy_obj(position_share_float: float = 10.0) -> PassiveTestStrategy:
    strategy_obj = PassiveTestStrategy(
        name="passive_test",
        benchmarks=[],
        capital_base=1_000.0,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    if not np.isclose(position_share_float, 0.0):
        strategy_obj.add_transaction(
            trade_id=1,
            bar=pd.Timestamp("2024-01-02"),
            asset="AAA",
            amount=position_share_float,
            price=10.0,
            total_value=position_share_float * 10.0,
            order_id=1,
            commission=0.0,
        )
    strategy_obj.previous_bar = pd.Timestamp("2024-01-02")
    strategy_obj.current_bar = pd.Timestamp("2024-01-03")
    return strategy_obj


def _pricing_data_df(
    *,
    dividend_float: float = 1.0,
    include_dividend_bool: bool = True,
) -> pd.DataFrame:
    date_idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    pricing_column_dict = {
        ("AAA", "Open"): [10.0, 9.0],
        ("AAA", "High"): [10.0, 9.0],
        ("AAA", "Low"): [10.0, 9.0],
        ("AAA", "Close"): [10.0, 9.0],
    }
    if include_dividend_bool:
        pricing_column_dict[("AAA", "Dividend")] = [dividend_float, 0.0]
    return pd.DataFrame(pricing_column_dict, index=date_idx)


def test_credit_uses_prior_entitlement_bar_and_preopen_long_position():
    strategy_obj = _strategy_obj(position_share_float=10.0)

    credited_cash_float = credit_dividend_cash_before_open(
        strategy_obj,
        _pricing_data_df(dividend_float=1.0),
    )

    assert credited_cash_float == pytest.approx(10.0)
    assert strategy_obj.cash == pytest.approx(1_010.0)
    ledger_df = get_dividend_ledger_df(strategy_obj)
    assert ledger_df.loc[0, "entitlement_date"] == pd.Timestamp("2024-01-02")
    assert ledger_df.loc[0, "ex_date"] == pd.Timestamp("2024-01-03")
    assert ledger_df.loc[0, "position_share_float"] == pytest.approx(10.0)


def test_credit_is_idempotent_for_same_ex_date():
    strategy_obj = _strategy_obj(position_share_float=10.0)
    pricing_data_df = _pricing_data_df(dividend_float=1.0)

    first_credit_float = credit_dividend_cash_before_open(
        strategy_obj,
        pricing_data_df,
    )
    second_credit_float = credit_dividend_cash_before_open(
        strategy_obj,
        pricing_data_df,
    )

    assert first_credit_float == pytest.approx(10.0)
    assert second_credit_float == pytest.approx(0.0)
    assert strategy_obj.cash == pytest.approx(1_010.0)
    assert len(get_dividend_ledger_df(strategy_obj)) == 1


def test_short_position_pays_full_manufactured_dividend():
    strategy_obj = _strategy_obj(position_share_float=-10.0)

    credited_cash_float = credit_dividend_cash_before_open(
        strategy_obj,
        _pricing_data_df(dividend_float=1.0),
        withholding_rate_float=0.25,
    )

    assert credited_cash_float == pytest.approx(-10.0)
    ledger_df = get_dividend_ledger_df(strategy_obj)
    assert ledger_df.loc[0, "gross_dividend_cash_float"] == pytest.approx(-10.0)
    assert ledger_df.loc[0, "withholding_cash_float"] == pytest.approx(0.0)
    assert ledger_df.loc[0, "net_dividend_cash_float"] == pytest.approx(-10.0)


def test_withholding_reduces_only_positive_dividend_cash():
    strategy_obj = _strategy_obj(position_share_float=10.0)

    credited_cash_float = credit_dividend_cash_before_open(
        strategy_obj,
        _pricing_data_df(dividend_float=2.0),
        withholding_rate_float=0.25,
    )

    assert credited_cash_float == pytest.approx(15.0)
    ledger_df = get_dividend_ledger_df(strategy_obj)
    assert ledger_df.loc[0, "gross_dividend_cash_float"] == pytest.approx(20.0)
    assert ledger_df.loc[0, "withholding_cash_float"] == pytest.approx(5.0)
    assert ledger_df.loc[0, "net_dividend_cash_float"] == pytest.approx(15.0)


def test_missing_or_nonfinite_dividend_fails_loudly():
    missing_strategy_obj = _strategy_obj(position_share_float=10.0)
    with pytest.raises(RuntimeError, match="Missing Norgate Dividend"):
        credit_dividend_cash_before_open(
            missing_strategy_obj,
            _pricing_data_df(include_dividend_bool=False),
        )

    invalid_strategy_obj = _strategy_obj(position_share_float=10.0)
    invalid_pricing_data_df = _pricing_data_df(dividend_float=np.nan)
    with pytest.raises(RuntimeError, match="Invalid Dividend"):
        credit_dividend_cash_before_open(
            invalid_strategy_obj,
            invalid_pricing_data_df,
        )


def test_multi_asset_validation_failure_is_atomic():
    strategy_obj = _strategy_obj(position_share_float=10.0)
    strategy_obj.add_transaction(
        trade_id=2,
        bar=pd.Timestamp("2024-01-02"),
        asset="BBB",
        amount=5.0,
        price=20.0,
        total_value=100.0,
        order_id=2,
        commission=0.0,
    )
    pricing_data_df = _pricing_data_df(dividend_float=1.0)
    pricing_data_df[("BBB", "Open")] = [20.0, 19.0]
    pricing_data_df[("BBB", "High")] = [20.0, 19.0]
    pricing_data_df[("BBB", "Low")] = [20.0, 19.0]
    pricing_data_df[("BBB", "Close")] = [20.0, 19.0]

    with pytest.raises(RuntimeError, match="BBB"):
        credit_dividend_cash_before_open(strategy_obj, pricing_data_df)

    assert strategy_obj.cash == pytest.approx(1_000.0)
    assert len(get_dividend_ledger_df(strategy_obj)) == 0
    assert strategy_obj.dividend_cash_gross_total_float == pytest.approx(0.0)


def test_same_open_buyer_gets_nothing_and_same_open_seller_keeps_credit():
    pricing_data_df = _pricing_data_df(dividend_float=1.0)

    buyer_strategy_obj = _strategy_obj(position_share_float=0.0)
    buyer_strategy_obj.order_target("AAA", 10.0)
    with research_dividend_cash_ledger_context():
        buyer_strategy_obj.process_orders(pricing_data_df)
    assert buyer_strategy_obj.get_position("AAA") == pytest.approx(10.0)
    assert len(get_dividend_ledger_df(buyer_strategy_obj)) == 0

    seller_strategy_obj = _strategy_obj(position_share_float=10.0)
    seller_strategy_obj.order_target("AAA", 0.0)
    with research_dividend_cash_ledger_context():
        seller_strategy_obj.process_orders(pricing_data_df)
    assert seller_strategy_obj.get_position("AAA") == pytest.approx(0.0)
    assert get_dividend_ledger_df(seller_strategy_obj).loc[
        0,
        "net_dividend_cash_float",
    ] == pytest.approx(10.0)


def test_vanilla_backtester_credits_entitlement_before_next_open_sale():
    date_idx = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    )
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): [10.0, 10.0, 9.0, 9.0],
            ("AAA", "High"): [10.0, 10.0, 9.0, 9.0],
            ("AAA", "Low"): [10.0, 10.0, 9.0, 9.0],
            ("AAA", "Close"): [10.0, 10.0, 9.0, 9.0],
            ("AAA", "Dividend"): [0.0, 1.0, 0.0, 0.0],
        },
        index=date_idx,
    )
    baseline_strategy_obj = EntitlementRoundTripTestStrategy(
        name="baseline",
        benchmarks=[],
        capital_base=1_000.0,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    dividend_strategy_obj = EntitlementRoundTripTestStrategy(
        name="dividend",
        benchmarks=[],
        capital_base=1_000.0,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )

    run_daily(
        baseline_strategy_obj,
        pricing_data_df,
        calendar=date_idx,
        show_progress=False,
    )
    with research_dividend_cash_ledger_context():
        run_daily(
            dividend_strategy_obj,
            pricing_data_df,
            calendar=date_idx,
            show_progress=False,
        )

    assert baseline_strategy_obj.total_value == pytest.approx(990.0)
    assert dividend_strategy_obj.total_value == pytest.approx(1_000.0)
    assert len(get_dividend_ledger_df(dividend_strategy_obj)) == 1


def test_context_restores_shared_engine_method():
    original_process_orders_fn = Strategy.process_orders

    with research_dividend_cash_ledger_context():
        assert Strategy.process_orders is not original_process_orders_fn

    assert Strategy.process_orders is original_process_orders_fn


def test_context_restores_method_after_exception_and_supports_prices_keyword():
    original_process_orders_fn = Strategy.process_orders
    strategy_obj = _strategy_obj(position_share_float=0.0)
    pricing_data_df = _pricing_data_df()

    with pytest.raises(RuntimeError, match="boom"):
        with research_dividend_cash_ledger_context():
            strategy_obj.process_orders(prices=pricing_data_df)
            raise RuntimeError("boom")

    assert Strategy.process_orders is original_process_orders_fn


def test_nested_context_is_rejected():
    with research_dividend_cash_ledger_context():
        with pytest.raises(RuntimeError, match="exclusive serial"):
            with research_dividend_cash_ledger_context():
                pass


def test_execution_skeleton_ignores_global_order_ids():
    first_strategy_obj = _strategy_obj(position_share_float=0.0)
    second_strategy_obj = _strategy_obj(position_share_float=0.0)
    for strategy_obj, order_id_int in (
        (first_strategy_obj, 101),
        (second_strategy_obj, 999),
    ):
        strategy_obj.add_transaction(
            trade_id=1,
            bar=pd.Timestamp("2024-01-03"),
            asset="AAA",
            amount=10.0,
            price=9.0,
            total_value=90.0,
            order_id=order_id_int,
            commission=0.0,
        )

    assert _execution_skeleton_df(first_strategy_obj).equals(
        _execution_skeleton_df(second_strategy_obj)
    )


def test_pair_input_equality_fails_on_changed_prices_or_calendar():
    baseline_strategy_obj = _strategy_obj(position_share_float=0.0)
    dividend_strategy_obj = _strategy_obj(position_share_float=0.0)
    baseline_strategy_obj.results = pd.DataFrame(
        {"cash": [1_000.0]},
        index=[pd.Timestamp("2024-01-03")],
    )
    dividend_strategy_obj.results = baseline_strategy_obj.results.copy()
    baseline_input_dict = {
        "strategy_obj": baseline_strategy_obj,
        "pricing_data_df": _pricing_data_df(),
    }
    dividend_input_dict = {
        "strategy_obj": dividend_strategy_obj,
        "pricing_data_df": _pricing_data_df(),
    }

    _assert_pair_inputs_equal(baseline_input_dict, dividend_input_dict)

    changed_price_input_dict = {
        **dividend_input_dict,
        "pricing_data_df": _pricing_data_df().copy(),
    }
    changed_price_input_dict["pricing_data_df"].loc[
        pd.Timestamp("2024-01-03"),
        ("AAA", "Close"),
    ] = 8.0
    with pytest.raises(RuntimeError, match="pricing values differ"):
        _assert_pair_inputs_equal(
            baseline_input_dict,
            changed_price_input_dict,
        )

    changed_calendar_input_dict = {
        **dividend_input_dict,
        "pricing_data_df": _pricing_data_df().iloc[:-1],
    }
    with pytest.raises(RuntimeError, match="pricing calendars differ"):
        _assert_pair_inputs_equal(
            baseline_input_dict,
            changed_calendar_input_dict,
        )


def test_cash_diagnostic_counts_days_episodes_and_severity():
    cash_ser = pd.Series([10.0, -1.0, -2.0, 5.0, -4.0])
    total_value_ser = pd.Series([100.0] * 5)

    diagnostic_dict = _cash_diagnostic_dict(
        cash_ser,
        total_value_ser,
        prefix_str="test",
    )

    assert diagnostic_dict["test_negative_cash_day_count_int"] == 3
    assert diagnostic_dict["test_negative_cash_episode_count_int"] == 2
    assert diagnostic_dict["test_negative_cash_day_fraction_float"] == pytest.approx(
        0.6
    )
    assert diagnostic_dict["test_minimum_cash_float"] == pytest.approx(-4.0)
    assert diagnostic_dict["test_minimum_cash_weight_float"] == pytest.approx(-0.04)
    assert diagnostic_dict["test_average_negative_cash_float"] == pytest.approx(
        -7.0 / 3.0
    )
    assert diagnostic_dict[
        "test_average_negative_cash_weight_float"
    ] == pytest.approx(-7.0 / 300.0)


def test_negative_cash_is_reported_as_known_gap_without_blocking_study():
    baseline_strategy_obj = _strategy_obj(position_share_float=0.0)
    dividend_strategy_obj = _strategy_obj(position_share_float=0.0)
    result_idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    baseline_strategy_obj.results = pd.DataFrame(
        {
            "total_value": [1_000.0, 1_010.0],
            "cash": [100.0, -1.0],
        },
        index=result_idx,
    )
    dividend_strategy_obj.results = pd.DataFrame(
        {
            "total_value": [1_000.0, 1_020.0],
            "cash": [100.0, 10.0],
        },
        index=result_idx,
    )
    baseline_strategy_obj.summary = pd.DataFrame(columns=["Strategy"])
    dividend_strategy_obj.summary = pd.DataFrame(columns=["Strategy"])

    result_row_dict = _strategy_result_row_dict(
        "test.strategy",
        baseline_strategy_obj=baseline_strategy_obj,
        dividend_strategy_obj=dividend_strategy_obj,
    )

    assert result_row_dict["negative_cash_known_gap_bool"] is True
    assert (
        result_row_dict["negative_cash_policy_status_str"]
        == "KNOWN_GAP_REPORTED"
    )
    assert result_row_dict["study_completed_bool"] is True
    assert result_row_dict["baseline_negative_cash_day_count_int"] == 1
    assert result_row_dict["baseline_negative_cash_episode_count_int"] == 1
    assert result_row_dict["baseline_minimum_cash_float"] == pytest.approx(-1.0)


@pytest.mark.parametrize("withholding_rate_float", [-0.01, 1.01, np.nan])
def test_invalid_withholding_rate_is_rejected(withholding_rate_float: float):
    with pytest.raises(ValueError, match="withholding_rate_float"):
        credit_dividend_cash_before_open(
            _strategy_obj(position_share_float=10.0),
            _pricing_data_df(),
            withholding_rate_float=withholding_rate_float,
        )
