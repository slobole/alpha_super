from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from alpha.engine.backtest import run_daily
from alpha.engine.strategy import Strategy
from scripts.research.run_wired_dividend_cash_ledger_study import (
    get_dividend_ledger_df,
    research_dividend_cash_ledger_context,
)


class DividendRoundTripTestStrategy(Strategy):
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


class PassiveDividendTestStrategy(Strategy):
    def iterate(
        self,
        data: pd.DataFrame,
        close: pd.DataFrame,
        open_prices: pd.Series,
    ):
        return None


def _pricing_data_df(
    *,
    dividend_float: float = 1.0,
    include_dividend_bool: bool = True,
) -> pd.DataFrame:
    date_idx = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    )
    pricing_column_dict = {
        ("AAA", "Open"): [10.0, 10.0, 9.0, 9.0],
        ("AAA", "High"): [10.0, 10.0, 9.0, 9.0],
        ("AAA", "Low"): [10.0, 10.0, 9.0, 9.0],
        ("AAA", "Close"): [10.0, 10.0, 9.0, 9.0],
    }
    if include_dividend_bool:
        pricing_column_dict[("AAA", "Dividend")] = [
            0.0,
            dividend_float,
            0.0,
            0.0,
        ]
    return pd.DataFrame(pricing_column_dict, index=date_idx)


def _passive_strategy_obj(
    *,
    position_share_float: float,
) -> PassiveDividendTestStrategy:
    strategy_obj = PassiveDividendTestStrategy(
        name="passive_dividend_test",
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


def test_vanilla_engine_credits_net_dividend_before_ex_date_open_sale():
    pricing_data_df = _pricing_data_df(dividend_float=1.0)
    strategy_obj = DividendRoundTripTestStrategy(
        name="net_dividend_round_trip",
        benchmarks=[],
        capital_base=1_000.0,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )

    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=pricing_data_df.index,
        show_progress=False,
    )

    dividend_ledger_df = strategy_obj.get_dividend_ledger()
    assert strategy_obj.total_value == pytest.approx(997.5)
    assert len(dividend_ledger_df) == 1
    assert dividend_ledger_df.loc[0, "position_share_float"] == pytest.approx(10.0)
    assert dividend_ledger_df.loc[0, "gross_dividend_cash_float"] == pytest.approx(
        10.0
    )
    assert dividend_ledger_df.loc[0, "withholding_cash_float"] == pytest.approx(
        2.5
    )
    assert dividend_ledger_df.loc[0, "net_dividend_cash_float"] == pytest.approx(
        7.5
    )
    assert strategy_obj._accounting_policy_dict[
        "accounting_contract_version_str"
    ] == "net_dividend_cash_ledger_v2"
    assert strategy_obj._accounting_policy_dict[
        "dividend_data_status_str"
    ] == "available_and_active"


def test_engine_matches_the_approved_research_ledger_at_25_percent():
    pricing_data_df = _pricing_data_df(dividend_float=1.0)
    engine_strategy_obj = DividendRoundTripTestStrategy(
        name="engine_v2",
        benchmarks=[],
        capital_base=1_000.0,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    research_strategy_obj = DividendRoundTripTestStrategy(
        name="research_candidate",
        benchmarks=[],
        capital_base=1_000.0,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )

    run_daily(
        engine_strategy_obj,
        pricing_data_df,
        calendar=pricing_data_df.index,
        show_progress=False,
    )
    with research_dividend_cash_ledger_context(withholding_rate_float=0.25):
        run_daily(
            research_strategy_obj,
            pricing_data_df,
            calendar=pricing_data_df.index,
            show_progress=False,
        )

    pd.testing.assert_series_equal(
        engine_strategy_obj.results["total_value"],
        research_strategy_obj.results["total_value"],
    )
    pd.testing.assert_frame_equal(
        engine_strategy_obj.get_transactions()
        .drop(columns=["order_id"])
        .reset_index(drop=True),
        research_strategy_obj.get_transactions()
        .drop(columns=["order_id"])
        .reset_index(drop=True),
    )
    engine_dividend_ledger_df = engine_strategy_obj.get_dividend_ledger()
    research_dividend_ledger_df = get_dividend_ledger_df(research_strategy_obj)
    pd.testing.assert_frame_equal(
        engine_dividend_ledger_df,
        research_dividend_ledger_df,
    )


def test_short_position_pays_full_gross_manufactured_dividend():
    strategy_obj = _passive_strategy_obj(position_share_float=-10.0)
    strategy_obj.configure_dividend_cash_ledger(
        enabled_bool=True,
        withholding_rate_float=0.25,
    )

    net_dividend_cash_float = strategy_obj._credit_dividend_cash_before_open(
        _pricing_data_df(dividend_float=1.0)
    )

    dividend_ledger_df = strategy_obj.get_dividend_ledger()
    assert net_dividend_cash_float == pytest.approx(-10.0)
    assert dividend_ledger_df.loc[0, "gross_dividend_cash_float"] == pytest.approx(
        -10.0
    )
    assert dividend_ledger_df.loc[0, "withholding_cash_float"] == pytest.approx(
        0.0
    )
    assert dividend_ledger_df.loc[0, "net_dividend_cash_float"] == pytest.approx(
        -10.0
    )


def test_dividend_event_is_idempotent_for_same_ex_date():
    strategy_obj = _passive_strategy_obj(position_share_float=10.0)
    pricing_data_df = _pricing_data_df(dividend_float=1.0)

    first_credit_float = strategy_obj._credit_dividend_cash_before_open(
        pricing_data_df
    )
    second_credit_float = strategy_obj._credit_dividend_cash_before_open(
        pricing_data_df
    )

    assert first_credit_float == pytest.approx(7.5)
    assert second_credit_float == pytest.approx(0.0)
    assert len(strategy_obj.get_dividend_ledger()) == 1


def test_dividend_event_stays_idempotent_after_pickle_resume():
    strategy_obj = _passive_strategy_obj(position_share_float=10.0)
    pricing_data_df = _pricing_data_df(dividend_float=1.0)
    strategy_obj._credit_dividend_cash_before_open(pricing_data_df)

    resumed_strategy_obj = pickle.loads(pickle.dumps(strategy_obj))
    repeated_credit_float = resumed_strategy_obj._credit_dividend_cash_before_open(
        pricing_data_df
    )

    assert repeated_credit_float == pytest.approx(0.0)
    assert resumed_strategy_obj.cash == pytest.approx(1_007.5)
    assert len(resumed_strategy_obj.get_dividend_ledger()) == 1


def test_same_open_buyer_does_not_receive_prior_entitlement():
    strategy_obj = _passive_strategy_obj(position_share_float=0.0)
    strategy_obj.order_target("AAA", 10.0)

    strategy_obj.process_orders(_pricing_data_df())

    assert strategy_obj.get_position("AAA") == pytest.approx(10.0)
    assert len(strategy_obj.get_dividend_ledger()) == 0


def test_explicit_disabled_mode_preserves_price_return_ledger():
    strategy_obj = _passive_strategy_obj(position_share_float=10.0)
    strategy_obj.configure_dividend_cash_ledger(enabled_bool=False)

    net_dividend_cash_float = strategy_obj._credit_dividend_cash_before_open(
        _pricing_data_df()
    )

    assert net_dividend_cash_float == pytest.approx(0.0)
    assert strategy_obj.cash == pytest.approx(1_000.0)
    assert len(strategy_obj.get_dividend_ledger()) == 0
    assert strategy_obj._accounting_policy_dict[
        "dividend_data_status_str"
    ] == "disabled_explicitly"


def test_explicit_engine_policy_fails_when_dividend_field_is_missing():
    strategy_obj = _passive_strategy_obj(position_share_float=10.0)
    strategy_obj.configure_dividend_cash_ledger(enabled_bool=True)

    with pytest.raises(RuntimeError, match="no Dividend field"):
        strategy_obj._credit_dividend_cash_before_open(
            _pricing_data_df(include_dividend_bool=False)
        )

    assert strategy_obj.cash == pytest.approx(1_000.0)
    assert len(strategy_obj.get_dividend_ledger()) == 0


def test_dividend_ledger_rejects_trading_a_declared_benchmark():
    strategy_obj = PassiveDividendTestStrategy(
        name="benchmark_overlap_test",
        benchmarks=["AAA"],
        capital_base=1_000.0,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-01-02")
    strategy_obj.current_bar = pd.Timestamp("2024-01-03")
    strategy_obj.order_target("AAA", 10.0)

    with pytest.raises(RuntimeError, match="TOTALRETURN"):
        strategy_obj.process_orders(_pricing_data_df())

    assert strategy_obj.get_position("AAA") == pytest.approx(0.0)
    assert len(strategy_obj.get_transactions()) == 0
    assert len(strategy_obj.get_dividend_ledger()) == 0


def test_dividend_ledger_rejects_total_return_tradeable_source_metadata():
    strategy_obj = _passive_strategy_obj(position_share_float=10.0)
    pricing_data_df = _pricing_data_df()
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        "AAA": "TOTALRETURN",
    }

    with pytest.raises(RuntimeError, match="requires CAPITALSPECIAL"):
        strategy_obj._credit_dividend_cash_before_open(pricing_data_df)

    assert strategy_obj.cash == pytest.approx(1_000.0)
    assert len(strategy_obj.get_dividend_ledger()) == 0


def test_dividend_ledger_rejects_unknown_adjustment_provenance():
    strategy_obj = _passive_strategy_obj(position_share_float=10.0)
    pricing_data_df = _pricing_data_df()
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        "AAA": "RAW",
    }

    with pytest.raises(RuntimeError, match="'AAA': 'RAW'"):
        strategy_obj._credit_dividend_cash_before_open(pricing_data_df)

    assert strategy_obj.cash == pytest.approx(1_000.0)
    assert len(strategy_obj.get_dividend_ledger()) == 0


def test_dividend_ledger_records_verified_capitalspecial_provenance():
    strategy_obj = _passive_strategy_obj(position_share_float=10.0)
    pricing_data_df = _pricing_data_df()
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        "AAA": "CAPITALSPECIAL",
    }

    strategy_obj._credit_dividend_cash_before_open(pricing_data_df)

    assert strategy_obj._data_adjustment_policy_dict[
        "execution_and_marks_adjustment_str"
    ] == "CAPITALSPECIAL"
    assert strategy_obj._data_adjustment_policy_dict[
        "dividend_ledger_execution_basis_validation_str"
    ] == "verified_from_norgate_source_metadata"


def test_auto_mode_declares_legacy_contract_when_dividend_field_is_absent():
    strategy_obj = _passive_strategy_obj(position_share_float=10.0)

    net_dividend_cash_float = strategy_obj._credit_dividend_cash_before_open(
        _pricing_data_df(include_dividend_bool=False)
    )

    assert net_dividend_cash_float == pytest.approx(0.0)
    assert strategy_obj._accounting_policy_dict[
        "accounting_contract_version_str"
    ] == "price_return_ledger_v1"
    assert strategy_obj._accounting_policy_dict[
        "dividend_data_status_str"
    ] == "not_available_legacy_input"


def test_pre_v2_strategy_state_is_hydrated_for_reporting_and_future_runs():
    strategy_obj = _passive_strategy_obj(position_share_float=10.0)
    for attribute_name_str in (
        "_dividend_cash_ledger_mode_str",
        "dividend_withholding_rate_float",
        "_dividend_processed_ex_date_set",
        "_dividend_ledger_row_dict_list",
        "dividend_cash_gross_total_float",
        "dividend_withholding_total_float",
        "dividend_cash_net_total_float",
    ):
        delattr(strategy_obj, attribute_name_str)
    strategy_obj._accounting_policy_dict = {
        "accounting_contract_version_str": "price_return_ledger_v1",
        "dividend_policy_str": "not_credited",
    }

    assert len(strategy_obj.get_dividend_ledger()) == 0
    net_dividend_cash_float = strategy_obj._credit_dividend_cash_before_open(
        _pricing_data_df()
    )

    assert net_dividend_cash_float == pytest.approx(7.5)
    assert len(strategy_obj.get_dividend_ledger()) == 1
    assert strategy_obj._accounting_policy_dict[
        "accounting_contract_version_str"
    ] == "net_dividend_cash_ledger_v2"


def test_active_asset_missing_dividend_column_fails_atomically():
    strategy_obj = _passive_strategy_obj(position_share_float=10.0)
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

    with pytest.raises(RuntimeError, match="BBB"):
        strategy_obj._credit_dividend_cash_before_open(_pricing_data_df())

    assert strategy_obj.cash == pytest.approx(1_000.0)
    assert len(strategy_obj.get_dividend_ledger()) == 0
    assert strategy_obj.dividend_cash_gross_total_float == pytest.approx(0.0)


def test_sparse_backtest_calendar_fails_instead_of_skipping_dividends():
    strategy_obj = _passive_strategy_obj(position_share_float=10.0)
    strategy_obj.current_bar = pd.Timestamp("2024-01-04")

    with pytest.raises(RuntimeError, match="consecutive pricing sessions"):
        strategy_obj._credit_dividend_cash_before_open(_pricing_data_df())

    assert strategy_obj.cash == pytest.approx(1_000.0)
    assert len(strategy_obj.get_dividend_ledger()) == 0


@pytest.mark.parametrize("withholding_rate_float", [-0.01, 1.01, np.nan])
def test_invalid_withholding_rate_is_rejected(withholding_rate_float: float):
    strategy_obj = _passive_strategy_obj(position_share_float=0.0)

    with pytest.raises(ValueError, match="withholding_rate_float"):
        strategy_obj.configure_dividend_cash_ledger(
            withholding_rate_float=withholding_rate_float
        )


def test_negative_cash_diagnostics_are_recorded_without_financing():
    strategy_obj = _passive_strategy_obj(position_share_float=0.0)
    strategy_obj.results = pd.DataFrame(
        {
            "cash": [10.0, -1.0, -2.0, 5.0, -4.0],
            "total_value": [100.0] * 5,
        }
    )

    strategy_obj._update_accounting_diagnostics()

    accounting_policy_dict = strategy_obj._accounting_policy_dict
    assert accounting_policy_dict["negative_cash_day_count_int"] == 3
    assert accounting_policy_dict["negative_cash_episode_count_int"] == 2
    assert accounting_policy_dict["minimum_cash_float"] == pytest.approx(-4.0)
    assert accounting_policy_dict["minimum_cash_weight_float"] == pytest.approx(
        -0.04
    )
    assert accounting_policy_dict["average_negative_cash_float"] == pytest.approx(
        -7.0 / 3.0
    )
