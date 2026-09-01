import numpy as np
import pandas as pd
import pytest

from alpha.engine.crisis import SUPPORTED_CRISIS_STRATEGY_SPEC_MAP
from alpha.strategy_registry import MaturityTier, tier_for
from strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd import (
    DEFAULT_CONFIG,
    FROZEN_FRED_SHA256_BY_SERIES_DICT,
    FROZEN_NORGATE_SHA256_BY_SYMBOL_DICT,
    FROZEN_SIGNAL_CONTRACT_SHA256_STR,
    TacticalYieldConfig,
    TacticalYieldStrategy,
    build_causal_cash_return_ser,
    build_canonical_signal_contract_df,
    canonical_dataframe_sha256_str,
    causal_expanding_median_state_ser,
    get_tactical_yield_data,
    historical_monthly_spread_records,
    load_frozen_yield_panel,
    run_variant,
    select_publication_safe_observation_date,
)


MODULE_IMPORT_STR = (
    "strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd"
)


def test_frozen_fred_snapshots_match_pakal_hashes() -> None:
    _yield_df, snapshot_tuple = load_frozen_yield_panel(DEFAULT_CONFIG)

    assert {snapshot_obj.series_id_str for snapshot_obj in snapshot_tuple} == {
        "DGS10",
        "DGS3MO",
        "DAAA",
        "DBAA",
    }
    assert {
        snapshot_obj.series_id_str: snapshot_obj.sha256_str
        for snapshot_obj in snapshot_tuple
    } == FROZEN_FRED_SHA256_BY_SERIES_DICT


def test_expanding_median_includes_current_and_ties_stay_in_cash() -> None:
    decision_index = pd.to_datetime(["2020-01-31", "2020-02-28"])
    observation_date_ser = pd.Series(
        pd.to_datetime(["2020-01-30", "2020-02-27"]),
        index=decision_index,
    )
    spread_ser = pd.Series([3.0, 2.0], index=decision_index, dtype=float)

    state_ser, threshold_ser = causal_expanding_median_state_ser(
        observation_date_ser=observation_date_ser,
        spread_ser=spread_ser,
        prehistory_record_list=[(pd.Timestamp("2019-12-30"), 1.0)],
    )

    assert threshold_ser.tolist() == pytest.approx([2.0, 2.0])
    assert state_ser.tolist() == [1.0, 0.0]


def test_prehistory_excludes_every_observation_from_first_decision_month() -> None:
    yield_df = pd.DataFrame(
        {
            "DGS10": [4.0, 4.1, 4.2, 4.3],
            "DGS3MO": [1.0, 1.0, 1.0, 1.0],
        },
        index=pd.to_datetime(
            ["2002-06-27", "2002-06-28", "2002-07-29", "2002-07-30"]
        ),
    )

    prehistory_record_list = historical_monthly_spread_records(
        proxy_str="DGS10-DGS3MO",
        yield_df=yield_df,
        before_date_ts=pd.Timestamp("2002-07-31"),
    )

    assert prehistory_record_list == [(pd.Timestamp("2002-06-27"), 3.0)]


def test_frozen_config_rejects_cost_or_sample_mutation() -> None:
    with pytest.raises(ValueError, match="5 bps"):
        TacticalYieldConfig(slippage_per_side_float=0.0)
    with pytest.raises(ValueError, match="start date"):
        TacticalYieldConfig(price_start_date_str="2003-01-01")
    with pytest.raises(ValueError, match="benchmark"):
        TacticalYieldConfig(benchmark_tuple=("$NDX",))


def test_publication_safe_selection_rejects_same_day_observation() -> None:
    session_index = pd.to_datetime(
        ["2026-07-29", "2026-07-30", "2026-07-31", "2026-08-03"]
    )
    yield_df = pd.DataFrame(
        {
            "DGS10": [4.0, 4.1, 4.2],
            "DGS3MO": [3.0, 3.0, 3.0],
            "DAAA": [5.0, 5.1, 5.2],
            "DBAA": [6.0, 6.1, 6.2],
        },
        index=pd.to_datetime(["2026-07-29", "2026-07-30", "2026-07-31"]),
    )

    selected_ts = select_publication_safe_observation_date(
        decision_date_ts=pd.Timestamp("2026-07-31"),
        yield_df=yield_df,
        session_index=session_index,
    )

    assert selected_ts == pd.Timestamp("2026-07-30")


def test_corrected_frozen_signal_endpoints_and_changed_ties() -> None:
    (
        _execution_price_df,
        _yield_df,
        signal_df,
        weight_df,
        _cash_return_ser,
        _snapshot_tuple,
    ) = get_tactical_yield_data(DEFAULT_CONFIG)

    first_signal_ser = signal_df.iloc[0]
    last_signal_ser = signal_df.iloc[-1]
    assert signal_df.index[0] == pd.Timestamp("2002-07-31")
    assert signal_df.index[-1] == pd.Timestamp("2026-07-31")
    assert len(signal_df) == 289
    assert first_signal_ser["observation_date"] == pd.Timestamp("2002-07-30")
    assert first_signal_ser["term_threshold_float"] == pytest.approx(1.69)
    assert first_signal_ser["credit_threshold_float"] == pytest.approx(2.915)
    assert last_signal_ser["observation_date"] == pd.Timestamp("2026-07-30")
    assert last_signal_ser["term_threshold_float"] == pytest.approx(1.60)
    assert last_signal_ser["credit_threshold_float"] == pytest.approx(3.19)
    assert signal_df.loc[pd.Timestamp("2007-12-31"), "credit_state_float"] == 1.0
    assert signal_df.loc[pd.Timestamp("2016-12-30"), "term_state_float"] == 1.0
    assert signal_df.loc[pd.Timestamp("2016-12-30"), "credit_state_float"] == 1.0
    assert weight_df.iloc[0][["IEF", "LQD", "Cash"]].tolist() == [0.5, 0.5, 0.0]
    assert weight_df.iloc[-1][["IEF", "LQD", "Cash"]].tolist() == [0.0, 0.0, 1.0]
    assert canonical_dataframe_sha256_str(
        build_canonical_signal_contract_df(signal_df, weight_df)
    ) == FROZEN_SIGNAL_CONTRACT_SHA256_STR


def test_frozen_norgate_price_frames_match_content_hashes() -> None:
    execution_price_df = get_tactical_yield_data(DEFAULT_CONFIG)[0]

    assert {
        symbol_str: canonical_dataframe_sha256_str(execution_price_df[symbol_str])
        for symbol_str in FROZEN_NORGATE_SHA256_BY_SYMBOL_DICT
    } == FROZEN_NORGATE_SHA256_BY_SYMBOL_DICT


def test_run_variant_honors_earlier_pm_end_date() -> None:
    strategy_obj = run_variant(
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2002-08-01",
        end_date_str="2003-12-31",
    )

    assert strategy_obj.results.index[-1] == pd.Timestamp("2003-12-31")
    assert strategy_obj.month_end_signal_df.index.max() <= pd.Timestamp("2003-12-31")
    assert strategy_obj.month_end_weight_df.index.max() <= pd.Timestamp("2003-12-31")
    assert strategy_obj.daily_target_weights.index.max() <= pd.Timestamp("2003-12-31")
    assert strategy_obj.cash_return_ser.index.max() <= pd.Timestamp("2003-12-31")


def test_run_variant_rejects_date_after_frozen_snapshot() -> None:
    with pytest.raises(ValueError, match="cannot run beyond"):
        run_variant(
            show_display_bool=False,
            save_results_bool=False,
            end_date_str="2026-08-20",
        )


def test_causal_cash_return_uses_t_minus_two_observation_and_act_365() -> None:
    session_index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    dgs3mo_value_ser = pd.Series(
        [3.65, 7.30, 10.95],
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]),
        dtype=float,
    )

    cash_return_ser = build_causal_cash_return_ser(
        session_index=session_index,
        dgs3mo_value_ser=dgs3mo_value_ser,
    )

    assert cash_return_ser.iloc[1] == pytest.approx(0.0365 * 3.0 / 365.0)
    assert cash_return_ser.iloc[2] == pytest.approx(0.0730 * 1.0 / 365.0)


def _strategy_obj(cash_return_ser: pd.Series) -> TacticalYieldStrategy:
    return TacticalYieldStrategy(
        name="test_tactical_yield",
        benchmarks=(),
        rebalance_weight_df=pd.DataFrame(
            {"IEF": [0.5], "LQD": [0.5], "Cash": [0.0]},
            index=[pd.Timestamp("2026-01-05")],
        ),
        cash_return_ser=cash_return_ser,
        tradeable_asset_list=("IEF", "LQD"),
        capital_base=100_000.0,
        slippage=0.0005,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )


def test_cash_interest_is_positive_cash_only_and_idempotent() -> None:
    current_bar_ts = pd.Timestamp("2026-01-05")
    strategy_obj = _strategy_obj(pd.Series([0.001], index=[current_bar_ts]))
    strategy_obj.current_bar = current_bar_ts

    assert strategy_obj._accrue_positive_cash_interest_float() == pytest.approx(100.0)
    assert strategy_obj._accrue_positive_cash_interest_float() == 0.0
    assert strategy_obj.cash == pytest.approx(100_100.0)

    next_bar_ts = pd.Timestamp("2026-01-06")
    strategy_obj.cash_return_ser.loc[next_bar_ts] = 0.001
    strategy_obj.current_bar = next_bar_ts
    strategy_obj.cash = -50.0
    assert strategy_obj._accrue_positive_cash_interest_float() == 0.0
    assert strategy_obj.cash == -50.0


def test_iterate_uses_close_t_budget_and_creates_two_target_value_orders() -> None:
    current_bar_ts = pd.Timestamp("2026-01-05")
    previous_bar_ts = pd.Timestamp("2026-01-02")
    strategy_obj = _strategy_obj(pd.Series([0.0], index=[current_bar_ts]))
    strategy_obj.current_bar = current_bar_ts
    strategy_obj.previous_bar = previous_bar_ts
    strategy_obj._total_value_history_list = [100_000.0]
    close_row_ser = pd.Series(
        {
            ("IEF", "Close"): 100.0,
            ("LQD", "Close"): 125.0,
        }
    )

    strategy_obj.iterate(pd.DataFrame(), close_row_ser, pd.Series(dtype=float))

    order_list = strategy_obj.get_orders()
    assert len(order_list) == 2
    assert {order_obj.asset for order_obj in order_list} == {"IEF", "LQD"}
    assert all(order_obj.unit == "value" for order_obj in order_list)
    assert all(order_obj.target for order_obj in order_list)
    assert [order_obj.amount for order_obj in order_list] == pytest.approx(
        [50_000.0, 50_000.0]
    )


def test_registry_and_stress_support_are_pm_ready_without_live_claim() -> None:
    assert tier_for(MODULE_IMPORT_STR) is MaturityTier.PM_READY
    assert "strategy_taa_tactical_fixed_income_ief_lqd" in SUPPORTED_CRISIS_STRATEGY_SPEC_MAP
    assert (
        SUPPORTED_CRISIS_STRATEGY_SPEC_MAP[
            "strategy_taa_tactical_fixed_income_ief_lqd"
        ].full_history_replay_bool
        is True
    )
    strategy_obj = _strategy_obj(pd.Series(dtype=float))
    assert strategy_obj._accounting_policy_dict["paper_live_authorized_bool"] is False
