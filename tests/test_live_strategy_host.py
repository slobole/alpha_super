from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from alpha.data import FredSeriesSnapshot
from alpha.live.models import LiveRelease, PodState
from alpha.live.strategy_host import build_decision_plan_for_release, preflight_decision_contract_for_release


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def make_release(
    strategy_import_str: str = "strategies.dv2.strategy_mr_dv2:DVO2Strategy",
    data_profile_str: str = "norgate_eod_sp500_pit",
    execution_policy_str: str = "next_open_moo",
    params_dict: dict | None = None,
) -> LiveRelease:
    return LiveRelease(
        release_id_str=f"release::{strategy_import_str.split(':')[0].split('.')[-1]}",
        user_id_str="user_001",
        pod_id_str="pod_001",
        account_route_str="DU1",
        strategy_import_str=strategy_import_str,
        mode_str="paper",
        session_calendar_id_str="XNYS",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str=execution_policy_str,
        data_profile_str=data_profile_str,
        params_dict=params_dict or {"capital_base_float": 100000.0, "max_positions_int": 1},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="manifest.yaml",
    )


def make_price_df(symbol_list: list[str], date_index: pd.DatetimeIndex) -> pd.DataFrame:
    frame_list: list[pd.DataFrame] = []
    for symbol_idx_int, symbol_str in enumerate(symbol_list):
        close_ser = pd.Series(
            100.0 + symbol_idx_int + np.arange(len(date_index), dtype=float) * 0.2,
            index=date_index,
        )
        price_df = pd.DataFrame(
            {
                (symbol_str, "Open"): close_ser - 0.1,
                (symbol_str, "High"): close_ser + 0.5,
                (symbol_str, "Low"): close_ser - 0.5,
                (symbol_str, "Close"): close_ser,
            },
            index=date_index,
        )
        frame_list.append(price_df)
    pricing_data_df = pd.concat(frame_list, axis=1)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def test_strategy_host_builds_dv2_decision_plan(monkeypatch):
    import strategies.dv2.strategy_mr_dv2 as dv2_module

    date_index = pd.bdate_range("2023-01-02", periods=260)
    pricing_data_df = make_price_df(["TEST", "$SPX"], date_index)
    universe_df = pd.DataFrame(1, index=date_index, columns=["TEST"])

    monkeypatch.setattr(dv2_module, "build_index_constituent_matrix", lambda indexname: (["TEST"], universe_df))
    monkeypatch.setattr(dv2_module, "get_prices", lambda symbols, benchmarks, start_date, end_date: pricing_data_df)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "compute_signals", lambda self, pricing_data: pricing_data)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "get_opportunities", lambda self, close: ["TEST"])

    release_obj = make_release()
    decision_plan_obj = build_decision_plan_for_release(
        release_obj=release_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=None,
    )

    assert decision_plan_obj.execution_policy_str == "next_open_moo"
    assert decision_plan_obj.decision_book_type_str == "incremental_entry_exit_book"
    assert decision_plan_obj.decision_base_position_map == {}
    assert decision_plan_obj.exit_asset_set == set()
    assert decision_plan_obj.entry_priority_list == ["TEST"]
    assert decision_plan_obj.entry_target_weight_map_dict == {"TEST": 1.0}
    assert decision_plan_obj.target_weight_map == {"TEST": 1.0}
    assert decision_plan_obj.preserve_untouched_positions_bool is True


def test_strategy_host_preflight_reports_dv2_contract_pass(monkeypatch):
    import strategies.dv2.strategy_mr_dv2 as dv2_module

    date_index = pd.bdate_range("2023-01-02", periods=260)
    pricing_data_df = make_price_df(["TEST", "$SPX"], date_index)
    universe_df = pd.DataFrame(1, index=date_index, columns=["TEST"])

    monkeypatch.setattr(dv2_module, "build_index_constituent_matrix", lambda indexname: (["TEST"], universe_df))
    monkeypatch.setattr(dv2_module, "get_prices", lambda symbols, benchmarks, start_date, end_date: pricing_data_df)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "compute_signals", lambda self, pricing_data: pricing_data)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "get_opportunities", lambda self, close: ["TEST"])

    preflight_detail_dict = preflight_decision_contract_for_release(
        release_obj=make_release(),
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=None,
    )

    assert preflight_detail_dict["decision_book_type_str"] == "incremental_entry_exit_book"
    assert preflight_detail_dict["contract_status_str"] == "pass"
    assert preflight_detail_dict["accepted_shape_count_int"] == 1
    assert preflight_detail_dict["unsupported_shape_count_int"] == 0
    assert preflight_detail_dict["unsupported_shape_example_dict_list"] == []


def test_strategy_host_accepts_zero_share_target_exit_for_dv2(monkeypatch):
    import strategies.dv2.strategy_mr_dv2 as dv2_module

    date_index = pd.bdate_range("2023-01-02", periods=260)
    pricing_data_df = make_price_df(["TEST", "$SPX"], date_index)
    pricing_data_df.loc[date_index[-2], ("TEST", "High")] = 100.0
    pricing_data_df.loc[date_index[-1], ("TEST", "Close")] = 101.0
    universe_df = pd.DataFrame(1, index=date_index, columns=["TEST"])

    monkeypatch.setattr(dv2_module, "build_index_constituent_matrix", lambda indexname: (["TEST"], universe_df))
    monkeypatch.setattr(dv2_module, "get_prices", lambda symbols, benchmarks, start_date, end_date: pricing_data_df)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "compute_signals", lambda self, pricing_data: pricing_data)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "get_opportunities", lambda self, close: [])

    release_obj = make_release()
    pod_state_obj = PodState(
        pod_id_str=release_obj.pod_id_str,
        user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str,
        position_amount_map={"TEST": 45.0},
        cash_float=1000.0,
        total_value_float=100000.0,
        strategy_state_dict={"trade_id_int": 7, "current_trade_map": {"TEST": 7}},
        updated_timestamp_ts=datetime(2024, 1, 30, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    decision_plan_obj = build_decision_plan_for_release(
        release_obj=release_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=pod_state_obj,
    )

    assert decision_plan_obj.decision_base_position_map == {"TEST": 45.0}
    assert decision_plan_obj.exit_asset_set == {"TEST"}
    assert decision_plan_obj.entry_target_weight_map_dict == {}
    assert decision_plan_obj.entry_priority_list == []


def test_strategy_host_builds_mixed_qpi_decision_plan(monkeypatch):
    import strategies.qpi.strategy_mr_qpi_ibs_rsi_exit as qpi_module

    date_index = pd.bdate_range("2023-01-02", periods=260)
    pricing_data_df = make_price_df(["OLD", "NEW", "$SPX"], date_index)
    universe_df = pd.DataFrame(1, index=date_index, columns=["OLD", "NEW"])

    def compute_signals_stub(self, pricing_data_df):
        signal_data_df = pricing_data_df.copy()
        feature_df = pd.DataFrame(
            {
                ("OLD", "ibs_value_ser"): pd.Series(0.95, index=date_index),
                ("OLD", "rsi2_value_ser"): pd.Series(95.0, index=date_index),
                ("NEW", "ibs_value_ser"): pd.Series(0.05, index=date_index),
                ("NEW", "rsi2_value_ser"): pd.Series(10.0, index=date_index),
            },
            index=date_index,
        )
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)
        return pd.concat([signal_data_df, feature_df], axis=1)

    monkeypatch.setattr(qpi_module, "build_index_constituent_matrix", lambda indexname: (["OLD", "NEW"], universe_df))
    monkeypatch.setattr(
        qpi_module,
        "get_prices",
        lambda symbol_list, benchmark_list, start_date_str, end_date_str: pricing_data_df,
    )
    monkeypatch.setattr(qpi_module.QPIIbsRsiExitStrategy, "compute_signals", compute_signals_stub)
    monkeypatch.setattr(qpi_module.QPIIbsRsiExitStrategy, "get_opportunity_list", lambda self, close_row_ser: ["NEW"])

    release_obj = make_release(
        strategy_import_str="strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy",
        params_dict={"capital_base_float": 100000.0, "max_positions_int": 1},
    )
    pod_state_obj = PodState(
        pod_id_str=release_obj.pod_id_str,
        user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str,
        position_amount_map={"OLD": 12.0},
        cash_float=1000.0,
        total_value_float=100000.0,
        strategy_state_dict={"trade_id_int": 7, "current_trade_map": {"OLD": 7}},
        updated_timestamp_ts=datetime(2024, 1, 30, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    decision_plan_obj = build_decision_plan_for_release(
        release_obj=release_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=pod_state_obj,
    )

    assert decision_plan_obj.decision_base_position_map == {"OLD": 12.0}
    assert decision_plan_obj.exit_asset_set == {"OLD"}
    assert decision_plan_obj.entry_target_weight_map_dict == {"NEW": 1.0}
    assert decision_plan_obj.entry_priority_list == ["NEW"]


def test_strategy_host_rejects_unsupported_incremental_limit_order_shape(monkeypatch):
    import strategies.dv2.strategy_mr_dv2 as dv2_module

    date_index = pd.bdate_range("2023-01-02", periods=260)
    pricing_data_df = make_price_df(["TEST", "$SPX"], date_index)
    universe_df = pd.DataFrame(1, index=date_index, columns=["TEST"])

    def iterate_limit_order_stub(self, data_df, close_row_ser, open_price_ser):
        self.order_value("TEST", 1000.0, limit_price=99.0, trade_id=1)

    monkeypatch.setattr(dv2_module, "build_index_constituent_matrix", lambda indexname: (["TEST"], universe_df))
    monkeypatch.setattr(dv2_module, "get_prices", lambda symbols, benchmarks, start_date, end_date: pricing_data_df)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "compute_signals", lambda self, pricing_data: pricing_data)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "iterate", iterate_limit_order_stub)

    release_obj = make_release()

    with pytest.raises(NotImplementedError) as exc_info:
        build_decision_plan_for_release(
            release_obj=release_obj,
            as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
            pod_state_obj=None,
        )

    error_str = str(exc_info.value)
    assert "Unsupported incremental research order shape" in error_str
    assert "strategy_import_str='strategies.dv2.strategy_mr_dv2:DVO2Strategy'" in error_str
    assert "order_class_str='LimitOrder'" in error_str
    assert "unit_str='value'" in error_str
    assert "target_bool=False" in error_str
    assert "asset_str='TEST'" in error_str


def test_strategy_host_preflight_reports_dv2_contract_failure(monkeypatch):
    import strategies.dv2.strategy_mr_dv2 as dv2_module

    date_index = pd.bdate_range("2023-01-02", periods=260)
    pricing_data_df = make_price_df(["TEST", "$SPX"], date_index)
    universe_df = pd.DataFrame(1, index=date_index, columns=["TEST"])

    def iterate_limit_order_stub(self, data_df, close_row_ser, open_price_ser):
        self.order_value("TEST", 1000.0, limit_price=99.0, trade_id=1)

    monkeypatch.setattr(dv2_module, "build_index_constituent_matrix", lambda indexname: (["TEST"], universe_df))
    monkeypatch.setattr(dv2_module, "get_prices", lambda symbols, benchmarks, start_date, end_date: pricing_data_df)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "compute_signals", lambda self, pricing_data: pricing_data)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "iterate", iterate_limit_order_stub)

    preflight_detail_dict = preflight_decision_contract_for_release(
        release_obj=make_release(),
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=None,
    )

    assert preflight_detail_dict["decision_book_type_str"] == "incremental_entry_exit_book"
    assert preflight_detail_dict["contract_status_str"] == "fail"
    assert preflight_detail_dict["accepted_shape_count_int"] == 0
    assert preflight_detail_dict["unsupported_shape_count_int"] == 1
    assert preflight_detail_dict["unsupported_shape_example_dict_list"] == [
        {
            "asset_str": "TEST",
            "order_class_str": "LimitOrder",
            "unit_str": "value",
            "target_bool": False,
            "amount_float": 1000.0,
            "trade_id_int": 1,
        }
    ]


def test_strategy_host_builds_full_target_taa_decision_plan(monkeypatch):
    import strategies.taa_df.strategy_taa_df as base_taa_module
    import strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils as vix_overlay_module

    date_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-30"),
            pd.Timestamp("2024-01-31"),
        ]
    )
    execution_price_df = make_price_df(["BTAL", "TLT", "SPY"], date_index)
    month_end_weight_df = pd.DataFrame(
        [{"BTAL": 0.4, "TLT": 0.3}],
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
    )
    diagnostic_df = pd.DataFrame(
        [{"cash_weight": 0.3}],
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
    )
    dtb3_snapshot_obj = FredSeriesSnapshot(
        value_ser=pd.Series(
            [5.20],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
            name="DTB3",
        ),
        source_name_str="FRED",
        series_id_str="DTB3",
        download_attempt_timestamp_ts=datetime(2024, 1, 31, 16, 5, tzinfo=MARKET_TIMEZONE_OBJ),
        download_status_str="download_success",
        latest_observation_date_ts=pd.Timestamp("2024-01-31"),
        used_cache_bool=False,
        freshness_business_days_int=0,
    )

    monkeypatch.setattr(
        base_taa_module,
        "get_defense_first_data_with_snapshot",
        lambda config_obj: (execution_price_df, None, month_end_weight_df, None, dtb3_snapshot_obj),
    )
    monkeypatch.setattr(
        vix_overlay_module,
        "_load_vrp_overlay_signal_frames",
        lambda config_obj: (None, pd.DataFrame(index=month_end_weight_df.index)),
    )
    monkeypatch.setattr(
        vix_overlay_module,
        "apply_vrp_cash_gate_to_month_end_weight_df",
        lambda base_month_end_weight_df, month_end_vrp_signal_df, config: (
            month_end_weight_df,
            diagnostic_df,
        ),
    )
    monkeypatch.setattr(base_taa_module.DefenseFirstStrategy, "iterate", lambda self, data_df, close_row_ser, open_price_ser: None)

    release_obj = make_release(
        strategy_import_str="strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
        data_profile_str="norgate_eod_etf_plus_vix_helper",
        execution_policy_str="next_month_first_open",
        params_dict={"capital_base_float": 100000.0},
    )
    pod_state_obj = PodState(
        pod_id_str=release_obj.pod_id_str,
        user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str,
        position_amount_map={"SPY": 5.0},
        cash_float=1000.0,
        total_value_float=100000.0,
        strategy_state_dict={},
        updated_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    decision_plan_obj = build_decision_plan_for_release(
        release_obj=release_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=pod_state_obj,
    )

    assert decision_plan_obj.decision_book_type_str == "full_target_weight_book"
    assert decision_plan_obj.full_target_weight_map_dict == {"BTAL": 0.4, "TLT": 0.3}
    assert decision_plan_obj.cash_reserve_weight_float == pytest.approx(0.3)
    assert decision_plan_obj.decision_base_position_map == {"SPY": 5.0}
    assert decision_plan_obj.preserve_untouched_positions_bool is False
    assert decision_plan_obj.rebalance_omitted_assets_to_zero_bool is True
    assert decision_plan_obj.snapshot_metadata_dict == {
        "strategy_family_str": "taa_df_btal_fallback_tqqq_vix_cash",
        "cash_weight_float": 0.3,
        "dtb3_source_name_str": "FRED",
        "dtb3_series_id_str": "DTB3",
        "dtb3_latest_observation_date_str": "2024-01-31",
        "dtb3_download_attempt_timestamp_str": "2024-01-31T16:05:00-05:00",
        "dtb3_download_status_str": "download_success",
        "dtb3_used_cache_bool": False,
        "dtb3_freshness_business_days_int": 0,
    }


def test_strategy_host_taa_live_path_does_not_use_backtest_fill_costs(monkeypatch):
    import strategies.taa_df.strategy_taa_df as base_taa_module
    import strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils as vix_overlay_module

    date_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-30"),
            pd.Timestamp("2024-01-31"),
        ]
    )
    execution_price_df = make_price_df(["BTAL", "TLT", "SPY"], date_index)
    month_end_weight_df = pd.DataFrame(
        [{"BTAL": 0.4, "TLT": 0.3}],
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
    )
    diagnostic_df = pd.DataFrame(
        [{"cash_weight": 0.3}],
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
    )
    dtb3_snapshot_obj = FredSeriesSnapshot(
        value_ser=pd.Series(
            [5.20],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
            name="DTB3",
        ),
        source_name_str="FRED",
        series_id_str="DTB3",
        download_attempt_timestamp_ts=datetime(2024, 1, 31, 16, 5, tzinfo=MARKET_TIMEZONE_OBJ),
        download_status_str="download_success",
        latest_observation_date_ts=pd.Timestamp("2024-01-31"),
        used_cache_bool=False,
        freshness_business_days_int=0,
    )

    monkeypatch.setattr(
        base_taa_module,
        "get_defense_first_data_with_snapshot",
        lambda config_obj: (execution_price_df, None, month_end_weight_df, None, dtb3_snapshot_obj),
    )
    monkeypatch.setattr(
        vix_overlay_module,
        "_load_vrp_overlay_signal_frames",
        lambda config_obj: (None, pd.DataFrame(index=month_end_weight_df.index)),
    )
    monkeypatch.setattr(
        vix_overlay_module,
        "apply_vrp_cash_gate_to_month_end_weight_df",
        lambda base_month_end_weight_df, month_end_vrp_signal_df, config: (
            month_end_weight_df,
            diagnostic_df,
        ),
    )

    def process_orders_should_not_run(self, pricing_data_df):
        raise AssertionError("Live TAA DecisionPlan generation must not call process_orders().")

    monkeypatch.setattr(
        base_taa_module.DefenseFirstStrategy,
        "process_orders",
        process_orders_should_not_run,
    )

    release_obj = make_release(
        strategy_import_str="strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
        data_profile_str="norgate_eod_etf_plus_vix_helper",
        execution_policy_str="next_month_first_open",
        params_dict={
            "capital_base_float": 100000.0,
            "slippage_float": 0.50,
            "commission_per_share_float": 99.0,
            "commission_minimum_float": 99.0,
        },
    )
    pod_state_obj = PodState(
        pod_id_str=release_obj.pod_id_str,
        user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str,
        position_amount_map={"SPY": 5.0},
        cash_float=1000.0,
        total_value_float=100000.0,
        strategy_state_dict={},
        updated_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    decision_plan_obj = build_decision_plan_for_release(
        release_obj=release_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=pod_state_obj,
    )

    assert decision_plan_obj.full_target_weight_map_dict == {"BTAL": 0.4, "TLT": 0.3}
    assert decision_plan_obj.cash_reserve_weight_float == pytest.approx(0.3)
    assert decision_plan_obj.decision_base_position_map == {"SPY": 5.0}


def test_strategy_host_taa_ignores_unavailable_partial_month_end(monkeypatch):
    import strategies.taa_df.strategy_taa_df as base_taa_module
    import strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils as vix_overlay_module

    date_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-31"),
            pd.Timestamp("2024-02-15"),
        ]
    )
    execution_price_df = make_price_df(["BTAL", "TLT", "SPY"], date_index)
    month_end_weight_df = pd.DataFrame(
        [
            {"BTAL": 0.4, "TLT": 0.3},
            {"BTAL": 0.0, "TLT": 1.0},
        ],
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]),
    )
    diagnostic_df = pd.DataFrame(
        [
            {"cash_weight": 0.3},
            {"cash_weight": 0.0},
        ],
        index=month_end_weight_df.index,
    )
    dtb3_snapshot_obj = FredSeriesSnapshot(
        value_ser=pd.Series(
            [5.20],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
            name="DTB3",
        ),
        source_name_str="FRED",
        series_id_str="DTB3",
        download_attempt_timestamp_ts=datetime(2024, 2, 15, 16, 5, tzinfo=MARKET_TIMEZONE_OBJ),
        download_status_str="download_success",
        latest_observation_date_ts=pd.Timestamp("2024-02-15"),
        used_cache_bool=False,
        freshness_business_days_int=0,
    )

    monkeypatch.setattr(
        base_taa_module,
        "get_defense_first_data_with_snapshot",
        lambda config_obj: (execution_price_df, None, month_end_weight_df, None, dtb3_snapshot_obj),
    )
    monkeypatch.setattr(
        vix_overlay_module,
        "_load_vrp_overlay_signal_frames",
        lambda config_obj: (None, pd.DataFrame(index=month_end_weight_df.index)),
    )
    monkeypatch.setattr(
        vix_overlay_module,
        "apply_vrp_cash_gate_to_month_end_weight_df",
        lambda base_month_end_weight_df, month_end_vrp_signal_df, config: (
            month_end_weight_df,
            diagnostic_df,
        ),
    )
    monkeypatch.setattr(base_taa_module.DefenseFirstStrategy, "iterate", lambda self, data_df, close_row_ser, open_price_ser: None)

    release_obj = make_release(
        strategy_import_str="strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
        data_profile_str="norgate_eod_etf_plus_vix_helper",
        execution_policy_str="next_month_first_open",
        params_dict={"capital_base_float": 100000.0},
    )

    decision_plan_obj = build_decision_plan_for_release(
        release_obj=release_obj,
        as_of_ts=datetime(2024, 2, 15, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=None,
    )

    assert decision_plan_obj.signal_timestamp_ts == datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ)
    assert decision_plan_obj.target_execution_timestamp_ts == datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ)
    assert decision_plan_obj.full_target_weight_map_dict == {"BTAL": 0.4, "TLT": 0.3}


def test_strategy_host_builds_full_target_atr_decision_plan(monkeypatch):
    import strategies.momentum.strategy_mo_atr_normalized_ndx as atr_module

    date_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-30"),
            pd.Timestamp("2024-01-31"),
        ]
    )
    pricing_data_df = make_price_df(["AAPL", "MSFT", "SPY"], date_index)
    universe_df = pd.DataFrame(1, index=date_index, columns=["AAPL", "MSFT"])
    monthly_decision_close_df = pd.DataFrame(
        {"AAPL": [100.0], "MSFT": [200.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
    )

    monkeypatch.setattr(
        atr_module,
        "get_atr_normalized_ndx_data",
        lambda config_obj: (pricing_data_df, universe_df, None),
    )
    monkeypatch.setattr(
        atr_module,
        "compute_atr_normalized_signal_tables",
        lambda price_close_df, price_high_df, price_low_df, regime_close_ser, config: (
            monthly_decision_close_df,
            monthly_decision_close_df.copy(),
            monthly_decision_close_df.copy(),
            monthly_decision_close_df.notna(),
            pd.Series([True], index=monthly_decision_close_df.index),
            pd.Series([True], index=monthly_decision_close_df.index),
            monthly_decision_close_df.copy(),
        ),
    )
    monkeypatch.setattr(atr_module.AtrNormalizedNdxStrategy, "compute_signals", lambda self, pricing_data_df: pricing_data_df)

    def get_target_weight_with_timing_assertion_ser(self, close_row_ser):
        assert self.previous_bar == pd.Timestamp("2024-01-31")
        assert self.current_bar == pd.Timestamp("2024-02-01")
        return pd.Series({"AAPL": 0.5, "MSFT": 0.5})

    monkeypatch.setattr(
        atr_module.AtrNormalizedNdxStrategy,
        "get_target_weight_ser",
        get_target_weight_with_timing_assertion_ser,
    )
    monkeypatch.setattr(atr_module.AtrNormalizedNdxStrategy, "iterate", lambda self, data_df, close_row_ser, open_price_ser: None)

    release_obj = make_release(
        strategy_import_str="strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy",
        data_profile_str="norgate_eod_ndx_pit",
        execution_policy_str="next_month_first_open",
        params_dict={"capital_base_float": 100000.0, "max_positions_int": 2},
    )
    pod_state_obj = PodState(
        pod_id_str=release_obj.pod_id_str,
        user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str,
        position_amount_map={"QQQ": 3.0},
        cash_float=1000.0,
        total_value_float=100000.0,
        strategy_state_dict={},
        updated_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    decision_plan_obj = build_decision_plan_for_release(
        release_obj=release_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=pod_state_obj,
    )

    assert decision_plan_obj.decision_book_type_str == "full_target_weight_book"
    assert decision_plan_obj.full_target_weight_map_dict == {"AAPL": 0.5, "MSFT": 0.5}
    assert decision_plan_obj.cash_reserve_weight_float == 0.0
    assert decision_plan_obj.decision_base_position_map == {"QQQ": 3.0}
    assert decision_plan_obj.preserve_untouched_positions_bool is False
    assert decision_plan_obj.rebalance_omitted_assets_to_zero_bool is True
