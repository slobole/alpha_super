from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from alpha.live import reference_compare
from alpha.live.models import BrokerSnapshot, DecisionPlan, LiveRelease, PodState, VPlan, VPlanRow
from alpha.live.release_manifest import SUPPORTED_MODE_TUPLE, SUPPORTED_STRATEGY_IMPORT_TUPLE
from alpha.live.runner import get_compare_reference_summary
from alpha.live.state_store_v2 import LiveStateStore


MARKET_TZ = ZoneInfo("America/New_York")


class FakeReferenceStrategy:
    def __init__(self) -> None:
        self.results = pd.DataFrame(
            {
                "total_value": [5000.0],
                "cash": [4000.0],
            },
            index=pd.to_datetime(["2024-02-02"]),
        )
        self.realized_weight_df = pd.DataFrame(
            {"AAPL": [0.2], "Cash": [0.8]},
            index=pd.to_datetime(["2024-02-02"]),
        )
        self.transaction_df = pd.DataFrame(
            [
                {
                    "trade_id": 1,
                    "bar": pd.Timestamp("2024-02-02"),
                    "asset": "AAPL",
                    "amount": 10.0,
                    "price": 100.0,
                    "total_value": 1000.0,
                    "order_id": 1,
                    "commission": 1.0,
                }
            ]
        )

    def get_transactions(self):
        return self.transaction_df

    def to_pickle(self, path):
        Path(path).write_bytes(b"fake-reference")


def _build_release(
    strategy_import_str: str = "strategies.dv2.strategy_mr_dv2:DVO2Strategy",
    mode_str: str = "paper",
    pod_id_str: str = "pod_test_01",
) -> LiveRelease:
    return LiveRelease(
        release_id_str=f"user_001.{pod_id_str}.{mode_str}.v1",
        user_id_str="user_001",
        pod_id_str=pod_id_str,
        account_route_str=("SIM_" + pod_id_str if mode_str == "incubation" else "DU1"),
        strategy_import_str=strategy_import_str,
        mode_str=mode_str,
        session_calendar_id_str="XNYS",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
        data_profile_str="norgate_eod_sp500_pit",
        params_dict={"capital_base_float": 10000.0},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="test.yaml",
        pod_budget_fraction_float=0.5,
        auto_submit_enabled_bool=True,
    )


def test_supported_deployment_strategies_expose_auto_reference_contract():
    for strategy_import_str in SUPPORTED_STRATEGY_IMPORT_TUPLE:
        release_obj = _build_release(strategy_import_str=strategy_import_str)

        support_dict = reference_compare.inspect_auto_reference_support_dict(release_obj)

        assert support_dict["supported_bool"] is True
        assert support_dict["missing_parameter_list"] == []


def _insert_vplan(
    state_store_obj: LiveStateStore,
    release_obj: LiveRelease,
    signal_ts: datetime,
    execution_ts: datetime,
    pod_budget_float: float,
) -> VPlan:
    decision_plan_obj = state_store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=signal_ts,
            submission_timestamp_ts=execution_ts.replace(hour=9, minute=20),
            target_execution_timestamp_ts=execution_ts,
            execution_policy_str=release_obj.execution_policy_str,
            decision_base_position_map={},
            snapshot_metadata_dict={},
            strategy_state_dict={},
            decision_book_type_str="incremental_entry_exit_book",
            entry_target_weight_map_dict={"AAPL": 0.2},
            target_weight_map={"AAPL": 0.2},
            exit_asset_set=set(),
            entry_priority_list=["AAPL"],
        )
    )
    return state_store_obj.insert_vplan(
        VPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            decision_plan_id_int=int(decision_plan_obj.decision_plan_id_int),
            signal_timestamp_ts=signal_ts,
            submission_timestamp_ts=execution_ts.replace(hour=9, minute=20),
            target_execution_timestamp_ts=execution_ts,
            execution_policy_str=release_obj.execution_policy_str,
            broker_snapshot_timestamp_ts=execution_ts.replace(hour=9, minute=15),
            live_reference_snapshot_timestamp_ts=execution_ts.replace(hour=9, minute=15),
            live_price_source_str="test",
            net_liq_float=pod_budget_float,
            available_funds_float=pod_budget_float,
            excess_liquidity_float=pod_budget_float,
            pod_budget_fraction_float=1.0,
            pod_budget_float=pod_budget_float,
            current_broker_position_map={},
            live_reference_price_map={"AAPL": 100.0},
            target_share_map={"AAPL": 10.0},
            order_delta_map={"AAPL": 10.0},
            vplan_row_list=[
                VPlanRow(
                    asset_str="AAPL",
                    current_share_float=0.0,
                    target_share_float=10.0,
                    order_delta_share_float=10.0,
                    live_reference_price_float=100.0,
                    estimated_target_notional_float=1000.0,
                    broker_order_type_str="MOO",
                    live_reference_source_str="test",
                )
            ],
        )
    )


def test_pod_state_history_is_append_only(tmp_path):
    state_store_obj = LiveStateStore(str(tmp_path / "live.sqlite3"))
    release_obj = _build_release()
    state_store_obj.upsert_release(release_obj)

    state_store_obj.upsert_pod_state(
        PodState(
            pod_id_str=release_obj.pod_id_str,
            user_id_str=release_obj.user_id_str,
            account_route_str=release_obj.account_route_str,
            position_amount_map={},
            cash_float=10000.0,
            total_value_float=10000.0,
            strategy_state_dict={},
            updated_timestamp_ts=datetime(2024, 2, 1, 16, 0, tzinfo=MARKET_TZ),
        )
    )
    state_store_obj.upsert_pod_state(
        PodState(
            pod_id_str=release_obj.pod_id_str,
            user_id_str=release_obj.user_id_str,
            account_route_str=release_obj.account_route_str,
            position_amount_map={"AAPL": 10.0},
            cash_float=9000.0,
            total_value_float=10050.0,
            strategy_state_dict={},
            updated_timestamp_ts=datetime(2024, 2, 2, 16, 0, tzinfo=MARKET_TZ),
        ),
        snapshot_stage_str="eod",
        snapshot_source_str="broker",
    )

    history_row_dict_list = state_store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str)
    latest_pod_state_obj = state_store_obj.get_pod_state(release_obj.pod_id_str)

    assert len(history_row_dict_list) == 2
    assert history_row_dict_list[0]["total_value_float"] == 10000.0
    assert history_row_dict_list[0]["snapshot_stage_str"] == "unknown"
    assert history_row_dict_list[1]["position_amount_map"] == {"AAPL": 10.0}
    assert history_row_dict_list[1]["snapshot_stage_str"] == "eod"
    assert history_row_dict_list[1]["snapshot_source_str"] == "broker"
    assert latest_pod_state_obj is not None
    assert latest_pod_state_obj.snapshot_stage_str == "eod"
    assert latest_pod_state_obj.snapshot_source_str == "broker"


def test_auto_reference_uses_first_vplan_execution_date_and_budget(tmp_path, monkeypatch):
    state_store_obj = LiveStateStore(str(tmp_path / "live.sqlite3"))
    release_obj = _build_release()
    state_store_obj.upsert_release(release_obj)
    releases_root_path_obj = tmp_path / "empty_releases"
    releases_root_path_obj.mkdir()

    _insert_vplan(
        state_store_obj,
        release_obj,
        datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TZ),
        datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TZ),
        5000.0,
    )
    _insert_vplan(
        state_store_obj,
        release_obj,
        datetime(2024, 2, 1, 16, 0, tzinfo=MARKET_TZ),
        datetime(2024, 2, 2, 9, 30, tzinfo=MARKET_TZ),
        5000.0,
    )

    captured_arg_dict = {}

    def fake_run_auto_reference_strategy(**kwarg_dict):
        captured_arg_dict.update(kwarg_dict)
        return FakeReferenceStrategy()

    monkeypatch.setattr(reference_compare, "run_auto_reference_strategy", fake_run_auto_reference_strategy)

    detail_dict = get_compare_reference_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 2, 17, 0, tzinfo=MARKET_TZ),
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        pod_id_str=release_obj.pod_id_str,
        output_dir_str=str(tmp_path / "results"),
    )

    compare_report_dict = detail_dict["compare_report_dict_list"][0]
    assert captured_arg_dict["deployment_start_date_str"] == "2024-02-01"
    assert captured_arg_dict["reference_end_date_str"] == "2024-02-02"
    assert captured_arg_dict["deployment_initial_cash_float"] == 5000.0
    assert compare_report_dict["deployment_start_date_str"] == "2024-02-01"
    assert compare_report_dict["deployment_initial_cash_float"] == 5000.0


def test_compare_reference_prefers_same_date_eod_equity(tmp_path, monkeypatch):
    state_store_obj = LiveStateStore(str(tmp_path / "live.sqlite3"))
    release_obj = _build_release()
    state_store_obj.upsert_release(release_obj)
    releases_root_path_obj = tmp_path / "empty_releases"
    releases_root_path_obj.mkdir()

    _insert_vplan(
        state_store_obj,
        release_obj,
        datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TZ),
        datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TZ),
        5000.0,
    )
    _insert_vplan(
        state_store_obj,
        release_obj,
        datetime(2024, 2, 1, 16, 0, tzinfo=MARKET_TZ),
        datetime(2024, 2, 2, 9, 30, tzinfo=MARKET_TZ),
        5000.0,
    )
    state_store_obj.upsert_pod_state(
        PodState(
            pod_id_str=release_obj.pod_id_str,
            user_id_str=release_obj.user_id_str,
            account_route_str=release_obj.account_route_str,
            position_amount_map={"AAPL": 10.0},
            cash_float=3500.0,
            total_value_float=4500.0,
            strategy_state_dict={},
            updated_timestamp_ts=datetime(2024, 2, 2, 9, 35, tzinfo=MARKET_TZ),
        ),
        snapshot_stage_str="post_execution",
        snapshot_source_str="broker",
    )
    state_store_obj.upsert_pod_state(
        PodState(
            pod_id_str=release_obj.pod_id_str,
            user_id_str=release_obj.user_id_str,
            account_route_str=release_obj.account_route_str,
            position_amount_map={"AAPL": 99.0},
            cash_float=4000.0,
            total_value_float=5050.0,
            strategy_state_dict={},
            updated_timestamp_ts=datetime(2024, 2, 2, 16, 10, tzinfo=MARKET_TZ),
        ),
        snapshot_stage_str="eod",
        snapshot_source_str="broker",
    )
    state_store_obj.upsert_broker_snapshot_cache(
        BrokerSnapshot(
            account_route_str=release_obj.account_route_str,
            snapshot_timestamp_ts=datetime(2024, 2, 2, 16, 10, tzinfo=MARKET_TZ),
            cash_float=4000.0,
            total_value_float=5050.0,
            net_liq_float=5050.0,
            position_amount_map={"AAPL": 99.0},
        )
    )

    monkeypatch.setattr(
        reference_compare,
        "run_auto_reference_strategy",
        lambda **_kwarg_dict: FakeReferenceStrategy(),
    )

    detail_dict = get_compare_reference_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 2, 17, 0, tzinfo=MARKET_TZ),
        releases_root_path_str=str(releases_root_path_obj),
        env_mode_str="paper",
        pod_id_str=release_obj.pod_id_str,
        output_dir_str=str(tmp_path / "results"),
    )

    compare_report_dict = detail_dict["compare_report_dict_list"][0]
    assert compare_report_dict["actual_equity_float"] == 5050.0
    assert compare_report_dict["actual_cash_float"] == 4000.0
    assert compare_report_dict["actual_equity_source_str"] == "pod_state_history.eod"
    assert abs(float(compare_report_dict["equity_tracking_error_float"]) - 0.01) < 1e-12
    assert compare_report_dict["compare_row_dict_list"][0]["actual_position_float"] == 10.0


def test_compare_reference_dispatches_for_supported_strategy_mode_matrix(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "empty_releases"
    releases_root_path_obj.mkdir()

    def fake_run_auto_reference_strategy(**_kwarg_dict):
        return FakeReferenceStrategy()

    monkeypatch.setattr(reference_compare, "run_auto_reference_strategy", fake_run_auto_reference_strategy)

    for strategy_index_int, strategy_import_str in enumerate(SUPPORTED_STRATEGY_IMPORT_TUPLE):
        for mode_str in SUPPORTED_MODE_TUPLE:
            pod_id_str = f"pod_matrix_{strategy_index_int}_{mode_str}"
            state_store_obj = LiveStateStore(str(tmp_path / f"{pod_id_str}.sqlite3"))
            release_obj = _build_release(
                strategy_import_str=strategy_import_str,
                mode_str=mode_str,
                pod_id_str=pod_id_str,
            )
            state_store_obj.upsert_release(release_obj)
            _insert_vplan(
                state_store_obj,
                release_obj,
                datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TZ),
                datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TZ),
                5000.0,
            )
            _insert_vplan(
                state_store_obj,
                release_obj,
                datetime(2024, 2, 1, 16, 0, tzinfo=MARKET_TZ),
                datetime(2024, 2, 2, 9, 30, tzinfo=MARKET_TZ),
                5000.0,
            )

            detail_dict = get_compare_reference_summary(
                state_store_obj=state_store_obj,
                as_of_ts=datetime(2024, 2, 2, 17, 0, tzinfo=MARKET_TZ),
                releases_root_path_str=str(releases_root_path_obj),
                env_mode_str=mode_str,
                pod_id_str=pod_id_str,
                output_dir_str=str(tmp_path / "results"),
            )

            compare_report_dict_list = detail_dict["compare_report_dict_list"]
            assert len(compare_report_dict_list) == 1
            assert compare_report_dict_list[0]["pod_id_str"] == pod_id_str
            assert compare_report_dict_list[0]["deployment_start_date_str"] == "2024-02-01"


def test_compare_status_catches_fill_mismatches():
    base_report_dict = {
        "compare_row_dict_list": [
            {
                "planned_order_delta_share_float": 10.0,
                "filled_share_float": 0.0,
                "quantity_diff_float": -10.0,
                "backtest_quantity_diff_float": -10.0,
                "reference_position_diff_float": -10.0,
                "backtest_fill_price_diff_float": None,
            }
        ]
    }

    status_dict = reference_compare.classify_compare_status_dict(base_report_dict)

    assert status_dict["status_str"] == "red"
    assert status_dict["open_issue_count_int"] == 1


def test_html_report_writes_expected_artifacts(tmp_path):
    reference_strategy_obj = FakeReferenceStrategy()
    reference_maps_dict = reference_compare.build_reference_maps_dict(reference_strategy_obj)
    compare_report_dict = {
        "pod_id_str": "pod_test_01",
        "target_session_date_str": "2024-02-02",
        "deployment_start_date_str": "2024-02-01",
        "deployment_initial_cash_float": 5000.0,
        "actual_equity_float": 5000.0,
        "backtest_equity_float": 5000.0,
        "equity_tracking_error_float": 0.0,
        "actual_cash_float": 4000.0,
        "backtest_cash_float": 4000.0,
        "cash_diff_float": 0.0,
        "status_str": "green",
        "open_issue_count_int": 0,
        "compare_row_dict_list": [
            {
                "asset_str": "AAPL",
                "planned_order_delta_share_float": 10.0,
                "filled_share_float": 10.0,
                "quantity_diff_float": 0.0,
                "target_position_float": 10.0,
                "actual_position_float": 10.0,
                "reference_position_float": 10.0,
                "position_diff_float": 0.0,
                "reference_position_diff_float": 0.0,
                "avg_fill_price_float": 100.0,
                "reference_price_float": 100.0,
                "fill_slippage_bps_float": 0.0,
                "backtest_quantity_diff_float": 0.0,
                "backtest_fill_price_diff_float": 0.0,
            }
        ],
    }

    artifact_path_dict = reference_compare.write_reference_compare_artifacts(
        output_dir_path_obj=tmp_path / "report",
        release_obj=_build_release(),
        compare_report_dict=compare_report_dict,
        reference_maps_dict=reference_maps_dict,
        live_history_row_dict_list=[],
        reference_strategy_pickle_path_str=str(tmp_path / "reference.pkl"),
    )

    assert Path(artifact_path_dict["html_path_str"]).exists()
    assert Path(artifact_path_dict["summary_json_path_str"]).exists()
    assert Path(artifact_path_dict["fill_compare_csv_path_str"]).read_text(encoding="utf-8").strip()


def test_dv2_uses_latest_prior_pit_universe_row_for_lagged_universe_date():
    from strategies.dv2.strategy_mr_dv2 import DVO2Strategy

    strategy_obj = DVO2Strategy(
        name="test_dv2",
        benchmarks=[],
        capital_base=10000.0,
    )
    strategy_obj.previous_bar = pd.Timestamp("2026-04-24")
    strategy_obj.universe_df = pd.DataFrame(
        {"AAPL": [1], "MSFT": [0]},
        index=pd.to_datetime(["2026-04-23"]),
    )
    close_row_ser = pd.Series(
        {
            ("AAPL", "Close"): 120.0,
            ("AAPL", "sma_200"): 100.0,
            ("AAPL", "p126d_return"): 0.10,
            ("AAPL", "dv2"): 5.0,
            ("AAPL", "natr"): 2.0,
            ("MSFT", "Close"): 120.0,
            ("MSFT", "sma_200"): 100.0,
            ("MSFT", "p126d_return"): 0.10,
            ("MSFT", "dv2"): 5.0,
            ("MSFT", "natr"): 3.0,
        }
    )

    opportunity_symbol_list = strategy_obj.get_opportunities(close_row_ser)

    assert opportunity_symbol_list == ["AAPL"]
