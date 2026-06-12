"""Tests for the read-only trade-sheet export (alpha/live/trade_sheet.py).

The export renders the persisted DecisionPlan + VPlan for one pod into an
xlsx an operator can hand-execute from. These tests seed a temporary
LiveStateStore the same way tests/test_live_runner.py does and assert the
sheet frames, the xlsx round-trip, the decision-only fallback, the loud
empty-DB error, and the runner CLI wiring.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import alpha.live.runner as runner_module
from alpha.live.models import DecisionPlan, VPlan, VPlanRow
from alpha.live.runner import main
from alpha.live.state_store_v2 import LiveStateStore
from alpha.live.trade_sheet import (
    build_trade_sheet_data,
    export_trade_sheet_detail_dict,
)


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
GENERATED_AT_TS = datetime(2024, 2, 1, 9, 25, tzinfo=MARKET_TIMEZONE_OBJ)
POD_ID_STR = "pod_test_01"


def _build_decision_plan_stub(pod_id_str: str = POD_ID_STR) -> DecisionPlan:
    return DecisionPlan(
        release_id_str="user_001.pod_test.daily.v2",
        user_id_str="user_001",
        pod_id_str=pod_id_str,
        account_route_str="DU1",
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        decision_base_position_map={},
        snapshot_metadata_dict={"strategy_family_str": "stub"},
        strategy_state_dict={"trade_id_int": 1},
        decision_book_type_str="incremental_entry_exit_book",
        entry_target_weight_map_dict={"AAPL": 0.2},
        exit_asset_set={"MSFT"},
        entry_priority_list=["AAPL"],
    )


def _insert_plans(
    state_store_obj: LiveStateStore,
    pod_id_str: str = POD_ID_STR,
    with_vplan_bool: bool = True,
) -> tuple[DecisionPlan, VPlan | None]:
    decision_plan_obj = state_store_obj.insert_decision_plan(
        _build_decision_plan_stub(pod_id_str=pod_id_str)
    )
    if not with_vplan_bool:
        return decision_plan_obj, None
    # The state store does not preserve row insertion order; the export sorts
    # rows into manual-execution order itself (SELL, then BUY, then HOLD).
    vplan_obj = state_store_obj.insert_vplan(
        VPlan(
            release_id_str=decision_plan_obj.release_id_str,
            user_id_str=decision_plan_obj.user_id_str,
            pod_id_str=decision_plan_obj.pod_id_str,
            account_route_str=decision_plan_obj.account_route_str,
            decision_plan_id_int=int(decision_plan_obj.decision_plan_id_int or 0),
            signal_timestamp_ts=decision_plan_obj.signal_timestamp_ts,
            submission_timestamp_ts=decision_plan_obj.submission_timestamp_ts,
            target_execution_timestamp_ts=decision_plan_obj.target_execution_timestamp_ts,
            execution_policy_str=decision_plan_obj.execution_policy_str,
            broker_snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            live_reference_snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            live_price_source_str="stub",
            net_liq_float=10000.0,
            available_funds_float=10000.0,
            excess_liquidity_float=10000.0,
            pod_budget_fraction_float=0.5,
            pod_budget_float=5000.0,
            current_broker_position_map={"MSFT": 5.0, "SPY": 3.0},
            live_reference_price_map={"AAPL": 100.0, "MSFT": 50.0, "SPY": 200.0},
            target_share_map={"AAPL": 10.0, "MSFT": 0.0, "SPY": 3.0},
            order_delta_map={"AAPL": 10.0, "MSFT": -5.0, "SPY": 0.0},
            vplan_row_list=[
                VPlanRow(
                    asset_str="MSFT",
                    current_share_float=5.0,
                    target_share_float=0.0,
                    order_delta_share_float=-5.0,
                    live_reference_price_float=50.0,
                    estimated_target_notional_float=0.0,
                    broker_order_type_str="MOO",
                ),
                VPlanRow(
                    asset_str="AAPL",
                    current_share_float=0.0,
                    target_share_float=10.0,
                    order_delta_share_float=10.0,
                    live_reference_price_float=100.0,
                    estimated_target_notional_float=1000.0,
                    broker_order_type_str="MOO",
                ),
                VPlanRow(
                    asset_str="SPY",
                    current_share_float=3.0,
                    target_share_float=3.0,
                    order_delta_share_float=0.0,
                    live_reference_price_float=200.0,
                    estimated_target_notional_float=600.0,
                    broker_order_type_str="MOO",
                ),
            ],
            status_str="ready",
        )
    )
    return decision_plan_obj, vplan_obj


@pytest.fixture(name="state_store_obj")
def fixture_state_store_obj(tmp_path) -> LiveStateStore:
    return LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))


def test_build_trade_sheet_orders_sorted_sell_buy_hold(state_store_obj) -> None:
    _insert_plans(state_store_obj)

    trade_sheet_dict = build_trade_sheet_data(
        state_store_obj=state_store_obj,
        pod_id_str=POD_ID_STR,
        generated_at_ts=GENERATED_AT_TS,
    )

    orders_df = trade_sheet_dict["orders_df"]
    assert list(orders_df["asset_str"]) == ["MSFT", "AAPL", "SPY"]
    assert list(orders_df["side_str"]) == ["SELL", "BUY", "HOLD"]
    assert list(orders_df["order_delta_share_float"]) == [-5.0, 10.0, 0.0]
    # delta notional = delta * live reference price (sell shows negative).
    assert list(orders_df["estimated_delta_notional_float"]) == [-250.0, 1000.0, 0.0]
    assert list(orders_df["broker_order_type_str"]) == ["MOO", "MOO", "MOO"]


def test_build_trade_sheet_decision_tab_has_entries_and_exits(state_store_obj) -> None:
    _insert_plans(state_store_obj)

    trade_sheet_dict = build_trade_sheet_data(
        state_store_obj=state_store_obj,
        pod_id_str=POD_ID_STR,
        generated_at_ts=GENERATED_AT_TS,
    )

    decision_df = trade_sheet_dict["decision_df"]
    aapl_row = decision_df[decision_df["asset_str"] == "AAPL"].iloc[0]
    assert aapl_row["role_str"] == "ENTRY"
    assert aapl_row["target_weight_float"] == 0.2
    assert aapl_row["entry_priority_int"] == 0
    msft_row = decision_df[decision_df["asset_str"] == "MSFT"].iloc[0]
    assert msft_row["role_str"] == "EXIT"
    assert msft_row["target_weight_float"] == 0.0


def test_export_writes_xlsx_with_three_tabs(state_store_obj, tmp_path) -> None:
    _insert_plans(state_store_obj)
    output_path_str = str(tmp_path / "sheet.xlsx")

    detail_dict = export_trade_sheet_detail_dict(
        state_store_obj=state_store_obj,
        pod_id_str=POD_ID_STR,
        env_mode_str="paper",
        generated_at_ts=GENERATED_AT_TS,
        output_path_str=output_path_str,
    )

    assert detail_dict["order_count_int"] == 3
    assert detail_dict["decision_only_bool"] is False
    assert detail_dict["vplan_status_str"] == "ready"
    sheet_df_dict = pd.read_excel(output_path_str, sheet_name=None)
    assert set(sheet_df_dict.keys()) == {"Orders", "Decision", "Context"}
    assert list(sheet_df_dict["Orders"]["asset_str"]) == ["MSFT", "AAPL", "SPY"]
    context_field_list = list(sheet_df_dict["Context"]["field_str"])
    assert "pod_id" in context_field_list
    assert "vplan_status" in context_field_list
    assert "net_liq" in context_field_list


def test_export_default_path_uses_mode_pod_and_timestamp(state_store_obj, tmp_path) -> None:
    _insert_plans(state_store_obj)

    detail_dict = export_trade_sheet_detail_dict(
        state_store_obj=state_store_obj,
        pod_id_str=POD_ID_STR,
        env_mode_str="paper",
        generated_at_ts=GENERATED_AT_TS,
        output_dir_str=str(tmp_path / "results"),
    )

    output_path_obj = Path(str(detail_dict["output_path_str"]))
    assert output_path_obj.exists()
    assert output_path_obj.parent == tmp_path / "results" / "trade_sheets" / "paper" / POD_ID_STR
    # 2024-02-01 09:25 America/New_York == 14:25 UTC.
    assert output_path_obj.name == "trade_sheet_20240201T142500Z.xlsx"


def test_decision_only_export_has_empty_orders(state_store_obj, tmp_path) -> None:
    _insert_plans(state_store_obj, with_vplan_bool=False)
    output_path_str = str(tmp_path / "sheet.xlsx")

    detail_dict = export_trade_sheet_detail_dict(
        state_store_obj=state_store_obj,
        pod_id_str=POD_ID_STR,
        env_mode_str="paper",
        generated_at_ts=GENERATED_AT_TS,
        output_path_str=output_path_str,
    )

    assert detail_dict["decision_only_bool"] is True
    assert detail_dict["order_count_int"] == 0
    assert detail_dict["vplan_id_int"] is None
    sheet_df_dict = pd.read_excel(output_path_str, sheet_name=None)
    assert len(sheet_df_dict["Orders"]) == 0
    assert "AAPL" in list(sheet_df_dict["Decision"]["asset_str"])


def test_full_target_weight_book_renders_target_role(state_store_obj, tmp_path) -> None:
    decision_plan_obj = state_store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str="user_001.pod_taa.monthly.v2",
            user_id_str="user_001",
            pod_id_str=POD_ID_STR,
            account_route_str="DU1",
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_month_first_open",
            decision_base_position_map={},
            snapshot_metadata_dict={},
            strategy_state_dict={},
            decision_book_type_str="full_target_weight_book",
            full_target_weight_map_dict={"SPY": 0.6, "TLT": 0.4},
        )
    )
    assert decision_plan_obj.decision_plan_id_int is not None
    output_path_str = str(tmp_path / "full_book_sheet.xlsx")

    detail_dict = export_trade_sheet_detail_dict(
        state_store_obj=state_store_obj,
        pod_id_str=POD_ID_STR,
        env_mode_str="paper",
        generated_at_ts=GENERATED_AT_TS,
        output_path_str=output_path_str,
    )

    assert detail_dict["decision_only_bool"] is True
    sheet_df_dict = pd.read_excel(output_path_str, sheet_name=None)
    decision_df = sheet_df_dict["Decision"]
    assert list(decision_df["asset_str"]) == ["SPY", "TLT"]
    assert set(decision_df["role_str"]) == {"TARGET"}
    assert list(decision_df["target_weight_float"]) == [0.6, 0.4]


def test_context_handles_none_available_funds(state_store_obj, tmp_path) -> None:
    decision_plan_obj, vplan_obj = _insert_plans(state_store_obj, with_vplan_bool=False)
    state_store_obj.insert_vplan(
        VPlan(
            release_id_str=decision_plan_obj.release_id_str,
            user_id_str=decision_plan_obj.user_id_str,
            pod_id_str=decision_plan_obj.pod_id_str,
            account_route_str=decision_plan_obj.account_route_str,
            decision_plan_id_int=int(decision_plan_obj.decision_plan_id_int or 0),
            signal_timestamp_ts=decision_plan_obj.signal_timestamp_ts,
            submission_timestamp_ts=decision_plan_obj.submission_timestamp_ts,
            target_execution_timestamp_ts=decision_plan_obj.target_execution_timestamp_ts,
            execution_policy_str=decision_plan_obj.execution_policy_str,
            broker_snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            live_reference_snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            live_price_source_str="stub",
            net_liq_float=10000.0,
            available_funds_float=None,
            excess_liquidity_float=None,
            pod_budget_fraction_float=0.5,
            pod_budget_float=5000.0,
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
                )
            ],
            status_str="ready",
        )
    )
    output_path_str = str(tmp_path / "none_funds_sheet.xlsx")

    export_trade_sheet_detail_dict(
        state_store_obj=state_store_obj,
        pod_id_str=POD_ID_STR,
        env_mode_str="paper",
        generated_at_ts=GENERATED_AT_TS,
        output_path_str=output_path_str,
    )

    context_df = pd.read_excel(output_path_str, sheet_name="Context")
    available_funds_row = context_df[context_df["field_str"] == "available_funds"]
    assert len(available_funds_row) == 1
    # None renders as an empty cell, not the string "None".
    value_obj = available_funds_row.iloc[0]["value_str"]
    assert pd.isna(value_obj) or str(value_obj) == ""


def test_empty_db_raises_plain_language_error(state_store_obj) -> None:
    with pytest.raises(ValueError, match="nothing to export"):
        build_trade_sheet_data(
            state_store_obj=state_store_obj,
            pod_id_str=POD_ID_STR,
            generated_at_ts=GENERATED_AT_TS,
        )


def test_vplan_id_pod_mismatch_is_refused(state_store_obj) -> None:
    _, vplan_obj = _insert_plans(state_store_obj, pod_id_str="pod_other_01")

    with pytest.raises(ValueError, match="belongs to pod"):
        build_trade_sheet_data(
            state_store_obj=state_store_obj,
            pod_id_str=POD_ID_STR,
            generated_at_ts=GENERATED_AT_TS,
            vplan_id_int=int(vplan_obj.vplan_id_int or 0),
        )


def test_runner_cli_exports_trade_sheet(state_store_obj, tmp_path, monkeypatch) -> None:
    _insert_plans(state_store_obj)
    output_path_str = str(tmp_path / "cli_sheet.xlsx")
    # Keep the test hermetic: do not load the developer's real config.env.
    monkeypatch.setattr(runner_module, "load_config_env_file", lambda **kwargs: {})
    monkeypatch.setattr(
        runner_module,
        "default_config_env_path_obj",
        lambda: tmp_path / "config.env",
    )

    exit_code_int = main(
        [
            "export_trade_sheet",
            "--mode",
            "paper",
            "--pod-id",
            POD_ID_STR,
            "--db-path",
            state_store_obj.db_path_str,
            "--output-path",
            output_path_str,
            "--log-path",
            str(tmp_path / "events.jsonl"),
        ]
    )

    assert exit_code_int == 0
    assert Path(output_path_str).exists()


def test_runner_cli_requires_pod_id(state_store_obj, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "load_config_env_file", lambda **kwargs: {})
    monkeypatch.setattr(
        runner_module,
        "default_config_env_path_obj",
        lambda: tmp_path / "config.env",
    )

    with pytest.raises(ValueError, match="requires --pod-id"):
        main(
            [
                "export_trade_sheet",
                "--mode",
                "paper",
                "--db-path",
                state_store_obj.db_path_str,
                "--log-path",
                str(tmp_path / "events.jsonl"),
            ]
        )
