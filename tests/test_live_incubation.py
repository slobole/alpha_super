from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from alpha.live.incubation import IncubationBrokerAdapter
from alpha.live.models import DecisionPlan, SessionOpenPrice
from alpha.live.release_manifest import load_release_list
from alpha.live.runner import (
    DEFAULT_DB_PATH_STR,
    DEFAULT_INCUBATION_DB_PATH_STR,
    _resolve_db_path_for_mode_str,
    build_vplans,
    get_compare_reference_summary,
    post_execution_reconcile,
    submit_ready_vplans,
)
from alpha.live.state_store_v2 import LiveStateStore


MARKET_TZ = ZoneInfo("America/New_York")


def _write_incubation_release(
    root_path_obj: Path,
    *,
    execution_policy_str: str = "next_open_moo",
) -> Path:
    release_path_obj = root_path_obj / "releases" / "user_001" / "pod_inc.yaml"
    release_path_obj.parent.mkdir(parents=True, exist_ok=True)
    release_path_obj.write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: user_001.pod_inc.incubation",
                "  user_id: user_001",
                "  pod_id: pod_inc",
                "deployment:",
                "  mode: incubation",
                "  enabled_bool: true",
                "broker:",
                "  account_route: SIM_pod_inc",
                "strategy:",
                "  strategy_import_str: strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy",
                "  data_profile_str: norgate_eod_sp500_pit",
                "params:",
                "  capital_base_float: 100000.0",
                "risk:",
                "  risk_profile_str: test",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                "  signal_clock_str: eod_snapshot_ready",
                f"  execution_policy_str: {execution_policy_str}",
                "execution:",
                "  pod_budget_fraction_float: 1.0",
                "  auto_submit_enabled_bool: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return release_path_obj


def _official_price_lookup_func(
    asset_str_list: list[str],
    session_date_str: str,
    price_field_str: str,
) -> dict[str, float]:
    price_map_dict = {
        ("2024-01-02", "Close", "AAPL"): 100.0,
    }
    official_price_map_dict: dict[str, float] = {}
    for asset_str in asset_str_list:
        key_tup = (session_date_str, price_field_str, asset_str)
        if key_tup not in price_map_dict:
            raise RuntimeError(f"missing test price {key_tup}")
        official_price_map_dict[asset_str] = float(price_map_dict[key_tup])
    return official_price_map_dict


def _ibkr_tick_open_lookup_func(
    account_route_str: str,
    asset_str_list: list[str],
    session_open_timestamp_ts: datetime,
    session_calendar_id_str: str,
) -> list[SessionOpenPrice]:
    del session_calendar_id_str
    session_date_str = session_open_timestamp_ts.astimezone(MARKET_TZ).date().isoformat()
    price_map_dict = {
        ("2024-01-03", "AAPL"): 101.0,
    }
    return [
        SessionOpenPrice(
            session_date_str=session_date_str,
            account_route_str=account_route_str,
            asset_str=str(asset_str),
            official_open_price_float=price_map_dict.get((session_date_str, str(asset_str))),
            open_price_source_str=(
                "ibkr.tick_open"
                if (session_date_str, str(asset_str)) in price_map_dict
                else None
            ),
            snapshot_timestamp_ts=session_open_timestamp_ts,
            raw_payload_dict={},
        )
        for asset_str in asset_str_list
    ]


def _insert_open_decision_plan(state_store_obj: LiveStateStore, root_path_obj: Path) -> None:
    release_obj = load_release_list(str(root_path_obj / "releases"))[0]
    state_store_obj.upsert_release(release_obj)
    state_store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=datetime(2024, 1, 2, 16, 0, tzinfo=MARKET_TZ),
            submission_timestamp_ts=datetime(2024, 1, 3, 9, 23, 30, tzinfo=MARKET_TZ),
            target_execution_timestamp_ts=datetime(2024, 1, 3, 9, 30, tzinfo=MARKET_TZ),
            execution_policy_str=release_obj.execution_policy_str,
            decision_base_position_map={},
            snapshot_metadata_dict={},
            strategy_state_dict={"cycle_int": 1},
            entry_target_weight_map_dict={"AAPL": 0.5},
            entry_priority_list=["AAPL"],
        )
    )


def test_incubation_manifest_accepts_sim_account_only_in_incubation(tmp_path: Path):
    _write_incubation_release(tmp_path)
    release_list = load_release_list(str(tmp_path / "releases"))

    assert release_list[0].mode_str == "incubation"
    assert release_list[0].account_route_str == "SIM_pod_inc"

    paper_release_path_obj = tmp_path / "releases" / "user_001" / "paper_bad.yaml"
    paper_release_path_obj.write_text(
        (tmp_path / "releases" / "user_001" / "pod_inc.yaml").read_text(encoding="utf-8").replace(
            "mode: incubation",
            "mode: paper",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="virtual incubation route"):
        load_release_list(str(tmp_path / "releases"))


def test_incubation_uses_separate_default_db_path():
    assert _resolve_db_path_for_mode_str(None, "incubation") == DEFAULT_INCUBATION_DB_PATH_STR
    assert _resolve_db_path_for_mode_str(None, "paper") == DEFAULT_DB_PATH_STR
    assert _resolve_db_path_for_mode_str(None, "live") == DEFAULT_DB_PATH_STR

    with pytest.raises(ValueError, match="paper/live default DB"):
        _resolve_db_path_for_mode_str(DEFAULT_DB_PATH_STR, "incubation")
    with pytest.raises(ValueError, match="incubation default DB"):
        _resolve_db_path_for_mode_str(DEFAULT_INCUBATION_DB_PATH_STR, "paper")


def test_incubation_settles_open_order_with_official_open_and_cash_ledger(tmp_path: Path):
    _write_incubation_release(tmp_path)
    state_store_obj = LiveStateStore(str(tmp_path / "live.sqlite3"))
    _insert_open_decision_plan(state_store_obj, tmp_path)

    build_as_of_ts = datetime(2024, 1, 3, 9, 25, tzinfo=MARKET_TZ)
    build_adapter_obj = IncubationBrokerAdapter(
        state_store_obj=state_store_obj,
        as_of_ts=build_as_of_ts,
        official_price_lookup_func=_official_price_lookup_func,
    )
    build_detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=build_adapter_obj,
        as_of_ts=build_as_of_ts,
        env_mode_str="incubation",
    )
    assert build_detail_dict["created_vplan_count_int"] == 1

    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_inc")
    assert latest_vplan_obj is not None
    assert latest_vplan_obj.live_reference_price_map == {"AAPL": 100.0}
    assert latest_vplan_obj.order_delta_map == {"AAPL": 500.0}

    submit_as_of_ts = datetime(2024, 1, 3, 9, 26, tzinfo=MARKET_TZ)
    submit_adapter_obj = IncubationBrokerAdapter(
        state_store_obj=state_store_obj,
        as_of_ts=submit_as_of_ts,
        official_price_lookup_func=_official_price_lookup_func,
    )
    submit_detail_dict = submit_ready_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=submit_adapter_obj,
        as_of_ts=submit_as_of_ts,
        env_mode_str="incubation",
        manual_only_bool=False,
        vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
    )
    assert submit_detail_dict["submitted_vplan_count_int"] == 1
    assert state_store_obj.get_fill_row_dict_list_for_vplan(int(latest_vplan_obj.vplan_id_int or 0)) == []
    assert state_store_obj.get_broker_order_row_dict_list_for_vplan(
        int(latest_vplan_obj.vplan_id_int or 0)
    )[0]["status_str"] == "PendingSubmit"

    early_reconcile_adapter_obj = IncubationBrokerAdapter(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 3, 9, 29, tzinfo=MARKET_TZ),
        official_price_lookup_func=_official_price_lookup_func,
    )
    early_reconcile_detail_dict = post_execution_reconcile(
        state_store_obj=state_store_obj,
        broker_adapter_obj=early_reconcile_adapter_obj,
        as_of_ts=datetime(2024, 1, 3, 9, 29, tzinfo=MARKET_TZ),
        env_mode_str="incubation",
    )
    assert early_reconcile_detail_dict["completed_vplan_count_int"] == 0

    settle_as_of_ts = datetime(2024, 1, 3, 9, 31, tzinfo=MARKET_TZ)
    settle_adapter_obj = IncubationBrokerAdapter(
        state_store_obj=state_store_obj,
        as_of_ts=settle_as_of_ts,
        official_price_lookup_func=_official_price_lookup_func,
        ibkr_tick_open_lookup_func=_ibkr_tick_open_lookup_func,
    )
    reconcile_detail_dict = post_execution_reconcile(
        state_store_obj=state_store_obj,
        broker_adapter_obj=settle_adapter_obj,
        as_of_ts=settle_as_of_ts,
        env_mode_str="incubation",
    )
    assert reconcile_detail_dict["completed_vplan_count_int"] == 1

    fill_row_dict_list = state_store_obj.get_fill_row_dict_list_for_vplan(int(latest_vplan_obj.vplan_id_int or 0))
    assert len(fill_row_dict_list) == 1
    assert fill_row_dict_list[0]["fill_amount_float"] == 500.0
    assert fill_row_dict_list[0]["fill_price_float"] == 101.0
    assert fill_row_dict_list[0]["official_open_price_float"] == 101.0
    assert fill_row_dict_list[0]["open_price_source_str"] == "ibkr.tick_open"

    session_open_price_map_dict = state_store_obj.get_session_open_price_map_dict(
        account_route_str="SIM_pod_inc",
        session_date_str="2024-01-03",
    )
    assert session_open_price_map_dict["AAPL"].official_open_price_float == 101.0
    assert session_open_price_map_dict["AAPL"].open_price_source_str == "ibkr.tick_open"

    cash_ledger_row_dict_list = state_store_obj.get_cash_ledger_row_dict_list_for_vplan(
        int(latest_vplan_obj.vplan_id_int or 0)
    )
    assert [row_dict["entry_type_str"] for row_dict in cash_ledger_row_dict_list] == [
        "trade_notional",
        "commission",
    ]
    assert sum(row_dict["cash_delta_float"] for row_dict in cash_ledger_row_dict_list) == -50502.5

    pod_state_obj = state_store_obj.get_pod_state("pod_inc")
    assert pod_state_obj is not None
    assert pod_state_obj.position_amount_map == {"AAPL": 500.0}
    assert pod_state_obj.cash_float == 49497.5

    second_reconcile_adapter_obj = IncubationBrokerAdapter(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 3, 9, 32, tzinfo=MARKET_TZ),
        official_price_lookup_func=_official_price_lookup_func,
        ibkr_tick_open_lookup_func=_ibkr_tick_open_lookup_func,
    )
    second_reconcile_detail_dict = post_execution_reconcile(
        state_store_obj=state_store_obj,
        broker_adapter_obj=second_reconcile_adapter_obj,
        as_of_ts=datetime(2024, 1, 3, 9, 32, tzinfo=MARKET_TZ),
        env_mode_str="incubation",
    )
    assert second_reconcile_detail_dict["completed_vplan_count_int"] == 0
    assert len(state_store_obj.get_fill_row_dict_list_for_vplan(int(latest_vplan_obj.vplan_id_int or 0))) == 1
    assert len(state_store_obj.get_cash_ledger_row_dict_list_for_vplan(int(latest_vplan_obj.vplan_id_int or 0))) == 2

    compare_summary_dict = get_compare_reference_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 3, 9, 32, tzinfo=MARKET_TZ),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="incubation",
        pod_id_str="pod_inc",
    )
    compare_row_dict = compare_summary_dict["compare_report_dict_list"][0]["compare_row_dict_list"][0]
    assert compare_row_dict["quantity_diff_float"] == 0.0
    assert compare_row_dict["position_diff_float"] == 0.0
    assert compare_row_dict["fill_slippage_bps_float"] == 0.0


def test_incubation_missing_ibkr_tick_open_writes_no_accounting(tmp_path: Path):
    _write_incubation_release(tmp_path)
    state_store_obj = LiveStateStore(str(tmp_path / "live.sqlite3"))
    _insert_open_decision_plan(state_store_obj, tmp_path)

    build_as_of_ts = datetime(2024, 1, 3, 9, 25, tzinfo=MARKET_TZ)
    build_adapter_obj = IncubationBrokerAdapter(
        state_store_obj=state_store_obj,
        as_of_ts=build_as_of_ts,
        official_price_lookup_func=_official_price_lookup_func,
    )
    build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=build_adapter_obj,
        as_of_ts=build_as_of_ts,
        env_mode_str="incubation",
    )
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_inc")
    assert latest_vplan_obj is not None

    submit_adapter_obj = IncubationBrokerAdapter(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 3, 9, 26, tzinfo=MARKET_TZ),
        official_price_lookup_func=_official_price_lookup_func,
    )
    submit_ready_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=submit_adapter_obj,
        as_of_ts=datetime(2024, 1, 3, 9, 26, tzinfo=MARKET_TZ),
        env_mode_str="incubation",
        manual_only_bool=False,
        vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
    )

    def _missing_ibkr_tick_open_lookup_func(
        account_route_str: str,
        asset_str_list: list[str],
        session_open_timestamp_ts: datetime,
        session_calendar_id_str: str,
    ) -> list[SessionOpenPrice]:
        del session_calendar_id_str
        session_date_str = session_open_timestamp_ts.astimezone(MARKET_TZ).date().isoformat()
        return [
            SessionOpenPrice(
                session_date_str=session_date_str,
                account_route_str=account_route_str,
                asset_str=str(asset_str),
                official_open_price_float=None,
                open_price_source_str=None,
                snapshot_timestamp_ts=session_open_timestamp_ts,
                raw_payload_dict={},
            )
            for asset_str in asset_str_list
        ]

    settle_adapter_obj = IncubationBrokerAdapter(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 3, 9, 31, tzinfo=MARKET_TZ),
        official_price_lookup_func=_official_price_lookup_func,
        ibkr_tick_open_lookup_func=_missing_ibkr_tick_open_lookup_func,
    )
    with pytest.raises(RuntimeError, match="Missing IBKR tick-open price"):
        post_execution_reconcile(
            state_store_obj=state_store_obj,
            broker_adapter_obj=settle_adapter_obj,
            as_of_ts=datetime(2024, 1, 3, 9, 31, tzinfo=MARKET_TZ),
            env_mode_str="incubation",
        )

    assert state_store_obj.get_fill_row_dict_list_for_vplan(int(latest_vplan_obj.vplan_id_int or 0)) == []
    assert state_store_obj.get_cash_ledger_row_dict_list_for_vplan(int(latest_vplan_obj.vplan_id_int or 0)) == []
    assert state_store_obj.get_pod_state("pod_inc") is None


def test_incubation_moc_blocks_without_preclose_snapshot(tmp_path: Path):
    _write_incubation_release(tmp_path, execution_policy_str="same_day_moc")
    state_store_obj = LiveStateStore(str(tmp_path / "live.sqlite3"))
    release_obj = load_release_list(str(tmp_path / "releases"))[0]
    state_store_obj.upsert_release(release_obj)
    state_store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=datetime(2024, 1, 2, 15, 45, tzinfo=MARKET_TZ),
            submission_timestamp_ts=datetime(2024, 1, 2, 15, 50, tzinfo=MARKET_TZ),
            target_execution_timestamp_ts=datetime(2024, 1, 2, 16, 0, tzinfo=MARKET_TZ),
            execution_policy_str="same_day_moc",
            decision_base_position_map={},
            snapshot_metadata_dict={},
            strategy_state_dict={},
            entry_target_weight_map_dict={"AAPL": 0.5},
            entry_priority_list=["AAPL"],
        )
    )
    as_of_ts = datetime(2024, 1, 2, 15, 55, tzinfo=MARKET_TZ)
    adapter_obj = IncubationBrokerAdapter(
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
        official_price_lookup_func=_official_price_lookup_func,
    )

    detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=adapter_obj,
        as_of_ts=as_of_ts,
        env_mode_str="incubation",
    )

    assert detail_dict["created_vplan_count_int"] == 0
    assert detail_dict["blocked_action_count_int"] == 1
    assert detail_dict["reason_count_map_dict"] == {"live_price_snapshot_error": 1}
    assert state_store_obj.get_latest_decision_plan_for_pod("pod_inc").status_str == "blocked"
