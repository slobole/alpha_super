from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from alpha.live.models import LiveRelease
from alpha.live.strategy_host import build_decision_plan_for_release


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def make_release() -> LiveRelease:
    return LiveRelease(
        release_id_str="release::dv2",
        user_id_str="user_001",
        pod_id_str="pod_001",
        account_route_str="DU1",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        mode_str="paper",
        session_calendar_id_str="XNYS",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
        data_profile_str="norgate_eod_sp500_pit",
        params_dict={"capital_base_float": 100000.0, "max_positions_int": 1},
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
