from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from alpha.live.models import LiveRelease
from alpha.live.strategy_host import build_order_plan_for_release


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def make_release(strategy_import_str: str, signal_clock_str: str, execution_policy_str: str) -> LiveRelease:
    return LiveRelease(
        release_id_str=f"release::{strategy_import_str}",
        user_id_str="user_001",
        pod_id_str="pod_001",
        account_route_str="U1",
        strategy_import_str=strategy_import_str,
        mode_str="paper",
        session_calendar_id_str="XNYS",
        signal_clock_str=signal_clock_str,
        execution_policy_str=execution_policy_str,
        data_profile_str="norgate_eod_sp500_pit",
        params_dict={"capital_base_float": 100000.0, "max_positions_int": 1},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="manifest.yaml",
    )


def make_price_df(
    symbol_list: list[str],
    date_index: pd.DatetimeIndex,
    close_start_float: float = 100.0,
) -> pd.DataFrame:
    frame_list: list[pd.DataFrame] = []
    for symbol_idx_int, symbol_str in enumerate(symbol_list):
        close_ser = pd.Series(
            close_start_float + symbol_idx_int + np.arange(len(date_index), dtype=float) * 0.2,
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


def test_strategy_host_builds_dv2_order_plan(monkeypatch):
    import strategies.dv2.strategy_mr_dv2 as dv2_module

    date_index = pd.bdate_range("2023-01-02", periods=260)
    pricing_data_df = make_price_df(["TEST", "$SPX"], date_index)
    universe_df = pd.DataFrame(1, index=date_index, columns=["TEST"])

    monkeypatch.setattr(dv2_module, "build_index_constituent_matrix", lambda indexname: (["TEST"], universe_df))
    monkeypatch.setattr(dv2_module, "get_prices", lambda symbols, benchmarks, start_date, end_date: pricing_data_df)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "compute_signals", lambda self, pricing_data: pricing_data)
    monkeypatch.setattr(dv2_module.DVO2Strategy, "get_opportunities", lambda self, close: ["TEST"])

    release_obj = make_release(
        "strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        "eod_snapshot_ready",
        "next_open_moo",
    )
    order_plan_obj = build_order_plan_for_release(
        release_obj=release_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=None,
    )

    assert order_plan_obj.execution_policy_str == "next_open_moo"
    assert len(order_plan_obj.order_intent_list) == 1
    assert order_plan_obj.order_intent_list[0].broker_order_type_str == "MOO"


def test_strategy_host_builds_taa_order_plan(monkeypatch):
    import strategies.taa_df.strategy_taa_df as base_taa_module
    import strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash as variant_module
    import strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils as vix_overlay_module

    signal_date_ts = pd.Timestamp("2024-01-31")
    execution_price_df = make_price_df(
        ["GLD", "UUP", "TLT", "DBC", "BTAL", "TQQQ", "$SPX"],
        pd.DatetimeIndex([signal_date_ts]),
    )
    base_month_end_weight_df = pd.DataFrame(
        {
            "GLD": [0.4],
            "UUP": [0.0],
            "TLT": [0.3],
            "DBC": [0.0],
            "BTAL": [0.0],
            "TQQQ": [0.3],
        },
        index=pd.DatetimeIndex([signal_date_ts]),
        dtype=float,
    )
    month_end_vrp_signal_df = pd.DataFrame(
        {
            "rv20_ann_pct": [12.0],
            "vix_close": [18.0],
            "vrp_gate": [1.0],
        },
        index=pd.DatetimeIndex([signal_date_ts]),
        dtype=float,
    )

    monkeypatch.setattr(
        base_taa_module,
        "get_defense_first_data",
        lambda config_obj: (execution_price_df, pd.DataFrame(), base_month_end_weight_df, pd.DataFrame()),
    )
    monkeypatch.setattr(
        vix_overlay_module,
        "_load_vrp_overlay_signal_frames",
        lambda config_obj: (pd.DataFrame(), month_end_vrp_signal_df),
    )

    release_obj = make_release(
        "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
        "month_end_snapshot_ready",
        "next_month_first_open",
    )
    order_plan_obj = build_order_plan_for_release(
        release_obj=release_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=None,
    )

    assert order_plan_obj.execution_policy_str == "next_month_first_open"
    assert len(order_plan_obj.order_intent_list) > 0
    assert order_plan_obj.snapshot_metadata_dict["strategy_family_str"] == "taa_df_btal_fallback_tqqq_vix_cash"


def test_strategy_host_builds_atr_normalized_ndx_order_plan(monkeypatch):
    import strategies.momentum.strategy_mo_atr_normalized_ndx as atr_module

    date_index = pd.bdate_range("2023-01-02", periods=260)
    pricing_data_df = make_price_df(["AAA", "SPY"], date_index)
    universe_df = pd.DataFrame(1, index=date_index, columns=["AAA"])
    signal_date_ts = pd.Timestamp(date_index[-1])

    def compute_signal_table_stub(price_close_df, price_high_df, price_low_df, regime_close_ser, config):
        monthly_index = pd.DatetimeIndex([signal_date_ts])
        monthly_decision_close_df = pd.DataFrame({"AAA": [float(price_close_df["AAA"].iloc[-1])]}, index=monthly_index)
        monthly_roc_df = pd.DataFrame({"AAA": [0.2]}, index=monthly_index)
        atr_decision_df = pd.DataFrame({"AAA": [1.0]}, index=monthly_index)
        stock_trend_pass_df = pd.DataFrame({"AAA": [True]}, index=monthly_index)
        regime_sma_ser = pd.Series([100.0], index=monthly_index)
        regime_pass_ser = pd.Series([True], index=monthly_index)
        risk_adj_score_df = pd.DataFrame({"AAA": [0.2]}, index=monthly_index)
        return (
            monthly_decision_close_df,
            monthly_roc_df,
            atr_decision_df,
            stock_trend_pass_df,
            regime_sma_ser,
            regime_pass_ser,
            risk_adj_score_df,
        )

    monkeypatch.setattr(
        atr_module,
        "get_atr_normalized_ndx_data",
        lambda config_obj: (pricing_data_df, universe_df, pd.DataFrame()),
    )
    monkeypatch.setattr(atr_module, "compute_atr_normalized_signal_tables", compute_signal_table_stub)

    release_obj = make_release(
        "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy",
        "month_end_snapshot_ready",
        "next_open_moo",
    )
    order_plan_obj = build_order_plan_for_release(
        release_obj=release_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=None,
    )

    assert order_plan_obj.execution_policy_str == "next_open_moo"
    assert len(order_plan_obj.order_intent_list) == 1
    assert order_plan_obj.order_intent_list[0].asset_str == "AAA"
