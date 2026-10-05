"""Saved-fill reporting separates opening slippage from a prior-close gap."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from alpha.live import runner
from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
from alpha.live.models import BrokerOrderFill, DecisionPlan, VPlan, VPlanRow
from alpha.live.state_store_v2 import LiveStateStore
from test_live_reference_compare import _build_release


STRATEGY_TUPLE = (
    CORE5_STRATEGY_IMPORT_STR,
    "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy",
    "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
)


def _report_row(tmp_path, monkeypatch, strategy_str, official_open_float, amount_float,
        *, legacy_quote_bool=False):
    store_obj = LiveStateStore(str(tmp_path / "report.sqlite3"))
    release_obj = _build_release(strategy_import_str=strategy_str)
    store_obj.upsert_release(release_obj)
    signal_ts = datetime(2024, 2, 1, 16, tzinfo=ZoneInfo("America/New_York"))
    execution_ts = datetime(2024, 2, 2, 9, 30, tzinfo=signal_ts.tzinfo)
    identity_dict = {"release_id_str": release_obj.release_id_str, "user_id_str": release_obj.user_id_str,
        "pod_id_str": release_obj.pod_id_str, "account_route_str": release_obj.account_route_str,
        "signal_timestamp_ts": signal_ts, "submission_timestamp_ts": execution_ts - timedelta(minutes=10),
        "target_execution_timestamp_ts": execution_ts, "execution_policy_str": "next_open_moo"}
    core5_bool = strategy_str == CORE5_STRATEGY_IMPORT_STR
    metadata_dict = {"sizing_close_price_map_dict": {"DBC": 100.0}} if core5_bool else {}
    decision_obj = store_obj.insert_decision_plan(DecisionPlan(**identity_dict,
        decision_base_position_map={}, snapshot_metadata_dict=metadata_dict, strategy_state_dict={},
        decision_book_type_str="full_target_weight_book", full_target_weight_map_dict={"DBC": 0.2}))
    reference_float = 101.0 if legacy_quote_bool else 100.0
    source_str = "core5.frozen_close_t" if core5_bool and not legacy_quote_bool else "fixture.quote"
    vplan_obj = store_obj.insert_vplan(VPlan(**identity_dict,
        decision_plan_id_int=decision_obj.decision_plan_id_int,
        broker_snapshot_timestamp_ts=identity_dict["submission_timestamp_ts"],
        live_reference_snapshot_timestamp_ts=signal_ts, live_price_source_str=source_str,
        net_liq_float=5000.0, available_funds_float=5000.0, excess_liquidity_float=5000.0,
        pod_budget_fraction_float=1.0, pod_budget_float=5000.0, current_broker_position_map={},
        live_reference_price_map={"DBC": reference_float}, target_share_map={"DBC": amount_float},
        order_delta_map={"DBC": amount_float}, vplan_row_list=[VPlanRow(asset_str="DBC",
            current_share_float=0.0, target_share_float=amount_float, order_delta_share_float=amount_float,
            live_reference_price_float=reference_float, estimated_target_notional_float=1000.0,
            broker_order_type_str="MOO", live_reference_source_str=source_str)]))
    store_obj.upsert_vplan_fill_list([BrokerOrderFill(broker_order_id_str="offline_order",
        decision_plan_id_int=decision_obj.decision_plan_id_int, vplan_id_int=vplan_obj.vplan_id_int,
        account_route_str=release_obj.account_route_str, asset_str="DBC", fill_amount_float=amount_float,
        fill_price_float=102.0, fill_timestamp_ts=execution_ts, raw_payload_dict={"exec_id_str": "offline_fill"},
        official_open_price_float=official_open_float,
        open_price_source_str="fixture.official_open" if official_open_float is not None else None)])
    monkeypatch.setattr(runner, "_load_backtest_reference_maps_dict", lambda _path_str: {
        "transaction_map_dict": {}, "equity_by_date_dict": {}, "cash_by_date_dict": {}})
    release_root_obj = tmp_path / "empty_releases"
    release_root_obj.mkdir()
    report_dict = runner.get_compare_reference_summary(store_obj, execution_ts + timedelta(hours=8),
        str(release_root_obj), reference_strategy_pickle_path_str="offline_saved_reference.pkl")
    return report_dict["compare_report_dict_list"][0]["compare_row_dict_list"][0]


@pytest.mark.parametrize("strategy_str", STRATEGY_TUPLE)
@pytest.mark.parametrize("official_open_float", [None, 101.0])
@pytest.mark.parametrize("amount_float", [10.0, -10.0])
def test_close_gap_is_never_reported_as_core5_opening_slippage(
        tmp_path, monkeypatch, strategy_str, official_open_float, amount_float):
    row_dict = _report_row(tmp_path, monkeypatch, strategy_str, official_open_float, amount_float)
    side_float = 1.0 if amount_float > 0 else -1.0
    close_gap_float = 200.0 * side_float
    opening_slippage_float = ((102.0 / 101.0) - 1.0) * 10000.0 * side_float
    assert row_dict["vplan_reference_slippage_bps_float"] == pytest.approx(close_gap_float)
    if official_open_float is None:
        assert row_dict["official_open_slippage_bps_float"] is None
        assert row_dict["reference_price_float"] == 100.0
    else:
        assert row_dict["official_open_slippage_bps_float"] == pytest.approx(opening_slippage_float)
        assert row_dict["fill_slippage_bps_float"] == pytest.approx(opening_slippage_float)
        assert row_dict["reference_price_float"] == 101.0
    if strategy_str == CORE5_STRATEGY_IMPORT_STR:
        assert row_dict["frozen_close_deviation_bps_float"] == pytest.approx(close_gap_float)
        assert row_dict["reference_price_source_str"] == (
            "official_session_open" if official_open_float is not None else "core5.frozen_close_t")
        if official_open_float is None:
            assert row_dict["fill_slippage_bps_float"] is None
    else:
        assert "frozen_close_deviation_bps_float" not in row_dict
        assert "reference_price_source_str" not in row_dict
        if official_open_float is None:
            assert row_dict["fill_slippage_bps_float"] == pytest.approx(close_gap_float)


def test_legacy_core5_quote_reference_keeps_its_source_and_distinct_close_gap(tmp_path, monkeypatch):
    row_dict = _report_row(tmp_path, monkeypatch, CORE5_STRATEGY_IMPORT_STR, None, 10.0,
        legacy_quote_bool=True)
    assert row_dict["fill_slippage_bps_float"] is None
    assert row_dict["reference_price_source_str"] == "fixture.quote"
    assert row_dict["reference_price_float"] == 101.0
    assert row_dict["frozen_close_deviation_bps_float"] == pytest.approx(200.0)
    assert row_dict["vplan_reference_slippage_bps_float"] == pytest.approx((102.0 / 101.0 - 1.0) * 10000.0)
