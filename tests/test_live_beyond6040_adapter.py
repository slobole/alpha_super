"""Beyond 60/40 trend-tilt (T1) decision adapter: frozen Close_T shares, the BIL reserve, receipts and the first fill."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alpha.live import beyond6040_adapter as adapter_module, scheduler_utils
from alpha.live.models import LiveRelease, PodState
from data.beyond6040_snapshot_contract import BEYOND6040_PROFILE_STR
from strategies.taa_beyond_6040 import strategy_taa_beyond_6040_trend_tilt as tilt_module


ASSET_TUPLE = adapter_module.BEYOND6040_ASSET_TUPLE
RISKY_TUPLE = adapter_module.BEYOND6040_RISK_ASSET_TUPLE
BIL_INCEPTION_STR = "2025-06-02"  # the fixture's BIL inception: decisions before it park the residual in cash


def _time(date_str, hour_int=18):
    return (pd.Timestamp(date_str).tz_localize("America/New_York") + pd.Timedelta(hours=hour_int)).to_pydatetime()


@pytest.fixture
def release_obj():
    return LiveRelease(
        release_id_str="beyond6040.test.v1", user_id_str="test", pod_id_str="beyond6040_test",
        account_route_str="DU_BEYOND6040", strategy_import_str=adapter_module.BEYOND6040_STRATEGY_IMPORT_STR,
        mode_str="paper", session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo", data_profile_str=BEYOND6040_PROFILE_STR, params_dict={},
        risk_profile_str="beyond6040_long_only", enabled_bool=True, source_path_str="test.yaml",
        pod_budget_fraction_float=1.0, auto_submit_enabled_bool=False,
    )


@pytest.fixture(scope="module")
def price_df():
    """Exact XNYS sessions; VTI trends up, TLT down, GLD wobbles; BIL has no bar before its inception."""
    date_idx = pd.DatetimeIndex(scheduler_utils.get_exchange_calendar_obj("XNYS").sessions_in_range("2024-01-02", "2026-09-15"))
    bar_vec = np.arange(len(date_idx), dtype=float)
    return_dict = {
        "VTI": 0.0008 + 0.0080 * np.sin(bar_vec / 4.0),
        "GLD": 0.0002 + 0.0040 * np.sin(bar_vec / 6.0 + 0.5),
        "TLT": -0.0006 + 0.0030 * np.cos(bar_vec / 7.0),
        "BIL": np.full(len(bar_vec), 0.00015),
    }
    price_dict = {}
    for asset_int, asset_str in enumerate(ASSET_TUPLE):
        close_vec = (100.0 + 10.0 * asset_int) * np.cumprod(1.0 + return_dict[asset_str])
        if asset_str == "BIL":
            close_vec = np.where(date_idx < pd.Timestamp(BIL_INCEPTION_STR), np.nan, close_vec)
        for field_str, multiplier_float in (("Open", 0.999), ("High", 1.004), ("Low", 0.996), ("Close", 1.0)):
            price_dict[(asset_str, field_str)] = close_vec * multiplier_float
        price_dict[(asset_str, "Dividend")] = np.where(np.isnan(close_vec), np.nan, 0.0)
        if asset_str in RISKY_TUPLE:
            price_dict[(asset_str, "TotalReturnClose")] = close_vec * (1.0 + 0.0001 * bar_vec)
    spx_vec = 4000.0 * np.cumprod(1.0 + 0.0004 + 0.0070 * np.sin(bar_vec / 5.0 + 0.25))
    for field_str in ("Open", "High", "Low", "Close"):
        price_dict[("$SPX", field_str)] = spx_vec
    result_df = pd.DataFrame(price_dict, index=date_idx)
    result_df.columns = pd.MultiIndex.from_tuples(result_df.columns)
    result_df.attrs["norgate_adjustment_by_symbol_dict"] = {**{a: "CAPITALSPECIAL" for a in ASSET_TUPLE}, "$SPX": "TOTALRETURN"}
    result_df.attrs["benchmark_data_symbol_dict"] = {"$SPX": "$SPXTR"}
    return result_df


@pytest.fixture(scope="module")
def engine_obj(price_df):
    """The research engine on the same frame: the oracle for every decision's weights and exposure."""
    with patch.object(tilt_module, "get_trend_tilt_data", return_value=price_df):
        return tilt_module.run_variant(show_display_bool=False, save_results_bool=False)


def _state(release_obj, date_str, position_dict=None, strategy_state_dict=None, cash_float=100_000.0):
    return PodState(
        pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str,
        position_amount_map=position_dict or {}, cash_float=cash_float, total_value_float=999_999.0,
        strategy_state_dict=strategy_state_dict or {}, updated_timestamp_ts=_time(date_str, 17),
        snapshot_stage_str="eod", snapshot_source_str="broker",
    )


def _metadata(date_str, **change_dict):
    return {"norgate_data_source_mode_str": "snapshot", "norgate_data_profile_str": BEYOND6040_PROFILE_STR,
            "norgate_snapshot_date_str": date_str, "norgate_manifest_hash_str": "fixture_hash", **change_dict}


def _build(release_obj, price_df, date_str, state_obj, **metadata_change_dict):
    return adapter_module.build_beyond6040_decision_from_prices(release_obj, _time(date_str), state_obj, price_df, _metadata(date_str, **metadata_change_dict))


def _committed_state_dict(decision_obj):
    return {**decision_obj.strategy_state_dict,
            "beyond6040_execution_receipt_dict": decision_obj.snapshot_metadata_dict["beyond6040_candidate_execution_receipt_dict"]}


def _weights(decision_obj):
    """The plan drops zero weights from its map; read the complete four-asset book."""
    return {a: float(decision_obj.full_target_weight_map_dict.get(a, 0.0)) for a in ASSET_TUPLE}


def _next_session_str(date_str):
    return pd.Timestamp(scheduler_utils.get_exchange_calendar_obj("XNYS").next_session(pd.Timestamp(date_str))).date().isoformat()


def _month_end_session_list(price_df, start_str, end_str):
    idx = price_df.loc[start_str:end_str].index
    return [ts for ts in idx if scheduler_utils.is_last_session_of_month_bool(ts, "XNYS")]


# ---------------------------------------------------------------- release rules
@pytest.mark.parametrize("field_str,value_obj", [
    ("pod_budget_fraction_float", .5), ("execution_policy_str", "next_close_moc"), ("data_profile_str", "norgate_eod_core5"),
    ("signal_clock_str", "month_end_eod_snapshot_ready"), ("session_calendar_id_str", "XLON"), ("params_dict", {"tilt_strength_float": 1.0}),
])
def test_release_requires_frozen_operational_contract(release_obj, field_str, value_obj):
    adapter_module.validate_beyond6040_release(release_obj)
    adapter_module.validate_beyond6040_release(replace(release_obj, params_dict={"capital_base_float": 50_000.0}))
    with pytest.raises(ValueError):
        adapter_module.validate_beyond6040_release(replace(release_obj, **{field_str: value_obj}))


def test_incubation_is_allowed_like_core5(release_obj):
    adapter_module.validate_beyond6040_release(replace(release_obj, mode_str="incubation"))


# ---------------------------------------------------------------- the decision against the engine
def _engine_rows(engine_obj, date_str):
    next_ts = pd.Timestamp(_next_session_str(date_str))
    return engine_obj.daily_target_weights.loc[next_ts], float(engine_obj.daily_exposure_ser.loc[next_ts])


def _decision_dates(price_df):
    month_end_list = _month_end_session_list(price_df, "2025-01-01", "2026-08-31")
    mid_month_list = [price_df.index[price_df.index.get_loc(ts) - 7] for ts in month_end_list]
    return sorted(month_end_list + mid_month_list)


def test_every_decision_equals_the_engines_weights_and_exposure(release_obj, price_df, engine_obj):
    checked_int = month_end_int = 0
    for decision_ts in _decision_dates(price_df):
        date_str = decision_ts.date().isoformat()
        decision_obj = _build(release_obj, price_df, date_str, _state(release_obj, date_str))
        metadata_dict = decision_obj.snapshot_metadata_dict
        engine_weight_ser, engine_exposure_float = _engine_rows(engine_obj, date_str)
        for asset_str in ASSET_TUPLE:
            assert _weights(decision_obj)[asset_str] == pytest.approx(float(engine_weight_ser[asset_str]), abs=1e-12), (date_str, asset_str)
        assert decision_obj.cash_reserve_weight_float == pytest.approx(float(engine_weight_ser["Cash"]), abs=1e-12)
        assert metadata_dict["beyond6040_gross_exposure_float"] == pytest.approx(engine_exposure_float, abs=1e-15)
        assert metadata_dict["month_end_bool"] == scheduler_utils.is_last_session_of_month_bool(decision_ts, "XNYS")
        month_end_int += int(metadata_dict["month_end_bool"])
        checked_int += 1
    assert checked_int >= 30 and month_end_int >= 15


def test_month_end_decision_reads_the_new_schedule_a_cut_history_cannot_show(release_obj, price_df, engine_obj):
    strategy_obj = adapter_module._build_strategy_obj()
    changed_int = 0
    for decision_ts in _month_end_session_list(price_df, "2025-01-01", "2026-08-31"):
        cut_signal_df = strategy_obj.compute_signals(price_df.loc[:decision_ts])
        cut_base_ser = strategy_obj._current_base_weight_ser(cut_signal_df.loc[decision_ts])
        full_base_ser = pd.Series({a: float(engine_obj.compute_signals(price_df).loc[decision_ts, (a, "base_weight_ser")]) for a in RISKY_TUPLE})
        decision_obj = _build(release_obj, price_df, decision_ts.date().isoformat(), _state(release_obj, decision_ts.date().isoformat()))
        base_dict = decision_obj.snapshot_metadata_dict["beyond6040_base_weight_map_dict"]
        assert all(base_dict[a] == pytest.approx(full_base_ser[a], abs=1e-12) for a in RISKY_TUPLE)
        changed_int += int(not np.allclose(cut_base_ser.to_numpy(), full_base_ser.to_numpy()))
    assert changed_int > 0  # the cut history really lacked the month-end's new weights on some sessions


def test_frozen_shares_are_trunc_nav_times_weight_over_close_t(release_obj, price_df):
    date_str = "2026-03-12"
    position_dict = {"VTI": 120.0, "GLD": 30.0, "BIL": 400.0}
    # A pod that has decided before (yesterday), so its holdings are its own; the receipt does not match them.
    prior_obj = _build(release_obj, price_df, "2026-03-11", _state(release_obj, "2026-03-11"))
    decision_obj = _build(release_obj, price_df, date_str, _state(release_obj, date_str, position_dict, _committed_state_dict(prior_obj), 5_000.0))
    metadata_dict = decision_obj.snapshot_metadata_dict
    close_dict = {a: float(price_df.loc[date_str, (a, "Close")]) for a in ASSET_TUPLE}
    nav_float = 5_000.0 + sum(shares_float * close_dict[a] for a, shares_float in position_dict.items())
    assert metadata_dict["sizing_close_nav_float"] == pytest.approx(nav_float, rel=1e-12)
    assert metadata_dict["sizing_close_price_map_dict"] == close_dict
    weight_dict = _weights(decision_obj)
    assert metadata_dict["fixed_target_share_map_dict"] == {a: float(int(nav_float * weight_dict[a] / close_dict[a])) for a in ASSET_TUPLE}
    assert metadata_dict["rebalance_bool"] and not metadata_dict["no_order_bool"]
    assert metadata_dict["beyond6040_reserve_tradable_bool"]
    assert weight_dict["BIL"] == pytest.approx(1.0 - sum(weight_dict[a] for a in RISKY_TUPLE), abs=1e-12)
    assert decision_obj.cash_reserve_weight_float == 0.0
    assert all(weight_float >= 0.0 for weight_float in weight_dict.values())
    assert decision_obj.target_execution_timestamp_ts == _time("2026-03-13", 9) + timedelta(minutes=30)
    assert decision_obj.decision_book_type_str == "full_target_weight_book"
    assert adapter_module.is_beyond6040_decision_bool(decision_obj)
    assert adapter_module.validated_beyond6040_target_share_dict(decision_obj, release_obj, position_dict) == metadata_dict["fixed_target_share_map_dict"]


def test_before_bil_inception_the_residual_is_cash_and_bil_is_not_sized(release_obj, price_df):
    date_str = "2025-04-15"
    assert np.isnan(price_df.loc[date_str, ("BIL", "Close")])
    decision_obj = _build(release_obj, price_df, date_str, _state(release_obj, date_str))
    metadata_dict = decision_obj.snapshot_metadata_dict
    weight_dict = _weights(decision_obj)
    assert not metadata_dict["beyond6040_reserve_tradable_bool"]
    assert weight_dict["BIL"] == 0.0
    assert decision_obj.cash_reserve_weight_float == pytest.approx(1.0 - sum(weight_dict[a] for a in RISKY_TUPLE), abs=1e-12)
    assert decision_obj.cash_reserve_weight_float > 0.0
    assert metadata_dict["sizing_close_price_map_dict"]["BIL"] is None
    assert metadata_dict["fixed_target_share_map_dict"]["BIL"] == 0.0
    assert adapter_module.validated_beyond6040_target_share_dict(decision_obj, release_obj, {}) == metadata_dict["fixed_target_share_map_dict"]
    with pytest.raises(ValueError, match="observed Close_T"):
        _build(release_obj, price_df, date_str, _state(release_obj, date_str, {"BIL": 10.0}))


def test_prices_after_close_t_cannot_change_the_decision(release_obj, price_df):
    date_str = "2026-02-19"
    state_obj = _state(release_obj, date_str)
    shocked_df = price_df.copy()
    shocked_df.loc[shocked_df.index > pd.Timestamp(date_str)] *= 3.0
    shocked_df.attrs = dict(price_df.attrs)
    base_obj = _build(release_obj, price_df, date_str, state_obj)
    shocked_obj = _build(release_obj, shocked_df, date_str, state_obj)
    cut_obj = _build(release_obj, price_df.loc[:date_str].copy(), date_str, state_obj)
    assert (base_obj.snapshot_metadata_dict["fixed_target_share_map_dict"] == shocked_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
            == cut_obj.snapshot_metadata_dict["fixed_target_share_map_dict"])
    assert _weights(base_obj) == _weights(shocked_obj) == _weights(cut_obj)


def test_signal_not_ready_blocks_the_decision(release_obj, price_df):
    with pytest.raises(ValueError, match="not ready"):
        _build(release_obj, price_df, "2024-03-15", _state(release_obj, "2024-03-15"))


# ---------------------------------------------------------------- state, receipts, catch-up, the first fill
def test_no_order_day_and_matching_receipt(release_obj, price_df):
    date_str = "2026-03-12"
    first_obj = _build(release_obj, price_df, date_str, _state(release_obj, date_str))
    share_dict = first_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    position_dict = {a: s for a, s in share_dict.items() if s}
    close_dict = first_obj.snapshot_metadata_dict["sizing_close_price_map_dict"]
    cash_float = 100_000.0 - sum(s * close_dict[a] for a, s in position_dict.items())
    second_obj = _build(release_obj, price_df, date_str, _state(release_obj, date_str, position_dict, _committed_state_dict(first_obj), cash_float))
    metadata_dict = second_obj.snapshot_metadata_dict
    assert metadata_dict["sizing_close_nav_float"] == pytest.approx(100_000.0, rel=1e-12)
    assert metadata_dict["fixed_target_share_map_dict"] == share_dict
    assert metadata_dict["no_order_bool"] and metadata_dict["rebalance_bool"]
    assert not metadata_dict["beyond6040_catch_up_bool"] and not metadata_dict["initialization_bool"]
    # Built on the same session as its cached state, so only the cache-gap note; the receipt itself is read as applied.
    assert metadata_dict["beyond6040_warning_code_list"] == ["beyond6040_strategy_state_cache_gap"]
    assert second_obj.decision_base_position_map == position_dict


def test_next_session_resizes_from_nav_and_reads_its_receipt_back(release_obj, price_df):
    first_str, second_str = "2026-03-12", "2026-03-13"
    first_obj = _build(release_obj, price_df, first_str, _state(release_obj, first_str))
    share_dict = first_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    position_dict = {a: s for a, s in share_dict.items() if s}
    cash_float = 100_000.0 - sum(s * float(price_df.loc[second_str, (a, "Open")]) for a, s in position_dict.items())
    second_obj = _build(release_obj, price_df, second_str, _state(release_obj, second_str, position_dict, _committed_state_dict(first_obj), cash_float))
    metadata_dict = second_obj.snapshot_metadata_dict
    assert not metadata_dict["beyond6040_catch_up_bool"] and metadata_dict["beyond6040_warning_code_list"] == []
    nav_float = cash_float + sum(s * float(price_df.loc[second_str, (a, "Close")]) for a, s in position_dict.items())
    assert metadata_dict["fixed_target_share_map_dict"] == {
        a: float(int(nav_float * _weights(second_obj)[a] / float(price_df.loc[second_str, (a, "Close")]))) for a in ASSET_TUPLE}
    assert second_obj.strategy_state_dict["beyond6040_execution_receipt_dict"] == first_obj.snapshot_metadata_dict["beyond6040_candidate_execution_receipt_dict"]
    assert second_obj.strategy_state_dict["last_signal_date_str"] == second_str


def test_unapplied_receipt_is_caught_up_and_a_malformed_one_warns(release_obj, price_df):
    first_str, second_str = "2026-03-12", "2026-03-13"
    first_obj = _build(release_obj, price_df, first_str, _state(release_obj, first_str))
    # Nothing filled: the account is still flat while the receipt names the frozen book.
    stale_obj = _build(release_obj, price_df, second_str, _state(release_obj, second_str, {}, _committed_state_dict(first_obj)))
    assert stale_obj.snapshot_metadata_dict["beyond6040_catch_up_bool"] and stale_obj.snapshot_metadata_dict["rebalance_bool"]
    assert not stale_obj.snapshot_metadata_dict["no_order_bool"]
    broken_state_dict = _committed_state_dict(first_obj)
    broken_state_dict["beyond6040_execution_receipt_dict"] = {"last_applied_target_share_map_dict": "oops"}
    broken_obj = _build(release_obj, price_df, second_str, _state(release_obj, second_str, {}, broken_state_dict))
    assert "beyond6040_execution_receipt_invalid" in broken_obj.snapshot_metadata_dict["beyond6040_warning_code_list"]
    assert broken_obj.snapshot_metadata_dict["beyond6040_catch_up_bool"]


def test_a_new_flat_pod_sizes_at_its_first_decision_for_the_next_open(release_obj, price_df):
    decision_obj = _build(release_obj, price_df, "2026-03-12", _state(release_obj, "2026-03-12"))
    metadata_dict = decision_obj.snapshot_metadata_dict
    assert metadata_dict["initialization_bool"] and metadata_dict["rebalance_bool"] and not metadata_dict["beyond6040_catch_up_bool"]
    assert metadata_dict["fixed_target_share_map_dict"] != {a: 0.0 for a in ASSET_TUPLE}
    assert metadata_dict["beyond6040_warning_code_list"] == []
    assert decision_obj.strategy_state_dict["beyond6040_execution_receipt_dict"] == {}


def test_a_failed_first_fill_is_caught_up_not_reinitialized(release_obj, price_df):
    first_str, second_str = "2026-03-12", "2026-03-13"
    first_obj = _build(release_obj, price_df, first_str, _state(release_obj, first_str))
    failed_state_dict = {**first_obj.strategy_state_dict, "beyond6040_execution_receipt_dict": {}}
    retry_obj = _build(release_obj, price_df, second_str, _state(release_obj, second_str, {}, failed_state_dict))
    metadata_dict = retry_obj.snapshot_metadata_dict
    assert metadata_dict["rebalance_bool"] and metadata_dict["beyond6040_catch_up_bool"] and not metadata_dict["initialization_bool"]
    assert metadata_dict["fixed_target_share_map_dict"] != {a: 0.0 for a in ASSET_TUPLE}


def test_a_partial_first_fill_is_resized_from_nav_and_completed(release_obj, price_df):
    first_str, second_str = "2026-03-12", "2026-03-13"
    first_obj = _build(release_obj, price_df, first_str, _state(release_obj, first_str))
    target_dict = first_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    filled_asset_str = next(a for a in RISKY_TUPLE if target_dict[a] > 0)
    partial_dict = {filled_asset_str: target_dict[filled_asset_str]}
    cash_float = 100_000.0 - partial_dict[filled_asset_str] * float(price_df.loc[second_str, (filled_asset_str, "Open")])
    failed_state_dict = {**first_obj.strategy_state_dict, "beyond6040_execution_receipt_dict": {}}
    retry_obj = _build(release_obj, price_df, second_str, _state(release_obj, second_str, partial_dict, failed_state_dict, cash_float))
    metadata_dict = retry_obj.snapshot_metadata_dict
    assert metadata_dict["rebalance_bool"] and metadata_dict["beyond6040_catch_up_bool"] and not metadata_dict["initialization_bool"]
    nav_float = cash_float + partial_dict[filled_asset_str] * float(price_df.loc[second_str, (filled_asset_str, "Close")])
    assert metadata_dict["sizing_close_nav_float"] == pytest.approx(nav_float, rel=1e-12)
    assert all(metadata_dict["fixed_target_share_map_dict"][a] > 0 for a in ASSET_TUPLE if _weights(retry_obj)[a] > 0)


@pytest.mark.parametrize("held_dict", [{"VTI": 10.0}, {"TLT": 5.0}, {"GLD": 3.0}, {"BIL": 100.0}, {"VTI": 10.0, "BIL": 50.0}])
def test_holdings_without_any_decision_need_a_reviewed_start(release_obj, price_df, held_dict):
    """A never-decided pod that already holds any of the four tradables is refused, as the Flow Vote does."""
    date_str = "2026-03-13"
    with pytest.raises(ValueError, match="holdings without a decision; a reviewed start is required"):
        _build(release_obj, price_df, date_str, _state(release_obj, date_str, held_dict, cash_float=50_000.0))


def test_a_flat_never_decided_pod_still_sizes_normally(release_obj, price_df):
    date_str = "2026-03-13"
    decision_obj = _build(release_obj, price_df, date_str, _state(release_obj, date_str))
    metadata_dict = decision_obj.snapshot_metadata_dict
    assert metadata_dict["initialization_bool"] and metadata_dict["rebalance_bool"] and not metadata_dict["beyond6040_catch_up_bool"]
    assert metadata_dict["beyond6040_warning_code_list"] == []
    close_dict = metadata_dict["sizing_close_price_map_dict"]
    assert metadata_dict["fixed_target_share_map_dict"] == {
        a: float(int(100_000.0 * _weights(decision_obj)[a] / close_dict[a])) for a in ASSET_TUPLE}
    assert any(shares_float > 0 for shares_float in metadata_dict["fixed_target_share_map_dict"].values())


def test_skipped_session_and_revised_history_warn_without_blocking(release_obj, price_df):
    first_obj = _build(release_obj, price_df, "2026-03-10", _state(release_obj, "2026-03-10"))
    skipped_obj = _build(release_obj, price_df, "2026-03-12", _state(release_obj, "2026-03-12", strategy_state_dict=_committed_state_dict(first_obj)))
    assert "beyond6040_strategy_state_cache_gap" in skipped_obj.snapshot_metadata_dict["beyond6040_warning_code_list"]
    assert skipped_obj.snapshot_metadata_dict["beyond6040_catch_up_bool"]
    revised_state_dict = deepcopy(_committed_state_dict(first_obj))
    revised_state_dict["last_target_weight_map_dict"]["VTI"] += 0.01
    revised_obj = _build(release_obj, price_df, "2026-03-11", _state(release_obj, "2026-03-11", strategy_state_dict=revised_state_dict))
    assert revised_obj.snapshot_metadata_dict["beyond6040_data_revision_warning_bool"]
    assert "beyond6040_strategy_state_cache_revised" in revised_obj.snapshot_metadata_dict["beyond6040_warning_code_list"]
    clean_obj = _build(release_obj, price_df, "2026-03-11", _state(release_obj, "2026-03-11", strategy_state_dict=_committed_state_dict(first_obj)))
    assert not clean_obj.snapshot_metadata_dict["beyond6040_data_revision_warning_bool"]


def test_last_rebalance_date_is_the_month_end_whose_book_is_in_force(release_obj, price_df):
    month_end_ts = _month_end_session_list(price_df, "2026-02-01", "2026-02-28")[0]
    month_end_obj = _build(release_obj, price_df, month_end_ts.date().isoformat(), _state(release_obj, month_end_ts.date().isoformat()))
    later_obj = _build(release_obj, price_df, "2026-03-12", _state(release_obj, "2026-03-12"))
    assert pd.Timestamp(later_obj.strategy_state_dict["last_rebalance_date_str"]) <= pd.Timestamp(month_end_ts)
    assert pd.Timestamp(month_end_obj.strategy_state_dict["last_rebalance_date_str"]) <= pd.Timestamp(month_end_ts)
    assert month_end_obj.snapshot_metadata_dict["beyond6040_candidate_execution_receipt_dict"]["last_applied_signal_date_str"] == month_end_ts.date().isoformat()


# ---------------------------------------------------------------- fail closed
@pytest.mark.parametrize("mutation_str", ["stale", "bootstrap", "wrong_account", "future", "fractional", "unknown_asset", "short", "pre_close"])
def test_invalid_account_state_blocks_decision(release_obj, price_df, mutation_str):
    state_obj = _state(release_obj, "2026-03-12")
    if mutation_str == "stale": state_obj = replace(state_obj, updated_timestamp_ts=_time("2026-03-11", 17))
    elif mutation_str == "bootstrap": state_obj = replace(state_obj, snapshot_stage_str="unknown", snapshot_source_str="pod_state")
    elif mutation_str == "wrong_account": state_obj = replace(state_obj, account_route_str="DU_OTHER")
    elif mutation_str == "future": state_obj = replace(state_obj, updated_timestamp_ts=_time("2026-03-13", 17))
    elif mutation_str == "fractional": state_obj = replace(state_obj, position_amount_map={"VTI": 1.5})
    elif mutation_str == "unknown_asset": state_obj = replace(state_obj, position_amount_map={"SPY": 2})
    elif mutation_str == "short": state_obj = replace(state_obj, position_amount_map={"TLT": -1})
    else: state_obj = replace(state_obj, updated_timestamp_ts=_time("2026-03-12", 15))
    with pytest.raises(ValueError):
        _build(release_obj, price_df, "2026-03-12", state_obj)


@pytest.mark.parametrize("change_dict", [
    {"norgate_snapshot_date_str": "2026-03-11"}, {"norgate_snapshot_date_str": "2026-03-13"},
    {"norgate_data_profile_str": "norgate_eod_core5"}, {"norgate_manifest_hash_str": ""},
])
def test_decision_requires_the_exact_snapshot_identity(release_obj, price_df, change_dict):
    with pytest.raises(ValueError, match="exact decision-session snapshot"):
        _build(release_obj, price_df, "2026-03-12", _state(release_obj, "2026-03-12"), **change_dict)


def test_disabled_live_release_and_missing_prior_session_block(release_obj, price_df):
    with pytest.raises(ValueError, match="disabled LIVE"):
        _build(replace(release_obj, mode_str="live", enabled_bool=False), price_df, "2026-03-12", _state(release_obj, "2026-03-12"))
    gapped_df = price_df.drop(index=pd.Timestamp("2026-03-11"))
    gapped_df.attrs = dict(price_df.attrs)
    with pytest.raises(ValueError):
        _build(release_obj, gapped_df, "2026-03-12", _state(release_obj, "2026-03-12"))


def test_frozen_plan_rejects_changed_positions_or_tampered_quantities(release_obj, price_df):
    decision_obj = _build(release_obj, price_df, "2026-03-12", _state(release_obj, "2026-03-12"))
    with pytest.raises(ValueError, match="positions changed"):
        adapter_module.validated_beyond6040_target_share_dict(decision_obj, release_obj, {"VTI": 1})
    decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]["GLD"] += 1
    with pytest.raises(ValueError, match="disagree"):
        adapter_module.validated_beyond6040_target_share_dict(decision_obj, release_obj, {})
    other_release_obj = replace(release_obj, pod_id_str="other_pod")
    with pytest.raises(ValueError, match="identity"):
        adapter_module.validated_beyond6040_target_share_dict(decision_obj, other_release_obj, {})


@pytest.mark.parametrize("weight_dict,tradable_bool", [
    ({"VTI": .5, "TLT": .3, "GLD": .3, "BIL": 0.0, "Cash": -.1}, True),
    ({"VTI": .5, "TLT": .2, "GLD": .1, "BIL": .1, "Cash": .1}, True),
    ({"VTI": .5, "TLT": .2, "GLD": .1, "BIL": .2, "Cash": 0.0}, False),
    ({"VTI": -.1, "TLT": .2, "GLD": .1, "BIL": .8, "Cash": 0.0}, True),
    ({"VTI": .5, "TLT": .2, "GLD": .1, "Cash": .2}, True),
])
def test_weight_book_validation_rejects_shorts_leverage_and_misplaced_residual(weight_dict, tradable_bool):
    with pytest.raises(ValueError):
        adapter_module._validate_target_weight_dict(weight_dict, tradable_bool)
    adapter_module._validate_target_weight_dict({"VTI": .5, "TLT": .2, "GLD": .1, "BIL": .2, "Cash": 0.0}, True)
    adapter_module._validate_target_weight_dict({"VTI": .5, "TLT": .2, "GLD": .1, "BIL": 0.0, "Cash": .2}, False)


# ---------------------------------------------------------------- the host path: snapshot mode required
@pytest.mark.parametrize("mutation_str", ["valid", "direct", "stale", "changed_manifest"])
def test_decision_plan_requires_snapshot_mode_and_a_stable_manifest(release_obj, price_df, monkeypatch, mutation_str):
    snapshot_module = adapter_module.snapshot_module
    monkeypatch.setattr(snapshot_module, "is_snapshot_mode_enabled_bool", lambda: mutation_str != "direct")

    def load_manifest(profile_str, minimum_snapshot_date_str):
        assert profile_str == BEYOND6040_PROFILE_STR and minimum_snapshot_date_str == "2026-03-12"
        return SimpleNamespace(manifest_hash_str="fixture_hash")

    monkeypatch.setattr(snapshot_module, "load_valid_snapshot_manifest", load_manifest)
    monkeypatch.setattr(tilt_module, "get_trend_tilt_data", lambda config: price_df)
    metadata_dict = {"norgate_data_profile_str": BEYOND6040_PROFILE_STR,
                     "norgate_manifest_hash_str": "different" if mutation_str == "changed_manifest" else "fixture_hash",
                     "norgate_snapshot_date_str": "2026-03-11" if mutation_str == "stale" else "2026-03-12"}
    monkeypatch.setattr(snapshot_module, "build_data_source_metadata_dict", lambda *args: metadata_dict)
    if mutation_str == "valid":
        decision_obj = adapter_module.build_beyond6040_decision_plan(release_obj, _time("2026-03-12"), _state(release_obj, "2026-03-12"))
        assert decision_obj.snapshot_metadata_dict["initialization_bool"]
    else:
        with pytest.raises(ValueError):
            adapter_module.build_beyond6040_decision_plan(release_obj, _time("2026-03-12"), _state(release_obj, "2026-03-12"))
