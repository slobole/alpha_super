"""Capsule host parity and fail-closed boundaries, without Norgate or a broker."""
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from alpha.live import mr_capsule_adapter as adapter_module, strategy_host
from alpha.live.models import PodState
from alpha.live.release_manifest import parse_release_manifest, validate_release_manifest
from strategies.hpi import stateful_long as hpi_module
from strategies.mr_capsule import dv2_vix_gated, hpi_vote_vix_gated, vix_stress_gate


SIGNAL_DATE_TS = pd.Timestamp("2024-01-12")  # Friday; next session is Tuesday (MLK holiday).
AS_OF_TS = datetime(2024, 1, 12, 18, tzinfo=ZoneInfo("America/New_York"))


def _release(pod_str="dv2_vix_gated", mode_str="bil"):
    return parse_release_manifest(f"docs/live/release_templates/pod_mr_{pod_str}_{mode_str}_daily_moo.yaml.example")


def _state(release_obj, position_map_dict=None, cash_float=100_000.0):
    return PodState(
        pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str, position_amount_map=position_map_dict or {},
        cash_float=cash_float, total_value_float=100_000.0,
        strategy_state_dict={"mr_capsule_strategy_import_str": release_obj.strategy_import_str},
        updated_timestamp_ts=datetime(2024, 1, 12, 17, tzinfo=ZoneInfo("America/New_York")),
        snapshot_stage_str="eod", snapshot_source_str="broker",
    )


def _inputs(monkeypatch, release_obj, gate_open_bool=True):
    pricing_index = pd.bdate_range("2023-12-01", SIGNAL_DATE_TS)
    pricing_data_df = pd.DataFrame({
        (symbol_str, field_str): value_float
        for symbol_str in ("AAA", "BIL", "SPMO", "$SPX", "$SPXTR")
        for field_str, value_float in {
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
            "Volume": 10000.0, "Dividend": 0.0,
            "ibs_value_ser": 0.0, "rsi2_value_ser": 0.0,
        }.items()
    }, index=pricing_index)
    universe_df = pd.DataFrame(1, index=pricing_index, columns=["AAA"])
    vix_close_ser = pd.Series(10.0, index=pd.bdate_range("1990-01-02", SIGNAL_DATE_TS))
    if gate_open_bool:
        vix_close_ser.loc[SIGNAL_DATE_TS] = 50.0
    metadata_dict = {
        "norgate_snapshot_date_str": "2024-01-12", "norgate_data_profile_str": release_obj.data_profile_str,
        "norgate_manifest_hash_str": "synthetic-manifest",
    }
    monkeypatch.setattr(adapter_module, "build_data_source_metadata_dict", lambda *_: metadata_dict)
    monkeypatch.setattr(strategy_host, "build_data_source_metadata_dict", lambda *_: metadata_dict)
    monkeypatch.setattr(dv2_vix_gated, "load_pricing_data", lambda *_: (pricing_data_df, universe_df))
    monkeypatch.setattr(hpi_module, "load_exact_hpi_inputs", lambda **_: (["AAA"], universe_df, pricing_data_df))
    # The shared capsule loader calls the binding in hpi_vote_vix_gated; patch it so no test reaches Norgate.
    monkeypatch.setattr(hpi_vote_vix_gated, "load_exact_hpi_inputs", lambda **_: (["AAA"], universe_df, pricing_data_df))
    monkeypatch.setattr(hpi_vote_vix_gated, "append_parking_prices", lambda frame_df, *_: frame_df)
    monkeypatch.setattr(vix_stress_gate, "load_vix_close_ser", lambda *_: vix_close_ser)

    def synthetic_signals(strategy_obj, frame_df):
        strategy_obj._prepare_capsule_state()
        return frame_df

    monkeypatch.setattr(dv2_vix_gated.DV2VixGatedStrategy, "compute_signals", synthetic_signals)
    monkeypatch.setattr(hpi_vote_vix_gated.HPIVoteVixGatedStrategy, "compute_signals", synthetic_signals)
    monkeypatch.setattr(dv2_vix_gated.DV2VixGatedStrategy, "get_opportunities", lambda *_: ["AAA"])
    monkeypatch.setattr(hpi_vote_vix_gated.HPIVoteVixGatedStrategy, "get_opportunity_list", lambda *_: ["AAA"])
    monkeypatch.setattr(strategy_host, "_validate_hpi_live_feature_readiness", lambda **_: None)
    return pricing_data_df, vix_close_ser, metadata_dict


@pytest.mark.parametrize("pod_str", ["dv2_vix_gated", "hpi_vote_vix_gated"])
@pytest.mark.parametrize("mode_str", ["cash", "bil", "spmo"])
def test_fresh_host_preserves_stock_entry_and_parking_targets(monkeypatch, pod_str, mode_str):
    release_obj = _release(pod_str, mode_str)
    _inputs(monkeypatch, release_obj)
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))
    assert decision_obj.entry_target_weight_map_dict == {"AAA": .1}
    assert decision_obj.target_share_map_dict == ({} if mode_str == "cash" else {"BIL": 890.0})
    assert decision_obj.target_execution_timestamp_ts.date().isoformat() == "2024-01-16"
    assert decision_obj.snapshot_metadata_dict["decision_nav_float"] == 100_000.0
    assert decision_obj.strategy_state_dict["mr_capsule_strategy_import_str"] == release_obj.strategy_import_str
    repeat_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))
    assert decision_obj == repeat_obj


@pytest.mark.parametrize("pod_str", ["dv2_vix_gated", "hpi_vote_vix_gated"])
def test_closed_gate_keeps_exits_enabled_and_bil_parking(monkeypatch, pod_str):
    release_obj = _release(pod_str)
    pricing_data_df, _, _ = _inputs(monkeypatch, release_obj, False)
    pricing_data_df.loc[SIGNAL_DATE_TS, ("AAA", "Close")] = 102.0
    pricing_data_df.loc[SIGNAL_DATE_TS, ("AAA", "ibs_value_ser")] = .95
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj, {"AAA": 100.0}, 89_800.0))
    assert decision_obj.exit_asset_set == {"AAA"}
    assert decision_obj.entry_target_weight_map_dict == {}
    assert decision_obj.target_share_map_dict == {"BIL": 990.0}


def test_same_open_entry_is_funded_by_bil_reduction(monkeypatch):
    release_obj = _release()
    _inputs(monkeypatch, release_obj)
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj, {"BIL": 990.0}, 1000.0))
    assert decision_obj.target_share_map_dict == {"BIL": 890.0}
    assert decision_obj.decision_base_position_map == {"BIL": 990.0}
    assert decision_obj.entry_target_weight_map_dict == {"AAA": .1}


@pytest.mark.parametrize("invalid_str", ["stale", "truncated", "gap", "nan", "duplicate"])
def test_invalid_vix_cannot_produce_any_decision(monkeypatch, invalid_str):
    release_obj = _release()
    _, vix_close_ser, _ = _inputs(monkeypatch, release_obj)
    if invalid_str == "stale":
        vix_close_ser = vix_close_ser.iloc[:-1]
    elif invalid_str == "truncated":
        vix_close_ser = vix_close_ser.iloc[1:]
    elif invalid_str == "gap":
        vix_close_ser = vix_close_ser.drop(pd.Timestamp("2024-01-11"))
    elif invalid_str == "duplicate":
        vix_close_ser = pd.concat([vix_close_ser, vix_close_ser.iloc[-1:]])
    else:
        vix_close_ser.loc[SIGNAL_DATE_TS] = np.nan
    monkeypatch.setattr(vix_stress_gate, "load_vix_close_ser", lambda *_: vix_close_ser)
    with pytest.raises(ValueError, match="VIX history"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))


@pytest.mark.parametrize("field_str,value_obj", [
    ("snapshot_stage_str", "unknown"), ("snapshot_source_str", "pod_state"),
    ("cash_float", float("nan")), ("total_value_float", 0.0),
    ("account_route_str", "DU_DIFFERENT"),
    ("updated_timestamp_ts", datetime(2024, 1, 11, 17, tzinfo=ZoneInfo("America/New_York"))),
])
def test_untrusted_account_state_is_rejected(monkeypatch, field_str, value_obj):
    release_obj = _release()
    _inputs(monkeypatch, release_obj)
    with pytest.raises(ValueError, match="account state|account snapshot"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, replace(_state(release_obj), **{field_str: value_obj}))


def test_missing_held_close_blocks_stock_and_funding_plan(monkeypatch):
    release_obj = _release()
    pricing_data_df, _, _ = _inputs(monkeypatch, release_obj)
    pricing_data_df.loc[SIGNAL_DATE_TS, ("AAA", "Close")] = np.nan
    with pytest.raises(ValueError, match="funding cannot be established"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj, {"AAA": 100.0}))


@pytest.mark.parametrize("volume_float", [float("nan"), float("inf"), -1.0])
def test_spmo_invalid_volume_cannot_silently_disable_parking(monkeypatch, volume_float):
    release_obj = _release(mode_str="spmo")
    pricing_data_df, _, _ = _inputs(monkeypatch, release_obj)
    pricing_data_df[("SPMO", "Volume")] = volume_float
    with pytest.raises(ValueError, match="trading-volume"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))


@pytest.mark.parametrize("change_dict", [
    {"pod_budget_fraction_float": .03}, {"execution_policy_str": "same_day_moc"},
    {"data_profile_str": "norgate_eod_sp500_pit"}, {"params_dict": {"max_positions_int": 1}},
    {"enabled_bool": True},
])
def test_release_requires_qualified_frozen_contract(change_dict):
    with pytest.raises(ValueError, match="MR capsule"):
        validate_release_manifest(replace(_release(), **change_dict))


def test_enabled_release_requires_explicit_margin_confirmation():
    release_obj = replace(_release(), enabled_bool=True, account_route_str="DU12345", params_dict={"margin_account_confirmed_bool": True})
    validate_release_manifest(release_obj)


def test_other_strategy_cannot_claim_capsule_profile():
    with pytest.raises(ValueError, match="reserved"):
        validate_release_manifest(replace(_release(), strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy"))


@pytest.mark.parametrize("mode_str,position_map_dict,state_identity_obj,match_str", [
    ("bil", {"AAA": 10.5}, "own", "whole-share"),
    ("bil", {"AAA": -10.0}, "own", "whole-share"),
    ("bil", {"AAA": 10.0}, None, "cannot adopt"),
    ("bil", {}, "strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_bil", "cannot adopt"),
    ("cash", {"BIL": 10.0}, "own", "parking holdings"),
    ("bil", {"SPMO": 10.0}, "own", "parking holdings"),
])
def test_malformed_or_foreign_holdings_cannot_produce_a_decision(monkeypatch, mode_str, position_map_dict, state_identity_obj, match_str):
    """Fail closed before any intent: the pod only trades from its own whole-share, mode-consistent state."""
    release_obj = _release("dv2_vix_gated", mode_str)
    _inputs(monkeypatch, release_obj)
    state_obj = _state(release_obj, position_map_dict)
    identity_dict = (
        {} if state_identity_obj is None
        else {"mr_capsule_strategy_import_str": release_obj.strategy_import_str if state_identity_obj == "own" else state_identity_obj}
    )
    with pytest.raises(ValueError, match=match_str):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, replace(state_obj, strategy_state_dict=identity_dict))


@pytest.mark.parametrize("key_str,value_obj", [
    ("norgate_snapshot_date_str", "2024-01-11"),
    ("norgate_data_profile_str", "norgate_eod_sp500_pit"),
    ("norgate_manifest_hash_str", ""),
])
def test_snapshot_identity_must_be_the_decision_session(monkeypatch, key_str, value_obj):
    """The real (unmocked-by-default) snapshot identity check: wrong date, profile or missing hash blocks."""
    release_obj = _release()
    _, _, metadata_dict = _inputs(monkeypatch, release_obj)
    bad_metadata_dict = {**metadata_dict, key_str: value_obj}
    monkeypatch.setattr(adapter_module, "build_data_source_metadata_dict", lambda *_: dict(bad_metadata_dict))
    with pytest.raises(ValueError, match="exact decision-session snapshot"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))


def test_missing_prior_session_row_blocks_the_decision(monkeypatch):
    """The gate-switch and week-end logic read the prior XNYS session; a gap there must not be bridged."""
    release_obj = _release()
    pricing_data_df, _, _ = _inputs(monkeypatch, release_obj)
    trimmed_df = pricing_data_df.drop(index=pd.Timestamp("2024-01-11"))
    trimmed_universe_df = pd.DataFrame(1, index=trimmed_df.index, columns=["AAA"])
    monkeypatch.setattr(dv2_vix_gated, "load_pricing_data", lambda *_: (trimmed_df, trimmed_universe_df))
    with pytest.raises(ValueError, match="prior XNYS session"):
        strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, _state(release_obj))


def test_non_positive_close_nav_blocks_the_decision(monkeypatch):
    """NAV_T = cash + sum(q * Close_T) must be positive before any entry dollars or parking shares exist."""
    release_obj = _release()
    _inputs(monkeypatch, release_obj)
    with pytest.raises(ValueError, match="Close_T NAV must be positive"):
        strategy_host.build_decision_plan_for_release(
            release_obj, AS_OF_TS, _state(release_obj, {"AAA": 100.0}, cash_float=-20_000.0),
        )
