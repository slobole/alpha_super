"""The Beyond 60/40 data path: exporter -> snapshot -> reader -> the same frame and the same decision as direct Norgate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha.live import beyond6040_adapter as adapter_module, scheduler_utils
from alpha.live.models import LiveRelease, PodState
from data import norgate_loader, norgate_snapshot_store as snapshot_module
from data.beyond6040_snapshot_contract import (
    BEYOND6040_CAPITAL_SYMBOL_TUPLE, BEYOND6040_HISTORY_START_STR, BEYOND6040_PROFILE_STR, BEYOND6040_TOTAL_RETURN_SYMBOL_TUPLE,
    beyond6040_price_coverage_dict, beyond6040_required_pair_set, validate_beyond6040_snapshot,
)
from scripts import export_norgate_snapshot as export_module
from strategies.taa_beyond_6040 import strategy_taa_beyond_6040_trend_tilt as tilt_module


END_STR = "2026-03-12"
INCEPTION_DICT = {"VTI": "2001-05-31", "TLT": "2002-07-30", "GLD": "2004-11-18", "BIL": "2007-05-30", "$SPX": "2001-05-31", "$SPXTR": "2001-05-31"}
ROUTE_STR = "DU_BEYOND6040_TEST"


class FakeNorgate:
    """Norgate's call shape for the nine Beyond 60/40 series: observed rows only, native float32 prices, per-ETF inception."""
    StockPriceAdjustmentType = SimpleNamespace(CAPITALSPECIAL="CAPITALSPECIAL", TOTALRETURN="TOTALRETURN")
    PaddingType = SimpleNamespace(ALLMARKETDAYS="ALLMARKETDAYS", NONE="NONE")

    def __init__(self, end_str=END_STR):
        import exchange_calendars as xcals

        self.date_idx = pd.DatetimeIndex(list(xcals.get_calendar("XNYS", start="2001-01-01").sessions_in_range(BEYOND6040_HISTORY_START_STR, end_str)), name="Date")
        rng_obj = np.random.default_rng(7)
        self.frame_dict, self.stale_key_tuple = {}, None
        for symbol_int, symbol_str in enumerate(INCEPTION_DICT):
            capital_vec = (100.0 + 10.0 * symbol_int) * np.cumprod(1.0 + rng_obj.normal(0.0002, 0.004 if symbol_str == "BIL" else 0.01, len(self.date_idx)))
            for adjustment_str, growth_obj in (("CAPITALSPECIAL", 1.0), ("TOTALRETURN", np.linspace(1.0, 1.6, len(self.date_idx)))):
                close_vec = (capital_vec * growth_obj).astype(np.float32)
                frame_df = pd.DataFrame({"Open": (close_vec * 0.999).astype(np.float32), "High": (close_vec * 1.01).astype(np.float32),
                    "Low": (close_vec * 0.99).astype(np.float32), "Close": close_vec,
                    "Volume": np.full(len(close_vec), 1_000_000, dtype=np.int64), "Turnover": close_vec.astype(float) * 1e6},
                    index=self.date_idx)
                if not symbol_str.startswith("$"):
                    frame_df["Unadjusted Close"] = close_vec
                    frame_df["Dividend"] = np.where(np.arange(len(close_vec)) % 63 == 0, 0.11, 0.0).astype(np.float32)
                self.frame_dict[(symbol_str, adjustment_str)] = frame_df.loc[INCEPTION_DICT[symbol_str]:]

    def price_timeseries(self, symbol_str, **option_dict):
        key_tuple = (symbol_str, option_dict["stock_price_adjustment_setting"])
        frame_df = self.frame_dict[key_tuple].iloc[:-1] if key_tuple == self.stale_key_tuple else self.frame_dict[key_tuple]
        return frame_df.loc[option_dict["start_date"]:option_dict["end_date"]].copy()


@pytest.fixture
def snapshot_case(tmp_path, monkeypatch):
    provider_obj = FakeNorgate()
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: provider_obj)
    monkeypatch.setattr(norgate_loader, "_load_direct_norgate_module", lambda: provider_obj)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    config_obj = replace(tilt_module.DEFAULT_CONFIG, end_date_str=END_STR)
    direct_df = tilt_module.get_trend_tilt_data(config_obj)
    snapshot_path_obj = export_module.export_profile_snapshot(snapshot_root_str=str(tmp_path), profile_str=BEYOND6040_PROFILE_STR, snapshot_date_str=END_STR)
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setattr(norgate_loader, "_load_direct_norgate_module", lambda: pytest.fail("the snapshot consumer called direct Norgate"))
    return SimpleNamespace(provider_obj=provider_obj, direct_df=direct_df, snapshot_path_obj=snapshot_path_obj, config_obj=config_obj, signal_ts=pd.Timestamp(END_STR))


def _release():
    return LiveRelease(
        release_id_str="beyond6040.test.v1", user_id_str="test", pod_id_str="beyond6040_test",
        account_route_str=ROUTE_STR, strategy_import_str=adapter_module.BEYOND6040_STRATEGY_IMPORT_STR,
        mode_str="paper", session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo", data_profile_str=BEYOND6040_PROFILE_STR, params_dict={},
        risk_profile_str="beyond6040_long_only", enabled_bool=True, source_path_str="test.yaml",
        pod_budget_fraction_float=1.0, auto_submit_enabled_bool=False,
    )


def _pod_state(release_obj, signal_ts):
    close_ts = scheduler_utils.get_session_close_timestamp_ts(signal_ts, "XNYS")
    return PodState(pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str,
        position_amount_map={}, cash_float=100_000.0, total_value_float=100_000.0, strategy_state_dict={},
        updated_timestamp_ts=close_ts + timedelta(minutes=12), snapshot_stage_str="eod", snapshot_source_str="broker")


def test_the_profile_contract_names_the_research_series():
    assert set(BEYOND6040_CAPITAL_SYMBOL_TUPLE) == set(adapter_module.BEYOND6040_ASSET_TUPLE)
    assert set(BEYOND6040_TOTAL_RETURN_SYMBOL_TUPLE) == set(adapter_module.BEYOND6040_RISK_ASSET_TUPLE) | {"$SPX", "$SPXTR"}
    assert len(beyond6040_required_pair_set()) == 9
    assert tilt_module.DEFAULT_CONFIG.start_date_str == BEYOND6040_HISTORY_START_STR == "2001-05-31"
    assert tilt_module.DEFAULT_CONFIG.reserve_asset_str == "BIL"
    assert export_module.PROFILE_EXPORT_SPEC_DICT[BEYOND6040_PROFILE_STR].padding_type_name_str == "NONE"


def test_the_snapshot_reader_returns_the_research_frame_and_the_same_decision(snapshot_case):
    case_obj = snapshot_case
    snapshot_df = tilt_module.get_trend_tilt_data(case_obj.config_obj)
    pd.testing.assert_frame_equal(snapshot_df, case_obj.direct_df, check_exact=True, check_freq=False)
    assert snapshot_df.attrs == case_obj.direct_df.attrs
    assert snapshot_df.index[0] == pd.Timestamp(BEYOND6040_HISTORY_START_STR) and snapshot_df.index[-1] == case_obj.signal_ts
    assert snapshot_df[("BIL", "Close")].loc[:"2007-05-29"].isna().all() and snapshot_df[("BIL", "Close")].loc["2007-05-30":].notna().all()
    assert ("$SPX", "Dividend") not in snapshot_df.columns and ("BIL", "TotalReturnClose") not in snapshot_df.columns
    pd.testing.assert_series_equal(snapshot_df[("$SPX", "Close")], case_obj.provider_obj.frame_dict[("$SPXTR", "TOTALRETURN")]["Close"], check_names=False, check_freq=False)
    assert snapshot_df[("VTI", "Close")].dtype == np.float32
    release_obj = _release()
    state_obj = _pod_state(release_obj, case_obj.signal_ts)
    as_of_ts = scheduler_utils.get_session_close_timestamp_ts(case_obj.signal_ts, "XNYS") + timedelta(minutes=30)
    host_obj = adapter_module.build_beyond6040_decision_plan(release_obj, as_of_ts, state_obj)
    direct_obj = adapter_module.build_beyond6040_decision_from_prices(release_obj, as_of_ts, state_obj, case_obj.direct_df,
        {key_str: host_obj.snapshot_metadata_dict[key_str] for key_str in ("norgate_snapshot_date_str", "norgate_data_profile_str", "norgate_manifest_hash_str")})
    assert host_obj.snapshot_metadata_dict["norgate_data_source_mode_str"] == "snapshot"
    for key_str in ("fixed_target_share_map_dict", "rebalance_bool", "sizing_close_price_map_dict", "beyond6040_gross_exposure_float", "beyond6040_base_weight_map_dict"):
        assert host_obj.snapshot_metadata_dict[key_str] == direct_obj.snapshot_metadata_dict[key_str]
    assert host_obj.full_target_weight_map_dict == direct_obj.full_target_weight_map_dict
    assert host_obj.full_target_weight_map_dict["BIL"] > 0.0 and sum(host_obj.full_target_weight_map_dict.values()) == pytest.approx(1.0, abs=1e-12)


def test_the_snapshot_engine_run_equals_the_direct_run(snapshot_case):
    case_obj = snapshot_case
    snapshot_df = tilt_module.get_trend_tilt_data(case_obj.config_obj)
    from unittest.mock import patch
    with patch.object(tilt_module, "get_trend_tilt_data", return_value=case_obj.direct_df):
        direct_strategy_obj = tilt_module.run_variant(show_display_bool=False, save_results_bool=False, end_date_str=END_STR, backtest_start_date_str="2024-01-01")
    with patch.object(tilt_module, "get_trend_tilt_data", return_value=snapshot_df):
        snapshot_strategy_obj = tilt_module.run_variant(show_display_bool=False, save_results_bool=False, end_date_str=END_STR, backtest_start_date_str="2024-01-01")
    for field_str in ("daily_target_weights", "results"):
        pd.testing.assert_frame_equal(getattr(direct_strategy_obj, field_str), getattr(snapshot_strategy_obj, field_str), check_exact=True, check_freq=False)
    assert (direct_strategy_obj.daily_target_weights["BIL"] > 0.0).any()
    assert np.allclose(direct_strategy_obj.daily_target_weights.sum(axis=1).to_numpy(dtype=float), 1.0, atol=1e-12)


def _rewrite_snapshot(case_obj, price_fn=None, contract_fn=None, recount_bool=False):
    price_path_obj = case_obj.snapshot_path_obj / "prices.parquet"
    price_df = pd.read_parquet(price_path_obj)
    if price_fn is not None:
        price_df = price_fn(price_df)
        price_df.to_parquet(price_path_obj, index=False)
    manifest_path_obj = case_obj.snapshot_path_obj / "manifest.json"
    manifest_dict = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
    if recount_bool:
        manifest_dict["data_contract"]["series_coverage_dict"] = beyond6040_price_coverage_dict(price_df)
    if contract_fn is not None:
        contract_fn(manifest_dict["data_contract"])
    manifest_dict["files"]["prices.parquet"]["sha256"] = hashlib.sha256(price_path_obj.read_bytes()).hexdigest()
    manifest_path_obj.write_text(json.dumps(manifest_dict), encoding="utf-8")


def _pair_mask_ser(price_df, symbol_str, adjustment_str):
    return price_df["symbol_str"].eq(symbol_str) & price_df["adjustment_str"].eq(adjustment_str)


@pytest.mark.parametrize("mutation_str,match_str", [
    ("endpoint", "stale endpoint"), ("gap", "missing session"), ("future", "future"), ("coverage", "declared source coverage"),
    ("padding", "contract mismatch"), ("history", "contract mismatch"), ("pair", "exact symbol/adjustment pairs"),
    ("extra_pair", "exact symbol/adjustment pairs"), ("dtype", "invalid native dtype"), ("request", "source request mismatch"),
    ("bil_gap", "missing session"),
])
def test_a_corrupt_snapshot_fails_closed_even_with_an_updated_hash(snapshot_case, mutation_str, match_str):
    case_obj = snapshot_case

    def price_fn(price_df):
        mask_ser = _pair_mask_ser(price_df, "GLD", "TOTALRETURN")
        if mutation_str == "endpoint":
            return price_df.drop(index=price_df.index[mask_ser][-1])
        if mutation_str in ("gap", "coverage"):
            return price_df.drop(index=price_df.index[mask_ser][100])
        if mutation_str == "bil_gap":
            return price_df.drop(index=price_df.index[_pair_mask_ser(price_df, "BIL", "CAPITALSPECIAL")][40])
        if mutation_str == "future":
            price_df.loc[price_df.index[mask_ser][-1], "date"] = pd.Timestamp(END_STR) + pd.Timedelta(days=1)
            return price_df
        if mutation_str == "pair":
            return price_df.loc[~_pair_mask_ser(price_df, "BIL", "CAPITALSPECIAL")]
        if mutation_str == "extra_pair":
            extra_df = price_df.loc[_pair_mask_ser(price_df, "BIL", "CAPITALSPECIAL")].copy()
            extra_df["adjustment_str"] = "TOTALRETURN"
            return pd.concat([price_df, extra_df], ignore_index=True)
        price_df["Close"] = price_df["Close"].astype(float)
        price_df.loc[price_df.index[mask_ser][50], "Close"] += 1e-7
        return price_df

    def contract_fn(contract_dict):
        if mutation_str == "padding":
            contract_dict["price_padding_setting_str"] = "ALLMARKETDAYS"
        elif mutation_str == "history":
            contract_dict["history_start_date_str"] = "1995-01-01"
        else:
            contract_dict["source_request_by_pair_dict"]["VTI|CAPITALSPECIAL"]["padding_type_name_str"] = "ALLMARKETDAYS"

    contract_mutation_bool = mutation_str in ("padding", "history", "request")
    _rewrite_snapshot(case_obj, price_fn=None if contract_mutation_bool else price_fn,
        contract_fn=contract_fn if contract_mutation_bool else None, recount_bool=mutation_str in ("endpoint", "gap", "future", "bil_gap", "pair", "extra_pair"))
    with pytest.raises((ValueError, snapshot_module.NorgateSnapshotValidationError), match=match_str):
        snapshot_module.load_valid_snapshot_manifest(BEYOND6040_PROFILE_STR)
    with pytest.raises((ValueError, snapshot_module.NorgateSnapshotValidationError)):
        tilt_module.get_trend_tilt_data(case_obj.config_obj)


def test_the_contract_validator_rejects_the_wrong_schema_and_a_wrong_manifest_contract(snapshot_case):
    case_obj = snapshot_case
    price_df = pd.read_parquet(case_obj.snapshot_path_obj / "prices.parquet")
    manifest_dict = json.loads((case_obj.snapshot_path_obj / "manifest.json").read_text(encoding="utf-8"))
    validate_beyond6040_snapshot(price_df, manifest_dict, pd.Timestamp(END_STR))
    with pytest.raises(ValueError, match="schema v2"):
        validate_beyond6040_snapshot(price_df, {**manifest_dict, "schema_version": 1}, pd.Timestamp(END_STR))
    with pytest.raises(ValueError, match="source request mismatch"):
        # A later session than the one exported: the recorded request's end date no longer certifies Close_T.
        validate_beyond6040_snapshot(price_df, manifest_dict, pd.Timestamp(END_STR) + pd.Timedelta(days=1))
    missing_field_dict = json.loads(json.dumps(manifest_dict))
    missing_field_dict["data_contract"]["source_field_by_pair_dict"]["BIL|CAPITALSPECIAL"].remove("Dividend")
    with pytest.raises(ValueError, match="provenance"):
        validate_beyond6040_snapshot(price_df, missing_field_dict, pd.Timestamp(END_STR))


def test_the_exporter_refuses_a_series_without_an_observed_bar_at_t(tmp_path, monkeypatch):
    provider_obj = FakeNorgate()
    provider_obj.stale_key_tuple = ("BIL", "CAPITALSPECIAL")
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: provider_obj)
    with pytest.raises(ValueError, match="observed bar"):
        export_module.export_profile_snapshot(snapshot_root_str=str(tmp_path), profile_str=BEYOND6040_PROFILE_STR, snapshot_date_str=END_STR)
    assert not (tmp_path / BEYOND6040_PROFILE_STR / END_STR).exists()


def test_the_exporter_refuses_an_end_date_other_than_the_snapshot_session(tmp_path, monkeypatch):
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: FakeNorgate())
    with pytest.raises(ValueError, match="must end on the snapshot session"):
        export_module.export_profile_snapshot(snapshot_root_str=str(tmp_path), profile_str=BEYOND6040_PROFILE_STR, snapshot_date_str=END_STR, end_date_str="2026-03-11")


def test_a_snapshot_published_between_the_reads_is_refused(snapshot_case, monkeypatch):
    original_fn, call_list = snapshot_module.load_valid_snapshot_manifest, []

    def changing_fn(*argument_tuple, **argument_dict):
        manifest_obj = original_fn(*argument_tuple, **argument_dict)
        call_list.append(manifest_obj)
        return manifest_obj if len(call_list) == 1 else replace(manifest_obj, manifest_hash_str="another vintage")

    monkeypatch.setattr(tilt_module, "load_valid_snapshot_manifest", changing_fn)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="changed during the data load"):
        tilt_module.get_trend_tilt_data(snapshot_case.config_obj)


def test_the_decision_refuses_metadata_from_another_snapshot(snapshot_case, monkeypatch):
    case_obj = snapshot_case
    monkeypatch.setattr(snapshot_module, "build_data_source_metadata_dict", lambda _profile_str: {"norgate_manifest_hash_str": "another"})
    release_obj = _release()
    as_of_ts = scheduler_utils.get_session_close_timestamp_ts(case_obj.signal_ts, "XNYS") + timedelta(minutes=30)
    with pytest.raises(ValueError, match="changed while preparing"):
        adapter_module.build_beyond6040_decision_plan(release_obj, as_of_ts, _pod_state(release_obj, case_obj.signal_ts))


def test_other_profiles_keep_their_routes():
    assert snapshot_module.default_profile_for_symbol_str("VTI") == "norgate_eod_etf_plus_vix_helper"
    assert BEYOND6040_PROFILE_STR not in snapshot_module.NDX_PROFILE_SET and BEYOND6040_PROFILE_STR not in snapshot_module.MR_CAPSULE_PROFILE_SET
    assert export_module.PROFILE_EXPORT_SPEC_DICT[snapshot_module.CORE5_PROFILE_STR].capital_symbol_tuple == snapshot_module.CORE5_CAPITAL_SYMBOL_TUPLE
    assert export_module.PROFILE_EXPORT_SPEC_DICT["norgate_eod_etf_plus_vix_helper"].capital_symbol_tuple == ("GLD", "UUP", "TLT", "DBC", "BTAL", "SPY", "QQQ", "TQQQ")
