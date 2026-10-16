"""Offline adversarial probes: passing tests document CURRENT permissive behavior.

These are audit demonstrations, not proposed production acceptance criteria.
All features and actual host dispatch run; only data transports are synthetic.
"""
from __future__ import annotations

import io
import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.research.wired_signoff_20261015.taa_ndx_native_replay import build_release_obj
from alpha.live import scheduler_utils, strategy_host

NEW_YORK_TZ = ZoneInfo("America/New_York")
SIGNAL_DATE_TS = pd.Timestamp("2024-01-31")


@pytest.fixture
def monthly_input_tuple():
    calendar_obj = scheduler_utils.get_exchange_calendar_obj("XNYS")
    session_idx = pd.DatetimeIndex(calendar_obj.sessions_in_range("2021-01-04", SIGNAL_DATE_TS)).tz_localize(None)
    elapsed_vec = np.arange(len(session_idx), dtype=float)
    field_dict = {}
    symbol_list = [f"STOCK{symbol_int:02}" for symbol_int in range(12)]
    for symbol_int, symbol_str in enumerate([*symbol_list, "SPY"]):
        close_vec = (20 + symbol_int * 15) * np.exp((0.0001 + symbol_int * 0.00007) * elapsed_vec)
        for field_str, factor_float in (("Open", 1), ("High", 1.01), ("Low", .99), ("Close", 1), ("Unadjusted Close", 1)):
            field_dict[(symbol_str, field_str)] = close_vec * factor_float
    pricing_df = pd.DataFrame(field_dict, index=session_idx)
    universe_df = pd.DataFrame(1, index=session_idx, columns=symbol_list)
    vxn_df = pd.DataFrame({"vxn_close": [44.0], "vxn_exposure_scale_float": [.5]}, index=[SIGNAL_DATE_TS])
    return pricing_df, universe_df, vxn_df


@pytest.mark.parametrize("variant_str", ["ndx_atr", "ndx_atr_vxn"])
def test_legacy_ndx_actual_host_returns_past_execution_without_stale_guard(monkeypatch, monthly_input_tuple, variant_str):
    pricing_df, universe_df, vxn_df = monthly_input_tuple
    module_str = ("strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled" if variant_str == "ndx_atr_vxn"
        else "strategies.momentum.strategy_mo_atr_normalized_ndx")
    module_obj = import_module(module_str)
    loader_name_str = "get_vxn_scaled_atr_normalized_ndx_data" if variant_str == "ndx_atr_vxn" else "get_atr_normalized_ndx_data"
    result_tuple = (pricing_df, universe_df, None, vxn_df) if variant_str == "ndx_atr_vxn" else (pricing_df, universe_df, None)
    monkeypatch.setattr(module_obj, loader_name_str, lambda config_obj: result_tuple)
    as_of_ts = datetime(2024, 2, 15, 16, 10, tzinfo=NEW_YORK_TZ)
    plan_obj = strategy_host.build_decision_plan_for_release(build_release_obj(variant_str, 1_000_000), as_of_ts, None)
    assert plan_obj.full_target_weight_map_dict
    assert plan_obj.target_execution_timestamp_ts < as_of_ts
    assert plan_obj.signal_timestamp_ts.date().isoformat() == "2024-01-31"


def test_legacy_ndx_actual_host_accepts_month_old_vxn_without_warning(monkeypatch, monthly_input_tuple):
    from strategies.momentum import strategy_mo_atr_normalized_ndx_vxn_scaled as module_obj
    pricing_df, universe_df, vxn_df = monthly_input_tuple
    vxn_df.index = pd.DatetimeIndex(["2023-12-29"])
    monkeypatch.setattr(module_obj, "get_vxn_scaled_atr_normalized_ndx_data", lambda config_obj: (pricing_df, universe_df, None, vxn_df))
    plan_obj = strategy_host.build_decision_plan_for_release(build_release_obj("ndx_atr_vxn", 1_000_000),
        datetime(2024, 1, 31, 16, 10, tzinfo=NEW_YORK_TZ), None)
    assert sum(plan_obj.full_target_weight_map_dict.values()) == pytest.approx(.5)
    assert plan_obj.snapshot_metadata_dict["vxn_reference_date_str"] == "2023-12-29"
    assert not any("warn" in str(key_str).lower() for key_str in plan_obj.snapshot_metadata_dict)


@pytest.mark.parametrize("variant_str", ["taa_rank_tqqq", "taa_equal_tqqq", "taa_linearity_qqq"])
def test_actual_taa_host_can_flip_full_fallback_on_one_missing_vix_close(monkeypatch, monthly_input_tuple, variant_str):
    import alpha.data.fred_loader as fred_module
    from strategies.taa_df import strategy_taa_df as base_module
    from strategies.taa_df import strategy_taa_df_fallback_vix_cash_variant_utils as overlay_module
    session_idx = monthly_input_tuple[0].index
    elapsed_vec = np.arange(len(session_idx), dtype=float)
    stale_dict = {"enabled_bool": False}

    def synthetic_price_fn(symbol_str, **keyword_dict):
        close_vec = (100 * np.exp(.0004 * elapsed_vec + .005 * np.sin(.7 * elapsed_vec)) if symbol_str in ("SPY", "$SPXTR")
            else 100 * np.exp(-.0004 * elapsed_vec))
        if symbol_str == "$VIX":
            close_vec = np.full(len(session_idx), 50.0)
            close_vec[-1] = .01
        price_df = pd.DataFrame({"Open": close_vec, "High": close_vec * 1.01, "Low": close_vec * .99,
            "Close": close_vec, "Unadjusted Close": close_vec, "Volume": 1_000_000, "Dividend": 0.0}, index=session_idx)
        if symbol_str == "$VIX" and stale_dict["enabled_bool"]:
            price_df = price_df.iloc[:-1]
        return price_df

    monkeypatch.setattr(base_module, "load_price_timeseries", synthetic_price_fn)
    monkeypatch.setattr(overlay_module, "load_price_timeseries", synthetic_price_fn)
    # FRED transport is isolated; deny its cache write so shared input files
    # cannot change. Include every historical month for valid mapping history.
    fred_bytes = pd.DataFrame({"observation_date": session_idx, "DTB3": 5.0}).to_csv(index=False).encode()
    monkeypatch.setattr(fred_module, "urlopen", lambda *_, **__: io.BytesIO(fred_bytes))
    monkeypatch.setattr(Path, "write_text", lambda *_, **__: (_ for _ in ()).throw(OSError("audit transport has no cache write")))
    release_obj = build_release_obj(variant_str, 1_000_000)
    as_of_ts = datetime(2024, 1, 31, 16, 10, tzinfo=NEW_YORK_TZ)
    fresh_obj = strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, None)
    stale_dict["enabled_bool"] = True
    stale_obj = strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, None)
    fallback_str = "QQQ" if variant_str == "taa_linearity_qqq" else "TQQQ"
    assert fresh_obj.full_target_weight_map_dict == {}
    assert stale_obj.full_target_weight_map_dict == {fallback_str: pytest.approx(1.0)}
    assert "vix_reference_date_str" not in stale_obj.snapshot_metadata_dict
    assert stale_obj.signal_timestamp_ts == fresh_obj.signal_timestamp_ts


@pytest.mark.parametrize("kind_str", ["dollar_atr", "natr20"])
def test_ndx_row_t_prefix_harness_detects_planted_one_session_feature_leak(monthly_input_tuple, kind_str):
    from strategies.momentum import strategy_mo_atr_normalized_ndx as atr_module
    from strategies.momentum import strategy_mo_natr20_ndx_vxn_scaled as natr_module
    pricing_df, universe_df, _ = monthly_input_tuple
    symbol_list = list(universe_df.columns)
    close_df = pricing_df.xs("Close", axis=1, level=1)[symbol_list].copy()
    elapsed_vec = np.arange(len(close_df), dtype=float)
    for symbol_int, symbol_str in enumerate(symbol_list):
        close_df[symbol_str] *= np.exp(.015 * np.sin(.11 * elapsed_vec + symbol_int))
    high_df, low_df, raw_df = close_df * 1.01, close_df * .99, close_df.copy()
    regime_ser = pricing_df[("SPY", "Close")]
    module_obj = atr_module if kind_str == "dollar_atr" else natr_module
    feature_fn = module_obj.compute_atr_normalized_signal_tables if kind_str == "dollar_atr" else module_obj.compute_natr20_signal_tables

    def score_frame_fn(close_input_df, high_input_df, low_input_df, raw_input_df, regime_input_ser):
        config_keyword_dict = {"config" if kind_str == "dollar_atr" else "config_obj": module_obj.DEFAULT_CONFIG}
        return feature_fn(price_close_df=close_input_df, price_high_df=high_input_df,
            price_low_df=low_input_df, regime_close_ser=regime_input_ser, **config_keyword_dict,
            price_unadjusted_close_df=raw_input_df)[6]

    full_score_df = score_frame_fn(close_df, high_df, low_df, raw_df, regime_ser)
    cutoff_list = list(full_score_df.dropna(how="all").index[-13:-1])
    assert len(cutoff_list) == 12
    detected_int = 0
    # *** CRITICAL *** Negative shift is an INTENTIONAL planted lookahead:
    # decision T receives T+1 OHLC. The same exact-row oracle must reject it.
    leaked_score_df = score_frame_fn(close_df.shift(-1), high_df.shift(-1), low_df.shift(-1), raw_df, regime_ser)
    for cutoff_ts in cutoff_list:
        prefix_df = score_frame_fn(close_df.loc[:cutoff_ts], high_df.loc[:cutoff_ts], low_df.loc[:cutoff_ts],
            raw_df.loc[:cutoff_ts], regime_ser.loc[:cutoff_ts])
        pd.testing.assert_series_equal(full_score_df.loc[cutoff_ts], prefix_df.loc[cutoff_ts])
        detected_int += not np.allclose(leaked_score_df.loc[cutoff_ts], prefix_df.loc[cutoff_ts], atol=1e-12, rtol=1e-12)
    assert detected_int == len(cutoff_list)


def test_taa_linearity_default_windows_prefix_invariance_and_one_session_positive_control(monthly_input_tuple):
    from strategies.taa_df.strategy_taa_df_btal_linearity import compute_daily_linearity_score_df
    pricing_df = monthly_input_tuple[0]
    close_df = pricing_df.xs("Close", axis=1, level=1).iloc[:, :5].copy()
    elapsed_vec = np.arange(len(close_df), dtype=float)
    for symbol_int, symbol_str in enumerate(close_df):
        close_df[symbol_str] *= np.exp(.015 * np.sin(.11 * elapsed_vec + symbol_int))
    full_score_df = compute_daily_linearity_score_df(close_df)
    scaled_df = close_df.copy()
    scaled_df.iloc[:, 0] /= 40
    pd.testing.assert_frame_equal(full_score_df, compute_daily_linearity_score_df(scaled_df), rtol=1e-9, atol=1e-12)
    # *** CRITICAL *** Intentional planted future-price read at T; the same
    # row-T oracle checks the untouched implementation against causal prefixes.
    leaked_score_df = compute_daily_linearity_score_df(close_df.shift(-1))
    cutoff_list = close_df.index.to_series().groupby(close_df.index.to_period("M")).max().iloc[-13:-1]
    for cutoff_ts in cutoff_list:
        prefix_df = compute_daily_linearity_score_df(close_df.loc[:cutoff_ts])
        pd.testing.assert_series_equal(full_score_df.loc[cutoff_ts], prefix_df.loc[cutoff_ts])
        assert not np.allclose(leaked_score_df.loc[cutoff_ts], prefix_df.loc[cutoff_ts], atol=1e-12, rtol=1e-12)
