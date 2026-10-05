"""Offline price cache tests; fixtures contain no vendor data or network calls."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
from threading import Event
import weakref

import pandas as pd
import pytest

from data import norgate_snapshot_store as snapshot_store_module


@pytest.fixture(autouse=True)
def empty_capsule_price_cache():
    snapshot_store_module.clear_snapshot_manifest_cache()
    snapshot_store_module._read_prices_cached_df.cache_clear()
    yield
    snapshot_store_module.clear_snapshot_manifest_cache()
    snapshot_store_module._read_prices_cached_df.cache_clear()


def _manifest_obj(tmp_path, profile_str, date_str="2026-10-02", manifest_hash_str="first"):
    return snapshot_store_module.NorgateSnapshotManifest(
        profile_str=profile_str, snapshot_date_ts=pd.Timestamp(date_str),
        snapshot_dir_path_obj=tmp_path / profile_str / date_str,
        manifest_dict={}, manifest_hash_str=manifest_hash_str,
    )


def _raw_price_df(close_float=100.5):
    return pd.DataFrame({
        "date": ["2026-10-02"], "symbol_str": ["BIL"],
        "adjustment_str": ["capitalspecial"], "Close": [close_float],
    })


def test_capsule_cache_reuses_one_snapshot_and_isolates_caller_mutations(tmp_path, monkeypatch):
    read_path_list = []
    def read_parquet(path_obj):
        read_path_list.append(path_obj)
        return _raw_price_df()
    monkeypatch.setattr(snapshot_store_module.pd, "read_parquet", read_parquet)
    manifest_obj = _manifest_obj(tmp_path, snapshot_store_module.MR_CAPSULE_DV2_PROFILE_STR)
    first_price_df = snapshot_store_module._read_prices_df(manifest_obj)
    first_price_df.loc[0, "Close"] = 1.0
    second_price_df = snapshot_store_module._read_prices_df(manifest_obj)
    assert second_price_df.loc[0, "Close"] == 100.5
    assert second_price_df.loc[0, "adjustment_str"] == "CAPITALSPECIAL"
    assert second_price_df.loc[0, "date"] == pd.Timestamp("2026-10-02")
    assert len(read_path_list) == 1
    assert snapshot_store_module._read_prices_cached_df.cache_info().currsize == 0


def test_each_capsule_profile_evicts_old_frames_and_retains_no_more_than_two(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot_store_module.pd, "read_parquet", lambda path_obj: _raw_price_df())
    for profile_str in sorted(snapshot_store_module.MR_CAPSULE_PROFILE_SET):
        first_manifest_obj = _manifest_obj(tmp_path, profile_str)
        snapshot_store_module._read_prices_df(first_manifest_obj)
        old_frame_ref = weakref.ref(snapshot_store_module._MR_CAPSULE_PRICE_CACHE_DICT[profile_str][1])
        for day_int in range(3, 15):
            snapshot_store_module._read_prices_df(_manifest_obj(tmp_path, profile_str, f"2026-10-{day_int:02}"))
        gc.collect()
        assert old_frame_ref() is None
    assert len(snapshot_store_module._MR_CAPSULE_PRICE_CACHE_DICT) == 2
    assert snapshot_store_module._read_prices_cached_df.cache_info().currsize == 0


def test_capsule_identity_includes_manifest_hash_and_snapshot_root(tmp_path, monkeypatch):
    read_path_list = []
    def read_parquet(path_obj):
        read_path_list.append(path_obj)
        return _raw_price_df(float(len(read_path_list)))
    monkeypatch.setattr(snapshot_store_module.pd, "read_parquet", read_parquet)
    first_manifest_obj = _manifest_obj(tmp_path, snapshot_store_module.MR_CAPSULE_DV2_PROFILE_STR)
    assert snapshot_store_module._read_prices_df(first_manifest_obj).loc[0, "Close"] == 1.0
    revised_manifest_obj = replace(first_manifest_obj, manifest_hash_str="revised")
    assert snapshot_store_module._read_prices_df(revised_manifest_obj).loc[0, "Close"] == 2.0
    other_root_manifest_obj = replace(revised_manifest_obj, snapshot_dir_path_obj=tmp_path / "other_root")
    assert snapshot_store_module._read_prices_df(other_root_manifest_obj).loc[0, "Close"] == 3.0
    snapshot_store_module.clear_snapshot_manifest_cache()
    assert snapshot_store_module._read_prices_df(other_root_manifest_obj).loc[0, "Close"] == 4.0


def test_concurrent_capsule_reads_share_load_and_return_independent_copies(tmp_path, monkeypatch):
    read_started_event = Event()
    release_read_event = Event()
    read_path_list = []
    def read_parquet(path_obj):
        read_path_list.append(path_obj)
        read_started_event.set()
        assert release_read_event.wait(timeout=5)
        return _raw_price_df()
    monkeypatch.setattr(snapshot_store_module.pd, "read_parquet", read_parquet)
    manifest_obj = _manifest_obj(tmp_path, snapshot_store_module.MR_CAPSULE_HPI_PROFILE_STR)
    with ThreadPoolExecutor(max_workers=4) as executor_obj:
        future_list = [executor_obj.submit(snapshot_store_module._read_prices_df, manifest_obj) for _ in range(4)]
        assert read_started_event.wait(timeout=5)
        release_read_event.set()
        price_frame_list = [future_obj.result(timeout=5) for future_obj in future_list]
    assert len(read_path_list) == 1
    assert len({id(price_df) for price_df in price_frame_list}) == 4
    price_frame_list[0].loc[0, "Close"] = 1.0
    assert all(price_df.loc[0, "Close"] == 100.5 for price_df in price_frame_list[1:])


def test_failed_capsule_refresh_does_not_serve_old_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot_store_module.pd, "read_parquet", lambda path_obj: _raw_price_df())
    manifest_obj = _manifest_obj(tmp_path, snapshot_store_module.MR_CAPSULE_DV2_PROFILE_STR)
    snapshot_store_module._read_prices_df(manifest_obj)
    monkeypatch.setattr(snapshot_store_module.pd, "read_parquet", lambda path_obj: pd.DataFrame({"Close": [1.0]}))
    with pytest.raises(snapshot_store_module.NorgateSnapshotValidationError, match="missing required columns"):
        snapshot_store_module._read_prices_df(replace(manifest_obj, manifest_hash_str="invalid"))
    assert manifest_obj.profile_str not in snapshot_store_module._MR_CAPSULE_PRICE_CACHE_DICT


@pytest.mark.parametrize("profile_str", ["norgate_eod_ndx_pit", "norgate_eod_etf_plus_vix_helper", "norgate_eod_core5"])
def test_non_capsule_price_cache_retains_prior_multi_snapshot_behavior(tmp_path, monkeypatch, profile_str):
    read_path_list = []
    def read_parquet(path_obj):
        read_path_list.append(path_obj)
        return _raw_price_df()
    monkeypatch.setattr(snapshot_store_module.pd, "read_parquet", read_parquet)
    first_manifest_obj = _manifest_obj(tmp_path, profile_str)
    second_manifest_obj = _manifest_obj(tmp_path, profile_str, "2026-10-05")
    for manifest_obj in [first_manifest_obj, second_manifest_obj, first_manifest_obj]:
        snapshot_store_module._read_prices_df(manifest_obj)
    assert len(read_path_list) == 2
    assert snapshot_store_module._read_prices_cached_df.cache_info().currsize == 2
    assert not snapshot_store_module._MR_CAPSULE_PRICE_CACHE_DICT
