from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from data.norgate_snapshot_store import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    NorgateSnapshotNotReadyError,
    NorgateSnapshotValidationError,
    build_data_source_metadata_dict,
    load_index_constituent_matrix_df,
    load_price_timeseries_df,
    load_valid_snapshot_manifest,
    write_snapshot_files,
)


PROFILE_STR = "norgate_eod_etf_plus_vix_helper"


def _price_snapshot_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol_str": "SPY",
                "adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
            },
            {
                "date": "2024-01-02",
                "symbol_str": "$VIX",
                "adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "Open": 13.0,
                "High": 14.0,
                "Low": 12.5,
                "Close": 13.5,
            },
        ]
    )


def _write_snapshot(tmp_path, profile_str: str = PROFILE_STR) -> None:
    write_snapshot_files(
        snapshot_root_str=str(tmp_path),
        profile_str=profile_str,
        snapshot_date_str="2024-01-02",
        price_df=_price_snapshot_df(),
        required_symbol_list=["SPY"],
        required_helper_symbol_list=["$VIX"],
        adjustment_mode_map_dict={
            "SPY": CAPITALSPECIAL_ADJUSTMENT_STR,
            "$VIX": CAPITALSPECIAL_ADJUSTMENT_STR,
        },
        generated_timestamp_ts=datetime(2024, 1, 2, 23, 15, tzinfo=UTC),
    )


def test_valid_manifest_loads_and_price_shape_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    _write_snapshot(tmp_path)

    snapshot_manifest_obj = load_valid_snapshot_manifest(PROFILE_STR)
    price_df = load_price_timeseries_df(
        "SPY",
        CAPITALSPECIAL_ADJUSTMENT_STR,
        data_profile_str=PROFILE_STR,
    )

    assert snapshot_manifest_obj.snapshot_date_ts == pd.Timestamp("2024-01-02")
    assert list(price_df.columns) == ["Open", "High", "Low", "Close"]
    assert float(price_df.loc[pd.Timestamp("2024-01-02"), "Close"]) == 100.5


def test_loader_snapshot_mode_preserves_raw_price_multiindex_shape(tmp_path, monkeypatch):
    from data.norgate_loader import load_raw_prices

    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    _write_snapshot(tmp_path)

    pricing_data_df = load_raw_prices(
        ["SPY"],
        [],
        start_date="2024-01-02",
        end_date="2024-01-02",
    )

    assert isinstance(pricing_data_df.columns, pd.MultiIndex)
    assert ("SPY", "Close") in pricing_data_df.columns
    assert float(pricing_data_df.loc[pd.Timestamp("2024-01-02"), ("SPY", "Close")]) == 100.5


def test_snapshot_metadata_includes_manifest_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    _write_snapshot(tmp_path)

    metadata_dict = build_data_source_metadata_dict(PROFILE_STR)

    assert metadata_dict["norgate_data_source_mode_str"] == "snapshot"
    assert metadata_dict["norgate_data_profile_str"] == PROFILE_STR
    assert metadata_dict["norgate_snapshot_date_str"] == "2024-01-02"
    assert len(str(metadata_dict["norgate_manifest_hash_str"])) == 64


def test_missing_manifest_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    (tmp_path / PROFILE_STR / "2024-01-02").mkdir(parents=True)

    with pytest.raises(NorgateSnapshotNotReadyError):
        load_valid_snapshot_manifest(PROFILE_STR)


def test_hash_mismatch_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    _write_snapshot(tmp_path)
    price_path_obj = tmp_path / PROFILE_STR / "2024-01-02" / "prices.parquet"
    price_path_obj.write_bytes(b"corrupted snapshot")

    with pytest.raises(NorgateSnapshotValidationError, match="SHA256 mismatch"):
        load_valid_snapshot_manifest(PROFILE_STR)


def test_stale_snapshot_blocks_when_minimum_date_is_requested(tmp_path, monkeypatch):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    _write_snapshot(tmp_path)

    with pytest.raises(NorgateSnapshotValidationError, match="stale"):
        load_valid_snapshot_manifest(PROFILE_STR, minimum_snapshot_date_str="2024-01-03")


def test_wrong_profile_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    _write_snapshot(tmp_path)
    manifest_path_obj = tmp_path / PROFILE_STR / "2024-01-02" / "manifest.json"
    manifest_dict = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
    manifest_dict["profile"] = "wrong_profile"
    manifest_path_obj.write_text(json.dumps(manifest_dict), encoding="utf-8")

    with pytest.raises(NorgateSnapshotValidationError, match="profile mismatch"):
        load_valid_snapshot_manifest(PROFILE_STR)


def test_missing_helper_symbol_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    price_df = _price_snapshot_df().loc[lambda frame_df: frame_df["symbol_str"] == "SPY"]
    write_snapshot_files(
        snapshot_root_str=str(tmp_path),
        profile_str=PROFILE_STR,
        snapshot_date_str="2024-01-02",
        price_df=price_df,
    )

    with pytest.raises(NorgateSnapshotValidationError, match="missing required symbol"):
        load_price_timeseries_df(
            "$VIX",
            CAPITALSPECIAL_ADJUSTMENT_STR,
            data_profile_str=PROFILE_STR,
        )


def test_universe_snapshot_loads_pit_matrix(tmp_path, monkeypatch):
    profile_str = "norgate_eod_sp500_pit"
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    universe_df = pd.DataFrame(
        {"AAPL": [1, 1], "MSFT": [0, 1]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
    )
    write_snapshot_files(
        snapshot_root_str=str(tmp_path),
        profile_str=profile_str,
        snapshot_date_str="2024-01-03",
        price_df=_price_snapshot_df(),
        universe_df=universe_df,
    )

    symbol_list, loaded_universe_df = load_index_constituent_matrix_df(
        "S&P 500",
        data_profile_str=profile_str,
    )

    assert symbol_list == ["AAPL", "MSFT"]
    assert int(loaded_universe_df.loc[pd.Timestamp("2024-01-03"), "MSFT"]) == 1
