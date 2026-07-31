from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest import mock

import pandas as pd
import pytest

from data.norgate_snapshot_store import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    LEGACY_SNAPSHOT_SCHEMA_VERSION_INT,
    NorgateSnapshotNotReadyError,
    NorgateSnapshotValidationError,
    SNAPSHOT_SCHEMA_VERSION_INT,
    TOTALRETURN_ADJUSTMENT_STR,
    build_data_source_metadata_dict,
    load_index_constituent_matrix_df,
    load_price_timeseries_df,
    load_raw_prices_df,
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


def test_legacy_schema_without_dividend_remains_readable(tmp_path, monkeypatch):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    _write_snapshot(tmp_path)

    snapshot_manifest_obj = load_valid_snapshot_manifest(PROFILE_STR)

    assert snapshot_manifest_obj.manifest_dict["schema_version"] == LEGACY_SNAPSHOT_SCHEMA_VERSION_INT


def test_current_schema_requires_dividend_when_written(tmp_path):
    with pytest.raises(NorgateSnapshotValidationError, match="Dividend"):
        write_snapshot_files(
            snapshot_root_str=str(tmp_path),
            profile_str=PROFILE_STR,
            snapshot_date_str="2024-01-02",
            price_df=_price_snapshot_df(),
            schema_version_int=SNAPSHOT_SCHEMA_VERSION_INT,
        )


def test_current_schema_rejects_null_dividend_values(tmp_path):
    price_df = _price_snapshot_df().assign(Dividend=[0.0, None])

    with pytest.raises(NorgateSnapshotValidationError, match="must not contain null"):
        write_snapshot_files(
            snapshot_root_str=str(tmp_path),
            profile_str=PROFILE_STR,
            snapshot_date_str="2024-01-02",
            price_df=price_df,
            schema_version_int=SNAPSHOT_SCHEMA_VERSION_INT,
        )


def test_current_schema_rejects_nonnumeric_dividend_values(tmp_path):
    price_df = _price_snapshot_df().assign(Dividend="not_numeric")

    with pytest.raises(NorgateSnapshotValidationError, match="must be numeric"):
        write_snapshot_files(
            snapshot_root_str=str(tmp_path),
            profile_str=PROFILE_STR,
            snapshot_date_str="2024-01-02",
            price_df=price_df,
            schema_version_int=SNAPSHOT_SCHEMA_VERSION_INT,
        )


def test_current_schema_without_dividend_blocks_when_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    _write_snapshot(tmp_path)
    manifest_path_obj = tmp_path / PROFILE_STR / "2024-01-02" / "manifest.json"
    manifest_dict = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
    manifest_dict["schema_version"] = SNAPSHOT_SCHEMA_VERSION_INT
    manifest_path_obj.write_text(json.dumps(manifest_dict), encoding="utf-8")

    with pytest.raises(NorgateSnapshotValidationError, match="Dividend"):
        load_valid_snapshot_manifest(PROFILE_STR)


def test_current_schema_with_null_dividend_blocks_when_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    price_df = _price_snapshot_df().assign(Dividend=[0.0, None])
    write_snapshot_files(
        snapshot_root_str=str(tmp_path),
        profile_str=PROFILE_STR,
        snapshot_date_str="2024-01-02",
        price_df=price_df,
        schema_version_int=LEGACY_SNAPSHOT_SCHEMA_VERSION_INT,
    )
    manifest_path_obj = tmp_path / PROFILE_STR / "2024-01-02" / "manifest.json"
    manifest_dict = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
    manifest_dict["schema_version"] = SNAPSHOT_SCHEMA_VERSION_INT
    manifest_path_obj.write_text(json.dumps(manifest_dict), encoding="utf-8")

    with pytest.raises(NorgateSnapshotValidationError, match="must not contain null"):
        load_valid_snapshot_manifest(PROFILE_STR)


def test_current_schema_with_dividend_loads_and_declares_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    price_df = _price_snapshot_df().assign(Dividend=0.0)
    write_snapshot_files(
        snapshot_root_str=str(tmp_path),
        profile_str=PROFILE_STR,
        snapshot_date_str="2024-01-02",
        price_df=price_df,
        schema_version_int=SNAPSHOT_SCHEMA_VERSION_INT,
    )

    snapshot_manifest_obj = load_valid_snapshot_manifest(PROFILE_STR)
    metadata_dict = build_data_source_metadata_dict(PROFILE_STR)

    assert snapshot_manifest_obj.manifest_dict["schema_version"] == SNAPSHOT_SCHEMA_VERSION_INT
    assert metadata_dict["norgate_snapshot_schema_version_int"] == SNAPSHOT_SCHEMA_VERSION_INT
    assert metadata_dict["norgate_dividend_field_required_bool"] is True


def test_current_ndx_schema_requires_spy_price_and_total_return_rows(tmp_path, monkeypatch):
    profile_str = "norgate_eod_ndx_pit"
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    price_df = _price_snapshot_df().loc[
        lambda frame_df: frame_df["symbol_str"] == "SPY"
    ].assign(Dividend=0.0)
    write_snapshot_files(
        snapshot_root_str=str(tmp_path),
        profile_str=profile_str,
        snapshot_date_str="2024-01-02",
        price_df=price_df,
        schema_version_int=SNAPSHOT_SCHEMA_VERSION_INT,
    )

    with pytest.raises(NorgateSnapshotValidationError, match="SPY adjustment"):
        load_valid_snapshot_manifest(profile_str)


def test_current_ndx_schema_accepts_spy_price_and_total_return_rows(tmp_path, monkeypatch):
    profile_str = "norgate_eod_ndx_pit"
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    spy_price_df = _price_snapshot_df().loc[
        lambda frame_df: frame_df["symbol_str"] == "SPY"
    ].assign(Dividend=0.0)
    spy_total_return_df = spy_price_df.copy()
    spy_total_return_df["adjustment_str"] = TOTALRETURN_ADJUSTMENT_STR
    write_snapshot_files(
        snapshot_root_str=str(tmp_path),
        profile_str=profile_str,
        snapshot_date_str="2024-01-02",
        price_df=pd.concat([spy_price_df, spy_total_return_df], ignore_index=True),
        schema_version_int=SNAPSHOT_SCHEMA_VERSION_INT,
    )

    snapshot_manifest_obj = load_valid_snapshot_manifest(profile_str)

    assert snapshot_manifest_obj.manifest_dict["schema_version"] == SNAPSHOT_SCHEMA_VERSION_INT


def test_ndx_export_profiles_include_price_and_total_return_spy():
    from scripts.export_norgate_snapshot import PROFILE_EXPORT_SPEC_DICT

    for profile_str in (
        "norgate_eod_ndx_pit",
        "norgate_eod_ndx_pit_plus_vxn_helper",
    ):
        profile_spec_obj = PROFILE_EXPORT_SPEC_DICT[profile_str]
        assert "SPY" in profile_spec_obj.capital_symbol_tuple
        assert "SPY" in profile_spec_obj.total_return_symbol_tuple


def test_hpi_export_profile_preserves_strict_data_contract():
    from scripts.export_norgate_snapshot import PROFILE_EXPORT_SPEC_DICT

    profile_spec_obj = PROFILE_EXPORT_SPEC_DICT[
        "norgate_eod_sp500_hpi_pit"
    ]

    assert profile_spec_obj.indexname_str == "S&P 500"
    assert profile_spec_obj.padding_type_name_str == "NONE"
    assert profile_spec_obj.trim_past_member_tail_bool is False
    assert profile_spec_obj.total_return_symbol_tuple == ("$SPX", "$SPXTR")


def test_hpi_export_writes_end_to_end_strict_contract(tmp_path, monkeypatch):
    from data.norgate_snapshot_store import (
        HPI_SP500_DATA_CONTRACT_DICT,
        HPI_SP500_PROFILE_STR,
    )
    from scripts import export_norgate_snapshot as export_module

    universe_df = pd.DataFrame(
        {"AAPL": [1]},
        index=[pd.Timestamp("2024-01-02")],
    )
    price_df = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol_str": "AAPL",
                "adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "Dividend": 0.0,
            },
            {
                "date": "2024-01-02",
                "symbol_str": "$SPX",
                "adjustment_str": TOTALRETURN_ADJUSTMENT_STR,
                "Open": 4_700.0,
                "High": 4_710.0,
                "Low": 4_690.0,
                "Close": 4_705.0,
                "Dividend": 0.0,
            },
            {
                "date": "2024-01-02",
                "symbol_str": "$SPXTR",
                "adjustment_str": TOTALRETURN_ADJUSTMENT_STR,
                "Open": 10_000.0,
                "High": 10_010.0,
                "Low": 9_990.0,
                "Close": 10_005.0,
                "Dividend": 0.0,
            },
        ]
    )
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))

    with mock.patch.object(
        export_module,
        "_load_index_constituent_matrix_df",
        return_value=(["AAPL"], universe_df),
    ) as universe_loader_mock, mock.patch.object(
        export_module,
        "_build_price_snapshot_df",
        return_value=price_df,
    ) as price_builder_mock:
        export_module._export_profile_to_root_path(
            snapshot_root_str=str(tmp_path),
            profile_str=HPI_SP500_PROFILE_STR,
            snapshot_date_str="2024-01-02",
            start_date_str="1998-01-01",
            end_date_str="2024-01-02",
            overwrite_bool=False,
        )

    universe_loader_mock.assert_called_once_with(
        "S&P 500",
        trim_past_member_tail_bool=False,
    )
    assert (
        price_builder_mock.call_args.kwargs["padding_type_name_str"]
        == "NONE"
    )
    assert price_builder_mock.call_args.kwargs[
        "total_return_symbol_list"
    ] == ["$SPX", "$SPXTR"]
    snapshot_manifest_obj = load_valid_snapshot_manifest(
        HPI_SP500_PROFILE_STR
    )
    assert (
        snapshot_manifest_obj.manifest_dict["data_contract"]
        == HPI_SP500_DATA_CONTRACT_DICT
    )


def test_hpi_snapshot_rejects_missing_strict_contract(tmp_path, monkeypatch):
    from data.norgate_snapshot_store import (
        HPI_SP500_PROFILE_STR,
        NorgateSnapshotValidationError,
        load_valid_snapshot_manifest,
    )

    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    write_snapshot_files(
        snapshot_root_str=str(tmp_path),
        profile_str=HPI_SP500_PROFILE_STR,
        snapshot_date_str="2024-01-02",
        price_df=_price_snapshot_df(),
    )

    with pytest.raises(
        NorgateSnapshotValidationError,
        match="data_contract mismatch",
    ):
        load_valid_snapshot_manifest(HPI_SP500_PROFILE_STR)


def test_hpi_export_requests_no_padding():
    from types import SimpleNamespace

    from scripts import export_norgate_snapshot as export_module

    raw_price_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Dividend": [0.0],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    direct_norgate_mock = mock.Mock()
    direct_norgate_mock.PaddingType = SimpleNamespace(NONE="none")
    direct_norgate_mock.price_timeseries.return_value = raw_price_df

    with mock.patch.object(
        export_module,
        "_load_direct_norgate_module",
        return_value=direct_norgate_mock,
    ), mock.patch.object(
        export_module,
        "_adjustment_type_obj",
        return_value=object(),
    ):
        export_module._load_price_frame_df(
            symbol_str="AAPL",
            adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR,
            start_date_str="2024-01-02",
            end_date_str="2024-01-02",
            padding_type_name_str="NONE",
        )

    assert (
        direct_norgate_mock.price_timeseries.call_args.kwargs[
            "padding_setting"
        ]
        == "none"
    )


def test_hpi_export_does_not_trim_past_member_tail():
    from scripts import export_norgate_snapshot as export_module

    date_idx = pd.bdate_range("2024-01-02", periods=10)
    membership_df = pd.DataFrame(
        {"Index Constituent": 1},
        index=date_idx,
    )
    heartbeat_df = pd.DataFrame(index=pd.bdate_range("2024-01-02", periods=15))
    direct_norgate_mock = mock.Mock()
    direct_norgate_mock.watchlist_symbols.return_value = ["OLD"]
    direct_norgate_mock.price_timeseries.return_value = heartbeat_df
    direct_norgate_mock.index_constituent_timeseries.return_value = (
        membership_df
    )

    with mock.patch.object(
        export_module,
        "_load_direct_norgate_module",
        return_value=direct_norgate_mock,
    ):
        _, exact_universe_df = (
            export_module._load_index_constituent_matrix_df(
                "S&P 500",
                trim_past_member_tail_bool=False,
            )
        )
        _, shared_universe_df = (
            export_module._load_index_constituent_matrix_df(
                "S&P 500",
                trim_past_member_tail_bool=True,
            )
        )

    assert len(exact_universe_df.index) == 10
    assert len(shared_universe_df.index) == 5


def test_snapshot_export_normalizes_dividend_only_for_non_distributing_indices():
    from scripts import export_norgate_snapshot as export_module

    raw_index_price_df = pd.DataFrame(
        {
            "Open": [20.0],
            "High": [21.0],
            "Low": [19.0],
            "Close": [20.5],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    direct_norgate_mock = mock.Mock()
    direct_norgate_mock.price_timeseries.return_value = raw_index_price_df

    with mock.patch.object(
        export_module,
        "_load_direct_norgate_module",
        return_value=direct_norgate_mock,
    ), mock.patch.object(
        export_module,
        "_adjustment_type_obj",
        return_value=object(),
    ):
        price_df = export_module._load_price_frame_df(
            symbol_str="$VXN",
            adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR,
            start_date_str="2024-01-02",
            end_date_str="2024-01-02",
        )

    assert float(price_df.loc[0, "Dividend"]) == 0.0


def test_snapshot_export_rejects_missing_dividend_for_etf():
    from scripts import export_norgate_snapshot as export_module

    raw_etf_price_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    direct_norgate_mock = mock.Mock()
    direct_norgate_mock.price_timeseries.return_value = raw_etf_price_df

    with mock.patch.object(
        export_module,
        "_load_direct_norgate_module",
        return_value=direct_norgate_mock,
    ), mock.patch.object(
        export_module,
        "_adjustment_type_obj",
        return_value=object(),
    ):
        with pytest.raises(RuntimeError, match="missing Dividend"):
            export_module._load_price_frame_df(
                symbol_str="SPY",
                adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR,
                start_date_str="2024-01-02",
                end_date_str="2024-01-02",
            )


def test_snapshot_export_accepts_spxtr_without_dividend():
    from scripts import export_norgate_snapshot as export_module

    raw_index_price_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    direct_norgate_mock = mock.Mock()
    direct_norgate_mock.price_timeseries.return_value = raw_index_price_df

    with mock.patch.object(
        export_module,
        "_load_direct_norgate_module",
        return_value=direct_norgate_mock,
    ), mock.patch.object(
        export_module,
        "_adjustment_type_obj",
        return_value=object(),
    ):
        price_df = export_module._load_price_frame_df(
            symbol_str="$SPXTR",
            adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
            start_date_str="2024-01-02",
            end_date_str="2024-01-02",
        )

    assert float(price_df.loc[0, "Dividend"]) == 0.0


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


def test_raw_snapshot_loader_reads_price_table_once(tmp_path, monkeypatch):
    import data.norgate_snapshot_store as snapshot_store_module

    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    _write_snapshot(tmp_path)

    with mock.patch.object(
        snapshot_store_module,
        "_read_prices_df",
        wraps=snapshot_store_module._read_prices_df,
    ) as read_prices_mock:
        pricing_data_df = load_raw_prices_df(
            symbols=["SPY"],
            benchmarks=[],
            start_date_str="2024-01-02",
            end_date_str="2024-01-02",
            data_profile_str=PROFILE_STR,
        )

    assert read_prices_mock.call_count == 1
    assert ("SPY", "Close") in pricing_data_df.columns


def test_bulk_raw_snapshot_loader_matches_repeated_symbol_loads(
    tmp_path,
    monkeypatch,
):
    from data.norgate_snapshot_store import (
        HPI_SP500_DATA_CONTRACT_DICT,
        HPI_SP500_PROFILE_STR,
    )

    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    price_df = pd.DataFrame(
        [
            {
                "date": date_str,
                "symbol_str": symbol_str,
                "adjustment_str": adjustment_str,
                "Open": close_float - 0.5,
                "High": close_float + 1.0,
                "Low": close_float - 1.0,
                "Close": close_float,
                "Dividend": 0.0,
            }
            for date_str, symbol_str, adjustment_str, close_float in [
                (
                    "2024-01-02",
                    "AAPL",
                    CAPITALSPECIAL_ADJUSTMENT_STR,
                    100.0,
                ),
                (
                    "2024-01-03",
                    "AAPL",
                    CAPITALSPECIAL_ADJUSTMENT_STR,
                    101.0,
                ),
                (
                    "2024-01-03",
                    "MSFT",
                    CAPITALSPECIAL_ADJUSTMENT_STR,
                    200.0,
                ),
                (
                    "2024-01-02",
                    "$SPXTR",
                    TOTALRETURN_ADJUSTMENT_STR,
                    10_000.0,
                ),
                (
                    "2024-01-03",
                    "$SPXTR",
                    TOTALRETURN_ADJUSTMENT_STR,
                    10_010.0,
                ),
            ]
        ]
    )
    write_snapshot_files(
        snapshot_root_str=str(tmp_path),
        profile_str=HPI_SP500_PROFILE_STR,
        snapshot_date_str="2024-01-03",
        price_df=price_df,
        required_symbol_list=["AAPL", "MSFT", "$SPXTR"],
        adjustment_mode_map_dict={
            "AAPL": CAPITALSPECIAL_ADJUSTMENT_STR,
            "MSFT": CAPITALSPECIAL_ADJUSTMENT_STR,
            "$SPXTR": TOTALRETURN_ADJUSTMENT_STR,
        },
        data_contract_dict=HPI_SP500_DATA_CONTRACT_DICT,
    )

    bulk_price_df = load_raw_prices_df(
        symbols=["AAPL", "MSFT"],
        benchmarks=["$SPXTR"],
        start_date_str="2024-01-02",
        end_date_str="2024-01-03",
        data_profile_str=HPI_SP500_PROFILE_STR,
    )
    expected_frame_list = []
    for symbol_str, adjustment_str in [
        ("AAPL", CAPITALSPECIAL_ADJUSTMENT_STR),
        ("MSFT", CAPITALSPECIAL_ADJUSTMENT_STR),
        ("$SPXTR", TOTALRETURN_ADJUSTMENT_STR),
    ]:
        symbol_price_df = load_price_timeseries_df(
            symbol_str,
            adjustment_str,
            start_date_str="2024-01-02",
            end_date_str="2024-01-03",
            data_profile_str=HPI_SP500_PROFILE_STR,
        )
        symbol_price_df.columns = pd.MultiIndex.from_tuples(
            [
                (symbol_str, field_str)
                for field_str in symbol_price_df.columns
            ]
        )
        expected_frame_list.append(symbol_price_df)
    expected_price_df = pd.concat(
        expected_frame_list,
        axis=1,
    ).sort_index()

    pd.testing.assert_frame_equal(bulk_price_df, expected_price_df)
    assert list(
        bulk_price_df.columns.get_level_values(0).unique()
    ) == ["AAPL", "MSFT", "$SPXTR"]
    assert pd.isna(
        bulk_price_df.loc[
            pd.Timestamp("2024-01-02"),
            ("MSFT", "Close"),
        ]
    )


def test_snapshot_raw_prices_skips_pit_symbols_outside_requested_date_range(tmp_path, monkeypatch):
    profile_str = "norgate_eod_sp500_pit"
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    price_df = pd.DataFrame(
        [
            {
                "date": "1997-01-02",
                "symbol_str": "AAL-199702",
                "adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "Open": 10.0,
                "High": 11.0,
                "Low": 9.0,
                "Close": 10.5,
            },
            {
                "date": "2024-01-02",
                "symbol_str": "$SPX",
                "adjustment_str": TOTALRETURN_ADJUSTMENT_STR,
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
            },
        ]
    )
    write_snapshot_files(
        snapshot_root_str=str(tmp_path),
        profile_str=profile_str,
        snapshot_date_str="2024-01-02",
        price_df=price_df,
    )

    pricing_data_df = load_raw_prices_df(
        symbols=["AAL-199702"],
        benchmarks=["$SPX"],
        start_date_str="1998-01-01",
        end_date_str="2024-01-02",
        data_profile_str=profile_str,
    )

    assert ("AAL-199702", "Close") not in pricing_data_df.columns
    assert ("$SPX", "Close") in pricing_data_df.columns


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
