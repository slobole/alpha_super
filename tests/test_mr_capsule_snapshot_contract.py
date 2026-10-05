"""Synthetic-only transport tests: no direct Norgate or active snapshots."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import exchange_calendars
import numpy as np
import pandas as pd
import pytest

from data import norgate_snapshot_store as store_mod
from scripts import export_norgate_snapshot as export_mod


SNAPSHOT_DATE_STR = "2020-01-10"
PROFILE_TUPLE = (store_mod.MR_CAPSULE_DV2_PROFILE_STR, store_mod.MR_CAPSULE_HPI_PROFILE_STR)


def _price_frame_df() -> pd.DataFrame:
    session_idx = exchange_calendars.get_calendar("XNYS", start="1990-01-02", end=SNAPSHOT_DATE_STR).sessions.tz_localize(None)
    frame_list = []
    for symbol_str in ("AAA", "BIL", "SPMO", "$VIX", "$SPX", "$SPXTR"):
        symbol_date_idx = session_idx[-30:] if symbol_str in {"AAA", "BIL", "SPMO"} else session_idx
        if symbol_str == "$VIX":
            symbol_date_idx = symbol_date_idx.difference(pd.DatetimeIndex(["1991-03-01"]))
        elif symbol_str in {"$SPX", "$SPXTR"}:
            symbol_date_idx = symbol_date_idx[symbol_date_idx >= pd.Timestamp("1998-01-01")]
        frame_list.append(pd.DataFrame({
            "date": symbol_date_idx, "symbol_str": symbol_str,
            "adjustment_str": "TOTALRETURN" if symbol_str in {"$SPX", "$SPXTR"} else "CAPITALSPECIAL",
            "Open": 20.0, "High": 21.0, "Low": 19.0, "Close": 20.0, "Volume": 1000.0, "Dividend": 0.0,
        }))
    return pd.concat(frame_list, ignore_index=True)


def _write_snapshot(tmp_path, profile_str, price_df=None, contract_change_dict=None, schema_version_int=2):
    price_df = _price_frame_df() if price_df is None else price_df
    contract_dict = copy.deepcopy(store_mod.MR_CAPSULE_DATA_CONTRACT_BY_PROFILE_DICT[profile_str])
    contract_dict["series_coverage_dict"] = store_mod.mr_capsule_price_coverage_dict(_price_frame_df(), pd.Timestamp(SNAPSHOT_DATE_STR))
    contract_dict["source_field_by_pair_dict"] = {
        f"{symbol_str}|{'TOTALRETURN' if symbol_str in {'$SPX', '$SPXTR'} else 'CAPITALSPECIAL'}": ["Open", "High", "Low", "Close", "Volume", "Dividend"]
        for symbol_str in ("AAA", "BIL", "SPMO", "$VIX", "$SPX", "$SPXTR")
    }
    contract_dict["source_dtype_by_pair_dict"] = {pair_str: {field_str: str(price_df[field_str].dtype) for field_str in field_list}
        for pair_str, field_list in contract_dict["source_field_by_pair_dict"].items()}
    contract_dict["source_request_by_pair_dict"] = {
        pair_str: {"requested_start_date_str": "1990-01-02" if pair_str.startswith("$VIX|") else "1998-01-01",
                   "effective_start_date_str": "1990-01-02" if pair_str.startswith("$VIX|") else "1998-01-01",
                   "end_date_str": SNAPSHOT_DATE_STR, "empty_primary_request_bool": False,
                   "padding_type_name_str": contract_dict["price_padding_by_symbol_dict"].get(pair_str.split("|")[0], contract_dict["price_padding_setting_str"])}
        for pair_str in contract_dict["source_field_by_pair_dict"]
    }
    contract_dict.update(contract_change_dict or {})
    return store_mod.write_snapshot_files(
        snapshot_root_str=str(tmp_path), profile_str=profile_str, snapshot_date_str=SNAPSHOT_DATE_STR,
        price_df=price_df, universe_df=pd.DataFrame({"AAA": [1]}, index=pd.DatetimeIndex([SNAPSHOT_DATE_STR])),
        required_symbol_list=["AAA", "BIL", "SPMO", "$SPX", "$SPXTR"], required_helper_symbol_list=["$VIX"],
        adjustment_mode_map_dict={symbol_str: "TOTALRETURN" if symbol_str in {"$SPX", "$SPXTR"} else "CAPITALSPECIAL"
                                  for symbol_str in ("AAA", "BIL", "SPMO", "$VIX", "$SPX", "$SPXTR")},
        data_contract_dict=contract_dict, schema_version_int=schema_version_int,
    )


@pytest.mark.parametrize("profile_str", PROFILE_TUPLE)
def test_capsule_snapshot_preserves_source_roles_and_reads_vix_inception(tmp_path, profile_str):
    _write_snapshot(tmp_path, profile_str)
    manifest_obj = store_mod.load_valid_snapshot_manifest(profile_str, snapshot_root_str=str(tmp_path))
    assert manifest_obj.snapshot_date_ts == pd.Timestamp(SNAPSHOT_DATE_STR)
    coverage_dict = manifest_obj.manifest_dict["data_contract"]["series_coverage_dict"]
    assert coverage_dict["$VIX|CAPITALSPECIAL"]["first_observed_date_str"] == "1990-01-02"
    assert coverage_dict["$VIX|CAPITALSPECIAL"]["last_observed_date_str"] == SNAPSHOT_DATE_STR
    assert coverage_dict["BIL|CAPITALSPECIAL"]["observed_row_count_int"] == 30


@pytest.mark.parametrize("profile_str", PROFILE_TUPLE)
def test_capsule_spx_benchmark_uses_true_total_return_series(tmp_path, monkeypatch, profile_str):
    price_df = _price_frame_df()
    price_df.loc[price_df["symbol_str"].eq("$SPXTR"), "Close"] = 42.0
    _write_snapshot(tmp_path, profile_str, price_df=price_df)
    manifest_obj = store_mod.load_valid_snapshot_manifest(profile_str, snapshot_root_str=str(tmp_path))
    monkeypatch.setattr(store_mod, "load_valid_snapshot_manifest", lambda *_: manifest_obj)
    loaded_price_df = store_mod.load_raw_prices_df(
        ["BIL"], ["$SPX"], start_date_str="2020-01-01", end_date_str=SNAPSHOT_DATE_STR,
        data_profile_str=profile_str,
    )
    assert loaded_price_df[("$SPX", "Close")].eq(42.0).all()
    assert loaded_price_df[("BIL", "Close")].eq(20.0).all()


@pytest.mark.parametrize("fault_str", ("stale", "gap", "duplicate", "late_start", "nan", "zero", "infinite"))
def test_bad_vix_history_fails_before_decisions(tmp_path, fault_str):
    price_df = _price_frame_df()
    vix_idx = price_df.index[price_df["symbol_str"].eq("$VIX")]
    if fault_str in {"stale", "gap", "late_start"}:
        remove_int = vix_idx[-1] if fault_str == "stale" else vix_idx[400] if fault_str == "gap" else vix_idx[0]
        price_df = price_df.drop(index=remove_int)
    elif fault_str == "duplicate":
        price_df = pd.concat([price_df, price_df.loc[[vix_idx[-1]]]], ignore_index=True)
    else:
        price_df.loc[vix_idx[400], "Close"] = {"nan": np.nan, "zero": 0.0, "infinite": np.inf}[fault_str]
    _write_snapshot(tmp_path, PROFILE_TUPLE[0], price_df=price_df)
    with pytest.raises(store_mod.NorgateSnapshotValidationError, match="MR capsule"):
        store_mod.load_valid_snapshot_manifest(PROFILE_TUPLE[0], snapshot_root_str=str(tmp_path))


@pytest.mark.parametrize("contract_change_dict", (
    {"price_padding_setting_str": "ALLMARKETDAYS"},
    {"price_padding_by_symbol_dict": {"$VIX": "ALLMARKETDAYS", "BIL": "ALLMARKETDAYS", "SPMO": "ALLMARKETDAYS"}},
    {"past_member_tail_policy_str": "trim_last_5"},
    {"vix_history_start_date_str": "1998-01-01"},
    {"vix_known_missing_session_list": []},
    {"source_request_by_pair_dict": {}},
    {"source_field_by_pair_dict": {}},
    {"series_coverage_dict": {}},
))
def test_hpi_capsule_rejects_declared_contract_drift(tmp_path, contract_change_dict):
    _write_snapshot(tmp_path, PROFILE_TUPLE[1], contract_change_dict=contract_change_dict)
    with pytest.raises(store_mod.NorgateSnapshotValidationError, match="MR capsule"):
        store_mod.load_valid_snapshot_manifest(PROFILE_TUPLE[1], snapshot_root_str=str(tmp_path))


@pytest.mark.parametrize("profile_str", PROFILE_TUPLE)
def test_capsule_known_vix_gap_preserves_observations_and_expanding_gate(tmp_path, monkeypatch, profile_str):
    from strategies.mr_capsule.vix_stress_gate import stress_gate_open_ser

    price_df = _price_frame_df()
    vix_mask_ser = price_df["symbol_str"].eq("$VIX")
    price_df.loc[vix_mask_ser, "Close"] = 20. + 5. * np.sin(np.arange(int(vix_mask_ser.sum())) / 50.)
    native_vix_ser = price_df.loc[vix_mask_ser].set_index("date")["Close"]
    native_vix_ser.index.name = None
    _write_snapshot(tmp_path, profile_str, price_df=price_df)
    manifest_obj = store_mod.load_valid_snapshot_manifest(profile_str, snapshot_root_str=str(tmp_path))
    assert manifest_obj.manifest_dict["data_contract"]["vix_known_missing_session_list"] == ["1991-03-01"]
    monkeypatch.setattr(store_mod, "load_valid_snapshot_manifest", lambda *_: manifest_obj)
    loaded_vix_ser = store_mod.load_price_timeseries_df("$VIX", data_profile_str=profile_str)["Close"]
    assert pd.Timestamp("1991-03-01") not in loaded_vix_ser.index
    pd.testing.assert_series_equal(loaded_vix_ser, native_vix_ser, check_freq=False)
    pd.testing.assert_series_equal(stress_gate_open_ser(loaded_vix_ser), stress_gate_open_ser(native_vix_ser), check_freq=False)


@pytest.mark.parametrize("profile_str", PROFILE_TUPLE)
def test_capsule_backfilled_known_vix_gap_requires_new_qualification(tmp_path, profile_str):
    price_df = _price_frame_df()
    backfill_df = price_df.loc[price_df["symbol_str"].eq("$VIX")].iloc[:1].copy()
    backfill_df["date"] = pd.Timestamp("1991-03-01")
    price_df = pd.concat([price_df, backfill_df], ignore_index=True)
    _write_snapshot(tmp_path, profile_str, price_df=price_df)
    with pytest.raises(store_mod.NorgateSnapshotValidationError, match="backfilled.*requalification"):
        store_mod.load_valid_snapshot_manifest(profile_str, snapshot_root_str=str(tmp_path))


@pytest.mark.parametrize("fault_str", ("legacy", "wrong_bil_adjustment", "missing_bil", "missing_universe", "missing_helper", "infinite_dividend"))
def test_capsule_rejects_incomplete_transport_contract(tmp_path, fault_str):
    price_df = _price_frame_df()
    if fault_str == "wrong_bil_adjustment":
        price_df.loc[price_df["symbol_str"].eq("BIL"), "adjustment_str"] = "TOTALRETURN"
    elif fault_str == "missing_bil":
        price_df = price_df.loc[~price_df["symbol_str"].eq("BIL")]
    elif fault_str == "infinite_dividend":
        price_df.loc[price_df["symbol_str"].eq("BIL"), "Dividend"] = np.inf
    snapshot_path_obj = _write_snapshot(tmp_path, PROFILE_TUPLE[0], price_df=price_df, schema_version_int=1 if fault_str == "legacy" else 2)
    if fault_str in {"missing_universe", "missing_helper"}:
        manifest_path_obj = snapshot_path_obj / "manifest.json"
        manifest_dict = json.loads(manifest_path_obj.read_text())
        if fault_str == "missing_universe":
            del manifest_dict["files"]["universe.parquet"]
        else:
            manifest_dict["required_helpers"] = []
        manifest_path_obj.write_text(json.dumps(manifest_dict))
    with pytest.raises(store_mod.NorgateSnapshotValidationError):
        store_mod.load_valid_snapshot_manifest(PROFILE_TUPLE[0], snapshot_root_str=str(tmp_path))


@pytest.mark.parametrize("volume_float", (np.nan, np.inf, -1.0))
def test_capsule_rejects_unknown_or_invalid_spmo_tradability(tmp_path, volume_float):
    price_df = _price_frame_df()
    price_df.loc[price_df["symbol_str"].eq("SPMO"), "Volume"] = volume_float
    _write_snapshot(tmp_path, PROFILE_TUPLE[1], price_df=price_df)
    with pytest.raises(store_mod.NorgateSnapshotValidationError, match="SPMO Volume"):
        store_mod.load_valid_snapshot_manifest(PROFILE_TUPLE[1], snapshot_root_str=str(tmp_path))


def test_capsule_accepts_known_zero_volume_padding_but_requires_native_volume_provenance(tmp_path):
    price_df = _price_frame_df()
    price_df.loc[price_df["symbol_str"].eq("SPMO"), "Volume"] = 0.0
    _write_snapshot(tmp_path, PROFILE_TUPLE[0], price_df=price_df)
    store_mod.load_valid_snapshot_manifest(PROFILE_TUPLE[0], snapshot_root_str=str(tmp_path))
    _write_snapshot(tmp_path, PROFILE_TUPLE[1], price_df=price_df, contract_change_dict={
        "source_field_by_pair_dict": {
            "BIL|CAPITALSPECIAL": ["Dividend"], "SPMO|CAPITALSPECIAL": ["Dividend"],
        },
    })
    with pytest.raises(store_mod.NorgateSnapshotValidationError, match="Volume.*provenance"):
        store_mod.load_valid_snapshot_manifest(PROFILE_TUPLE[1], snapshot_root_str=str(tmp_path))


@pytest.mark.parametrize("profile_str", PROFILE_TUPLE)
@pytest.mark.parametrize("requested_start_str", ["1990-01-01", "1998-01-01"])
def test_export_uses_per_symbol_padding_and_full_unpadded_vix(monkeypatch, tmp_path, profile_str, requested_start_str):
    call_dict = {}
    source_df = _price_frame_df()

    def fake_price_loader(*, symbol_str, adjustment_str, start_date_str, end_date_str, padding_type_name_str, empty_history_fallback_start_date_str=None):
        call_dict[symbol_str] = (adjustment_str, start_date_str, end_date_str, padding_type_name_str)
        symbol_df = source_df.loc[source_df["symbol_str"].eq(symbol_str)].copy()
        symbol_df.attrs.update(source_field_list=["Open", "High", "Low", "Close", "Volume", "Dividend"],
                               source_index_name_str=None, source_dtype_dict={field_str: str(symbol_df[field_str].dtype)
                                   for field_str in ("Open", "High", "Low", "Close", "Volume", "Dividend")})
        symbol_df.attrs["source_request_dict"] = {"requested_start_date_str": start_date_str, "effective_start_date_str": start_date_str,
            "end_date_str": end_date_str, "padding_type_name_str": padding_type_name_str, "empty_primary_request_bool": False}
        return symbol_df

    monkeypatch.setattr(export_mod, "_load_price_frame_df", fake_price_loader)
    monkeypatch.setattr(export_mod, "_load_index_constituent_matrix_df", lambda *args, **kwargs: (
        ["AAA"], pd.DataFrame({"AAA": [1]}, index=pd.DatetimeIndex([SNAPSHOT_DATE_STR])),
    ))
    export_mod.export_profile_snapshot(snapshot_root_str=str(tmp_path), profile_str=profile_str,
                                       snapshot_date_str=SNAPSHOT_DATE_STR, start_date_str=requested_start_str)
    stock_padding_str = "NONE" if profile_str == PROFILE_TUPLE[1] else "ALLMARKETDAYS"
    assert call_dict["AAA"] == ("CAPITALSPECIAL", "1998-01-01", SNAPSHOT_DATE_STR, stock_padding_str)
    assert call_dict["$VIX"] == ("CAPITALSPECIAL", "1990-01-02", SNAPSHOT_DATE_STR, "NONE")
    for symbol_str in ("BIL", "SPMO"):
        assert call_dict[symbol_str] == ("CAPITALSPECIAL", "1998-01-01", SNAPSHOT_DATE_STR, "ALLMARKETDAYS")
    assert call_dict["$SPXTR"][0] == "TOTALRETURN"
    assert call_dict["$SPXTR"][1] == "1998-01-01"


def test_existing_sp500_profile_contracts_are_preserved():
    dv2_spec_obj = export_mod.PROFILE_EXPORT_SPEC_DICT["norgate_eod_sp500_pit"]
    hpi_spec_obj = export_mod.PROFILE_EXPORT_SPEC_DICT[store_mod.HPI_SP500_PROFILE_STR]
    assert dv2_spec_obj.padding_type_name_str == "ALLMARKETDAYS"
    assert dv2_spec_obj.capital_symbol_tuple == dv2_spec_obj.helper_symbol_tuple == ()
    assert dv2_spec_obj.total_return_symbol_tuple == ("$SPX",)
    assert hpi_spec_obj.padding_type_name_str == "NONE"
    assert hpi_spec_obj.capital_symbol_tuple == hpi_spec_obj.helper_symbol_tuple == ()
    assert hpi_spec_obj.trim_past_member_tail_bool is False


@pytest.mark.parametrize("profile_str", PROFILE_TUPLE)
def test_capsule_rejects_missing_native_dtype_metadata(tmp_path, profile_str):
    _write_snapshot(tmp_path, profile_str, contract_change_dict={"source_dtype_by_pair_dict": {}})
    with pytest.raises(store_mod.NorgateSnapshotValidationError, match="dtype provenance"):
        store_mod.load_valid_snapshot_manifest(profile_str, snapshot_root_str=str(tmp_path))


@pytest.mark.parametrize("dtype_str", ["float32", "int64"])
def test_capsule_rejects_lossy_or_invalid_native_dtype_restoration(dtype_str):
    contract_dict = {"source_field_by_pair_dict": {"AAA|CAPITALSPECIAL": ["Close"]},
        "source_dtype_by_pair_dict": {"AAA|CAPITALSPECIAL": {"Close": dtype_str}}}
    with pytest.raises(store_mod.NorgateSnapshotValidationError, match="invalid or lossy"):
        store_mod._restore_mr_capsule_native_fields(pd.DataFrame({"Close": [100.1]}), contract_dict, "AAA|CAPITALSPECIAL")


def test_capsule_rejects_nonnumeric_declared_native_dtype(tmp_path):
    price_df = _price_frame_df()
    price_df["Open"] = price_df["Open"].astype(object)
    _write_snapshot(tmp_path, PROFILE_TUPLE[0], price_df=price_df)
    with pytest.raises(store_mod.NorgateSnapshotValidationError, match="native dtype is invalid"):
        store_mod.load_valid_snapshot_manifest(PROFILE_TUPLE[0], snapshot_root_str=str(tmp_path))


def test_capsule_integer_precision_cannot_be_lost_during_float_restore():
    contract_dict = {"source_field_by_pair_dict": {"AAA|CAPITALSPECIAL": ["Volume"]},
        "source_dtype_by_pair_dict": {"AAA|CAPITALSPECIAL": {"Volume": "float64"}}}
    with pytest.raises(store_mod.NorgateSnapshotValidationError, match="invalid or lossy"):
        store_mod._restore_mr_capsule_native_fields(pd.DataFrame({"Volume": [2**53 + 1]}), contract_dict, "AAA|CAPITALSPECIAL")


@pytest.mark.parametrize("profile_str", PROFILE_TUPLE)
def test_capsule_native_dtypes_preserve_actual_dv2_threshold_after_mixed_export(tmp_path, monkeypatch, profile_str):
    from strategies.dv2.strategy_mr_dv2 import DVO2Strategy

    source_df = _price_frame_df()
    history_index = pd.DatetimeIndex(sorted(source_df["date"].unique()))[-300:]
    native_df = pd.DataFrame({field_str: np.full(300, value_float, dtype=np.float32)
        for field_str, value_float in {"Open": 100., "High": 102., "Low": 98., "Close": 100.}.items()}, index=history_index)
    native_df["Volume"], native_df["Dividend"] = 1000., 0.
    native_df.loc[history_index[-1], ["Close", "High", "Low"]] = [105., 108., 104.]
    long_stock_df = native_df.rename_axis("date").reset_index().assign(symbol_str="AAA", adjustment_str="CAPITALSPECIAL")
    mixed_df = pd.concat([source_df.loc[~source_df["symbol_str"].eq("AAA")], long_stock_df], ignore_index=True)
    field_list = ["Open", "High", "Low", "Close", "Volume", "Dividend"]
    native_dtype_dict = {f"{symbol_str}|{adjustment_str}": {field_str: str(mixed_df[field_str].dtype) for field_str in field_list}
        for symbol_str, adjustment_str in mixed_df[["symbol_str", "adjustment_str"]].drop_duplicates().itertuples(index=False, name=None)}
    native_dtype_dict["AAA|CAPITALSPECIAL"] = {field_str: str(native_df[field_str].dtype) for field_str in field_list}
    _write_snapshot(tmp_path, profile_str, price_df=mixed_df, contract_change_dict={"source_dtype_by_pair_dict": native_dtype_dict})
    manifest_obj = store_mod.load_valid_snapshot_manifest(profile_str, snapshot_root_str=str(tmp_path))
    monkeypatch.setattr(store_mod, "load_valid_snapshot_manifest", lambda *_: manifest_obj)
    restored_df = store_mod.load_raw_prices_df(["AAA"], [], start_date_str=str(history_index[0].date()),
        end_date_str=SNAPSHOT_DATE_STR, data_profile_str=profile_str)
    assert restored_df[("AAA", "Close")].dtype == np.dtype("float32")
    assert store_mod.load_price_timeseries_df("AAA", data_profile_str=profile_str)["Close"].dtype == np.dtype("float32")
    direct_df = native_df.copy()
    direct_df.columns = pd.MultiIndex.from_tuples([("AAA", field_str) for field_str in field_list])
    opportunity_dict, return_dict = {}, {}
    for mode_str, pricing_df in (("direct", direct_df), ("promoted", direct_df.astype("float64")), ("restored", restored_df)):
        strategy_obj = DVO2Strategy(name=mode_str, benchmarks=[], capital_base=100000.)
        strategy_obj.universe_df = pd.DataFrame({"AAA": 1}, index=history_index)
        strategy_obj.previous_bar = history_index[-1]
        signal_df = strategy_obj.compute_signals(pricing_df)
        return_dict[mode_str] = float(signal_df.iloc[-1][("AAA", "p126d_return")])
        opportunity_dict[mode_str] = strategy_obj.get_opportunities(signal_df.iloc[-1])
    assert return_dict["direct"] < .05 < return_dict["promoted"]
    assert return_dict["restored"] == return_dict["direct"]
    assert opportunity_dict == {"direct": [], "promoted": ["AAA"], "restored": []}


def _native_frame(date_str="1998-02-05"):
    return pd.DataFrame({"Open": [100.], "High": [101.], "Low": [99.], "Close": [100.],
        "Volume": [1000.], "Dividend": [.018267]}, index=pd.DatetimeIndex([date_str], name="Date")).astype("float32")


def _native_stub(monkeypatch, response_list):
    call_list = []

    def native_prices(symbol_str, **request_dict):
        call_list.append(request_dict)
        return response_list.pop(0)

    provider_obj = SimpleNamespace(price_timeseries=native_prices,
        PaddingType=SimpleNamespace(ALLMARKETDAYS="ALLMARKETDAYS", NONE="NONE"),
        StockPriceAdjustmentType=SimpleNamespace(CAPITALSPECIAL="CAPITALSPECIAL", TOTALRETURN="TOTALRETURN"))
    monkeypatch.setattr(export_mod, "_load_direct_norgate_module", lambda: provider_obj)
    return call_list


def test_nonempty_native_request_preserves_dividends_and_never_uses_fallback(monkeypatch):
    native_df = _native_frame()
    call_list = _native_stub(monkeypatch, [native_df.copy()])
    exported_df = export_mod._load_price_frame_df(symbol_str="VLO", adjustment_str="CAPITALSPECIAL",
        start_date_str="1998-01-01", end_date_str=SNAPSHOT_DATE_STR, empty_history_fallback_start_date_str="1990-01-01")
    assert [request_dict["start_date"] for request_dict in call_list] == ["1998-01-01"]
    np.testing.assert_array_equal(exported_df["Dividend"].to_numpy(), native_df["Dividend"].to_numpy())
    assert exported_df.attrs["source_request_dict"]["empty_primary_request_bool"] is False


def test_empty_native_primary_request_preserves_recorded_pre_window_history(monkeypatch):
    call_list = _native_stub(monkeypatch, [pd.DataFrame(), _native_frame("1997-02-05")])
    exported_df = export_mod._load_price_frame_df(symbol_str="OLD", adjustment_str="CAPITALSPECIAL",
        start_date_str="1998-01-01", end_date_str=SNAPSHOT_DATE_STR, empty_history_fallback_start_date_str="1990-01-01")
    assert [request_dict["start_date"] for request_dict in call_list] == ["1998-01-01", "1990-01-01"]
    assert exported_df["date"].lt(pd.Timestamp("1998-01-01")).all()
    assert exported_df.attrs["source_request_dict"] == {"requested_start_date_str": "1998-01-01",
        "effective_start_date_str": "1990-01-01", "end_date_str": SNAPSHOT_DATE_STR,
        "padding_type_name_str": "ALLMARKETDAYS", "empty_primary_request_bool": True}


@pytest.mark.parametrize("fallback_kind_str", ["overlap", "empty", "nat"])
def test_fallback_cannot_replace_primary_history_or_hide_empty_or_invalid_dates(monkeypatch, fallback_kind_str):
    fallback_df = pd.DataFrame() if fallback_kind_str == "empty" else _native_frame(None if fallback_kind_str == "nat" else "1998-02-05")
    _native_stub(monkeypatch, [pd.DataFrame(), fallback_df])
    with pytest.raises(RuntimeError, match="fallback|no price data"):
        export_mod._load_price_frame_df(symbol_str="OLD", adjustment_str="CAPITALSPECIAL",
            start_date_str="1998-01-01", end_date_str=SNAPSHOT_DATE_STR, empty_history_fallback_start_date_str="1990-01-01")


@pytest.mark.parametrize("fallback_kind_str", ["pre_window", "overlap", "nat"])
def test_snapshot_enforces_declared_empty_request_fallback_scope(tmp_path, monkeypatch, fallback_kind_str):
    price_df = _price_frame_df()
    stock_mask_ser = price_df["symbol_str"].eq("AAA")
    if fallback_kind_str != "overlap":
        price_df.loc[stock_mask_ser, "date"] = pd.bdate_range(end="1997-12-31", periods=int(stock_mask_ser.sum()))
    if fallback_kind_str == "nat":
        price_df.loc[price_df.index[stock_mask_ser][0], "date"] = pd.NaT
    snapshot_path_obj = _write_snapshot(tmp_path, PROFILE_TUPLE[0], price_df=price_df)
    manifest_path_obj = snapshot_path_obj / "manifest.json"
    manifest_dict = json.loads(manifest_path_obj.read_text())
    manifest_dict["data_contract"]["source_request_by_pair_dict"]["AAA|CAPITALSPECIAL"].update(
        effective_start_date_str="1990-01-01", empty_primary_request_bool=True)
    manifest_path_obj.write_text(json.dumps(manifest_dict))
    if fallback_kind_str != "pre_window":
        with pytest.raises(store_mod.NorgateSnapshotValidationError, match="pre-window fallback"):
            store_mod.load_valid_snapshot_manifest(PROFILE_TUPLE[0], snapshot_root_str=str(tmp_path))
        return
    manifest_obj = store_mod.load_valid_snapshot_manifest(PROFILE_TUPLE[0], snapshot_root_str=str(tmp_path))
    monkeypatch.setattr(store_mod, "load_valid_snapshot_manifest", lambda *_: manifest_obj)
    loaded_df = store_mod.load_raw_prices_df(["AAA", "BIL"], [], start_date_str="1998-01-01", data_profile_str=PROFILE_TUPLE[0])
    assert set(loaded_df.columns.get_level_values(0)) == {"BIL"}
