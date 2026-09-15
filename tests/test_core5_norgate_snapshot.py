from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha.live import release_manifest, scheduler_utils
from data import norgate_loader, norgate_snapshot_store as snapshot_module
from scripts import export_norgate_snapshot as export_module
from scripts.review import verify_core5_snapshot_parity as qualification_module
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module


class FakeNorgate:
    StockPriceAdjustmentType = SimpleNamespace(CAPITALSPECIAL="CAPITALSPECIAL", TOTALRETURN="TOTALRETURN")
    PaddingType = SimpleNamespace(ALLMARKETDAYS="ALLMARKETDAYS", NONE="NONE")

    def __init__(self):
        self.date_idx = pd.bdate_range("2006-01-03", periods=520)
        self.date_idx.name = "Date"
        self.call_list = []
        self.stale_unpadded_symbol_str = None
        self.invalid_unpadded_symbol_str = None
        self.price_frame_dict = {}
        symbol_tuple = tuple(dict.fromkeys(snapshot_module.CORE5_CAPITAL_SYMBOL_TUPLE + snapshot_module.CORE5_TOTAL_RETURN_SYMBOL_TUPLE))
        for symbol_idx_int, symbol_str in enumerate(symbol_tuple):
            for adjustment_str in ("CAPITALSPECIAL", "TOTALRETURN"):
                bar_vec = np.arange(len(self.date_idx), dtype=float)
                close_vec = 100.0 + symbol_idx_int * 20.0 + bar_vec * 0.03 + 12.0 * np.sin(bar_vec / (13.0 + symbol_idx_int))
                if adjustment_str == "TOTALRETURN":
                    close_vec = close_vec * 1.8
                if symbol_str == "$SPXTR":
                    close_vec = close_vec + 900.0
                price_df = pd.DataFrame({
                    "Open": close_vec * 1.002,
                    "High": close_vec * 1.005,
                    "Low": close_vec * 0.995,
                    "Close": close_vec,
                    "Volume": np.full(len(bar_vec), 1_000_000.0),
                    "Turnover": close_vec * 1_000_000.0,
                }, index=self.date_idx)
                if not symbol_str.startswith("$"):
                    price_df["Unadjusted Close"] = close_vec / 1.8 if adjustment_str == "TOTALRETURN" else close_vec
                    price_df["Dividend"] = np.where(bar_vec.astype(int) % 31 == 0, 0.12, 0.0).astype(np.float32)
                # Actual per-ETF inception differs. Union alignment must preserve
                # unavailable prefixes without inventing historical observations.
                first_bar_int = 35 if symbol_str == "BIL" else (20 if symbol_str == "GLD" else 0)
                self.price_frame_dict[(symbol_str, adjustment_str)] = price_df.iloc[first_bar_int:].copy()

    def price_timeseries(self, symbol_str, **option_dict):
        self.call_list.append((symbol_str, option_dict.copy()))
        adjustment_str = option_dict["stock_price_adjustment_setting"]
        price_df = self.price_frame_dict[(symbol_str, adjustment_str)].copy()
        price_df = price_df.loc[option_dict["start_date"]:option_dict["end_date"]]
        if option_dict["padding_setting"] == "NONE" and symbol_str == self.stale_unpadded_symbol_str:
            return price_df.iloc[:0]
        if option_dict["padding_setting"] == "NONE" and symbol_str == self.invalid_unpadded_symbol_str:
            price_df["Close"] = np.inf
        return price_df


@pytest.fixture
def core5_snapshot(tmp_path, monkeypatch):
    provider_obj = FakeNorgate()
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: provider_obj)
    monkeypatch.setattr(norgate_loader, "_load_direct_norgate_module", lambda: provider_obj)
    snapshot_date_str = provider_obj.date_idx[-1].date().isoformat()
    snapshot_path = export_module.export_profile_snapshot(
        snapshot_root_str=str(tmp_path), profile_str=snapshot_module.CORE5_PROFILE_STR,
        snapshot_date_str=snapshot_date_str,
    )
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    return provider_obj, snapshot_path, snapshot_date_str


def _rewrite_price_file(snapshot_path, price_df):
    price_path = snapshot_path / "prices.parquet"
    price_df.to_parquet(price_path, index=False)
    manifest_path = snapshot_path / "manifest.json"
    manifest_dict = json.loads(manifest_path.read_text())
    manifest_dict["files"]["prices.parquet"]["sha256"] = hashlib.sha256(price_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest_dict))


def test_core5_export_and_direct_snapshot_strategy_parity(core5_snapshot, monkeypatch):
    provider_obj, snapshot_path, snapshot_date_str = core5_snapshot
    config_obj = replace(core5_module.DEFAULT_CONFIG, end_date_str=snapshot_date_str)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    direct_price_df = core5_module.get_adaptive_macro_core5_data(config_obj)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setattr(norgate_loader, "_load_direct_norgate_module", lambda: pytest.fail("Snapshot must not access direct Norgate"))
    with snapshot_module.use_norgate_data_profile("outer_profile"):
        snapshot_price_df = core5_module.get_adaptive_macro_core5_data(config_obj)
        assert snapshot_module.get_active_data_profile_str() == "outer_profile"
    pd.testing.assert_frame_equal(direct_price_df, snapshot_price_df, check_exact=True, check_freq=False)
    assert direct_price_df.attrs == snapshot_price_df.attrs
    pd.testing.assert_series_equal(
        snapshot_price_df[("$SPX", "Close")],
        provider_obj.price_frame_dict[("$SPXTR", "TOTALRETURN")]["Close"],
        check_names=False, check_freq=False,
    )
    assert snapshot_price_df[("BIL", "Close")].iloc[:35].isna().all()
    assert snapshot_price_df[("SPY", "Dividend")].sum() > 0.0
    assert ("$SPX", "Dividend") not in snapshot_price_df.columns
    direct_strategy_obj = core5_module.run_variant(end_date_str=snapshot_date_str, pricing_data_df=direct_price_df, show_display_bool=False, save_results_bool=False)
    snapshot_strategy_obj = core5_module.run_variant(end_date_str=snapshot_date_str, pricing_data_df=snapshot_price_df, show_display_bool=False, save_results_bool=False)
    pd.testing.assert_frame_equal(direct_strategy_obj.compute_signals(direct_price_df), snapshot_strategy_obj.compute_signals(snapshot_price_df), check_exact=True, check_freq=False)
    for field_str in ("daily_target_weights", "rebalance_target_weight_df", "results"):
        pd.testing.assert_frame_equal(getattr(direct_strategy_obj, field_str), getattr(snapshot_strategy_obj, field_str), check_exact=True, check_freq=False)
    qualification_module.compare_core5_transactions(direct_strategy_obj.get_transactions(), snapshot_strategy_obj.get_transactions())
    assert (snapshot_strategy_obj.daily_target_weights["DBC"] < 0.0).any()
    manifest_dict = json.loads((snapshot_path / "manifest.json").read_text())
    assert manifest_dict["data_contract"]["history_start_date_str"] == "1990-01-01"
    assert all(option_dict["padding_setting"] == "ALLMARKETDAYS" for _, option_dict in provider_obj.call_list if option_dict["start_date"] == "1990-01-01")


@pytest.mark.parametrize("mutation_str", [
    "missing_bil", "missing_ief_signal", "missing_benchmark", "stale", "latest_nan",
    "future", "duplicate", "bad_price", "bad_dividend", "truncated_prefix", "interior_gap",
])
def test_core5_rejects_invalid_payload_even_with_updated_hash(core5_snapshot, mutation_str):
    _, snapshot_path, snapshot_date_str = core5_snapshot
    snapshot_module.load_valid_snapshot_manifest(snapshot_module.CORE5_PROFILE_STR)
    price_df = pd.read_parquet(snapshot_path / "prices.parquet")
    signal_mask_ser = (price_df["symbol_str"] == "IEF") & (price_df["adjustment_str"] == "TOTALRETURN")
    latest_mask_ser = signal_mask_ser & price_df["date"].eq(pd.Timestamp(snapshot_date_str))
    if mutation_str == "missing_bil":
        price_df = price_df.loc[price_df["symbol_str"] != "BIL"]
    elif mutation_str == "missing_ief_signal":
        price_df = price_df.loc[~signal_mask_ser]
    elif mutation_str == "missing_benchmark":
        price_df = price_df.loc[price_df["symbol_str"] != "$SPXTR"]
    elif mutation_str == "stale":
        price_df = price_df.loc[~latest_mask_ser]
    elif mutation_str == "latest_nan":
        price_df.loc[latest_mask_ser, "Close"] = np.nan
    elif mutation_str == "future":
        price_df.loc[latest_mask_ser, "date"] += pd.Timedelta(days=1)
    elif mutation_str == "duplicate":
        price_df = pd.concat([price_df, price_df.loc[latest_mask_ser]], ignore_index=True)
    elif mutation_str == "bad_price":
        price_df.loc[latest_mask_ser, "Open"] = np.inf
    elif mutation_str == "bad_dividend":
        price_df.loc[latest_mask_ser, "Dividend"] = np.inf
    elif mutation_str == "truncated_prefix":
        first_signal_date_ts = price_df.loc[signal_mask_ser, "date"].min()
        price_df = price_df.loc[~(signal_mask_ser & price_df["date"].eq(first_signal_date_ts))]
    elif mutation_str == "interior_gap":
        interior_date_ts = price_df.loc[signal_mask_ser, "date"].iloc[100]
        price_df.loc[signal_mask_ser & price_df["date"].eq(interior_date_ts), "Close"] = np.nan
    _rewrite_price_file(snapshot_path, price_df)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError):
        snapshot_module.load_valid_snapshot_manifest(snapshot_module.CORE5_PROFILE_STR)


@pytest.mark.parametrize("mutation_str", ["padding", "history", "schema", "adjustment", "native_dividend", "observation", "lossy_dtype", "nonnumeric_dtype"])
def test_core5_rejects_wrong_manifest_contract(core5_snapshot, mutation_str):
    _, snapshot_path, _ = core5_snapshot
    manifest_path = snapshot_path / "manifest.json"
    manifest_dict = json.loads(manifest_path.read_text())
    contract_dict = manifest_dict["data_contract"]
    if mutation_str == "padding":
        contract_dict["price_padding_setting_str"] = "NONE"
    elif mutation_str == "history":
        contract_dict["history_start_date_str"] = "2020-01-01"
    elif mutation_str == "schema":
        manifest_dict["schema_version"] = 1
    elif mutation_str == "adjustment":
        manifest_dict["adjustment_modes"]["SPY"] = "CAPITALSPECIAL"
    elif mutation_str == "native_dividend":
        contract_dict["source_field_by_pair_dict"]["SPY|CAPITALSPECIAL"].remove("Dividend")
    elif mutation_str == "observation":
        contract_dict["observed_endpoint_date_by_symbol_dict"]["BIL"] = "2006-01-03"
    elif mutation_str == "lossy_dtype":
        contract_dict["source_dtype_by_pair_dict"]["SPY|CAPITALSPECIAL"]["Close"] = "int64"
    elif mutation_str == "nonnumeric_dtype":
        contract_dict["source_dtype_by_pair_dict"]["SPY|CAPITALSPECIAL"]["Close"] = "str"
    manifest_path.write_text(json.dumps(manifest_dict))
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError):
        snapshot_module.load_valid_snapshot_manifest(snapshot_module.CORE5_PROFILE_STR)


def test_core5_corruption_is_rechecked_after_cached_success(core5_snapshot):
    _, snapshot_path, snapshot_date_str = core5_snapshot
    snapshot_module.load_valid_snapshot_manifest(snapshot_module.CORE5_PROFILE_STR)
    with (snapshot_path / "prices.parquet").open("ab") as price_file_obj:
        price_file_obj.write(b"corrupted")
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="SHA256"):
        snapshot_module.load_valid_snapshot_manifest(snapshot_module.CORE5_PROFILE_STR)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="SHA256"):
        export_module.export_profile_snapshot(snapshot_root_str=str(snapshot_path.parent.parent), profile_str=snapshot_module.CORE5_PROFILE_STR, snapshot_date_str=snapshot_date_str)


def test_core5_rejects_stale_snapshot_for_requested_cycle(core5_snapshot):
    _, _, snapshot_date_str = core5_snapshot
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="stale"):
        snapshot_module.load_valid_snapshot_manifest(snapshot_module.CORE5_PROFILE_STR, minimum_snapshot_date_str=str(pd.Timestamp(snapshot_date_str) + pd.Timedelta(days=1)))


@pytest.mark.parametrize("start_date_str,end_date_str", [("2007-01-01", None), ("1990-01-01", "2007-01-01")])
def test_core5_rejects_shortened_export_request(tmp_path, start_date_str, end_date_str):
    with pytest.raises(ValueError, match="history from 1990"):
        export_module.export_profile_snapshot(snapshot_root_str=str(tmp_path), profile_str=snapshot_module.CORE5_PROFILE_STR, snapshot_date_str="2008-01-01", start_date_str=start_date_str, end_date_str=end_date_str)
    assert not list(tmp_path.iterdir())


def test_core5_padded_latest_price_cannot_hide_stale_source(tmp_path, monkeypatch):
    provider_obj = FakeNorgate()
    provider_obj.stale_unpadded_symbol_str = "IEF"
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: provider_obj)
    snapshot_date_str = provider_obj.date_idx[-1].date().isoformat()
    with pytest.raises(RuntimeError, match="no price data for IEF"):
        export_module.export_profile_snapshot(snapshot_root_str=str(tmp_path), profile_str=snapshot_module.CORE5_PROFILE_STR, snapshot_date_str=snapshot_date_str)
    assert not (tmp_path / snapshot_module.CORE5_PROFILE_STR / snapshot_date_str).exists()


def test_core5_profile_context_resets_when_load_fails(monkeypatch):
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    monkeypatch.setattr(core5_module, "load_raw_prices", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("test failure")))
    with snapshot_module.use_norgate_data_profile("outer_profile"):
        with pytest.raises(RuntimeError, match="test failure"):
            core5_module.get_adaptive_macro_core5_data()
        assert snapshot_module.get_active_data_profile_str() == "outer_profile"


def test_core5_profile_registration_preserves_existing_routes():
    assert snapshot_module.CORE5_PROFILE_STR in release_manifest.SUPPORTED_DATA_PROFILE_TUPLE
    assert scheduler_utils.DATA_PROFILE_HEARTBEAT_SYMBOL_MAP[snapshot_module.CORE5_PROFILE_STR] == "$SPX"
    assert core5_module.__name__ in release_manifest.SUPPORTED_STRATEGY_IMPORT_TUPLE
    assert snapshot_module.default_profile_for_symbol_str("IEF") == "norgate_eod_etf_plus_vix_helper"
    old_profile_obj = export_module.PROFILE_EXPORT_SPEC_DICT["norgate_eod_etf_plus_vix_helper"]
    assert old_profile_obj.capital_symbol_tuple == ("GLD", "UUP", "TLT", "DBC", "BTAL", "SPY", "QQQ", "TQQQ")
    assert old_profile_obj.total_return_symbol_tuple == ("GLD", "UUP", "TLT", "DBC", "BTAL", "$SPX", "$SPXTR")


def test_core5_unpadded_endpoint_must_be_finite(tmp_path, monkeypatch):
    provider_obj = FakeNorgate()
    provider_obj.invalid_unpadded_symbol_str = "IEF"
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: provider_obj)
    snapshot_date_str = provider_obj.date_idx[-1].date().isoformat()
    with pytest.raises(ValueError, match="no unpadded source observation"):
        export_module.export_profile_snapshot(snapshot_root_str=str(tmp_path), profile_str=snapshot_module.CORE5_PROFILE_STR, snapshot_date_str=snapshot_date_str)
    assert not (tmp_path / snapshot_module.CORE5_PROFILE_STR / snapshot_date_str).exists()


def test_core5_rejects_snapshot_republication_between_price_reads(core5_snapshot, monkeypatch):
    _, snapshot_path, snapshot_date_str = core5_snapshot
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    original_loader_fn = core5_module.load_raw_prices
    load_count_int = 0

    def load_with_publication(**option_dict):
        nonlocal load_count_int
        price_df = original_loader_fn(**option_dict)
        load_count_int += 1
        if load_count_int == 1:
            # A valid revised history is published after execution prices were
            # read, before the next read selects its TOTALRETURN signal data.
            revised_price_df = pd.read_parquet(snapshot_path / "prices.parquet")
            signal_mask_ser = revised_price_df["symbol_str"].eq("IEF") & revised_price_df["adjustment_str"].eq("TOTALRETURN")
            revised_price_df.loc[signal_mask_ser, "Close"] *= 1.0001
            _rewrite_price_file(snapshot_path, revised_price_df)
        return price_df

    monkeypatch.setattr(core5_module, "load_raw_prices", load_with_publication)
    with snapshot_module.use_norgate_data_profile("outer_profile"):
        with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="changed during"):
            core5_module.get_adaptive_macro_core5_data(replace(core5_module.DEFAULT_CONFIG, end_date_str=snapshot_date_str))
        assert snapshot_module.get_active_data_profile_str() == "outer_profile"
    assert load_count_int == 2


@pytest.mark.parametrize("mutation_str", ["historical_signal", "truncated_prefix"])
def test_qualification_comparator_rejects_changed_history(core5_snapshot, monkeypatch, mutation_str):
    _, _, snapshot_date_str = core5_snapshot
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    direct_price_df = core5_module.get_adaptive_macro_core5_data(replace(core5_module.DEFAULT_CONFIG, end_date_str=snapshot_date_str))
    snapshot_price_df = direct_price_df.copy()
    if mutation_str == "historical_signal":
        snapshot_price_df.loc[snapshot_price_df.index[100], ("ADAPTIVE_TR_IEF", "Close")] *= 1.001
    else:
        snapshot_price_df = snapshot_price_df.iloc[1:]
    with pytest.raises(AssertionError):
        qualification_module.compare_core5_frames(direct_price_df, snapshot_price_df)


def test_transaction_comparator_preserves_trade_fields_and_id_gaps():
    direct_transaction_df = pd.DataFrame({"order_id": [10, 11, 13], "price": [10.25, 11.5, 12.75], "amount": [2, -3, 4]})
    snapshot_transaction_df = direct_transaction_df.copy()
    snapshot_transaction_df["order_id"] += 100
    qualification_module.compare_core5_transactions(direct_transaction_df, snapshot_transaction_df)
    for field_str in ("order_id", "price", "amount"):
        changed_transaction_df = snapshot_transaction_df.copy()
        changed_transaction_df.loc[1, field_str] += 1
        with pytest.raises(AssertionError):
            qualification_module.compare_core5_transactions(direct_transaction_df, changed_transaction_df)


def test_qualification_records_original_failure_when_package_is_missing(tmp_path, monkeypatch):
    def missing_version(package_str):
        raise qualification_module.PackageNotFoundError(package_str)

    def failed_load(config_obj):
        raise RuntimeError("source unavailable")

    monkeypatch.setattr(qualification_module, "version", missing_version)
    monkeypatch.setattr(core5_module, "get_adaptive_macro_core5_data", failed_load)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "original")
    output_path = tmp_path / "failed_qualification"
    with pytest.raises(RuntimeError, match="source unavailable"):
        qualification_module.run_qualification(output_path, "2026-09-11")
    report_dict = json.loads((output_path / "qualification.json").read_text())
    assert report_dict["status_str"] == "failed"
    assert report_dict["error_str"] == "RuntimeError: source unavailable"
    assert report_dict["package_version_dict"]["norgatedata"] == "unavailable"
    assert qualification_module.os.environ["ALPHA_USE_NORGATE_SNAPSHOT_BOOL"] == "original"
