"""E2 transports current GICS with its dated prices; no client provider fallback."""
import json
from types import SimpleNamespace

import numpy as np
import exchange_calendars as calendar_module
import pandas as pd
import pytest

from data import norgate_snapshot_store as snapshot_module
from scripts import export_norgate_snapshot as export_module
from strategies.momentum import strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap as sector_module


class FakeE2Provider:
    StockPriceAdjustmentType = SimpleNamespace(CAPITALSPECIAL="CAPITALSPECIAL", TOTALRETURN="TOTALRETURN")
    PaddingType = SimpleNamespace(ALLMARKETDAYS="ALLMARKETDAYS", NONE="NONE")

    def __init__(self):
        self.date_idx = calendar_module.get_calendar("XNYS", start="2021-12-01", end="2024-02-01").sessions_in_range("2022-01-01", "2024-01-31").tz_localize(None).rename("Date")
        self.sector_dict = {"AAA": "45000000", "BBB": "35000000", "OLD": None}
        self.failure_bool = False
        self.call_list = []

    def classification_at_level(self, symbol_str, scheme_str, field_str, level_int):
        assert (scheme_str, field_str, level_int) == ("GICS", "ClassificationId", 1)
        self.call_list.append(symbol_str)
        if self.failure_bool:
            raise ConnectionError("Classification service unavailable")
        return self.sector_dict[symbol_str]

    def price_timeseries(self, symbol_str, **option_dict):
        price_float = 22.0 if symbol_str == "$VXN" else 100.0
        price_df = pd.DataFrame({"Open": price_float, "High": price_float + 1,
            "Low": price_float - 1, "Close": price_float, "Volume": 1000000.0,
            "Unadjusted Close": price_float * 2, "Dividend": 0.0}, index=self.date_idx)
        if symbol_str == "$VXN":
            price_df = price_df.drop(columns=["Unadjusted Close", "Dividend"])
        return price_df.loc[option_dict["start_date"]:option_dict["end_date"]]


@pytest.fixture
def e2_snapshot(tmp_path, monkeypatch):
    provider_obj = FakeE2Provider()
    universe_df = pd.DataFrame({"AAA": 1, "BBB": 1, "OLD": 0}, index=provider_obj.date_idx)
    universe_df.loc[universe_df.index[:20], "OLD"] = 1
    universe_df["PRE1999"] = 0
    universe_df["FUTURE"] = 0
    universe_df.loc[pd.Timestamp("1995-01-03")] = [0, 0, 0, 1, 0]
    universe_df.loc[pd.Timestamp("2024-02-01")] = [0, 0, 0, 0, 1]
    universe_df = universe_df.sort_index()
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: provider_obj)
    monkeypatch.setattr(export_module, "_load_index_constituent_matrix_df",
        lambda *args, **kwargs: (list(universe_df.columns), universe_df))
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    snapshot_path_obj = export_module.export_profile_snapshot(snapshot_root_str=str(tmp_path),
        profile_str=snapshot_module.NDX_E2_PROFILE_STR, snapshot_date_str="2024-01-31")
    return provider_obj, snapshot_path_obj


def rewrite_manifest(snapshot_path_obj, mutate_fn):
    manifest_path_obj = snapshot_path_obj / "manifest.json"
    manifest_dict = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
    mutate_fn(manifest_dict)
    manifest_path_obj.write_text(json.dumps(manifest_dict), encoding="utf-8")


def rewrite_prices(snapshot_path_obj, price_df):
    price_path_obj = snapshot_path_obj / "prices.parquet"
    price_df.to_parquet(price_path_obj, index=False)
    rewrite_manifest(snapshot_path_obj, lambda manifest_dict: manifest_dict["files"]["prices.parquet"].update(
        sha256=snapshot_module._hash_file_path(price_path_obj)))


def test_export_and_client_share_complete_frozen_map_and_nominal_prices(e2_snapshot, monkeypatch):
    provider_obj, snapshot_path_obj = e2_snapshot
    assert provider_obj.call_list == ["AAA", "BBB", "OLD"]
    provider_obj.failure_bool = True
    monkeypatch.setattr(sector_module, "build_current_gics_sector_map", lambda *_: pytest.fail("Client called Norgate"))
    with snapshot_module.use_norgate_data_profile(snapshot_module.NDX_E2_PROFILE_STR):
        sector_dict = sector_module.sector_map_for_universe(pd.DataFrame(columns=["AAA", "BBB", "OLD"]))
        price_df = snapshot_module.load_raw_prices_df(["AAA"], [], start_date_str="2024-01-02", end_date_str="2024-01-31")
    assert sector_dict == {"AAA": "45000000", "BBB": "35000000", "OLD": "UNKNOWN"}
    assert price_df[("AAA", "Unadjusted Close")].eq(200.0).all()
    assert price_df[("AAA", "Close")].eq(100.0).all()
    assert price_df.index.min() == pd.Timestamp("2024-01-02")
    manifest_obj = snapshot_module.load_valid_snapshot_manifest(snapshot_module.NDX_E2_PROFILE_STR)
    assert manifest_obj.manifest_dict["data_contract"]["sector_map_hash_str"] == snapshot_module.ndx_e2_sector_map_hash_str(sector_dict)
    assert manifest_obj.snapshot_dir_path_obj == snapshot_path_obj


@pytest.mark.parametrize("mutation_str", ["missing_map", "missing_symbol", "wrong_hash", "wrong_date", "bad_sector", "missing_time", "wrong_basis", "legacy_schema", "missing_helper"])
def test_sector_contract_refuses_incomplete_or_wrong_provenance_after_prior_read(e2_snapshot, mutation_str):
    _, snapshot_path_obj = e2_snapshot
    snapshot_module.load_valid_snapshot_manifest(snapshot_module.NDX_E2_PROFILE_STR)
    def mutate(manifest_dict):
        contract_dict = manifest_dict["data_contract"]
        if mutation_str == "missing_map":
            contract_dict.pop("sector_by_symbol_dict")
        elif mutation_str == "missing_symbol":
            contract_dict["sector_by_symbol_dict"].pop("OLD")
            contract_dict["sector_map_hash_str"] = snapshot_module.ndx_e2_sector_map_hash_str(contract_dict["sector_by_symbol_dict"])
        elif mutation_str == "wrong_hash":
            contract_dict["sector_map_hash_str"] = "0" * 64
        elif mutation_str == "wrong_date":
            contract_dict["sector_snapshot_date_str"] = "2024-01-30"
        elif mutation_str == "bad_sector":
            contract_dict["sector_by_symbol_dict"]["AAA"] = "4510"
        elif mutation_str == "missing_time":
            contract_dict.pop("sector_observed_timestamp_str")
        elif mutation_str == "wrong_basis":
            contract_dict["sector_classification_basis_str"] = "historical"
        elif mutation_str == "legacy_schema":
            manifest_dict["schema_version"] = 1
        else:
            manifest_dict["required_helpers"] = []
    rewrite_manifest(snapshot_path_obj, mutate)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError):
        snapshot_module.load_ndx_e2_sector_map(["AAA"])


@pytest.mark.parametrize("mutation_str", ["missing_anchor", "zero_anchor", "nan_anchor", "missing_member", "future_price", "duplicate_price", "zero_vxn"])
def test_e2_prices_fail_closed_without_inventing_nominal_anchors(e2_snapshot, mutation_str):
    _, snapshot_path_obj = e2_snapshot
    price_df = pd.read_parquet(snapshot_path_obj / "prices.parquet")
    row_mask_ser = (price_df["symbol_str"] == "AAA") & (price_df["date"] == pd.Timestamp("2024-01-31"))
    if mutation_str == "missing_anchor":
        price_df = price_df.drop(columns=["Unadjusted Close"])
    elif mutation_str in {"zero_anchor", "nan_anchor"}:
        price_df.loc[row_mask_ser, "Unadjusted Close"] = 0.0 if mutation_str == "zero_anchor" else np.nan
    elif mutation_str == "missing_member":
        price_df = price_df.loc[~row_mask_ser]
    elif mutation_str == "future_price":
        price_df.loc[row_mask_ser, "date"] = pd.Timestamp("2024-02-01")
    elif mutation_str == "duplicate_price":
        price_df = pd.concat([price_df, price_df.loc[row_mask_ser]])
    else:
        price_df.loc[price_df["symbol_str"] == "$VXN", "Close"] = 0.0
    rewrite_prices(snapshot_path_obj, price_df)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError):
        snapshot_module.load_valid_snapshot_manifest(snapshot_module.NDX_E2_PROFILE_STR)


def test_prior_asof_vxn_is_allowed_without_inventing_same_day_value(e2_snapshot):
    _, snapshot_path_obj = e2_snapshot
    price_df = pd.read_parquet(snapshot_path_obj / "prices.parquet")
    price_df = price_df.loc[~((price_df["symbol_str"] == "$VXN") & (price_df["date"] == pd.Timestamp("2024-01-31")))]
    rewrite_prices(snapshot_path_obj, price_df)
    snapshot_module.load_valid_snapshot_manifest(snapshot_module.NDX_E2_PROFILE_STR)


def test_classification_service_failure_is_not_converted_to_unknown(e2_snapshot):
    provider_obj, _ = e2_snapshot
    provider_obj.failure_bool = True
    with pytest.raises(ConnectionError, match="Classification service"):
        export_module._capture_ndx_e2_sector_contract(["AAA"], "2024-01-31")


def test_direct_research_sector_loading_stays_unchanged(monkeypatch):
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    monkeypatch.setattr(sector_module, "build_current_gics_sector_map", lambda symbols: {symbol: "35" for symbol in symbols})
    with snapshot_module.use_norgate_data_profile(None):
        assert sector_module.sector_map_for_universe(pd.DataFrame(columns=["AAA"])) == {"AAA": "35"}
    with snapshot_module.use_norgate_data_profile(snapshot_module.NDX_E2_PROFILE_STR):
        with pytest.raises(RuntimeError, match="snapshot mode"):
            sector_module.sector_map_for_universe(pd.DataFrame(columns=["AAA"]))


@pytest.mark.parametrize("mutation_str", ["spy_nan", "whole_missing_session", "stock_missing_session", "missing_benchmark"])
def test_historical_gaps_cannot_become_false_regime_or_shortened_rolling_windows(e2_snapshot, mutation_str):
    _, snapshot_path_obj = e2_snapshot
    price_df = pd.read_parquet(snapshot_path_obj / "prices.parquet")
    missing_date_ts = pd.Timestamp("2024-01-10")
    if mutation_str == "spy_nan":
        price_df.loc[(price_df["symbol_str"] == "SPY") & (price_df["date"] == missing_date_ts), "Close"] = np.nan
    elif mutation_str == "whole_missing_session":
        price_df = price_df.loc[price_df["date"] != missing_date_ts]
    elif mutation_str == "stock_missing_session":
        price_df = price_df.loc[~((price_df["symbol_str"] == "AAA") & (price_df["date"] == missing_date_ts))]
    else:
        price_df = price_df.loc[price_df["symbol_str"] != "$SPXTR"]
    rewrite_prices(snapshot_path_obj, price_df)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError):
        snapshot_module.load_valid_snapshot_manifest(snapshot_module.NDX_E2_PROFILE_STR)


def test_actual_both_parent_reference_loaders_can_read_total_return_benchmark(e2_snapshot):
    from dataclasses import replace
    from strategies.momentum import strategy_mo_natr20_ndx_vxn_scaled_sector_cap as natr_sector_module
    for strategy_module, loader_fn in (
        (sector_module, sector_module.get_vxn_scaled_atr_normalized_ndx_data),
        (natr_sector_module, natr_sector_module.get_natr20_vxn_scaled_ndx_data),
    ):
        config_obj = replace(strategy_module.DEFAULT_CONFIG, end_date_str="2024-01-31")
        with snapshot_module.use_norgate_data_profile(snapshot_module.NDX_E2_PROFILE_STR):
            pricing_data_df, _, _, _ = loader_fn(config_obj, include_total_return_benchmark_bool=True)
        assert (config_obj.performance_benchmark_data_symbol_str, "Close") in pricing_data_df.columns
        assert pricing_data_df[(config_obj.performance_benchmark_data_symbol_str, "Close")].notna().all()
