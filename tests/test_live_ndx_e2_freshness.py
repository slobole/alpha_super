"""Padded E2 history cannot certify a missing or inconsistent Close_T observation."""
import numpy as np
import pandas as pd
import pytest

from data import norgate_snapshot_store as snapshot_module
from test_ndx_e2_norgate_snapshot import FakeE2Provider, e2_snapshot, rewrite_manifest


@pytest.mark.parametrize("symbol_str", ["SPY", "AAA", "$SPXTR"])
@pytest.mark.parametrize("mutation_str", ["missing", "stale", "nan", "zero", "duplicate", "mismatched_close"])
def test_export_rejects_invalid_unpadded_endpoint_before_publication(
    monkeypatch, request, tmp_path, symbol_str, mutation_str,
):
    original_price_fn = FakeE2Provider.price_timeseries
    observed_request_list = []

    def endpoint_price_fn(provider_obj, requested_symbol_str, **option_dict):
        price_df = original_price_fn(provider_obj, requested_symbol_str, **option_dict)
        if requested_symbol_str == symbol_str and option_dict["padding_setting"] == "NONE":
            observed_request_list.append(option_dict)
            if mutation_str == "missing":
                return price_df.iloc[:0]
            if mutation_str == "stale":
                price_df.index = pd.DatetimeIndex(["2024-01-30"], name="Date")
            elif mutation_str == "nan":
                price_df["Close"] = np.nan
            elif mutation_str == "zero":
                price_df["Close"] = 0.0
            elif mutation_str == "duplicate":
                price_df = pd.concat([price_df, price_df])
            else:
                price_df["Close"] += 1.0
        return price_df

    monkeypatch.setattr(FakeE2Provider, "price_timeseries", endpoint_price_fn)
    with pytest.raises((ValueError, RuntimeError)):
        request.getfixturevalue("e2_snapshot")
    assert len(observed_request_list) == 1
    assert observed_request_list[0]["start_date"] == "2024-01-31"
    assert observed_request_list[0]["end_date"] == "2024-01-31"
    assert observed_request_list[0]["stock_price_adjustment_setting"] == (
        "TOTALRETURN" if symbol_str == "$SPXTR" else "CAPITALSPECIAL"
    )
    assert not (tmp_path / snapshot_module.NDX_E2_PROFILE_STR / "2024-01-31").exists()


@pytest.mark.parametrize("mutation_str", [
    "missing_map", "missing_spy", "missing_member", "missing_benchmark",
    "wrong_date", "extra_retired_member", "not_mapping",
])
def test_client_rejects_missing_or_tampered_endpoint_attestation(e2_snapshot, mutation_str):
    _, snapshot_path_obj = e2_snapshot
    snapshot_module.load_valid_snapshot_manifest(snapshot_module.NDX_E2_PROFILE_STR)

    def mutate_manifest(manifest_dict):
        contract_dict = manifest_dict["data_contract"]
        endpoint_dict = contract_dict["observed_endpoint_date_by_symbol_dict"]
        if mutation_str == "missing_map":
            contract_dict.pop("observed_endpoint_date_by_symbol_dict")
        elif mutation_str == "missing_spy":
            endpoint_dict.pop("SPY")
        elif mutation_str == "missing_member":
            endpoint_dict.pop("AAA")
        elif mutation_str == "missing_benchmark":
            endpoint_dict.pop("$SPXTR")
        elif mutation_str == "wrong_date":
            endpoint_dict["SPY"] = "2024-01-30"
        elif mutation_str == "extra_retired_member":
            endpoint_dict["OLD"] = "2024-01-31"
        else:
            contract_dict["observed_endpoint_date_by_symbol_dict"] = []

    rewrite_manifest(snapshot_path_obj, mutate_manifest)
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="unpadded source"):
        snapshot_module.load_valid_snapshot_manifest(snapshot_module.NDX_E2_PROFILE_STR)


def test_fresh_endpoint_checks_preserve_padded_history_and_prior_asof_vxn(monkeypatch, request):
    original_price_fn = FakeE2Provider.price_timeseries
    request_list = []

    def recorded_price_fn(provider_obj, symbol_str, **option_dict):
        request_list.append((symbol_str, option_dict["padding_setting"]))
        price_df = original_price_fn(provider_obj, symbol_str, **option_dict)
        if symbol_str == "$VXN":
            price_df = price_df.loc[price_df.index < pd.Timestamp("2024-01-31")]
        if symbol_str == "OLD" and option_dict["padding_setting"] == "NONE":
            pytest.fail("A retired constituent must not require a current endpoint.")
        return price_df

    monkeypatch.setattr(FakeE2Provider, "price_timeseries", recorded_price_fn)
    _, snapshot_path_obj = request.getfixturevalue("e2_snapshot")
    manifest_obj = snapshot_module.load_valid_snapshot_manifest(snapshot_module.NDX_E2_PROFILE_STR)
    assert manifest_obj.manifest_dict["data_contract"]["observed_endpoint_date_by_symbol_dict"] == {
        "AAA": "2024-01-31", "BBB": "2024-01-31", "SPY": "2024-01-31", "$SPXTR": "2024-01-31",
    }
    assert {symbol_str for symbol_str, padding_str in request_list if padding_str == "NONE"} == {
        "AAA", "BBB", "SPY", "$SPXTR",
    }
    assert {symbol_str for symbol_str, padding_str in request_list if padding_str == "ALLMARKETDAYS"} == {
        "AAA", "BBB", "OLD", "SPY", "$SPXTR", "$VXN",
    }
    price_df = pd.read_parquet(snapshot_path_obj / "prices.parquet")
    assert price_df.loc[price_df["symbol_str"] == "SPY", "Close"].eq(100.0).all()
    assert pd.Timestamp(price_df.loc[price_df["symbol_str"] == "$VXN", "date"].max()) == pd.Timestamp("2024-01-30")
