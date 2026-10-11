"""E2 comparison records the current sector reconstruction and sizing bridge."""
from datetime import datetime
from functools import wraps
import importlib
import json
from pathlib import Path

import exchange_calendars
import pandas as pd
import pytest

from alpha.live import reference_compare
from alpha.live.ndx_e2_adapter import NDX_E2_STRATEGY_IMPORT_TUPLE
from alpha.live.release_manifest import parse_release_manifest
from alpha.live.runner import get_compare_reference_summary
from alpha.live.state_store_v2 import LiveStateStore
from data import norgate_snapshot_store as snapshot_module
from test_live_reference_compare import FakeReferenceStrategy, MARKET_TZ, _build_release, _insert_vplan
from test_ndx_e2_norgate_snapshot import FakeE2Provider, e2_snapshot, rewrite_manifest


TEMPLATE_NAME_TUPLE = (
    "pod_ndx_atr_normalized_vxn_scaled_sector_cap_monthly_open.yaml.example",
    "pod_ndx_natr20_vxn_scaled_sector_cap_monthly_open.yaml.example",
)


@pytest.fixture
def e2_reference_snapshot(monkeypatch, request):
    original_init_fn = FakeE2Provider.__init__

    def extended_history_init(provider_obj):
        original_init_fn(provider_obj)
        provider_obj.date_idx = exchange_calendars.get_calendar("XNYS").sessions_in_range(
            "2022-01-03", "2024-01-31"
        ).tz_localize(None).rename("Date")

    monkeypatch.setattr(FakeE2Provider, "__init__", extended_history_init)
    return request.getfixturevalue("e2_snapshot")


@pytest.fixture(params=range(2))
def e2_release_obj(request):
    return parse_release_manifest(str(Path("docs/live/release_templates") / TEMPLATE_NAME_TUPLE[request.param]))


def _run_reference(release_obj, tmp_path, end_date_str="2024-01-31"):
    return reference_compare.run_auto_reference_strategy(
        release_obj=release_obj, deployment_start_date_str="2024-01-02",
        reference_end_date_str=end_date_str, deployment_initial_cash_float=100000.0,
        output_dir_path_obj=tmp_path / "reference",
    )


def _write_artifacts(release_obj, reference_maps_dict, tmp_path):
    return reference_compare.write_reference_compare_artifacts(
        output_dir_path_obj=tmp_path / "report", release_obj=release_obj,
        compare_report_dict={"target_session_date_str": "2024-01-31"},
        reference_maps_dict=reference_maps_dict, live_history_row_dict_list=[],
        reference_strategy_pickle_path_str=None,
    )


def test_actual_e2_reference_preserves_validated_provenance_in_pickle_and_report(
    e2_reference_snapshot, e2_release_obj, tmp_path,
):
    provider_obj, _ = e2_reference_snapshot
    provider_obj.failure_bool = True
    strategy_obj = _run_reference(e2_release_obj, tmp_path)
    manifest_obj = snapshot_module.load_valid_snapshot_manifest(snapshot_module.NDX_E2_PROFILE_STR)
    provenance_dict = strategy_obj.reference_provenance_dict
    assert provenance_dict["scope_str"] == "diagnostic_current_sector_map_reconstruction"
    assert provenance_dict["provenance_available_bool"] is True
    assert provenance_dict["data_profile_str"] == snapshot_module.NDX_E2_PROFILE_STR
    assert provenance_dict["snapshot_date_str"] == "2024-01-31"
    assert provenance_dict["snapshot_manifest_hash_str"] == manifest_obj.manifest_hash_str
    assert provenance_dict["sector_map_hash_str"] == manifest_obj.manifest_dict["data_contract"]["sector_map_hash_str"]
    assert provenance_dict["sector_by_symbol_dict"] == manifest_obj.manifest_dict["data_contract"]["sector_by_symbol_dict"]
    assert provenance_dict["sector_by_symbol_dict"]["OLD"] == "UNKNOWN"
    assert strategy_obj.sector_by_symbol_map == {
        symbol_str: provenance_dict["sector_by_symbol_dict"][symbol_str]
        for symbol_str in strategy_obj.universe_df.columns
    }
    assert "OLD" not in strategy_obj.sector_by_symbol_map
    assert provenance_dict["research_sizing_basis_str"] == "Close_T_NAV_and_prices"
    assert provenance_dict["live_sizing_contract_str"] == "legacy_monthly_preopen_v1"
    assert provenance_dict["exact_share_parity_bool"] is False
    assert provenance_dict["exact_pnl_parity_bool"] is False
    assert provenance_dict["historical_pit_gics_bool"] is False
    pickle_path_obj = tmp_path / "reference.pkl"
    strategy_obj.to_pickle(pickle_path_obj)
    reference_maps_dict = reference_compare.load_reference_maps_from_pickle(str(pickle_path_obj))
    assert reference_maps_dict["reference_provenance_dict"] == provenance_dict
    assert reference_maps_dict["reference_accounting_contract_version_str"] == "price_return_ledger_v1"
    assert reference_maps_dict["reference_dividend_event_count_int"] == 0
    artifact_dict = _write_artifacts(e2_release_obj, reference_maps_dict, tmp_path)
    summary_dict = json.loads(Path(artifact_dict["summary_json_path_str"]).read_text(encoding="utf-8"))
    assert summary_dict["reference_provenance_dict"] == provenance_dict
    html_str = Path(artifact_dict["html_path_str"]).read_text(encoding="utf-8")
    assert "E2 diagnostic reference" in html_str
    assert "Close_T" in html_str and "pre-open" in html_str
    assert "not exact share or P&amp;L parity" in html_str
    assert "not historical point-in-time GICS" in html_str


def test_manual_e2_pickle_without_metadata_is_explicitly_unavailable(
    e2_reference_snapshot, e2_release_obj, tmp_path,
):
    strategy_obj = _run_reference(e2_release_obj, tmp_path)
    del strategy_obj.reference_provenance_dict
    pickle_path_obj = tmp_path / "manual.pkl"
    strategy_obj.to_pickle(pickle_path_obj)
    reference_maps_dict = reference_compare.load_reference_maps_from_pickle(str(pickle_path_obj))
    provenance_dict = reference_maps_dict["reference_provenance_dict"]
    assert provenance_dict["provenance_available_bool"] is False
    assert provenance_dict["scope_str"] == "diagnostic_current_sector_map_reconstruction"
    assert provenance_dict["snapshot_manifest_hash_str"] is None
    assert provenance_dict["sector_by_symbol_dict"] is None
    assert provenance_dict["exact_share_parity_bool"] is False
    artifact_dict = _write_artifacts(e2_release_obj, reference_maps_dict, tmp_path)
    html_str = Path(artifact_dict["html_path_str"]).read_text(encoding="utf-8")
    assert "Snapshot provenance unavailable" in html_str


def test_e2_reference_refuses_direct_provider_fallback(e2_reference_snapshot, e2_release_obj, tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    with pytest.raises(RuntimeError, match="snapshot mode"):
        _run_reference(e2_release_obj, tmp_path)


def test_e2_reference_refuses_stale_snapshot(e2_reference_snapshot, e2_release_obj, tmp_path):
    with pytest.raises(snapshot_module.NorgateSnapshotError):
        _run_reference(e2_release_obj, tmp_path, end_date_str="2024-02-01")


def test_e2_reference_refuses_invalid_sector_provenance(e2_reference_snapshot, e2_release_obj, tmp_path):
    _, snapshot_path_obj = e2_reference_snapshot
    rewrite_manifest(snapshot_path_obj, lambda manifest_dict: manifest_dict["data_contract"].pop("sector_map_hash_str"))
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError):
        _run_reference(e2_release_obj, tmp_path)


@pytest.mark.parametrize("mutation_str", ["manifest", "price_bytes", "used_sector_map"])
def test_e2_reference_refuses_snapshot_change_during_run(
    e2_reference_snapshot, e2_release_obj, tmp_path, monkeypatch, mutation_str,
):
    _, snapshot_path_obj = e2_reference_snapshot
    strategy_module_obj = importlib.import_module(e2_release_obj.strategy_import_str.split(":")[0])
    original_run_fn = strategy_module_obj.run_variant

    @wraps(original_run_fn)
    def changed_run_fn(**kwarg_dict):
        strategy_obj = original_run_fn(**kwarg_dict)
        if mutation_str == "manifest":
            rewrite_manifest(snapshot_path_obj, lambda manifest_dict: manifest_dict.update(test_revision_str="changed"))
        elif mutation_str == "price_bytes":
            price_path_obj = snapshot_path_obj / "prices.parquet"
            price_df = pd.read_parquet(price_path_obj)
            price_df["Close"] += 1.0
            price_df.to_parquet(price_path_obj, index=False)
        else:
            strategy_obj.sector_by_symbol_map["AAA"] = strategy_obj.sector_by_symbol_map["BBB"]
        return strategy_obj

    monkeypatch.setattr(strategy_module_obj, "run_variant", changed_run_fn)
    with pytest.raises(RuntimeError, match="changed|hash mismatch|sector map|SHA256 mismatch"):
        _run_reference(e2_release_obj, tmp_path)


def test_reference_json_includes_e2_unavailable_diagnostic_only(tmp_path, monkeypatch):
    for strategy_import_str in (NDX_E2_STRATEGY_IMPORT_TUPLE[0], _build_release().strategy_import_str):
        run_path_obj = tmp_path / strategy_import_str.rsplit(":", 1)[1]
        run_path_obj.mkdir()
        release_obj = _build_release(strategy_import_str=strategy_import_str)
        state_store_obj = LiveStateStore(str(run_path_obj / "live.sqlite3"))
        state_store_obj.upsert_release(release_obj)
        releases_path_obj = run_path_obj / "releases"
        releases_path_obj.mkdir()
        _insert_vplan(state_store_obj, release_obj,
            datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TZ),
            datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TZ), 5000.0)
        monkeypatch.setattr(reference_compare, "run_auto_reference_strategy", lambda **_: FakeReferenceStrategy())
        report_dict = get_compare_reference_summary(state_store_obj=state_store_obj,
            as_of_ts=datetime(2024, 2, 1, 17, 0, tzinfo=MARKET_TZ),
            releases_root_path_str=str(releases_path_obj), env_mode_str="paper",
            pod_id_str=release_obj.pod_id_str, output_dir_str=str(run_path_obj / "results"),
            html_output_bool=False)["compare_report_dict_list"][0]
        if strategy_import_str in NDX_E2_STRATEGY_IMPORT_TUPLE:
            assert report_dict["reference_provenance_dict"]["provenance_available_bool"] is False
        else:
            assert "reference_provenance_dict" not in report_dict


def test_non_e2_maps_and_report_have_no_e2_provenance(tmp_path):
    reference_maps_dict = reference_compare.build_reference_maps_dict(FakeReferenceStrategy())
    assert "reference_provenance_dict" not in reference_maps_dict
    artifact_dict = _write_artifacts(_build_release(), reference_maps_dict, tmp_path)
    summary_dict = json.loads(Path(artifact_dict["summary_json_path_str"]).read_text(encoding="utf-8"))
    assert "reference_provenance_dict" not in summary_dict
    assert "E2 diagnostic reference" not in Path(artifact_dict["html_path_str"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("stored_provenance_obj", [None, {}, {
    "provenance_available_bool": True, "scope_str": "exact",
    "exact_share_parity_bool": True, "exact_pnl_parity_bool": True,
}])
def test_partial_manual_provenance_cannot_claim_exact_parity(stored_provenance_obj):
    provenance_dict = reference_compare.get_ndx_e2_reference_provenance_dict({
        "reference_provenance_dict": stored_provenance_obj,
    })
    assert provenance_dict["provenance_available_bool"] is False
    assert provenance_dict["scope_str"] == "diagnostic_current_sector_map_reconstruction"
    assert provenance_dict["exact_share_parity_bool"] is False
    assert provenance_dict["exact_pnl_parity_bool"] is False
    assert provenance_dict["historical_pit_gics_bool"] is False
