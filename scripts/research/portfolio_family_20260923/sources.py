"""Extract frozen saved research sources; never import or execute a strategy.

This is retrospective artifact extraction. It computes no performance ranking,
fills no missing history, and never changes the original saved artifacts.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd


ROOT_PATH = Path(__file__).resolve().parents[3]
OUTPUT_PATH = ROOT_PATH / "results/research/portfolio_family_20260923"
REGISTRY_PATH = ROOT_PATH / "alpha/strategy_registry.py"
SHARED_PATH_TUPLE = (
    "alpha/engine/backtest.py", "alpha/engine/backtester.py",
    "alpha/engine/strategy.py", "alpha/engine/order.py", "alpha/engine/metrics.py",
    "alpha/engine/portfolio.py", "alpha/engine/portfolio_manager.py",
    "data/norgate_loader.py",
)


class SavedObject:
    """Attribute container replacing local strategy/engine classes in pickles."""


class SavedUnpickler(pickle.Unpickler):
    """Load trusted local artifacts without importing production strategy code."""

    def find_class(self, module_str: str, name_str: str):
        if module_str.startswith(("alpha.", "strategies.")) or module_str == "__main__":
            return SavedObject
        return super().find_class(module_str, name_str)


def sha256_str(file_path: Path) -> str:
    digest_obj = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk_bytes in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest_obj.update(chunk_bytes)
    return digest_obj.hexdigest()


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(file_path: Path, payload_dict: dict) -> None:
    file_path.write_text(
        json.dumps(payload_dict, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        + "\n", encoding="utf-8",
    )


def promoted_source_list() -> list[dict]:
    registry_tree_obj = ast.parse(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_node_obj = next(
        node_obj for node_obj in registry_tree_obj.body
        if isinstance(node_obj, ast.AnnAssign)
        and getattr(node_obj.target, "id", "") == "STRATEGY_TIER_DICT"
    )
    source_list = []
    for key_node_obj, tier_node_obj in zip(
        registry_node_obj.value.keys, registry_node_obj.value.values
    ):
        if tier_node_obj.attr not in {"PM_READY", "WIRED"}:
            continue
        strategy_import_str = ast.literal_eval(key_node_obj)
        module_str = strategy_import_str.split(":", maxsplit=1)[0]
        source_id_str = module_str.rsplit(".", maxsplit=1)[-1]
        source_root_path = ROOT_PATH / "results/research/strategy" / source_id_str
        metadata_path_list = sorted((source_root_path / "vanilla_backtest").glob("*/metadata.json"))
        if not metadata_path_list:
            raise ValueError(f"No saved Vanilla source for {source_id_str}")
        metadata_path = metadata_path_list[-1]
        pickle_path = metadata_path.parent / f"{source_id_str}.pkl"
        current_module_path = ROOT_PATH / (module_str.replace(".", "/") + ".py")
        input_path_dict = {
            "pickle": pickle_path, "metadata": metadata_path,
            "run_info": metadata_path.parent / "run_info.json",
            "current_module": current_module_path,
        }
        source_list.append({
            "source_id_str": source_id_str, "strategy_import_str": strategy_import_str,
            "tier_str": tier_node_obj.attr, "saved_run_str": metadata_path.parent.name,
            "input_file_dict": {
                label_str: {"path_str": str(input_path.resolve()),
                            "sha256_str": sha256_str(input_path)}
                for label_str, input_path in input_path_dict.items()
            },
        })
    if len(source_list) != 25:
        raise ValueError("Expected the frozen 25 promoted strategy sources")
    return source_list


def validate_nav(nav_df: pd.DataFrame, source_id_str: str) -> None:
    if not isinstance(nav_df.index, pd.DatetimeIndex):
        raise ValueError(f"{source_id_str}: date index required")
    if nav_df.empty or not nav_df.index.is_unique or not nav_df.index.is_monotonic_increasing:
        raise ValueError(f"{source_id_str}: invalid date sequence")
    if nav_df.index.tz is not None or not nav_df.index.equals(nav_df.index.normalize()):
        raise ValueError(f"{source_id_str}: date index must be naive midnight dates")
    core_df = nav_df[["total_value", "cash", "portfolio_value"]].astype(float)
    if not np.isfinite(core_df.to_numpy()).all() or core_df["total_value"].le(0).any():
        raise ValueError(f"{source_id_str}: invalid account values")
    # *** CRITICAL *** Accounting check only, using contemporaneous saved EOD
    # facts: NAV_t = cash_t + signed holdings_t. No signals or prices are joined.
    if not np.allclose(
        core_df["total_value"], core_df["cash"] + core_df["portfolio_value"],
        rtol=1e-12, atol=1e-7,
    ):
        raise ValueError(f"{source_id_str}: NAV identity failed")


def validate_calendars(index_by_source_dict: dict[str, pd.DatetimeIndex]) -> dict:
    union_idx = pd.DatetimeIndex([])
    for date_idx in index_by_source_dict.values():
        union_idx = union_idx.union(date_idx)
    for source_id_str, date_idx in index_by_source_dict.items():
        expected_idx = union_idx[(union_idx >= date_idx[0]) & (union_idx <= date_idx[-1])]
        missing_idx = expected_idx.difference(date_idx)
        if len(missing_idx):
            raise ValueError(f"{source_id_str}: internal calendar holes {missing_idx[:5].tolist()}")
    common_start_ts = max(date_idx[0] for date_idx in index_by_source_dict.values())
    common_end_ts = min(date_idx[-1] for date_idx in index_by_source_dict.values())
    common_idx = union_idx[(union_idx >= common_start_ts) & (union_idx <= common_end_ts)]
    if len(common_idx) < 2:
        raise ValueError("No useful shared close window")
    return {
        "common_close_start_date_str": str(common_start_ts.date()),
        "common_close_end_date_str": str(common_end_ts.date()),
        "common_close_row_count_int": len(common_idx),
        "close_to_close_return_count_int": len(common_idx) - 1,
        "calendar_basis_str": "union_of_all_saved_native_calendars_within_each_source_endpoints",
        "exchange_calendar_independently_verified_bool": False,
        "internal_missing_date_count_int": 0,
        "common_index_sha256_str": hashlib.sha256(
            "\n".join(str(date_ts.date()) for date_ts in common_idx).encode()
        ).hexdigest(),
    }


def export_frame(frame_df: pd.DataFrame, file_path: Path, index_bool: bool) -> dict:
    frame_df.to_csv(
        file_path, index=index_bool, index_label="date" if index_bool else None,
        date_format="%Y-%m-%d", float_format="%.17g", na_rep="",
        compression={"method": "gzip", "mtime": 0}, lineterminator="\n",
    )
    return {
        "path_str": str(file_path.resolve()), "sha256_str": sha256_str(file_path),
        "row_count_int": len(frame_df), "column_list": list(map(str, frame_df.columns)),
        "null_cell_count_int": int(frame_df.isna().sum().sum()),
    }


def benchmark_identity_str(benchmark_symbol_str: str | None) -> str:
    if benchmark_symbol_str in {"$SPXTR", "$NDXTR", "$RUITR"}:
        return "total_return_index_symbol"
    if benchmark_symbol_str in {"$SPX", "$NDX", "$RUI"}:
        # Legacy saved identity maps may retain a display alias even when the
        # stored benchmark path is total return. The map alone proves neither.
        return "unverified_identity_map"
    return "unverified_from_saved_metadata"


def validate_cash_interest(
    interest_df: pd.DataFrame, nav_idx: pd.DatetimeIndex, recorded_total_float: float,
) -> None:
    required_column_list = [
        "date", "positive_cash_base_float", "cash_return_float", "cash_interest_float",
    ]
    if any(column_str not in interest_df for column_str in required_column_list):
        raise ValueError("Cash-interest ledger schema is incomplete")
    interest_date_idx = pd.DatetimeIndex(pd.to_datetime(interest_df["date"]))
    if not interest_date_idx.equals(nav_idx):
        raise ValueError("Cash-interest ledger calendar differs from native NAV")
    numeric_interest_df = interest_df[required_column_list[1:]].astype(float)
    if not np.isfinite(numeric_interest_df.to_numpy()).all():
        raise ValueError("Cash-interest ledger has non-finite amounts")
    if interest_df["positive_cash_base_float"].lt(0).any():
        raise ValueError("Positive cash-interest base is negative")
    # *** CRITICAL *** Validate the saved accrual; never re-create its rate
    # timing. interest_t = saved positive_cash_base_t * saved cash_return_t.
    if not np.allclose(
        interest_df["cash_interest_float"],
        interest_df["positive_cash_base_float"] * interest_df["cash_return_float"],
        rtol=1e-12, atol=1e-9,
    ):
        raise ValueError("Cash-interest amount differs from its saved base/rate")
    if not np.isclose(
        interest_df["cash_interest_float"].sum(), recorded_total_float,
        rtol=1e-12, atol=1e-7,
    ):
        raise ValueError("Cash-interest ledger differs from the recorded total")


def extract_source(source_dict: dict) -> tuple[dict, pd.DatetimeIndex]:
    source_id_str = source_dict["source_id_str"]
    for input_dict in source_dict["input_file_dict"].values():
        if sha256_str(Path(input_dict["path_str"])) != input_dict["sha256_str"]:
            raise ValueError(f"{source_id_str}: frozen input changed")
    metadata_dict = json.loads(Path(source_dict["input_file_dict"]["metadata"]["path_str"]).read_text())
    with Path(source_dict["input_file_dict"]["pickle"]["path_str"]).open("rb") as file_obj:
        strategy_obj = SavedUnpickler(file_obj).load()
    benchmark_list = list(getattr(strategy_obj, "_benchmarks", []))
    benchmark_column_list = [str(item_str) for item_str in benchmark_list if item_str in strategy_obj.results]
    nav_df = strategy_obj.results[["total_value", "portfolio_value", "cash"] + benchmark_column_list].copy()
    validate_nav(nav_df, source_id_str)
    if benchmark_column_list and (
        not np.isfinite(nav_df[benchmark_column_list].to_numpy(dtype=float)).all()
        or nav_df[benchmark_column_list].le(0).any().any()
    ):
        raise ValueError(f"{source_id_str}: invalid stored benchmark path")
    source_output_path = OUTPUT_PATH / "data" / source_id_str
    source_output_path.mkdir(parents=True, exist_ok=True)
    output_file_dict = {"nav": export_frame(nav_df, source_output_path / "nav.csv.gz", True)}

    transaction_df = strategy_obj._transactions.copy()
    required_transaction_list = ["bar", "asset", "amount", "price", "total_value", "commission"]
    if any(column_str not in transaction_df for column_str in required_transaction_list):
        raise ValueError(f"{source_id_str}: transaction schema incomplete")
    numeric_transaction_list = ["amount", "price", "total_value", "commission"]
    if not np.isfinite(transaction_df[numeric_transaction_list].to_numpy(dtype=float)).all():
        raise ValueError(f"{source_id_str}: invalid transaction numbers")
    if transaction_df["price"].le(0).any() or transaction_df["commission"].lt(0).any():
        raise ValueError(f"{source_id_str}: invalid transaction prices/fees")
    transaction_date_idx = pd.DatetimeIndex(pd.to_datetime(transaction_df["bar"]))
    if not transaction_date_idx.is_monotonic_increasing or len(transaction_date_idx.difference(nav_df.index)):
        raise ValueError(f"{source_id_str}: transaction dates outside native calendar")
    output_file_dict["transactions"] = export_frame(transaction_df, source_output_path / "transactions.csv.gz", False)

    weight_df = getattr(strategy_obj, "realized_weight_df", pd.DataFrame()).copy()
    weight_available_bool = not weight_df.empty
    if weight_available_bool:
        if not weight_df.index.equals(nav_df.index) or "Cash" not in weight_df:
            raise ValueError(f"{source_id_str}: incomplete holdings snapshot calendar/cash")
        finite_weight_vec = weight_df.to_numpy(dtype=float)
        if np.isinf(finite_weight_vec).any() or weight_df["Cash"].isna().any():
            raise ValueError(f"{source_id_str}: invalid holdings weights")
        # *** CRITICAL *** Missing asset cells are preserved exactly. Native
        # snapshots omit unheld assets; only the row-total audit skips those
        # cells. Cash is retained and must be excluded from later gross exposure.
        if not np.allclose(weight_df.sum(axis=1), 1.0, rtol=1e-10, atol=1e-10):
            raise ValueError(f"{source_id_str}: holdings weights do not sum to one")
        if not np.allclose(weight_df["Cash"], nav_df["cash"] / nav_df["total_value"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"{source_id_str}: holdings cash differs from saved NAV")
        output_file_dict["realized_weights"] = export_frame(weight_df, source_output_path / "realized_weights.csv.gz", True)

    dividend_df = pd.DataFrame(getattr(strategy_obj, "_dividend_ledger_row_dict_list", []))
    borrow_df = getattr(strategy_obj, "borrow_fee_df", pd.DataFrame()).copy()
    interest_df = pd.DataFrame(getattr(strategy_obj, "cash_interest_ledger_row_dict_list", []))
    if not interest_df.empty:
        validate_cash_interest(interest_df, nav_df.index, float(strategy_obj.cash_interest_total_float))
    for ledger_str, ledger_df in [
        ("dividends", dividend_df), ("borrow", borrow_df), ("cash_interest", interest_df),
    ]:
        if not ledger_df.empty:
            numeric_ledger_df = ledger_df.select_dtypes(include=[np.number])
            if not np.isfinite(numeric_ledger_df.to_numpy(dtype=float)).all():
                raise ValueError(f"{source_id_str}: invalid {ledger_str} ledger")
            output_file_dict[ledger_str] = export_frame(ledger_df, source_output_path / f"{ledger_str}.csv.gz", False)

    price_panel_coverage_list = []
    for attribute_str, value_obj in vars(strategy_obj).items():
        if isinstance(value_obj, pd.DataFrame) and isinstance(value_obj.columns, pd.MultiIndex):
            if "Close" in value_obj.columns.get_level_values(-1):
                price_panel_coverage_list.append({
                    "attribute_str": attribute_str, "row_count_int": len(value_obj),
                    "start_str": str(value_obj.index.min()), "end_str": str(value_obj.index.max()),
                    "instrument_list": sorted(set(map(str, value_obj.columns.get_level_values(0)))),
                })
    accounting_dict = metadata_dict.get("accounting_policy", {})
    benchmark_map_dict = metadata_dict.get("benchmark_data_symbol_map", {})
    source_record_dict = {
        "source_id_str": source_id_str, "actual_start_date_str": str(nav_df.index[0].date()),
        "actual_end_date_str": str(nav_df.index[-1].date()), "native_row_count_int": len(nav_df),
        "native_capital_float": float(strategy_obj._capital_base),
        "slippage_per_side_float": float(strategy_obj._slippage),
        "commission_per_share_float": float(strategy_obj._commission_per_share),
        "commission_minimum_float": float(strategy_obj._commission_minimum),
        "accounting_policy_dict": accounting_dict,
        "data_adjustment_policy_dict": metadata_dict.get("data_adjustment_policy", {}),
        "saved_metadata_source_hash_available_bool": any("sha256" in key_str or "hash" in key_str for key_str in metadata_dict),
        "current_module_hash_is_historical_run_proof_bool": False,
        "realized_weights_available_bool": weight_available_bool,
        "cash_weight_column_str": "Cash" if weight_available_bool else None,
        "weight_null_semantics_str": "preserved_native_sparse_unheld_asset_cells_not_missing_return_history",
        "gross_exposure_rule_str": "sum(abs(asset weights)); exclude Cash; short proceeds cash is not exposure",
        "benchmark_column_list": benchmark_column_list,
        "benchmark_data_symbol_map_dict": benchmark_map_dict,
        "benchmark_basis_warning_str": "saved capital-normalized benchmark paths, not prices; adjustment labels and legacy identity maps alone do not establish price versus total-return identity",
        "benchmark_identity_by_column_dict": {
            column_str: benchmark_identity_str(benchmark_map_dict.get(column_str))
            for column_str in benchmark_column_list
        },
        "saved_price_panel_coverage_list": price_panel_coverage_list,
        "price_panel_synthesis_performed_bool": False,
        "output_file_dict": output_file_dict,
    }
    write_json(source_output_path / "source_metadata.json", source_record_dict)
    source_record_dict["source_metadata_sha256_str"] = sha256_str(source_output_path / "source_metadata.json")
    source_index_idx = nav_df.index.copy()
    del strategy_obj
    gc.collect()
    return source_record_dict, source_index_idx


def write_audit_addendum() -> None:
    """Append the reviewed corrections without rewriting any frozen artifact."""
    manifest_path = OUTPUT_PATH / "source_manifest.json"
    spec_path = OUTPUT_PATH / "research_spec_frozen.json"
    addendum_path = OUTPUT_PATH / "source_audit_addendum.json"
    interest_path = OUTPUT_PATH / "data/strategy_taa_tactical_fixed_income_ief_lqd/cash_interest.csv.gz"
    if addendum_path.exists() or interest_path.exists():
        raise FileExistsError("The audit addendum and added ledger must not overwrite existing files")
    manifest_hash_str = sha256_str(manifest_path)
    manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec_dict = json.loads(spec_path.read_text(encoding="utf-8"))
    if manifest_dict["status_str"] != "complete" or manifest_hash_str != spec_dict["data"]["source_manifest_sha256"]:
        raise ValueError("Original source manifest is not the completed frozen input")
    source_by_id_dict = {source_dict["source_id_str"]: source_dict for source_dict in manifest_dict["source_selection_list"]}
    record_by_id_dict = {record_dict["source_id_str"]: record_dict for record_dict in manifest_dict["source_record_list"]}
    metadata_hash_dict = {}
    correction_list = []
    for source_id_str, record_dict in record_by_id_dict.items():
        metadata_path = OUTPUT_PATH / "data" / source_id_str / "source_metadata.json"
        metadata_hash_dict[str(metadata_path)] = sha256_str(metadata_path)
        if metadata_hash_dict[str(metadata_path)] != record_dict["source_metadata_sha256_str"]:
            raise ValueError(f"{source_id_str}: original source metadata changed")
        for column_str, old_identity_str in record_dict["benchmark_identity_by_column_dict"].items():
            new_identity_str = benchmark_identity_str(record_dict["benchmark_data_symbol_map_dict"].get(column_str))
            if new_identity_str != old_identity_str:
                correction_list.append({
                    "source_id_str": source_id_str, "benchmark_column_str": column_str,
                    "previous_identity_str": old_identity_str, "corrected_identity_str": new_identity_str,
                    "reason_str": "Legacy identity map is not evidence of price-only data; retain uncertainty without independently verified identity.",
                    "saved_metadata_sha256_str": metadata_hash_dict[str(metadata_path)],
                    "saved_nav_sha256_str": record_dict["output_file_dict"]["nav"]["sha256_str"],
                })

    tactical_id_str = "strategy_taa_tactical_fixed_income_ief_lqd"
    pickle_dict = source_by_id_dict[tactical_id_str]["input_file_dict"]["pickle"]
    pickle_path = Path(pickle_dict["path_str"])
    if sha256_str(pickle_path) != pickle_dict["sha256_str"]:
        raise ValueError("Frozen Tactical FI pickle changed")
    with pickle_path.open("rb") as file_obj:
        strategy_obj = SavedUnpickler(file_obj).load()
    interest_df = pd.DataFrame(strategy_obj.cash_interest_ledger_row_dict_list)
    recorded_total_float = float(strategy_obj.cash_interest_total_float)
    validate_cash_interest(interest_df, strategy_obj.results.index, recorded_total_float)
    policy_total_float = float(strategy_obj._accounting_policy_dict["cash_interest_total_float"])
    if not np.isclose(recorded_total_float, policy_total_float, rtol=1e-12, atol=1e-7):
        raise ValueError("Cash-interest object and accounting totals differ")

    benchmark_evidence_list = []
    reference_id_str = "strategy_mr_hpi_sp500_2_3_5_vote"
    reference_nav_dict = record_by_id_dict[reference_id_str]["output_file_dict"]["nav"]
    reference_nav_path = Path(reference_nav_dict["path_str"])
    if sha256_str(reference_nav_path) != reference_nav_dict["sha256_str"]:
        raise ValueError("Frozen benchmark reference export changed")
    reference_df = pd.read_csv(reference_nav_path, index_col="date", parse_dates=["date"])
    for source_id_str in ["strategy_mr_dv2", "strategy_mr_qpi_ibs_rsi_exit", "strategy_crisis_trend_core"]:
        nav_dict = record_by_id_dict[source_id_str]["output_file_dict"]["nav"]
        nav_path = Path(nav_dict["path_str"])
        if sha256_str(nav_path) != nav_dict["sha256_str"]:
            raise ValueError(f"{source_id_str}: frozen benchmark evidence export changed")
        evidence_df = pd.read_csv(nav_path, index_col="date", parse_dates=["date"])
        common_idx = evidence_df.index.intersection(reference_df.index)
        difference_vec = evidence_df.loc[common_idx, "$SPX"].to_numpy() - reference_df.loc[common_idx, "$SPXTR"].to_numpy()
        benchmark_evidence_list.append({
            "source_id_str": source_id_str, "reference_source_id_str": reference_id_str,
            "comparison_str": "direct_saved_benchmark_levels_no_normalization_or_returns",
            "common_row_count_int": len(common_idx),
            "maximum_absolute_difference_float": float(np.max(np.abs(difference_vec))),
            "exactly_equal_bool": bool(np.array_equal(difference_vec, np.zeros(len(common_idx)))),
            "source_nav_sha256_str": nav_dict["sha256_str"],
            "reference_nav_sha256_str": reference_nav_dict["sha256_str"],
            "interpretation_str": "Matches an explicit-TR saved path; contradicts blanket price-only classification; does not recreate vendor lineage.",
        })
    interest_output_dict = export_frame(interest_df, interest_path, False)
    if sha256_str(manifest_path) != manifest_hash_str or any(
        sha256_str(Path(path_str)) != hash_str for path_str, hash_str in metadata_hash_dict.items()
    ):
        raise ValueError("Original source manifest or metadata changed during audit")
    addendum_dict = {
        "schema_version_str": "portfolio_source_audit_addendum_v1", "study_id_str": "portfolio_family_20260923",
        "recorded_at_utc_str": utc_now_str(), "phase_str": "review_correction_before_new_portfolio_outcomes",
        "original_source_manifest_sha256_str": manifest_hash_str,
        "frozen_research_spec_sha256_str": sha256_str(spec_path),
        "original_extractor_sha256_str": manifest_dict["extractor_sha256_str"],
        "updated_extractor_sha256_str": sha256_str(Path(__file__)),
        "precedence_str": "Apply these metadata overrides and added ledger alongside the unchanged frozen source manifest; no NAV or portfolio definition changes.",
        "original_source_metadata_hashes_preserved_bool": True,
        "benchmark_identity_override_list": correction_list,
        "benchmark_level_evidence_list": benchmark_evidence_list,
        "cash_interest_addition_dict": {
            "source_id_str": tactical_id_str, "source_attribute_str": "cash_interest_ledger_row_dict_list",
            "input_pickle_dict": pickle_dict, "output_file_dict": interest_output_dict,
            "recorded_total_float": recorded_total_float, "accounting_policy_total_float": policy_total_float,
            "validation_str": "exact native dates; finite values; nonnegative cash base; amount equals saved base times saved rate; ledger sum matches both saved totals",
            "pnl_treatment_str": "Interest is already embedded in native NAV; this export must not be added again.",
        },
        "performance_metrics_computed_bool": False,
    }
    write_json(addendum_path, addendum_dict)
    print(json.dumps({"addendum_path": str(addendum_path), "sha256": sha256_str(addendum_path),
                      "benchmark_corrections": len(correction_list), "cash_interest_rows": len(interest_df)}))


def main() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_PATH / "source_manifest.json"
    if manifest_path.exists():
        manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_dict["status_str"] == "complete":
            raise ValueError("Extraction already complete; do not replace a completed source freeze")
    else:
        manifest_dict = {
            "status_str": "selection_frozen_before_unpickle", "selection_frozen_at_utc_str": utc_now_str(),
            "study_id_str": "portfolio_family_20260923", "research_only_bool": True,
            "selection_rule_str": "lexicographically_latest_saved_vanilla_run_for_each_current_promoted_module",
            "registry_sha256_str": sha256_str(REGISTRY_PATH),
            "extractor_sha256_str": sha256_str(Path(__file__)),
            "current_shared_dependency_hash_dict": {path_str: sha256_str(ROOT_PATH / path_str) for path_str in SHARED_PATH_TUPLE},
            "source_selection_list": promoted_source_list(),
            "performance_metrics_computed_bool": False,
        }
        write_json(manifest_path, manifest_dict)
    source_record_list = []
    index_by_source_dict = {}
    for source_dict in manifest_dict["source_selection_list"]:
        source_record_dict, source_index_idx = extract_source(source_dict)
        source_record_list.append(source_record_dict)
        index_by_source_dict[source_dict["source_id_str"]] = source_index_idx
        print(f"extracted {source_dict['source_id_str']} ({len(source_index_idx)} rows)", flush=True)
    manifest_dict["calendar_validation_dict"] = validate_calendars(index_by_source_dict)
    manifest_dict["source_record_list"] = source_record_list
    manifest_dict["status_str"] = "complete"
    manifest_dict["completed_at_utc_str"] = utc_now_str()
    write_json(manifest_path, manifest_dict)
    print(json.dumps(manifest_dict["calendar_validation_dict"], indent=2), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--audit-addendum"]:
        write_audit_addendum()
    elif sys.argv[1:]:
        raise ValueError("Supported invocation: sources.py [--audit-addendum]")
    else:
        main()
