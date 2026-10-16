"""Describe saved BTAL executed-order liquidity; never rerun strategies."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
OUTPUT_ROOT_PATH = REPO_ROOT_PATH / "results/review/wired_signoff_20261015"
VARIANT_KEY_DICT = {
    "taa_rank_tqqq": "strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "taa_equal_tqqq": "strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
    "taa_linearity_qqq": "strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
}


def hash_dict(path_obj: Path) -> dict:
    return {"path": str(path_obj.relative_to(REPO_ROOT_PATH)).replace("\\", "/"),
        "bytes": path_obj.stat().st_size, "sha256": hashlib.sha256(path_obj.read_bytes()).hexdigest()}


def records_list(order_df: pd.DataFrame) -> list[dict]:
    field_list = ["execution_date", "decision_date", "signed_adjusted_shares", "recorded_fill_price",
        "actual_ledger_notional", "decision_nav", "constant_1m_notional", "native_trailing20_median_turnover",
        "adv_status", "constant_1m_participation", "actual_participation"]
    return json.loads(order_df[field_list].to_json(orient="records", date_format="iso"))


def period_dict(order_df: pd.DataFrame) -> dict:
    covered_df = order_df.loc[order_df["adv_status"].eq("available")]
    missing_df = order_df.loc[~order_df["adv_status"].eq("available")]
    result_dict = {"executed_order_count_int": len(order_df), "covered_order_count_int": len(covered_df),
        "missing_adv_order_count_int": len(missing_df),
        "missing_adv_actual_notional_float": float(missing_df["actual_ledger_notional"].sum()),
        "known_fixed_1m_above_1pct_ADV_count_int": int(covered_df["constant_1m_participation"].gt(.01).sum()),
        "known_fixed_1m_above_100pct_ADV_count_int": int(covered_df["constant_1m_participation"].gt(1).sum()),
        "top_fixed_1m_participation_order_list": records_list(covered_df.nlargest(5, "constant_1m_participation")),
        "missing_adv_order_list": records_list(missing_df)}
    for basis_str, field_str in (("fixed_1m", "constant_1m_participation"), ("original_compounded", "actual_participation")):
        participation_ser = covered_df[field_str]
        result_dict[basis_str + "_participation_available_subset_dict"] = {
            "max_float": None if participation_ser.empty else float(participation_ser.max()),
            "p99_float": None if participation_ser.empty else float(participation_ser.quantile(.99)),
            "median_float": None if participation_ser.empty else float(participation_ser.median())}
    return result_dict


def main() -> None:
    variant_result_dict, input_manifest_list = {}, []
    for variant_str, key_str in VARIANT_KEY_DICT.items():
        statistics_path = OUTPUT_ROOT_PATH / "audit_statistics" / key_str / "statistics.json"
        orders_path = statistics_path.parent / "order_diagnostics.csv.gz"
        before_manifest_list = [hash_dict(path_obj) for path_obj in (statistics_path, orders_path)]
        statistics_dict = json.loads(statistics_path.read_text(encoding="utf-8"))
        order_df = pd.read_csv(orders_path, parse_dates=["execution_date", "decision_date"])
        btal_df = order_df.loc[order_df["asset"].eq("BTAL")].sort_values("execution_date")
        endpoint_ts = pd.Timestamp(statistics_dict["standalone_metrics"]["end"])
        # *** CRITICAL *** These are descriptive subsets of already-executed
        # historical orders. Their per-order ADV was computed from preceding
        #20 complete nativeTurnover sessions, excluding execution-day volume.
        period_result_dict = {"full": period_dict(btal_df),
            "last3y": period_dict(btal_df.loc[btal_df["execution_date"] >= endpoint_ts - pd.DateOffset(years=3)]),
            "last1y": period_dict(btal_df.loc[btal_df["execution_date"] >= endpoint_ts - pd.DateOffset(years=1)])}
        endpoint_dict = next(row_dict["current_endpoint"] for row_dict in
            statistics_dict["liquidity"]["period_details"]["asset_list"] if row_dict["asset"] == "BTAL")
        variant_result_dict[variant_str] = {"key_str": key_str, "endpoint_str": str(endpoint_ts.date()),
            "period_dict": period_result_dict, "current_endpoint_diagnostic_dict": endpoint_dict}
        if before_manifest_list != [hash_dict(path_obj) for path_obj in (statistics_path, orders_path)]:
            raise ValueError("Saved statistics/order diagnostics changed during the bounded read.")
        input_manifest_list.extend(before_manifest_list)
    result_dict = {"schema_version_str": "wired_signoff.btal_capacity.v1", "status_str": "pass",
        "generated_utc_str": datetime.now(UTC).isoformat(), "variant_result_dict": variant_result_dict,
        "formula_str": "fixed1M_notional=abs(recorded_adjusted_shares*recorded_fill_price)/captured_CloseT_NAV*1M; historical participation=fixed1M_notional/median(nativeTurnover preceding20 complete sessions). Original_compounded uses recordednotional without1M normalization.",
        "scope_str": "Executed BTAL legs only; no new engines/provider/strategyvariants. Full/last3y/last1y are order-date subsets sharing2026-10-14endpoint; zero last1y orders means no recent executed-order evidence, never a capacity pass.",
        "current_endpoint_basis_str": "Stored endpoint20session median includes completed endpoint bar, then scales historical last3y order sizes. It is retrospective current-scale context, not a current intended order or historical fill qualification.",
        "qualification_str": "Baseline2.5bps per side cannot be treated as verified executable historical cost at1M when order sizes exceed multiple days of aggregateADV. G012 requires everyMOO order>1%ADV to be flagged as extrapolation; this20median diagnostic is not CapacityAnalysisv2.1's full lower-of10mean/20median auction-impact rerun. Better recent liquidity cannot repair early-history fills.",
        "input_manifest_list": input_manifest_list, "source_manifest_list": [hash_dict(Path(__file__))],
        "policy_source_str": "ASSUMPTIONS_AND_GAPS.md:238"}
    output_path = OUTPUT_ROOT_PATH / "taa_ndx_btal_capacity.json"
    output_path.write_text(json.dumps(result_dict, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({variant_str: {period_str: {field_str: period_row_dict[field_str] for field_str in
        ("executed_order_count_int", "missing_adv_order_count_int", "known_fixed_1m_above_1pct_ADV_count_int",
        "fixed_1m_participation_available_subset_dict")} for period_str, period_row_dict in row_dict["period_dict"].items()}
        for variant_str, row_dict in variant_result_dict.items()}, indent=2))


if __name__ == "__main__":
    main()
