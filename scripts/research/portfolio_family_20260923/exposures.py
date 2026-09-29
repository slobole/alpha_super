"""Additional descriptive exposure diagnostics on already-seen portfolio history.

No allocation search, new candidate, hypothesis test, or physical account replay.
The immutable diagnostic specification is written before loading numerical paths.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.portfolio_family_20260923 import analyze
from scripts.research.portfolio_family_20260923.freeze import ROOT_PATH, STUDY_PATH, sha256_str


PRIMARY_ID_DICT = {
    "CORE5": "strategy_taa_adaptive_macro_core5",
    "NDX_VXN": "strategy_mo_atr_normalized_ndx_vxn_scaled",
    "MOSAIC": "strategy_mo_mosaic_russell1000",
    "DV2": "strategy_mr_dv2",
    "HPI_VOTE": "strategy_mr_hpi_sp500_2_3_5_vote",
}
CORE_ID_LIST = ["CORE_100", "CORE_075", "CORE_050", "CORE_025", "CORE_000"]


def freeze_diagnostic() -> tuple[dict, dict]:
    """Freeze definitions and exact dependencies before numerical source reads."""
    spec_path = STUDY_PATH / "exposures_spec.json"
    main_spec_path = STUDY_PATH / "research_spec_frozen.json"
    main_spec_dict = json.loads(main_spec_path.read_text(encoding="utf-8"))
    if spec_path.exists():
        diagnostic_dict = json.loads(spec_path.read_text(encoding="utf-8"))
        verify_dependencies(diagnostic_dict)
        return diagnostic_dict, main_spec_dict
    dependency_path_list = [
        main_spec_path, STUDY_PATH / "source_manifest.json",
        STUDY_PATH / "source_audit_addendum.json",
        STUDY_PATH / "benchmarks/benchmark_manifest.json",
        STUDY_PATH / "catalog_rules.json", STUDY_PATH / "defensive_rules.json",
        Path(__file__), Path(analyze.__file__), Path(analyze.__file__).with_name("freeze.py"),
    ]
    source_manifest_dict = json.loads((STUDY_PATH / "source_manifest.json").read_text(encoding="utf-8"))
    diagnostic_dict = {
        "schema_version_str": "portfolio-exposures-diagnostic-v1",
        "frozen_at_str": datetime.now(timezone.utc).isoformat(),
        "status_str": "additional_descriptive_diagnostic_on_previously_seen_history",
        "new_allocation_variants_int": 0, "new_inference_tests_int": 0,
        "outcomes_seen_statement_str": "Main portfolio analysis already began. No outputs of these additional exposure calculations were inspected before this specification; historical strategy/portfolio development and main research mean none of these dates is an untouched holdout.",
        "dependency_file_list": [{"path_str": str(file_path.resolve()), "sha256_str": sha256_str(file_path)} for file_path in dependency_path_list],
        "source_id_list": [source_dict["source_id_str"] for source_dict in source_manifest_dict["source_selection_list"]],
        "primary_id_dict": PRIMARY_ID_DICT,
        "scenario_str": "common_account",
        "primary_anchor_str": main_spec_dict["data"]["primary_anchor_close"],
        "primary_end_str": main_spec_dict["data"]["primary_last_close"],
        "all25_anchor_str": main_spec_dict["data"]["all_25_anchor_close"],
        "all25_end_str": main_spec_dict["data"]["all_25_last_close"],
        "partition_list": main_spec_dict["evaluation"]["partitions"],
        "rolling_sessions_int": 126,
        "return_diagnostics_list": ["all25 same-calendar daily Pearson correlation; native and common_account separately", "all ten pairs among the five representatives in full primary sample, each frozen partition, SPY-negative days and SPY worst5% days", "trailing126-session NDX/MOSAIC Pearson correlation, inclusive of reporting close; no trading signal"],
        "conditioning_str": "SPY common-account return on the exact primary calendar; down means <0; worst5% means <= full-primary empirical linear 5th percentile. Contemporaneous explanatory masks, not predictors; no filtering of strategy returns before calendar validation.",
        "joint_loss_formula_str": "mean((r_a<0)&(r_b<0)) on the named subset; conditional losses use only the named subset's loss days as denominator; zero-loss denominator is undefined.",
        "holdings_formula_str": "long_NAV_overlap_t=sum_asset min(max(w_NDX,0),max(w_MOSAIC,0)); conditional invested_long_overlap_t=sum min(w_NDX_long/sum(w_NDX_long),w_MOSAIC_long/sum(w_MOSAIC_long)) only when both have positive long exposure. Cash excluded; sparse native unheld cells=0; no date filling.",
        "core_candidate_id_list": CORE_ID_LIST,
        "outer_rebalance_str": "annual_fixed",
        "outer_cost_float": main_spec_dict["portfolio"]["outer_added_cost_bps_per_absolute_weight_change"] / 10000,
        "concentration_formula_str": "EOD sleeve weight q_i,t = b_i,t*(1+r_i,t)/sum_j b_j,t*(1+r_j,t). L_asset=sum_i q_i*max(w_i,asset,0), S_asset=sum_i q_i*max(-w_i,asset,0); unnetted gross=sum(L+S), HHI_long=sum((L/sumL)^2), effective_long_names=1/HHI. Topk long weights are shares of book NAV; normalized topk are shares of total long exposure. Cash excluded from asset concentration; opposing pod legs remain separate.",
        "ladder4_reference_str": "Only already-frozen LADDER_4_GROWTH, original25%NDX+8%MOSAIC. Report capital drift, Cov(component daily contribution,book daily return)/Var(book return), and sum(component contribution on book worst5% days)/sum(book return on those days). Fees included in additive contributions; negative risk contribution possible. No optimal ratio claim.",
        "limitations_list": [
            "All periods are seen history; correlations, tail conditioning, asset summaries and risk attribution are descriptive, without p-values or selection.",
            "Native saved whole-share holdings are mixed with adjusted synthetic sleeve NAV: indicative saved-unit exposure, not stateful re-sizing, a broker account, or an executable transfer ledger.",
            "Same-date EOD weights explain same-date exposures; they are not inputs to a same-date trading decision.",
            "Ticker concentration is not factor concentration. ETF constituents, derivatives and leveraged ETF economic look-through are unavailable; no ETF look-through or fabricated holdings are added.",
            "Long/short legs remain unnetted across pods; an explicitly named nettable diagnostic is not permission or proof of physical account netting.",
            "Trinity has no saved security weights; it participates in return correlations only and is not in the five CORE portfolios.",
            "Exact historical ticker identifiers are preserved; no current-universe substitution. Common stock risk can be large despite low name overlap.",
            "Current source hashes do not prove the code used when heterogeneous historical native runs were generated.",
        ],
    }
    spec_path.write_text(json.dumps(diagnostic_dict, indent=2) + "\n", encoding="utf-8")
    return diagnostic_dict, main_spec_dict


def verify_dependencies(diagnostic_dict: dict) -> None:
    for file_dict in diagnostic_dict["dependency_file_list"]:
        analyze.verify_file(file_dict)


def pair_statistics(return_df: pd.DataFrame, left_str: str, right_str: str) -> dict:
    left_series, right_series = return_df[left_str], return_df[right_str]
    left_loss_series, right_loss_series = left_series.lt(0), right_series.lt(0)
    both_loss_series = left_loss_series & right_loss_series
    correlation_float = (float(left_series.corr(right_series)) if len(return_df) > 1
                         and left_series.std() > 0 and right_series.std() > 0 else np.nan)
    return {
        "left_source": left_str, "right_source": right_str, "sessions": len(return_df),
        "correlation": correlation_float,
        "left_loss_fraction": float(left_loss_series.mean()),
        "right_loss_fraction": float(right_loss_series.mean()),
        "joint_loss_fraction": float(both_loss_series.mean()),
        "right_loss_given_left_loss": float(both_loss_series.sum() / left_loss_series.sum()) if left_loss_series.any() else np.nan,
        "left_loss_given_right_loss": float(both_loss_series.sum() / right_loss_series.sum()) if right_loss_series.any() else np.nan,
        "left_mean_daily_return": float(left_series.mean()),
        "right_mean_daily_return": float(right_series.mean()),
    }


def clean_holdings(weight_df: pd.DataFrame, calendar_idx: pd.DatetimeIndex) -> pd.DataFrame:
    if weight_df.index.has_duplicates or not weight_df.index.is_monotonic_increasing:
        raise ValueError("Ordered unique holdings dates required")
    selected_df = weight_df.loc[calendar_idx[0]:calendar_idx[-1]].copy()
    if not selected_df.index.equals(calendar_idx) or "Cash" not in selected_df:
        raise ValueError("Holdings require every exact session and the native Cash column")
    if not np.isfinite(selected_df["Cash"]).all():
        raise ValueError("Cash observations cannot be filled")
    # *** CRITICAL *** Sparse native asset snapshots omit UNHELD securities at
    # the same observed close. Only those cells become zero; no dates are filled.
    selected_df.loc[:, selected_df.columns != "Cash"] = selected_df.loc[:, selected_df.columns != "Cash"].fillna(0.)
    if not np.isfinite(selected_df.to_numpy(float)).all():
        raise ValueError("Non-finite saved holdings")
    if not np.allclose(selected_df.sum(axis=1), 1., rtol=0, atol=1e-7):
        raise ValueError("Native asset plus cash weights do not reconcile to NAV")
    return selected_df


def holdings_overlap(left_df: pd.DataFrame, right_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not left_df.index.equals(right_df.index):
        raise ValueError("Overlap calendars must match exactly")
    asset_list = sorted(set(left_df.columns).union(right_df.columns) - {"Cash"})
    left_long_df = left_df.reindex(columns=asset_list, fill_value=0.).clip(lower=0.)
    right_long_df = right_df.reindex(columns=asset_list, fill_value=0.).clip(lower=0.)
    overlap_df = pd.DataFrame(np.minimum(left_long_df.to_numpy(), right_long_df.to_numpy()), index=left_df.index, columns=asset_list)
    left_total_series, right_total_series = left_long_df.sum(axis=1), right_long_df.sum(axis=1)
    left_normal_df = left_long_df.div(left_total_series.where(left_total_series.gt(0)), axis=0)
    right_normal_df = right_long_df.div(right_total_series.where(right_total_series.gt(0)), axis=0)
    normalized_mat = np.minimum(left_normal_df.to_numpy(), right_normal_df.to_numpy())
    daily_df = pd.DataFrame({
        "long_nav_overlap": overlap_df.sum(axis=1),
        "invested_long_overlap": normalized_mat.sum(axis=1),
        "left_long_weight": left_total_series, "right_long_weight": right_total_series,
        "shared_positive_assets": ((left_long_df > 0) & (right_long_df > 0)).sum(axis=1),
    }, index=left_df.index)
    asset_df = pd.DataFrame({
        "asset": asset_list, "mean_overlap_nav": overlap_df.mean().to_numpy(),
        "both_held_sessions": ((left_long_df > 0) & (right_long_df > 0)).sum().to_numpy(),
        "both_held_fraction": ((left_long_df > 0) & (right_long_df > 0)).mean().to_numpy(),
        "mean_left_long_nav": left_long_df.mean().to_numpy(),
        "mean_right_long_nav": right_long_df.mean().to_numpy(),
    })
    return daily_df, asset_df.sort_values(["mean_overlap_nav", "asset"], ascending=[False, True])


def end_weights(begin_weight_df: pd.DataFrame, return_df: pd.DataFrame) -> pd.DataFrame:
    if not begin_weight_df.index.equals(return_df.index):
        raise ValueError("Sleeve weights and returns require the same calendar")
    # *** CRITICAL *** Reporting EOD exposures uses EOD sleeve NAV. These weights
    # are NOT the weights used to earn the current return or decide a rebalance.
    end_weight_df = begin_weight_df * (1 + return_df[begin_weight_df.columns])
    return end_weight_df.div(end_weight_df.sum(axis=1), axis=0)


def aggregate_holdings(end_weight_df: pd.DataFrame, holdings_dict: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_list = sorted(set().union(*(set(holdings_dict[source_str].columns) for source_str in end_weight_df)) - {"Cash"})
    long_df = pd.DataFrame(0., index=end_weight_df.index, columns=asset_list)
    short_df, cash_series = long_df.copy(), pd.Series(0., index=end_weight_df.index)
    for source_str in end_weight_df:
        native_df = holdings_dict[source_str]
        if not native_df.index.equals(end_weight_df.index):
            raise ValueError("Native holdings and book dates differ")
        native_asset_df = native_df.drop(columns="Cash").reindex(columns=asset_list, fill_value=0.)
        long_df += native_asset_df.clip(lower=0.).mul(end_weight_df[source_str], axis=0)
        short_df += -native_asset_df.clip(upper=0.).mul(end_weight_df[source_str], axis=0)
        cash_series += native_df.Cash * end_weight_df[source_str]
    long_total_series, short_total_series = long_df.sum(axis=1), short_df.sum(axis=1)
    normalized_long_df = long_df.div(long_total_series.where(long_total_series.gt(0)), axis=0)
    hhi_series = normalized_long_df.pow(2).sum(axis=1).where(long_total_series.gt(0))
    ordered_long_mat = np.sort(long_df.to_numpy(), axis=1)[:, ::-1]
    daily_df = pd.DataFrame({
        "long_nav": long_total_series, "short_nav": short_total_series,
        "gross_unnetted_nav": long_total_series + short_total_series,
        "net_signed_nav": long_total_series - short_total_series,
        "gross_if_same_ticker_netted_nav": (long_df-short_df).abs().sum(axis=1),
        "cash_nav": cash_series, "long_hhi": hhi_series,
        "effective_long_names": 1 / hhi_series,
        "positive_long_names": long_df.gt(0).sum(axis=1),
        "largest_long_asset": long_df.idxmax(axis=1).where(long_total_series.gt(0)),
    })
    for count_int in (1, 5, 10):
        daily_df[f"top{count_int}_long_nav"] = ordered_long_mat[:, :count_int].sum(axis=1)
        daily_df[f"top{count_int}_fraction_long"] = daily_df[f"top{count_int}_long_nav"] / long_total_series.where(long_total_series.gt(0))
    if not np.allclose(daily_df.net_signed_nav + daily_df.cash_nav, 1., rtol=0, atol=1e-7):
        raise AssertionError("Aggregate holdings fail NAV reconciliation")
    asset_df = pd.DataFrame({
        "asset": asset_list, "mean_long_nav": long_df.mean().to_numpy(),
        "mean_short_nav": short_df.mean().to_numpy(),
        "max_long_nav": long_df.max().to_numpy(), "max_short_nav": short_df.max().to_numpy(),
        "long_held_fraction": long_df.gt(0).mean().to_numpy(),
        "short_held_fraction": short_df.gt(0).mean().to_numpy(),
    })
    return daily_df, asset_df.sort_values(["mean_long_nav", "asset"], ascending=[False, True])


def descriptive_summary(frame_df: pd.DataFrame) -> pd.DataFrame:
    record_list = []
    for column_str in frame_df.select_dtypes(include="number"):
        value_series = frame_df[column_str].dropna()
        record_list.append({"field": column_str, "observations": len(value_series),
                            "mean": value_series.mean(), "minimum": value_series.min(),
                            "median": value_series.median(), "p05": value_series.quantile(.05),
                            "p95": value_series.quantile(.95), "maximum": value_series.max()})
    return pd.DataFrame(record_list)


def main() -> None:
    diagnostic_dict, main_spec_dict = freeze_diagnostic()
    verify_dependencies(diagnostic_dict)
    table_path = STUDY_PATH / "tables"
    manifest_path = table_path / "exposures_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Completed exposure diagnostic exists; preserve it")
    addendum_dict = json.loads((STUDY_PATH / "source_audit_addendum.json").read_text(encoding="utf-8"))
    source_id_list = analyze.preflight_source_ids(main_spec_dict, addendum_dict)
    if source_id_list != diagnostic_dict["source_id_list"]:
        raise ValueError("Diagnostic source selection changed")
    component_dict, metadata_dict = {}, {}
    for source_str in source_id_list:
        component_dict[source_str], metadata_dict[source_str] = analyze.source_components(STUDY_PATH / "data" / source_str)
    component_dict["benchmark_spy"], metadata_dict["benchmark_spy"] = analyze.source_components(STUDY_PATH / "benchmarks/benchmark_spy")
    output_list = []

    def save_table(name_str: str, frame_df: pd.DataFrame, index_bool: bool = False) -> None:
        file_path = table_path / f"exposures_{name_str}.csv.gz"
        frame_df.to_csv(file_path, index=index_bool, index_label="date" if index_bool else None,
                        float_format="%.17g", compression={"method": "gzip", "mtime": 0})
        output_list.append({"path_str": str(file_path), "sha256_str": sha256_str(file_path),
                            "rows_int": len(frame_df), "columns_list": list(frame_df.columns)})

    for scenario_str in ("native", "common_account"):
        all_return_df = analyze.strict_return_panel(component_dict, source_id_list,
            diagnostic_dict["all25_anchor_str"], diagnostic_dict["all25_end_str"], scenario_str)
        correlation_df = all_return_df.corr()
        correlation_df.index.name = "source_id"
        save_table(f"all25_correlation_{scenario_str}", correlation_df.reset_index())
    primary_id_list = list(PRIMARY_ID_DICT.values())
    primary_return_df = analyze.strict_return_panel(component_dict, primary_id_list + ["benchmark_spy"],
        diagnostic_dict["primary_anchor_str"], diagnostic_dict["primary_end_str"], "common_account")
    spy_series = primary_return_df.benchmark_spy
    spy_cutoff_float = float(spy_series.quantile(.05))
    subset_dict = {"full_primary": primary_return_df,
                   "spy_negative": primary_return_df.loc[spy_series.lt(0)],
                   "spy_worst5pct": primary_return_df.loc[spy_series.le(spy_cutoff_float)]}
    for partition_int, (start_str, end_str) in enumerate(diagnostic_dict["partition_list"], start=1):
        subset_dict[f"partition_{partition_int}"] = primary_return_df.loc[start_str:end_str]
    pair_record_list = []
    for subset_str, subset_df in subset_dict.items():
        for left_str, right_str in combinations(primary_id_list, 2):
            pair_record_list.append({"subset": subset_str, "start": str(subset_df.index[0].date()),
                                    "end": str(subset_df.index[-1].date()),
                                    **pair_statistics(subset_df, left_str, right_str)})
    save_table("primary_pair_diagnostics", pd.DataFrame(pair_record_list))
    left_str, right_str = PRIMARY_ID_DICT["NDX_VXN"], PRIMARY_ID_DICT["MOSAIC"]
    # *** CRITICAL *** Trailing inclusive reporting window uses only returns up
    # through each displayed close. It never determines a trade or allocation.
    rolling_series = primary_return_df[left_str].rolling(126, min_periods=126).corr(primary_return_df[right_str])
    save_table("ndx_mosaic_rolling126", rolling_series.rename("correlation").to_frame(), True)
    save_table("ndx_mosaic_rolling126_summary", descriptive_summary(rolling_series.rename("correlation").to_frame()))
    holdings_dict = {source_str: clean_holdings(analyze.read_frame(STUDY_PATH / "data" / source_str / "realized_weights.csv.gz", True), primary_return_df.index) for source_str in primary_id_list}
    overlap_df, shared_asset_df = holdings_overlap(holdings_dict[left_str], holdings_dict[right_str])
    save_table("ndx_mosaic_overlap_daily", overlap_df, True)
    save_table("ndx_mosaic_overlap_summary", descriptive_summary(overlap_df))
    save_table("ndx_mosaic_shared_assets", shared_asset_df)
    core_daily_list, core_summary_list, core_asset_list = [], [], []
    candidate_dict = {candidate_dict["candidate_id"]: candidate_dict for candidate_dict in main_spec_dict["portfolio"]["candidates"]}
    anchor_ts = pd.Timestamp(diagnostic_dict["primary_anchor_str"])
    for candidate_str in CORE_ID_LIST:
        _, begin_weight_df, _, _ = analyze.simulate_book(primary_return_df,
            candidate_dict[candidate_str]["weights"], anchor_ts, "annual_fixed", diagnostic_dict["outer_cost_float"])
        daily_df, asset_df = aggregate_holdings(end_weights(begin_weight_df, primary_return_df), holdings_dict)
        summary_df = descriptive_summary(daily_df)
        daily_df = daily_df.reset_index(names="date")
        for frame_df in (daily_df, summary_df, asset_df):
            frame_df.insert(0, "candidate_id", candidate_str)
        core_daily_list.append(daily_df)
        core_summary_list.append(summary_df)
        core_asset_list.append(asset_df)
    save_table("core_concentration_daily", pd.concat(core_daily_list, ignore_index=True))
    save_table("core_concentration_summary", pd.concat(core_summary_list, ignore_index=True))
    save_table("core_assets", pd.concat(core_asset_list, ignore_index=True))
    ladder_dict = candidate_dict["LADDER_4_GROWTH"]["weights"]
    ladder_return_df = analyze.strict_return_panel(component_dict, list(ladder_dict),
        diagnostic_dict["primary_anchor_str"], diagnostic_dict["primary_end_str"], "common_account")
    _, begin_weight_df, contribution_df, _ = analyze.simulate_book(ladder_return_df, ladder_dict,
        anchor_ts, "annual_fixed", diagnostic_dict["outer_cost_float"])
    end_weight_df = end_weights(begin_weight_df, ladder_return_df)
    book_return_series = contribution_df.sum(axis=1)
    tail_mask_series = book_return_series.le(book_return_series.quantile(.05))
    risk_record_list = []
    for group_str, member_list in (("NDX_VXN", [left_str]), ("MOSAIC", [right_str]), ("MOMENTUM_COMBINED", [left_str, right_str])):
        group_contribution_series = contribution_df[member_list].sum(axis=1)
        risk_record_list.append({"candidate_id": "LADDER_4_GROWTH", "group": group_str,
            "initial_capital_weight": sum(ladder_dict[source_str] for source_str in member_list),
            "mean_begin_capital_weight": begin_weight_df[member_list].sum(axis=1).mean(),
            "mean_end_capital_weight": end_weight_df[member_list].sum(axis=1).mean(),
            "min_end_capital_weight": end_weight_df[member_list].sum(axis=1).min(),
            "max_end_capital_weight": end_weight_df[member_list].sum(axis=1).max(),
            "variance_contribution_fraction": group_contribution_series.cov(book_return_series) / book_return_series.var(),
            "book_tail_sessions": int(tail_mask_series.sum()),
            "book_worst5pct_contribution_fraction": group_contribution_series.loc[tail_mask_series].sum() / book_return_series.loc[tail_mask_series].sum()})
    save_table("ladder4_momentum_risk", pd.DataFrame(risk_record_list))
    save_table("source_accounting_context", pd.DataFrame([{
        "source_id": source_str, "native_start": source_dict["actual_start_date_str"],
        "native_end": source_dict["actual_end_date_str"], "native_capital": source_dict["native_capital_float"],
        "native_dividend_withholding": source_dict["accounting_policy_dict"]["dividend_withholding_rate_float"],
        "saved_asset_holdings_available": source_dict.get("realized_weights_available_bool", True),
        "historical_source_hash_available": source_dict.get("saved_metadata_source_hash_available_bool", False),
    } for source_str, source_dict in metadata_dict.items() if source_str in source_id_list]))
    verify_dependencies(diagnostic_dict)
    manifest_dict = {"status_str": "complete", "completed_at_str": datetime.now(timezone.utc).isoformat(),
        "spec_path_str": str(STUDY_PATH / "exposures_spec.json"),
        "spec_sha256_str": sha256_str(STUDY_PATH / "exposures_spec.json"),
        "primary_return_sessions_int": len(primary_return_df), "all25_return_sessions_int": len(all_return_df),
        "spy_primary_5pct_return_cutoff_float": spy_cutoff_float,
        "source_preflight_bool": True, "dependencies_unchanged_bool": True,
        "new_allocation_variants_int": 0, "new_inference_tests_int": 0,
        "limitations_list": diagnostic_dict["limitations_list"], "output_file_list": output_list}
    manifest_path.write_text(json.dumps(manifest_dict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "spec_sha256": manifest_dict["spec_sha256_str"],
                      "tables": len(output_list), "primary_sessions": len(primary_return_df),
                      "all25_sessions": len(all_return_df)}))


if __name__ == "__main__":
    main()
