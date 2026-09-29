"""Transparent saved-unit portfolio diagnostics under a frozen research contract.

No strategy rules, production allocations, or orders are changed. Adjusted
returns are fixed-holdings sensitivities; outer rebalancing trades synthetic
sleeve units. Neither operation is a physical broker replay.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.portfolio_family_20260923.freeze import ROOT_PATH, STUDY_PATH, sha256_str


def verify_file(record_dict: dict) -> None:
    file_path = Path(record_dict["path_str"])
    if not file_path.is_file() or sha256_str(file_path) != record_dict["sha256_str"]:
        raise ValueError(f"Frozen input missing or changed: {file_path}")


def preflight_source_ids(spec_dict: dict, addendum_dict: dict) -> list[str]:
    manifest_dict = json.loads((STUDY_PATH/"source_manifest.json").read_text(encoding="utf-8"))
    if manifest_dict["status_str"] != "complete":
        raise ValueError("Source extraction incomplete")
    selected_id_list = [record_dict["source_id_str"] for record_dict in manifest_dict["source_selection_list"]]
    exported_id_list = [record_dict["source_id_str"] for record_dict in manifest_dict["source_record_list"]]
    if len(selected_id_list) != 25 or len(set(selected_id_list)) != 25 or set(selected_id_list) != set(exported_id_list):
        raise ValueError("Frozen source identities do not match all25 exports")
    for selection_dict in manifest_dict["source_selection_list"]:
        for record_dict in selection_dict["input_file_dict"].values():
            verify_file(record_dict)
    for source_dict in manifest_dict["source_record_list"]:
        metadata_path = STUDY_PATH/"data"/source_dict["source_id_str"]/"source_metadata.json"
        verify_file({"path_str":str(metadata_path), "sha256_str":source_dict["source_metadata_sha256_str"]})
        for record_dict in source_dict["output_file_dict"].values():
            verify_file(record_dict)
    verify_file(addendum_dict["cash_interest_addition_dict"]["output_file_dict"])
    benchmark_dict = json.loads((STUDY_PATH/"benchmarks/benchmark_manifest.json").read_text(encoding="utf-8"))
    if benchmark_dict["status_str"] != "complete" or set(benchmark_dict["controls"]) != {"benchmark_bil","benchmark_spy"}:
        raise ValueError("Benchmark source set incomplete")
    if benchmark_dict["input_sha256_str"] != spec_dict["data"]["benchmark_input_sha256"]:
        raise ValueError("Benchmark snapshot differs from frozen specification")
    verify_file({"path_str":benchmark_dict["input_path_str"],"sha256_str":benchmark_dict["input_sha256_str"]})
    for relative_str, hash_str in benchmark_dict["current_source_hash_dict"].items():
        verify_file({"path_str":str(ROOT_PATH/relative_str),"sha256_str":hash_str})
    for source_dict in benchmark_dict["source_metadata_list"]:
        metadata_path = STUDY_PATH/"benchmarks"/source_dict["source_id_str"]/"source_metadata.json"
        if json.loads(metadata_path.read_text(encoding="utf-8")) != source_dict:
            raise ValueError("Benchmark metadata differs from completed manifest")
        for record_dict in source_dict["output_file_dict"].values():
            verify_file(record_dict)
    return selected_id_list


def read_frame(file_path: Path, date_index_bool: bool = False) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()
    frame_df = pd.read_csv(file_path, index_col=0 if date_index_bool else None)
    if date_index_bool:
        frame_df.index = pd.to_datetime(frame_df.index)
    return frame_df


def event_totals(event_df: pd.DataFrame, date_column_str: str,
                 amount_series: pd.Series, calendar_idx: pd.DatetimeIndex) -> pd.Series:
    """Zero means no recorded event on an observed session, never a missing return."""
    event_date_idx = pd.to_datetime(event_df[date_column_str]).dt.normalize()
    if not event_date_idx.isin(calendar_idx).all():
        raise ValueError(f"Event outside native NAV calendar: {date_column_str}")
    event_series = pd.Series(amount_series.to_numpy(float), index=event_date_idx)
    return event_series.groupby(level=0).sum().reindex(calendar_idx, fill_value=0.)


def source_components(source_path: Path) -> tuple[pd.DataFrame, dict]:
    """Keep every native date, and disclose all retrospective cost adjustments."""
    metadata_dict = json.loads((source_path / "source_metadata.json").read_text(encoding="utf-8"))
    nav_df = read_frame(source_path / "nav.csv.gz", True)
    calendar_idx = nav_df.index
    if nav_df.empty or calendar_idx.has_duplicates or not calendar_idx.is_monotonic_increasing:
        raise ValueError(f"Invalid native calendar: {source_path}")
    if not np.isfinite(nav_df[["total_value", "portfolio_value", "cash"]]).all().all():
        raise ValueError(f"Non-finite source NAV: {source_path}")
    if not (nav_df.total_value > 0).all():
        raise ValueError("Positive NAV required")
    accounting_dict = metadata_dict["accounting_policy_dict"]
    native_tax_float = float(accounting_dict["dividend_withholding_rate_float"])
    # *** CRITICAL *** These shifts measure realized EOD returns, not signals.
    # r_t = NAV_t / NAV_(t-1) - 1; first source row is an anchor, never zero-filled.
    prior_nav_series = nav_df.total_value.shift(1)
    component_df = pd.DataFrame(index=calendar_idx)
    component_df["native"] = nav_df.total_value / prior_nav_series - 1
    component_df["cash_weight"] = nav_df.cash / nav_df.total_value
    day_count_vec = np.r_[np.diff(calendar_idx.values).astype("timedelta64[D]").astype(float), 0.]
    weight_df = read_frame(source_path / "realized_weights.csv.gz", True)
    if weight_df.empty:
        # Only the audited long-only Trinity source lacks an asset-weight panel.
        source_id_str = metadata_dict.get("source_id_str", source_path.name)
        if source_id_str != "strategy_taa_trinity_vol_control_8_bil":
            raise ValueError(f"Unknown missing holdings: {source_id_str}")
        component_df["gross"] = nav_df.portfolio_value / nav_df.total_value
        component_df["short"] = 0.
        component_df["embedded_equity_leverage_extra"] = 0.
    else:
        if not weight_df.index.equals(calendar_idx):
            raise ValueError(f"Holdings calendar mismatch: {source_path}")
        # *** CRITICAL *** Native snapshot rows enumerate only positions held at
        # Close_T. NaN in this sparse ASSET matrix means unheld, not missing prices
        # or missing returns. Cash is excluded from traded-security gross exposure.
        asset_weight_df = weight_df.drop(columns=["Cash"], errors="ignore").fillna(0.)
        component_df["gross"] = asset_weight_df.abs().sum(axis=1)
        component_df["short"] = -asset_weight_df.clip(upper=0).sum(axis=1)
        extra_series = pd.Series(0., index=calendar_idx)
        for asset_str, multiplier_float in {"TQQQ": 3., "QLD": 2., "SSO": 2., "UPRO": 3.}.items():
            if asset_str in asset_weight_df:
                extra_series += asset_weight_df[asset_str].abs() * (multiplier_float-1)
        component_df["embedded_equity_leverage_extra"] = extra_series
    collateral_series = 1.02 * component_df["short"] * nav_df.total_value
    borrow_df = read_frame(source_path / "borrow.csv.gz")
    native_borrow_series = pd.Series(0., index=calendar_idx)
    extra_borrow_series = native_borrow_series.copy()
    future_native_borrow_series = native_borrow_series.copy()
    future_extra_borrow_series = native_borrow_series.copy()
    if not borrow_df.empty:
        date_column_str = next(column_str for column_str in
            ("accrual_start_date_ts", "accrual_start_date", "date_ts") if column_str in borrow_df)
        native_borrow_series = event_totals(borrow_df, date_column_str, borrow_df.borrow_fee_float, calendar_idx)
        if "collateral_value_float" in borrow_df:
            exact_collateral_series = event_totals(borrow_df, date_column_str, borrow_df.collateral_value_float, calendar_idx)
            # Ledger absent on a non-short day => collateral zero. All actual short
            # sessions must be present; no future collateral or backward fill.
            borrow_date_idx = pd.DatetimeIndex(pd.to_datetime(borrow_df[date_column_str]))
            uncovered_short_series = component_df["short"].gt(1e-10) & ~calendar_idx.isin(borrow_date_idx)
            # Native CORE5 does not accrue beyond its terminal mark. That one
            # session has zero modeled calendar days in this exported sensitivity.
            if uncovered_short_series.iloc[:-1].any():
                raise ValueError("Missing collateral ledger on a short session")
            collateral_series = exact_collateral_series
        native_rate_series = (borrow_df.annual_borrow_rate_float.astype(float)
                              if "annual_borrow_rate_float" in borrow_df
                              else pd.Series(.01, index=borrow_df.index))
        if not (native_rate_series > 0).all():
            raise ValueError("Positive native borrow rate required for fee increment")
        # Increment only: native borrow is already debited inside saved NAV.
        incremental_fee_series = borrow_df.borrow_fee_float * ((.05-native_rate_series).clip(lower=0) / native_rate_series)
        extra_borrow_series = event_totals(borrow_df, date_column_str, incremental_fee_series, calendar_idx)
        if date_column_str in {"accrual_start_date_ts", "accrual_start_date"}:
            # CORE5/flow pre-accrue a future calendar interval at current close.
            # Crisis borrow is for the current trading day and is NOT refunded.
            future_native_borrow_series = native_borrow_series.copy()
            future_extra_borrow_series = extra_borrow_series.copy()
            extra_borrow_series.iloc[-1] = 0.
            future_extra_borrow_series.iloc[-1] = 0.
    elif component_df["short"].gt(1e-10).any():
        raise ValueError("Short source without a native borrow ledger")
    funding_base_series = (collateral_series-nav_df.cash).clip(lower=0)
    component_df["funding_base_weight"] = funding_base_series / nav_df.total_value
    # *** CRITICAL *** Post current-close fee for a known calendar interval; future
    # prices never enter. The native FINAL interval is zero, not invented carry.
    funding_unit_series = funding_base_series * day_count_vec / 360 / prior_nav_series
    tax_cost_series = pd.Series(0., index=calendar_idx)
    dividend_df = read_frame(source_path / "dividends.csv.gz")
    if not dividend_df.empty:
        incremental_tax_series = dividend_df.gross_dividend_cash_float.clip(lower=0) * (.25-native_tax_float)
        tax_cost_series = event_totals(dividend_df, "ex_date", incremental_tax_series, calendar_idx) / prior_nav_series
    transaction_df = read_frame(source_path / "transactions.csv.gz")
    turnover_series = pd.Series(0., index=calendar_idx)
    commission_series = turnover_series.copy()
    if not transaction_df.empty:
        notional_series = (transaction_df.amount * transaction_df.price).abs()
        turnover_series = event_totals(transaction_df, "bar", notional_series, calendar_idx) / prior_nav_series
        commission_series = event_totals(transaction_df, "bar", transaction_df.commission, calendar_idx) / prior_nav_series
    component_df["turnover"] = turnover_series
    component_df["commission_rate"] = commission_series
    component_df["native_borrow_rate"] = native_borrow_series / prior_nav_series
    component_df["tax_adjustment"] = tax_cost_series
    component_df["funding_common"] = funding_unit_series * .05
    component_df["funding_conservative"] = funding_unit_series * .08
    component_df["slippage_extra"] = turnover_series * .001
    component_df["borrow_extra"] = extra_borrow_series / prior_nav_series
    component_df["future_native_borrow"] = future_native_borrow_series / prior_nav_series
    component_df["future_extra_borrow"] = future_extra_borrow_series / prior_nav_series
    component_df["common_account"] = component_df.native-tax_cost_series-component_df.funding_common
    component_df["conservative"] = (component_df.native-tax_cost_series-component_df.funding_conservative
                                     -component_df.slippage_extra-component_df.borrow_extra)
    if not np.isfinite(component_df.iloc[1:].to_numpy(float)).all():
        raise ValueError(f"Non-finite derived components: {source_path}")
    if component_df[["native", "common_account", "conservative"]].iloc[1:].le(-1).any().any():
        raise ValueError("Sensitivity bankrupts source; cannot compound as positive fund units")
    return component_df, metadata_dict


def strict_return_panel(component_dict: dict[str, pd.DataFrame], source_id_list: list[str],
                        anchor_str: str, end_str: str, scenario_str: str) -> pd.DataFrame:
    expected_idx = None
    series_list = []
    for source_id_str in source_id_list:
        source_df = component_dict[source_id_str].loc[anchor_str:end_str]
        if source_df.empty or source_df.index[0] != pd.Timestamp(anchor_str) or source_df.index[-1] != pd.Timestamp(end_str):
            raise ValueError(f"Source does not cover exact window: {source_id_str}")
        if expected_idx is None:
            expected_idx = source_df.index
        if not source_df.index.equals(expected_idx):
            raise ValueError(f"Internal session mismatch: {source_id_str}")
        # *** CRITICAL *** Anchor is an observed ongoing-account close. Only returns
        # AFTER that close belong to this allocation comparison; no zero return is
        # manufactured. Inception orders require the separate native replay.
        return_series = source_df[scenario_str].iloc[1:].copy()
        if scenario_str != "native":
            # *** CRITICAL *** The EVALUATION terminal mark is common to all
            # sources, irrespective of later saved history. No hypothetical
            # financing or prepaid future borrow survives past that mark.
            # Preserve unadjusted native NAV in the separately labeled native arm.
            funding_column_str = "funding_common" if scenario_str == "common_account" else "funding_conservative"
            return_series.iloc[-1] += source_df[funding_column_str].iloc[-1] + source_df.future_native_borrow.iloc[-1]
            if scenario_str == "conservative":
                return_series.iloc[-1] += source_df.future_extra_borrow.iloc[-1]
        series_list.append(return_series.rename(source_id_str))
    panel_df = pd.concat(series_list, axis=1)
    if not np.isfinite(panel_df.to_numpy()).all():
        raise ValueError("Missing returns forbidden")
    return panel_df


def simulate_book(return_df: pd.DataFrame, weight_dict: dict[str, float],
                  anchor_ts: pd.Timestamp, rebalance_str: str,
                  outer_cost_float: float = .001) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    if return_df.empty or not return_df.index.is_unique or not return_df.index.is_monotonic_increasing:
        raise ValueError("Ordered unique return dates required")
    if anchor_ts >= return_df.index[0] or rebalance_str not in {"annual_fixed", "none_drift"}:
        raise ValueError("Invalid anchor or rebalance mode")
    source_id_list = list(weight_dict)
    target_vec = np.array([weight_dict[source_id_str] for source_id_str in source_id_list], float)
    if (target_vec < 0).any() or not np.isclose(target_vec.sum(), 1., atol=1e-12):
        raise ValueError("Long-only nonnegative sleeve budgets must sum to one")
    return_mat = return_df[source_id_list].to_numpy(float)
    if not np.isfinite(return_mat).all() or (return_mat <= -1).any():
        raise ValueError("Invalid component returns")
    sleeve_vec = target_vec.copy()
    nav_vec = np.ones(len(return_df)+1)
    weight_mat = np.zeros_like(return_mat)
    contribution_mat = np.zeros_like(return_mat)
    cost_vec = np.zeros(len(return_df))
    prior_year_int = anchor_ts.year
    for row_int, session_ts in enumerate(return_df.index):
        previous_nav_float = float(sleeve_vec.sum())
        previous_weight_vec = sleeve_vec/previous_nav_float
        if row_int == 0:
            cost_float = outer_cost_float  # declared initial fund-unit allocation proxy
            sleeve_vec *= 1-cost_float
        elif rebalance_str == "annual_fixed" and session_ts.year != prior_year_int:
            # *** CRITICAL *** Reset from PRIOR close only. Current return row has
            # not been read into NAV or weights. Fee = abs(target-prior weights)*c.
            cost_float = float(np.abs(target_vec-previous_weight_vec).sum()*outer_cost_float)
            sleeve_vec = previous_nav_float*(1-cost_float)*target_vec
        else:
            cost_float = 0.
        cost_vec[row_int] = cost_float
        weight_mat[row_int] = sleeve_vec/sleeve_vec.sum()
        sleeve_vec *= 1+return_mat[row_int]
        # Transfers are not profits. Allocate rebalance fees pro rata to new
        # target units; return contribution is weight*r minus allocated cost.
        contribution_mat[row_int] = weight_mat[row_int]*((1-cost_float)*(1+return_mat[row_int])-1)
        nav_vec[row_int+1] = sleeve_vec.sum()
        if not np.isclose(contribution_mat[row_int].sum(), nav_vec[row_int+1]/previous_nav_float-1, atol=1e-12):
            raise AssertionError("Component return attribution does not reconcile")
        prior_year_int = session_ts.year
    nav_series = pd.Series(nav_vec, index=pd.DatetimeIndex([anchor_ts]).append(return_df.index), name="nav")
    return (nav_series, pd.DataFrame(weight_mat, index=return_df.index, columns=source_id_list),
            pd.DataFrame(contribution_mat, index=return_df.index, columns=source_id_list),
            pd.Series(cost_vec, index=return_df.index, name="outer_cost"))


def metrics_dict(return_series: pd.Series, market_series: pd.Series | None = None,
                 cash_series: pd.Series | None = None) -> dict:
    return_vec = return_series.to_numpy(float)
    if len(return_vec) < 2 or not np.isfinite(return_vec).all() or (return_vec <= -1).any():
        raise ValueError("At least two finite solvent returns required")
    nav_vec = np.r_[1., np.cumprod(1+return_vec)]
    peak_vec = np.maximum.accumulate(nav_vec)
    drawdown_vec = nav_vec/peak_vec-1
    vol_float = float(np.std(return_vec, ddof=1)*np.sqrt(252))
    quantile_float = float(np.quantile(return_vec, .05))
    month_series = (1+return_series).resample("ME").prod()-1
    # *** CRITICAL *** Backward-only rolling realized wealth; never a signal.
    rolling_year_series = np.exp(np.log1p(return_series).rolling(252, min_periods=252).sum())-1
    underwater_int = 0
    max_underwater_int = 0
    for drawdown_float in drawdown_vec[1:]:
        underwater_int = underwater_int+1 if drawdown_float < -1e-12 else 0
        max_underwater_int = max(max_underwater_int, underwater_int)
    metric_dict = {"start": str(return_series.index[0].date()), "end": str(return_series.index[-1].date()),
                   "observations": len(return_vec), "cagr": nav_vec[-1]**(252/len(return_vec))-1,
                   "volatility": vol_float, "sharpe_zero": np.mean(return_vec)*252/vol_float if vol_float else np.nan,
                   "max_drawdown": float(drawdown_vec.min()), "es5_loss": -float(return_vec[return_vec <= quantile_float].mean()),
                   "worst_day": float(return_vec.min()), "worst_month": float(month_series.min()),
                   "worst_rolling252": float(rolling_year_series.min()) if len(return_vec) >=252 else np.nan,
                   "max_underwater_sessions": max_underwater_int, "negative_month_fraction": float((month_series<0).mean()),
                   "total_return": nav_vec[-1]-1}
    if market_series is not None:
        if not market_series.index.equals(return_series.index):
            raise ValueError("Market must use identical dates")
        market_vec = market_series.to_numpy(float)
        market_variance_float = np.var(market_vec, ddof=1)
        metric_dict["beta"] = np.cov(return_vec, market_vec, ddof=1)[0,1]/market_variance_float if market_variance_float > 0 else np.nan
        metric_dict["market_correlation"] = np.corrcoef(return_vec, market_vec)[0,1] if vol_float > 0 and market_variance_float > 0 else np.nan
        market_month_series = (1+market_series).resample("ME").prod()-1
        metric_dict["monthly_market_correlation"] = (month_series.corr(market_month_series)
            if len(month_series) >= 2 and month_series.std() > 0 and market_month_series.std() > 0 else np.nan)
    if cash_series is not None:
        if not cash_series.index.equals(return_series.index):
            raise ValueError("Cash control must use identical dates")
        excess_vec = return_vec-cash_series.to_numpy(float)
        excess_vol_float = np.std(excess_vec, ddof=1)*np.sqrt(252)
        metric_dict["sharpe_excess_bil"] = excess_vec.mean()*252/excess_vol_float if excess_vol_float else np.nan
        metric_dict["annualized_mean_excess_bil"] = excess_vec.mean()*252
    return metric_dict


def holm_adjust(pvalue_vec: np.ndarray) -> np.ndarray:
    order_vec = np.argsort(pvalue_vec)
    corrected_vec = np.minimum(1., np.maximum.accumulate((len(pvalue_vec)-np.arange(len(pvalue_vec)))*pvalue_vec[order_vec]))
    output_vec = np.empty_like(corrected_vec)
    output_vec[order_vec] = corrected_vec
    return output_vec


def paired_bootstrap(return_df: pd.DataFrame, test_list: list[dict], seed_int: int,
                     replicates_int: int, block_int: int) -> pd.DataFrame:
    rng_obj = np.random.default_rng(seed_int)
    observation_int = len(return_df)
    block_count_int = math.ceil(observation_int/block_int)
    start_mat = rng_obj.integers(0, observation_int, (replicates_int, block_count_int))
    sample_idx_mat = ((start_mat[:,:,None]+np.arange(block_int)) % observation_int).reshape(replicates_int,-1)[:,:observation_int]
    row_list = []
    for test_dict in test_list:
        delta_vec = (return_df[test_dict["candidate"]]-return_df[test_dict["comparator"]]).to_numpy()
        observed_mean_float = delta_vec.mean()
        bootstrap_mean_vec = delta_vec[sample_idx_mat].mean(axis=1)
        pvalue_float = (1+np.count_nonzero(bootstrap_mean_vec-observed_mean_float >= observed_mean_float))/(replicates_int+1)
        lower_float, upper_float = np.quantile(bootstrap_mean_vec*252, [.025,.975])
        row_list.append({**test_dict, "annualized_paired_mean": observed_mean_float*252,
                         "ci95_low": lower_float, "ci95_high": upper_float, "p_one_sided": pvalue_float})
    result_df = pd.DataFrame(row_list)
    result_df["p_holm"] = holm_adjust(result_df.p_one_sided.to_numpy())
    return result_df


def main() -> None:
    spec_path = STUDY_PATH / "research_spec_frozen.json"
    spec_dict = json.loads(spec_path.read_text(encoding="utf-8"))
    if sha256_str(STUDY_PATH / "source_manifest.json") != spec_dict["data"]["source_manifest_sha256"]:
        raise ValueError("Frozen source manifest changed")
    amendment_path = STUDY_PATH / "pre_result_amendment_01.json"
    amendment_dict = json.loads(amendment_path.read_text(encoding="utf-8"))
    addendum_path = STUDY_PATH / "source_audit_addendum.json"
    if sha256_str(addendum_path) != amendment_dict["source_audit_addendum_sha256"]:
        raise ValueError("Source audit addendum changed")
    addendum_dict = json.loads(addendum_path.read_text(encoding="utf-8"))
    selected_id_list = preflight_source_ids(spec_dict, addendum_dict)
    table_path = STUDY_PATH / "tables"
    table_path.mkdir(exist_ok=True)
    derived_path = STUDY_PATH / "derived"
    derived_path.mkdir(exist_ok=True)
    component_dict, metadata_dict = {}, {}
    for source_id_str in selected_id_list:
        source_path = STUDY_PATH/"data"/source_id_str
        source_df, source_metadata_dict = source_components(source_path)
        component_dict[source_path.name] = source_df
        metadata_dict[source_path.name] = source_metadata_dict
        source_df.to_csv(derived_path/f"{source_path.name}.csv.gz", index_label="date")
    standalone_id_list = list(component_dict)
    if len(standalone_id_list) != 25:
        raise ValueError("Expected25 native strategy series")
    for correction_dict in addendum_dict["benchmark_identity_override_list"]:
        metadata_dict[correction_dict["source_id_str"]].setdefault("benchmark_identity_by_column_dict", {})[
            correction_dict["benchmark_column_str"]] = correction_dict["corrected_identity_str"]
    (STUDY_PATH/"effective_source_metadata.json").write_text(json.dumps(metadata_dict,indent=2),encoding="utf-8")
    for source_id_str in ("benchmark_bil", "benchmark_spy"):
        source_df, source_metadata_dict = source_components(STUDY_PATH/"benchmarks"/source_id_str)
        component_dict[source_id_str] = source_df
        metadata_dict[source_id_str] = source_metadata_dict
    anchor_str, end_str = spec_dict["data"]["primary_anchor_close"], spec_dict["data"]["primary_last_close"]
    candidate_list = spec_dict["portfolio"]["candidates"]
    required_id_list = sorted(set().union(*(set(candidate_dict["weights"]) for candidate_dict in candidate_list)))
    metric_row_list, period_row_list, crisis_row_list, attribution_row_list = [], [], [], []
    yearly_row_list, standalone_row_list, long_history_row_list = [], [], []
    primary_return_df = pd.DataFrame()
    primary_nav_df = pd.DataFrame()
    primary_weight_dict = {}
    for scenario_str in ("native", "common_account", "conservative"):
        return_panel_df = strict_return_panel(component_dict, required_id_list+["benchmark_spy"], anchor_str, end_str, scenario_str)
        market_series, cash_series = return_panel_df.benchmark_spy, return_panel_df.benchmark_bil
        all_return_df = strict_return_panel(component_dict, standalone_id_list+["benchmark_spy","benchmark_bil"],
                                          spec_dict["data"]["all_25_anchor_close"], end_str, scenario_str)
        for source_id_str in standalone_id_list:
            standalone_row_list.append({"source_id": source_id_str, "scenario": scenario_str,
                **metrics_dict(all_return_df[source_id_str], all_return_df.benchmark_spy, all_return_df.benchmark_bil)})
            long_anchor_ts = max(component_dict[source_id_str].index[0], component_dict["benchmark_bil"].index[0])
            long_df = strict_return_panel(component_dict,[source_id_str,"benchmark_spy","benchmark_bil"],
                                         str(long_anchor_ts.date()),end_str,scenario_str)
            long_history_row_list.append({"source_id":source_id_str,"scenario":scenario_str,"period":"own_long_history",
                **metrics_dict(long_df[source_id_str],long_df.benchmark_spy,long_df.benchmark_bil)})
            for crisis_start_str, crisis_end_str in spec_dict["evaluation"]["long_history_crises"]:
                if long_df.index[0] > pd.Timestamp(crisis_start_str):
                    long_history_row_list.append({"source_id":source_id_str,"scenario":scenario_str,"period":crisis_start_str,"status":"unavailable_no_proxy"})
                    continue
                crisis_df = long_df.loc[crisis_start_str:crisis_end_str]
                long_history_row_list.append({"source_id":source_id_str,"scenario":scenario_str,"period":crisis_start_str,"status":"observed",
                    **metrics_dict(crisis_df[source_id_str],crisis_df.benchmark_spy,crisis_df.benchmark_bil)})
        for rebalance_str in ("annual_fixed", "none_drift"):
            for candidate_dict in candidate_list:
                candidate_id_str = candidate_dict["candidate_id"]
                nav_series, weight_df, contribution_df, outer_cost_series = simulate_book(
                    return_panel_df, candidate_dict["weights"], pd.Timestamp(anchor_str), rebalance_str)
                book_return_series = pd.Series(nav_series.to_numpy()[1:]/nav_series.to_numpy()[:-1]-1, index=return_panel_df.index)
                identity_dict = {"candidate_id": candidate_id_str, "family": candidate_dict["family"],
                                 "scenario": scenario_str, "rebalance": rebalance_str}
                metric_dict = metrics_dict(book_return_series, market_series, cash_series)
                # Same-close exposure uses END weights, not the return-period
                # starting weights. A strongly moving sleeve changes its share.
                end_weight_df = weight_df*(1+return_panel_df[weight_df.columns])
                end_weight_df = end_weight_df.div(end_weight_df.sum(axis=1),axis=0)
                for column_str in ("turnover", "gross", "short", "embedded_equity_leverage_extra", "funding_base_weight"):
                    component_panel_df = pd.DataFrame({source_id_str: component_dict[source_id_str].loc[book_return_series.index,column_str]
                                                       for source_id_str in weight_df.columns})
                    aggregation_weight_df = weight_df.mul(1-outer_cost_series,axis=0) if column_str == "turnover" else end_weight_df
                    weighted_series = (aggregation_weight_df*component_panel_df).sum(axis=1)
                    metric_dict[f"average_{column_str}"] = weighted_series.mean()
                    metric_dict[f"maximum_{column_str}"] = weighted_series.max()
                metric_dict["annual_turnover"] = metric_dict["average_turnover"]*252
                metric_dict["annual_outer_cost"] = outer_cost_series.mean()*252
                for cost_column_str in ("commission_rate","native_borrow_rate","tax_adjustment","funding_common",
                                        "funding_conservative","slippage_extra","borrow_extra"):
                    component_cost_df = pd.DataFrame({source_id_str:component_dict[source_id_str].loc[book_return_series.index,cost_column_str]
                                                       for source_id_str in weight_df.columns})
                    if cost_column_str.startswith("funding_"):
                        component_cost_df.iloc[-1] = 0.
                    elif cost_column_str == "borrow_extra":
                        for source_id_str in component_cost_df:
                            component_cost_df.loc[component_cost_df.index[-1],source_id_str] -= component_dict[source_id_str].loc[component_cost_df.index[-1],"future_extra_borrow"]
                    elif cost_column_str == "native_borrow_rate" and scenario_str != "native":
                        for source_id_str in component_cost_df:
                            component_cost_df.loc[component_cost_df.index[-1],source_id_str] -= component_dict[source_id_str].loc[component_cost_df.index[-1],"future_native_borrow"]
                    # Scenario-inactive components are zero so cost columns are
                    # actual charged components, never mutually exclusive sums.
                    active_cost_bool = (cost_column_str in ("commission_rate", "native_borrow_rate")
                        or (cost_column_str == "tax_adjustment" and scenario_str != "native")
                        or (cost_column_str == "funding_common" and scenario_str == "common_account")
                        or (cost_column_str in ("funding_conservative", "slippage_extra", "borrow_extra") and scenario_str == "conservative"))
                    if not active_cost_bool:
                        component_cost_df *= 0.
                    metric_dict[f"annual_{cost_column_str}"] = (weight_df.mul(1-outer_cost_series,axis=0)*component_cost_df).sum(axis=1).mean()*252
                metric_row_list.append({**identity_dict, **metric_dict})
                for period_int, (start_str, stop_str) in enumerate(spec_dict["evaluation"]["partitions"],1):
                    part_series = book_return_series.loc[start_str:stop_str]
                    period_row_list.append({**identity_dict,"period":period_int,
                        **metrics_dict(part_series, market_series.loc[part_series.index], cash_series.loc[part_series.index])})
                for start_str, stop_str in spec_dict["evaluation"]["crises"]:
                    part_series = book_return_series.loc[start_str:stop_str]
                    crisis_row_list.append({**identity_dict,"crisis":start_str,
                        **metrics_dict(part_series, market_series.loc[part_series.index], cash_series.loc[part_series.index])})
                if scenario_str == "common_account" and rebalance_str == "annual_fixed":
                    primary_return_df[candidate_id_str] = book_return_series
                    primary_nav_df[candidate_id_str] = nav_series
                    primary_weight_dict[candidate_id_str] = weight_df
                    year_series = (1+book_return_series).groupby(book_return_series.index.year).prod()-1
                    yearly_row_list.extend({"candidate_id":candidate_id_str,"year":int(year_int),"return":float(return_float)}
                                           for year_int,return_float in year_series.items())
                    variance_float = np.var(book_return_series.to_numpy(),ddof=1)
                    tail_mask_series = book_return_series <= book_return_series.quantile(.05)
                    for source_id_str in weight_df:
                        risk_share_float = np.cov(contribution_df[source_id_str],book_return_series,ddof=1)[0,1]/variance_float
                        attribution_row_list.append({"candidate_id":candidate_id_str,"source_id":source_id_str,
                            "capital_weight":candidate_dict["weights"][source_id_str],
                            "average_weight":weight_df[source_id_str].mean(),
                            "final_prior_weight":weight_df[source_id_str].iloc[-1],
                            "variance_contribution_fraction":risk_share_float,
                            "average_daily_return_contribution":contribution_df[source_id_str].mean(),
                            "tail_loss_contribution_fraction":contribution_df.loc[tail_mask_series,source_id_str].sum()/book_return_series.loc[tail_mask_series].sum()})
    pd.DataFrame(metric_row_list).to_csv(table_path/"portfolio_metrics.csv",index=False)
    pd.DataFrame(period_row_list).to_csv(table_path/"subperiod_metrics.csv",index=False)
    pd.DataFrame(crisis_row_list).to_csv(table_path/"crisis_metrics.csv",index=False)
    pd.DataFrame(attribution_row_list).to_csv(table_path/"risk_contributions.csv",index=False)
    pd.DataFrame(yearly_row_list).to_csv(table_path/"annual_returns.csv",index=False)
    pd.DataFrame(standalone_row_list).to_csv(table_path/"all25_common_metrics.csv",index=False)
    pd.DataFrame(long_history_row_list).to_csv(table_path/"long_history_metrics.csv",index=False)
    primary_return_df.to_csv(table_path/"primary_portfolio_returns.csv.gz",index_label="date")
    primary_nav_df.to_csv(table_path/"primary_portfolio_nav.csv.gz",index_label="date")
    primary_return_df.corr().to_csv(table_path/"portfolio_daily_correlations.csv",index_label="candidate_id")
    common_source_df = strict_return_panel(component_dict, required_id_list, anchor_str, end_str, "common_account")
    common_source_df.to_csv(table_path/"primary_source_returns.csv.gz",index_label="date")
    common_source_df.corr().to_csv(table_path/"source_daily_correlations.csv",index_label="source_id")
    ((1+common_source_df).resample("ME").prod()-1).corr().to_csv(table_path/"source_monthly_correlations.csv",index_label="source_id")
    market_series = strict_return_panel(component_dict,["benchmark_spy","benchmark_bil"],anchor_str,end_str,"common_account").benchmark_spy
    cash_series = strict_return_panel(component_dict,["benchmark_spy","benchmark_bil"],anchor_str,end_str,"common_account").benchmark_bil
    pd.DataFrame({"SPY_net25":market_series,"BIL_net25":cash_series}).to_csv(table_path/"benchmark_returns.csv.gz",index_label="date")
    primary_return_df.rolling(126,min_periods=126).corr(market_series).to_csv(table_path/"rolling126_market_correlation.csv.gz",index_label="date")
    bootstrap_dict = spec_dict["evaluation"]["bootstrap"]
    bootstrap_df = paired_bootstrap(primary_return_df,bootstrap_dict["tests"],bootstrap_dict["seed"],
                                    bootstrap_dict["replicates"],bootstrap_dict["block_sessions"])
    bootstrap_df.to_csv(table_path/"paired_bootstrap.csv",index=False)
    for candidate_id_str, weight_df in primary_weight_dict.items():
        weight_df.to_csv(derived_path/f"weights_{candidate_id_str}.csv.gz",index_label="date")
    run_dict = {"status":"complete_diagnostic_calculation", "spec_sha256":sha256_str(spec_path),
                "portfolio_cells":len(metric_row_list), "standalone_cells":len(standalone_row_list),
                "anchor":anchor_str,"end":end_str,"returns":len(primary_return_df),
                "claim":"saved-unit historical diagnostics, not native client inception or physical rebalancing replay"}
    (STUDY_PATH/"analysis_run.json").write_text(json.dumps(run_dict,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(run_dict))


if __name__ == "__main__":
    main()
