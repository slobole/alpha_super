"""Phase 1 inventory: standalone facts for every PM_READY + WIRED sleeve.

Reads the fresh source runs written by run_sources.py and writes, under
results/research/portfolio/fund_product_menu_20260923/inventory/:

- sleeve_returns.csv.gz     daily returns per sleeve (NaN before its first position)
- benchmark_returns.csv.gz  S&P 500 TR, 60/40 SPY/AGG, QQQ TR, T-bill accrual
- sleeve_stats_full.csv     metrics on each sleeve's own full history
- sleeve_stats_common.csv   metrics on the window every sleeve shares
- sleeve_activity.csv       trading cadence, turnover, cash usage per sleeve
- corr_daily_common.csv / corr_monthly_common.csv / corr_stress_common.csv
- correlation_clusters.csv  hierarchical clustering on 1 - corr (a check, not the engine map)

Descriptive only: nothing here chooses weights or products.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

INVENTORY_DIR_PATH = common.STUDY_DIR_PATH / "inventory"


def activity_row_dict(alias_str: str, path_df: pd.DataFrame, transaction_df: pd.DataFrame, first_date_str: str) -> dict:
    """Cadence and resource usage from the transaction ledger and daily cash."""
    live_path_df = path_df.loc[pd.Timestamp(first_date_str):]
    nav_ser = live_path_df["total_value_float"]
    year_count_float = (nav_ser.index[-1] - nav_ser.index[0]).days / 365.25
    live_transaction_df = transaction_df[transaction_df["date"] >= pd.Timestamp(first_date_str)]
    trade_day_count_int = int(live_transaction_df["date"].nunique())
    traded_notional_ser = live_transaction_df.groupby("date")["signed_notional_float"].apply(lambda s: s.abs().sum())
    turnover_ser = (traded_notional_ser / nav_ser.reindex(traded_notional_ser.index)).dropna()
    cash_weight_ser = live_path_df["cash_float"] / nav_ser
    invested_weight_ser = live_path_df["portfolio_value_float"] / nav_ser
    distinct_asset_int = int(live_transaction_df["asset_str"].nunique())
    return {
        "alias_str": alias_str,
        "trade_days_per_year_float": trade_day_count_int / year_count_float,
        "one_way_turnover_per_year_float": float(turnover_ser.sum() / 2.0 / year_count_float),
        "distinct_assets_traded_int": distinct_asset_int,
        "assets_sample_str": ",".join(sorted(live_transaction_df["asset_str"].unique())[:14]),
        "mean_cash_weight_float": float(cash_weight_ser.mean()),
        "mean_positive_cash_weight_float": float(cash_weight_ser.clip(lower=0.0).mean()),
        "share_days_fully_in_cash_float": float((invested_weight_ser.abs() < 1e-6).mean()),
        "negative_cash_day_share_float": float((live_path_df["cash_float"] < 0).mean()),
        "min_cash_weight_float": float(cash_weight_ser.min()),
        "mean_gross_invested_weight_float": float(invested_weight_ser.mean()),
    }


def main() -> int:
    metadata_by_alias_dict = common.load_sleeve_metadata_dict()
    path_by_alias_dict = common.load_sleeve_path_dict()
    allow_partial_bool = "--allow-partial" in sys.argv
    if (len(metadata_by_alias_dict) != 25 and not allow_partial_bool) or set(metadata_by_alias_dict) != set(path_by_alias_dict):
        raise RuntimeError(f"Expected 25 complete sleeves, found {sorted(metadata_by_alias_dict)}")
    output_dir_path = common.STUDY_DIR_PATH / "inventory_partial_debug" if allow_partial_bool else INVENTORY_DIR_PATH
    output_dir_path.mkdir(parents=True, exist_ok=True)

    nav_df = common.sleeve_nav_df(path_by_alias_dict, metadata_by_alias_dict)
    sleeve_return_df = nav_df.pct_change(fill_method=None)
    session_index = nav_df.index
    benchmark_return_df = common.build_benchmark_return_df(
        session_index, session_index[0].date().isoformat(), session_index[-1].date().isoformat()
    )
    sleeve_return_df.to_csv(output_dir_path / "sleeve_returns.csv.gz", float_format="%.10g", compression="gzip")
    benchmark_return_df.to_csv(output_dir_path / "benchmark_returns.csv.gz", float_format="%.10g", compression="gzip")

    # Full-history standalone metrics (each sleeve on its own window).
    full_row_list = []
    first_return_date_by_alias_dict = {}
    for alias_str in sleeve_return_df.columns:
        return_ser = sleeve_return_df[alias_str].dropna()
        first_return_date_by_alias_dict[alias_str] = return_ser.index[0]
        base_ts = nav_df[alias_str].dropna().index[0]
        row_dict = common.metric_dict(return_ser, benchmark_return_df["SPXTR"], benchmark_return_df["TBILL"], base_ts)
        row_dict.update(
            alias_str=alias_str,
            tier_str=metadata_by_alias_dict[alias_str]["tier_str"],
            first_invested_date_str=metadata_by_alias_dict[alias_str]["first_invested_date_str"],
            negative_cash_day_count_int=metadata_by_alias_dict[alias_str]["negative_cash_day_count_int"],
            min_cash_nav_weight_float=metadata_by_alias_dict[alias_str]["minimum_cash_nav_weight_float"],
        )
        full_row_list.append(row_dict)
    full_stats_df = pd.DataFrame(full_row_list).set_index("alias_str").sort_values("first_invested_date_str")
    full_stats_df.to_csv(output_dir_path / "sleeve_stats_full.csv", float_format="%.6g")

    # Common window: every sleeve with history since at least 2013 has a realised
    # return on every session. Sleeves that start later (the XLC-based dispersion
    # pair, 2018+) would shrink the window for everyone; they are listed as short
    # history and enter the pairwise-overlap correlation table only.
    short_history_alias_list = sorted(
        alias_str for alias_str, first_ts in first_return_date_by_alias_dict.items() if first_ts > pd.Timestamp("2013-12-31")
    )
    long_history_alias_list = [a for a in sleeve_return_df.columns if a not in short_history_alias_list]
    common_start_ts = max(first_return_date_by_alias_dict[a] for a in long_history_alias_list)
    common_return_df = sleeve_return_df.loc[common_start_ts:, long_history_alias_list]
    sleeve_return_df.corr(min_periods=500).to_csv(output_dir_path / "corr_daily_pairwise_overlap.csv", float_format="%.4f")
    print(f"short-history sleeves excluded from the common window: {short_history_alias_list}")
    if common_return_df.isna().any().any():
        gap_ser = common_return_df.isna().sum()
        raise RuntimeError(f"Gaps inside the common window: {gap_ser[gap_ser > 0].to_dict()}")
    common_base_ts = sleeve_return_df.index[sleeve_return_df.index.get_loc(common_start_ts) - 1]
    common_row_list = []
    for alias_str in common_return_df.columns:
        row_dict = common.metric_dict(
            common_return_df[alias_str], benchmark_return_df["SPXTR"], benchmark_return_df["TBILL"], common_base_ts
        )
        row_dict["alias_str"] = alias_str
        common_row_list.append(row_dict)
    common_stats_df = pd.DataFrame(common_row_list).set_index("alias_str")
    common_stats_df.to_csv(output_dir_path / "sleeve_stats_common.csv", float_format="%.6g")

    # Correlations on the common window: daily, monthly, and on S&P 500 stress days.
    corr_daily_df = common_return_df.corr()
    monthly_return_df = (1.0 + common_return_df).resample("ME").prod() - 1.0
    corr_monthly_df = monthly_return_df.corr()
    spx_common_ser = benchmark_return_df["SPXTR"].reindex(common_return_df.index)
    stress_mask = spx_common_ser <= spx_common_ser.quantile(0.10)
    corr_stress_df = common_return_df[stress_mask].corr()
    corr_daily_df.to_csv(output_dir_path / "corr_daily_common.csv", float_format="%.4f")
    corr_monthly_df.to_csv(output_dir_path / "corr_monthly_common.csv", float_format="%.4f")
    corr_stress_df.to_csv(output_dir_path / "corr_stress_common.csv", float_format="%.4f")

    # Hierarchical clustering as an independent check of the mechanism-based engine map.
    distance_mat = np.clip(1.0 - corr_monthly_df.to_numpy(), 0.0, 2.0)
    np.fill_diagonal(distance_mat, 0.0)
    linkage_mat = linkage(squareform(distance_mat, checks=False), method="average")
    cluster_df = pd.DataFrame(
        {
            "cluster_at_corr_0p5_int": fcluster(linkage_mat, t=0.5, criterion="distance"),
            "cluster_at_corr_0p7_int": fcluster(linkage_mat, t=0.3, criterion="distance"),
            "cluster_at_corr_0p9_int": fcluster(linkage_mat, t=0.1, criterion="distance"),
        },
        index=corr_monthly_df.index,
    ).sort_values(["cluster_at_corr_0p5_int", "cluster_at_corr_0p7_int"])
    cluster_df.to_csv(output_dir_path / "correlation_clusters.csv")

    # Cadence and resource usage.
    activity_row_list = []
    for alias_str, path_df in path_by_alias_dict.items():
        transaction_df = pd.read_csv(
            common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"]
        )
        activity_row_list.append(
            activity_row_dict(alias_str, path_df, transaction_df, metadata_by_alias_dict[alias_str]["first_invested_date_str"])
        )
    activity_df = pd.DataFrame(activity_row_list).set_index("alias_str")
    activity_df.to_csv(output_dir_path / "sleeve_activity.csv", float_format="%.6g")

    # Console digest.
    pd.set_option("display.width", 250)
    print(f"common window: {common_start_ts.date()} -> {common_return_df.index[-1].date()} ({len(common_return_df)} sessions)")
    digest_col_list = ["first_invested_date_str", "cagr_float", "volatility_float", "sharpe_rf0_float", "max_drawdown_float", "beta_spx_float", "corr_spx_daily_float"]
    print(full_stats_df[digest_col_list].round(3).to_string())
    print(common_stats_df[["cagr_float", "volatility_float", "sharpe_rf0_float", "max_drawdown_float", "beta_spx_float"]].round(3).to_string())
    print(activity_df[["trade_days_per_year_float", "one_way_turnover_per_year_float", "mean_positive_cash_weight_float", "negative_cash_day_share_float", "distinct_assets_traded_int"]].round(3).to_string())
    print(cluster_df.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
