from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.report import build_research_output_path
from scripts.research.run_sector_dispersion_marginal_universe_study import (
    ACCEPTANCE_RULE_DICT,
    IN_SAMPLE_END_TS,
    OUT_OF_SAMPLE_START_TS,
    STRESS_RULE_DICT,
    _active_bool_ser,
    _correlation_float,
    _daily_return_ser,
    _json_default_obj,
    _markdown_table_str,
    _run_strategy_variant,
    _safe_delta_float,
    _safe_float,
    _strategy_summary_row_dict,
    build_candidate_manifest_df,
    compute_tail_stress_metric_dict,
    evaluate_acceptance_rule,
    evaluate_stress_rule,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    DEFAULT_CONFIG,
    ORIGINAL_SYMBOL_TUPLE,
    UNIVERSE_A_SYMBOL_TUPLE,
    UNIVERSE_B_SYMBOL_TUPLE,
    UNIVERSE_C_SYMBOL_TUPLE,
    get_sector_dispersion_ibs_data,
)


FULL_UNIVERSE_ADDITION_DICT = {
    "full_universe_a": tuple(symbol_str for symbol_str in UNIVERSE_A_SYMBOL_TUPLE if symbol_str not in ORIGINAL_SYMBOL_TUPLE),
    "full_universe_b": tuple(symbol_str for symbol_str in UNIVERSE_B_SYMBOL_TUPLE if symbol_str not in ORIGINAL_SYMBOL_TUPLE),
    "full_universe_c": tuple(symbol_str for symbol_str in UNIVERSE_C_SYMBOL_TUPLE if symbol_str not in ORIGINAL_SYMBOL_TUPLE),
}

DEFAULT_TRIPLE_POOL_SYMBOL_TUPLE = (
    "KIE",
    "XLRE",
    "IHI",
    "IYT",
    "ITA",
    "XLC",
    "XLB",
    "XLF",
    "XLI",
)

COMBINATION_ACCEPTANCE_RULE_DICT = {
    "min_delta_oos_sharpe_float": 0.0,
    "min_delta_oos_max_drawdown_pct_float": -3.0,
    "min_delta_full_sharpe_float": -0.10,
    "max_delta_cost_drag_ann_pct_float": 0.35,
    "min_base_tail_delta_mean_return_pct_float": 0.0,
    "min_market_tail_delta_mean_return_pct_float": 0.0,
}

COMBINATION_STRESS_RULE_DICT = {
    "min_base_tail_delta_mean_return_pct_float": 0.0,
    "min_market_tail_delta_mean_return_pct_float": 0.0,
    "min_base_tail_active_pct_float": 5.0,
    "min_market_tail_active_pct_float": 5.0,
}

COMPOSITE_WEIGHT_DICT = {
    "delta_oos_sharpe_float": 0.30,
    "base_tail_delta_mean_return_pct_float": 0.22,
    "market_tail_delta_mean_return_pct_float": 0.18,
    "delta_full_sharpe_float": 0.12,
    "delta_oos_max_drawdown_pct_float": 0.10,
    "delta_cost_drag_ann_pct_float": -0.05,
    "addition_count_int": -0.03,
}


def _slug_str(raw_value_str: str) -> str:
    keep_char_list: list[str] = []
    for char_str in str(raw_value_str).lower():
        keep_char_list.append(char_str if char_str.isalnum() else "_")
    return "_".join(filter(None, "".join(keep_char_list).split("_")))


def _addition_tuple_str(addition_tuple: tuple[str, ...]) -> str:
    return ",".join(addition_tuple)


def _variant_name_str(addition_tuple: tuple[str, ...], variant_kind_str: str) -> str:
    if len(addition_tuple) == 0:
        return "strategy_mr_sector_dispersion_ibs_combo_base"
    return f"strategy_mr_sector_dispersion_ibs_combo_{variant_kind_str}_{_slug_str('_'.join(addition_tuple))}"


def _variant_label_str(addition_tuple: tuple[str, ...]) -> str:
    if len(addition_tuple) == 0:
        return "Base"
    return "Base+" + "+".join(addition_tuple)


def build_combination_manifest_df(
    include_pairs_bool: bool = True,
    include_triples_bool: bool = True,
    include_full_universes_bool: bool = True,
    triple_pool_symbol_tuple: tuple[str, ...] = DEFAULT_TRIPLE_POOL_SYMBOL_TUPLE,
    pair_scope_str: str = "challenger_pool",
) -> pd.DataFrame:
    candidate_manifest_df = build_candidate_manifest_df()
    candidate_symbol_tuple = tuple(candidate_manifest_df["symbol_str"].astype(str).tolist())
    candidate_symbol_set = set(candidate_symbol_tuple)
    triple_pool_tuple = tuple(symbol_str for symbol_str in triple_pool_symbol_tuple if symbol_str in candidate_symbol_set)
    if pair_scope_str == "all":
        pair_symbol_tuple = candidate_symbol_tuple
        pair_note_str = "All two-asset combinations from the frozen candidate manifest."
    elif pair_scope_str == "challenger_pool":
        pair_symbol_tuple = triple_pool_tuple
        pair_note_str = "All two-asset combinations from the fixed stress-pass challenger pool."
    else:
        raise ValueError("pair_scope_str must be 'challenger_pool' or 'all'.")

    row_dict_list: list[dict[str, object]] = [
        {
            "variant_rank_int": 1,
            "variant_kind_str": "baseline",
            "variant_label_str": "Base",
            "addition_count_int": 0,
            "addition_tuple_str": "",
            "symbol_tuple_str": _addition_tuple_str(ORIGINAL_SYMBOL_TUPLE),
            "selection_note_str": "Original paper basket baseline.",
        }
    ]

    variant_rank_int = 2
    for candidate_symbol_str in candidate_symbol_tuple:
        addition_tuple = (candidate_symbol_str,)
        row_dict_list.append(
            {
                "variant_rank_int": variant_rank_int,
                "variant_kind_str": "single_add",
                "variant_label_str": _variant_label_str(addition_tuple),
                "addition_count_int": len(addition_tuple),
                "addition_tuple_str": _addition_tuple_str(addition_tuple),
                "symbol_tuple_str": _addition_tuple_str(ORIGINAL_SYMBOL_TUPLE + addition_tuple),
                "selection_note_str": "All predeclared Universe A/B/C single additions.",
            }
        )
        variant_rank_int += 1

    if include_pairs_bool:
        for addition_tuple in itertools.combinations(pair_symbol_tuple, 2):
            row_dict_list.append(
                {
                    "variant_rank_int": variant_rank_int,
                    "variant_kind_str": "pair_add",
                    "variant_label_str": _variant_label_str(addition_tuple),
                    "addition_count_int": len(addition_tuple),
                    "addition_tuple_str": _addition_tuple_str(addition_tuple),
                    "symbol_tuple_str": _addition_tuple_str(ORIGINAL_SYMBOL_TUPLE + addition_tuple),
                    "selection_note_str": pair_note_str,
                }
            )
            variant_rank_int += 1

    if include_triples_bool:
        for addition_tuple in itertools.combinations(triple_pool_tuple, 3):
            row_dict_list.append(
                {
                    "variant_rank_int": variant_rank_int,
                    "variant_kind_str": "selected_triple_add",
                    "variant_label_str": _variant_label_str(addition_tuple),
                    "addition_count_int": len(addition_tuple),
                    "addition_tuple_str": _addition_tuple_str(addition_tuple),
                    "symbol_tuple_str": _addition_tuple_str(ORIGINAL_SYMBOL_TUPLE + addition_tuple),
                    "selection_note_str": (
                        "Triples only from the fixed stress-pass challenger pool; this is a second-stage search."
                    ),
                }
            )
            variant_rank_int += 1

    if include_full_universes_bool:
        for variant_kind_str, addition_tuple in FULL_UNIVERSE_ADDITION_DICT.items():
            row_dict_list.append(
                {
                    "variant_rank_int": variant_rank_int,
                    "variant_kind_str": variant_kind_str,
                    "variant_label_str": _variant_label_str(addition_tuple),
                    "addition_count_int": len(addition_tuple),
                    "addition_tuple_str": _addition_tuple_str(addition_tuple),
                    "symbol_tuple_str": _addition_tuple_str(ORIGINAL_SYMBOL_TUPLE + addition_tuple),
                    "selection_note_str": "Full-universe diagnostic only; not a direct default candidate.",
                }
            )
            variant_rank_int += 1

    manifest_df = pd.DataFrame(row_dict_list)
    duplicate_addition_df = manifest_df.loc[manifest_df["addition_tuple_str"].duplicated(keep=False)]
    if len(duplicate_addition_df) > 0:
        raise RuntimeError(f"Duplicate combination rows: {duplicate_addition_df['addition_tuple_str'].tolist()}")
    return manifest_df


def _benchmark_return_ser(pricing_data_df: pd.DataFrame, benchmark_symbol_str: str) -> pd.Series:
    benchmark_close_ser = pd.to_numeric(pricing_data_df[(benchmark_symbol_str, "Close")], errors="coerce")
    benchmark_close_ser.index = pd.to_datetime(benchmark_close_ser.index).normalize()
    # *** CRITICAL*** Benchmark returns are post-run diagnostics only:
    # r_t = Close_t / Close_{t-1} - 1. They must not feed signal generation,
    # sizing, or candidate inclusion during the backtest loop.
    return benchmark_close_ser.pct_change(fill_method=None)


def _downside_beta_float(strategy_return_ser: pd.Series, benchmark_return_ser: pd.Series, benchmark_quantile_float: float) -> float:
    aligned_return_df = pd.concat(
        [strategy_return_ser.rename("strategy"), benchmark_return_ser.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned_return_df) < 3:
        return float("nan")
    threshold_float = float(aligned_return_df["benchmark"].quantile(benchmark_quantile_float))
    tail_return_df = aligned_return_df.loc[aligned_return_df["benchmark"].le(threshold_float)]
    benchmark_variance_float = float(tail_return_df["benchmark"].var())
    if len(tail_return_df) < 3 or benchmark_variance_float == 0.0:
        return float("nan")
    return float(tail_return_df["strategy"].cov(tail_return_df["benchmark"]) / benchmark_variance_float)


def _variant_diagnostic_row_dict(
    manifest_row_ser: pd.Series,
    baseline_strategy_obj,
    variant_strategy_obj,
    benchmark_return_ser: pd.Series,
    baseline_summary_dict: dict[str, object],
    variant_summary_dict: dict[str, object],
) -> dict[str, object]:
    baseline_return_ser = _daily_return_ser(baseline_strategy_obj)
    variant_return_ser = _daily_return_ser(variant_strategy_obj)
    variant_active_bool_ser = _active_bool_ser(variant_strategy_obj)
    tail_metric_dict = compute_tail_stress_metric_dict(
        baseline_return_ser=baseline_return_ser,
        standalone_return_ser=variant_return_ser,
        marginal_return_ser=variant_return_ser,
        benchmark_return_ser=benchmark_return_ser,
        standalone_active_bool_ser=variant_active_bool_ser,
    )
    addition_tuple_str = str(manifest_row_ser["addition_tuple_str"])
    addition_tuple = tuple(filter(None, addition_tuple_str.split(",")))
    diagnostic_row_dict: dict[str, object] = {
        "variant_rank_int": int(manifest_row_ser["variant_rank_int"]),
        "variant_kind_str": str(manifest_row_ser["variant_kind_str"]),
        "variant_label_str": str(manifest_row_ser["variant_label_str"]),
        "addition_count_int": int(manifest_row_ser["addition_count_int"]),
        "addition_tuple_str": addition_tuple_str,
        "symbol_tuple_str": str(manifest_row_ser["symbol_tuple_str"]),
        "selection_note_str": str(manifest_row_ser["selection_note_str"]),
        "tested_addition_symbol_tuple": addition_tuple,
        "return_corr_to_baseline_float": _correlation_float(variant_return_ser, baseline_return_ser),
        "downside_beta_to_spx_float": _downside_beta_float(
            strategy_return_ser=variant_return_ser,
            benchmark_return_ser=benchmark_return_ser,
            benchmark_quantile_float=0.10,
        ),
        "active_day_count_int": int(variant_active_bool_ser.sum()),
        "active_day_pct_float": float(variant_active_bool_ser.mean() * 100.0),
        "ann_return_pct_float": variant_summary_dict.get("ann_return_pct_float"),
        "volatility_ann_pct_float": variant_summary_dict.get("volatility_ann_pct_float"),
        "sharpe_float": variant_summary_dict.get("sharpe_float"),
        "max_drawdown_pct_float": variant_summary_dict.get("max_drawdown_pct_float"),
        "mar_float": variant_summary_dict.get("mar_float"),
        "turnover_ann_pct_float": variant_summary_dict.get("turnover_ann_pct_float"),
        "cost_drag_ann_pct_float": variant_summary_dict.get("cost_drag_ann_pct_float"),
        "trade_count_int": variant_summary_dict.get("trade_count_int"),
        "oos_sharpe_float": variant_summary_dict.get("oos_sharpe_float"),
        "oos_max_drawdown_pct_float": variant_summary_dict.get("oos_max_drawdown_pct_float"),
        "in_sample_sharpe_float": variant_summary_dict.get("in_sample_sharpe_float"),
        "delta_full_ann_return_pct_float": _safe_delta_float(
            variant_summary_dict.get("ann_return_pct_float"),
            baseline_summary_dict.get("ann_return_pct_float"),
        ),
        "delta_full_sharpe_float": _safe_delta_float(
            variant_summary_dict.get("sharpe_float"),
            baseline_summary_dict.get("sharpe_float"),
        ),
        "delta_full_max_drawdown_pct_float": _safe_delta_float(
            variant_summary_dict.get("max_drawdown_pct_float"),
            baseline_summary_dict.get("max_drawdown_pct_float"),
        ),
        "delta_oos_sharpe_float": _safe_delta_float(
            variant_summary_dict.get("oos_sharpe_float"),
            baseline_summary_dict.get("oos_sharpe_float"),
        ),
        "delta_oos_max_drawdown_pct_float": _safe_delta_float(
            variant_summary_dict.get("oos_max_drawdown_pct_float"),
            baseline_summary_dict.get("oos_max_drawdown_pct_float"),
        ),
        "delta_trade_count_int": _safe_delta_float(
            variant_summary_dict.get("trade_count_int"),
            baseline_summary_dict.get("trade_count_int"),
        ),
        "delta_cost_drag_ann_pct_float": _safe_delta_float(
            variant_summary_dict.get("cost_drag_ann_pct_float"),
            baseline_summary_dict.get("cost_drag_ann_pct_float"),
        ),
    }
    diagnostic_row_dict.update(tail_metric_dict)
    accept_bool, reject_reason_str = evaluate_combination_acceptance_rule(diagnostic_row_dict)
    diagnostic_row_dict["accept_bool"] = bool(accept_bool)
    diagnostic_row_dict["reject_reason_str"] = reject_reason_str
    stress_pass_bool, stress_reject_reason_str = evaluate_combination_stress_rule(diagnostic_row_dict)
    diagnostic_row_dict["stress_pass_bool"] = bool(stress_pass_bool)
    diagnostic_row_dict["stress_reject_reason_str"] = stress_reject_reason_str
    return diagnostic_row_dict


def evaluate_combination_acceptance_rule(diagnostic_row_dict: dict[str, object]) -> tuple[bool, str]:
    reject_reason_list: list[str] = []
    delta_oos_sharpe_float = _safe_float(diagnostic_row_dict.get("delta_oos_sharpe_float"))
    delta_oos_drawdown_float = _safe_float(diagnostic_row_dict.get("delta_oos_max_drawdown_pct_float"))
    delta_full_sharpe_float = _safe_float(diagnostic_row_dict.get("delta_full_sharpe_float"))
    delta_cost_drag_float = _safe_float(diagnostic_row_dict.get("delta_cost_drag_ann_pct_float"))
    base_tail_delta_float = _safe_float(diagnostic_row_dict.get("base_tail_delta_mean_return_pct_float"))
    market_tail_delta_float = _safe_float(diagnostic_row_dict.get("market_tail_delta_mean_return_pct_float"))

    if (
        not np.isfinite(delta_oos_sharpe_float)
        or delta_oos_sharpe_float < COMBINATION_ACCEPTANCE_RULE_DICT["min_delta_oos_sharpe_float"]
    ):
        reject_reason_list.append("oos_sharpe_not_better")
    if (
        not np.isfinite(delta_oos_drawdown_float)
        or delta_oos_drawdown_float < COMBINATION_ACCEPTANCE_RULE_DICT["min_delta_oos_max_drawdown_pct_float"]
    ):
        reject_reason_list.append("oos_drawdown_worse_than_limit")
    if (
        not np.isfinite(delta_full_sharpe_float)
        or delta_full_sharpe_float < COMBINATION_ACCEPTANCE_RULE_DICT["min_delta_full_sharpe_float"]
    ):
        reject_reason_list.append("full_sharpe_degraded_too_much")
    if (
        not np.isfinite(delta_cost_drag_float)
        or delta_cost_drag_float > COMBINATION_ACCEPTANCE_RULE_DICT["max_delta_cost_drag_ann_pct_float"]
    ):
        reject_reason_list.append("cost_drag_too_high")
    if (
        not np.isfinite(base_tail_delta_float)
        or base_tail_delta_float < COMBINATION_ACCEPTANCE_RULE_DICT["min_base_tail_delta_mean_return_pct_float"]
    ):
        reject_reason_list.append("base_tail_not_helpful")
    if (
        not np.isfinite(market_tail_delta_float)
        or market_tail_delta_float < COMBINATION_ACCEPTANCE_RULE_DICT["min_market_tail_delta_mean_return_pct_float"]
    ):
        reject_reason_list.append("market_tail_not_helpful")
    return len(reject_reason_list) == 0, ";".join(reject_reason_list)


def evaluate_combination_stress_rule(diagnostic_row_dict: dict[str, object]) -> tuple[bool, str]:
    reject_reason_list: list[str] = []
    if not bool(diagnostic_row_dict.get("accept_bool", False)):
        reject_reason_list.append("average_rule_failed")

    base_tail_delta_float = _safe_float(diagnostic_row_dict.get("base_tail_delta_mean_return_pct_float"))
    market_tail_delta_float = _safe_float(diagnostic_row_dict.get("market_tail_delta_mean_return_pct_float"))
    base_tail_active_float = _safe_float(diagnostic_row_dict.get("base_tail_candidate_active_pct_float"))
    market_tail_active_float = _safe_float(diagnostic_row_dict.get("market_tail_candidate_active_pct_float"))

    if (
        not np.isfinite(base_tail_delta_float)
        or base_tail_delta_float < COMBINATION_STRESS_RULE_DICT["min_base_tail_delta_mean_return_pct_float"]
    ):
        reject_reason_list.append("base_tail_not_helpful")
    if (
        not np.isfinite(market_tail_delta_float)
        or market_tail_delta_float < COMBINATION_STRESS_RULE_DICT["min_market_tail_delta_mean_return_pct_float"]
    ):
        reject_reason_list.append("market_tail_not_helpful")
    if (
        not np.isfinite(base_tail_active_float)
        or base_tail_active_float < COMBINATION_STRESS_RULE_DICT["min_base_tail_active_pct_float"]
    ):
        reject_reason_list.append("base_tail_variant_inactive")
    if (
        not np.isfinite(market_tail_active_float)
        or market_tail_active_float < COMBINATION_STRESS_RULE_DICT["min_market_tail_active_pct_float"]
    ):
        reject_reason_list.append("market_tail_variant_inactive")
    return len(reject_reason_list) == 0, ";".join(reject_reason_list)


def add_composite_score_df(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    scored_df = diagnostic_df.copy()
    rank_component_df = pd.DataFrame(index=scored_df.index)
    for metric_name_str, weight_float in COMPOSITE_WEIGHT_DICT.items():
        metric_ser = pd.to_numeric(scored_df[metric_name_str], errors="coerce")
        ascending_bool = weight_float > 0.0
        filled_metric_ser = metric_ser.fillna(metric_ser.median())
        if filled_metric_ser.nunique(dropna=True) <= 1:
            rank_component_df[f"{metric_name_str}_rank_component_float"] = 0.5
            continue
        rank_component_df[f"{metric_name_str}_rank_component_float"] = filled_metric_ser.rank(
            pct=True,
            ascending=ascending_bool,
        )

    score_ser = pd.Series(0.0, index=scored_df.index, dtype=float)
    total_weight_float = 0.0
    for metric_name_str, weight_float in COMPOSITE_WEIGHT_DICT.items():
        component_ser = rank_component_df[f"{metric_name_str}_rank_component_float"]
        abs_weight_float = abs(float(weight_float))
        score_ser = score_ser + component_ser * abs_weight_float
        total_weight_float += abs_weight_float

    if total_weight_float > 0.0:
        score_ser = score_ser / total_weight_float

    scored_df["composite_score_float"] = score_ser.astype(float)
    scored_df = scored_df.sort_values(
        by=[
            "accept_bool",
            "stress_pass_bool",
            "composite_score_float",
            "delta_oos_sharpe_float",
            "base_tail_delta_mean_return_pct_float",
            "addition_count_int",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    scored_df["leaderboard_rank_int"] = np.arange(1, len(scored_df) + 1)
    return scored_df


def build_asset_recommendation_df(
    candidate_manifest_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
) -> pd.DataFrame:
    candidate_info_df = candidate_manifest_df.set_index("symbol_str", drop=False)
    active_leaderboard_df = leaderboard_df.loc[
        leaderboard_df["addition_count_int"].gt(0)
        & ~leaderboard_df["variant_kind_str"].astype(str).str.startswith("full_universe")
    ].copy()
    top20_rank_set = set(active_leaderboard_df.head(20)["leaderboard_rank_int"].astype(int).tolist())
    top50_rank_set = set(active_leaderboard_df.head(50)["leaderboard_rank_int"].astype(int).tolist())
    row_dict_list: list[dict[str, object]] = []

    for symbol_str, info_ser in candidate_info_df.iterrows():
        contains_bool_ser = active_leaderboard_df["addition_tuple_str"].astype(str).str.split(",").apply(
            lambda addition_list: symbol_str in addition_list
        )
        asset_variant_df = active_leaderboard_df.loc[contains_bool_ser].copy()
        single_df = asset_variant_df.loc[asset_variant_df["addition_tuple_str"].eq(symbol_str)]
        best_df = asset_variant_df.sort_values("leaderboard_rank_int").head(1)
        pair_df = asset_variant_df.loc[asset_variant_df["addition_count_int"].eq(2)]
        best_pair_df = pair_df.sort_values("leaderboard_rank_int").head(1)
        top20_count_int = int(asset_variant_df["leaderboard_rank_int"].astype(int).isin(top20_rank_set).sum())
        top50_count_int = int(asset_variant_df["leaderboard_rank_int"].astype(int).isin(top50_rank_set).sum())
        pass_count_int = int(asset_variant_df["accept_bool"].astype(bool).sum())
        stress_pass_count_int = int(asset_variant_df["stress_pass_bool"].astype(bool).sum())

        best_row_ser = best_df.iloc[0] if len(best_df) > 0 else pd.Series(dtype=object)
        single_row_ser = single_df.iloc[0] if len(single_df) > 0 else pd.Series(dtype=object)
        best_pair_row_ser = best_pair_df.iloc[0] if len(best_pair_df) > 0 else pd.Series(dtype=object)
        tier_str, reason_str = _asset_tier_reason_str(
            single_row_ser=single_row_ser,
            best_row_ser=best_row_ser,
            top20_count_int=top20_count_int,
            top50_count_int=top50_count_int,
            pass_count_int=pass_count_int,
            stress_pass_count_int=stress_pass_count_int,
        )
        row_dict_list.append(
            {
                "symbol_str": symbol_str,
                "source_universe_str": info_ser["source_universe_str"],
                "bucket_str": info_ser["bucket_str"],
                "description_str": info_ser["description_str"],
                "recommendation_tier_str": tier_str,
                "recommendation_reason_str": reason_str,
                "single_accept_bool": bool(single_row_ser.get("accept_bool", False)),
                "single_stress_pass_bool": bool(single_row_ser.get("stress_pass_bool", False)),
                "single_delta_oos_sharpe_float": single_row_ser.get("delta_oos_sharpe_float", np.nan),
                "single_base_tail_delta_mean_return_pct_float": single_row_ser.get(
                    "base_tail_delta_mean_return_pct_float",
                    np.nan,
                ),
                "single_market_tail_delta_mean_return_pct_float": single_row_ser.get(
                    "market_tail_delta_mean_return_pct_float",
                    np.nan,
                ),
                "best_variant_label_str": best_row_ser.get("variant_label_str", ""),
                "best_variant_rank_int": best_row_ser.get("leaderboard_rank_int", np.nan),
                "best_variant_score_float": best_row_ser.get("composite_score_float", np.nan),
                "best_pair_label_str": best_pair_row_ser.get("variant_label_str", ""),
                "best_pair_rank_int": best_pair_row_ser.get("leaderboard_rank_int", np.nan),
                "top20_count_int": top20_count_int,
                "top50_count_int": top50_count_int,
                "accept_variant_count_int": pass_count_int,
                "stress_pass_variant_count_int": stress_pass_count_int,
                "tested_variant_count_int": int(len(asset_variant_df)),
            }
        )

    asset_df = pd.DataFrame(row_dict_list)
    tier_order_dict = {
        "core_candidate": 0,
        "combo_candidate": 1,
        "defensive_candidate": 2,
        "watchlist": 3,
        "reject_for_now": 4,
    }
    asset_df["tier_sort_int"] = asset_df["recommendation_tier_str"].map(tier_order_dict).fillna(9).astype(int)
    return asset_df.sort_values(
        by=["tier_sort_int", "best_variant_rank_int", "top20_count_int", "single_delta_oos_sharpe_float"],
        ascending=[True, True, False, False],
    ).drop(columns=["tier_sort_int"])


def _asset_tier_reason_str(
    single_row_ser: pd.Series,
    best_row_ser: pd.Series,
    top20_count_int: int,
    top50_count_int: int,
    pass_count_int: int,
    stress_pass_count_int: int,
) -> tuple[str, str]:
    single_accept_bool = bool(single_row_ser.get("accept_bool", False))
    single_stress_bool = bool(single_row_ser.get("stress_pass_bool", False))
    single_oos_sharpe_float = _safe_float(single_row_ser.get("delta_oos_sharpe_float"))
    single_base_tail_delta_float = _safe_float(single_row_ser.get("base_tail_delta_mean_return_pct_float"))
    best_rank_float = _safe_float(best_row_ser.get("leaderboard_rank_int"))

    if single_accept_bool and single_stress_bool and top20_count_int > 0:
        return (
            "core_candidate",
            "Passed as a single addition, passed stress, and appears in the top combination set.",
        )
    if top20_count_int > 0 and pass_count_int > 0 and stress_pass_count_int > 0:
        return (
            "combo_candidate",
            "Useful mainly inside combinations; at least one top-20 accepted stress variant contains it.",
        )
    if np.isfinite(single_base_tail_delta_float) and single_base_tail_delta_float > 0.25 and not single_accept_bool:
        return (
            "defensive_candidate",
            "Tail help exists, but average/OOS acceptance is not clean enough for a default asset.",
        )
    if top50_count_int > 0 or (np.isfinite(single_oos_sharpe_float) and single_oos_sharpe_float > 0.0):
        return (
            "watchlist",
            "Some evidence is positive, but it did not pass enough fixed rules to promote.",
        )
    if np.isfinite(best_rank_float):
        return ("reject_for_now", "Did not rank well under the fixed combination score.")
    return ("reject_for_now", "No usable evidence in the tested variants.")


def _leaderboard_column_list(leaderboard_df: pd.DataFrame) -> list[str]:
    preferred_column_list = [
        "leaderboard_rank_int",
        "variant_label_str",
        "variant_kind_str",
        "addition_count_int",
        "accept_bool",
        "stress_pass_bool",
        "composite_score_float",
        "ann_return_pct_float",
        "sharpe_float",
        "max_drawdown_pct_float",
        "oos_sharpe_float",
        "delta_oos_sharpe_float",
        "delta_full_sharpe_float",
        "base_tail_delta_mean_return_pct_float",
        "market_tail_delta_mean_return_pct_float",
        "cost_drag_ann_pct_float",
        "delta_cost_drag_ann_pct_float",
    ]
    return [column_str for column_str in preferred_column_list if column_str in leaderboard_df.columns]


def _asset_table_column_list(asset_recommendation_df: pd.DataFrame) -> list[str]:
    preferred_column_list = [
        "symbol_str",
        "recommendation_tier_str",
        "source_universe_str",
        "description_str",
        "single_accept_bool",
        "single_stress_pass_bool",
        "single_delta_oos_sharpe_float",
        "single_base_tail_delta_mean_return_pct_float",
        "single_market_tail_delta_mean_return_pct_float",
        "best_variant_label_str",
        "best_variant_rank_int",
        "best_pair_label_str",
        "top20_count_int",
        "top50_count_int",
        "recommendation_reason_str",
    ]
    return [column_str for column_str in preferred_column_list if column_str in asset_recommendation_df.columns]


def _write_recommendations_md(
    output_path: Path,
    combination_manifest_df: pd.DataFrame,
    baseline_summary_dict: dict[str, object],
    leaderboard_df: pd.DataFrame,
    asset_recommendation_df: pd.DataFrame,
    pair_scope_str: str,
) -> None:
    top_leaderboard_df = leaderboard_df.head(20).copy()
    top_asset_df = asset_recommendation_df.head(25).copy()
    accepted_count_int = int(leaderboard_df["accept_bool"].astype(bool).sum())
    stress_pass_count_int = int(leaderboard_df["stress_pass_bool"].astype(bool).sum())
    top_variant_label_str = str(top_leaderboard_df.iloc[0]["variant_label_str"]) if len(top_leaderboard_df) else ""
    top_symbols_str = str(top_leaderboard_df.iloc[0]["symbol_tuple_str"]) if len(top_leaderboard_df) else ""
    pair_kie_xlre_df = leaderboard_df.loc[leaderboard_df["addition_tuple_str"].eq("KIE,XLRE")]
    if len(pair_kie_xlre_df) == 0:
        kie_xlre_text_str = "`Base+KIE+XLRE` was not present in the manifest."
    else:
        pair_row_ser = pair_kie_xlre_df.iloc[0]
        kie_xlre_text_str = (
            f"`Base+KIE+XLRE` rank `{int(pair_row_ser['leaderboard_rank_int'])}`, "
            f"accepted `{bool(pair_row_ser['accept_bool'])}`, stress pass `{bool(pair_row_ser['stress_pass_bool'])}`, "
            f"OOS Sharpe delta `{float(pair_row_ser['delta_oos_sharpe_float']):.3f}`, "
            f"base-tail delta `{float(pair_row_ser['base_tail_delta_mean_return_pct_float']):.3f}%`."
        )

    recommendations_md_str = f"""# Sector Dispersion Combination Universe Study

## TL;DR

- Research-only; no live/release wiring.
- Execution semantics unchanged: `signal T -> Open T+1`.
- Costs unchanged from the base strategy.
- Tested variants: `{len(combination_manifest_df)}` total rows.
- Accepted variants: `{accepted_count_int}`.
- Stress-pass variants: `{stress_pass_count_int}`.
- Current top fixed-score variant: `{top_variant_label_str}`.
- Top symbol tuple: `{top_symbols_str}`.
- KIE + XLRE check: {kie_xlre_text_str}

## Search Space

- Baseline: original basket `{", ".join(ORIGINAL_SYMBOL_TUPLE)}`.
- Singles: every predeclared Universe A/B/C candidate.
- Pair scope: `{pair_scope_str}`.
- Pairs: challenger-pool pairs by default; rerun with `--pair-scope all` for every candidate pair.
- Triples: only from the fixed challenger pool `{", ".join(DEFAULT_TRIPLE_POOL_SYMBOL_TUPLE)}`.
- Full universes A/B/C are diagnostic rows, not promotion candidates.
- In-sample ends `{IN_SAMPLE_END_TS.date().isoformat()}`.
- OOS starts `{OUT_OF_SAMPLE_START_TS.date().isoformat()}`.
- Combination stress-pass intentionally does not require low correlation to the baseline because every tested
  combination contains the baseline assets.

## Baseline

- Annual return [%]: `{baseline_summary_dict.get("ann_return_pct_float"):.3f}`
- Sharpe: `{baseline_summary_dict.get("sharpe_float"):.3f}`
- Max drawdown [%]: `{baseline_summary_dict.get("max_drawdown_pct_float"):.3f}`
- OOS Sharpe: `{baseline_summary_dict.get("oos_sharpe_float"):.3f}`
- OOS max drawdown [%]: `{baseline_summary_dict.get("oos_max_drawdown_pct_float"):.3f}`

## Top Variants

{_markdown_table_str(top_leaderboard_df, _leaderboard_column_list(top_leaderboard_df), max_rows_int=20)}

## Asset Recommendation Table

{_markdown_table_str(top_asset_df, _asset_table_column_list(top_asset_df), max_rows_int=25)}

## Caveat

This is a broad research sweep. The winner is a candidate for follow-up, not proof of edge.
The table reports the trial count so the search cost is visible.
"""
    (output_path / "recommendations.md").write_text(recommendations_md_str, encoding="utf-8")


def run_combination_universe_study(
    output_dir_str: str = "results",
    end_date_str: str | None = None,
    show_progress_bool: bool = False,
    include_pairs_bool: bool = True,
    include_triples_bool: bool = True,
    include_full_universes_bool: bool = True,
    pair_scope_str: str = "challenger_pool",
    max_variants_int: int | None = None,
) -> Path:
    candidate_manifest_df = build_candidate_manifest_df()
    combination_manifest_df = build_combination_manifest_df(
        include_pairs_bool=include_pairs_bool,
        include_triples_bool=include_triples_bool,
        include_full_universes_bool=include_full_universes_bool,
        pair_scope_str=pair_scope_str,
    )
    if max_variants_int is not None:
        combination_manifest_df = combination_manifest_df.head(int(max_variants_int)).copy()

    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str="strategy_mr_sector_dispersion_ibs",
        analysis_type_str="combination_universe_study",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)
    candidate_manifest_df.to_csv(output_path / "candidate_manifest.csv", index=False)
    combination_manifest_df.to_csv(output_path / "combination_manifest.csv", index=False)

    all_candidate_symbol_tuple = tuple(candidate_manifest_df["symbol_str"].astype(str).tolist())
    all_symbol_tuple = tuple(dict.fromkeys(ORIGINAL_SYMBOL_TUPLE + all_candidate_symbol_tuple))
    base_config_obj = replace(
        DEFAULT_CONFIG,
        symbol_tuple=all_symbol_tuple,
        universe_name_str="original",
        end_date_str=end_date_str,
    )
    pricing_data_df = get_sector_dispersion_ibs_data(config_obj=base_config_obj)
    benchmark_return_ser = _benchmark_return_ser(
        pricing_data_df=pricing_data_df,
        benchmark_symbol_str=base_config_obj.benchmark_symbol_str,
    )
    baseline_strategy_obj = _run_strategy_variant(
        strategy_name_str="strategy_mr_sector_dispersion_ibs_combination_base",
        symbol_tuple=ORIGINAL_SYMBOL_TUPLE,
        base_config_obj=base_config_obj,
        pricing_data_df=pricing_data_df,
        show_progress_bool=show_progress_bool,
    )
    baseline_summary_dict = _strategy_summary_row_dict(
        strategy_obj=baseline_strategy_obj,
        variant_kind_str="baseline",
        candidate_symbol_str=None,
        bucket_str=None,
    )
    pd.DataFrame([baseline_summary_dict]).to_csv(output_path / "baseline_summary.csv", index=False)

    summary_row_dict_list: list[dict[str, object]] = []
    diagnostic_row_dict_list: list[dict[str, object]] = []
    for row_index_int, manifest_row_ser in combination_manifest_df.iterrows():
        addition_tuple_str = str(manifest_row_ser["addition_tuple_str"])
        addition_tuple = tuple(filter(None, addition_tuple_str.split(",")))
        symbol_tuple = tuple(dict.fromkeys(ORIGINAL_SYMBOL_TUPLE + addition_tuple))
        variant_kind_str = str(manifest_row_ser["variant_kind_str"])
        strategy_name_str = _variant_name_str(addition_tuple=addition_tuple, variant_kind_str=variant_kind_str)
        print(
            f"Running {row_index_int + 1}/{len(combination_manifest_df)} {manifest_row_ser['variant_label_str']}...",
            flush=True,
        )
        if len(addition_tuple) == 0:
            variant_strategy_obj = baseline_strategy_obj
        else:
            variant_strategy_obj = _run_strategy_variant(
                strategy_name_str=strategy_name_str,
                symbol_tuple=symbol_tuple,
                base_config_obj=base_config_obj,
                pricing_data_df=pricing_data_df,
                show_progress_bool=show_progress_bool,
            )
        variant_summary_dict = _strategy_summary_row_dict(
            strategy_obj=variant_strategy_obj,
            variant_kind_str=variant_kind_str,
            candidate_symbol_str=addition_tuple_str,
            bucket_str=None,
        )
        variant_summary_dict["variant_label_str"] = str(manifest_row_ser["variant_label_str"])
        variant_summary_dict["addition_count_int"] = int(manifest_row_ser["addition_count_int"])
        variant_summary_dict["addition_tuple_str"] = addition_tuple_str
        summary_row_dict_list.append(variant_summary_dict)
        diagnostic_row_dict_list.append(
            _variant_diagnostic_row_dict(
                manifest_row_ser=manifest_row_ser,
                baseline_strategy_obj=baseline_strategy_obj,
                variant_strategy_obj=variant_strategy_obj,
                benchmark_return_ser=benchmark_return_ser,
                baseline_summary_dict=baseline_summary_dict,
                variant_summary_dict=variant_summary_dict,
            )
        )

        if (row_index_int + 1) % 25 == 0 or row_index_int + 1 == len(combination_manifest_df):
            partial_summary_df = pd.DataFrame(summary_row_dict_list)
            partial_diagnostic_df = pd.DataFrame(diagnostic_row_dict_list)
            partial_summary_df.to_csv(output_path / "combination_summary.partial.csv", index=False)
            partial_diagnostic_df.to_csv(output_path / "combination_diagnostics.partial.csv", index=False)

    summary_df = pd.DataFrame(summary_row_dict_list)
    diagnostic_df = pd.DataFrame(diagnostic_row_dict_list)
    leaderboard_df = add_composite_score_df(diagnostic_df)
    asset_recommendation_df = build_asset_recommendation_df(
        candidate_manifest_df=candidate_manifest_df,
        leaderboard_df=leaderboard_df,
    )

    summary_df.to_csv(output_path / "combination_summary.csv", index=False)
    diagnostic_df.to_csv(output_path / "combination_diagnostics.csv", index=False)
    leaderboard_df.to_csv(output_path / "leaderboard.csv", index=False)
    asset_recommendation_df.to_csv(output_path / "asset_recommendations.csv", index=False)
    _write_recommendations_md(
        output_path=output_path,
        combination_manifest_df=combination_manifest_df,
        baseline_summary_dict=baseline_summary_dict,
        leaderboard_df=leaderboard_df,
        asset_recommendation_df=asset_recommendation_df,
        pair_scope_str=pair_scope_str,
    )

    metadata_dict = {
        "strategy_id_str": "strategy_mr_sector_dispersion_ibs",
        "analysis_type_str": "combination_universe_study",
        "research_only_bool": True,
        "output_path_str": str(output_path.resolve()),
        "candidate_count_int": int(len(candidate_manifest_df)),
        "variant_count_int": int(len(combination_manifest_df)),
        "pair_scope_str": pair_scope_str,
        "all_symbol_tuple": all_symbol_tuple,
        "baseline_symbol_tuple": ORIGINAL_SYMBOL_TUPLE,
        "default_triple_pool_symbol_tuple": DEFAULT_TRIPLE_POOL_SYMBOL_TUPLE,
        "benchmark_symbol_str": base_config_obj.benchmark_symbol_str,
        "history_start_date_str": base_config_obj.history_start_date_str,
        "backtest_start_date_str": base_config_obj.backtest_start_date_str,
        "end_date_str": end_date_str,
        "in_sample_end_date_str": IN_SAMPLE_END_TS.date().isoformat(),
        "out_of_sample_start_date_str": OUT_OF_SAMPLE_START_TS.date().isoformat(),
        "acceptance_rule_dict": ACCEPTANCE_RULE_DICT,
        "stress_rule_dict": STRESS_RULE_DICT,
        "combination_acceptance_rule_dict": COMBINATION_ACCEPTANCE_RULE_DICT,
        "combination_stress_rule_dict": COMBINATION_STRESS_RULE_DICT,
        "composite_weight_dict": COMPOSITE_WEIGHT_DICT,
        "execution_mapping_str": "signal daily bar T -> Open T+1",
        "notes_str": (
            "Broad research sweep. Singles are exhaustive over the frozen candidate manifest; "
            "pairs follow pair_scope_str; triples are explicitly second-stage tests from a fixed challenger pool."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True, default=_json_default_obj),
        encoding="utf-8",
    )

    print(f"Saved combination universe study to {output_path.resolve()}", flush=True)
    return output_path


def parse_args(argv_list: list[str] | None = None) -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(description="Run sector-dispersion IBS combination universe study.")
    parser_obj.add_argument("--output-dir", default="results", help="Root output directory.")
    parser_obj.add_argument("--end-date", default=None, help="Optional inclusive Norgate end date.")
    parser_obj.add_argument("--show-progress", action="store_true", help="Show Vanilla progress bars.")
    parser_obj.add_argument("--no-pairs", action="store_true", help="Skip exhaustive pair additions.")
    parser_obj.add_argument("--no-triples", action="store_true", help="Skip selected triple additions.")
    parser_obj.add_argument("--no-full-universes", action="store_true", help="Skip full Universe A/B/C diagnostics.")
    parser_obj.add_argument(
        "--pair-scope",
        choices=("challenger_pool", "all"),
        default="challenger_pool",
        help="Pair grid to run. Use 'all' for every Universe A/B/C pair; default is the fixed challenger pool.",
    )
    parser_obj.add_argument("--max-variants", type=int, default=None, help="Optional smoke-test cap.")
    return parser_obj.parse_args(argv_list)


def main(argv_list: list[str] | None = None) -> None:
    args_obj = parse_args(argv_list)
    run_combination_universe_study(
        output_dir_str=args_obj.output_dir,
        end_date_str=args_obj.end_date,
        show_progress_bool=bool(args_obj.show_progress),
        include_pairs_bool=not bool(args_obj.no_pairs),
        include_triples_bool=not bool(args_obj.no_triples),
        include_full_universes_bool=not bool(args_obj.no_full_universes),
        pair_scope_str=str(args_obj.pair_scope),
        max_variants_int=args_obj.max_variants,
    )


if __name__ == "__main__":
    main()
