"""Engine parity for non-default cells of the NDX parameter-robustness study (PREREG section 6; research only).

The real repo engine (`run_daily`) executes the replica's target weights through a subclass of the live strategy
class: only `get_target_weight_ser` and the rebalance schedule are replaced. Everything else (share sizing on the
decision close, next-open fills, slippage, commissions, dividend ledger, missing-price liquidation) is the
engine's own code. The engine daily returns are then compared with the replica's.

    uv run python scripts/research/run_ndx_param_robustness_engine_parity.py "<cell key>"
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ndx_param_robustness_core as core  # noqa: E402

from alpha.engine.backtest import run_daily  # noqa: E402
from strategies.momentum.strategy_mo_atr_normalized_ndx import configure_total_return_benchmark_provenance  # noqa: E402
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (  # noqa: E402
    DEFAULT_CONFIG,
    VxnScaledAtrNormalizedNdxStrategy,
    get_vxn_scaled_atr_normalized_ndx_data,
)


class ReplicaTargetStrategy(VxnScaledAtrNormalizedNdxStrategy):
    """Live strategy class whose target weights come from the replica (already VXN-scaled)."""

    def __init__(self, *args, target_weight_by_decision_dict: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_weight_by_decision_dict = target_weight_by_decision_dict

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        # *** CRITICAL *** keyed by previous_bar = the decision close; iterate() already asserts the schedule.
        return self.target_weight_by_decision_dict.get(pd.Timestamp(self.previous_bar), pd.Series(dtype=float))


def find_cell(key_str: str) -> core.Cell:
    for cell in core.all_cell_list():
        if cell.key_str == key_str:
            return cell
    raise KeyError(key_str)


def main() -> None:
    key_str = sys.argv[1]
    cell = find_cell(key_str)
    start_float = time.perf_counter()
    universe_dict = core.load_universe("NDX")
    feature_obj = core.FeatureBook(universe_dict)
    target_list = core.build_target_list(feature_obj, cell)
    replica_dict = core.simulate(universe_dict, target_list)
    date_index = universe_dict["date_index"]
    symbol_list = universe_dict["symbol_list"]
    target_weight_by_decision_dict = {
        date_index[t["decision_pos"]]: pd.Series(t["weight_vec"], index=[symbol_list[i] for i in t["symbol_idx_vec"]], dtype=float)
        for t in target_list
    }
    schedule_df = pd.DataFrame(
        {"decision_date_ts": [date_index[t["decision_pos"]] for t in target_list]},
        index=pd.DatetimeIndex([date_index[t["execution_pos"]] for t in target_list], name="execution_date_ts"),
    )

    config_obj = DEFAULT_CONFIG
    pricing_data_df, universe_df, _repo_schedule_df, vxn_scale_signal_df = get_vxn_scaled_atr_normalized_ndx_data(
        config_obj, include_total_return_benchmark_bool=True
    )
    strategy_obj = ReplicaTargetStrategy(
        name="ndx_param_robustness_engine_parity",
        benchmarks=[config_obj.performance_benchmark_symbol_str],
        rebalance_schedule_df=schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_month_int=config_obj.lookback_month_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
        target_weight_by_decision_dict=target_weight_by_decision_dict,
    )
    strategy_obj.universe_df = universe_df
    configure_total_return_benchmark_provenance(strategy_obj=strategy_obj, config_obj=config_obj)
    # *** CRITICAL *** same executable calendar as the repo's run_variant; fills at the next open.
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    run_daily(strategy_obj, pricing_data_df, calendar=calendar_idx, show_progress=False,
              show_signal_progress_bool=False, audit_override_bool=None)

    engine_nav_ser = strategy_obj.results["total_value"].astype(float)
    engine_nav_ser.index = pd.to_datetime(engine_nav_ser.index)
    engine_ret_ser = engine_nav_ser.pct_change().iloc[1:]
    replica_ret_ser = replica_dict["return_ser"]
    common_index = engine_ret_ser.index.intersection(replica_ret_ser.index)
    common_index = common_index[common_index <= pd.Timestamp("2026-07-24")]
    e_ser, r_ser = engine_ret_ser.reindex(common_index), replica_ret_ser.reindex(common_index)

    def cagr(return_ser: pd.Series) -> float:
        return float((1 + return_ser).prod() ** (252.0 / len(return_ser)) - 1)

    report_dict = {
        "cell_key_str": key_str,
        "daily_return_corr_float": float(np.corrcoef(e_ser, r_ser)[0, 1]),
        "max_abs_daily_diff_float": float((e_ser - r_ser).abs().max()),
        "cagr_engine_float": cagr(e_ser),
        "cagr_replica_float": cagr(r_ser),
        "final_nav_engine_float": float(engine_nav_ser.iloc[-1]),
        "final_nav_replica_float": float(replica_dict["total_ser"].iloc[-1]),
        "seconds_float": round(time.perf_counter() - start_float, 1),
    }
    report_dict["passed_bool"] = bool(
        report_dict["daily_return_corr_float"] >= 0.9999
        and abs(report_dict["cagr_engine_float"] - report_dict["cagr_replica_float"]) <= 0.0005
    )
    safe_str = "".join(c if c.isalnum() else "_" for c in key_str)
    parity_dir_path = core.RESULTS_DIR_PATH / "engine_parity"
    parity_dir_path.mkdir(parents=True, exist_ok=True)
    (parity_dir_path / f"{safe_str}.json").write_text(json.dumps(report_dict, indent=2))
    print(json.dumps(report_dict, indent=2))


if __name__ == "__main__":
    main()
