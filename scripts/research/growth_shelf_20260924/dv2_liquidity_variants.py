"""DV2 with a liquidity rule: two pre-declared variants plus a reproduction check (owner request, 2026-09-25).

Why: in the growth capacity study DV2 was the pod that capped every book holding it (thin S&P 500 names such as NWS
in the auction; 0.50%/yr at $25M in ladder_4 vs 0.16% for HPI at the same weight). DV2 ranks its candidates by NATR,
i.e. it buys the MOST volatile dips first; HPI ranks by dollar turnover and already has a liquidity floor in code.

PRE-DECLARED variants (research only; strategies/dv2/strategy_mr_dv2.py is not modified):
  dv2_check          the WIRED DVO2Strategy unchanged, run through this runner; must reproduce the fund-menu dv2 source
  dv2_turnover_rank  same entry and exit rules; candidates ranked by 63-day mean dollar volume (raw close x volume),
                     highest first, instead of NATR
  dv2_liq_floor      HPI's own liquidity rule (strategies/hpi/stateful_long.py, raw_price_5_adv63_above_median): raw
                     price > $5 and 63-day mean dollar volume above the median of that day's S&P 500 members; NATR
                     ranking kept
Settings exactly as the fund-menu sources: $1M, start 2000-01-03, end 2026-08-19, next-open fills, same costs.
No other variant will be run; the evaluation rule is written in dv2_liquidity_eval.py before these results are read.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

import pandas as pd  # noqa: E402

from alpha.engine.backtest import run_daily  # noqa: E402
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR, build_index_constituent_matrix  # noqa: E402
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy, default_trade_id_int, get_asof_universe_symbol_list, get_prices  # noqa: E402

OUT_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "growth_shelf_20260924" / "dv2_sources"
REFERENCE_CAPITAL_FLOAT = 1_000_000.0
START_DATE_STR = "2000-01-03"
END_DATE_STR = "2026-08-19"
ADV_WINDOW_INT = 63
RAW_PRICE_MIN_FLOAT = 5.0


def add_liquidity_features(signal_df: pd.DataFrame, pricing_df: pd.DataFrame) -> pd.DataFrame:
    """ADV63_T = mean of raw close x volume over [T-62, T] (HPI's definition); known after Close_T for Open_(T+1)."""
    feature_dict = {}
    for symbol_str in pricing_df.columns.get_level_values(0).unique():
        if str(symbol_str).startswith("$") or (symbol_str, "Unadjusted Close") not in pricing_df.columns:
            continue
        raw_close_ser = pricing_df[(symbol_str, "Unadjusted Close")].astype(float)
        dollar_volume_ser = raw_close_ser * pricing_df[(symbol_str, "Volume")].astype(float)
        # *** CRITICAL*** trailing window ending at T only; the engine reads this row for the NEXT open.
        feature_dict[(symbol_str, "adv_63")] = dollar_volume_ser.rolling(ADV_WINDOW_INT, min_periods=ADV_WINDOW_INT).mean()
        feature_dict[(symbol_str, "raw_price")] = raw_close_ser
    return pd.concat([signal_df, pd.DataFrame(feature_dict, index=signal_df.index)], axis=1).copy()


def entry_candidate_df(close_row_ser: pd.Series) -> pd.DataFrame:
    frame_df = close_row_ser.unstack().dropna()
    frame_df = frame_df[~frame_df.index.astype(str).str.startswith("$")]
    return frame_df


class DV2TurnoverRankStrategy(DVO2Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return add_liquidity_features(super().compute_signals(pricing_data), pricing_data)

    def get_opportunities(self, close) -> list:
        frame_df = entry_candidate_df(close)
        frame_df = frame_df[(frame_df["dv2"] < 10) & (frame_df["Close"] > frame_df["sma_200"]) & (frame_df["p126d_return"] > 0.05)]
        frame_df = frame_df.sort_values("adv_63", ascending=False)
        member_list = get_asof_universe_symbol_list(self.universe_df, pd.Timestamp(self.previous_bar))
        return frame_df[frame_df.index.isin(member_list)].index.tolist()


class DV2LiquidityFloorStrategy(DVO2Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return add_liquidity_features(super().compute_signals(pricing_data), pricing_data)

    def get_opportunities(self, close) -> list:
        frame_df = entry_candidate_df(close)
        member_list = get_asof_universe_symbol_list(self.universe_df, pd.Timestamp(self.previous_bar))
        member_df = frame_df[frame_df.index.isin(member_list)]
        # *** CRITICAL*** the median is taken over that day's index members, before the entry rules (as in HPI).
        median_adv_float = float(member_df["adv_63"].median())
        if not pd.notna(median_adv_float):
            return []
        member_df = member_df[(member_df["raw_price"] > RAW_PRICE_MIN_FLOAT) & (member_df["adv_63"] > median_adv_float)]
        member_df = member_df[(member_df["dv2"] < 10) & (member_df["Close"] > member_df["sma_200"]) & (member_df["p126d_return"] > 0.05)]
        return member_df.sort_values("natr", ascending=False).index.tolist()


CLASS_BY_ALIAS_DICT = {"dv2_check": DVO2Strategy, "dv2_turnover_rank": DV2TurnoverRankStrategy, "dv2_liq_floor": DV2LiquidityFloorStrategy}


def run_one(alias_str: str) -> dict:
    from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner

    started_float = time.time()
    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_df = get_prices(symbol_list, benchmark_list, start_date="1998-01-01", end_date=END_DATE_STR)
    strategy_obj = CLASS_BY_ALIAS_DICT[alias_str](
        name="strategy_mr_dv2", benchmarks=benchmark_list, capital_base=REFERENCE_CAPITAL_FLOAT, slippage=0.00025,
        commission_per_share=0.005, commission_minimum=1.0, performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR)
    strategy_obj.universe_df = universe_df
    strategy_obj.trade_id = 0
    strategy_obj.current_trade = defaultdict(default_trade_id_int)
    calendar_idx = pricing_df.index[pricing_df.index >= pd.Timestamp(START_DATE_STR)]
    run_daily(strategy_obj, pricing_df, calendar_idx, show_progress=False, show_signal_progress_bool=False)
    strategy_obj.universe_df = None
    result_df = ladder_runner.extract_source_result_df(strategy_obj)
    transaction_df = ladder_runner.extract_source_transaction_df(strategy_obj, alias_str)
    if result_df.index[-1] != pd.Timestamp(END_DATE_STR):
        raise RuntimeError(f"{alias_str} ended {result_df.index[-1].date()}, not {END_DATE_STR}.")
    OUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    ladder_runner.write_csv_gzip(result_df, OUT_DIR_PATH / f"{alias_str}__path.csv.gz", index_bool=True, index_label_str="date")
    ladder_runner.write_csv_gzip(transaction_df, OUT_DIR_PATH / f"{alias_str}__transactions.csv.gz", index_bool=False)
    metadata_dict = {"alias_str": alias_str, "strategy_class_str": CLASS_BY_ALIAS_DICT[alias_str].__name__, "start_date_str": START_DATE_STR,
                     "end_date_str": END_DATE_STR, "transaction_count_int": int(len(transaction_df)),
                     "runtime_seconds_float": round(time.time() - started_float, 1)}
    (OUT_DIR_PATH / f"{alias_str}__metadata.json").write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
    return metadata_dict


def main() -> int:
    with ProcessPoolExecutor(max_workers=3, max_tasks_per_child=1) as executor_obj:
        future_dict = {executor_obj.submit(run_one, a): a for a in CLASS_BY_ALIAS_DICT}
        for future_obj in as_completed(future_dict):
            print(json.dumps(future_obj.result()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
