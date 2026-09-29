"""Phase 5c: finalists through the real Vanilla engine (research-only classes; no strategy file is modified).

Each class is the WIRED DV2 / liquidity-floor strategy with exactly one documented change:
  F1  rank candidates by ADV63 (raw close x volume, 63-day mean) instead of NATR14
  F2  F1 with 15 slots
  F3  entry needs >= 5 of 9 DV percentiles < 10 (DV smoothing 2/3/5 x rank window 63/126/252), floor + trend kept
  F4  DV2 rank window 252 instead of 126
  ETF WIRED rules on the declared industry-ETF list; eligible after 252 sessions of own history
Usage: python engine_finalists.py F1 | F2 | F3 | F4 | ETF
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from alpha.engine.backtest import run_daily  # noqa: E402
from alpha.indicators import dv2_indicator  # noqa: E402
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR, build_index_constituent_matrix, load_raw_prices  # noqa: E402
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy, default_trade_id_int, get_asof_universe_symbol_list  # noqa: E402
from strategies.dv2.strategy_mr_dv2_liquidity_floor import DVO2LiquidityFloorStrategy, RAW_PRICE_MIN_FLOAT  # noqa: E402
import replica as rp  # noqa: E402
import phase4_universes as p4  # noqa: E402

OUT = REPO / "results/research/dv2_deep_20260925/engine"


def floor_members(strategy, close):
    candidate_df = close.unstack().dropna()
    candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]
    member_list = get_asof_universe_symbol_list(strategy.universe_df, pd.Timestamp(strategy.previous_bar))
    member_df = candidate_df[candidate_df.index.isin(member_list)]
    median_adv_float = float(member_df["adv_63"].median())
    if not pd.notna(median_adv_float):
        return None
    return member_df[(member_df["raw_price"] > RAW_PRICE_MIN_FLOAT) & (member_df["adv_63"] > median_adv_float)]


class F1(DVO2LiquidityFloorStrategy):
    def get_opportunities(self, close) -> list:
        m = floor_members(self, close)
        if m is None:
            return []
        m = m[(m["dv2"] < 10) & (m["Close"] > m["sma_200"]) & (m["p126d_return"] > 0.05)]
        return m.sort_values("adv_63", ascending=False, kind="stable").index.tolist()


class F2(F1):
    max_positions = 15


class F3(DVO2LiquidityFloorStrategy):
    def compute_signals(self, pricing_data):
        signal_df = super().compute_signals(pricing_data)
        feats = {}
        for s in pricing_data.columns.get_level_values(0).unique():
            if str(s).startswith("$") or (s, "Close") not in pricing_data.columns:
                continue
            C = pricing_data[(s, "Close")].to_numpy(float)[:, None]
            H = pricing_data[(s, "High")].to_numpy(float)[:, None]
            L = pricing_data[(s, "Low")].to_numpy(float)[:, None]
            dv1 = rp._dv1(C, H, L)
            votes = np.zeros(len(C))
            finite = np.ones(len(C), dtype=bool)
            for k in (2, 3, 5):
                dvk = rp._roll_mean_all(dv1, k)
                for w in (63, 126, 252):
                    # *** CRITICAL*** trailing percentile of DV_k over [t-w+1, t]; known after Close_t
                    pc = rp._pct_rank(dvk, w)[:, 0]
                    finite &= np.isfinite(pc)
                    votes += np.where(np.isfinite(pc) & (pc < 10), 1, 0)
            feats[(s, "dv_votes")] = pd.Series(np.where(finite, votes, np.nan), index=pricing_data.index)
        return pd.concat([signal_df, pd.DataFrame(feats, index=signal_df.index)], axis=1).copy()

    def get_opportunities(self, close) -> list:
        m = floor_members(self, close)
        if m is None:
            return []
        m = m[(m["dv_votes"] >= 5) & (m["Close"] > m["sma_200"]) & (m["p126d_return"] > 0.05)]
        return m.sort_values("natr", ascending=False).index.tolist()


class F4(DVO2LiquidityFloorStrategy):
    def compute_signals(self, pricing_data):
        signal_df = super().compute_signals(pricing_data)
        for s in pricing_data.columns.get_level_values(0).unique():
            if str(s).startswith("$") or (s, "Close") not in pricing_data.columns:
                continue
            signal_df[(s, "dv2")] = dv2_indicator(pricing_data[(s, "Close")], pricing_data[(s, "High")], pricing_data[(s, "Low")], length_int=252)
        return signal_df


def run(which: str):
    t0 = time.time()
    if which == "ETF":
        symbols = p4.GROUPS["industries"]
        pricing = load_raw_prices(symbols, ["$SPX"], start_date="1989-01-01", end_date="2026-08-19")
        closes = pricing.xs("Close", axis=1, level=1)[[s for s in symbols if s in pricing.columns.get_level_values(0)]]
        universe_df = (closes.notna().cumsum() >= 252).astype(int)
        cls = DVO2Strategy
    else:
        symbols, universe_df = build_index_constituent_matrix(indexname="S&P 500")
        pricing = load_raw_prices(symbols, ["$SPX"], start_date="1998-01-01", end_date="2026-08-19")
        cls = {"F1": F1, "F2": F2, "F3": F3, "F4": F4}[which]
    st = cls(name=f"dv2_deep_{which}", benchmarks=["$SPX"], capital_base=1_000_000.0, slippage=0.00025,
             commission_per_share=0.005, commission_minimum=1.0, performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR)
    st.universe_df = universe_df
    st.trade_id = 0
    st.current_trade = defaultdict(default_trade_id_int)
    cal = pricing.index[pricing.index >= pd.Timestamp("2000-01-03")]
    run_daily(st, pricing, cal, show_progress=False, show_signal_progress_bool=False)
    OUT.mkdir(parents=True, exist_ok=True)
    st.results[["total_value"]].to_csv(OUT / f"{which}__path.csv")
    st.get_transactions().to_csv(OUT / f"{which}__transactions.csv", index=False)
    (OUT / f"{which}__meta.json").write_text(json.dumps({"which": which, "runtime_s": round(time.time() - t0, 1),
                                                          "fills": int(len(st.get_transactions()))}), encoding="utf-8")
    print(which, "done", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
