"""Real-engine run of SH1 (mirror DV2 short, floor with the complete-row median) to validate replica shorts."""
from collections import defaultdict
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
import pandas as pd
from alpha.engine.backtest import run_daily
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR, build_index_constituent_matrix, load_raw_prices
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy, default_trade_id_int
from strategies.dv2.strategy_mr_dv2_liquidity_floor import DVO2LiquidityFloorStrategy
import engine_finalists as ef

class SH1(DVO2LiquidityFloorStrategy):
    def iterate(self, data, close, open_prices):
        pos = self.get_positions(); shorts = pos[pos < 0]
        slots = self.max_positions - len(shorts)
        for s in shorts.index:
            if close[(s, "Close")] < data[(s, "Low")].iloc[-2]:
                self.order_target_value(s, 0, trade_id=self.current_trade[s]); slots += 1
        cap = self.previous_total_value / self.max_positions
        opp = self.get_opportunities(close)
        while slots > 0 and opp:
            s = opp.pop(0)
            if self.get_position(s) != 0:
                continue
            self.trade_id += 1; self.current_trade[s] = self.trade_id
            self.order_value(s, -cap, trade_id=self.trade_id); slots -= 1
    def get_opportunities(self, close):
        m = ef.floor_members(self, close)
        if m is None:
            return []
        m = m[(m["dv2"] > 90) & (m["Close"] < m["sma_200"]) & (m["p126d_return"] < -0.05)]
        return m.sort_values("natr", ascending=False).index.tolist()

t0 = time.time()
syms, uni = build_index_constituent_matrix(indexname="S&P 500")
pr = load_raw_prices(syms, ["$SPX"], start_date="1998-01-01", end_date="2026-08-19")
st = SH1(name="sh1", benchmarks=["$SPX"], capital_base=1_000_000.0, slippage=0.00025, commission_per_share=0.005,
         commission_minimum=1.0, performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR)
st.universe_df = uni; st.trade_id = 0; st.current_trade = defaultdict(default_trade_id_int)
run_daily(st, pr, pr.index[pr.index >= pd.Timestamp("2000-01-03")], show_progress=False, show_signal_progress_bool=False)
out = REPO / "results/research/dv2_deep_20260925/shorts"; out.mkdir(parents=True, exist_ok=True)
st.results[["total_value"]].to_csv(out / "SH1_engine__path.csv"); st.get_transactions().to_csv(out / "SH1_engine__transactions.csv", index=False)
print("done", round(time.time() - t0, 1))
