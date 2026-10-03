"""DV2 (wired rules) behind the MR capsule's VIX stress gate, idle cash parked in SPMO / BIL (research-only).

Stock rules are those of strategies/dv2/strategy_mr_dv2.py (DVO2Strategy), unchanged:
    entry_i,T = 1[DV2(126)_i,T < 10] * 1[Close_i,T > SMA200_i,T] * 1[Return126_i,T > 0.05] * 1[i in S&P 500 at T]
    rank: NATR14 descending; 10 slots of previous_total_value / 10; exit: Close_i,T > High_i,(T-1)
Capsule additions (docs/research/MR_CAPSULE_20261003.md):
    new entries only when the VIX stress gate is open at Close_T (strategies/mr_capsule/vix_stress_gate.py);
    exits are never gated; SPMO / BIL are excluded from the slot count and the exit rule;
    idle cash: SPMO (8% vol target, weekly) while the gate is closed, BIL otherwise (strategies/mr_capsule/parking.py).
Decisions after Close_T, fills at Open_(T+1) (engine contract). CAPITALSPECIAL fills and marks; dividends credited
net of the house 25% withholding (G-024).

Known caveats (kept with the strategy, owner request; measured in the build record, docs/research/
MR_CAPSULE_20261003.md "Build record (2026-10-04)", and docs/strategies/book-strategy-caveats.md):
- issue: the gate and the parking were chosen after about 100 gate variants and about 10 parking forks on 2000-2026
  data (SPMO missed its pre-registered bar; "SPMO only while the gate is closed" was chosen on the same window);
  bias: optimistic; impact: medium; mitigation: frozen rules, plateau checks, DSR 0.97 (N = 110, engine capsule
  2004-2026), forward paper before capital.
- issue: SPMO's research edge over T-bills came from 2015-11 to 2018-01, when SPMO did not trade most days; from 2018
  SPMO parking adds about 0.5 pp/yr of CAGR but costs about 0.04 Sharpe and 2 pp of drawdown (engine); bias:
  optimistic in the research record; impact: medium; mitigation: spmo_parking_enabled_bool=False runs BIL only.
- issue: SPMO is used only once it traded on each of the last 20 sessions (guard B1: from 2018); BIL exists from
  2007-05-30 and idle cash earns 0% before; bias: conservative before 2018; impact: low-medium.
- issue: BIL dividends carry the house 25% withholding (interest-related dividends may be exempt, G-029) and BIL
  trades pay the engine's 2.5 bps (BIL's spread is about 1 bp); bias: conservative; impact: low (about 0.4 pp/yr of
  capsule CAGR together, estimate).
- issue: negative cash comes from the parent's 10 x 10% sizing (next-open gaps), not from the parking (DV2-G 136
  sessions with parking, 133 without; minimum -8.8% of NAV); it is not financed (G-023); bias: optimistic, small.
- issue: the gated pod lags the ungated one in 2008-2011 (book 0.78 vs 0.81 in research) and in calm 1990s-style
  markets; bias: n/a (a real cost of the rule); impact: medium in such regimes.
- issue: about 52 parking orders a year at the $1 minimum commission; bias: costs higher for small pods; impact:
  about -0.35 pp/yr at USD 15K per pod, -0.1 pp/yr at USD 50K.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR, build_index_constituent_matrix, load_raw_prices
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy, default_trade_id_int
from strategies.mr_capsule.capsule_pod import CapsulePodMixin
from strategies.mr_capsule.parking import PARKING_SYMBOL_TUPLE
from strategies.mr_capsule.vix_stress_gate import load_vix_close_ser

STRATEGY_NAME_STR = "strategy_mr_dv2_vix_gated"
BENCHMARK_LIST = ["$SPX"]


class DV2VixGatedStrategy(CapsulePodMixin, DVO2Strategy):
    """DVO2Strategy with the capsule gate on entries and SPMO / BIL parking of idle cash."""

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = super().compute_signals(pricing_data)
        self._prepare_capsule_state()
        return signal_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        # Same decision logic as DVO2Strategy.iterate, with SPMO / BIL outside the slot count and the exit rule,
        # and entries only while the gate is open (tests/test_strategy_mr_capsule_pods.py pins the gate-open, no-parking
        # version to DVO2Strategy trade for trade).
        stock_position_ser = self._stock_position_ser()
        long_slots = self.max_positions - len(stock_position_ser)
        exiting_symbol_set = set()
        for symbol in stock_position_ser.index:
            c = close[(symbol, 'Close')]
            yh = data[(symbol, 'High')].iloc[-2]
            if c > yh:
                self.order_target_value(symbol, 0, trade_id=self.current_trade[symbol])
                exiting_symbol_set.add(symbol)
                long_slots += 1
        capital_to_allocate_per_trade = self.previous_total_value / self.max_positions
        gate_open_bool = self._gate_open_at_decision()
        entry_value_float = 0.0
        if gate_open_bool:
            long_opportunities = self.get_opportunities(close)
            while long_slots > 0 and len(long_opportunities) > 0:
                symbol = long_opportunities.pop(0)
                if self.get_position(symbol) != 0:
                    continue
                self.trade_id += 1
                self.current_trade[symbol] = self.trade_id
                self.order_value(symbol, capital_to_allocate_per_trade, trade_id=self.trade_id)
                entry_value_float += capital_to_allocate_per_trade
                long_slots -= 1
        elif long_slots > 0:
            self._gate_closed_free_slot_session_int += 1
        # *** CRITICAL*** stock value after this decision's orders, at Close_T (the engine's sizing price).
        kept_value_float = sum(
            float(share_float) * float(close[(symbol, 'Close')])
            for symbol, share_float in stock_position_ser.items()
            if symbol not in exiting_symbol_set
        )
        self._place_parking_orders(data, close, kept_value_float + entry_value_float, gate_open_bool)

    def finalize(self, current_data):
        self._record_capsule_diagnostics()
        parent_finalize = getattr(super(), "finalize", None)
        if callable(parent_finalize):
            parent_finalize(current_data)


def load_pricing_data(end_date_str: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S&P 500 PIT universe plus SPMO and BIL as CAPITALSPECIAL symbols (never benchmarks: G-024 dividend guard)."""
    index_symbol_list, universe_df = build_index_constituent_matrix(indexname='S&P 500')
    symbol_list = list(dict.fromkeys([*index_symbol_list, *PARKING_SYMBOL_TUPLE]))
    pricing_data_df = load_raw_prices(symbol_list, BENCHMARK_LIST, start_date='1998-01-01', end_date=end_date_str)
    return pricing_data_df, universe_df


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
    parking_enabled_bool: bool = True,
    slippage_float: float = 0.00025,
    spmo_parking_enabled_bool: bool = True,
):
    pricing_data_df, universe_df = load_pricing_data(end_date_str)
    strategy = DV2VixGatedStrategy(
        name=STRATEGY_NAME_STR,
        benchmarks=BENCHMARK_LIST,
        capital_base=capital_base_float,
        slippage=slippage_float,
        commission_per_share=0.005,
        commission_minimum=1.0,
        performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
    )
    strategy.universe_df = universe_df
    strategy.trade_id = 0
    strategy.current_trade = defaultdict(default_trade_id_int)
    strategy.vix_close_ser = load_vix_close_ser(end_date_str)
    strategy.parking_enabled_bool = parking_enabled_bool
    strategy.spmo_parking_enabled_bool = spmo_parking_enabled_bool
    # *** CRITICAL*** full pre-start history stays in the frame for indicators; trading starts at the first
    # deployment fill session (same convention as DVO2Strategy.run_variant).
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(strategy, pricing_data_df, calendar_idx, show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    strategy.universe_df = None
    if show_display_bool:
        display(strategy.summary)
        display(strategy.summary_trades)
    if save_results_bool:
        save_results(strategy, output_dir=output_dir_str)
    return strategy


if __name__ == "__main__":
    run_variant()
