"""Check B: HPI 2/3/5 vote with and without the DV2-G gate, real engine (SPEC_FROZEN.md). Usage: python check_b_hpi_engine.py ungated|gated"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import strategies.hpi.stateful_long as sl  # noqa: E402
from alpha.engine.backtest import run_daily  # noqa: E402
from data.norgate_loader import load_price_timeseries  # noqa: E402

OUT = REPO / "results/research/mr_gate_final_checks_20261003"
MEMORY = 15


def gate_series() -> pd.Series:
    v = load_price_timeseries("$VIX", start_date_str="1990-01-01")["Close"].astype(float)
    v.index = pd.to_datetime(v.index)
    thr = v.expanding(min_periods=500).mean()
    out, state, held = [], False, 0
    for x, t in zip(v.to_numpy(), thr.to_numpy()):
        if np.isfinite(x) and np.isfinite(t):
            c = x > t
            if not state and c:
                state, held = True, 0
            elif state:
                held += 1
                if not c and held >= MEMORY:
                    state = False
        out.append(state)
    return pd.Series(out, index=v.index)


class GatedHPI(sl.HPIStatefulLongStrategy):
    gate_ser: pd.Series | None = None

    def get_opportunity_list(self, close_row_ser, member_symbol_set=None):
        # *** CRITICAL*** the decision is taken after Close(previous_bar); use the gate known at that close.
        d = pd.Timestamp(self.previous_bar)
        g = self.gate_ser.loc[:d]
        if len(g) == 0 or not bool(g.iloc[-1]):
            return []
        return super().get_opportunity_list(close_row_ser, member_symbol_set)


def main(arm):
    OUT.mkdir(parents=True, exist_ok=True)
    _, universe_df, pricing_df = sl.load_exact_hpi_inputs(indexname_str="S&P 500", benchmark_symbol_str="$SPXTR", start_date_str="1998-01-01", end_date_str=None)
    syms = pricing_df.columns.get_level_values(0).unique().astype(str)
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {s: ("TOTALRETURN" if s == "$SPXTR" else "CAPITALSPECIAL") for s in syms}
    cls = GatedHPI if arm == "gated" else sl.HPIStatefulLongStrategy
    s = cls(name=f"hpi_vote_{arm}", benchmarks=["$SPXTR"], ranking_field_str=sl.TURNOVER_FIELD_STR, capital_base=100_000.0,
            entry_mode_str=sl.ENTRY_HORIZON_VOTE_STR, backtest_start_date_str="2004-01-01")
    if arm == "gated":
        s.gate_ser = gate_series()
    s.universe_df = universe_df
    cal = pricing_df.index[pricing_df.index >= pd.Timestamp("2004-01-01")]
    run_daily(s, pricing_df, cal, show_progress=False, show_signal_progress_bool=False)
    s.results[["total_value", "cash"]].to_csv(OUT / f"hpi_{arm}_nav.csv")
    pd.DataFrame(s.get_transactions()).to_csv(OUT / f"hpi_{arm}_transactions.csv", index=False)
    print("done", arm, len(s.results))


if __name__ == "__main__":
    main(sys.argv[1])
