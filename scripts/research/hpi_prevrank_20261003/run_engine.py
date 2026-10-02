"""HPI 2/3/5 vote with current vs previous-session turnover rank, real engine (SPEC_FROZEN.md, research-only).

Usage: python run_engine.py current|prev
The strategy file is not modified: a research subclass adds Turnover_{T-1} as an extra feature column.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import strategies.hpi.stateful_long as sl  # noqa: E402
from alpha.engine.backtest import run_daily  # noqa: E402

OUT = REPO / "results/research/hpi_prevrank_20261003"
PREV = "TurnoverPrev"


class PrevRankHPI(sl.HPIStatefulLongStrategy):
    def compute_signals(self, pricing_data_df):
        out = super().compute_signals(pricing_data_df)
        syms = [s for s in pricing_data_df.columns.get_level_values(0).unique()
                if not str(s).startswith("$") and (s, "Turnover") in pricing_data_df.columns]
        # *** CRITICAL*** previous SESSION's turnover: shift by one calendar row, known before Close_T.
        prev = pd.DataFrame({(s, PREV): pricing_data_df[(s, "Turnover")].shift(1) for s in syms}, index=out.index)
        return pd.concat([out, prev], axis=1)


def main(arm):
    OUT.mkdir(parents=True, exist_ok=True)
    sl.RANKING_FIELD_SET.add(PREV)  # in-process only
    _, universe_df, pricing_df = sl.load_exact_hpi_inputs(indexname_str="S&P 500", benchmark_symbol_str="$SPXTR",
                                                          start_date_str="1998-01-01", end_date_str=None)
    syms = pricing_df.columns.get_level_values(0).unique().astype(str)
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {s: ("TOTALRETURN" if s == "$SPXTR" else "CAPITALSPECIAL") for s in syms}
    s = PrevRankHPI(name=f"hpi_vote_{arm}", benchmarks=["$SPXTR"], ranking_field_str=PREV if arm == "prev" else sl.TURNOVER_FIELD_STR,
                    capital_base=100_000.0, entry_mode_str=sl.ENTRY_HORIZON_VOTE_STR, backtest_start_date_str="2004-01-01")
    s.universe_df = universe_df
    cal = pricing_df.index[pricing_df.index >= pd.Timestamp("2004-01-01")]
    run_daily(s, pricing_df, cal, show_progress=False, show_signal_progress_bool=False)
    res = getattr(s, "results", None)
    if res is None:
        res = getattr(s, "_results")
    res[["total_value"]].to_csv(OUT / f"nav_{arm}.csv")
    pd.DataFrame(s.get_transactions()).to_csv(OUT / f"transactions_{arm}.csv", index=False)
    print("done", arm, len(res))


if __name__ == "__main__":
    main(sys.argv[1])
