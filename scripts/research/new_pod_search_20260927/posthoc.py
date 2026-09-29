"""Post-hoc diagnostics for the report (research only; computed after the rule was applied, change no verdict).

- family M: concurrent positions over time, deal P&L in dollars and per year, the T-bill share of the M pod's return;
- family S: calendar-year returns of the two anchors against L and SPY.
Writes posthoc.json.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from new_pod_search_20260927 import cells as cells_module  # noqa: E402
from new_pod_search_20260927 import common  # noqa: E402
from new_pod_search_20260927 import analyze  # noqa: E402

R = common.RESULTS_DIR_PATH


def main() -> None:
    out: dict = {"note": "POST-HOC diagnostics computed after the rule was applied; they change no verdict."}
    bil_ser = common.load_bil_ret_ser()
    trades = pickle.load(open(R / "trades_M_R1000.pkl", "rb"))
    date_index = pd.DatetimeIndex(pd.read_parquet(R / "returns_M_R1000_engine.parquet").index)
    start_pos = 0
    m_out = {}
    for key_str in (cells_module.M0_CELL.key_str, "M|J10|th1.5|W5|K20|brk95", "M|J15|th1|W5|K10|brk95"):
        trade_df = trades[key_str]
        concurrency = np.zeros(len(date_index) + 400, dtype=int)
        for row in trade_df.itertuples(index=False):
            concurrency[row.entry_pos : row.exit_pos] += 1
        years = float((common.END_TS - common.TRADING_START_TS).days / 365.25)
        m_out[key_str] = {
            "positions_mean": float(concurrency[: len(date_index)].mean()), "positions_max": int(concurrency.max()),
            "share_of_sessions_with_no_position": float((concurrency[: len(date_index)] == 0).mean()),
            "deal_pnl_total_usd": float(trade_df["pnl"].sum()), "deal_pnl_per_year_usd_on_100k_pod": float(trade_df["pnl"].sum() / years),
            "deal_pnl_per_year_pct_of_initial_capital": float(trade_df["pnl"].sum() / years / common.CAPITAL_BASE_FLOAT),
            "mean_return_per_deal": float((trade_df["pnl"] / trade_df["cost"]).mean()), "median_holding_sessions": float(trade_df["holding_sessions"].median()),
            "pnl_by_reason_usd": {str(k): float(v) for k, v in trade_df.groupby("reason")["pnl"].sum().items()},
            "worst_five_episodes_return": sorted((trade_df["pnl"] / trade_df["cost"]).round(4).tolist())[:5],
            "best_five_episodes_return": sorted((trade_df["pnl"] / trade_df["cost"]).round(4).tolist())[-5:],
        }
    # T-bill share of the M0 pod return with the sweep
    sweep_df = analyze.pod_returns("M_R1000", "engine", bil_ser, True)
    nosweep_df = analyze.pod_returns("M_R1000", "engine", bil_ser, False)
    k0 = cells_module.M0_CELL.key_str
    for label, frame in (("sweep", sweep_df), ("nosweep", nosweep_df)):
        for win, (s, e) in (("FULL", common.BLOCK_DICT["FULL"]), ("2012-2026", ("2012-10-02", "2026-08-19"))):
            m_out.setdefault("M0_return_decomposition", {})[f"{label}_{win}"] = common.metric_dict(frame[k0].loc[s:e])
    bil_full = common.metric_dict(bil_ser.loc["2007-05-31":common.END_TS])
    m_out["BIL_standalone_2007_2026"] = bil_full
    out["M"] = m_out
    # S anchors: calendar-year returns vs L and SPY TR
    s_sweep = analyze.pod_returns("S_SP500", "engine", bil_ser, True)
    l_ser = common.load_l_ret_ser("engine")
    spy_ser = common.load_spy_tr_ret_ser()
    frame = pd.DataFrame({"GATED_anchor": s_sweep[cells_module.S_ANCHOR_DICT["GATED"].key_str], "HEDGED_anchor": s_sweep[cells_module.S_ANCHOR_DICT["HEDGED"].key_str],
                          "GATED_centre_SE_1_20_N50": s_sweep["S|SE_1_20|N50|GATED|k+0"], "L": l_ser, "SPY_TR": spy_ser}).loc["2000-01-04":common.END_TS]
    yearly = (1 + frame.fillna(0.0)).groupby(frame.index.year).prod() - 1
    out["S_calendar_year_returns"] = {str(y): {k: round(float(v), 4) for k, v in row.items()} for y, row in yearly.iterrows()}
    (R / "posthoc.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["M"], indent=1, default=str))
    print(yearly.round(3).T.to_string())


if __name__ == "__main__":
    main()
