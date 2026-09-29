"""POST-HOC diagnostic (lead, 2026-09-27; not pre-registered, changes no verdict).

Question: does the passing breakout leg (family B, stage B2, rank R2 = quietest breakout first) earn its book effect by
buying takeover targets after the announcement jump and holding them to the deal close (merger arbitrage in disguise)?
And is the PREREG's flat 25% haircut on every terminal liquidation a fair stress for it?

For every holding episode of the B2 neighbourhood cells (and B0 for contrast) on S&P 500:
- terminal episodes (position liquidated because the price series ended) are classified by the price path:
    acquisition-like: last close within 10% of the post-entry highest close and NATR20 over the final 20 sessions < 1.5%
    distress-like:    last close below 50% of the post-entry highest close
    other:            everything else
- "pre-delisting entries": episodes of any exit reason whose symbol stops trading within 250 sessions after entry.
A targeted haircut applies the 25% haircut only to distress-like terminal liquidations (NAV shock on the liquidation
day, compounded forward through the daily returns).

    python scripts/research/trend_breakout_20260927/posthoc_terminal_diagnostic.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/research
from trend_breakout_20260927 import common as tb_common  # noqa: E402
from trend_breakout_20260927 import data as tb_data  # noqa: E402

CELL_LIST = ["B|N100|k5|K10|R2|noVXN|noRX", "B|N100|k5|K20|R2|noVXN|noRX", "B|N100|k5|K20|R1|noVXN|noRX"]
END_TS = pd.Timestamp("2026-08-19")


def main() -> None:
    out_path = tb_common.RESULTS_DIR_PATH
    universe_dict = tb_data.load_universe("SP500")
    date_index = universe_dict["date_index"]
    close_arr = np.asarray(universe_dict["close_arr"], dtype=np.float64)
    high_arr = np.asarray(universe_dict["high_arr"], dtype=np.float64)
    low_arr = np.asarray(universe_dict["low_arr"], dtype=np.float64)
    prior_close_arr = np.vstack([np.full((1, close_arr.shape[1]), np.nan), close_arr[:-1]])
    tr_arr = np.maximum(np.maximum(high_arr - low_arr, np.abs(high_arr - prior_close_arr)), np.abs(low_arr - prior_close_arr))
    # *** CRITICAL *** diagnostic only (uses the whole path after entry by design); never a signal.
    natr20_arr = pd.DataFrame(tr_arr).rolling(20, min_periods=20).mean().to_numpy() / close_arr
    last_valid_pos_vec = np.array([
        int(np.flatnonzero(np.isfinite(close_arr[:, j]))[-1]) if np.isfinite(close_arr[:, j]).any() else -1
        for j in range(close_arr.shape[1])
    ])
    trades_dict = pickle.load(open(out_path / "trades_B_SP500.pkl", "rb"))
    returns_df = pd.read_parquet(out_path / "returns_B_SP500_engine.parquet")

    result_dict: dict = {"note": "POST-HOC diagnostic, not pre-registered; changes no verdict"}
    for cell_str in CELL_LIST:
        trade_df = trades_dict[cell_str].copy()
        trade_df = trade_df[date_index[trade_df["entry_pos"].to_numpy()] <= END_TS]
        rows = []
        for row in trade_df.itertuples():
            j, e, x = int(row.symbol_idx), int(row.entry_pos), int(row.exit_pos)
            path = close_arr[e: min(x, last_valid_pos_vec[j]) + 1, j]
            hwm = float(np.nanmax(path)) if len(path) else np.nan
            last_close = float(close_arr[last_valid_pos_vec[j], j])
            jump_20 = float(close_arr[e - 1, j] / close_arr[e - 21, j] - 1.0) if e >= 21 else np.nan
            delisted_within_250 = bool(last_valid_pos_vec[j] <= e + 250 and date_index[last_valid_pos_vec[j]] < date_index[-5])
            cls = ""
            if row.reason == "terminal":
                end_natr = float(np.nanmean(natr20_arr[max(last_valid_pos_vec[j] - 19, 0): last_valid_pos_vec[j] + 1, j]))
                if last_close >= 0.9 * hwm and end_natr < 0.015:
                    cls = "acquisition_like"
                elif last_close < 0.5 * hwm:
                    cls = "distress_like"
                else:
                    cls = "other_terminal"
            rows.append({"reason": row.reason, "class": cls, "pnl": float(row.pnl), "proceeds": float(row.proceeds),
                         "exit_pos": x, "jump_20_before_entry": jump_20,
                         "natr20_at_decision": float(natr20_arr[e - 1, j]) if e >= 1 else np.nan,
                         "delisted_within_250": delisted_within_250, "holding": int(row.holding_sessions)})
        ep_df = pd.DataFrame(rows)
        total_pnl = float(ep_df["pnl"].sum())
        term = ep_df[ep_df["reason"] == "terminal"]
        pre_delist = ep_df[ep_df["delisted_within_250"]]
        # targeted haircut: 25% of liquidation proceeds lost on distress-like terminal days only
        ret = returns_df[cell_str].copy()
        nav = 100_000.0 * (1.0 + ret).cumprod()
        adj_ret = ret.copy()
        for r in term[term["class"] == "distress_like"].itertuples():
            ts = date_index[int(r.exit_pos)]
            if ts in adj_ret.index:
                prev_nav = float(nav.shift(1).get(ts, np.nan))
                if np.isfinite(prev_nav) and prev_nav > 0:
                    adj_ret.loc[ts] -= 0.25 * r.proceeds / prev_nav

        def stats(r: pd.Series) -> dict:
            m = tb_common.metric_dict(r.loc["2000-01-04":END_TS])
            return {"cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"]}

        result_dict[cell_str] = {
            "episodes_int": int(len(ep_df)),
            "reason_counts": ep_df["reason"].value_counts().to_dict(),
            "terminal_class_counts": term["class"].value_counts().to_dict(),
            "terminal_pnl_by_class": term.groupby("class")["pnl"].sum().to_dict(),
            "terminal_pnl_share_of_total": float(term["pnl"].sum() / total_pnl) if total_pnl else np.nan,
            "pre_delisting_entries_int": int(len(pre_delist)),
            "pre_delisting_pnl_share_of_total": float(pre_delist["pnl"].sum() / total_pnl) if total_pnl else np.nan,
            "median_jump_20_before_entry_all": float(ep_df["jump_20_before_entry"].median()),
            "median_jump_20_before_entry_terminal": float(term["jump_20_before_entry"].median()) if len(term) else np.nan,
            "median_natr20_at_decision_all": float(ep_df["natr20_at_decision"].median()),
            "median_natr20_at_decision_terminal": float(term["natr20_at_decision"].median()) if len(term) else np.nan,
            "standalone_engine": stats(ret),
            "standalone_targeted_haircut": stats(adj_ret),
        }
    (out_path / "posthoc_terminal_diagnostic.json").write_text(json.dumps(result_dict, indent=1, default=float))
    print(json.dumps(result_dict, indent=1, default=float))


if __name__ == "__main__":
    main()
