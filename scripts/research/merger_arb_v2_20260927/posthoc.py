"""Post-hoc diagnostics for the report (research only; computed after the rule was applied, change no verdict).

- V0: deal P&L in dollars and per year, the T-bill share of the swept return, calendar-year returns against the
  version-1 anchor M0, MNA and BIL; concurrency and the split of confirmations / entries by Russell half.
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
from merger_arb_v2_20260927 import cells as cells_module  # noqa: E402
from merger_arb_v2_20260927 import common  # noqa: E402
from merger_arb_v2_20260927 import analyze  # noqa: E402

R = common.RESULTS_DIR_PATH


def main() -> None:
    out: dict = {"note": "POST-HOC diagnostics computed after the rule was applied; they change no verdict."}
    bil_ser = common.load_bil_ret_ser()
    key0 = cells_module.V0_CELL.key_str
    trades = pickle.load(open(R / "trades_V_ALL.pkl", "rb"))
    years_float = float((common.END_TS - common.TRADING_START_TS).days / 365.25)
    v0_out = {}
    for key_str in (key0, "V|J10|th0.8|W5|K10|brk95", "V|J15|th0.5|W3|K20|brk95"):
        if key_str not in trades:
            continue
        trade_df = trades[key_str]
        ret_vec = (trade_df["pnl"] / trade_df["cost"]).to_numpy() if len(trade_df) else np.array([])
        v0_out[key_str] = {"deal_pnl_total_usd": float(trade_df["pnl"].sum()), "deal_pnl_per_year_pct_of_100k": float(trade_df["pnl"].sum() / years_float / common.CAPITAL_BASE_FLOAT),
                           "mean_return_per_episode": float(ret_vec.mean()) if len(ret_vec) else float("nan"), "median_return_per_episode": float(np.median(ret_vec)) if len(ret_vec) else float("nan"),
                           "worst_five_episodes": sorted(np.round(ret_vec, 4).tolist())[:5], "best_five_episodes": sorted(np.round(ret_vec, 4).tolist())[-5:],
                           "pnl_by_reason_usd": {str(k): float(v) for k, v in trade_df.groupby("reason")["pnl"].sum().items()} if len(trade_df) else {},
                           "share_of_episodes_negative": float((ret_vec < 0).mean()) if len(ret_vec) else float("nan")}
    out["V_cells"] = v0_out
    sweep_df = analyze.pod_returns("V_ALL", "engine", bil_ser, True)
    nosweep_df = analyze.pod_returns("V_ALL", "engine", bil_ser, False)
    for label, frame in (("sweep", sweep_df), ("nosweep", nosweep_df)):
        for win, (s, e) in (("FULL", common.BLOCK_DICT["FULL"]), ("2012-2026", ("2012-10-02", "2026-08-19"))):
            out.setdefault("V0_return_decomposition", {})[f"{label}_{win}"] = common.metric_dict(frame[key0].loc[s:e])
    out["BIL_2007_2026"] = common.metric_dict(bil_ser.loc["2007-05-31":common.END_TS])
    mna_ser = common.load_mna_ret_ser()
    m0_ser = common.load_v1_m0_ret_ser(bil_ser)
    frame = pd.DataFrame({"V0_sweep": sweep_df[key0], "V0_nosweep": nosweep_df[key0], "M0_v1_sweep": m0_ser, "MNA": mna_ser, "BIL": bil_ser}).loc["2000-01-04":common.END_TS]
    yearly = (1 + frame.fillna(0.0)).groupby(frame.index.year).prod() - 1
    out["calendar_year_returns"] = {str(y): {k: round(float(v), 4) for k, v in row.items()} for y, row in yearly.iterrows()}
    out["corr_V0_vs_M0_v1_2000_2026"] = float(frame["V0_sweep"].corr(frame["M0_v1_sweep"])) if m0_ser is not None else None
    # halves split of V0 entries
    halves_meta = {h: json.loads((R / f"meta_V_{h}.json").read_text()) for h in cells_module.HALF_TUPLE if (R / f"meta_V_{h}.json").exists()}
    out["V0_by_half"] = {h: {k: m[f"{key0}|{h}"]["engine"]["v"][k] for k in ("events_per_year", "confirmations_per_year", "entries_per_year", "entries_total_int", "precision_terminal_within_252", "positions_mean")} for h, m in halves_meta.items() if f"{key0}|{h}" in m}
    (R / "posthoc.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "calendar_year_returns"}, indent=1, default=str))
    print(yearly.round(3).T.to_string())


if __name__ == "__main__":
    main()
