"""POST-HOC controls (lead, 2026-09-27; not pre-registered, change no verdict).

The passing candidate (family B, stage B2: S&P 500 quiet 100-day breakouts, 5xATR20 trailing exit, K 10, addition
role 0.5 TAA + 0.25 L + 0.25 B) improves G3. The quant-pitfalls review asked how much of that is simply adding
S&P 500 exposure. Controls in the same quarter slot, same official pod model (annual reset, one run per window):
- passive SPY: SPY CAPITALSPECIAL daily price return (no dividends, no costs - favours the control slightly on costs,
  penalises it by the missing dividend yield);
- gated SPY: SPY price return on days after a close with SPY > SMA200, else cash (close-to-close approximation of a
  next-open switch).
Also: B2 after the study END (2026-08-20..2026-09-25, outside every block).

    python scripts/research/trend_breakout_20260927/posthoc_spy_controls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/research
from trend_breakout_20260927 import common as tb_common  # noqa: E402
from trend_breakout_20260927 import data as tb_data  # noqa: E402

END_STR = "2026-08-19"
WINDOW_DICT = {
    "G-P1": ("2008-03-04", "2011-12-31"),
    "G-P2": ("2012-10-02", "2021-12-31"),
    "G-P3": ("2022-01-01", END_STR),
    "G-FULL": ("2012-10-02", END_STR),
    "G-LONG": ("2008-03-04", END_STR),
}


def book(leg_dict: dict, weight_dict: dict, start_str: str, end_str: str) -> pd.Series:
    frame_df = pd.DataFrame(leg_dict).loc[start_str:end_str].dropna()
    return tb_common.book_return_ser(frame_df, weight_dict, "annual")[0]


def main() -> None:
    out_path = tb_common.RESULTS_DIR_PATH
    taa_ser = tb_common.load_taa_ser()
    l_ser = pd.read_parquet(out_path / "returns_A_NDX_engine.parquet")["A|none|CASH|k+0"]
    b_df = pd.read_parquet(out_path / "returns_B_SP500_engine.parquet")
    spy_close_ser = tb_data.load_universe("SP500")["spy_close_ser"].astype(float)
    spy_ret_ser = spy_close_ser.pct_change()
    # *** CRITICAL *** gate known at close t applies to the next session's return (no same-day use).
    gate_ser = (spy_close_ser > spy_close_ser.rolling(200, min_periods=200).mean()).astype(float).shift(1)
    gated_ret_ser = (spy_ret_ser * gate_ser).fillna(0.0)
    leg_dict = {
        "B2_centre_K10": b_df["B|N100|k5|K10|R2|noVXN|noRX"],
        "B2_interior_K20": b_df["B|N100|k5|K20|R2|noVXN|noRX"],
        "passive_SPY_price": spy_ret_ser,
        "gated_SPY_SMA200_price": gated_ret_ser,
    }
    result_dict: dict = {"note": "POST-HOC controls, not pre-registered; change no verdict", "addition_margin_vs_G3": {}}
    for leg_str, leg_ser in leg_dict.items():
        rows = {}
        for window_str, (start_str, end_str) in WINDOW_DICT.items():
            g3_ser = book({"taa": taa_ser, "L": l_ser}, {"taa": 0.5, "L": 0.5}, start_str, end_str)
            add_ser = book({"taa": taa_ser, "L": l_ser, "X": leg_ser}, {"taa": 0.5, "L": 0.25, "X": 0.25}, start_str, end_str)
            g3_m, add_m = tb_common.metric_dict(g3_ser), tb_common.metric_dict(add_ser)
            rows[window_str] = {"sharpe_margin": add_m["sharpe"] - g3_m["sharpe"], "dd_gap_pp": 100 * (add_m["max_dd"] - g3_m["max_dd"]),
                                "book": add_m}
        standalone_m = tb_common.metric_dict(leg_ser.loc["2000-01-04":END_STR])
        recent_m = tb_common.metric_dict(leg_ser.loc["2022-01-01":END_STR])
        result_dict["addition_margin_vs_G3"][leg_str] = {"windows": rows, "standalone_2000_2026": standalone_m, "standalone_2022_2026": recent_m}
    post_slice = slice("2026-08-20", "2026-09-25")
    result_dict["after_END_2026_08_20_to_09_25"] = {
        "B2_centre_K10": float((1 + leg_dict["B2_centre_K10"].loc[post_slice]).prod() - 1),
        "passive_SPY_price": float((1 + spy_ret_ser.loc[post_slice]).prod() - 1),
        "L": float((1 + l_ser.loc[post_slice]).prod() - 1),
    }
    result_dict["corr_B2_vs_SPY_2012_2026"] = float(leg_dict["B2_centre_K10"].loc["2012-10-02":END_STR].corr(spy_ret_ser.loc["2012-10-02":END_STR]))
    (out_path / "posthoc_spy_controls.json").write_text(json.dumps(result_dict, indent=1, default=float))
    for leg_str, payload in result_dict["addition_margin_vs_G3"].items():
        print(leg_str, {w: round(v["sharpe_margin"], 3) for w, v in payload["windows"].items()},
              "standalone", round(payload["standalone_2000_2026"]["sharpe"], 3), "2022-26", round(payload["standalone_2022_2026"]["sharpe"], 3))
    print(result_dict["after_END_2026_08_20_to_09_25"], result_dict["corr_B2_vs_SPY_2012_2026"])


if __name__ == "__main__":
    main()
