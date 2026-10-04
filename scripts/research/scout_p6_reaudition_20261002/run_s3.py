"""P6: S3 for the two LIVE pods (class W: TAA 3x; class X: NDX VXN), in sample (to 2022-12-30).

TAA 3x: the blended 1-3-6-12 month momentum of the five defensive ETFs against their next-month total return, over the
ETFs' full histories (not only the pod's 2012 start); the VIX gate (SPY rv20 < VIX) against next-month QQQ risk.
NDX VXN: ROC12 / dollar ATR20 among eligible members (member and above SMA100) against next-month return.

    uv run python scripts/research/scout_p6_reaudition_20261002/run_s3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.stations.s3_allocation import (
    gate_split,
    predictive_tests,
    ranking_tests,
)

OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "reaudition"
SEAL_END_STR = "2022-12-30"


def taa_s3() -> dict:
    from alpha.scout.specs import taa_3x
    from data.norgate_loader import load_price_timeseries

    def total_return(symbol_str: str) -> pd.Series:
        return load_price_timeseries(symbol_str, adjustment_str="TOTALRETURN", start_date_str="1998-01-01")["Close"]

    month_close_df = pd.DataFrame({s: total_return(s) for s in taa_3x.DEFENSIVE_TUPLE}).resample("ME").last().loc[:SEAL_END_STR]
    score_df = sum(month_close_df.pct_change(k, fill_method=None) for k in taa_3x.MOMENTUM_MONTH_TUPLE) / len(taa_3x.MOMENTUM_MONTH_TUPLE)
    dtb3_ser = taa_3x.load_inputs().dtb3_ser
    hurdle_ser = ((1.0 + dtb3_ser / 100.0) ** (1.0 / 12.0) - 1.0).resample("ME").last()
    # *** CRITICAL*** next-month return: label only (month m's score against month m+1's return).
    next_return_df = month_close_df.pct_change(fill_method=None).shift(-1).sub(hurdle_ser.reindex(month_close_df.index), axis=0)
    result_dict = predictive_tests(score_df, next_return_df, hurdle_ser.reindex(score_df.index))

    spy_ser, vix_ser = load_price_timeseries("SPY", adjustment_str="CAPITALSPECIAL", start_date_str="1998-01-01")["Close"], load_price_timeseries("$VIX", adjustment_str="CAPITALSPECIAL", start_date_str="1998-01-01")["Close"]
    helper_df = pd.concat([spy_ser, vix_ser], axis=1, join="inner").dropna()
    realized_ser = (helper_df.iloc[:, 0].pct_change()).rolling(20).std(ddof=0) * np.sqrt(252.0) * 100.0
    gate_ser = (realized_ser < helper_df.iloc[:, 1]).resample("ME").last()
    qqq_month_ser = total_return("QQQ").resample("ME").last()
    result_dict["vix_gate"] = gate_split(qqq_month_ser.pct_change().shift(-1).loc[:SEAL_END_STR], gate_ser.loc[:SEAL_END_STR])
    return result_dict


def ndx_s3() -> dict:
    from alpha.scout.panel import load_panel

    panel = load_panel("Nasdaq 100")
    close_df, high_df, low_df = panel.field("Close"), panel.field("High"), panel.field("Low")
    previous_close_df = close_df.shift(1)
    true_range_df = np.maximum(high_df - low_df, np.maximum((high_df - previous_close_df).abs(), (low_df - previous_close_df).abs()))
    atr_dollar_df = true_range_df.rolling(20, min_periods=20).mean() * panel.field("Unadjusted Close") / close_df
    position_ser = pd.Series(np.arange(len(panel.date_index)), index=panel.date_index)
    decision_index = panel.date_index[position_ser.groupby(panel.date_index.to_period("M")).max().to_numpy()]
    month_close_df = close_df.loc[decision_index]
    score_df = (month_close_df / month_close_df.shift(12) - 1.0) / atr_dollar_df.loc[decision_index]
    eligible_df = (panel.member_df.loc[decision_index] == 1) & (month_close_df > close_df.rolling(100, min_periods=100).mean().loc[decision_index])
    next_return_df = month_close_df.shift(-1) / month_close_df - 1.0  # label only
    return ranking_tests(score_df.replace([np.inf, -np.inf], np.nan), next_return_df, eligible_df, top_int=10)


def main() -> None:
    output_dict = {"TAA 3x": taa_s3(), "NDX VXN": ndx_s3()}
    for pod_str, result_dict in output_dict.items():
        pod_dir_path = OUTPUT_DIR_PATH / pod_str.replace(" ", "_")
        pod_dir_path.mkdir(parents=True, exist_ok=True)
        (pod_dir_path / "s3.json").write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")
        print("==", pod_str)
        for name_str, verdict_str, detail_str in result_dict["check_list"] + ([result_dict["vix_gate"]["check"]] if "vix_gate" in result_dict else []):
            print(f"  {verdict_str:5s} {name_str}: {detail_str}")
        for row in result_dict.get("per_asset_list", []):
            print("   ", row)
        if "vix_gate" in result_dict:
            print("   gate", {k: round(v, 4) if isinstance(v, float) else v for k, v in result_dict["vix_gate"].items() if k != "check"})


if __name__ == "__main__":
    main()
