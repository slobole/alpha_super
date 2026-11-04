"""SPX source audit and exact before/after replay of the 2026-11-04 validation fix."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from strategies.vix_stuff import strategy_vrp_svxy_vixm_rsi2 as strategy_module


def main() -> None:
    evidence_path = REPO_ROOT_PATH / "results/research/vrp_spx_validation_20261104"
    baseline_spec_obj = importlib.util.spec_from_file_location("vrp_spx_before", evidence_path / "strategy_before.py")
    baseline_module = importlib.util.module_from_spec(baseline_spec_obj)
    sys.modules[baseline_spec_obj.name] = baseline_module
    baseline_spec_obj.loader.exec_module(baseline_module)
    config_obj = strategy_module.VrpRsi2Config(end_date_str="2026-10-30")
    pricing_path = evidence_path / "pricing_data.pkl"
    pricing_data_df = pd.read_pickle(pricing_path) if pricing_path.exists() else strategy_module.get_vrp_data(config_obj)
    if not pricing_path.exists():
        pricing_data_df.to_pickle(pricing_path)
    signal_df = strategy_module.signal_frame_df(pricing_data_df)
    spx_close_ser = signal_df["spx_close"].astype(float)
    invalid_spx_ser = ~np.isfinite(spx_close_ser) | (spx_close_ser <= 0.0)
    invalid_date_list = [str(date_ts.date()) for date_ts in spx_close_ser.index[invalid_spx_ser]]
    # Audit unpadded source bars too: the normal loader pads ALLMARKETDAYS.
    raw_spx_path = evidence_path / "spx_unpadded.csv"
    if raw_spx_path.exists():
        raw_spx_df = pd.read_csv(raw_spx_path, index_col=0, parse_dates=True)
    else:
        import norgatedata
        raw_spx_df = norgatedata.price_timeseries(
            "$SPX", stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.NONE,
            padding_setting=norgatedata.PaddingType.NONE,
            start_date=config_obj.history_start_date_str, end_date=config_obj.end_date_str,
            timeseriesformat="pandas-dataframe",
        )
        raw_spx_df.to_csv(raw_spx_path)
    source_close_ser = raw_spx_df["Close"].reindex(signal_df.index).astype(float)
    source_invalid_ser = ~np.isfinite(source_close_ser) | (source_close_ser <= 0.0)
    source_invalid_date_list = [str(date_ts.date()) for date_ts in source_close_ser.index[source_invalid_ser]]
    audit_dict = {
        "history_start_str": str(signal_df.index[0].date()),
        "history_end_str": str(signal_df.index[-1].date()),
        "history_row_int": len(signal_df),
        "loaded_invalid_spx_date_list": invalid_date_list,
        "unpadded_invalid_or_missing_spx_date_list": source_invalid_date_list,
    }
    print(json.dumps(audit_dict), flush=True)
    (evidence_path / "source_audit.json").write_text(json.dumps(audit_dict, indent=2), encoding="utf-8")
    assert not invalid_date_list, "Historical loaded SPX data is invalid; cannot claim unchanged replay."
    calendar_idx = strategy_module.build_execution_calendar_idx(pricing_data_df, config_obj)
    first_decision_ts = strategy_module.first_decision_date_ts(pricing_data_df.index, config_obj)
    baseline_feature_df = baseline_module.compute_feature_df(signal_df, config_obj, first_decision_ts)
    corrected_feature_df = strategy_module.compute_feature_df(signal_df, config_obj, first_decision_ts)
    pd.testing.assert_frame_equal(baseline_feature_df, corrected_feature_df, check_exact=True)
    print("Features exactly equal; replaying pre-fix and corrected engines.", flush=True)
    baseline_strategy_obj = baseline_module.VrpRsi2Strategy(config_obj)
    corrected_strategy_obj = strategy_module.VrpRsi2Strategy(config_obj)
    for label_str, replay_strategy_obj in (("before", baseline_strategy_obj), ("after", corrected_strategy_obj)):
        run_daily(replay_strategy_obj, pricing_data_df.copy(), calendar=calendar_idx,
                  show_progress=False, show_signal_progress_bool=False)
        replay_strategy_obj.results.to_pickle(evidence_path / f"{label_str}_results.pkl")
        replay_strategy_obj.decision_df.to_csv(evidence_path / f"{label_str}_decisions.csv", index=False)
        print(f"{label_str} replay complete", flush=True)
    pd.testing.assert_frame_equal(baseline_strategy_obj.results, corrected_strategy_obj.results, check_exact=True)
    pd.testing.assert_frame_equal(baseline_strategy_obj.decision_df, corrected_strategy_obj.decision_df, check_exact=True)
    audit_dict.update({
        "features_exact_bool": True, "results_exact_bool": True, "decisions_exact_bool": True,
        "scoring_session_int": len(calendar_idx),
        "final_equity_float": float(corrected_strategy_obj.results["total_value"].iloc[-1]),
        "decision_row_int": len(corrected_strategy_obj.decision_df),
    })
    (evidence_path / "verification.json").write_text(json.dumps(audit_dict, indent=2), encoding="utf-8")
    print(json.dumps(audit_dict, indent=2), flush=True)


if __name__ == "__main__":
    main()
