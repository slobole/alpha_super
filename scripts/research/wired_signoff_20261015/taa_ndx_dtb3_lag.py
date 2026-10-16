"""Bound a one-XNYS-session DTB3 availability delay on frozen TAA tables."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import exchange_calendars as calendar_module
import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH))
from strategies.taa_df import strategy_taa_df as base_module
from strategies.taa_df import strategy_taa_df_btal_fallback_tqqq_vix_cash as rank_module
from strategies.taa_df import strategy_taa_df_btal_1n_fallback_tqqq_vix_cash as equal_module

OUTPUT_ROOT_PATH = REPO_ROOT_PATH / "results/review/wired_signoff_20261015"
SESSION_INDEX = calendar_module.get_calendar("XNYS", start="2010-01-01", end="2027-01-01").sessions.tz_localize(None)


def hash_dict(path_obj: Path) -> dict:
    return {"path": str(path_obj.relative_to(REPO_ROOT_PATH)).replace("\\", "/"),
        "bytes": path_obj.stat().st_size, "sha256": hashlib.sha256(path_obj.read_bytes()).hexdigest()}


def delay_table(observation_ser: pd.Series) -> pd.DataFrame:
    """A dated observation is assumed available at the first strictly later XNYS session."""
    # *** CRITICAL *** observation dates are not publication timestamps. This
    # diagnostic shifts to the NEXT exchange session, including holiday gaps;
    # it is an assumed lag bound, never reconstructed archived FRED vintages.
    delayed_df = pd.DataFrame({"observation_date": observation_ser.index, "rate": observation_ser.to_numpy()})
    delayed_df["availability_date"] = SESSION_INDEX[SESSION_INDEX.searchsorted(observation_ser.index, side="right")]
    return delayed_df.sort_values("observation_date").groupby("availability_date", sort=True).last()


def hurdle_series(rate_ser: pd.Series) -> pd.Series:
    # Literal canonical formula h_T=(1+DTB3_T/100)^(1/12)-1.
    return ((1 + rate_ser.astype(float) / 100) ** (1 / 12) - 1).rename("cash_return")


def main() -> None:
    replay_path = OUTPUT_ROOT_PATH / "taa_ndx_native/taa_actual_host_replay.json"
    replay_dict = json.loads(replay_path.read_text(encoding="utf-8"))
    fred_path = OUTPUT_ROOT_PATH / "inputs/DTB3.csv"
    input_path_list = [replay_path, fred_path]
    price_map_dict = {}
    for row_dict in replay_dict["native_input_manifest_list"]:
        argument_tuple, keyword_pair_list = ast.literal_eval(row_dict["request_str"])
        if (dict(keyword_pair_list)["adjustment_str"] == "TOTALRETURN"
                and argument_tuple[0] in rank_module.DEFAULT_CONFIG.defensive_asset_list):
            input_path = REPO_ROOT_PATH / row_dict["path_str"]
            if hash_dict(input_path)["sha256"] != row_dict["sha256_str"]:
                raise ValueError("Captured raw TAA price changed.")
            input_path_list.append(input_path)
            price_map_dict[argument_tuple[0]] = pd.read_parquet(input_path)["Close"].astype(float)
    if set(price_map_dict) != set(rank_module.DEFAULT_CONFIG.defensive_asset_list):
        raise ValueError("Expected exactly the five frozen defensive TR price frames.")
    signal_close_df = pd.DataFrame(price_map_dict).sort_index()
    fred_df = pd.read_csv(fred_path, index_col=0, parse_dates=True)
    dtb3_rate_ser = pd.to_numeric(fred_df["DTB3"], errors="coerce").dropna()
    dtb3_rate_ser = dtb3_rate_ser.loc["2010-01-01":"2026-10-14"]
    baseline_hurdle_ser = hurdle_series(dtb3_rate_ser)
    availability_df = delay_table(dtb3_rate_ser)
    delayed_hurdle_ser = hurdle_series(availability_df["rate"])
    calendar_control_dict = {"2024-01-05": "2024-01-08", "2024-05-24": "2024-05-28", "2024-03-28": "2024-04-01"}
    for observation_str, expected_str in calendar_control_dict.items():
        actual_ts = SESSION_INDEX[SESSION_INDEX.searchsorted(pd.Timestamp(observation_str), side="right")]
        if actual_ts != pd.Timestamp(expected_str):
            raise AssertionError("XNYS weekend/holiday control failed.")
    complete_period = pd.Timestamp("2026-09-30")
    variant_result_dict = {}
    for variant_str, config_obj in (("taa_rank_tqqq", rank_module.DEFAULT_CONFIG), ("taa_equal_tqqq", equal_module.DEFAULT_CONFIG)):
        momentum_score_df, baseline_weight_df = base_module.compute_month_end_weight_df(signal_close_df, baseline_hurdle_ser, config_obj)
        _, delayed_weight_df = base_module.compute_month_end_weight_df(signal_close_df, delayed_hurdle_ser, config_obj)
        baseline_weight_df = baseline_weight_df.loc[:complete_period]
        delayed_weight_df = delayed_weight_df.loc[baseline_weight_df.index]
        weight_delta_df = delayed_weight_df - baseline_weight_df
        flipped_index = weight_delta_df.index[(weight_delta_df.abs() > 1e-12).any(axis=1)]
        margin_row_list = []
        for decision_ts in baseline_weight_df.index:
            observation_ts = dtb3_rate_ser.loc[:decision_ts].index[-1]
            availability_ts = availability_df.loc[:decision_ts].index[-1]
            delayed_observation_ts = availability_df.loc[availability_ts, "observation_date"]
            for asset_str in config_obj.defensive_asset_list:
                score_float = float(momentum_score_df.loc[decision_ts, asset_str])
                for basis_str, hurdle_float, source_ts in (
                        ("canonical_observation_date", float(baseline_hurdle_ser.loc[observation_ts]), observation_ts),
                        ("one_XNYS_session_delay", float(delayed_hurdle_ser.loc[availability_ts]), delayed_observation_ts)):
                    margin_row_list.append({"decision_month_str": str(decision_ts.date()), "asset_str": asset_str,
                        "basis_str": basis_str, "momentum_score_float": score_float, "hurdle_float": hurdle_float,
                        "signed_margin_float": score_float - hurdle_float,
                        "observation_date_str": str(pd.Timestamp(source_ts).date())})
        nearest_dict = {basis_str: min((row_dict for row_dict in margin_row_list if row_dict["basis_str"] == basis_str),
            key=lambda row_dict: abs(row_dict["signed_margin_float"])) for basis_str in ("canonical_observation_date", "one_XNYS_session_delay")}
        variant_result_dict[variant_str] = {"complete_month_count_int": len(baseline_weight_df),
            "first_month_str": str(baseline_weight_df.index[0].date()), "last_month_str": str(baseline_weight_df.index[-1].date()),
            "base_target_changed_months_int": len(flipped_index),
            "max_abs_weight_change_float": float(weight_delta_df.abs().to_numpy().max()),
            "nearest_score_hurdle_margin_dict": nearest_dict,
            "changed_month_list": [{"month_str": str(decision_ts.date()),
                "canonical_weights_dict": baseline_weight_df.loc[decision_ts].to_dict(),
                "delayed_weights_dict": delayed_weight_df.loc[decision_ts].to_dict()} for decision_ts in flipped_index]}
    # Positive control: Sep30's planted high hurdle is unavailable until Oct1;
    # literal canonical weight tables must flip from fallback to defenses.
    planted_index = pd.date_range("2023-01-31", "2024-09-30", freq="ME")
    planted_price_df = pd.DataFrame({asset_str: 1.01 ** np.arange(len(planted_index)) for asset_str in price_map_dict}, index=planted_index)
    planted_rate_ser = pd.Series([0., 500.], index=pd.to_datetime(["2024-09-27", "2024-09-30"]))
    planted_delayed_ser = hurdle_series(delay_table(planted_rate_ser)["rate"])
    positive_control_dict = {}
    for variant_str, config_obj in (("taa_rank_tqqq", rank_module.DEFAULT_CONFIG), ("taa_equal_tqqq", equal_module.DEFAULT_CONFIG)):
        _, original_df = base_module.compute_month_end_weight_df(planted_price_df, hurdle_series(planted_rate_ser), config_obj)
        _, delayed_df = base_module.compute_month_end_weight_df(planted_price_df, planted_delayed_ser, config_obj)
        detected_bool = bool((original_df.loc["2024-09-30"] - delayed_df.loc["2024-09-30"]).abs().max() > .9)
        if not detected_bool:
            raise AssertionError("Planted availability leak was not detected.")
        positive_control_dict[variant_str] = {"detected_bool": detected_bool, "canonical_fallback_float": float(original_df.loc["2024-09-30", "TQQQ"]),
            "delayed_fallback_float": float(delayed_df.loc["2024-09-30", "TQQQ"])}
    source_path_list = [Path(__file__), Path(base_module.__file__), Path(rank_module.__file__), Path(equal_module.__file__)]
    result_dict = {"schema_version_str": "wired_signoff.dtb3_lag.v1", "status_str": "pass",
        "generated_at_utc_str": datetime.now(UTC).isoformat(), "variant_result_dict": variant_result_dict,
        "formula_str": "m_i,T=mean(C_TR_i,T/C_TR_i,T-L-1),L={1,3,6,12}months; h_T=(1+DTB3_T/100)^(1/12)-1; keep iff m>h",
        "availability_assumption_str": "Each observation dated d becomes available on first XNYS session strictly after d. Multiple bank observations mapping to one session keep latest observation. Months through2026-09 only; no partialOctober. This bounds one-session publication lag including weekends/holidays; it is not archived release-vintage reconstruction.",
        "scope_str": "Both canonical pre-VIX month-end momentum target tables. A zero base-target difference guarantees zero final VIX-gated target difference for the same helper inputs. No engine/backtest/new strategy search. Linearity has no DTB3 input.",
        "accepted_gap_str": "G-015 current-vintage/publication approximation remains an explicit model assumption; unchanged sampled targets do not establish historical availability or bound larger revisions/lags.",
        "source_line_list": ["strategies/taa_df/strategy_taa_df.py:183-190", "strategies/taa_df/strategy_taa_df.py:294-307,315-325", "ASSUMPTIONS_AND_GAPS.md:G-015"],
        "calendar_positive_control_dict": calendar_control_dict, "planted_positive_control_dict": positive_control_dict,
        "input_manifest_list": [hash_dict(path_obj) for path_obj in input_path_list],
        "source_manifest_list": [hash_dict(path_obj) for path_obj in source_path_list]}
    output_path = OUTPUT_ROOT_PATH / "taa_ndx_dtb3_lag.json"
    output_path.write_text(json.dumps(result_dict, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output_str": str(output_path), "variants_dict": variant_result_dict, "positive_controls_dict": positive_control_dict}))


if __name__ == "__main__":
    main()
