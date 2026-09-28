"""Tactical FI (L14): frozen contract versus ALFRED point-in-time replay.

Runs, on the same Norgate prices and the same causal cash accrual:

1. ``frozen_current_vintage``: the governed 289-row contract.
2. ``alfred_pit_decision_date``: decisions from 2014-04-30 recomputed from the
   ALFRED vintage dated T, with the stale-input rule (block and hold).
3. ``alfred_pit_previous_session``: the same with the vintage of session T-1
   (conservative about the publication time of day).
4. ``leakage_hunt_reproduction``: the 2026-09-27 leakage hunt's faithful mode
   (vintage T for the last 45 calendar days, current-vintage history before
   that, no stale rule). It exists only to prove that this snapshot reproduces
   the study's 2 flipped decisions and its metrics. It is NOT point in time:
   during the 2016-17 Moody's outage its history still holds later backfill.

Outputs go to results/research/tactical_fi_alfred_pit_20260928/ (gitignored):
metrics.csv, decision_comparison.csv and summary.json.

    uv run python scripts/research/run_tactical_fi_alfred_replay.py
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from strategies.taa_beyond_6040 import (  # noqa: E402
    strategy_taa_tactical_fixed_income_ief_lqd as tactical_module,
)


OUTPUT_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "tactical_fi_alfred_pit_20260928"
BOOK_START_DATE_STR = "2012-10-02"
BOOK_END_DATE_STR = "2026-08-19"
LEAKAGE_HUNT_WINDOW_DAYS_INT = 45
METRIC_WINDOW_DICT = {
    "full": (None, None),
    "book": (BOOK_START_DATE_STR, BOOK_END_DATE_STR),
    "since_2014-05": ("2014-05-01", BOOK_END_DATE_STR),
}


def return_metric_dict(
    daily_return_ser: pd.Series,
    start_date_str: str | None,
    end_date_str: str | None,
) -> dict[str, object]:
    """Same definitions as the leakage hunt (def_common.metrics_from_returns).

    CAGR = equity_end^(365.25 / calendar_days) - 1; Sharpe uses rf = 0,
    daily mean / daily std (ddof=1) * sqrt(252).
    """
    return_ser = daily_return_ser.astype(float).dropna()
    if start_date_str is not None:
        return_ser = return_ser[return_ser.index >= pd.Timestamp(start_date_str)]
    if end_date_str is not None:
        return_ser = return_ser[return_ser.index <= pd.Timestamp(end_date_str)]
    equity_ser = (1.0 + return_ser).cumprod()
    year_count_float = (return_ser.index[-1] - return_ser.index[0]).days / 365.25
    daily_std_float = float(return_ser.std(ddof=1))
    return {
        "cagr": float(equity_ser.iloc[-1] ** (1.0 / year_count_float) - 1.0),
        "sharpe": float(return_ser.mean() / daily_std_float * np.sqrt(252.0)),
        "vol": daily_std_float * float(np.sqrt(252.0)),
        "maxdd": float((equity_ser / equity_ser.cummax() - 1.0).min()),
        "n": int(len(return_ser)),
        "start": return_ser.index[0].date().isoformat(),
        "end": return_ser.index[-1].date().isoformat(),
    }


def leakage_hunt_reproduction_weight_df(
    frozen_yield_df: pd.DataFrame,
    frozen_signal_df: pd.DataFrame,
    frozen_weight_df: pd.DataFrame,
    alfred_snapshot_by_series_dict: dict,
    session_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the study's ``vintage_T__history_current`` mode from the snapshot.

    For each decision T >= 2014-04-30 and v = T: observations dated in
    (v - 45 days, v] come from vintage v; older ones from the current vintage.
    The rule is recomputed on that panel and row T replaces the frozen target.
    """
    signal_row_list: list[pd.Series] = []
    reproduction_weight_df = frozen_weight_df.copy()
    first_decision_ts = pd.Timestamp(tactical_module.FIRST_ALFRED_DECISION_DATE_STR)
    for decision_date_ts in frozen_signal_df.index[frozen_signal_df.index >= first_decision_ts]:
        decision_date_ts = pd.Timestamp(decision_date_ts)
        window_start_ts = decision_date_ts - pd.Timedelta(days=LEAKAGE_HUNT_WINDOW_DAYS_INT)
        part_list = []
        for series_id_str in tactical_module.FRED_SERIES_ID_TUPLE:
            vintage_value_ser = alfred_snapshot_by_series_dict[series_id_str].value_ser_as_of(
                decision_date_ts
            )
            window_value_ser = vintage_value_ser[vintage_value_ser.index > window_start_ts]
            current_value_ser = frozen_yield_df[series_id_str].dropna()
            history_value_ser = current_value_ser[current_value_ser.index <= window_start_ts]
            part_list.append(
                pd.concat([history_value_ser, window_value_ser]).sort_index().rename(series_id_str)
            )
        panel_df = pd.concat(part_list, axis=1).sort_index().loc[:decision_date_ts]
        panel_signal_df, _panel_weight_df = tactical_module.build_month_end_signal_and_weight_df(
            yield_df=panel_df,
            session_index=session_index,
            last_complete_signal_month_str=str(decision_date_ts.to_period("M")),
        )
        signal_row_ser = panel_signal_df.loc[decision_date_ts]
        signal_row_list.append(signal_row_ser)
        rebalance_date_ts = reproduction_weight_df.index[
            reproduction_weight_df["decision_date"] == decision_date_ts
        ][0]
        ief_weight_float = 0.5 * float(signal_row_ser["term_state_float"])
        lqd_weight_float = 0.5 * float(signal_row_ser["credit_state_float"])
        reproduction_weight_df.loc[rebalance_date_ts, ["IEF", "LQD", "Cash"]] = [
            ief_weight_float,
            lqd_weight_float,
            1.0 - ief_weight_float - lqd_weight_float,
        ]
    reproduction_signal_df = pd.DataFrame(signal_row_list)
    reproduction_signal_df.index = pd.DatetimeIndex(
        reproduction_signal_df.index,
        name="decision_date",
    )
    return reproduction_signal_df, reproduction_weight_df


def flipped_decision_df(
    frozen_signal_df: pd.DataFrame,
    candidate_signal_df: pd.DataFrame,
) -> pd.DataFrame:
    common_index = candidate_signal_df.index.intersection(frozen_signal_df.index)
    frozen_df = frozen_signal_df.loc[common_index]
    candidate_df = candidate_signal_df.loc[common_index]
    flip_bool_ser = (frozen_df["term_state_float"] != candidate_df["term_state_float"]) | (
        frozen_df["credit_state_float"] != candidate_df["credit_state_float"]
    )
    return pd.DataFrame(
        {
            "observation_date_frozen": frozen_df["observation_date"],
            "observation_date_candidate": candidate_df["observation_date"],
            "term_state_frozen": frozen_df["term_state_float"],
            "term_state_candidate": candidate_df["term_state_float"],
            "credit_state_frozen": frozen_df["credit_state_float"],
            "credit_state_candidate": candidate_df["credit_state_float"],
        }
    ).loc[flip_bool_ser]


def run_backtest(
    config_obj: tactical_module.TacticalYieldConfig,
    execution_price_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    cash_return_ser: pd.Series,
    fred_snapshot_tuple: tuple,
) -> tactical_module.TacticalYieldStrategy:
    return tactical_module._run_strategy(
        config_obj=config_obj,
        execution_price_df=execution_price_df,
        signal_df=signal_df,
        rebalance_weight_df=weight_df,
        cash_return_ser=cash_return_ser,
        fred_snapshot_tuple=fred_snapshot_tuple,
        backtest_start_date_str="2002-08-01",
        end_date_str=None,
        show_progress_bool=False,
    )


def main() -> int:
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    frozen_config_obj = tactical_module.DEFAULT_CONFIG
    (
        execution_price_df,
        frozen_yield_df,
        frozen_signal_df,
        frozen_weight_df,
        cash_return_ser,
        fred_snapshot_tuple,
    ) = tactical_module.get_tactical_yield_data(frozen_config_obj)
    session_index = pd.DatetimeIndex(execution_price_df.index)
    _manifest_dict, alfred_snapshot_by_series_dict = (
        tactical_module.load_alfred_point_in_time_snapshots(frozen_config_obj)
    )

    variant_dict: dict[str, tuple] = {
        "frozen_current_vintage": (frozen_config_obj, frozen_signal_df, frozen_weight_df),
    }
    for vintage_policy_str in tactical_module.SUPPORTED_ALFRED_VINTAGE_POLICY_TUPLE:
        pit_config_obj = replace(
            frozen_config_obj,
            fred_data_mode_str=tactical_module.FRED_DATA_MODE_ALFRED_PIT_STR,
            alfred_vintage_policy_str=vintage_policy_str,
        )
        (
            _price_df,
            _yield_df,
            pit_signal_df,
            pit_weight_df,
            _cash_ser,
            _snapshot_tuple,
        ) = tactical_module.get_tactical_yield_data(pit_config_obj)
        variant_dict[f"alfred_pit_{vintage_policy_str}"] = (
            pit_config_obj,
            pit_signal_df,
            pit_weight_df,
        )
    reproduction_signal_df, reproduction_weight_df = leakage_hunt_reproduction_weight_df(
        frozen_yield_df,
        frozen_signal_df,
        frozen_weight_df,
        alfred_snapshot_by_series_dict,
        session_index,
    )
    variant_dict["leakage_hunt_reproduction"] = (
        frozen_config_obj,
        frozen_signal_df,
        reproduction_weight_df,
    )

    metric_row_list: list[dict[str, object]] = []
    for variant_str, (config_obj, signal_df, weight_df) in variant_dict.items():
        strategy_obj = run_backtest(
            config_obj,
            execution_price_df,
            signal_df,
            weight_df,
            cash_return_ser,
            fred_snapshot_tuple,
        )
        for window_str, (start_date_str, end_date_str) in METRIC_WINDOW_DICT.items():
            metric_row_list.append(
                {
                    "variant": variant_str,
                    "window": window_str,
                    **return_metric_dict(
                        strategy_obj.results["daily_returns"],
                        start_date_str,
                        end_date_str,
                    ),
                }
            )
        print(variant_str, "done", flush=True)
    metric_df = pd.DataFrame(metric_row_list)
    metric_df.to_csv(OUTPUT_DIR_PATH / "metrics.csv", index=False)

    comparison_df_list = []
    summary_dict: dict[str, object] = {}
    candidate_signal_dict = {
        "alfred_pit_decision_date": variant_dict["alfred_pit_decision_date"][1],
        "alfred_pit_previous_session": variant_dict["alfred_pit_previous_session"][1],
        "leakage_hunt_reproduction": reproduction_signal_df,
    }
    for variant_str, candidate_signal_df in candidate_signal_dict.items():
        if "stale_input_blocked_bool" in candidate_signal_df.columns:
            blocked_index = candidate_signal_df.index[candidate_signal_df["stale_input_blocked_bool"]]
            unblocked_signal_df = candidate_signal_df.loc[
                ~candidate_signal_df["stale_input_blocked_bool"]
            ]
        else:
            blocked_index = pd.DatetimeIndex([])
            unblocked_signal_df = candidate_signal_df
        replayed_signal_df = unblocked_signal_df.loc[
            unblocked_signal_df.index >= pd.Timestamp(tactical_module.FIRST_ALFRED_DECISION_DATE_STR)
        ]
        flip_df = flipped_decision_df(frozen_signal_df, replayed_signal_df)
        comparison_df_list.append(flip_df.assign(variant=variant_str))
        summary_dict[variant_str] = {
            "decisions_replayed_int": int(len(replayed_signal_df)),
            "stale_blocked_decision_list": [
                pd.Timestamp(date_ts).date().isoformat() for date_ts in blocked_index
            ],
            "flipped_decision_list": [
                pd.Timestamp(date_ts).date().isoformat() for date_ts in flip_df.index
            ],
        }
    pd.concat(comparison_df_list).to_csv(OUTPUT_DIR_PATH / "decision_comparison.csv")
    for variant_str in ("alfred_pit_decision_date", "alfred_pit_previous_session"):
        variant_dict[variant_str][1].to_csv(OUTPUT_DIR_PATH / f"{variant_str}_signal_table.csv")
    (OUTPUT_DIR_PATH / "summary.json").write_text(
        json.dumps(summary_dict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(metric_df.to_string(index=False))
    print(json.dumps(summary_dict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
