"""Re-run the owner's two Pakal edge notebooks through Scout S1-S3 and compare with the notebook method.

Notebooks (Pakal/pakal-research): edge_qpi_indicator_qpi_mimush.ipynb and edge_dvo2_sp500.ipynb.
    QPI study: regime Close > SMA200 and 3-day return < 0; event QPI(3, 5y) < 15; hold 5 sessions.
    DV2 study: regime 126-day return > 0 and Close > SMA200; event DV2(126) < 10; hold 5 sessions.
    Forward return (both): Close(t+5) / Open(t+1) − 1.
    Universe: "S&P 500 Current & Past" with no membership-by-date filter.

For each study:
    notebook method  the notebook's statistics re-run on the sealed Scout panel (every panel symbol on every date,
                     membership ignored; events vs regime rows with the indicator at or above the threshold;
                     per-event Welch t). The panel holds symbols that were members from 1998 on, so it is not the
                     notebooks' exact universe (their "Current & Past" list, unpadded data through 2026-03);
    Scout S3         point-in-time membership; excess over the same-date eligible mean; date-level Newey-West t;
                     eras, liquidity, lag decay, concentration, costs;
    decomposition    Scout with the membership filter off, to separate the membership effect from the inference fix;
    non-members      the events S3 drops, split by where they fall in the stock's membership history.
Both on the sealed panel (to 2022-12-30): the vault stays closed.

    uv run python scripts/research/scout_p4_edge_rerun_20261001/run.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout import features as F
from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.panel import load_panel
from alpha.scout.stations.s1_causality import membership_integrity, run_s1
from alpha.scout.stations.s2_indicator import run_s2
from alpha.scout.stations.s3_edge import forward_return_df, run_s3

OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "p4_edge_rerun"
HORIZON_INT = 5


def notebook_method(panel, regime_mask_df, event_mask_df, indicator_df) -> dict:
    """The notebook's statistics: raw forward returns, events vs non-events in the regime, per-event Welch t.
    Non-events are regime rows whose indicator exists and is not an event (the notebook's `indicator >= threshold`)."""
    forward_df = forward_return_df(panel, HORIZON_INT)
    regime_df = regime_mask_df.fillna(False).astype(bool)
    event_df = event_mask_df.fillna(False).astype(bool) & regime_df
    event_vec = forward_df.where(event_df).stack().to_numpy()
    non_event_vec = forward_df.where(regime_df & indicator_df.notna() & ~event_df).stack().to_numpy()
    welch = stats.ttest_ind(event_vec, non_event_vec, equal_var=False)
    return {
        "events_int": int(event_vec.size),
        "event_mean_float": float(event_vec.mean()),
        "non_event_mean_float": float(non_event_vec.mean()),
        "difference_float": float(event_vec.mean() - non_event_vec.mean()),
        "welch_t_float": float(welch.statistic),
        "event_win_rate_float": float(np.mean(event_vec > 0)),
    }


def non_member_breakdown(panel, regime_mask_df, event_mask_df) -> dict:
    """Events in the regime on non-member sessions, by position in the stock's membership history (5-day raw return)."""
    forward_df = forward_return_df(panel, HORIZON_INT)
    event_df = event_mask_df.fillna(False).astype(bool) & regime_mask_df.fillna(False).astype(bool) & forward_df.notna()
    member_df = panel.member_df == 1
    date_arr = np.broadcast_to(np.arange(len(panel.date_index))[:, None], member_df.shape)
    first_arr = np.where(member_df, date_arr, np.iinfo(np.int64).max).min(axis=0)
    last_arr = np.where(member_df, date_arr, -1).max(axis=0)
    never_arr = ~member_df.to_numpy().any(axis=0)
    group_mask_dict = {
        "member": member_df.to_numpy(),
        "never a member before the seal": ~member_df.to_numpy() & never_arr[None, :],
        "before first inclusion": ~member_df.to_numpy() & ~never_arr[None, :] & (date_arr < first_arr[None, :]),
        "after last removal": ~member_df.to_numpy() & ~never_arr[None, :] & (date_arr > last_arr[None, :]),
        "gap between membership spells": ~member_df.to_numpy() & ~never_arr[None, :] & (date_arr > first_arr[None, :]) & (date_arr < last_arr[None, :]),
    }
    event_arr, forward_arr = event_df.to_numpy(), forward_df.to_numpy()
    non_member_count_int = int((event_arr & ~member_df.to_numpy()).sum())
    row_list = []
    for group_str, group_arr in group_mask_dict.items():
        value_vec = forward_arr[event_arr & group_arr]
        row_list.append({
            "group_str": group_str,
            "events_int": int(value_vec.size),
            "share_of_non_member_events_float": float(value_vec.size / non_member_count_int) if group_str != "member" else float("nan"),
            "mean_forward_return_float": float(value_vec.mean()) if value_vec.size else float("nan"),
        })
    return {"non_member_share_float": non_member_count_int / int(event_arr.sum()), "groups": row_list}


def run_study(name_str, panel, regime_mask_df, event_mask_df, indicator_df, threshold_fn, feature_list, liquidity_df) -> dict:
    started_float = time.time()
    s1_list = run_s1(feature_list, panel)
    eligible_df = regime_mask_df.fillna(False).astype(bool) & (panel.member_df == 1)
    s2_dict = run_s2(indicator_df, eligible_df, threshold_fn, panel, F.reference_feature_list())
    s3 = run_s3(name_str, panel, regime_mask_df, event_mask_df, HORIZON_INT, indicator_df, liquidity_df)
    # Decomposition: the same Scout statistics with every symbol treated as a member (the notebook's universe).
    all_member_panel = replace(panel, member_df=pd.DataFrame(1, index=panel.date_index, columns=panel.symbol_list, dtype=np.int8))
    s3_no_pit = run_s3(name_str + " (no membership filter)", all_member_panel, regime_mask_df, event_mask_df, HORIZON_INT)
    return {
        "name_str": name_str,
        "seconds_float": time.time() - started_float,
        "s1": [vars(result) | {"hard_fail_bool": result.hard_fail_bool} for result in s1_list],
        "s2": s2_dict,
        "notebook": notebook_method(panel, regime_mask_df, event_mask_df, indicator_df),
        "non_member_breakdown": non_member_breakdown(panel, regime_mask_df, event_mask_df),
        "s3": {"headline": s3.headline_dict, "tables": s3.table_dict, "checks": s3.check_list, "verdict": s3.verdict_str},
        "s3_no_pit_headline": s3_no_pit.headline_dict,
    }


def main() -> None:
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    panel = load_panel("S&P 500")
    print("panel", panel.snapshot_id_str, panel.date_index[0].date(), panel.date_index[-1].date(), flush=True)
    integrity_dict = membership_integrity(panel)
    print("membership integrity", integrity_dict, flush=True)
    close_over_sma200_df = F.close_over_sma(200).compute_fn(panel)
    return_3d_df = F.trailing_return(3).compute_fn(panel)
    return_126d_df = F.trailing_return(126).compute_fn(panel)
    liquidity_df = F.turnover_rank(63).compute_fn(panel)
    qpi_df = F.qpi(3, 5).compute_fn(panel)
    dv2_df = F.dv2(126).compute_fn(panel)
    uptrend_df = close_over_sma200_df > 0

    result_list = [
        run_study(
            "QPI pullback (QPI < 15, uptrend, 3-day return < 0)", panel,
            uptrend_df & (return_3d_df < 0), qpi_df < 15, qpi_df, lambda value_vec: value_vec < 15,
            [F.qpi(3, 5), F.trailing_return(3), F.close_over_sma(200), F.turnover_rank(63)], liquidity_df,
        ),
        run_study(
            "DV2 oversold (DV2 < 10, uptrend, 126-day return > 0)", panel,
            uptrend_df & (return_126d_df > 0), dv2_df < 10, dv2_df, lambda value_vec: value_vec < 10,
            [F.dv2(126), F.trailing_return(126), F.close_over_sma(200), F.turnover_rank(63)], liquidity_df,
        ),
    ]
    output_dict = {"panel_snapshot_str": panel.snapshot_id_str, "membership_integrity": integrity_dict, "studies": result_list}
    (OUTPUT_DIR_PATH / "edge_rerun.json").write_text(json.dumps(output_dict, indent=2, default=str), encoding="utf-8")
    for result_dict in result_list:
        headline_dict = result_dict["s3"]["headline"]
        print("\n==", result_dict["name_str"], f"({result_dict['seconds_float']:.0f}s)")
        print("notebook:", {k: round(v, 5) if isinstance(v, float) else v for k, v in result_dict["notebook"].items()})
        print("scout   :", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in headline_dict.items()})
        print("non-mem :", result_dict["non_member_breakdown"])
        print("deciles :", [(row["decile_int"], round(row["mean_float"] * 1e4, 1), round(row["t_float"], 1)) for row in result_dict["s3"]["tables"]["deciles"]])
        print("no-PIT  :", {k: round(result_dict["s3_no_pit_headline"][k], 5) for k in ("date_mean_excess_float", "nw_t_float", "naive_event_t_float")},
              "events", result_dict["s3_no_pit_headline"]["events_int"])
        print("verdict :", result_dict["s3"]["verdict"], [name for name, passed, kind in result_dict["s3"]["checks"] if not passed])
        print("S1 fails:", [row["feature_name_str"] for row in result_dict["s1"] if row["hard_fail_bool"]], "| S2 warns:", result_dict["s2"]["warn_list"])


if __name__ == "__main__":
    main()
