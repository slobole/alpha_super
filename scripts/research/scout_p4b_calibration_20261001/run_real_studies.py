"""P4b first use of the new S2/S3 pieces: the QPI and DV2 studies on the S&P 500 and on its sibling, the Nasdaq-100.

Same study definitions as `scout_p4_edge_rerun_20261001/run.py`. New here: Masters' S2 battery (with the forward
excess for mutual information), the S3 shift placebo, and the replication check across the two universes. Both
panels are sealed at the vault (to 2022-12-30).

    uv run python scripts/research/scout_p4b_calibration_20261001/run_real_studies.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout import features as F
from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.panel import load_panel
from alpha.scout.stations.s1_causality import membership_integrity
from alpha.scout.stations.s2_indicator import run_s2
from alpha.scout.stations.s3_edge import (
    add_replication,
    forward_return_df,
    run_s3,
)

OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "p4b_calibration"
HORIZON_INT = 5


def _studies(panel) -> dict:
    close_over_sma200_df = F.close_over_sma(200).compute_fn(panel)
    return_3d_df = F.trailing_return(3).compute_fn(panel)
    return_126d_df = F.trailing_return(126).compute_fn(panel)
    qpi_df, dv2_df = F.qpi(3, 5).compute_fn(panel), F.dv2(126).compute_fn(panel)
    uptrend_df = close_over_sma200_df > 0
    return {
        "QPI": (uptrend_df & (return_3d_df < 0), qpi_df < 15, qpi_df, lambda v: v < 15),
        "DV2": (uptrend_df & (return_126d_df > 0), dv2_df < 10, dv2_df, lambda v: v < 10),
    }


def main() -> None:
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    report_dict, output_dict = {}, {}
    for universe_str in ("S&P 500", "Nasdaq 100"):
        panel = load_panel(universe_str)
        liquidity_df = F.turnover_rank(63).compute_fn(panel)
        forward_df = forward_return_df(panel, HORIZON_INT)
        output_dict[universe_str] = {"snapshot_str": panel.snapshot_id_str, "membership_integrity": membership_integrity(panel)}
        print(universe_str, panel.snapshot_id_str, output_dict[universe_str]["membership_integrity"]["trimmed_share_float"], flush=True)
        for study_str, (regime_df, event_df, indicator_df, threshold_fn) in _studies(panel).items():
            eligible_df = regime_df.fillna(False).astype(bool) & (panel.member_df == 1)
            excess_df = forward_df.sub(forward_df.where(eligible_df).mean(axis=1), axis=0)
            s2_dict = run_s2(indicator_df, eligible_df, threshold_fn, panel, F.reference_feature_list(), forward_excess_df=excess_df)
            report = run_s3(f"{study_str} | {universe_str}", panel, regime_df, event_df, HORIZON_INT, indicator_df, liquidity_df)
            report_dict[(study_str, universe_str)] = report
            output_dict[universe_str][study_str] = {"s2": s2_dict, "s3_headline": report.headline_dict, "s3_tables": report.table_dict}
            print(study_str, universe_str, {k: round(v, 4) if isinstance(v, float) else v for k, v in report.headline_dict.items()}, flush=True)
            print("  S2 masters:", {k: v for k, v in s2_dict["masters"].items() if k != "warn_list"}, "| warns:", s2_dict["warn_list"], flush=True)

    for study_str in ("QPI", "DV2"):
        for universe_str, sibling_str in (("S&P 500", "Nasdaq 100"), ("Nasdaq 100", "S&P 500")):
            report = add_replication(report_dict[(study_str, universe_str)], [report_dict[(study_str, sibling_str)]])
            output_dict[universe_str][study_str]["s3_checks"] = report.check_list
            output_dict[universe_str][study_str]["s3_verdict"] = report.verdict_str
            print(study_str, universe_str, report.verdict_str, [name for name, passed, _ in report.check_list if not passed], flush=True)
    (OUTPUT_DIR_PATH / "real_studies.json").write_text(json.dumps(output_dict, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
