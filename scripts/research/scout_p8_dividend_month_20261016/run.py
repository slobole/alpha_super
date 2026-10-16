"""P8, first discovery family: the dividend month premium through S1-S6 (registered as ledger row 52 before any result).

S1 causality of the four predicted-payer features and the membership check; S3 edge study on total returns with the
Nasdaq-100 replication; S4-S6 through the re-audition runner (alpha/scout/reaudit.py) with MCPT kind "panel" and the
registered fixed-calendar null (alpha/scout/specs/dividend_month.py); the T-bill slot test as a 20% sleeve of the live
60/40 book. Writes the bundle, the card, and one station_verdict row per station plus test_result rows (S3 p, MCPT p)
to the ledger.

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/scout_p8_dividend_month_20261016/run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.card import grade_str
from alpha.scout.family import ndx_vxn_family, taa_3x_family
from alpha.scout.fdr_report import record_station_verdict, record_test_result
from alpha.scout.ledger import Ledger
from alpha.scout.metrics import tbill_daily_ser
from alpha.scout.reaudit import PodPlan, factor_daily_df, reaudit
from alpha.scout.specs import dividend_month as dm

S3_NOTE_STR = ("Class X: predicted payers (ex-dividend in m-3 or m-12) at each month-end decision, month-m total return "
               "(entry next open, 21 sessions) minus the same-date eligible mean, one date per month, date-level Newey-West t. "
               "Total returns: the panel's dividend is moved to its ex-date (the column is booked on the last cum-dividend "
               "session). S3's hard and soft criteria count toward the grade.")
SLEEVE_BOOK_WEIGHT_DICT = {"TAA 3x": 0.48, "NDX VXN": 0.32, dm.NAME_STR: 0.20}


def s3_bundle() -> tuple[dict, object]:
    from alpha.scout.panel import load_panel
    from alpha.scout.specs.sector_ibs import s3_result
    from alpha.scout.stations.s1_causality import membership_integrity, run_s1
    from alpha.scout.stations.s3_edge import add_replication

    panel = load_panel(dm.PANEL_NAME_STR)
    s1_list = run_s1([dm.predicted_feature(rule_str) for rule_str in dm.LOOKBACK_RULE_TUPLE], panel)
    membership_dict = membership_integrity(panel)
    report = s3_result(dm.s3_inputs(panel))["edge_report"]
    replication_report = s3_result(dm.s3_inputs(load_panel(dm.REPLICATION_PANEL_NAME_STR), panel_name_str=dm.REPLICATION_PANEL_NAME_STR))["edge_report"]
    add_replication(report, [replication_report])
    headline = report.headline_dict
    detail_str = (f"h {report.horizon_int}: {headline['events_int']} events on {headline['event_dates_int']} dates, date-mean excess "
                  f"{headline['date_mean_excess_float'] * 1e4:+.1f} bp, NW t {headline['nw_t_float']:.2f}, cost coverage "
                  f"{headline['cost_coverage_float']:.2f}, placebo p {headline['placebo_p_float']:.3f}")
    check_list = [(name_str, "PASS" if ok_bool else ("FAIL" if level_str == "hard" else "WARN"), detail_str)
                  for name_str, ok_bool, level_str in report.check_list]
    s1_dict = {
        "features": [(r.feature_name_str, r.prefix_pass_bool, r.split_pass_bool, r.prefix_detail_str) for r in s1_list],
        "hard_fail_bool": any(r.hard_fail_bool for r in s1_list) or not membership_dict["pass_bool"],
        "membership": membership_dict,
    }
    return {"check_list": check_list, "verdict_str": report.verdict_str, "edge_report": report,
            "replication_report": replication_report, "s1": s1_dict}, report


def main() -> None:
    from alpha.scout.panel import load_panel
    from alpha.scout.specs import ndx_vxn, taa_3x

    started_float = time.time()
    ledger = Ledger()
    s3_dict, s3_report = s3_bundle()
    if s3_dict["s1"]["hard_fail_bool"]:
        raise SystemExit("S1 hard fail: the family stops here.")
    print(f"S1 PASS; S3 {s3_report.verdict_str} ({time.time() - started_float:.0f}s)", flush=True)

    signal_dict = dm.mcpt_signal_dict(load_panel(dm.PANEL_NAME_STR), dm.family().config_list())
    plan = PodPlan(
        name_str=dm.NAME_STR, family_fn=dm.family, inputs_fn=dm.load_inputs, mcpt_kind_str="panel",
        adoption_date_str="2026-10-16", prior_trial_count_int=0, slot_str="NDX VXN",
        s3_fn=lambda: ("X", S3_NOTE_STR, s3_dict), book_weight_dict=SLEEVE_BOOK_WEIGHT_DICT,
        option_dict={"panel": {"module_str": "alpha.scout.specs.dividend_month", "panel_name_str": dm.PANEL_NAME_STR,
                               "fast_kwarg_dict": {"signal_dict": signal_dict}, "warm_int": 260, "worker_count_int": 8}},
        strategy_module_str="",
    )
    taa_inputs, ndx_inputs = taa_3x.load_inputs(), ndx_vxn.load_inputs()
    live_net_dict = {pod: fam.run_config(fam.live_config_dict).daily_return_ser
                     for pod, fam in (("TAA 3x", taa_3x_family(taa_inputs)), ("NDX VXN", ndx_vxn_family(ndx_inputs)))}
    full_index = ndx_inputs.close_df.index.union(taa_inputs.open_df.index)
    tbill_ser = tbill_daily_ser(taa_inputs.dtb3_ser, full_index)
    bundle = reaudit(plan, live_net_dict, factor_daily_df(full_index, tbill_ser), tbill_ser)
    grade = grade_str(bundle)
    print(f"\n== {dm.NAME_STR}: {grade} ({time.time() - started_float:.0f}s)", flush=True)

    registration_id_str = dm.REGISTRATION_ID_STR
    if any(row.get("registration_id_str") == registration_id_str for row in ledger.rows("station_verdict")):
        print("Station verdicts already recorded for this registration: the first run is final; nothing written.")
        return
    headline = s3_report.headline_dict
    record_test_result(ledger, registration_id_str, "s3_date_level_nw", "S3", headline["nw_one_sided_p_float"],
                       {"date_mean_excess_float": headline["date_mean_excess_float"], "nw_t_float": headline["nw_t_float"],
                        "event_dates_int": headline["event_dates_int"]}, primary_bool=True,
                       source_str="results/scout/reaudition/Dividend_month_premium/bundle.pkl")
    for component in bundle["s5"].mcpt_list if hasattr(bundle["s5"], "mcpt_list") else []:
        record_test_result(ledger, registration_id_str, f"s5_mcpt_{component.name_str.replace(' ', '_')}", "S5", component.p_value_float,
                           {"observed_float": component.observed_float, "permutations_int": int(component.null_score_vec.size)},
                           primary_bool=False, source_str="results/scout/reaudition/Dividend_month_premium/summary.json")
    for station_str, check_list in (("S1", [("causality and membership", "PASS", "4 features prefix/split invariant")]),
                                    ("S3", s3_dict["check_list"]), ("S4", bundle["s4"].check_list),
                                    ("S5", bundle["s5"].check_list), ("S6", bundle["s6"].check_list)):
        record_station_verdict(ledger, registration_id_str, station_str, check_list)
        for row_name_str, verdict_str, detail_str in check_list:
            print(f"  {station_str} {verdict_str:8s} {row_name_str}: {detail_str[:150]}", flush=True)
    record_station_verdict(ledger, registration_id_str, "grade", [("grade", grade, "S0-S6 (card results/scout/cards)")])
    print(f"ledger chain OK: {ledger.verify()} rows")


if __name__ == "__main__":
    main()
