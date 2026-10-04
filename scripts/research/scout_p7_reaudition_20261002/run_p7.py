"""P7: re-audit the point-in-time stock event pods (DV2 S&P 500 [WIRED], DV2 Nasdaq-100, HPI) through S3-S6, with
cards and RETRO ledger registrations. MCPT kind "panel": the per-asset null of P4b on the sealed Scout panel.
Each pod is tested in NDX VXN's slot of the live 60/40 book, as in P6b/P6c.

    uv run python scripts/research/scout_p7_reaudition_20261002/run_p7.py [name ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.card import grade_str
from alpha.scout.family import ndx_vxn_family, taa_3x_family
from alpha.scout.ledger import Ledger
from alpha.scout.metrics import tbill_daily_ser
from alpha.scout.reaudit import PodPlan, factor_daily_df, reaudit
from alpha.scout.registration import Registration, register, registration_rows

EVENT_NOTE_STR = ("Class E: the rule's raw entry signal as events among eligible point-in-time members, excess over the same-date "
                  "eligible mean, date-level Newey-West t at the horizon nearest the median hold; S3's hard and soft criteria "
                  "count toward the grade. A long-only pod also earns the regime's own return: that part is judged in S5/S6.")


def plan_dict() -> dict[str, tuple[PodPlan, Registration]]:
    from alpha.scout.family import DV2_FAMILY_NAME_DICT, dv2_variant_family
    from alpha.scout.specs import dv2, sector_ibs

    out_dict = {}
    for variant_str, (label_str, grid_dict) in DV2_FAMILY_NAME_DICT.items():
        variant = dv2.VARIANT_DICT[variant_str]
        is_live_bool = variant_str == "dv2"
        out_dict[variant_str] = (
            PodPlan(
                name_str=label_str, family_fn=lambda inputs, v=variant_str: dv2_variant_family(v, inputs),
                inputs_fn=lambda v=variant_str: dv2.load_inputs(variant_name_str=v), mcpt_kind_str="panel",
                adoption_date_str="2026-04-04", prior_trial_count_int=200, slot_str="NDX VXN",
                s3_fn=lambda v=variant_str: ("E", EVENT_NOTE_STR, sector_ibs.s3_result(dv2.s3_inputs(variant_name_str=v))),
                option_dict={"panel": {"module_str": "alpha.scout.specs.dv2", "panel_name_str": variant.index_name_str,
                                       "fast_kwarg_dict": {"base_config": variant.config}}},
                strategy_module_str=variant.strategy_import_str,
            ),
            Registration(
                registration_id_str=f"{variant_str}_reaudition_20261002", family_id_str="us_equity_short_term_reversal",
                hypothesis_str=(f"{label_str}: buying uptrending {variant.index_name_str} members whose DV2 (two-day close vs mid-range, "
                                "ranked over its history) is very low, selling on a rebound close, beats the equal-weight members."),
                mechanism_str="Liquidity provision to impatient sellers in uptrending large caps (short-term reversal).",
                expected_sign_and_location_str="Positive active return over the equal-weight members, concentrated in stress.",
                hypothesis_class_str="E", universe_str=f"{variant.index_name_str} point-in-time members", horizon_str="days",
                schedule_str="daily close decision", execution_str="next session's open",
                param_grid_dict={k: tuple(v) for k, v in grid_dict.items()}, primary_metric_str="S5 MCPT (per-asset null); S6 T-bill slot",
                kill_criteria_str="S8 CUSUM or Cold Blood Index red", source_str=variant.strategy_import_str, retro_bool=True,
                prior_trials_int=200, parent_id_str=None if is_live_bool else "dv2_reaudition_20261002",
                universe_choice_str=("The S&P 500 was the pod's universe from the start; the 2026-09-25 deep research studied many "
                                     "variants (counted: 200, an estimate above the 50 floor)." if is_live_bool else
                                     "Nasdaq-100 as a sibling universe; P4b's S3 found the signal stronger there before this run."),
                universe_chosen_after_results_bool=not is_live_bool,
            ),
        )
    from alpha.scout.family import HPI_FAMILY_NAME_DICT, HPI_GRID_DICT, hpi_family
    from alpha.scout.specs import hpi

    for variant_str, label_str in HPI_FAMILY_NAME_DICT.items():
        variant = hpi.VARIANT_DICT[variant_str]
        is_live_bool = variant_str == "hpi_vote"
        out_dict[variant_str] = (
            PodPlan(
                name_str=label_str, family_fn=lambda inputs, v=variant_str: hpi_family(v, inputs), inputs_fn=hpi.load_inputs,
                mcpt_kind_str="panel", adoption_date_str="2026-04-04", prior_trial_count_int=100, slot_str="NDX VXN",
                s3_fn=lambda v=variant_str: ("E", EVENT_NOTE_STR, sector_ibs.s3_result(hpi.s3_inputs(v))),
                option_dict={"panel": {"module_str": "alpha.scout.specs.hpi", "panel_name_str": "S&P 500",
                                       "fast_kwarg_dict": {"base_config": variant.config}, "worker_count_int": 6}},
                strategy_module_str=variant.strategy_import_str,
            ),
            Registration(
                registration_id_str=f"{variant_str}_reaudition_20261002", family_id_str="us_equity_short_term_reversal",
                hypothesis_str=(f"{label_str}: buying uptrending S&P 500 members whose recent pullback is rare against their own "
                                "five-year history (HPI) and whose close sits low in the day's range, selling on a rebound close, "
                                "beats the equal-weight members."),
                mechanism_str="Liquidity provision to impatient sellers in uptrending large caps (short-term reversal).",
                expected_sign_and_location_str="Positive active return over the equal-weight members, concentrated in stress.",
                hypothesis_class_str="E", universe_str="S&P 500 point-in-time members", horizon_str="days",
                schedule_str="daily close decision", execution_str="next session's open (exit slots refilled in the same open)",
                param_grid_dict={k: tuple(v) for k, v in HPI_GRID_DICT.items()}, primary_metric_str="S5 MCPT (per-asset null); S6 T-bill slot",
                kill_criteria_str="S8 CUSUM or Cold Blood Index red", source_str=variant.strategy_import_str, retro_bool=True,
                prior_trials_int=100, parent_id_str=None if is_live_bool else "hpi_vote_reaudition_20261002",
                universe_choice_str="The S&P 500 was the pods' universe from the start; HPI variants documented in strategies/hpi and "
                                    "Pakal (counted: 100, an estimate above the 50 floor).",
                universe_chosen_after_results_bool=False,
            ),
        )
    return out_dict


def main() -> None:
    from alpha.scout.specs import ndx_vxn, taa_3x

    plans = plan_dict()
    name_list = sys.argv[1:] or list(plans)
    ledger = Ledger()
    for name_str in name_list:  # parents first: dv2 is registered before dv2_ndx
        registration = plans[name_str][1]
        if registration.registration_id_str not in registration_rows(ledger):
            register(ledger, registration)
            print("registered", registration.registration_id_str, flush=True)
    started_float = time.time()
    taa_inputs, ndx_inputs = taa_3x.load_inputs(), ndx_vxn.load_inputs()
    live_net_dict = {pod: fam.run_config(fam.live_config_dict).daily_return_ser
                     for pod, fam in (("TAA 3x", taa_3x_family(taa_inputs)), ("NDX VXN", ndx_vxn_family(ndx_inputs)))}
    full_index = ndx_inputs.close_df.index.union(taa_inputs.open_df.index)
    tbill_ser = tbill_daily_ser(taa_inputs.dtb3_ser, full_index)
    factor_df = factor_daily_df(full_index, tbill_ser)
    for name_str in name_list:
        bundle = reaudit(plans[name_str][0], live_net_dict, factor_df, tbill_ser)
        print(f"\n== {bundle['pod_str']}: {grade_str(bundle)} ({time.time() - started_float:.0f}s)", flush=True)
        s3_rows = bundle["s3"]["result"]["check_list"] if bundle.get("s3") else []
        for station_str, check_list in (("S3", s3_rows[:1] + [r for r in s3_rows if r[1] != "PASS"]), ("S4", bundle["s4"].check_list),
                                        ("S5", bundle["s5"].check_list), ("S6", bundle["s6"].check_list)):
            for row_name_str, verdict_str, detail_str in check_list:
                print(f"  {station_str} {verdict_str:16s} {row_name_str}: {detail_str[:150]}", flush=True)


if __name__ == "__main__":
    main()
