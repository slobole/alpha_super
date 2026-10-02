"""P7b: re-audit the DV2 industry-ETF pod (RESEARCH tier) through S3-S6, with a card and a RETRO ledger registration.
MCPT kind "spec" (A9 date shuffle of the 19 ETFs' bars; eligibility stays on real dates). The pod is tested in NDX VXN's
slot of the live 60/40 book, as in P6b-P7. It is idle about half the time and the engine earns nothing on idle cash,
so the slot test is also shown with idle cash credited at the T-bill rate (information).

    uv run python scripts/research/scout_p7b_reaudition_20261002/run_p7b.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.card import grade_str
from alpha.scout.family import (
    DV2_INDUSTRY_GRID_DICT,
    dv2_industry_family,
    ndx_vxn_family,
    taa_3x_family,
)
from alpha.scout.ledger import Ledger
from alpha.scout.metrics import tbill_daily_ser
from alpha.scout.reaudit import (
    BOOK_WEIGHT_DICT,
    SEAL_END_STR,
    PodPlan,
    factor_daily_df,
    reaudit,
)
from alpha.scout.registration import Registration, register, registration_rows
from alpha.scout.stations.s6_book import tbill_slot_test

NAME_STR = "DV2 industry ETF"
EVENT_NOTE_STR = ("Class E: the rule's raw entry signal as events among eligible ETFs, excess over the same-date eligible "
                  "mean, date-level Newey-West t at the horizon nearest the median hold; S3's hard and soft criteria count "
                  "toward the grade. A long-only pod also earns the regime's own return: that part is judged in S5/S6.")


def plan_and_registration() -> tuple[PodPlan, Registration]:
    from alpha.scout.specs import dv2_industry_etf, sector_ibs

    plan = PodPlan(
        name_str=NAME_STR, family_fn=dv2_industry_family, inputs_fn=dv2_industry_etf.load_inputs, mcpt_kind_str="spec",
        adoption_date_str="2026-09-27", prior_trial_count_int=50, slot_str="NDX VXN",
        s3_fn=lambda: ("E", EVENT_NOTE_STR, sector_ibs.s3_result(dv2_industry_etf.s3_inputs())),
        option_dict={"spec": dv2_industry_etf.mcpt_option_dict()}, strategy_module_str=dv2_industry_etf.STRATEGY_IMPORT_STR,
    )
    registration = Registration(
        registration_id_str="dv2_industry_etf_reaudition_20261002", family_id_str="etf_short_term_reversal",
        hypothesis_str=("DV2 industry ETF: buying uptrending, liquid US industry ETFs whose DV2 is very low, selling on a rebound "
                        "close, beats the equal-weight ETFs."),
        mechanism_str="Liquidity provision to impatient sellers of industry baskets (short-term reversal).",
        expected_sign_and_location_str="Positive active return over the equal-weight ETFs, concentrated in stress.",
        hypothesis_class_str="E", universe_str="19 US industry ETFs with 252 sessions of history and ADV63 > $50M",
        horizon_str="days", schedule_str="daily close decision", execution_str="next session's open",
        param_grid_dict={k: tuple(v) for k, v in DV2_INDUSTRY_GRID_DICT.items()},
        primary_metric_str="S5 MCPT (date shuffle, eligibility on real dates); S6 T-bill slot",
        kill_criteria_str="S8 CUSUM or Cold Blood Index red", source_str=dv2_industry_etf.STRATEGY_IMPORT_STR, retro_bool=True,
        prior_trials_int=50, parent_id_str=None,  # its own family (ETF reversal); the stock DV2 pod is a sibling idea
        universe_choice_str=("The 2026-09-25 DV2 research tried 23 universe and ETF runs and found industry ETFs the one new "
                             "universe that works; the 19-ETF list and the $50M screen were chosen then (counted: 50, the floor)."),
        universe_chosen_after_results_bool=True,
    )
    return plan, registration


def idle_cash_slot(inputs, tbill_ser, live_net_dict) -> dict:
    """The NDX VXN slot test with the pod's idle cash credited at the T-bill rate (information)."""
    family = dv2_industry_family(inputs)
    result = family.run_config(family.live_config_dict)
    close_df = inputs.base.close_df
    exposure_ser = ((result.daily_position_df * close_df.reindex(result.daily_position_df.index)).sum(axis=1)
                    / result.total_value_ser.reindex(result.daily_position_df.index)).clip(0.0, 1.0)
    cash_ser = (1.0 - exposure_ser.shift(1)).fillna(1.0)
    net_ser = (result.daily_return_ser + cash_ser.reindex(result.daily_return_ser.index).fillna(1.0)
               * tbill_ser.reindex(result.daily_return_ser.index).fillna(0.0)).loc[:SEAL_END_STR]
    book_weight_dict = {(NAME_STR if p == "NDX VXN" else p): w for p, w in BOOK_WEIGHT_DICT.items()}
    component_dict = {p: live_net_dict[p].loc[:SEAL_END_STR] for p in book_weight_dict if p != NAME_STR}
    out_dict = tbill_slot_test(net_ser, book_weight_dict, component_dict, NAME_STR, tbill_ser)
    out_dict["mean_exposure_float"] = float(exposure_ser.loc[:SEAL_END_STR].mean())
    out_dict["idle_share_float"] = float((exposure_ser.loc[:SEAL_END_STR] < 1e-9).mean())
    return out_dict


def main() -> None:
    from alpha.scout.specs import ndx_vxn, taa_3x

    plan, registration = plan_and_registration()
    ledger = Ledger()
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
    bundle = reaudit(plan, live_net_dict, factor_df, tbill_ser)
    print(f"\n== {bundle['pod_str']}: {grade_str(bundle)} ({time.time() - started_float:.0f}s)", flush=True)
    s3_rows = bundle["s3"]["result"]["check_list"] if bundle.get("s3") else []
    for station_str, check_list in (("S3", s3_rows[:1] + [r for r in s3_rows if r[1] != "PASS"]), ("S4", bundle["s4"].check_list),
                                    ("S5", bundle["s5"].check_list), ("S6", bundle["s6"].check_list)):
        for row_name_str, verdict_str, detail_str in check_list:
            print(f"  {station_str} {verdict_str:16s} {row_name_str}: {detail_str[:150]}", flush=True)
    from alpha.scout.specs import dv2_industry_etf

    cash_dict = idle_cash_slot(dv2_industry_etf.load_inputs(), tbill_ser, live_net_dict)
    print("  info idle cash at T-bills:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cash_dict.items()
                                           if not hasattr(v, "__len__")}, flush=True)


if __name__ == "__main__":
    main()
