"""P6c: re-audit the calendar-flow and ETF reversal pods through S3-S6 (alpha.scout.reaudit, MCPT kind "spec"), with
cards and RETRO ledger registrations. Each pod is tested in NDX VXN's slot of the live 60/40 book (as in P6b).

    uv run python scripts/research/scout_p6c_reaudition_20261002/run_p6c.py [name ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.card import grade_str
from alpha.scout.family import ndx_vxn_family, taa_3x_family
from alpha.scout.ledger import Ledger
from alpha.scout.metrics import tbill_daily_ser
from alpha.scout.reaudit import PodPlan, factor_daily_df, reaudit
from alpha.scout.registration import Registration, register, registration_rows
from alpha.scout.stations.s3_allocation import predictive_tests


def _registration(name_str: str, family_id_str: str, label_str: str, module, grid_dict: dict, hypothesis_str: str, mechanism_str: str,
                  class_str: str, universe_str: str, schedule_str: str, execution_str: str, prior_int: int, choice_str: str,
                  after_results_bool: bool = False) -> Registration:
    return Registration(
        registration_id_str=f"{name_str}_reaudition_20261002", family_id_str=family_id_str, hypothesis_str=hypothesis_str,
        mechanism_str=mechanism_str, expected_sign_and_location_str="Positive active return over the volatility-targeted equal weight.",
        hypothesis_class_str=class_str, universe_str=universe_str, horizon_str="days", schedule_str=schedule_str,
        execution_str=execution_str, param_grid_dict={k: tuple(v) for k, v in grid_dict.items()},
        primary_metric_str="S5 MCPT (score SD); S6 T-bill slot", kill_criteria_str="S8 CUSUM or Cold Blood Index red",
        source_str=module.STRATEGY_IMPORT_STR if hasattr(module, "STRATEGY_IMPORT_STR") else label_str, retro_bool=True,
        prior_trials_int=prior_int, universe_choice_str=choice_str, universe_chosen_after_results_bool=after_results_bool,
    )


def plan_dict() -> dict[str, tuple[PodPlan, Registration]]:
    from alpha.scout.family import EOM_GRID_DICT, eom_family
    from alpha.scout.specs import eom

    out_dict = {}

    def eom_s3():
        result_dict = predictive_tests(**eom.s3_inputs()["predictive_tests"])
        return "W", ("Class W, calendar flow: the causal pressure percentile against the two windows it trades (TLT minus SPY into "
                     "month end, SPY minus TLT after it). With two columns only the per-window slopes are meaningful; the "
                     "cross-sectional slope and on/off spread need more assets and read NaN."), result_dict

    out_dict["eom"] = (
        PodPlan(
            name_str="EOM", family_fn=lambda inputs: eom_family(inputs), inputs_fn=eom.load_inputs, mcpt_kind_str="spec",
            adoption_date_str="2026-09-30", prior_trial_count_int=50, slot_str="NDX VXN", s3_fn=eom_s3,
            option_dict={"spec": {"module_str": "alpha.scout.specs.eom", "matrix_fn": lambda m, inputs: m.mcpt_matrix(inputs), "asset_count_int": 2}},
            strategy_module_str=eom.STRATEGY_IMPORT_STR,
        ),
        _registration(
            "eom", "calendar_and_flow", "EOM", eom, EOM_GRID_DICT,
            "Month-end rebalancing flow: when the 60/40 SPY/IEF drift is extreme at dtme=7, pension rebalancing pushes TLT against SPY "
            "into month end and reverses after it; trading both legs at the close beats a volatility-targeted equal weight.",
            "Predictable institutional rebalancing flows at month end (calendar and flow).", "W", "SPY, TLT (signal SPY/IEF)",
            "dtme 6 entry, month-end reversal, session 5 exit", "same-session close (MOC)", 50,
            "SPY/TLT from the owner's month-end-flow research (2026-09); documented variants not counted, so N = 50 (rule).",
        ),
    )
    from alpha.scout.family import (
        DISPERSION_IBS_GRID_DICT,
        DISPERSION_IBS_NAME_DICT,
        SECTOR_IBS_GRID_DICT,
        dispersion_ibs_family,
        sector_ibs_family,
    )
    from alpha.scout.specs import sector_dispersion_ibs, sector_ibs

    event_note_str = ("Class E: the raw entry signal as events among ETFs with a valid bar, excess over the same-date basket mean, "
                      "date-level Newey-West t; S3's hard and soft criteria count toward the grade.")
    ibs_hypothesis_str = ("Buying a sector ETF after a close near its low on a down-shock day and selling on a strong rebound close "
                          "beats a volatility-targeted equal weight of the basket.")
    out_dict["sector_ibs_vox_iyr"] = (
        PodPlan(
            name_str="Sector IBS VOX IYR", family_fn=lambda inputs: sector_ibs_family(inputs), inputs_fn=sector_ibs.load_inputs,
            mcpt_kind_str="spec", adoption_date_str="2026-09-30", prior_trial_count_int=50, slot_str="NDX VXN",
            s3_fn=lambda: ("E", event_note_str, sector_ibs.s3_result(sector_ibs.s3_inputs())),
            option_dict={"spec": {"module_str": "alpha.scout.specs.sector_ibs", "matrix_fn": lambda m, inputs: m.mcpt_matrix(inputs),
                                  "asset_count_int": len(sector_ibs.TRADED_TUPLE)}},
            strategy_module_str=sector_ibs.STRATEGY_IMPORT_STR,
        ),
        _registration("sector_ibs_vox_iyr", "etf_short_term_reversal", "Sector IBS VOX IYR", sector_ibs, SECTOR_IBS_GRID_DICT,
                      ibs_hypothesis_str, "Liquidity provision after sharp intraday selling in sector ETFs.", "E",
                      ", ".join(sector_ibs.TRADED_TUPLE), "daily close decision", "next session's open", 50,
                      "The 11 US sector ETFs with VOX/IYR for the late-listed sectors; documented variants not counted, so N = 50 (rule)."),
    )
    for variant_str, label_str in DISPERSION_IBS_NAME_DICT.items():
        variant = sector_dispersion_ibs.VARIANT_DICT[variant_str]
        out_dict[variant_str] = (
            PodPlan(
                name_str=label_str, family_fn=lambda inputs, v=variant_str: dispersion_ibs_family(v, inputs),
                inputs_fn=lambda v=variant_str: sector_dispersion_ibs.load_inputs(v), mcpt_kind_str="spec",
                adoption_date_str="2026-09-30", prior_trial_count_int=50, slot_str="NDX VXN",
                s3_fn=lambda v=variant_str: ("E", event_note_str, sector_ibs.s3_result(sector_dispersion_ibs.s3_inputs(v))),
                option_dict={"spec": {"module_str": "alpha.scout.specs.sector_dispersion_ibs", "matrix_fn": lambda m, inputs: m.mcpt_matrix(inputs),
                                      "asset_count_int": len(variant.symbol_tuple), "fast_kwarg_dict": {"base_config": variant.config}}},
                strategy_module_str=variant.strategy_import_str,
            ),
            _registration(variant_str, "etf_short_term_reversal", label_str, SimpleNamespace(STRATEGY_IMPORT_STR=variant.strategy_import_str), DISPERSION_IBS_GRID_DICT,
                          ibs_hypothesis_str.replace("on a down-shock day", "on a wide-range day"),
                          "Liquidity provision in dispersed industry ETFs.", "E", ", ".join(variant.symbol_tuple),
                          "daily close decision", "next session's open", 50,
                          "Industry ETF basket picked by the owner's combination and market-correlation studies (after results); N = 50.",
                          after_results_bool=True),
        )
    return out_dict


def main() -> None:
    from alpha.scout.specs import ndx_vxn, taa_3x

    plans = plan_dict()
    name_list = sys.argv[1:] or list(plans)
    ledger = Ledger()
    existing_dict = registration_rows(ledger)
    for name_str in name_list:
        registration = plans[name_str][1]
        if registration.registration_id_str not in existing_dict:
            register(ledger, registration)
            print("registered", registration.registration_id_str, flush=True)
    started_float = time.time()
    taa_inputs, ndx_inputs = taa_3x.load_inputs(), ndx_vxn.load_inputs()
    live_net_dict = {}
    for pod_str, family in (("TAA 3x", taa_3x_family(taa_inputs)), ("NDX VXN", ndx_vxn_family(ndx_inputs))):
        live_net_dict[pod_str] = family.run_config(family.live_config_dict).daily_return_ser
    full_index = ndx_inputs.close_df.index.union(taa_inputs.open_df.index)
    tbill_ser = tbill_daily_ser(taa_inputs.dtb3_ser, full_index)
    factor_df = factor_daily_df(full_index, tbill_ser)
    for name_str in name_list:
        bundle = reaudit(plans[name_str][0], live_net_dict, factor_df, tbill_ser)
        print(f"\n== {bundle['pod_str']}: {grade_str(bundle)} ({time.time() - started_float:.0f}s)", flush=True)
        for station_str, check_list in (("S3", bundle["s3"]["result"]["check_list"]), ("S4", bundle["s4"].check_list),
                                        ("S5", bundle["s5"].check_list), ("S6", bundle["s6"].check_list)):
            for row_name_str, verdict_str, detail_str in check_list:
                print(f"  {station_str} {verdict_str:16s} {row_name_str}: {detail_str}", flush=True)


if __name__ == "__main__":
    main()
