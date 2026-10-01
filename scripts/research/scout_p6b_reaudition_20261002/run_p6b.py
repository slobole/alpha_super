"""P6b: re-audit the macro pods through S3-S6 (alpha.scout.reaudit, MCPT kind "spec"), with cards and RETRO
ledger registrations. Pods join as their specs pass the exact identity gate.

Slot in the reference (live 60/40) book: each macro pod is tested in NDX VXN's slot, the slot P5/P6a found adds
nothing to this book, so the question is whether a macro pod is a better use of it than T-bills.

    uv run python scripts/research/scout_p6b_reaudition_20261002/run_p6b.py [name ...]
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
from alpha.scout.stations.s3_allocation import gate_split, predictive_tests


def spec_s3_fn(module_str: str, variant_str: str, note_str: str):
    """S3 from a spec module's `s3_inputs` (keys: a predictive_tests block and/or a gate_split block)."""
    def s3_fn():
        import importlib

        module = importlib.import_module(module_str)
        input_dict = module.s3_inputs(module.VARIANT_DICT[variant_str].config)
        predictive_dict = next(v for v in input_dict.values() if "score_df" in v)
        result_dict = predictive_tests(predictive_dict["score_df"], predictive_dict["next_return_df"], predictive_dict.get("hurdle_ser"))
        gate_dict = next((v for v in input_dict.values() if "gate_on_ser" in v), None)
        if gate_dict is not None:
            result_dict["vix_gate"] = gate_split(gate_dict["next_return_ser"], gate_dict["gate_on_ser"])
        return "W", note_str, result_dict

    return s3_fn


def plan_dict() -> dict[str, tuple[PodPlan, Registration]]:
    from alpha.scout.family import compass_family
    from alpha.scout.specs import compass

    out_dict = {}
    for variant_str, label_str in (("compass", "Compass"), ("compass_qqq", "Compass QQQ")):
        config = compass.VARIANT_DICT[variant_str].config
        plan = PodPlan(
            name_str=label_str,
            family_fn=lambda inputs, v=variant_str: compass_family(v, inputs),
            inputs_fn=lambda c=config: compass.load_inputs(config=c),
            mcpt_kind_str="spec", adoption_date_str="2026-09-28", prior_trial_count_int=826, slot_str="NDX VXN",
            s3_fn=spec_s3_fn("alpha.scout.specs.compass", variant_str, (
                "Class W: Compass gates rather than ranks. The regime's pick among its four sleeves against the others next month "
                "(predictive tests), and the SPY > SMA200 growth gate against next-month SPY risk. The regime map was designed on this "
                "history, so these are diagnostics.")),
            option_dict={"spec": {
                "module_str": "alpha.scout.specs.compass",
                "matrix_fn": lambda module, inputs, c=config: module.mcpt_matrix(inputs, c),
                "fast_kwarg_dict": {"base_config": config}, "asset_count_int": 5,
            }},
            strategy_module_str=compass.VARIANT_DICT[variant_str].strategy_import_str,
        )
        registration = Registration(
            registration_id_str=f"{variant_str}_reaudition_20261002", family_id_str="macro_regime_allocation",
            hypothesis_str=f"{label_str}: a growth (SPY trend) x inflation (T5YIE trend and level) regime map picks one of four "
                           "ETF sleeves each month and beats a volatility-targeted equal weight of the traded ETFs.",
            mechanism_str="Asset classes respond differently to growth and inflation regimes; breakeven inflation and the equity trend "
                          "are timely, published reads of those regimes.",
            expected_sign_and_location_str="Positive active return, mostly from avoiding equities in growth-down months.",
            hypothesis_class_str="W", universe_str="SPY-regime sleeves: XLE, " + config.goldilocks_str + ", XLU, XLP + IEF",
            horizon_str="one month", schedule_str="decision at the month's last session", execution_str="next session's open",
            param_grid_dict={"growth_sma_int": (150, 200, 250), "inflation_threshold_float": (1.8, 2.0, 2.2), "trend_lookback_int": (40, 60, 80)},
            primary_metric_str="S5 MCPT (score SD); S6 T-bill slot", kill_criteria_str="S8 CUSUM or Cold Blood Index red",
            source_str=compass.VARIANT_DICT[variant_str].strategy_import_str, retro_bool=True, prior_trials_int=826,
            universe_choice_str="Sector sleeves chosen by the owner's 2026-09 Compass design and its 826-trial study.",
            universe_chosen_after_results_bool=True,
        )
        out_dict[variant_str] = (plan, registration)

    from alpha.scout.family import CORE5_GRID_DICT, core5_family
    from alpha.scout.specs import core5

    def core5_s3():
        score_df, next_return_df, threshold_ser = core5.s3_inputs()
        return "W", ("Class W: CORE5 does not rank; each sleeve is its own trend signal (SMA / adaptive MA - 1 at month end, > 0 = long) "
                     "against its next-month return over BIL. Only the on/off spread and per-asset slopes test the idea; the "
                     "cross-sectional slope is reported for completeness."), predictive_tests(score_df, next_return_df, threshold_ser)

    out_dict["core5"] = (
        PodPlan(
            name_str="CORE5", family_fn=lambda inputs: core5_family(inputs), inputs_fn=core5.load_inputs, mcpt_kind_str="spec",
            adoption_date_str="2026-09-28", prior_trial_count_int=50, slot_str="NDX VXN", s3_fn=core5_s3,
            option_dict={"spec": {"module_str": "alpha.scout.specs.core5", "matrix_fn": lambda module, inputs: module.mcpt_matrix(inputs),
                                  "asset_count_int": len(core5.TRADED_TUPLE)}},
            strategy_module_str=core5.STRATEGY_IMPORT_STR,
        ),
        Registration(
            registration_id_str="core5_reaudition_20261002", family_id_str="time_series_trend_and_breakout",
            hypothesis_str="CORE5: each of SPY, IEF, GLD, DBC and UUP is held while its short price average is above an adaptive moving "
                           "average (speed set by its drawdown percentile), DBC is shorted when below, idle weight goes to BIL; it beats a "
                           "volatility-targeted equal weight of the six ETFs.",
            mechanism_str="Time-series trend across asset classes, with an adaptive average that speeds up in drawdowns.",
            expected_sign_and_location_str="Positive active return, mostly in sustained trends and crisis exits.",
            hypothesis_class_str="W", universe_str="SPY, IEF, GLD, DBC, UUP; reserve BIL", horizon_str="month-end and state changes",
            schedule_str="month end", execution_str="next session's open (shorts with a borrow fee)",
            param_grid_dict={k: tuple(v) for k, v in CORE5_GRID_DICT.items()},
            primary_metric_str="S5 MCPT (score SD); S6 T-bill slot", kill_criteria_str="S8 CUSUM or Cold Blood Index red",
            source_str=core5.STRATEGY_IMPORT_STR, retro_bool=True, prior_trials_int=50,
            universe_choice_str="A five-asset macro ETF set from the owner's adaptive-macro design; documented variants not counted, so N = 50 (rule).",
            universe_chosen_after_results_bool=False,
        ),
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
