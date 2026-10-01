"""P6a: re-audit the gated siblings of the two LIVE families through S3-S6 (alpha.scout.reaudit), with cards.

TAA family (slot: TAA 3x in the live book): taa_3x_1n (WIRED), taa_lin_1n_qqq (WIRED), taa_2x_1n_qld,
taa_nobtal_2x_1n_qld, taa_nobtal_2x_1n_sso (PM_READY).
NDX family (slot: NDX VXN): ndx_atr (WIRED), ndx_natr20, ndx_natr20_vxn (research only, not in the registry).
Each gets a RETRO ledger registration, child of its family's P5 registration.

    uv run python scripts/research/scout_p6_reaudition_20261002/run_p6a.py [name ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.card import grade_str
from alpha.scout.family import (
    NDX_GRID_DICT,
    TAA_GRID_DICT,
    TAA_LINEARITY_GRID_DICT,
    _ndx_family,
    ndx_vxn_family,
    taa_3x_family,
    taa_variant_family,
)
from alpha.scout.ledger import Ledger
from alpha.scout.metrics import tbill_daily_ser
from alpha.scout.reaudit import PodPlan, factor_daily_df, reaudit
from alpha.scout.registration import (
    Registration,
    register,
    registration_rows,
)
from alpha.scout.stations.s3_allocation import (
    gate_split,
    predictive_tests,
    ranking_tests,
)

SEAL_END_STR = "2022-12-30"
TAA_NAME_LIST = ["taa_3x_1n", "taa_lin_1n_qqq", "taa_2x_1n_qld", "taa_nobtal_2x_1n_qld", "taa_nobtal_2x_1n_sso"]
NDX_NAME_LIST = ["ndx_atr", "ndx_natr20", "ndx_natr20_vxn"]


# ---------------------------------------------------------------- S3 per family member
def _total_return(symbol_str: str) -> pd.Series:
    from data.norgate_loader import load_price_timeseries

    return load_price_timeseries(symbol_str, adjustment_str="TOTALRETURN", start_date_str="1998-01-01")["Close"]


def taa_s3_fn(variant_name_str: str):
    def s3_fn():
        from alpha.scout.specs import taa_3x

        config = taa_3x.VARIANT_DICT[variant_name_str].config
        daily_close_df = pd.DataFrame({s: _total_return(s) for s in config.defensive_tuple})
        month_close_df = daily_close_df.resample("ME").last().loc[:SEAL_END_STR]
        hurdle_ser = ((1.0 + taa_3x.load_inputs(config=config).dtb3_ser / 100.0) ** (1.0 / 12.0) - 1.0).resample("ME").last()
        if config.score_str == "momentum":
            score_df = sum(month_close_df.pct_change(k, fill_method=None) for k in config.momentum_month_tuple) / len(config.momentum_month_tuple)
            threshold_ser = hurdle_ser.reindex(score_df.index)
        else:
            daily_score_df = sum(taa_3x._linearity_lookback_df(np.log(daily_close_df), d) for d in config.linearity_day_tuple) / len(config.linearity_day_tuple)
            score_df = daily_score_df.resample("ME").last().loc[:SEAL_END_STR]
            threshold_ser = pd.Series(config.linearity_threshold_float, index=score_df.index)
        # *** CRITICAL*** next-month return: a label only.
        next_return_df = month_close_df.pct_change(fill_method=None).shift(-1).sub(hurdle_ser.reindex(month_close_df.index), axis=0)
        result_dict = predictive_tests(score_df, next_return_df, threshold_ser)
        spy_ser, vix_ser = _total_return("SPY"), _total_return("$VIX")
        helper_df = pd.concat([spy_ser, vix_ser], axis=1, join="inner").dropna()
        gate_ser = (helper_df.iloc[:, 0].pct_change().rolling(config.realized_vol_window_int).std(ddof=0) * np.sqrt(252.0) * 100.0 < helper_df.iloc[:, 1]).resample("ME").last()
        risk_proxy_str = "SPY" if config.fallback_str == "SSO" else "QQQ"  # the fallback's unlevered index, for a long history
        result_dict["vix_gate"] = gate_split(_total_return(risk_proxy_str).resample("ME").last().pct_change().shift(-1).loc[:SEAL_END_STR], gate_ser.loc[:SEAL_END_STR])
        note_str = (f"Class W: the {config.score_str} score of {', '.join(config.defensive_tuple)} against next-month excess returns over the ETFs' "
                    f"full histories (to 2022), and the VIX gate against next-month {risk_proxy_str} risk. Diagnostics: the burden is on S5 and S6.")
        return "W", note_str, result_dict

    return s3_fn


def ndx_s3_fn(atr_unit_str: str):
    def s3_fn():
        from alpha.scout.panel import load_panel

        panel = load_panel("Nasdaq 100")
        close_df, high_df, low_df = panel.field("Close"), panel.field("High"), panel.field("Low")
        previous_close_df = close_df.shift(1)
        true_range_df = np.maximum(high_df - low_df, np.maximum((high_df - previous_close_df).abs(), (low_df - previous_close_df).abs()))
        atr_df = true_range_df.rolling(20, min_periods=20).mean()
        denominator_df = atr_df * panel.field("Unadjusted Close") / close_df if atr_unit_str == "dollar" else atr_df / close_df
        position_ser = pd.Series(np.arange(len(panel.date_index)), index=panel.date_index)
        decision_index = panel.date_index[position_ser.groupby(panel.date_index.to_period("M")).max().to_numpy()]
        month_close_df = close_df.loc[decision_index]
        score_df = ((month_close_df / month_close_df.shift(12) - 1.0) / denominator_df.loc[decision_index]).replace([np.inf, -np.inf], np.nan)
        eligible_df = (panel.member_df.loc[decision_index] == 1) & (month_close_df > close_df.rolling(100, min_periods=100).mean().loc[decision_index])
        result_dict = ranking_tests(score_df, month_close_df.shift(-1) / month_close_df - 1.0, eligible_df, top_int=10)
        return "X", f"Class X: ROC12 / {'dollar ATR20' if atr_unit_str == 'dollar' else 'NATR20'} among eligible members against next-month return.", result_dict

    return s3_fn


# ---------------------------------------------------------------- plans
def plan_dict() -> dict[str, PodPlan]:
    from alpha.scout.family import TAA_FAMILY_NAME_DICT
    from alpha.scout.specs import ndx_vxn, taa_3x

    out_dict = {}
    for name_str in TAA_NAME_LIST:
        config = taa_3x.VARIANT_DICT[name_str].config
        out_dict[name_str] = PodPlan(
            name_str=TAA_FAMILY_NAME_DICT[name_str],
            family_fn=lambda inputs, n=name_str: taa_variant_family(n, inputs),
            inputs_fn=lambda c=config: taa_3x.load_inputs(config=c),
            mcpt_kind_str="taa", adoption_date_str="2026-05-15", prior_trial_count_int=103, slot_str="TAA 3x",
            s3_fn=taa_s3_fn(name_str),
            option_dict={"taa": {"asset_tuple": config.traded_tuple, "slot_weight_str": config.slot_weight_str, "score_str": config.score_str}},
        )
    family_name_dict = {"ndx_atr": "NDX ATR", "ndx_natr20": "NDX NATR20", "ndx_natr20_vxn": "NDX NATR20 VXN"}
    for name_str in NDX_NAME_LIST:
        config = ndx_vxn.NDX_VARIANT_DICT[name_str].config
        out_dict[name_str] = PodPlan(
            name_str=family_name_dict[name_str],
            family_fn=lambda inputs, n=name_str: _ndx_family(n, family_name_dict[n], inputs),
            inputs_fn=ndx_vxn.load_inputs, mcpt_kind_str="ndx", adoption_date_str="2026-05-15", prior_trial_count_int=150,
            slot_str="NDX VXN", s3_fn=ndx_s3_fn(config.atr_unit_str), option_dict={"ndx_config": config},
        )
    return out_dict


def register_variants(name_list: list[str]) -> None:
    from alpha.scout.family import TAA_FAMILY_NAME_DICT
    from alpha.scout.specs import ndx_vxn, taa_3x

    ledger = Ledger()
    existing_dict = registration_rows(ledger)
    for name_str in name_list:
        registration_id_str = f"{name_str}_reaudition_20261002"
        if registration_id_str in existing_dict:
            continue
        is_taa_bool = name_str.startswith("taa")
        parent_str = "taa_3x_reaudition_20261002" if is_taa_bool else "ndx_vxn_reaudition_20261002"
        parent_row_dict = existing_dict[parent_str]
        module_str = taa_3x.VARIANT_DICT[name_str].strategy_import_str if is_taa_bool else ndx_vxn.NDX_VARIANT_DICT[name_str].strategy_module_str
        grid_dict = (TAA_LINEARITY_GRID_DICT if is_taa_bool and taa_3x.VARIANT_DICT[name_str].config.score_str == "linearity" else TAA_GRID_DICT) if is_taa_bool else NDX_GRID_DICT
        register(ledger, Registration(
            registration_id_str=registration_id_str, family_id_str=parent_row_dict["family_id_str"],
            hypothesis_str=f"Sibling of {parent_row_dict['registration_id_str']}: {TAA_FAMILY_NAME_DICT.get(name_str, name_str)}, same mechanism with a different "
                           "slot weighting, score, fallback, universe or ranking unit.",
            mechanism_str=parent_row_dict["mechanism_str"], expected_sign_and_location_str=parent_row_dict["expected_sign_and_location_str"],
            hypothesis_class_str=parent_row_dict["hypothesis_class_str"], universe_str=parent_row_dict["universe_str"],
            horizon_str=parent_row_dict["horizon_str"], schedule_str=parent_row_dict["schedule_str"], execution_str=parent_row_dict["execution_str"],
            param_grid_dict={k: tuple(v) for k, v in grid_dict.items()}, primary_metric_str=parent_row_dict["primary_metric_str"],
            kill_criteria_str=parent_row_dict["kill_criteria_str"], source_str=module_str, retro_bool=True,
            prior_trials_int=parent_row_dict["prior_trials_int"], parent_id_str=parent_str,
            universe_choice_str=parent_row_dict["universe_choice_str"], universe_chosen_after_results_bool=parent_row_dict["universe_chosen_after_results_bool"],
        ))
        print("registered", registration_id_str, flush=True)


def main() -> None:
    from alpha.scout.specs import ndx_vxn, taa_3x

    name_list = sys.argv[1:] or TAA_NAME_LIST + NDX_NAME_LIST
    register_variants(name_list)
    started_float = time.time()
    taa_inputs, ndx_inputs = taa_3x.load_inputs(), ndx_vxn.load_inputs()
    live_net_dict = {}
    for pod_str, family in (("TAA 3x", taa_3x_family(taa_inputs)), ("NDX VXN", ndx_vxn_family(ndx_inputs))):
        live_net_dict[pod_str] = family.run_config(family.live_config_dict).daily_return_ser
    full_index = ndx_inputs.close_df.index.union(taa_inputs.open_df.index)
    tbill_ser = tbill_daily_ser(taa_inputs.dtb3_ser, full_index)
    factor_df = factor_daily_df(full_index, tbill_ser)
    plans = plan_dict()
    for name_str in name_list:
        bundle = reaudit(plans[name_str], live_net_dict, factor_df, tbill_ser)
        s4, s5, s6 = bundle["s4"], bundle["s5"], bundle["s6"]
        print(f"\n== {bundle['pod_str']}: {grade_str(bundle)} ({time.time() - started_float:.0f}s)", flush=True)
        for station_str, check_list in (("S3", bundle["s3"]["result"]["check_list"]), ("S4", s4.check_list), ("S5", s5.check_list), ("S6", s6.check_list)):
            for row_name_str, verdict_str, detail_str in check_list:
                print(f"  {station_str} {verdict_str:16s} {row_name_str}: {detail_str}", flush=True)


if __name__ == "__main__":
    main()

