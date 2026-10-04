"""Scout discovery study: a Bitcoin + gold sleeve (GQResearch "Dual Momentum Without the Dual"; audited in Pakal).

Two registrations, written to the ledger BEFORE this script computes any Scout result:
    btc_gold_trend    the article's "hold both" rule (trend signal + per-asset 20% volatility cap); 18-configuration grid
    btc_gold_static   the no-signal control: 50/50 Bitcoin / gold with the same cap; 6-configuration grid
Contamination: the owner's Pakal audit (866 engine runs, 2018-2026) has seen this history, so the vault (2023 on) is
contaminated and those runs count as prior trials. In sample = 2017-10 to 2022-12-30 (about five years; Bitcoin ETF
history is short). The book question: a 10% sleeve (54% TAA 3x / 36% NDX VXN / 10% sleeve) against 10% T-bills.

    uv run python scripts/research/scout_btc_gold_20261002/run.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from alpha.scout.card import grade_str
from alpha.scout.engines.weights import CostModel
from alpha.scout.family import FamilyRunner, ndx_vxn_family, taa_3x_family
from alpha.scout.ledger import Ledger
from alpha.scout.metrics import sharpe_float, tbill_daily_ser
from alpha.scout.reaudit import PodPlan, factor_daily_df, reaudit
from alpha.scout.registration import Registration, register, registration_rows
from alpha.scout.specs import btc_gold
from alpha.scout.stations.s3_allocation import predictive_tests
from alpha.scout.stations.s6_book import tbill_slot_test

TREND_GRID_DICT = {"cap_float": (0.15, 0.20, 0.25), "lookback_tuple": ((10, 21, 42), (21, 42, 63), (42, 63, 126)), "vol_mode_str": ("63", "max21_63")}
STATIC_GRID_DICT = {"cap_float": (0.15, 0.20, 0.25), "vol_mode_str": ("63", "max21_63")}
SLEEVE_BOOK_DICT = {"TAA 3x": 0.54, "NDX VXN": 0.36}
SLEEVE_FLOAT = 0.10


def family(name_str: str, grid_dict: dict, base_config: btc_gold.BtcGoldConfig, inputs) -> FamilyRunner:
    def simulate_fn(config_dict: dict, cost_model: CostModel, capital_float: float):
        return btc_gold.simulate_config(inputs, replace(base_config, **config_dict), cost_model, capital_float)

    return FamilyRunner(name_str=name_str, family_id_str="time_series_trend_and_breakout", param_grid_dict=grid_dict,
                        live_config_dict={k: getattr(base_config, k) for k in grid_dict}, simulate_fn=simulate_fn, offset_count_int=5)


def trend_s3():
    """Per asset (BTC, GLD): the weekly trend allocation against the next week's return over BIL (class W diagnostic)."""
    inputs = btc_gold.load_inputs()
    frame = inputs.return_df.loc[btc_gold.START_STR:"2022-12-30", ["BTC_splice", "GLD", "BIL"]]
    log_price_df = np.log1p(frame[["BTC_splice", "GLD"]]).cumsum()
    decision_vec = btc_gold._decision_rows(frame.index, 0)
    score_df = sum((log_price_df - log_price_df.shift(L) > 0).astype(float) for L in (21, 42, 63)).iloc[decision_vec] / 3.0
    week_return_df = (1 + frame).cumprod().iloc[decision_vec].pct_change().shift(-1)
    next_excess_df = week_return_df[["BTC_splice", "GLD"]].sub(week_return_df["BIL"], axis=0)
    result_dict = predictive_tests(score_df.iloc[63:], next_excess_df.iloc[63:], pd.Series(0.5, index=score_df.index[63:]), min_asset_int=2)
    return "W", ("Class W: each asset's weekly trend score (share of 21/42/63-session returns above zero) against its next-week "
                 "return over BIL. With two assets the per-asset slopes are the meaningful part."), result_dict


def registration(name_str: str, grid_dict: dict, hypothesis_str: str) -> Registration:
    return Registration(
        registration_id_str=f"{name_str}_20261002", family_id_str="time_series_trend_and_breakout", hypothesis_str=hypothesis_str,
        mechanism_str="Trend persistence in Bitcoin and gold, plus a per-asset volatility cap that shrinks Bitcoin in turbulent spells.",
        expected_sign_and_location_str="Positive active return over a volatility-targeted Bitcoin + gold mix; a better book than T-bills in a 10% sleeve.",
        hypothesis_class_str="W", universe_str="BTC (GBTC -> IBIT splice; spot BTC as a check), GLD, BIL", horizon_str="one week",
        schedule_str="weekly, first session of the week; the volatility cap daily", execution_str="decided at close T, held from close T+1",
        param_grid_dict={k: tuple(v) for k, v in grid_dict.items()}, primary_metric_str="S5 MCPT (score SD); S6 10% sleeve vs T-bills",
        kill_criteria_str="S8 CUSUM or Cold Blood Index red", source_str="https://gqresearch.substack.com/p/dual-momentum-without-the-dual",
        source_published_date_str="2026-09-01", retro_bool=True, prior_trials_int=866,
        universe_choice_str="Bitcoin and gold chosen by the source author; the owner's Pakal audit (866 runs) has seen 2018-2026.",
        universe_chosen_after_results_bool=True,
    )


def main() -> None:
    from alpha.scout.specs import ndx_vxn, taa_3x

    ledger = Ledger()
    existing_dict = registration_rows(ledger)
    registration_list = [
        registration("btc_gold_trend", TREND_GRID_DICT, "Holding Bitcoin and gold while their 21/42/63-session trends are up (50/50 when both, "
                     "100% in one, else T-bills), each capped at 20% volatility, beats a volatility-targeted Bitcoin + gold mix."),
        registration("btc_gold_static", STATIC_GRID_DICT, "A fixed 50/50 Bitcoin + gold mix, each capped at 20% volatility (no signal), "
                     "beats a volatility-targeted Bitcoin + gold mix; the control for the trend signal."),
    ]
    for reg in registration_list:
        if reg.registration_id_str not in existing_dict:
            register(ledger, reg)
            print("registered", reg.registration_id_str, flush=True)

    started_float = time.time()
    taa_inputs, ndx_inputs = taa_3x.load_inputs(), ndx_vxn.load_inputs()
    live_net_dict = {pod: fam.run_config(fam.live_config_dict).daily_return_ser
                     for pod, fam in (("TAA 3x", taa_3x_family(taa_inputs)), ("NDX VXN", ndx_vxn_family(ndx_inputs)))}
    full_index = ndx_inputs.close_df.index.union(taa_inputs.open_df.index)
    tbill_ser = tbill_daily_ser(taa_inputs.dtb3_ser, full_index)
    factor_df = factor_daily_df(full_index, tbill_ser)

    plan_list = []
    for name_str, grid_dict, base_config, s3_fn in (
        ("BTC-Gold trend", TREND_GRID_DICT, btc_gold.LIVE_CONFIG, trend_s3),
        ("BTC-Gold static", STATIC_GRID_DICT, replace(btc_gold.LIVE_CONFIG, signal_bool=False), None),
    ):
        plan_list.append(PodPlan(
            name_str=name_str, family_fn=lambda inputs, n=name_str, g=grid_dict, c=base_config: family(n, g, c, inputs),
            inputs_fn=btc_gold.load_inputs, mcpt_kind_str="spec", adoption_date_str="2026-09-01", prior_trial_count_int=866,
            slot_str="NDX VXN", s3_fn=s3_fn,
            option_dict={"spec": {"module_str": "alpha.scout.specs.btc_gold", "matrix_fn": lambda m, inputs: m.mcpt_matrix(inputs),
                                  "asset_count_int": 2, "fast_kwarg_dict": {"base_config": base_config}}},
            strategy_module_str="", book_weight_dict={**SLEEVE_BOOK_DICT, name_str: SLEEVE_FLOAT},
        ))
    for plan in plan_list:
        bundle = reaudit(plan, live_net_dict, factor_df, tbill_ser)
        print(f"\n== {bundle['pod_str']}: {grade_str(bundle)} ({time.time() - started_float:.0f}s)", flush=True)
        for station_str, check_list in (("S3", bundle["s3"]["result"]["check_list"] if bundle.get("s3") else []), ("S4", bundle["s4"].check_list),
                                        ("S5", bundle["s5"].check_list), ("S6", bundle["s6"].check_list)):
            for row_name_str, verdict_str, detail_str in check_list:
                print(f"  {station_str} {verdict_str:16s} {row_name_str}: {detail_str}", flush=True)
        # Full period 2017-10..2026-09 (contaminated: the Pakal audit saw it), the sleeve question only.
        candidate_ser = bundle["s4"].grid_df[bundle["s4"].live_label_str]
        full_dict = tbill_slot_test(candidate_ser, {**SLEEVE_BOOK_DICT, plan.name_str: SLEEVE_FLOAT}, live_net_dict, plan.name_str, tbill_ser)
        print(f"  FULL (2017-10..2026-09, contaminated): book {full_dict['book_sharpe_with_float']:.2f} vs {full_dict['book_sharpe_tbills_float']:.2f}, "
              f"P {full_dict['probability_better_float']:.2f}; sleeve Sharpe {sharpe_float(candidate_ser.loc['2017-12':]):.2f}", flush=True)


if __name__ == "__main__":
    main()
