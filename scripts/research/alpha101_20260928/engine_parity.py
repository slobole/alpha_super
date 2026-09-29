"""Engine parity of A0 (C_EQ, N 20, B 2) on U1 (PREREG section 9; research only).

The real repo engine replays the replica's order intents through the trend study's ResearchOrderIntentStrategy
(historical share units; only order_target(0) / order_value / order_target_percent on the bar whose previous_bar is
the decision close). Pass: daily-return correlation >= 0.9999, CAGR within 0.05 pp, identical positions.
"""

from __future__ import annotations

import pickle
import time

from alpha101_20260928 import checks
from alpha101_20260928 import common
from alpha101_20260928 import run_pods
from alpha101_20260928.simulate import simulate
from new_pod_search_20260927 import engine_parity as nps_parity
from trend_breakout_20260927 import engine_parity as trend_parity


def run_parity(universe_str: str = "U1") -> dict:
    context_dict = run_pods.load_context(universe_str, hedged_bool=False)
    feature_obj = context_dict["features"]
    symbol_list = feature_obj.symbol_list
    policy_obj = run_pods.build_policy(context_dict, common.A0_COMPOSITE_STR, common.A0_N_INT, common.A0_B_INT, "LONG")
    start_float = time.perf_counter()
    replica_sim = simulate(feature_obj, policy_obj, slippage_float=common.ENGINE_SLIPPAGE_FLOAT, record_positions_bool=True)
    common.log_progress(f"parity replica A0: {time.perf_counter() - start_float:.0f}s, {len(replica_sim['intent_df'])} intents, final NAV {replica_sim['final_total_float']:.0f}")
    intents_dict = nps_parity.intents_from_intent_df(replica_sim["intent_df"], symbol_list)
    traded_symbol_list = sorted({symbol_str for intent_list in intents_dict.values() for symbol_str, _, _ in intent_list})
    with open(common.RESULTS_DIR_PATH / "intents_A0_U1.pkl", "wb") as file_obj:
        pickle.dump({"intent_df": replica_sim["intent_df"], "position_log": replica_sim["position_log"], "trade_df": replica_sim["trade_df"]}, file_obj)
    engine_obj = nps_parity.run_engine_replay(intents_dict, traded_symbol_list + ["SPY"], "parity_A0_C_EQ_N20_B2")
    report_dict = nps_parity.parity_report(replica_sim, engine_obj, feature_obj.date_index, symbol_list)
    report_dict["cell"] = common.cell_key(common.A0_COMPOSITE_STR, common.A0_N_INT, common.A0_B_INT)
    report_dict["universe"] = universe_str
    report_dict["traded_symbols_int"] = len(traded_symbol_list)
    report_dict["thresholds"] = {"corr_min": trend_parity.CORR_THRESHOLD_FLOAT, "cagr_tolerance_pp": trend_parity.CAGR_TOLERANCE_FLOAT * 100}
    checks.update_checks("engine_parity_A0", report_dict)
    common.write_json("parity_gate.json", report_dict)
    common.log_progress(f"PARITY A0: corr {report_dict['daily_return_corr_float']:.7f}, cagr gap {report_dict['cagr_gap_pp_float']:.4f} pp, max daily diff {report_dict['max_abs_daily_diff_float']:.2e}, "
                        f"position mismatches {report_dict['position_mismatch_sessions_int']}, passed {report_dict['passed_bool']}")
    return report_dict
