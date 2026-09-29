"""Condensed report extract from results.json (research only): every number quoted in the final report comes from here.

    python scripts/research/new_pod_search_20260927/report_extract.py
Writes report_extract.json next to results.json and prints it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from new_pod_search_20260927 import common  # noqa: E402

R = common.RESULTS_DIR_PATH


def r3(x):
    return round(float(x), 3) if isinstance(x, (int, float, np.floating)) and x is not None and np.isfinite(x) else x


def main() -> None:
    res = json.loads((R / "results.json").read_text())
    out: dict = {"decision": res["decision"], "controls": {}, "candidates": {}, "cells": {}, "m_diagnostics": {}, "s_diagnostics": {}, "multiplicity": {}, "timing_luck_S": {}}
    for name_str, blocks in res["controls"].items():
        out["controls"][name_str] = {b: (r3(v["sharpe"]), r3(v["max_dd"]), r3(v["cagr"])) for b, v in blocks.items()}
    out["controls"]["C_BIL_stress"] = {b: r3(v["sharpe"]) for b, v in res["controls_stress"]["C_BIL"].items()}
    for c in res["candidates"]:
        r = c["rule"]
        out["candidates"][c["stage"]] = {
            "centre": c["centre"], "neighbourhood": c["neighbourhood"], "plateau_full_sharpe_sweep": r3(c["plateau_full_sharpe"]),
            "centre_standalone_sweep": {b: (r3(v["sharpe"]), r3(v["cagr"]), r3(v["max_dd"])) for b, v in c["centre_standalone_sweep"].items()},
            "margins_vs_C_BIL": {b: r3(v) for b, v in r["engine"]["sharpe_margin"].items()}, "min_margin": r3(r["engine"]["min_margin"]),
            "dd_gap_pp": {b: r3(v) for b, v in r["engine"]["dd_gap_pp"].items()}, "R1": r["engine"]["R1"], "R2": r["engine"]["R2"],
            "stress_margins": {b: r3(v) for b, v in r["stress"]["sharpe_margin"].items()}, "R3": r["R3"],
            "R4": {u: (r3(v["neigh_median_book_g_full_sharpe"]), r3(v["C_BIL_g_full_sharpe"]), v["pass"]) for u, v in r["R4"].items()},
            "R5": {"aum_max_5pct_M": r3((r["R5"]["aum_max_5pct_usd_2021_2026"] or 0) / 1e6), "aum_p95_1pct_M": r3((r["R5"]["aum_p95_1pct_usd_2021_2026"] or 0) / 1e6), "pass": r["R5"]["pass"]},
            "passes": r["passes"],
            "labels": {"vs_G3_margins": {b: r3(v) for b, v in r["labels"]["vs_G3"]["sharpe_margin"].items()}, "beats_G3_all_blocks": r["labels"]["vs_G3"]["R1"],
                       "vs_C_SPY_margins": {b: r3(v) for b, v in r["labels"]["vs_C_SPY"]["sharpe_margin"].items()}, "beats_C_SPY_all_blocks": r["labels"]["vs_C_SPY"]["R1"],
                       "nosweep_vs_C_CASH0_margins": {b: r3(v) for b, v in r["labels"]["nosweep_vs_C_CASH0"]["sharpe_margin"].items()}, "nosweep_beats_C_CASH0_all_blocks": r["labels"]["nosweep_vs_C_CASH0"]["R1"],
                       "owner_gate_centre": {k: r3(v) if k != "pass" else v for k, v in r["labels"]["owner_gate_centre"].items()},
                       "owner_gate_neigh_median": {k: r3(v) if k != "pass" else v for k, v in r["labels"]["owner_gate_neighbourhood_median"].items()}},
            "neigh_median_books": {b: (r3(v["sharpe"]), r3(v["max_dd"]), r3(v["cagr"])) for b, v in r["neigh_median_books_sweep_engine"].items()},
            "walk_forward": {"centre_by_P1": c["walk_forward"]["centre_by_P1"], "standalone_P2_P3": {b: r3(v) for b, v in c["walk_forward"]["standalone_sweep_neigh_median_sharpe"].items()},
                             "book_G-P2_G-P3": {b: r3(v) for b, v in c["walk_forward"]["book_neigh_median_sharpe"].items()}, "C_BIL": {b: r3(v) for b, v in c["walk_forward"]["C_BIL"].items()}},
        }
    for key_str, entry in res["cells"].items():
        meta_e = res["cell_meta_primary"].get(key_str, {}).get("engine", {})
        out["cells"][key_str] = {
            "sa_sweep": {b: (r3(v["sharpe"]), r3(v["cagr"]), r3(v["max_dd"])) for b, v in entry["standalone_sweep"].items()},
            "sa_nosweep_FULL": (r3(entry["standalone_nosweep"]["FULL"]["sharpe"]), r3(entry["standalone_nosweep"]["FULL"]["cagr"]), r3(entry["standalone_nosweep"]["FULL"]["max_dd"])),
            "beta_spy": r3(entry["beta_spy_full"]), "corr_taa": r3(entry["corr_taa_2008_2026"]), "corr_L": r3(entry["corr_L_full"]),
            "book_sweep": {b: (r3(v["sharpe"]), r3(v["max_dd"])) for b, v in entry["books"]["sweep_engine"].items()},
            "book_margins_vs_C_BIL": {b: r3(v["sharpe"] - res["controls"]["C_BIL"][b]["sharpe"]) for b, v in entry["books"]["sweep_engine"].items()},
            "book_nosweep_G_FULL": r3(entry["books"]["nosweep_engine"]["G-FULL"]["sharpe"]),
            "cross_books_G_FULL": {u: r3(v["G-FULL"]["sharpe"]) for u, v in entry["cross_universe_books_sweep_engine"].items()},
            "cross_sa_FULL": {u: r3(v["sharpe"]) for u, v in entry["cross_universe_standalone_full_sweep"].items()},
            "turnover": r3(meta_e.get("turnover_x_per_year")), "exposure": r3(meta_e.get("mean_exposure")), "round_trips_per_year": r3(meta_e.get("round_trips_per_year")),
            "holding_median": r3(meta_e.get("holding_sessions_median")), "terminal_pnl_share": r3(meta_e.get("terminal_pnl_share")),
            "cap21_max5pct_M": r3(meta_e.get("capacity_2021_2026", {}).get("aum_max_5pct_usd", np.nan) / 1e6), "cap21_p95_1pct_M": r3(meta_e.get("capacity_2021_2026", {}).get("aum_p95_1pct_usd", np.nan) / 1e6),
        }
        if "m" in meta_e:
            m = meta_e["m"]
            out["m_diagnostics"][key_str] = {"events_per_year": r3(m["events_per_year"]), "confirmations_per_year": r3(m["confirmations_per_year"]), "entries_per_year": r3(m["entries_per_year"]),
                                             "entries_total": m["entries_total_int"], "entries_by_year": m["entries_by_year"], "precision_terminal_within_252": r3(m["precision_terminal_within_252"]),
                                             "exits_by_reason": {k: {kk: r3(vv) for kk, vv in v.items()} for k, v in m["exits_by_reason"].items()},
                                             "break_fill_vs_level_mean": r3(m["break_fill_vs_level_mean"]), "break_fill_vs_level_p5": r3(m["break_fill_vs_level_p5"]),
                                             "queue_dropped_zero_share": m["queue_dropped_zero_share_int"], "queue_dropped_delisted": m["queue_dropped_delisted_int"], "queue_left_at_end": m["queue_left_at_end_int"]}
        if "s" in meta_e:
            out["s_diagnostics"][key_str] = {"score_coverage": r3(meta_e["s"]["score_coverage_mean"]), "decisions": meta_e["s"]["decisions_int"], "invested_decisions": meta_e["s"]["invested_decisions_int"],
                                             "turnover": r3(meta_e.get("turnover_x_per_year")), "beta_spy": r3(entry["beta_spy_full"])}
    mult = res["multiplicity"]
    rc = mult["reality_check_book_g_full_vs_C_BIL"]
    out["multiplicity"] = {"configs": rc["configs_int"], "best_config": rc["best_config"], "best_obs_diff": r3(rc["best_obs_sharpe_diff"]), "reality_check_p": r3(rc["reality_check_p"]),
                           "share_above_C_BIL": r3(rc["share_configs_above_benchmark"]),
                           "paired": {k: {"obs": r3(v["obs_diff"]), "p": r3(v["p_one_sided"]), "ci90": [r3(x) for x in v["ci90"]]} for k, v in rc["paired"].items()},
                           "dsr": {k: {"sharpe": r3(v["sharpe_ann"]), "sr0": r3(v["sr0_ann"]), "dsr_prob": r3(v["dsr_prob"]), "cross_trial_sd": r3(v["cross_trial_sd_ann"])} for k, v in mult["deflated_sharpe_standalone_sweep"].items()}}
    for form_str, d in res.get("timing_luck_S", {}).items():
        out["timing_luck_S"][form_str] = {"standalone": {k: r3(v) for k, v in d["standalone_full"].items() if k != "by_offset"}, "book": {k: r3(v) for k, v in d["book_g_full"].items() if k != "by_offset"}}
    out["cross_universe_spearman"] = {f: d["spearman_vs_primary"] for f, d in res["cross_universe"].items()}
    out["invariance"] = {"v1": {k: {"differing_cells": v["cells_with_position_or_decision_difference"], "phantom_free_nav": v["max_rel_nav_diff_phantom_free_over_cells_float"], "engine_rule_nav": v["max_rel_nav_diff_engine_rule_over_cells_float"]} for k, v in (res.get("invariance_v1") or {}).items()},
                         "v2": res.get("invariance_v2")}
    out["parity"] = {k: {kk: v[kk] for kk in ("daily_return_corr_float", "max_abs_daily_diff_float", "cagr_gap_pp_float", "position_mismatch_sessions_int", "passed_bool")} for k, v in (res.get("parity_gate") or {}).items() if k != "gate_passed_bool"}
    (R / "report_extract.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
