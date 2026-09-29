"""Condensed report extract from results.json (research only): every number quoted in the final report comes from here.

    python scripts/research/merger_arb_v2_20260927/report_extract.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merger_arb_v2_20260927 import common  # noqa: E402

R = common.RESULTS_DIR_PATH


def r3(x, n=3):
    return round(float(x), n) if isinstance(x, (int, float, np.floating)) and x is not None and np.isfinite(x) else x


def main() -> None:
    res = json.loads((R / "results.json").read_text())
    c_bil = res["controls"]["C_BIL"]
    out: dict = {"decision": res["decision"], "controls": {k: {b: (r3(v["sharpe"]), r3(v["max_dd"]), r3(v["cagr"])) for b, v in blocks.items()} for k, blocks in res["controls"].items()},
                 "controls_stress_C_BIL": {b: r3(v["sharpe"]) for b, v in res["controls_stress"]["C_BIL"].items()}, "legs": res["legs"], "candidates": {}, "cells": {}, "sensitivities_V0": {},
                 "multiplicity": {}, "small_account_V0": res.get("small_account_V0"), "detection": {"pass1": res.get("detection_pass1"), "stats": res.get("detection_detection_stats")},
                 "parity": {k: v for k, v in (res.get("parity_gate") or {}).items()}, "invariance_v1": res.get("invariance_v1"), "invariance_v2": res.get("invariance_v2")}
    for c in res["candidates"]:
        r = c["rule"]
        out["candidates"][c["stage"]] = {
            "centre": c["centre"], "neighbourhood": c["neighbourhood"], "plateau_full_sharpe_sweep": r3(c["plateau_full_sharpe"]),
            "centre_standalone_sweep": {b: (r3(v["sharpe"]), r3(v["cagr"], 4), r3(v["max_dd"], 4)) for b, v in c["centre_standalone_sweep"].items()},
            "margins_vs_C_BIL": {b: r3(v, 4) for b, v in r["engine"]["sharpe_margin"].items()}, "min_margin": r3(r["engine"]["min_margin"], 4), "dd_gap_pp": {b: r3(v) for b, v in r["engine"]["dd_gap_pp"].items()},
            "R1": r["engine"]["R1"], "R2": r["engine"]["R2"], "stress_margins": {b: r3(v, 4) for b, v in r["stress"]["sharpe_margin"].items()}, "R3": r["R3"],
            "R4": {h: (r3(v["neigh_median_book_g_full_sharpe"], 4), r3(v["C_BIL_g_full_sharpe"], 4), v["pass"]) for h, v in r["R4"].items()},
            "R5": {"aum_max_5pct_M": r3((r["R5"]["aum_max_5pct_usd_2021_2026"] or 0) / 1e6, 1), "aum_p95_1pct_M": r3((r["R5"]["aum_p95_1pct_usd_2021_2026"] or 0) / 1e6, 1), "pass": r["R5"]["pass"]}, "passes": r["passes"],
            "labels": {"vs_G3": {b: r3(v) for b, v in r["labels"]["vs_G3"]["sharpe_margin"].items()}, "beats_G3_all": r["labels"]["vs_G3"]["R1"],
                       "vs_C_SPY": {b: r3(v) for b, v in r["labels"]["vs_C_SPY"]["sharpe_margin"].items()}, "beats_C_SPY_all": r["labels"]["vs_C_SPY"]["R1"],
                       "vs_C_MNA": {b: r3(v) for b, v in r["labels"]["vs_C_MNA"]["sharpe_margin"].items()}, "vs_C_MNA_g_full": r3(r["labels"]["vs_C_MNA"]["g_full_margin"]), "beats_C_MNA_P2_P3": r["labels"]["vs_C_MNA"]["R1"],
                       "nosweep_vs_C_CASH0": {b: r3(v, 4) for b, v in r["labels"]["nosweep_vs_C_CASH0"]["sharpe_margin"].items()}, "nosweep_beats_C_CASH0_all": r["labels"]["nosweep_vs_C_CASH0"]["R1"],
                       "owner_gate_centre": {k: (r3(v, 4) if k != "pass" else v) for k, v in r["labels"]["owner_gate_centre"].items()},
                       "owner_gate_neigh": {k: (r3(v, 4) if k != "pass" else v) for k, v in r["labels"]["owner_gate_neighbourhood_median"].items()}},
            "neigh_median_books": {b: (r3(v["sharpe"], 4), r3(v["max_dd"], 4), r3(v["cagr"], 4)) for b, v in r["neigh_median_books_sweep_engine"].items()},
            "walk_forward": {"centre_by_P1": c["walk_forward"]["centre_by_P1"], "standalone_P2_P3": {b: r3(v) for b, v in c["walk_forward"]["standalone_sweep_neigh_median_sharpe"].items()},
                             "book_G-P2_G-P3": {b: r3(v, 4) for b, v in c["walk_forward"]["book_neigh_median_sharpe"].items()}, "C_BIL": {b: r3(v, 4) for b, v in c["walk_forward"]["C_BIL"].items()}},
        }
    for key_str, e in res["cells"].items():
        m = res["cell_meta"].get(key_str, {}).get("engine", {})
        v = m.get("v", {})
        out["cells"][key_str] = {
            "sa_sweep_FULL": (r3(e["standalone_sweep"]["FULL"]["sharpe"]), r3(e["standalone_sweep"]["FULL"]["cagr"], 4), r3(e["standalone_sweep"]["FULL"]["max_dd"], 4)),
            "sa_sweep_blocks_sharpe": {b: r3(vv["sharpe"]) for b, vv in e["standalone_sweep"].items()},
            "sa_nosweep_FULL": (r3(e["standalone_nosweep"]["FULL"]["sharpe"]), r3(e["standalone_nosweep"]["FULL"]["cagr"], 4), r3(e["standalone_nosweep"]["FULL"]["max_dd"], 4)),
            "beta_spy": r3(e["beta_spy_full"]), "beta_spy_nosweep": r3(e["beta_spy_nosweep_full"]), "corr_taa": r3(e["corr_taa_2008_2026"]), "corr_L": r3(e["corr_L_full"]), "corr_mna": r3(e["corr_mna_2009_2026"]),
            "book_sweep": {b: (r3(vv["sharpe"], 4), r3(vv["max_dd"], 4)) for b, vv in e["books"]["sweep_engine"].items()},
            "book_margins_vs_C_BIL": {b: r3(vv["sharpe"] - c_bil[b]["sharpe"], 4) for b, vv in e["books"]["sweep_engine"].items()},
            "book_nosweep_G_FULL": r3(e["books"]["nosweep_engine"]["G-FULL"]["sharpe"], 4),
            "half_books_G_FULL": {h: r3(vv["G-FULL"]["sharpe"], 4) for h, vv in e["half_books_sweep_engine"].items()}, "half_sa_FULL": {h: r3(vv["sharpe"]) for h, vv in e["half_standalone_full_sweep"].items()},
            "events_per_year": r3(v.get("events_per_year"), 1), "confirmations_per_year": r3(v.get("confirmations_per_year"), 1), "entries_per_year": r3(v.get("entries_per_year"), 1), "entries_total": v.get("entries_total_int"),
            "positions_mean": r3(v.get("positions_mean"), 2), "positions_max": v.get("positions_max"), "share_sessions_no_position": r3(v.get("share_sessions_no_position")),
            "exposure": r3(m.get("mean_exposure")), "turnover": r3(m.get("turnover_x_per_year"), 2), "holding_median": r3(m.get("holding_sessions_median"), 1),
            "precision": r3(v.get("precision_terminal_within_252")), "exits": {k: {kk: (r3(vv, 4) if kk != "count_int" else vv) for kk, vv in d.items()} for k, d in v.get("exits_by_reason", {}).items()},
            "break_fill_mean_p5": (r3(v.get("break_fill_vs_level_mean"), 4), r3(v.get("break_fill_vs_level_p5"), 4)), "worst_episode": r3(v.get("worst_episode_return"), 4),
            "queue_drops": v.get("queue_drops"), "entries_by_year": v.get("entries_by_year"), "terminal_pnl_share": r3(m.get("terminal_pnl_share")),
            "cap21_max5pct_M": r3(m.get("capacity_2021_2026", {}).get("aum_max_5pct_usd", np.nan) / 1e6, 1), "cap21_p95_1pct_M": r3(m.get("capacity_2021_2026", {}).get("aum_p95_1pct_usd", np.nan) / 1e6, 1),
            "capfull_max5pct_M": r3(m.get("capacity_full", {}).get("aum_max_5pct_usd", np.nan) / 1e6, 2), "commission_total": r3(v.get("commission_total_usd"), 0), "total_pnl": r3(v.get("total_pnl_usd"), 0),
        }
    for label_str, d in res.get("sensitivities_V0", {}).items():
        out["sensitivities_V0"][label_str] = {"key": d["key"], "sa_sweep_FULL_sharpe": r3(d["standalone_sweep_FULL"]["sharpe"]), "sa_nosweep_FULL": (r3(d["standalone_nosweep_FULL"]["sharpe"]), r3(d["standalone_nosweep_FULL"]["cagr"], 4)),
                                              "book_margins_vs_C_BIL": {b: r3(v, 4) for b, v in d["book_margins_vs_C_BIL"].items()}, "precision": r3((d.get("meta_v") or {}).get("precision_terminal_within_252")),
                                              "exits": {k: (dd["count_int"], r3(dd["pnl_usd"], 0), r3(dd["mean_return"], 4)) for k, dd in (d.get("meta_v") or {}).get("exits_by_reason", {}).items()}}
    rc = res["multiplicity"]["reality_check_book_g_full_vs_C_BIL"]
    out["multiplicity"] = {"configs": rc["configs_int"], "best_config": rc["best_config"], "best_obs_diff": r3(rc["best_obs_sharpe_diff"], 4), "reality_check_p": r3(rc["reality_check_p"]), "share_above_C_BIL": r3(rc["share_configs_above_benchmark"]),
                           "paired": {k: {"obs": r3(v["obs_diff"], 4), "p": r3(v["p_one_sided"]), "ci90": [r3(x, 4) for x in v["ci90"]]} for k, v in rc["paired"].items()},
                           "dsr_sweep": {k: {"sharpe": r3(v["sharpe_ann"]), "sr0": r3(v["sr0_ann"]), "dsr_prob": r3(v["dsr_prob"]), "cross_trial_sd": r3(v["cross_trial_sd_ann"])} for k, v in res["multiplicity"]["deflated_sharpe_standalone_sweep"].items()},
                           "dsr_nosweep": {k: {"sharpe": r3(v["sharpe_ann"]), "sr0": r3(v["sr0_ann"]), "dsr_prob": r3(v["dsr_prob"]), "cross_trial_sd": r3(v["cross_trial_sd_ann"])} for k, v in res["multiplicity"]["deflated_sharpe_standalone_nosweep"].items()}}
    (R / "report_extract.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
