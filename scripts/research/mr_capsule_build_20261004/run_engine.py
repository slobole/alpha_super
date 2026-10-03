"""MR capsule build check: run one capsule pod in the real engine (2004 -> latest data) and save NAV, trades, diagnostics.

Usage: python run_engine.py dv2|hpi [parked|cash|bil]
    parked = the capsule spec (SPMO while the gate is closed, BIL otherwise); cash = parking disabled (idle cash at
    0%); bil = idle cash all in BIL (the research "T-bills" reference, with BIL's real costs and withholding).
Writes results/research/mr_capsule_build_20261004/<pod>_<mode>_{nav.csv,transactions.csv,diagnostics.json}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/mr_capsule_build_20261004"


def main(pod_str: str, mode_str: str = "parked") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if mode_str not in ("parked", "cash", "bil"):
        raise ValueError(mode_str)
    parking_bool = mode_str != "cash"
    start_float = time.time()
    if pod_str == "dv2":
        from strategies.mr_capsule.strategy_mr_dv2_vix_gated import run_variant
    elif pod_str == "hpi":
        from strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated import run_variant
    else:
        raise ValueError(pod_str)
    strategy = run_variant(show_display_bool=False, save_results_bool=mode_str == "parked", parking_enabled_bool=parking_bool,
                           spmo_parking_enabled_bool=mode_str == "parked")
    tag_str = f"{pod_str}_{mode_str}"
    result_df = strategy.results
    result_df[[c for c in ("total_value", "cash", "portfolio_value") if c in result_df.columns]].to_csv(OUT / f"{tag_str}_nav.csv")
    strategy.get_transactions().to_csv(OUT / f"{tag_str}_transactions.csv", index=False)
    cash_fraction_ser = result_df["cash"].astype(float) / result_df["total_value"].astype(float)
    diagnostic_dict = dict(strategy._accounting_policy_dict)
    diagnostic_dict.update(
        runtime_sec_float=time.time() - start_float,
        session_int=len(result_df),
        negative_cash_session_int=int((result_df["cash"].astype(float) < 0).sum()),
        min_cash_fraction_float=float(cash_fraction_ser.min()),
        median_cash_fraction_float=float(cash_fraction_ser.median()),
    )
    (OUT / f"{tag_str}_diagnostics.json").write_text(json.dumps(diagnostic_dict, indent=2, default=str), encoding="utf-8")
    print("done", tag_str, len(result_df), round(diagnostic_dict["runtime_sec_float"]), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
