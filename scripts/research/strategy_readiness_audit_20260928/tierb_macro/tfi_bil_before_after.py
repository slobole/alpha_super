"""Tactical FI cash-vehicle change (2026-09-28): before/after evidence.

before = legacy ledger (cash_vehicle_str="dgs3mo_accrual", withholding 0%): must reproduce the
         published contract (book window 2012-10-02..2026-08-19: 2.706% CAGR, Sharpe 1.063).
after  = BIL position for the cash sleeve, 0% on residual cash, 25% withholding (new default).
Also reported: the two steps separately, the ALFRED point-in-time mode, and decision identity
(the monthly IEF/LQD/Cash targets must not change).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))

import strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd as tfi  # noqa: E402

OUTPUT_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/tierb_macro/tfi_bil_before_after.json"
WINDOW_DICT = {
    "full": ("2002-08-01", "2026-08-19"),
    "book": ("2012-10-02", "2026-08-19"),
    "from_2014_05": ("2014-05-01", "2026-08-19"),
    "last3y": ("2023-08-21", "2026-08-19"),
}


def metrics(nav_ser: pd.Series) -> dict:
    out = {}
    for label_str, (start_str, end_str) in WINDOW_DICT.items():
        window_ser = nav_ser.loc[start_str:end_str]
        ret_ser = window_ser.pct_change().dropna()
        years_float = len(ret_ser) / 252.0
        std_float = float(ret_ser.std())
        out[label_str] = {
            "cagr_pct": round(((window_ser.iloc[-1] / window_ser.iloc[0]) ** (1 / years_float) - 1) * 100, 3),
            "sharpe": round(float(ret_ser.mean() / std_float * np.sqrt(252)), 3) if std_float > 0 else None,
            "max_dd_pct": round(float((window_ser / window_ser.cummax() - 1).min()) * 100, 2),
        }
    return out


def run(label_str: str, **config_kwargs) -> tuple[dict, object]:
    config_obj = replace(tfi.DEFAULT_CONFIG, **config_kwargs)
    strategy_obj = tfi.run_variant(show_display_bool=False, save_results_bool=False, config_obj=config_obj,
                                   fred_data_mode_str=config_kwargs.get("fred_data_mode_str"))
    nav_ser = strategy_obj.results["total_value"].astype(float)
    nav_ser.index = pd.to_datetime(nav_ser.index)
    tx_df = strategy_obj.get_transactions()
    result_dict = {
        "metrics": metrics(nav_ser),
        "cash_interest_total_usd": round(float(strategy_obj.cash_interest_total_float), 0),
        "dividend_net_usd": round(float(getattr(strategy_obj, "dividend_cash_net_total_float", 0.0)), 0),
        "dividend_withheld_usd": round(float(getattr(strategy_obj, "dividend_withholding_total_float", 0.0)), 0),
        "fills_by_asset": tx_df["asset"].astype(str).value_counts().to_dict(),
        "min_cash_frac": round(float((strategy_obj.results["cash"].astype(float) / nav_ser.values).min()), 5),
        "accounting_policy_subset": {
            k: strategy_obj._accounting_policy_dict.get(k)
            for k in ("cash_vehicle_str", "positive_cash_rate_policy_str", "dividend_withholding_rate_float")
        },
    }
    print(label_str, json.dumps(result_dict["metrics"]["book"]), json.dumps(result_dict["metrics"]["full"]), flush=True)
    return result_dict, strategy_obj


def main() -> None:
    out = {}
    out["before_legacy_dgs3mo_0pct"], legacy_obj = run(
        "before", cash_vehicle_str=tfi.CASH_VEHICLE_DGS3MO_ACCRUAL_STR, dividend_withholding_rate_float=0.0
    )
    out["step1_legacy_accrual_25pct"], _ = run(
        "step1", cash_vehicle_str=tfi.CASH_VEHICLE_DGS3MO_ACCRUAL_STR, dividend_withholding_rate_float=0.25
    )
    out["step2_bil_0pct"], _ = run("step2", cash_vehicle_str=tfi.CASH_VEHICLE_BIL_STR, dividend_withholding_rate_float=0.0)
    out["after_bil_25pct_default"], new_obj = run("after")
    out["after_alfred_pit"], _ = run("after_alfred", fred_data_mode_str=tfi.FRED_DATA_MODE_ALFRED_PIT_STR)
    decisions_identical_bool = legacy_obj.rebalance_weight_df[["IEF", "LQD", "Cash"]].equals(
        new_obj.rebalance_weight_df[["IEF", "LQD", "Cash"]]
    )
    out["decision_targets_identical_bool"] = bool(decisions_identical_bool)
    out["published_contract_reproduced_bool"] = (
        abs(out["before_legacy_dgs3mo_0pct"]["metrics"]["book"]["cagr_pct"] - 2.706) < 0.01  # window-edge convention; bit-identity vs HEAD checked separately
        and abs(out["before_legacy_dgs3mo_0pct"]["metrics"]["book"]["sharpe"] - 1.063) < 0.002
    )
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("decisions identical:", decisions_identical_bool, "| contract reproduced:", out["published_contract_reproduced_bool"])


if __name__ == "__main__":
    main()
