"""TAA follow-ups.

(1) Fee-only counterfactual: re-run TAA 3x / 1/N with the commission charged on RAW shares (adjusted shares / k,
    k = Unadjusted/Adjusted close at the fill bar) and everything else unchanged. Measures the CAGR effect of the
    adjusted-unit fee (TQQQ k up to 96).
(2) DTB3 hurdle sanity: prove the lag patch is live (the lagged monthly hurdle differs) and report the smallest
    |momentum_score - cash_hurdle| margin across all decisions vs the largest one-session DTB3 change, which bounds
    how far a publication lag could move any decision.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import alpha.data.fred_loader as fred_loader_module  # noqa: E402

fred_loader_module.urlopen = lambda *a, **k: (_ for _ in ()).throw(URLError("offline"))

from alpha.engine.strategy import Strategy  # noqa: E402
import rq_taa_checks as rq  # noqa: E402

OUT_DIR_PATH = rq.OUT_DIR_PATH


def _raw_fee(self, prices_df, asset_str, amount_float, anchor_bar_ts):
    key_u, key_c = (asset_str, "Unadjusted Close"), (asset_str, "Close")
    if key_u in prices_df.columns and anchor_bar_ts in prices_df.index:
        k_float = float(prices_df.loc[anchor_bar_ts, key_u]) / float(prices_df.loc[anchor_bar_ts, key_c])
        if np.isfinite(k_float) and k_float > 0:
            return self._compute_commission(np.floor(abs(amount_float) / k_float + 1e-9))
    return self._compute_commission(amount_float)


def fee_counterfactual(variant_key_str: str) -> dict:
    base = rq._run(variant_key_str)
    real = Strategy._compute_execution_commission_float
    Strategy._compute_execution_commission_float = _raw_fee
    try:
        raw = rq._run(variant_key_str)
    finally:
        Strategy._compute_execution_commission_float = real
    mb, mr = rq._metrics(base.results["total_value"]), rq._metrics(raw.results["total_value"])
    return {"production": mb, "raw_share_fees": mr,
            "fees_production": float(base.get_transactions()["commission"].sum()),
            "fees_raw": float(raw.get_transactions()["commission"].sum()),
            "delta_cagr_pp": 100 * (mr["cagr"] - mb["cagr"]), "delta_sharpe": mr["sharpe"] - mb["sharpe"]}


def hurdle_margin(variant_key_str: str) -> dict:
    captured = {}
    real_fn = rq.base_module.compute_month_end_weight_df

    def spy(signal_close_df, cash_return_ser, config):
        captured.setdefault("cash", []).append(cash_return_ser.resample("ME").last())
        score_df, weight_df = real_fn(signal_close_df, cash_return_ser, config)
        captured.setdefault("score", []).append(score_df)
        return score_df, weight_df

    rq.base_module.compute_month_end_weight_df = spy
    try:
        rq._month_end_weights(variant_key_str, rq._config(variant_key_str))
        real_loader, lagged = rq._patched_cash_loader("session_lag_1")
        rq.base_module.load_cash_return_ser_and_snapshot = lagged
        try:
            rq._month_end_weights(variant_key_str, rq._config(variant_key_str))
        finally:
            rq.base_module.load_cash_return_ser_and_snapshot = real_loader
    finally:
        rq.base_module.compute_month_end_weight_df = real_fn
    cash_base, cash_lag = captured["cash"][0], captured["cash"][1]
    score_df = captured["score"][0]
    common = score_df.dropna().index.intersection(cash_base.dropna().index).intersection(cash_lag.dropna().index)
    margin_df = score_df.loc[common].sub(cash_base.loc[common], axis=0).abs()
    hurdle_shift = (cash_lag.loc[common] - cash_base.loc[common]).abs()
    return {
        "months": int(len(common)),
        "months_where_lag_changes_hurdle": int((hurdle_shift > 0).sum()),
        "max_hurdle_change_monthly_return": float(hurdle_shift.max()),
        "min_score_minus_hurdle_margin": float(margin_df.min().min()),
        "min_margin_month": str(margin_df.min(axis=1).idxmin().date()),
    }


def main() -> None:
    out = {}
    for v in rq.VARIANT_DICT:
        out[v] = {"fee_counterfactual": fee_counterfactual(v), "hurdle_margin": hurdle_margin(v)}
        print(v, json.dumps(out[v]), flush=True)
    (OUT_DIR_PATH / "rq_taa_fee_and_hurdle_margin.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
