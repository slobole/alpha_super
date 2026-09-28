"""QPI candidate-filter NaN diagnostics and RSI2 / NATR NaN-poisoning check on real data (research-only).

QPI drops a member from the candidate list when any of Close, Turnover, qpi_value_ser, sma_200_price_ser,
three_day_return_ser, ibs_value_ser is NaN (strategy_mr_qpi_ibs_rsi_exit.py:436).  This measures, over PIT
member-days 2004-01-02..2026-09-25, how many are dropped and why, and whether any series goes permanently NaN after
it first becomes valid (a mid-series NaN would poison talib's recursive RSI2 / NATR and silently disable exits or
ranking for that name).

Usage: uv run python aud_nan_diag.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import aud_common as ac
import aud_data
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import QPIIbsRsiExitStrategy


def main() -> None:
    data = aud_data.load()
    pricing, universe = data["pricing_df"], data["universe_trimmed"]
    del data
    sig = ac.make("qpi", universe, cls=QPIIbsRsiExitStrategy).compute_signals(pricing)
    cal = pricing.index[(pricing.index >= pd.Timestamp("2004-01-02")) & (pricing.index <= ac.STUDY_END)]
    uni = universe.reindex(index=cal, method="ffill").fillna(0).astype(int)
    req = ["Close", "Turnover", "qpi_value_ser", "sma_200_price_ser", "three_day_return_ser", "ibs_value_ser"]
    member_days = 0
    drop = {f: 0 for f in req}
    drop_any = 0
    drop_close_ok_only_qpi = 0
    ibs_nan_zero_range = 0
    for s in uni.columns:
        if (s, "Close") not in sig.columns:
            continue
        m = uni[s].to_numpy() == 1
        if not m.any():
            continue
        member_days += int(m.sum())
        frame = sig.loc[cal, [(s, f) for f in req]]
        frame.columns = req
        nan = frame.isna().to_numpy()[m]
        for j, f in enumerate(req):
            drop[f] += int(nan[:, j].sum())
        drop_any += int(nan.any(axis=1).sum())
        only_qpi = nan[:, req.index("qpi_value_ser")] & ~np.delete(nan, req.index("qpi_value_ser"), axis=1).any(axis=1)
        drop_close_ok_only_qpi += int(only_qpi.sum())
        hl = (sig.loc[cal, (s, "High")] - sig.loc[cal, (s, "Low")]).to_numpy()[m]
        ibs_nan_zero_range += int((hl == 0).sum())
    # permanent NaN after first valid value (talib recursive poisoning or data gaps)
    poison = {"rsi2_value_ser": [], "qpi_value_ser": []}
    for s in sig.columns.get_level_values(0).unique():
        if (s, "rsi2_value_ser") not in sig.columns:
            continue
        c = sig[(s, "Close")]
        for f in poison:
            x = sig[(s, f)]
            fv = x.first_valid_index()
            if fv is None:
                continue
            after = x.loc[fv:]
            close_after = c.loc[fv:]
            n_bad = int((after.isna() & close_after.notna()).sum())
            if f == "rsi2_value_ser" and n_bad > 0:
                poison[f].append((str(s), n_bad))
            if f == "qpi_value_ser" and n_bad > 0:
                poison[f].append((str(s), n_bad))
    del sig
    dsig = ac.make("dv2", universe, cls=DVO2Strategy).compute_signals(pricing)
    natr_poison = []
    for s in dsig.columns.get_level_values(0).unique():
        if (s, "natr") not in dsig.columns:
            continue
        x, c = dsig[(s, "natr")], dsig[(s, "Close")]
        fv = x.first_valid_index()
        if fv is None:
            continue
        n_bad = int((x.loc[fv:].isna() & c.loc[fv:].notna()).sum())
        if n_bad:
            natr_poison.append((str(s), n_bad))
    out = {"qpi_member_days": member_days, "qpi_member_days_dropped_any_nan": drop_any,
           "qpi_share_dropped": drop_any / member_days, "qpi_drop_by_field": drop,
           "qpi_dropped_only_because_qpi_nan": drop_close_ok_only_qpi,
           "qpi_member_days_zero_range_bar": ibs_nan_zero_range,
           "rsi2_nan_after_first_valid_symbols": len(poison["rsi2_value_ser"]),
           "rsi2_examples": poison["rsi2_value_ser"][:10],
           "qpi_nan_after_first_valid_symbols": len(poison["qpi_value_ser"]),
           "qpi_nan_examples": sorted(poison["qpi_value_ser"], key=lambda t: -t[1])[:10],
           "natr_nan_after_first_valid_symbols": len(natr_poison), "natr_examples": natr_poison[:10]}
    (ac.OUT / "nan_diag.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
