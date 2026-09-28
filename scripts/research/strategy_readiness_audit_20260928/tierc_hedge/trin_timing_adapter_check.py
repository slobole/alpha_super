"""Handoff claim check: does the Trinity ExecutionTimingAnalysis adapter double-subtract dividends in raw mode, and
does it change the vol-overlay decision?

Runs only the default cell (same_open/same_open, vanilla_current_bar) of ExecutionTimingAnalyzer for:
  legacy  + TrinityVolControlTimingStrategy (production adapter)
  raw     + TrinityVolControlTimingStrategy (historical_share_units_bool=True)
  legacy  + TrinityVolControlStrategy (no adapter)
and compares decisions (daily target weights) and NAV to the vanilla run_daily baseline.
"""
import pickle
import numpy as np
import pandas as pd
import tc_common as c
from alpha.engine.execution_timing import ExecutionTimingAnalyzer

trin = c.trin
inputs = trin.build_execution_timing_analysis_inputs()
pricing = inputs["pricing_data_df"]
cal = inputs["calendar_idx"]
base = pickle.load(open(c.CACHE / "trin_baseline.pkl", "rb"))
base_tw = base["daily_target"]
base_tv = base["results"]["total_value"].astype(float)


def factory(cls, raw):
    def f():
        s = trin._build_trinity_strategy(trin.DEFAULT_CONFIG, float(trin.DEFAULT_CONFIG.capital_base_float), cls)
        if hasattr(s, "timing_pricing_data_df"):
            s.timing_pricing_data_df = pricing
        s.historical_share_units_bool = bool(raw)
        return s
    return f


def run(cls, raw):
    an = ExecutionTimingAnalyzer(
        strategy_factory_fn=factory(cls, raw), pricing_data_df=pricing, calendar_idx=cal,
        entry_timing_str_tuple=("same_open",), exit_timing_str_tuple=("same_open",),
        save_output_bool=False, audit_override_bool=False, order_generation_mode_str="vanilla_current_bar",
        risk_model_str="taa_rebalance", default_entry_timing_str="same_open", default_exit_timing_str="same_open")
    res = an.run()
    s = list(res.strategy_map.values())[0] if hasattr(res, "strategy_map") else None
    return res, s


def cmp_tw(a, b):
    idx = a.index.intersection(b.index)
    cols = [x for x in a.columns if x in b.columns]
    d = (a.loc[idx, cols].astype(float) - b.loc[idx, cols].astype(float)).abs().max(axis=1)
    return {"n": int(len(idx)), "n_diff_gt_1e-9": int((d > 1e-9).sum()), "max_abs": float(d.max()),
            "first": [str(x.date()) for x in d[d > 1e-9].index[:5]]}


out = {}
for label, cls, raw in (("legacy_adapter", trin.TrinityVolControlTimingStrategy, False),
                        ("raw_adapter", trin.TrinityVolControlTimingStrategy, True),
                        ("legacy_no_adapter", trin.TrinityVolControlStrategy, False)):
    res, s = run(cls, raw)
    if s is None:
        out[label] = {"error": "no strategy_map on result", "attrs": [a for a in dir(res) if not a.startswith("_")]}
        continue
    tv = s.results["total_value"].astype(float)
    tw = s.daily_target_weights
    out[label] = {
        "final_tv": float(tv.iloc[-1]),
        "final_tv_vanilla": float(base_tv.iloc[-1]),
        "decisions_vs_vanilla": cmp_tw(tw, base_tw),
        "dividend_net_total": float(getattr(s, "dividend_cash_net_total_float", np.nan)),
        "dividend_events": int(len(s.get_dividend_ledger())),
    }
    out[label + "_tw"] = None
    with (c.CACHE / f"trin_timing_{label}.pkl").open("wb") as fh:
        pickle.dump({"tw": tw, "tv": tv}, fh)
    print(label, out[label], flush=True)

a = pickle.load(open(c.CACHE / "trin_timing_legacy_adapter.pkl", "rb"))
b = pickle.load(open(c.CACHE / "trin_timing_raw_adapter.pkl", "rb"))
n = pickle.load(open(c.CACHE / "trin_timing_legacy_no_adapter.pkl", "rb"))
out["raw_vs_legacy_adapter_decisions"] = cmp_tw(b["tw"], a["tw"])
out["adapter_vs_no_adapter_decisions"] = cmp_tw(a["tw"], n["tw"])
out = {k: v for k, v in out.items() if v is not None}
c.dump(out, "trin/timing_adapter_check.json")
print(out)
