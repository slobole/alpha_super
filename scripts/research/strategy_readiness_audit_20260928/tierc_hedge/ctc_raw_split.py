"""CTC future-split check in raw historical share units (engine opt-in historical_share_units_bool=True):
the legacy-unit fill differences at SPY k=0.1 should vanish."""
import numpy as np
import pandas as pd
import tc_common as c


class RawCtc(c.ctc.CrisisTrendCoreStrategy):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.historical_share_units_bool = True


def keys(s):
    tx = s.get_transactions()
    return set(zip(pd.to_datetime(tx["bar"]), tx["asset"], np.sign(tx["amount"].astype(float)).astype(int)))


df = c.load_ctc_workaround()
b = c.run_ctc(df, strategy_cls=RawCtc)
out = {"raw_baseline_cagr": c.metrics(c.strat_returns(b))["cagr_pct"]}
for sym, k in (("SPY", 0.1), ("SPY", 40.0)):
    pdf = c.rescale_namespace(c.rescale_namespace(df, sym, k), c.ctc.signal_namespace_str(sym), k)
    o = c.run_ctc(pdf, strategy_cls=RawCtc)
    kb, ko = keys(b), keys(o)
    out[f"{sym}_{k}"] = {"fills_only_base": len(kb - ko), "fills_only_other": len(ko - kb),
                         "final_tv_rel_diff": float(o.total_value_series.iloc[-1] / b.total_value_series.iloc[-1] - 1)}
    print(sym, k, out[f"{sym}_{k}"], flush=True)
c.dump(out, "ctc/a2_split_raw_units.json")
