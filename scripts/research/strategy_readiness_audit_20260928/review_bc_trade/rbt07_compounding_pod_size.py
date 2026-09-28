"""Review BC-trade RBT07: how large a "USD 30K" backtest pod really is during the governing last-3y window.

The auditors' small-account runs start at USD 30K on the backtest start date and compound. The owner's pods are
rebalanced inside a USD 30K book, so they stay near 12-30K. This prints NAV(2023-09-25)/NAV(start) from the
auditors' saved 100K equity curves (capital scaling is linear to <0.1 pp, A9), i.e. the size a 30K-start run has
when the last-3y window begins.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rbt02_uniform_capacity_and_friction as R  # noqa: E402
RES = R.RES

def mult(s: pd.Series) -> dict:
    s = s.copy(); s.index = pd.to_datetime(s.index); s = s.sort_index()
    return {"start": str(s.index[0].date()), "mult_at_2023_09_25": round(float(s.asof(R.L3Y) / s.iloc[0]), 2),
            "mult_at_end": round(float(s.iloc[-1] / s.iloc[0]), 2),
            "size_of_30k_start_pod_at_2023_09_25_usd": round(30000 * float(s.asof(R.L3Y) / s.iloc[0]), -3)}

out = {}
for k in ("vox", "eom", "xlc", "xlc200", "kie200"):
    out[k] = mult(pd.read_parquet(RES / f"tierb_etf_mr/equity_{k}.parquet")["total_value"])
out["etf_dv2"] = mult(pd.read_csv(RES / "tierbc_dv2etf_taa2x/etf/baseline_nav.csv", index_col=0)["total_value"])
for k in ("qld_1n", "btal_qld_1n", "lin_qqq"):
    out[k] = mult(pd.read_csv(RES / f"tierbc_dv2etf_taa2x/taa2x/nav_{k}_100k.csv", index_col=0)["total_value"])
for k in ("ctc", "vixm", "trin"):
    out[k] = mult(pickle.load(open(RES / f"tierc_hedge/_cache/{k}_baseline.pkl", "rb"))["results"]["portfolio_value"])
for k, v in out.items():
    print(k, v)
(R.OUT / "rbt07_compounding_pod_size.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
