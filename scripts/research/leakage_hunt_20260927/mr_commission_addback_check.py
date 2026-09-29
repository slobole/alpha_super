"""Is growth_shelf_v2 commission_fix.py's fee-only add-back consistent with the engine's historical_share_units path?

Applies commission_fix.real_commission (R = Unadjusted Close / Close on the fill date) to this study's dv2_base and
hpi_base ledgers, adds the daily add-back to the engine returns, and compares with the *_hsu arms.
Writes mr/commission_addback_check.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import mr_common as mc
import mr_data

sys.path.insert(0, str(mc.REPO / "scripts" / "research" / "growth_shelf_v2_20260926"))
sys.path.insert(0, str(mc.REPO / "scripts" / "research" / "fund_menu_20260923"))
from commission_fix import real_commission  # noqa: E402


def stats(ret):
    nav = (1 + ret.fillna(0)).cumprod()
    years = (ret.index[-1] - ret.index[0]).days / 365.25
    return {"cagr": float(nav.iloc[-1] ** (1 / years) - 1), "sharpe": float(ret.mean() / ret.std() * np.sqrt(252))}


def main():
    out = {}
    for fam, base, hsu in (("dv2", "dv2_base", "dv2_hsu"), ("hpi", "hpi_base", "hpi_hsu")):
        pricing = mr_data.load(fam)["pricing_df"]
        ratio_df = pricing.xs("Unadjusted Close", axis=1, level=1) / pricing.xs("Close", axis=1, level=1)
        daily = pd.read_csv(mc.OUT / "full_runs" / base / "daily.csv.gz", index_col=0, parse_dates=True)
        tx = pd.read_csv(mc.OUT / "full_runs" / base / "transactions.csv.gz", parse_dates=["bar"])
        stacked = ratio_df.ffill().stack()
        r = stacked.reindex(pd.MultiIndex.from_arrays([tx["bar"], tx["asset"]])).to_numpy()
        real = real_commission(tx["commission"].to_numpy(), r)
        saving = pd.Series(tx["commission"].to_numpy() - real, index=tx["bar"]).groupby(level=0).sum()
        nav = daily["total_value"]
        add = (saving.reindex(nav.index).fillna(0) / nav.shift(1)).fillna(0)
        ret = nav.pct_change().fillna(0)
        hsu_nav = pd.read_csv(mc.OUT / "full_runs" / hsu / "daily.csv.gz", index_col=0, parse_dates=True)["total_value"]
        out[fam] = {"engine": stats(ret), "engine_plus_addback": stats(ret + add), "hsu_engine_path": stats(hsu_nav.pct_change().fillna(0)),
                    "engine_commission": float(tx["commission"].sum()), "real_commission_formula": float(real.sum()),
                    "missing_ratio": int(np.isnan(r).sum())}
    (mc.OUT / "commission_addback_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
