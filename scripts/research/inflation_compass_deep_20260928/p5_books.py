"""Phase 5: Compass's role in the fund-menu books versus T-bills in the same slot - see SPEC_FROZEN.md.

Reuses the leakage-hunt book machinery (main checkout, research scripts): fund-menu sleeve inventory with the
fixed momentum sleeves, the HPI live-slot restatement and the causal Compass sleeve; independent pods with an
annual reset (fund_menu common.book_return_ser); window 2012-10-02..2026-08-19.
Compass variants enter as daily-return deltas (replica variant minus replica baseline) added to the causal sleeve.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
spec = importlib.util.spec_from_file_location("cm", HERE / "common.py")
cm = importlib.util.module_from_spec(spec)
sys.modules["cm"] = cm
spec.loader.exec_module(cm)

# the book machinery lives in the main checkout; its modules import each other as `common`, `shelf_books`, ...
for p in (MAIN / "scripts" / "research" / "fund_menu_20260923", MAIN / "scripts" / "research" / "growth_shelf_20260924",
          MAIN / "scripts" / "research" / "growth_shelf_v2_20260926", MAIN / "scripts" / "research" / "leakage_hunt_20260927"):
    sys.path.insert(0, str(p))
import book_impact as bi  # noqa: E402
import common as fm  # noqa: E402  (fund-menu common)
import shelf_books as sb  # noqa: E402
import yaml  # noqa: E402


def stats(r: pd.Series) -> dict:
    v = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = v.iloc[-1] ** (1 / yrs) - 1
    dd = (v / v.cummax() - 1).min()
    return {"cagr": cagr * 100, "sharpe": r.mean() / r.std() * np.sqrt(252), "maxdd": dd * 100,
            "calmar": cagr / abs(dd)}


def tbill_returns(index: pd.DatetimeIndex) -> pd.Series:
    d = cm.fred_series("DTB3")
    full = d.reindex(d.index.union(index)).ffill()
    # *** CRITICAL*** yesterday's published rate accrues over the calendar gap to today
    rate = full.shift(1).reindex(index) / 100.0
    gap = pd.Series(np.r_[1.0, np.diff(index.values).astype("timedelta64[D]").astype(float)], index=index)
    return (rate * gap / 360.0).fillna(0.0)


def main():
    sleeve = sb.load_inputs(fixed=False)["sleeve"]
    refresh = pd.read_csv(bi.REFRESH / "sleeve_series_incl_2008.csv.gz", index_col=0, parse_dates=True)
    for new in bi.FIXED_MOMENTUM.values():
        sleeve[new] = refresh[new].reindex(sleeve.index)
    inventory = pd.read_csv(fm.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True)
    for alias in ("infl_compass", "disp_kie_ihi_sma"):
        sleeve[alias] = inventory[alias].reindex(sleeve.index)
    window = sleeve.loc[bi.START:bi.END].copy()
    window["hpi_vote"] = window["hpi_vote"] + bi.hpi_delta().reindex(window.index).fillna(0.0)
    window["infl_compass"] = window["infl_compass"] + bi.compass_delta().reindex(window.index).fillna(0.0)
    window["tbill"] = tbill_returns(window.index)

    # Compass variants as deltas vs the replica baseline
    p4 = pd.read_pickle(cm.OUT / "p4_navs.pkl")
    offs = pd.read_pickle(cm.OUT / "p2b_offset_avg_nav.pkl")["offset_avg_nav"]
    base_r = p4["BASE"].pct_change()
    variants = {"C1 ensemble 27": p4["C1 ensemble 27"], "C4 tranching x4": p4["C4 tranching x4"],
                "C7 ensemble + tranching": p4["C7 ensemble + tranching"], "C2 QQQ for XLK": p4["C2 QQQ for XLK"],
                "offset-average (diagnostic)": offs}
    for name, nav in variants.items():
        delta = (nav.pct_change() - base_r).reindex(window.index).fillna(0.0)
        window[f"compass::{name}"] = window["infl_compass"] + delta

    rows = []
    for path in sorted((MAIN / "portfolios").glob("fund_menu_*.yaml")):
        spec_d = yaml.safe_load(path.read_text(encoding="utf-8"))
        pods = spec_d.get("pods") or spec_d.get("pod_list") or []
        w = {}
        for pod in pods:
            a = bi.MENU_ALIAS[pod["strategy_import_str"]]
            a = bi.FIXED_MOMENTUM.get(a, a)
            w[a] = w.get(a, 0.0) + float(pod["weight_float"])
        tot = sum(w.values())
        w = {a: v / tot for a, v in w.items()}
        cw = w.get("infl_compass", 0.0)
        if cw == 0:
            continue
        book = path.stem.replace("fund_menu_", "")
        others = {a: v for a, v in w.items() if a != "infl_compass"}
        so = sum(others.values())
        cases = {"(a) as is": w,
                 "(b) Compass slot -> T-bills": {**others, "tbill": cw},
                 "(c) Compass slot -> other sleeves pro rata": {a: v / so for a, v in others.items()}}
        for name in variants:
            cases[f"(d) {name}"] = {**others, f"compass::{name}": cw}
        for case, weights in cases.items():
            r, _ = fm.book_return_ser(window, weights, "annual")
            ex = stats(r - window["tbill"])  # excess of T-bill (spec: report both conventions)
            rows.append({"book": book, "compass_weight": cw, "case": case, **stats(r),
                         "xs_sharpe": ex["sharpe"], "xs_calmar": ex["calmar"]})
    t = pd.DataFrame(rows)
    t.to_csv(cm.OUT / "p5_books.csv", index=False)
    print(t.round(3).to_string(index=False))
    # sleeve-level: Compass vs T-bills and correlations with the book's other sleeves (window)
    comp = window["infl_compass"]
    corr = window.drop(columns=[c for c in window.columns if c.startswith("compass::")]).corr()["infl_compass"]
    out = {"compass_sleeve": stats(comp), "tbill": stats(window["tbill"]),
           "corr_with_compass": corr.drop("infl_compass").dropna().round(3).sort_values().to_dict()}
    (cm.OUT / "p5_meta.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
