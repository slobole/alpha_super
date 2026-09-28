"""Phase 3: sensitivity, reported as distributions (not used for selection) - see SPEC_FROZEN.md."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

import common as cm

THRESH = [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5]
BE = [20, 40, 60, 80, 120]
SLOPE = [20, 40, 60, 80, 120]
SMA = [100, 150, 200, 250]


def run(w, lag=1, fill="open", **kw):
    return cm.run_replica(cm.map_to_execution(w, cm.sessions(), lag=lag), fill=fill, **kw)


def row(name, nav, **extra):
    pt = cm.period_table(nav)
    r = {"variant": name, **extra}
    for per, m in pt.items():
        r[f"{per}_cagr"] = m["cagr"]; r[f"{per}_sharpe"] = m["sharpe"]
    r["maxdd"] = pt["ALL"]["maxdd"]
    return r


def main():
    sig = cm.signal_close_df()
    t5 = cm.fred_series("T5YIE")
    t5 = t5[t5.index <= pd.Timestamp(cm.MAIN_END_STR)]
    base_w, base_reg = cm.compass_weights(sig, t5)

    # 1) joint grid
    grid = []
    for th, be, sl, sma in itertools.product(THRESH, BE, SLOPE, SMA):
        p = cm.Params(threshold=th, be_lookback=be, slope_lookback=sl, sma=sma)
        w, reg = cm.compass_weights(sig, t5, p)
        nav = run(w)
        grid.append(row("grid", nav, threshold=th, be=be, slope=sl, sma=sma,
                        flips_vs_base=int((reg.reindex(base_reg.index)[["growth", "infl"]]
                                           != base_reg[["growth", "infl"]]).any(axis=1).sum())))
    g = pd.DataFrame(grid)
    g.to_csv(cm.OUT / "p3_grid.csv", index=False)
    desc = g[["ALL_cagr", "ALL_sharpe", "maxdd", "P1_sharpe", "P2_sharpe", "P3_sharpe"]].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3)
    print(desc.to_string())
    base_row = g[(g.threshold == 2.0) & (g.be == 60) & (g.slope == 60) & (g.sma == 200)].iloc[0]
    out = {"grid_n": len(g), "base_sharpe": base_row["ALL_sharpe"],
           "base_sharpe_percentile_in_grid": float((g["ALL_sharpe"] < base_row["ALL_sharpe"]).mean()),
           "grid_sharpe_std": float(g["ALL_sharpe"].std()),
           "grid_describe": desc.to_dict()}
    # Varadi's 1-D grid reproduced (others at base)
    one_d = []
    for col, vals in (("threshold", THRESH), ("be", BE), ("slope", SLOPE), ("sma", SMA)):
        base = {"threshold": 2.0, "be": 60, "slope": 60, "sma": 200}
        for v in vals:
            q = dict(base); q[col] = v
            r = g[(g.threshold == q["threshold"]) & (g.be == q["be"]) & (g.slope == q["slope"]) & (g.sma == q["sma"])]
            one_d.append({"axis": col, "value": v, "cagr": r["ALL_cagr"].iloc[0], "sharpe": r["ALL_sharpe"].iloc[0],
                          "maxdd": r["maxdd"].iloc[0]})
    od = pd.DataFrame(one_d)
    od.to_csv(cm.OUT / "p3_one_dim.csv", index=False)
    print(od.round(3).to_string(index=False))

    rows = []
    # 2) ties / anchor
    for name, p in (("base", cm.Params()), ("ties >=", cm.Params(strict_ties=False)),
                    ("anchor lagged 1 more session", cm.Params(anchor_lag_extra=1))):
        w, _ = cm.compass_weights(sig, t5, p)
        rows.append(row(name, run(w), group="ties_anchor"))
    # 3) rebalance-day offsets
    for off in range(-10, 11):
        w, _ = cm.compass_weights(sig, t5, cm.Params(), offset=off)
        rows.append(row(f"offset {off:+d}", run(w), group="offset", offset=off))
    # 4) execution timing
    rows.append(row("next close", run(base_w, lag=1, fill="close"), group="execution"))
    rows.append(row("second open", run(base_w, lag=2, fill="open"), group="execution"))
    # 5) instruments
    alt_maps = {
        "QQQ for XLK": {(True, True): {"XLE": 1.0}, (True, False): {"QQQ": 1.0}, (False, True): {"XLU": 1.0},
                        (False, False): {"XLP": 0.5, "IEF": 0.5}},
        "TLT for IEF": {**cm.REGIME_WEIGHTS, (False, False): {"XLP": 0.5, "TLT": 0.5}},
        "SHY for IEF": {**cm.REGIME_WEIGHTS, (False, False): {"XLP": 0.5, "SHY": 0.5}},
        "XLP only in slowdown": {**cm.REGIME_WEIGHTS, (False, False): {"XLP": 1.0}},
        "IEF only in slowdown": {**cm.REGIME_WEIGHTS, (False, False): {"IEF": 1.0}},
    }
    for name, cmap in alt_maps.items():
        w = cm.weights_from_regimes(base_reg, cmap)
        rows.append(row(name, run(w), group="instrument"))
    s = pd.DataFrame(rows)
    s.to_csv(cm.OUT / "p3_other.csv", index=False)
    print(s.drop(columns=[c for c in s.columns if c.endswith("_cagr") and c != "ALL_cagr"]).round(3)
          .to_string(index=False))
    off = s[s.group == "offset"]
    out["offset_sharpe_range"] = [float(off["ALL_sharpe"].min()), float(off["ALL_sharpe"].max())]
    out["offset_cagr_range"] = [float(off["ALL_cagr"].min()), float(off["ALL_cagr"].max())]
    out["offset_sharpe_median"] = float(off["ALL_sharpe"].median())
    (cm.OUT / "p3_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "grid_describe"}, indent=1))


if __name__ == "__main__":
    main()
