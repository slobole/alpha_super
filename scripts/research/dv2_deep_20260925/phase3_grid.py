"""Phase 1 luck band + Phase 3 pre-declared robustness map (exactly the configurations in SPEC_FROZEN.md)."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch  # noqa: E402
from replica import Rule  # noqa: E402

WIRED = Rule()
FLOOR = Rule(floor=True)
SOURCE = Rule(mom_thr=0.0)


def grid() -> dict[str, dict[str, Rule]]:
    g: dict[str, dict[str, Rule]] = {}
    g["baselines"] = {"wired": WIRED, "floor": FLOOR, "source": SOURCE, "source_floor": SOURCE.with_(floor=True)}
    g["luck"] = {f"random_{s:03d}": FLOOR.with_(rank="random", seed=s) for s in range(200)}
    ax = {}
    for k in (1, 2, 3, 4, 5, 10):
        ax[f"k{k}"] = FLOOR.with_(k=k)
    for w in (63, 126, 252):
        ax[f"w{w}"] = FLOOR.with_(dv_window=w)
    for th in (5, 10, 15, 20):
        ax[f"thr{th}"] = FLOOR.with_(dv_thr=th)
    for k in (1, 2, 3, 4, 5, 10):
        for w in (63, 126, 252):
            ax[f"k{k}_w{w}"] = FLOOR.with_(k=k, dv_window=w)
        for th in (5, 10, 15, 20):
            ax[f"k{k}_thr{th}"] = FLOOR.with_(k=k, dv_thr=th)
    ax["mom_none"] = FLOOR.with_(mom_lb=None)
    for lb in (63, 126, 252):
        for th in (0.0, 0.05, 0.10):
            ax[f"mom{lb}_{int(th * 100)}"] = FLOOR.with_(mom_lb=lb, mom_thr=th)
    for n in (None, 100, 150, 200, 250):
        ax[f"sma{n}"] = FLOOR.with_(sma_n=n)
    for rk in ("natr5", "natr30", "adv", "dv"):
        ax[f"rank_{rk}"] = FLOOR.with_(rank=rk)
    for s in (5, 15, 20):
        ax[f"slots{s}"] = FLOOR.with_(slots=s)
    g["axes"] = ax
    g["robust"] = {"E_avg": FLOOR.with_(ensemble="avg"), "E_vote": FLOOR.with_(ensemble="vote"),
                   "T_vote": FLOOR.with_(mom_vote=True), "E_avg_T_vote": FLOOR.with_(ensemble="avg", mom_vote=True),
                   "E_vote_T_vote": FLOOR.with_(ensemble="vote", mom_vote=True)}
    ex = {}
    for x in ("X1", "X2", "X3", "X4", "X5", "X6"):
        ex[f"floor_{x}"] = FLOOR.with_(exit=x)
        ex[f"wired_{x}"] = WIRED.with_(exit=x)
    g["exits"] = ex
    return g


def main():
    g = grid()
    out = []
    for tag, workers in (("baselines", 4), ("luck", 8), ("axes", 6), ("robust", 3), ("exits", 6)):
        df = batch.run_many("sp500", g[tag], workers=workers, tag=tag)
        df["group"] = tag
        out.append(df)
        print(tag, len(df), flush=True)
    res = pd.concat(out)
    res.to_csv(batch.OUT / "phase3_grid_summary.csv")
    # stressed costs (+5 bps per side) for everything except the luck band
    stress = {f"{n}__stress": r.with_(slippage=r.slippage + 0.0005) for tag in ("baselines", "axes", "robust", "exits") for n, r in g[tag].items()}
    sdf = batch.run_many("sp500", stress, workers=6, tag="stress")
    sdf.to_csv(batch.OUT / "phase3_grid_stress_summary.csv")
    print("done", len(res), len(sdf))


if __name__ == "__main__":
    main()
