"""Contracts of the defensive shelf v2 check code (scripts/research/defensive_v2_20260929, research only).

The A2 conclusions rely on: the block-sum CSCV Sharpe equals the Sharpe of the concatenated rows, the omega rank
convention, the plateau variants keep weights summing to one with the moved pod at w +- 10 points, the selection
applies the frozen tie-break order and champion rule, and the family has the declared size.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1] / "scripts" / "research" / "defensive_v2_20260929"
sys.path.insert(0, str(STUDY.parent / "shelf_rebuild_20260929"))
sys.path.insert(0, str(STUDY))

import defensive_v2 as dv  # noqa: E402
import sharpe_checks as sc  # noqa: E402


def test_family_sizes():
    assert len(dv.family("MAIN")) == 949
    assert len(dv.family("LOW")) == 34


def test_block_sum_cscv_sharpe_matches_concatenated_rows():
    rng = np.random.default_rng(3)
    X = rng.normal(0.0004, 0.01, size=(1603, 5))
    X[:, 2] += 0.0006  # a clearly better column
    out = sc.cscv_sharpe(X, {"best": 2, "other": 0}, blocks=8)
    n = X.shape[0] - X.shape[0] % 8
    parts = np.array_split(np.arange(n), 8)
    ranks, logits = [], []
    for combo in combinations(range(8), 4):
        ins = np.concatenate([parts[i] for i in combo])
        oos = np.concatenate([parts[i] for i in range(8) if i not in combo])
        v_in = X[ins].mean(0) / X[ins].std(0, ddof=1)
        v_out = X[oos].mean(0) / X[oos].std(0, ddof=1)
        best = int(np.argmax(v_in))
        w = (np.sum(v_out < v_out[best]) + 1.0) / (X.shape[1] + 1)
        logits.append(np.log(w / (1 - w)))
        ranks.append((np.sum(v_out < v_out[2]) + 1.0) / (X.shape[1] + 1))
    assert out["splits"] == 70
    assert np.isclose(out["pbo_argmax"], np.mean(np.array(logits) <= 0))
    assert np.isclose(out["picks"]["best"]["median_oos_rank"], np.median(ranks))


def test_plateau_variants_sum_to_one_and_move_ten_points():
    base = dv.DefBook("B", ("core5", "eom_flow", "disp"), "IV", {}, 0.0)
    w = {"core5": 0.4285, "eom_flow": 0.2277, "disp": 0.3438}
    books = sc.plateau_books(base, w)
    assert len(books) == 1 + 2 * 3
    for b in books:
        assert abs(sum(b.weights.values()) - 1.0) < 1e-12
        assert b.rule == "FIXED"
    moved = {b.name.split("| ")[1]: b.weights for b in books[1:]}
    assert np.isclose(moved["EOM +10"]["eom_flow"], 0.3277)
    assert np.isclose(moved["CORE5 -10"]["core5"], 0.3285)
    # the other pods keep their ratio
    r = moved["EOM +10"]
    assert np.isclose(r["core5"] / r["disp"], w["core5"] / w["disp"])


def test_select_tie_break_and_champion_rule():
    names = [sc.CHAMPION, "A", "B", "C"]
    t = pd.DataFrame({"obj": [1.0, 1.50, 1.45, 1.20], "pods": [2, 3, 2, 2], "not_live_share": [0, 0.2, 0.2, 0],
                      "p_breach10": [0.13, 0.01, 0.02, 0.05], "trade_days_per_year": [29, 120, 90, 30]}, index=names)
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, (2000, 1))
    boot = np.hstack([base + 1.0, base + 1.5, base + 1.5 + rng.normal(0, 0.05, (2000, 1)), base + 1.2])
    res = sc.select(t, "obj", names, boot, ["A", "B", "C"])
    assert res["top"] == "A"
    assert "B" in res["band"] and "C" not in res["band"]  # A beats C on every path -> outside the band
    assert res["pick"] == "B"  # fewer pods wins the tie-break
    assert res["champion_holds"] is False and res["recommendation"] == "B"
    t.loc["B", "p_breach10"] = 0.20  # breach worse than the champion's by more than 2 points -> champion holds
    assert sc.select(t, "obj", names, boot, ["A", "B", "C"])["recommendation"] == sc.CHAMPION
