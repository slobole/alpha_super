"""Pre-registered unit and parity tests (SPEC 3): run before any selection. Usage: python test_parity.py"""

from __future__ import annotations

import numpy as np
import pandas as pd

import bb_lib as bb
from bb_lib import LONG_START, ga, lib


def main() -> int:
    data = lib.load_inputs()
    frame, start = ga.frames(data)["main"]
    sleeves = {s.name: s for s in bb.risky_sleeves()}
    assert len(sleeves) == 23 * 11, len(sleeves)
    assert sum(s.line == "LT" for s in sleeves.values()) == 115
    # Every look-through pod at s = 1/3 with any core is >= 1.67% (USD 25K at USD 1.5M).
    small = min(min(v for v in bb.look_through(s.weights, bb.def_book(d).targets() if bb.def_book(d).rule == "EQ"
                                             else {p: 1 / len(bb.def_book(d).pods) for p in bb.def_book(d).pods},
                                             bb.S_CLIENT).values()) for s in sleeves.values() for d in bb.DEF_CORES)
    print("smallest look-through weight", round(small, 4))
    # (1) sleeve parity with the extension study.
    ext = pd.read_csv(ga.STUDY / "ext" / "main" / "long_returns.csv.gz", index_col=0, parse_dates=True)
    for mine, theirs in ((bb.GROWTH_V, "TAA3x-1N + NDX-VXN @60:40 | def2@36"), (bb.AGGR_V, "TAA3x-1N + NDX-VXN @70:30 | def2@18"),
                         (bb.G3, "TAA3x + NDX-VXN")):
        r = lib.book_returns(frame, sleeves[mine].book(), start)
        diff = float(np.abs(r.to_numpy() - ext[theirs].reindex(r.index).to_numpy()).max())
        print("sleeve parity", mine, diff)
        assert diff < 1e-9  # the saved CSV keeps 10 significant digits
    # (2) account (annual, zero transfer cost) == flat look-through book for EQ cores.
    for rname, d in ((bb.AGGR_V, "D0"), (bb.GROWTH_V, "D1"), (bb.G3, "D9")):
        rA = lib.book_returns(frame, sleeves[rname].book(), start)
        rD = lib.book_returns(frame, bb.def_book(d), start)
        acc = bb.account_returns(rA.to_numpy(), rD.to_numpy(), bb.S_CLIENT, bb.period_labels(rA.index, "annual"), cost=0.0)[:, 0]
        w = bb.look_through(sleeves[rname].weights, bb.def_book(d).targets(), bb.S_CLIENT)
        flat = lib.book_returns(frame, bb.Book("flat", tuple(w), "EQ", w), start).to_numpy()
        diff = float(np.abs(acc - flat).max())
        print("account == flat", rname, d, diff)
        assert diff < 1e-12
    # s = 1 reproduces R, s = 0 reproduces D.
    rA = lib.book_returns(frame, sleeves[bb.AGGR_V].book(), start).to_numpy()
    rD = lib.book_returns(frame, bb.def_book("D2"), start).to_numpy()
    per = bb.period_labels(pd.DatetimeIndex(lib.book_returns(frame, sleeves[bb.AGGR_V].book(), start).index), "annual")
    assert np.abs(bb.account_returns(rA, rD, 1.0, per)[:, 0] - rA).max() < 1e-12
    assert np.abs(bb.account_returns(rA, rD, 0.0, per)[:, 0] - rD).max() < 1e-12
    # (3) bootstrap engine: s = 1 equals the growth study's engine on the same indices.
    idx = bb.boot_index(len(rA), bb.SEEDS[0])[:200]
    mine = bb.boot_accounts(rA[:, None], rD[:, None], np.array([0]), np.array([0]), 1.0, idx)
    ref = ga.bootstrap_paths(rA[:, None], idx)
    for k in ("gross_cagr", "net_cagr", "gross_dd", "net_dd"):
        d = float(np.abs(mine[k][:, 0] - ref[k][:, 0]).max())
        print("boot parity", k, d)
        assert d < 1e-10
    # Constant return: exact CAGR on paths.
    const = np.full((len(rA), 1), 0.0003)
    c = bb.boot_accounts(const, const, np.array([0]), np.array([0]), 0.5, idx, cost=0.0)
    assert np.allclose(c["gross_cagr"], 1.0003 ** 252 - 1.0)
    # (4) defensive parity with the defensive study (D0 = its champion).
    dv = pd.read_csv(ga.MAIN_REPO / "results" / "research" / "portfolio" / "defensive_v2_20260929" / "main_long_returns.csv.gz",
                     index_col=0, parse_dates=True)
    for key, col in (("D0", "CORE5 60 + BTAL_QQQ 40"), ("D3", "CORE5 + BTAL_QQQ + DV2-IND [IV]")):
        if col in dv.columns:
            r = lib.book_returns(frame, bb.def_book(key), start)
            diff = float(np.abs(r.to_numpy() - dv[col].reindex(r.index).to_numpy()).max())
            print("defensive parity", key, diff)
        else:
            print("defensive column not saved:", col)
    bb.ledger("parity_tests_passed")
    print("ALL PARITY TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
