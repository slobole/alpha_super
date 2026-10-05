"""SPEC 9-10 (descriptive, never selects): GROWTH vs AGGRESSIVE for the USD 0.5M, iso-risk split, redundant-defence
test, the s-frontier and its envelope, the lever table from CH, and the drawdown governor.

Usage: python part_b.py   (after run_select.py main)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bb_lib as bb
from bb_lib import TBILL, ga, lib

OUT = bb.STUDY / "part_b"
S_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
FRONT_SLEEVES = [bb.G3, bb.GROWTH_V, bb.AGGR_V, "TAA3x-1N + NDX-VXN 50:50", "TAA3x-1N + NDX-VXN 60:40",
                 "TAA3x-1N + NDX-VXN 70:30", "TAA3x + NDX-VXN 70:30", "TAA3x-1N only", "TAA3x only", "TAA2x-1N + NDX-VXN 70:30"]
FRONT_CORES = ["D0", "D2", "D8", "D3", "D5"]


def fee_nav_params(r: np.ndarray, year_end: np.ndarray, mgmt: float, perf: float) -> np.ndarray:
    """Net NAV under mgmt/perf with an HWM, daily accrual, yearly payment (as ga.fee_nav, other rates)."""
    pre, hwm = np.ones(r.shape[1]), np.ones(r.shape[1])
    out = np.empty_like(r)
    keep = 1.0 - mgmt / 252.0
    for t in range(r.shape[0]):
        pre *= (1.0 + r[t]) * keep
        acc = perf * np.maximum(pre - hwm, 0.0)
        out[t] = pre - acc
        if year_end[t]:
            paid = acc > 0
            pre = pre - acc
            hwm = np.where(paid, pre, hwm)
    return out


def tails(b: dict, key: str = "gross") -> dict:
    d = b[f"{key}_dd"]
    return {"ddar10": np.percentile(d, 10, axis=0), **{f"p{int(round(-B * 1000))}": (d < B).mean(axis=0) for B in bb.B_GRID}}


def seeds_tails(RA, RD, iA, jD, s, n, seeds=bb.SEEDS) -> dict:
    acc = {}
    for k, seed in enumerate(seeds):
        b = bb.boot_accounts(RA, RD, iA, jD, s, bb.boot_index(n, seed), dtype=np.float64 if k == 0 else np.float32)
        for mode in ("gross", "net"):
            for kk, v in tails(b, mode).items():
                acc.setdefault(f"{mode}_{kk}", []).append(v)
        acc.setdefault("gross_cagr_paths0", [b["gross_cagr"]]) if k == 0 else None
        acc.setdefault("net_cagr_paths0", [b["net_cagr"]]) if k == 0 else None
    return acc


def governor(rA: np.ndarray, rD: np.ndarray, index: pd.DatetimeIndex, s: float, X: float) -> np.ndarray:
    """Month-end drawdown brake: account DD below -X -> risky share s/2 until DD above -X/2; annual reset to target."""
    n = len(rA)
    month_end = np.r_[index.month[1:] != index.month[:-1], True]
    year_end = np.r_[index.year[1:] != index.year[:-1], True]
    vR, vD = s, 1.0 - s
    peak, target, out, prev = 1.0, s, np.empty(n), 1.0
    for t in range(n):
        vR *= 1.0 + rA[t]
        vD *= 1.0 + rD[t]
        v = vR + vD
        out[t] = v / prev - 1.0
        peak = max(peak, v)
        prev = v                                  # the next return is measured from this close
        new_target = target
        if month_end[t]:
            ddn = v / peak - 1.0
            if target == s and ddn < -X:
                new_target = s / 2.0
            elif target < s and ddn > -X / 2.0:
                new_target = s
        if new_target != target or year_end[t]:
            target = new_target
            moved = abs(vR - target * v)
            v_after = v - 2.0 * bb.TRANSFER_BPS * moved   # *** CRITICAL*** the cost lands in the next session's return
            vR, vD = target * v_after, (1.0 - target) * v_after
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = lib.load_inputs()
    frame, start = ga.frames(data)["main"]
    index_all = data["index"]
    sl = {s.name: s for s in bb.risky_sleeves()}
    r_names = sorted(set(FRONT_SLEEVES) | {bb.GROWTH_V, bb.AGGR_V, "TAA3x-1N + NDX-VXN 70:30", "TAA3x-1N + NDX-VXN 60:40"})
    d_names = sorted(set(FRONT_CORES) | set(bb.FUNDED) | {"D3"})
    RA = pd.DataFrame({n: lib.book_returns(frame, sl[n].book(), start) for n in r_names})
    RD = pd.DataFrame({d: lib.book_returns(frame, bb.def_book(d), start) for d in d_names})
    index = RA.index
    n = len(index)
    periods = bb.period_labels(index, "annual")
    RAv, RDv = RA.to_numpy(), RD.to_numpy()
    ri = {k: i for i, k in enumerate(r_names)}
    di = {k: i for i, k in enumerate(d_names)}

    def hist(combos):
        a = bb.account_returns(RAv[:, [ri[c[0]] for c in combos]], RDv[:, [di[c[1]] for c in combos]],
                               np.array([c[2] for c in combos]), periods)
        st = ga.window_stats(pd.DataFrame(a, index=index), index_all)
        return a, st

    # 1. GROWTH vs AGGRESSIVE at s = 1/3 with every FUNDED core and D3 (pre-specified comparison).
    cmp_combos = [(r, d, bb.S_CLIENT) for d in bb.FUNDED + ["D3"] for r in (bb.GROWTH_V, bb.AGGR_V)]
    a, st = hist(cmp_combos)
    sd = seeds_tails(RAv, RDv, np.array([ri[c[0]] for c in cmp_combos]), np.array([di[c[1]] for c in cmp_combos]),
                     np.array([c[2] for c in cmp_combos]), n)
    comparison = []
    for k, (r, d, s) in enumerate(cmp_combos):
        row = {"risky": r, "core": d, "s": s, **{f"hist_{kk}": float(v[k]) for kk, v in st.items()},
               **{kk: float(np.mean([x[k] for x in v])) for kk, v in sd.items() if not kk.endswith("paths0")}}
        comparison.append(row)
    for d in bb.FUNDED + ["D3"]:
        kg = cmp_combos.index((bb.GROWTH_V, d, bb.S_CLIENT))
        ka = cmp_combos.index((bb.AGGR_V, d, bb.S_CLIENT))
        for mode in ("gross", "net"):
            p = sd[f"{mode}_cagr_paths0"][0]
            comparison[ka][f"aggr_beats_growth_{mode}"] = float(np.mean(p[:, ka] > p[:, kg]))
    pd.DataFrame(comparison).to_csv(OUT / "growth_vs_aggressive.csv", index=False, float_format="%.6g")

    # 2. Iso-risk: the s at which each sleeve with D0 matches CH's 10-seed gross DDaR10 (bisection on the frozen seed,
    #    then 10 seeds at the solution), and its CAGR there. Includes the redundant-defence cores (no def2 slice).
    ch_k = cmp_combos.index((bb.AGGR_V, "D0", bb.S_CLIENT))
    ch_ddar = comparison[ch_k]["gross_ddar10"]
    idx0 = bb.boot_index(n, bb.SEEDS[0])
    iso = []
    for r in (bb.GROWTH_V, bb.AGGR_V, "TAA3x-1N + NDX-VXN 70:30", "TAA3x-1N + NDX-VXN 60:40", bb.G3):
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            b = bb.boot_accounts(RAv, RDv, np.array([ri[r]]), np.array([di["D0"]]), mid, idx0)
            if np.percentile(b["gross_dd"][:, 0], 10) >= ch_ddar:
                lo = mid
            else:
                hi = mid
        s_star = round(lo, 4)
        a_iso, st_iso = hist([(r, "D0", s_star)])
        sdi = seeds_tails(RAv, RDv, np.array([ri[r]]), np.array([di["D0"]]), s_star, n)
        iso.append({"risky": r, "s_star": s_star, "gross_cagr": float(st_iso["gross_cagr"][0]), "net_cagr": float(st_iso["net_cagr"][0]),
                    "gross_dd": float(st_iso["gross_dd"][0]), "gross_ddar10_10seed": float(np.mean(sdi["gross_ddar10"])),
                    "ch_ddar10": float(ch_ddar)})
    pd.DataFrame(iso).to_csv(OUT / "iso_risk.csv", index=False, float_format="%.6g")

    # 3. Frontier over s (frozen seed for all; 10 seeds on the envelope and neighbours).
    fr = [(r, d, float(s)) for r in FRONT_SLEEVES for d in FRONT_CORES for s in S_GRID]
    a_fr, st_fr = hist(fr)
    b = bb.boot_accounts(RAv, RDv, np.array([ri[c[0]] for c in fr]), np.array([di[c[1]] for c in fr]),
                         np.array([c[2] for c in fr]), idx0)
    tg, tn = tails(b, "gross"), tails(b, "net")
    F = pd.DataFrame({"risky": [c[0] for c in fr], "core": [c[1] for c in fr], "s": [c[2] for c in fr],
                      "core_line": [bb.DEF_CORES[c[1]][3] for c in fr],
                      "taa_share": [c[2] * sum(sl[c[0]].weights.get(p, 0) for p in bb.TAA_PODS) for c in fr],
                      **{k: v for k, v in st_fr.items()}, **{f"g_{k}": v for k, v in tg.items()}, **{f"n_{k}": v for k, v in tn.items()}})
    budgets = np.round(np.arange(-0.04, -0.301, -0.01), 2)
    env_rows, env_idx = [], set()
    for line in ("FUNDED", "TARGET"):
        pool = F[F["core_line"].eq(line) | F["core"].eq("D8")] if line == "TARGET" else F[F["core_line"].eq("FUNDED")]
        for cap in (True, False):
            pp = pool[pool["taa_share"] <= bb.TAA_CAP + 1e-12] if cap else pool
            for B in budgets:
                ok = pp[pp["g_ddar10"] >= B]
                if not len(ok):
                    continue
                k = ok["gross_cagr"].idxmax()
                env_idx.add(k)
                env_rows.append({"line": line, "taa_cap": cap, "ddar10_budget": float(B), "book": k, "risky": F.at[k, "risky"],
                                 "core": F.at[k, "core"], "s": F.at[k, "s"], "gross_cagr": F.at[k, "gross_cagr"],
                                 "net_cagr": F.at[k, "net_cagr"], "gross_dd": F.at[k, "gross_dd"], "g_ddar10": F.at[k, "g_ddar10"]})
    E = pd.DataFrame(env_rows)
    # 10 seeds on envelope books.
    ek = sorted(env_idx)
    sde = seeds_tails(RAv, RDv, np.array([ri[F.at[k, "risky"]] for k in ek]), np.array([di[F.at[k, "core"]] for k in ek]),
                      F.loc[ek, "s"].to_numpy(), n)
    F.loc[ek, "g_ddar10_10seed"] = np.mean(np.vstack(sde["gross_ddar10"]), axis=0)
    F.loc[ek, "g_p150_10seed"] = np.mean(np.vstack(sde["gross_p150"]), axis=0)
    E["g_ddar10_10seed"] = F.loc[E["book"], "g_ddar10_10seed"].to_numpy()
    F.to_csv(OUT / "frontier.csv", float_format="%.6g")
    E.to_csv(OUT / "envelope.csv", index=False, float_format="%.6g")

    # 4. Lever table from CH (each lever alone).
    levers = [("CH (plan)", bb.AGGR_V, "D0", bb.S_CLIENT),
              ("split 0.40", bb.AGGR_V, "D0", 0.40), ("split 0.50", bb.AGGR_V, "D0", 0.50),
              ("sleeve GROWTH", bb.GROWTH_V, "D0", bb.S_CLIENT),
              ("sleeve AGGR core, no def2", "TAA3x-1N + NDX-VXN 70:30", "D0", bb.S_CLIENT),
              ("sleeve TAA3x-1N only (breaks the 25% TAA cap)", "TAA3x-1N only", "D0", bb.S_CLIENT),
              ("ballast D2 (IV)", bb.AGGR_V, "D2", bb.S_CLIENT), ("ballast D3 (target, DV2-IND)", bb.AGGR_V, "D3", bb.S_CLIENT),
              ("ballast T-bills", bb.AGGR_V, "D8", bb.S_CLIENT)]
    lv_combos = [(r, d, s) for _, r, d, s in levers]
    a_lv, st_lv = hist(lv_combos)
    sdl = seeds_tails(RAv, RDv, np.array([ri[c[0]] for c in lv_combos]), np.array([di[c[1]] for c in lv_combos]),
                      np.array([c[2] for c in lv_combos]), n)
    rows = []
    for k, (label, r, d, s) in enumerate(levers):
        pg = sdl["gross_cagr_paths0"][0]
        rows.append({"lever": label, "risky": r, "core": d, "s": s, "gross_cagr": float(st_lv["gross_cagr"][k]),
                     "net_cagr": float(st_lv["net_cagr"][k]), "gross_dd": float(st_lv["gross_dd"][k]),
                     "gross_sharpe": float(st_lv["gross_sharpe"][k]),
                     "ddar10_10seed": float(np.mean([x[k] for x in sdl["gross_ddar10"]])),
                     "p150_10seed": float(np.mean([x[k] for x in sdl["gross_p150"]])),
                     "beats_ch": float(np.mean(pg[:, k] > pg[:, 0]))})
    # Fee lever and rebalancing lever on CH itself.
    ch_a = a_lv[:, 0]
    ye = ga.year_end_flags(index)
    days = (index[-1] - lib.base_date(index_all, pd.Series(ch_a, index=index))).days
    for label, mg, pf in (("fee none (gross)", 0.0, 0.0), ("fee 2/20 total account (F1)", 0.02, 0.20), ("fee 1/10 total (F3)", 0.01, 0.10)):
        nav = fee_nav_params(ch_a[:, None], ye, mg, pf)[:, 0]
        rows.append({"lever": label, "risky": bb.AGGR_V, "core": "D0", "s": bb.S_CLIENT, "net_cagr": float(nav[-1] ** (365.25 / days) - 1)})
    rA_net = ga.net_return_series(RA[bb.AGGR_V]).to_numpy()
    rD_net = ga.net_return_series(RD["D0"]).to_numpy()
    f2 = bb.account_returns(rA_net, rD_net, bb.S_CLIENT, periods)[:, 0]
    rows.append({"lever": "fee 2/20 per sleeve (F2)", "risky": bb.AGGR_V, "core": "D0", "s": bb.S_CLIENT,
                 "net_cagr": float(np.prod(1 + f2) ** (365.25 / days) - 1)})
    for label, pol in (("rebalance quarterly (P3)", "quarterly"), ("no rebalance, 18 years (P1, descriptive)", "drift")):
        acc = bb.account_returns(RA[bb.AGGR_V].to_numpy(), RD["D0"].to_numpy(), bb.S_CLIENT, bb.period_labels(index, pol))[:, 0]
        st = ga.window_stats(pd.DataFrame(acc, index=index), index_all)
        rows.append({"lever": label, "risky": bb.AGGR_V, "core": "D0", "s": bb.S_CLIENT, "gross_cagr": float(st["gross_cagr"][0]),
                     "net_cagr": float(st["net_cagr"][0]), "gross_dd": float(st["gross_dd"][0])})
    pd.DataFrame(rows).to_csv(OUT / "levers.csv", index=False, float_format="%.6g")

    # 5. Drawdown governor (descriptive). Unit check: with an unreachable trigger it equals the annual policy.
    chk = governor(RA[bb.AGGR_V].to_numpy(), RD["D0"].to_numpy(), index, bb.S_CLIENT, 10.0)
    ref = bb.account_returns(RA[bb.AGGR_V].to_numpy(), RD["D0"].to_numpy(), bb.S_CLIENT, periods)[:, 0]
    assert np.abs(chk - ref).max() < 1e-12, np.abs(chk - ref).max()
    gov = []
    for r in (bb.AGGR_V, bb.GROWTH_V):
        for X in (0.06, 0.08):
            g = governor(RA[r].to_numpy(), RD["D0"].to_numpy(), index, bb.S_CLIENT, X)
            st = ga.window_stats(pd.DataFrame(g, index=index), index_all)
            gov.append({"risky": r, "core": "D0", "X": X, "gross_cagr": float(st["gross_cagr"][0]), "gross_dd": float(st["gross_dd"][0]),
                        "net_cagr": float(st["net_cagr"][0])})
    pd.DataFrame(gov).to_csv(OUT / "governor.csv", index=False, float_format="%.6g")
    bb.ledger("part_b_finished")
    print(pd.DataFrame(comparison)[["risky", "core", "hist_gross_cagr", "hist_net_cagr", "hist_gross_dd", "gross_ddar10", "gross_p150"]].round(4).to_string())
    print(pd.DataFrame(iso).round(4).to_string())
    print(pd.DataFrame(rows).round(4).to_string())
    print(pd.DataFrame(gov).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
