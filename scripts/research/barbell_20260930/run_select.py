"""SPEC 5-8: client-account selection at s = 1/3 for every line, mode and rung, on one frame.

Usage: python run_select.py <frame> [--lines LT-FUNDED,...]
Frames: main, s1_house_cash, s2_proxy_unscaled, s3_plus_5bps, s5_hpi_live_gap, s6_exact, s7_block126, s8_block21.
Outputs under <study>/<frame>/: books.csv (every client book), selection.json, seeds.json, frozen_paths.npz (contenders).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import bb_lib as bb
from bb_lib import TBILL, ga, lib

BLOCK = {"s7_block126": 126.0, "s8_block21": 21.0}
ALL_LINES = ("LT-FUNDED", "MAIN-FUNDED", "TARGET", "NOW", "LT-FUNDED-UNCAPPED")
PRIMARY_ONLY = ("LT-FUNDED", "LT-FUNDED-UNCAPPED")
RUNGS = ["RM"] + [f"B{int(round(-b * 1000))}" for b in bb.B_GRID]      # B100 = -10%, B125 = -12.5%, ...
B_OF = {f"B{int(round(-b * 1000))}": b for b in bb.B_GRID}


def line_members(sleeves: list, lines) -> dict[str, list[tuple[str, str]]]:
    lt = [s for s in sleeves if s.line == "LT"]
    now = [s for s in lt if set(s.weights) <= bb.WIRED_NOW]
    out = {"LT-FUNDED": [(s.name, d) for s in lt for d in bb.FUNDED],
           "LT-FUNDED-UNCAPPED": [(s.name, d) for s in lt for d in bb.FUNDED],
           "MAIN-FUNDED": [(s.name, d) for s in sleeves for d in bb.FUNDED],
           "TARGET": [(s.name, d) for s in lt for d in bb.TARGET],
           "NOW": [(s.name, d) for s in now for d in bb.NOW_CORES]}
    return {k: v for k, v in out.items() if k in lines}


def hist_stats(acc: np.ndarray, index: pd.DatetimeIndex, index_all: pd.DatetimeIndex, tb: pd.Series) -> dict:
    df = pd.DataFrame(acc, index=index)
    st = ga.window_stats(df, index_all)
    out = {f"{k}": v for k, v in st.items()}
    ex = df.loc[bb.EXACT_START:]
    se = ga.window_stats(ex, index_all)
    out["gross_dd_exact"], out["net_dd_exact"] = se["gross_dd"], se["net_dd"]
    out["gross_cagr_exact"], out["net_cagr_exact"] = se["gross_cagr"], se["net_cagr"]
    for block in ("A", "B", "C", "RECENT"):
        lo, hi = ga.BLOCK_DICT[block]
        sub = df.loc[max(lo, df.index[0]):hi]
        if len(sub) < 60:
            out[f"gross_xs_{block}"] = out[f"net_xs_{block}"] = np.full(df.shape[1], np.nan)
            continue
        s = ga.window_stats(sub, index_all)
        tbc = lib.cagr(tb.loc[sub.index], lib.base_date(index_all, sub.iloc[:, 0]))
        out[f"gross_xs_{block}"], out[f"net_xs_{block}"] = s["gross_cagr"] - tbc, s["net_cagr"] - tbc
        if block == "C":
            out["gross_cagr_C"], out["net_cagr_C"] = s["gross_cagr"], s["net_cagr"]
        if block == "RECENT":
            out["gross_cagr_RECENT"], out["net_cagr_RECENT"] = s["gross_cagr"], s["net_cagr"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("--lines", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    fname = args.frame
    smoke = bool(os.environ.get("BB_SMOKE"))
    if smoke:  # quick end-to-end check of the code path; never a result
        ga.BOOT_REPS = 40
        bb.SEEDS = bb.SEEDS[:2]
    lines = tuple(args.lines.split(",")) if args.lines else (ALL_LINES if fname in ("main", "s5_hpi_live_gap") else PRIMARY_ONLY)
    if fname == "s5_hpi_live_gap" and not args.lines:
        lines = ("MAIN-FUNDED",)
    block = BLOCK.get(fname, 63.0)
    t0 = time.time()
    data = lib.load_inputs()
    frame, start = ga.frames(data)[fname]
    index_all = data["index"]
    sleeves = bb.risky_sleeves()
    sl = {s.name: s for s in sleeves}
    members = line_members(sleeves, lines)
    combos = sorted({c for v in members.values() for c in v} | {bb.CH, bb.CH_G})
    r_names = sorted({c[0] for c in combos})
    d_names = sorted({c[1] for c in combos})
    RA = pd.DataFrame({n: lib.book_returns(frame, sl[n].book(), start) for n in r_names})
    RD = pd.DataFrame({d: lib.book_returns(frame, bb.def_book(d), start) for d in d_names})
    index = RA.index
    iA = np.array([r_names.index(c[0]) for c in combos])
    jD = np.array([d_names.index(c[1]) for c in combos])
    periods = bb.period_labels(index, "annual")
    acc = bb.account_returns(RA.to_numpy()[:, iA], RD.to_numpy()[:, jD], bb.S_CLIENT, periods)
    tb = frame[TBILL].reindex(index)
    H = hist_stats(acc, index, index_all, tb)
    names = [bb.client_name(r, d) for r, d in combos]
    T = pd.DataFrame(H, index=names)
    T["risky"], T["core"] = [c[0] for c in combos], [c[1] for c in combos]
    lt_w = {}
    dw_cache = {d: bb.def_avg_weights(frame, d, start) for d in d_names}
    for (r, d), n in zip(combos, names):
        dw = dw_cache[d]
        w = bb.look_through(sl[r].weights, dw, bb.S_CLIENT)
        lt_w[n] = w
        T.at[n, "taa_share"] = sum(w.get(p, 0.0) for p in bb.TAA_PODS)
        T.at[n, "taa_df_share"] = sum(w.get(p, 0.0) for p in bb.TAA_DF_PODS)
        T.at[n, "pods"] = sum(1 for p, v in w.items() if v > 1e-12 and p != TBILL)
        T.at[n, "weights"] = json.dumps({k: round(v, 4) for k, v in w.items()})
        T.at[n, "risky_line"] = sl[r].line
        T.at[n, "satellite"] = sl[r].tags["satellite"]
    t1 = time.time()
    # Slot test (G3), gross and net, LONG and RECENT: replace one pod with BIL wherever it appears.
    lo, hi = ga.BLOCK_DICT["RECENT"]
    pods = sorted({p for w in lt_w.values() for p, v in w.items() if v > 1e-12 and p != TBILL})
    fail_long = {m: {n: [] for n in names} for m in ("gross", "net")}
    fail_recent = {m: {n: [] for n in names} for m in ("gross", "net")}
    for p in pods:
        fp = lib.replace_pod(frame, p)
        ra_p = RA.copy()
        for n in r_names:
            if p in sl[n].weights:
                ra_p[n] = lib.book_returns(fp, sl[n].book(), start)
        rd_p = RD.copy()
        for d in d_names:
            if p in bb.def_book(d).pods:
                rd_p[d] = lib.book_returns(fp, bb.def_book(d), start, weight_source=frame)
        cols = [k for k, n in enumerate(names) if lt_w[n].get(p, 0.0) > 1e-12]
        if not cols:
            continue
        a_p = bb.account_returns(ra_p.to_numpy()[:, iA[cols]], rd_p.to_numpy()[:, jD[cols]], bb.S_CLIENT, periods)
        dfp = pd.DataFrame(a_p, index=index)
        sl_long = ga.window_stats(dfp, index_all)
        sl_rec = ga.window_stats(dfp.loc[lo:hi], index_all)
        for m in ("gross", "net"):
            for kk, k in enumerate(cols):
                n = names[k]
                if T.at[n, f"{m}_cagr"] <= sl_long[f"{m}_cagr"][kk]:
                    fail_long[m][n].append(p)
                if T.at[n, f"{m}_cagr_RECENT"] <= sl_rec[f"{m}_cagr"][kk]:
                    fail_recent[m][n].append(p)
    for m in ("gross", "net"):
        T[f"slot_fail_{m}"] = [",".join(fail_long[m][n]) for n in names]
        T[f"slot_recent_pass_{m}"] = [not fail_recent[m][n] for n in names]
        T[f"g2_{m}"] = (T[[f"{m}_xs_B", f"{m}_xs_C", f"{m}_xs_RECENT"]] > 0).all(axis=1)
        T[f"g3_{m}"] = T[f"slot_fail_{m}"] == ""
    T["g4"] = T["taa_share"] <= bb.TAA_CAP + 1e-12
    t2 = time.time()
    # Frozen-seed bootstrap on every combo.
    RAv, RDv = RA.to_numpy(), RD.to_numpy()
    idx0 = bb.boot_index(len(index), bb.SEEDS[0], block)
    bp = bb.boot_accounts(RAv, RDv, iA, jD, bb.S_CLIENT, idx0)
    print('frozen seed done', len(combos), 'books', round(time.time() - t0), flush=True)
    t3 = time.time()

    def tail_stats(b: dict) -> dict:
        out = {}
        for m in ("gross", "net"):
            out[f"ddar10_{m}"] = np.percentile(b[f"{m}_dd"], 10, axis=0)
            for rung, B in B_OF.items():
                out[f"p_{rung}_{m}"] = (b[f"{m}_dd"] < B).mean(axis=0)
        out["p_fy10"] = (b["fy_dd"] < -0.10).mean(axis=0)
        return out

    frozen = tail_stats(bp)
    for k, v in frozen.items():
        T[f"{k}_s0"] = v
    ch_n, chg_n = bb.client_name(*bb.CH), bb.client_name(*bb.CH_G)
    ch_i = names.index(ch_n)
    for m in ("gross", "net"):
        T[f"beats_ch_{m}_s0"] = (bp[f"{m}_cagr"] > bp[f"{m}_cagr"][:, [ch_i]]).mean(axis=0)
    # Contenders: pass the deterministic gates (G2-G4 in either mode, uncapped line ignores G4) and are within 3 pp of a
    # risk threshold on the frozen seed (SPEC 7).
    det = (T["g2_gross"] & T["g3_gross"]) | (T["g2_net"] & T["g3_net"])
    near = pd.Series(False, index=names)
    for m in ("gross", "net"):
        near |= (T[f"ddar10_{m}_s0"] >= T.at[ch_n, f"ddar10_{m}_s0"] - 0.03)
        for rung in B_OF:
            near |= T[f"p_{rung}_{m}_s0"] <= bb.MAX_BREACH_ABS + 0.03
    cont = list(T.index[det & near])
    for extra in (ch_n, chg_n):
        if extra not in cont:
            cont.append(extra)
    ci = np.array([names.index(n) for n in cont])
    seed_vals = {k: [frozen[k][ci]] for k in frozen}
    beats = {m: [T.loc[cont, f"beats_ch_{m}_s0"].to_numpy()] for m in ("gross", "net")}
    ch_pos = cont.index(ch_n)
    for seed in bb.SEEDS[1:]:
        idx = bb.boot_index(len(index), seed, block)
        b = bb.boot_accounts(RAv, RDv, iA[ci], jD[ci], bb.S_CLIENT, idx, dtype=np.float32)  # SPEC amendment P1
        print('seed done', seed, round(time.time() - t0), flush=True)
        ts = tail_stats(b)
        for k in ts:
            seed_vals[k].append(ts[k])
        for m in ("gross", "net"):
            beats[m].append((b[f"{m}_cagr"] > b[f"{m}_cagr"][:, [ch_pos]]).mean(axis=0))
    t4 = time.time()
    for k, v in seed_vals.items():
        arr = np.vstack(v)
        T[f"{k}"] = T[f"{k}_s0"]
        T.loc[cont, k] = arr.mean(axis=0)
        T.loc[cont, f"{k}_max"] = arr.max(axis=0)
        T.loc[cont, f"{k}_min"] = arr.min(axis=0)
    for m in ("gross", "net"):
        arr = np.vstack(beats[m])
        T.loc[cont, f"beats_ch_{m}_min"] = arr.min(axis=0)
        T.loc[cont, f"beats_ch_{m}_max"] = arr.max(axis=0)
    T["contender"] = T.index.isin(cont)

    # Selection per line, mode, rung (SPEC 8).
    sel = {"frame": fname, "block": block, "lines": list(lines), "contenders": len(cont), "books": len(names),
           "seconds": [round(x) for x in (t1 - t0, t2 - t1, t3 - t2, t4 - t3)], "results": {}}
    ch = T.loc[ch_n]
    for line, mem in members.items():
        mem_names = [bb.client_name(r, d) for r, d in mem]
        for mode in ("GROSS", "NET"):
            cur = mode.lower()
            curs = ("gross",) if mode == "GROSS" else ("gross", "net")
            pc = bp[f"{cur}_cagr"]
            for rung in RUNGS:
                F = T.loc[mem_names]
                ok = F[f"g2_{cur}"] & F[f"g3_{cur}"]
                if line != "LT-FUNDED-UNCAPPED":
                    ok &= F["g4"]
                for c in curs:
                    if rung == "RM":
                        ok &= (F[f"{c}_dd"] >= ch[f"{c}_dd"] - 1e-12) & (F[f"{c}_dd_exact"] >= ch[f"{c}_dd_exact"] - 1e-12)
                        ok &= F[f"ddar10_{c}"] >= ch[f"ddar10_{c}"] - 1e-12
                    else:
                        B = B_OF[rung]
                        ok &= (F[f"{c}_dd"] >= B + 0.03) & (F[f"p_{rung}_{c}"] <= bb.MAX_BREACH_ABS)
                P = F[ok]
                e = {"members": len(mem_names), "passers": int(len(P))}
                if len(P):
                    top = P[f"{cur}_cagr"].idxmax()
                    tcol = names.index(top)
                    share = pd.Series({n: float(np.mean(pc[:, tcol] > pc[:, names.index(n)])) for n in P.index})
                    share[top] = 0.0
                    band = P.loc[share.index[share < bb.TIE_SHARE]].copy()
                    tail_key = f"ddar10_{cur}" if rung == "RM" else f"p_{rung}_{cur}"
                    band["_tail"] = -band[tail_key] if rung == "RM" else band[tail_key]
                    band = band.sort_values(["_tail", "taa_df_share", f"slot_recent_pass_{cur}", "pods", f"{cur}_cagr"],
                                            ascending=[True, True, False, True, False])
                    pick = band.index[0]
                    pk = T.loc[pick]
                    beat = float(np.mean(pc[:, names.index(pick)] > pc[:, ch_i]))
                    tail_ok = (pk[tail_key] >= ch[tail_key] - 1e-12) if rung == "RM" else (pk[tail_key] <= ch[tail_key] + 1e-12)
                    hist_ok = (pk[f"{cur}_cagr_exact"] > ch[f"{cur}_cagr_exact"]) and (pk[f"{cur}_cagr_C"] > ch[f"{cur}_cagr_C"])
                    replaces = bool(pick != ch_n and beat >= bb.CHAMP_SHARE and tail_ok and hist_ok)
                    e.update({"top": top, "pick": pick, "band_size": int(len(band)), "band": list(band.index[:15]),
                              "pick_obj": float(pk[f"{cur}_cagr"]), "top_obj": float(T.at[top, f"{cur}_cagr"]),
                              "ch_obj": float(ch[f"{cur}_cagr"]), "pick_beats_ch": beat,
                              "pick_beats_ch_range": [pk.get(f"beats_ch_{cur}_min"), pk.get(f"beats_ch_{cur}_max")],
                              "tail_ok": bool(tail_ok), "hist_ok": bool(hist_ok), "ch_passes": bool(ch_n in P.index),
                              "product": pick if replaces else ch_n,
                              "how": ("pick replaces CH" if replaces else ("CH is the pick" if pick == ch_n else "CH kept"))})
                sel["results"][f"{line}|{mode}|{rung}"] = e
    out = bb.STUDY / (fname + args.tag + ("_smoke" if smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "books.csv", float_format="%.6g")
    (out / "selection.json").write_text(json.dumps(sel, indent=1, default=str), encoding="utf-8")
    if fname == "main" and not args.tag:
        pd.DataFrame(acc, index=index, columns=names).to_csv(out / "account_returns.csv.gz", float_format="%.10g", compression="gzip")
        RA.to_csv(out / "risky_returns.csv.gz", float_format="%.10g", compression="gzip")
        RD.to_csv(out / "core_returns.csv.gz", float_format="%.10g", compression="gzip")
        np.savez_compressed(out / "frozen_paths.npz", names=np.array(cont),
                            **{f"{m}_{k}": bp[f"{m}_{k}"][:, ci].astype(np.float32) for m in ("gross", "net") for k in ("cagr", "dd")})
    if not smoke:
        bb.ledger("select_finished", frame=fname + args.tag, seconds=sel["seconds"], contenders=len(cont))
    print(json.dumps({k: (v.get("pick"), v.get("product"), v.get("how"), v.get("passers")) for k, v in sel["results"].items()}, indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
