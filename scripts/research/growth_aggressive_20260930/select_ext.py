"""Extension E (SPEC_EXT_FROZEN.md): gross and net modes, GROWTH / AGGRESSIVE / MAX rungs, NDX-RM, TAA:NDX ratios.

Usage: python select_ext.py [frame ...]   (default: every frame; outputs under <study>/ext/<frame>)
"""

from __future__ import annotations

from itertools import product
import json
import sys
import time

import numpy as np
import pandas as pd

import ga_lib as ga
from ga_lib import TBILL, Book, lib

EXT = ga.STUDY / "ext"
RM_CSV = ga.MAIN_REPO.parent / "pakal" / "pakal-research" / "reports" / "momentum_trend_universe_search" / "final" / "ndx_rm_daily_returns.csv"
RUNGS = {"GROWTH": (-0.17, -0.20), "AGGRESSIVE": (-0.22, -0.25), "MAX": (-0.27, -0.30)}
NDX_LEGS = {"ndx_vxn": "NDX-VXN", "ndx_atr": "NDX-ATR", "ndx_natr20": "NDX-NATR20", "ndx_rm": "NDX-RM", "vxn_rm": "NDX-VXN/RM"}
RATIOS = (0.5, 0.6, 0.7)
MODES = ("GROSS", "NET")
LINES = ("MAIN", "LOW-TOUCH")
FRAMES = ["main", "s1_house_cash", "s3_plus_5bps", "s5_hpi_live_gap", "s6_exact", "s7_block126", "s8_block21", "s9_rm_cost20"]
BLOCK = {"s7_block126": 126.0, "s8_block21": 21.0}


def family() -> list[Book]:
    books = []
    for taa, ndx, ratio in product(ga.TAA_LEGS, NDX_LEGS, RATIOS):
        core = f"{ga.TAA_LEGS[taa]} + {NDX_LEGS[ndx]}" + ("" if ratio == 0.5 else f" @{int(ratio * 100)}:{int(round((1 - ratio) * 100))}")
        twin_core = None
        if ndx in ("ndx_rm", "vxn_rm"):
            twin_core = core.replace(NDX_LEGS[ndx], "NDX-VXN")
        for sat, share in [("none", 0.0)] + [(s, sh) for s in ga.SATELLITES for sh in ga.SHARES]:
            core_w = 1.0 - share
            weights = {taa: core_w * ratio}
            ndx_share = core_w * (1.0 - ratio)
            if ndx == "vxn_rm":
                weights["ndx_vxn"], weights["ndx_rm"] = ndx_share / 2, ndx_share / 2
            else:
                weights[ndx] = ndx_share
            if sat != "none":
                pods = ga.SATELLITES[sat]
                for p in pods:
                    weights[p] = weights.get(p, 0.0) + share / len(pods)
            suffix = "" if sat == "none" else f" | {sat}@{int(round(share * 100))}"
            tags = {"taa_leg": taa, "ndx_leg": ndx, "ratio": ratio, "satellite": sat, "share": share,
                    "low_touch": sat in ga.LOW_TOUCH_SATELLITES, "rm": ndx in ("ndx_rm", "vxn_rm"),
                    "twin": (twin_core + suffix) if twin_core else ""}
            books.append(Book(core + suffix, tuple(weights), "EQ", weights, "annual", "EXT", tags))
    return books


def ext_frames(data: dict) -> dict[str, tuple[pd.DataFrame, pd.Timestamp]]:
    rm = pd.read_csv(RM_CSV, index_col=0, parse_dates=True)
    base = ga.frames(data)
    out = {}
    for name in FRAMES:
        src = "main" if name == "s9_rm_cost20" else name
        frame, start = base[src]
        frame = frame.copy()
        col = "ndx_rm_cost20" if name in ("s3_plus_5bps", "s9_rm_cost20") else "ndx_rm"
        frame["ndx_rm"] = rm[col].reindex(frame.index)
        out[name] = (frame, start)
    return out


def evaluate(frame_name: str, data: dict, books: list[Book], frame: pd.DataFrame, start: pd.Timestamp) -> dict:
    index_all = data["index"]
    tier = {a: m["tier_str"] for a, m in data["meta"].items()}
    tier["ndx_rm"] = "shadow"
    t0 = time.time()
    R = pd.DataFrame({b.name: lib.book_returns(frame, b, start) for b in books})
    tb = frame[TBILL].reindex(R.index)
    st = ga.window_stats(R, index_all)
    T = pd.DataFrame(st, index=R.columns)
    for b in books:
        for k, v in b.tags.items():
            T.at[b.name, k] = v
        tw = b.targets()
        T.at[b.name, "pods"] = len(b.pods)
        T.at[b.name, "weights"] = json.dumps({k: round(v, 4) for k, v in tw.items()})
        T.at[b.name, "shadow_share"] = sum(w for p, w in tw.items() if p != TBILL and tier[p] == "shadow")
    T["no_shadow"] = T["shadow_share"] <= 1e-12
    for block in ("A", "B", "C", "RECENT"):
        lo, hi = ga.BLOCK_DICT[block]
        sub = R.loc[max(lo, R.index[0]):hi]
        if len(sub) < 60:
            T[f"gross_xs_{block}"] = T[f"net_xs_{block}"] = np.nan
            continue
        s = ga.window_stats(sub, index_all)
        tbc = lib.cagr(tb.loc[sub.index], lib.base_date(index_all, sub.iloc[:, 0]))
        T[f"gross_xs_{block}"], T[f"net_xs_{block}"] = s["gross_cagr"] - tbc, s["net_cagr"] - tbc
        if block == "RECENT":
            T["gross_cagr_RECENT"], T["net_cagr_RECENT"] = s["gross_cagr"], s["net_cagr"]
    series, keys = {}, []
    for b in books:
        for pod in b.pods:
            if pod != TBILL:
                key = f"{b.name}##{pod}"
                series[key] = lib.book_returns(lib.replace_pod(frame, pod), b, start)
                keys.append((b.name, pod, key))
    S = pd.DataFrame(series)
    lo, hi = ga.BLOCK_DICT["RECENT"]
    s_long, s_rec = ga.window_stats(S, index_all), ga.window_stats(S.loc[lo:hi], index_all)
    pos = {k: i for i, k in enumerate(S.columns)}
    for cur in ("gross", "net"):
        lf, rf = {}, {}
        for name, pod, key in keys:
            if T.at[name, f"{cur}_cagr"] <= s_long[f"{cur}_cagr"][pos[key]]:
                lf.setdefault(name, []).append(pod)
            if T.at[name, f"{cur}_cagr_RECENT"] <= s_rec[f"{cur}_cagr"][pos[key]]:
                rf.setdefault(name, []).append(pod)
        T[f"slot_long_fail_{cur}"] = [",".join(lf.get(n, [])) for n in T.index]
        T[f"slot_recent_pass_{cur}"] = [n not in rf for n in T.index]
        T[f"r4_{cur}"] = T[f"slot_long_fail_{cur}"] == ""
    t1 = time.time()
    idx = ga.boot_index(len(R), BLOCK.get(frame_name, 63.0))
    boot = ga.bootstrap_paths(R.to_numpy(dtype=float), idx)
    t2 = time.time()
    names = list(R.columns)
    col = {n: i for i, n in enumerate(names)}
    for rung, (_, lim) in RUNGS.items():
        T[f"p_gross_{rung}"] = (boot["gross_dd"] < lim).mean(axis=0)
        T[f"p_net_{rung}"] = (boot["net_dd"] < lim).mean(axis=0)
    g3 = ga.G3_NAME
    sel = {"frame": frame_name, "sessions": len(R), "seconds": [round(t1 - t0), round(t2 - t1)], "modes": {}}
    for mode in MODES:
        cur = mode.lower()
        pc = boot[f"{cur}_cagr"]
        T[f"beats_g3_{cur}"] = [float(np.mean(pc[:, col[n]] > pc[:, col[g3]])) for n in names]
        rm_share = {n: float(np.mean(pc[:, col[n]] > pc[:, col[T.at[n, "twin"]]])) for n in names if T.at[n, "twin"]}
        T[f"rm_beats_twin_{cur}"] = pd.Series(rm_share)
        T[f"r6_{cur}"] = T[f"rm_beats_twin_{cur}"].isna() | (T[f"rm_beats_twin_{cur}"] >= 0.90)
        T[f"r3_{cur}"] = (T[[f"{cur}_xs_B", f"{cur}_xs_C", f"{cur}_xs_RECENT"]] > 0).all(axis=1)
        msel = {}
        for rung, (hist, lim) in RUNGS.items():
            if mode == "GROSS":
                r1 = T["gross_dd"] >= hist
                r2 = T[f"p_gross_{rung}"] <= ga.MAX_BREACH
            else:
                r1 = (T["gross_dd"] >= hist) & (T["net_dd"] >= hist)
                r2 = (T[f"p_gross_{rung}"] <= ga.MAX_BREACH) & (T[f"p_net_{rung}"] <= ga.MAX_BREACH)
            T[f"pass_{mode}_{rung}"] = r1 & r2 & T[f"r3_{cur}"] & T[f"r4_{cur}"] & T[f"r6_{cur}"]
            msel[rung] = {"g3_passes": bool(T.at[g3, f"pass_{mode}_{rung}"]),
                          "funnel": [int(r1.sum()), int((r1 & r2).sum()), int((r1 & r2 & T[f"r3_{cur}"]).sum()),
                                     int((r1 & r2 & T[f"r3_{cur}"] & T[f"r4_{cur}"]).sum()), int(T[f"pass_{mode}_{rung}"].sum())]}
            for line in LINES:
                fam = T if line == "MAIN" else T[T["low_touch"].astype(bool)]
                passers = fam[fam[f"pass_{mode}_{rung}"]]
                e = {"family": int(len(fam)), "passers": int(len(passers))}
                if len(passers):
                    top = passers[f"{cur}_cagr"].idxmax()
                    share = pd.Series({n: float(np.mean(pc[:, col[top]] > pc[:, col[n]])) for n in passers.index})
                    share[top] = 0.0
                    band = passers.loc[share.index[share < ga.TIE_SHARE]].copy()
                    band = band.sort_values([f"slot_recent_pass_{cur}", "no_shadow", f"p_gross_{rung}", "pods", f"{cur}_cagr"],
                                            ascending=[False, False, True, True, False])
                    pick = band.index[0]
                    e.update({"top": top, "pick": pick, "band_size": int(len(band)), "band": list(band.index[:20]),
                              "pick_obj": float(T.at[pick, f"{cur}_cagr"]), "top_obj": float(T.at[top, f"{cur}_cagr"]),
                              "pick_p": float(T.at[pick, f"p_gross_{rung}"]), "top_p": float(T.at[top, f"p_gross_{rung}"]),
                              "pick_beats_g3": float(np.mean(pc[:, col[pick]] > pc[:, col[g3]])),
                              "top_beats_g3": float(np.mean(pc[:, col[top]] > pc[:, col[g3]]))})
                msel[rung][line] = e
        for line in LINES:
            ge = msel["GROWTH"][line]
            g3p = float(T.at[g3, "p_gross_GROWTH"])
            if "pick" not in ge:
                prod, how = g3, "no passer; G3 kept"
            elif not msel["GROWTH"]["g3_passes"]:
                prod, how = ge["pick"], "pick by default (G3 fails a GROWTH gate)"
            elif ge["pick"] == g3:
                prod, how = g3, "G3 is the pick"
            elif ge["pick_beats_g3"] >= ga.CHAMP_SHARE and ge["pick_p"] <= g3p:
                prod, how = ge["pick"], "pick replaces G3"
            else:
                prod, how = g3, "G3 kept (champion test not passed)"
            ge["product"], ge["product_how"] = prod, how
            prev = prod
            for rung in ("AGGRESSIVE", "MAX"):
                e = msel[rung][line]
                if "pick" in e:
                    e["beats_prev_product"] = float(np.mean(pc[:, col[e["pick"]]] > pc[:, col[prev]]))
                    e["separate_product"] = bool(e["beats_prev_product"] >= ga.CHAMP_SHARE and e["pick"] != prev)
                    e["product"] = e["pick"] if e["separate_product"] else prev
                    prev = e["product"]
                else:
                    e["product"] = prev
        sel["modes"][mode] = msel
    return {"T": T, "sel": sel, "R": R, "boot": boot}


def main(argv: list[str]) -> int:
    data = lib.load_inputs()
    data["meta"]["ndx_rm"] = {"tier_str": "shadow"}
    books = family()
    assert len(books) == 1395, len(books)
    frames = ext_frames(data)
    for name in argv or FRAMES:
        ga.ledger("ext_frame_started", frame=name)
        frame, start = frames[name]
        res = evaluate(name, data, books, frame, start)
        out = EXT / name
        out.mkdir(parents=True, exist_ok=True)
        res["T"].to_csv(out / "books.csv", float_format="%.6g")
        (out / "selection.json").write_text(json.dumps(res["sel"], indent=1, default=str), encoding="utf-8")
        if name == "main":
            res["R"].to_csv(out / "long_returns.csv.gz", float_format="%.10g", compression="gzip")
        ga.ledger("ext_frame_finished", frame=name)
        print("finished", name, res["sel"]["seconds"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
