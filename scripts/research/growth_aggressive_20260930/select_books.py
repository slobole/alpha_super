"""SPEC 5-6 and 8: evaluate the family on one frame, apply the gates, tie band, tie-break and champion tests.

Usage: python select_books.py [frame ...]     (default: every frame; "main" first)
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import ga_lib as ga
from ga_lib import END, LONG_START, TBILL, lib

LINES = {"MAIN": lambda t: pd.Series(True, index=t.index), "LOW-TOUCH": lambda t: t["low_touch"].astype(bool)}


def build_returns(frame: pd.DataFrame, books: list, start: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({b.name: lib.book_returns(frame, b, start) for b in books})


def evaluate(frame_name: str, data: dict, books: list) -> dict:
    frame, start = ga.frames(data)[frame_name]
    index_all = data["index"]
    t0 = time.time()
    R = build_returns(frame, books, start)
    tb = frame[TBILL].reindex(R.index)
    st = ga.window_stats(R, index_all)
    table = pd.DataFrame({k: v for k, v in st.items()}, index=R.columns)
    for b in books:
        for k, v in b.tags.items():
            table.at[b.name, k] = v
        table.at[b.name, "pods"] = len(b.pods)
        table.at[b.name, "pods_list"] = "+".join(b.pods)
        table.at[b.name, "weights"] = json.dumps({k: round(v, 4) for k, v in b.targets().items()})
        table.at[b.name, "shadow_share"] = sum(w for p, w in b.targets().items()
                                               if p != TBILL and data["meta"][p]["tier_str"] == "shadow")
        table.at[b.name, "pm_ready_share"] = sum(w for p, w in b.targets().items()
                                                 if p != TBILL and data["meta"][p]["tier_str"] == "pm-ready")
    table["no_shadow"] = table["shadow_share"] <= 1e-12
    # R3: investor net excess over T-bills per block (new investor at each block start).
    tb_df = tb.to_frame("tb")
    for block in ("A", "B", "C", "RECENT"):
        lo, hi = ga.BLOCK_DICT[block]
        sub = R.loc[max(lo, R.index[0]):hi]
        if len(sub) < 60:
            table[f"net_xs_{block}"] = np.nan
            continue
        s = ga.window_stats(sub, index_all)
        tbc = lib.cagr(tb_df.loc[sub.index, "tb"], lib.base_date(index_all, sub.iloc[:, 0]))
        table[f"net_xs_{block}"] = s["net_cagr"] - tbc
        table[f"gross_xs_{block}"] = s["gross_cagr"] - tbc
        if block == "RECENT":
            table["net_cagr_RECENT"] = s["net_cagr"]
    # R4 and the RECENT slot test: one T-bill replacement per non-T-bill pod.
    slot_series, slot_keys = {}, []
    for b in books:
        for pod in b.pods:
            if pod == TBILL:
                continue
            key = f"{b.name}##{pod}"
            slot_series[key] = lib.book_returns(lib.replace_pod(frame, pod), b, start)
            slot_keys.append((b.name, pod, key))
    S = pd.DataFrame(slot_series)
    s_long = ga.window_stats(S, index_all)["net_cagr"]
    lo, hi = ga.BLOCK_DICT["RECENT"]
    s_recent = ga.window_stats(S.loc[lo:hi], index_all)["net_cagr"]
    pos = {k: i for i, k in enumerate(S.columns)}
    long_fail, recent_fail, margin = {}, {}, {}
    for name, pod, key in slot_keys:
        d_long = table.at[name, "net_cagr"] - s_long[pos[key]]
        d_recent = table.at[name, "net_cagr_RECENT"] - s_recent[pos[key]]
        margin.setdefault(name, {})[pod] = round(float(d_long), 5)
        if d_long <= 0:
            long_fail.setdefault(name, []).append(pod)
        if d_recent <= 0:
            recent_fail.setdefault(name, []).append(pod)
    table["slot_long_fail_pods"] = [",".join(long_fail.get(n, [])) for n in table.index]
    table["slot_recent_fail_pods"] = [",".join(recent_fail.get(n, [])) for n in table.index]
    table["slot_long_pass"] = table["slot_long_fail_pods"] == ""
    table["slot_recent_pass"] = table["slot_recent_fail_pods"] == ""
    table["slot_long_margin"] = [json.dumps(margin.get(n, {})) for n in table.index]
    t1 = time.time()
    # Bootstrap on this frame's sessions.
    idx = ga.boot_index(len(R), ga.BLOCK_BY_FRAME.get(frame_name, 63.0))
    boot = ga.bootstrap_paths(R.to_numpy(dtype=float), idx)
    t2 = time.time()
    names = list(R.columns)
    col = {n: i for i, n in enumerate(names)}
    for rung, (hist, lim) in ga.RUNGS.items():
        table[f"p_gross_{rung}"] = (boot["gross_dd"] < lim).mean(axis=0)
        table[f"p_net_{rung}"] = (boot["net_dd"] < lim).mean(axis=0)
    table["boot_net_cagr_p50"] = np.median(boot["net_cagr"], axis=0)
    table["boot_net_cagr_p05"] = np.percentile(boot["net_cagr"], 5, axis=0)
    table["boot_gross_dd_p50"] = np.median(boot["gross_dd"], axis=0)
    table["boot_gross_dd_p05"] = np.percentile(boot["gross_dd"], 5, axis=0)
    nc = boot["net_cagr"]
    compass_share = {}
    for n in names:
        if bool(table.at[n, "compass"]):
            compass_share[n] = float(np.mean(nc[:, col[n]] > nc[:, col[table.at[n, "twin"]]]))
    table["compass_beats_twin"] = pd.Series(compass_share)
    table["gate_r5"] = table["compass_beats_twin"].isna() | (table["compass_beats_twin"] >= ga.COMPASS_SHARE)
    table["gate_r3"] = (table[["net_xs_B", "net_xs_C", "net_xs_RECENT"]] > 0).all(axis=1)
    table["gate_r4"] = table["slot_long_pass"]
    g3 = ga.G3_NAME
    table["beats_g3_share"] = [float(np.mean(nc[:, col[n]] > nc[:, col[g3]])) for n in names]

    selection = {"frame": frame_name, "sessions": len(R), "start": str(R.index[0].date()),
                 "seconds": {"books_and_slots": round(t1 - t0, 1), "bootstrap": round(t2 - t1, 1)}, "rungs": {}}
    for rung, (hist, lim) in ga.RUNGS.items():
        table[f"gate_r1_{rung}"] = (table["gross_dd"] >= hist) & (table["net_dd"] >= hist)
        table[f"gate_r2_{rung}"] = (table[f"p_gross_{rung}"] <= ga.MAX_BREACH) & (table[f"p_net_{rung}"] <= ga.MAX_BREACH)
        table[f"pass_{rung}"] = table[f"gate_r1_{rung}"] & table[f"gate_r2_{rung}"] & table["gate_r3"] \
            & table["gate_r4"] & table["gate_r5"]
        g3_row = table.loc[g3]
        selection["rungs"][rung] = {"g3_passes": bool(g3_row[f"pass_{rung}"]),
                                    "g3_failed_gates": [g for g in ("gate_r1_" + rung, "gate_r2_" + rung, "gate_r3", "gate_r4", "gate_r5")
                                                        if not bool(g3_row[g])]}
        for line, mask_fn in LINES.items():
            fam = table[mask_fn(table)]
            passers = fam[fam[f"pass_{rung}"]]
            entry: dict = {"family": int(len(fam)), "passers": int(len(passers))}
            if len(passers):
                top = passers["net_cagr"].idxmax()
                share = pd.Series({n: float(np.mean(nc[:, col[top]] > nc[:, col[n]])) for n in passers.index})
                share[top] = 0.0
                band = passers.loc[share.index[share < ga.TIE_SHARE]].copy()
                band["beaten_by_top"] = share.reindex(band.index)
                band = band.sort_values(["slot_recent_pass", "no_shadow", f"p_gross_{rung}", "pods", "net_cagr"],
                                        ascending=[False, False, True, True, False])
                pick = band.index[0]
                entry.update({"top": top, "band_size": int(len(band)), "pick": pick,
                              "band": list(band.index[:25]),
                              "pick_net_cagr": float(table.at[pick, "net_cagr"]), "top_net_cagr": float(table.at[top, "net_cagr"])})
                for label, name in (("pick", pick), ("top", top)):
                    entry[f"{label}_beats_g3"] = float(np.mean(nc[:, col[name]] > nc[:, col[g3]]))
                    entry[f"{label}_p_breach_gross"] = float(table.at[name, f"p_gross_{rung}"])
                entry["g3_p_breach_gross"] = float(table.at[g3, f"p_gross_{rung}"])
                table.loc[band.index, f"band_{rung}_{line}"] = True
            selection["rungs"][rung][line] = entry
    # Champion tests (SPEC 6).
    for line in LINES:
        ge = selection["rungs"]["GROWTH"][line]
        if "pick" in ge:
            pick = ge["pick"]
            replaces = ge["pick_beats_g3"] >= ga.CHAMP_SHARE and ge["pick_p_breach_gross"] <= ge["g3_p_breach_gross"]
            if not selection["rungs"]["GROWTH"]["g3_passes"]:
                product, how = pick, "pick by default (G3 fails a GROWTH gate)"
            elif pick == g3:
                product, how = g3, "G3 is the pick"
            else:
                product, how = (pick, "pick replaces G3") if replaces else (g3, "G3 kept (champion test not passed)")
            top = ge["top"]
            ge["top_would_replace"] = bool(ge["top_beats_g3"] >= ga.CHAMP_SHARE
                                           and ge["top_p_breach_gross"] <= ge["g3_p_breach_gross"])
        else:
            product, how = g3, "no GROWTH passer; G3 kept"
        ge["product"], ge["product_how"] = product, how
        ae = selection["rungs"]["AGGRESSIVE"][line]
        if "pick" in ae:
            a_pick = ae["pick"]
            share = float(np.mean(nc[:, col[a_pick]] > nc[:, col[product]]))
            top_share = float(np.mean(nc[:, col[ae["top"]]] > nc[:, col[product]]))
            ae["pick_beats_growth_product"] = share
            ae["top_beats_growth_product"] = top_share
            ae["separate_product"] = bool(share >= ga.CHAMP_SHARE and a_pick != product)
            ae["product"] = a_pick if ae["separate_product"] else product
    return {"table": table, "selection": selection, "R": R, "boot": boot, "idx_block": ga.BLOCK_BY_FRAME.get(frame_name, 63.0)}


def main(argv: list[str]) -> int:
    ga.STUDY.mkdir(parents=True, exist_ok=True)
    data = lib.load_inputs()
    books = ga.family_books()
    assert len(books) == 558, len(books)
    assert sum(b.tags["low_touch"] for b in books) == 90
    wanted = argv or list(ga.frames(data))
    summary = {}
    for name in wanted:
        ga.ledger("frame_started", frame=name)
        res = evaluate(name, data, books)
        out = ga.STUDY / name
        out.mkdir(exist_ok=True)
        res["table"].to_csv(out / "books.csv", float_format="%.6g")
        (out / "selection.json").write_text(json.dumps(res["selection"], indent=2, default=str), encoding="utf-8")
        if name == "main":
            res["R"].to_csv(out / "long_returns.csv.gz", float_format="%.10g", compression="gzip")
            np.savez_compressed(out / "boot.npz", **{k: v.astype(np.float32) for k, v in res["boot"].items()},
                                names=np.array(res["R"].columns))
        summary[name] = res["selection"]
        ga.ledger("frame_finished", frame=name, selection=res["selection"])
        print(json.dumps(res["selection"], indent=1, default=str)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
