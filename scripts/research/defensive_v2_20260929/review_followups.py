"""Amendment A3 (after the A2 results and the independent review; post-result, labelled).

  1. Tie-break (2) under both readings of "without a live route" (code: etf_dv2, eom_flow, trinity; SPEC-literal:
     also the daily PM_READY pods disp and downshock).
  2. EOM stress frames: +10 / +20 bps per side on EOM's own fills; the rule on the EXACT window only.
  3. Forward split: the rule on one half of the LONG window, the chosen book scored on the other half.

Every rule run here re-evaluates R1, R3 (blocks inside its window), R4 (slot test) and R2 (own bootstrap) for the
whole MAIN family and selects with LONG-rule mechanics on its own bootstrap. R5 is not re-run inside these frames.

Usage: python review_followups.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "research" / "shelf_rebuild_20260929"))
sys.path.insert(0, str(HERE))

import lib  # noqa: E402
from lib import END, EXACT_START, LONG_START, TBILL  # noqa: E402
import defensive_v2 as dv  # noqa: E402
import sharpe_checks as sc  # noqa: E402

OUT = dv.OUT / "sharpe_checks"
RUNNER_UP = "CORE5 + BTAL_QQQ + EOM [IV]"
NOT_LIVE = {"code": set(dv.NOT_LIVE), "spec_literal": set(dv.NOT_LIVE) | {"disp", "downshock"}}
MID = pd.Timestamp("2017-06-30")
T = pd.Timestamp


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ledger(event: str, **fields) -> None:
    """Append to this study's ledger with the current SPEC hash (lib.ledger writes to the shelf rebuild's)."""
    import hashlib
    from datetime import datetime, timezone
    rec = {"event_str": event, "recorded_at_utc_str": datetime.now(timezone.utc).isoformat(),
           "spec_sha256_str": hashlib.sha256((HERE / "SPEC_FROZEN.md").read_bytes()).hexdigest(), **fields}
    with (dv.OUT / "experiment_ledger.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps(rec) + "\n")


def not_live_share(t: pd.DataFrame, reading: str) -> pd.Series:
    out = {}
    for b in t.index:
        w = json.loads(t.at[b, "avg_weights"])
        out[b] = sum(v for p, v in w.items() if p in NOT_LIVE[reading])
    return pd.Series(out)


def series_set(frame: pd.DataFrame, books: list[dv.DefBook]) -> tuple[pd.DataFrame, dict]:
    """Book series and slot-test series (each pod, and G3, replaced by T-bills) for the whole family."""
    main, slots = {}, {}
    for i, b in enumerate(books):
        main[b.name] = dv.book_series(frame, b)
        names = list(b.pods) + (["G3"] if b.slice_ > 0 else [])
        if len(names) > 1:
            slots[b.name] = {p: dv.book_series(frame, b, replace=p, weight_source=frame) for p in names}
        else:
            slots[b.name] = {}
        if i % 200 == 0:
            log(f"  series {i}/{len(books)}")
    return pd.DataFrame(main), slots


def rule(R: pd.DataFrame, slots: dict, tb: pd.Series, idx, lo, hi, blocks: dict, base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """R1-R4 on [lo, hi] (R3 over `blocks`), R2 on own bootstrap, then the Sharpe / excess-Sharpe selection."""
    Rw = R.loc[lo:hi]
    t = base[["pods", "pods_list", "avg_weights", "trade_days_per_year"]].copy()
    rows = {}
    for b in Rw.columns:
        r = Rw[b]
        ecal = lib.excess_calmar(r, tb, idx)
        xs_blocks = [lib.excess_cagr(lib.window(r, a, z), tb, idx) for a, z in blocks.values()]
        if slots[b]:
            slot_ok = all(lib.excess_calmar(s.loc[lo:hi], tb, idx) < ecal for s in slots[b].values())
        else:
            slot_ok = lib.excess_cagr(r, tb, idx) > 0
        rows[b] = {"maxdd_w": lib.maxdd(r), "r3_w": all(x > 0 for x in xs_blocks), "r4_w": slot_ok,
                   "sharpe": sc.sharpe(r), "xs_sharpe": sc.xs_sharpe(r, tb)}
    t = t.join(pd.DataFrame(rows).T)
    t["r1_w"] = t["maxdd_w"] >= dv.DD_HIST
    cand = list(t.index[t[["r1_w", "r3_w", "r4_w"]].astype(bool).all(axis=1)])
    cols = list(dict.fromkeys(cand + [sc.CHAMPION]))
    sh, xs, dd = sc.boot_sharpe_dd(Rw[cols].to_numpy(), tb.reindex(Rw.index).to_numpy())
    t["p_breach10"] = pd.Series((dd < dv.DD_LIMIT).mean(axis=0), index=cols)
    t["gates_pass"] = t[["r1_w", "r3_w", "r4_w"]].astype(bool).all(axis=1) & (t["p_breach10"] <= dv.BREACH_MAX)
    for c in ("sharpe", "xs_sharpe", "maxdd_w"):
        t[c] = t[c].astype(float)
    out = {"gate_passers": int(t["gates_pass"].sum()), "champion_passes": bool(t.at[sc.CHAMPION, "gates_pass"])}
    for reading in NOT_LIVE:
        t["not_live_share"] = not_live_share(t, reading)
        out[reading] = {obj: {k: sc.select(t, obj, cols, boot, v) for k, v in sc.pools(t).items()}
                        for obj, boot in (("sharpe", sh), ("xs_sharpe", xs))}
    return t, out


def score(R: pd.DataFrame, tb: pd.Series, idx, lo, hi, books: list[str]) -> dict:
    Rw = R.loc[lo:hi]
    sh = Rw.mean() / Rw.std() * np.sqrt(252)
    rank = sh.rank(pct=True)
    return {b: {"sharpe": float(sh[b]), "xs_sharpe": sc.xs_sharpe(Rw[b], tb), "cagr": lib.cagr(Rw[b], lib.base_date(idx, Rw[b])),
                "maxdd": lib.maxdd(Rw[b]), "sharpe_pct_rank_all_books": float(rank[b])} for b in books}


def picks_of(res: dict) -> set:
    s = set()
    for reading in NOT_LIVE:
        for obj in ("sharpe", "xs_sharpe"):
            for k in ("all", "no_eom"):
                r = res[reading][obj].get(k, {})
                for key in ("pick", "top", "recommendation"):
                    if r.get(key):
                        s.add(r[key])
    return s


def slim(res: dict) -> dict:
    """Drop the long band lists but keep their size."""
    out = {k: v for k, v in res.items() if k not in NOT_LIVE}
    for reading in NOT_LIVE:
        out[reading] = {obj: {k: {kk: (len(vv) if kk == "band" else vv) for kk, vv in r.items()} for k, r in d.items()}
                        for obj, d in res[reading].items()}
    return out


def main() -> int:
    ledger("A3_review_followups_started")
    log("loading")
    data = lib.load_inputs()
    idx = data["index"]
    fair = data["cash_long"]
    tb = fair[TBILL]
    base = pd.read_csv(dv.OUT / "main_books_sharpe.csv", index_col=0)
    house_t = pd.read_csv(OUT / "house_main_books.csv", index_col=0)
    books = dv.family("MAIN")
    result: dict = {}

    # 1. classification: re-select the A1/A2 main runs under the SPEC-literal reading
    log("classification")
    R = pd.read_csv(dv.OUT / "main_long_returns.csv.gz", index_col=0, parse_dates=True)
    names = list(R.columns)
    sh_f, xs_f, _ = sc.boot_sharpe_dd(R.to_numpy(), tb.reindex(R.index).to_numpy())
    t = base.copy()
    t["xs_sharpe"] = [sc.xs_sharpe(R[b], tb) for b in t.index]
    cls = {}
    for reading in NOT_LIVE:
        t["not_live_share"] = not_live_share(t, reading)
        cls[f"fair|{reading}"] = {obj: {k: sc.select(t, obj, names, boot, v) for k, v in sc.pools(t).items()}
                                  for obj, boot in (("sharpe", sh_f), ("xs_sharpe", xs_f))}
    # house: bootstrap only the house gate-passers (+ champion), as in sharpe_checks
    frames_h = sc.house_frames(data)
    hp = list(house_t.index[house_t["gates_pass"].astype(bool)])
    hcols = list(dict.fromkeys(hp + [sc.CHAMPION]))
    by_name = {b.name: b for b in books}
    HR = pd.DataFrame({b: dv.book_series(frames_h["main"], by_name[b]) for b in hcols})
    sh_h, xs_h, _ = sc.boot_sharpe_dd(HR.to_numpy(), tb.reindex(HR.index).to_numpy())
    th = house_t.copy()
    th["avg_weights"] = base["avg_weights"].reindex(th.index)
    for reading in NOT_LIVE:
        th["not_live_share"] = not_live_share(th, reading)
        cls[f"house|{reading}"] = {obj: {k: sc.select(th, obj, hcols, boot, [b for b in v if b in hcols])
                                         for k, v in sc.pools(th).items()}
                                   for obj, boot in (("sharpe", sh_h), ("xs_sharpe", xs_h))}
    result["classification"] = cls

    # 2 and 3. frames for the full rule
    log("fair series + slots")
    R_fair, S_fair = series_set(fair, books)
    frames = {}
    for bps in (10, 20):
        f = fair.copy()
        drag = lib.evaluation.extra_slippage_cost_ser(data["tx"]["eom_flow"], data["nav"]["eom_flow"], bps / 1e4)
        drag = drag.reindex(f.index).fillna(0.0)
        live = f["eom_flow"].notna()
        f.loc[live, "eom_flow"] = f.loc[live, "eom_flow"] - drag[live]
        frames[f"eom_plus_{bps}bps"] = f
    blocks_long = dict(lib.BLOCK_DICT)
    blocks_exact = {k: v for k, v in lib.BLOCK_DICT.items() if k != "A"}
    runs = {}
    report_books = {sc.A1_PICK, RUNNER_UP, sc.CHAMPION, sc.A1_NOEOM, sc.DPRIME}
    for label, f in frames.items():
        log(f"rule: {label}")
        Rf, Sf = series_set(f, books)
        _, runs[label] = rule(Rf, Sf, tb, idx, LONG_START, END, blocks_long, base)
        runs[label]["scores"] = score(Rf, tb, idx, LONG_START, END, sorted(report_books | picks_of(runs[label])))
    log("rule: EXACT only")
    _, runs["exact_only"] = rule(R_fair, S_fair, tb, idx, EXACT_START, END, blocks_exact, base)
    runs["exact_only"]["scores"] = score(R_fair, tb, idx, EXACT_START, END, sorted(report_books | picks_of(runs["exact_only"])))

    halves = {"first_half": ((LONG_START, MID), (MID + pd.Timedelta(days=1), END),
                             {"A": (LONG_START, T("2012-10-01")), "B1": (T("2012-10-02"), MID)}),
              "second_half": ((MID + pd.Timedelta(days=1), END), (LONG_START, MID),
                              {"B2": (MID + pd.Timedelta(days=1), T("2021-12-31")), "C": (T("2022-01-03"), END)})}
    fwd = {}
    for label, ((lo, hi), (olo, ohi), blocks) in halves.items():
        log(f"forward: select on {label}")
        _, res = rule(R_fair, S_fair, tb, idx, lo, hi, blocks, base)
        chosen = picks_of(res) | report_books
        res["in_sample"] = score(R_fair, tb, idx, lo, hi, sorted(chosen))
        res["out_of_sample"] = score(R_fair, tb, idx, olo, ohi, sorted(chosen))
        res["windows"] = {"select": [str(lo.date()), str(hi.date())], "score": [str(olo.date()), str(ohi.date())]}
        fwd[label] = res
    result["stress"] = {k: slim(v) for k, v in runs.items()}
    result["forward"] = {k: slim(v) for k, v in fwd.items()}
    (OUT / "review_followups.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    ledger("A3_review_followups_finished")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
