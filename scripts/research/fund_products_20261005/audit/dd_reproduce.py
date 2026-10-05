"""defensive_delta audit, step 2: reload the A6 inputs and recompute every defensive slot (read-only on the old study).

- The frame fingerprint (a6.Lab.fp) is compared with the fingerprint stored in the A6 tail cache keys.
- Every a6d / a6 defensive slot is rebuilt from its stored weights: quick metrics, frame stats, halves.
- The bootstrap tails of the slots are recomputed from scratch (10 seeds x 2,000 paths, block 63), model costs and
  +5 bps, without reading the cache, and compared with the stored json.

Outputs: results/research/portfolio/fund_products_20261005/audit/defensive_delta/reproduce.json
Nothing is written into the 2026-09-30 study folder or into the main checkout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import data.norgate_loader  # noqa: F401,E402  (bind the `data` package to this worktree's HEAD copy first)

WT = Path(__file__).resolve().parents[4]
FP_DIR = WT / "scripts" / "research" / "fund_products_20260930"
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "defensive_delta"
sys.path.insert(0, str(FP_DIR))

import numpy as np  # noqa: E402
import fp_lib as fp  # noqa: E402
import a6  # noqa: E402
from a6 import LIMITS, Lab, gross_dd  # noqa: E402

OLD = fp.STUDY / "report"
PLUS5 = "s3_plus_5bps"


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                out[f"{prefix}{k}[{i}]"] = x
        else:
            out[f"{prefix}{k}"] = v
    return out


def max_diff(a: dict, b: dict) -> tuple[float, str]:
    fa, fb = flat(a), flat(b)
    worst, where = 0.0, ""
    for k in fa:
        if k not in fb or not isinstance(fa[k], (int, float)) or isinstance(fa[k], bool):
            continue
        d = abs(float(fa[k]) - float(fb[k]))
        if d > worst:
            worst, where = d, k
    return worst, where


def main() -> int:
    t0 = time.time()
    lab = Lab()
    cache_raw = json.loads((OLD / "a6_tail_cache.json").read_text(encoding="utf-8"))
    cache_fps = sorted({json.loads(k)[0] for k in cache_raw if isinstance(json.loads(k)[0], str)})
    import data as data_pkg
    out: dict = {"fingerprint_now": lab.fp, "fingerprints_in_cache": cache_fps, "fingerprint_match": cache_fps == [lab.fp],
                 "frame_start": str(lab.start), "frame_end": str(fp.END), "n_days": int(len(lab.ret(a6.GROWTH))),
                 "data_pkg_path": str(Path(data_pkg.__file__).parent), "lib_path": str(Path(fp.lib.__file__)), "slots": {}}
    print("fingerprint now", lab.fp, "| in cache", cache_fps, "| match", out["fingerprint_match"], flush=True)

    stored = {"a6d": json.loads((OLD / "a6d.json").read_text(encoding="utf-8")),
              "a6": json.loads((OLD / "a6.json").read_text(encoding="utf-8"))}
    # Rebuild each stored slot from its stored weights.
    todo = []
    for src, js in stored.items():
        for slot, v in js["defensive"].items():
            if v is None:
                continue
            n = f"{src}:{slot}"
            lab.add(n, v["weights"], base=n, stage="audit")
            lab.add(n + "@5", v["weights"], frame=PLUS5, base=n, stage="audit")
            todo.append((src, slot, n, v))

    # Tails from scratch (no cache): one batched pass per seed over all slots, both frames.
    names = [n for _, _, n, _ in todo] + [n + "@5" for _, _, n, _ in todo]
    R = np.column_stack([lab.ret(lab.cands[n]["w"], 1.0, lab.cands[n].get("frame", "main")).to_numpy() for n in names])
    assert not np.isnan(R).any()
    acc = {n: [] for n in names}
    for s in range(10):
        dd = gross_dd(R, lab.idx(s))
        for j, n in enumerate(names):
            acc[n].append({L: float((dd[:, j] < L).mean()) for L in LIMITS})
        print(f"  seed {s} done ({time.time() - t0:.0f}s)", flush=True)

    def tails(n: str) -> dict:
        t = {}
        for L in LIMITS:
            key = f"p{int(round(-L * 100))}"
            vals = [d[L] for d in acc[n]]
            t[key], t[key + "_max"] = float(np.mean(vals)), float(np.max(vals))
        return t

    worst_all = 0.0
    for src, slot, n, v in todo:
        q = lab.cands[n]["q"]
        new = {"q": q, "tails": tails(n), "frames": {f: lab.frame_stats(n, f) for f in (PLUS5, "s6_exact", "s1_house_cash")},
               "halves_xs": list(lab.halves(n)), "floor_ok": lab.floor_ok(n)}
        old = {"q": v["q"], "tails": v["tails"], "frames": v["frames"], "halves_xs": v["halves_xs"], "floor_ok": v["floor_ok"]}
        if v.get("tails_plus5"):
            new["tails_plus5"], old["tails_plus5"] = tails(n + "@5"), v["tails_plus5"]
        d, where = max_diff(old, new)
        worst_all = max(worst_all, d)
        g = {"cagr": q["cagr"], "xs": q["xs"], "dd": q["dd"], "vol": q["vol"], "gfc": q["crises"]["gfc"],
             "bear_2022": q["crises"]["bear_2022"], "worst_crisis": q["worst_crisis"], "worst_year": q["worst_year"],
             "p10": new["tails"]["p10"], "p10_max": new["tails"]["p10_max"], "p7": new["tails"]["p7"], "p7_max": new["tails"]["p7_max"]}
        if "tails_plus5" in new:
            g |= {"p10_plus5": new["tails_plus5"]["p10"], "p10_plus5_max": new["tails_plus5"]["p10_max"],
                  "p7_plus5": new["tails_plus5"]["p7"], "p7_plus5_max": new["tails_plus5"]["p7_max"]}
        out["slots"][n] = {"name": v["name"], "weights": v["weights"], "max_abs_diff": d, "max_diff_field": where,
                           "floor_ok_same": new["floor_ok"] == old["floor_ok"], "recomputed": g, "new": new}
        print(f"{n:22s} {v['name']:26s} max|diff| {d:.3e} ({where}) cagr {q['cagr']:.4%} xs {q['xs']:.4f} dd {q['dd']:.4%} "
              f"gfc {q['crises']['gfc']:+.4%} 2022 {q['crises']['bear_2022']:+.4%} p10 {g['p10']:.4f}/{g['p10_max']:.4f}", flush=True)
    out["max_abs_diff_all"] = worst_all
    out["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reproduce.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("MAX |diff| over all slots and fields:", worst_all, "| seconds", round(out["seconds"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
