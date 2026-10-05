"""Fund products, final pass: the monthly-only growth books (amendment O3, owner decision 2026-10-05, after results).

Exploratory, not pre-registered. Three questions the owner asked:
  1. Does the momentum capsule earn a weight in a monthly book of one TAA variant and CORE5?  (grid: 2 TAA variants x
     4 TAA:CORE5 ratios x momentum 0 / 15 / 30%, with the paired test against the same book without momentum)
  2. Is CORE5 needed, and what do the 2026-10-01 four-pod books add?  (reference rows)
  3. Which monthly book reaches 20% CAGR, and does leverage on a calmer mix do better than more TAA 3x 1N?

Usage: PYTHONDONTWRITEBYTECODE=1 python monthly.py   (after study.py). Writes <study>/report/monthly.json.
"""

from __future__ import annotations

import json

import g_lib as g
from g_lib import Lab


def main() -> int:
    lab = Lab()
    rf = lab.rf

    def row(name: str, w: dict, base: str | None = None) -> dict:
        n = lab.add(name, w, kind="monthly")
        q, t = lab.cands[n]["q"], lab.tails(n)
        out = {"weights": {k: float(v) for k, v in w.items()}, "q": q, "tails": t, "plus5": g.stats(lab.ret(w, "s3_plus_5bps"), rf),
               "rung_growth": lab.rung_ok(n, "GROWTH"), "rung_growth_plus": lab.rung_ok(n, "GROWTH PLUS")}
        if base:
            p = lab.paired(lab.r(n), lab.r(base))
            out["vs_no_momentum"] = {"share_xs": p["share_xs"], "share_cagr": p["share_cagr"]}
        return out

    mk = lambda taa, t: g.blend((t, {taa: 1.0}), (1 - t, {"core5": 1.0}))  # noqa: E731
    out: dict = {"grid": {}, "ladder": {}, "reference": {}, "corr": {}}
    for taa in ("taa3x", "taa3x_1n"):
        for t in (0.4, 0.5, 0.6, 0.7):
            base = f"{taa} {t:.0%}:{1 - t:.0%} mom 0%"
            for m in (0.0, 0.15, 0.30):
                w = g.blend(((1 - m) * t, {taa: 1.0}), ((1 - m) * (1 - t), {"core5": 1.0}), *(((m, g.MOM),) if m else ()))
                name = f"{taa} {t:.0%}:{1 - t:.0%} mom {m:.0%}"
                out["grid"][name] = {"taa": taa, "taa_to_core5": t, "momentum": m, **row(name, w, base if m else None)}
    for t in (0.40, 0.50, 0.60, 0.65, 0.70):
        out["ladder"][f"1N {t:.0%} / CORE5 {1 - t:.0%}"] = row(f"ladder 1N {t:.2f}", mk("taa3x_1n", t))
    for name, w, L in (("1N 50 / CORE5 50 x1.25", mk("taa3x_1n", 0.5), 1.25), ("1N 40 / CORE5 60 x1.50", mk("taa3x_1n", 0.4), 1.5)):
        out["ladder"][name] = {**row(name, g.levered(w, L)), "L": L}
    out["reference"] = {"old monthly (2026-10-01)": row("ref old monthly", g.INCUMBENT), "old monthly plus (2026-10-01)": row("ref old plus", g.OLD_PLUS),
                        "TAA 3x 57 / momentum 43, no CORE5": row("ref no core5", g.blend((4, {"taa3x": 1.0}), (3, g.MOM))),
                        "TAA 3x 40 / momentum 30 / CORE5 30": row("ref A", g.blend((0.4, {"taa3x": 1.0}), (0.3, g.MOM), (0.3, {"core5": 1.0})))}
    fr = lab.frame
    out["corr"] = {"btal_qqq | taa3x_1n": float(fr["btal_qqq"].corr(fr["taa3x_1n"])), "core5 | taa3x_1n": float(fr["core5"].corr(fr["taa3x_1n"])),
                   "core5 | taa3x": float(fr["core5"].corr(fr["taa3x"])), "core5 | MOM": float(fr["core5"].corr(lab.ret(g.MOM))),
                   "ndx_vxn | MOM": float(fr["ndx_vxn"].corr(lab.ret(g.MOM)))}
    (g.OUT / "monthly.json").write_text(json.dumps(g.r6(out), indent=1, default=str), encoding="utf-8")
    g.ledger("monthly_finished")
    print("monthly done", len(out["grid"]), "grid books")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
