"""Read-only: dump the JSON shape of the v4 report inputs (report_a6.json, a7.json) for the report_map audit."""
import json
from pathlib import Path

WT = Path(__file__).resolve().parents[4]
REP = WT / "results" / "research" / "portfolio" / "fund_products_20260930" / "report"
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "report_map"
OUT.mkdir(parents=True, exist_ok=True)


def shape(x, depth=0, maxd=4, maxk=40):
    if isinstance(x, dict):
        if depth >= maxd:
            return f"dict[{len(x)}] keys={list(x)[:8]}"
        ks = list(x)
        return {k: shape(x[k], depth + 1, maxd, maxk) for k in ks[:maxk]} | ({"...": f"+{len(ks) - maxk} more"} if len(ks) > maxk else {})
    if isinstance(x, list):
        if not x:
            return "list[0]"
        return [f"list[{len(x)}]", shape(x[0], depth + 1, maxd, maxk)]
    return type(x).__name__ + (f"={x!r}" if isinstance(x, (int, float, str, bool)) and len(str(x)) < 40 else "")


lines = []
D = json.loads((REP / "report_a6.json").read_text(encoding="utf-8"))
lines.append("== report_a6.json top keys: " + str(list(D)))
lines.append("rows keys: " + str(list(D["rows"])))
r = D["rows"]["g_launch"]
lines.append("row g_launch shape:\n" + json.dumps(shape(r, 0, 3), indent=1, ensure_ascii=False))
lines.append("row g_launch m keys: " + str(list(r["m"])))
lines.append("row g_launch m (scalars): " + json.dumps({k: v for k, v in r["m"].items() if not isinstance(v, (dict, list))}, indent=1))
lines.append("m.crises keys: " + str(list(r["m"]["crises"])))
lines.append("m.crises_dd keys: " + str(list(r["m"]["crises_dd"])))
lines.append("m.years keys: " + str(list(r["m"]["years"])))
lines.append(f"m.nav: len {len(r['m']['nav'])} first {r['m']['nav'][:2]} last {r['m']['nav'][-2:]}")
lines.append("tail: " + json.dumps(r["tail"]))
lines.append("ops: " + json.dumps(r["ops"]))
lines.append("frames: " + json.dumps(r["frames"]))
for k in ("weights", "lever", "needs_wiring", "daily_pods", "capacity", "capacity_top", "cash", "g", "plus10_cagr", "financing", "reg_t", "reg_t_flag", "halves_xs", "a6_name", "gated", "tails_plus5", "product"):
    lines.append(f"g_launch.{k} = {json.dumps(r.get(k))}")
r2 = D["rows"]["g_g22_target"]
for k in ("weights", "lever", "capacity", "capacity_top", "plus10_cagr", "financing", "reg_t", "reg_t_flag", "halves_xs", "a6_name"):
    lines.append(f"g_g22_target.{k} = {json.dumps(r2.get(k))}")
r3 = D["rows"]["d_launch"]
for k in ("weights", "lever", "capacity", "capacity_top", "cash", "g", "plus10_cagr", "financing", "reg_t", "halves_xs", "a6_name", "tails_plus5", "tail", "needs_wiring", "ops"):
    lines.append(f"d_launch.{k} = {json.dumps(r3.get(k))}")
lines.append("bench keys: " + str(list(D["bench"])) + " ; bench[S&P 500] keys: " + str(list(D["bench"]["S&P 500"])))
lines.append("alpha: len %d first %s" % (len(D["alpha"]), json.dumps(D["alpha"][0])))
lines.append("alpha books: " + str(sorted({a["book"] for a in D["alpha"]})) + " bases " + str(sorted({a["basis"] for a in D["alpha"]})) + " models " + str(sorted({a["model"] for a in D["alpha"]})))
lines.append("corr: names %s all %dx%d" % (D["corr"]["names"], len(D["corr"]["all"]), len(D["corr"]["all"][0])))
lines.append("sweep: len %d first %s" % (len(D["sweep"]), json.dumps(shape(D["sweep"][0], 0, 2))))
lines.append("challenges keys: " + str({k: len(v) for k, v in D["challenges"].items()}))
lines.append("challenge example: " + json.dumps(D["challenges"]["g22_target"][0]))
lines.append("next_vs_launch: " + json.dumps(D["next_vs_launch"]))
lines.append("notes keys: " + str(list(D["notes"])))
lines.append("profile keys: " + str(list(D["profile"])) + " -> " + str(list(D["profile"]["core5"])))
A7 = json.loads((REP / "a7.json").read_text(encoding="utf-8"))
lines.append("== a7.json top keys: " + str(list(A7)))
lines.append("targets: " + str(A7["targets"]))
lines.append("unlevered: " + json.dumps(A7["unlevered"]))
lines.append("levered keys: " + str(list(A7["levered"])) + " launch_x: " + json.dumps(A7["levered"]["launch_x"], ensure_ascii=False))
k0 = next(iter(A7["rows"]))
lines.append(f"rows: {len(A7['rows'])}; first {k0}: " + json.dumps(A7["rows"][k0]))
lines.append("ceiling: " + json.dumps(A7.get("ceiling")))
# one-line table of the v4 rows
lines.append("== v4 rows summary (key, weights, lever, cagr, net_cagr, xsharpe, sharpe, maxdd, capacity)")
for k, r in D["rows"].items():
    m = r["m"]
    lines.append(f"{k:18s} L={r['lever']:<5} cagr={m['cagr']:.4f} net={m['net_cagr']:.4f} xs={m['xsharpe']:.3f} sharpe={m.get('sharpe')} maxdd={m['maxdd']:.4f} cap={r['capacity']} top={r['capacity_top']} w={r['weights']}")
(OUT / "shape.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
