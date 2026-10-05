"""Reviewer scratch (report lens): spot-check rendered table cells against the JSONs with an independent mapping."""
import json, re, sys
from pathlib import Path
from bs4 import BeautifulSoup
WT = Path(__file__).resolve().parents[5]
REP = WT / "results/research/portfolio/fund_products_20261005/report"
OLD = WT / "results/research/portfolio/fund_products_20260930/report"
ld = lambda p: json.loads(p.read_text(encoding="utf-8"))
S, B, C, X, D = (ld(REP / f"{n}.json") for n in ("study", "battery", "capacity", "exposure", "defensive"))
A6 = ld(OLD / "report_a6.json")
K = S["books"]; HL = B["edge_decay"]["headline"]; ED = B["edge_decay"]; BT = B["bootstrap"]
soup = BeautifulSoup(Path(sys.argv[1]).read_text(encoding="utf-8"), "lxml")
clean = lambda s: re.sub(r"\s+", " ", s).strip()
tables = []
for t in soup.find_all("table"):
    rows = [[clean(c.get_text(" ")) for c in r.find_all(["th", "td"])] for r in t.find_all("tr")]
    tables.append(rows)


def find(header_first, contains=None):
    for rows in tables:
        if rows and rows[0] and rows[0][0].startswith(header_first) and (contains is None or any(contains in h for h in rows[0])):
            return rows
    raise KeyError(header_first)


def row(rows, start):
    for r in rows:
        if r and r[0].startswith(start):
            return r
    raise KeyError(start)


def col(rows, name):
    return next(i for i, h in enumerate(rows[0]) if h == name)


P = lambda x, d=1: f"{x * 100:.{d}f}%"
SG = lambda x, d=1: ("+" if x > 0 else "−" if x < 0 else "") + f"{abs(x) * 100:.{d}f}%"
F = lambda x, d=2: f"{x:.{d}f}"
S9 = "S9 incumbent launch"
checks = []


def chk(label, got, exp):
    checks.append((label, got, exp, got == exp or got.startswith(exp)))


# 1 products menu
t = find("Option", "Planning CAGR")
for he, k in (("צמיחה 33%", "GR1"), ("צמיחה פלוס", "GR2"), ("אגרסיבי", "GR3"), ("התיק החודשי", S9)):
    r = row(t, he)
    chk(f"menu {k} CAGR", r[col(t, "CAGR")], P(K[k]["q"]["cagr"]))
    chk(f"menu {k} Planning", r[col(t, "Planning CAGR")], P(HL[k]["planning"]["cagr"]))
    chk(f"menu {k} Floor", r[col(t, "Floor CAGR")], P(HL[k]["floor"]["cagr"]))
    chk(f"menu {k} xs", r[col(t, "Excess Sharpe")], F(K[k]["q"]["xs"]))
    chk(f"menu {k} Vol", r[col(t, "Vol")], P(K[k]["q"]["vol"]))
    chk(f"menu {k} MaxDD", r[col(t, "Max DD")], SG(K[k]["q"]["dd"]))
    chk(f"menu {k} P20", r[col(t, "P(DD<−20%)")], P(K[k]["tails"]["p20"]))
    chk(f"menu {k} 2008", r[col(t, "2008")], SG(K[k]["q"]["crises"]["gfc"]))
    chk(f"menu {k} +5bps", r[col(t, "CAGR +5 bps")], P(K[k]["frames"]["s3_plus_5bps"]["cagr"]))
    chk(f"menu {k} cash0", r[col(t, "CAGR cash 0%")], P(K[k]["frames"]["s1_house_cash"]["cagr"]))
    chk(f"menu {k} net", r[col(t, "Net 2/20")], P(K[k]["net"]["cagr"]))
# 2 dial map
t = find("TAA share")
for lab, k in (("TAA 3x 40%", "dial taa3x 40"), ("TAA 3x 60%", "dial taa3x 60"), ("TAA 3x 1N 67%", "dial taa3x_1n 67")):
    r = row(t, lab)
    chk(f"dial {k} CAGR", r[1], P(K[k]["q"]["cagr"]))
    chk(f"dial {k} xs", r[2], F(K[k]["q"]["xs"]))
    chk(f"dial {k} dd", r[4], SG(K[k]["q"]["dd"]))
    chk(f"dial {k} p25", r[6], P(K[k]["tails"]["p25"]))
    chk(f"dial {k} nasdaq peak", r[10], f'{X["books"][k]["nasdaq_lookthrough"]["max"]:.2f}×')
    chk(f"dial {k} rung", r[12], K[k]["strictest_rung"] or "none")
# 3 slots
t = find("GR1 with one capsule replaced")
for i, k in ((2, "T1 MOM -> BIL"), (3, "T2 MR -> BIL (GR1-L)"), (4, "T3 TAA -> BIL"), (5, "T4 MOM -> QQQ")):
    r = t[i]
    s = S["slots"][k]
    chk(f"slot {k} CAGR", r[1], P(s["q"]["cagr"]))
    chk(f"slot {k} xs", r[2], F(s["q"]["xs"]))
    chk(f"slot {k} shares", r[5], " / ".join(P(s["frames"][f]["share_xs"], 0) for f in ("main", "plus5", "plus10")))
    chk(f"slot {k} blocks", r[7], " / ".join(P(s["blocks"][b]["share_xs"], 0) for b in "ABC"))
# 4 challengers
t = find("Challenger against Growth")
for c in S["challenges"]:
    r = row(t, c["challenger"])
    chk(f"chal {c['challenger']} share", r[col(t, "Higher Excess Sharpe: paths")].replace(" ~", ""), P(c["share_xs"], 0))
    chk(f"chal {c['challenger']} passed", r[col(t, "Passed")], "yes" if c["passed"] else "no")
    chk(f"chal {c['challenger']} c5", r[col(t, "5 H1")], "yes" if c["checks"]["c5_h1_higher"] else "no")
# 5 edge decay
t = find("Book: CAGR / Excess Sharpe")
for he, k in (("צמיחה", "GR1"), ("אגרסיבי", "GR3"), ("התיק החודשי", S9)):
    r = row(t, he)
    for j, sc in enumerate(["all at 0.75", "all at 0.5", "TAA dead", "MOM dead", "MR dead"]):
        e = ED["scenarios"][sc][k]
        chk(f"decay {k} {sc}", r[2 + j], f'{P(e["cagr"])} / {F(e["xs"])}')
# 6 believe
t = find("Book", "Deflated Sharpe, N = 100 / 1,000 trials")
for i, k in ((1, "GR1"), (2, "GR2"), (3, "GR3"), (4, S9)):
    r = t[i]
    v = B["believe"][k]
    chk(f"believe {k} planning", r[2], f'{P(HL[k]["planning"]["cagr"])} / {F(HL[k]["planning"]["xs"])}')
    chk(f"believe {k} xs interval", r[4], f'{F(v["k1"]["xs_p5_50_95"][0])} … {F(v["k1"]["xs_p5_50_95"][2])}')
    chk(f"believe {k} P(xs<1)", r[6].split(" / ")[0], P(v["k1"]["p_xs_below_1.0"], 0))
    chk(f"believe {k} DSR", r[8], f'{F(v["dsr_N100"]["dsr"])} / {F(v["dsr_N1000"]["dsr"])}')
# 7 rung reading
t = find("Breach figure at the product's limit")
for i, k, hk in ((1, "GR1", "p20"), (2, "GR2", "p25"), (3, "GR3", "p30"), (4, S9, "p20")):
    r = t[i]
    v21 = BT["blocks"]["21"][k][hk]
    chk(f"rung {k} block21", r[2], P(v21, 0 if v21 > 0.1 else 1))
    chk(f"rung {k} edge 3/4", r[6], P(ED["scenarios"]["all at 0.75"][k][hk]))
    chk(f"rung {k} exact", r[10], P(BT["frames"]["s6_exact"][k][hk]))
# 8 crises + years
t = find("Event")
r = row(t, "COVID 2020")
for j, k in enumerate(("GR1", "GR2", "GR3", S9)):
    chk(f"crisis covid {k}", r[1 + j], f'{SG(K[k]["q"]["crises"]["covid"])} ({SG(K[k]["crises_dd"]["covid"])})')
r = row(t, "2022 bear")
chk("crisis 2022 SPX", r[5], SG(A6["bench"]["S&P 500"]["crises"]["bear_2022"]))
chk("crisis 2022 QQQ", r[6], SG(A6["bench"]["QQQ"]["crises"]["bear_2022"]))
t = find("Year")
for y in ("2013", "2020", "2022", "2023"):
    r = row(t, y)
    for j, k in enumerate(("GR1", "GR2", "GR3", S9)):
        chk(f"year {y} {k}", r[1 + j], SG(K[k]["years"][y]))
# 9 alpha
t = find("Book, basis")


def al(book, basis, window, model):
    a = next(a for a in B["alpha"] if a["book"] == book and a["basis"] == basis and a["window"] == window and a["model"] == model)
    return f'{SG(a["alpha_ann"])} (t {a["alpha_t"]:.1f})'


chk("alpha GR1 gross M2 long", row(t, "Growth, gross")[3], al("GR1", "gross", "long", "M2"))
chk("alpha GR1 net h2", row(t, "Growth, net")[5], al("GR1", "net", "h2", "M2"))
chk("alpha Monthly gross M0", row(t, "Monthly, gross")[1], al(S9, "gross", "long", "M0"))
chk("alpha MOM gross h2", row(t, "Momentum capsule alone, gross")[5], al("capsule MOM", "gross", "h2", "M2"))
# 10 capacity
t = find("Book", "House model: close auction")
usd = lambda x: "$" + (f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:.0f}K")
for he, k in (("צמיחה", "GR1"), ("התיק החודשי", S9)):
    r = row(t, he)
    for j, route in enumerate(("MOO", "MOC", "worked+blocks")):
        rec = C["books"][k]["routes"][route]["recommended"]
        got = r[1 + j].split(" ")[0].replace("≥", "")
        chk(f"capacity {k} {route}", got, usd(rec))
# 11 ease
t = find("Book", "Margin accounts")
r = row(t, "צמיחה")
e = C["books"]["GR1"]
chk("ease GR1 pods", r[1], f'{e["ease"]["research_pods"]} / {e["ease"]["live_pods"]}')
chk("ease GR1 orders", r[4], f'{e["orders_per_year"]:.0f}')
chk("ease GR1 trade days", r[3], f'{e["ease"]["trade_days_per_year"]:.0f}')
# 12 after window
t = find("Book", "Return after the window")
for i, k in ((1, "GR1"), (4, S9)):
    chk(f"after {k}", t[i][1], SG(B["after_window"]["books"][k]["after_window"]))
bad = [c for c in checks if not c[3]]
print(f"cells checked: {len(checks)}; mismatches: {len(bad)}")
for c in bad:
    print("  MISMATCH", c[:3])
